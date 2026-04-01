static constexpr unsigned int TALU_ATTN_WARP_SIZE = 32u;

static __forceinline__ __device__ float talu_attn_warp_sum_f32(float value) {
    const unsigned int mask = 0xFFFFffffu;
    for (unsigned int offset = TALU_ATTN_WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(mask, value, offset);
    }
    return value;
}

static __forceinline__ __device__ float talu_attn_warp_max_f32(float value) {
    const unsigned int mask = 0xFFFFffffu;
    for (unsigned int offset = TALU_ATTN_WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        const float other = __shfl_down_sync(mask, value, offset);
        value = fmaxf(value, other);
    }
    return value;
}

extern "C" __global__ void talu_attn_scores_heads_f16_kv(
    float* scores,
    const float* query,
    const unsigned short* key_cache,
    unsigned int n_heads,
    unsigned int seq_len,
    unsigned int row_stride,
    unsigned int kv_groups,
    unsigned int head_dim,
    float scale
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid / TALU_ATTN_WARP_SIZE;
    const unsigned int lane = tid & (TALU_ATTN_WARP_SIZE - 1u);
    const unsigned int warps_per_block = blockDim.x / TALU_ATTN_WARP_SIZE;
    const unsigned int token_index = blockIdx.x * warps_per_block + warp;
    const unsigned int head = blockIdx.y;
    if (head >= n_heads || token_index >= seq_len) return;
    if (kv_groups == 0) return;

    const unsigned int kv_head = head / kv_groups;
    const unsigned int head_offset = kv_head * head_dim;
    const float* query_head = query + (unsigned long long)head * head_dim;
    const unsigned short* key_row = key_cache + ((unsigned long long)token_index * row_stride + head_offset);

    float partial = 0.0f;
    for (unsigned int d = lane; d < head_dim; d += TALU_ATTN_WARP_SIZE) {
        partial += query_head[d] * __half2float(*reinterpret_cast<const __half*>(&key_row[d]));
    }
    const float dot = talu_attn_warp_sum_f32(partial);
    if (lane == 0) {
        scores[(unsigned long long)head * seq_len + token_index] = dot * scale;
    }
}

extern "C" __global__ void talu_attn_scores_heads_f32(
    float* scores,
    const float* query,
    const float* key_cache,
    unsigned int n_heads,
    unsigned int seq_len,
    unsigned int row_stride,
    unsigned int kv_groups,
    unsigned int head_dim,
    float scale
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid / TALU_ATTN_WARP_SIZE;
    const unsigned int lane = tid & (TALU_ATTN_WARP_SIZE - 1u);
    const unsigned int warps_per_block = blockDim.x / TALU_ATTN_WARP_SIZE;
    const unsigned int token_index = blockIdx.x * warps_per_block + warp;
    const unsigned int head = blockIdx.y;
    if (head >= n_heads || token_index >= seq_len) return;
    if (kv_groups == 0) return;

    const unsigned int kv_head = head / kv_groups;
    const unsigned int head_offset = kv_head * head_dim;
    const float* query_head = query + (unsigned long long)head * head_dim;
    const float* key_row = key_cache + ((unsigned long long)token_index * row_stride + head_offset);

    float partial = 0.0f;
    for (unsigned int d = lane; d < head_dim; d += TALU_ATTN_WARP_SIZE) {
        partial += query_head[d] * key_row[d];
    }
    const float dot = talu_attn_warp_sum_f32(partial);
    if (lane == 0) {
        scores[(unsigned long long)head * seq_len + token_index] = dot * scale;
    }
}

extern "C" __global__ void talu_softmax_rows_f32(
    float* out,
    const float* input,
    unsigned int rows,
    unsigned int cols
) {
    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & (TALU_ATTN_WARP_SIZE - 1u);
    const unsigned int warp = tid / TALU_ATTN_WARP_SIZE;
    const unsigned int warp_count = (blockDim.x + TALU_ATTN_WARP_SIZE - 1u) / TALU_ATTN_WARP_SIZE;
    if (row >= rows || cols == 0) return;

    const float* row_in = input + (unsigned long long)row * cols;
    float* row_out = out + (unsigned long long)row * cols;
    __shared__ float warp_max[8];
    __shared__ float warp_sum[8];
    __shared__ float row_max;
    __shared__ float row_inv_sum;

    float local_max = -3.402823466e+38f;
    for (unsigned int i = tid; i < cols; i += blockDim.x) {
        local_max = fmaxf(local_max, row_in[i]);
    }
    const float max_lane = talu_attn_warp_max_f32(local_max);
    if (lane == 0) warp_max[warp] = max_lane;
    __syncthreads();

    if (warp == 0) {
        float block_max = (lane < warp_count) ? warp_max[lane] : -3.402823466e+38f;
        block_max = talu_attn_warp_max_f32(block_max);
        if (lane == 0) row_max = block_max;
    }
    __syncthreads();

    const float max_v = row_max;
    float local_sum = 0.0f;
    for (unsigned int i = tid; i < cols; i += blockDim.x) {
        local_sum += expf(row_in[i] - max_v);
    }
    const float sum_lane = talu_attn_warp_sum_f32(local_sum);
    if (lane == 0) warp_sum[warp] = sum_lane;
    __syncthreads();

    if (warp == 0) {
        float block_sum = (lane < warp_count) ? warp_sum[lane] : 0.0f;
        block_sum = talu_attn_warp_sum_f32(block_sum);
        if (lane == 0) row_inv_sum = 1.0f / fmaxf(block_sum, 1.0e-20f);
    }
    __syncthreads();

    const float inv_sum = row_inv_sum;
    for (unsigned int i = tid; i < cols; i += blockDim.x) {
        row_out[i] = expf(row_in[i] - max_v) * inv_sum;
    }
}

extern "C" __global__ void talu_attn_weighted_sum_heads_f16_kv(
    float* out,
    const float* probs,
    const unsigned short* value_cache,
    unsigned int n_heads,
    unsigned int seq_len,
    unsigned int row_stride,
    unsigned int kv_groups,
    unsigned int head_dim
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid / TALU_ATTN_WARP_SIZE;
    const unsigned int lane = tid & (TALU_ATTN_WARP_SIZE - 1u);
    const unsigned int warps_per_block = blockDim.x / TALU_ATTN_WARP_SIZE;
    const unsigned int d = blockIdx.x * warps_per_block + warp;
    const unsigned int head = blockIdx.y;
    if (head >= n_heads || d >= head_dim) return;
    if (kv_groups == 0) return;

    const unsigned int kv_head = head / kv_groups;
    const unsigned int head_offset = kv_head * head_dim;
    const float* probs_row = probs + (unsigned long long)head * seq_len;

    float partial = 0.0f;
    for (unsigned int t = lane; t < seq_len; t += TALU_ATTN_WARP_SIZE) {
        const unsigned long long value_index = (unsigned long long)t * row_stride + head_offset + d;
        partial += probs_row[t] * __half2float(*reinterpret_cast<const __half*>(&value_cache[value_index]));
    }
    const float acc = talu_attn_warp_sum_f32(partial);
    if (lane == 0) {
        out[(unsigned long long)head * head_dim + d] = acc;
    }
}

extern "C" __global__ void talu_attn_weighted_sum_heads_f32(
    float* out,
    const float* probs,
    const float* value_cache,
    unsigned int n_heads,
    unsigned int seq_len,
    unsigned int row_stride,
    unsigned int kv_groups,
    unsigned int head_dim
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid / TALU_ATTN_WARP_SIZE;
    const unsigned int lane = tid & (TALU_ATTN_WARP_SIZE - 1u);
    const unsigned int warps_per_block = blockDim.x / TALU_ATTN_WARP_SIZE;
    const unsigned int d = blockIdx.x * warps_per_block + warp;
    const unsigned int head = blockIdx.y;
    if (head >= n_heads || d >= head_dim) return;
    if (kv_groups == 0) return;

    const unsigned int kv_head = head / kv_groups;
    const unsigned int head_offset = kv_head * head_dim;
    const float* probs_row = probs + (unsigned long long)head * seq_len;

    float partial = 0.0f;
    for (unsigned int t = lane; t < seq_len; t += TALU_ATTN_WARP_SIZE) {
        const unsigned long long value_index = (unsigned long long)t * row_stride + head_offset + d;
        partial += probs_row[t] * value_cache[value_index];
    }
    const float acc = talu_attn_warp_sum_f32(partial);
    if (lane == 0) {
        out[(unsigned long long)head * head_dim + d] = acc;
    }
}

extern "C" __global__ void talu_attn_fused_heads_f16_kv(
    float* out,
    const float* query,
    const unsigned short* key_cache,
    const unsigned short* value_cache,
    unsigned int n_heads,
    unsigned int seq_len,
    unsigned int row_stride,
    unsigned int kv_groups,
    unsigned int head_dim,
    float scale,
    unsigned int rope_dim,
    unsigned int position,
    float theta
) {
    const unsigned int head = blockIdx.x;
    const unsigned int lane = threadIdx.x & 31u;
    if (head >= n_heads || kv_groups == 0 || seq_len == 0 || head_dim == 0) return;
    if (rope_dim == 0 || rope_dim > head_dim || (rope_dim & 1u) != 0u) return;

    const unsigned int kv_head = head / kv_groups;
    const unsigned int head_offset = kv_head * head_dim;
    const float* query_head = query + (unsigned long long)head * head_dim;
    const unsigned int mask = 0xFFFFffffu;
    const float log2_theta = log2f(theta);

    // One warp per head. Each lane owns dimensions d = lane + k*32.
    // Use online softmax update to avoid score/prob buffers and avoid recomputing dots.
    const unsigned int dims_per_lane = (head_dim + 31u) >> 5;
    float out_acc[16];
    float q_rot[16];
    #pragma unroll
    for (unsigned int i = 0; i < 16; ++i) {
        out_acc[i] = 0.0f;
        q_rot[i] = 0.0f;
    }
    if (dims_per_lane > 16u) return;

    const unsigned int half = rope_dim >> 1;
    #pragma unroll
    for (unsigned int k = 0; k < 16; ++k) {
        if (k >= dims_per_lane) break;
        const unsigned int d = lane + (k << 5);
        if (d >= head_dim) continue;

        float qv = query_head[d];
        if (d < rope_dim) {
            const unsigned int pair = (d < half) ? d : (d - half);
            const float q_lo = query_head[pair];
            const float q_hi = query_head[half + pair];
            const float inv_freq = exp2f(log2_theta * (-2.0f * (float)pair / (float)rope_dim));
            const float angle = (float)position * inv_freq;
            float s = 0.0f;
            float c = 0.0f;
            __sincosf(angle, &s, &c);
            qv = (d < half) ? fmaf(q_lo, c, -q_hi * s) : fmaf(q_lo, s, q_hi * c);
        }
        q_rot[k] = qv;
    }

    float m = -3.402823466e+38f;
    float s = 0.0f;

    for (unsigned int t = 0; t < seq_len; ++t) {
        const unsigned short* key_row = key_cache + ((unsigned long long)t * row_stride + head_offset);
        float partial = 0.0f;
        #pragma unroll
        for (unsigned int k = 0; k < 16; ++k) {
            if (k >= dims_per_lane) break;
            const unsigned int d = lane + (k << 5);
            if (d >= head_dim) continue;
            partial += q_rot[k] * __half2float(*reinterpret_cast<const __half*>(&key_row[d]));
        }
        for (unsigned int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_down_sync(mask, partial, offset);
        }
        const float dot = __shfl_sync(mask, partial, 0);
        const float score = dot * scale;

        float alpha = 0.0f;
        float beta = 0.0f;
        if (lane == 0) {
            const float m_new = fmaxf(m, score);
            alpha = expf(m - m_new);
            beta = expf(score - m_new);
            s = s * alpha + beta;
            m = m_new;
        }
        alpha = __shfl_sync(mask, alpha, 0);
        beta = __shfl_sync(mask, beta, 0);

        const unsigned short* value_row = value_cache + ((unsigned long long)t * row_stride + head_offset);
        #pragma unroll
        for (unsigned int k = 0; k < 16; ++k) {
            if (k >= dims_per_lane) break;
            const unsigned int d = lane + (k << 5);
            if (d < head_dim) {
                const float v = __half2float(*reinterpret_cast<const __half*>(&value_row[d]));
                out_acc[k] = out_acc[k] * alpha + v * beta;
            }
        }
    }

    s = __shfl_sync(mask, s, 0);
    const float inv_s = 1.0f / fmaxf(s, 1.0e-20f);
    #pragma unroll
    for (unsigned int k = 0; k < 16; ++k) {
        if (k >= dims_per_lane) break;
        const unsigned int d = lane + (k << 5);
        if (d < head_dim) {
            out[(unsigned long long)head * head_dim + d] = out_acc[k] * inv_s;
        }
    }
}

extern "C" __global__ __launch_bounds__(512)
void talu_attn_fused_decode_heads_f16_kv_ptrs(
    float* out,
    const float* query,
    const unsigned long long* key_cache_ptrs,
    const unsigned long long* value_cache_ptrs,
    const unsigned int* seq_lens,
    const unsigned int* positions,
    unsigned int batch_rows,
    unsigned int n_heads,
    unsigned int row_stride,
    unsigned int kv_groups,
    unsigned int head_dim,
    float scale,
    unsigned int rope_dim,
    unsigned int sliding_window,
    float theta,
    const float* gate_proj,
    unsigned int gate_proj_stride
) {
    const unsigned int head = blockIdx.x;
    const unsigned int row = blockIdx.y;
    const unsigned int warp_id = threadIdx.x / 32u;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int num_warps = blockDim.x / 32u;
    if (row >= batch_rows || head >= n_heads || kv_groups == 0 || head_dim == 0) return;
    if (rope_dim == 0 || rope_dim > head_dim || (rope_dim & 1u) != 0u) return;

    const unsigned int seq_len = seq_lens[row];
    if (seq_len == 0) return;
    const unsigned int position = positions[row];
    const unsigned int effective_seq = min(seq_len, position + 1u);
    const unsigned int start_t = (sliding_window > 0u && effective_seq > sliding_window)
        ? (effective_seq - sliding_window)
        : 0u;

    const unsigned short* key_cache = reinterpret_cast<const unsigned short*>(key_cache_ptrs[row]);
    const unsigned short* value_cache = reinterpret_cast<const unsigned short*>(value_cache_ptrs[row]);
    if (key_cache == nullptr || value_cache == nullptr) return;

    const unsigned int kv_head = head / kv_groups;
    const unsigned int head_offset = kv_head * head_dim;
    const float* query_head = query + ((unsigned long long)row * n_heads + head) * head_dim;
    float* out_head = out + ((unsigned long long)row * n_heads + head) * head_dim;
    const unsigned int wmask = 0xFFFFffffu;
    const float log2_theta = log2f(theta);

    const unsigned int dims_per_lane = (head_dim + 31u) >> 5;
    float out_acc[16];
    float q_rot[16];
    #pragma unroll
    for (unsigned int i = 0; i < 16; ++i) {
        out_acc[i] = 0.0f;
        q_rot[i] = 0.0f;
    }
    if (dims_per_lane > 16u) return;

    // Q RoPE: identical across all warps (same head, same row).
    const unsigned int half = rope_dim >> 1;
    #pragma unroll
    for (unsigned int k = 0; k < 16; ++k) {
        if (k >= dims_per_lane) break;
        const unsigned int d = lane + (k << 5);
        if (d >= head_dim) continue;

        float qv = query_head[d];
        if (d < rope_dim) {
            const unsigned int pair = (d < half) ? d : (d - half);
            const float q_lo = query_head[pair];
            const float q_hi = query_head[half + pair];
            const float inv_freq = exp2f(log2_theta * (-2.0f * (float)pair / (float)rope_dim));
            const float angle = (float)position * inv_freq;
            float sn = 0.0f;
            float cn = 0.0f;
            __sincosf(angle, &sn, &cn);
            qv = (d < half) ? fmaf(q_lo, cn, -q_hi * sn) : fmaf(q_lo, sn, q_hi * cn);
        }
        q_rot[k] = qv;
    }

    // Split token range across warps (FlashDecoding pattern).
    const unsigned int total_tokens = effective_seq - start_t;
    const unsigned int tokens_per_warp = (total_tokens + num_warps - 1u) / num_warps;
    const unsigned int my_start = start_t + warp_id * tokens_per_warp;
    const unsigned int my_end = min(my_start + tokens_per_warp, effective_seq);

    // Online softmax over this warp's token chunk.
    float m = -3.402823466e+38f;
    float s = 0.0f;

    for (unsigned int t = my_start; t < my_end; ++t) {
        const unsigned short* key_row = key_cache + ((unsigned long long)t * row_stride + head_offset);
        float partial = 0.0f;
        #pragma unroll
        for (unsigned int k = 0; k < 16; ++k) {
            if (k >= dims_per_lane) break;
            const unsigned int d = lane + (k << 5);
            if (d >= head_dim) continue;
            partial += q_rot[k] * __half2float(*reinterpret_cast<const __half*>(&key_row[d]));
        }
        for (unsigned int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_down_sync(wmask, partial, offset);
        }
        const float dot = __shfl_sync(wmask, partial, 0);
        const float score = dot * scale;

        float alpha = 0.0f;
        float beta = 0.0f;
        if (lane == 0) {
            const float m_new = fmaxf(m, score);
            alpha = expf(m - m_new);
            beta = expf(score - m_new);
            s = s * alpha + beta;
            m = m_new;
        }
        alpha = __shfl_sync(wmask, alpha, 0);
        beta = __shfl_sync(wmask, beta, 0);

        const unsigned short* value_row = value_cache + ((unsigned long long)t * row_stride + head_offset);
        #pragma unroll
        for (unsigned int k = 0; k < 16; ++k) {
            if (k >= dims_per_lane) break;
            const unsigned int d = lane + (k << 5);
            if (d < head_dim) {
                const float v = __half2float(*reinterpret_cast<const __half*>(&value_row[d]));
                out_acc[k] = out_acc[k] * alpha + v * beta;
            }
        }
    }

    // Broadcast m and s from lane 0 to all lanes within each warp.
    m = __shfl_sync(wmask, m, 0);
    s = __shfl_sync(wmask, s, 0);

    // Combine partial results from all warps via shared memory.
    // Layout: [num_warps] m, [num_warps] s, [num_warps * head_dim] out.
    extern __shared__ float smem[];
    float* smem_m = smem;
    float* smem_s = smem + num_warps;
    float* smem_out = smem + 2u * num_warps;

    if (lane == 0) {
        smem_m[warp_id] = m;
        smem_s[warp_id] = s;
    }
    #pragma unroll
    for (unsigned int k = 0; k < 16; ++k) {
        if (k >= dims_per_lane) break;
        const unsigned int d = lane + (k << 5);
        if (d < head_dim) {
            smem_out[warp_id * head_dim + d] = out_acc[k];
        }
    }
    __syncthreads();

    // Only warp 0 combines and writes output.
    if (warp_id != 0) return;

    // Start with warp 0's values.
    m = smem_m[0];
    s = smem_s[0];
    #pragma unroll
    for (unsigned int k = 0; k < 16; ++k) {
        if (k >= dims_per_lane) break;
        const unsigned int d = lane + (k << 5);
        if (d < head_dim) {
            out_acc[k] = smem_out[d];
        }
    }

    // Merge warps 1..num_warps-1 using online-softmax merge formula.
    for (unsigned int w = 1; w < num_warps; ++w) {
        const float mw = smem_m[w];
        const float sw = smem_s[w];
        const float m_new = fmaxf(m, mw);
        const float alpha = expf(m - m_new);
        const float beta = expf(mw - m_new);
        s = s * alpha + sw * beta;
        m = m_new;

        #pragma unroll
        for (unsigned int k = 0; k < 16; ++k) {
            if (k >= dims_per_lane) break;
            const unsigned int d = lane + (k << 5);
            if (d < head_dim) {
                out_acc[k] = out_acc[k] * alpha + smem_out[w * head_dim + d] * beta;
            }
        }
    }

    // Normalize and write output.
    const float inv_s = 1.0f / fmaxf(s, 1.0e-20f);
    #pragma unroll
    for (unsigned int k = 0; k < 16; ++k) {
        if (k >= dims_per_lane) break;
        const unsigned int d = lane + (k << 5);
        if (d < head_dim) {
            float result = out_acc[k] * inv_s;
            if (gate_proj) {
                const float gate = gate_proj[(unsigned long long)row * gate_proj_stride + (unsigned long long)head * head_dim * 2u + head_dim + d];
                result *= 1.0f / (1.0f + expf(-gate));
            }
            out_head[d] = result;
        }
    }
}

extern "C" __global__ void talu_attn_fused_prefill_heads_f16_kv(
    float* out,
    const float* query,
    const unsigned short* key_cache,
    const unsigned short* value_cache,
    unsigned int n_heads,
    unsigned int q_rows,
    unsigned int seq_len,
    unsigned int row_stride,
    unsigned int kv_groups,
    unsigned int head_dim,
    float scale,
    unsigned int rope_dim,
    unsigned int position_base,
    unsigned int sliding_window,
    float theta
) {
    const unsigned int head = blockIdx.x;
    const unsigned int q_idx = blockIdx.y;
    const unsigned int lane = threadIdx.x & 31u;
    if (head >= n_heads || q_idx >= q_rows || kv_groups == 0 || seq_len == 0 || head_dim == 0) return;
    if (rope_dim == 0 || rope_dim > head_dim || (rope_dim & 1u) != 0u) return;

    const unsigned int kv_head = head / kv_groups;
    const unsigned int head_offset = kv_head * head_dim;
    const unsigned int mask = 0xFFFFffffu;
    const unsigned int query_pos = position_base + q_idx;
    const unsigned int effective_seq = min(seq_len, query_pos + 1u);
    const unsigned int start_t = (sliding_window > 0u && effective_seq > sliding_window) ? (effective_seq - sliding_window) : 0u;
    const float* query_head = query + ((unsigned long long)q_idx * n_heads + head) * head_dim;
    float* out_head = out + ((unsigned long long)q_idx * n_heads + head) * head_dim;
    const float log2_theta = log2f(theta);

    const unsigned int dims_per_lane = (head_dim + 31u) >> 5;
    float out_acc[16];
    float q_rot[16];
    #pragma unroll
    for (unsigned int i = 0; i < 16; ++i) {
        out_acc[i] = 0.0f;
        q_rot[i] = 0.0f;
    }
    if (dims_per_lane > 16u) return;

    const unsigned int half = rope_dim >> 1;
    #pragma unroll
    for (unsigned int k = 0; k < 16; ++k) {
        if (k >= dims_per_lane) break;
        const unsigned int d = lane + (k << 5);
        if (d >= head_dim) continue;

        float qv = query_head[d];
        if (d < rope_dim) {
            const unsigned int pair = (d < half) ? d : (d - half);
            const float q_lo = query_head[pair];
            const float q_hi = query_head[half + pair];
            const float inv_freq = exp2f(log2_theta * (-2.0f * (float)pair / (float)rope_dim));
            const float angle = (float)query_pos * inv_freq;
            float s = 0.0f;
            float c = 0.0f;
            __sincosf(angle, &s, &c);
            qv = (d < half) ? fmaf(q_lo, c, -q_hi * s) : fmaf(q_lo, s, q_hi * c);
        }
        q_rot[k] = qv;
    }

    float m = -3.402823466e+38f;
    float s = 0.0f;
    for (unsigned int t = start_t; t < effective_seq; ++t) {
        const unsigned short* key_row = key_cache + ((unsigned long long)t * row_stride + head_offset);
        float partial = 0.0f;
        #pragma unroll
        for (unsigned int k = 0; k < 16; ++k) {
            if (k >= dims_per_lane) break;
            const unsigned int d = lane + (k << 5);
            if (d >= head_dim) continue;
            partial += q_rot[k] * __half2float(*reinterpret_cast<const __half*>(&key_row[d]));
        }
        for (unsigned int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_down_sync(mask, partial, offset);
        }
        const float dot = __shfl_sync(mask, partial, 0);
        const float score = dot * scale;

        float alpha = 0.0f;
        float beta = 0.0f;
        if (lane == 0) {
            const float m_new = fmaxf(m, score);
            alpha = expf(m - m_new);
            beta = expf(score - m_new);
            s = s * alpha + beta;
            m = m_new;
        }
        alpha = __shfl_sync(mask, alpha, 0);
        beta = __shfl_sync(mask, beta, 0);

        const unsigned short* value_row = value_cache + ((unsigned long long)t * row_stride + head_offset);
        #pragma unroll
        for (unsigned int k = 0; k < 16; ++k) {
            if (k >= dims_per_lane) break;
            const unsigned int d = lane + (k << 5);
            if (d < head_dim) {
                const float v = __half2float(*reinterpret_cast<const __half*>(&value_row[d]));
                out_acc[k] = out_acc[k] * alpha + v * beta;
            }
        }
    }

    s = __shfl_sync(mask, s, 0);
    const float inv_s = 1.0f / fmaxf(s, 1.0e-20f);
    #pragma unroll
    for (unsigned int k = 0; k < 16; ++k) {
        if (k >= dims_per_lane) break;
        const unsigned int d = lane + (k << 5);
        if (d < head_dim) {
            out_head[d] = out_acc[k] * inv_s;
        }
    }
}

// GQA-aware variant: groups Q heads sharing a KV head into one block,
// loads KV tiles into shared memory once, and serves all warps from smem.
// Reduces global memory bandwidth by kv_groups× compared to the non-GQA kernel.
// Grid: (n_kv_heads, q_rows).  Block: kv_groups * 32 threads.
// Dynamic shared memory: 2 * GQA_KV_TILE * head_dim * sizeof(__half).

#define GQA_KV_TILE 32u

extern "C" __global__ void talu_attn_fused_prefill_heads_f16_kv_gqa(
    float* __restrict__ out,
    const float* __restrict__ query,
    const unsigned short* __restrict__ key_cache,
    const unsigned short* __restrict__ value_cache,
    unsigned int n_heads,
    unsigned int q_rows,
    unsigned int seq_len,
    unsigned int row_stride,
    unsigned int kv_groups,
    unsigned int head_dim,
    float scale,
    unsigned int rope_dim,
    unsigned int position_base,
    unsigned int sliding_window,
    float theta
) {
    const unsigned int n_kv_heads = n_heads / kv_groups;
    const unsigned int kv_head = blockIdx.x;
    const unsigned int q_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / TALU_ATTN_WARP_SIZE;
    const unsigned int lane = tid & (TALU_ATTN_WARP_SIZE - 1u);
    const unsigned int mask = 0xFFFFffffu;

    if (kv_head >= n_kv_heads || q_idx >= q_rows || warp_id >= kv_groups) return;
    if (kv_groups == 0 || seq_len == 0 || head_dim == 0) return;
    if (rope_dim == 0 || rope_dim > head_dim || (rope_dim & 1u) != 0u) return;

    const unsigned int head = kv_head * kv_groups + warp_id;
    if (head >= n_heads) return;

    const unsigned int query_pos = position_base + q_idx;
    const unsigned int effective_seq = min(seq_len, query_pos + 1u);
    const unsigned int start_t = (sliding_window > 0u && effective_seq > sliding_window)
                                 ? (effective_seq - sliding_window) : 0u;
    const unsigned int head_offset = kv_head * head_dim;
    const float* query_head = query + ((unsigned long long)q_idx * n_heads + head) * head_dim;
    float* out_head = out + ((unsigned long long)q_idx * n_heads + head) * head_dim;
    const float log2_theta = log2f(theta);

    const unsigned int dims_per_lane = (head_dim + 31u) >> 5;
    float out_acc[16];
    float q_rot[16];
    #pragma unroll
    for (unsigned int i = 0; i < 16; ++i) {
        out_acc[i] = 0.0f;
        q_rot[i] = 0.0f;
    }
    if (dims_per_lane > 16u) return;

    // RoPE rotate query (identical to non-GQA kernel).
    const unsigned int half_rope = rope_dim >> 1;
    #pragma unroll
    for (unsigned int k = 0; k < 16; ++k) {
        if (k >= dims_per_lane) break;
        const unsigned int d = lane + (k << 5);
        if (d >= head_dim) continue;

        float qv = query_head[d];
        if (d < rope_dim) {
            const unsigned int pair = (d < half_rope) ? d : (d - half_rope);
            const float q_lo = query_head[pair];
            const float q_hi = query_head[half_rope + pair];
            const float inv_freq = exp2f(log2_theta * (-2.0f * (float)pair / (float)rope_dim));
            const float angle = (float)query_pos * inv_freq;
            float sin_v = 0.0f;
            float cos_v = 0.0f;
            __sincosf(angle, &sin_v, &cos_v);
            qv = (d < half_rope) ? fmaf(q_lo, cos_v, -q_hi * sin_v)
                                 : fmaf(q_lo, sin_v, q_hi * cos_v);
        }
        q_rot[k] = qv;
    }

    // Shared memory: [GQA_KV_TILE * head_dim] K values then [GQA_KV_TILE * head_dim] V values.
    extern __shared__ unsigned short kv_smem[];
    const unsigned int kv_tile_stride = GQA_KV_TILE * head_dim;
    unsigned short* k_smem = kv_smem;
    unsigned short* v_smem = kv_smem + kv_tile_stride;

    const unsigned int total_threads = blockDim.x;
    float m = -3.402823466e+38f;
    float acc_sum = 0.0f;

    for (unsigned int tile_start = start_t; tile_start < effective_seq; tile_start += GQA_KV_TILE) {
        const unsigned int tile_end = min(tile_start + GQA_KV_TILE, effective_seq);
        const unsigned int tile_size = tile_end - tile_start;

        // Cooperative load: all warps load KV tile into shared memory.
        const unsigned int total_elems = tile_size * head_dim;
        for (unsigned int idx = tid; idx < total_elems; idx += total_threads) {
            const unsigned int t_local = idx / head_dim;
            const unsigned int d = idx - t_local * head_dim;
            const unsigned long long cache_idx =
                (unsigned long long)(tile_start + t_local) * row_stride + head_offset + d;
            k_smem[t_local * head_dim + d] = key_cache[cache_idx];
            v_smem[t_local * head_dim + d] = value_cache[cache_idx];
        }
        __syncthreads();

        // Each warp processes its Q head against the tile from shared memory.
        for (unsigned int t_local = 0; t_local < tile_size; ++t_local) {
            const unsigned int smem_row = t_local * head_dim;

            // Q · K dot product.
            float partial = 0.0f;
            #pragma unroll
            for (unsigned int k = 0; k < 16; ++k) {
                if (k >= dims_per_lane) break;
                const unsigned int d = lane + (k << 5);
                if (d < head_dim) {
                    partial += q_rot[k] * __half2float(
                        *reinterpret_cast<const __half*>(&k_smem[smem_row + d]));
                }
            }
            for (unsigned int offset = 16; offset > 0; offset >>= 1) {
                partial += __shfl_down_sync(mask, partial, offset);
            }
            const float dot = __shfl_sync(mask, partial, 0);
            const float score = dot * scale;

            // Online softmax update.
            float alpha = 0.0f;
            float beta = 0.0f;
            if (lane == 0) {
                const float m_new = fmaxf(m, score);
                alpha = expf(m - m_new);
                beta = expf(score - m_new);
                acc_sum = acc_sum * alpha + beta;
                m = m_new;
            }
            alpha = __shfl_sync(mask, alpha, 0);
            beta = __shfl_sync(mask, beta, 0);

            // Value accumulation from shared memory.
            #pragma unroll
            for (unsigned int k = 0; k < 16; ++k) {
                if (k >= dims_per_lane) break;
                const unsigned int d = lane + (k << 5);
                if (d < head_dim) {
                    const float v = __half2float(
                        *reinterpret_cast<const __half*>(&v_smem[smem_row + d]));
                    out_acc[k] = out_acc[k] * alpha + v * beta;
                }
            }
        }
        __syncthreads();
    }

    // Normalize and write output.
    acc_sum = __shfl_sync(mask, acc_sum, 0);
    const float inv_sum = 1.0f / fmaxf(acc_sum, 1.0e-20f);
    #pragma unroll
    for (unsigned int k = 0; k < 16; ++k) {
        if (k >= dims_per_lane) break;
        const unsigned int d = lane + (k << 5);
        if (d < head_dim) {
            out_head[d] = out_acc[k] * inv_sum;
        }
    }
}

// Causal attention softmax: applies causal mask and row-wise softmax in one pass.
// Grid: (total_rows, 1).  Block: (128, 1).
// total_rows = kv_groups * q_rows (or n_heads * q_rows depending on batching).
// For row r: q_idx = r % q_rows, valid range [0, position_base + q_idx].
// Positions outside the valid range are treated as -inf before softmax.

extern "C" __global__ void talu_causal_attn_softmax_f32(
    float* scores,
    unsigned int total_rows,
    unsigned int cols,
    unsigned int q_rows,
    unsigned int position_base,
    unsigned int sliding_window
) {
    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & (TALU_ATTN_WARP_SIZE - 1u);
    const unsigned int warp = tid / TALU_ATTN_WARP_SIZE;
    const unsigned int warp_count = (blockDim.x + TALU_ATTN_WARP_SIZE - 1u) / TALU_ATTN_WARP_SIZE;
    if (row >= total_rows || cols == 0 || q_rows == 0) return;

    const unsigned int q_idx = row % q_rows;
    const unsigned int q_pos = position_base + q_idx;
    const unsigned int mask_end = (q_pos < cols) ? q_pos : (cols - 1u);
    const unsigned int mask_start = (sliding_window > 0u && q_pos >= sliding_window)
                                    ? (q_pos - sliding_window + 1u) : 0u;

    float* row_data = scores + (unsigned long long)row * cols;
    __shared__ float warp_max[8];
    __shared__ float warp_sum[8];
    __shared__ float row_max;
    __shared__ float row_inv_sum;

    // Phase 1: find max over valid elements.
    float local_max = -3.402823466e+38f;
    for (unsigned int i = tid; i <= mask_end; i += blockDim.x) {
        if (i >= mask_start) {
            local_max = fmaxf(local_max, row_data[i]);
        }
    }
    const float max_lane = talu_attn_warp_max_f32(local_max);
    if (lane == 0) warp_max[warp] = max_lane;
    __syncthreads();

    if (warp == 0) {
        float block_max = (lane < warp_count) ? warp_max[lane] : -3.402823466e+38f;
        block_max = talu_attn_warp_max_f32(block_max);
        if (lane == 0) row_max = block_max;
    }
    __syncthreads();

    const float max_v = row_max;

    // Phase 2: exp and sum over valid elements, zero masked.
    float local_sum = 0.0f;
    for (unsigned int i = tid; i < cols; i += blockDim.x) {
        if (i >= mask_start && i <= mask_end) {
            const float e = expf(row_data[i] - max_v);
            row_data[i] = e;
            local_sum += e;
        } else {
            row_data[i] = 0.0f;
        }
    }
    const float sum_lane = talu_attn_warp_sum_f32(local_sum);
    if (lane == 0) warp_sum[warp] = sum_lane;
    __syncthreads();

    if (warp == 0) {
        float block_sum = (lane < warp_count) ? warp_sum[lane] : 0.0f;
        block_sum = talu_attn_warp_sum_f32(block_sum);
        if (lane == 0) row_inv_sum = 1.0f / fmaxf(block_sum, 1.0e-20f);
    }
    __syncthreads();

    // Phase 3: normalize.
    const float inv_sum = row_inv_sum;
    for (unsigned int i = tid; i < cols; i += blockDim.x) {
        row_data[i] *= inv_sum;
    }
}

// --- Batched separate attention kernels (graph-compatible) ---
// These replicate the proven-fast separate kernel design but read position/seq_len
// from device buffers and KV cache from pointer tables, enabling CUDA graph capture.

// Batched RoPE: apply rotary position embedding to data for multiple rows,
// reading per-row positions from a device buffer.
// Grid: (ceil(n_heads * (rope_dim/2) / blockDim.x), batch_rows), Block: 128
extern "C" __global__ void talu_rope_rows_ptrs(
    float* io,
    const unsigned int* positions,
    unsigned int batch_rows,
    unsigned int n_heads,
    unsigned int head_dim,
    unsigned int rope_dim,
    float theta
) {
    const unsigned int row = blockIdx.y;
    if (row >= batch_rows) return;

    const unsigned int half = rope_dim >> 1;
    const unsigned int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total_pairs = n_heads * half;
    if (pair_index >= total_pairs) return;

    const unsigned int position = positions[row];
    const unsigned int head = pair_index / half;
    const unsigned int pair = pair_index % half;
    const unsigned int base = (row * n_heads + head) * head_dim;
    const float log2_theta = log2f(theta);
    const float inv_freq = exp2f(log2_theta * (-2.0f * (float)pair / (float)rope_dim));
    const float angle = (float)position * inv_freq;
    float s = 0.0f, c = 0.0f;
    __sincosf(angle, &s, &c);

    const unsigned int lo_idx = base + pair;
    const unsigned int hi_idx = base + half + pair;
    const float x0 = io[lo_idx];
    const float x1 = io[hi_idx];
    io[lo_idx] = fmaf(x0, c, -x1 * s);
    io[hi_idx] = fmaf(x0, s, x1 * c);
}

// Batched attention scores with f16 KV pointer tables.
// Computes Q*K dot product for each (token, head, row), reading KV cache pointers
// and seq_lens from device buffers. Out-of-range tokens write -inf for softmax.
// Grid: (ceil(max_seq_len / warps_per_block), n_heads, batch_rows), Block: 128
extern "C" __global__ void talu_attn_scores_heads_f16_kv_ptrs(
    float* scores,
    const float* query,
    const unsigned long long* key_cache_ptrs,
    const unsigned int* seq_lens,
    const unsigned int* positions,
    unsigned int n_heads,
    unsigned int max_seq_len,
    unsigned int row_stride,
    unsigned int kv_groups,
    unsigned int head_dim,
    float scale,
    unsigned int sliding_window
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid / TALU_ATTN_WARP_SIZE;
    const unsigned int lane = tid & (TALU_ATTN_WARP_SIZE - 1u);
    const unsigned int warps_per_block = blockDim.x / TALU_ATTN_WARP_SIZE;
    const unsigned int token_index = blockIdx.x * warps_per_block + warp;
    const unsigned int head = blockIdx.y;
    const unsigned int row = blockIdx.z;
    if (head >= n_heads || kv_groups == 0) return;

    const unsigned int seq_len = seq_lens[row];
    const unsigned int position = positions[row];
    const unsigned int effective_seq = min(seq_len, position + 1u);
    const unsigned int start_t = (sliding_window > 0u && effective_seq > sliding_window)
        ? (effective_seq - sliding_window) : 0u;
    const unsigned int visible_len = effective_seq - start_t;

    // Scores layout: [row, head, max_seq_len]. We write to visible_len slots
    // starting at offset 0, and -inf beyond.
    float* scores_head = scores + ((unsigned long long)row * n_heads + head) * max_seq_len;

    if (token_index >= visible_len) {
        if (token_index < max_seq_len && lane == 0) {
            scores_head[token_index] = -3.402823466e+38f;
        }
        return;
    }

    const unsigned int kv_head = head / kv_groups;
    const unsigned int head_offset = kv_head * head_dim;
    const float* query_head = query + ((unsigned long long)row * n_heads + head) * head_dim;
    const unsigned short* key_cache = reinterpret_cast<const unsigned short*>(key_cache_ptrs[row]);
    const unsigned int actual_t = start_t + token_index;
    const unsigned short* key_row = key_cache + ((unsigned long long)actual_t * row_stride + head_offset);

    float partial = 0.0f;
    for (unsigned int d = lane; d < head_dim; d += TALU_ATTN_WARP_SIZE) {
        partial += query_head[d] * __half2float(*reinterpret_cast<const __half*>(&key_row[d]));
    }
    const float dot = talu_attn_warp_sum_f32(partial);
    if (lane == 0) {
        scores_head[token_index] = dot * scale;
    }
}

// Softmax with per-row dynamic column counts read from device buffer.
// Each block handles one row of the [batch_rows * n_heads, max_cols] matrix.
// Reads actual column count from seq_lens[row / n_heads] so only real scores
// are softmaxed (no -inf padding waste).
// Grid: (batch_rows * n_heads), Block: 128
extern "C" __global__ void talu_softmax_rows_dynamic_cols_ptrs(
    float* data,
    const unsigned int* seq_lens,
    const unsigned int* positions,
    unsigned int n_heads,
    unsigned int max_cols,
    unsigned int sliding_window
) {
    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & (TALU_ATTN_WARP_SIZE - 1u);
    const unsigned int warp = tid / TALU_ATTN_WARP_SIZE;
    const unsigned int warp_count = (blockDim.x + TALU_ATTN_WARP_SIZE - 1u) / TALU_ATTN_WARP_SIZE;

    const unsigned int batch_row = row / n_heads;
    const unsigned int seq_len = seq_lens[batch_row];
    const unsigned int position = positions[batch_row];
    const unsigned int effective_seq = min(seq_len, position + 1u);
    const unsigned int start_t = (sliding_window > 0u && effective_seq > sliding_window)
        ? (effective_seq - sliding_window) : 0u;
    const unsigned int cols = effective_seq - start_t;
    if (cols == 0) return;

    float* row_data = data + (unsigned long long)row * max_cols;
    __shared__ float warp_max[8];
    __shared__ float warp_sum[8];
    __shared__ float row_max;
    __shared__ float row_inv_sum;

    // Phase 1: find max.
    float local_max = -3.402823466e+38f;
    for (unsigned int i = tid; i < cols; i += blockDim.x) {
        local_max = fmaxf(local_max, row_data[i]);
    }
    const float max_lane = talu_attn_warp_max_f32(local_max);
    if (lane == 0) warp_max[warp] = max_lane;
    __syncthreads();

    if (warp == 0) {
        float block_max = (lane < warp_count) ? warp_max[lane] : -3.402823466e+38f;
        block_max = talu_attn_warp_max_f32(block_max);
        if (lane == 0) row_max = block_max;
    }
    __syncthreads();

    // Phase 2: exp + sum.
    const float max_v = row_max;
    float local_sum = 0.0f;
    for (unsigned int i = tid; i < cols; i += blockDim.x) {
        const float e = expf(row_data[i] - max_v);
        row_data[i] = e;
        local_sum += e;
    }
    const float sum_lane = talu_attn_warp_sum_f32(local_sum);
    if (lane == 0) warp_sum[warp] = sum_lane;
    __syncthreads();

    if (warp == 0) {
        float block_sum = (lane < warp_count) ? warp_sum[lane] : 0.0f;
        block_sum = talu_attn_warp_sum_f32(block_sum);
        if (lane == 0) row_inv_sum = 1.0f / fmaxf(block_sum, 1.0e-20f);
    }
    __syncthreads();

    // Phase 3: normalize.
    const float inv_sum = row_inv_sum;
    for (unsigned int i = tid; i < cols; i += blockDim.x) {
        row_data[i] *= inv_sum;
    }
}

// Batched attention weighted sum with f16 KV pointer tables.
// Each warp computes the weighted sum for one dimension of the output.
// Grid: (ceil(head_dim / warps_per_block), n_heads, batch_rows), Block: 128
extern "C" __global__ void talu_attn_weighted_sum_heads_f16_kv_ptrs(
    float* out,
    const float* probs,
    const unsigned long long* value_cache_ptrs,
    const unsigned int* seq_lens,
    const unsigned int* positions,
    unsigned int n_heads,
    unsigned int max_seq_len,
    unsigned int row_stride,
    unsigned int kv_groups,
    unsigned int head_dim,
    unsigned int sliding_window
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid / TALU_ATTN_WARP_SIZE;
    const unsigned int lane = tid & (TALU_ATTN_WARP_SIZE - 1u);
    const unsigned int warps_per_block = blockDim.x / TALU_ATTN_WARP_SIZE;
    const unsigned int d = blockIdx.x * warps_per_block + warp;
    const unsigned int head = blockIdx.y;
    const unsigned int row = blockIdx.z;
    if (head >= n_heads || d >= head_dim || kv_groups == 0) return;

    const unsigned int seq_len = seq_lens[row];
    const unsigned int position = positions[row];
    const unsigned int effective_seq = min(seq_len, position + 1u);
    const unsigned int start_t = (sliding_window > 0u && effective_seq > sliding_window)
        ? (effective_seq - sliding_window) : 0u;
    const unsigned int visible_len = effective_seq - start_t;

    const unsigned int kv_head = head / kv_groups;
    const unsigned int head_offset = kv_head * head_dim;
    const float* probs_row = probs + ((unsigned long long)row * n_heads + head) * max_seq_len;
    const unsigned short* value_cache = reinterpret_cast<const unsigned short*>(value_cache_ptrs[row]);

    float partial = 0.0f;
    for (unsigned int t = lane; t < visible_len; t += TALU_ATTN_WARP_SIZE) {
        const unsigned int actual_t = start_t + t;
        const unsigned long long value_index = (unsigned long long)actual_t * row_stride + head_offset + d;
        partial += probs_row[t] * __half2float(*reinterpret_cast<const __half*>(&value_cache[value_index]));
    }
    const float acc = talu_attn_warp_sum_f32(partial);
    if (lane == 0) {
        out[((unsigned long long)row * n_heads + head) * head_dim + d] = acc;
    }
}

// --- INT8 KV cache attention kernels ---
// Per-head-per-token symmetric quantization: dequant = (float)(int8) * scale.
// Scale layout: [capacity, n_kv_heads] where scales[t * n_kv_heads + kv_head].
// K dot products factor out the scale: dot = scale * sum(q[d] * (float)k_i8[d]).

// Separate Q*K scores with INT8 K cache.
// Grid: (ceil(seq_len / warps_per_block), n_heads), Block: 128
extern "C" __global__ void talu_attn_scores_heads_i8_kv(
    float* scores,
    const float* query,
    const signed char* key_cache,
    const float* k_scales,
    unsigned int n_heads,
    unsigned int n_kv_heads,
    unsigned int seq_len,
    unsigned int row_stride,
    unsigned int kv_groups,
    unsigned int head_dim,
    float scale
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid / TALU_ATTN_WARP_SIZE;
    const unsigned int lane = tid & (TALU_ATTN_WARP_SIZE - 1u);
    const unsigned int warps_per_block = blockDim.x / TALU_ATTN_WARP_SIZE;
    const unsigned int token_index = blockIdx.x * warps_per_block + warp;
    const unsigned int head = blockIdx.y;
    if (head >= n_heads || token_index >= seq_len) return;
    if (kv_groups == 0) return;

    const unsigned int kv_head = head / kv_groups;
    const unsigned int head_offset = kv_head * head_dim;
    const float* query_head = query + (unsigned long long)head * head_dim;
    const signed char* key_row = key_cache + ((unsigned long long)token_index * row_stride + head_offset);
    const float k_scale = k_scales[token_index * n_kv_heads + kv_head];

    float partial = 0.0f;
    for (unsigned int d = lane; d < head_dim; d += TALU_ATTN_WARP_SIZE) {
        partial += query_head[d] * (float)(key_row[d]);
    }
    const float dot = talu_attn_warp_sum_f32(partial) * k_scale;
    if (lane == 0) {
        scores[(unsigned long long)head * seq_len + token_index] = dot * scale;
    }
}

// Separate weighted sum with INT8 V cache.
// Grid: (ceil(head_dim / warps_per_block), n_heads), Block: 128
extern "C" __global__ void talu_attn_weighted_sum_heads_i8_kv(
    float* out,
    const float* probs,
    const signed char* value_cache,
    const float* v_scales,
    unsigned int n_heads,
    unsigned int n_kv_heads,
    unsigned int seq_len,
    unsigned int row_stride,
    unsigned int kv_groups,
    unsigned int head_dim
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid / TALU_ATTN_WARP_SIZE;
    const unsigned int lane = tid & (TALU_ATTN_WARP_SIZE - 1u);
    const unsigned int warps_per_block = blockDim.x / TALU_ATTN_WARP_SIZE;
    const unsigned int d = blockIdx.x * warps_per_block + warp;
    const unsigned int head = blockIdx.y;
    if (head >= n_heads || d >= head_dim) return;
    if (kv_groups == 0) return;

    const unsigned int kv_head = head / kv_groups;
    const unsigned int head_offset = kv_head * head_dim;
    const float* probs_row = probs + (unsigned long long)head * seq_len;

    float partial = 0.0f;
    for (unsigned int t = lane; t < seq_len; t += TALU_ATTN_WARP_SIZE) {
        const float v_scale = v_scales[t * n_kv_heads + kv_head];
        const unsigned long long value_index = (unsigned long long)t * row_stride + head_offset + d;
        partial += probs_row[t] * v_scale * (float)(value_cache[value_index]);
    }
    const float acc = talu_attn_warp_sum_f32(partial);
    if (lane == 0) {
        out[(unsigned long long)head * head_dim + d] = acc;
    }
}

// Fused single-token decode attention with INT8 KV (online softmax).
// One warp per head. Grid: (n_heads,), Block: (32,)
extern "C" __global__ void talu_attn_fused_heads_i8_kv(
    float* out,
    const float* query,
    const signed char* key_cache,
    const signed char* value_cache,
    const float* k_scales,
    const float* v_scales,
    unsigned int n_heads,
    unsigned int n_kv_heads,
    unsigned int seq_len,
    unsigned int row_stride,
    unsigned int kv_groups,
    unsigned int head_dim,
    float scale,
    unsigned int rope_dim,
    unsigned int position,
    float theta
) {
    const unsigned int head = blockIdx.x;
    const unsigned int lane = threadIdx.x & 31u;
    if (head >= n_heads || kv_groups == 0 || seq_len == 0 || head_dim == 0) return;
    if (rope_dim == 0 || rope_dim > head_dim || (rope_dim & 1u) != 0u) return;

    const unsigned int kv_head = head / kv_groups;
    const unsigned int head_offset = kv_head * head_dim;
    const float* query_head = query + (unsigned long long)head * head_dim;
    const unsigned int mask = 0xFFFFffffu;
    const float log2_theta = log2f(theta);

    const unsigned int dims_per_lane = (head_dim + 31u) >> 5;
    float out_acc[16];
    float q_rot[16];
    #pragma unroll
    for (unsigned int i = 0; i < 16; ++i) {
        out_acc[i] = 0.0f;
        q_rot[i] = 0.0f;
    }
    if (dims_per_lane > 16u) return;

    const unsigned int half = rope_dim >> 1;
    #pragma unroll
    for (unsigned int k = 0; k < 16; ++k) {
        if (k >= dims_per_lane) break;
        const unsigned int d = lane + (k << 5);
        if (d >= head_dim) continue;

        float qv = query_head[d];
        if (d < rope_dim) {
            const unsigned int pair = (d < half) ? d : (d - half);
            const float q_lo = query_head[pair];
            const float q_hi = query_head[half + pair];
            const float inv_freq = exp2f(log2_theta * (-2.0f * (float)pair / (float)rope_dim));
            const float angle = (float)position * inv_freq;
            float s = 0.0f;
            float c = 0.0f;
            __sincosf(angle, &s, &c);
            qv = (d < half) ? fmaf(q_lo, c, -q_hi * s) : fmaf(q_lo, s, q_hi * c);
        }
        q_rot[k] = qv;
    }

    float m = -3.402823466e+38f;
    float s = 0.0f;

    for (unsigned int t = 0; t < seq_len; ++t) {
        const signed char* key_row = key_cache + ((unsigned long long)t * row_stride + head_offset);
        const float k_sc = k_scales[t * n_kv_heads + kv_head];
        float partial = 0.0f;
        #pragma unroll
        for (unsigned int k = 0; k < 16; ++k) {
            if (k >= dims_per_lane) break;
            const unsigned int d = lane + (k << 5);
            if (d >= head_dim) continue;
            partial += q_rot[k] * (float)(key_row[d]);
        }
        for (unsigned int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_down_sync(mask, partial, offset);
        }
        const float dot = __shfl_sync(mask, partial, 0) * k_sc;
        const float score = dot * scale;

        float alpha = 0.0f;
        float beta = 0.0f;
        if (lane == 0) {
            const float m_new = fmaxf(m, score);
            alpha = expf(m - m_new);
            beta = expf(score - m_new);
            s = s * alpha + beta;
            m = m_new;
        }
        alpha = __shfl_sync(mask, alpha, 0);
        beta = __shfl_sync(mask, beta, 0);

        const signed char* value_row = value_cache + ((unsigned long long)t * row_stride + head_offset);
        const float v_sc = v_scales[t * n_kv_heads + kv_head];
        #pragma unroll
        for (unsigned int k = 0; k < 16; ++k) {
            if (k >= dims_per_lane) break;
            const unsigned int d = lane + (k << 5);
            if (d < head_dim) {
                const float v = (float)(value_row[d]) * v_sc;
                out_acc[k] = out_acc[k] * alpha + v * beta;
            }
        }
    }

    s = __shfl_sync(mask, s, 0);
    const float inv_s = 1.0f / fmaxf(s, 1.0e-20f);
    #pragma unroll
    for (unsigned int k = 0; k < 16; ++k) {
        if (k >= dims_per_lane) break;
        const unsigned int d = lane + (k << 5);
        if (d < head_dim) {
            out[(unsigned long long)head * head_dim + d] = out_acc[k] * inv_s;
        }
    }
}

// Fused batched decode attention with INT8 KV + pointer tables.
// Multi-warp FlashDecoding with online softmax merge.
// Grid: (n_heads, batch_rows), Block: (512,)
extern "C" __global__ __launch_bounds__(512)
void talu_attn_fused_decode_heads_i8_kv_ptrs(
    float* out,
    const float* query,
    const unsigned long long* key_cache_ptrs,
    const unsigned long long* value_cache_ptrs,
    const unsigned long long* k_scale_ptrs,
    const unsigned long long* v_scale_ptrs,
    const unsigned int* seq_lens,
    const unsigned int* positions,
    unsigned int batch_rows,
    unsigned int n_heads,
    unsigned int n_kv_heads,
    unsigned int row_stride,
    unsigned int kv_groups,
    unsigned int head_dim,
    float scale,
    unsigned int rope_dim,
    unsigned int sliding_window,
    float theta,
    const float* gate_proj,
    unsigned int gate_proj_stride
) {
    const unsigned int head = blockIdx.x;
    const unsigned int row = blockIdx.y;
    const unsigned int warp_id = threadIdx.x / 32u;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int num_warps = blockDim.x / 32u;
    if (row >= batch_rows || head >= n_heads || kv_groups == 0 || head_dim == 0) return;
    if (rope_dim == 0 || rope_dim > head_dim || (rope_dim & 1u) != 0u) return;

    const unsigned int seq_len = seq_lens[row];
    if (seq_len == 0) return;
    const unsigned int position = positions[row];
    const unsigned int effective_seq = min(seq_len, position + 1u);
    const unsigned int start_t = (sliding_window > 0u && effective_seq > sliding_window)
        ? (effective_seq - sliding_window)
        : 0u;

    const signed char* key_cache = reinterpret_cast<const signed char*>(key_cache_ptrs[row]);
    const signed char* value_cache = reinterpret_cast<const signed char*>(value_cache_ptrs[row]);
    const float* k_scales = reinterpret_cast<const float*>(k_scale_ptrs[row]);
    const float* v_scales = reinterpret_cast<const float*>(v_scale_ptrs[row]);
    if (key_cache == nullptr || value_cache == nullptr) return;

    const unsigned int kv_head = head / kv_groups;
    const unsigned int head_offset = kv_head * head_dim;
    const float* query_head = query + ((unsigned long long)row * n_heads + head) * head_dim;
    float* out_head = out + ((unsigned long long)row * n_heads + head) * head_dim;
    const unsigned int wmask = 0xFFFFffffu;
    const float log2_theta = log2f(theta);

    const unsigned int dims_per_lane = (head_dim + 31u) >> 5;
    float out_acc[16];
    float q_rot[16];
    #pragma unroll
    for (unsigned int i = 0; i < 16; ++i) {
        out_acc[i] = 0.0f;
        q_rot[i] = 0.0f;
    }
    if (dims_per_lane > 16u) return;

    // Q RoPE.
    const unsigned int half = rope_dim >> 1;
    #pragma unroll
    for (unsigned int k = 0; k < 16; ++k) {
        if (k >= dims_per_lane) break;
        const unsigned int d = lane + (k << 5);
        if (d >= head_dim) continue;

        float qv = query_head[d];
        if (d < rope_dim) {
            const unsigned int pair = (d < half) ? d : (d - half);
            const float q_lo = query_head[pair];
            const float q_hi = query_head[half + pair];
            const float inv_freq = exp2f(log2_theta * (-2.0f * (float)pair / (float)rope_dim));
            const float angle = (float)position * inv_freq;
            float sn = 0.0f;
            float cn = 0.0f;
            __sincosf(angle, &sn, &cn);
            qv = (d < half) ? fmaf(q_lo, cn, -q_hi * sn) : fmaf(q_lo, sn, q_hi * cn);
        }
        q_rot[k] = qv;
    }

    // Split tokens across warps.
    const unsigned int total_tokens = effective_seq - start_t;
    const unsigned int tokens_per_warp = (total_tokens + num_warps - 1u) / num_warps;
    const unsigned int my_start = start_t + warp_id * tokens_per_warp;
    const unsigned int my_end = min(my_start + tokens_per_warp, effective_seq);

    float m = -3.402823466e+38f;
    float s = 0.0f;

    for (unsigned int t = my_start; t < my_end; ++t) {
        const signed char* key_row = key_cache + ((unsigned long long)t * row_stride + head_offset);
        const float k_sc = k_scales[t * n_kv_heads + kv_head];
        float partial = 0.0f;
        #pragma unroll
        for (unsigned int k = 0; k < 16; ++k) {
            if (k >= dims_per_lane) break;
            const unsigned int d = lane + (k << 5);
            if (d >= head_dim) continue;
            partial += q_rot[k] * (float)(key_row[d]);
        }
        for (unsigned int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_down_sync(wmask, partial, offset);
        }
        const float dot = __shfl_sync(wmask, partial, 0) * k_sc;
        const float score = dot * scale;

        float alpha = 0.0f;
        float beta = 0.0f;
        if (lane == 0) {
            const float m_new = fmaxf(m, score);
            alpha = expf(m - m_new);
            beta = expf(score - m_new);
            s = s * alpha + beta;
            m = m_new;
        }
        alpha = __shfl_sync(wmask, alpha, 0);
        beta = __shfl_sync(wmask, beta, 0);

        const signed char* value_row = value_cache + ((unsigned long long)t * row_stride + head_offset);
        const float v_sc = v_scales[t * n_kv_heads + kv_head];
        #pragma unroll
        for (unsigned int k = 0; k < 16; ++k) {
            if (k >= dims_per_lane) break;
            const unsigned int d = lane + (k << 5);
            if (d < head_dim) {
                const float v = (float)(value_row[d]) * v_sc;
                out_acc[k] = out_acc[k] * alpha + v * beta;
            }
        }
    }

    m = __shfl_sync(wmask, m, 0);
    s = __shfl_sync(wmask, s, 0);

    // Cross-warp merge via shared memory.
    extern __shared__ float smem[];
    float* smem_m = smem;
    float* smem_s = smem + num_warps;
    float* smem_out = smem + 2u * num_warps;

    if (lane == 0) {
        smem_m[warp_id] = m;
        smem_s[warp_id] = s;
    }
    #pragma unroll
    for (unsigned int k = 0; k < 16; ++k) {
        if (k >= dims_per_lane) break;
        const unsigned int d = lane + (k << 5);
        if (d < head_dim) {
            smem_out[warp_id * head_dim + d] = out_acc[k];
        }
    }
    __syncthreads();

    if (warp_id != 0) return;

    m = smem_m[0];
    s = smem_s[0];
    #pragma unroll
    for (unsigned int k = 0; k < 16; ++k) {
        if (k >= dims_per_lane) break;
        const unsigned int d = lane + (k << 5);
        if (d < head_dim) {
            out_acc[k] = smem_out[d];
        }
    }

    for (unsigned int w = 1; w < num_warps; ++w) {
        const float mw = smem_m[w];
        const float sw = smem_s[w];
        const float m_new = fmaxf(m, mw);
        const float alpha = expf(m - m_new);
        const float beta = expf(mw - m_new);
        s = s * alpha + sw * beta;
        m = m_new;

        #pragma unroll
        for (unsigned int k = 0; k < 16; ++k) {
            if (k >= dims_per_lane) break;
            const unsigned int d = lane + (k << 5);
            if (d < head_dim) {
                out_acc[k] = out_acc[k] * alpha + smem_out[w * head_dim + d] * beta;
            }
        }
    }

    const float inv_s = 1.0f / fmaxf(s, 1.0e-20f);
    #pragma unroll
    for (unsigned int k = 0; k < 16; ++k) {
        if (k >= dims_per_lane) break;
        const unsigned int d = lane + (k << 5);
        if (d < head_dim) {
            float result = out_acc[k] * inv_s;
            if (gate_proj) {
                const float gate = gate_proj[(unsigned long long)row * gate_proj_stride + (unsigned long long)head * head_dim * 2u + head_dim + d];
                result *= 1.0f / (1.0f + expf(-gate));
            }
            out_head[d] = result;
        }
    }
}

// Fused prefill attention with INT8 KV (single-warp online softmax).
// Grid: (n_heads, q_rows), Block: (32,)
extern "C" __global__ void talu_attn_fused_prefill_heads_i8_kv(
    float* out,
    const float* query,
    const signed char* key_cache,
    const signed char* value_cache,
    const float* k_scales,
    const float* v_scales,
    unsigned int n_heads,
    unsigned int n_kv_heads,
    unsigned int q_rows,
    unsigned int seq_len,
    unsigned int row_stride,
    unsigned int kv_groups,
    unsigned int head_dim,
    float scale,
    unsigned int rope_dim,
    unsigned int position_base,
    unsigned int sliding_window,
    float theta
) {
    const unsigned int head = blockIdx.x;
    const unsigned int q_idx = blockIdx.y;
    const unsigned int lane = threadIdx.x & 31u;
    if (head >= n_heads || q_idx >= q_rows || kv_groups == 0 || seq_len == 0 || head_dim == 0) return;
    if (rope_dim == 0 || rope_dim > head_dim || (rope_dim & 1u) != 0u) return;

    const unsigned int kv_head = head / kv_groups;
    const unsigned int head_offset = kv_head * head_dim;
    const unsigned int mask = 0xFFFFffffu;
    const unsigned int query_pos = position_base + q_idx;
    const unsigned int effective_seq = min(seq_len, query_pos + 1u);
    const unsigned int start_t = (sliding_window > 0u && effective_seq > sliding_window) ? (effective_seq - sliding_window) : 0u;
    const float* query_head = query + ((unsigned long long)q_idx * n_heads + head) * head_dim;
    float* out_head = out + ((unsigned long long)q_idx * n_heads + head) * head_dim;
    const float log2_theta = log2f(theta);

    const unsigned int dims_per_lane = (head_dim + 31u) >> 5;
    float out_acc[16];
    float q_rot[16];
    #pragma unroll
    for (unsigned int i = 0; i < 16; ++i) {
        out_acc[i] = 0.0f;
        q_rot[i] = 0.0f;
    }
    if (dims_per_lane > 16u) return;

    const unsigned int half_rope = rope_dim >> 1;
    #pragma unroll
    for (unsigned int k = 0; k < 16; ++k) {
        if (k >= dims_per_lane) break;
        const unsigned int d = lane + (k << 5);
        if (d >= head_dim) continue;

        float qv = query_head[d];
        if (d < rope_dim) {
            const unsigned int pair = (d < half_rope) ? d : (d - half_rope);
            const float q_lo = query_head[pair];
            const float q_hi = query_head[half_rope + pair];
            const float inv_freq = exp2f(log2_theta * (-2.0f * (float)pair / (float)rope_dim));
            const float angle = (float)query_pos * inv_freq;
            float sn = 0.0f;
            float cn = 0.0f;
            __sincosf(angle, &sn, &cn);
            qv = (d < half_rope) ? fmaf(q_lo, cn, -q_hi * sn) : fmaf(q_lo, sn, q_hi * cn);
        }
        q_rot[k] = qv;
    }

    float m = -3.402823466e+38f;
    float s = 0.0f;
    for (unsigned int t = start_t; t < effective_seq; ++t) {
        const signed char* key_row = key_cache + ((unsigned long long)t * row_stride + head_offset);
        const float k_sc = k_scales[t * n_kv_heads + kv_head];
        float partial = 0.0f;
        #pragma unroll
        for (unsigned int k = 0; k < 16; ++k) {
            if (k >= dims_per_lane) break;
            const unsigned int d = lane + (k << 5);
            if (d >= head_dim) continue;
            partial += q_rot[k] * (float)(key_row[d]);
        }
        for (unsigned int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_down_sync(mask, partial, offset);
        }
        const float dot = __shfl_sync(mask, partial, 0) * k_sc;
        const float score = dot * scale;

        float alpha = 0.0f;
        float beta = 0.0f;
        if (lane == 0) {
            const float m_new = fmaxf(m, score);
            alpha = expf(m - m_new);
            beta = expf(score - m_new);
            s = s * alpha + beta;
            m = m_new;
        }
        alpha = __shfl_sync(mask, alpha, 0);
        beta = __shfl_sync(mask, beta, 0);

        const signed char* value_row = value_cache + ((unsigned long long)t * row_stride + head_offset);
        const float v_sc = v_scales[t * n_kv_heads + kv_head];
        #pragma unroll
        for (unsigned int k = 0; k < 16; ++k) {
            if (k >= dims_per_lane) break;
            const unsigned int d = lane + (k << 5);
            if (d < head_dim) {
                const float v = (float)(value_row[d]) * v_sc;
                out_acc[k] = out_acc[k] * alpha + v * beta;
            }
        }
    }

    s = __shfl_sync(mask, s, 0);
    const float inv_s = 1.0f / fmaxf(s, 1.0e-20f);
    #pragma unroll
    for (unsigned int k = 0; k < 16; ++k) {
        if (k >= dims_per_lane) break;
        const unsigned int d = lane + (k << 5);
        if (d < head_dim) {
            out_head[d] = out_acc[k] * inv_s;
        }
    }
}

// GQA-aware fused prefill with INT8 KV. Loads KV tiles into shared memory.
// Grid: (n_kv_heads, q_rows). Block: kv_groups * 32.
// Dynamic shared memory: 2 * GQA_KV_TILE * head_dim * sizeof(signed char) + 2 * GQA_KV_TILE * sizeof(float).
#define GQA_I8_KV_TILE 32u

extern "C" __global__ void talu_attn_fused_prefill_heads_i8_kv_gqa(
    float* __restrict__ out,
    const float* __restrict__ query,
    const signed char* __restrict__ key_cache,
    const signed char* __restrict__ value_cache,
    const float* __restrict__ k_scales,
    const float* __restrict__ v_scales,
    unsigned int n_heads,
    unsigned int n_kv_heads,
    unsigned int q_rows,
    unsigned int seq_len,
    unsigned int row_stride,
    unsigned int kv_groups,
    unsigned int head_dim,
    float scale,
    unsigned int rope_dim,
    unsigned int position_base,
    unsigned int sliding_window,
    float theta
) {
    const unsigned int kv_head = blockIdx.x;
    const unsigned int q_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / TALU_ATTN_WARP_SIZE;
    const unsigned int lane = tid & (TALU_ATTN_WARP_SIZE - 1u);
    const unsigned int mask = 0xFFFFffffu;

    if (kv_head >= n_kv_heads || q_idx >= q_rows || warp_id >= kv_groups) return;
    if (kv_groups == 0 || seq_len == 0 || head_dim == 0) return;
    if (rope_dim == 0 || rope_dim > head_dim || (rope_dim & 1u) != 0u) return;

    const unsigned int head = kv_head * kv_groups + warp_id;
    if (head >= n_heads) return;

    const unsigned int query_pos = position_base + q_idx;
    const unsigned int effective_seq = min(seq_len, query_pos + 1u);
    const unsigned int start_t = (sliding_window > 0u && effective_seq > sliding_window)
                                 ? (effective_seq - sliding_window) : 0u;
    const unsigned int head_offset = kv_head * head_dim;
    const float* query_head = query + ((unsigned long long)q_idx * n_heads + head) * head_dim;
    float* out_head = out + ((unsigned long long)q_idx * n_heads + head) * head_dim;
    const float log2_theta = log2f(theta);

    const unsigned int dims_per_lane = (head_dim + 31u) >> 5;
    float out_acc[16];
    float q_rot[16];
    #pragma unroll
    for (unsigned int i = 0; i < 16; ++i) {
        out_acc[i] = 0.0f;
        q_rot[i] = 0.0f;
    }
    if (dims_per_lane > 16u) return;

    const unsigned int half_rope = rope_dim >> 1;
    #pragma unroll
    for (unsigned int k = 0; k < 16; ++k) {
        if (k >= dims_per_lane) break;
        const unsigned int d = lane + (k << 5);
        if (d >= head_dim) continue;

        float qv = query_head[d];
        if (d < rope_dim) {
            const unsigned int pair = (d < half_rope) ? d : (d - half_rope);
            const float q_lo = query_head[pair];
            const float q_hi = query_head[half_rope + pair];
            const float inv_freq = exp2f(log2_theta * (-2.0f * (float)pair / (float)rope_dim));
            const float angle = (float)query_pos * inv_freq;
            float sin_v = 0.0f;
            float cos_v = 0.0f;
            __sincosf(angle, &sin_v, &cos_v);
            qv = (d < half_rope) ? fmaf(q_lo, cos_v, -q_hi * sin_v)
                                 : fmaf(q_lo, sin_v, q_hi * cos_v);
        }
        q_rot[k] = qv;
    }

    // Shared memory layout: [GQA_I8_KV_TILE * head_dim] i8 K, [GQA_I8_KV_TILE * head_dim] i8 V,
    // [GQA_I8_KV_TILE] k_scales, [GQA_I8_KV_TILE] v_scales.
    extern __shared__ char smem_raw[];
    signed char* k_smem = reinterpret_cast<signed char*>(smem_raw);
    signed char* v_smem = k_smem + GQA_I8_KV_TILE * head_dim;
    float* k_sc_smem = reinterpret_cast<float*>(v_smem + GQA_I8_KV_TILE * head_dim);
    float* v_sc_smem = k_sc_smem + GQA_I8_KV_TILE;

    const unsigned int total_threads = blockDim.x;
    float m_val = -3.402823466e+38f;
    float acc_sum = 0.0f;

    for (unsigned int tile_start = start_t; tile_start < effective_seq; tile_start += GQA_I8_KV_TILE) {
        const unsigned int tile_end = min(tile_start + GQA_I8_KV_TILE, effective_seq);
        const unsigned int tile_size = tile_end - tile_start;

        // Cooperative load of KV tile + scales into shared memory.
        const unsigned int total_elems = tile_size * head_dim;
        for (unsigned int idx = tid; idx < total_elems; idx += total_threads) {
            const unsigned int t_local = idx / head_dim;
            const unsigned int d = idx - t_local * head_dim;
            const unsigned long long cache_idx =
                (unsigned long long)(tile_start + t_local) * row_stride + head_offset + d;
            k_smem[t_local * head_dim + d] = key_cache[cache_idx];
            v_smem[t_local * head_dim + d] = value_cache[cache_idx];
        }
        // Load scales (one per token per kv_head).
        for (unsigned int t_local = tid; t_local < tile_size; t_local += total_threads) {
            const unsigned int t_abs = tile_start + t_local;
            k_sc_smem[t_local] = k_scales[t_abs * n_kv_heads + kv_head];
            v_sc_smem[t_local] = v_scales[t_abs * n_kv_heads + kv_head];
        }
        __syncthreads();

        for (unsigned int t_local = 0; t_local < tile_size; ++t_local) {
            const unsigned int smem_row = t_local * head_dim;
            const float k_sc = k_sc_smem[t_local];

            float partial = 0.0f;
            #pragma unroll
            for (unsigned int k = 0; k < 16; ++k) {
                if (k >= dims_per_lane) break;
                const unsigned int d = lane + (k << 5);
                if (d < head_dim) {
                    partial += q_rot[k] * (float)(k_smem[smem_row + d]);
                }
            }
            for (unsigned int offset = 16; offset > 0; offset >>= 1) {
                partial += __shfl_down_sync(mask, partial, offset);
            }
            const float dot = __shfl_sync(mask, partial, 0) * k_sc;
            const float score = dot * scale;

            float alpha = 0.0f;
            float beta = 0.0f;
            if (lane == 0) {
                const float m_new = fmaxf(m_val, score);
                alpha = expf(m_val - m_new);
                beta = expf(score - m_new);
                acc_sum = acc_sum * alpha + beta;
                m_val = m_new;
            }
            alpha = __shfl_sync(mask, alpha, 0);
            beta = __shfl_sync(mask, beta, 0);

            const float v_sc = v_sc_smem[t_local];
            #pragma unroll
            for (unsigned int k = 0; k < 16; ++k) {
                if (k >= dims_per_lane) break;
                const unsigned int d = lane + (k << 5);
                if (d < head_dim) {
                    const float v = (float)(v_smem[smem_row + d]) * v_sc;
                    out_acc[k] = out_acc[k] * alpha + v * beta;
                }
            }
        }
        __syncthreads();
    }

    acc_sum = __shfl_sync(mask, acc_sum, 0);
    const float inv_sum = 1.0f / fmaxf(acc_sum, 1.0e-20f);
    #pragma unroll
    for (unsigned int k = 0; k < 16; ++k) {
        if (k >= dims_per_lane) break;
        const unsigned int d = lane + (k << 5);
        if (d < head_dim) {
            out_head[d] = out_acc[k] * inv_sum;
        }
    }
}

// Batched separate attention scores with INT8 K cache + pointer tables.
// Grid: (ceil(max_seq_len / warps_per_block), n_heads, batch_rows), Block: 128
extern "C" __global__ void talu_attn_scores_heads_i8_kv_ptrs(
    float* scores,
    const float* query,
    const unsigned long long* key_cache_ptrs,
    const unsigned long long* k_scale_ptrs,
    const unsigned int* seq_lens,
    const unsigned int* positions,
    unsigned int n_heads,
    unsigned int n_kv_heads,
    unsigned int max_seq_len,
    unsigned int row_stride,
    unsigned int kv_groups,
    unsigned int head_dim,
    float scale,
    unsigned int sliding_window
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid / TALU_ATTN_WARP_SIZE;
    const unsigned int lane = tid & (TALU_ATTN_WARP_SIZE - 1u);
    const unsigned int warps_per_block = blockDim.x / TALU_ATTN_WARP_SIZE;
    const unsigned int token_index = blockIdx.x * warps_per_block + warp;
    const unsigned int head = blockIdx.y;
    const unsigned int row = blockIdx.z;
    if (head >= n_heads || kv_groups == 0) return;

    const unsigned int seq_len = seq_lens[row];
    const unsigned int position = positions[row];
    const unsigned int effective_seq = min(seq_len, position + 1u);
    const unsigned int start_t = (sliding_window > 0u && effective_seq > sliding_window)
        ? (effective_seq - sliding_window) : 0u;
    const unsigned int visible_len = effective_seq - start_t;

    float* scores_head = scores + ((unsigned long long)row * n_heads + head) * max_seq_len;

    if (token_index >= visible_len) {
        if (token_index < max_seq_len && lane == 0) {
            scores_head[token_index] = -3.402823466e+38f;
        }
        return;
    }

    const unsigned int kv_head = head / kv_groups;
    const unsigned int head_offset = kv_head * head_dim;
    const float* query_head = query + ((unsigned long long)row * n_heads + head) * head_dim;
    const signed char* key_cache = reinterpret_cast<const signed char*>(key_cache_ptrs[row]);
    const float* k_scales = reinterpret_cast<const float*>(k_scale_ptrs[row]);
    const unsigned int actual_t = start_t + token_index;
    const signed char* key_row = key_cache + ((unsigned long long)actual_t * row_stride + head_offset);
    const float k_sc = k_scales[actual_t * n_kv_heads + kv_head];

    float partial = 0.0f;
    for (unsigned int d = lane; d < head_dim; d += TALU_ATTN_WARP_SIZE) {
        partial += query_head[d] * (float)(key_row[d]);
    }
    const float dot = talu_attn_warp_sum_f32(partial) * k_sc;
    if (lane == 0) {
        scores_head[token_index] = dot * scale;
    }
}

// Batched separate weighted sum with INT8 V cache + pointer tables.
// Grid: (ceil(head_dim / warps_per_block), n_heads, batch_rows), Block: 128
extern "C" __global__ void talu_attn_weighted_sum_heads_i8_kv_ptrs(
    float* out,
    const float* probs,
    const unsigned long long* value_cache_ptrs,
    const unsigned long long* v_scale_ptrs,
    const unsigned int* seq_lens,
    const unsigned int* positions,
    unsigned int n_heads,
    unsigned int n_kv_heads,
    unsigned int max_seq_len,
    unsigned int row_stride,
    unsigned int kv_groups,
    unsigned int head_dim,
    unsigned int sliding_window
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid / TALU_ATTN_WARP_SIZE;
    const unsigned int lane = tid & (TALU_ATTN_WARP_SIZE - 1u);
    const unsigned int warps_per_block = blockDim.x / TALU_ATTN_WARP_SIZE;
    const unsigned int d = blockIdx.x * warps_per_block + warp;
    const unsigned int head = blockIdx.y;
    const unsigned int row = blockIdx.z;
    if (head >= n_heads || d >= head_dim || kv_groups == 0) return;

    const unsigned int seq_len = seq_lens[row];
    const unsigned int position = positions[row];
    const unsigned int effective_seq = min(seq_len, position + 1u);
    const unsigned int start_t = (sliding_window > 0u && effective_seq > sliding_window)
        ? (effective_seq - sliding_window) : 0u;
    const unsigned int visible_len = effective_seq - start_t;

    const unsigned int kv_head = head / kv_groups;
    const unsigned int head_offset = kv_head * head_dim;
    const float* probs_row = probs + ((unsigned long long)row * n_heads + head) * max_seq_len;
    const signed char* value_cache = reinterpret_cast<const signed char*>(value_cache_ptrs[row]);
    const float* v_scales = reinterpret_cast<const float*>(v_scale_ptrs[row]);

    float partial = 0.0f;
    for (unsigned int t = lane; t < visible_len; t += TALU_ATTN_WARP_SIZE) {
        const unsigned int actual_t = start_t + t;
        const float v_sc = v_scales[actual_t * n_kv_heads + kv_head];
        const unsigned long long value_index = (unsigned long long)actual_t * row_stride + head_offset + d;
        partial += probs_row[t] * v_sc * (float)(value_cache[value_index]);
    }
    const float acc = talu_attn_warp_sum_f32(partial);
    if (lane == 0) {
        out[((unsigned long long)row * n_heads + head) * head_dim + d] = acc;
    }
}

// --- FP8 E4M3 KV cache attention kernels ---
// Same structure as INT8 variants but dequantizes via FP8 E4M3 → half → float.
// Per-head scale: dequantized = fp8_to_f32(byte) * scale.
// Requires sm_89+ for __nv_cvt_fp8_to_halfraw intrinsic.

#if __CUDA_ARCH__ >= 890

static __device__ __forceinline__ float talu_attn_fp8e4m3_to_f32(__nv_fp8_storage_t x) {
    __half_raw hr = __nv_cvt_fp8_to_halfraw(x, __NV_E4M3);
    __half h;
    memcpy(&h, &hr, sizeof(h));
    return __half2float(h);
}

// Separate attention scores with FP8 K cache.
// Grid: (ceil(seq_len / warps_per_block), n_heads), Block: 128
extern "C" __global__ void talu_attn_scores_heads_fp8_kv(
    float* scores,
    const float* query,
    const unsigned char* key_cache,
    const float* k_scales,
    unsigned int n_heads,
    unsigned int n_kv_heads,
    unsigned int seq_len,
    unsigned int row_stride,
    unsigned int kv_groups,
    unsigned int head_dim,
    float scale
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid / TALU_ATTN_WARP_SIZE;
    const unsigned int lane = tid & (TALU_ATTN_WARP_SIZE - 1u);
    const unsigned int warps_per_block = blockDim.x / TALU_ATTN_WARP_SIZE;
    const unsigned int token_index = blockIdx.x * warps_per_block + warp;
    const unsigned int head = blockIdx.y;
    if (head >= n_heads || token_index >= seq_len) return;
    if (kv_groups == 0) return;

    const unsigned int kv_head = head / kv_groups;
    const unsigned int head_offset = kv_head * head_dim;
    const float* query_head = query + (unsigned long long)head * head_dim;
    const unsigned char* key_row = key_cache + ((unsigned long long)token_index * row_stride + head_offset);
    const float k_scale = k_scales[token_index * n_kv_heads + kv_head];

    float partial = 0.0f;
    for (unsigned int d = lane; d < head_dim; d += TALU_ATTN_WARP_SIZE) {
        partial += query_head[d] * talu_attn_fp8e4m3_to_f32(static_cast<__nv_fp8_storage_t>(key_row[d]));
    }
    const float dot = talu_attn_warp_sum_f32(partial) * k_scale;
    if (lane == 0) {
        scores[(unsigned long long)head * seq_len + token_index] = dot * scale;
    }
}

// Separate weighted sum with FP8 V cache.
// Grid: (ceil(head_dim / warps_per_block), n_heads), Block: 128
extern "C" __global__ void talu_attn_weighted_sum_heads_fp8_kv(
    float* out,
    const float* probs,
    const unsigned char* value_cache,
    const float* v_scales,
    unsigned int n_heads,
    unsigned int n_kv_heads,
    unsigned int seq_len,
    unsigned int row_stride,
    unsigned int kv_groups,
    unsigned int head_dim
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid / TALU_ATTN_WARP_SIZE;
    const unsigned int lane = tid & (TALU_ATTN_WARP_SIZE - 1u);
    const unsigned int warps_per_block = blockDim.x / TALU_ATTN_WARP_SIZE;
    const unsigned int d = blockIdx.x * warps_per_block + warp;
    const unsigned int head = blockIdx.y;
    if (head >= n_heads || d >= head_dim) return;
    if (kv_groups == 0) return;

    const unsigned int kv_head = head / kv_groups;
    const unsigned int head_offset = kv_head * head_dim;
    const float* probs_row = probs + (unsigned long long)head * seq_len;

    float partial = 0.0f;
    for (unsigned int t = lane; t < seq_len; t += TALU_ATTN_WARP_SIZE) {
        const float v_scale = v_scales[t * n_kv_heads + kv_head];
        const unsigned long long value_index = (unsigned long long)t * row_stride + head_offset + d;
        partial += probs_row[t] * v_scale * talu_attn_fp8e4m3_to_f32(static_cast<__nv_fp8_storage_t>(value_cache[value_index]));
    }
    const float acc = talu_attn_warp_sum_f32(partial);
    if (lane == 0) {
        out[(unsigned long long)head * head_dim + d] = acc;
    }
}

// Fused single-token decode attention with FP8 KV (online softmax).
// One warp per head. Grid: (n_heads,), Block: (32,)
extern "C" __global__ void talu_attn_fused_heads_fp8_kv(
    float* out,
    const float* query,
    const unsigned char* key_cache,
    const unsigned char* value_cache,
    const float* k_scales,
    const float* v_scales,
    unsigned int n_heads,
    unsigned int n_kv_heads,
    unsigned int seq_len,
    unsigned int row_stride,
    unsigned int kv_groups,
    unsigned int head_dim,
    float scale,
    unsigned int rope_dim,
    unsigned int position,
    float theta
) {
    const unsigned int head = blockIdx.x;
    const unsigned int lane = threadIdx.x & 31u;
    if (head >= n_heads || kv_groups == 0 || seq_len == 0 || head_dim == 0) return;
    if (rope_dim == 0 || rope_dim > head_dim || (rope_dim & 1u) != 0u) return;

    const unsigned int kv_head = head / kv_groups;
    const unsigned int head_offset = kv_head * head_dim;
    const float* query_head = query + (unsigned long long)head * head_dim;
    const unsigned int mask = 0xFFFFffffu;
    const float log2_theta = log2f(theta);

    const unsigned int dims_per_lane = (head_dim + 31u) >> 5;
    float out_acc[16];
    float q_rot[16];
    #pragma unroll
    for (unsigned int i = 0; i < 16; ++i) {
        out_acc[i] = 0.0f;
        q_rot[i] = 0.0f;
    }
    if (dims_per_lane > 16u) return;

    const unsigned int half = rope_dim >> 1;
    #pragma unroll
    for (unsigned int k = 0; k < 16; ++k) {
        if (k >= dims_per_lane) break;
        const unsigned int d = lane + (k << 5);
        if (d >= head_dim) continue;

        float qv = query_head[d];
        if (d < rope_dim) {
            const unsigned int pair = (d < half) ? d : (d - half);
            const float q_lo = query_head[pair];
            const float q_hi = query_head[half + pair];
            const float inv_freq = exp2f(log2_theta * (-2.0f * (float)pair / (float)rope_dim));
            const float angle = (float)position * inv_freq;
            float s = 0.0f;
            float c = 0.0f;
            __sincosf(angle, &s, &c);
            qv = (d < half) ? fmaf(q_lo, c, -q_hi * s) : fmaf(q_lo, s, q_hi * c);
        }
        q_rot[k] = qv;
    }

    float m = -3.402823466e+38f;
    float s = 0.0f;

    for (unsigned int t = 0; t < seq_len; ++t) {
        const unsigned char* key_row = key_cache + ((unsigned long long)t * row_stride + head_offset);
        const float k_sc = k_scales[t * n_kv_heads + kv_head];
        float partial = 0.0f;
        #pragma unroll
        for (unsigned int k = 0; k < 16; ++k) {
            if (k >= dims_per_lane) break;
            const unsigned int d = lane + (k << 5);
            if (d >= head_dim) continue;
            partial += q_rot[k] * talu_attn_fp8e4m3_to_f32(static_cast<__nv_fp8_storage_t>(key_row[d]));
        }
        for (unsigned int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_down_sync(mask, partial, offset);
        }
        const float dot = __shfl_sync(mask, partial, 0) * k_sc;
        const float score = dot * scale;

        float alpha = 0.0f;
        float beta = 0.0f;
        if (lane == 0) {
            const float m_new = fmaxf(m, score);
            alpha = expf(m - m_new);
            beta = expf(score - m_new);
            s = s * alpha + beta;
            m = m_new;
        }
        alpha = __shfl_sync(mask, alpha, 0);
        beta = __shfl_sync(mask, beta, 0);

        const unsigned char* value_row = value_cache + ((unsigned long long)t * row_stride + head_offset);
        const float v_sc = v_scales[t * n_kv_heads + kv_head];
        #pragma unroll
        for (unsigned int k = 0; k < 16; ++k) {
            if (k >= dims_per_lane) break;
            const unsigned int d = lane + (k << 5);
            if (d < head_dim) {
                const float v = talu_attn_fp8e4m3_to_f32(static_cast<__nv_fp8_storage_t>(value_row[d])) * v_sc;
                out_acc[k] = out_acc[k] * alpha + v * beta;
            }
        }
    }

    s = __shfl_sync(mask, s, 0);
    const float inv_s = 1.0f / fmaxf(s, 1.0e-20f);
    #pragma unroll
    for (unsigned int k = 0; k < 16; ++k) {
        if (k >= dims_per_lane) break;
        const unsigned int d = lane + (k << 5);
        if (d < head_dim) {
            out[(unsigned long long)head * head_dim + d] = out_acc[k] * inv_s;
        }
    }
}

// Fused batched decode attention with FP8 KV + pointer tables.
// Multi-warp FlashDecoding with online softmax merge.
// Grid: (n_heads, batch_rows), Block: (512,)
extern "C" __global__ __launch_bounds__(512)
void talu_attn_fused_decode_heads_fp8_kv_ptrs(
    float* out,
    const float* query,
    const unsigned long long* key_cache_ptrs,
    const unsigned long long* value_cache_ptrs,
    const unsigned long long* k_scale_ptrs,
    const unsigned long long* v_scale_ptrs,
    const unsigned int* seq_lens,
    const unsigned int* positions,
    unsigned int batch_rows,
    unsigned int n_heads,
    unsigned int n_kv_heads,
    unsigned int row_stride,
    unsigned int kv_groups,
    unsigned int head_dim,
    float scale,
    unsigned int rope_dim,
    unsigned int sliding_window,
    float theta,
    const float* gate_proj,
    unsigned int gate_proj_stride
) {
    const unsigned int head = blockIdx.x;
    const unsigned int row = blockIdx.y;
    const unsigned int warp_id = threadIdx.x / 32u;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int num_warps = blockDim.x / 32u;
    if (row >= batch_rows || head >= n_heads || kv_groups == 0 || head_dim == 0) return;
    if (rope_dim == 0 || rope_dim > head_dim || (rope_dim & 1u) != 0u) return;

    const unsigned int seq_len = seq_lens[row];
    if (seq_len == 0) return;
    const unsigned int position = positions[row];
    const unsigned int effective_seq = min(seq_len, position + 1u);
    const unsigned int start_t = (sliding_window > 0u && effective_seq > sliding_window)
        ? (effective_seq - sliding_window)
        : 0u;

    const unsigned char* key_cache = reinterpret_cast<const unsigned char*>(key_cache_ptrs[row]);
    const unsigned char* value_cache = reinterpret_cast<const unsigned char*>(value_cache_ptrs[row]);
    const float* k_scales = reinterpret_cast<const float*>(k_scale_ptrs[row]);
    const float* v_scales = reinterpret_cast<const float*>(v_scale_ptrs[row]);
    if (key_cache == nullptr || value_cache == nullptr) return;

    const unsigned int kv_head = head / kv_groups;
    const unsigned int head_offset = kv_head * head_dim;
    const float* query_head = query + ((unsigned long long)row * n_heads + head) * head_dim;
    float* out_head = out + ((unsigned long long)row * n_heads + head) * head_dim;
    const unsigned int wmask = 0xFFFFffffu;
    const float log2_theta = log2f(theta);

    const unsigned int dims_per_lane = (head_dim + 31u) >> 5;
    float out_acc[16];
    float q_rot[16];
    #pragma unroll
    for (unsigned int i = 0; i < 16; ++i) {
        out_acc[i] = 0.0f;
        q_rot[i] = 0.0f;
    }
    if (dims_per_lane > 16u) return;

    // Q RoPE.
    const unsigned int half = rope_dim >> 1;
    #pragma unroll
    for (unsigned int k = 0; k < 16; ++k) {
        if (k >= dims_per_lane) break;
        const unsigned int d = lane + (k << 5);
        if (d >= head_dim) continue;

        float qv = query_head[d];
        if (d < rope_dim) {
            const unsigned int pair = (d < half) ? d : (d - half);
            const float q_lo = query_head[pair];
            const float q_hi = query_head[half + pair];
            const float inv_freq = exp2f(log2_theta * (-2.0f * (float)pair / (float)rope_dim));
            const float angle = (float)position * inv_freq;
            float sn = 0.0f;
            float cn = 0.0f;
            __sincosf(angle, &sn, &cn);
            qv = (d < half) ? fmaf(q_lo, cn, -q_hi * sn) : fmaf(q_lo, sn, q_hi * cn);
        }
        q_rot[k] = qv;
    }

    // Split tokens across warps.
    const unsigned int total_tokens = effective_seq - start_t;
    const unsigned int tokens_per_warp = (total_tokens + num_warps - 1u) / num_warps;
    const unsigned int my_start = start_t + warp_id * tokens_per_warp;
    const unsigned int my_end = min(my_start + tokens_per_warp, effective_seq);

    float m = -3.402823466e+38f;
    float s = 0.0f;

    for (unsigned int t = my_start; t < my_end; ++t) {
        const unsigned char* key_row = key_cache + ((unsigned long long)t * row_stride + head_offset);
        const float k_sc = k_scales[t * n_kv_heads + kv_head];
        float partial = 0.0f;
        #pragma unroll
        for (unsigned int k = 0; k < 16; ++k) {
            if (k >= dims_per_lane) break;
            const unsigned int d = lane + (k << 5);
            if (d >= head_dim) continue;
            partial += q_rot[k] * talu_attn_fp8e4m3_to_f32(static_cast<__nv_fp8_storage_t>(key_row[d]));
        }
        for (unsigned int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_down_sync(wmask, partial, offset);
        }
        const float dot = __shfl_sync(wmask, partial, 0) * k_sc;
        const float score = dot * scale;

        float alpha = 0.0f;
        float beta = 0.0f;
        if (lane == 0) {
            const float m_new = fmaxf(m, score);
            alpha = expf(m - m_new);
            beta = expf(score - m_new);
            s = s * alpha + beta;
            m = m_new;
        }
        alpha = __shfl_sync(wmask, alpha, 0);
        beta = __shfl_sync(wmask, beta, 0);

        const unsigned char* value_row = value_cache + ((unsigned long long)t * row_stride + head_offset);
        const float v_sc = v_scales[t * n_kv_heads + kv_head];
        #pragma unroll
        for (unsigned int k = 0; k < 16; ++k) {
            if (k >= dims_per_lane) break;
            const unsigned int d = lane + (k << 5);
            if (d < head_dim) {
                const float v = talu_attn_fp8e4m3_to_f32(static_cast<__nv_fp8_storage_t>(value_row[d])) * v_sc;
                out_acc[k] = out_acc[k] * alpha + v * beta;
            }
        }
    }

    m = __shfl_sync(wmask, m, 0);
    s = __shfl_sync(wmask, s, 0);

    // Cross-warp merge via shared memory.
    extern __shared__ float smem[];
    float* smem_m = smem;
    float* smem_s = smem + num_warps;
    float* smem_out = smem + 2u * num_warps;

    if (lane == 0) {
        smem_m[warp_id] = m;
        smem_s[warp_id] = s;
    }
    #pragma unroll
    for (unsigned int k = 0; k < 16; ++k) {
        if (k >= dims_per_lane) break;
        const unsigned int d = lane + (k << 5);
        if (d < head_dim) {
            smem_out[warp_id * head_dim + d] = out_acc[k];
        }
    }
    __syncthreads();

    if (warp_id != 0) return;

    m = smem_m[0];
    s = smem_s[0];
    #pragma unroll
    for (unsigned int k = 0; k < 16; ++k) {
        if (k >= dims_per_lane) break;
        const unsigned int d = lane + (k << 5);
        if (d < head_dim) {
            out_acc[k] = smem_out[d];
        }
    }

    for (unsigned int w = 1; w < num_warps; ++w) {
        const float mw = smem_m[w];
        const float sw = smem_s[w];
        const float m_new = fmaxf(m, mw);
        const float alpha = expf(m - m_new);
        const float beta = expf(mw - m_new);
        s = s * alpha + sw * beta;
        m = m_new;

        #pragma unroll
        for (unsigned int k = 0; k < 16; ++k) {
            if (k >= dims_per_lane) break;
            const unsigned int d = lane + (k << 5);
            if (d < head_dim) {
                out_acc[k] = out_acc[k] * alpha + smem_out[w * head_dim + d] * beta;
            }
        }
    }

    const float inv_s = 1.0f / fmaxf(s, 1.0e-20f);
    #pragma unroll
    for (unsigned int k = 0; k < 16; ++k) {
        if (k >= dims_per_lane) break;
        const unsigned int d = lane + (k << 5);
        if (d < head_dim) {
            float result = out_acc[k] * inv_s;
            if (gate_proj) {
                const float gate = gate_proj[(unsigned long long)row * gate_proj_stride + (unsigned long long)head * head_dim * 2u + head_dim + d];
                result *= 1.0f / (1.0f + expf(-gate));
            }
            out_head[d] = result;
        }
    }
}

// Fused prefill attention with FP8 KV (single-warp online softmax).
// Grid: (n_heads, q_rows), Block: (32,)
extern "C" __global__ void talu_attn_fused_prefill_heads_fp8_kv(
    float* out,
    const float* query,
    const unsigned char* key_cache,
    const unsigned char* value_cache,
    const float* k_scales,
    const float* v_scales,
    unsigned int n_heads,
    unsigned int n_kv_heads,
    unsigned int q_rows,
    unsigned int seq_len,
    unsigned int row_stride,
    unsigned int kv_groups,
    unsigned int head_dim,
    float scale,
    unsigned int rope_dim,
    unsigned int position_base,
    unsigned int sliding_window,
    float theta
) {
    const unsigned int head = blockIdx.x;
    const unsigned int q_idx = blockIdx.y;
    const unsigned int lane = threadIdx.x & 31u;
    if (head >= n_heads || q_idx >= q_rows || kv_groups == 0 || seq_len == 0 || head_dim == 0) return;
    if (rope_dim == 0 || rope_dim > head_dim || (rope_dim & 1u) != 0u) return;

    const unsigned int kv_head = head / kv_groups;
    const unsigned int head_offset = kv_head * head_dim;
    const unsigned int mask = 0xFFFFffffu;
    const unsigned int query_pos = position_base + q_idx;
    const unsigned int effective_seq = min(seq_len, query_pos + 1u);
    const unsigned int start_t = (sliding_window > 0u && effective_seq > sliding_window) ? (effective_seq - sliding_window) : 0u;
    const float* query_head = query + ((unsigned long long)q_idx * n_heads + head) * head_dim;
    float* out_head = out + ((unsigned long long)q_idx * n_heads + head) * head_dim;
    const float log2_theta = log2f(theta);

    const unsigned int dims_per_lane = (head_dim + 31u) >> 5;
    float out_acc[16];
    float q_rot[16];
    #pragma unroll
    for (unsigned int i = 0; i < 16; ++i) {
        out_acc[i] = 0.0f;
        q_rot[i] = 0.0f;
    }
    if (dims_per_lane > 16u) return;

    const unsigned int half_rope = rope_dim >> 1;
    #pragma unroll
    for (unsigned int k = 0; k < 16; ++k) {
        if (k >= dims_per_lane) break;
        const unsigned int d = lane + (k << 5);
        if (d >= head_dim) continue;

        float qv = query_head[d];
        if (d < rope_dim) {
            const unsigned int pair = (d < half_rope) ? d : (d - half_rope);
            const float q_lo = query_head[pair];
            const float q_hi = query_head[half_rope + pair];
            const float inv_freq = exp2f(log2_theta * (-2.0f * (float)pair / (float)rope_dim));
            const float angle = (float)query_pos * inv_freq;
            float sn = 0.0f;
            float cn = 0.0f;
            __sincosf(angle, &sn, &cn);
            qv = (d < half_rope) ? fmaf(q_lo, cn, -q_hi * sn) : fmaf(q_lo, sn, q_hi * cn);
        }
        q_rot[k] = qv;
    }

    float m = -3.402823466e+38f;
    float s = 0.0f;
    for (unsigned int t = start_t; t < effective_seq; ++t) {
        const unsigned char* key_row = key_cache + ((unsigned long long)t * row_stride + head_offset);
        const float k_sc = k_scales[t * n_kv_heads + kv_head];
        float partial = 0.0f;
        #pragma unroll
        for (unsigned int k = 0; k < 16; ++k) {
            if (k >= dims_per_lane) break;
            const unsigned int d = lane + (k << 5);
            if (d >= head_dim) continue;
            partial += q_rot[k] * talu_attn_fp8e4m3_to_f32(static_cast<__nv_fp8_storage_t>(key_row[d]));
        }
        for (unsigned int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_down_sync(mask, partial, offset);
        }
        const float dot = __shfl_sync(mask, partial, 0) * k_sc;
        const float score = dot * scale;

        float alpha = 0.0f;
        float beta = 0.0f;
        if (lane == 0) {
            const float m_new = fmaxf(m, score);
            alpha = expf(m - m_new);
            beta = expf(score - m_new);
            s = s * alpha + beta;
            m = m_new;
        }
        alpha = __shfl_sync(mask, alpha, 0);
        beta = __shfl_sync(mask, beta, 0);

        const unsigned char* value_row = value_cache + ((unsigned long long)t * row_stride + head_offset);
        const float v_sc = v_scales[t * n_kv_heads + kv_head];
        #pragma unroll
        for (unsigned int k = 0; k < 16; ++k) {
            if (k >= dims_per_lane) break;
            const unsigned int d = lane + (k << 5);
            if (d < head_dim) {
                const float v = talu_attn_fp8e4m3_to_f32(static_cast<__nv_fp8_storage_t>(value_row[d])) * v_sc;
                out_acc[k] = out_acc[k] * alpha + v * beta;
            }
        }
    }

    s = __shfl_sync(mask, s, 0);
    const float inv_s = 1.0f / fmaxf(s, 1.0e-20f);
    #pragma unroll
    for (unsigned int k = 0; k < 16; ++k) {
        if (k >= dims_per_lane) break;
        const unsigned int d = lane + (k << 5);
        if (d < head_dim) {
            out_head[d] = out_acc[k] * inv_s;
        }
    }
}

// GQA-aware fused prefill with FP8 KV. Loads KV tiles into shared memory.
// Grid: (n_kv_heads, q_rows). Block: kv_groups * 32.
// Dynamic shared memory: 2 * GQA_FP8_KV_TILE * head_dim + 2 * GQA_FP8_KV_TILE * sizeof(float).
#define GQA_FP8_KV_TILE 32u

extern "C" __global__ void talu_attn_fused_prefill_heads_fp8_kv_gqa(
    float* __restrict__ out,
    const float* __restrict__ query,
    const unsigned char* __restrict__ key_cache,
    const unsigned char* __restrict__ value_cache,
    const float* __restrict__ k_scales,
    const float* __restrict__ v_scales,
    unsigned int n_heads,
    unsigned int n_kv_heads,
    unsigned int q_rows,
    unsigned int seq_len,
    unsigned int row_stride,
    unsigned int kv_groups,
    unsigned int head_dim,
    float scale,
    unsigned int rope_dim,
    unsigned int position_base,
    unsigned int sliding_window,
    float theta
) {
    const unsigned int kv_head = blockIdx.x;
    const unsigned int q_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / TALU_ATTN_WARP_SIZE;
    const unsigned int lane = tid & (TALU_ATTN_WARP_SIZE - 1u);
    const unsigned int mask = 0xFFFFffffu;

    if (kv_head >= n_kv_heads || q_idx >= q_rows || warp_id >= kv_groups) return;
    if (kv_groups == 0 || seq_len == 0 || head_dim == 0) return;
    if (rope_dim == 0 || rope_dim > head_dim || (rope_dim & 1u) != 0u) return;

    const unsigned int head = kv_head * kv_groups + warp_id;
    if (head >= n_heads) return;

    const unsigned int query_pos = position_base + q_idx;
    const unsigned int effective_seq = min(seq_len, query_pos + 1u);
    const unsigned int start_t = (sliding_window > 0u && effective_seq > sliding_window)
                                 ? (effective_seq - sliding_window) : 0u;
    const unsigned int head_offset = kv_head * head_dim;
    const float* query_head = query + ((unsigned long long)q_idx * n_heads + head) * head_dim;
    float* out_head = out + ((unsigned long long)q_idx * n_heads + head) * head_dim;
    const float log2_theta = log2f(theta);

    const unsigned int dims_per_lane = (head_dim + 31u) >> 5;
    float out_acc[16];
    float q_rot[16];
    #pragma unroll
    for (unsigned int i = 0; i < 16; ++i) {
        out_acc[i] = 0.0f;
        q_rot[i] = 0.0f;
    }
    if (dims_per_lane > 16u) return;

    const unsigned int half_rope = rope_dim >> 1;
    #pragma unroll
    for (unsigned int k = 0; k < 16; ++k) {
        if (k >= dims_per_lane) break;
        const unsigned int d = lane + (k << 5);
        if (d >= head_dim) continue;

        float qv = query_head[d];
        if (d < rope_dim) {
            const unsigned int pair = (d < half_rope) ? d : (d - half_rope);
            const float q_lo = query_head[pair];
            const float q_hi = query_head[half_rope + pair];
            const float inv_freq = exp2f(log2_theta * (-2.0f * (float)pair / (float)rope_dim));
            const float angle = (float)query_pos * inv_freq;
            float sin_v = 0.0f;
            float cos_v = 0.0f;
            __sincosf(angle, &sin_v, &cos_v);
            qv = (d < half_rope) ? fmaf(q_lo, cos_v, -q_hi * sin_v)
                                 : fmaf(q_lo, sin_v, q_hi * cos_v);
        }
        q_rot[k] = qv;
    }

    // Shared memory layout: [GQA_FP8_KV_TILE * head_dim] u8 K, [GQA_FP8_KV_TILE * head_dim] u8 V,
    // [GQA_FP8_KV_TILE] k_scales, [GQA_FP8_KV_TILE] v_scales.
    extern __shared__ char smem_raw[];
    unsigned char* k_smem = reinterpret_cast<unsigned char*>(smem_raw);
    unsigned char* v_smem = k_smem + GQA_FP8_KV_TILE * head_dim;
    float* k_sc_smem = reinterpret_cast<float*>(v_smem + GQA_FP8_KV_TILE * head_dim);
    float* v_sc_smem = k_sc_smem + GQA_FP8_KV_TILE;

    const unsigned int total_threads = blockDim.x;
    float m_val = -3.402823466e+38f;
    float acc_sum = 0.0f;

    for (unsigned int tile_start = start_t; tile_start < effective_seq; tile_start += GQA_FP8_KV_TILE) {
        const unsigned int tile_end = min(tile_start + GQA_FP8_KV_TILE, effective_seq);
        const unsigned int tile_size = tile_end - tile_start;

        // Cooperative load of KV tile + scales into shared memory.
        const unsigned int total_elems = tile_size * head_dim;
        for (unsigned int idx = tid; idx < total_elems; idx += total_threads) {
            const unsigned int t_local = idx / head_dim;
            const unsigned int d = idx - t_local * head_dim;
            const unsigned long long cache_idx =
                (unsigned long long)(tile_start + t_local) * row_stride + head_offset + d;
            k_smem[t_local * head_dim + d] = key_cache[cache_idx];
            v_smem[t_local * head_dim + d] = value_cache[cache_idx];
        }
        for (unsigned int t_local = tid; t_local < tile_size; t_local += total_threads) {
            const unsigned int t_abs = tile_start + t_local;
            k_sc_smem[t_local] = k_scales[t_abs * n_kv_heads + kv_head];
            v_sc_smem[t_local] = v_scales[t_abs * n_kv_heads + kv_head];
        }
        __syncthreads();

        for (unsigned int t_local = 0; t_local < tile_size; ++t_local) {
            const unsigned int smem_row = t_local * head_dim;
            const float k_sc = k_sc_smem[t_local];

            float partial = 0.0f;
            #pragma unroll
            for (unsigned int k = 0; k < 16; ++k) {
                if (k >= dims_per_lane) break;
                const unsigned int d = lane + (k << 5);
                if (d < head_dim) {
                    partial += q_rot[k] * talu_attn_fp8e4m3_to_f32(static_cast<__nv_fp8_storage_t>(k_smem[smem_row + d]));
                }
            }
            for (unsigned int offset = 16; offset > 0; offset >>= 1) {
                partial += __shfl_down_sync(mask, partial, offset);
            }
            const float dot = __shfl_sync(mask, partial, 0) * k_sc;
            const float score = dot * scale;

            float alpha = 0.0f;
            float beta = 0.0f;
            if (lane == 0) {
                const float m_new = fmaxf(m_val, score);
                alpha = expf(m_val - m_new);
                beta = expf(score - m_new);
                acc_sum = acc_sum * alpha + beta;
                m_val = m_new;
            }
            alpha = __shfl_sync(mask, alpha, 0);
            beta = __shfl_sync(mask, beta, 0);

            const float v_sc = v_sc_smem[t_local];
            #pragma unroll
            for (unsigned int k = 0; k < 16; ++k) {
                if (k >= dims_per_lane) break;
                const unsigned int d = lane + (k << 5);
                if (d < head_dim) {
                    const float v = talu_attn_fp8e4m3_to_f32(static_cast<__nv_fp8_storage_t>(v_smem[smem_row + d])) * v_sc;
                    out_acc[k] = out_acc[k] * alpha + v * beta;
                }
            }
        }
        __syncthreads();
    }

    acc_sum = __shfl_sync(mask, acc_sum, 0);
    const float inv_sum = 1.0f / fmaxf(acc_sum, 1.0e-20f);
    #pragma unroll
    for (unsigned int k = 0; k < 16; ++k) {
        if (k >= dims_per_lane) break;
        const unsigned int d = lane + (k << 5);
        if (d < head_dim) {
            out_head[d] = out_acc[k] * inv_sum;
        }
    }
}

// Batched separate attention scores with FP8 K cache + pointer tables.
// Grid: (ceil(max_seq_len / warps_per_block), n_heads, batch_rows), Block: 128
extern "C" __global__ void talu_attn_scores_heads_fp8_kv_ptrs(
    float* scores,
    const float* query,
    const unsigned long long* key_cache_ptrs,
    const unsigned long long* k_scale_ptrs,
    const unsigned int* seq_lens,
    const unsigned int* positions,
    unsigned int n_heads,
    unsigned int n_kv_heads,
    unsigned int max_seq_len,
    unsigned int row_stride,
    unsigned int kv_groups,
    unsigned int head_dim,
    float scale,
    unsigned int sliding_window
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid / TALU_ATTN_WARP_SIZE;
    const unsigned int lane = tid & (TALU_ATTN_WARP_SIZE - 1u);
    const unsigned int warps_per_block = blockDim.x / TALU_ATTN_WARP_SIZE;
    const unsigned int token_index = blockIdx.x * warps_per_block + warp;
    const unsigned int head = blockIdx.y;
    const unsigned int row = blockIdx.z;
    if (head >= n_heads || kv_groups == 0) return;

    const unsigned int seq_len = seq_lens[row];
    const unsigned int position = positions[row];
    const unsigned int effective_seq = min(seq_len, position + 1u);
    const unsigned int start_t = (sliding_window > 0u && effective_seq > sliding_window)
        ? (effective_seq - sliding_window) : 0u;
    const unsigned int visible_len = effective_seq - start_t;

    float* scores_head = scores + ((unsigned long long)row * n_heads + head) * max_seq_len;

    if (token_index >= visible_len) {
        if (token_index < max_seq_len && lane == 0) {
            scores_head[token_index] = -3.402823466e+38f;
        }
        return;
    }

    const unsigned int kv_head = head / kv_groups;
    const unsigned int head_offset = kv_head * head_dim;
    const float* query_head = query + ((unsigned long long)row * n_heads + head) * head_dim;
    const unsigned char* key_cache = reinterpret_cast<const unsigned char*>(key_cache_ptrs[row]);
    const float* k_scales = reinterpret_cast<const float*>(k_scale_ptrs[row]);
    const unsigned int actual_t = start_t + token_index;
    const unsigned char* key_row = key_cache + ((unsigned long long)actual_t * row_stride + head_offset);
    const float k_sc = k_scales[actual_t * n_kv_heads + kv_head];

    float partial = 0.0f;
    for (unsigned int d = lane; d < head_dim; d += TALU_ATTN_WARP_SIZE) {
        partial += query_head[d] * talu_attn_fp8e4m3_to_f32(static_cast<__nv_fp8_storage_t>(key_row[d]));
    }
    const float dot = talu_attn_warp_sum_f32(partial) * k_sc;
    if (lane == 0) {
        scores_head[token_index] = dot * scale;
    }
}

// Batched separate weighted sum with FP8 V cache + pointer tables.
// Grid: (ceil(head_dim / warps_per_block), n_heads, batch_rows), Block: 128
extern "C" __global__ void talu_attn_weighted_sum_heads_fp8_kv_ptrs(
    float* out,
    const float* probs,
    const unsigned long long* value_cache_ptrs,
    const unsigned long long* v_scale_ptrs,
    const unsigned int* seq_lens,
    const unsigned int* positions,
    unsigned int n_heads,
    unsigned int n_kv_heads,
    unsigned int max_seq_len,
    unsigned int row_stride,
    unsigned int kv_groups,
    unsigned int head_dim,
    unsigned int sliding_window
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid / TALU_ATTN_WARP_SIZE;
    const unsigned int lane = tid & (TALU_ATTN_WARP_SIZE - 1u);
    const unsigned int warps_per_block = blockDim.x / TALU_ATTN_WARP_SIZE;
    const unsigned int d = blockIdx.x * warps_per_block + warp;
    const unsigned int head = blockIdx.y;
    const unsigned int row = blockIdx.z;
    if (head >= n_heads || d >= head_dim || kv_groups == 0) return;

    const unsigned int seq_len = seq_lens[row];
    const unsigned int position = positions[row];
    const unsigned int effective_seq = min(seq_len, position + 1u);
    const unsigned int start_t = (sliding_window > 0u && effective_seq > sliding_window)
        ? (effective_seq - sliding_window) : 0u;
    const unsigned int visible_len = effective_seq - start_t;

    const unsigned int kv_head = head / kv_groups;
    const unsigned int head_offset = kv_head * head_dim;
    const float* probs_row = probs + ((unsigned long long)row * n_heads + head) * max_seq_len;
    const unsigned char* value_cache = reinterpret_cast<const unsigned char*>(value_cache_ptrs[row]);
    const float* v_scales = reinterpret_cast<const float*>(v_scale_ptrs[row]);

    float partial = 0.0f;
    for (unsigned int t = lane; t < visible_len; t += TALU_ATTN_WARP_SIZE) {
        const unsigned int actual_t = start_t + t;
        const float v_sc = v_scales[actual_t * n_kv_heads + kv_head];
        const unsigned long long value_index = (unsigned long long)actual_t * row_stride + head_offset + d;
        partial += probs_row[t] * v_sc * talu_attn_fp8e4m3_to_f32(static_cast<__nv_fp8_storage_t>(value_cache[value_index]));
    }
    const float acc = talu_attn_warp_sum_f32(partial);
    if (lane == 0) {
        out[((unsigned long long)row * n_heads + head) * head_dim + d] = acc;
    }
}

#endif // __CUDA_ARCH__ >= 890
