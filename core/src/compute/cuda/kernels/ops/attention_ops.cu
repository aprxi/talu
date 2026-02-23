extern "C" __global__ void talu_attn_scores_f32(
    float* scores,
    const float* query_head,
    const float* key_cache,
    unsigned int seq_len,
    unsigned int row_stride,
    unsigned int head_offset,
    unsigned int head_dim,
    float scale
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= seq_len) return;

    const float* key_row = key_cache + ((unsigned long long)index * row_stride + head_offset);
    float dot = 0.0f;
    for (unsigned int d = 0; d < head_dim; ++d) {
        dot += query_head[d] * key_row[d];
    }
    scores[index] = dot * scale;
}

extern "C" __global__ void talu_attn_scores_f16_kv(
    float* scores,
    const float* query_head,
    const unsigned short* key_cache,
    unsigned int seq_len,
    unsigned int row_stride,
    unsigned int head_offset,
    unsigned int head_dim,
    float scale
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= seq_len) return;

    const unsigned short* key_row = key_cache + ((unsigned long long)index * row_stride + head_offset);
    float dot = 0.0f;
    for (unsigned int d = 0; d < head_dim; ++d) {
        dot += query_head[d] * __half2float(*reinterpret_cast<const __half*>(&key_row[d]));
    }
    scores[index] = dot * scale;
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
    const unsigned int token_index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int head = blockIdx.y;
    if (head >= n_heads || token_index >= seq_len) return;
    if (kv_groups == 0) return;

    const unsigned int kv_head = head / kv_groups;
    const unsigned int head_offset = kv_head * head_dim;
    const float* query_head = query + (unsigned long long)head * head_dim;
    const unsigned short* key_row = key_cache + ((unsigned long long)token_index * row_stride + head_offset);

    float dot = 0.0f;
    for (unsigned int d = 0; d < head_dim; ++d) {
        dot += query_head[d] * __half2float(*reinterpret_cast<const __half*>(&key_row[d]));
    }
    scores[(unsigned long long)head * seq_len + token_index] = dot * scale;
}

extern "C" __global__ void talu_softmax_f32(
    float* out,
    const float* input,
    unsigned int count
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (count == 0) return;

    float max_v = input[0];
    for (unsigned int i = 1; i < count; ++i) {
        const float v = input[i];
        if (v > max_v) max_v = v;
    }

    float sum = 0.0f;
    for (unsigned int i = 0; i < count; ++i) {
        sum += expf(input[i] - max_v);
    }

    const float inv_sum = 1.0f / sum;
    for (unsigned int i = 0; i < count; ++i) {
        out[i] = expf(input[i] - max_v) * inv_sum;
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
    if (row >= rows || cols == 0) return;

    const float* row_in = input + (unsigned long long)row * cols;
    float* row_out = out + (unsigned long long)row * cols;

    float local_max = -3.402823466e+38f;
    for (unsigned int i = tid; i < cols; i += blockDim.x) {
        const float v = row_in[i];
        if (v > local_max) local_max = v;
    }

    __shared__ float scratch[256];
    scratch[tid] = local_max;
    __syncthreads();

    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other = scratch[tid + stride];
            if (other > scratch[tid]) scratch[tid] = other;
        }
        __syncthreads();
    }
    const float max_v = scratch[0];

    float local_sum = 0.0f;
    for (unsigned int i = tid; i < cols; i += blockDim.x) {
        local_sum += expf(row_in[i] - max_v);
    }
    scratch[tid] = local_sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    const float inv_sum = 1.0f / fmaxf(scratch[0], 1.0e-20f);

    for (unsigned int i = tid; i < cols; i += blockDim.x) {
        row_out[i] = expf(row_in[i] - max_v) * inv_sum;
    }
}

extern "C" __global__ void talu_attn_weighted_sum_f32(
    float* out_head,
    const float* probs,
    const float* value_cache,
    unsigned int seq_len,
    unsigned int row_stride,
    unsigned int head_offset,
    unsigned int head_dim
) {
    const unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= head_dim) return;

    float acc = 0.0f;
    for (unsigned int t = 0; t < seq_len; ++t) {
        const float p = probs[t];
        const unsigned long long value_index = (unsigned long long)t * row_stride + head_offset + d;
        acc += p * value_cache[value_index];
    }
    out_head[d] = acc;
}

extern "C" __global__ void talu_attn_weighted_sum_f16_kv(
    float* out_head,
    const float* probs,
    const unsigned short* value_cache,
    unsigned int seq_len,
    unsigned int row_stride,
    unsigned int head_offset,
    unsigned int head_dim
) {
    const unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= head_dim) return;

    float acc = 0.0f;
    for (unsigned int t = 0; t < seq_len; ++t) {
        const float p = probs[t];
        const unsigned long long value_index = (unsigned long long)t * row_stride + head_offset + d;
        acc += p * __half2float(*reinterpret_cast<const __half*>(&value_cache[value_index]));
    }
    out_head[d] = acc;
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
    const unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int head = blockIdx.y;
    if (head >= n_heads || d >= head_dim) return;
    if (kv_groups == 0) return;

    const unsigned int kv_head = head / kv_groups;
    const unsigned int head_offset = kv_head * head_dim;
    const float* probs_row = probs + (unsigned long long)head * seq_len;

    float acc = 0.0f;
    for (unsigned int t = 0; t < seq_len; ++t) {
        const unsigned long long value_index = (unsigned long long)t * row_stride + head_offset + d;
        acc += probs_row[t] * __half2float(*reinterpret_cast<const __half*>(&value_cache[value_index]));
    }
    out[(unsigned long long)head * head_dim + d] = acc;
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
            const float inv_freq = powf(theta, -2.0f * (float)pair / (float)rope_dim);
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
