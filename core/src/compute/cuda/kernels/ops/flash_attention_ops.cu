// Flash attention decode kernel with GQA optimization.
//
// Current fused decode: grid(n_heads, batch_rows) — each block reads KV
// for one query head. With kv_groups=4, 4 blocks read the same KV data.
//
// Flash decode: grid(n_kv_heads, batch_rows) — each block reads KV once
// and computes attention for all kv_groups query heads simultaneously.
// Reduces KV cache DRAM reads by kv_groups×.
//
// Online softmax (per-head): numerically stable streaming softmax with
// running max + exp-sum, merged across warps via shared memory.
//
// Max head_dim = 256, max kv_groups = 4.

static constexpr unsigned int FD_WARP_SIZE = 32u;
static constexpr unsigned int FD_MAX_GROUPS = 4u;
static constexpr unsigned int FD_MAX_DIMS_PER_LANE = 8u; // head_dim 256 / 32

static __forceinline__ __device__ float fd_warp_sum(float v) {
    for (unsigned int o = FD_WARP_SIZE >> 1; o > 0; o >>= 1)
        v += __shfl_down_sync(0xFFFFffffu, v, o);
    return v;
}

// Template for KV cache data type.
// LOAD_K(ptr, idx)  → float  (dequant K element)
// LOAD_V(ptr, idx)  → float  (dequant V element)
// HAS_SCALES        → 0/1    (whether per-head scales exist)
// SCALE_K(raw_dot, scale) → float  (apply K scale to raw dot product)

// ============================================================
// F16 KV cache
// ============================================================
#define FD_LOAD_K_F16(ptr, idx) __half2float(*reinterpret_cast<const __half*>(&(ptr)[(idx)]))
#define FD_LOAD_V_F16(ptr, idx) __half2float(*reinterpret_cast<const __half*>(&(ptr)[(idx)]))

extern "C" __global__ __launch_bounds__(256)
void talu_flash_decode_f16(
    float* __restrict__ out,
    const float* __restrict__ query,
    const unsigned long long* __restrict__ key_cache_ptrs,
    const unsigned long long* __restrict__ value_cache_ptrs,
    const unsigned int* __restrict__ seq_lens,
    const unsigned int* __restrict__ positions,
    unsigned int batch_rows,
    unsigned int n_heads,
    unsigned int n_kv_heads,
    unsigned int row_stride,       // KV cache row stride in ELEMENTS (not bytes)
    unsigned int kv_groups,
    unsigned int head_dim,
    float scale,
    unsigned int rope_dim,
    unsigned int sliding_window,
    float theta,
    const float* __restrict__ gate_proj,
    unsigned int gate_proj_stride,
    unsigned int n_seq_chunks,
    float* __restrict__ partial_m,
    float* __restrict__ partial_s,
    float* __restrict__ partial_out
) {
    const unsigned int kv_head = blockIdx.x / n_seq_chunks;
    const unsigned int chunk_id = blockIdx.x % n_seq_chunks;
    const unsigned int row = blockIdx.y;
    const unsigned int warp_id = threadIdx.x / 32u;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int num_warps = blockDim.x / 32u; // 8
    if (row >= batch_rows || kv_head >= n_kv_heads) return;
    if (kv_groups == 0 || head_dim == 0) return;

    const unsigned int seq_len = seq_lens[row];
    if (seq_len == 0) return;
    const unsigned int position = positions[row];
    const unsigned int effective_seq = min(seq_len, position + 1u);
    const unsigned int start_t = (sliding_window > 0u && effective_seq > sliding_window)
        ? (effective_seq - sliding_window) : 0u;
    const unsigned int total_tokens = effective_seq - start_t;
    if (total_tokens == 0) return;

    const unsigned short* key_cache = reinterpret_cast<const unsigned short*>(key_cache_ptrs[row]);
    const unsigned short* value_cache = reinterpret_cast<const unsigned short*>(value_cache_ptrs[row]);
    if (key_cache == nullptr || value_cache == nullptr) return;

    const unsigned int head_offset = kv_head * head_dim;
    const unsigned int first_head = kv_head * kv_groups;
    const unsigned int actual_groups = min(kv_groups, n_heads - first_head);
    const unsigned int dims_per_lane = (head_dim + 31u) >> 5;
    if (dims_per_lane > FD_MAX_DIMS_PER_LANE) return;

    const float log2_theta = log2f(theta);
    const unsigned int half_rope = rope_dim >> 1;

    // Load Q for all grouped heads into registers + apply RoPE.
    // q_rot[g][k]: g = group index, k = dims_per_lane chunk.
    float q_rot[FD_MAX_GROUPS][FD_MAX_DIMS_PER_LANE];
    #pragma unroll
    for (unsigned int g = 0; g < FD_MAX_GROUPS; ++g) {
        #pragma unroll
        for (unsigned int k = 0; k < FD_MAX_DIMS_PER_LANE; ++k) {
            q_rot[g][k] = 0.0f;
        }
    }
    for (unsigned int g = 0; g < actual_groups; ++g) {
        const unsigned int head = first_head + g;
        const float* query_head = query + ((unsigned long long)row * n_heads + head) * head_dim;
        #pragma unroll
        for (unsigned int k = 0; k < FD_MAX_DIMS_PER_LANE; ++k) {
            if (k >= dims_per_lane) break;
            const unsigned int d = lane + (k << 5);
            if (d >= head_dim) continue;

            float qv = query_head[d];
            if (d < rope_dim) {
                const unsigned int pair = (d < half_rope) ? d : (d - half_rope);
                const float q_lo = query_head[pair];
                const float q_hi = query_head[half_rope + pair];
                const float inv_freq = exp2f(log2_theta * (-2.0f * (float)pair / (float)rope_dim));
                const float angle = (float)position * inv_freq;
                float sn, cn;
                __sincosf(angle, &sn, &cn);
                qv = (d < half_rope) ? fmaf(q_lo, cn, -q_hi * sn) : fmaf(q_lo, sn, q_hi * cn);
            }
            q_rot[g][k] = qv;
        }
    }

    // Per-head online softmax accumulators.
    float m_val[FD_MAX_GROUPS];
    float s_val[FD_MAX_GROUPS];
    float out_acc[FD_MAX_GROUPS][FD_MAX_DIMS_PER_LANE];
    #pragma unroll
    for (unsigned int g = 0; g < FD_MAX_GROUPS; ++g) {
        m_val[g] = -3.402823466e+38f;
        s_val[g] = 0.0f;
        #pragma unroll
        for (unsigned int k = 0; k < FD_MAX_DIMS_PER_LANE; ++k) {
            out_acc[g][k] = 0.0f;
        }
    }

    // Split token range across chunks, then warps within each chunk.
    const unsigned int tokens_per_chunk = (total_tokens + n_seq_chunks - 1u) / n_seq_chunks;
    const unsigned int chunk_start = start_t + chunk_id * tokens_per_chunk;
    const unsigned int chunk_end = min(chunk_start + tokens_per_chunk, effective_seq);
    const unsigned int chunk_tokens = (chunk_end > chunk_start) ? (chunk_end - chunk_start) : 0u;
    const unsigned int tokens_per_warp = (chunk_tokens + num_warps - 1u) / num_warps;
    const unsigned int my_start = chunk_start + warp_id * tokens_per_warp;
    const unsigned int my_end = min(my_start + tokens_per_warp, chunk_end);
    const unsigned int wmask = 0xFFFFffffu;

    // Iterate over this warp's token range. Read K/V once, score all Q heads.
    for (unsigned int t = my_start; t < my_end; ++t) {
        const unsigned short* key_row = key_cache + ((unsigned long long)t * row_stride + head_offset);
        const unsigned short* value_row = value_cache + ((unsigned long long)t * row_stride + head_offset);

        // Compute Q·K dot for all grouped heads, sharing the K read.
        float dots[FD_MAX_GROUPS];
        #pragma unroll
        for (unsigned int g = 0; g < FD_MAX_GROUPS; ++g) {
            float partial = 0.0f;
            if (g < actual_groups) {
                #pragma unroll
                for (unsigned int k = 0; k < FD_MAX_DIMS_PER_LANE; ++k) {
                    if (k >= dims_per_lane) break;
                    const unsigned int d = lane + (k << 5);
                    if (d < head_dim) {
                        partial += q_rot[g][k] * FD_LOAD_K_F16(key_row, d);
                    }
                }
                // Warp reduce.
                for (unsigned int o = 16; o > 0; o >>= 1)
                    partial += __shfl_down_sync(wmask, partial, o);
            }
            dots[g] = __shfl_sync(wmask, partial, 0) * scale;
        }

        // Online softmax update + V accumulation for each head.
        #pragma unroll
        for (unsigned int g = 0; g < FD_MAX_GROUPS; ++g) {
            if (g >= actual_groups) continue;
            const float score = dots[g];
            const float m_new = fmaxf(m_val[g], score);
            const float alpha = expf(m_val[g] - m_new);
            const float beta = expf(score - m_new);
            s_val[g] = s_val[g] * alpha + beta;
            m_val[g] = m_new;

            #pragma unroll
            for (unsigned int k = 0; k < FD_MAX_DIMS_PER_LANE; ++k) {
                if (k >= dims_per_lane) break;
                const unsigned int d = lane + (k << 5);
                if (d < head_dim) {
                    const float v = FD_LOAD_V_F16(value_row, d);
                    out_acc[g][k] = out_acc[g][k] * alpha + v * beta;
                }
            }
        }
    }

    // Cross-warp merge via shared memory.
    // Layout per group: [num_warps] m, [num_warps] s, [num_warps * head_dim] out
    // Total groups in smem = actual_groups.
    extern __shared__ float smem[];
    const unsigned int per_group = 2u * num_warps + num_warps * head_dim;

    for (unsigned int g = 0; g < actual_groups; ++g) {
        float* g_smem = smem + g * per_group;
        float* g_m = g_smem;
        float* g_s = g_smem + num_warps;
        float* g_out = g_smem + 2u * num_warps;

        if (lane == 0) {
            g_m[warp_id] = m_val[g];
            g_s[warp_id] = s_val[g];
        }
        #pragma unroll
        for (unsigned int k = 0; k < FD_MAX_DIMS_PER_LANE; ++k) {
            if (k >= dims_per_lane) break;
            const unsigned int d = lane + (k << 5);
            if (d < head_dim) {
                g_out[warp_id * head_dim + d] = out_acc[g][k];
            }
        }
    }
    __syncthreads();

    // Only warp 0 merges and writes output for all heads.
    if (warp_id != 0) return;

    for (unsigned int g = 0; g < actual_groups; ++g) {
        const unsigned int head = first_head + g;
        float* g_smem = smem + g * per_group;
        float* g_m = g_smem;
        float* g_s = g_smem + num_warps;
        float* g_out = g_smem + 2u * num_warps;

        float mg = g_m[0];
        float sg = g_s[0];
        float merged[FD_MAX_DIMS_PER_LANE];
        #pragma unroll
        for (unsigned int k = 0; k < FD_MAX_DIMS_PER_LANE; ++k) {
            if (k >= dims_per_lane) break;
            const unsigned int d = lane + (k << 5);
            merged[k] = (d < head_dim) ? g_out[d] : 0.0f;
        }

        for (unsigned int w = 1; w < num_warps; ++w) {
            const float mw = g_m[w];
            const float sw = g_s[w];
            const float m_new = fmaxf(mg, mw);
            const float a = expf(mg - m_new);
            const float b = expf(mw - m_new);
            sg = sg * a + sw * b;
            mg = m_new;
            #pragma unroll
            for (unsigned int k = 0; k < FD_MAX_DIMS_PER_LANE; ++k) {
                if (k >= dims_per_lane) break;
                const unsigned int d = lane + (k << 5);
                if (d < head_dim) {
                    merged[k] = merged[k] * a + g_out[w * head_dim + d] * b;
                }
            }
        }

        if (n_seq_chunks == 1u) {
            // Direct output: normalize and write.
            float* out_head = out + ((unsigned long long)row * n_heads + head) * head_dim;
            const float inv_s = 1.0f / fmaxf(sg, 1.0e-20f);
            #pragma unroll
            for (unsigned int k = 0; k < FD_MAX_DIMS_PER_LANE; ++k) {
                if (k >= dims_per_lane) break;
                const unsigned int d = lane + (k << 5);
                if (d < head_dim) {
                    float result = merged[k] * inv_s;
                    if (gate_proj) {
                        const float gate = gate_proj[(unsigned long long)row * gate_proj_stride + (unsigned long long)head * head_dim * 2u + head_dim + d];
                        result *= 1.0f / (1.0f + expf(-gate));
                    }
                    out_head[d] = result;
                }
            }
        } else {
            // Split-K: write partial (m, s, out_unnormalized) for this chunk.
            const unsigned long long partial_idx =
                ((unsigned long long)row * n_heads + head) * n_seq_chunks + chunk_id;
            if (lane == 0) {
                partial_m[partial_idx] = mg;
                partial_s[partial_idx] = sg;
            }
            #pragma unroll
            for (unsigned int k = 0; k < FD_MAX_DIMS_PER_LANE; ++k) {
                if (k >= dims_per_lane) break;
                const unsigned int d = lane + (k << 5);
                if (d < head_dim)
                    partial_out[partial_idx * head_dim + d] = merged[k];
            }
        }
    }
}

// ============================================================
// I8 KV cache (per-head f32 scales)
// ============================================================
extern "C" __global__ __launch_bounds__(256)
void talu_flash_decode_i8(
    float* __restrict__ out,
    const float* __restrict__ query,
    const unsigned long long* __restrict__ key_cache_ptrs,
    const unsigned long long* __restrict__ value_cache_ptrs,
    const unsigned long long* __restrict__ k_scale_ptrs,
    const unsigned long long* __restrict__ v_scale_ptrs,
    const unsigned int* __restrict__ seq_lens,
    const unsigned int* __restrict__ positions,
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
    const float* __restrict__ gate_proj,
    unsigned int gate_proj_stride,
    unsigned int n_seq_chunks,
    float* __restrict__ partial_m,
    float* __restrict__ partial_s,
    float* __restrict__ partial_out
) {
    const unsigned int kv_head = blockIdx.x / n_seq_chunks;
    const unsigned int chunk_id = blockIdx.x % n_seq_chunks;
    const unsigned int row = blockIdx.y;
    const unsigned int warp_id = threadIdx.x / 32u;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int num_warps = blockDim.x / 32u;
    if (row >= batch_rows || kv_head >= n_kv_heads) return;
    if (kv_groups == 0 || head_dim == 0) return;

    const unsigned int seq_len = seq_lens[row];
    if (seq_len == 0) return;
    const unsigned int position = positions[row];
    const unsigned int effective_seq = min(seq_len, position + 1u);
    const unsigned int start_t = (sliding_window > 0u && effective_seq > sliding_window)
        ? (effective_seq - sliding_window) : 0u;
    const unsigned int total_tokens = effective_seq - start_t;
    if (total_tokens == 0) return;

    const signed char* key_cache = reinterpret_cast<const signed char*>(key_cache_ptrs[row]);
    const signed char* value_cache = reinterpret_cast<const signed char*>(value_cache_ptrs[row]);
    const float* k_scales = reinterpret_cast<const float*>(k_scale_ptrs[row]);
    const float* v_scales = reinterpret_cast<const float*>(v_scale_ptrs[row]);
    if (key_cache == nullptr || value_cache == nullptr) return;

    const unsigned int head_offset = kv_head * head_dim;
    const unsigned int first_head = kv_head * kv_groups;
    const unsigned int actual_groups = min(kv_groups, n_heads - first_head);
    const unsigned int dims_per_lane = (head_dim + 31u) >> 5;
    if (dims_per_lane > FD_MAX_DIMS_PER_LANE) return;

    const float log2_theta = log2f(theta);
    const unsigned int half_rope = rope_dim >> 1;
    const unsigned int per_group = 2u * num_warps + num_warps * head_dim;

    // Shared memory layout per group:
    // [num_warps] m, [num_warps] s, [num_warps * head_dim] out
    // Reuse out[0..head_dim) of each group as temporary Q-RoPE staging.
    extern __shared__ float smem[];

    // Load Q + RoPE for all grouped heads.
    float q_rot[FD_MAX_GROUPS][FD_MAX_DIMS_PER_LANE];
    #pragma unroll
    for (unsigned int g = 0; g < FD_MAX_GROUPS; ++g) {
        #pragma unroll
        for (unsigned int k = 0; k < FD_MAX_DIMS_PER_LANE; ++k)
            q_rot[g][k] = 0.0f;
    }
    if (warp_id == 0u) {
        for (unsigned int g = 0; g < actual_groups; ++g) {
            const unsigned int head = first_head + g;
            const float* query_head = query + ((unsigned long long)row * n_heads + head) * head_dim;
            float* g_m = smem + g * per_group;
            float* g_out = g_m + 2u * num_warps;
            #pragma unroll
            for (unsigned int k = 0; k < FD_MAX_DIMS_PER_LANE; ++k) {
                if (k >= dims_per_lane) break;
                const unsigned int d = lane + (k << 5);
                if (d >= head_dim) continue;
                float qv = query_head[d];
                if (d < rope_dim) {
                    const unsigned int pair = (d < half_rope) ? d : (d - half_rope);
                    const float q_lo = query_head[pair];
                    const float q_hi = query_head[half_rope + pair];
                    const float inv_freq = exp2f(log2_theta * (-2.0f * (float)pair / (float)rope_dim));
                    const float angle = (float)position * inv_freq;
                    float sn, cn;
                    __sincosf(angle, &sn, &cn);
                    qv = (d < half_rope) ? fmaf(q_lo, cn, -q_hi * sn) : fmaf(q_lo, sn, q_hi * cn);
                }
                g_out[d] = qv;
            }
        }
    }
    __syncthreads();
    for (unsigned int g = 0; g < actual_groups; ++g) {
        float* g_m = smem + g * per_group;
        float* g_out = g_m + 2u * num_warps;
        #pragma unroll
        for (unsigned int k = 0; k < FD_MAX_DIMS_PER_LANE; ++k) {
            if (k >= dims_per_lane) break;
            const unsigned int d = lane + (k << 5);
            if (d < head_dim) q_rot[g][k] = g_out[d];
        }
    }

    // Per-head accumulators.
    float m_val[FD_MAX_GROUPS], s_val[FD_MAX_GROUPS];
    float out_acc[FD_MAX_GROUPS][FD_MAX_DIMS_PER_LANE];
    #pragma unroll
    for (unsigned int g = 0; g < FD_MAX_GROUPS; ++g) {
        m_val[g] = -3.402823466e+38f;
        s_val[g] = 0.0f;
        #pragma unroll
        for (unsigned int k = 0; k < FD_MAX_DIMS_PER_LANE; ++k)
            out_acc[g][k] = 0.0f;
    }

    // Split token range across chunks, then warps within each chunk.
    const unsigned int tokens_per_chunk = (total_tokens + n_seq_chunks - 1u) / n_seq_chunks;
    const unsigned int chunk_start = start_t + chunk_id * tokens_per_chunk;
    const unsigned int chunk_end = min(chunk_start + tokens_per_chunk, effective_seq);
    const unsigned int chunk_tokens = (chunk_end > chunk_start) ? (chunk_end - chunk_start) : 0u;
    const unsigned int tokens_per_warp = (chunk_tokens + num_warps - 1u) / num_warps;
    const unsigned int my_start = chunk_start + warp_id * tokens_per_warp;
    const unsigned int my_end = min(my_start + tokens_per_warp, chunk_end);
    const unsigned int wmask = 0xFFFFffffu;

    for (unsigned int t = my_start; t < my_end; ++t) {
        const signed char* key_row = key_cache + ((unsigned long long)t * row_stride + head_offset);
        const signed char* value_row = value_cache + ((unsigned long long)t * row_stride + head_offset);
        const float k_sc = k_scales[t * n_kv_heads + kv_head];
        const float v_sc = v_scales[t * n_kv_heads + kv_head];

        // Q·K for all heads (K read once, shared).
        float dots[FD_MAX_GROUPS];
        #pragma unroll
        for (unsigned int g = 0; g < FD_MAX_GROUPS; ++g) {
            float partial = 0.0f;
            if (g < actual_groups) {
                #pragma unroll
                for (unsigned int k = 0; k < FD_MAX_DIMS_PER_LANE; ++k) {
                    if (k >= dims_per_lane) break;
                    const unsigned int d = lane + (k << 5);
                    if (d < head_dim) {
                        partial += q_rot[g][k] * (float)(key_row[d]);
                    }
                }
                for (unsigned int o = 16; o > 0; o >>= 1)
                    partial += __shfl_down_sync(wmask, partial, o);
            }
            dots[g] = __shfl_sync(wmask, partial, 0) * k_sc * scale;
        }

        // Online softmax + V accumulation.
        #pragma unroll
        for (unsigned int g = 0; g < FD_MAX_GROUPS; ++g) {
            if (g >= actual_groups) continue;
            const float sc = dots[g];
            const float m_new = fmaxf(m_val[g], sc);
            const float alpha = __expf(m_val[g] - m_new);
            const float beta = __expf(sc - m_new);
            s_val[g] = s_val[g] * alpha + beta;
            m_val[g] = m_new;
            #pragma unroll
            for (unsigned int k = 0; k < FD_MAX_DIMS_PER_LANE; ++k) {
                if (k >= dims_per_lane) break;
                const unsigned int d = lane + (k << 5);
                if (d < head_dim) {
                    const float v = (float)(value_row[d]) * v_sc;
                    out_acc[g][k] = out_acc[g][k] * alpha + v * beta;
                }
            }
        }
    }

    // Cross-warp merge.
    for (unsigned int g = 0; g < actual_groups; ++g) {
        float* g_m = smem + g * per_group;
        float* g_s = g_m + num_warps;
        float* g_out = g_s + num_warps;
        if (lane == 0) {
            g_m[warp_id] = m_val[g];
            g_s[warp_id] = s_val[g];
        }
        #pragma unroll
        for (unsigned int k = 0; k < FD_MAX_DIMS_PER_LANE; ++k) {
            if (k >= dims_per_lane) break;
            const unsigned int d = lane + (k << 5);
            if (d < head_dim)
                g_out[warp_id * head_dim + d] = out_acc[g][k];
        }
    }
    __syncthreads();

    if (warp_id >= actual_groups) return;
    const unsigned int g = warp_id;
    const unsigned int head = first_head + g;
    float* g_m = smem + g * per_group;
    float* g_s = g_m + num_warps;
    float* g_out = g_s + num_warps;

    float mg = g_m[0], sg = g_s[0];
    float merged[FD_MAX_DIMS_PER_LANE];
    #pragma unroll
    for (unsigned int k = 0; k < FD_MAX_DIMS_PER_LANE; ++k) {
        const unsigned int d = lane + (k << 5);
        merged[k] = (k < dims_per_lane && d < head_dim) ? g_out[d] : 0.0f;
    }
    for (unsigned int w = 1; w < num_warps; ++w) {
        const float mw = g_m[w], sw = g_s[w];
        const float m_new = fmaxf(mg, mw);
        const float a = __expf(mg - m_new), b = __expf(mw - m_new);
        sg = sg * a + sw * b;
        mg = m_new;
        #pragma unroll
        for (unsigned int k = 0; k < FD_MAX_DIMS_PER_LANE; ++k) {
            if (k >= dims_per_lane) break;
            const unsigned int d = lane + (k << 5);
            if (d < head_dim)
                merged[k] = merged[k] * a + g_out[w * head_dim + d] * b;
        }
    }

    if (n_seq_chunks == 1u) {
        float* out_head = out + ((unsigned long long)row * n_heads + head) * head_dim;
        const float inv_s = 1.0f / fmaxf(sg, 1.0e-20f);
        #pragma unroll
        for (unsigned int k = 0; k < FD_MAX_DIMS_PER_LANE; ++k) {
            if (k >= dims_per_lane) break;
            const unsigned int d = lane + (k << 5);
            if (d < head_dim) {
                float result = merged[k] * inv_s;
                if (gate_proj) {
                    const float gate = gate_proj[(unsigned long long)row * gate_proj_stride + (unsigned long long)head * head_dim * 2u + head_dim + d];
                    result *= 1.0f / (1.0f + __expf(-gate));
                }
                out_head[d] = result;
            }
        }
    } else {
        const unsigned long long partial_idx =
            ((unsigned long long)row * n_heads + head) * n_seq_chunks + chunk_id;
        if (lane == 0) {
            partial_m[partial_idx] = mg;
            partial_s[partial_idx] = sg;
        }
        #pragma unroll
        for (unsigned int k = 0; k < FD_MAX_DIMS_PER_LANE; ++k) {
            if (k >= dims_per_lane) break;
            const unsigned int d = lane + (k << 5);
            if (d < head_dim)
                partial_out[partial_idx * head_dim + d] = merged[k];
        }
    }
}

// ============================================================
// FP8 E4M3 KV cache (per-head f32 scales)
// ============================================================
#if __CUDA_ARCH__ >= 890

static __device__ __forceinline__ float fd_fp8e4m3_to_f32(__nv_fp8_storage_t x) {
    __half_raw hr = __nv_cvt_fp8_to_halfraw(x, __NV_E4M3);
    __half h;
    memcpy(&h, &hr, sizeof(h));
    return __half2float(h);
}

extern "C" __global__ __launch_bounds__(256)
void talu_flash_decode_fp8(
    float* __restrict__ out,
    const float* __restrict__ query,
    const unsigned long long* __restrict__ key_cache_ptrs,
    const unsigned long long* __restrict__ value_cache_ptrs,
    const unsigned long long* __restrict__ k_scale_ptrs,
    const unsigned long long* __restrict__ v_scale_ptrs,
    const unsigned int* __restrict__ seq_lens,
    const unsigned int* __restrict__ positions,
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
    const float* __restrict__ gate_proj,
    unsigned int gate_proj_stride,
    unsigned int n_seq_chunks,
    float* __restrict__ partial_m,
    float* __restrict__ partial_s,
    float* __restrict__ partial_out
) {
    const unsigned int kv_head = blockIdx.x / n_seq_chunks;
    const unsigned int chunk_id = blockIdx.x % n_seq_chunks;
    const unsigned int row = blockIdx.y;
    const unsigned int warp_id = threadIdx.x / 32u;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int num_warps = blockDim.x / 32u;
    if (row >= batch_rows || kv_head >= n_kv_heads) return;
    if (kv_groups == 0 || head_dim == 0) return;

    const unsigned int seq_len = seq_lens[row];
    if (seq_len == 0) return;
    const unsigned int position = positions[row];
    const unsigned int effective_seq = min(seq_len, position + 1u);
    const unsigned int start_t = (sliding_window > 0u && effective_seq > sliding_window)
        ? (effective_seq - sliding_window) : 0u;
    const unsigned int total_tokens = effective_seq - start_t;
    if (total_tokens == 0) return;

    const unsigned char* key_cache = reinterpret_cast<const unsigned char*>(key_cache_ptrs[row]);
    const unsigned char* value_cache = reinterpret_cast<const unsigned char*>(value_cache_ptrs[row]);
    const float* k_scales = reinterpret_cast<const float*>(k_scale_ptrs[row]);
    const float* v_scales = reinterpret_cast<const float*>(v_scale_ptrs[row]);
    if (key_cache == nullptr || value_cache == nullptr) return;

    const unsigned int head_offset = kv_head * head_dim;
    const unsigned int first_head = kv_head * kv_groups;
    const unsigned int actual_groups = min(kv_groups, n_heads - first_head);
    const unsigned int dims_per_lane = (head_dim + 31u) >> 5;
    if (dims_per_lane > FD_MAX_DIMS_PER_LANE) return;

    const float log2_theta = log2f(theta);
    const unsigned int half_rope = rope_dim >> 1;
    const unsigned int per_group = 2u * num_warps + num_warps * head_dim;

    // Shared memory layout per group:
    // [num_warps] m, [num_warps] s, [num_warps * head_dim] out
    // Reuse out[0..head_dim) of each group as temporary Q-RoPE staging.
    extern __shared__ float smem[];

    // Load Q + RoPE.
    float q_rot[FD_MAX_GROUPS][FD_MAX_DIMS_PER_LANE];
    #pragma unroll
    for (unsigned int g = 0; g < FD_MAX_GROUPS; ++g) {
        #pragma unroll
        for (unsigned int k = 0; k < FD_MAX_DIMS_PER_LANE; ++k)
            q_rot[g][k] = 0.0f;
    }
    if (warp_id == 0u) {
        for (unsigned int g = 0; g < actual_groups; ++g) {
            const unsigned int head = first_head + g;
            const float* query_head = query + ((unsigned long long)row * n_heads + head) * head_dim;
            float* g_m = smem + g * per_group;
            float* g_out = g_m + 2u * num_warps;
            #pragma unroll
            for (unsigned int k = 0; k < FD_MAX_DIMS_PER_LANE; ++k) {
                if (k >= dims_per_lane) break;
                const unsigned int d = lane + (k << 5);
                if (d >= head_dim) continue;
                float qv = query_head[d];
                if (d < rope_dim) {
                    const unsigned int pair = (d < half_rope) ? d : (d - half_rope);
                    const float q_lo = query_head[pair];
                    const float q_hi = query_head[half_rope + pair];
                    const float inv_freq = exp2f(log2_theta * (-2.0f * (float)pair / (float)rope_dim));
                    const float angle = (float)position * inv_freq;
                    float sn, cn;
                    __sincosf(angle, &sn, &cn);
                    qv = (d < half_rope) ? fmaf(q_lo, cn, -q_hi * sn) : fmaf(q_lo, sn, q_hi * cn);
                }
                g_out[d] = qv;
            }
        }
    }
    __syncthreads();
    for (unsigned int g = 0; g < actual_groups; ++g) {
        float* g_m = smem + g * per_group;
        float* g_out = g_m + 2u * num_warps;
        #pragma unroll
        for (unsigned int k = 0; k < FD_MAX_DIMS_PER_LANE; ++k) {
            if (k >= dims_per_lane) break;
            const unsigned int d = lane + (k << 5);
            if (d < head_dim) q_rot[g][k] = g_out[d];
        }
    }

    float m_val[FD_MAX_GROUPS], s_val[FD_MAX_GROUPS];
    float out_acc[FD_MAX_GROUPS][FD_MAX_DIMS_PER_LANE];
    #pragma unroll
    for (unsigned int g = 0; g < FD_MAX_GROUPS; ++g) {
        m_val[g] = -3.402823466e+38f;
        s_val[g] = 0.0f;
        #pragma unroll
        for (unsigned int k = 0; k < FD_MAX_DIMS_PER_LANE; ++k)
            out_acc[g][k] = 0.0f;
    }

    const unsigned int tokens_per_chunk = (total_tokens + n_seq_chunks - 1u) / n_seq_chunks;
    const unsigned int chunk_start = start_t + chunk_id * tokens_per_chunk;
    const unsigned int chunk_end = min(chunk_start + tokens_per_chunk, effective_seq);
    const unsigned int chunk_tokens = (chunk_end > chunk_start) ? (chunk_end - chunk_start) : 0u;
    const unsigned int tokens_per_warp = (chunk_tokens + num_warps - 1u) / num_warps;
    const unsigned int my_start = chunk_start + warp_id * tokens_per_warp;
    const unsigned int my_end = min(my_start + tokens_per_warp, chunk_end);
    const unsigned int wmask = 0xFFFFffffu;

    for (unsigned int t = my_start; t < my_end; ++t) {
        const unsigned char* key_row = key_cache + ((unsigned long long)t * row_stride + head_offset);
        const unsigned char* value_row = value_cache + ((unsigned long long)t * row_stride + head_offset);
        const float k_sc = k_scales[t * n_kv_heads + kv_head];
        const float v_sc = v_scales[t * n_kv_heads + kv_head];

        float dots[FD_MAX_GROUPS];
        #pragma unroll
        for (unsigned int g = 0; g < FD_MAX_GROUPS; ++g) {
            float partial = 0.0f;
            if (g < actual_groups) {
                #pragma unroll
                for (unsigned int k = 0; k < FD_MAX_DIMS_PER_LANE; ++k) {
                    if (k >= dims_per_lane) break;
                    const unsigned int d = lane + (k << 5);
                    if (d < head_dim) {
                        partial += q_rot[g][k] * fd_fp8e4m3_to_f32(
                            static_cast<__nv_fp8_storage_t>(key_row[d]));
                    }
                }
                for (unsigned int o = 16; o > 0; o >>= 1)
                    partial += __shfl_down_sync(wmask, partial, o);
            }
            dots[g] = __shfl_sync(wmask, partial, 0) * k_sc * scale;
        }

        #pragma unroll
        for (unsigned int g = 0; g < FD_MAX_GROUPS; ++g) {
            if (g >= actual_groups) continue;
            const float sc = dots[g];
            const float m_new = fmaxf(m_val[g], sc);
            const float alpha = __expf(m_val[g] - m_new);
            const float beta = __expf(sc - m_new);
            s_val[g] = s_val[g] * alpha + beta;
            m_val[g] = m_new;
            #pragma unroll
            for (unsigned int k = 0; k < FD_MAX_DIMS_PER_LANE; ++k) {
                if (k >= dims_per_lane) break;
                const unsigned int d = lane + (k << 5);
                if (d < head_dim) {
                    const float v = fd_fp8e4m3_to_f32(
                        static_cast<__nv_fp8_storage_t>(value_row[d])) * v_sc;
                    out_acc[g][k] = out_acc[g][k] * alpha + v * beta;
                }
            }
        }
    }

    // Cross-warp merge.
    for (unsigned int g = 0; g < actual_groups; ++g) {
        float* g_m = smem + g * per_group;
        float* g_s = g_m + num_warps;
        float* g_out = g_s + num_warps;
        if (lane == 0) {
            g_m[warp_id] = m_val[g];
            g_s[warp_id] = s_val[g];
        }
        #pragma unroll
        for (unsigned int k = 0; k < FD_MAX_DIMS_PER_LANE; ++k) {
            if (k >= dims_per_lane) break;
            const unsigned int d = lane + (k << 5);
            if (d < head_dim)
                g_out[warp_id * head_dim + d] = out_acc[g][k];
        }
    }
    __syncthreads();

    if (warp_id >= actual_groups) return;
    const unsigned int g = warp_id;
    const unsigned int head = first_head + g;
    float* g_m = smem + g * per_group;
    float* g_s = g_m + num_warps;
    float* g_out = g_s + num_warps;

    float mg = g_m[0], sg = g_s[0];
    float merged[FD_MAX_DIMS_PER_LANE];
    #pragma unroll
    for (unsigned int k = 0; k < FD_MAX_DIMS_PER_LANE; ++k) {
        const unsigned int d = lane + (k << 5);
        merged[k] = (k < dims_per_lane && d < head_dim) ? g_out[d] : 0.0f;
    }
    for (unsigned int w = 1; w < num_warps; ++w) {
        const float mw = g_m[w], sw = g_s[w];
        const float m_new = fmaxf(mg, mw);
        const float a = __expf(mg - m_new), b = __expf(mw - m_new);
        sg = sg * a + sw * b;
        mg = m_new;
        #pragma unroll
        for (unsigned int k = 0; k < FD_MAX_DIMS_PER_LANE; ++k) {
            if (k >= dims_per_lane) break;
            const unsigned int d = lane + (k << 5);
            if (d < head_dim)
                merged[k] = merged[k] * a + g_out[w * head_dim + d] * b;
        }
    }

    if (n_seq_chunks == 1u) {
        float* out_head = out + ((unsigned long long)row * n_heads + head) * head_dim;
        const float inv_s = 1.0f / fmaxf(sg, 1.0e-20f);
        #pragma unroll
        for (unsigned int k = 0; k < FD_MAX_DIMS_PER_LANE; ++k) {
            if (k >= dims_per_lane) break;
            const unsigned int d = lane + (k << 5);
            if (d < head_dim) {
                float result = merged[k] * inv_s;
                if (gate_proj) {
                    const float gate = gate_proj[(unsigned long long)row * gate_proj_stride + (unsigned long long)head * head_dim * 2u + head_dim + d];
                    result *= 1.0f / (1.0f + __expf(-gate));
                }
                out_head[d] = result;
            }
        }
    } else {
        const unsigned long long partial_idx =
            ((unsigned long long)row * n_heads + head) * n_seq_chunks + chunk_id;
        if (lane == 0) {
            partial_m[partial_idx] = mg;
            partial_s[partial_idx] = sg;
        }
        #pragma unroll
        for (unsigned int k = 0; k < FD_MAX_DIMS_PER_LANE; ++k) {
            if (k >= dims_per_lane) break;
            const unsigned int d = lane + (k << 5);
            if (d < head_dim)
                partial_out[partial_idx * head_dim + d] = merged[k];
        }
    }
}

#endif // __CUDA_ARCH__ >= 890

// ============================================================
// Flash Decode Reduce — merge split-K partial results
// ============================================================
//
// Grid: (n_heads, batch_rows)
// Block: 32 threads (1 warp)
//
// Merges n_seq_chunks partial (m, s, out) results per (row, head)
// using online softmax, then normalizes and writes final output.

extern "C" __global__ __launch_bounds__(32)
void talu_flash_decode_reduce(
    float* __restrict__ out,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_s,
    const float* __restrict__ partial_out,
    unsigned int batch_rows,
    unsigned int n_heads,
    unsigned int head_dim,
    unsigned int n_seq_chunks,
    const float* __restrict__ gate_proj,
    unsigned int gate_proj_stride
) {
    const unsigned int head = blockIdx.x;
    const unsigned int row = blockIdx.y;
    const unsigned int lane = threadIdx.x;
    if (head >= n_heads || row >= batch_rows) return;

    const unsigned int dims_per_lane = (head_dim + 31u) >> 5;
    const unsigned long long base =
        ((unsigned long long)row * n_heads + head) * n_seq_chunks;

    // Initialize from chunk 0.
    float mg = partial_m[base];
    float sg = partial_s[base];
    float merged[FD_MAX_DIMS_PER_LANE];
    #pragma unroll
    for (unsigned int k = 0; k < FD_MAX_DIMS_PER_LANE; ++k) {
        const unsigned int d = lane + (k << 5);
        merged[k] = (k < dims_per_lane && d < head_dim)
            ? partial_out[base * head_dim + d] : 0.0f;
    }

    // Merge remaining chunks.
    for (unsigned int c = 1; c < n_seq_chunks; ++c) {
        const float mc = partial_m[base + c];
        const float sc = partial_s[base + c];
        if (sc == 0.0f) continue; // empty chunk
        if (sg == 0.0f) {
            // First non-empty chunk — adopt directly.
            mg = mc;
            sg = sc;
            #pragma unroll
            for (unsigned int k = 0; k < FD_MAX_DIMS_PER_LANE; ++k) {
                const unsigned int d = lane + (k << 5);
                if (k < dims_per_lane && d < head_dim)
                    merged[k] = partial_out[(base + c) * head_dim + d];
            }
            continue;
        }

        const float m_new = fmaxf(mg, mc);
        const float a = expf(mg - m_new);
        const float b = expf(mc - m_new);
        sg = sg * a + sc * b;
        mg = m_new;
        #pragma unroll
        for (unsigned int k = 0; k < FD_MAX_DIMS_PER_LANE; ++k) {
            const unsigned int d = lane + (k << 5);
            if (k < dims_per_lane && d < head_dim)
                merged[k] = merged[k] * a + partial_out[(base + c) * head_dim + d] * b;
        }
    }

    // Normalize and write.
    float* out_head = out + ((unsigned long long)row * n_heads + head) * head_dim;
    const float inv_s = 1.0f / fmaxf(sg, 1.0e-20f);
    #pragma unroll
    for (unsigned int k = 0; k < FD_MAX_DIMS_PER_LANE; ++k) {
        if (k >= dims_per_lane) break;
        const unsigned int d = lane + (k << 5);
        if (d < head_dim) {
            float result = merged[k] * inv_s;
            if (gate_proj) {
                const float gate = gate_proj[(unsigned long long)row * gate_proj_stride + (unsigned long long)head * head_dim * 2u + head_dim + d];
                result *= 1.0f / (1.0f + expf(-gate));
            }
            out_head[d] = result;
        }
    }
}

// ============================================================
// Flash Prefill Attention
// ============================================================
//
// Tiled prefill with online softmax and KV tile sharing.
//
// Grid: (ceil(q_rows / FP_BR), n_heads, 1)
// Block: FP_BR × 32 threads (FP_BR warps)
//
// Each block processes FP_BR query rows for one attention head.
// K/V tiles loaded into shared memory and reused across all FP_BR warps,
// reducing KV cache DRAM reads by FP_BR×.
//
// Causal masking: each warp has its own q_position; future KV tokens skipped.
// Sliding window: per-warp skip for tokens before the window.

static constexpr unsigned int FP_BR = 4u;      // query rows per block
static constexpr unsigned int FP_TILE = 16u;    // KV tokens per smem tile
static constexpr unsigned int FP_BLOCK = FP_BR * 32u; // 128 threads
static constexpr unsigned int FP_MAX_DPL = 8u;  // head_dim ≤ 256

// ============================================================
// F16 KV cache — flash prefill
// ============================================================
extern "C" __global__ __launch_bounds__(128)
void talu_flash_prefill_f16(
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
    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int head = blockIdx.y;
    const unsigned int q_start = blockIdx.x * FP_BR;
    const unsigned int q_idx = q_start + warp_id;
    const bool valid_q = (q_idx < q_rows);

    const unsigned int kv_head = head / kv_groups;
    const unsigned int head_offset = kv_head * head_dim;
    const unsigned int dims_per_lane = (head_dim + 31u) >> 5;

    const float log2_theta = log2f(theta);
    const unsigned int half_rope = rope_dim >> 1;
    const unsigned int q_position = position_base + q_idx;

    // Common KV range across all warps in the block (for tile loading).
    const unsigned int last_valid_q = min(q_start + FP_BR - 1u, q_rows - 1u);
    const unsigned int last_q_pos = position_base + last_valid_q;
    const unsigned int common_eff = min(seq_len, last_q_pos + 1u);
    const unsigned int first_q_pos = position_base + q_start;
    const unsigned int first_eff = min(seq_len, first_q_pos + 1u);
    const unsigned int common_kv_start = (sliding_window > 0u && first_eff > sliding_window)
        ? (first_eff - sliding_window) : 0u;
    const unsigned int common_kv_len = (common_eff > common_kv_start)
        ? (common_eff - common_kv_start) : 0u;

    // Per-warp sliding window start.
    const unsigned int my_eff = valid_q ? min(seq_len, q_position + 1u) : 0u;
    const unsigned int my_kv_start = (sliding_window > 0u && my_eff > sliding_window)
        ? (my_eff - sliding_window) : 0u;

    // Load Q + RoPE into registers.
    float q_reg[FP_MAX_DPL];
    #pragma unroll
    for (unsigned int k = 0; k < FP_MAX_DPL; ++k) q_reg[k] = 0.0f;
    if (valid_q && dims_per_lane <= FP_MAX_DPL) {
        const float* qh = query + ((unsigned long long)q_idx * n_heads + head) * head_dim;
        #pragma unroll
        for (unsigned int k = 0; k < FP_MAX_DPL; ++k) {
            if (k >= dims_per_lane) break;
            const unsigned int d = lane + (k << 5);
            if (d >= head_dim) continue;
            float qv = qh[d];
            if (d < rope_dim) {
                const unsigned int pair = (d < half_rope) ? d : (d - half_rope);
                const float q_lo = qh[pair];
                const float q_hi = qh[half_rope + pair];
                const float inv_freq = exp2f(log2_theta * (-2.0f * (float)pair / (float)rope_dim));
                const float angle = (float)q_position * inv_freq;
                float sn, cn;
                __sincosf(angle, &sn, &cn);
                qv = (d < half_rope) ? fmaf(q_lo, cn, -q_hi * sn) : fmaf(q_lo, sn, q_hi * cn);
            }
            q_reg[k] = qv;
        }
    }

    // Online softmax accumulators.
    float m_val = -3.402823466e+38f;
    float s_val = 0.0f;
    float out_acc[FP_MAX_DPL];
    #pragma unroll
    for (unsigned int k = 0; k < FP_MAX_DPL; ++k) out_acc[k] = 0.0f;

    // Shared memory: K tile [FP_TILE × head_dim] u16 + V tile [FP_TILE × head_dim] u16.
    extern __shared__ unsigned short fp_smem_u16[];
    unsigned short* smem_k = fp_smem_u16;
    unsigned short* smem_v = fp_smem_u16 + FP_TILE * head_dim;

    const unsigned int wmask = 0xFFFFffffu;

    for (unsigned int tile_off = 0; tile_off < common_kv_len; tile_off += FP_TILE) {
        const unsigned int tile_len = min(FP_TILE, common_kv_len - tile_off);

        // Collaborative load K+V tile.
        __syncthreads();
        const unsigned int total_elems = tile_len * head_dim;
        for (unsigned int i = threadIdx.x; i < total_elems; i += FP_BLOCK) {
            const unsigned int t_local = i / head_dim;
            const unsigned int d = i - t_local * head_dim;
            const unsigned long long idx =
                (unsigned long long)(common_kv_start + tile_off + t_local) * row_stride
                + head_offset + d;
            smem_k[t_local * head_dim + d] = key_cache[idx];
            smem_v[t_local * head_dim + d] = value_cache[idx];
        }
        __syncthreads();

        if (!valid_q || dims_per_lane > FP_MAX_DPL) continue;

        // Per-token attention with online softmax.
        for (unsigned int t = 0; t < tile_len; ++t) {
            const unsigned int kv_pos = common_kv_start + tile_off + t;
            if (kv_pos > q_position) break;         // causal
            if (kv_pos < my_kv_start) continue;     // sliding window

            float dot = 0.0f;
            #pragma unroll
            for (unsigned int k = 0; k < FP_MAX_DPL; ++k) {
                if (k >= dims_per_lane) break;
                const unsigned int d = lane + (k << 5);
                if (d < head_dim) {
                    dot += q_reg[k] * __half2float(
                        *reinterpret_cast<const __half*>(&smem_k[t * head_dim + d]));
                }
            }
            for (unsigned int o = 16; o > 0; o >>= 1)
                dot += __shfl_down_sync(wmask, dot, o);
            const float score = __shfl_sync(wmask, dot, 0) * scale;

            const float m_new = fmaxf(m_val, score);
            const float alpha = expf(m_val - m_new);
            const float beta = expf(score - m_new);
            s_val = s_val * alpha + beta;
            m_val = m_new;

            #pragma unroll
            for (unsigned int k = 0; k < FP_MAX_DPL; ++k) {
                if (k >= dims_per_lane) break;
                const unsigned int d = lane + (k << 5);
                if (d < head_dim) {
                    const float v = __half2float(
                        *reinterpret_cast<const __half*>(&smem_v[t * head_dim + d]));
                    out_acc[k] = out_acc[k] * alpha + v * beta;
                }
            }
        }
    }

    if (!valid_q || dims_per_lane > FP_MAX_DPL) return;

    // Normalize and write.
    float* out_head = out + ((unsigned long long)q_idx * n_heads + head) * head_dim;
    const float inv_s = 1.0f / fmaxf(s_val, 1.0e-20f);
    #pragma unroll
    for (unsigned int k = 0; k < FP_MAX_DPL; ++k) {
        if (k >= dims_per_lane) break;
        const unsigned int d = lane + (k << 5);
        if (d < head_dim)
            out_head[d] = out_acc[k] * inv_s;
    }
}

// ============================================================
// I8 KV cache — flash prefill (per-head f32 scales)
// ============================================================
extern "C" __global__ __launch_bounds__(128)
void talu_flash_prefill_i8(
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
    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int head = blockIdx.y;
    const unsigned int q_start = blockIdx.x * FP_BR;
    const unsigned int q_idx = q_start + warp_id;
    const bool valid_q = (q_idx < q_rows);

    const unsigned int kv_head = head / kv_groups;
    const unsigned int head_offset = kv_head * head_dim;
    const unsigned int dims_per_lane = (head_dim + 31u) >> 5;

    const float log2_theta = log2f(theta);
    const unsigned int half_rope = rope_dim >> 1;
    const unsigned int q_position = position_base + q_idx;

    const unsigned int last_valid_q = min(q_start + FP_BR - 1u, q_rows - 1u);
    const unsigned int last_q_pos = position_base + last_valid_q;
    const unsigned int common_eff = min(seq_len, last_q_pos + 1u);
    const unsigned int first_q_pos = position_base + q_start;
    const unsigned int first_eff = min(seq_len, first_q_pos + 1u);
    const unsigned int common_kv_start = (sliding_window > 0u && first_eff > sliding_window)
        ? (first_eff - sliding_window) : 0u;
    const unsigned int common_kv_len = (common_eff > common_kv_start)
        ? (common_eff - common_kv_start) : 0u;

    const unsigned int my_eff = valid_q ? min(seq_len, q_position + 1u) : 0u;
    const unsigned int my_kv_start = (sliding_window > 0u && my_eff > sliding_window)
        ? (my_eff - sliding_window) : 0u;

    float q_reg[FP_MAX_DPL];
    #pragma unroll
    for (unsigned int k = 0; k < FP_MAX_DPL; ++k) q_reg[k] = 0.0f;
    if (valid_q && dims_per_lane <= FP_MAX_DPL) {
        const float* qh = query + ((unsigned long long)q_idx * n_heads + head) * head_dim;
        #pragma unroll
        for (unsigned int k = 0; k < FP_MAX_DPL; ++k) {
            if (k >= dims_per_lane) break;
            const unsigned int d = lane + (k << 5);
            if (d >= head_dim) continue;
            float qv = qh[d];
            if (d < rope_dim) {
                const unsigned int pair = (d < half_rope) ? d : (d - half_rope);
                const float q_lo = qh[pair];
                const float q_hi = qh[half_rope + pair];
                const float inv_freq = exp2f(log2_theta * (-2.0f * (float)pair / (float)rope_dim));
                const float angle = (float)q_position * inv_freq;
                float sn, cn;
                __sincosf(angle, &sn, &cn);
                qv = (d < half_rope) ? fmaf(q_lo, cn, -q_hi * sn) : fmaf(q_lo, sn, q_hi * cn);
            }
            q_reg[k] = qv;
        }
    }

    float m_val = -3.402823466e+38f;
    float s_val = 0.0f;
    float out_acc[FP_MAX_DPL];
    #pragma unroll
    for (unsigned int k = 0; k < FP_MAX_DPL; ++k) out_acc[k] = 0.0f;

    // Shared memory: K tile [FP_TILE × head_dim] i8 + V tile [FP_TILE × head_dim] i8.
    extern __shared__ signed char fp_smem_i8[];
    signed char* smem_k = fp_smem_i8;
    signed char* smem_v = fp_smem_i8 + FP_TILE * head_dim;

    const unsigned int wmask = 0xFFFFffffu;

    for (unsigned int tile_off = 0; tile_off < common_kv_len; tile_off += FP_TILE) {
        const unsigned int tile_len = min(FP_TILE, common_kv_len - tile_off);

        __syncthreads();
        const unsigned int total_elems = tile_len * head_dim;
        for (unsigned int i = threadIdx.x; i < total_elems; i += FP_BLOCK) {
            const unsigned int t_local = i / head_dim;
            const unsigned int d = i - t_local * head_dim;
            const unsigned long long idx =
                (unsigned long long)(common_kv_start + tile_off + t_local) * row_stride
                + head_offset + d;
            smem_k[t_local * head_dim + d] = key_cache[idx];
            smem_v[t_local * head_dim + d] = value_cache[idx];
        }
        __syncthreads();

        if (!valid_q || dims_per_lane > FP_MAX_DPL) continue;

        for (unsigned int t = 0; t < tile_len; ++t) {
            const unsigned int kv_pos = common_kv_start + tile_off + t;
            if (kv_pos > q_position) break;
            if (kv_pos < my_kv_start) continue;

            const float k_sc = k_scales[kv_pos * n_kv_heads + kv_head];
            const float v_sc = v_scales[kv_pos * n_kv_heads + kv_head];

            float dot = 0.0f;
            #pragma unroll
            for (unsigned int k = 0; k < FP_MAX_DPL; ++k) {
                if (k >= dims_per_lane) break;
                const unsigned int d = lane + (k << 5);
                if (d < head_dim)
                    dot += q_reg[k] * (float)(smem_k[t * head_dim + d]);
            }
            for (unsigned int o = 16; o > 0; o >>= 1)
                dot += __shfl_down_sync(wmask, dot, o);
            const float score = __shfl_sync(wmask, dot, 0) * k_sc * scale;

            const float m_new = fmaxf(m_val, score);
            const float alpha = expf(m_val - m_new);
            const float beta = expf(score - m_new);
            s_val = s_val * alpha + beta;
            m_val = m_new;

            #pragma unroll
            for (unsigned int k = 0; k < FP_MAX_DPL; ++k) {
                if (k >= dims_per_lane) break;
                const unsigned int d = lane + (k << 5);
                if (d < head_dim) {
                    const float v = (float)(smem_v[t * head_dim + d]) * v_sc;
                    out_acc[k] = out_acc[k] * alpha + v * beta;
                }
            }
        }
    }

    if (!valid_q || dims_per_lane > FP_MAX_DPL) return;

    float* out_head = out + ((unsigned long long)q_idx * n_heads + head) * head_dim;
    const float inv_s = 1.0f / fmaxf(s_val, 1.0e-20f);
    #pragma unroll
    for (unsigned int k = 0; k < FP_MAX_DPL; ++k) {
        if (k >= dims_per_lane) break;
        const unsigned int d = lane + (k << 5);
        if (d < head_dim)
            out_head[d] = out_acc[k] * inv_s;
    }
}

// ============================================================
// FP8 E4M3 KV cache — flash prefill (per-head f32 scales)
// ============================================================
#if __CUDA_ARCH__ >= 890

extern "C" __global__ __launch_bounds__(128)
void talu_flash_prefill_fp8(
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
    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int head = blockIdx.y;
    const unsigned int q_start = blockIdx.x * FP_BR;
    const unsigned int q_idx = q_start + warp_id;
    const bool valid_q = (q_idx < q_rows);

    const unsigned int kv_head = head / kv_groups;
    const unsigned int head_offset = kv_head * head_dim;
    const unsigned int dims_per_lane = (head_dim + 31u) >> 5;

    const float log2_theta = log2f(theta);
    const unsigned int half_rope = rope_dim >> 1;
    const unsigned int q_position = position_base + q_idx;

    const unsigned int last_valid_q = min(q_start + FP_BR - 1u, q_rows - 1u);
    const unsigned int last_q_pos = position_base + last_valid_q;
    const unsigned int common_eff = min(seq_len, last_q_pos + 1u);
    const unsigned int first_q_pos = position_base + q_start;
    const unsigned int first_eff = min(seq_len, first_q_pos + 1u);
    const unsigned int common_kv_start = (sliding_window > 0u && first_eff > sliding_window)
        ? (first_eff - sliding_window) : 0u;
    const unsigned int common_kv_len = (common_eff > common_kv_start)
        ? (common_eff - common_kv_start) : 0u;

    const unsigned int my_eff = valid_q ? min(seq_len, q_position + 1u) : 0u;
    const unsigned int my_kv_start = (sliding_window > 0u && my_eff > sliding_window)
        ? (my_eff - sliding_window) : 0u;

    float q_reg[FP_MAX_DPL];
    #pragma unroll
    for (unsigned int k = 0; k < FP_MAX_DPL; ++k) q_reg[k] = 0.0f;
    if (valid_q && dims_per_lane <= FP_MAX_DPL) {
        const float* qh = query + ((unsigned long long)q_idx * n_heads + head) * head_dim;
        #pragma unroll
        for (unsigned int k = 0; k < FP_MAX_DPL; ++k) {
            if (k >= dims_per_lane) break;
            const unsigned int d = lane + (k << 5);
            if (d >= head_dim) continue;
            float qv = qh[d];
            if (d < rope_dim) {
                const unsigned int pair = (d < half_rope) ? d : (d - half_rope);
                const float q_lo = qh[pair];
                const float q_hi = qh[half_rope + pair];
                const float inv_freq = exp2f(log2_theta * (-2.0f * (float)pair / (float)rope_dim));
                const float angle = (float)q_position * inv_freq;
                float sn, cn;
                __sincosf(angle, &sn, &cn);
                qv = (d < half_rope) ? fmaf(q_lo, cn, -q_hi * sn) : fmaf(q_lo, sn, q_hi * cn);
            }
            q_reg[k] = qv;
        }
    }

    float m_val = -3.402823466e+38f;
    float s_val = 0.0f;
    float out_acc[FP_MAX_DPL];
    #pragma unroll
    for (unsigned int k = 0; k < FP_MAX_DPL; ++k) out_acc[k] = 0.0f;

    extern __shared__ unsigned char fp_smem_u8[];
    unsigned char* smem_k = fp_smem_u8;
    unsigned char* smem_v = fp_smem_u8 + FP_TILE * head_dim;

    const unsigned int wmask = 0xFFFFffffu;

    for (unsigned int tile_off = 0; tile_off < common_kv_len; tile_off += FP_TILE) {
        const unsigned int tile_len = min(FP_TILE, common_kv_len - tile_off);

        __syncthreads();
        const unsigned int total_elems = tile_len * head_dim;
        for (unsigned int i = threadIdx.x; i < total_elems; i += FP_BLOCK) {
            const unsigned int t_local = i / head_dim;
            const unsigned int d = i - t_local * head_dim;
            const unsigned long long idx =
                (unsigned long long)(common_kv_start + tile_off + t_local) * row_stride
                + head_offset + d;
            smem_k[t_local * head_dim + d] = key_cache[idx];
            smem_v[t_local * head_dim + d] = value_cache[idx];
        }
        __syncthreads();

        if (!valid_q || dims_per_lane > FP_MAX_DPL) continue;

        for (unsigned int t = 0; t < tile_len; ++t) {
            const unsigned int kv_pos = common_kv_start + tile_off + t;
            if (kv_pos > q_position) break;
            if (kv_pos < my_kv_start) continue;

            const float k_sc = k_scales[kv_pos * n_kv_heads + kv_head];
            const float v_sc = v_scales[kv_pos * n_kv_heads + kv_head];

            float dot = 0.0f;
            #pragma unroll
            for (unsigned int k = 0; k < FP_MAX_DPL; ++k) {
                if (k >= dims_per_lane) break;
                const unsigned int d = lane + (k << 5);
                if (d < head_dim) {
                    dot += q_reg[k] * fd_fp8e4m3_to_f32(
                        static_cast<__nv_fp8_storage_t>(smem_k[t * head_dim + d]));
                }
            }
            for (unsigned int o = 16; o > 0; o >>= 1)
                dot += __shfl_down_sync(wmask, dot, o);
            const float score = __shfl_sync(wmask, dot, 0) * k_sc * scale;

            const float m_new = fmaxf(m_val, score);
            const float alpha = expf(m_val - m_new);
            const float beta = expf(score - m_new);
            s_val = s_val * alpha + beta;
            m_val = m_new;

            #pragma unroll
            for (unsigned int k = 0; k < FP_MAX_DPL; ++k) {
                if (k >= dims_per_lane) break;
                const unsigned int d = lane + (k << 5);
                if (d < head_dim) {
                    const float v = fd_fp8e4m3_to_f32(
                        static_cast<__nv_fp8_storage_t>(smem_v[t * head_dim + d])) * v_sc;
                    out_acc[k] = out_acc[k] * alpha + v * beta;
                }
            }
        }
    }

    if (!valid_q || dims_per_lane > FP_MAX_DPL) return;

    float* out_head = out + ((unsigned long long)q_idx * n_heads + head) * head_dim;
    const float inv_s = 1.0f / fmaxf(s_val, 1.0e-20f);
    #pragma unroll
    for (unsigned int k = 0; k < FP_MAX_DPL; ++k) {
        if (k >= dims_per_lane) break;
        const unsigned int d = lane + (k << 5);
        if (d < head_dim)
            out_head[d] = out_acc[k] * inv_s;
    }
}

#endif // __CUDA_ARCH__ >= 890 (flash prefill fp8)
