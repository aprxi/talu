#include <cuda_fp16.h>

extern "C" __global__ void talu_vector_add_f32(
    float* out,
    const float* a,
    const float* b,
    unsigned int count
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    out[index] = a[index] + b[index];
}

extern "C" __global__ void talu_mul_f32(
    float* out,
    const float* a,
    const float* b,
    unsigned int count
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    out[index] = a[index] * b[index];
}

extern "C" __global__ void talu_copy_f32(
    float* out,
    const float* input,
    unsigned int count
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    out[index] = input[index];
}

extern "C" __global__ void talu_copy_u16(
    unsigned short* out,
    const unsigned short* input,
    unsigned int count
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    out[index] = input[index];
}

extern "C" __global__ void talu_cast_f32_to_f16(
    unsigned short* out,
    const float* input,
    unsigned int count
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const __half h = __float2half_rn(input[index]);
    out[index] = __half_as_ushort(h);
}

extern "C" __global__ void talu_rmsnorm_f32(
    float* out,
    const float* input,
    const float* weight,
    unsigned int rows,
    unsigned int cols,
    float eps,
    float weight_offset
) {
    const unsigned int row = blockIdx.x;
    if (row >= rows) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int base = row * cols;

    // Simple reference kernel: lane 0 computes row RMS; all lanes normalize.
    __shared__ float inv_rms;
    if (tid == 0) {
        float sum_sq = 0.0f;
        for (unsigned int i = 0; i < cols; ++i) {
            const float v = input[base + i];
            sum_sq += v * v;
        }
        const float mean_sq = sum_sq / (float)cols;
        inv_rms = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    for (unsigned int col = tid; col < cols; col += blockDim.x) {
        const float normalized = input[base + col] * inv_rms;
        out[base + col] = normalized * (weight[col] + weight_offset);
    }
}

extern "C" __global__ void talu_rope_f32(
    float* io,
    unsigned int n_heads,
    unsigned int head_dim,
    unsigned int rope_dim,
    unsigned int position,
    float theta
) {
    // Llama/Qwen-style half-rotation layout: first half rotates with second half.
    const unsigned int half = rope_dim >> 1;
    const unsigned int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total_pairs = n_heads * half;
    if (pair_index >= total_pairs) return;

    const unsigned int head = pair_index / half;
    const unsigned int pair = pair_index % half;
    const unsigned int base = head * head_dim;

    const float inv_freq = powf(theta, -2.0f * (float)pair / (float)rope_dim);
    const float angle = (float)position * inv_freq;
    const float c = cosf(angle);
    const float s = sinf(angle);
    const unsigned int lo_idx = base + pair;
    const unsigned int hi_idx = base + half + pair;
    const float x0 = io[lo_idx];
    const float x1 = io[hi_idx];
    io[lo_idx] = x0 * c - x1 * s;
    io[hi_idx] = x0 * s + x1 * c;
}

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
    float scale
) {
    const unsigned int head = blockIdx.x;
    const unsigned int lane = threadIdx.x & 31u;
    if (head >= n_heads || kv_groups == 0 || seq_len == 0 || head_dim == 0) return;

    const unsigned int kv_head = head / kv_groups;
    const unsigned int head_offset = kv_head * head_dim;
    const float* query_head = query + (unsigned long long)head * head_dim;
    const unsigned int mask = 0xFFFFffffu;

    // One warp per head. Each lane owns dimensions d = lane + k*32.
    // Use online softmax update to avoid score/prob buffers and avoid recomputing dots.
    const unsigned int dims_per_lane = (head_dim + 31u) >> 5;
    float out_acc[16];
    #pragma unroll
    for (unsigned int i = 0; i < 16; ++i) out_acc[i] = 0.0f;
    if (dims_per_lane > 16u) return;

    float m = -3.402823466e+38f;
    float s = 0.0f;

    for (unsigned int t = 0; t < seq_len; ++t) {
        const unsigned short* key_row = key_cache + ((unsigned long long)t * row_stride + head_offset);
        float partial = 0.0f;
        for (unsigned int d = lane; d < head_dim; d += 32u) {
            partial += query_head[d] * __half2float(*reinterpret_cast<const __half*>(&key_row[d]));
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

extern "C" __global__ void talu_silu_f32(
    float* out,
    const float* input,
    unsigned int count
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float x = input[index];
    out[index] = x / (1.0f + expf(-x));
}

extern "C" __global__ void talu_argmax_f32(
    const float* input,
    unsigned int count,
    unsigned int* out_index
) {
    const unsigned int tid = threadIdx.x;
    float best_val = -3.402823466e+38f;
    unsigned int best_idx = 0;

    for (unsigned int idx = tid; idx < count; idx += blockDim.x) {
        const float v = input[idx];
        if (v > best_val || (v == best_val && idx < best_idx)) {
            best_val = v;
            best_idx = idx;
        }
    }

    __shared__ float shared_val[256];
    __shared__ unsigned int shared_idx[256];
    shared_val[tid] = best_val;
    shared_idx[tid] = best_idx;
    __syncthreads();

    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other_val = shared_val[tid + stride];
            const unsigned int other_idx = shared_idx[tid + stride];
            const float cur_val = shared_val[tid];
            const unsigned int cur_idx = shared_idx[tid];
            if (other_val > cur_val || (other_val == cur_val && other_idx < cur_idx)) {
                shared_val[tid] = other_val;
                shared_idx[tid] = other_idx;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_index[0] = shared_idx[0];
    }
}

static __device__ __forceinline__ float talu_decode_f16_u16(unsigned short raw) {
    return __half2float(*reinterpret_cast<const __half*>(&raw));
}

static __device__ __forceinline__ float talu_decode_bf16_u16(unsigned short raw) {
    const unsigned int bits = static_cast<unsigned int>(raw) << 16;
    return __uint_as_float(bits);
}

static __device__ __forceinline__ float talu_decode_scale_bias_u16(unsigned short raw, unsigned int dtype_tag) {
    // dtype_tag: 0 => f16, 1 => bf16
    return (dtype_tag == 0) ? talu_decode_f16_u16(raw) : talu_decode_bf16_u16(raw);
}

extern "C" __global__ void talu_matvec_f16_f32(
    const float* input,
    const unsigned short* weight,
    float* out,
    unsigned int in_dim,
    unsigned int out_dim
) {
    extern __shared__ float shared_input[];
    const unsigned int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = out_idx < out_dim;
    if (in_dim == 0) {
        if (active) out[out_idx] = 0.0f;
        return;
    }

    float acc = 0.0f;
    for (unsigned int tile_base = 0; tile_base < in_dim; tile_base += blockDim.x) {
        const unsigned int in_idx = tile_base + threadIdx.x;
        shared_input[threadIdx.x] = (in_idx < in_dim) ? input[in_idx] : 0.0f;
        __syncthreads();

        if (active) {
            const unsigned int tile_remaining = in_dim - tile_base;
            const unsigned int tile_count = (tile_remaining < blockDim.x) ? tile_remaining : blockDim.x;
            const unsigned long long row_base = (unsigned long long)out_idx * in_dim + tile_base;

            #pragma unroll 4
            for (unsigned int k = 0; k < tile_count; ++k) {
                const unsigned long long w_idx = row_base + k;
                const float w = talu_decode_f16_u16(weight[w_idx]);
                acc = fmaf(shared_input[k], w, acc);
            }
        }
        __syncthreads();
    }
    if (active) out[out_idx] = acc;
}

extern "C" __global__ void talu_matvec_bf16_f32(
    const float* input,
    const unsigned short* weight,
    float* out,
    unsigned int in_dim,
    unsigned int out_dim
) {
    extern __shared__ float shared_input[];
    const unsigned int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = out_idx < out_dim;
    if (in_dim == 0) {
        if (active) out[out_idx] = 0.0f;
        return;
    }

    float acc = 0.0f;
    for (unsigned int tile_base = 0; tile_base < in_dim; tile_base += blockDim.x) {
        const unsigned int in_idx = tile_base + threadIdx.x;
        shared_input[threadIdx.x] = (in_idx < in_dim) ? input[in_idx] : 0.0f;
        __syncthreads();

        if (active) {
            const unsigned int tile_remaining = in_dim - tile_base;
            const unsigned int tile_count = (tile_remaining < blockDim.x) ? tile_remaining : blockDim.x;
            const unsigned long long row_base = (unsigned long long)out_idx * in_dim + tile_base;

            #pragma unroll 4
            for (unsigned int k = 0; k < tile_count; ++k) {
                const unsigned long long w_idx = row_base + k;
                const float w = talu_decode_bf16_u16(weight[w_idx]);
                acc = fmaf(shared_input[k], w, acc);
            }
        }
        __syncthreads();
    }
    if (active) out[out_idx] = acc;
}

extern "C" __global__ void talu_gaffine_u4_matvec_f32(
    const float* input,
    const unsigned int* packed_weight,
    const unsigned short* scales,
    const unsigned short* biases,
    float* out,
    unsigned int in_dim,
    unsigned int out_dim,
    unsigned int group_size,
    unsigned int scales_dtype_tag
) {
    const unsigned int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= out_dim) return;
    if (group_size == 0 || (in_dim % group_size) != 0 || (group_size % 8) != 0) {
        return;
    }

    const unsigned int groups_per_row = in_dim / group_size;
    const unsigned int words_per_group = group_size / 8;
    const unsigned int words_per_row = in_dim / 8;
    const unsigned int row_word_base = out_idx * words_per_row;
    const unsigned int row_sb_base = out_idx * groups_per_row;

    float acc = 0.0f;
    for (unsigned int group_idx = 0; group_idx < groups_per_row; ++group_idx) {
        const float scale = talu_decode_scale_bias_u16(scales[row_sb_base + group_idx], scales_dtype_tag);
        const float bias = talu_decode_scale_bias_u16(biases[row_sb_base + group_idx], scales_dtype_tag);
        const unsigned int group_input_base = group_idx * group_size;
        const unsigned int group_word_base = row_word_base + group_idx * words_per_group;

        for (unsigned int w = 0; w < words_per_group; ++w) {
            const unsigned int packed = packed_weight[group_word_base + w];
            const unsigned int value_base = group_input_base + w * 8;

            #pragma unroll
            for (unsigned int nib = 0; nib < 8; ++nib) {
                const unsigned int quant = (packed >> (nib * 4)) & 0xF;
                const float dequant = static_cast<float>(quant) * scale + bias;
                acc += input[value_base + nib] * dequant;
            }
        }
    }

    out[out_idx] = acc;
}

static __device__ __forceinline__ float talu_gaffine_u4_dot_row(
    const float* input,
    const unsigned int* packed_weight,
    const unsigned short* scales,
    const unsigned short* biases,
    unsigned int in_dim,
    unsigned int out_idx,
    unsigned int group_size,
    unsigned int scales_dtype_tag
) {
    const unsigned int groups_per_row = in_dim / group_size;
    const unsigned int words_per_group = group_size / 8;
    const unsigned int words_per_row = in_dim / 8;
    const unsigned int row_word_base = out_idx * words_per_row;
    const unsigned int row_sb_base = out_idx * groups_per_row;

    float acc = 0.0f;
    for (unsigned int group_idx = 0; group_idx < groups_per_row; ++group_idx) {
        const float scale = talu_decode_scale_bias_u16(scales[row_sb_base + group_idx], scales_dtype_tag);
        const float bias = talu_decode_scale_bias_u16(biases[row_sb_base + group_idx], scales_dtype_tag);
        const unsigned int group_input_base = group_idx * group_size;
        const unsigned int group_word_base = row_word_base + group_idx * words_per_group;

        for (unsigned int w = 0; w < words_per_group; ++w) {
            const unsigned int packed = packed_weight[group_word_base + w];
            const unsigned int value_base = group_input_base + w * 8;

            #pragma unroll
            for (unsigned int nib = 0; nib < 8; ++nib) {
                const unsigned int quant = (packed >> (nib * 4)) & 0xF;
                const float dequant = static_cast<float>(quant) * scale + bias;
                acc += input[value_base + nib] * dequant;
            }
        }
    }
    return acc;
}

extern "C" __global__ void talu_gaffine_u4_matvec_qkv_f32(
    const float* input,
    const unsigned int* q_packed_weight,
    const unsigned short* q_scales,
    const unsigned short* q_biases,
    float* q_out,
    unsigned int q_out_dim,
    unsigned int q_group_size,
    unsigned int q_scales_dtype_tag,
    const unsigned int* k_packed_weight,
    const unsigned short* k_scales,
    const unsigned short* k_biases,
    float* k_out,
    unsigned int k_out_dim,
    unsigned int k_group_size,
    unsigned int k_scales_dtype_tag,
    const unsigned int* v_packed_weight,
    const unsigned short* v_scales,
    const unsigned short* v_biases,
    float* v_out,
    unsigned int v_out_dim,
    unsigned int v_group_size,
    unsigned int v_scales_dtype_tag,
    unsigned int in_dim
) {
    if (in_dim == 0 || (in_dim % 8) != 0) return;

    const unsigned int out_index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int qk_dim = q_out_dim + k_out_dim;
    const unsigned int total_dim = qk_dim + v_out_dim;
    if (out_index >= total_dim) return;

    if (out_index < q_out_dim) {
        if (q_group_size == 0 || (in_dim % q_group_size) != 0 || (q_group_size % 8) != 0) return;
        q_out[out_index] = talu_gaffine_u4_dot_row(
            input,
            q_packed_weight,
            q_scales,
            q_biases,
            in_dim,
            out_index,
            q_group_size,
            q_scales_dtype_tag
        );
        return;
    }

    if (out_index < qk_dim) {
        if (k_group_size == 0 || (in_dim % k_group_size) != 0 || (k_group_size % 8) != 0) return;
        const unsigned int k_row = out_index - q_out_dim;
        k_out[k_row] = talu_gaffine_u4_dot_row(
            input,
            k_packed_weight,
            k_scales,
            k_biases,
            in_dim,
            k_row,
            k_group_size,
            k_scales_dtype_tag
        );
        return;
    }

    if (v_group_size == 0 || (in_dim % v_group_size) != 0 || (v_group_size % 8) != 0) return;
    const unsigned int v_row = out_index - qk_dim;
    v_out[v_row] = talu_gaffine_u4_dot_row(
        input,
        v_packed_weight,
        v_scales,
        v_biases,
        in_dim,
        v_row,
        v_group_size,
        v_scales_dtype_tag
    );
}

extern "C" __global__ void talu_gaffine_u4_matvec_gate_up_f32(
    const float* input,
    const unsigned int* gate_packed_weight,
    const unsigned short* gate_scales,
    const unsigned short* gate_biases,
    float* gate_out,
    unsigned int gate_out_dim,
    unsigned int gate_group_size,
    unsigned int gate_scales_dtype_tag,
    const unsigned int* up_packed_weight,
    const unsigned short* up_scales,
    const unsigned short* up_biases,
    float* up_out,
    unsigned int up_out_dim,
    unsigned int up_group_size,
    unsigned int up_scales_dtype_tag,
    unsigned int in_dim
) {
    if (in_dim == 0 || (in_dim % 8) != 0) return;

    const unsigned int out_index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total_dim = gate_out_dim + up_out_dim;
    if (out_index >= total_dim) return;

    if (out_index < gate_out_dim) {
        if (gate_group_size == 0 || (in_dim % gate_group_size) != 0 || (gate_group_size % 8) != 0) return;
        gate_out[out_index] = talu_gaffine_u4_dot_row(
            input,
            gate_packed_weight,
            gate_scales,
            gate_biases,
            in_dim,
            out_index,
            gate_group_size,
            gate_scales_dtype_tag
        );
        return;
    }

    if (up_group_size == 0 || (in_dim % up_group_size) != 0 || (up_group_size % 8) != 0) return;
    const unsigned int up_row = out_index - gate_out_dim;
    up_out[up_row] = talu_gaffine_u4_dot_row(
        input,
        up_packed_weight,
        up_scales,
        up_biases,
        in_dim,
        up_row,
        up_group_size,
        up_scales_dtype_tag
    );
}
