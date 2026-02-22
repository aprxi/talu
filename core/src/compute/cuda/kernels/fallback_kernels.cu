extern "C" __global__ void talu_vector_add_f32_v1(
    float* out,
    const float* a,
    const float* b,
    unsigned int count
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    out[index] = a[index] + b[index];
}

extern "C" __global__ void talu_mul_f32_v1(
    float* out,
    const float* a,
    const float* b,
    unsigned int count
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    out[index] = a[index] * b[index];
}

extern "C" __global__ void talu_copy_f32_v1(
    float* out,
    const float* input,
    unsigned int count
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    out[index] = input[index];
}

extern "C" __global__ void talu_rmsnorm_f32_v1(
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

extern "C" __global__ void talu_rope_f32_v1(
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

extern "C" __global__ void talu_attn_scores_f32_v1(
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

extern "C" __global__ void talu_softmax_f32_v1(
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

extern "C" __global__ void talu_attn_weighted_sum_f32_v1(
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

extern "C" __global__ void talu_silu_f32_v1(
    float* out,
    const float* input,
    unsigned int count
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float x = input[index];
    out[index] = x / (1.0f + expf(-x));
}

extern "C" __global__ void talu_argmax_f32_v1(
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
