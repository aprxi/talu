__inline__ __device__ float talu_warp_reduce_sum(float value) {
    #pragma unroll
    for (unsigned int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xFFFFffffu, value, offset);
    }
    return value;
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
    const float* row_in = input + (unsigned long long)row * cols;
    float* row_out = out + (unsigned long long)row * cols;

    float sum_sq = 0.0f;
    const unsigned int cols4 = cols / 4;
    const bool vectorizable = ((reinterpret_cast<uintptr_t>(row_in) & 0xFu) == 0u);

    if (vectorizable) {
        const float4* row_in4 = reinterpret_cast<const float4*>(row_in);
        for (unsigned int i = tid; i < cols4; i += blockDim.x) {
            const float4 v = row_in4[i];
            sum_sq = fmaf(v.x, v.x, sum_sq);
            sum_sq = fmaf(v.y, v.y, sum_sq);
            sum_sq = fmaf(v.z, v.z, sum_sq);
            sum_sq = fmaf(v.w, v.w, sum_sq);
        }
        for (unsigned int i = cols4 * 4 + tid; i < cols; i += blockDim.x) {
            const float v = row_in[i];
            sum_sq = fmaf(v, v, sum_sq);
        }
    } else {
        for (unsigned int i = tid; i < cols; i += blockDim.x) {
            const float v = row_in[i];
            sum_sq = fmaf(v, v, sum_sq);
        }
    }

    const float lane_sum = talu_warp_reduce_sum(sum_sq);
    __shared__ float warp_sums[32];
    if ((tid & 31u) == 0u) {
        warp_sums[tid >> 5] = lane_sum;
    }
    __syncthreads();

    const unsigned int warp_count = (blockDim.x + 31u) >> 5;
    float block_sum = (tid < warp_count) ? warp_sums[tid] : 0.0f;
    if (tid < 32u) {
        block_sum = talu_warp_reduce_sum(block_sum);
    }

    __shared__ float inv_rms;
    if (tid == 0) {
        const float mean_sq = block_sum / (float)cols;
        inv_rms = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    const float irms = inv_rms;
    if (vectorizable and ((reinterpret_cast<uintptr_t>(row_out) & 0xFu) == 0u) and
        ((reinterpret_cast<uintptr_t>(weight) & 0xFu) == 0u))
    {
        const float4* row_in4 = reinterpret_cast<const float4*>(row_in);
        float4* row_out4 = reinterpret_cast<float4*>(row_out);
        const float4* weight4 = reinterpret_cast<const float4*>(weight);
        for (unsigned int i = tid; i < cols4; i += blockDim.x) {
            const float4 in_v = row_in4[i];
            const float4 w_v = weight4[i];
            float4 out_v;
            out_v.x = in_v.x * irms * (w_v.x + weight_offset);
            out_v.y = in_v.y * irms * (w_v.y + weight_offset);
            out_v.z = in_v.z * irms * (w_v.z + weight_offset);
            out_v.w = in_v.w * irms * (w_v.w + weight_offset);
            row_out4[i] = out_v;
        }
        for (unsigned int i = cols4 * 4 + tid; i < cols; i += blockDim.x) {
            row_out[i] = row_in[i] * irms * (weight[i] + weight_offset);
        }
    } else {
        for (unsigned int i = tid; i < cols; i += blockDim.x) {
            row_out[i] = row_in[i] * irms * (weight[i] + weight_offset);
        }
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
    float s = 0.0f;
    float c = 0.0f;
    __sincosf(angle, &s, &c);
    const unsigned int lo_idx = base + pair;
    const unsigned int hi_idx = base + half + pair;
    const float x0 = io[lo_idx];
    const float x1 = io[hi_idx];
    io[lo_idx] = fmaf(x0, c, -x1 * s);
    io[hi_idx] = fmaf(x0, s, x1 * c);
}

extern "C" __global__ void talu_rope_store_f16(
    unsigned short* out_f16,
    const float* input_f32,
    unsigned int n_heads,
    unsigned int head_dim,
    unsigned int rope_dim,
    unsigned int position,
    float theta
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = n_heads * head_dim;
    if (idx >= total) return;

    const unsigned int head = idx / head_dim;
    const unsigned int dim = idx % head_dim;
    const unsigned int base = head * head_dim;

    float out_v = input_f32[idx];
    if (dim < rope_dim) {
        const unsigned int half = rope_dim >> 1;
        const unsigned int pair = (dim < half) ? dim : (dim - half);
        const unsigned int lo_idx = base + pair;
        const unsigned int hi_idx = base + half + pair;
        const float x0 = input_f32[lo_idx];
        const float x1 = input_f32[hi_idx];
        const float inv_freq = powf(theta, -2.0f * (float)pair / (float)rope_dim);
        const float angle = (float)position * inv_freq;
        float s = 0.0f;
        float c = 0.0f;
        __sincosf(angle, &s, &c);
        out_v = (dim < half) ? fmaf(x0, c, -x1 * s) : fmaf(x0, s, x1 * c);
    }

    __half* out = reinterpret_cast<__half*>(out_f16);
    out[idx] = __float2half_rn(out_v);
}

extern "C" __global__ void talu_kv_write_f16(
    unsigned short* out_k_f16,
    unsigned short* out_v_f16,
    const float* input_k_f32,
    const float* input_v_f32,
    unsigned int n_heads,
    unsigned int head_dim,
    unsigned int rope_dim,
    unsigned int position,
    float theta
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = n_heads * head_dim;
    if (idx >= total) return;

    const unsigned int head = idx / head_dim;
    const unsigned int dim = idx % head_dim;
    const unsigned int base = head * head_dim;

    float k_out = input_k_f32[idx];
    if (dim < rope_dim) {
        const unsigned int half = rope_dim >> 1;
        const unsigned int pair = (dim < half) ? dim : (dim - half);
        const unsigned int lo_idx = base + pair;
        const unsigned int hi_idx = base + half + pair;
        const float x0 = input_k_f32[lo_idx];
        const float x1 = input_k_f32[hi_idx];
        const float inv_freq = powf(theta, -2.0f * (float)pair / (float)rope_dim);
        const float angle = (float)position * inv_freq;
        float s = 0.0f;
        float c = 0.0f;
        __sincosf(angle, &s, &c);
        k_out = (dim < half) ? fmaf(x0, c, -x1 * s) : fmaf(x0, s, x1 * c);
    }

    __half* out_k = reinterpret_cast<__half*>(out_k_f16);
    __half* out_v = reinterpret_cast<__half*>(out_v_f16);
    out_k[idx] = __float2half_rn(k_out);
    out_v[idx] = __float2half_rn(input_v_f32[idx]);
}
