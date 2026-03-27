__inline__ __device__ float talu_warp_reduce_sum(float value) {
    #pragma unroll
    for (unsigned int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xFFFFffffu, value, offset);
    }
    return value;
}

__inline__ __device__ float talu_warp_reduce_max(float value) {
    #pragma unroll
    for (unsigned int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xFFFFffffu, value, offset));
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
    const float log2_theta = log2f(theta);
    const unsigned int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total_pairs = n_heads * half;
    if (pair_index >= total_pairs) return;

    const unsigned int head = pair_index / half;
    const unsigned int pair = pair_index % half;
    const unsigned int base = head * head_dim;

    const float inv_freq = exp2f(log2_theta * (-2.0f * (float)pair / (float)rope_dim));
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
    const float log2_theta = log2f(theta);

    float out_v = input_f32[idx];
    if (dim < rope_dim) {
        const unsigned int half = rope_dim >> 1;
        const unsigned int pair = (dim < half) ? dim : (dim - half);
        const unsigned int lo_idx = base + pair;
        const unsigned int hi_idx = base + half + pair;
        const float x0 = input_f32[lo_idx];
        const float x1 = input_f32[hi_idx];
        const float inv_freq = exp2f(log2_theta * (-2.0f * (float)pair / (float)rope_dim));
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
    const float log2_theta = log2f(theta);

    float k_out = input_k_f32[idx];
    if (dim < rope_dim) {
        const unsigned int half = rope_dim >> 1;
        const unsigned int pair = (dim < half) ? dim : (dim - half);
        const unsigned int lo_idx = base + pair;
        const unsigned int hi_idx = base + half + pair;
        const float x0 = input_k_f32[lo_idx];
        const float x1 = input_k_f32[hi_idx];
        const float inv_freq = exp2f(log2_theta * (-2.0f * (float)pair / (float)rope_dim));
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

extern "C" __global__ void talu_kv_write_f16_rows(
    unsigned short* out_k_f16,
    unsigned short* out_v_f16,
    const float* input_k_f32,
    const float* input_v_f32,
    unsigned int n_heads,
    unsigned int head_dim,
    unsigned int rope_dim,
    unsigned int q_rows,
    unsigned int row_stride,
    unsigned int position_base,
    float theta
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row_width = n_heads * head_dim;
    const unsigned int total = q_rows * row_width;
    if (idx >= total) return;

    const unsigned int row = idx / row_width;
    const unsigned int row_offset = idx - (row * row_width);
    const unsigned int head = row_offset / head_dim;
    const unsigned int dim = row_offset % head_dim;
    const unsigned int row_base = row * row_width;
    const unsigned int head_base = row_base + (head * head_dim);
    const float log2_theta = log2f(theta);

    float k_out = input_k_f32[idx];
    if (dim < rope_dim) {
        const unsigned int half = rope_dim >> 1;
        const unsigned int pair = (dim < half) ? dim : (dim - half);
        const unsigned int lo_idx = head_base + pair;
        const unsigned int hi_idx = head_base + half + pair;
        const float x0 = input_k_f32[lo_idx];
        const float x1 = input_k_f32[hi_idx];
        const float inv_freq = exp2f(log2_theta * (-2.0f * (float)pair / (float)rope_dim));
        const float angle = (float)(position_base + row) * inv_freq;
        float s = 0.0f;
        float c = 0.0f;
        __sincosf(angle, &s, &c);
        k_out = (dim < half) ? fmaf(x0, c, -x1 * s) : fmaf(x0, s, x1 * c);
    }

    const unsigned int out_idx = row * row_stride + row_offset;
    __half* out_k = reinterpret_cast<__half*>(out_k_f16);
    __half* out_v = reinterpret_cast<__half*>(out_v_f16);
    out_k[out_idx] = __float2half_rn(k_out);
    out_v[out_idx] = __float2half_rn(input_v_f32[idx]);
}

extern "C" __global__ void talu_kv_write_f16_rows_ptrs(
    const unsigned long long* out_k_ptrs,
    const unsigned long long* out_v_ptrs,
    const unsigned int* positions,
    const float* input_k_f32,
    const float* input_v_f32,
    unsigned int n_heads,
    unsigned int head_dim,
    unsigned int rope_dim,
    unsigned int q_rows,
    unsigned int row_stride,
    float theta
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row_width = n_heads * head_dim;
    const unsigned int total = q_rows * row_width;
    if (idx >= total) return;

    const unsigned int row = idx / row_width;
    const unsigned int row_offset = idx - (row * row_width);
    const unsigned int head = row_offset / head_dim;
    const unsigned int dim = row_offset % head_dim;
    const unsigned int row_base = row * row_width;
    const unsigned int head_base = row_base + (head * head_dim);
    const float log2_theta = log2f(theta);

    float k_out = input_k_f32[idx];
    if (dim < rope_dim) {
        const unsigned int half = rope_dim >> 1;
        const unsigned int pair = (dim < half) ? dim : (dim - half);
        const unsigned int lo_idx = head_base + pair;
        const unsigned int hi_idx = head_base + half + pair;
        const float x0 = input_k_f32[lo_idx];
        const float x1 = input_k_f32[hi_idx];
        const float inv_freq = exp2f(log2_theta * (-2.0f * (float)pair / (float)rope_dim));
        const float angle = (float)positions[row] * inv_freq;
        float s = 0.0f;
        float c = 0.0f;
        __sincosf(angle, &s, &c);
        k_out = (dim < half) ? fmaf(x0, c, -x1 * s) : fmaf(x0, s, x1 * c);
    }

    __half* out_k = reinterpret_cast<__half*>(out_k_ptrs[row]);
    __half* out_v = reinterpret_cast<__half*>(out_v_ptrs[row]);
    if (out_k == nullptr || out_v == nullptr) return;

    const unsigned int out_idx = positions[row] * row_stride + row_offset;
    out_k[out_idx] = __float2half_rn(k_out);
    out_v[out_idx] = __float2half_rn(input_v_f32[idx]);
}

// --- INT8 KV quantization kernels ---
// Per-head symmetric quantization: scale = max_abs / 127, val = round(x / scale).
// Grid: (n_heads, ...), Block: (256,). One block per head (per row for multi-row).
// Each thread caches up to 4 elements in registers between the max_abs reduction
// and quantization passes, avoiding recomputation.

// Single-token KV write: RoPE on K, quantize both K and V to INT8.
// Grid: (n_heads,), Block: (256,)
extern "C" __global__ void talu_kv_write_i8(
    signed char* out_k_i8,
    signed char* out_v_i8,
    float* k_scales,
    float* v_scales,
    const float* input_k_f32,
    const float* input_v_f32,
    unsigned int n_heads,
    unsigned int head_dim,
    unsigned int rope_dim,
    unsigned int position,
    float theta
) {
    const unsigned int head = blockIdx.x;
    if (head >= n_heads) return;

    const unsigned int base = head * head_dim;
    const float log2_theta = log2f(theta);
    const unsigned int half_rd = rope_dim >> 1;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp_id = tid / 32u;
    const unsigned int num_warps = (blockDim.x + 31u) / 32u;

    // Pass 1: compute RoPE'd K and raw V, find per-head max_abs.
    float k_vals[4];
    float v_vals[4];
    float k_max = 0.0f;
    float v_max = 0.0f;
    unsigned int count = 0;

    for (unsigned int d = tid; d < head_dim; d += blockDim.x) {
        float k_val = input_k_f32[base + d];
        if (d < rope_dim) {
            const unsigned int pair = (d < half_rd) ? d : (d - half_rd);
            const float x0 = input_k_f32[base + pair];
            const float x1 = input_k_f32[base + half_rd + pair];
            const float inv_freq = exp2f(log2_theta * (-2.0f * (float)pair / (float)rope_dim));
            const float angle = (float)position * inv_freq;
            float s, c;
            __sincosf(angle, &s, &c);
            k_val = (d < half_rd) ? fmaf(x0, c, -x1 * s) : fmaf(x0, s, x1 * c);
        }
        k_vals[count] = k_val;
        v_vals[count] = input_v_f32[base + d];
        k_max = fmaxf(k_max, fabsf(k_val));
        v_max = fmaxf(v_max, fabsf(v_vals[count]));
        count++;
    }

    // Warp-level max reduction.
    k_max = talu_warp_reduce_max(k_max);
    v_max = talu_warp_reduce_max(v_max);

    // Cross-warp reduction via shared memory.
    __shared__ float smem[16]; // [0..7] k_max per warp, [8..15] v_max
    if (lane == 0) {
        smem[warp_id] = k_max;
        smem[warp_id + 8] = v_max;
    }
    __syncthreads();

    if (warp_id == 0) {
        k_max = (lane < num_warps) ? smem[lane] : 0.0f;
        v_max = (lane < num_warps) ? smem[lane + 8] : 0.0f;
        k_max = talu_warp_reduce_max(k_max);
        v_max = talu_warp_reduce_max(v_max);
    }

    // Broadcast scales.
    __shared__ float k_scale_s, v_scale_s;
    if (tid == 0) {
        k_scale_s = (k_max > 0.0f) ? (k_max / 127.0f) : 0.0f;
        v_scale_s = (v_max > 0.0f) ? (v_max / 127.0f) : 0.0f;
        k_scales[head] = k_scale_s;
        v_scales[head] = v_scale_s;
    }
    __syncthreads();

    // Pass 2: quantize from cached register values.
    const float k_inv = (k_scale_s > 0.0f) ? (1.0f / k_scale_s) : 0.0f;
    const float v_inv = (v_scale_s > 0.0f) ? (1.0f / v_scale_s) : 0.0f;

    count = 0;
    for (unsigned int d = tid; d < head_dim; d += blockDim.x) {
        out_k_i8[base + d] = (signed char)fminf(fmaxf(roundf(k_vals[count] * k_inv), -127.0f), 127.0f);
        out_v_i8[base + d] = (signed char)fminf(fmaxf(roundf(v_vals[count] * v_inv), -127.0f), 127.0f);
        count++;
    }
}

// Single-token K-only RoPE + INT8 store (no V).
// Grid: (n_heads,), Block: (256,)
extern "C" __global__ void talu_rope_store_i8(
    signed char* out_i8,
    float* scales,
    const float* input_f32,
    unsigned int n_heads,
    unsigned int head_dim,
    unsigned int rope_dim,
    unsigned int position,
    float theta
) {
    const unsigned int head = blockIdx.x;
    if (head >= n_heads) return;

    const unsigned int base = head * head_dim;
    const float log2_theta = log2f(theta);
    const unsigned int half_rd = rope_dim >> 1;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp_id = tid / 32u;
    const unsigned int num_warps = (blockDim.x + 31u) / 32u;

    float vals[4];
    float abs_max = 0.0f;
    unsigned int count = 0;

    for (unsigned int d = tid; d < head_dim; d += blockDim.x) {
        float out_v = input_f32[base + d];
        if (d < rope_dim) {
            const unsigned int pair = (d < half_rd) ? d : (d - half_rd);
            const float x0 = input_f32[base + pair];
            const float x1 = input_f32[base + half_rd + pair];
            const float inv_freq = exp2f(log2_theta * (-2.0f * (float)pair / (float)rope_dim));
            const float angle = (float)position * inv_freq;
            float s, c;
            __sincosf(angle, &s, &c);
            out_v = (d < half_rd) ? fmaf(x0, c, -x1 * s) : fmaf(x0, s, x1 * c);
        }
        vals[count] = out_v;
        abs_max = fmaxf(abs_max, fabsf(out_v));
        count++;
    }

    abs_max = talu_warp_reduce_max(abs_max);

    __shared__ float smem[8];
    if (lane == 0) smem[warp_id] = abs_max;
    __syncthreads();

    if (warp_id == 0) {
        abs_max = (lane < num_warps) ? smem[lane] : 0.0f;
        abs_max = talu_warp_reduce_max(abs_max);
    }

    __shared__ float scale_s;
    if (tid == 0) {
        scale_s = (abs_max > 0.0f) ? (abs_max / 127.0f) : 0.0f;
        scales[head] = scale_s;
    }
    __syncthreads();

    const float inv = (scale_s > 0.0f) ? (1.0f / scale_s) : 0.0f;

    count = 0;
    for (unsigned int d = tid; d < head_dim; d += blockDim.x) {
        out_i8[base + d] = (signed char)fminf(fmaxf(roundf(vals[count] * inv), -127.0f), 127.0f);
        count++;
    }
}

// Multi-row KV write with INT8 quantization (prefill).
// Grid: (n_heads, q_rows), Block: (256,)
extern "C" __global__ void talu_kv_write_i8_rows(
    signed char* out_k_i8,
    signed char* out_v_i8,
    float* k_scales,
    float* v_scales,
    const float* input_k_f32,
    const float* input_v_f32,
    unsigned int n_heads,
    unsigned int head_dim,
    unsigned int rope_dim,
    unsigned int q_rows,
    unsigned int row_stride,
    unsigned int position_base,
    float theta
) {
    const unsigned int head = blockIdx.x;
    const unsigned int row = blockIdx.y;
    if (head >= n_heads || row >= q_rows) return;

    const unsigned int input_base = (row * n_heads + head) * head_dim;
    const float log2_theta = log2f(theta);
    const unsigned int half_rd = rope_dim >> 1;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp_id = tid / 32u;
    const unsigned int num_warps = (blockDim.x + 31u) / 32u;
    const unsigned int position = position_base + row;

    float k_vals[4];
    float v_vals[4];
    float k_max = 0.0f;
    float v_max = 0.0f;
    unsigned int count = 0;

    for (unsigned int d = tid; d < head_dim; d += blockDim.x) {
        float k_val = input_k_f32[input_base + d];
        if (d < rope_dim) {
            const unsigned int pair = (d < half_rd) ? d : (d - half_rd);
            const float x0 = input_k_f32[input_base + pair];
            const float x1 = input_k_f32[input_base + half_rd + pair];
            const float inv_freq = exp2f(log2_theta * (-2.0f * (float)pair / (float)rope_dim));
            const float angle = (float)position * inv_freq;
            float s, c;
            __sincosf(angle, &s, &c);
            k_val = (d < half_rd) ? fmaf(x0, c, -x1 * s) : fmaf(x0, s, x1 * c);
        }
        k_vals[count] = k_val;
        v_vals[count] = input_v_f32[input_base + d];
        k_max = fmaxf(k_max, fabsf(k_val));
        v_max = fmaxf(v_max, fabsf(v_vals[count]));
        count++;
    }

    k_max = talu_warp_reduce_max(k_max);
    v_max = talu_warp_reduce_max(v_max);

    __shared__ float smem[16];
    if (lane == 0) {
        smem[warp_id] = k_max;
        smem[warp_id + 8] = v_max;
    }
    __syncthreads();

    if (warp_id == 0) {
        k_max = (lane < num_warps) ? smem[lane] : 0.0f;
        v_max = (lane < num_warps) ? smem[lane + 8] : 0.0f;
        k_max = talu_warp_reduce_max(k_max);
        v_max = talu_warp_reduce_max(v_max);
    }

    __shared__ float k_scale_s, v_scale_s;
    if (tid == 0) {
        k_scale_s = (k_max > 0.0f) ? (k_max / 127.0f) : 0.0f;
        v_scale_s = (v_max > 0.0f) ? (v_max / 127.0f) : 0.0f;
        k_scales[row * n_heads + head] = k_scale_s;
        v_scales[row * n_heads + head] = v_scale_s;
    }
    __syncthreads();

    const float k_inv = (k_scale_s > 0.0f) ? (1.0f / k_scale_s) : 0.0f;
    const float v_inv = (v_scale_s > 0.0f) ? (1.0f / v_scale_s) : 0.0f;

    const unsigned int out_base = row * row_stride + head * head_dim;
    count = 0;
    for (unsigned int d = tid; d < head_dim; d += blockDim.x) {
        out_k_i8[out_base + d] = (signed char)fminf(fmaxf(roundf(k_vals[count] * k_inv), -127.0f), 127.0f);
        out_v_i8[out_base + d] = (signed char)fminf(fmaxf(roundf(v_vals[count] * v_inv), -127.0f), 127.0f);
        count++;
    }
}

// Batched KV write with INT8 quantization using per-row pointer tables.
// Grid: (n_heads, q_rows), Block: (256,)
extern "C" __global__ void talu_kv_write_i8_rows_ptrs(
    const unsigned long long* out_k_ptrs,
    const unsigned long long* out_v_ptrs,
    const unsigned long long* k_scale_ptrs,
    const unsigned long long* v_scale_ptrs,
    const unsigned int* positions,
    const float* input_k_f32,
    const float* input_v_f32,
    unsigned int n_heads,
    unsigned int head_dim,
    unsigned int rope_dim,
    unsigned int q_rows,
    unsigned int row_stride,
    float theta
) {
    const unsigned int head = blockIdx.x;
    const unsigned int row = blockIdx.y;
    if (head >= n_heads || row >= q_rows) return;

    signed char* out_k = reinterpret_cast<signed char*>(out_k_ptrs[row]);
    signed char* out_v = reinterpret_cast<signed char*>(out_v_ptrs[row]);
    float* k_scales = reinterpret_cast<float*>(k_scale_ptrs[row]);
    float* v_scales = reinterpret_cast<float*>(v_scale_ptrs[row]);
    if (out_k == nullptr || out_v == nullptr) return;

    const unsigned int position = positions[row];
    const unsigned int input_base = (row * n_heads + head) * head_dim;
    const float log2_theta = log2f(theta);
    const unsigned int half_rd = rope_dim >> 1;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp_id = tid / 32u;
    const unsigned int num_warps = (blockDim.x + 31u) / 32u;

    float k_vals[4];
    float v_vals[4];
    float k_max = 0.0f;
    float v_max = 0.0f;
    unsigned int count = 0;

    for (unsigned int d = tid; d < head_dim; d += blockDim.x) {
        float k_val = input_k_f32[input_base + d];
        if (d < rope_dim) {
            const unsigned int pair = (d < half_rd) ? d : (d - half_rd);
            const float x0 = input_k_f32[input_base + pair];
            const float x1 = input_k_f32[input_base + half_rd + pair];
            const float inv_freq = exp2f(log2_theta * (-2.0f * (float)pair / (float)rope_dim));
            const float angle = (float)position * inv_freq;
            float s, c;
            __sincosf(angle, &s, &c);
            k_val = (d < half_rd) ? fmaf(x0, c, -x1 * s) : fmaf(x0, s, x1 * c);
        }
        k_vals[count] = k_val;
        v_vals[count] = input_v_f32[input_base + d];
        k_max = fmaxf(k_max, fabsf(k_val));
        v_max = fmaxf(v_max, fabsf(v_vals[count]));
        count++;
    }

    k_max = talu_warp_reduce_max(k_max);
    v_max = talu_warp_reduce_max(v_max);

    __shared__ float smem[16];
    if (lane == 0) {
        smem[warp_id] = k_max;
        smem[warp_id + 8] = v_max;
    }
    __syncthreads();

    if (warp_id == 0) {
        k_max = (lane < num_warps) ? smem[lane] : 0.0f;
        v_max = (lane < num_warps) ? smem[lane + 8] : 0.0f;
        k_max = talu_warp_reduce_max(k_max);
        v_max = talu_warp_reduce_max(v_max);
    }

    __shared__ float k_scale_s, v_scale_s;
    if (tid == 0) {
        k_scale_s = (k_max > 0.0f) ? (k_max / 127.0f) : 0.0f;
        v_scale_s = (v_max > 0.0f) ? (v_max / 127.0f) : 0.0f;
        k_scales[position * n_heads + head] = k_scale_s;
        v_scales[position * n_heads + head] = v_scale_s;
    }
    __syncthreads();

    const float k_inv = (k_scale_s > 0.0f) ? (1.0f / k_scale_s) : 0.0f;
    const float v_inv = (v_scale_s > 0.0f) ? (1.0f / v_scale_s) : 0.0f;

    const unsigned int out_base = position * row_stride + head * head_dim;
    count = 0;
    for (unsigned int d = tid; d < head_dim; d += blockDim.x) {
        out_k[out_base + d] = (signed char)fminf(fmaxf(roundf(k_vals[count] * k_inv), -127.0f), 127.0f);
        out_v[out_base + d] = (signed char)fminf(fmaxf(roundf(v_vals[count] * v_inv), -127.0f), 127.0f);
        count++;
    }
}
