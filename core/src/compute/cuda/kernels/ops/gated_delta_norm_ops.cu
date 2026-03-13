__inline__ __device__ float talu_gd_warp_reduce_sum(float value) {
    #pragma unroll
    for (unsigned int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xFFFFffffu, value, offset);
    }
    return value;
}

extern "C" __global__ void talu_gated_delta_qk_norm_f32(
    float* query,
    float* key,
    unsigned int n_heads,
    unsigned int d_head
) {
    const unsigned int head = blockIdx.x;
    if (head >= n_heads || d_head == 0) return;

    const unsigned int tid = threadIdx.x;
    float* query_head = query + (unsigned long long)head * d_head;
    float* key_head = key + (unsigned long long)head * d_head;

    float q_sum_sq = 0.0f;
    float k_sum_sq = 0.0f;
    for (unsigned int idx = tid; idx < d_head; idx += blockDim.x) {
        const float q = query_head[idx];
        const float k = key_head[idx];
        q_sum_sq = fmaf(q, q, q_sum_sq);
        k_sum_sq = fmaf(k, k, k_sum_sq);
    }

    const float q_lane_sum = talu_gd_warp_reduce_sum(q_sum_sq);
    const float k_lane_sum = talu_gd_warp_reduce_sum(k_sum_sq);
    __shared__ float q_warp_sums[32];
    __shared__ float k_warp_sums[32];
    if ((tid & 31u) == 0u) {
        q_warp_sums[tid >> 5] = q_lane_sum;
        k_warp_sums[tid >> 5] = k_lane_sum;
    }
    __syncthreads();

    const unsigned int warp_count = (blockDim.x + 31u) >> 5;
    float q_block_sum = (tid < warp_count) ? q_warp_sums[tid] : 0.0f;
    float k_block_sum = (tid < warp_count) ? k_warp_sums[tid] : 0.0f;
    if (tid < 32u) {
        q_block_sum = talu_gd_warp_reduce_sum(q_block_sum);
        k_block_sum = talu_gd_warp_reduce_sum(k_block_sum);
    }

    __shared__ float q_scale;
    __shared__ float k_scale;
    if (tid == 0) {
        const float inv_sqrt_d = rsqrtf((float)d_head);
        q_scale = rsqrtf(q_block_sum + 1.0e-6f) * inv_sqrt_d;
        k_scale = rsqrtf(k_block_sum + 1.0e-6f);
    }
    __syncthreads();

    for (unsigned int idx = tid; idx < d_head; idx += blockDim.x) {
        query_head[idx] *= q_scale;
        key_head[idx] *= k_scale;
    }
}

extern "C" __global__ void talu_gated_delta_rmsnorm_silu_mul_f32(
    float* out,
    const float* input,
    const float* gate,
    const float* weight,
    unsigned int rows,
    unsigned int cols,
    float eps,
    unsigned int weight_row_stride
) {
    const unsigned int row = blockIdx.x;
    if (row >= rows || cols == 0) return;

    const unsigned int tid = threadIdx.x;
    const unsigned long long row_offset = (unsigned long long)row * cols;
    const float* in_row = input + row_offset;
    const float* gate_row = gate + row_offset;
    float* out_row = out + row_offset;

    float sum_sq = 0.0f;
    for (unsigned int idx = tid; idx < cols; idx += blockDim.x) {
        const float v = in_row[idx];
        sum_sq = fmaf(v, v, sum_sq);
    }

    const float lane_sum = talu_gd_warp_reduce_sum(sum_sq);
    __shared__ float warp_sums[32];
    if ((tid & 31u) == 0u) {
        warp_sums[tid >> 5] = lane_sum;
    }
    __syncthreads();

    const unsigned int warp_count = (blockDim.x + 31u) >> 5;
    float block_sum = (tid < warp_count) ? warp_sums[tid] : 0.0f;
    if (tid < 32u) {
        block_sum = talu_gd_warp_reduce_sum(block_sum);
    }

    __shared__ float inv_rms;
    if (tid == 0u) {
        inv_rms = rsqrtf((block_sum / (float)cols) + eps);
    }
    __syncthreads();

    const unsigned int weight_base = (weight_row_stride == 0u) ? 0u : row * weight_row_stride;
    for (unsigned int idx = tid; idx < cols; idx += blockDim.x) {
        const float normed = in_row[idx] * inv_rms * weight[weight_base + idx];
        const float g = gate_row[idx];
        const float sigma = 1.0f / (1.0f + expf(-g));
        out_row[idx] = normed * (g * sigma);
    }
}

extern "C" __global__ void talu_gated_delta_rmsnorm_silu_mul_rows_f32(
    float* out,
    const float* input,
    const float* gate_rows,
    const float* weight,
    unsigned int rows_total,
    unsigned int cols,
    unsigned int n_v_heads,
    unsigned int gate_row_stride,
    unsigned int gate_offset,
    float eps,
    unsigned int weight_row_stride
) {
    const unsigned int row = blockIdx.x;
    if (row >= rows_total || cols == 0 || n_v_heads == 0) return;
    if (gate_row_stride == 0 || gate_offset + cols > gate_row_stride) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int token_idx = row / n_v_heads;
    const unsigned int head_idx = row - token_idx * n_v_heads;

    const unsigned long long in_row_offset = (unsigned long long)row * cols;
    const unsigned long long gate_base = (unsigned long long)token_idx * gate_row_stride + gate_offset + (unsigned long long)head_idx * cols;
    const float* in_row = input + in_row_offset;
    const float* gate_row = gate_rows + gate_base;
    float* out_row = out + in_row_offset;

    float sum_sq = 0.0f;
    for (unsigned int idx = tid; idx < cols; idx += blockDim.x) {
        const float v = in_row[idx];
        sum_sq = fmaf(v, v, sum_sq);
    }

    const float lane_sum = talu_gd_warp_reduce_sum(sum_sq);
    __shared__ float warp_sums[32];
    if ((tid & 31u) == 0u) {
        warp_sums[tid >> 5] = lane_sum;
    }
    __syncthreads();

    const unsigned int warp_count = (blockDim.x + 31u) >> 5;
    float block_sum = (tid < warp_count) ? warp_sums[tid] : 0.0f;
    if (tid < 32u) {
        block_sum = talu_gd_warp_reduce_sum(block_sum);
    }

    __shared__ float inv_rms;
    if (tid == 0u) {
        inv_rms = rsqrtf((block_sum / (float)cols) + eps);
    }
    __syncthreads();

    const unsigned int weight_base = (weight_row_stride == 0u) ? 0u : head_idx * weight_row_stride;
    for (unsigned int idx = tid; idx < cols; idx += blockDim.x) {
        const float normed = in_row[idx] * inv_rms * weight[weight_base + idx];
        const float g = gate_row[idx];
        const float sigma = 1.0f / (1.0f + expf(-g));
        out_row[idx] = normed * (g * sigma);
    }
}
