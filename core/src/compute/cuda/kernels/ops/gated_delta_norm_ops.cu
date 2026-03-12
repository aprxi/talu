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
