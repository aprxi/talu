extern "C" __global__ void talu_rmsnorm_f32_v1(
    float* out,
    const float* input,
    const float* weight,
    unsigned int rows,
    unsigned int cols,
    float eps
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
        out[base + col] = normalized * weight[col];
    }
}

