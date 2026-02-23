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

extern "C" __global__ void talu_silu_mul_f32(
    float* out,
    const float* gate,
    const float* up,
    unsigned int count
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float x = gate[index];
    const float silu_x = x / (1.0f + expf(-x));
    out[index] = silu_x * up[index];
}

extern "C" __global__ void talu_gelu_mul_f32(
    float* out,
    const float* gate,
    const float* up,
    unsigned int count
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float x = gate[index];
    const float x3 = x * x * x;
    const float inner = 0.7978845608028654f * (x + 0.044715f * x3);
    const float gelu_x = 0.5f * x * (1.0f + tanhf(inner));
    out[index] = gelu_x * up[index];
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
