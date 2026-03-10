static __device__ __forceinline__ float talu_fast_exp_scalar(float x) {
    const float log2e = 1.4426950408889634f;
    const float ln2_hi = 0.693359375f;
    const float ln2_lo = -2.12194440e-4f;
    const float exp_hi = 88.3762626647949f;
    const float exp_lo = -88.3762626647949f;
    const float p0 = 1.9875691500e-4f;
    const float p1 = 1.3981999507e-3f;
    const float p2 = 8.3334519073e-3f;
    const float p3 = 4.1665795894e-2f;
    const float p4 = 1.6666665459e-1f;
    const float p5 = 5.0000001201e-1f;

    float x_clamped = fminf(fmaxf(x, exp_lo), exp_hi);
    const float fx = floorf(x_clamped * log2e + 0.5f);
    const int fxi = (int)fx;
    x_clamped = x_clamped - fx * ln2_hi - fx * ln2_lo;

    float y = p0;
    y = y * x_clamped + p1;
    y = y * x_clamped + p2;
    y = y * x_clamped + p3;
    y = y * x_clamped + p4;
    y = y * x_clamped + p5;
    y = y * x_clamped * x_clamped + x_clamped + 1.0f;

    union {
        unsigned int u;
        float f;
    } pow2n;
    pow2n.u = (unsigned int)((fxi + 127) << 23);
    return y * pow2n.f;
}

extern "C" __global__ void talu_silu_f32(
    float* out,
    const float* input,
    unsigned int count
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float x = input[index];
    out[index] = x / (1.0f + talu_fast_exp_scalar(-x));
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
    const float silu_x = x / (1.0f + talu_fast_exp_scalar(-x));
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
