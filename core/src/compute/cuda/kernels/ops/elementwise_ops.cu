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
