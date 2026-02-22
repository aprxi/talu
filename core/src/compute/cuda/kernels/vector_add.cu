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

