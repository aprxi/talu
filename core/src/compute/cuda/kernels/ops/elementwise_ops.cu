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

extern "C" __global__ void talu_vector_add_scaled_f32(
    float* out,
    const float* a,
    const float* b,
    float scale,
    unsigned int count
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    out[index] = a[index] + b[index] * scale;
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

extern "C" __global__ void talu_embedding_lookup_f32(
    float* out,
    const float* embeddings,
    unsigned int dim0,
    unsigned int dim1,
    unsigned int hidden_dim,
    unsigned int token,
    unsigned int layout_tag,
    float multiplier
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= hidden_dim) return;

    if (layout_tag == 0) {
        // Layout: [vocab, hidden]
        if (token >= dim0 || index >= dim1) return;
        out[index] = embeddings[(unsigned long long)token * dim1 + index] * multiplier;
        return;
    }
    if (layout_tag == 1) {
        // Layout: [hidden, vocab]
        if (token >= dim1 || index >= dim0) return;
        out[index] = embeddings[(unsigned long long)index * dim1 + token] * multiplier;
    }
}

__device__ __forceinline__ float talu_bf16_to_f32(unsigned short raw) {
    const unsigned int bits = ((unsigned int)raw) << 16;
    return __uint_as_float(bits);
}

extern "C" __global__ void talu_embedding_lookup_u16_f32(
    float* out,
    const unsigned short* embeddings,
    unsigned int dim0,
    unsigned int dim1,
    unsigned int hidden_dim,
    unsigned int token,
    unsigned int layout_tag,
    unsigned int dtype_tag,
    float multiplier
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= hidden_dim) return;

    unsigned long long src_index = 0;
    if (layout_tag == 0) {
        // Layout: [vocab, hidden]
        if (token >= dim0 || index >= dim1) return;
        src_index = (unsigned long long)token * dim1 + index;
    } else if (layout_tag == 1) {
        // Layout: [hidden, vocab]
        if (token >= dim1 || index >= dim0) return;
        src_index = (unsigned long long)index * dim1 + token;
    } else {
        return;
    }

    const unsigned short raw = embeddings[src_index];
    float value = 0.0f;
    if (dtype_tag == 0) {
        value = __half2float(__ushort_as_half(raw));
    } else if (dtype_tag == 1) {
        value = talu_bf16_to_f32(raw);
    } else {
        return;
    }
    out[index] = value * multiplier;
}
