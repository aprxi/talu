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

extern "C" __global__ void talu_vector_add_rows_strided_f32(
    float* out,
    const float* a,
    const float* b,
    unsigned int rows,
    unsigned int cols,
    unsigned int out_stride,
    unsigned int a_stride,
    unsigned int b_stride
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = rows * cols;
    if (index >= total) return;

    const unsigned int row = index / cols;
    const unsigned int col = index - row * cols;
    const unsigned long long out_idx = (unsigned long long)row * out_stride + col;
    const unsigned long long a_idx = (unsigned long long)row * a_stride + col;
    const unsigned long long b_idx = (unsigned long long)row * b_stride + col;
    out[out_idx] = a[a_idx] + b[b_idx];
}

extern "C" __global__ void talu_vector_add_scaled_rows_strided_f32(
    float* out,
    const float* a,
    const float* b,
    float scale,
    unsigned int rows,
    unsigned int cols,
    unsigned int out_stride,
    unsigned int a_stride,
    unsigned int b_stride
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = rows * cols;
    if (index >= total) return;

    const unsigned int row = index / cols;
    const unsigned int col = index - row * cols;
    const unsigned long long out_idx = (unsigned long long)row * out_stride + col;
    const unsigned long long a_idx = (unsigned long long)row * a_stride + col;
    const unsigned long long b_idx = (unsigned long long)row * b_stride + col;
    out[out_idx] = a[a_idx] + b[b_idx] * scale;
}

extern "C" __global__ void talu_residual_add_scaled_rows_strided_f32(
    float* out,
    const float* a,
    const float* b,
    float residual_scale,
    float output_scale,
    unsigned int rows,
    unsigned int cols,
    unsigned int out_stride,
    unsigned int a_stride,
    unsigned int b_stride
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = rows * cols;
    if (index >= total) return;

    const unsigned int row = index / cols;
    const unsigned int col = index - row * cols;
    const unsigned long long out_idx = (unsigned long long)row * out_stride + col;
    const unsigned long long a_idx = (unsigned long long)row * a_stride + col;
    const unsigned long long b_idx = (unsigned long long)row * b_stride + col;
    const float residual = a[a_idx] + b[b_idx] * residual_scale;
    out[out_idx] = residual * output_scale;
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

extern "C" __global__ void talu_cast_f32_to_bf16(
    unsigned short* out,
    const float* input,
    unsigned int count
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;

    const float value = input[index];
    const unsigned int bits = __float_as_uint(value);
    // Round-to-nearest-even for bf16 truncation.
    const unsigned int lsb = (bits >> 16) & 1u;
    const unsigned int rounding_bias = 0x7FFFu + lsb;
    out[index] = (unsigned short)((bits + rounding_bias) >> 16);
}

extern "C" __global__ void talu_cast_bf16_to_f32(
    float* out,
    const unsigned short* input,
    unsigned int count
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    out[index] = __uint_as_float(static_cast<unsigned int>(input[index]) << 16);
}

extern "C" __global__ void talu_decode_u32_increment(
    unsigned int* seq_lens,
    unsigned int* positions,
    unsigned int count
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    seq_lens[index] += 1u;
    positions[index] += 1u;
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

extern "C" __global__ void talu_embedding_lookup_u16_rows_f32(
    float* out,
    const unsigned short* embeddings,
    const unsigned int* tokens,
    unsigned int rows,
    unsigned int dim0,
    unsigned int dim1,
    unsigned int hidden_dim,
    unsigned int layout_tag,
    unsigned int dtype_tag,
    float multiplier
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = rows * hidden_dim;
    if (index >= total) return;

    const unsigned int row = index / hidden_dim;
    const unsigned int col = index - row * hidden_dim;
    if (row >= rows) return;
    const unsigned int token = tokens[row];

    unsigned long long src_index = 0;
    if (layout_tag == 0) {
        // Layout: [vocab, hidden]
        if (token >= dim0 || col >= dim1) return;
        src_index = (unsigned long long)token * dim1 + col;
    } else if (layout_tag == 1) {
        // Layout: [hidden, vocab]
        if (token >= dim1 || col >= dim0) return;
        src_index = (unsigned long long)col * dim1 + token;
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
