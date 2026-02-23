static __device__ __forceinline__ float talu_matvec_dot_row_f16(
    const float* input,
    const unsigned short* weight_row,
    unsigned int in_dim
) {
    float acc = 0.0f;
    const unsigned int in_dim4 = in_dim / 4;
    const bool vectorizable = ((reinterpret_cast<uintptr_t>(input) & 0xFu) == 0u) and
        ((reinterpret_cast<uintptr_t>(weight_row) & 0x7u) == 0u);

    if (vectorizable) {
        const float4* in_ptr4 = reinterpret_cast<const float4*>(input);
        const uint2* w_ptr2 = reinterpret_cast<const uint2*>(weight_row);
        for (unsigned int i = 0; i < in_dim4; ++i) {
            const float4 x = in_ptr4[i];
            const uint2 w_pack = w_ptr2[i];

            const float w0 = talu_decode_f16_u16(static_cast<unsigned short>(w_pack.x & 0xFFFFu));
            const float w1 = talu_decode_f16_u16(static_cast<unsigned short>(w_pack.x >> 16));
            const float w2 = talu_decode_f16_u16(static_cast<unsigned short>(w_pack.y & 0xFFFFu));
            const float w3 = talu_decode_f16_u16(static_cast<unsigned short>(w_pack.y >> 16));

            acc = fmaf(x.x, w0, acc);
            acc = fmaf(x.y, w1, acc);
            acc = fmaf(x.z, w2, acc);
            acc = fmaf(x.w, w3, acc);
        }
        for (unsigned int i = in_dim4 * 4; i < in_dim; ++i) {
            acc = fmaf(input[i], talu_decode_f16_u16(weight_row[i]), acc);
        }
        return acc;
    }

    for (unsigned int i = 0; i < in_dim; ++i) {
        acc = fmaf(input[i], talu_decode_f16_u16(weight_row[i]), acc);
    }
    return acc;
}

static __device__ __forceinline__ float talu_matvec_dot_row_bf16(
    const float* input,
    const unsigned short* weight_row,
    unsigned int in_dim
) {
    float acc = 0.0f;
    const unsigned int in_dim4 = in_dim / 4;
    const bool vectorizable = ((reinterpret_cast<uintptr_t>(input) & 0xFu) == 0u) and
        ((reinterpret_cast<uintptr_t>(weight_row) & 0x7u) == 0u);

    if (vectorizable) {
        const float4* in_ptr4 = reinterpret_cast<const float4*>(input);
        const uint2* w_ptr2 = reinterpret_cast<const uint2*>(weight_row);
        for (unsigned int i = 0; i < in_dim4; ++i) {
            const float4 x = in_ptr4[i];
            const uint2 w_pack = w_ptr2[i];

            const float w0 = __uint_as_float((w_pack.x & 0xFFFFu) << 16);
            const float w1 = __uint_as_float((w_pack.x >> 16) << 16);
            const float w2 = __uint_as_float((w_pack.y & 0xFFFFu) << 16);
            const float w3 = __uint_as_float((w_pack.y >> 16) << 16);

            acc = fmaf(x.x, w0, acc);
            acc = fmaf(x.y, w1, acc);
            acc = fmaf(x.z, w2, acc);
            acc = fmaf(x.w, w3, acc);
        }
        for (unsigned int i = in_dim4 * 4; i < in_dim; ++i) {
            acc = fmaf(input[i], talu_decode_bf16_u16(weight_row[i]), acc);
        }
        return acc;
    }

    for (unsigned int i = 0; i < in_dim; ++i) {
        acc = fmaf(input[i], talu_decode_bf16_u16(weight_row[i]), acc);
    }
    return acc;
}

extern "C" __global__ void talu_matvec_f16_f32(
    const float* input,
    const unsigned short* weight,
    float* out,
    unsigned int in_dim,
    unsigned int out_dim
) {
    const unsigned int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= out_dim) return;
    const unsigned short* row = weight + (unsigned long long)out_idx * in_dim;
    out[out_idx] = talu_matvec_dot_row_f16(input, row, in_dim);
}

extern "C" __global__ void talu_matvec_bf16_f32(
    const float* input,
    const unsigned short* weight,
    float* out,
    unsigned int in_dim,
    unsigned int out_dim
) {
    const unsigned int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= out_dim) return;
    const unsigned short* row = weight + (unsigned long long)out_idx * in_dim;
    out[out_idx] = talu_matvec_dot_row_bf16(input, row, in_dim);
}

extern "C" __global__ void talu_matvec_qkv_f16_f32(
    const float* input,
    const unsigned short* q_weight,
    float* q_out,
    unsigned int q_out_dim,
    const unsigned short* k_weight,
    float* k_out,
    unsigned int k_out_dim,
    const unsigned short* v_weight,
    float* v_out,
    unsigned int v_out_dim,
    unsigned int in_dim
) {
    const unsigned int out_index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int qk_dim = q_out_dim + k_out_dim;
    const unsigned int total_dim = qk_dim + v_out_dim;
    if (out_index >= total_dim) return;

    if (out_index < q_out_dim) {
        const unsigned short* row = q_weight + (unsigned long long)out_index * in_dim;
        q_out[out_index] = talu_matvec_dot_row_f16(input, row, in_dim);
        return;
    }
    if (out_index < qk_dim) {
        const unsigned int k_idx = out_index - q_out_dim;
        const unsigned short* row = k_weight + (unsigned long long)k_idx * in_dim;
        k_out[k_idx] = talu_matvec_dot_row_f16(input, row, in_dim);
        return;
    }
    const unsigned int v_idx = out_index - qk_dim;
    const unsigned short* row = v_weight + (unsigned long long)v_idx * in_dim;
    v_out[v_idx] = talu_matvec_dot_row_f16(input, row, in_dim);
}

extern "C" __global__ void talu_matvec_qkv_bf16_f32(
    const float* input,
    const unsigned short* q_weight,
    float* q_out,
    unsigned int q_out_dim,
    const unsigned short* k_weight,
    float* k_out,
    unsigned int k_out_dim,
    const unsigned short* v_weight,
    float* v_out,
    unsigned int v_out_dim,
    unsigned int in_dim
) {
    const unsigned int out_index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int qk_dim = q_out_dim + k_out_dim;
    const unsigned int total_dim = qk_dim + v_out_dim;
    if (out_index >= total_dim) return;

    if (out_index < q_out_dim) {
        const unsigned short* row = q_weight + (unsigned long long)out_index * in_dim;
        q_out[out_index] = talu_matvec_dot_row_bf16(input, row, in_dim);
        return;
    }
    if (out_index < qk_dim) {
        const unsigned int k_idx = out_index - q_out_dim;
        const unsigned short* row = k_weight + (unsigned long long)k_idx * in_dim;
        k_out[k_idx] = talu_matvec_dot_row_bf16(input, row, in_dim);
        return;
    }
    const unsigned int v_idx = out_index - qk_dim;
    const unsigned short* row = v_weight + (unsigned long long)v_idx * in_dim;
    v_out[v_idx] = talu_matvec_dot_row_bf16(input, row, in_dim);
}

extern "C" __global__ void talu_matvec_gate_up_f16_f32(
    const float* input,
    const unsigned short* gate_weight,
    float* gate_out,
    unsigned int gate_out_dim,
    const unsigned short* up_weight,
    float* up_out,
    unsigned int up_out_dim,
    unsigned int in_dim
) {
    const unsigned int out_index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total_dim = gate_out_dim + up_out_dim;
    if (out_index >= total_dim) return;

    if (out_index < gate_out_dim) {
        const unsigned short* row = gate_weight + (unsigned long long)out_index * in_dim;
        gate_out[out_index] = talu_matvec_dot_row_f16(input, row, in_dim);
        return;
    }
    const unsigned int up_idx = out_index - gate_out_dim;
    const unsigned short* row = up_weight + (unsigned long long)up_idx * in_dim;
    up_out[up_idx] = talu_matvec_dot_row_f16(input, row, in_dim);
}

extern "C" __global__ void talu_matvec_gate_up_bf16_f32(
    const float* input,
    const unsigned short* gate_weight,
    float* gate_out,
    unsigned int gate_out_dim,
    const unsigned short* up_weight,
    float* up_out,
    unsigned int up_out_dim,
    unsigned int in_dim
) {
    const unsigned int out_index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total_dim = gate_out_dim + up_out_dim;
    if (out_index >= total_dim) return;

    if (out_index < gate_out_dim) {
        const unsigned short* row = gate_weight + (unsigned long long)out_index * in_dim;
        gate_out[out_index] = talu_matvec_dot_row_bf16(input, row, in_dim);
        return;
    }
    const unsigned int up_idx = out_index - gate_out_dim;
    const unsigned short* row = up_weight + (unsigned long long)up_idx * in_dim;
    up_out[up_idx] = talu_matvec_dot_row_bf16(input, row, in_dim);
}
