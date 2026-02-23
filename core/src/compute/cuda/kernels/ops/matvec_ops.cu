static constexpr unsigned int TALU_WARP_SIZE = 32;

static __device__ __forceinline__ float talu_warp_sum_f32(float value) {
    value += __shfl_down_sync(0xFFFFFFFFu, value, 16);
    value += __shfl_down_sync(0xFFFFFFFFu, value, 8);
    value += __shfl_down_sync(0xFFFFFFFFu, value, 4);
    value += __shfl_down_sync(0xFFFFFFFFu, value, 2);
    value += __shfl_down_sync(0xFFFFFFFFu, value, 1);
    return value;
}

extern "C" __global__ void talu_matvec_f16_f32(
    const float* input,
    const unsigned short* weight,
    float* out,
    unsigned int in_dim,
    unsigned int out_dim
) {
    const unsigned int lane = threadIdx.x & (TALU_WARP_SIZE - 1u);
    const unsigned int warp_id = threadIdx.x / TALU_WARP_SIZE;
    const unsigned int warps_per_block = blockDim.x / TALU_WARP_SIZE;
    const unsigned int out_idx = blockIdx.x * warps_per_block + warp_id;
    if (out_idx >= out_dim) return;
    const unsigned short* row = weight + (unsigned long long)out_idx * in_dim;
    float acc = 0.0f;
    for (unsigned int i = lane; i < in_dim; i += TALU_WARP_SIZE) {
        acc = fmaf(input[i], talu_decode_f16_u16(row[i]), acc);
    }
    acc = talu_warp_sum_f32(acc);
    if (lane == 0) out[out_idx] = acc;
}

extern "C" __global__ void talu_matvec_bf16_f32(
    const float* input,
    const unsigned short* weight,
    float* out,
    unsigned int in_dim,
    unsigned int out_dim
) {
    const unsigned int lane = threadIdx.x & (TALU_WARP_SIZE - 1u);
    const unsigned int warp_id = threadIdx.x / TALU_WARP_SIZE;
    const unsigned int warps_per_block = blockDim.x / TALU_WARP_SIZE;
    const unsigned int out_idx = blockIdx.x * warps_per_block + warp_id;
    if (out_idx >= out_dim) return;
    const unsigned short* row = weight + (unsigned long long)out_idx * in_dim;
    float acc = 0.0f;
    for (unsigned int i = lane; i < in_dim; i += TALU_WARP_SIZE) {
        acc = fmaf(input[i], talu_decode_bf16_u16(row[i]), acc);
    }
    acc = talu_warp_sum_f32(acc);
    if (lane == 0) out[out_idx] = acc;
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
    const unsigned int lane = threadIdx.x & (TALU_WARP_SIZE - 1u);
    const unsigned int warp_id = threadIdx.x / TALU_WARP_SIZE;
    const unsigned int warps_per_block = blockDim.x / TALU_WARP_SIZE;
    const unsigned int out_index = blockIdx.x * warps_per_block + warp_id;
    const unsigned int qk_dim = q_out_dim + k_out_dim;
    const unsigned int total_dim = qk_dim + v_out_dim;
    if (out_index >= total_dim) return;

    const unsigned short* row = nullptr;
    float* out_ptr = nullptr;
    unsigned int out_row = 0;
    if (out_index < q_out_dim) {
        out_row = out_index;
        row = q_weight + (unsigned long long)out_row * in_dim;
        out_ptr = q_out + out_row;
    } else if (out_index < qk_dim) {
        out_row = out_index - q_out_dim;
        row = k_weight + (unsigned long long)out_row * in_dim;
        out_ptr = k_out + out_row;
    } else {
        out_row = out_index - qk_dim;
        row = v_weight + (unsigned long long)out_row * in_dim;
        out_ptr = v_out + out_row;
    }

    float acc = 0.0f;
    for (unsigned int i = lane; i < in_dim; i += TALU_WARP_SIZE) {
        acc = fmaf(input[i], talu_decode_f16_u16(row[i]), acc);
    }
    acc = talu_warp_sum_f32(acc);
    if (lane == 0) *out_ptr = acc;
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
    const unsigned int lane = threadIdx.x & (TALU_WARP_SIZE - 1u);
    const unsigned int warp_id = threadIdx.x / TALU_WARP_SIZE;
    const unsigned int warps_per_block = blockDim.x / TALU_WARP_SIZE;
    const unsigned int out_index = blockIdx.x * warps_per_block + warp_id;
    const unsigned int qk_dim = q_out_dim + k_out_dim;
    const unsigned int total_dim = qk_dim + v_out_dim;
    if (out_index >= total_dim) return;

    const unsigned short* row = nullptr;
    float* out_ptr = nullptr;
    unsigned int out_row = 0;
    if (out_index < q_out_dim) {
        out_row = out_index;
        row = q_weight + (unsigned long long)out_row * in_dim;
        out_ptr = q_out + out_row;
    } else if (out_index < qk_dim) {
        out_row = out_index - q_out_dim;
        row = k_weight + (unsigned long long)out_row * in_dim;
        out_ptr = k_out + out_row;
    } else {
        out_row = out_index - qk_dim;
        row = v_weight + (unsigned long long)out_row * in_dim;
        out_ptr = v_out + out_row;
    }

    float acc = 0.0f;
    for (unsigned int i = lane; i < in_dim; i += TALU_WARP_SIZE) {
        acc = fmaf(input[i], talu_decode_bf16_u16(row[i]), acc);
    }
    acc = talu_warp_sum_f32(acc);
    if (lane == 0) *out_ptr = acc;
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
    const unsigned int lane = threadIdx.x & (TALU_WARP_SIZE - 1u);
    const unsigned int warp_id = threadIdx.x / TALU_WARP_SIZE;
    const unsigned int warps_per_block = blockDim.x / TALU_WARP_SIZE;
    const unsigned int out_index = blockIdx.x * warps_per_block + warp_id;
    const unsigned int total_dim = gate_out_dim + up_out_dim;
    if (out_index >= total_dim) return;

    const unsigned short* row = nullptr;
    float* out_ptr = nullptr;
    unsigned int out_row = 0;
    if (out_index < gate_out_dim) {
        out_row = out_index;
        row = gate_weight + (unsigned long long)out_row * in_dim;
        out_ptr = gate_out + out_row;
    } else {
        out_row = out_index - gate_out_dim;
        row = up_weight + (unsigned long long)out_row * in_dim;
        out_ptr = up_out + out_row;
    }

    float acc = 0.0f;
    for (unsigned int i = lane; i < in_dim; i += TALU_WARP_SIZE) {
        acc = fmaf(input[i], talu_decode_f16_u16(row[i]), acc);
    }
    acc = talu_warp_sum_f32(acc);
    if (lane == 0) *out_ptr = acc;
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
    const unsigned int lane = threadIdx.x & (TALU_WARP_SIZE - 1u);
    const unsigned int warp_id = threadIdx.x / TALU_WARP_SIZE;
    const unsigned int warps_per_block = blockDim.x / TALU_WARP_SIZE;
    const unsigned int out_index = blockIdx.x * warps_per_block + warp_id;
    const unsigned int total_dim = gate_out_dim + up_out_dim;
    if (out_index >= total_dim) return;

    const unsigned short* row = nullptr;
    float* out_ptr = nullptr;
    unsigned int out_row = 0;
    if (out_index < gate_out_dim) {
        out_row = out_index;
        row = gate_weight + (unsigned long long)out_row * in_dim;
        out_ptr = gate_out + out_row;
    } else {
        out_row = out_index - gate_out_dim;
        row = up_weight + (unsigned long long)out_row * in_dim;
        out_ptr = up_out + out_row;
    }

    float acc = 0.0f;
    for (unsigned int i = lane; i < in_dim; i += TALU_WARP_SIZE) {
        acc = fmaf(input[i], talu_decode_bf16_u16(row[i]), acc);
    }
    acc = talu_warp_sum_f32(acc);
    if (lane == 0) *out_ptr = acc;
}
