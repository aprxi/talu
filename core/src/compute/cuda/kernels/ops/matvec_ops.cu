static constexpr unsigned int TALU_WARP_SIZE = 32;
static constexpr unsigned int TALU_MATMUL_TILE_K = 2048;

static __device__ __forceinline__ float talu_warp_sum_f32(float value) {
    value += __shfl_down_sync(0xFFFFFFFFu, value, 16);
    value += __shfl_down_sync(0xFFFFFFFFu, value, 8);
    value += __shfl_down_sync(0xFFFFFFFFu, value, 4);
    value += __shfl_down_sync(0xFFFFFFFFu, value, 2);
    value += __shfl_down_sync(0xFFFFFFFFu, value, 1);
    return value;
}

static __device__ __forceinline__ float talu_dot_f16_u16_vec8(
    const float* input,
    const unsigned short* row,
    unsigned int in_dim,
    unsigned int lane
) {
    float acc = 0.0f;
    const bool can_vec8 = ((((unsigned long long)row) & 0xFu) == 0u);
    if (!can_vec8) {
        for (unsigned int i = lane; i < in_dim; i += TALU_WARP_SIZE) {
            acc = fmaf(input[i], talu_decode_f16_u16(row[i]), acc);
        }
        return acc;
    }
    const unsigned int vec_elems = in_dim & ~7u;
    const uint4* row_vec = reinterpret_cast<const uint4*>(row);
    for (unsigned int i = lane * 8u; i < vec_elems; i += TALU_WARP_SIZE * 8u) {
        const uint4 packed = row_vec[i >> 3];
        const unsigned int w0 = packed.x;
        const unsigned int w1 = packed.y;
        const unsigned int w2 = packed.z;
        const unsigned int w3 = packed.w;
        acc = fmaf(input[i + 0], talu_decode_f16_u16((unsigned short)(w0 & 0xFFFFu)), acc);
        acc = fmaf(input[i + 1], talu_decode_f16_u16((unsigned short)(w0 >> 16)), acc);
        acc = fmaf(input[i + 2], talu_decode_f16_u16((unsigned short)(w1 & 0xFFFFu)), acc);
        acc = fmaf(input[i + 3], talu_decode_f16_u16((unsigned short)(w1 >> 16)), acc);
        acc = fmaf(input[i + 4], talu_decode_f16_u16((unsigned short)(w2 & 0xFFFFu)), acc);
        acc = fmaf(input[i + 5], talu_decode_f16_u16((unsigned short)(w2 >> 16)), acc);
        acc = fmaf(input[i + 6], talu_decode_f16_u16((unsigned short)(w3 & 0xFFFFu)), acc);
        acc = fmaf(input[i + 7], talu_decode_f16_u16((unsigned short)(w3 >> 16)), acc);
    }
    for (unsigned int i = vec_elems + lane; i < in_dim; i += TALU_WARP_SIZE) {
        acc = fmaf(input[i], talu_decode_f16_u16(row[i]), acc);
    }
    return acc;
}

static __device__ __forceinline__ float talu_dot_bf16_u16_vec8(
    const float* input,
    const unsigned short* row,
    unsigned int in_dim,
    unsigned int lane
) {
    float acc = 0.0f;
    const bool can_vec8 = ((((unsigned long long)row) & 0xFu) == 0u);
    if (!can_vec8) {
        for (unsigned int i = lane; i < in_dim; i += TALU_WARP_SIZE) {
            acc = fmaf(input[i], talu_decode_bf16_u16(row[i]), acc);
        }
        return acc;
    }
    const unsigned int vec_elems = in_dim & ~7u;
    const uint4* row_vec = reinterpret_cast<const uint4*>(row);
    for (unsigned int i = lane * 8u; i < vec_elems; i += TALU_WARP_SIZE * 8u) {
        const uint4 packed = row_vec[i >> 3];
        const unsigned int w0 = packed.x;
        const unsigned int w1 = packed.y;
        const unsigned int w2 = packed.z;
        const unsigned int w3 = packed.w;
        acc = fmaf(input[i + 0], talu_decode_bf16_u16((unsigned short)(w0 & 0xFFFFu)), acc);
        acc = fmaf(input[i + 1], talu_decode_bf16_u16((unsigned short)(w0 >> 16)), acc);
        acc = fmaf(input[i + 2], talu_decode_bf16_u16((unsigned short)(w1 & 0xFFFFu)), acc);
        acc = fmaf(input[i + 3], talu_decode_bf16_u16((unsigned short)(w1 >> 16)), acc);
        acc = fmaf(input[i + 4], talu_decode_bf16_u16((unsigned short)(w2 & 0xFFFFu)), acc);
        acc = fmaf(input[i + 5], talu_decode_bf16_u16((unsigned short)(w2 >> 16)), acc);
        acc = fmaf(input[i + 6], talu_decode_bf16_u16((unsigned short)(w3 & 0xFFFFu)), acc);
        acc = fmaf(input[i + 7], talu_decode_bf16_u16((unsigned short)(w3 >> 16)), acc);
    }
    for (unsigned int i = vec_elems + lane; i < in_dim; i += TALU_WARP_SIZE) {
        acc = fmaf(input[i], talu_decode_bf16_u16(row[i]), acc);
    }
    return acc;
}

extern "C" __global__ void talu_matmul_f16_f32(
    const float* input,
    const unsigned short* weight,
    float* out,
    unsigned int rows,
    unsigned int in_dim,
    unsigned int out_dim
) {
    const unsigned int lane = threadIdx.x;
    const unsigned int warp_id = threadIdx.y;
    const unsigned int warps_per_block = blockDim.y;
    const unsigned int out_idx = blockIdx.x * warps_per_block + warp_id;
    const unsigned int row_idx = blockIdx.y;
    if (row_idx >= rows || out_idx >= out_dim) return;

    const float* input_row = input + (unsigned long long)row_idx * in_dim;
    const unsigned short* weight_row = weight + (unsigned long long)out_idx * in_dim;
    __shared__ float input_tile[TALU_MATMUL_TILE_K];
    const unsigned int threads_per_block = TALU_WARP_SIZE * warps_per_block;
    const unsigned int linear_tid = warp_id * TALU_WARP_SIZE + lane;
    float acc = 0.0f;

    for (unsigned int base = 0; base < in_dim; base += TALU_MATMUL_TILE_K) {
        for (unsigned int t = linear_tid; t < TALU_MATMUL_TILE_K; t += threads_per_block) {
            const unsigned int idx = base + t;
            input_tile[t] = (idx < in_dim) ? input_row[idx] : 0.0f;
        }
        __syncthreads();

        const unsigned int tile_elems = min(TALU_MATMUL_TILE_K, in_dim - base);
        for (unsigned int i = lane; i < tile_elems; i += TALU_WARP_SIZE) {
            const unsigned int idx = base + i;
            acc = fmaf(input_tile[i], talu_decode_f16_u16(weight_row[idx]), acc);
        }
        __syncthreads();
    }
    acc = talu_warp_sum_f32(acc);
    if (lane == 0) {
        out[(unsigned long long)row_idx * out_dim + out_idx] = acc;
    }
}

extern "C" __global__ void talu_matmul_bf16_f32(
    const float* input,
    const unsigned short* weight,
    float* out,
    unsigned int rows,
    unsigned int in_dim,
    unsigned int out_dim
) {
    const unsigned int lane = threadIdx.x;
    const unsigned int warp_id = threadIdx.y;
    const unsigned int warps_per_block = blockDim.y;
    const unsigned int out_idx = blockIdx.x * warps_per_block + warp_id;
    const unsigned int row_idx = blockIdx.y;
    if (row_idx >= rows || out_idx >= out_dim) return;

    const float* input_row = input + (unsigned long long)row_idx * in_dim;
    const unsigned short* weight_row = weight + (unsigned long long)out_idx * in_dim;
    __shared__ float input_tile[TALU_MATMUL_TILE_K];
    const unsigned int threads_per_block = TALU_WARP_SIZE * warps_per_block;
    const unsigned int linear_tid = warp_id * TALU_WARP_SIZE + lane;
    float acc = 0.0f;

    for (unsigned int base = 0; base < in_dim; base += TALU_MATMUL_TILE_K) {
        for (unsigned int t = linear_tid; t < TALU_MATMUL_TILE_K; t += threads_per_block) {
            const unsigned int idx = base + t;
            input_tile[t] = (idx < in_dim) ? input_row[idx] : 0.0f;
        }
        __syncthreads();

        const unsigned int tile_elems = min(TALU_MATMUL_TILE_K, in_dim - base);
        for (unsigned int i = lane; i < tile_elems; i += TALU_WARP_SIZE) {
            const unsigned int idx = base + i;
            acc = fmaf(input_tile[i], talu_decode_bf16_u16(weight_row[idx]), acc);
        }
        __syncthreads();
    }
    acc = talu_warp_sum_f32(acc);
    if (lane == 0) {
        out[(unsigned long long)row_idx * out_dim + out_idx] = acc;
    }
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
    float acc = talu_dot_f16_u16_vec8(input, row, in_dim, lane);
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
    float acc = talu_dot_bf16_u16_vec8(input, row, in_dim, lane);
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

    float acc = talu_dot_f16_u16_vec8(input, row, in_dim, lane);
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

    float acc = talu_dot_bf16_u16_vec8(input, row, in_dim, lane);
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

    float acc = talu_dot_f16_u16_vec8(input, row, in_dim, lane);
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

    float acc = talu_dot_bf16_u16_vec8(input, row, in_dim, lane);
    acc = talu_warp_sum_f32(acc);
    if (lane == 0) *out_ptr = acc;
}

extern "C" __global__ void talu_matvec_gate_up_silu_f16_f32(
    const float* input,
    const unsigned short* gate_weight,
    const unsigned short* up_weight,
    float* out,
    unsigned int out_dim,
    unsigned int in_dim
) {
    const unsigned int lane = threadIdx.x & (TALU_WARP_SIZE - 1u);
    const unsigned int warp_id = threadIdx.x / TALU_WARP_SIZE;
    const unsigned int warps_per_block = blockDim.x / TALU_WARP_SIZE;
    const unsigned int out_idx = blockIdx.x * warps_per_block + warp_id;
    if (out_idx >= out_dim) return;

    const unsigned short* gate_row = gate_weight + (unsigned long long)out_idx * in_dim;
    const unsigned short* up_row = up_weight + (unsigned long long)out_idx * in_dim;
    float gate_acc = talu_dot_f16_u16_vec8(input, gate_row, in_dim, lane);
    float up_acc = talu_dot_f16_u16_vec8(input, up_row, in_dim, lane);
    gate_acc = talu_warp_sum_f32(gate_acc);
    up_acc = talu_warp_sum_f32(up_acc);
    if (lane == 0) {
        const float sigma = 1.0f / (1.0f + expf(-gate_acc));
        out[out_idx] = gate_acc * sigma * up_acc;
    }
}

extern "C" __global__ void talu_matvec_gate_up_silu_bf16_f32(
    const float* input,
    const unsigned short* gate_weight,
    const unsigned short* up_weight,
    float* out,
    unsigned int out_dim,
    unsigned int in_dim
) {
    const unsigned int lane = threadIdx.x & (TALU_WARP_SIZE - 1u);
    const unsigned int warp_id = threadIdx.x / TALU_WARP_SIZE;
    const unsigned int warps_per_block = blockDim.x / TALU_WARP_SIZE;
    const unsigned int out_idx = blockIdx.x * warps_per_block + warp_id;
    if (out_idx >= out_dim) return;

    const unsigned short* gate_row = gate_weight + (unsigned long long)out_idx * in_dim;
    const unsigned short* up_row = up_weight + (unsigned long long)out_idx * in_dim;
    float gate_acc = talu_dot_bf16_u16_vec8(input, gate_row, in_dim, lane);
    float up_acc = talu_dot_bf16_u16_vec8(input, up_row, in_dim, lane);
    gate_acc = talu_warp_sum_f32(gate_acc);
    up_acc = talu_warp_sum_f32(up_acc);
    if (lane == 0) {
        const float sigma = 1.0f / (1.0f + expf(-gate_acc));
        out[out_idx] = gate_acc * sigma * up_acc;
    }
}
