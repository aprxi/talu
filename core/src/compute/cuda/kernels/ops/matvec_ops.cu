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

static __device__ __forceinline__ float2 talu_decode_f16_pair_u32(unsigned int packed) {
    const __half2 pair = *reinterpret_cast<const __half2*>(&packed);
    return __half22float2(pair);
}

static __device__ __forceinline__ float talu_dot_f16_u16_vec8(
    const float* input,
    const unsigned short* row,
    unsigned int in_dim,
    unsigned int lane
) {
    float acc = 0.0f;
    const bool can_vec8 = ((((unsigned long long)row) & 0xFu) == 0u) and
        ((((unsigned long long)input) & 0xFu) == 0u);
    if (!can_vec8) {
        for (unsigned int i = lane; i < in_dim; i += TALU_WARP_SIZE) {
            acc = fmaf(input[i], talu_decode_f16_u16(row[i]), acc);
        }
        return acc;
    }
    const unsigned int vec_elems = in_dim & ~7u;
    const uint4* row_vec = reinterpret_cast<const uint4*>(row);
    const float4* input_vec = reinterpret_cast<const float4*>(input);
    for (unsigned int i = lane * 8u; i < vec_elems; i += TALU_WARP_SIZE * 8u) {
        uint4 packed;
        asm volatile(
            "ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
            : "=r"(packed.x), "=r"(packed.y), "=r"(packed.z), "=r"(packed.w)
            : "l"(row_vec + (i >> 3)));
        const unsigned int w0 = packed.x;
        const unsigned int w1 = packed.y;
        const unsigned int w2 = packed.z;
        const unsigned int w3 = packed.w;
        const unsigned int input_vec_idx = i >> 2;
        const float4 in0 = input_vec[input_vec_idx];
        const float4 in1 = input_vec[input_vec_idx + 1];
        const float2 f0 = talu_decode_f16_pair_u32(w0);
        const float2 f1 = talu_decode_f16_pair_u32(w1);
        const float2 f2 = talu_decode_f16_pair_u32(w2);
        const float2 f3 = talu_decode_f16_pair_u32(w3);
        acc = fmaf(in0.x, f0.x, acc);
        acc = fmaf(in0.y, f0.y, acc);
        acc = fmaf(in0.z, f1.x, acc);
        acc = fmaf(in0.w, f1.y, acc);
        acc = fmaf(in1.x, f2.x, acc);
        acc = fmaf(in1.y, f2.y, acc);
        acc = fmaf(in1.z, f3.x, acc);
        acc = fmaf(in1.w, f3.y, acc);
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
    const bool can_vec8 = ((((unsigned long long)row) & 0xFu) == 0u) and
        ((((unsigned long long)input) & 0xFu) == 0u);
    if (!can_vec8) {
        for (unsigned int i = lane; i < in_dim; i += TALU_WARP_SIZE) {
            acc = fmaf(input[i], talu_decode_bf16_u16(row[i]), acc);
        }
        return acc;
    }
    const unsigned int vec_elems = in_dim & ~7u;
    const uint4* row_vec = reinterpret_cast<const uint4*>(row);
    const float4* input_vec = reinterpret_cast<const float4*>(input);
    for (unsigned int i = lane * 8u; i < vec_elems; i += TALU_WARP_SIZE * 8u) {
        uint4 packed;
        asm volatile(
            "ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
            : "=r"(packed.x), "=r"(packed.y), "=r"(packed.z), "=r"(packed.w)
            : "l"(row_vec + (i >> 3)));
        const unsigned int w0 = packed.x;
        const unsigned int w1 = packed.y;
        const unsigned int w2 = packed.z;
        const unsigned int w3 = packed.w;
        const unsigned int input_vec_idx = i >> 2;
        const float4 in0 = input_vec[input_vec_idx];
        const float4 in1 = input_vec[input_vec_idx + 1];
        acc = fmaf(in0.x, talu_decode_bf16_u16((unsigned short)(w0 & 0xFFFFu)), acc);
        acc = fmaf(in0.y, talu_decode_bf16_u16((unsigned short)(w0 >> 16)), acc);
        acc = fmaf(in0.z, talu_decode_bf16_u16((unsigned short)(w1 & 0xFFFFu)), acc);
        acc = fmaf(in0.w, talu_decode_bf16_u16((unsigned short)(w1 >> 16)), acc);
        acc = fmaf(in1.x, talu_decode_bf16_u16((unsigned short)(w2 & 0xFFFFu)), acc);
        acc = fmaf(in1.y, talu_decode_bf16_u16((unsigned short)(w2 >> 16)), acc);
        acc = fmaf(in1.z, talu_decode_bf16_u16((unsigned short)(w3 & 0xFFFFu)), acc);
        acc = fmaf(in1.w, talu_decode_bf16_u16((unsigned short)(w3 >> 16)), acc);
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
    unsigned int out_dim,
    const float* residual
) {
    const unsigned int lane = threadIdx.x & (TALU_WARP_SIZE - 1u);
    const unsigned int warp_id = threadIdx.x / TALU_WARP_SIZE;
    const unsigned int warps_per_block = blockDim.x / TALU_WARP_SIZE;
    const unsigned int out_idx = blockIdx.x * warps_per_block + warp_id;
    if (out_idx >= out_dim) return;
    const unsigned short* row = weight + (unsigned long long)out_idx * in_dim;
    float acc = talu_dot_f16_u16_vec8(input, row, in_dim, lane);
    acc = talu_warp_sum_f32(acc);
    if (lane == 0) out[out_idx] = residual ? acc + residual[out_idx] : acc;
}

extern "C" __global__ void talu_matvec_bf16_f32(
    const float* input,
    const unsigned short* weight,
    float* out,
    unsigned int in_dim,
    unsigned int out_dim,
    const float* residual
) {
    const unsigned int lane = threadIdx.x & (TALU_WARP_SIZE - 1u);
    const unsigned int warp_id = threadIdx.x / TALU_WARP_SIZE;
    const unsigned int warps_per_block = blockDim.x / TALU_WARP_SIZE;
    const unsigned int out_idx = blockIdx.x * warps_per_block + warp_id;
    if (out_idx >= out_dim) return;
    const unsigned short* row = weight + (unsigned long long)out_idx * in_dim;
    float acc = talu_dot_bf16_u16_vec8(input, row, in_dim, lane);
    acc = talu_warp_sum_f32(acc);
    if (lane == 0) out[out_idx] = residual ? acc + residual[out_idx] : acc;
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
    float gr0 = 0.0f, gr1 = 0.0f, ur0 = 0.0f, ur1 = 0.0f;
    const bool can_vec8 = ((((unsigned long long)gate_row) & 0xFu) == 0u) and
        ((((unsigned long long)up_row) & 0xFu) == 0u) and
        ((((unsigned long long)input) & 0xFu) == 0u);
    if (can_vec8) {
        const unsigned int vec_elems = in_dim & ~7u;
        const uint4* gate_vec = reinterpret_cast<const uint4*>(gate_row);
        const uint4* up_vec = reinterpret_cast<const uint4*>(up_row);
        const float4* input_vec = reinterpret_cast<const float4*>(input);
        for (unsigned int i = lane * 8u; i < vec_elems; i += TALU_WARP_SIZE * 8u) {
            const unsigned int vi = i >> 3;
            uint4 g_packed;
            asm volatile(
                "ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
                : "=r"(g_packed.x), "=r"(g_packed.y), "=r"(g_packed.z), "=r"(g_packed.w)
                : "l"(gate_vec + vi));
            uint4 u_packed;
            asm volatile(
                "ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
                : "=r"(u_packed.x), "=r"(u_packed.y), "=r"(u_packed.z), "=r"(u_packed.w)
                : "l"(up_vec + vi));
            const unsigned int input_vec_idx = i >> 2;
            const float4 in0 = input_vec[input_vec_idx];
            const float4 in1 = input_vec[input_vec_idx + 1];
            const float2 gf0 = talu_decode_f16_pair_u32(g_packed.x);
            const float2 gf1 = talu_decode_f16_pair_u32(g_packed.y);
            const float2 gf2 = talu_decode_f16_pair_u32(g_packed.z);
            const float2 gf3 = talu_decode_f16_pair_u32(g_packed.w);
            const float2 uf0 = talu_decode_f16_pair_u32(u_packed.x);
            const float2 uf1 = talu_decode_f16_pair_u32(u_packed.y);
            const float2 uf2 = talu_decode_f16_pair_u32(u_packed.z);
            const float2 uf3 = talu_decode_f16_pair_u32(u_packed.w);
            gr0 = fmaf(in0.x, gf0.x, gr0);
            gr1 = fmaf(in0.y, gf0.y, gr1);
            gr0 = fmaf(in0.z, gf1.x, gr0);
            gr1 = fmaf(in0.w, gf1.y, gr1);
            gr0 = fmaf(in1.x, gf2.x, gr0);
            gr1 = fmaf(in1.y, gf2.y, gr1);
            gr0 = fmaf(in1.z, gf3.x, gr0);
            gr1 = fmaf(in1.w, gf3.y, gr1);
            ur0 = fmaf(in0.x, uf0.x, ur0);
            ur1 = fmaf(in0.y, uf0.y, ur1);
            ur0 = fmaf(in0.z, uf1.x, ur0);
            ur1 = fmaf(in0.w, uf1.y, ur1);
            ur0 = fmaf(in1.x, uf2.x, ur0);
            ur1 = fmaf(in1.y, uf2.y, ur1);
            ur0 = fmaf(in1.z, uf3.x, ur0);
            ur1 = fmaf(in1.w, uf3.y, ur1);
        }
        for (unsigned int i = vec_elems + lane; i < in_dim; i += TALU_WARP_SIZE) {
            const float in_val = input[i];
            gr0 = fmaf(in_val, talu_decode_f16_u16(gate_row[i]), gr0);
            ur0 = fmaf(in_val, talu_decode_f16_u16(up_row[i]), ur0);
        }
    } else {
        for (unsigned int i = lane; i < in_dim; i += TALU_WARP_SIZE) {
            const float in_val = input[i];
            gr0 = fmaf(in_val, talu_decode_f16_u16(gate_row[i]), gr0);
            ur0 = fmaf(in_val, talu_decode_f16_u16(up_row[i]), ur0);
        }
    }
    float gate_acc = talu_warp_sum_f32(gr0 + gr1);
    float up_acc = talu_warp_sum_f32(ur0 + ur1);
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
    float gr0 = 0.0f, gr1 = 0.0f, ur0 = 0.0f, ur1 = 0.0f;
    const bool can_vec8 = ((((unsigned long long)gate_row) & 0xFu) == 0u) and
        ((((unsigned long long)up_row) & 0xFu) == 0u) and
        ((((unsigned long long)input) & 0xFu) == 0u);
    if (can_vec8) {
        const unsigned int vec_elems = in_dim & ~7u;
        const uint4* gate_vec = reinterpret_cast<const uint4*>(gate_row);
        const uint4* up_vec = reinterpret_cast<const uint4*>(up_row);
        const float4* input_vec = reinterpret_cast<const float4*>(input);
        for (unsigned int i = lane * 8u; i < vec_elems; i += TALU_WARP_SIZE * 8u) {
            const unsigned int vi = i >> 3;
            uint4 g_packed;
            asm volatile(
                "ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
                : "=r"(g_packed.x), "=r"(g_packed.y), "=r"(g_packed.z), "=r"(g_packed.w)
                : "l"(gate_vec + vi));
            uint4 u_packed;
            asm volatile(
                "ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
                : "=r"(u_packed.x), "=r"(u_packed.y), "=r"(u_packed.z), "=r"(u_packed.w)
                : "l"(up_vec + vi));
            const unsigned int input_vec_idx = i >> 2;
            const float4 in0 = input_vec[input_vec_idx];
            const float4 in1 = input_vec[input_vec_idx + 1];
            gr0 = fmaf(in0.x, __uint_as_float(g_packed.x << 16), gr0);
            gr1 = fmaf(in0.y, __uint_as_float(g_packed.x & 0xFFFF0000u), gr1);
            gr0 = fmaf(in0.z, __uint_as_float(g_packed.y << 16), gr0);
            gr1 = fmaf(in0.w, __uint_as_float(g_packed.y & 0xFFFF0000u), gr1);
            gr0 = fmaf(in1.x, __uint_as_float(g_packed.z << 16), gr0);
            gr1 = fmaf(in1.y, __uint_as_float(g_packed.z & 0xFFFF0000u), gr1);
            gr0 = fmaf(in1.z, __uint_as_float(g_packed.w << 16), gr0);
            gr1 = fmaf(in1.w, __uint_as_float(g_packed.w & 0xFFFF0000u), gr1);
            ur0 = fmaf(in0.x, __uint_as_float(u_packed.x << 16), ur0);
            ur1 = fmaf(in0.y, __uint_as_float(u_packed.x & 0xFFFF0000u), ur1);
            ur0 = fmaf(in0.z, __uint_as_float(u_packed.y << 16), ur0);
            ur1 = fmaf(in0.w, __uint_as_float(u_packed.y & 0xFFFF0000u), ur1);
            ur0 = fmaf(in1.x, __uint_as_float(u_packed.z << 16), ur0);
            ur1 = fmaf(in1.y, __uint_as_float(u_packed.z & 0xFFFF0000u), ur1);
            ur0 = fmaf(in1.z, __uint_as_float(u_packed.w << 16), ur0);
            ur1 = fmaf(in1.w, __uint_as_float(u_packed.w & 0xFFFF0000u), ur1);
        }
        for (unsigned int i = vec_elems + lane; i < in_dim; i += TALU_WARP_SIZE) {
            const float in_val = input[i];
            gr0 = fmaf(in_val, talu_decode_bf16_u16(gate_row[i]), gr0);
            ur0 = fmaf(in_val, talu_decode_bf16_u16(up_row[i]), ur0);
        }
    } else {
        for (unsigned int i = lane; i < in_dim; i += TALU_WARP_SIZE) {
            const float in_val = input[i];
            gr0 = fmaf(in_val, talu_decode_bf16_u16(gate_row[i]), gr0);
            ur0 = fmaf(in_val, talu_decode_bf16_u16(up_row[i]), ur0);
        }
    }
    float gate_acc = talu_warp_sum_f32(gr0 + gr1);
    float up_acc = talu_warp_sum_f32(ur0 + ur1);
    if (lane == 0) {
        const float sigma = 1.0f / (1.0f + expf(-gate_acc));
        out[out_idx] = gate_acc * sigma * up_acc;
    }
}
