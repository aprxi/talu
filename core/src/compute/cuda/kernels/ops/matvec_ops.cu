static constexpr unsigned int TALU_WARP_SIZE = 32;
static constexpr unsigned int TALU_MATMUL_TILE_K = 2048;
static constexpr unsigned int TALU_BATCH_TILE = 8;

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
            "ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
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
            "ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
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

template <unsigned int BATCH>
static __device__ void talu_matvec_f16_batched(
    const float* input,
    const unsigned short* weight,
    float* out,
    unsigned int in_dim,
    unsigned int out_dim,
    unsigned int batch_rows,
    const float* residual
) {
    const unsigned int lane = threadIdx.x & (TALU_WARP_SIZE - 1u);
    const unsigned int warp_id = threadIdx.x / TALU_WARP_SIZE;
    const unsigned int warps_per_block = blockDim.x / TALU_WARP_SIZE;
    const unsigned int out_idx = blockIdx.x * warps_per_block + warp_id;
    if (out_idx >= out_dim) return;

    const unsigned short* row = weight + (unsigned long long)out_idx * in_dim;
    const bool can_vec8 = ((((unsigned long long)row) & 0xFu) == 0u) and
        ((((unsigned long long)input) & 0xFu) == 0u);

    float acc[BATCH] = {};

    if (can_vec8) {
        const unsigned int vec_elems = in_dim & ~7u;
        const uint4* row_vec = reinterpret_cast<const uint4*>(row);
        for (unsigned int i = lane * 8u; i < vec_elems; i += TALU_WARP_SIZE * 8u) {
            uint4 packed;
            asm volatile(
                "ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
                : "=r"(packed.x), "=r"(packed.y), "=r"(packed.z), "=r"(packed.w)
                : "l"(row_vec + (i >> 3)));
            const float2 f0 = talu_decode_f16_pair_u32(packed.x);
            const float2 f1 = talu_decode_f16_pair_u32(packed.y);
            const float2 f2 = talu_decode_f16_pair_u32(packed.z);
            const float2 f3 = talu_decode_f16_pair_u32(packed.w);
            const unsigned int input_vec_idx = i >> 2;
            #pragma unroll
            for (unsigned int b = 0; b < BATCH; b++) {
                if (b >= batch_rows) break;
                const float4* input_vec = reinterpret_cast<const float4*>(input + (unsigned long long)b * in_dim);
                const float4 in0 = input_vec[input_vec_idx];
                const float4 in1 = input_vec[input_vec_idx + 1];
                acc[b] = fmaf(in0.x, f0.x, acc[b]);
                acc[b] = fmaf(in0.y, f0.y, acc[b]);
                acc[b] = fmaf(in0.z, f1.x, acc[b]);
                acc[b] = fmaf(in0.w, f1.y, acc[b]);
                acc[b] = fmaf(in1.x, f2.x, acc[b]);
                acc[b] = fmaf(in1.y, f2.y, acc[b]);
                acc[b] = fmaf(in1.z, f3.x, acc[b]);
                acc[b] = fmaf(in1.w, f3.y, acc[b]);
            }
        }
        for (unsigned int i = vec_elems + lane; i < in_dim; i += TALU_WARP_SIZE) {
            const float w = talu_decode_f16_u16(row[i]);
            #pragma unroll
            for (unsigned int b = 0; b < BATCH; b++) {
                if (b >= batch_rows) break;
                acc[b] = fmaf(input[(unsigned long long)b * in_dim + i], w, acc[b]);
            }
        }
    } else {
        for (unsigned int i = lane; i < in_dim; i += TALU_WARP_SIZE) {
            const float w = talu_decode_f16_u16(row[i]);
            #pragma unroll
            for (unsigned int b = 0; b < BATCH; b++) {
                if (b >= batch_rows) break;
                acc[b] = fmaf(input[(unsigned long long)b * in_dim + i], w, acc[b]);
            }
        }
    }

    #pragma unroll
    for (unsigned int b = 0; b < BATCH; b++) {
        if (b >= batch_rows) break;
        float result = talu_warp_sum_f32(acc[b]);
        if (lane == 0) {
            float* out_row = out + (unsigned long long)b * out_dim;
            if (residual) {
                out_row[out_idx] = result + residual[(unsigned long long)b * out_dim + out_idx];
            } else {
                out_row[out_idx] = result;
            }
        }
    }
}

extern "C" __global__ void talu_matvec_f16_f32(
    const float* input, const unsigned short* weight, float* out,
    unsigned int in_dim, unsigned int out_dim, unsigned int batch_rows,
    const float* residual
) {
    const unsigned int row_idx = blockIdx.y;
    if (row_idx >= batch_rows) return;
    const float* input_row = input + (unsigned long long)row_idx * in_dim;
    float* out_row = out + (unsigned long long)row_idx * out_dim;
    const float* residual_row = residual ? (residual + (unsigned long long)row_idx * out_dim) : nullptr;
    talu_matvec_f16_batched<1>(input_row, weight, out_row, in_dim, out_dim, 1, residual_row);
}

// Inner-batch GEMV: weight loaded once from DRAM, reused across batch rows
// from registers.  BATCH=1 compiles to identical code as the original
// single-row kernel.  BATCH=2/4/8 adds one accumulator per extra row.
template <unsigned int BATCH>
static __device__ void talu_matvec_bf16_batched(
    const float* input,
    const unsigned short* weight,
    float* out,
    unsigned int in_dim,
    unsigned int out_dim,
    unsigned int batch_rows,
    const float* residual
) {
    const unsigned int lane = threadIdx.x & (TALU_WARP_SIZE - 1u);
    const unsigned int warp_id = threadIdx.x / TALU_WARP_SIZE;
    const unsigned int warps_per_block = blockDim.x / TALU_WARP_SIZE;
    const unsigned int out_idx = blockIdx.x * warps_per_block + warp_id;
    if (out_idx >= out_dim) return;

    const unsigned short* row = weight + (unsigned long long)out_idx * in_dim;
    const bool can_vec8 = ((((unsigned long long)row) & 0xFu) == 0u) and
        ((((unsigned long long)input) & 0xFu) == 0u);

    float acc[BATCH] = {};

    if (can_vec8) {
        const unsigned int vec_elems = in_dim & ~7u;
        const uint4* row_vec = reinterpret_cast<const uint4*>(row);
        for (unsigned int i = lane * 8u; i < vec_elems; i += TALU_WARP_SIZE * 8u) {
            uint4 packed;
            asm volatile(
                "ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
                : "=r"(packed.x), "=r"(packed.y), "=r"(packed.z), "=r"(packed.w)
                : "l"(row_vec + (i >> 3)));
            const float w0 = talu_decode_bf16_u16((unsigned short)(packed.x & 0xFFFFu));
            const float w1 = talu_decode_bf16_u16((unsigned short)(packed.x >> 16));
            const float w2 = talu_decode_bf16_u16((unsigned short)(packed.y & 0xFFFFu));
            const float w3 = talu_decode_bf16_u16((unsigned short)(packed.y >> 16));
            const float w4 = talu_decode_bf16_u16((unsigned short)(packed.z & 0xFFFFu));
            const float w5 = talu_decode_bf16_u16((unsigned short)(packed.z >> 16));
            const float w6 = talu_decode_bf16_u16((unsigned short)(packed.w & 0xFFFFu));
            const float w7 = talu_decode_bf16_u16((unsigned short)(packed.w >> 16));
            const unsigned int input_vec_idx = i >> 2;
            #pragma unroll
            for (unsigned int b = 0; b < BATCH; b++) {
                if (b >= batch_rows) break;
                const float4* input_vec = reinterpret_cast<const float4*>(input + (unsigned long long)b * in_dim);
                const float4 in0 = input_vec[input_vec_idx];
                const float4 in1 = input_vec[input_vec_idx + 1];
                acc[b] = fmaf(in0.x, w0, acc[b]);
                acc[b] = fmaf(in0.y, w1, acc[b]);
                acc[b] = fmaf(in0.z, w2, acc[b]);
                acc[b] = fmaf(in0.w, w3, acc[b]);
                acc[b] = fmaf(in1.x, w4, acc[b]);
                acc[b] = fmaf(in1.y, w5, acc[b]);
                acc[b] = fmaf(in1.z, w6, acc[b]);
                acc[b] = fmaf(in1.w, w7, acc[b]);
            }
        }
        for (unsigned int i = vec_elems + lane; i < in_dim; i += TALU_WARP_SIZE) {
            const float w = talu_decode_bf16_u16(row[i]);
            #pragma unroll
            for (unsigned int b = 0; b < BATCH; b++) {
                if (b >= batch_rows) break;
                acc[b] = fmaf(input[(unsigned long long)b * in_dim + i], w, acc[b]);
            }
        }
    } else {
        for (unsigned int i = lane; i < in_dim; i += TALU_WARP_SIZE) {
            const float w = talu_decode_bf16_u16(row[i]);
            #pragma unroll
            for (unsigned int b = 0; b < BATCH; b++) {
                if (b >= batch_rows) break;
                acc[b] = fmaf(input[(unsigned long long)b * in_dim + i], w, acc[b]);
            }
        }
    }

    #pragma unroll
    for (unsigned int b = 0; b < BATCH; b++) {
        if (b >= batch_rows) break;
        float result = talu_warp_sum_f32(acc[b]);
        if (lane == 0) {
            float* out_row = out + (unsigned long long)b * out_dim;
            if (residual) {
                out_row[out_idx] = result + residual[(unsigned long long)b * out_dim + out_idx];
            } else {
                out_row[out_idx] = result;
            }
        }
    }
}

extern "C" __global__ void talu_matvec_bf16_f32(
    const float* input, const unsigned short* weight, float* out,
    unsigned int in_dim, unsigned int out_dim, unsigned int batch_rows,
    const float* residual
) {
    const unsigned int row_idx = blockIdx.y;
    if (row_idx >= batch_rows) return;
    const float* input_row = input + (unsigned long long)row_idx * in_dim;
    float* out_row = out + (unsigned long long)row_idx * out_dim;
    const float* residual_row = residual ? (residual + (unsigned long long)row_idx * out_dim) : nullptr;
    talu_matvec_bf16_batched<1>(input_row, weight, out_row, in_dim, out_dim, 1, residual_row);
}

extern "C" __global__
void talu_matvec_f16_f32_batch(
    const float* input, const unsigned short* weight, float* out,
    unsigned int in_dim, unsigned int out_dim, unsigned int batch_rows,
    const float* residual
) {
    const unsigned int batch_base = blockIdx.y * TALU_BATCH_TILE;
    if (batch_base >= batch_rows) return;
    const unsigned int rows_remaining = batch_rows - batch_base;
    const unsigned int tile_rows = rows_remaining < TALU_BATCH_TILE ? rows_remaining : TALU_BATCH_TILE;

    const float* input_tile = input + (unsigned long long)batch_base * in_dim;
    float* out_tile = out + (unsigned long long)batch_base * out_dim;
    const float* residual_tile = residual ? (residual + (unsigned long long)batch_base * out_dim) : nullptr;

    switch (tile_rows) {
        case 1u: talu_matvec_f16_batched<1>(input_tile, weight, out_tile, in_dim, out_dim, 1u, residual_tile); break;
        case 2u: talu_matvec_f16_batched<2>(input_tile, weight, out_tile, in_dim, out_dim, 2u, residual_tile); break;
        case 3u: talu_matvec_f16_batched<3>(input_tile, weight, out_tile, in_dim, out_dim, 3u, residual_tile); break;
        case 4u: talu_matvec_f16_batched<4>(input_tile, weight, out_tile, in_dim, out_dim, 4u, residual_tile); break;
        case 5u: talu_matvec_f16_batched<5>(input_tile, weight, out_tile, in_dim, out_dim, 5u, residual_tile); break;
        case 6u: talu_matvec_f16_batched<6>(input_tile, weight, out_tile, in_dim, out_dim, 6u, residual_tile); break;
        case 7u: talu_matvec_f16_batched<7>(input_tile, weight, out_tile, in_dim, out_dim, 7u, residual_tile); break;
        case 8u: talu_matvec_f16_batched<8>(input_tile, weight, out_tile, in_dim, out_dim, 8u, residual_tile); break;
        default: return;
    }
}

extern "C" __global__
void talu_matvec_bf16_f32_batch(
    const float* input, const unsigned short* weight, float* out,
    unsigned int in_dim, unsigned int out_dim, unsigned int batch_rows,
    const float* residual
) {
    const unsigned int batch_base = blockIdx.y * TALU_BATCH_TILE;
    if (batch_base >= batch_rows) return;
    const unsigned int rows_remaining = batch_rows - batch_base;
    const unsigned int tile_rows = rows_remaining < TALU_BATCH_TILE ? rows_remaining : TALU_BATCH_TILE;

    const float* input_tile = input + (unsigned long long)batch_base * in_dim;
    float* out_tile = out + (unsigned long long)batch_base * out_dim;
    const float* residual_tile = residual ? (residual + (unsigned long long)batch_base * out_dim) : nullptr;

    switch (tile_rows) {
        case 1u: talu_matvec_bf16_batched<1>(input_tile, weight, out_tile, in_dim, out_dim, 1u, residual_tile); break;
        case 2u: talu_matvec_bf16_batched<2>(input_tile, weight, out_tile, in_dim, out_dim, 2u, residual_tile); break;
        case 3u: talu_matvec_bf16_batched<3>(input_tile, weight, out_tile, in_dim, out_dim, 3u, residual_tile); break;
        case 4u: talu_matvec_bf16_batched<4>(input_tile, weight, out_tile, in_dim, out_dim, 4u, residual_tile); break;
        case 5u: talu_matvec_bf16_batched<5>(input_tile, weight, out_tile, in_dim, out_dim, 5u, residual_tile); break;
        case 6u: talu_matvec_bf16_batched<6>(input_tile, weight, out_tile, in_dim, out_dim, 6u, residual_tile); break;
        case 7u: talu_matvec_bf16_batched<7>(input_tile, weight, out_tile, in_dim, out_dim, 7u, residual_tile); break;
        case 8u: talu_matvec_bf16_batched<8>(input_tile, weight, out_tile, in_dim, out_dim, 8u, residual_tile); break;
        default: return;
    }
}

template <unsigned int BATCH, bool BF16>
__device__ __forceinline__ void talu_matvec_qkv_u16_batched(
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
    unsigned int in_dim,
    unsigned int batch_rows
) {
    const unsigned int lane = threadIdx.x & (TALU_WARP_SIZE - 1u);
    const unsigned int warp_id = threadIdx.x / TALU_WARP_SIZE;
    const unsigned int warps_per_block = blockDim.x / TALU_WARP_SIZE;
    const unsigned int out_index = blockIdx.x * warps_per_block + warp_id;
    const unsigned int qk_dim = q_out_dim + k_out_dim;
    const unsigned int total_dim = qk_dim + v_out_dim;
    if (out_index >= total_dim) return;

    const unsigned short* row = nullptr;
    float* out_base = nullptr;
    unsigned int out_dim = 0;
    unsigned int out_row = 0;
    if (out_index < q_out_dim) {
        out_row = out_index;
        row = q_weight + (unsigned long long)out_row * in_dim;
        out_base = q_out;
        out_dim = q_out_dim;
    } else if (out_index < qk_dim) {
        out_row = out_index - q_out_dim;
        row = k_weight + (unsigned long long)out_row * in_dim;
        out_base = k_out;
        out_dim = k_out_dim;
    } else {
        out_row = out_index - qk_dim;
        row = v_weight + (unsigned long long)out_row * in_dim;
        out_base = v_out;
        out_dim = v_out_dim;
    }

    const bool can_vec8 = ((((unsigned long long)row) & 0xFu) == 0u) and
        ((((unsigned long long)input) & 0xFu) == 0u);

    float acc[BATCH] = {};

    if (can_vec8) {
        const unsigned int vec_elems = in_dim & ~7u;
        const uint4* row_vec = reinterpret_cast<const uint4*>(row);
        for (unsigned int i = lane * 8u; i < vec_elems; i += TALU_WARP_SIZE * 8u) {
            uint4 packed;
            asm volatile(
                "ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
                : "=r"(packed.x), "=r"(packed.y), "=r"(packed.z), "=r"(packed.w)
                : "l"(row_vec + (i >> 3)));
            const unsigned int input_vec_idx = i >> 2;
            if (BF16) {
                const float w0 = talu_decode_bf16_u16((unsigned short)(packed.x & 0xFFFFu));
                const float w1 = talu_decode_bf16_u16((unsigned short)(packed.x >> 16));
                const float w2 = talu_decode_bf16_u16((unsigned short)(packed.y & 0xFFFFu));
                const float w3 = talu_decode_bf16_u16((unsigned short)(packed.y >> 16));
                const float w4 = talu_decode_bf16_u16((unsigned short)(packed.z & 0xFFFFu));
                const float w5 = talu_decode_bf16_u16((unsigned short)(packed.z >> 16));
                const float w6 = talu_decode_bf16_u16((unsigned short)(packed.w & 0xFFFFu));
                const float w7 = talu_decode_bf16_u16((unsigned short)(packed.w >> 16));
                #pragma unroll
                for (unsigned int b = 0; b < BATCH; b++) {
                    if (b >= batch_rows) break;
                    const float4* input_vec = reinterpret_cast<const float4*>(input + (unsigned long long)b * in_dim);
                    const float4 in0 = input_vec[input_vec_idx];
                    const float4 in1 = input_vec[input_vec_idx + 1];
                    acc[b] = fmaf(in0.x, w0, acc[b]);
                    acc[b] = fmaf(in0.y, w1, acc[b]);
                    acc[b] = fmaf(in0.z, w2, acc[b]);
                    acc[b] = fmaf(in0.w, w3, acc[b]);
                    acc[b] = fmaf(in1.x, w4, acc[b]);
                    acc[b] = fmaf(in1.y, w5, acc[b]);
                    acc[b] = fmaf(in1.z, w6, acc[b]);
                    acc[b] = fmaf(in1.w, w7, acc[b]);
                }
            } else {
                const float2 f0 = talu_decode_f16_pair_u32(packed.x);
                const float2 f1 = talu_decode_f16_pair_u32(packed.y);
                const float2 f2 = talu_decode_f16_pair_u32(packed.z);
                const float2 f3 = talu_decode_f16_pair_u32(packed.w);
                #pragma unroll
                for (unsigned int b = 0; b < BATCH; b++) {
                    if (b >= batch_rows) break;
                    const float4* input_vec = reinterpret_cast<const float4*>(input + (unsigned long long)b * in_dim);
                    const float4 in0 = input_vec[input_vec_idx];
                    const float4 in1 = input_vec[input_vec_idx + 1];
                    acc[b] = fmaf(in0.x, f0.x, acc[b]);
                    acc[b] = fmaf(in0.y, f0.y, acc[b]);
                    acc[b] = fmaf(in0.z, f1.x, acc[b]);
                    acc[b] = fmaf(in0.w, f1.y, acc[b]);
                    acc[b] = fmaf(in1.x, f2.x, acc[b]);
                    acc[b] = fmaf(in1.y, f2.y, acc[b]);
                    acc[b] = fmaf(in1.z, f3.x, acc[b]);
                    acc[b] = fmaf(in1.w, f3.y, acc[b]);
                }
            }
        }
        for (unsigned int i = vec_elems + lane; i < in_dim; i += TALU_WARP_SIZE) {
            const float w = BF16 ? talu_decode_bf16_u16(row[i]) : talu_decode_f16_u16(row[i]);
            #pragma unroll
            for (unsigned int b = 0; b < BATCH; ++b) {
                if (b >= batch_rows) break;
                acc[b] = fmaf(input[(unsigned long long)b * in_dim + i], w, acc[b]);
            }
        }
    } else {
        for (unsigned int i = lane; i < in_dim; i += TALU_WARP_SIZE) {
            const float w = BF16 ? talu_decode_bf16_u16(row[i]) : talu_decode_f16_u16(row[i]);
            #pragma unroll
            for (unsigned int b = 0; b < BATCH; ++b) {
                if (b >= batch_rows) break;
                acc[b] = fmaf(input[(unsigned long long)b * in_dim + i], w, acc[b]);
            }
        }
    }

    #pragma unroll
    for (unsigned int b = 0; b < BATCH; ++b) {
        if (b >= batch_rows) break;
        const float result = talu_warp_sum_f32(acc[b]);
        if (lane == 0) {
            float* out_row_ptr = out_base + (unsigned long long)b * out_dim;
            out_row_ptr[out_row] = result;
        }
    }
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
    const unsigned int row_idx = blockIdx.y;
    const float* input_row = input + (unsigned long long)row_idx * in_dim;
    float* q_out_row = q_out + (unsigned long long)row_idx * q_out_dim;
    float* k_out_row = k_out + (unsigned long long)row_idx * k_out_dim;
    float* v_out_row = v_out + (unsigned long long)row_idx * v_out_dim;
    talu_matvec_qkv_u16_batched<1, false>(
        input_row,
        q_weight,
        q_out_row,
        q_out_dim,
        k_weight,
        k_out_row,
        k_out_dim,
        v_weight,
        v_out_row,
        v_out_dim,
        in_dim,
        1
    );
}

extern "C" __global__
void talu_matvec_qkv_f16_f32_batch(
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
    unsigned int in_dim,
    unsigned int batch_rows
) {
    const unsigned int batch_base = blockIdx.y * TALU_BATCH_TILE;
    if (batch_base >= batch_rows) return;
    const unsigned int rows_remaining = batch_rows - batch_base;
    const unsigned int tile_rows = rows_remaining < TALU_BATCH_TILE ? rows_remaining : TALU_BATCH_TILE;

    const float* input_tile = input + (unsigned long long)batch_base * in_dim;
    float* q_out_tile = q_out + (unsigned long long)batch_base * q_out_dim;
    float* k_out_tile = k_out + (unsigned long long)batch_base * k_out_dim;
    float* v_out_tile = v_out + (unsigned long long)batch_base * v_out_dim;

    switch (tile_rows) {
        case 1u: talu_matvec_qkv_u16_batched<1, false>(input_tile, q_weight, q_out_tile, q_out_dim, k_weight, k_out_tile, k_out_dim, v_weight, v_out_tile, v_out_dim, in_dim, 1u); break;
        case 2u: talu_matvec_qkv_u16_batched<2, false>(input_tile, q_weight, q_out_tile, q_out_dim, k_weight, k_out_tile, k_out_dim, v_weight, v_out_tile, v_out_dim, in_dim, 2u); break;
        case 3u: talu_matvec_qkv_u16_batched<3, false>(input_tile, q_weight, q_out_tile, q_out_dim, k_weight, k_out_tile, k_out_dim, v_weight, v_out_tile, v_out_dim, in_dim, 3u); break;
        case 4u: talu_matvec_qkv_u16_batched<4, false>(input_tile, q_weight, q_out_tile, q_out_dim, k_weight, k_out_tile, k_out_dim, v_weight, v_out_tile, v_out_dim, in_dim, 4u); break;
        case 5u: talu_matvec_qkv_u16_batched<5, false>(input_tile, q_weight, q_out_tile, q_out_dim, k_weight, k_out_tile, k_out_dim, v_weight, v_out_tile, v_out_dim, in_dim, 5u); break;
        case 6u: talu_matvec_qkv_u16_batched<6, false>(input_tile, q_weight, q_out_tile, q_out_dim, k_weight, k_out_tile, k_out_dim, v_weight, v_out_tile, v_out_dim, in_dim, 6u); break;
        case 7u: talu_matvec_qkv_u16_batched<7, false>(input_tile, q_weight, q_out_tile, q_out_dim, k_weight, k_out_tile, k_out_dim, v_weight, v_out_tile, v_out_dim, in_dim, 7u); break;
        case 8u: talu_matvec_qkv_u16_batched<8, false>(input_tile, q_weight, q_out_tile, q_out_dim, k_weight, k_out_tile, k_out_dim, v_weight, v_out_tile, v_out_dim, in_dim, 8u); break;
        default: return;
    }
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
    const unsigned int row_idx = blockIdx.y;
    const float* input_row = input + (unsigned long long)row_idx * in_dim;
    float* q_out_row = q_out + (unsigned long long)row_idx * q_out_dim;
    float* k_out_row = k_out + (unsigned long long)row_idx * k_out_dim;
    float* v_out_row = v_out + (unsigned long long)row_idx * v_out_dim;
    talu_matvec_qkv_u16_batched<1, true>(
        input_row,
        q_weight,
        q_out_row,
        q_out_dim,
        k_weight,
        k_out_row,
        k_out_dim,
        v_weight,
        v_out_row,
        v_out_dim,
        in_dim,
        1
    );
}

extern "C" __global__
void talu_matvec_qkv_bf16_f32_batch(
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
    unsigned int in_dim,
    unsigned int batch_rows
) {
    const unsigned int batch_base = blockIdx.y * TALU_BATCH_TILE;
    if (batch_base >= batch_rows) return;
    const unsigned int rows_remaining = batch_rows - batch_base;
    const unsigned int tile_rows = rows_remaining < TALU_BATCH_TILE ? rows_remaining : TALU_BATCH_TILE;

    const float* input_tile = input + (unsigned long long)batch_base * in_dim;
    float* q_out_tile = q_out + (unsigned long long)batch_base * q_out_dim;
    float* k_out_tile = k_out + (unsigned long long)batch_base * k_out_dim;
    float* v_out_tile = v_out + (unsigned long long)batch_base * v_out_dim;

    switch (tile_rows) {
        case 1u: talu_matvec_qkv_u16_batched<1, true>(input_tile, q_weight, q_out_tile, q_out_dim, k_weight, k_out_tile, k_out_dim, v_weight, v_out_tile, v_out_dim, in_dim, 1u); break;
        case 2u: talu_matvec_qkv_u16_batched<2, true>(input_tile, q_weight, q_out_tile, q_out_dim, k_weight, k_out_tile, k_out_dim, v_weight, v_out_tile, v_out_dim, in_dim, 2u); break;
        case 3u: talu_matvec_qkv_u16_batched<3, true>(input_tile, q_weight, q_out_tile, q_out_dim, k_weight, k_out_tile, k_out_dim, v_weight, v_out_tile, v_out_dim, in_dim, 3u); break;
        case 4u: talu_matvec_qkv_u16_batched<4, true>(input_tile, q_weight, q_out_tile, q_out_dim, k_weight, k_out_tile, k_out_dim, v_weight, v_out_tile, v_out_dim, in_dim, 4u); break;
        case 5u: talu_matvec_qkv_u16_batched<5, true>(input_tile, q_weight, q_out_tile, q_out_dim, k_weight, k_out_tile, k_out_dim, v_weight, v_out_tile, v_out_dim, in_dim, 5u); break;
        case 6u: talu_matvec_qkv_u16_batched<6, true>(input_tile, q_weight, q_out_tile, q_out_dim, k_weight, k_out_tile, k_out_dim, v_weight, v_out_tile, v_out_dim, in_dim, 6u); break;
        case 7u: talu_matvec_qkv_u16_batched<7, true>(input_tile, q_weight, q_out_tile, q_out_dim, k_weight, k_out_tile, k_out_dim, v_weight, v_out_tile, v_out_dim, in_dim, 7u); break;
        case 8u: talu_matvec_qkv_u16_batched<8, true>(input_tile, q_weight, q_out_tile, q_out_dim, k_weight, k_out_tile, k_out_dim, v_weight, v_out_tile, v_out_dim, in_dim, 8u); break;
        default: return;
    }
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

// Inner-batch gate-up-SiLU (F16): gate+up weights loaded once from DRAM,
// reused across batch rows from registers.
template <unsigned int BATCH>
static __device__ void talu_gate_up_silu_f16_inner(
    const float* input,
    const unsigned short* gate_weight,
    const unsigned short* up_weight,
    float* out,
    unsigned int out_dim,
    unsigned int in_dim,
    unsigned int batch_rows
) {
    const unsigned int lane = threadIdx.x & (TALU_WARP_SIZE - 1u);
    const unsigned int warp_id = threadIdx.x / TALU_WARP_SIZE;
    const unsigned int warps_per_block = blockDim.x / TALU_WARP_SIZE;
    const unsigned int out_idx = blockIdx.x * warps_per_block + warp_id;
    if (out_idx >= out_dim) return;

    const unsigned short* gate_row = gate_weight + (unsigned long long)out_idx * in_dim;
    const unsigned short* up_row = up_weight + (unsigned long long)out_idx * in_dim;

    float gacc[BATCH] = {}, uacc[BATCH] = {};

    const bool can_vec8 = ((((unsigned long long)gate_row) & 0xFu) == 0u) and
        ((((unsigned long long)up_row) & 0xFu) == 0u) and
        ((((unsigned long long)input) & 0xFu) == 0u);
    if (can_vec8) {
        const unsigned int vec_elems = in_dim & ~7u;
        const uint4* gate_vec = reinterpret_cast<const uint4*>(gate_row);
        const uint4* up_vec = reinterpret_cast<const uint4*>(up_row);
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
            const float2 gf0 = talu_decode_f16_pair_u32(g_packed.x);
            const float2 gf1 = talu_decode_f16_pair_u32(g_packed.y);
            const float2 gf2 = talu_decode_f16_pair_u32(g_packed.z);
            const float2 gf3 = talu_decode_f16_pair_u32(g_packed.w);
            const float2 uf0 = talu_decode_f16_pair_u32(u_packed.x);
            const float2 uf1 = talu_decode_f16_pair_u32(u_packed.y);
            const float2 uf2 = talu_decode_f16_pair_u32(u_packed.z);
            const float2 uf3 = talu_decode_f16_pair_u32(u_packed.w);
            const unsigned int input_vec_idx = i >> 2;
            #pragma unroll
            for (unsigned int b = 0; b < BATCH; b++) {
                if (b >= batch_rows) break;
                const float4* input_vec = reinterpret_cast<const float4*>(input + (unsigned long long)b * in_dim);
                const float4 in0 = input_vec[input_vec_idx];
                const float4 in1 = input_vec[input_vec_idx + 1];
                gacc[b] = fmaf(in0.x, gf0.x, gacc[b]);  gacc[b] = fmaf(in0.y, gf0.y, gacc[b]);
                gacc[b] = fmaf(in0.z, gf1.x, gacc[b]);  gacc[b] = fmaf(in0.w, gf1.y, gacc[b]);
                gacc[b] = fmaf(in1.x, gf2.x, gacc[b]);  gacc[b] = fmaf(in1.y, gf2.y, gacc[b]);
                gacc[b] = fmaf(in1.z, gf3.x, gacc[b]);  gacc[b] = fmaf(in1.w, gf3.y, gacc[b]);
                uacc[b] = fmaf(in0.x, uf0.x, uacc[b]);  uacc[b] = fmaf(in0.y, uf0.y, uacc[b]);
                uacc[b] = fmaf(in0.z, uf1.x, uacc[b]);  uacc[b] = fmaf(in0.w, uf1.y, uacc[b]);
                uacc[b] = fmaf(in1.x, uf2.x, uacc[b]);  uacc[b] = fmaf(in1.y, uf2.y, uacc[b]);
                uacc[b] = fmaf(in1.z, uf3.x, uacc[b]);  uacc[b] = fmaf(in1.w, uf3.y, uacc[b]);
            }
        }
        for (unsigned int i = vec_elems + lane; i < in_dim; i += TALU_WARP_SIZE) {
            const float gw = talu_decode_f16_u16(gate_row[i]);
            const float uw = talu_decode_f16_u16(up_row[i]);
            #pragma unroll
            for (unsigned int b = 0; b < BATCH; b++) {
                if (b >= batch_rows) break;
                const float in_val = input[(unsigned long long)b * in_dim + i];
                gacc[b] = fmaf(in_val, gw, gacc[b]);
                uacc[b] = fmaf(in_val, uw, uacc[b]);
            }
        }
    } else {
        for (unsigned int i = lane; i < in_dim; i += TALU_WARP_SIZE) {
            const float gw = talu_decode_f16_u16(gate_row[i]);
            const float uw = talu_decode_f16_u16(up_row[i]);
            #pragma unroll
            for (unsigned int b = 0; b < BATCH; b++) {
                if (b >= batch_rows) break;
                const float in_val = input[(unsigned long long)b * in_dim + i];
                gacc[b] = fmaf(in_val, gw, gacc[b]);
                uacc[b] = fmaf(in_val, uw, uacc[b]);
            }
        }
    }

    #pragma unroll
    for (unsigned int b = 0; b < BATCH; b++) {
        if (b >= batch_rows) break;
        float gate_val = talu_warp_sum_f32(gacc[b]);
        float up_val = talu_warp_sum_f32(uacc[b]);
        if (lane == 0) {
            const float sigma = 1.0f / (1.0f + expf(-gate_val));
            out[(unsigned long long)b * out_dim + out_idx] = gate_val * sigma * up_val;
        }
    }
}

extern "C" __global__ void talu_matvec_gate_up_silu_f16_f32(
    const float* input, const unsigned short* gate_weight, const unsigned short* up_weight,
    float* out, unsigned int out_dim, unsigned int in_dim, unsigned int batch_rows
) {
    const unsigned int row_idx = blockIdx.y;
    if (row_idx >= batch_rows) return;
    const float* input_row = input + (unsigned long long)row_idx * in_dim;
    float* out_row = out + (unsigned long long)row_idx * out_dim;
    talu_gate_up_silu_f16_inner<1>(input_row, gate_weight, up_weight, out_row, out_dim, in_dim, 1);
}

extern "C" __global__
void talu_matvec_gate_up_silu_f16_f32_batch(
    const float* input, const unsigned short* gate_weight, const unsigned short* up_weight,
    float* out, unsigned int out_dim, unsigned int in_dim, unsigned int batch_rows
) {
    const unsigned int batch_base = blockIdx.y * TALU_BATCH_TILE;
    if (batch_base >= batch_rows) return;
    const unsigned int rows_remaining = batch_rows - batch_base;
    const unsigned int tile_rows = rows_remaining < TALU_BATCH_TILE ? rows_remaining : TALU_BATCH_TILE;

    const float* input_tile = input + (unsigned long long)batch_base * in_dim;
    float* out_tile = out + (unsigned long long)batch_base * out_dim;
    switch (tile_rows) {
        case 1u: talu_gate_up_silu_f16_inner<1>(input_tile, gate_weight, up_weight, out_tile, out_dim, in_dim, 1u); break;
        case 2u: talu_gate_up_silu_f16_inner<2>(input_tile, gate_weight, up_weight, out_tile, out_dim, in_dim, 2u); break;
        case 3u: talu_gate_up_silu_f16_inner<3>(input_tile, gate_weight, up_weight, out_tile, out_dim, in_dim, 3u); break;
        case 4u: talu_gate_up_silu_f16_inner<4>(input_tile, gate_weight, up_weight, out_tile, out_dim, in_dim, 4u); break;
        case 5u: talu_gate_up_silu_f16_inner<5>(input_tile, gate_weight, up_weight, out_tile, out_dim, in_dim, 5u); break;
        case 6u: talu_gate_up_silu_f16_inner<6>(input_tile, gate_weight, up_weight, out_tile, out_dim, in_dim, 6u); break;
        case 7u: talu_gate_up_silu_f16_inner<7>(input_tile, gate_weight, up_weight, out_tile, out_dim, in_dim, 7u); break;
        case 8u: talu_gate_up_silu_f16_inner<8>(input_tile, gate_weight, up_weight, out_tile, out_dim, in_dim, 8u); break;
        default: return;
    }
}

// Inner-batch gate-up-SiLU (BF16): gate+up weights loaded once from DRAM,
// reused across batch rows from registers.
template <unsigned int BATCH>
static __device__ void talu_gate_up_silu_bf16_inner(
    const float* input,
    const unsigned short* gate_weight,
    const unsigned short* up_weight,
    float* out,
    unsigned int out_dim,
    unsigned int in_dim,
    unsigned int batch_rows
) {
    const unsigned int lane = threadIdx.x & (TALU_WARP_SIZE - 1u);
    const unsigned int warp_id = threadIdx.x / TALU_WARP_SIZE;
    const unsigned int warps_per_block = blockDim.x / TALU_WARP_SIZE;
    const unsigned int out_idx = blockIdx.x * warps_per_block + warp_id;
    if (out_idx >= out_dim) return;

    const unsigned short* gate_row = gate_weight + (unsigned long long)out_idx * in_dim;
    const unsigned short* up_row = up_weight + (unsigned long long)out_idx * in_dim;

    float gacc[BATCH] = {}, uacc[BATCH] = {};

    const bool can_vec8 = ((((unsigned long long)gate_row) & 0xFu) == 0u) and
        ((((unsigned long long)up_row) & 0xFu) == 0u) and
        ((((unsigned long long)input) & 0xFu) == 0u);
    if (can_vec8) {
        const unsigned int vec_elems = in_dim & ~7u;
        const uint4* gate_vec = reinterpret_cast<const uint4*>(gate_row);
        const uint4* up_vec = reinterpret_cast<const uint4*>(up_row);
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
            const float gw0 = __uint_as_float(g_packed.x << 16);
            const float gw1 = __uint_as_float(g_packed.x & 0xFFFF0000u);
            const float gw2 = __uint_as_float(g_packed.y << 16);
            const float gw3 = __uint_as_float(g_packed.y & 0xFFFF0000u);
            const float gw4 = __uint_as_float(g_packed.z << 16);
            const float gw5 = __uint_as_float(g_packed.z & 0xFFFF0000u);
            const float gw6 = __uint_as_float(g_packed.w << 16);
            const float gw7 = __uint_as_float(g_packed.w & 0xFFFF0000u);
            const float uw0 = __uint_as_float(u_packed.x << 16);
            const float uw1 = __uint_as_float(u_packed.x & 0xFFFF0000u);
            const float uw2 = __uint_as_float(u_packed.y << 16);
            const float uw3 = __uint_as_float(u_packed.y & 0xFFFF0000u);
            const float uw4 = __uint_as_float(u_packed.z << 16);
            const float uw5 = __uint_as_float(u_packed.z & 0xFFFF0000u);
            const float uw6 = __uint_as_float(u_packed.w << 16);
            const float uw7 = __uint_as_float(u_packed.w & 0xFFFF0000u);
            const unsigned int input_vec_idx = i >> 2;
            #pragma unroll
            for (unsigned int b = 0; b < BATCH; b++) {
                if (b >= batch_rows) break;
                const float4* input_vec = reinterpret_cast<const float4*>(input + (unsigned long long)b * in_dim);
                const float4 in0 = input_vec[input_vec_idx];
                const float4 in1 = input_vec[input_vec_idx + 1];
                gacc[b] = fmaf(in0.x, gw0, gacc[b]);  gacc[b] = fmaf(in0.y, gw1, gacc[b]);
                gacc[b] = fmaf(in0.z, gw2, gacc[b]);  gacc[b] = fmaf(in0.w, gw3, gacc[b]);
                gacc[b] = fmaf(in1.x, gw4, gacc[b]);  gacc[b] = fmaf(in1.y, gw5, gacc[b]);
                gacc[b] = fmaf(in1.z, gw6, gacc[b]);  gacc[b] = fmaf(in1.w, gw7, gacc[b]);
                uacc[b] = fmaf(in0.x, uw0, uacc[b]);  uacc[b] = fmaf(in0.y, uw1, uacc[b]);
                uacc[b] = fmaf(in0.z, uw2, uacc[b]);  uacc[b] = fmaf(in0.w, uw3, uacc[b]);
                uacc[b] = fmaf(in1.x, uw4, uacc[b]);  uacc[b] = fmaf(in1.y, uw5, uacc[b]);
                uacc[b] = fmaf(in1.z, uw6, uacc[b]);  uacc[b] = fmaf(in1.w, uw7, uacc[b]);
            }
        }
        for (unsigned int i = vec_elems + lane; i < in_dim; i += TALU_WARP_SIZE) {
            const float gw = talu_decode_bf16_u16(gate_row[i]);
            const float uw = talu_decode_bf16_u16(up_row[i]);
            #pragma unroll
            for (unsigned int b = 0; b < BATCH; b++) {
                if (b >= batch_rows) break;
                const float in_val = input[(unsigned long long)b * in_dim + i];
                gacc[b] = fmaf(in_val, gw, gacc[b]);
                uacc[b] = fmaf(in_val, uw, uacc[b]);
            }
        }
    } else {
        for (unsigned int i = lane; i < in_dim; i += TALU_WARP_SIZE) {
            const float gw = talu_decode_bf16_u16(gate_row[i]);
            const float uw = talu_decode_bf16_u16(up_row[i]);
            #pragma unroll
            for (unsigned int b = 0; b < BATCH; b++) {
                if (b >= batch_rows) break;
                const float in_val = input[(unsigned long long)b * in_dim + i];
                gacc[b] = fmaf(in_val, gw, gacc[b]);
                uacc[b] = fmaf(in_val, uw, uacc[b]);
            }
        }
    }

    #pragma unroll
    for (unsigned int b = 0; b < BATCH; b++) {
        if (b >= batch_rows) break;
        float gate_val = talu_warp_sum_f32(gacc[b]);
        float up_val = talu_warp_sum_f32(uacc[b]);
        if (lane == 0) {
            const float sigma = 1.0f / (1.0f + expf(-gate_val));
            out[(unsigned long long)b * out_dim + out_idx] = gate_val * sigma * up_val;
        }
    }
}

extern "C" __global__ void talu_matvec_gate_up_silu_bf16_f32(
    const float* input, const unsigned short* gate_weight, const unsigned short* up_weight,
    float* out, unsigned int out_dim, unsigned int in_dim, unsigned int batch_rows
) {
    const unsigned int row_idx = blockIdx.y;
    if (row_idx >= batch_rows) return;
    const float* input_row = input + (unsigned long long)row_idx * in_dim;
    float* out_row = out + (unsigned long long)row_idx * out_dim;
    talu_gate_up_silu_bf16_inner<1>(input_row, gate_weight, up_weight, out_row, out_dim, in_dim, 1);
}

extern "C" __global__
void talu_matvec_gate_up_silu_bf16_f32_batch(
    const float* input, const unsigned short* gate_weight, const unsigned short* up_weight,
    float* out, unsigned int out_dim, unsigned int in_dim, unsigned int batch_rows
) {
    const unsigned int batch_base = blockIdx.y * TALU_BATCH_TILE;
    if (batch_base >= batch_rows) return;
    const unsigned int rows_remaining = batch_rows - batch_base;
    const unsigned int tile_rows = rows_remaining < TALU_BATCH_TILE ? rows_remaining : TALU_BATCH_TILE;

    const float* input_tile = input + (unsigned long long)batch_base * in_dim;
    float* out_tile = out + (unsigned long long)batch_base * out_dim;
    switch (tile_rows) {
        case 1u: talu_gate_up_silu_bf16_inner<1>(input_tile, gate_weight, up_weight, out_tile, out_dim, in_dim, 1u); break;
        case 2u: talu_gate_up_silu_bf16_inner<2>(input_tile, gate_weight, up_weight, out_tile, out_dim, in_dim, 2u); break;
        case 3u: talu_gate_up_silu_bf16_inner<3>(input_tile, gate_weight, up_weight, out_tile, out_dim, in_dim, 3u); break;
        case 4u: talu_gate_up_silu_bf16_inner<4>(input_tile, gate_weight, up_weight, out_tile, out_dim, in_dim, 4u); break;
        case 5u: talu_gate_up_silu_bf16_inner<5>(input_tile, gate_weight, up_weight, out_tile, out_dim, in_dim, 5u); break;
        case 6u: talu_gate_up_silu_bf16_inner<6>(input_tile, gate_weight, up_weight, out_tile, out_dim, in_dim, 6u); break;
        case 7u: talu_gate_up_silu_bf16_inner<7>(input_tile, gate_weight, up_weight, out_tile, out_dim, in_dim, 7u); break;
        case 8u: talu_gate_up_silu_bf16_inner<8>(input_tile, gate_weight, up_weight, out_tile, out_dim, in_dim, 8u); break;
        default: return;
    }
}
