// NVFP4 (FP4 E2M1 + FP8 E4M3 per-group scales) quantized GEMV and dequantization kernels.

// ---------------------------------------------------------------------------
// NVFP4 activation quantization: F32 → FP4(E2M1) + UE4M3 block-16 scales
// ---------------------------------------------------------------------------

// Compute interleaved offset for cuBLASLt VEC16_UE4M3 scale tensor.
// Layout matches cuBLASLt tiled tensor format: 128×4 scale tiles (512 bytes).
__device__ __forceinline__ unsigned int nvfp4_scale_offset(
    unsigned int m,             // outer index (row for activations, col for weights)
    unsigned int k,             // inner scale index (0..sf_k-1), block size = 16
    unsigned int n_col_tiles    // padded_sf_k / 4
) {
    return (m / 128) * n_col_tiles * 512 +
        (k / 4) * 512 +
        (m % 32) * 16 +
        ((m % 128) / 32) * 4 +
        (k % 4);
}

#if __CUDA_ARCH__ >= 1000

// NVFP4 activation quantization: F32 → packed FP4 E2M1 + interleaved UE4M3 scales.
// Each block handles one 16-element group for one row.
// Launch: grid=(sf_k, padded_outer), block=(32). Only lanes 0-15 are active.
extern "C" __global__ void talu_quantize_f32_to_nvfp4(
    const float* __restrict__ input,        // [rows × in_dim]
    unsigned char* __restrict__ out_fp4,    // [rows × ceil(in_dim/2)] packed FP4 E2M1
    unsigned char* __restrict__ out_scales, // [padded_outer × padded_sf_k] interleaved UE4M3
    unsigned int in_dim,
    unsigned int rows,
    unsigned int padded_outer,              // = roundoff(rows, 128)
    unsigned int padded_sf_k                // = roundoff(ceil(in_dim/16), 4)
) {
    const unsigned int row = blockIdx.y;
    const unsigned int group = blockIdx.x;
    const unsigned int lane = threadIdx.x;

    if (row >= padded_outer) return;

    if (row >= rows) {
        if (lane == 0) {
            const unsigned int n_col_tiles = padded_sf_k / 4;
            out_scales[nvfp4_scale_offset(row, group, n_col_tiles)] = 0;
        }
        return;
    }

    if (lane >= 16) return;

    const unsigned int col = group * 16 + lane;
    if (col >= in_dim) return;

    const unsigned long long input_offset = (unsigned long long)row * in_dim + col;
    const float val = input[input_offset];

    float absmax = fabsf(val);
    for (int s = 8; s >= 1; s >>= 1) {
        absmax = fmaxf(absmax, __shfl_down_sync(0xFFFFu, absmax, s, 16));
    }
    const float block_absmax = __shfl_sync(0xFFFFu, absmax, 0, 16);

    const float scale = (block_absmax > 0.0f) ? (block_absmax / 6.0f) : 1.0f;
    const unsigned char scale_e4m3 = __nv_cvt_float_to_fp8(scale, __NV_SATFINITE, __NV_E4M3);

    if (lane == 0) {
        const unsigned int n_col_tiles = padded_sf_k / 4;
        out_scales[nvfp4_scale_offset(row, group, n_col_tiles)] = scale_e4m3;
    }

    const float scaled = val / scale;
    const unsigned char fp4_nibble = static_cast<unsigned char>(
        __nv_cvt_float_to_fp4(scaled, __NV_E2M1, cudaRoundNearest)
    ) & 0x0Fu;
    // All 16 lanes must participate in the shuffle before the even-lane write.
    const unsigned char next_nibble = __shfl_down_sync(0xFFFFu, fp4_nibble, 1u, 16) & 0x0Fu;
    if ((lane & 1u) == 0u) {
        const unsigned int packed_cols = (in_dim + 1u) >> 1;
        const unsigned int pair = lane >> 1;
        const unsigned long long out_offset = (unsigned long long)row * packed_cols + (unsigned long long)group * 8u + pair;
        out_fp4[out_offset] = static_cast<unsigned char>(fp4_nibble | (next_nibble << 4));
    }
}

#else

extern "C" __global__ void talu_quantize_f32_to_nvfp4(
    const float* __restrict__ input,
    void* __restrict__ out_fp4,
    void* __restrict__ out_scales,
    unsigned int in_dim,
    unsigned int rows,
    unsigned int padded_outer,
    unsigned int padded_sf_k
) {}

#endif // __CUDA_ARCH__ >= 1000

#if __CUDA_ARCH__ >= 890

static __device__ __forceinline__ void nvfp4_lut_init(float* lut, unsigned int tid) {
    if (tid < 16u) {
        const unsigned int s = tid >> 3;
        const unsigned int e = (tid >> 1) & 3u;
        const unsigned int m = tid & 1u;
        const unsigned int is_nz = min(e | m, 1u);
        const unsigned int e_nz = min(e, 1u);
        lut[tid] = __uint_as_float(
            (s << 31) | (((e + 126u) * is_nz) << 23) | (e_nz * (m << 22)));
    }
}

// Process one NVFP4 word (4 bytes = 8 nibbles) against 2 float4 inputs.
// Uses shared memory LUT for branchless FP4 decode.
#define NVFP4_PROCESS_WORD(word, inp0, inp1, acc, lut) do { \
    (acc) = fmaf((inp0).x, (lut)[(word) & 0xFu], (acc)); \
    (acc) = fmaf((inp0).y, (lut)[((word) >> 4) & 0xFu], (acc)); \
    (acc) = fmaf((inp0).z, (lut)[((word) >> 8) & 0xFu], (acc)); \
    (acc) = fmaf((inp0).w, (lut)[((word) >> 12) & 0xFu], (acc)); \
    (acc) = fmaf((inp1).x, (lut)[((word) >> 16) & 0xFu], (acc)); \
    (acc) = fmaf((inp1).y, (lut)[((word) >> 20) & 0xFu], (acc)); \
    (acc) = fmaf((inp1).z, (lut)[((word) >> 24) & 0xFu], (acc)); \
    (acc) = fmaf((inp1).w, (lut)[(word) >> 28], (acc)); \
} while (0)

// ---------------------------------------------------------------------------
// NVFP4 GEMV (packed FP4 E2M1 + FP8 E4M3 per-group scales)
// ---------------------------------------------------------------------------

static __device__ __forceinline__ float nvfp4_nibble_to_f32(unsigned int nibble) {
    // Branchless FP4 E2M1 → IEEE 754 f32 decode via bit construction.
    // nibble layout: bit3=sign, bits1-2=exponent(bias=1), bit0=mantissa
    //
    // Normal (e>0): (-1)^s × 2^(e-1) × (1 + m×0.5)
    //   → f32: s<<31 | (e+126)<<23 | m<<22
    //
    // Subnormal (e=0, m=1): ±0.5 → f32: s<<31 | 126<<23
    // Zero (e=0, m=0): ±0.0 → f32: s<<31
    const unsigned int s = nibble >> 3;
    const unsigned int e = (nibble >> 1) & 3u;
    const unsigned int m = nibble & 1u;
    const unsigned int is_nz = min(e | m, 1u);   // 0 when zero, 1 otherwise
    const unsigned int e_nz = min(e, 1u);         // 0 when subnormal, 1 when normal
    const unsigned int f32_exp = (e + 126u) * is_nz;
    const unsigned int f32_mant = e_nz * (m << 22);
    return __uint_as_float((s << 31) | (f32_exp << 23) | f32_mant);
}

// ---------------------------------------------------------------------------
// NVFP4 → I8 weight dequant cache for INT8 tensor core prefill
// ---------------------------------------------------------------------------

// Dequantize NVFP4 weights to I8 with per-row F32 scales for INT8 tensor core GEMM.
// Two-pass: (1) dequant to F32 + find per-row absmax, (2) quantize to I8.
// Launch: grid=(out_dim), block=(256)
extern "C" __global__ void talu_nvfp4_to_i8(
    const unsigned char* __restrict__ weight_packed,  // [out_dim × packed_cols]
    const unsigned char* __restrict__ scales,         // [out_dim × scale_cols] FP8 E4M3
    signed char* __restrict__ out_i8,                 // [out_dim × in_dim]
    float* __restrict__ out_row_scales,               // [out_dim]
    unsigned int in_dim,
    unsigned int scale_cols,
    float weight_global_scale
) {
    const unsigned int row = blockIdx.x;
    const unsigned int packed_cols = (in_dim + 1u) >> 1;
    const unsigned char* row_packed = weight_packed + (unsigned long long)row * packed_cols;
    const unsigned char* scale_row = scales + (unsigned long long)row * scale_cols;
    signed char* row_out = out_i8 + (unsigned long long)row * in_dim;

    // Pass 1: dequant to F32 in registers and find per-row absmax.
    __shared__ float smem[8];
    float local_max = 0.0f;
    for (unsigned int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        const unsigned char packed_byte = row_packed[i >> 1];
        const unsigned int nibble = (i & 1u) ? (packed_byte >> 4) : (packed_byte & 0x0Fu);
        const float fp4_val = nvfp4_nibble_to_f32(nibble);
        const float block_scale = fp8e4m3_to_f32(
            static_cast<__nv_fp8_storage_t>(scale_row[i >> 4]));
        const float val = fp4_val * block_scale / weight_global_scale;
        local_max = fmaxf(local_max, fabsf(val));
    }
    for (int offset = 16; offset >= 1; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFFu, local_max, offset));
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = local_max;
    __syncthreads();
    if (threadIdx.x < 8) {
        local_max = smem[threadIdx.x];
        for (int offset = 4; offset >= 1; offset >>= 1)
            local_max = fmaxf(local_max, __shfl_down_sync(0xFFu, local_max, offset));
    }
    __syncthreads();
    if (threadIdx.x == 0) smem[0] = local_max;
    __syncthreads();
    const float absmax = smem[0];
    const float scale = (absmax > 0.0f) ? (absmax / 127.0f) : 1.0f;
    const float inv_scale = 1.0f / scale;
    if (threadIdx.x == 0) out_row_scales[row] = scale;

    // Pass 2: dequant + quantize to I8.
    for (unsigned int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        const unsigned char packed_byte = row_packed[i >> 1];
        const unsigned int nibble = (i & 1u) ? (packed_byte >> 4) : (packed_byte & 0x0Fu);
        const float fp4_val = nvfp4_nibble_to_f32(nibble);
        const float block_scale = fp8e4m3_to_f32(
            static_cast<__nv_fp8_storage_t>(scale_row[i >> 4]));
        const float val = fp4_val * block_scale / weight_global_scale;
        int q = __float2int_rn(val * inv_scale);
        q = max(-128, min(127, q));
        row_out[i] = static_cast<signed char>(q);
    }
}

// ---------------------------------------------------------------------------
// NVFP4 dequantization: packed FP4 E2M1 + FP8 E4M3 per-group scales → BF16
// ---------------------------------------------------------------------------

// Dequantize NVFP4 weights to BF16 for cuBLAS GEMM prefill path.
// Each thread processes one element: unpack nibble, fp4→f32, scale, f32→bf16.
// Launch: grid=(ceil(in_dim/256), out_dim), block=(256)
extern "C" __global__ void talu_dequant_nvfp4_to_bf16(
    const unsigned char* __restrict__ weight_packed,  // [out_dim × packed_cols] (2 FP4 per byte)
    const unsigned char* __restrict__ scales,         // [out_dim × scale_cols] FP8 E4M3
    unsigned short* __restrict__ out_bf16,            // [out_dim × in_dim] BF16
    unsigned int in_dim,
    unsigned int out_dim,
    unsigned int scale_cols,
    float weight_global_scale
) {
    const unsigned int row = blockIdx.y;
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim || col >= in_dim) return;

    const unsigned int packed_cols = (in_dim + 1u) >> 1;
    const unsigned long long byte_idx = (unsigned long long)row * packed_cols + (col >> 1);
    const unsigned char packed_byte = weight_packed[byte_idx];
    const unsigned int nibble = (col & 1u) ? (packed_byte >> 4) : (packed_byte & 0x0Fu);

    const float fp4_val = nvfp4_nibble_to_f32(nibble);
    const float scale = fp8e4m3_to_f32(
        static_cast<__nv_fp8_storage_t>(scales[(unsigned long long)row * scale_cols + (col >> 4)]));
    const float val = fp4_val * scale / weight_global_scale;

    // F32 → BF16: round-to-nearest-even (same as FP8 dequant kernel).
    unsigned int fbits;
    memcpy(&fbits, &val, sizeof(fbits));
    const unsigned int lsb = (fbits >> 16) & 1u;
    const unsigned int rounding_bias = 0x7FFFu + lsb;
    const unsigned long long out_idx = (unsigned long long)row * in_dim + col;
    out_bf16[out_idx] = static_cast<unsigned short>((fbits + rounding_bias) >> 16);
}

// Inner-batch NVFP4 GEMV: vectorized 64-bit weight loads + float4 input loads.
// Each iteration processes 2 u32 words = 16 FP4 elements = 1 scale group (group_size=16).
// Scale is applied once per group via post-multiply on the accumulated partial.
// Requires: group_size == 16, in_dim % 16 == 0.
template <unsigned int BATCH>
static __device__ __forceinline__ void talu_nvfp4_matvec_batched(
    const float* input,                    // [batch_rows × in_dim]
    const unsigned char* weight_packed,    // [out_dim × packed_in]
    const unsigned char* scales,           // [out_dim × scale_cols] FP8 E4M3
    float* out,                            // [batch_rows × out_dim]
    unsigned int in_dim,
    unsigned int out_dim,
    unsigned int out_idx,
    unsigned int scale_cols,
    float inv_global,
    const float* fp4_lut,
    unsigned int lane,
    unsigned int batch_rows
) {
    // scale_cols = in_dim / 16 = number of word-pairs per row.
    const unsigned int scale_row_off = out_idx * scale_cols;
    const unsigned int* weight_words = reinterpret_cast<const unsigned int*>(weight_packed);

    float acc[BATCH] = {};

    for (unsigned int wp = lane; wp < scale_cols; wp += TALU_QUANT_WARP_SIZE) {
        // 64-bit streaming load: 2 u32 words = 8 bytes = 16 FP4 elements.
        unsigned int w0, w1;
        asm volatile(
            "ld.global.cs.v2.u32 {%0, %1}, [%2];"
            : "=r"(w0), "=r"(w1)
            : "l"(&weight_words[wp << 1]));

        const unsigned int base_elem = wp << 4;  // 16 elements per word-pair

        // 1 FP8 E4M3 scale per 16-element group.
        const float scale = fp8e4m3_to_f32(
            static_cast<__nv_fp8_storage_t>(scales[scale_row_off + wp])) * inv_global;

        #pragma unroll
        for (unsigned int b = 0; b < BATCH; b++) {
            if (b >= batch_rows) break;
            const unsigned long long row_off = (unsigned long long)b * in_dim;

            float partial = 0.0f;

            // Phase 1: w0 (elements 0-7)
            {
                const float4 i0 = *reinterpret_cast<const float4*>(&input[row_off + base_elem]);
                const float4 i1 = *reinterpret_cast<const float4*>(&input[row_off + base_elem + 4]);
                NVFP4_PROCESS_WORD(w0, i0, i1, partial, fp4_lut);
            }

            // Phase 2: w1 (elements 8-15)
            {
                const float4 i2 = *reinterpret_cast<const float4*>(&input[row_off + base_elem + 8]);
                const float4 i3 = *reinterpret_cast<const float4*>(&input[row_off + base_elem + 12]);
                NVFP4_PROCESS_WORD(w1, i2, i3, partial, fp4_lut);
            }

            acc[b] = fmaf(partial, scale, acc[b]);
        }
    }

    #pragma unroll
    for (unsigned int b = 0; b < BATCH; b++) {
        if (b >= batch_rows) break;
        const float result = talu_quant_warp_sum_f32(acc[b]);
        if (lane == 0) out[(unsigned long long)b * out_dim + out_idx] = result;
    }
}

extern "C" __global__ __launch_bounds__(128, 3) void talu_nvfp4_matvec_f32(
    const float* __restrict__ input,
    const unsigned char* __restrict__ weight_packed,
    const unsigned char* __restrict__ scales,
    float* __restrict__ out,
    unsigned int in_dim,
    unsigned int out_dim,
    unsigned int scale_cols,
    unsigned int group_size,
    float weight_global_scale,
    unsigned int batch_rows
) {
    const unsigned int batch_base = blockIdx.y * FP8_BATCH_TILE;
    if (batch_base >= batch_rows) return;
    const unsigned int tile_rows = batch_rows - batch_base < FP8_BATCH_TILE
        ? batch_rows - batch_base : FP8_BATCH_TILE;
    if (group_size != 16u || (in_dim & 15u) != 0u || weight_global_scale == 0.0f) return;

    __shared__ float fp4_lut[16];
    nvfp4_lut_init(fp4_lut, threadIdx.x);
    __syncthreads();

    const unsigned int warp_id = threadIdx.x / TALU_QUANT_WARP_SIZE;
    const unsigned int lane = threadIdx.x & (TALU_QUANT_WARP_SIZE - 1u);
    const unsigned int out_idx = blockIdx.x * FP8_WARPS_PER_BLOCK + warp_id;
    if (out_idx >= out_dim) return;

    const unsigned int packed_cols = (in_dim + 1u) >> 1;
    const float inv_global = 1.0f / weight_global_scale;
    const float* input_tile = input + (unsigned long long)batch_base * in_dim;
    float* out_tile = out + (unsigned long long)batch_base * out_dim;
    const unsigned char* row_weight = weight_packed + (unsigned long long)out_idx * packed_cols;

    switch (tile_rows) {
        case 1u: talu_nvfp4_matvec_batched<1>(input_tile, row_weight, scales, out_tile, in_dim, out_dim, out_idx, scale_cols, inv_global, fp4_lut, lane, 1u); break;
        case 2u: talu_nvfp4_matvec_batched<2>(input_tile, row_weight, scales, out_tile, in_dim, out_dim, out_idx, scale_cols, inv_global, fp4_lut, lane, 2u); break;
        case 3u: talu_nvfp4_matvec_batched<3>(input_tile, row_weight, scales, out_tile, in_dim, out_dim, out_idx, scale_cols, inv_global, fp4_lut, lane, 3u); break;
        case 4u: talu_nvfp4_matvec_batched<4>(input_tile, row_weight, scales, out_tile, in_dim, out_dim, out_idx, scale_cols, inv_global, fp4_lut, lane, 4u); break;
        default: return;
    }
}

extern "C" __global__ __launch_bounds__(128, 2) void talu_nvfp4_matvec_f32_tile8(
    const float* __restrict__ input,
    const unsigned char* __restrict__ weight_packed,
    const unsigned char* __restrict__ scales,
    float* __restrict__ out,
    unsigned int in_dim,
    unsigned int out_dim,
    unsigned int scale_cols,
    unsigned int group_size,
    float weight_global_scale,
    unsigned int batch_rows
) {
    const unsigned int batch_base = blockIdx.y * FP8_BATCH_TILE_X8;
    if (batch_base >= batch_rows) return;
    const unsigned int tile_rows = batch_rows - batch_base < FP8_BATCH_TILE_X8
        ? batch_rows - batch_base : FP8_BATCH_TILE_X8;
    if (group_size != 16u || (in_dim & 15u) != 0u || weight_global_scale == 0.0f) return;

    __shared__ float fp4_lut[16];
    nvfp4_lut_init(fp4_lut, threadIdx.x);
    __syncthreads();

    const unsigned int warp_id = threadIdx.x / TALU_QUANT_WARP_SIZE;
    const unsigned int lane = threadIdx.x & (TALU_QUANT_WARP_SIZE - 1u);
    const unsigned int out_idx = blockIdx.x * FP8_WARPS_PER_BLOCK + warp_id;
    if (out_idx >= out_dim) return;

    const unsigned int packed_cols = (in_dim + 1u) >> 1;
    const float inv_global = 1.0f / weight_global_scale;
    const float* input_tile = input + (unsigned long long)batch_base * in_dim;
    float* out_tile = out + (unsigned long long)batch_base * out_dim;
    const unsigned char* row_weight = weight_packed + (unsigned long long)out_idx * packed_cols;

    switch (tile_rows) {
        case 1u: talu_nvfp4_matvec_batched<1>(input_tile, row_weight, scales, out_tile, in_dim, out_dim, out_idx, scale_cols, inv_global, fp4_lut, lane, 1u); break;
        case 2u: talu_nvfp4_matvec_batched<2>(input_tile, row_weight, scales, out_tile, in_dim, out_dim, out_idx, scale_cols, inv_global, fp4_lut, lane, 2u); break;
        case 3u: talu_nvfp4_matvec_batched<3>(input_tile, row_weight, scales, out_tile, in_dim, out_dim, out_idx, scale_cols, inv_global, fp4_lut, lane, 3u); break;
        case 4u: talu_nvfp4_matvec_batched<4>(input_tile, row_weight, scales, out_tile, in_dim, out_dim, out_idx, scale_cols, inv_global, fp4_lut, lane, 4u); break;
        case 5u: talu_nvfp4_matvec_batched<5>(input_tile, row_weight, scales, out_tile, in_dim, out_dim, out_idx, scale_cols, inv_global, fp4_lut, lane, 5u); break;
        case 6u: talu_nvfp4_matvec_batched<6>(input_tile, row_weight, scales, out_tile, in_dim, out_dim, out_idx, scale_cols, inv_global, fp4_lut, lane, 6u); break;
        case 7u: talu_nvfp4_matvec_batched<7>(input_tile, row_weight, scales, out_tile, in_dim, out_dim, out_idx, scale_cols, inv_global, fp4_lut, lane, 7u); break;
        case 8u: talu_nvfp4_matvec_batched<8>(input_tile, row_weight, scales, out_tile, in_dim, out_dim, out_idx, scale_cols, inv_global, fp4_lut, lane, 8u); break;
        default: return;
    }
}

extern "C" __global__ __launch_bounds__(128, 3) void talu_nvfp4_matvec_qkv_f32(
    const float* __restrict__ input,
    const unsigned char* __restrict__ q_weight_packed,
    const unsigned char* __restrict__ q_scales,
    float* __restrict__ q_out,
    unsigned int q_out_dim,
    unsigned int q_scale_cols,
    unsigned int q_group_size,
    float q_weight_global_scale,
    const unsigned char* __restrict__ k_weight_packed,
    const unsigned char* __restrict__ k_scales,
    float* __restrict__ k_out,
    unsigned int k_out_dim,
    unsigned int k_scale_cols,
    unsigned int k_group_size,
    float k_weight_global_scale,
    const unsigned char* __restrict__ v_weight_packed,
    const unsigned char* __restrict__ v_scales,
    float* __restrict__ v_out,
    unsigned int v_out_dim,
    unsigned int v_scale_cols,
    unsigned int v_group_size,
    float v_weight_global_scale,
    unsigned int in_dim,
    unsigned int batch_rows
) {
    const unsigned int batch_base = blockIdx.y * FP8_BATCH_TILE;
    if (batch_base >= batch_rows) return;
    const unsigned int tile_rows = batch_rows - batch_base < FP8_BATCH_TILE
        ? batch_rows - batch_base : FP8_BATCH_TILE;

    __shared__ float fp4_lut[16];
    nvfp4_lut_init(fp4_lut, threadIdx.x);
    __syncthreads();

    const unsigned int warp_id = threadIdx.x / TALU_QUANT_WARP_SIZE;
    const unsigned int lane = threadIdx.x & (TALU_QUANT_WARP_SIZE - 1u);
    const unsigned int out_index = blockIdx.x * FP8_WARPS_PER_BLOCK + warp_id;
    const unsigned int qk_dim = q_out_dim + k_out_dim;
    const unsigned int total_dim = qk_dim + v_out_dim;
    if (out_index >= total_dim) return;

    const unsigned char* wt;
    const unsigned char* sc;
    float* out_ptr;
    unsigned int out_dim_proj;
    unsigned int row_idx;
    unsigned int scale_cols;
    unsigned int group_size;
    float weight_global_scale;
    if (out_index < q_out_dim) {
        wt = q_weight_packed;
        sc = q_scales;
        out_ptr = q_out;
        out_dim_proj = q_out_dim;
        row_idx = out_index;
        scale_cols = q_scale_cols;
        group_size = q_group_size;
        weight_global_scale = q_weight_global_scale;
    } else if (out_index < qk_dim) {
        wt = k_weight_packed;
        sc = k_scales;
        out_ptr = k_out;
        out_dim_proj = k_out_dim;
        row_idx = out_index - q_out_dim;
        scale_cols = k_scale_cols;
        group_size = k_group_size;
        weight_global_scale = k_weight_global_scale;
    } else {
        wt = v_weight_packed;
        sc = v_scales;
        out_ptr = v_out;
        out_dim_proj = v_out_dim;
        row_idx = out_index - qk_dim;
        scale_cols = v_scale_cols;
        group_size = v_group_size;
        weight_global_scale = v_weight_global_scale;
    }
    if (group_size != 16u || (in_dim & 15u) != 0u || weight_global_scale == 0.0f) return;

    const unsigned int packed_cols = (in_dim + 1u) >> 1;
    const float inv_global = 1.0f / weight_global_scale;
    const float* input_tile = input + (unsigned long long)batch_base * in_dim;
    float* out_tile = out_ptr + (unsigned long long)batch_base * out_dim_proj;
    const unsigned char* row_weight = wt + (unsigned long long)row_idx * packed_cols;

    switch (tile_rows) {
        case 1u: talu_nvfp4_matvec_batched<1>(input_tile, row_weight, sc, out_tile, in_dim, out_dim_proj, row_idx, scale_cols, inv_global, fp4_lut, lane, 1u); break;
        case 2u: talu_nvfp4_matvec_batched<2>(input_tile, row_weight, sc, out_tile, in_dim, out_dim_proj, row_idx, scale_cols, inv_global, fp4_lut, lane, 2u); break;
        case 3u: talu_nvfp4_matvec_batched<3>(input_tile, row_weight, sc, out_tile, in_dim, out_dim_proj, row_idx, scale_cols, inv_global, fp4_lut, lane, 3u); break;
        case 4u: talu_nvfp4_matvec_batched<4>(input_tile, row_weight, sc, out_tile, in_dim, out_dim_proj, row_idx, scale_cols, inv_global, fp4_lut, lane, 4u); break;
        default: return;
    }
}

extern "C" __global__ __launch_bounds__(128, 2) void talu_nvfp4_matvec_qkv_f32_tile8(
    const float* __restrict__ input,
    const unsigned char* __restrict__ q_weight_packed,
    const unsigned char* __restrict__ q_scales,
    float* __restrict__ q_out,
    unsigned int q_out_dim,
    unsigned int q_scale_cols,
    unsigned int q_group_size,
    float q_weight_global_scale,
    const unsigned char* __restrict__ k_weight_packed,
    const unsigned char* __restrict__ k_scales,
    float* __restrict__ k_out,
    unsigned int k_out_dim,
    unsigned int k_scale_cols,
    unsigned int k_group_size,
    float k_weight_global_scale,
    const unsigned char* __restrict__ v_weight_packed,
    const unsigned char* __restrict__ v_scales,
    float* __restrict__ v_out,
    unsigned int v_out_dim,
    unsigned int v_scale_cols,
    unsigned int v_group_size,
    float v_weight_global_scale,
    unsigned int in_dim,
    unsigned int batch_rows
) {
    const unsigned int batch_base = blockIdx.y * FP8_BATCH_TILE_X8;
    if (batch_base >= batch_rows) return;
    const unsigned int tile_rows = batch_rows - batch_base < FP8_BATCH_TILE_X8
        ? batch_rows - batch_base : FP8_BATCH_TILE_X8;

    __shared__ float fp4_lut[16];
    nvfp4_lut_init(fp4_lut, threadIdx.x);
    __syncthreads();

    const unsigned int warp_id = threadIdx.x / TALU_QUANT_WARP_SIZE;
    const unsigned int lane = threadIdx.x & (TALU_QUANT_WARP_SIZE - 1u);
    const unsigned int out_index = blockIdx.x * FP8_WARPS_PER_BLOCK + warp_id;
    const unsigned int qk_dim = q_out_dim + k_out_dim;
    const unsigned int total_dim = qk_dim + v_out_dim;
    if (out_index >= total_dim) return;

    const unsigned char* wt;
    const unsigned char* sc;
    float* out_ptr;
    unsigned int out_dim_proj;
    unsigned int row_idx;
    unsigned int scale_cols;
    unsigned int group_size;
    float weight_global_scale;
    if (out_index < q_out_dim) {
        wt = q_weight_packed;
        sc = q_scales;
        out_ptr = q_out;
        out_dim_proj = q_out_dim;
        row_idx = out_index;
        scale_cols = q_scale_cols;
        group_size = q_group_size;
        weight_global_scale = q_weight_global_scale;
    } else if (out_index < qk_dim) {
        wt = k_weight_packed;
        sc = k_scales;
        out_ptr = k_out;
        out_dim_proj = k_out_dim;
        row_idx = out_index - q_out_dim;
        scale_cols = k_scale_cols;
        group_size = k_group_size;
        weight_global_scale = k_weight_global_scale;
    } else {
        wt = v_weight_packed;
        sc = v_scales;
        out_ptr = v_out;
        out_dim_proj = v_out_dim;
        row_idx = out_index - qk_dim;
        scale_cols = v_scale_cols;
        group_size = v_group_size;
        weight_global_scale = v_weight_global_scale;
    }
    if (group_size != 16u || (in_dim & 15u) != 0u || weight_global_scale == 0.0f) return;

    const unsigned int packed_cols = (in_dim + 1u) >> 1;
    const float inv_global = 1.0f / weight_global_scale;
    const float* input_tile = input + (unsigned long long)batch_base * in_dim;
    float* out_tile = out_ptr + (unsigned long long)batch_base * out_dim_proj;
    const unsigned char* row_weight = wt + (unsigned long long)row_idx * packed_cols;

    switch (tile_rows) {
        case 1u: talu_nvfp4_matvec_batched<1>(input_tile, row_weight, sc, out_tile, in_dim, out_dim_proj, row_idx, scale_cols, inv_global, fp4_lut, lane, 1u); break;
        case 2u: talu_nvfp4_matvec_batched<2>(input_tile, row_weight, sc, out_tile, in_dim, out_dim_proj, row_idx, scale_cols, inv_global, fp4_lut, lane, 2u); break;
        case 3u: talu_nvfp4_matvec_batched<3>(input_tile, row_weight, sc, out_tile, in_dim, out_dim_proj, row_idx, scale_cols, inv_global, fp4_lut, lane, 3u); break;
        case 4u: talu_nvfp4_matvec_batched<4>(input_tile, row_weight, sc, out_tile, in_dim, out_dim_proj, row_idx, scale_cols, inv_global, fp4_lut, lane, 4u); break;
        case 5u: talu_nvfp4_matvec_batched<5>(input_tile, row_weight, sc, out_tile, in_dim, out_dim_proj, row_idx, scale_cols, inv_global, fp4_lut, lane, 5u); break;
        case 6u: talu_nvfp4_matvec_batched<6>(input_tile, row_weight, sc, out_tile, in_dim, out_dim_proj, row_idx, scale_cols, inv_global, fp4_lut, lane, 6u); break;
        case 7u: talu_nvfp4_matvec_batched<7>(input_tile, row_weight, sc, out_tile, in_dim, out_dim_proj, row_idx, scale_cols, inv_global, fp4_lut, lane, 7u); break;
        case 8u: talu_nvfp4_matvec_batched<8>(input_tile, row_weight, sc, out_tile, in_dim, out_dim_proj, row_idx, scale_cols, inv_global, fp4_lut, lane, 8u); break;
        default: return;
    }
}

extern "C" __global__ __launch_bounds__(128, 3) void talu_nvfp4_matvec_gate_up_f32(
    const float* __restrict__ input,
    const unsigned char* __restrict__ gate_weight_packed,
    const unsigned char* __restrict__ gate_scales,
    float* __restrict__ gate_out,
    unsigned int gate_out_dim,
    unsigned int gate_scale_cols,
    unsigned int gate_group_size,
    float gate_weight_global_scale,
    const unsigned char* __restrict__ up_weight_packed,
    const unsigned char* __restrict__ up_scales,
    float* __restrict__ up_out,
    unsigned int up_out_dim,
    unsigned int up_scale_cols,
    unsigned int up_group_size,
    float up_weight_global_scale,
    unsigned int in_dim,
    unsigned int batch_rows
) {
    const unsigned int batch_base = blockIdx.y * FP8_BATCH_TILE;
    if (batch_base >= batch_rows) return;
    const unsigned int tile_rows = batch_rows - batch_base < FP8_BATCH_TILE
        ? batch_rows - batch_base : FP8_BATCH_TILE;

    __shared__ float fp4_lut[16];
    nvfp4_lut_init(fp4_lut, threadIdx.x);
    __syncthreads();

    const unsigned int warp_id = threadIdx.x / TALU_QUANT_WARP_SIZE;
    const unsigned int lane = threadIdx.x & (TALU_QUANT_WARP_SIZE - 1u);
    const unsigned int out_index = blockIdx.x * FP8_WARPS_PER_BLOCK + warp_id;
    const unsigned int total_dim = gate_out_dim + up_out_dim;
    if (out_index >= total_dim) return;

    const unsigned char* wt;
    const unsigned char* sc;
    float* out_ptr;
    unsigned int out_dim_proj;
    unsigned int row_idx;
    unsigned int scale_cols;
    unsigned int group_size;
    float weight_global_scale;
    if (out_index < gate_out_dim) {
        wt = gate_weight_packed;
        sc = gate_scales;
        out_ptr = gate_out;
        out_dim_proj = gate_out_dim;
        row_idx = out_index;
        scale_cols = gate_scale_cols;
        group_size = gate_group_size;
        weight_global_scale = gate_weight_global_scale;
    } else {
        wt = up_weight_packed;
        sc = up_scales;
        out_ptr = up_out;
        out_dim_proj = up_out_dim;
        row_idx = out_index - gate_out_dim;
        scale_cols = up_scale_cols;
        group_size = up_group_size;
        weight_global_scale = up_weight_global_scale;
    }
    if (group_size != 16u || (in_dim & 15u) != 0u || weight_global_scale == 0.0f) return;

    const unsigned int packed_cols = (in_dim + 1u) >> 1;
    const float inv_global = 1.0f / weight_global_scale;
    const float* input_tile = input + (unsigned long long)batch_base * in_dim;
    float* out_tile = out_ptr + (unsigned long long)batch_base * out_dim_proj;
    const unsigned char* row_weight = wt + (unsigned long long)row_idx * packed_cols;

    switch (tile_rows) {
        case 1u: talu_nvfp4_matvec_batched<1>(input_tile, row_weight, sc, out_tile, in_dim, out_dim_proj, row_idx, scale_cols, inv_global, fp4_lut, lane, 1u); break;
        case 2u: talu_nvfp4_matvec_batched<2>(input_tile, row_weight, sc, out_tile, in_dim, out_dim_proj, row_idx, scale_cols, inv_global, fp4_lut, lane, 2u); break;
        case 3u: talu_nvfp4_matvec_batched<3>(input_tile, row_weight, sc, out_tile, in_dim, out_dim_proj, row_idx, scale_cols, inv_global, fp4_lut, lane, 3u); break;
        case 4u: talu_nvfp4_matvec_batched<4>(input_tile, row_weight, sc, out_tile, in_dim, out_dim_proj, row_idx, scale_cols, inv_global, fp4_lut, lane, 4u); break;
        default: return;
    }
}

extern "C" __global__ __launch_bounds__(128, 2) void talu_nvfp4_matvec_gate_up_f32_tile8(
    const float* __restrict__ input,
    const unsigned char* __restrict__ gate_weight_packed,
    const unsigned char* __restrict__ gate_scales,
    float* __restrict__ gate_out,
    unsigned int gate_out_dim,
    unsigned int gate_scale_cols,
    unsigned int gate_group_size,
    float gate_weight_global_scale,
    const unsigned char* __restrict__ up_weight_packed,
    const unsigned char* __restrict__ up_scales,
    float* __restrict__ up_out,
    unsigned int up_out_dim,
    unsigned int up_scale_cols,
    unsigned int up_group_size,
    float up_weight_global_scale,
    unsigned int in_dim,
    unsigned int batch_rows
) {
    const unsigned int batch_base = blockIdx.y * FP8_BATCH_TILE_X8;
    if (batch_base >= batch_rows) return;
    const unsigned int tile_rows = batch_rows - batch_base < FP8_BATCH_TILE_X8
        ? batch_rows - batch_base : FP8_BATCH_TILE_X8;

    __shared__ float fp4_lut[16];
    nvfp4_lut_init(fp4_lut, threadIdx.x);
    __syncthreads();

    const unsigned int warp_id = threadIdx.x / TALU_QUANT_WARP_SIZE;
    const unsigned int lane = threadIdx.x & (TALU_QUANT_WARP_SIZE - 1u);
    const unsigned int out_index = blockIdx.x * FP8_WARPS_PER_BLOCK + warp_id;
    const unsigned int total_dim = gate_out_dim + up_out_dim;
    if (out_index >= total_dim) return;

    const unsigned char* wt;
    const unsigned char* sc;
    float* out_ptr;
    unsigned int out_dim_proj;
    unsigned int row_idx;
    unsigned int scale_cols;
    unsigned int group_size;
    float weight_global_scale;
    if (out_index < gate_out_dim) {
        wt = gate_weight_packed;
        sc = gate_scales;
        out_ptr = gate_out;
        out_dim_proj = gate_out_dim;
        row_idx = out_index;
        scale_cols = gate_scale_cols;
        group_size = gate_group_size;
        weight_global_scale = gate_weight_global_scale;
    } else {
        wt = up_weight_packed;
        sc = up_scales;
        out_ptr = up_out;
        out_dim_proj = up_out_dim;
        row_idx = out_index - gate_out_dim;
        scale_cols = up_scale_cols;
        group_size = up_group_size;
        weight_global_scale = up_weight_global_scale;
    }
    if (group_size != 16u || (in_dim & 15u) != 0u || weight_global_scale == 0.0f) return;

    const unsigned int packed_cols = (in_dim + 1u) >> 1;
    const float inv_global = 1.0f / weight_global_scale;
    const float* input_tile = input + (unsigned long long)batch_base * in_dim;
    float* out_tile = out_ptr + (unsigned long long)batch_base * out_dim_proj;
    const unsigned char* row_weight = wt + (unsigned long long)row_idx * packed_cols;

    switch (tile_rows) {
        case 1u: talu_nvfp4_matvec_batched<1>(input_tile, row_weight, sc, out_tile, in_dim, out_dim_proj, row_idx, scale_cols, inv_global, fp4_lut, lane, 1u); break;
        case 2u: talu_nvfp4_matvec_batched<2>(input_tile, row_weight, sc, out_tile, in_dim, out_dim_proj, row_idx, scale_cols, inv_global, fp4_lut, lane, 2u); break;
        case 3u: talu_nvfp4_matvec_batched<3>(input_tile, row_weight, sc, out_tile, in_dim, out_dim_proj, row_idx, scale_cols, inv_global, fp4_lut, lane, 3u); break;
        case 4u: talu_nvfp4_matvec_batched<4>(input_tile, row_weight, sc, out_tile, in_dim, out_dim_proj, row_idx, scale_cols, inv_global, fp4_lut, lane, 4u); break;
        case 5u: talu_nvfp4_matvec_batched<5>(input_tile, row_weight, sc, out_tile, in_dim, out_dim_proj, row_idx, scale_cols, inv_global, fp4_lut, lane, 5u); break;
        case 6u: talu_nvfp4_matvec_batched<6>(input_tile, row_weight, sc, out_tile, in_dim, out_dim_proj, row_idx, scale_cols, inv_global, fp4_lut, lane, 6u); break;
        case 7u: talu_nvfp4_matvec_batched<7>(input_tile, row_weight, sc, out_tile, in_dim, out_dim_proj, row_idx, scale_cols, inv_global, fp4_lut, lane, 7u); break;
        case 8u: talu_nvfp4_matvec_batched<8>(input_tile, row_weight, sc, out_tile, in_dim, out_dim_proj, row_idx, scale_cols, inv_global, fp4_lut, lane, 8u); break;
        default: return;
    }
}

// Fused gate+up+silu NVFP4 GEMV with vectorized loads.
// Loads gate and up weights simultaneously, sharing input loads.
// Requires: group_size == 16 for both gate and up, in_dim % 16 == 0.
template <unsigned int BATCH>
static __device__ __forceinline__ void talu_nvfp4_gate_up_silu_batched(
    const float* input,
    const unsigned char* gate_weight_packed,
    const unsigned char* gate_scales,
    const unsigned char* up_weight_packed,
    const unsigned char* up_scales,
    float* out,
    unsigned int in_dim,
    unsigned int out_dim,
    unsigned int out_idx,
    unsigned int gate_scale_cols,
    float gate_inv_global,
    unsigned int up_scale_cols,
    float up_inv_global,
    const float* fp4_lut,
    unsigned int lane,
    unsigned int batch_rows
) {
    const unsigned int gate_scale_row_off = out_idx * gate_scale_cols;
    const unsigned int up_scale_row_off = out_idx * up_scale_cols;
    const unsigned int* gate_words = reinterpret_cast<const unsigned int*>(gate_weight_packed);
    const unsigned int* up_words = reinterpret_cast<const unsigned int*>(up_weight_packed);

    float gate_acc[BATCH] = {};
    float up_acc[BATCH] = {};

    for (unsigned int wp = lane; wp < gate_scale_cols; wp += TALU_QUANT_WARP_SIZE) {
        // 64-bit streaming loads for gate and up weights.
        unsigned int gw0, gw1, uw0, uw1;
        asm volatile(
            "ld.global.cs.v2.u32 {%0, %1}, [%2];"
            : "=r"(gw0), "=r"(gw1)
            : "l"(&gate_words[wp << 1]));
        asm volatile(
            "ld.global.cs.v2.u32 {%0, %1}, [%2];"
            : "=r"(uw0), "=r"(uw1)
            : "l"(&up_words[wp << 1]));

        const unsigned int base_elem = wp << 4;

        // 1 scale per 16-element group for gate and up.
        const float gs = fp8e4m3_to_f32(
            static_cast<__nv_fp8_storage_t>(gate_scales[gate_scale_row_off + wp])) * gate_inv_global;
        const float us = fp8e4m3_to_f32(
            static_cast<__nv_fp8_storage_t>(up_scales[up_scale_row_off + wp])) * up_inv_global;

        #pragma unroll
        for (unsigned int b = 0; b < BATCH; b++) {
            if (b >= batch_rows) break;
            const unsigned long long row_off = (unsigned long long)b * in_dim;

            float gp = 0.0f, up_p = 0.0f;

            // Phase 1: first 8 elements (word 0)
            {
                const float4 i0 = *reinterpret_cast<const float4*>(&input[row_off + base_elem]);
                const float4 i1 = *reinterpret_cast<const float4*>(&input[row_off + base_elem + 4]);
                NVFP4_PROCESS_WORD(gw0, i0, i1, gp, fp4_lut);
                NVFP4_PROCESS_WORD(uw0, i0, i1, up_p, fp4_lut);
            }

            // Phase 2: next 8 elements (word 1)
            {
                const float4 i2 = *reinterpret_cast<const float4*>(&input[row_off + base_elem + 8]);
                const float4 i3 = *reinterpret_cast<const float4*>(&input[row_off + base_elem + 12]);
                NVFP4_PROCESS_WORD(gw1, i2, i3, gp, fp4_lut);
                NVFP4_PROCESS_WORD(uw1, i2, i3, up_p, fp4_lut);
            }

            gate_acc[b] = fmaf(gp, gs, gate_acc[b]);
            up_acc[b] = fmaf(up_p, us, up_acc[b]);
        }
    }

    #pragma unroll
    for (unsigned int b = 0; b < BATCH; b++) {
        if (b >= batch_rows) break;
        const float gate = talu_quant_warp_sum_f32(gate_acc[b]);
        const float up = talu_quant_warp_sum_f32(up_acc[b]);
        if (lane == 0) {
            const float sigma = 1.0f / (1.0f + expf(-gate));
            out[(unsigned long long)b * out_dim + out_idx] = gate * sigma * up;
        }
    }
}

extern "C" __global__ __launch_bounds__(128, 3) void talu_nvfp4_matvec_gate_up_silu_f32(
    const float* __restrict__ input,
    const unsigned char* __restrict__ gate_weight_packed,
    const unsigned char* __restrict__ gate_scales,
    const unsigned char* __restrict__ up_weight_packed,
    const unsigned char* __restrict__ up_scales,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim,
    unsigned int gate_scale_cols,
    unsigned int gate_group_size,
    float gate_weight_global_scale,
    unsigned int up_scale_cols,
    unsigned int up_group_size,
    float up_weight_global_scale,
    unsigned int batch_rows
) {
    const unsigned int batch_base = blockIdx.y * FP8_BATCH_TILE;
    if (batch_base >= batch_rows) return;
    const unsigned int tile_rows = batch_rows - batch_base < FP8_BATCH_TILE
        ? batch_rows - batch_base : FP8_BATCH_TILE;
    if (gate_group_size != 16u || up_group_size != 16u) return;
    if ((in_dim & 15u) != 0u) return;
    if (gate_weight_global_scale == 0.0f || up_weight_global_scale == 0.0f) return;

    __shared__ float fp4_lut[16];
    nvfp4_lut_init(fp4_lut, threadIdx.x);
    __syncthreads();

    const unsigned int warp_id = threadIdx.x / TALU_QUANT_WARP_SIZE;
    const unsigned int lane = threadIdx.x & (TALU_QUANT_WARP_SIZE - 1u);
    const unsigned int out_idx = blockIdx.x * FP8_WARPS_PER_BLOCK + warp_id;
    if (out_idx >= out_dim) return;

    const unsigned int packed_cols = (in_dim + 1u) >> 1;
    const float g_inv = 1.0f / gate_weight_global_scale;
    const float u_inv = 1.0f / up_weight_global_scale;
    const float* input_tile = input + (unsigned long long)batch_base * in_dim;
    float* out_tile = out + (unsigned long long)batch_base * out_dim;
    const unsigned char* gate_row_weight = gate_weight_packed + (unsigned long long)out_idx * packed_cols;
    const unsigned char* up_row_weight = up_weight_packed + (unsigned long long)out_idx * packed_cols;

    switch (tile_rows) {
        case 1u: talu_nvfp4_gate_up_silu_batched<1>(input_tile, gate_row_weight, gate_scales, up_row_weight, up_scales, out_tile, in_dim, out_dim, out_idx, gate_scale_cols, g_inv, up_scale_cols, u_inv, fp4_lut, lane, 1u); break;
        case 2u: talu_nvfp4_gate_up_silu_batched<2>(input_tile, gate_row_weight, gate_scales, up_row_weight, up_scales, out_tile, in_dim, out_dim, out_idx, gate_scale_cols, g_inv, up_scale_cols, u_inv, fp4_lut, lane, 2u); break;
        case 3u: talu_nvfp4_gate_up_silu_batched<3>(input_tile, gate_row_weight, gate_scales, up_row_weight, up_scales, out_tile, in_dim, out_dim, out_idx, gate_scale_cols, g_inv, up_scale_cols, u_inv, fp4_lut, lane, 3u); break;
        case 4u: talu_nvfp4_gate_up_silu_batched<4>(input_tile, gate_row_weight, gate_scales, up_row_weight, up_scales, out_tile, in_dim, out_dim, out_idx, gate_scale_cols, g_inv, up_scale_cols, u_inv, fp4_lut, lane, 4u); break;
        default: return;
    }
}

extern "C" __global__ __launch_bounds__(128, 2) void talu_nvfp4_matvec_gate_up_silu_f32_tile8(
    const float* __restrict__ input,
    const unsigned char* __restrict__ gate_weight_packed,
    const unsigned char* __restrict__ gate_scales,
    const unsigned char* __restrict__ up_weight_packed,
    const unsigned char* __restrict__ up_scales,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim,
    unsigned int gate_scale_cols,
    unsigned int gate_group_size,
    float gate_weight_global_scale,
    unsigned int up_scale_cols,
    unsigned int up_group_size,
    float up_weight_global_scale,
    unsigned int batch_rows
) {
    const unsigned int batch_base = blockIdx.y * FP8_BATCH_TILE_X8;
    if (batch_base >= batch_rows) return;
    const unsigned int tile_rows = batch_rows - batch_base < FP8_BATCH_TILE_X8
        ? batch_rows - batch_base : FP8_BATCH_TILE_X8;
    if (gate_group_size != 16u || up_group_size != 16u) return;
    if ((in_dim & 15u) != 0u) return;
    if (gate_weight_global_scale == 0.0f || up_weight_global_scale == 0.0f) return;

    __shared__ float fp4_lut[16];
    nvfp4_lut_init(fp4_lut, threadIdx.x);
    __syncthreads();

    const unsigned int warp_id = threadIdx.x / TALU_QUANT_WARP_SIZE;
    const unsigned int lane = threadIdx.x & (TALU_QUANT_WARP_SIZE - 1u);
    const unsigned int out_idx = blockIdx.x * FP8_WARPS_PER_BLOCK + warp_id;
    if (out_idx >= out_dim) return;

    const unsigned int packed_cols = (in_dim + 1u) >> 1;
    const float g_inv = 1.0f / gate_weight_global_scale;
    const float u_inv = 1.0f / up_weight_global_scale;
    const float* input_tile = input + (unsigned long long)batch_base * in_dim;
    float* out_tile = out + (unsigned long long)batch_base * out_dim;
    const unsigned char* gate_row_weight = gate_weight_packed + (unsigned long long)out_idx * packed_cols;
    const unsigned char* up_row_weight = up_weight_packed + (unsigned long long)out_idx * packed_cols;

    switch (tile_rows) {
        case 1u: talu_nvfp4_gate_up_silu_batched<1>(input_tile, gate_row_weight, gate_scales, up_row_weight, up_scales, out_tile, in_dim, out_dim, out_idx, gate_scale_cols, g_inv, up_scale_cols, u_inv, fp4_lut, lane, 1u); break;
        case 2u: talu_nvfp4_gate_up_silu_batched<2>(input_tile, gate_row_weight, gate_scales, up_row_weight, up_scales, out_tile, in_dim, out_dim, out_idx, gate_scale_cols, g_inv, up_scale_cols, u_inv, fp4_lut, lane, 2u); break;
        case 3u: talu_nvfp4_gate_up_silu_batched<3>(input_tile, gate_row_weight, gate_scales, up_row_weight, up_scales, out_tile, in_dim, out_dim, out_idx, gate_scale_cols, g_inv, up_scale_cols, u_inv, fp4_lut, lane, 3u); break;
        case 4u: talu_nvfp4_gate_up_silu_batched<4>(input_tile, gate_row_weight, gate_scales, up_row_weight, up_scales, out_tile, in_dim, out_dim, out_idx, gate_scale_cols, g_inv, up_scale_cols, u_inv, fp4_lut, lane, 4u); break;
        case 5u: talu_nvfp4_gate_up_silu_batched<5>(input_tile, gate_row_weight, gate_scales, up_row_weight, up_scales, out_tile, in_dim, out_dim, out_idx, gate_scale_cols, g_inv, up_scale_cols, u_inv, fp4_lut, lane, 5u); break;
        case 6u: talu_nvfp4_gate_up_silu_batched<6>(input_tile, gate_row_weight, gate_scales, up_row_weight, up_scales, out_tile, in_dim, out_dim, out_idx, gate_scale_cols, g_inv, up_scale_cols, u_inv, fp4_lut, lane, 6u); break;
        case 7u: talu_nvfp4_gate_up_silu_batched<7>(input_tile, gate_row_weight, gate_scales, up_row_weight, up_scales, out_tile, in_dim, out_dim, out_idx, gate_scale_cols, g_inv, up_scale_cols, u_inv, fp4_lut, lane, 7u); break;
        case 8u: talu_nvfp4_gate_up_silu_batched<8>(input_tile, gate_row_weight, gate_scales, up_row_weight, up_scales, out_tile, in_dim, out_dim, out_idx, gate_scale_cols, g_inv, up_scale_cols, u_inv, fp4_lut, lane, 8u); break;
        default: return;
    }
}

#else

extern "C" __global__ void talu_nvfp4_to_i8(
    const unsigned char* weight_packed, const unsigned char* scales,
    signed char* out_i8, float* out_row_scales, unsigned int in_dim,
    unsigned int scale_cols, float weight_global_scale) {}
extern "C" __global__ void talu_dequant_nvfp4_to_bf16(
    const unsigned char* weight_packed, const unsigned char* scales,
    unsigned short* out_bf16, unsigned int in_dim, unsigned int out_dim,
    unsigned int scale_cols, float weight_global_scale) {}
extern "C" __global__ void talu_nvfp4_matvec_f32(
    const float* input, const unsigned char* weight_packed, const unsigned char* scales,
    float* out, unsigned int in_dim, unsigned int out_dim, unsigned int scale_cols,
    unsigned int group_size, float weight_global_scale, unsigned int batch_rows) {}
extern "C" __global__ void talu_nvfp4_matvec_f32_tile8(
    const float* input, const unsigned char* weight_packed, const unsigned char* scales,
    float* out, unsigned int in_dim, unsigned int out_dim, unsigned int scale_cols,
    unsigned int group_size, float weight_global_scale, unsigned int batch_rows) {}
extern "C" __global__ void talu_nvfp4_matvec_qkv_f32(
    const float* input, const unsigned char* q_weight_packed, const unsigned char* q_scales,
    float* q_out, unsigned int q_out_dim, unsigned int q_scale_cols, unsigned int q_group_size,
    float q_weight_global_scale, const unsigned char* k_weight_packed, const unsigned char* k_scales,
    float* k_out, unsigned int k_out_dim, unsigned int k_scale_cols, unsigned int k_group_size,
    float k_weight_global_scale, const unsigned char* v_weight_packed, const unsigned char* v_scales,
    float* v_out, unsigned int v_out_dim, unsigned int v_scale_cols, unsigned int v_group_size,
    float v_weight_global_scale, unsigned int in_dim, unsigned int batch_rows) {}
extern "C" __global__ void talu_nvfp4_matvec_qkv_f32_tile8(
    const float* input, const unsigned char* q_weight_packed, const unsigned char* q_scales,
    float* q_out, unsigned int q_out_dim, unsigned int q_scale_cols, unsigned int q_group_size,
    float q_weight_global_scale, const unsigned char* k_weight_packed, const unsigned char* k_scales,
    float* k_out, unsigned int k_out_dim, unsigned int k_scale_cols, unsigned int k_group_size,
    float k_weight_global_scale, const unsigned char* v_weight_packed, const unsigned char* v_scales,
    float* v_out, unsigned int v_out_dim, unsigned int v_scale_cols, unsigned int v_group_size,
    float v_weight_global_scale, unsigned int in_dim, unsigned int batch_rows) {}
extern "C" __global__ void talu_nvfp4_matvec_gate_up_f32(
    const float* input, const unsigned char* gate_weight_packed, const unsigned char* gate_scales,
    float* gate_out, unsigned int gate_out_dim, unsigned int gate_scale_cols, unsigned int gate_group_size,
    float gate_weight_global_scale, const unsigned char* up_weight_packed, const unsigned char* up_scales,
    float* up_out, unsigned int up_out_dim, unsigned int up_scale_cols, unsigned int up_group_size,
    float up_weight_global_scale, unsigned int in_dim, unsigned int batch_rows) {}
extern "C" __global__ void talu_nvfp4_matvec_gate_up_f32_tile8(
    const float* input, const unsigned char* gate_weight_packed, const unsigned char* gate_scales,
    float* gate_out, unsigned int gate_out_dim, unsigned int gate_scale_cols, unsigned int gate_group_size,
    float gate_weight_global_scale, const unsigned char* up_weight_packed, const unsigned char* up_scales,
    float* up_out, unsigned int up_out_dim, unsigned int up_scale_cols, unsigned int up_group_size,
    float up_weight_global_scale, unsigned int in_dim, unsigned int batch_rows) {}
extern "C" __global__ void talu_nvfp4_matvec_gate_up_silu_f32(
    const float* input, const unsigned char* gate_weight_packed, const unsigned char* gate_scales,
    const unsigned char* up_weight_packed, const unsigned char* up_scales, float* out,
    unsigned int out_dim, unsigned int in_dim, unsigned int gate_scale_cols, unsigned int gate_group_size,
    float gate_weight_global_scale, unsigned int up_scale_cols, unsigned int up_group_size,
    float up_weight_global_scale, unsigned int batch_rows) {}
extern "C" __global__ void talu_nvfp4_matvec_gate_up_silu_f32_tile8(
    const float* input, const unsigned char* gate_weight_packed, const unsigned char* gate_scales,
    const unsigned char* up_weight_packed, const unsigned char* up_scales, float* out,
    unsigned int out_dim, unsigned int in_dim, unsigned int gate_scale_cols, unsigned int gate_group_size,
    float gate_weight_global_scale, unsigned int up_scale_cols, unsigned int up_group_size,
    float up_weight_global_scale, unsigned int batch_rows) {}

#endif
