// MXFP8 (E4M3 + UE8M0 block-32 scales) quantized GEMV and dequantization kernels.

// ---------------------------------------------------------------------------
// MXFP8 activation quantization: F32 → E4M3 + UE8M0 block-32 scales
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// cuBLASLt interleaved scale offset helper (used by both quant + dequant)
// ---------------------------------------------------------------------------

// Compute interleaved offset for cuBLASLt VEC32_UE8M0 scale tensor.
// Maps (outer_idx, inner_scale_idx) to flat byte offset in the interleaved buffer.
// Layout: 128×4 tiles (512 bytes each) in row-major order across (outer, inner) dimensions.
// Within each tile: 4 sub-tiles of 32 rows are interleaved (32×16 physical layout).
// Matches PyTorch/AO to_blocked() and cuBLAS 13.2 section 3.1.4.4.2.
__device__ __forceinline__ unsigned int mxfp8_scale_offset(
    unsigned int m,             // outer index (row for activations, col for weights)
    unsigned int k,             // inner scale index (0..sf_k-1)
    unsigned int n_col_tiles    // padded_sf_k / 4 (number of inner-dimension tiles)
) {
    return (m / 128) * n_col_tiles * 512 +
           (k / 4) * 512 +
           (m % 32) * 16 +
           ((m % 128) / 32) * 4 +
           (k % 4);
}

// ---------------------------------------------------------------------------
// MXFP8 activation quantization: F32 → E4M3 + UE8M0 block-32 scales
// ---------------------------------------------------------------------------

// Quantize F32 activations to MXFP8 format for cuBLASLt block-scaled GEMM.
// Each warp processes one 32-element group: finds absmax, computes UE8M0 scale,
// quantizes all 32 elements to FP8 E4M3.
//
// Scales are written in cuBLASLt interleaved tile layout (128×4 tiles).
// Padded rows (row >= rows but < padded_outer) get zero scales.
//
// Launch: grid=(ceil(in_dim/32), padded_outer), block=(32)
// in_dim must be a multiple of 32.

#if __CUDA_ARCH__ >= 890

extern "C" __global__ void talu_quantize_f32_to_mxfp8(
    const float* __restrict__ input,        // [rows × in_dim]
    unsigned char* __restrict__ out_fp8,     // [rows × in_dim] E4M3 bytes
    unsigned char* __restrict__ out_scales,  // [padded_outer × padded_sf_k] interleaved UE8M0
    unsigned int in_dim,
    unsigned int rows,
    unsigned int padded_outer,              // = roundoff(rows, 128)
    unsigned int padded_sf_k                // = roundoff(ceil(in_dim/32), 4)
) {
    const unsigned int row = blockIdx.y;
    const unsigned int group = blockIdx.x;  // which 32-element group in this row
    const unsigned int lane = threadIdx.x;  // 0..31

    if (row >= padded_outer) return;

    const unsigned int col = group * 32 + lane;
    if (col >= in_dim) return;

    // Padded rows: write zero scale (e8m0=0 → scale=2^-127 ≈ 0), skip E4M3 output
    if (row >= rows) {
        if (lane == 0) {
            const unsigned int n_col_tiles = padded_sf_k / 4;
            out_scales[mxfp8_scale_offset(row, group, n_col_tiles)] = 0;
        }
        return;
    }

    const unsigned long long offset = (unsigned long long)row * in_dim + col;
    const float val = input[offset];

    // Warp-wide absmax reduction
    float absval = fabsf(val);
    for (int s = 16; s >= 1; s >>= 1)
        absval = fmaxf(absval, __shfl_xor_sync(0xFFFFFFFFu, absval, s));
    // absval now holds group absmax in all lanes

    // Compute UE8M0 exponent: smallest power-of-2 scale such that absmax / scale <= 448
    // E8M0 value = ceil(log2(absmax / 448)) + 127
    // Special case: absmax == 0 → e8m0 = 127 (scale = 1.0)
    unsigned char e8m0;
    float scale;
    if (absval == 0.0f) {
        e8m0 = 127;
        scale = 1.0f;
    } else {
        // Get exponent of (absmax / 448): use integer bit extraction
        // log2(absmax) is the IEEE754 exponent, and 448 = 2^8 * 1.75
        // We need ceil(log2(absmax / 448)) = ceil(log2(absmax) - log2(448))
        // Simpler: find smallest 2^n >= absmax/448
        float ratio = absval / 448.0f;
        // Round up to next power of 2
        unsigned int ratio_bits = __float_as_uint(ratio);
        // If ratio is exactly a power of 2, keep it; otherwise round up
        int exponent = ((int)(ratio_bits >> 23) & 0xFF) - 127;
        // Check if mantissa is non-zero (not exact power of 2)
        if (ratio_bits & 0x007FFFFFu) exponent += 1;
        // Clamp exponent to E8M0 range [-127, 128]
        exponent = max(exponent, -127);
        exponent = min(exponent, 127);
        e8m0 = (unsigned char)(exponent + 127);
        // Reconstruct scale = 2^exponent
        scale = __uint_as_float((unsigned int)(e8m0) << 23);
    }

    // Lane 0 writes the scale at interleaved offset
    if (lane == 0) {
        const unsigned int n_col_tiles = padded_sf_k / 4;
        out_scales[mxfp8_scale_offset(row, group, n_col_tiles)] = e8m0;
    }

    // Quantize this element to E4M3
    float scaled_val = val / scale;
    out_fp8[offset] = __nv_cvt_float_to_fp8(scaled_val, __NV_SATFINITE, __NV_E4M3);
}

#else

// Stub for architectures below sm_89
extern "C" __global__ void talu_quantize_f32_to_mxfp8(
    const float* __restrict__ input,
    void* __restrict__ out_fp8,
    void* __restrict__ out_scales,
    unsigned int in_dim,
    unsigned int rows,
    unsigned int padded_outer,
    unsigned int padded_sf_k
) {}

#endif // __CUDA_ARCH__ >= 890

// ---------------------------------------------------------------------------
// MXFP8 dequantization: E4M3 + UE8M0 block-32 scales → BF16
// ---------------------------------------------------------------------------

// Dequantize MXFP8 weights to BF16 for cuBLAS GEMM fallback on non-Blackwell GPUs.
// Each thread processes one element: fp8_to_f32(byte) * e8m0_to_f32(scale) → bf16.
// Scale tensor is in cuBLASLt interleaved layout (same buffer used for both paths).
// Launch: grid=(ceil(in_dim/256), out_dim), block=(256)

#if __CUDA_ARCH__ >= 890

extern "C" __global__ void talu_dequant_mxfp8_to_bf16(
    const unsigned char* __restrict__ weight,   // [out_dim × in_dim] FP8 E4M3 bytes
    const unsigned char* __restrict__ scales,   // interleaved UE8M0 bytes
    unsigned short* __restrict__ out_bf16,      // [out_dim × in_dim] BF16
    unsigned int in_dim,
    unsigned int out_dim,
    unsigned int padded_outer,                  // = roundoff(out_dim, 128)
    unsigned int padded_sf_k                    // = roundoff(ceil(in_dim/32), 4)
) {
    const unsigned int row = blockIdx.y;
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim || col >= in_dim) return;

    const unsigned long long idx = (unsigned long long)row * in_dim + col;

    // Read scale from interleaved position: (m=row, k=col/32)
    const unsigned int k = col / 32;
    const unsigned int n_col_tiles = padded_sf_k / 4;
    const unsigned int scale_idx = mxfp8_scale_offset(row, k, n_col_tiles);

    // UE8M0 → F32: scale = 2^(e8m0 - 127), encoded as IEEE754 with e8m0 in exponent field
    const float block_scale = __uint_as_float((unsigned int)(scales[scale_idx]) << 23);

    // FP8 E4M3 → F32 → scale → BF16
    const float val = fp8e4m3_to_f32(static_cast<__nv_fp8_storage_t>(weight[idx])) * block_scale;
    // F32 → BF16: round-to-nearest-even
    unsigned int fbits;
    memcpy(&fbits, &val, sizeof(fbits));
    const unsigned int lsb = (fbits >> 16) & 1u;
    const unsigned int rounding_bias = 0x7FFFu + lsb;
    out_bf16[idx] = static_cast<unsigned short>((fbits + rounding_bias) >> 16);
}

// ---------------------------------------------------------------------------
// MXFP8 GEMV: E4M3 weights + UE8M0 per-32 scales, F32 input → F32 output
// ---------------------------------------------------------------------------
// Like the FP8 GEMV but reads 1-byte UE8M0 scales per 32 elements instead of
// 2-byte BF16 per-block scales. Eliminates the separate activation quantization
// kernel needed by cuBLASLt. Used for small batch sizes (M≤8) where cuBLASLt
// overhead dominates.

// Convert UE8M0 scale byte to float: value = 2^(e - 127).
// In IEEE 754 this is just the exponent field (e=0 maps to 0.0f which is fine).
static __device__ __forceinline__ float e8m0_to_f32(unsigned char e) {
    return __uint_as_float(static_cast<unsigned int>(e) << 23);
}

// MXFP8 batched inner GEMV: weight stays in registers, reused across BATCH rows.
// Scales are simple row-major UE8M0: scales[out_idx * scale_cols + elem/32].
template <unsigned int BATCH>
static __device__ __forceinline__ void talu_mxfp8_matvec_batched(
    const float* input,
    const unsigned int* weight_words,
    const unsigned char* scales,
    float* out,
    unsigned int in_dim,
    unsigned int out_dim,
    unsigned int out_idx,
    unsigned int scale_cols,
    unsigned int lane,
    unsigned int batch_rows
) {
    const unsigned int words_per_row = in_dim >> 2;
    const unsigned int scale_row_off = out_idx * scale_cols;

    float acc[BATCH] = {};

    for (unsigned int w = lane << 1; w < words_per_row; w += TALU_QUANT_WARP_SIZE << 1) {
        // 64-bit streaming load: 2 words = 8 FP8 bytes per lane per iteration.
        unsigned int w0, w1;
        asm volatile(
            "ld.global.cs.v2.u32 {%0, %1}, [%2];"
            : "=r"(w0), "=r"(w1)
            : "l"(&weight_words[w]));

        const unsigned int base_elem = w << 2;

        // Per-32-element UE8M0 scales.
        const float s0 = e8m0_to_f32(scales[scale_row_off + base_elem / 32]);
        const float s1 = e8m0_to_f32(scales[scale_row_off + (base_elem + 4) / 32]);

        // Per-batch input load + accumulate: weight stays in registers.
        #pragma unroll
        for (unsigned int b = 0; b < BATCH; b++) {
            if (b >= batch_rows) break;
            const unsigned long long row_off = (unsigned long long)b * in_dim;
            float r0, r1;

            r0 = 0.0f; r1 = 0.0f;
            const float4 i0 = *reinterpret_cast<const float4*>(&input[row_off + base_elem]);
            FP8_PROCESS_WORD(w0, i0, r0, r1);
            acc[b] = fmaf(r0 + r1, s0, acc[b]);

            r0 = 0.0f; r1 = 0.0f;
            const float4 i1 = *reinterpret_cast<const float4*>(&input[row_off + base_elem + 4]);
            FP8_PROCESS_WORD(w1, i1, r0, r1);
            acc[b] = fmaf(r0 + r1, s1, acc[b]);
        }
    }

    #pragma unroll
    for (unsigned int b = 0; b < BATCH; b++) {
        if (b >= batch_rows) break;
        float result = talu_quant_warp_sum_f32(acc[b]);
        if (lane == 0) {
            out[(unsigned long long)b * out_dim + out_idx] = result;
        }
    }
}

// MXFP8 GEMV entry point: tile-4 (up to 4 batch rows per block).
extern "C" __global__ __launch_bounds__(128, 3) void talu_mxfp8_matvec_f32(
    const float* __restrict__ input,
    const unsigned char* __restrict__ weight,
    const unsigned char* __restrict__ scales,
    float* __restrict__ out,
    unsigned int in_dim,
    unsigned int out_dim,
    unsigned int scale_cols,
    unsigned int batch_rows
) {
    const unsigned int batch_base = blockIdx.y * FP8_BATCH_TILE;
    if (batch_base >= batch_rows) return;
    const unsigned int tile_rows = batch_rows - batch_base < FP8_BATCH_TILE
        ? batch_rows - batch_base : FP8_BATCH_TILE;

    const unsigned int warp_id = threadIdx.x / TALU_QUANT_WARP_SIZE;
    const unsigned int lane = threadIdx.x & (TALU_QUANT_WARP_SIZE - 1u);
    const unsigned int out_idx = blockIdx.x * FP8_WARPS_PER_BLOCK + warp_id;
    if (out_idx >= out_dim) return;

    const float* input_tile = input + (unsigned long long)batch_base * in_dim;
    float* out_tile = out + (unsigned long long)batch_base * out_dim;
    const unsigned int* weight_words = reinterpret_cast<const unsigned int*>(
        weight + (unsigned long long)out_idx * in_dim);

    switch (tile_rows) {
        case 1u: talu_mxfp8_matvec_batched<1>(input_tile, weight_words, scales, out_tile, in_dim, out_dim, out_idx, scale_cols, lane, 1u); break;
        case 2u: talu_mxfp8_matvec_batched<2>(input_tile, weight_words, scales, out_tile, in_dim, out_dim, out_idx, scale_cols, lane, 2u); break;
        case 3u: talu_mxfp8_matvec_batched<3>(input_tile, weight_words, scales, out_tile, in_dim, out_dim, out_idx, scale_cols, lane, 3u); break;
        case 4u: talu_mxfp8_matvec_batched<4>(input_tile, weight_words, scales, out_tile, in_dim, out_dim, out_idx, scale_cols, lane, 4u); break;
        default: return;
    }
}

// MXFP8 GEMV entry point: tile-8 (up to 8 batch rows per block).
extern "C" __global__ __launch_bounds__(128, 2) void talu_mxfp8_matvec_f32_tile8(
    const float* __restrict__ input,
    const unsigned char* __restrict__ weight,
    const unsigned char* __restrict__ scales,
    float* __restrict__ out,
    unsigned int in_dim,
    unsigned int out_dim,
    unsigned int scale_cols,
    unsigned int batch_rows
) {
    const unsigned int batch_base = blockIdx.y * FP8_BATCH_TILE_X8;
    if (batch_base >= batch_rows) return;
    const unsigned int tile_rows = batch_rows - batch_base < FP8_BATCH_TILE_X8
        ? batch_rows - batch_base : FP8_BATCH_TILE_X8;

    const unsigned int warp_id = threadIdx.x / TALU_QUANT_WARP_SIZE;
    const unsigned int lane = threadIdx.x & (TALU_QUANT_WARP_SIZE - 1u);
    const unsigned int out_idx = blockIdx.x * FP8_WARPS_PER_BLOCK + warp_id;
    if (out_idx >= out_dim) return;

    const float* input_tile = input + (unsigned long long)batch_base * in_dim;
    float* out_tile = out + (unsigned long long)batch_base * out_dim;
    const unsigned int* weight_words = reinterpret_cast<const unsigned int*>(
        weight + (unsigned long long)out_idx * in_dim);

    switch (tile_rows) {
        case 1u: talu_mxfp8_matvec_batched<1>(input_tile, weight_words, scales, out_tile, in_dim, out_dim, out_idx, scale_cols, lane, 1u); break;
        case 2u: talu_mxfp8_matvec_batched<2>(input_tile, weight_words, scales, out_tile, in_dim, out_dim, out_idx, scale_cols, lane, 2u); break;
        case 3u: talu_mxfp8_matvec_batched<3>(input_tile, weight_words, scales, out_tile, in_dim, out_dim, out_idx, scale_cols, lane, 3u); break;
        case 4u: talu_mxfp8_matvec_batched<4>(input_tile, weight_words, scales, out_tile, in_dim, out_dim, out_idx, scale_cols, lane, 4u); break;
        case 5u: talu_mxfp8_matvec_batched<5>(input_tile, weight_words, scales, out_tile, in_dim, out_dim, out_idx, scale_cols, lane, 5u); break;
        case 6u: talu_mxfp8_matvec_batched<6>(input_tile, weight_words, scales, out_tile, in_dim, out_dim, out_idx, scale_cols, lane, 6u); break;
        case 7u: talu_mxfp8_matvec_batched<7>(input_tile, weight_words, scales, out_tile, in_dim, out_dim, out_idx, scale_cols, lane, 7u); break;
        case 8u: talu_mxfp8_matvec_batched<8>(input_tile, weight_words, scales, out_tile, in_dim, out_dim, out_idx, scale_cols, lane, 8u); break;
        default: return;
    }
}

extern "C" __global__ __launch_bounds__(128, 2) void talu_mxfp8_matvec_gate_up_silu_f32(
    const float* __restrict__ input,
    const unsigned char* __restrict__ gate_weight,
    const unsigned char* __restrict__ gate_scales,
    const unsigned char* __restrict__ up_weight,
    const unsigned char* __restrict__ up_scales,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim,
    unsigned int gate_scale_cols,
    unsigned int up_scale_cols,
    unsigned int batch_rows
) {
    const unsigned int batch_base = blockIdx.y * FP8_BATCH_TILE;
    if (batch_base >= batch_rows) return;
    const unsigned int tile_rows = batch_rows - batch_base < FP8_BATCH_TILE
        ? batch_rows - batch_base : FP8_BATCH_TILE;

    const unsigned int warp_id = threadIdx.x / TALU_QUANT_WARP_SIZE;
    const unsigned int lane = threadIdx.x & (TALU_QUANT_WARP_SIZE - 1u);
    const unsigned int out_idx = blockIdx.x * FP8_WARPS_PER_BLOCK + warp_id;
    if (out_idx >= out_dim) return;

    const float* input_tile = input + (unsigned long long)batch_base * in_dim;
    const unsigned int* gw = reinterpret_cast<const unsigned int*>(
        gate_weight + (unsigned long long)out_idx * in_dim);
    const unsigned int* uw = reinterpret_cast<const unsigned int*>(
        up_weight + (unsigned long long)out_idx * in_dim);
    const unsigned int g_sro = out_idx * gate_scale_cols;
    const unsigned int u_sro = out_idx * up_scale_cols;
    const unsigned int words_per_row = in_dim >> 2;

    float gate_acc[FP8_BATCH_TILE] = {};
    float up_acc[FP8_BATCH_TILE] = {};

    for (unsigned int w = lane << 1; w < words_per_row; w += TALU_QUANT_WARP_SIZE << 1) {
        unsigned int gw0, gw1;
        asm volatile("ld.global.cs.v2.u32 {%0, %1}, [%2];"
            : "=r"(gw0), "=r"(gw1) : "l"(&gw[w]));
        unsigned int uw0, uw1;
        asm volatile("ld.global.cs.v2.u32 {%0, %1}, [%2];"
            : "=r"(uw0), "=r"(uw1) : "l"(&uw[w]));

        const unsigned int base = w << 2;
        const float gs0 = e8m0_to_f32(gate_scales[g_sro + base / 32]);
        const float gs1 = e8m0_to_f32(gate_scales[g_sro + (base + 4) / 32]);
        const float us0 = e8m0_to_f32(up_scales[u_sro + base / 32]);
        const float us1 = e8m0_to_f32(up_scales[u_sro + (base + 4) / 32]);

        #pragma unroll
        for (unsigned int b = 0; b < FP8_BATCH_TILE; b++) {
            if (b >= tile_rows) break;
            const unsigned long long row_off = (unsigned long long)b * in_dim;
            float r0, r1;

            const float4 inp0 = *reinterpret_cast<const float4*>(&input_tile[row_off + base]);
            r0 = 0.0f; r1 = 0.0f; FP8_PROCESS_WORD(gw0, inp0, r0, r1);
            gate_acc[b] = fmaf(r0 + r1, gs0, gate_acc[b]);
            r0 = 0.0f; r1 = 0.0f; FP8_PROCESS_WORD(uw0, inp0, r0, r1);
            up_acc[b] = fmaf(r0 + r1, us0, up_acc[b]);

            const float4 inp1 = *reinterpret_cast<const float4*>(&input_tile[row_off + base + 4]);
            r0 = 0.0f; r1 = 0.0f; FP8_PROCESS_WORD(gw1, inp1, r0, r1);
            gate_acc[b] = fmaf(r0 + r1, gs1, gate_acc[b]);
            r0 = 0.0f; r1 = 0.0f; FP8_PROCESS_WORD(uw1, inp1, r0, r1);
            up_acc[b] = fmaf(r0 + r1, us1, up_acc[b]);
        }
    }

    #pragma unroll
    for (unsigned int b = 0; b < FP8_BATCH_TILE; b++) {
        if (b >= tile_rows) break;
        float g = talu_quant_warp_sum_f32(gate_acc[b]);
        float u = talu_quant_warp_sum_f32(up_acc[b]);
        if (lane == 0) {
            const float sigma = 1.0f / (1.0f + expf(-g));
            out[(unsigned long long)(batch_base + b) * out_dim + out_idx] = g * sigma * u;
        }
    }
}

// MXFP8 fused gate+up+silu GEMV: tile-8 (up to 8 batch rows per block).
extern "C" __global__ __launch_bounds__(128, 2) void talu_mxfp8_matvec_gate_up_silu_f32_tile8(
    const float* __restrict__ input,
    const unsigned char* __restrict__ gate_weight,
    const unsigned char* __restrict__ gate_scales,
    const unsigned char* __restrict__ up_weight,
    const unsigned char* __restrict__ up_scales,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim,
    unsigned int gate_scale_cols,
    unsigned int up_scale_cols,
    unsigned int batch_rows
) {
    const unsigned int batch_base = blockIdx.y * FP8_BATCH_TILE_X8;
    if (batch_base >= batch_rows) return;
    const unsigned int tile_rows = batch_rows - batch_base < FP8_BATCH_TILE_X8
        ? batch_rows - batch_base : FP8_BATCH_TILE_X8;

    const unsigned int warp_id = threadIdx.x / TALU_QUANT_WARP_SIZE;
    const unsigned int lane = threadIdx.x & (TALU_QUANT_WARP_SIZE - 1u);
    const unsigned int out_idx = blockIdx.x * FP8_WARPS_PER_BLOCK + warp_id;
    if (out_idx >= out_dim) return;

    const float* input_tile = input + (unsigned long long)batch_base * in_dim;
    const unsigned int* gw = reinterpret_cast<const unsigned int*>(
        gate_weight + (unsigned long long)out_idx * in_dim);
    const unsigned int* uw = reinterpret_cast<const unsigned int*>(
        up_weight + (unsigned long long)out_idx * in_dim);
    const unsigned int g_sro = out_idx * gate_scale_cols;
    const unsigned int u_sro = out_idx * up_scale_cols;
    const unsigned int words_per_row = in_dim >> 2;

    float gate_acc[FP8_BATCH_TILE_X8] = {};
    float up_acc[FP8_BATCH_TILE_X8] = {};

    for (unsigned int w = lane << 1; w < words_per_row; w += TALU_QUANT_WARP_SIZE << 1) {
        unsigned int gw0, gw1;
        asm volatile("ld.global.cs.v2.u32 {%0, %1}, [%2];"
            : "=r"(gw0), "=r"(gw1) : "l"(&gw[w]));
        unsigned int uw0, uw1;
        asm volatile("ld.global.cs.v2.u32 {%0, %1}, [%2];"
            : "=r"(uw0), "=r"(uw1) : "l"(&uw[w]));

        const unsigned int base = w << 2;
        const float gs0 = e8m0_to_f32(gate_scales[g_sro + base / 32]);
        const float gs1 = e8m0_to_f32(gate_scales[g_sro + (base + 4) / 32]);
        const float us0 = e8m0_to_f32(up_scales[u_sro + base / 32]);
        const float us1 = e8m0_to_f32(up_scales[u_sro + (base + 4) / 32]);

        #pragma unroll
        for (unsigned int b = 0; b < FP8_BATCH_TILE_X8; b++) {
            if (b >= tile_rows) break;
            const unsigned long long row_off = (unsigned long long)b * in_dim;
            float r0, r1;

            const float4 inp0 = *reinterpret_cast<const float4*>(&input_tile[row_off + base]);
            r0 = 0.0f; r1 = 0.0f; FP8_PROCESS_WORD(gw0, inp0, r0, r1);
            gate_acc[b] = fmaf(r0 + r1, gs0, gate_acc[b]);
            r0 = 0.0f; r1 = 0.0f; FP8_PROCESS_WORD(uw0, inp0, r0, r1);
            up_acc[b] = fmaf(r0 + r1, us0, up_acc[b]);

            const float4 inp1 = *reinterpret_cast<const float4*>(&input_tile[row_off + base + 4]);
            r0 = 0.0f; r1 = 0.0f; FP8_PROCESS_WORD(gw1, inp1, r0, r1);
            gate_acc[b] = fmaf(r0 + r1, gs1, gate_acc[b]);
            r0 = 0.0f; r1 = 0.0f; FP8_PROCESS_WORD(uw1, inp1, r0, r1);
            up_acc[b] = fmaf(r0 + r1, us1, up_acc[b]);
        }
    }

    #pragma unroll
    for (unsigned int b = 0; b < FP8_BATCH_TILE_X8; b++) {
        if (b >= tile_rows) break;
        float g = talu_quant_warp_sum_f32(gate_acc[b]);
        float u = talu_quant_warp_sum_f32(up_acc[b]);
        if (lane == 0) {
            const float sigma = 1.0f / (1.0f + expf(-g));
            out[(unsigned long long)(batch_base + b) * out_dim + out_idx] = g * sigma * u;
        }
    }
}

// MXFP8 fused gate+up GEMV (separate outputs).
extern "C" __global__ __launch_bounds__(128, 2) void talu_mxfp8_matvec_gate_up_f32(
    const float* __restrict__ input,
    const unsigned char* __restrict__ gate_weight,
    const unsigned char* __restrict__ gate_scales,
    float* __restrict__ gate_out,
    unsigned int gate_out_dim,
    const unsigned char* __restrict__ up_weight,
    const unsigned char* __restrict__ up_scales,
    float* __restrict__ up_out,
    unsigned int up_out_dim,
    unsigned int in_dim,
    unsigned int gate_scale_cols,
    unsigned int up_scale_cols,
    unsigned int batch_rows
) {
    const unsigned int batch_base = blockIdx.y * FP8_BATCH_TILE;
    if (batch_base >= batch_rows) return;
    const unsigned int tile_rows = batch_rows - batch_base < FP8_BATCH_TILE
        ? batch_rows - batch_base : FP8_BATCH_TILE;

    const unsigned int warp_id = threadIdx.x / TALU_QUANT_WARP_SIZE;
    const unsigned int lane = threadIdx.x & (TALU_QUANT_WARP_SIZE - 1u);
    const unsigned int out_index = blockIdx.x * FP8_WARPS_PER_BLOCK + warp_id;
    const unsigned int total_dim = gate_out_dim + up_out_dim;
    if (out_index >= total_dim) return;

    const unsigned char* wt;
    const unsigned char* sc;
    float* out_ptr;
    unsigned int out_dim_proj, row_idx, sc_cols;

    if (out_index < gate_out_dim) {
        wt = gate_weight; sc = gate_scales; out_ptr = gate_out;
        out_dim_proj = gate_out_dim; row_idx = out_index; sc_cols = gate_scale_cols;
    } else {
        wt = up_weight; sc = up_scales; out_ptr = up_out;
        out_dim_proj = up_out_dim; row_idx = out_index - gate_out_dim; sc_cols = up_scale_cols;
    }

    const float* input_tile = input + (unsigned long long)batch_base * in_dim;
    float* out_tile = out_ptr + (unsigned long long)batch_base * out_dim_proj;
    const unsigned int* weight_words = reinterpret_cast<const unsigned int*>(
        wt + (unsigned long long)row_idx * in_dim);

    switch (tile_rows) {
        case 1u: talu_mxfp8_matvec_batched<1>(input_tile, weight_words, sc, out_tile, in_dim, out_dim_proj, row_idx, sc_cols, lane, 1u); break;
        case 2u: talu_mxfp8_matvec_batched<2>(input_tile, weight_words, sc, out_tile, in_dim, out_dim_proj, row_idx, sc_cols, lane, 2u); break;
        case 3u: talu_mxfp8_matvec_batched<3>(input_tile, weight_words, sc, out_tile, in_dim, out_dim_proj, row_idx, sc_cols, lane, 3u); break;
        case 4u: talu_mxfp8_matvec_batched<4>(input_tile, weight_words, sc, out_tile, in_dim, out_dim_proj, row_idx, sc_cols, lane, 4u); break;
        default: return;
    }
}

// MXFP8 fused gate+up GEMV (separate outputs): tile-8.
extern "C" __global__ __launch_bounds__(128, 2) void talu_mxfp8_matvec_gate_up_f32_tile8(
    const float* __restrict__ input,
    const unsigned char* __restrict__ gate_weight,
    const unsigned char* __restrict__ gate_scales,
    float* __restrict__ gate_out,
    unsigned int gate_out_dim,
    const unsigned char* __restrict__ up_weight,
    const unsigned char* __restrict__ up_scales,
    float* __restrict__ up_out,
    unsigned int up_out_dim,
    unsigned int in_dim,
    unsigned int gate_scale_cols,
    unsigned int up_scale_cols,
    unsigned int batch_rows
) {
    const unsigned int batch_base = blockIdx.y * FP8_BATCH_TILE_X8;
    if (batch_base >= batch_rows) return;
    const unsigned int tile_rows = batch_rows - batch_base < FP8_BATCH_TILE_X8
        ? batch_rows - batch_base : FP8_BATCH_TILE_X8;

    const unsigned int warp_id = threadIdx.x / TALU_QUANT_WARP_SIZE;
    const unsigned int lane = threadIdx.x & (TALU_QUANT_WARP_SIZE - 1u);
    const unsigned int out_index = blockIdx.x * FP8_WARPS_PER_BLOCK + warp_id;
    const unsigned int total_dim = gate_out_dim + up_out_dim;
    if (out_index >= total_dim) return;

    const unsigned char* wt;
    const unsigned char* sc;
    float* out_ptr;
    unsigned int out_dim_proj, row_idx, sc_cols;

    if (out_index < gate_out_dim) {
        wt = gate_weight; sc = gate_scales; out_ptr = gate_out;
        out_dim_proj = gate_out_dim; row_idx = out_index; sc_cols = gate_scale_cols;
    } else {
        wt = up_weight; sc = up_scales; out_ptr = up_out;
        out_dim_proj = up_out_dim; row_idx = out_index - gate_out_dim; sc_cols = up_scale_cols;
    }

    const float* input_tile = input + (unsigned long long)batch_base * in_dim;
    float* out_tile = out_ptr + (unsigned long long)batch_base * out_dim_proj;
    const unsigned int* weight_words = reinterpret_cast<const unsigned int*>(
        wt + (unsigned long long)row_idx * in_dim);

    switch (tile_rows) {
        case 1u: talu_mxfp8_matvec_batched<1>(input_tile, weight_words, sc, out_tile, in_dim, out_dim_proj, row_idx, sc_cols, lane, 1u); break;
        case 2u: talu_mxfp8_matvec_batched<2>(input_tile, weight_words, sc, out_tile, in_dim, out_dim_proj, row_idx, sc_cols, lane, 2u); break;
        case 3u: talu_mxfp8_matvec_batched<3>(input_tile, weight_words, sc, out_tile, in_dim, out_dim_proj, row_idx, sc_cols, lane, 3u); break;
        case 4u: talu_mxfp8_matvec_batched<4>(input_tile, weight_words, sc, out_tile, in_dim, out_dim_proj, row_idx, sc_cols, lane, 4u); break;
        case 5u: talu_mxfp8_matvec_batched<5>(input_tile, weight_words, sc, out_tile, in_dim, out_dim_proj, row_idx, sc_cols, lane, 5u); break;
        case 6u: talu_mxfp8_matvec_batched<6>(input_tile, weight_words, sc, out_tile, in_dim, out_dim_proj, row_idx, sc_cols, lane, 6u); break;
        case 7u: talu_mxfp8_matvec_batched<7>(input_tile, weight_words, sc, out_tile, in_dim, out_dim_proj, row_idx, sc_cols, lane, 7u); break;
        case 8u: talu_mxfp8_matvec_batched<8>(input_tile, weight_words, sc, out_tile, in_dim, out_dim_proj, row_idx, sc_cols, lane, 8u); break;
        default: return;
    }
}

#else

extern "C" __global__ void talu_dequant_mxfp8_to_bf16(
    const unsigned char* __restrict__ weight,
    const unsigned char* __restrict__ scales,
    unsigned short* __restrict__ out_bf16,
    unsigned int in_dim,
    unsigned int out_dim,
    unsigned int padded_outer,
    unsigned int padded_sf_k
) {}

extern "C" __global__ void talu_mxfp8_matvec_f32(
    const float* input, const unsigned char* weight, const unsigned char* scales,
    float* out, unsigned int in_dim, unsigned int out_dim, unsigned int scale_cols,
    unsigned int batch_rows) {}
extern "C" __global__ void talu_mxfp8_matvec_f32_tile8(
    const float* input, const unsigned char* weight, const unsigned char* scales,
    float* out, unsigned int in_dim, unsigned int out_dim, unsigned int scale_cols,
    unsigned int batch_rows) {}
extern "C" __global__ void talu_mxfp8_matvec_gate_up_silu_f32(
    const float* input, const unsigned char* gate_weight, const unsigned char* gate_scales,
    const unsigned char* up_weight, const unsigned char* up_scales, float* out,
    unsigned int out_dim, unsigned int in_dim, unsigned int gate_scale_cols,
    unsigned int up_scale_cols, unsigned int batch_rows) {}
extern "C" __global__ void talu_mxfp8_matvec_gate_up_silu_f32_tile8(
    const float* input, const unsigned char* gate_weight, const unsigned char* gate_scales,
    const unsigned char* up_weight, const unsigned char* up_scales, float* out,
    unsigned int out_dim, unsigned int in_dim, unsigned int gate_scale_cols,
    unsigned int up_scale_cols, unsigned int batch_rows) {}
extern "C" __global__ void talu_mxfp8_matvec_gate_up_f32(
    const float* input, const unsigned char* gate_weight, const unsigned char* gate_scales,
    float* gate_out, unsigned int gate_out_dim, const unsigned char* up_weight,
    const unsigned char* up_scales, float* up_out, unsigned int up_out_dim,
    unsigned int in_dim, unsigned int gate_scale_cols, unsigned int up_scale_cols,
    unsigned int batch_rows) {}
extern "C" __global__ void talu_mxfp8_matvec_gate_up_f32_tile8(
    const float* input, const unsigned char* gate_weight, const unsigned char* gate_scales,
    float* gate_out, unsigned int gate_out_dim, const unsigned char* up_weight,
    const unsigned char* up_scales, float* up_out, unsigned int up_out_dim,
    unsigned int in_dim, unsigned int gate_scale_cols, unsigned int up_scale_cols,
    unsigned int batch_rows) {}

#endif // __CUDA_ARCH__ >= 890
