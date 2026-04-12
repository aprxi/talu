// Shared helpers for quantized GEMV kernels. Included from quant_*_ops.cu files.

static constexpr unsigned int TALU_QUANT_WARP_SIZE = 32;

static __device__ __forceinline__ float talu_quant_warp_sum_f32(float value) {
    value += __shfl_down_sync(0xFFFFFFFFu, value, 16);
    value += __shfl_down_sync(0xFFFFFFFFu, value, 8);
    value += __shfl_down_sync(0xFFFFFFFFu, value, 4);
    value += __shfl_down_sync(0xFFFFFFFFu, value, 2);
    value += __shfl_down_sync(0xFFFFFFFFu, value, 1);
    return value;
}

static __device__ __forceinline__ float talu_quant_warp_min_f32(float value) {
    value = fminf(value, __shfl_down_sync(0xFFFFFFFFu, value, 16));
    value = fminf(value, __shfl_down_sync(0xFFFFFFFFu, value, 8));
    value = fminf(value, __shfl_down_sync(0xFFFFFFFFu, value, 4));
    value = fminf(value, __shfl_down_sync(0xFFFFFFFFu, value, 2));
    value = fminf(value, __shfl_down_sync(0xFFFFFFFFu, value, 1));
    return value;
}

static __device__ __forceinline__ float talu_quant_warp_max_f32(float value) {
    value = fmaxf(value, __shfl_down_sync(0xFFFFFFFFu, value, 16));
    value = fmaxf(value, __shfl_down_sync(0xFFFFFFFFu, value, 8));
    value = fmaxf(value, __shfl_down_sync(0xFFFFFFFFu, value, 4));
    value = fmaxf(value, __shfl_down_sync(0xFFFFFFFFu, value, 2));
    value = fmaxf(value, __shfl_down_sync(0xFFFFFFFFu, value, 1));
    return value;
}

static __device__ __forceinline__ unsigned short talu_quant_f32_to_bf16_rne(float value) {
    // Truncation (matching CPU f32ToBf16: bits >> 16).
    // Must match the CPU converter path to produce bit-identical model files
    // regardless of whether CUDA or CPU quantization is used.
    const unsigned int fbits = __float_as_uint(value);
    return static_cast<unsigned short>(fbits >> 16);
}

static constexpr unsigned int U8_BLOCK_THREADS = 128;
static constexpr unsigned int U8_ELEMS_PER_THREAD = 16;
static constexpr unsigned int U8_ELEMS_PER_STEP = U8_BLOCK_THREADS * U8_ELEMS_PER_THREAD;

static __device__ __forceinline__ float talu_block_reduce_sum_128(float val, float* smem) {
    const unsigned int warp_id = threadIdx.x / TALU_QUANT_WARP_SIZE;
    const unsigned int lane = threadIdx.x & (TALU_QUANT_WARP_SIZE - 1u);
    val = talu_quant_warp_sum_f32(val);
    if (lane == 0) smem[warp_id] = val;
    __syncthreads();
    if (threadIdx.x < 4) {
        val = smem[threadIdx.x];
        val += __shfl_down_sync(0xFu, val, 2);
        val += __shfl_down_sync(0xFu, val, 1);
    }
    return val;
}

// ---------------------------------------------------------------------------
// FP8 E4M3 per-block GEMV kernel
// ---------------------------------------------------------------------------

// FP8 E4M3 matrix-vector product with per-block BF16 scales.
// Warp-per-row design: 4 warps per block, each computes one output row.
// Weight layout: [out_dim, in_dim] in FP8 E4M3 (1 byte per element).
// Scales layout: [scale_rows, scale_cols] in BF16, where
//   scale_rows = out_dim / block_size, scale_cols = in_dim / block_size.
// FP8 E4M3 GEMV with per-block [128,128] BF16 scales.
// Each block_size×block_size tile of weights shares one BF16 scale.
//
// Optimizations:
// - 128-bit streaming weight loads (v4.u32 = 16 FP8 bytes per lane per iter)
// - Batched template: weight loaded ONCE from DRAM, reused across BATCH input rows
// - Factored accumulation with 2-way ILP (r0/r1 per 4-element word)
// - Grid: (ceil(out_dim/4), ceil(batch_rows/8)), Block: (128 = 4 warps)
static constexpr unsigned int FP8_WARPS_PER_BLOCK = 4;
static constexpr unsigned int FP8_BATCH_TILE = 4;
static constexpr unsigned int FP8_BATCH_TILE_X8 = 8;

#if __CUDA_ARCH__ >= 890

// Convert FP8 E4M3 byte to float via half intermediate.
static __device__ __forceinline__ float fp8e4m3_to_f32(__nv_fp8_storage_t x) {
    __half_raw hr = __nv_cvt_fp8_to_halfraw(x, __NV_E4M3);
    __half h;
    memcpy(&h, &hr, sizeof(h));
    return __half2float(h);
}

// Convert BF16 (stored as unsigned short) to float.
static __device__ __forceinline__ float fp8_bf16_to_f32(unsigned short raw) {
    return __uint_as_float(static_cast<unsigned int>(raw) << 16);
}

// Process one word (4 FP8 bytes) against float4 input with 2-way ILP.
// Accumulates raw products to r0/r1, caller applies scale once per word.
#define FP8_PROCESS_WORD(word, inp, r0, r1) do { \
    (r0) = fmaf((inp).x, fp8e4m3_to_f32(static_cast<__nv_fp8_storage_t>((word) & 0xFFu)), (r0)); \
    (r1) = fmaf((inp).y, fp8e4m3_to_f32(static_cast<__nv_fp8_storage_t>(((word) >> 8) & 0xFFu)), (r1)); \
    (r0) = fmaf((inp).z, fp8e4m3_to_f32(static_cast<__nv_fp8_storage_t>(((word) >> 16) & 0xFFu)), (r0)); \
    (r1) = fmaf((inp).w, fp8e4m3_to_f32(static_cast<__nv_fp8_storage_t>((word) >> 24)), (r1)); \
} while (0)

#endif // __CUDA_ARCH__ >= 890
