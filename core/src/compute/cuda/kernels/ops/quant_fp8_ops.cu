// FP8 E4M3 quantized GEMV, quantization, and dequantization kernels.

// ---------------------------------------------------------------------------
// FP8 E4M3 quantization / dequantization kernels (sm_89+)
// ---------------------------------------------------------------------------

#if __CUDA_ARCH__ >= 890

// Quantize F32 input rows to FP8 E4M3 with per-row absmax scaling.
// Launch: grid=(rows), block=(256)
extern "C" __global__ void talu_quantize_f32_to_fp8_e4m3(
    const float* __restrict__ input,        // [rows × in_dim]
    unsigned char* __restrict__ out_fp8,    // [rows × in_dim] FP8 E4M3 bytes
    float* __restrict__ out_row_scales,     // [rows]
    unsigned int in_dim
) {
    const unsigned int row = blockIdx.x;
    const float* row_in = input + (unsigned long long)row * in_dim;
    unsigned char* row_out = out_fp8 + (unsigned long long)row * in_dim;

    // Phase 1: per-row absmax via shared memory reduction.
    __shared__ float smem[8];
    float local_max = 0.0f;
    for (unsigned int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(row_in[i]));
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
    // E4M3 max representable value = 448.0
    const float scale = (absmax > 0.0f) ? (absmax / 448.0f) : 1.0f;
    const float inv_scale = 1.0f / scale;
    if (threadIdx.x == 0) out_row_scales[row] = scale;

    // Phase 2: quantize to FP8 E4M3.
    for (unsigned int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        float val = row_in[i] * inv_scale;
        row_out[i] = __nv_cvt_float_to_fp8(val, __NV_SATFINITE, __NV_E4M3);
    }
}

#else

// Stub for architectures below sm_89 — kernel symbol exists but should never be called.
extern "C" __global__ void talu_quantize_f32_to_fp8_e4m3(
    const float* __restrict__ input,
    void* __restrict__ out_fp8,
    float* __restrict__ out_row_scales,
    unsigned int in_dim
) {}

#endif // __CUDA_ARCH__ >= 890

#if __CUDA_ARCH__ >= 890

// Inner-batch FP8 GEMV: weight + scales loaded once from DRAM, reused across
// BATCH input rows from registers. BATCH=1 compiles to identical code as a
// single-row kernel. BATCH=2..4 adds one accumulator per extra row.
// Uses 64-bit streaming loads (v2.u32 = 8 FP8 bytes = 2 words per lane per iter).
template <unsigned int BATCH>
static __device__ __forceinline__ void talu_fp8_e4m3_matvec_batched(
    const float* input,
    const unsigned int* weight_words,
    const unsigned short* scales,
    float* out,
    unsigned int in_dim,
    unsigned int out_dim,
    unsigned int out_idx,
    unsigned int block_size,
    unsigned int scale_cols,
    unsigned int lane,
    unsigned int batch_rows
) {
    const unsigned int words_per_row = in_dim >> 2;
    const unsigned int scale_row_off = (out_idx / block_size) * scale_cols;

    float acc[BATCH] = {};

    for (unsigned int w = lane << 1; w < words_per_row; w += TALU_QUANT_WARP_SIZE << 1) {
        // 64-bit streaming load: 2 words = 8 FP8 bytes per lane per iteration.
        unsigned int w0, w1;
        asm volatile(
            "ld.global.cs.v2.u32 {%0, %1}, [%2];"
            : "=r"(w0), "=r"(w1)
            : "l"(&weight_words[w]));

        const unsigned int base_elem = w << 2;

        // Per-block BF16 scales.
        const float s0 = fp8_bf16_to_f32(scales[scale_row_off + base_elem / block_size]);
        const float s1 = fp8_bf16_to_f32(scales[scale_row_off + (base_elem + 4) / block_size]);

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

// Plain FP8 GEMV with batch tiling.
// Grid: (ceil(out_dim/4), ceil(batch_rows/FP8_BATCH_TILE)), Block: (128 = 4 warps)
extern "C" __global__ __launch_bounds__(128, 3) void talu_fp8_e4m3_matvec_f32(
    const float* __restrict__ input,                // [batch_rows × in_dim]
    const unsigned char* __restrict__ weight,        // [out_dim × in_dim] FP8 E4M3 bytes
    const unsigned short* __restrict__ scales,       // [scale_rows × scale_cols] BF16
    float* __restrict__ out,                         // [batch_rows × out_dim]
    unsigned int in_dim,
    unsigned int out_dim,
    unsigned int block_size,
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
        case 1u: talu_fp8_e4m3_matvec_batched<1>(input_tile, weight_words, scales, out_tile, in_dim, out_dim, out_idx, block_size, scale_cols, lane, 1u); break;
        case 2u: talu_fp8_e4m3_matvec_batched<2>(input_tile, weight_words, scales, out_tile, in_dim, out_dim, out_idx, block_size, scale_cols, lane, 2u); break;
        case 3u: talu_fp8_e4m3_matvec_batched<3>(input_tile, weight_words, scales, out_tile, in_dim, out_dim, out_idx, block_size, scale_cols, lane, 3u); break;
        case 4u: talu_fp8_e4m3_matvec_batched<4>(input_tile, weight_words, scales, out_tile, in_dim, out_dim, out_idx, block_size, scale_cols, lane, 4u); break;
        default: return;
    }
}

// Tile-8 variant: same as above but processes up to 8 rows per tile.
// Separate entry point to avoid register pressure overhead in the tile-4 kernel.
extern "C" __global__ __launch_bounds__(128, 2) void talu_fp8_e4m3_matvec_f32_tile8(
    const float* __restrict__ input,
    const unsigned char* __restrict__ weight,
    const unsigned short* __restrict__ scales,
    float* __restrict__ out,
    unsigned int in_dim,
    unsigned int out_dim,
    unsigned int block_size,
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
        case 1u: talu_fp8_e4m3_matvec_batched<1>(input_tile, weight_words, scales, out_tile, in_dim, out_dim, out_idx, block_size, scale_cols, lane, 1u); break;
        case 2u: talu_fp8_e4m3_matvec_batched<2>(input_tile, weight_words, scales, out_tile, in_dim, out_dim, out_idx, block_size, scale_cols, lane, 2u); break;
        case 3u: talu_fp8_e4m3_matvec_batched<3>(input_tile, weight_words, scales, out_tile, in_dim, out_dim, out_idx, block_size, scale_cols, lane, 3u); break;
        case 4u: talu_fp8_e4m3_matvec_batched<4>(input_tile, weight_words, scales, out_tile, in_dim, out_dim, out_idx, block_size, scale_cols, lane, 4u); break;
        case 5u: talu_fp8_e4m3_matvec_batched<5>(input_tile, weight_words, scales, out_tile, in_dim, out_dim, out_idx, block_size, scale_cols, lane, 5u); break;
        case 6u: talu_fp8_e4m3_matvec_batched<6>(input_tile, weight_words, scales, out_tile, in_dim, out_dim, out_idx, block_size, scale_cols, lane, 6u); break;
        case 7u: talu_fp8_e4m3_matvec_batched<7>(input_tile, weight_words, scales, out_tile, in_dim, out_dim, out_idx, block_size, scale_cols, lane, 7u); break;
        case 8u: talu_fp8_e4m3_matvec_batched<8>(input_tile, weight_words, scales, out_tile, in_dim, out_dim, out_idx, block_size, scale_cols, lane, 8u); break;
        default: return;
    }
}

// Fused FP8 gate+up+SiLU with batch tiling: each warp reads BOTH gate and up
// weights per iteration, reuses input from L2, fuses silu(gate)*up after reduction.
// Grid: (ceil(out_dim/4), ceil(batch_rows/FP8_BATCH_TILE)), Block: (128 = 4 warps)
extern "C" __global__ __launch_bounds__(128, 2) void talu_fp8_e4m3_matvec_gate_up_silu_f32(
    const float* __restrict__ input,           // [batch_rows × in_dim]
    const unsigned char* __restrict__ gate_weight, // [out_dim × in_dim] FP8
    const unsigned short* __restrict__ gate_scales,// [scale_rows × scale_cols] BF16
    const unsigned char* __restrict__ up_weight,   // [out_dim × in_dim] FP8
    const unsigned short* __restrict__ up_scales,  // [scale_rows × scale_cols] BF16
    float* __restrict__ out,                       // [batch_rows × out_dim]
    unsigned int out_dim,
    unsigned int in_dim,
    unsigned int block_size,
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
    const unsigned int g_sro = (out_idx / block_size) * gate_scale_cols;
    const unsigned int u_sro = (out_idx / block_size) * up_scale_cols;
    const unsigned int words_per_row = in_dim >> 2;

    float gate_acc[FP8_BATCH_TILE] = {};
    float up_acc[FP8_BATCH_TILE] = {};

    for (unsigned int w = lane << 1; w < words_per_row; w += TALU_QUANT_WARP_SIZE << 1) {
        // 64-bit loads for gate + up weights (2 DRAM reads per iteration).
        unsigned int gw0, gw1;
        asm volatile("ld.global.cs.v2.u32 {%0, %1}, [%2];"
            : "=r"(gw0), "=r"(gw1) : "l"(&gw[w]));
        unsigned int uw0, uw1;
        asm volatile("ld.global.cs.v2.u32 {%0, %1}, [%2];"
            : "=r"(uw0), "=r"(uw1) : "l"(&uw[w]));

        const unsigned int base = w << 2;
        const float gs0 = fp8_bf16_to_f32(gate_scales[g_sro + base / block_size]);
        const float gs1 = fp8_bf16_to_f32(gate_scales[g_sro + (base + 4) / block_size]);
        const float us0 = fp8_bf16_to_f32(up_scales[u_sro + base / block_size]);
        const float us1 = fp8_bf16_to_f32(up_scales[u_sro + (base + 4) / block_size]);

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

// Tile-8 variant of gate+up+SiLU.
extern "C" __global__ __launch_bounds__(128, 2) void talu_fp8_e4m3_matvec_gate_up_silu_f32_tile8(
    const float* __restrict__ input,
    const unsigned char* __restrict__ gate_weight,
    const unsigned short* __restrict__ gate_scales,
    const unsigned char* __restrict__ up_weight,
    const unsigned short* __restrict__ up_scales,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim,
    unsigned int block_size,
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
    const unsigned int g_sro = (out_idx / block_size) * gate_scale_cols;
    const unsigned int u_sro = (out_idx / block_size) * up_scale_cols;
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
        const float gs0 = fp8_bf16_to_f32(gate_scales[g_sro + base / block_size]);
        const float gs1 = fp8_bf16_to_f32(gate_scales[g_sro + (base + 4) / block_size]);
        const float us0 = fp8_bf16_to_f32(up_scales[u_sro + base / block_size]);
        const float us1 = fp8_bf16_to_f32(up_scales[u_sro + (base + 4) / block_size]);

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

// Fused FP8 gate+up (separate outputs, no SiLU) with batch tiling: warps handle
// either gate or up rows in a single grid.
// Grid: (ceil(total_dim/4), ceil(batch_rows/FP8_BATCH_TILE)), Block: (128 = 4 warps)
extern "C" __global__ __launch_bounds__(128, 2) void talu_fp8_e4m3_matvec_gate_up_f32(
    const float* __restrict__ input,
    const unsigned char* __restrict__ gate_weight,
    const unsigned short* __restrict__ gate_scales,
    float* __restrict__ gate_out,
    unsigned int gate_out_dim,
    const unsigned char* __restrict__ up_weight,
    const unsigned short* __restrict__ up_scales,
    float* __restrict__ up_out,
    unsigned int up_out_dim,
    unsigned int in_dim,
    unsigned int block_size,
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
    const unsigned short* sc;
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
        case 1u: talu_fp8_e4m3_matvec_batched<1>(input_tile, weight_words, sc, out_tile, in_dim, out_dim_proj, row_idx, block_size, sc_cols, lane, 1u); break;
        case 2u: talu_fp8_e4m3_matvec_batched<2>(input_tile, weight_words, sc, out_tile, in_dim, out_dim_proj, row_idx, block_size, sc_cols, lane, 2u); break;
        case 3u: talu_fp8_e4m3_matvec_batched<3>(input_tile, weight_words, sc, out_tile, in_dim, out_dim_proj, row_idx, block_size, sc_cols, lane, 3u); break;
        case 4u: talu_fp8_e4m3_matvec_batched<4>(input_tile, weight_words, sc, out_tile, in_dim, out_dim_proj, row_idx, block_size, sc_cols, lane, 4u); break;
        default: return;
    }
}

// Tile-8 variant of gate+up (separate outputs).
extern "C" __global__ __launch_bounds__(128, 2) void talu_fp8_e4m3_matvec_gate_up_f32_tile8(
    const float* __restrict__ input,
    const unsigned char* __restrict__ gate_weight,
    const unsigned short* __restrict__ gate_scales,
    float* __restrict__ gate_out,
    unsigned int gate_out_dim,
    const unsigned char* __restrict__ up_weight,
    const unsigned short* __restrict__ up_scales,
    float* __restrict__ up_out,
    unsigned int up_out_dim,
    unsigned int in_dim,
    unsigned int block_size,
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
    const unsigned short* sc;
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
        case 1u: talu_fp8_e4m3_matvec_batched<1>(input_tile, weight_words, sc, out_tile, in_dim, out_dim_proj, row_idx, block_size, sc_cols, lane, 1u); break;
        case 2u: talu_fp8_e4m3_matvec_batched<2>(input_tile, weight_words, sc, out_tile, in_dim, out_dim_proj, row_idx, block_size, sc_cols, lane, 2u); break;
        case 3u: talu_fp8_e4m3_matvec_batched<3>(input_tile, weight_words, sc, out_tile, in_dim, out_dim_proj, row_idx, block_size, sc_cols, lane, 3u); break;
        case 4u: talu_fp8_e4m3_matvec_batched<4>(input_tile, weight_words, sc, out_tile, in_dim, out_dim_proj, row_idx, block_size, sc_cols, lane, 4u); break;
        case 5u: talu_fp8_e4m3_matvec_batched<5>(input_tile, weight_words, sc, out_tile, in_dim, out_dim_proj, row_idx, block_size, sc_cols, lane, 5u); break;
        case 6u: talu_fp8_e4m3_matvec_batched<6>(input_tile, weight_words, sc, out_tile, in_dim, out_dim_proj, row_idx, block_size, sc_cols, lane, 6u); break;
        case 7u: talu_fp8_e4m3_matvec_batched<7>(input_tile, weight_words, sc, out_tile, in_dim, out_dim_proj, row_idx, block_size, sc_cols, lane, 7u); break;
        case 8u: talu_fp8_e4m3_matvec_batched<8>(input_tile, weight_words, sc, out_tile, in_dim, out_dim_proj, row_idx, block_size, sc_cols, lane, 8u); break;
        default: return;
    }
}

#else

extern "C" __global__ void talu_fp8_e4m3_matvec_f32(
    const float* __restrict__ input,
    const unsigned char* __restrict__ weight,
    const unsigned short* __restrict__ scales,
    float* __restrict__ out,
    unsigned int in_dim,
    unsigned int out_dim,
    unsigned int block_size,
    unsigned int scale_cols,
    unsigned int batch_rows
) {}

extern "C" __global__ void talu_fp8_e4m3_matvec_gate_up_silu_f32(
    const float* __restrict__ input,
    const unsigned char* __restrict__ gate_weight,
    const unsigned short* __restrict__ gate_scales,
    const unsigned char* __restrict__ up_weight,
    const unsigned short* __restrict__ up_scales,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim,
    unsigned int block_size,
    unsigned int gate_scale_cols,
    unsigned int up_scale_cols,
    unsigned int batch_rows
) {}

extern "C" __global__ void talu_fp8_e4m3_matvec_gate_up_f32(
    const float* __restrict__ input,
    const unsigned char* __restrict__ gate_weight,
    const unsigned short* __restrict__ gate_scales,
    float* __restrict__ gate_out,
    unsigned int gate_out_dim,
    const unsigned char* __restrict__ up_weight,
    const unsigned short* __restrict__ up_scales,
    float* __restrict__ up_out,
    unsigned int up_out_dim,
    unsigned int in_dim,
    unsigned int block_size,
    unsigned int gate_scale_cols,
    unsigned int up_scale_cols,
    unsigned int batch_rows
) {}

extern "C" __global__ void talu_fp8_e4m3_matvec_f32_tile8(
    const float* __restrict__ input,
    const unsigned char* __restrict__ weight,
    const unsigned short* __restrict__ scales,
    float* __restrict__ out,
    unsigned int in_dim,
    unsigned int out_dim,
    unsigned int block_size,
    unsigned int scale_cols,
    unsigned int batch_rows
) {}

extern "C" __global__ void talu_fp8_e4m3_matvec_gate_up_silu_f32_tile8(
    const float* __restrict__ input,
    const unsigned char* __restrict__ gate_weight,
    const unsigned short* __restrict__ gate_scales,
    const unsigned char* __restrict__ up_weight,
    const unsigned short* __restrict__ up_scales,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim,
    unsigned int block_size,
    unsigned int gate_scale_cols,
    unsigned int up_scale_cols,
    unsigned int batch_rows
) {}

extern "C" __global__ void talu_fp8_e4m3_matvec_gate_up_f32_tile8(
    const float* __restrict__ input,
    const unsigned char* __restrict__ gate_weight,
    const unsigned short* __restrict__ gate_scales,
    float* __restrict__ gate_out,
    unsigned int gate_out_dim,
    const unsigned char* __restrict__ up_weight,
    const unsigned short* __restrict__ up_scales,
    float* __restrict__ up_out,
    unsigned int up_out_dim,
    unsigned int in_dim,
    unsigned int block_size,
    unsigned int gate_scale_cols,
    unsigned int up_scale_cols,
    unsigned int batch_rows
) {}

#endif // __CUDA_ARCH__ >= 890

// ---------------------------------------------------------------------------
// FP8 E4M3 → BF16 dequantization with per-block scales (for prefill via cuBLAS)
// ---------------------------------------------------------------------------
// Converts FP8 E4M3 weights to BF16 by applying per-block BF16 scales.
// Launch: grid=(ceil(in_dim/256), out_dim), block=(256)

#if __CUDA_ARCH__ >= 890

extern "C" __global__ void talu_dequant_fp8_e4m3_to_bf16(
    const unsigned char* __restrict__ weight,   // [out_dim × in_dim] FP8 E4M3 bytes
    const unsigned short* __restrict__ scales,  // [scale_rows × scale_cols] BF16
    unsigned short* __restrict__ out_bf16,      // [out_dim × in_dim] BF16
    unsigned int in_dim,
    unsigned int out_dim,
    unsigned int block_size,
    unsigned int scale_cols
) {
    const unsigned int row = blockIdx.y;
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim || col >= in_dim) return;

    const unsigned long long idx = (unsigned long long)row * in_dim + col;
    const unsigned int scale_row = row / block_size;
    const unsigned int scale_col = col / block_size;
    const float block_scale = fp8_bf16_to_f32(scales[scale_row * scale_cols + scale_col]);

    // FP8 → F32 → scale → BF16
    const float val = fp8e4m3_to_f32(static_cast<__nv_fp8_storage_t>(weight[idx])) * block_scale;
    // F32 → BF16: round-to-nearest-even
    unsigned int fbits;
    memcpy(&fbits, &val, sizeof(fbits));
    const unsigned int lsb = (fbits >> 16) & 1u;
    const unsigned int rounding_bias = 0x7FFFu + lsb;
    out_bf16[idx] = static_cast<unsigned short>((fbits + rounding_bias) >> 16);
}

#else

extern "C" __global__ void talu_dequant_fp8_e4m3_to_bf16(
    const unsigned char* __restrict__ weight,
    const unsigned short* __restrict__ scales,
    unsigned short* __restrict__ out_bf16,
    unsigned int in_dim,
    unsigned int out_dim,
    unsigned int block_size,
    unsigned int scale_cols
) {}

#endif // __CUDA_ARCH__ >= 890

// Scale F32 output rows by per-row activation scales (FP8 dequant step).
// Launch: grid=(ceil(out_dim/256), rows), block=(256)
extern "C" __global__ void talu_scale_rows_f32(
    float* __restrict__ output,             // [rows × out_dim]
    const float* __restrict__ row_scales,   // [rows]
    unsigned int rows,
    unsigned int out_dim
) {
    const unsigned int m = blockIdx.y;
    const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= rows || n >= out_dim) return;
    const unsigned long long idx = (unsigned long long)m * out_dim + n;
    output[idx] *= row_scales[m];
}
