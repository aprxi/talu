// INT8 quantized GEMV and quantization/dequantization kernels.

// Grid: (ceil(out_dim/4), batch_rows), Block: (128 = 4 warps)
static constexpr unsigned int I8_WARPS_PER_BLOCK = 4;
static constexpr unsigned int I8_WARP_ELEMS_PER_THREAD = 16;
static constexpr unsigned int I8_WARP_STEP = TALU_QUANT_WARP_SIZE * I8_WARP_ELEMS_PER_THREAD; // 512

extern "C" __global__ __launch_bounds__(128) void talu_i8_matvec_f32(
    const float* input,
    const signed char* weight,
    const float* weight_scales,
    float* out,
    unsigned int in_dim,
    unsigned int out_dim,
    unsigned int batch_rows,
    const float* residual
) {
    const unsigned int batch = blockIdx.y;
    if (batch >= batch_rows) return;

    const unsigned int warp_id = threadIdx.x / TALU_QUANT_WARP_SIZE;
    const unsigned int lane = threadIdx.x & (TALU_QUANT_WARP_SIZE - 1u);
    const unsigned int out_idx = blockIdx.x * I8_WARPS_PER_BLOCK + warp_id;
    if (out_idx >= out_dim) return;

    const float* input_row = input + (unsigned long long)batch * in_dim;
    float* out_row = out + (unsigned long long)batch * out_dim;
    const signed char* weight_row = weight + (unsigned long long)out_idx * in_dim;

    float acc = 0.0f;
    for (unsigned int i = lane * I8_WARP_ELEMS_PER_THREAD; i < in_dim; i += I8_WARP_STEP) {
        uint4 w4;
        const unsigned int* waddr = reinterpret_cast<const unsigned int*>(weight_row + i);
        asm volatile(
            "ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
            : "=r"(w4.x), "=r"(w4.y), "=r"(w4.z), "=r"(w4.w)
            : "l"(waddr));

        const float4 in0 = *reinterpret_cast<const float4*>(&input_row[i]);
        const float4 in1 = *reinterpret_cast<const float4*>(&input_row[i + 4]);
        const float4 in2 = *reinterpret_cast<const float4*>(&input_row[i + 8]);
        const float4 in3 = *reinterpret_cast<const float4*>(&input_row[i + 12]);

        #pragma unroll
        for (unsigned int j = 0; j < 4; ++j) {
            const unsigned int q32 = (j == 0) ? w4.x : (j == 1) ? w4.y : (j == 2) ? w4.z : w4.w;
            const float4 inv = (j == 0) ? in0 : (j == 1) ? in1 : (j == 2) ? in2 : in3;
            acc = fmaf(inv.x, static_cast<float>(static_cast<signed char>(q32 & 0xFFu)), acc);
            acc = fmaf(inv.y, static_cast<float>(static_cast<signed char>((q32 >> 8) & 0xFFu)), acc);
            acc = fmaf(inv.z, static_cast<float>(static_cast<signed char>((q32 >> 16) & 0xFFu)), acc);
            acc = fmaf(inv.w, static_cast<float>(static_cast<signed char>(q32 >> 24)), acc);
        }
    }

    // Warp-level reduction (no shared memory, no __syncthreads).
    acc = talu_quant_warp_sum_f32(acc);
    if (lane == 0) {
        const float result = acc * weight_scales[out_idx];
        out_row[out_idx] = residual ? result + residual[(unsigned long long)batch * out_dim + out_idx] : result;
    }
}

extern "C" __global__ void talu_quantize_f32_to_i8(
    const float* __restrict__ input,
    signed char* __restrict__ out_i8,
    float* __restrict__ out_row_scales,
    float* __restrict__ out_group_sums,
    unsigned int in_dim,
    unsigned int group_size
) {
    const unsigned int row = blockIdx.x;
    const float* row_in = input + (unsigned long long)row * in_dim;
    signed char* row_out = out_i8 + (unsigned long long)row * in_dim;
    const unsigned int num_groups = in_dim / group_size;

    // Phase 1: find per-row absmax via block reduction.
    __shared__ float smem[8]; // 256/32 = 8 warps
    float local_max = 0.0f;
    for (unsigned int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(row_in[i]));
    }
    // Warp reduce.
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

    // Phase 2: quantize to I8 and compute per-group sums.
    // Each thread handles a strided subset of elements.
    // Use shared memory for group sum accumulation.
    extern __shared__ float group_smem[]; // num_groups floats per warp → too complex
    // Simpler: each thread accumulates its own group sums, then warp/block reduce.
    // But num_groups can be up to ~80. Use a different approach:
    // Two-pass or combined pass.
    for (unsigned int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        float val = row_in[i];
        int q = __float2int_rn(val * inv_scale);
        q = max(-128, min(127, q));
        row_out[i] = static_cast<signed char>(q);
    }

    // Per-group sums: each thread accumulates sums for each group.
    // With group_size typically 128 and blockDim=256, 2 threads per group element.
    float* row_gsums = out_group_sums + (unsigned long long)row * num_groups;
    for (unsigned int g = 0; g < num_groups; g++) {
        float gsum = 0.0f;
        const unsigned int base = g * group_size;
        for (unsigned int j = threadIdx.x; j < group_size; j += blockDim.x) {
            gsum += row_in[base + j];
        }
        // Block reduce this group sum.
        gsum = talu_block_reduce_sum_128(gsum, smem);
        if (threadIdx.x == 0) row_gsums[g] = gsum;
        __syncthreads();
    }
}

// Dequantize I32 GEMM output to F32 with grouped-affine correction.
// gemm_i32[m][n] = sum_k(input_i8[m][k] * weight_i8[n][k])
//   where weight_i8 = weight_u8 - 128 (XOR 0x80).
//
// Reconstruct: y[m][n] = sum_g(scale[n][g] * group_dot_u8x[m][n][g]) + bias_term[m][n]
//
// Since we only have the TOTAL dot product (not per-group), we use:
//   y[m][n] ≈ mean_scale[n] * (gemm_i32[m][n] + 128 * i8_rowsum[m]) * row_scale[m]
//            + bias_term[m][n]
//
// Where bias_term[m][n] = sum_g(bias[n][g] * group_input_sum[m][g])
//
// Launch: grid=(out_dim), block=(256)
extern "C" __global__ void talu_dequant_i32_gaffine(
    const int* __restrict__ gemm_i32,       // [rows × out_dim]
    const float* __restrict__ row_scales,    // [rows]
    const int* __restrict__ i8_rowsums,      // [rows]
    const float* __restrict__ group_sums,    // [rows × num_groups]
    const unsigned short* __restrict__ scales, // [out_dim × num_groups]
    const unsigned short* __restrict__ biases, // [out_dim × num_groups]
    float* __restrict__ output,              // [rows × out_dim]
    unsigned int rows,
    unsigned int out_dim,
    unsigned int num_groups,
    unsigned int scales_dtype_tag
) {
    const unsigned int n = blockIdx.x; // output dimension index
    if (n >= out_dim) return;
    const unsigned short* scale_row = scales + (unsigned long long)n * num_groups;
    const unsigned short* bias_row = biases + (unsigned long long)n * num_groups;

    // Compute mean scale for this output row.
    float mean_scale = 0.0f;
    for (unsigned int g = 0; g < num_groups; g++) {
        mean_scale += talu_decode_scale_bias_u16(scale_row[g], scales_dtype_tag);
    }
    mean_scale /= static_cast<float>(num_groups);

    for (unsigned int m = threadIdx.x; m < rows; m += blockDim.x) {
        const float rs = row_scales[m];
        const int raw = gemm_i32[(unsigned long long)m * out_dim + n];
        const int i8sum = i8_rowsums[m];

        // Main term: mean_scale * (raw + 128 * i8_rowsum) * input_scale
        float result = mean_scale * static_cast<float>(raw + 128 * i8sum) * rs;

        // Bias correction: sum_g(bias[n][g] * group_input_sum[m][g])
        const float* m_gsums = group_sums + (unsigned long long)m * num_groups;
        for (unsigned int g = 0; g < num_groups; g++) {
            result += talu_decode_scale_bias_u16(bias_row[g], scales_dtype_tag) * m_gsums[g];
        }

        output[(unsigned long long)m * out_dim + n] = result;
    }
}

// Compute per-row I8 sums (for U8→I8 offset correction).
// Launch: grid=(rows), block=(256)
extern "C" __global__ void talu_i8_rowsum(
    const signed char* __restrict__ input_i8,
    int* __restrict__ out_rowsums,
    unsigned int in_dim
) {
    const unsigned int row = blockIdx.x;
    const signed char* row_in = input_i8 + (unsigned long long)row * in_dim;
    __shared__ float smem[8];
    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        sum += static_cast<float>(row_in[i]);
    }
    sum = talu_block_reduce_sum_128(sum, smem);
    if (threadIdx.x == 0) out_rowsums[row] = __float2int_rn(sum);
}

// Convert U8 weights to I8 by XOR'ing each byte with 0x80 (subtract 128).
// Launch: grid=ceil(num_words/256), block=(256)

// Launch: grid=(num_rows), block=(256)
extern "C" __global__ void talu_quantize_f16_to_i8(
    const unsigned short* __restrict__ weight_f16,  // [num_rows × in_dim] as raw __half bits
    signed char* __restrict__ out_i8,               // [num_rows × in_dim]
    float* __restrict__ out_row_scales,             // [num_rows]
    unsigned int in_dim
) {
    const unsigned int row = blockIdx.x;
    const unsigned short* row_in = weight_f16 + (unsigned long long)row * in_dim;
    signed char* row_out = out_i8 + (unsigned long long)row * in_dim;

    // Phase 1: find per-row absmax.
    __shared__ float smem[8];
    float local_max = 0.0f;
    for (unsigned int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(__half2float(*reinterpret_cast<const __half*>(&row_in[i]))));
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

    // Phase 2: quantize to I8.
    for (unsigned int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        float val = __half2float(*reinterpret_cast<const __half*>(&row_in[i]));
        int q = __float2int_rn(val * inv_scale);
        q = max(-128, min(127, q));
        row_out[i] = static_cast<signed char>(q);
    }
}

// Quantize F32 input rows to I8 with per-row absmax scaling (no group sums).
// Launch: grid=(rows), block=(256)
extern "C" __global__ void talu_quantize_f32_to_i8_simple(
    const float* __restrict__ input,        // [rows × in_dim]
    signed char* __restrict__ out_i8,       // [rows × in_dim]
    float* __restrict__ out_row_scales,     // [rows]
    unsigned int in_dim
) {
    const unsigned int row = blockIdx.x;
    const float* row_in = input + (unsigned long long)row * in_dim;
    signed char* row_out = out_i8 + (unsigned long long)row * in_dim;

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
    const float scale = (absmax > 0.0f) ? (absmax / 127.0f) : 1.0f;
    const float inv_scale = 1.0f / scale;
    if (threadIdx.x == 0) out_row_scales[row] = scale;

    for (unsigned int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        float val = row_in[i];
        int q = __float2int_rn(val * inv_scale);
        q = max(-128, min(127, q));
        row_out[i] = static_cast<signed char>(q);
    }
}

// Quantize F32 weights to grouped-affine U4 layout.
// Grid: (group_count, row_count), Block: (128)
