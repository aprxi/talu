// Grouped-affine U8 quantized GEMV, dequantization, and converter kernels.

static __device__ __forceinline__ float talu_gaffine_u8_partial_dot_block(
    const float* input,
    const unsigned int* packed_weight,
    const unsigned short* scales,
    const unsigned short* biases,
    unsigned int in_dim,
    unsigned int out_idx,
    unsigned int group_size,
    unsigned int scales_dtype_tag
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int words_per_row = in_dim / 4;
    const unsigned int groups_per_row = in_dim / group_size;
    const unsigned int* weight_row = packed_weight + (unsigned long long)out_idx * words_per_row;
    const unsigned short* scale_row = scales + (unsigned long long)out_idx * groups_per_row;
    const unsigned short* bias_row = biases + (unsigned long long)out_idx * groups_per_row;

    float acc = 0.0f;
    for (unsigned int i = tid * U8_ELEMS_PER_THREAD; i < in_dim; i += U8_ELEMS_PER_STEP) {
        uint4 w4;
        const unsigned int* waddr = weight_row + (i >> 2);
        asm volatile(
            "ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
            : "=r"(w4.x), "=r"(w4.y), "=r"(w4.z), "=r"(w4.w)
            : "l"(waddr));

        const float4 in0 = *reinterpret_cast<const float4*>(&input[i]);
        const float4 in1 = *reinterpret_cast<const float4*>(&input[i + 4]);
        const float4 in2 = *reinterpret_cast<const float4*>(&input[i + 8]);
        const float4 in3 = *reinterpret_cast<const float4*>(&input[i + 12]);

        const unsigned int group_idx = i / group_size;
        const float scale = talu_decode_scale_bias_u16(scale_row[group_idx], scales_dtype_tag);
        const float bias = talu_decode_scale_bias_u16(bias_row[group_idx], scales_dtype_tag);

        #pragma unroll
        for (unsigned int j = 0; j < 4; ++j) {
            const unsigned int q32 = (j == 0) ? w4.x : (j == 1) ? w4.y : (j == 2) ? w4.z : w4.w;
            const float4 inv = (j == 0) ? in0 : (j == 1) ? in1 : (j == 2) ? in2 : in3;
            acc = fmaf(inv.x, static_cast<float>(q32 & 0xFFu) * scale + bias, acc);
            acc = fmaf(inv.y, static_cast<float>((q32 >> 8) & 0xFFu) * scale + bias, acc);
            acc = fmaf(inv.z, static_cast<float>((q32 >> 16) & 0xFFu) * scale + bias, acc);
            acc = fmaf(inv.w, static_cast<float>(q32 >> 24) * scale + bias, acc);
        }
    }
    return acc;
}

extern "C" __global__ void talu_gaffine_u8_matvec_f32(
    const float* input,
    const unsigned int* packed_weight,
    const unsigned short* scales,
    const unsigned short* biases,
    float* out,
    unsigned int in_dim,
    unsigned int out_dim,
    unsigned int group_size,
    unsigned int scales_dtype_tag,
    unsigned int batch_rows,
    const float* residual
) {
    const unsigned int batch = blockIdx.y;
    const unsigned int out_idx = blockIdx.x;
    if (batch >= batch_rows) return;
    if (out_idx >= out_dim) return;
    if (group_size == 0 || (in_dim % group_size) != 0 ||
        (in_dim % 16) != 0 || (group_size % 16) != 0) {
        return;
    }
    const float* input_row = input + (unsigned long long)batch * in_dim;
    float* out_row = out + (unsigned long long)batch * out_dim;

    __shared__ float smem[4];
    const float acc = talu_block_reduce_sum_128(
        talu_gaffine_u8_partial_dot_block(
            input_row, packed_weight, scales, biases,
            in_dim, out_idx, group_size, scales_dtype_tag),
        smem);

    if (threadIdx.x == 0) out_row[out_idx] = residual ? acc + residual[(unsigned long long)batch * out_dim + out_idx] : acc;
}

// Symmetric I8 GEMV: warp-per-row design with 4 output rows per block.
// Each warp (32 threads) computes one output row's dot product independently.
// No __syncthreads needed — only warp-level shuffles for reduction.
// For in_dim=2560: 32 threads × 16 elems = 512 per step → 5 exact steps (no tail waste).
// Grid: (ceil(out_dim/4), batch_rows), Block: (128 = 4 warps)

extern "C" __global__ void talu_gaffine_u8_matvec_qkv_f32(
    const float* input,
    const unsigned int* q_packed_weight,
    const unsigned short* q_scales,
    const unsigned short* q_biases,
    float* q_out,
    unsigned int q_out_dim,
    unsigned int q_group_size,
    unsigned int q_scales_dtype_tag,
    const unsigned int* k_packed_weight,
    const unsigned short* k_scales,
    const unsigned short* k_biases,
    float* k_out,
    unsigned int k_out_dim,
    unsigned int k_group_size,
    unsigned int k_scales_dtype_tag,
    const unsigned int* v_packed_weight,
    const unsigned short* v_scales,
    const unsigned short* v_biases,
    float* v_out,
    unsigned int v_out_dim,
    unsigned int v_group_size,
    unsigned int v_scales_dtype_tag,
    unsigned int in_dim,
    unsigned int batch_rows
) {
    const unsigned int batch = blockIdx.y;
    if (batch >= batch_rows) return;
    if (in_dim == 0 || (in_dim % 16) != 0) return;
    const float* input_row = input + (unsigned long long)batch * in_dim;
    float* q_out_row = q_out + (unsigned long long)batch * q_out_dim;
    float* k_out_row = k_out + (unsigned long long)batch * k_out_dim;
    float* v_out_row = v_out + (unsigned long long)batch * v_out_dim;

    const unsigned int out_index = blockIdx.x;
    const unsigned int qk_dim = q_out_dim + k_out_dim;
    const unsigned int total_dim = qk_dim + v_out_dim;
    if (out_index >= total_dim) return;

    __shared__ float smem[4];

    if (out_index < q_out_dim) {
        if (q_group_size == 0 || (in_dim % q_group_size) != 0 || (q_group_size % 16) != 0) return;
        const float acc = talu_block_reduce_sum_128(
            talu_gaffine_u8_partial_dot_block(
                input_row, q_packed_weight, q_scales, q_biases,
                in_dim, out_index, q_group_size, q_scales_dtype_tag),
            smem);
        if (threadIdx.x == 0) q_out_row[out_index] = acc;
        return;
    }

    if (out_index < qk_dim) {
        if (k_group_size == 0 || (in_dim % k_group_size) != 0 || (k_group_size % 16) != 0) return;
        const unsigned int k_row = out_index - q_out_dim;
        const float acc = talu_block_reduce_sum_128(
            talu_gaffine_u8_partial_dot_block(
                input_row, k_packed_weight, k_scales, k_biases,
                in_dim, k_row, k_group_size, k_scales_dtype_tag),
            smem);
        if (threadIdx.x == 0) k_out_row[k_row] = acc;
        return;
    }

    if (v_group_size == 0 || (in_dim % v_group_size) != 0 || (v_group_size % 16) != 0) return;
    const unsigned int v_row = out_index - qk_dim;
    const float acc = talu_block_reduce_sum_128(
        talu_gaffine_u8_partial_dot_block(
            input_row, v_packed_weight, v_scales, v_biases,
            in_dim, v_row, v_group_size, v_scales_dtype_tag),
        smem);
    if (threadIdx.x == 0) v_out_row[v_row] = acc;
}

extern "C" __global__ void talu_gaffine_u8_matvec_gate_up_f32(
    const float* input,
    const unsigned int* gate_packed_weight,
    const unsigned short* gate_scales,
    const unsigned short* gate_biases,
    float* gate_out,
    unsigned int gate_out_dim,
    unsigned int gate_group_size,
    unsigned int gate_scales_dtype_tag,
    const unsigned int* up_packed_weight,
    const unsigned short* up_scales,
    const unsigned short* up_biases,
    float* up_out,
    unsigned int up_out_dim,
    unsigned int up_group_size,
    unsigned int up_scales_dtype_tag,
    unsigned int in_dim,
    unsigned int batch_rows
) {
    const unsigned int batch = blockIdx.y;
    if (batch >= batch_rows) return;
    if (in_dim == 0 || (in_dim % 16) != 0) return;
    const float* input_row = input + (unsigned long long)batch * in_dim;
    float* gate_out_row = gate_out + (unsigned long long)batch * gate_out_dim;
    float* up_out_row = up_out + (unsigned long long)batch * up_out_dim;

    const unsigned int out_index = blockIdx.x;
    const unsigned int total_dim = gate_out_dim + up_out_dim;
    if (out_index >= total_dim) return;

    __shared__ float smem[4];

    if (out_index < gate_out_dim) {
        if (gate_group_size == 0 || (in_dim % gate_group_size) != 0 || (gate_group_size % 16) != 0) return;
        const float acc = talu_block_reduce_sum_128(
            talu_gaffine_u8_partial_dot_block(
                input_row, gate_packed_weight, gate_scales, gate_biases,
                in_dim, out_index, gate_group_size, gate_scales_dtype_tag),
            smem);
        if (threadIdx.x == 0) gate_out_row[out_index] = acc;
        return;
    }

    if (up_group_size == 0 || (in_dim % up_group_size) != 0 || (up_group_size % 16) != 0) return;
    const unsigned int up_row = out_index - gate_out_dim;
    const float acc = talu_block_reduce_sum_128(
        talu_gaffine_u8_partial_dot_block(
            input_row, up_packed_weight, up_scales, up_biases,
            in_dim, up_row, up_group_size, up_scales_dtype_tag),
        smem);
    if (threadIdx.x == 0) up_out_row[up_row] = acc;
}

extern "C" __global__ void talu_gaffine_u8_matvec_gate_up_silu_f32(
    const float* input,
    const unsigned int* gate_packed_weight,
    const unsigned short* gate_scales,
    const unsigned short* gate_biases,
    const unsigned int* up_packed_weight,
    const unsigned short* up_scales,
    const unsigned short* up_biases,
    float* out,
    unsigned int out_dim,
    unsigned int gate_group_size,
    unsigned int gate_scales_dtype_tag,
    unsigned int up_group_size,
    unsigned int up_scales_dtype_tag,
    unsigned int in_dim,
    unsigned int batch_rows
) {
    const unsigned int batch = blockIdx.y;
    if (batch >= batch_rows) return;
    if (in_dim == 0 || (in_dim % 16) != 0) return;
    const float* input_row = input + (unsigned long long)batch * in_dim;
    float* out_row = out + (unsigned long long)batch * out_dim;

    const unsigned int out_index = blockIdx.x;
    if (out_index >= out_dim) return;
    if (gate_group_size == 0 || (in_dim % gate_group_size) != 0 || (gate_group_size % 16) != 0) return;
    if (up_group_size == 0 || (in_dim % up_group_size) != 0 || (up_group_size % 16) != 0) return;

    __shared__ float smem[4];

    const float gate_acc = talu_block_reduce_sum_128(
        talu_gaffine_u8_partial_dot_block(
            input_row, gate_packed_weight, gate_scales, gate_biases,
            in_dim, out_index, gate_group_size, gate_scales_dtype_tag),
        smem);

    float gate_result = 0.0f;
    if (threadIdx.x == 0) gate_result = gate_acc;

    const float up_acc = talu_block_reduce_sum_128(
        talu_gaffine_u8_partial_dot_block(
            input_row, up_packed_weight, up_scales, up_biases,
            in_dim, out_index, up_group_size, up_scales_dtype_tag),
        smem);

    if (threadIdx.x == 0) {
        const float sigma = 1.0f / (1.0f + expf(-gate_result));
        out_row[out_index] = gate_result * sigma * up_acc;
    }
}

extern "C" __global__ void talu_gaffine_u8_dequantize_to_f16(
    const unsigned int* __restrict__ packed_weight,
    const unsigned short* __restrict__ scales,
    const unsigned short* __restrict__ biases,
    unsigned short* __restrict__ out_f16,
    unsigned int out_dim,
    unsigned int in_dim,
    unsigned int group_size,
    unsigned int scales_dtype_tag
) {
    const unsigned int row = blockIdx.x;
    if (row >= out_dim) return;

    const unsigned int words_per_row = in_dim >> 2;
    const unsigned int groups_per_row = in_dim / group_size;
    const unsigned int* weight_row = packed_weight + (unsigned long long)row * words_per_row;
    const unsigned short* scale_row = scales + (unsigned long long)row * groups_per_row;
    const unsigned short* bias_row = biases + (unsigned long long)row * groups_per_row;
    unsigned short* out_row = out_f16 + (unsigned long long)row * in_dim;

    // Process 16 U8 values per thread via 128-bit loads and stores.
    for (unsigned int i = threadIdx.x * 16; i < in_dim; i += blockDim.x * 16) {
        // 128-bit streaming load: 4 uint32 words = 16 packed U8 values.
        uint4 p;
        asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(p.x), "=r"(p.y), "=r"(p.z), "=r"(p.w)
            : "l"(&weight_row[i >> 2]));

        float4 store0, store1;
        unsigned int g;
        float s, b;

        // Elements i..i+3 (word 0)
        g = i / group_size;
        s = talu_decode_scale_bias_u16(scale_row[g], scales_dtype_tag);
        b = talu_decode_scale_bias_u16(bias_row[g], scales_dtype_tag);
        reinterpret_cast<__half2*>(&store0)[0] = __floats2half2_rn(
            static_cast<float>(p.x & 0xFFu) * s + b,
            static_cast<float>((p.x >> 8) & 0xFFu) * s + b);
        reinterpret_cast<__half2*>(&store0)[1] = __floats2half2_rn(
            static_cast<float>((p.x >> 16) & 0xFFu) * s + b,
            static_cast<float>(p.x >> 24) * s + b);

        // Elements i+4..i+7 (word 1)
        g = (i + 4) / group_size;
        s = talu_decode_scale_bias_u16(scale_row[g], scales_dtype_tag);
        b = talu_decode_scale_bias_u16(bias_row[g], scales_dtype_tag);
        reinterpret_cast<__half2*>(&store0)[2] = __floats2half2_rn(
            static_cast<float>(p.y & 0xFFu) * s + b,
            static_cast<float>((p.y >> 8) & 0xFFu) * s + b);
        reinterpret_cast<__half2*>(&store0)[3] = __floats2half2_rn(
            static_cast<float>((p.y >> 16) & 0xFFu) * s + b,
            static_cast<float>(p.y >> 24) * s + b);

        // 128-bit store for first 8 F16 values.
        *reinterpret_cast<float4*>(&out_row[i]) = store0;

        // Elements i+8..i+11 (word 2)
        g = (i + 8) / group_size;
        s = talu_decode_scale_bias_u16(scale_row[g], scales_dtype_tag);
        b = talu_decode_scale_bias_u16(bias_row[g], scales_dtype_tag);
        reinterpret_cast<__half2*>(&store1)[0] = __floats2half2_rn(
            static_cast<float>(p.z & 0xFFu) * s + b,
            static_cast<float>((p.z >> 8) & 0xFFu) * s + b);
        reinterpret_cast<__half2*>(&store1)[1] = __floats2half2_rn(
            static_cast<float>((p.z >> 16) & 0xFFu) * s + b,
            static_cast<float>(p.z >> 24) * s + b);

        // Elements i+12..i+15 (word 3)
        g = (i + 12) / group_size;
        s = talu_decode_scale_bias_u16(scale_row[g], scales_dtype_tag);
        b = talu_decode_scale_bias_u16(bias_row[g], scales_dtype_tag);
        reinterpret_cast<__half2*>(&store1)[2] = __floats2half2_rn(
            static_cast<float>(p.w & 0xFFu) * s + b,
            static_cast<float>((p.w >> 8) & 0xFFu) * s + b);
        reinterpret_cast<__half2*>(&store1)[3] = __floats2half2_rn(
            static_cast<float>((p.w >> 16) & 0xFFu) * s + b,
            static_cast<float>(p.w >> 24) * s + b);

        // 128-bit store for next 8 F16 values.
        *reinterpret_cast<float4*>(&out_row[i + 8]) = store1;
    }
}

// Launch: grid=ceil(num_words/256), block=(256)
extern "C" __global__ void talu_u8_xor_to_i8(
    const unsigned int* __restrict__ in,
    unsigned int* __restrict__ out,
    unsigned int num_words
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_words) out[idx] = in[idx] ^ 0x80808080u;
}

// ─── Symmetric INT8 GEMM support ───

// Fused grouped-affine U8 dequant → I8 requant with per-row absmax scaling.
// Reads packed U8 weights + per-group scales/biases, dequantizes in registers,
// finds per-row absmax, and requantizes to I8. No F16 intermediate needed.
// Launch: grid=(out_dim), block=(256)
extern "C" __global__ void talu_gaffine_u8_to_i8(
    const unsigned char* __restrict__ packed_u8,    // [out_dim × in_dim]
    const unsigned short* __restrict__ scales,      // [out_dim × num_groups]
    const unsigned short* __restrict__ biases,      // [out_dim × num_groups]
    signed char* __restrict__ out_i8,               // [out_dim × in_dim]
    float* __restrict__ out_row_scales,             // [out_dim]
    unsigned int in_dim,
    unsigned int group_size,
    unsigned int scales_dtype_tag
) {
    const unsigned int row = blockIdx.x;
    const unsigned char* row_u8 = packed_u8 + (unsigned long long)row * in_dim;
    signed char* row_out = out_i8 + (unsigned long long)row * in_dim;
    const unsigned int num_groups = in_dim / group_size;
    const unsigned short* scale_row = scales + (unsigned long long)row * num_groups;
    const unsigned short* bias_row = biases + (unsigned long long)row * num_groups;

    // Pass 1: dequant to F32 in registers and find per-row absmax.
    __shared__ float smem[8];
    float local_max = 0.0f;
    for (unsigned int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        const unsigned int g = i / group_size;
        const float s = talu_decode_scale_bias_u16(scale_row[g], scales_dtype_tag);
        const float b = talu_decode_scale_bias_u16(bias_row[g], scales_dtype_tag);
        const float val = s * static_cast<float>(row_u8[i]) + b;
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
        const unsigned int g = i / group_size;
        const float s = talu_decode_scale_bias_u16(scale_row[g], scales_dtype_tag);
        const float b = talu_decode_scale_bias_u16(bias_row[g], scales_dtype_tag);
        const float val = s * static_cast<float>(row_u8[i]) + b;
        int q = __float2int_rn(val * inv_scale);
        q = max(-128, min(127, q));
        row_out[i] = static_cast<signed char>(q);
    }
}

// Grid: (group_count, row_count), Block: (128)
extern "C" __global__ void talu_gaffine_quantize_u8_f32(
    const float* __restrict__ input,                // [row_count × col_count]
    const float* __restrict__ group_scale_factors,  // [group_count]
    const float* __restrict__ group_bias_shifts,    // [group_count]
    const float* __restrict__ group_round_shifts,   // [group_count]
    unsigned int* __restrict__ packed_out,          // [row_count × packed_col_count]
    unsigned short* __restrict__ scales_out,        // [row_count × group_count] BF16
    unsigned short* __restrict__ biases_out,        // [row_count × group_count] BF16
    unsigned int row_count,
    unsigned int col_count,
    unsigned int group_size,
    unsigned int packed_col_count
) {
    if (row_count == 0u || col_count == 0u || group_size == 0u) return;
    if ((col_count % group_size) != 0u || (group_size % 4u) != 0u) return;

    const unsigned int row = blockIdx.y;
    const unsigned int group_idx = blockIdx.x;
    const unsigned int group_count = col_count / group_size;
    if (row >= row_count || group_idx >= group_count) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & (TALU_QUANT_WARP_SIZE - 1u);
    const unsigned int warp_id = tid / TALU_QUANT_WARP_SIZE;
    const unsigned int warp_count = (blockDim.x + TALU_QUANT_WARP_SIZE - 1u) / TALU_QUANT_WARP_SIZE;

    const float* row_in = input + (unsigned long long)row * col_count;
    const unsigned int group_start = group_idx * group_size;
    const unsigned int words_per_group = group_size / 4u;

    float local_min = 3.402823466e+38F;
    float local_max = -3.402823466e+38F;
    for (unsigned int i = tid; i < group_size; i += blockDim.x) {
        const float value = row_in[group_start + i];
        local_min = fminf(local_min, value);
        local_max = fmaxf(local_max, value);
    }

    local_min = talu_quant_warp_min_f32(local_min);
    local_max = talu_quant_warp_max_f32(local_max);

    __shared__ float warp_min[8];
    __shared__ float warp_max[8];
    __shared__ float shared_scale;
    __shared__ float shared_bias;
    __shared__ float shared_round_shift;
    if (lane == 0u) {
        warp_min[warp_id] = local_min;
        warp_max[warp_id] = local_max;
    }
    __syncthreads();

    if (warp_id == 0u) {
        float block_min = (lane < warp_count) ? warp_min[lane] : 3.402823466e+38F;
        float block_max = (lane < warp_count) ? warp_max[lane] : -3.402823466e+38F;
        block_min = talu_quant_warp_min_f32(block_min);
        block_max = talu_quant_warp_max_f32(block_max);
        if (lane == 0u) {
            const float base_scale = (block_max > block_min) ? ((block_max - block_min) / 255.0f) : 0.0f;
            const float scale_factor = group_scale_factors[group_idx];
            const float group_scale = base_scale * scale_factor;
            const float group_bias_shift = group_bias_shifts[group_idx];
            const float group_bias = (group_scale > 0.0f)
                ? (block_min + group_bias_shift * group_scale)
                : block_min;
            const float group_round_shift = group_round_shifts[group_idx];

            shared_scale = group_scale;
            shared_bias = group_bias;
            shared_round_shift = group_round_shift;

            const unsigned long long sb_index = (unsigned long long)row * group_count + group_idx;
            scales_out[sb_index] = talu_quant_f32_to_bf16_rne(group_scale);
            biases_out[sb_index] = talu_quant_f32_to_bf16_rne(group_bias);
        }
    }
    __syncthreads();

    const float group_scale = shared_scale;
    const float group_bias = shared_bias;
    const float group_round_shift = shared_round_shift;

    const unsigned int group_word_offset = group_start / 4u;
    const unsigned long long row_word_base = (unsigned long long)row * packed_col_count + group_word_offset;
    for (unsigned int word_idx = tid; word_idx < words_per_group; word_idx += blockDim.x) {
        const unsigned int value_base = group_start + word_idx * 4u;
        unsigned int packed_word = 0u;

        #pragma unroll
        for (unsigned int value_idx = 0; value_idx < 4u; ++value_idx) {
            const float value = row_in[value_base + value_idx];
            unsigned int quantized = 0u;
            if (group_scale > 0.0f) {
                const float normalized = (value - group_bias) / group_scale + group_round_shift;
                float rounded = roundf(normalized);
                rounded = fminf(255.0f, fmaxf(0.0f, rounded));
                quantized = static_cast<unsigned int>(rounded);
            }
            packed_word |= (quantized << (value_idx * 8u));
        }

        packed_out[row_word_base + word_idx] = packed_word;
    }
}

// Build dequantized grouped-affine weights for calibration scoring.
// Inputs are sampled source rows + per-row/group min/base-scale statistics.
// Output layout matches matmul weight expectation: [col_count × sample_rows] (col-major).
// Grid: (ceil(col_count/256), sample_rows), Block: (256)
