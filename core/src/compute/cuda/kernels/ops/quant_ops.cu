static constexpr unsigned int TALU_QUANT_WARP_SIZE = 32;

static __device__ __forceinline__ float talu_quant_warp_sum_f32(float value) {
    value += __shfl_down_sync(0xFFFFFFFFu, value, 16);
    value += __shfl_down_sync(0xFFFFFFFFu, value, 8);
    value += __shfl_down_sync(0xFFFFFFFFu, value, 4);
    value += __shfl_down_sync(0xFFFFFFFFu, value, 2);
    value += __shfl_down_sync(0xFFFFFFFFu, value, 1);
    return value;
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

extern "C" __global__ void talu_embedding_lookup_gaffine_u4_f32(
    float* out,
    const unsigned int* packed_vals,
    const unsigned short* scales,
    const unsigned short* biases,
    unsigned int vocab_size,
    unsigned int hidden_dim,
    unsigned int token,
    unsigned int group_size,
    unsigned int scales_dtype_tag,
    float multiplier
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= hidden_dim) return;
    if (token >= vocab_size) return;
    if (group_size == 0 || (hidden_dim % group_size) != 0 || (group_size % 8) != 0) return;

    const unsigned int groups_per_row = hidden_dim / group_size;
    const unsigned int words_per_group = group_size / 8;
    const unsigned int words_per_row = hidden_dim / 8;
    const unsigned int group_idx = index / group_size;
    const unsigned int in_group = index - group_idx * group_size;
    const unsigned int word_in_group = in_group / 8;
    const unsigned int nibble_in_word = in_group & 7;

    const unsigned int row_word_base = token * words_per_row;
    const unsigned int packed_word = packed_vals[row_word_base + group_idx * words_per_group + word_in_group];
    const unsigned int quant = (packed_word >> (nibble_in_word * 4)) & 0xF;
    const unsigned int sb_index = token * groups_per_row + group_idx;
    const float scale = talu_decode_scale_bias_u16(scales[sb_index], scales_dtype_tag);
    const float bias = talu_decode_scale_bias_u16(biases[sb_index], scales_dtype_tag);
    out[index] = (static_cast<float>(quant) * scale + bias) * multiplier;
}

extern "C" __global__ void talu_gaffine_u4_matvec_f32(
    const float* input,
    const unsigned int* packed_weight,
    const unsigned short* scales,
    const unsigned short* biases,
    float* out,
    unsigned int in_dim,
    unsigned int out_dim,
    unsigned int group_size,
    unsigned int scales_dtype_tag,
    unsigned int batch_rows
) {
    const unsigned int batch = blockIdx.y;
    const unsigned int lane = threadIdx.x & (TALU_QUANT_WARP_SIZE - 1u);
    const unsigned int warp_id = threadIdx.x / TALU_QUANT_WARP_SIZE;
    const unsigned int warps_per_block = blockDim.x / TALU_QUANT_WARP_SIZE;
    const unsigned int out_idx = blockIdx.x * warps_per_block + warp_id;
    if (batch >= batch_rows) return;
    if (out_idx >= out_dim) return;
    if (group_size == 0 || (in_dim % group_size) != 0 || (group_size % 8) != 0) {
        return;
    }
    const float* input_row = input + (unsigned long long)batch * in_dim;
    float* out_row = out + (unsigned long long)batch * out_dim;

    const unsigned int groups_per_row = in_dim / group_size;
    const unsigned int words_per_group = group_size / 8;
    const unsigned int words_per_row = in_dim / 8;
    const unsigned int row_word_base = out_idx * words_per_row;
    const unsigned int row_sb_base = out_idx * groups_per_row;

    float acc = 0.0f;
    for (unsigned int i = lane; i < in_dim; i += TALU_QUANT_WARP_SIZE) {
        const unsigned int group_idx = i / group_size;
        const unsigned int in_group = i - group_idx * group_size;
        const unsigned int word_in_group = in_group / 8;
        const unsigned int nib_in_word = in_group & 7;
        const unsigned int packed_word = packed_weight[row_word_base + group_idx * words_per_group + word_in_group];
        const unsigned int quant = (packed_word >> (nib_in_word * 4)) & 0xF;
        const float scale = talu_decode_scale_bias_u16(scales[row_sb_base + group_idx], scales_dtype_tag);
        const float bias = talu_decode_scale_bias_u16(biases[row_sb_base + group_idx], scales_dtype_tag);
        const float dequant = static_cast<float>(quant) * scale + bias;
        acc = fmaf(input_row[i], dequant, acc);
    }
    acc = talu_quant_warp_sum_f32(acc);

    if (lane == 0) out_row[out_idx] = acc;
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
    unsigned int batch_rows
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

    if (threadIdx.x == 0) out_row[out_idx] = acc;
}

static __device__ __forceinline__ float talu_gaffine_u4_dot_row_warp(
    const float* input,
    const unsigned int* packed_weight,
    const unsigned short* scales,
    const unsigned short* biases,
    unsigned int in_dim,
    unsigned int out_idx,
    unsigned int group_size,
    unsigned int scales_dtype_tag,
    unsigned int lane
) {
    const unsigned int groups_per_row = in_dim / group_size;
    const unsigned int words_per_group = group_size / 8;
    const unsigned int row_word_base = out_idx * (in_dim / 8);
    const unsigned int row_sb_base = out_idx * groups_per_row;

    float acc = 0.0f;
    for (unsigned int i = lane; i < in_dim; i += TALU_QUANT_WARP_SIZE) {
        const unsigned int group_idx = i / group_size;
        const unsigned int in_group = i - group_idx * group_size;
        const unsigned int word_in_group = in_group / 8;
        const unsigned int nib_in_word = in_group & 7;
        const unsigned int packed_word = packed_weight[row_word_base + group_idx * words_per_group + word_in_group];
        const unsigned int quant = (packed_word >> (nib_in_word * 4)) & 0xF;
        const float scale = talu_decode_scale_bias_u16(scales[row_sb_base + group_idx], scales_dtype_tag);
        const float bias = talu_decode_scale_bias_u16(biases[row_sb_base + group_idx], scales_dtype_tag);
        const float dequant = static_cast<float>(quant) * scale + bias;
        acc = fmaf(input[i], dequant, acc);
    }
    return talu_quant_warp_sum_f32(acc);
}

extern "C" __global__ void talu_gaffine_u4_matvec_qkv_f32(
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
    if (in_dim == 0 || (in_dim % 8) != 0) return;
    const float* input_row = input + (unsigned long long)batch * in_dim;
    float* q_out_row = q_out + (unsigned long long)batch * q_out_dim;
    float* k_out_row = k_out + (unsigned long long)batch * k_out_dim;
    float* v_out_row = v_out + (unsigned long long)batch * v_out_dim;

    const unsigned int lane = threadIdx.x & (TALU_QUANT_WARP_SIZE - 1u);
    const unsigned int warp_id = threadIdx.x / TALU_QUANT_WARP_SIZE;
    const unsigned int warps_per_block = blockDim.x / TALU_QUANT_WARP_SIZE;
    const unsigned int out_index = blockIdx.x * warps_per_block + warp_id;
    const unsigned int qk_dim = q_out_dim + k_out_dim;
    const unsigned int total_dim = qk_dim + v_out_dim;
    if (out_index >= total_dim) return;

    float acc = 0.0f;
    if (out_index < q_out_dim) {
        if (q_group_size == 0 || (in_dim % q_group_size) != 0 || (q_group_size % 8) != 0) return;
        acc = talu_gaffine_u4_dot_row_warp(
            input_row,
            q_packed_weight,
            q_scales,
            q_biases,
            in_dim,
            out_index,
            q_group_size,
            q_scales_dtype_tag,
            lane
        );
        if (lane == 0) q_out_row[out_index] = acc;
        return;
    }

    if (out_index < qk_dim) {
        if (k_group_size == 0 || (in_dim % k_group_size) != 0 || (k_group_size % 8) != 0) return;
        const unsigned int k_row = out_index - q_out_dim;
        acc = talu_gaffine_u4_dot_row_warp(
            input_row,
            k_packed_weight,
            k_scales,
            k_biases,
            in_dim,
            k_row,
            k_group_size,
            k_scales_dtype_tag,
            lane
        );
        if (lane == 0) k_out_row[k_row] = acc;
        return;
    }

    if (v_group_size == 0 || (in_dim % v_group_size) != 0 || (v_group_size % 8) != 0) return;
    const unsigned int v_row = out_index - qk_dim;
    acc = talu_gaffine_u4_dot_row_warp(
        input_row,
        v_packed_weight,
        v_scales,
        v_biases,
        in_dim,
        v_row,
        v_group_size,
        v_scales_dtype_tag,
        lane
    );
    if (lane == 0) v_out_row[v_row] = acc;
}

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

extern "C" __global__ void talu_gaffine_u4_matvec_gate_up_f32(
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
    if (in_dim == 0 || (in_dim % 8) != 0) return;
    const float* input_row = input + (unsigned long long)batch * in_dim;
    float* gate_out_row = gate_out + (unsigned long long)batch * gate_out_dim;
    float* up_out_row = up_out + (unsigned long long)batch * up_out_dim;

    const unsigned int lane = threadIdx.x & (TALU_QUANT_WARP_SIZE - 1u);
    const unsigned int warp_id = threadIdx.x / TALU_QUANT_WARP_SIZE;
    const unsigned int warps_per_block = blockDim.x / TALU_QUANT_WARP_SIZE;
    const unsigned int out_index = blockIdx.x * warps_per_block + warp_id;
    const unsigned int total_dim = gate_out_dim + up_out_dim;
    if (out_index >= total_dim) return;

    float acc = 0.0f;
    if (out_index < gate_out_dim) {
        if (gate_group_size == 0 || (in_dim % gate_group_size) != 0 || (gate_group_size % 8) != 0) return;
        acc = talu_gaffine_u4_dot_row_warp(
            input_row,
            gate_packed_weight,
            gate_scales,
            gate_biases,
            in_dim,
            out_index,
            gate_group_size,
            gate_scales_dtype_tag,
            lane
        );
        if (lane == 0) gate_out_row[out_index] = acc;
        return;
    }

    if (up_group_size == 0 || (in_dim % up_group_size) != 0 || (up_group_size % 8) != 0) return;
    const unsigned int up_row = out_index - gate_out_dim;
    acc = talu_gaffine_u4_dot_row_warp(
        input_row,
        up_packed_weight,
        up_scales,
        up_biases,
        in_dim,
        up_row,
        up_group_size,
        up_scales_dtype_tag,
        lane
    );
    if (lane == 0) up_out_row[up_row] = acc;
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
    const unsigned int* packed_weight,
    const unsigned short* scales,
    const unsigned short* biases,
    unsigned short* out_f16,
    unsigned int out_dim,
    unsigned int in_dim,
    unsigned int group_size,
    unsigned int scales_dtype_tag
) {
    const unsigned int row = blockIdx.x;
    if (row >= out_dim) return;
    if (group_size == 0 || (in_dim % group_size) != 0 || (in_dim % 4) != 0 || (group_size % 4) != 0) return;

    const unsigned int words_per_row = in_dim / 4;
    const unsigned int groups_per_row = in_dim / group_size;
    const unsigned int* weight_row = packed_weight + (unsigned long long)row * words_per_row;
    const unsigned short* scale_row = scales + (unsigned long long)row * groups_per_row;
    const unsigned short* bias_row = biases + (unsigned long long)row * groups_per_row;
    unsigned short* out_row = out_f16 + (unsigned long long)row * in_dim;

    for (unsigned int i = threadIdx.x * 4; i < in_dim; i += blockDim.x * 4) {
        const unsigned int packed = weight_row[i >> 2];

        const unsigned int group_idx = i / group_size;
        const float scale = talu_decode_scale_bias_u16(scale_row[group_idx], scales_dtype_tag);
        const float bias = talu_decode_scale_bias_u16(bias_row[group_idx], scales_dtype_tag);

        const float v0 = static_cast<float>(packed & 0xFFu) * scale + bias;
        const float v1 = static_cast<float>((packed >> 8) & 0xFFu) * scale + bias;
        const float v2 = static_cast<float>((packed >> 16) & 0xFFu) * scale + bias;
        const float v3 = static_cast<float>(packed >> 24) * scale + bias;

        __half2 h01 = __floats2half2_rn(v0, v1);
        __half2 h23 = __floats2half2_rn(v2, v3);
        *reinterpret_cast<__half2*>(&out_row[i]) = h01;
        *reinterpret_cast<__half2*>(&out_row[i + 2]) = h23;
    }
}
