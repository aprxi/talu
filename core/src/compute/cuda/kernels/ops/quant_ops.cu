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
    unsigned int scales_dtype_tag
) {
    const unsigned int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= out_dim) return;
    if (group_size == 0 || (in_dim % group_size) != 0 || (group_size % 8) != 0) {
        return;
    }

    const unsigned int groups_per_row = in_dim / group_size;
    const unsigned int words_per_group = group_size / 8;
    const unsigned int words_per_row = in_dim / 8;
    const unsigned int row_word_base = out_idx * words_per_row;
    const unsigned int row_sb_base = out_idx * groups_per_row;

    float acc = 0.0f;
    for (unsigned int group_idx = 0; group_idx < groups_per_row; ++group_idx) {
        const float scale = talu_decode_scale_bias_u16(scales[row_sb_base + group_idx], scales_dtype_tag);
        const float bias = talu_decode_scale_bias_u16(biases[row_sb_base + group_idx], scales_dtype_tag);
        const unsigned int group_input_base = group_idx * group_size;
        const unsigned int group_word_base = row_word_base + group_idx * words_per_group;

        for (unsigned int w = 0; w < words_per_group; ++w) {
            const unsigned int packed = packed_weight[group_word_base + w];
            const unsigned int value_base = group_input_base + w * 8;

            #pragma unroll
            for (unsigned int nib = 0; nib < 8; ++nib) {
                const unsigned int quant = (packed >> (nib * 4)) & 0xF;
                const float dequant = static_cast<float>(quant) * scale + bias;
                acc += input[value_base + nib] * dequant;
            }
        }
    }

    out[out_idx] = acc;
}

static __device__ __forceinline__ float talu_gaffine_u4_dot_row(
    const float* input,
    const unsigned int* packed_weight,
    const unsigned short* scales,
    const unsigned short* biases,
    unsigned int in_dim,
    unsigned int out_idx,
    unsigned int group_size,
    unsigned int scales_dtype_tag
) {
    const unsigned int groups_per_row = in_dim / group_size;
    const unsigned int words_per_group = group_size / 8;
    const unsigned int words_per_row = in_dim / 8;
    const unsigned int row_word_base = out_idx * words_per_row;
    const unsigned int row_sb_base = out_idx * groups_per_row;

    float acc = 0.0f;
    for (unsigned int group_idx = 0; group_idx < groups_per_row; ++group_idx) {
        const float scale = talu_decode_scale_bias_u16(scales[row_sb_base + group_idx], scales_dtype_tag);
        const float bias = talu_decode_scale_bias_u16(biases[row_sb_base + group_idx], scales_dtype_tag);
        const unsigned int group_input_base = group_idx * group_size;
        const unsigned int group_word_base = row_word_base + group_idx * words_per_group;

        for (unsigned int w = 0; w < words_per_group; ++w) {
            const unsigned int packed = packed_weight[group_word_base + w];
            const unsigned int value_base = group_input_base + w * 8;

            #pragma unroll
            for (unsigned int nib = 0; nib < 8; ++nib) {
                const unsigned int quant = (packed >> (nib * 4)) & 0xF;
                const float dequant = static_cast<float>(quant) * scale + bias;
                acc += input[value_base + nib] * dequant;
            }
        }
    }
    return acc;
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
    unsigned int in_dim
) {
    if (in_dim == 0 || (in_dim % 8) != 0) return;

    const unsigned int out_index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int qk_dim = q_out_dim + k_out_dim;
    const unsigned int total_dim = qk_dim + v_out_dim;
    if (out_index >= total_dim) return;

    if (out_index < q_out_dim) {
        if (q_group_size == 0 || (in_dim % q_group_size) != 0 || (q_group_size % 8) != 0) return;
        q_out[out_index] = talu_gaffine_u4_dot_row(
            input,
            q_packed_weight,
            q_scales,
            q_biases,
            in_dim,
            out_index,
            q_group_size,
            q_scales_dtype_tag
        );
        return;
    }

    if (out_index < qk_dim) {
        if (k_group_size == 0 || (in_dim % k_group_size) != 0 || (k_group_size % 8) != 0) return;
        const unsigned int k_row = out_index - q_out_dim;
        k_out[k_row] = talu_gaffine_u4_dot_row(
            input,
            k_packed_weight,
            k_scales,
            k_biases,
            in_dim,
            k_row,
            k_group_size,
            k_scales_dtype_tag
        );
        return;
    }

    if (v_group_size == 0 || (in_dim % v_group_size) != 0 || (v_group_size % 8) != 0) return;
    const unsigned int v_row = out_index - qk_dim;
    v_out[v_row] = talu_gaffine_u4_dot_row(
        input,
        v_packed_weight,
        v_scales,
        v_biases,
        in_dim,
        v_row,
        v_group_size,
        v_scales_dtype_tag
    );
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
    unsigned int in_dim
) {
    if (in_dim == 0 || (in_dim % 8) != 0) return;

    const unsigned int out_index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total_dim = gate_out_dim + up_out_dim;
    if (out_index >= total_dim) return;

    if (out_index < gate_out_dim) {
        if (gate_group_size == 0 || (in_dim % gate_group_size) != 0 || (gate_group_size % 8) != 0) return;
        gate_out[out_index] = talu_gaffine_u4_dot_row(
            input,
            gate_packed_weight,
            gate_scales,
            gate_biases,
            in_dim,
            out_index,
            gate_group_size,
            gate_scales_dtype_tag
        );
        return;
    }

    if (up_group_size == 0 || (in_dim % up_group_size) != 0 || (up_group_size % 8) != 0) return;
    const unsigned int up_row = out_index - gate_out_dim;
    up_out[up_row] = talu_gaffine_u4_dot_row(
        input,
        up_packed_weight,
        up_scales,
        up_biases,
        in_dim,
        up_row,
        up_group_size,
        up_scales_dtype_tag
    );
}
