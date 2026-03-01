void* mlx_lazy_shortconv_mixer_bf16(
    const void* input,
    const void* in_proj,
    const void* conv_weight,
    const void* conv_bias,
    const void* out_proj,
    void* shortconv_cache_ptr,
    size_t layer_idx,
    size_t d_conv,
    size_t conv_dim
) {
    mlx_count_op();
    const auto& input_arr = *static_cast<const array*>(input);
    const auto& in_proj_arr = *static_cast<const array*>(in_proj);
    const auto& conv_weight_arr = *static_cast<const array*>(conv_weight);
    const auto& out_proj_arr = *static_cast<const array*>(out_proj);
    const auto* conv_bias_arr = static_cast<const array*>(conv_bias);

    const int seq_len = input_arr.shape(1);
    const int d_conv_i = static_cast<int>(d_conv);
    const int conv_dim_i = static_cast<int>(conv_dim);

    // in_proj is stored as [3*conv_dim, d_model], so transpose for [d_model, 3*conv_dim].
    array bcx = matmul(input_arr, transpose(in_proj_arr));
    // FP32 allowlist: shortconv recurrence is stateful; keep conv inputs in FP32
    // to avoid drift across long prefill/decode sequences.
    bcx = astype(bcx, float32);

    array b_gate = slice(bcx, {0, 0, 0}, {1, seq_len, conv_dim_i});
    array c_gate = slice(bcx, {0, 0, conv_dim_i}, {1, seq_len, 2 * conv_dim_i});
    array x_proj = slice(bcx, {0, 0, 2 * conv_dim_i}, {1, seq_len, 3 * conv_dim_i});
    array bx = b_gate * x_proj;

    array conv_kernel = conv_weight_arr;
    const auto ensure_f32 = [](const array& arr) -> array {
        // FP32 allowlist: depthwise conv kernel/state math remains in FP32.
        return arr.dtype() == float32 ? arr : astype(arr, float32);
    };
    array conv_kernel_broadcast(0.0f, float32);
    if (conv_kernel.ndim() == 3 &&
        conv_kernel.shape(0) == 1 &&
        conv_kernel.shape(1) == d_conv_i &&
        conv_kernel.shape(2) == conv_dim_i) {
        conv_kernel_broadcast = ensure_f32(conv_kernel);
    } else {
        if (conv_kernel.ndim() == 3) {
            // [conv_dim, 1, d_conv] -> [conv_dim, d_conv]
            conv_kernel = reshape(conv_kernel, {conv_kernel.shape(0), conv_kernel.shape(2)});
        }
        // [conv_dim, d_conv] -> [d_conv, conv_dim] for contiguous time-major access.
        array conv_kernel_t = ensure_f32(transpose(conv_kernel));
        conv_kernel_broadcast = reshape(conv_kernel_t, {1, d_conv_i, conv_dim_i});
    }

    std::optional<array> conv_bias_row;
    if (conv_bias_arr != nullptr) {
        const array conv_bias_f32 = ensure_f32(*conv_bias_arr);
        if (conv_bias_f32.ndim() == 2 && conv_bias_f32.shape(0) == 1 && conv_bias_f32.shape(1) == conv_dim_i) {
            conv_bias_row = conv_bias_f32;
        } else {
            conv_bias_row = reshape(conv_bias_f32, {1, conv_dim_i});
        }
    }

    auto* state_cache = static_cast<MLXShortConvCache*>(shortconv_cache_ptr);
    ShortConvLayer* layer_state = nullptr;
    if (state_cache != nullptr && layer_idx < state_cache->layers.size()) {
        layer_state = &state_cache->layers[layer_idx];
    }

    // FP32 allowlist: recurrent shortconv state accumulation.
    array conv_state(0.0f, float32);
    if (layer_state != nullptr) {
        const bool need_init =
            layer_state->conv_state == nullptr ||
            layer_state->conv_state->shape(1) != d_conv_i ||
            layer_state->conv_state->shape(2) != conv_dim_i;
        if (need_init) {
            delete layer_state->conv_state;
            layer_state->conv_state = new array(zeros({1, d_conv_i, conv_dim_i}, float32));
        }
        conv_state = *layer_state->conv_state;
    } else {
        conv_state = zeros({1, d_conv_i, conv_dim_i}, float32);
    }

    array conv_seq = shortconv_run_sequence_conv(
        &conv_state,
        bx,
        conv_kernel_broadcast,
        seq_len,
        d_conv_i,
        conv_dim_i
    );
    if (conv_bias_row) {
        conv_seq = conv_seq + reshape(*conv_bias_row, {1, 1, conv_dim_i});
    }

    const array out_proj_t = transpose(out_proj_arr);
    const array gated = conv_seq * c_gate;
    const array out = matmul(gated, out_proj_t);

    if (layer_state != nullptr) {
        *layer_state->conv_state = conv_state;
    }

    return pool_array(out);
}

void* mlx_lazy_shortconv_mixer_quantized(
    const void* input,
    const void* in_w,
    const void* in_s,
    const void* in_b,
    const void* conv_weight,
    const void* conv_bias,
    const void* out_w,
    const void* out_s,
    const void* out_b,
    size_t group_size,
    size_t bits,
    void* shortconv_cache_ptr,
    size_t layer_idx,
    size_t d_conv,
    size_t conv_dim
) {
    mlx_count_op();
    const auto& input_arr = *static_cast<const array*>(input);
    const auto& in_w_arr = *static_cast<const array*>(in_w);
    const auto& in_s_arr = *static_cast<const array*>(in_s);
    const auto& in_b_arr = *static_cast<const array*>(in_b);
    const auto& conv_weight_arr = *static_cast<const array*>(conv_weight);
    const auto* conv_bias_arr = static_cast<const array*>(conv_bias);
    const auto& out_w_arr = *static_cast<const array*>(out_w);
    const auto& out_s_arr = *static_cast<const array*>(out_s);
    const auto& out_b_arr = *static_cast<const array*>(out_b);

    const int seq_len = input_arr.shape(1);
    const int d_conv_i = static_cast<int>(d_conv);
    const int conv_dim_i = static_cast<int>(conv_dim);
    const int group_size_i = static_cast<int>(group_size);
    const int bits_i = static_cast<int>(bits);

    array bcx = quantized_matmul(
        input_arr,
        in_w_arr,
        in_s_arr,
        in_b_arr,
        true,
        group_size_i,
        bits_i,
        "affine"
    );
    // FP32 allowlist: shortconv recurrence is stateful; keep conv inputs in FP32
    // to avoid drift across long prefill/decode sequences.
    bcx = astype(bcx, float32);

    array b_gate = slice(bcx, {0, 0, 0}, {1, seq_len, conv_dim_i});
    array c_gate = slice(bcx, {0, 0, conv_dim_i}, {1, seq_len, 2 * conv_dim_i});
    array x_proj = slice(bcx, {0, 0, 2 * conv_dim_i}, {1, seq_len, 3 * conv_dim_i});
    array bx = b_gate * x_proj;

    array conv_kernel = conv_weight_arr;
    const auto ensure_f32 = [](const array& arr) -> array {
        // FP32 allowlist: depthwise conv kernel/state math remains in FP32.
        return arr.dtype() == float32 ? arr : astype(arr, float32);
    };
    array conv_kernel_broadcast(0.0f, float32);
    if (conv_kernel.ndim() == 3 &&
        conv_kernel.shape(0) == 1 &&
        conv_kernel.shape(1) == d_conv_i &&
        conv_kernel.shape(2) == conv_dim_i) {
        conv_kernel_broadcast = ensure_f32(conv_kernel);
    } else {
        if (conv_kernel.ndim() == 3) {
            conv_kernel = reshape(conv_kernel, {conv_kernel.shape(0), conv_kernel.shape(2)});
        }
        array conv_kernel_t = ensure_f32(transpose(conv_kernel));
        conv_kernel_broadcast = reshape(conv_kernel_t, {1, d_conv_i, conv_dim_i});
    }

    std::optional<array> conv_bias_row;
    if (conv_bias_arr != nullptr) {
        const array conv_bias_f32 = ensure_f32(*conv_bias_arr);
        if (conv_bias_f32.ndim() == 2 && conv_bias_f32.shape(0) == 1 && conv_bias_f32.shape(1) == conv_dim_i) {
            conv_bias_row = conv_bias_f32;
        } else {
            conv_bias_row = reshape(conv_bias_f32, {1, conv_dim_i});
        }
    }

    auto* state_cache = static_cast<MLXShortConvCache*>(shortconv_cache_ptr);
    ShortConvLayer* layer_state = nullptr;
    if (state_cache != nullptr && layer_idx < state_cache->layers.size()) {
        layer_state = &state_cache->layers[layer_idx];
    }

    // FP32 allowlist: recurrent shortconv state accumulation.
    array conv_state(0.0f, float32);
    if (layer_state != nullptr) {
        const bool need_init =
            layer_state->conv_state == nullptr ||
            layer_state->conv_state->shape(1) != d_conv_i ||
            layer_state->conv_state->shape(2) != conv_dim_i;
        if (need_init) {
            delete layer_state->conv_state;
            layer_state->conv_state = new array(zeros({1, d_conv_i, conv_dim_i}, float32));
        }
        conv_state = *layer_state->conv_state;
    } else {
        conv_state = zeros({1, d_conv_i, conv_dim_i}, float32);
    }

    array conv_seq = shortconv_run_sequence_conv(
        &conv_state,
        bx,
        conv_kernel_broadcast,
        seq_len,
        d_conv_i,
        conv_dim_i
    );
    if (conv_bias_row) {
        conv_seq = conv_seq + reshape(*conv_bias_row, {1, 1, conv_dim_i});
    }

    const array gated = conv_seq * c_gate;
    const array out = quantized_matmul(
        gated,
        out_w_arr,
        out_s_arr,
        out_b_arr,
        true,
        group_size_i,
        bits_i,
        "affine"
    );

    if (layer_state != nullptr) {
        *layer_state->conv_state = conv_state;
    }

    return pool_array(out);
}
