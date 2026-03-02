void* mlx_lazy_causal_conv_mixer_bf16(
    const void* input,
    const void* in_proj,
    const void* conv_weight,
    const void* conv_bias,
    const void* out_proj,
    void* causal_conv_cache_ptr,
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
    const int hidden_dim = input_arr.shape(2);

    const array in_proj_rhs = orient_matmul_rhs_cached(in_proj, in_proj_arr, hidden_dim, 3 * conv_dim_i, true);
    const array bcx = matmul_lastdim(input_arr, in_proj_rhs);
    // Keep recurrence update in FP32, but preserve model dtype for gating/output
    // to avoid unnecessary FP32 residency and per-token recasts in dense decode.
    array b_gate = slice(bcx, {0, 0, 0}, {1, seq_len, conv_dim_i});
    const array c_gate = slice(bcx, {0, 0, conv_dim_i}, {1, seq_len, 2 * conv_dim_i});
    array x_proj = slice(bcx, {0, 0, 2 * conv_dim_i}, {1, seq_len, 3 * conv_dim_i});
    if (b_gate.dtype() != float32) b_gate = astype(b_gate, float32);
    if (x_proj.dtype() != float32) x_proj = astype(x_proj, float32);
    // FP32 allowlist: causal_conv recurrence is stateful; keep conv stream update
    // in FP32 while preserving bf16/f16 for output projection.
    array bx = b_gate * x_proj;

    array conv_kernel_broadcast = resolve_causal_conv_kernel(conv_weight, conv_weight_arr, d_conv_i, conv_dim_i);

    std::optional<array> conv_bias_row = resolve_causal_conv_bias_row(conv_bias, conv_bias_arr, conv_dim_i);

    auto* state_cache = static_cast<MLXCausalConvCache*>(causal_conv_cache_ptr);
    CausalConvLayer* layer_state = nullptr;
    if (state_cache != nullptr && layer_idx < state_cache->layers.size()) {
        layer_state = &state_cache->layers[layer_idx];
    }

    // FP32 allowlist: recurrent causal_conv state accumulation.
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

    array conv_seq = causal_conv_run_sequence(
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

    const array conv_seq_model = (conv_seq.dtype() == c_gate.dtype()) ? conv_seq : astype(conv_seq, c_gate.dtype());
    const array gated = conv_seq_model * c_gate;
    const array out_proj_rhs = orient_matmul_rhs_cached(out_proj, out_proj_arr, conv_dim_i, hidden_dim, true);
    const array out = matmul_lastdim(gated, out_proj_rhs);

    if (layer_state != nullptr) {
        *layer_state->conv_state = conv_state;
    }

    return pool_array(out);
}

void* mlx_lazy_causal_conv_mixer_quantized(
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
    void* causal_conv_cache_ptr,
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

    array bcx = quantized_linear_no_bias(
        input_arr,
        in_w_arr,
        in_s_arr,
        in_b_arr,
        group_size_i,
        bits_i,
        in_s,
        in_b
    );
    // FP32 allowlist: causal_conv recurrence is stateful; keep conv inputs in FP32
    // to avoid drift across long prefill/decode sequences.
    bcx = astype(bcx, float32);

    array b_gate = slice(bcx, {0, 0, 0}, {1, seq_len, conv_dim_i});
    array c_gate = slice(bcx, {0, 0, conv_dim_i}, {1, seq_len, 2 * conv_dim_i});
    array x_proj = slice(bcx, {0, 0, 2 * conv_dim_i}, {1, seq_len, 3 * conv_dim_i});
    array bx = b_gate * x_proj;

    array conv_kernel_broadcast = resolve_causal_conv_kernel(conv_weight, conv_weight_arr, d_conv_i, conv_dim_i);

    std::optional<array> conv_bias_row = resolve_causal_conv_bias_row(conv_bias, conv_bias_arr, conv_dim_i);

    auto* state_cache = static_cast<MLXCausalConvCache*>(causal_conv_cache_ptr);
    CausalConvLayer* layer_state = nullptr;
    if (state_cache != nullptr && layer_idx < state_cache->layers.size()) {
        layer_state = &state_cache->layers[layer_idx];
    }

    // FP32 allowlist: recurrent causal_conv state accumulation.
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

    array conv_seq = causal_conv_run_sequence(
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
    const array out = quantized_linear_no_bias(
        gated,
        out_w_arr,
        out_s_arr,
        out_b_arr,
        group_size_i,
        bits_i,
        out_s,
        out_b
    );

    if (layer_state != nullptr) {
        *layer_state->conv_state = conv_state;
    }

    return pool_array(out);
}
