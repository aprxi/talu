// ============================================================================
// Forward Pass Implementation
// ============================================================================

static array fused_forward_logits_from_token(
    FusedModelWeights* fused_weights,
    MLXCache* cache_state,
    MLXShortConvCache* shortconv_cache_state,
    MLXMambaCache* mamba_cache_state,
    const array& token_idx,
    size_t pos_offset
) {
    const int group_size = fused_weights->group_size;
    const int bit_width = fused_weights->bits;
    const int n_layers = static_cast<int>(fused_weights->layers.size());

    // Attention scale:
    // - Custom attention_multiplier: use directly (e.g., 0.015625 = 1/64)
    // - query_pre_attn_scalar: use 1/sqrt(val) for custom scaling
    // - Default: use 1/sqrt(head_dim)
    const float attn_scale = (fused_weights->attention_multiplier > 0.0f)
        ? fused_weights->attention_multiplier
        : (fused_weights->query_pre_attn_scalar > 0.0f)
            ? (1.0f / std::sqrt(fused_weights->query_pre_attn_scalar))
            : (1.0f / std::sqrt(static_cast<float>(fused_weights->head_dim)));
    const Shape q_shape = {1, 1, fused_weights->n_heads, fused_weights->head_dim};
    const Shape kv_shape = {1, 1, fused_weights->n_kv_heads, fused_weights->head_dim};
    const Shape attn_out_shape = {1, 1, fused_weights->n_heads * fused_weights->head_dim};

    // Embedding lookup (quantized)
    array idx = reshape(astype(token_idx, int32), {1, 1});
    array embed_rows = take(fused_weights->embed_w, idx, 0);
    array scale_rows = take(fused_weights->embed_s, idx, 0);
    array bias_rows = take(fused_weights->embed_b, idx, 0);
    array hidden = dequantize(embed_rows, scale_rows, bias_rows, group_size, bit_width, "affine");

    // Embedding scaling (architecture-specific)
    if (fused_weights->embedding_multiplier != 1.0f) {
        hidden = hidden * fused_weights->embedding_multiplier;
    } else if (fused_weights->has_norm_weight_offset) {
        // Architectures with (1+w) norms also use sqrt(hidden_dim) embedding scaling
        hidden = hidden * std::sqrt(static_cast<float>(fused_weights->hidden_dim));
    }

    for (int layer_idx = 0; layer_idx < n_layers; layer_idx++) {
        const auto& l = fused_weights->layers[layer_idx];
        if (l.kind == FusedModelWeights::Layer::LayerKind::mamba) {
            auto* mamba_out = static_cast<const array*>(mlx_lazy_mamba_block_quantized(
                &hidden,
                &l.ln1_w,
                &l.mamba_in_w,
                &l.mamba_in_s,
                &l.mamba_in_b,
                &l.mamba_conv_w,
                l.mamba_conv_b ? &*l.mamba_conv_b : nullptr,
                &l.mamba_a_log,
                &l.mamba_d_skip,
                l.mamba_dt_bias ? &*l.mamba_dt_bias : nullptr,
                l.mamba_norm_weight ? &*l.mamba_norm_weight : nullptr,
                &l.mamba_out_w,
                &l.mamba_out_s,
                &l.mamba_out_b,
                &l.ln2_w,
                l.mamba_gate_up_w ? &*l.mamba_gate_up_w : nullptr,
                l.mamba_gate_up_s ? &*l.mamba_gate_up_s : nullptr,
                l.mamba_gate_up_b ? &*l.mamba_gate_up_b : nullptr,
                l.mamba_down_w ? &*l.mamba_down_w : nullptr,
                l.mamba_down_s ? &*l.mamba_down_s : nullptr,
                l.mamba_down_b ? &*l.mamba_down_b : nullptr,
                static_cast<size_t>(group_size),
                static_cast<size_t>(bit_width),
                fused_weights->use_gelu,
                fused_weights->residual_multiplier,
                fused_weights->rms_eps,
                mamba_cache_state,
                static_cast<size_t>(layer_idx),
                static_cast<size_t>(l.mamba_d_state),
                static_cast<size_t>(l.mamba_d_conv),
                static_cast<size_t>(l.mamba_n_heads),
                static_cast<size_t>(l.mamba_d_head),
                static_cast<size_t>(l.mamba_n_groups),
                l.mamba_gate_up_layout
            ));
            hidden = *mamba_out;
            continue;
        }
        auto& cl = cache_state->layers[layer_idx];

        array normed = fast::rms_norm(hidden, l.ln1_w, fused_weights->rms_eps);
        array attn_proj(0.0f, float32);
        if (l.kind == FusedModelWeights::Layer::LayerKind::shortconv) {
            const int d_conv_i = l.shortconv_d_conv;
            const int conv_dim_i = l.shortconv_conv_dim;

            array bcx = quantized_matmul(
                normed,
                l.shortconv_in_w,
                l.shortconv_in_s,
                l.shortconv_in_b,
                true,
                group_size,
                bit_width,
                "affine"
            );
            bcx = astype(bcx, float32);

            array b_gate = slice(bcx, {0, 0, 0}, {1, 1, conv_dim_i});
            array c_gate = slice(bcx, {0, 0, conv_dim_i}, {1, 1, 2 * conv_dim_i});
            array x_proj = slice(bcx, {0, 0, 2 * conv_dim_i}, {1, 1, 3 * conv_dim_i});
            array bx = b_gate * x_proj;

            array conv_state(0.0f, float32);
            if (shortconv_cache_state != nullptr &&
                layer_idx < static_cast<int>(shortconv_cache_state->layers.size())) {
                auto& scl = shortconv_cache_state->layers[layer_idx];
                const bool need_init =
                    scl.conv_state == nullptr ||
                    scl.conv_state->shape(1) != d_conv_i ||
                    scl.conv_state->shape(2) != conv_dim_i;
                if (need_init) {
                    delete scl.conv_state;
                    scl.conv_state = new array(zeros({1, d_conv_i, conv_dim_i}, float32));
                }
                conv_state = *scl.conv_state;
            } else {
                conv_state = zeros({1, d_conv_i, conv_dim_i}, float32);
            }

            const array bx_t = reshape(bx, {1, 1, conv_dim_i});
            if (d_conv_i > 1) {
                const array state_tail = slice(conv_state, {0, 1, 0}, {1, d_conv_i, conv_dim_i});
                conv_state = concatenate({state_tail, bx_t}, 1);
            } else {
                conv_state = bx_t;
            }

            if (!l.shortconv_kernel_broadcast.has_value()) {
                throw std::invalid_argument("[fused] shortconv kernel precompute missing");
            }
            const array& conv_kernel_broadcast = *l.shortconv_kernel_broadcast;
            array conv_t = sum(conv_state * conv_kernel_broadcast, 1);
            if (l.shortconv_bias_row) {
                conv_t = conv_t + *l.shortconv_bias_row;
            }

            const array c_t = reshape(c_gate, {1, conv_dim_i});
            const array gated = reshape(conv_t * c_t, {1, 1, conv_dim_i});
            attn_proj = quantized_matmul(
                gated,
                l.shortconv_out_w,
                l.shortconv_out_s,
                l.shortconv_out_b,
                true,
                group_size,
                bit_width,
                "affine"
            );

            if (shortconv_cache_state != nullptr &&
                layer_idx < static_cast<int>(shortconv_cache_state->layers.size())) {
                *shortconv_cache_state->layers[layer_idx].conv_state = conv_state;
            }
        } else if (l.use_mla) {
            auto* mla_out = static_cast<const array*>(mlx_lazy_mla_attention_quantized(
                &normed,
                &l.mla_q_a_w,
                &l.mla_q_a_s,
                &l.mla_q_a_b,
                &l.mla_q_a_norm,
                &l.mla_q_b_w,
                &l.mla_q_b_s,
                &l.mla_q_b_b,
                &l.mla_kv_a_w,
                &l.mla_kv_a_s,
                &l.mla_kv_a_b,
                &l.mla_kv_a_norm,
                &l.mla_kv_b_w,
                &l.mla_kv_b_s,
                &l.mla_kv_b_b,
                &l.mla_o_w,
                &l.mla_o_s,
                &l.mla_o_b,
                cache_state,
                static_cast<size_t>(layer_idx),
                static_cast<size_t>(l.mla_n_heads),
                static_cast<size_t>(l.mla_q_lora_rank),
                static_cast<size_t>(l.mla_kv_lora_rank),
                static_cast<size_t>(l.mla_qk_head_dim),
                static_cast<size_t>(l.mla_qk_rope_head_dim),
                static_cast<size_t>(l.mla_qk_nope_head_dim),
                static_cast<size_t>(l.mla_v_head_dim),
                pos_offset,
                fused_weights->rope_theta,
                nullptr,
                nullptr,
                0,
                fused_weights->rms_eps,
                static_cast<size_t>(group_size),
                static_cast<size_t>(bit_width)
            ));
            attn_proj = *mla_out;
        } else {
            array q(0.0f), k(0.0f), v(0.0f);
            if (l.use_fused_qkv && l.qkv_w) {
                // Fused QKV: single matmul then split
                array qkv = quantized_matmul(normed, *l.qkv_w, *l.qkv_s, *l.qkv_b, true, group_size, bit_width, "affine");
                // Split: q=[n_heads*head_dim], k=[n_kv_heads*head_dim], v=[n_kv_heads*head_dim]
                int q_dim = fused_weights->n_heads * fused_weights->head_dim;
                int kv_dim = fused_weights->n_kv_heads * fused_weights->head_dim;
                q = slice(qkv, {0, 0, 0}, {1, 1, q_dim});
                k = slice(qkv, {0, 0, q_dim}, {1, 1, q_dim + kv_dim});
                v = slice(qkv, {0, 0, q_dim + kv_dim}, {1, 1, q_dim + 2 * kv_dim});
            } else {
                q = quantized_matmul(normed, l.q_w, l.q_s, l.q_b, true, group_size, bit_width, "affine");
                k = quantized_matmul(normed, l.k_w, l.k_s, l.k_b, true, group_size, bit_width, "affine");
                v = quantized_matmul(normed, l.v_w, l.v_s, l.v_b, true, group_size, bit_width, "affine");
            }

            q = reshape(q, q_shape);
            k = reshape(k, kv_shape);
            v = reshape(v, kv_shape);

            if (l.q_norm) q = fast::rms_norm(q, *l.q_norm, fused_weights->rms_eps);
            if (l.k_norm) k = fast::rms_norm(k, *l.k_norm, fused_weights->rms_eps);

            q = transpose(q, g_transpose_perm);
            k = transpose(k, g_transpose_perm);
            v = transpose(v, g_transpose_perm);

            // Apply RoPE with custom frequencies (Llama3) or standard base
            if (fused_weights->rope_freqs) {
                q = fast::rope(q, fused_weights->head_dim, false, std::nullopt, 1.0f, static_cast<int>(pos_offset), fused_weights->rope_freqs);
                k = fast::rope(k, fused_weights->head_dim, false, std::nullopt, 1.0f, static_cast<int>(pos_offset), fused_weights->rope_freqs);
            } else {
                q = fast::rope(q, fused_weights->head_dim, false, fused_weights->rope_theta, 1.0f, static_cast<int>(pos_offset));
                k = fast::rope(k, fused_weights->head_dim, false, fused_weights->rope_theta, 1.0f, static_cast<int>(pos_offset));
            }

            // Cache update
            size_t prev = cl.offset;
            int offset = static_cast<int>(prev + 1);

            if (cl.k_bfloat16 == nullptr) {
                const int capacity = mlx_next_cache_capacity(cl, static_cast<size_t>(offset), 0, "[fused]");
                Shape shape = {1, fused_weights->n_kv_heads, capacity, fused_weights->head_dim};
                cl.k_bfloat16 = new array(zeros(shape, bfloat16));
                cl.v_bfloat16 = new array(zeros(shape, bfloat16));
            } else if (offset > cl.k_bfloat16->shape(2)) {
                const int current_capacity = cl.k_bfloat16->shape(2);
                const int new_capacity = mlx_next_cache_capacity(cl, static_cast<size_t>(offset), current_capacity, "[fused]");
                Shape shape = {1, fused_weights->n_kv_heads, new_capacity, fused_weights->head_dim};
                array new_k = zeros(shape, bfloat16);
                array new_v = zeros(shape, bfloat16);
                if (prev > 0) {
                    Shape copy_stop = {1, fused_weights->n_kv_heads, static_cast<int>(prev), fused_weights->head_dim};
                    new_k = slice_update(new_k, slice(*cl.k_bfloat16, g_slice_start, copy_stop), g_slice_start, copy_stop);
                    new_v = slice_update(new_v, slice(*cl.v_bfloat16, g_slice_start, copy_stop), g_slice_start, copy_stop);
                }
                *cl.k_bfloat16 = new_k;
                *cl.v_bfloat16 = new_v;
            }

            Shape update_start = {0, 0, static_cast<int>(prev), 0};
            Shape update_stop = {1, fused_weights->n_kv_heads, offset, fused_weights->head_dim};
            *cl.k_bfloat16 = slice_update(*cl.k_bfloat16, k, update_start, update_stop);
            *cl.v_bfloat16 = slice_update(*cl.v_bfloat16, v, update_start, update_stop);
            cl.offset = offset;

            const Shape slice_stop = {1, fused_weights->n_kv_heads, offset, fused_weights->head_dim};
            array k_full = slice(*cl.k_bfloat16, g_slice_start, slice_stop);
            array v_full = slice(*cl.v_bfloat16, g_slice_start, slice_stop);

            array attn_k(0.0f, float32);
            array attn_v(0.0f, float32);
            gqa_expand_attention_kv(q, k_full, v_full, &attn_k, &attn_v);
            array attn_out = fast::scaled_dot_product_attention(q, attn_k, attn_v, attn_scale, "");

            attn_out = transpose(attn_out, g_transpose_perm);
            attn_out = reshape(attn_out, attn_out_shape);

            attn_proj = quantized_matmul(attn_out, l.o_w, l.o_s, l.o_b, true, group_size, bit_width, "affine");
        }

        // Apply post-attention norm to attn output before residual (if architecture uses it)
        if (fused_weights->has_norm_weight_offset) {
            attn_proj = fast::rms_norm(attn_proj, l.ln2_w, fused_weights->rms_eps);
        }
        // Scale layer output by residual_multiplier (NOT the residual input)
        array hidden_1 = (fused_weights->residual_multiplier != 1.0f)
            ? hidden + attn_proj * fused_weights->residual_multiplier
            : hidden + attn_proj;

        // FFN normalization: use pre_ffn_norm if present, otherwise use ln2
        array normed_2 = (fused_weights->has_norm_weight_offset && l.pre_ffn_norm)
            ? fast::rms_norm(hidden_1, *l.pre_ffn_norm, fused_weights->rms_eps)
            : fast::rms_norm(hidden_1, l.ln2_w, fused_weights->rms_eps);

        array down(0.0f, float32);
        if (l.use_moe) {
            auto* moe_out = static_cast<const array*>(mlx_lazy_fused_moe_ffn_mxfp4(
                &normed_2,
                &l.moe_router_w,
                l.moe_router_s ? &*l.moe_router_s : nullptr,
                l.moe_router_b ? &*l.moe_router_b : nullptr,
                l.moe_router_bias ? &*l.moe_router_bias : nullptr,
                &l.moe_gate_w,
                &l.moe_gate_s,
                &l.moe_up_w,
                &l.moe_up_s,
                &l.moe_down_w,
                &l.moe_down_s,
                l.moe_gate_bias ? &*l.moe_gate_bias : nullptr,
                l.moe_up_bias ? &*l.moe_up_bias : nullptr,
                l.moe_down_bias ? &*l.moe_down_bias : nullptr,
                static_cast<size_t>(l.moe_num_experts),
                static_cast<size_t>(l.moe_experts_per_token),
                static_cast<size_t>(l.moe_router_group_size),
                static_cast<size_t>(l.moe_expert_group_size)));
            down = *moe_out;
        } else {
            array gate(0.0f), up(0.0f);
            if (l.use_fused_gate_up && l.gate_up_w) {
                // Fused gate/up: single matmul then split
                array gate_up = quantized_matmul(normed_2, *l.gate_up_w, *l.gate_up_s, *l.gate_up_b, true, group_size, bit_width, "affine");
                int d_ff = l.gate_w.shape(0);  // FFN intermediate size
                gate = slice(gate_up, {0, 0, 0}, {1, 1, d_ff});
                up = slice(gate_up, {0, 0, d_ff}, {1, 1, 2 * d_ff});
            } else {
                gate = quantized_matmul(normed_2, l.gate_w, l.gate_s, l.gate_b, true, group_size, bit_width, "affine");
                up = quantized_matmul(normed_2, l.up_w, l.up_s, l.up_b, true, group_size, bit_width, "affine");
            }

            // Activation: GELU or SiLU based on model config
            array mid = [&]() -> array {
                if (fused_weights->use_gelu) {
                    // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                    const float sqrt_2_over_pi = 0.7978845608f;
                    array x3 = gate * gate * gate;
                    array inner = sqrt_2_over_pi * (gate + 0.044715f * x3);
                    return gate * 0.5f * (1.0f + tanh(inner)) * up;
                } else {
                    return (gate * sigmoid(gate)) * up;
                }
            }();
            down = quantized_matmul(mid, l.down_w, l.down_s, l.down_b, true, group_size, bit_width, "affine");
        }

        // Apply post-FFN norm if present
        if (l.post_ffn_norm) {
            down = fast::rms_norm(down, *l.post_ffn_norm, fused_weights->rms_eps);
        }

        // Scale layer output by residual_multiplier (NOT the residual input)
        hidden = (fused_weights->residual_multiplier != 1.0f)
            ? hidden_1 + down * fused_weights->residual_multiplier
            : hidden_1 + down;
    }

    array final_normed = fast::rms_norm(hidden, fused_weights->ln_final, fused_weights->rms_eps);
    array logits = quantized_matmul(final_normed, fused_weights->lm_head_w, fused_weights->lm_head_s, fused_weights->lm_head_b, true, group_size, bit_width, "affine");

    // Scale logits by logits_scaling (divide, not multiply!)
    if (fused_weights->logits_scaling != 1.0f) {
        logits = logits / fused_weights->logits_scaling;
    }

    return reshape(logits, {-1});
}

static array fused_forward_from_token(
    FusedModelWeights* fused_weights,
    MLXCache* cache_state,
    MLXShortConvCache* shortconv_cache_state,
    MLXMambaCache* mamba_cache_state,
    const array& token_idx,
    size_t pos_offset
) {
    array logits = fused_forward_logits_from_token(
        fused_weights,
        cache_state,
        shortconv_cache_state,
        mamba_cache_state,
        token_idx,
        pos_offset
    );
    return argmax(logits, 0);
}
