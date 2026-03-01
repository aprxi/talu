// ============================================================================ 
// Forward Pass Implementation
// ============================================================================

static inline void ensure_dense_decode_shapes(FusedDenseModel* m) {
    if (m->decode_shapes_initialized) return;
    m->attn_scale = (m->attention_multiplier > 0.0f)
        ? m->attention_multiplier
        : (m->query_pre_attn_scalar > 0.0f)
            ? (1.0f / std::sqrt(m->query_pre_attn_scalar))
            : (1.0f / std::sqrt(static_cast<float>(m->head_dim)));
    m->q_shape = {1, 1, m->n_heads, m->head_dim};
    m->kv_shape = {1, 1, m->n_kv_heads, m->head_dim};
    m->attn_out_shape = {1, 1, m->n_heads * m->head_dim};
    m->token_shape = {1, 1};
    m->decode_shapes_initialized = true;
}

static array dense_forward_logits_from_token(
    FusedDenseModel* m,
    MLXCache* cache,
    MLXShortConvCache* shortconv_cache,
    MLXMambaCache* mamba_cache,
    const array& token_idx,
    size_t pos_offset
) {
    const int n_layers = static_cast<int>(m->layers.size());
    ensure_dense_decode_shapes(m);

    // Embedding lookup
    array hidden = take(m->embed_tokens, reshape(token_idx, m->token_shape), 0);
    if (m->embedding_multiplier != 1.0f) {
        hidden = hidden * m->embedding_multiplier;
    } else if (m->has_norm_weight_offset) {
        hidden = hidden * std::sqrt(static_cast<float>(m->hidden_dim));
    }

    for (int layer_idx = 0; layer_idx < n_layers; layer_idx++) {
        const auto& l = m->layers[layer_idx];
        if (l.kind == FusedDenseModel::Layer::LayerKind::mamba) {
            auto* mamba_out = static_cast<const array*>(mlx_lazy_mamba_block_bf16(
                &hidden,
                &l.ln1_w,
                &l.mamba_in_proj,
                &l.mamba_conv_weight,
                l.mamba_conv_bias ? &*l.mamba_conv_bias : nullptr,
                &l.mamba_a_log,
                &l.mamba_d_skip,
                l.mamba_dt_bias ? &*l.mamba_dt_bias : nullptr,
                l.mamba_norm_weight ? &*l.mamba_norm_weight : nullptr,
                &l.mamba_out_proj,
                &l.ln2_w,
                l.mamba_gate_up ? &*l.mamba_gate_up : nullptr,
                l.mamba_down_proj ? &*l.mamba_down_proj : nullptr,
                m->use_gelu,
                m->residual_multiplier,
                m->rms_eps,
                mamba_cache,
                static_cast<size_t>(layer_idx),
                static_cast<size_t>(l.mamba_d_state),
                static_cast<size_t>(l.mamba_d_conv),
                static_cast<size_t>(l.mamba_n_heads),
                static_cast<size_t>(l.mamba_d_head),
                static_cast<size_t>(l.mamba_n_groups),
                l.mamba_gate_up_layout
            ));
            hidden = ensure_float16(*mamba_out);
            continue;
        }

        // RMS norm
        array normed = ensure_float16(fast::rms_norm(hidden, l.ln1_w, m->rms_eps));
        array mixer_out(0.0f, float32);

        if (l.kind == FusedDenseModel::Layer::LayerKind::shortconv) {
            const int seq_len = normed.shape(1);
            const int d_conv_i = l.shortconv_d_conv;
            const int conv_dim_i = l.shortconv_conv_dim;

            array bcx = matmul(normed, l.shortconv_in_proj_t);
            bcx = astype(bcx, float32);

            array b_gate = slice(bcx, {0, 0, 0}, {1, seq_len, conv_dim_i});
            array c_gate = slice(bcx, {0, 0, conv_dim_i}, {1, seq_len, 2 * conv_dim_i});
            array x_proj = slice(bcx, {0, 0, 2 * conv_dim_i}, {1, seq_len, 3 * conv_dim_i});
            array bx = b_gate * x_proj;

            ShortConvLayer* layer_state = nullptr;
            if (shortconv_cache != nullptr && layer_idx < static_cast<int>(shortconv_cache->layers.size())) {
                layer_state = &shortconv_cache->layers[layer_idx];
            }

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

            if (seq_len == 1) {
                const array bx_t = reshape(bx, {1, 1, conv_dim_i});
                if (d_conv_i > 1) {
                    const array state_tail = slice(conv_state, {0, 1, 0}, {1, d_conv_i, conv_dim_i});
                    conv_state = concatenate({state_tail, bx_t}, 1);
                } else {
                    conv_state = bx_t;
                }

                array conv_t = sum(conv_state * *l.shortconv_kernel_broadcast, 1);
                if (l.shortconv_bias_row) {
                    conv_t = conv_t + *l.shortconv_bias_row;
                }

                array c_t = reshape(c_gate, {1, conv_dim_i});
                const array gated = ensure_float16(reshape(conv_t * c_t, {1, 1, conv_dim_i}));
                mixer_out = matmul(gated, l.shortconv_out_proj_t);
            } else {
                std::vector<array> token_outputs;
                token_outputs.reserve(seq_len);
                for (int token_idx_i = 0; token_idx_i < seq_len; token_idx_i++) {
                    const array bx_t = slice(bx, {0, token_idx_i, 0}, {1, token_idx_i + 1, conv_dim_i});
                    if (d_conv_i > 1) {
                        const array state_tail = slice(conv_state, {0, 1, 0}, {1, d_conv_i, conv_dim_i});
                        conv_state = concatenate({state_tail, bx_t}, 1);
                    } else {
                        conv_state = bx_t;
                    }

                    array conv_t = sum(conv_state * *l.shortconv_kernel_broadcast, 1);
                    if (l.shortconv_bias_row) {
                        conv_t = conv_t + *l.shortconv_bias_row;
                    }

                    array c_t = slice(c_gate, {0, token_idx_i, 0}, {1, token_idx_i + 1, conv_dim_i});
                    c_t = reshape(c_t, {1, conv_dim_i});
                    const array gated = ensure_float16(reshape(conv_t * c_t, {1, 1, conv_dim_i}));
                    token_outputs.push_back(matmul(gated, l.shortconv_out_proj_t));
                }

                mixer_out = (token_outputs.size() == 1) ? token_outputs[0] : concatenate(token_outputs, 1);
            }

            if (layer_state != nullptr) {
                *layer_state->conv_state = conv_state;
            }
        } else if (l.use_mla) {
            auto* mla_out = static_cast<const array*>(mlx_lazy_mla_attention_bf16(
                &normed,
                &l.mla_q_a_w,
                &l.mla_q_a_norm,
                &l.mla_q_b_w,
                &l.mla_kv_a_w,
                &l.mla_kv_a_norm,
                &l.mla_kv_b_w,
                &l.mla_o_w,
                cache,
                static_cast<size_t>(layer_idx),
                static_cast<size_t>(l.mla_n_heads),
                static_cast<size_t>(l.mla_q_lora_rank),
                static_cast<size_t>(l.mla_kv_lora_rank),
                static_cast<size_t>(l.mla_qk_head_dim),
                static_cast<size_t>(l.mla_qk_rope_head_dim),
                static_cast<size_t>(l.mla_qk_nope_head_dim),
                static_cast<size_t>(l.mla_v_head_dim),
                pos_offset,
                m->rope_theta,
                nullptr,
                nullptr,
                0,
                m->rms_eps
            ));
            mixer_out = ensure_float16(*mla_out);
        } else {
            auto& cl = cache->layers[layer_idx];

            // Single matmul for Q+K+V (weights pre-concatenated)
            array qkv = matmul(normed, l.qkv_proj);
            auto qkv_parts = split(qkv, {l.q_size, l.q_size + l.kv_size}, -1);
            array q = reshape(qkv_parts[0], m->q_shape);
            array k = reshape(qkv_parts[1], m->kv_shape);
            array v = reshape(qkv_parts[2], m->kv_shape);

            if (l.q_norm) q = fast::rms_norm(q, *l.q_norm, m->rms_eps);
            if (l.k_norm) k = fast::rms_norm(k, *l.k_norm, m->rms_eps);

            q = transpose(q, g_transpose_perm);
            k = transpose(k, g_transpose_perm);
            v = transpose(v, g_transpose_perm);

            q = fast::rope(q, m->head_dim, false, m->rope_theta, 1.0f, static_cast<int>(pos_offset));
            k = fast::rope(k, m->head_dim, false, m->rope_theta, 1.0f, static_cast<int>(pos_offset));

            // Cache update
            size_t prev = cl.offset;
            int offset = static_cast<int>(prev + 1);

            // Dense decode keeps KV cache in float16 so q/k/v updates do not
            // cast on every token before attention.
            if (cl.k_bfloat16 == nullptr) {
                const int capacity = mlx_next_cache_capacity(cl, static_cast<size_t>(offset), 0, "[dense]");
                Shape shape = {1, m->n_kv_heads, capacity, m->head_dim};
                cl.k_bfloat16 = new array(zeros(shape, float16));
                cl.v_bfloat16 = new array(zeros(shape, float16));
            } else if (offset > cl.k_bfloat16->shape(2)) {
                const int current_capacity = cl.k_bfloat16->shape(2);
                const int new_capacity = mlx_next_cache_capacity(cl, static_cast<size_t>(offset), current_capacity, "[dense]");
                Shape new_shape = {1, m->n_kv_heads, new_capacity, m->head_dim};
                array new_k = zeros(new_shape, cl.k_bfloat16->dtype());
                array new_v = zeros(new_shape, cl.v_bfloat16->dtype());
                const int prev_capacity = current_capacity;
                if (prev_capacity > 0) {
                    const Shape copy_start = {0, 0, 0, 0};
                    const Shape copy_stop = {1, m->n_kv_heads, prev_capacity, m->head_dim};
                    new_k = slice_update(new_k, slice(*cl.k_bfloat16, copy_start, copy_stop), copy_start, copy_stop);
                    new_v = slice_update(new_v, slice(*cl.v_bfloat16, copy_start, copy_stop), copy_start, copy_stop);
                }
                *cl.k_bfloat16 = new_k;
                *cl.v_bfloat16 = new_v;
            }

            Shape update_start = {0, 0, static_cast<int>(prev), 0};
            Shape update_stop = {1, m->n_kv_heads, offset, m->head_dim};
            *cl.k_bfloat16 = slice_update(*cl.k_bfloat16, k, update_start, update_stop);
            *cl.v_bfloat16 = slice_update(*cl.v_bfloat16, v, update_start, update_stop);
            cl.offset = offset;

            const Shape slice_stop = {1, m->n_kv_heads, offset, m->head_dim};
            array k_full = slice(*cl.k_bfloat16, g_slice_start, slice_stop);
            array v_full = slice(*cl.v_bfloat16, g_slice_start, slice_stop);
            array attn_k(0.0f, float32);
            array attn_v(0.0f, float32);
            gqa_expand_attention_kv(q, k_full, v_full, &attn_k, &attn_v);
            array attn_out = fast::scaled_dot_product_attention(q, attn_k, attn_v, m->attn_scale, "");

            attn_out = transpose(attn_out, g_transpose_perm);
            attn_out = reshape(attn_out, m->attn_out_shape);

            mixer_out = matmul(attn_out, l.o_proj);
        }
        const array attn_for_residual = m->has_norm_weight_offset
            ? ensure_float16(fast::rms_norm(mixer_out, l.ln2_w, m->rms_eps))
            : ensure_float16(mixer_out);
        const array scaled_attn = (m->residual_multiplier != 1.0f)
            ? ensure_float16(attn_for_residual * m->residual_multiplier)
            : attn_for_residual;
        array hidden_1 = ensure_float16(hidden + scaled_attn);

        // FFN - single matmul for gate+up (weights pre-concatenated)
        array normed_2 = ensure_float16(fast::rms_norm(hidden_1, l.ln2_w, m->rms_eps));
        array gate_up = matmul(normed_2, l.gate_up_proj);
        auto parts = split(gate_up, 2, -1);
        array& gate = parts[0];
        array& up = parts[1];
        const array ffn_mid = [&]() -> array {
            if (m->use_gelu) {
                const float sqrt_2_over_pi = 0.7978845608f;
                const array x3 = gate * gate * gate;
                const array inner = sqrt_2_over_pi * (gate + 0.044715f * x3);
                return gate * 0.5f * (1.0f + tanh(inner)) * up;
            }
            return gate * sigmoid(gate) * up;
        }();
        const array down = matmul(ensure_float16(ffn_mid), l.down_proj);
        const array scaled_down = (m->residual_multiplier != 1.0f)
            ? ensure_float16(down * m->residual_multiplier)
            : ensure_float16(down);

        hidden = ensure_float16(hidden_1 + scaled_down);
    }

    // Final norm + LM head
    array final_normed = ensure_float16(fast::rms_norm(hidden, m->ln_final, m->rms_eps));
    array logits = matmul(final_normed, m->lm_head);
    if (m->logits_scaling != 1.0f) {
        logits = logits / m->logits_scaling;
    }
    return reshape(logits, {-1});
}

static array dense_forward_from_token(
    FusedDenseModel* m,
    MLXCache* cache,
    MLXShortConvCache* shortconv_cache,
    MLXMambaCache* mamba_cache,
    const array& token_idx,
    size_t pos_offset
) {
    array logits = dense_forward_logits_from_token(m, cache, shortconv_cache, mamba_cache, token_idx, pos_offset);
    return argmax(logits, 0);
}
