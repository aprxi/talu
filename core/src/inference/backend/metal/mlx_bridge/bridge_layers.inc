// bridge_layers.inc — Layer implementations: run_full_attention_layer, weight combining
// (combine_*/compact_*), run_linear_attention_layer, run_shortconv_layer.

array run_full_attention_layer(mlx_ctx* ctx, int layer_idx, const array& x_norm) {
    LayerWeights& lw = ctx->layers[static_cast<size_t>(layer_idx)];
    KVCacheState& cache = ctx->kv_cache[static_cast<size_t>(layer_idx)];

    const int B = x_norm.shape(0);
    const int S = x_norm.shape(1);

    const bool prefill_only_qmm = false;
    const int q_proj_width = lw.attn_q_proj_width;
    const int k_proj_width = lw.attn_k_proj_width;
    const int v_proj_width = lw.attn_v_proj_width;
    const bool use_fused_qkv_proj =
        (lw.attn_qkv_rhs.size() > 1 || lw.attn_qkv_has_q) &&
        q_proj_width > 0 &&
        k_proj_width > 0 &&
        v_proj_width > 0;

    array q_proj_out = array(0.0f);
    array k_proj_out = array(0.0f);
    array v_proj_out = array(0.0f);
    if (use_fused_qkv_proj) {
        const array qkv_proj_out = linear_decode_maybe_quantized(
            ctx->fp8_decode_qmm_enabled,
            ctx->cfg.quant_group_size,
            ctx->cfg.quant_bits,
            x_norm,
            lw.attn_qkv_rhs,
            lw.attn_qkv_has_q,
            lw.attn_qkv_q_w,
            lw.attn_qkv_q_scales,
            lw.attn_qkv_q_biases,
            prefill_only_qmm,
            layer_idx,
            "attn.qkv_proj",
            &lw.attn_qkv_prefill_rhs
        );
        q_proj_out = slice(qkv_proj_out, {0, 0, 0}, {B, S, q_proj_width});
        k_proj_out = slice(qkv_proj_out, {0, 0, q_proj_width}, {B, S, q_proj_width + k_proj_width});
        v_proj_out = slice(
            qkv_proj_out,
            {0, 0, q_proj_width + k_proj_width},
            {B, S, q_proj_width + k_proj_width + v_proj_width}
        );
    } else {
        q_proj_out = linear_decode_maybe_quantized(
            ctx->fp8_decode_qmm_enabled,
            ctx->cfg.quant_group_size,
            ctx->cfg.quant_bits,
            x_norm,
            lw.attn_q_rhs,
            lw.attn_q_has_q,
            lw.attn_q_q_w,
            lw.attn_q_q_scales,
            lw.attn_q_q_biases,
            prefill_only_qmm,
            layer_idx,
            "attn.q_proj"
        );
        k_proj_out = linear_decode_maybe_quantized(
            ctx->fp8_decode_qmm_enabled,
            ctx->cfg.quant_group_size,
            ctx->cfg.quant_bits,
            x_norm,
            lw.attn_k_rhs,
            lw.attn_k_has_q,
            lw.attn_k_q_w,
            lw.attn_k_q_scales,
            lw.attn_k_q_biases,
            prefill_only_qmm
        );
        v_proj_out = linear_decode_maybe_quantized(
            ctx->fp8_decode_qmm_enabled,
            ctx->cfg.quant_group_size,
            ctx->cfg.quant_bits,
            x_norm,
            lw.attn_v_rhs,
            lw.attn_v_has_q,
            lw.attn_v_q_w,
            lw.attn_v_q_scales,
            lw.attn_v_q_biases,
            prefill_only_qmm
        );
    }
    const int q_last_dim = q_proj_out.shape(2);
    const int q_base_dim = lw.attn_num_heads * lw.attn_head_dim;
    array queries_raw = array(0.0f);
    array gate = array(0.0f);
    bool has_query_gate = false;
    if (q_last_dim == 2 * q_base_dim) {
        // Qwen3.5 full-attention branch: q projection carries both q and gate.
        has_query_gate = true;
        const array q_reshaped = reshape(q_proj_out, {B, S, lw.attn_num_heads, 2 * lw.attn_head_dim});
        queries_raw = slice(q_reshaped, {0, 0, 0, 0}, {B, S, lw.attn_num_heads, lw.attn_head_dim});
        const array gate_raw = slice(
            q_reshaped,
            {0, 0, 0, lw.attn_head_dim},
            {B, S, lw.attn_num_heads, 2 * lw.attn_head_dim}
        );
        gate = reshape(gate_raw, {B, S, q_base_dim});
    } else if (q_last_dim == q_base_dim) {
        // Qwen3 attention branch: q projection has no gating half.
        queries_raw = reshape(q_proj_out, {B, S, lw.attn_num_heads, lw.attn_head_dim});
    } else {
        throw std::runtime_error("unexpected self_attn.q_proj output shape");
    }

    KVCacheState* read_cache = &cache;
    bool use_shared_read_cache = false;
    if (lw.kv_shared_source_layer >= 0) {
        const int src = lw.kv_shared_source_layer;
        if (src >= 0 && src < static_cast<int>(ctx->kv_cache.size())) {
            KVCacheState* src_cache = &ctx->kv_cache[static_cast<size_t>(src)];
            if (src_cache->keys.has_value() && src_cache->values.has_value() && src_cache->offset > 0) {
                read_cache = src_cache;
                use_shared_read_cache = true;
            }
        }
    }
    int rope_offset = cache.offset;
    if (use_shared_read_cache) {
        rope_offset = std::max(0, read_cache->offset - S);
    }
    const int rope_dims = lw.attn_rope_dims > 0
        ? lw.attn_rope_dims
        : static_cast<int>(std::round(lw.attn_head_dim * lw.attn_partial_rotary_factor));
    const bool use_proportional_full_head_rope =
        ctx->cfg.global_head_dim > 0 &&
        lw.attn_head_dim == ctx->cfg.global_head_dim &&
        rope_dims > 0 && rope_dims < lw.attn_head_dim;

    array queries = fast::rms_norm(queries_raw, std::optional<array>(lw.attn_q_norm_w), ctx->cfg.rms_norm_eps);
    array keys = reshape(k_proj_out, {B, S, lw.attn_num_kv_heads, lw.attn_head_dim});
    keys = fast::rms_norm(keys, std::optional<array>(lw.attn_k_norm_w), ctx->cfg.rms_norm_eps);
    array values = reshape(v_proj_out, {B, S, lw.attn_num_kv_heads, lw.attn_head_dim});
    if (ctx->cfg.use_v_norm) {
        values = fast::rms_norm(values, std::nullopt, ctx->cfg.rms_norm_eps);
    }

    queries = transpose(queries, {0, 2, 1, 3});
    keys = transpose(keys, {0, 2, 1, 3});
    values = transpose(values, {0, 2, 1, 3});

    if (use_proportional_full_head_rope) {
        const int hd = lw.attn_head_dim;
        const int half_rope = rope_dims / 2;
        const int half_hd = hd / 2;
        auto apply_rope_full_head = [&](array t) -> array {
            const int nh = t.shape(1);
            array t_first = slice(t, {0,0,0,0},               {B,nh,S,half_rope});
            array t_pass1 = slice(t, {0,0,0,half_rope},        {B,nh,S,half_hd});
            array t_third = slice(t, {0,0,0,half_hd},          {B,nh,S,half_hd+half_rope});
            array t_pass2 = slice(t, {0,0,0,half_hd+half_rope},{B,nh,S,hd});
            array rope_in = concatenate({t_first, t_third}, 3);
            rope_in = fast::rope(rope_in, rope_dims, false,
                                 std::optional<float>(lw.attn_rope_theta), 1.0f, rope_offset);
            array first_rot = slice(rope_in, {0,0,0,0},        {B,nh,S,half_rope});
            array third_rot = slice(rope_in, {0,0,0,half_rope},{B,nh,S,rope_dims});
            return concatenate({first_rot, t_pass1, third_rot, t_pass2}, 3);
        };
        queries = apply_rope_full_head(queries);
        keys = apply_rope_full_head(keys);
    } else {
        queries = fast::rope(queries, rope_dims, false,
                             std::optional<float>(lw.attn_rope_theta), 1.0f, rope_offset);
        keys = fast::rope(keys, rope_dims, false,
                          std::optional<float>(lw.attn_rope_theta), 1.0f, rope_offset);
    }

    const int prev = cache.offset;
    if (!use_shared_read_cache) {
        ensure_kv_cache_capacity(
            ctx,
            &cache,
            B,
            lw.attn_num_kv_heads,
            lw.attn_head_dim,
            prev + S,
            keys.dtype(),
            values.dtype()
        );
        if (!cache.keys.has_value() || !cache.values.has_value()) {
            throw std::runtime_error("failed to allocate KV cache capacity");
        }
        cache.keys = slice_update(
            *cache.keys,
            keys,
            {0, 0, prev, 0},
            {B, lw.attn_num_kv_heads, prev + S, lw.attn_head_dim}
        );
        cache.values = slice_update(
            *cache.values,
            values,
            {0, 0, prev, 0},
            {B, lw.attn_num_kv_heads, prev + S, lw.attn_head_dim}
        );
        cache.offset = prev + S;
    } else {
        cache.offset = read_cache->offset;
    }

    if (!read_cache->keys.has_value() || !read_cache->values.has_value() || read_cache->offset <= 0) {
        throw std::runtime_error("invalid KV read cache state");
    }
    const int read_offset = read_cache->offset;
    const int window = (lw.attn_is_sliding && ctx->cfg.sliding_window > 0) ? ctx->cfg.sliding_window : 0;
    const int read_start = (window > 0 && read_offset > window) ? (read_offset - window) : 0;
    const array keys_view = slice(
        *read_cache->keys,
        {0, 0, read_start, 0},
        {B, lw.attn_num_kv_heads, read_offset, lw.attn_head_dim}
    );
    const array values_view = slice(
        *read_cache->values,
        {0, 0, read_start, 0},
        {B, lw.attn_num_kv_heads, read_offset, lw.attn_head_dim}
    );

    const std::string mask_mode = (S > 1) ? "causal" : "";
    const bool force_f32_attn =
        ctx->cfg.hidden_size_per_layer_input > 0 &&
        env_truthy("TALU_METAL_GEMMA4_FORCE_F32_ATTN", false);
    array attn_out = array(0.0f);
    if (force_f32_attn) {
        const array q_f32 = astype(queries, float32);
        const array k_f32 = astype(keys_view, float32);
        const array v_f32 = astype(values_view, float32);
        const array attn_f32 = fast::scaled_dot_product_attention(
            q_f32,
            k_f32,
            v_f32,
            lw.attn_scale,
            mask_mode
        );
        attn_out = astype(attn_f32, queries.dtype());
    } else {
        attn_out = fast::scaled_dot_product_attention(
            queries,
            keys_view,
            values_view,
            lw.attn_scale,
            mask_mode
        );
    }

    attn_out = transpose(attn_out, {0, 2, 1, 3});
    attn_out = reshape(attn_out, {B, S, lw.attn_num_heads * lw.attn_head_dim});
    if (has_query_gate) {
        static const auto compiled_sigmoid_gate = compile(
            [](const std::vector<array>& inputs) {
                return std::vector<array>{inputs[0] * sigmoid(inputs[1])};
            },
            true
        );
        attn_out = compiled_sigmoid_gate({attn_out, gate}).at(0);
    }

    return linear_decode_maybe_quantized(
        ctx->fp8_decode_qmm_enabled,
        ctx->cfg.quant_group_size,
        ctx->cfg.quant_bits,
        attn_out,
        lw.attn_o_rhs,
        lw.attn_o_has_q,
        lw.attn_o_q_w,
        lw.attn_o_q_scales,
        lw.attn_o_q_biases,
        prefill_only_qmm
    );
}

bool combine_linear_in_proj_weights(LayerWeights* lw) {
    if (lw == nullptr) return false;

    if (lw->lin_in_proj_qkvz_rhs.size() > 1 && lw->lin_in_proj_ba_rhs.size() > 1) {
        if (lw->lin_in_proj_qkvz_rhs.ndim() == 2 &&
            lw->lin_in_proj_ba_rhs.ndim() == 2 &&
            lw->lin_in_proj_qkvz_rhs.shape(0) == lw->lin_in_proj_ba_rhs.shape(0))
        {
            lw->lin_in_proj_all_rhs = stop_gradient(copy(concatenate(
                {lw->lin_in_proj_qkvz_rhs, lw->lin_in_proj_ba_rhs},
                1
            )));
        }
    }

    if (!lw->lin_in_proj_all_has_q &&
        lw->lin_in_proj_qkvz_has_q &&
        lw->lin_in_proj_ba_has_q &&
        lw->lin_in_proj_qkvz_q_w.ndim() == 2 &&
        lw->lin_in_proj_ba_q_w.ndim() == 2 &&
        lw->lin_in_proj_qkvz_q_w.shape(1) == lw->lin_in_proj_ba_q_w.shape(1) &&
        lw->lin_in_proj_qkvz_q_scales.ndim() == 2 &&
        lw->lin_in_proj_ba_q_scales.ndim() == 2 &&
        lw->lin_in_proj_qkvz_q_scales.shape(1) == lw->lin_in_proj_ba_q_scales.shape(1) &&
        lw->lin_in_proj_qkvz_q_biases.ndim() == 2 &&
        lw->lin_in_proj_ba_q_biases.ndim() == 2 &&
        lw->lin_in_proj_qkvz_q_biases.shape(1) == lw->lin_in_proj_ba_q_biases.shape(1))
    {
        lw->lin_in_proj_all_q_w = stop_gradient(copy(concatenate(
            {lw->lin_in_proj_qkvz_q_w, lw->lin_in_proj_ba_q_w},
            0
        )));
        lw->lin_in_proj_all_q_scales = stop_gradient(copy(concatenate(
            {lw->lin_in_proj_qkvz_q_scales, lw->lin_in_proj_ba_q_scales},
            0
        )));
        lw->lin_in_proj_all_q_biases = stop_gradient(copy(concatenate(
            {lw->lin_in_proj_qkvz_q_biases, lw->lin_in_proj_ba_q_biases},
            0
        )));
        lw->lin_in_proj_all_has_q = true;
    }

    return lw->lin_in_proj_all_has_q || lw->lin_in_proj_all_rhs.size() > 1;
}

void compact_linear_in_proj_weights(LayerWeights* lw) {
    if (lw == nullptr) return;
    if (!lw->lin_in_proj_all_has_q && lw->lin_in_proj_all_rhs.size() <= 1) return;

    // Clear individual qkvz/ba weights — the combined all_* version is used.
    // NOTE: the combined arrays were created via concatenate()+copy() and may
    // still hold lazy graph references to the originals. MLX ref-counting
    // keeps the underlying buffers alive until the combined arrays are
    // evaluated, so this only frees our C++ array handles, not the Metal
    // buffers. The real memory savings happen after the first forward pass
    // materializes the combined arrays and releases the graph references.
    lw->lin_in_proj_qkvz_rhs = array(0.0f);
    lw->lin_in_proj_ba_rhs = array(0.0f);
    lw->lin_in_proj_qkvz_q_w = array(0.0f);
    lw->lin_in_proj_qkvz_q_scales = array(0.0f);
    lw->lin_in_proj_qkvz_q_biases = array(0.0f);
    lw->lin_in_proj_qkvz_has_q = false;
    lw->lin_in_proj_ba_q_w = array(0.0f);
    lw->lin_in_proj_ba_q_scales = array(0.0f);
    lw->lin_in_proj_ba_q_biases = array(0.0f);
    lw->lin_in_proj_ba_has_q = false;
    // Also clear individual separate projections if they exist.
    lw->lin_in_proj_qkv_rhs = array(0.0f);
    lw->lin_in_proj_z_rhs = array(0.0f);
    lw->lin_in_proj_a_rhs = array(0.0f);
    lw->lin_in_proj_b_rhs = array(0.0f);
}

bool combine_shared_expert_gate_up_weights(LayerWeights* lw) {
    if (lw == nullptr) return false;

    if (lw->moe_shared_gate_rhs.size() > 1 && lw->moe_shared_up_rhs.size() > 1) {
        if (lw->moe_shared_gate_rhs.ndim() == 2 &&
            lw->moe_shared_up_rhs.ndim() == 2 &&
            lw->moe_shared_gate_rhs.shape(0) == lw->moe_shared_up_rhs.shape(0))
        {
            lw->moe_shared_gate_up_rhs = stop_gradient(copy(concatenate(
                {lw->moe_shared_gate_rhs, lw->moe_shared_up_rhs},
                1
            )));
        }
    }

    if (lw->moe_shared_gate_has_q &&
        lw->moe_shared_up_has_q &&
        lw->moe_shared_gate_q_w.ndim() == 2 &&
        lw->moe_shared_up_q_w.ndim() == 2 &&
        lw->moe_shared_gate_q_w.shape(1) == lw->moe_shared_up_q_w.shape(1) &&
        lw->moe_shared_gate_q_scales.ndim() == 2 &&
        lw->moe_shared_up_q_scales.ndim() == 2 &&
        lw->moe_shared_gate_q_scales.shape(1) == lw->moe_shared_up_q_scales.shape(1) &&
        lw->moe_shared_gate_q_biases.ndim() == 2 &&
        lw->moe_shared_up_q_biases.ndim() == 2 &&
        lw->moe_shared_gate_q_biases.shape(1) == lw->moe_shared_up_q_biases.shape(1))
    {
        lw->moe_shared_gate_up_q_w = stop_gradient(copy(concatenate(
            {lw->moe_shared_gate_q_w, lw->moe_shared_up_q_w},
            0
        )));
        lw->moe_shared_gate_up_q_scales = stop_gradient(copy(concatenate(
            {lw->moe_shared_gate_q_scales, lw->moe_shared_up_q_scales},
            0
        )));
        lw->moe_shared_gate_up_q_biases = stop_gradient(copy(concatenate(
            {lw->moe_shared_gate_q_biases, lw->moe_shared_up_q_biases},
            0
        )));
        lw->moe_shared_gate_up_has_q = true;
    }

    return lw->moe_shared_gate_up_has_q || lw->moe_shared_gate_up_rhs.size() > 1;
}

bool combine_dense_mlp_gate_up_weights(LayerWeights* lw) {
    if (lw == nullptr) return false;

    if (lw->mlp_gate_width <= 0) {
        if (lw->mlp_gate_has_q && lw->mlp_gate_q_w.ndim() == 2) {
            lw->mlp_gate_width = lw->mlp_gate_q_w.shape(0);
        } else if (lw->mlp_gate_rhs.ndim() == 2) {
            lw->mlp_gate_width = lw->mlp_gate_rhs.shape(1);
        }
    }

    if (lw->mlp_gate_rhs.size() > 1 && lw->mlp_up_rhs.size() > 1) {
        if (lw->mlp_gate_rhs.ndim() == 2 &&
            lw->mlp_up_rhs.ndim() == 2 &&
            lw->mlp_gate_rhs.shape(0) == lw->mlp_up_rhs.shape(0))
        {
            lw->mlp_gate_up_rhs = stop_gradient(copy(concatenate(
                {lw->mlp_gate_rhs, lw->mlp_up_rhs},
                1
            )));
        }
    }

    if (lw->mlp_gate_has_q &&
        lw->mlp_up_has_q &&
        lw->mlp_gate_q_w.ndim() == 2 &&
        lw->mlp_up_q_w.ndim() == 2 &&
        lw->mlp_gate_q_w.shape(1) == lw->mlp_up_q_w.shape(1) &&
        lw->mlp_gate_q_scales.ndim() == 2 &&
        lw->mlp_up_q_scales.ndim() == 2 &&
        lw->mlp_gate_q_scales.shape(1) == lw->mlp_up_q_scales.shape(1) &&
        lw->mlp_gate_q_biases.ndim() == 2 &&
        lw->mlp_up_q_biases.ndim() == 2 &&
        lw->mlp_gate_q_biases.shape(1) == lw->mlp_up_q_biases.shape(1))
    {
        lw->mlp_gate_up_q_w = stop_gradient(copy(concatenate(
            {lw->mlp_gate_q_w, lw->mlp_up_q_w},
            0
        )));
        lw->mlp_gate_up_q_scales = stop_gradient(copy(concatenate(
            {lw->mlp_gate_q_scales, lw->mlp_up_q_scales},
            0
        )));
        lw->mlp_gate_up_q_biases = stop_gradient(copy(concatenate(
            {lw->mlp_gate_q_biases, lw->mlp_up_q_biases},
            0
        )));
        lw->mlp_gate_up_has_q = true;
    }

    return lw->mlp_gate_up_has_q || lw->mlp_gate_up_rhs.size() > 1;
}

void compact_dense_mlp_gate_up_weights(LayerWeights* lw) {
    if (lw == nullptr) return;
    if (lw->mlp_gate_up_rhs.size() <= 1 && !lw->mlp_gate_up_has_q) return;
    if (lw->mlp_gate_width <= 0) return;

    lw->mlp_gate_rhs = array(0.0f);
    lw->mlp_up_rhs = array(0.0f);
    lw->mlp_gate_q_w = array(0.0f);
    lw->mlp_gate_q_scales = array(0.0f);
    lw->mlp_gate_q_biases = array(0.0f);
    lw->mlp_up_q_w = array(0.0f);
    lw->mlp_up_q_scales = array(0.0f);
    lw->mlp_up_q_biases = array(0.0f);
    lw->mlp_gate_has_q = false;
    lw->mlp_up_has_q = false;
}

bool combine_full_attention_qkv_weights(LayerWeights* lw) {
    if (lw == nullptr) return false;

    if (lw->attn_q_proj_width <= 0) {
        if (lw->attn_q_has_q && lw->attn_q_q_w.ndim() == 2) {
            lw->attn_q_proj_width = lw->attn_q_q_w.shape(0);
        } else if (lw->attn_q_rhs.ndim() == 2) {
            lw->attn_q_proj_width = lw->attn_q_rhs.shape(1);
        }
    }
    if (lw->attn_k_proj_width <= 0) {
        if (lw->attn_k_has_q && lw->attn_k_q_w.ndim() == 2) {
            lw->attn_k_proj_width = lw->attn_k_q_w.shape(0);
        } else if (lw->attn_k_rhs.ndim() == 2) {
            lw->attn_k_proj_width = lw->attn_k_rhs.shape(1);
        }
    }
    if (lw->attn_v_proj_width <= 0) {
        if (lw->attn_v_has_q && lw->attn_v_q_w.ndim() == 2) {
            lw->attn_v_proj_width = lw->attn_v_q_w.shape(0);
        } else if (lw->attn_v_rhs.ndim() == 2) {
            lw->attn_v_proj_width = lw->attn_v_rhs.shape(1);
        }
    }

    if (lw->attn_q_rhs.size() > 1 && lw->attn_k_rhs.size() > 1 && lw->attn_v_rhs.size() > 1) {
        if (lw->attn_q_rhs.ndim() == 2 &&
            lw->attn_k_rhs.ndim() == 2 &&
            lw->attn_v_rhs.ndim() == 2 &&
            lw->attn_q_rhs.shape(0) == lw->attn_k_rhs.shape(0) &&
            lw->attn_q_rhs.shape(0) == lw->attn_v_rhs.shape(0))
        {
            lw->attn_qkv_rhs = stop_gradient(copy(concatenate(
                {lw->attn_q_rhs, lw->attn_k_rhs, lw->attn_v_rhs},
                1
            )));
        }
    }

    if (lw->attn_q_has_q &&
        lw->attn_k_has_q &&
        lw->attn_v_has_q &&
        lw->attn_q_q_w.ndim() == 2 &&
        lw->attn_k_q_w.ndim() == 2 &&
        lw->attn_v_q_w.ndim() == 2 &&
        lw->attn_q_q_w.shape(1) == lw->attn_k_q_w.shape(1) &&
        lw->attn_q_q_w.shape(1) == lw->attn_v_q_w.shape(1) &&
        lw->attn_q_q_scales.ndim() == 2 &&
        lw->attn_k_q_scales.ndim() == 2 &&
        lw->attn_v_q_scales.ndim() == 2 &&
        lw->attn_q_q_scales.shape(1) == lw->attn_k_q_scales.shape(1) &&
        lw->attn_q_q_scales.shape(1) == lw->attn_v_q_scales.shape(1) &&
        lw->attn_q_q_biases.ndim() == 2 &&
        lw->attn_k_q_biases.ndim() == 2 &&
        lw->attn_v_q_biases.ndim() == 2 &&
        lw->attn_q_q_biases.shape(1) == lw->attn_k_q_biases.shape(1) &&
        lw->attn_q_q_biases.shape(1) == lw->attn_v_q_biases.shape(1))
    {
        lw->attn_qkv_q_w = stop_gradient(copy(concatenate(
            {lw->attn_q_q_w, lw->attn_k_q_w, lw->attn_v_q_w},
            0
        )));
        lw->attn_qkv_q_scales = stop_gradient(copy(concatenate(
            {lw->attn_q_q_scales, lw->attn_k_q_scales, lw->attn_v_q_scales},
            0
        )));
        lw->attn_qkv_q_biases = stop_gradient(copy(concatenate(
            {lw->attn_q_q_biases, lw->attn_k_q_biases, lw->attn_v_q_biases},
            0
        )));
        lw->attn_qkv_has_q = true;
    }

    return lw->attn_qkv_has_q || lw->attn_qkv_rhs.size() > 1;
}

void compact_full_attention_qkv_weights(LayerWeights* lw) {
    if (lw == nullptr) return;
    if (lw->attn_qkv_rhs.size() <= 1 && !lw->attn_qkv_has_q) return;
    if (lw->attn_q_proj_width <= 0 || lw->attn_k_proj_width <= 0 || lw->attn_v_proj_width <= 0) return;

    lw->attn_q_rhs = array(0.0f);
    lw->attn_k_rhs = array(0.0f);
    lw->attn_v_rhs = array(0.0f);
    lw->attn_q_q_w = array(0.0f);
    lw->attn_q_q_scales = array(0.0f);
    lw->attn_q_q_biases = array(0.0f);
    lw->attn_k_q_w = array(0.0f);
    lw->attn_k_q_scales = array(0.0f);
    lw->attn_k_q_biases = array(0.0f);
    lw->attn_v_q_w = array(0.0f);
    lw->attn_v_q_scales = array(0.0f);
    lw->attn_v_q_biases = array(0.0f);
    lw->attn_q_has_q = false;
    lw->attn_k_has_q = false;
    lw->attn_v_has_q = false;
}

array run_linear_attention_layer_reference(mlx_ctx* ctx, int layer_idx, const array& x_norm) {
    LayerWeights& lw = ctx->layers[static_cast<size_t>(layer_idx)];
    LinearCacheState& cache = ctx->linear_cache[static_cast<size_t>(layer_idx)];

    const int B = x_norm.shape(0);
    const int S = x_norm.shape(1);

    const bool prefill_only_qmm = false;
    // Use combined in_proj when available (one matmul instead of two).
    const bool use_combined_in_proj =
        (lw.lin_in_proj_all_has_q || lw.lin_in_proj_all_rhs.size() > 1);
    const int qkvz_cols = lw.lin_conv_dim + lw.lin_value_dim;
    const int ba_cols = 2 * lw.lin_num_v_heads;
    array qkvz = array(0.0f);
    array ba = array(0.0f);
    if (use_combined_in_proj) {
        const array proj_all = linear_decode_maybe_quantized(
            ctx->fp8_decode_qmm_enabled,
            ctx->cfg.quant_group_size,
            ctx->cfg.quant_bits,
            x_norm,
            lw.lin_in_proj_all_rhs,
            lw.lin_in_proj_all_has_q,
            lw.lin_in_proj_all_q_w,
            lw.lin_in_proj_all_q_scales,
            lw.lin_in_proj_all_q_biases,
            prefill_only_qmm
        );
        qkvz = slice(proj_all, {0, 0, 0}, {B, S, qkvz_cols});
        ba = slice(proj_all, {0, 0, qkvz_cols}, {B, S, qkvz_cols + ba_cols});
    } else {
        qkvz = linear_decode_maybe_quantized(
            ctx->fp8_decode_qmm_enabled,
            ctx->cfg.quant_group_size,
            ctx->cfg.quant_bits,
            x_norm,
            lw.lin_in_proj_qkvz_rhs,
            lw.lin_in_proj_qkvz_has_q,
            lw.lin_in_proj_qkvz_q_w,
            lw.lin_in_proj_qkvz_q_scales,
            lw.lin_in_proj_qkvz_q_biases,
            prefill_only_qmm,
            -1,
            nullptr,
            &lw.lin_in_proj_qkvz_prefill_rhs
        );
        ba = linear_decode_maybe_quantized(
            ctx->fp8_decode_qmm_enabled,
            ctx->cfg.quant_group_size,
            ctx->cfg.quant_bits,
            x_norm,
            lw.lin_in_proj_ba_rhs,
            lw.lin_in_proj_ba_has_q,
            lw.lin_in_proj_ba_q_w,
            lw.lin_in_proj_ba_q_scales,
            lw.lin_in_proj_ba_q_biases,
            prefill_only_qmm
        );
    }
    const array qkv = slice(qkvz, {0, 0, 0}, {B, S, lw.lin_conv_dim});
    const array z_flat = slice(qkvz, {0, 0, lw.lin_conv_dim}, {B, S, lw.lin_conv_dim + lw.lin_value_dim});
    const array z = reshape(z_flat, {B, S, lw.lin_num_v_heads, lw.lin_head_v_dim});
    const array b = slice(ba, {0, 0, 0}, {B, S, lw.lin_num_v_heads});
    const array a = slice(ba, {0, 0, lw.lin_num_v_heads}, {B, S, 2 * lw.lin_num_v_heads});

    if (!cache.conv_state.has_value()) {
        cache.conv_state = zeros({B, lw.lin_conv_kernel - 1, lw.lin_conv_dim}, x_norm.dtype());
    }
    if (!cache.state.has_value()) {
        cache.state = zeros({B, lw.lin_num_v_heads, lw.lin_head_v_dim, lw.lin_head_k_dim}, x_norm.dtype());
    }

    array conv_out = array(0.0f);
    bool conv_silu_fused = false;
    if (S == 1 && lw.lin_conv1d_decode_w.size() > 1) {
        auto [step_out, next_state] = depthwise_conv1d_decode_step(
            cache.conv_state,
            qkv,
            lw.lin_conv1d_decode_w,
            lw.lin_conv_kernel,
            lw.lin_conv_dim,
            "linear_attn.conv1d"
        );
        conv_out = step_out;
        // The fused Metal kernel already applied SiLU when B==1 and
        // kernel_size > 1. Only the fallback path needs a separate silu.
        conv_silu_fused = (B == 1 && lw.lin_conv_kernel > 1);
        if (next_state.has_value()) {
            cache.conv_state = *next_state;
        } else {
            cache.conv_state.reset();
        }
    } else {
        const array conv_input = concatenate({*cache.conv_state, qkv}, 1);
        cache.conv_state = slice(
            conv_input,
            {0, S, 0},
            {B, S + lw.lin_conv_kernel - 1, lw.lin_conv_dim}
        );
        conv_out = conv1d(conv_input, lw.lin_conv1d_w, 1, 0, 1, lw.lin_conv_dim);
    }
    if (!conv_silu_fused) {
        conv_out = silu(conv_out);
    }

    const array q_flat = slice(conv_out, {0, 0, 0}, {B, S, lw.lin_key_dim});
    const array k_flat = slice(conv_out, {0, 0, lw.lin_key_dim}, {B, S, 2 * lw.lin_key_dim});
    const array v_flat = slice(conv_out, {0, 0, 2 * lw.lin_key_dim}, {B, S, 2 * lw.lin_key_dim + lw.lin_value_dim});

    array q = reshape(q_flat, {B, S, lw.lin_num_k_heads, lw.lin_head_k_dim});
    array k = reshape(k_flat, {B, S, lw.lin_num_k_heads, lw.lin_head_k_dim});
    const array v = reshape(v_flat, {B, S, lw.lin_num_v_heads, lw.lin_head_v_dim});

    // For S=1 decode, the fused SSM kernel handles Q/K norms internally.
    // For prefill (S>1), normalize on the host side.
    if (S > 1) {
        const float inv_scale = 1.0f / std::sqrt(static_cast<float>(lw.lin_head_k_dim));
        q = fast::rms_norm(q, std::nullopt, 1.0e-6f) * array(inv_scale * inv_scale, q.dtype());
        k = fast::rms_norm(k, std::nullopt, 1.0e-6f) * array(inv_scale, k.dtype());
    }
    // gated_delta_update dispatches the fused kernel for T=1 (raw Q/K)
    // or the base kernel for T>1 (pre-normalized Q/K).
    auto [out, new_state] = gated_delta_update(q, k, v, a, b, lw.lin_A_exp_f32, lw.lin_dt_bias, cache.state);
    cache.state = new_state;

    const array normed = rmsnorm_gated(out, z, lw.lin_norm_w, ctx->cfg.rms_norm_eps);
    const array merged = reshape(normed, {B, S, lw.lin_value_dim});
    return linear_decode_maybe_quantized(
        ctx->fp8_decode_qmm_enabled,
        ctx->cfg.quant_group_size,
        ctx->cfg.quant_bits,
        merged,
        lw.lin_out_proj_rhs,
        lw.lin_out_proj_has_q,
        lw.lin_out_proj_q_w,
        lw.lin_out_proj_q_scales,
        lw.lin_out_proj_q_biases,
        prefill_only_qmm
    );
}

array run_linear_attention_layer(mlx_ctx* ctx, int layer_idx, const array& x_norm) {
    return run_linear_attention_layer_reference(ctx, layer_idx, x_norm);
}

array run_shortconv_layer(mlx_ctx* ctx, int layer_idx, const array& x_norm) {
    LayerWeights& lw = ctx->layers[static_cast<size_t>(layer_idx)];
    LinearCacheState& cache = ctx->linear_cache[static_cast<size_t>(layer_idx)];

    const int B = x_norm.shape(0);
    const int S = x_norm.shape(1);

    const bool prefill_only_qmm = false;
    const array proj = linear_decode_maybe_quantized(
        ctx->fp8_decode_qmm_enabled,
        ctx->cfg.quant_group_size,
        ctx->cfg.quant_bits,
        x_norm,
        lw.sc_in_proj_rhs,
        lw.sc_in_proj_has_q,
        lw.sc_in_proj_q_w,
        lw.sc_in_proj_q_scales,
        lw.sc_in_proj_q_biases,
        prefill_only_qmm
    ); // [B,S,3*conv_dim]
    const array b_gate = slice(proj, {0, 0, 0}, {B, S, lw.sc_conv_dim});
    const array c_gate = slice(proj, {0, 0, lw.sc_conv_dim}, {B, S, 2 * lw.sc_conv_dim});
    const array x_proj = slice(proj, {0, 0, 2 * lw.sc_conv_dim}, {B, S, 3 * lw.sc_conv_dim});
    const array bx = b_gate * x_proj;

    if (!cache.conv_state.has_value()) {
        cache.conv_state = zeros({B, lw.sc_conv_kernel - 1, lw.sc_conv_dim}, x_norm.dtype());
    }

    const array conv_input = concatenate({*cache.conv_state, bx}, 1);
    cache.conv_state = slice(
        conv_input,
        {0, S, 0},
        {B, S + lw.sc_conv_kernel - 1, lw.sc_conv_dim}
    );

    array conv_out = conv1d(conv_input, lw.sc_conv1d_w, 1, 0, 1, lw.sc_conv_dim);
    if (lw.sc_has_bias) {
        conv_out = conv_out + reshape(lw.sc_conv_bias, {1, 1, lw.sc_conv_dim});
    }

    const array gated = c_gate * conv_out;
    return linear_decode_maybe_quantized(
        ctx->fp8_decode_qmm_enabled,
        ctx->cfg.quant_group_size,
        ctx->cfg.quant_bits,
        gated,
        lw.sc_out_proj_rhs,
        lw.sc_out_proj_has_q,
        lw.sc_out_proj_q_w,
        lw.sc_out_proj_q_scales,
        lw.sc_out_proj_q_biases,
        prefill_only_qmm
    );
}
