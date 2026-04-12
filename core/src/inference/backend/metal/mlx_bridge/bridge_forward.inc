// bridge_forward.inc — Compiled decode path, embedding lookup, forward_logits,
// fused LM head, chunked prefill, reset_runtime_state.

bool can_use_compiled_decode(const mlx_ctx* ctx) {
    if (!ctx || ctx->layers.empty()) return false;
    if (ctx->per_layer_input_enabled) return false;
    if (ctx->cfg.hidden_size_per_layer_input > 0) return false;
    for (const auto& lw : ctx->layers) {
        if (lw.is_linear || lw.is_shortconv) return false;
        if (lw.mlp_is_moe) return false;
        if (lw.kv_shared_source_layer >= 0) return false;
        if (lw.attn_is_sliding) return false;
        if (lw.rope_freq.size() <= 1) return false;
    }
    return true;
}

using CompiledDecodeFn = std::function<std::vector<array>(const std::vector<array>&)>;

CompiledDecodeFn build_compiled_decode_core(mlx_ctx* ctx) {
    const int num_layers = static_cast<int>(ctx->layers.size());
    const float eps = ctx->cfg.rms_norm_eps;
    const int group_size = ctx->cfg.quant_group_size;
    const int bits = ctx->cfg.quant_bits;
    const bool use_gelu = ctx->cfg.use_gelu;
    const bool use_v_norm = ctx->cfg.use_v_norm;
    const int global_head_dim = ctx->cfg.global_head_dim;

    return compile(
        [ctx, num_layers, eps, group_size, bits, use_gelu, use_v_norm, global_head_dim](
            const std::vector<array>& inputs
        ) {
            array hidden = inputs[0];
            const array& offset_arr = inputs[1];
            const array& attn_mask = inputs[2];

            std::vector<array> new_kvs;
            new_kvs.reserve(static_cast<size_t>(num_layers) * 2);

            for (int li = 0; li < num_layers; ++li) {
                const LayerWeights& lw = ctx->layers[static_cast<size_t>(li)];
                const array& cache_k = inputs[static_cast<size_t>(3 + li * 2)];
                const array& cache_v = inputs[static_cast<size_t>(3 + li * 2 + 1)];

                // --- input norm ---
                const array x_norm = fast::rms_norm(
                    hidden, std::optional<array>(lw.input_layernorm_w), eps);

                // --- QKV projection ---
                const array qkv = lw.attn_qkv_has_q
                    ? quantized_matmul(x_norm, lw.attn_qkv_q_w, lw.attn_qkv_q_scales,
                          std::optional<array>(lw.attn_qkv_q_biases), true, group_size, bits, "affine")
                    : matmul(x_norm, lw.attn_qkv_rhs);
                const int q_w = lw.attn_q_proj_width;
                const int k_w = lw.attn_k_proj_width;
                const int v_w = lw.attn_v_proj_width;
                const array q_proj = slice(qkv, {0, 0, 0}, {1, 1, q_w});
                const array k_proj = slice(qkv, {0, 0, q_w}, {1, 1, q_w + k_w});
                const array v_proj = slice(qkv, {0, 0, q_w + k_w}, {1, 1, q_w + k_w + v_w});

                // --- Q/K/V reshape + norm ---
                const int q_base = lw.attn_num_heads * lw.attn_head_dim;
                const bool has_gate = (q_w == 2 * q_base);
                array queries_raw = array(0.0f);
                array gate = array(0.0f);
                if (has_gate) {
                    const array q_4d = reshape(q_proj, {1, 1, lw.attn_num_heads, 2 * lw.attn_head_dim});
                    queries_raw = slice(q_4d, {0,0,0,0}, {1,1,lw.attn_num_heads,lw.attn_head_dim});
                    gate = reshape(
                        slice(q_4d, {0,0,0,lw.attn_head_dim}, {1,1,lw.attn_num_heads,2*lw.attn_head_dim}),
                        {1, 1, q_base});
                } else {
                    queries_raw = reshape(q_proj, {1, 1, lw.attn_num_heads, lw.attn_head_dim});
                }

                array queries = fast::rms_norm(queries_raw, std::optional<array>(lw.attn_q_norm_w), eps);
                array keys = reshape(k_proj, {1, 1, lw.attn_num_kv_heads, lw.attn_head_dim});
                keys = fast::rms_norm(keys, std::optional<array>(lw.attn_k_norm_w), eps);
                array values = reshape(v_proj, {1, 1, lw.attn_num_kv_heads, lw.attn_head_dim});
                if (use_v_norm) {
                    values = fast::rms_norm(values, std::nullopt, eps);
                }

                queries = transpose(queries, {0, 2, 1, 3});
                keys = transpose(keys, {0, 2, 1, 3});
                values = transpose(values, {0, 2, 1, 3});

                // --- array-based RoPE ---
                const int rope_dims = lw.attn_rope_dims;
                const bool prop_rope = global_head_dim > 0 &&
                    lw.attn_head_dim == global_head_dim &&
                    rope_dims > 0 && rope_dims < lw.attn_head_dim;
                if (prop_rope) {
                    queries = apply_rope_proportional_array(
                        queries, lw.rope_freq, offset_arr, rope_dims, lw.attn_head_dim);
                    keys = apply_rope_proportional_array(
                        keys, lw.rope_freq, offset_arr, rope_dims, lw.attn_head_dim);
                } else {
                    queries = apply_rope_array(queries, lw.rope_freq, offset_arr, rope_dims);
                    keys = apply_rope_array(keys, lw.rope_freq, offset_arr, rope_dims);
                }

                // Emit new K/V for host-side cache commit.
                new_kvs.push_back(keys);
                new_kvs.push_back(values);

                // --- SDPA over concat(cache, new_kv) with mask ---
                const array sdpa_keys = concatenate({cache_k, keys}, 2);
                const array sdpa_vals = concatenate({cache_v, values}, 2);
                array attn_out = fast::scaled_dot_product_attention(
                    queries, sdpa_keys, sdpa_vals, lw.attn_scale, "",
                    std::optional<array>(attn_mask));

                attn_out = transpose(attn_out, {0, 2, 1, 3});
                attn_out = reshape(attn_out, {1, 1, lw.attn_num_heads * lw.attn_head_dim});
                if (has_gate) {
                    attn_out = attn_out * sigmoid(gate);
                }

                // --- O projection ---
                const array o_out = lw.attn_o_has_q
                    ? quantized_matmul(attn_out, lw.attn_o_q_w, lw.attn_o_q_scales,
                          std::optional<array>(lw.attn_o_q_biases), true, group_size, bits, "affine")
                    : matmul(attn_out, lw.attn_o_rhs);

                // --- residual + post-attention norm ---
                const array h = hidden + o_out;
                const array mlp_in = fast::rms_norm(
                    h, std::optional<array>(lw.post_attention_layernorm_w), eps);

                // --- MLP: gate_up + silu/gelu + down ---
                const array gate_up = lw.mlp_gate_up_has_q
                    ? quantized_matmul(mlp_in, lw.mlp_gate_up_q_w, lw.mlp_gate_up_q_scales,
                          std::optional<array>(lw.mlp_gate_up_q_biases), true, group_size, bits, "affine")
                    : matmul(mlp_in, lw.mlp_gate_up_rhs);
                const int gw = lw.mlp_gate_width;
                const array mlp_gate = slice(gate_up, {0, 0, 0}, {1, 1, gw});
                const array mlp_up = slice(gate_up, {0, 0, gw}, {1, 1, 2 * gw});
                const array ff = (use_gelu ? gelu_approx(mlp_gate) : silu(mlp_gate)) * mlp_up;

                const array down = lw.mlp_down_has_q
                    ? quantized_matmul(ff, lw.mlp_down_q_w, lw.mlp_down_q_scales,
                          std::optional<array>(lw.mlp_down_q_biases), true, group_size, bits, "affine")
                    : matmul(ff, lw.mlp_down_rhs);

                hidden = h + down;
            }

            // Final norm.
            hidden = fast::rms_norm(hidden, std::optional<array>(ctx->final_norm_w), eps);

            // Pack outputs: [hidden, new_k_0, new_v_0, ..., new_k_N-1, new_v_N-1]
            std::vector<array> outputs;
            outputs.reserve(1 + new_kvs.size());
            outputs.push_back(hidden);
            for (auto& kv : new_kvs) {
                outputs.push_back(std::move(kv));
            }
            return outputs;
        },
        false // shape-specific: traces once for S=1, caches
    );
}

int embedding_vocab_size(const mlx_ctx* ctx) {
    if (ctx->embed_tokens_has_q && ctx->embed_tokens_q_w.ndim() == 2) {
        return ctx->embed_tokens_q_w.shape(0);
    }
    if (ctx->embed_tokens.ndim() == 2) {
        return ctx->embed_tokens.shape(0);
    }
    return 0;
}

int embedding_hidden_size(const mlx_ctx* ctx) {
    if (ctx->embed_tokens_has_q && ctx->embed_tokens_q_w.ndim() == 2) {
        const int bits = ctx->embed_tokens_q_bits > 0 ? ctx->embed_tokens_q_bits : ctx->cfg.quant_bits;
        if (bits == 4 || bits == 8) {
            return (ctx->embed_tokens_q_w.shape(1) * 32) / bits;
        }
        return 0;
    }
    if (ctx->embed_tokens.ndim() == 2) {
        return ctx->embed_tokens.shape(1);
    }
    return 0;
}

array embedding_lookup(mlx_ctx* ctx, const array& ids) {
    if (!ctx->embed_tokens_has_q) {
        return take(ctx->embed_tokens, ids, 0);
    }
    const int bits = ctx->embed_tokens_q_bits > 0 ? ctx->embed_tokens_q_bits : ctx->cfg.quant_bits;
    if ((bits != 4 && bits != 8) || ctx->cfg.quant_group_size <= 0) {
        throw std::runtime_error("quantized embedding lookup: invalid quantization parameters");
    }
    const int hidden_size = embedding_hidden_size(ctx);
    if (hidden_size <= 0) {
        throw std::runtime_error("quantized embedding lookup: invalid hidden size");
    }
    const int rank = ids.ndim();
    if (rank <= 0) {
        throw std::runtime_error("quantized embedding lookup: ids must have rank >= 1");
    }
    Shape out_shape;
    out_shape.reserve(static_cast<size_t>(rank) + 1);
    for (int axis = 0; axis < rank; ++axis) {
        out_shape.push_back(ids.shape(axis));
    }
    out_shape.push_back(hidden_size);
    const array flat_ids = reshape(astype(ids, int32), {static_cast<int>(ids.size())});
    const array q_w = take(ctx->embed_tokens_q_w, flat_ids, 0);
    const array q_scales = take(ctx->embed_tokens_q_scales, flat_ids, 0);
    const array q_biases = take(ctx->embed_tokens_q_biases, flat_ids, 0);
    const array dequant = dequantize(
        q_w,
        q_scales,
        std::optional<array>(q_biases),
        ctx->cfg.quant_group_size,
        bits,
        "affine",
        std::nullopt,
        bfloat16
    );
    return reshape(dequant, out_shape);
}

array forward_hidden(
    mlx_ctx* ctx,
    const array& input_ids,
    bool apply_final_norm,
    const TraceFrame* trace_frame = nullptr
) {
    if (embedding_vocab_size(ctx) <= 0) {
        throw std::runtime_error("forward_hidden: embedding table is not initialized");
    }
    array hidden = embedding_lookup(ctx, input_ids); // [B,S,H]
    if (std::fabs(ctx->cfg.embedding_multiplier - 1.0f) > 1.0e-6f) {
        hidden = hidden * array(ctx->cfg.embedding_multiplier, hidden.dtype());
    }
    const array source_embeddings = hidden;
    if (trace_frame && trace_frame->emit_embed) {
        xray_emit_array_f32(
            XRAY_POINT_EMBED,
            XRAY_GLOBAL_LAYER,
            trace_frame->token,
            trace_frame->layer_position,
            hidden,
            "mlx_embed_tokens_host"
        );
    }

    if (!ctx->profile_layers) {
        const int n_layers = static_cast<int>(ctx->layers.size());
        for (int i = 0; i < n_layers; ++i) {
            hidden = run_layer(ctx, i, hidden, source_embeddings, input_ids, trace_frame);
            if (i < n_layers - 1) {
                async_eval(hidden);
            }
        }
    } else {
        for (int i = 0; i < static_cast<int>(ctx->layers.size()); ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            hidden = run_layer(ctx, i, hidden, source_embeddings, input_ids, trace_frame);
            eval(hidden);
            synchronize();
            const auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            const LayerWeights& lw = ctx->layers[static_cast<size_t>(i)];
            const char* kind = lw.is_linear ? "linear" : (lw.is_shortconv ? "shortconv" : "full");
            std::fprintf(stderr, "[mlx][layer %d][%s] %.3f ms\n", i, kind, ms);
        }
    }

    if (apply_final_norm) {
        hidden = fast::rms_norm(hidden, std::optional<array>(ctx->final_norm_w), ctx->cfg.rms_norm_eps);
        if (trace_frame) {
            xray_emit_array_f32(
                XRAY_POINT_FINAL_NORM,
                XRAY_GLOBAL_LAYER,
                trace_frame->token,
                1,
                hidden,
                "mlx_final_norm_host"
            );
        }
    }
    return hidden;
}

array forward_logits(mlx_ctx* ctx, const array& input_ids, const TraceFrame* trace_frame = nullptr) {
    const array hidden = forward_hidden(ctx, input_ids, true, trace_frame);
    array lm_input = hidden;
    if ((ctx->has_fp8_meta or ctx->has_grouped_affine_meta) &&
        hidden.ndim() == 3 &&
        hidden.shape(1) > 1 &&
        ctx->cfg.hidden_size_per_layer_input <= 0)
    {
        // Prefill only needs next-token logits; avoid full-sequence lm_head.
        const int b = hidden.shape(0);
        const int t = hidden.shape(1);
        const int h = hidden.shape(2);
        lm_input = slice(hidden, {0, t - 1, 0}, {b, t, h});
    }
    const bool lm_head_q_ready = ctx->lm_head_q_decode_enabled &&
        ctx->lm_head_q_w.size() > 0 && ctx->lm_head_q_scales.size() > 0;
    const int lm_bits = ctx->lm_head_q_bits > 0 ? ctx->lm_head_q_bits : ctx->cfg.quant_bits;
    const bool lm_head_affine_ready = lm_head_q_ready &&
        ctx->lm_head_q_biases.ndim() == 2 &&
        ctx->lm_head_q_biases.size() > 1 &&
        ctx->cfg.quant_group_size > 0 &&
        (lm_bits == 4 || lm_bits == 8);
    const bool lm_head_nvfp4_ready =
        lm_head_q_ready &&
        !lm_head_affine_ready &&
        ctx->lm_head_q_w.dtype() == uint32 &&
        ctx->lm_head_q_scales.dtype() == uint8 &&
        ctx->lm_head_q_biases.size() == 1 &&
        ctx->lm_head_q_biases.dtype() == float32 &&
        ctx->cfg.quant_group_size == 16 &&
        ctx->cfg.quant_bits == 4;
    array logits = lm_head_q_ready
        ? (lm_head_affine_ready
              ? quantized_matmul(
                    lm_input,
                    ctx->lm_head_q_w,
                    ctx->lm_head_q_scales,
                    std::optional<array>(ctx->lm_head_q_biases),
                    true,
                    ctx->cfg.quant_group_size,
                    lm_bits,
                    "affine"
                )
              : (lm_head_nvfp4_ready
                    ? reshape(
                          qqmm(
                              reshape(lm_input, {lm_input.shape(0), lm_input.shape(2)}),
                              ctx->lm_head_q_w,
                              std::optional<array>(ctx->lm_head_q_scales),
                              ctx->cfg.quant_group_size,
                              ctx->cfg.quant_bits,
                              "nvfp4",
                              std::optional<array>(nvfp4_unit_global_scale()),
                              std::optional<array>(ctx->lm_head_q_biases)
                          ),
                          {lm_input.shape(0), 1, ctx->lm_head_q_w.shape(0)}
                      )
              : quantized_matmul(
                    lm_input,
                    ctx->lm_head_q_w,
                    ctx->lm_head_q_scales,
                    std::nullopt,
                    true,
                    std::nullopt,
                    std::nullopt,
                    "mxfp8"
                )))
        : ((ctx->cfg.hidden_size_per_layer_input > 0 && ctx->lm_head_rhs.dtype() == float32)
              ? matmul(astype(lm_input, float32), ctx->lm_head_rhs)
              : matmul(lm_input, ctx->lm_head_rhs));
    if (ctx->cfg.final_logit_softcapping > 0.0f) {
        const array softcap = array(ctx->cfg.final_logit_softcapping, logits.dtype());
        logits = tanh(logits / softcap) * softcap;
    }
    if (trace_frame) {
        xray_emit_array_f32(
            XRAY_POINT_LM_HEAD,
            XRAY_GLOBAL_LAYER,
            trace_frame->token,
            0,
            last_token_logits(logits),
            "mlx_lm_head_host"
        );
    }
    return logits;
}

// Apply LM head to an already-normed hidden state [1,1,H].
// Used by the compiled decode path where the compiled core handles
// all layers + final norm, and this function adds just the lm_head.
array forward_logits_from_hidden(mlx_ctx* ctx, const array& normed_hidden) {
    const bool lm_head_q_ready = ctx->lm_head_q_decode_enabled &&
        ctx->lm_head_q_w.size() > 0 && ctx->lm_head_q_scales.size() > 0;
    const int lm_bits = ctx->lm_head_q_bits > 0 ? ctx->lm_head_q_bits : ctx->cfg.quant_bits;
    const bool lm_head_affine = lm_head_q_ready &&
        ctx->lm_head_q_biases.ndim() == 2 && ctx->lm_head_q_biases.size() > 1 &&
        ctx->cfg.quant_group_size > 0 && (lm_bits == 4 || lm_bits == 8);
    if (lm_head_affine) {
        return quantized_matmul(normed_hidden, ctx->lm_head_q_w, ctx->lm_head_q_scales,
            std::optional<array>(ctx->lm_head_q_biases), true,
            ctx->cfg.quant_group_size, lm_bits, "affine");
    }
    if (lm_head_q_ready) {
        return quantized_matmul(normed_hidden, ctx->lm_head_q_w, ctx->lm_head_q_scales,
            std::nullopt, true, std::nullopt, std::nullopt, "mxfp8");
    }
    return matmul(normed_hidden, ctx->lm_head_rhs);
}

bool can_use_fused_lm_head_argmax(const mlx_ctx* ctx, const TraceFrame* trace_frame = nullptr) {
    return ctx &&
        env_truthy("TALU_METAL_FUSED_LM_HEAD_ARGMAX", false) &&
        ctx->lm_head_q_decode_enabled &&
        ctx->lm_head_q_w.size() > 0 &&
        ctx->lm_head_q_scales.size() > 0 &&
        ctx->lm_head_q_biases.ndim() == 2 &&
        ctx->lm_head_q_biases.size() > 1 &&
        ctx->cfg.quant_group_size > 0 &&
        ctx->cfg.quant_bits == 4 &&
        trace_frame == nullptr;
}

bool can_use_fused_lm_head_topk(
    const mlx_ctx* ctx,
    int requested_top_k,
    const TraceFrame* trace_frame = nullptr
) {
    return can_use_fused_lm_head_argmax(ctx, trace_frame) &&
        env_truthy("TALU_METAL_FUSED_LM_HEAD_TOPK", false) &&
        requested_top_k > 1 &&
        requested_top_k <= 64;
}

int resolve_fused_lm_head_row_chunk() {
    const char* raw = std::getenv("TALU_METAL_FUSED_LM_HEAD_ROWS");
    if (!raw || raw[0] == '\0') return 8192;
    char* end = nullptr;
    const long parsed = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
        return 8192;
    }
    return static_cast<int>(parsed);
}

array slice_rows_2d(const array& input, int row_start, int row_end) {
    return slice(input, {row_start, 0}, {row_end, input.shape(1)});
}

[[noreturn]] void throw_with_context(const char* stage, const std::exception& e) {
    throw std::runtime_error(std::string(stage) + ": " + e.what());
}

std::pair<array, array> fused_lm_head_topk_candidate_arrays(
    mlx_ctx* ctx,
    const array& input_ids,
    int requested_top_k
);

array fused_lm_head_argmax_token(mlx_ctx* ctx, const array& input_ids) {
    if (!can_use_fused_lm_head_argmax(ctx)) {
        throw std::runtime_error("fused_lm_head_argmax_token called without supported lm_head");
    }

    auto [candidate_logits, candidate_ids] = fused_lm_head_topk_candidate_arrays(ctx, input_ids, 1);
    g_fused_lm_head_argmax_hits += 1;
    const array candidate_logits_3d = reshape(candidate_logits, {1, 1, candidate_logits.shape(0)});
    const array candidate_ids_3d = reshape(candidate_ids, {1, 1, candidate_ids.shape(0)});
    const array best_local = reshape(astype(argmax(candidate_logits_3d, -1), int32), {1, 1, 1});
    const array next_token_3d = take_along_axis(candidate_ids_3d, best_local, -1);
    return reshape(astype(next_token_3d, int32), {1, 1});
}

TopKCandidateBatch fused_lm_head_topk_candidates(
    mlx_ctx* ctx,
    const array& input_ids,
    int requested_top_k
) {
    if (!can_use_fused_lm_head_topk(ctx, requested_top_k)) {
        throw std::runtime_error("fused_lm_head_topk_candidates called without supported lm_head");
    }

    auto [candidate_logits_flat, candidate_ids_flat] =
        fused_lm_head_topk_candidate_arrays(ctx, input_ids, requested_top_k);

    const int candidate_count = candidate_logits_flat.shape(0);
    const array candidate_logits = reshape(candidate_logits_flat, {1, 1, candidate_count});
    const array candidate_ids = reshape(candidate_ids_flat, {1, 1, candidate_count});

    auto topk_fn = compiled_topk_candidates(requested_top_k);
    const std::vector<array> topk_outputs = (*topk_fn)({candidate_logits});
    if (topk_outputs.size() != 2) {
        throw std::runtime_error("fused_lm_head_topk_candidates: invalid top-k output");
    }

    const array topk_logits = reshape(topk_outputs[0], {1, 1, requested_top_k});
    const array candidate_local_indices = reshape(topk_outputs[1], {1, 1, requested_top_k});
    const array topk_ids = take_along_axis(candidate_ids, candidate_local_indices, -1);
    return TopKCandidateBatch{
        .logits = topk_logits,
        .indices = astype(topk_ids, int32),
        .k = requested_top_k,
    };
}

std::pair<array, array> fused_lm_head_topk_candidate_arrays(
    mlx_ctx* ctx,
    const array& input_ids,
    int requested_top_k
) {
    const bool supported =
        (requested_top_k == 1)
            ? can_use_fused_lm_head_argmax(ctx)
            : can_use_fused_lm_head_topk(ctx, requested_top_k);
    if (!supported) {
        throw std::runtime_error("fused_lm_head_topk_candidate_arrays called without supported lm_head");
    }

    array hidden = array(0.0f);
    try {
        hidden = forward_hidden(ctx, input_ids, true);
    } catch (const std::exception& e) {
        throw_with_context("fused_lm_head_topk_candidate_arrays/forward_hidden", e);
    }
    if (hidden.ndim() != 3 || hidden.shape(1) != 1) {
        throw std::runtime_error("fused_lm_head_topk_candidate_arrays expects decode hidden shape [B,1,H]");
    }

    const int vocab_size = ctx->lm_head_q_w.shape(0);
    const int row_chunk = std::max(requested_top_k, resolve_fused_lm_head_row_chunk());
    std::vector<array> candidate_logits_parts;
    std::vector<array> candidate_ids_parts;
    candidate_logits_parts.reserve(static_cast<size_t>((vocab_size + row_chunk - 1) / row_chunk));
    candidate_ids_parts.reserve(candidate_logits_parts.capacity());

    for (int row_start = 0; row_start < vocab_size; row_start += row_chunk) {
        const int row_end = std::min(vocab_size, row_start + row_chunk);
        const int local_vocab = row_end - row_start;
        const int local_k = std::min(requested_top_k, local_vocab);
        std::vector<array> topk_outputs;
        try {
            const array w_chunk = slice_rows_2d(ctx->lm_head_q_w, row_start, row_end);
            const array scales_chunk = slice_rows_2d(ctx->lm_head_q_scales, row_start, row_end);
            const array biases_chunk = slice_rows_2d(ctx->lm_head_q_biases, row_start, row_end);
            const int lm_bits_chunk = ctx->lm_head_q_bits > 0 ? ctx->lm_head_q_bits : ctx->cfg.quant_bits;
            const array logits_chunk = quantized_matmul(
                hidden,
                w_chunk,
                scales_chunk,
                std::optional<array>(biases_chunk),
                true,
                ctx->cfg.quant_group_size,
                lm_bits_chunk,
                "affine"
            );
            auto topk_fn = compiled_topk_candidates(local_k);
            topk_outputs = (*topk_fn)({logits_chunk});
        } catch (const std::exception& e) {
            throw std::runtime_error(
                "fused_lm_head_topk_candidate_arrays/chunk row_start=" + std::to_string(row_start) +
                " row_end=" + std::to_string(row_end) + ": " + e.what()
            );
        }
        if (topk_outputs.size() != 2) {
            throw std::runtime_error("fused_lm_head_topk_candidate_arrays: invalid chunk top-k output");
        }
        const array row_offset = full(topk_outputs[1].shape(), row_start, int32);
        candidate_logits_parts.push_back(topk_outputs[0]);
        candidate_ids_parts.push_back(astype(topk_outputs[1], int32) + row_offset);
    }

    if (candidate_logits_parts.empty() || candidate_ids_parts.empty()) {
        throw std::runtime_error("fused_lm_head_topk_candidate_arrays: no chunk candidates");
    }

    const array merged_logits = candidate_logits_parts.size() == 1
        ? candidate_logits_parts.front()
        : concatenate(candidate_logits_parts, -1);
    const array merged_ids_i32 = candidate_ids_parts.size() == 1
        ? candidate_ids_parts.front()
        : concatenate(candidate_ids_parts, -1);
    try {
        return {
            reshape(merged_logits, {merged_logits.shape(0)}),
            reshape(astype(merged_ids_i32, uint32), {merged_ids_i32.shape(0)})
        };
    } catch (const std::exception& e) {
        throw_with_context("fused_lm_head_topk_candidate_arrays/merge", e);
    }
}

void prefill_prefix_chunks(mlx_ctx* ctx, const std::vector<int32_t>& prompt_vec, int32_t prefix_len) {
    if (prefix_len <= 0) return;
    const int chunk_size = resolve_prefill_chunk_size(ctx);
    const int sync_interval_chunks = std::max(resolve_prefill_sync_interval_chunks(), 1);
    int32_t offset = 0;
    int chunks_since_sync = 0;
    array pending_hidden = array(0.0f);
    bool has_pending_hidden = false;
    while (offset < prefix_len) {
        const int32_t remaining = prefix_len - offset;
        const int32_t take_count = std::min<int32_t>(remaining, chunk_size);
        const auto* chunk_ptr = prompt_vec.data() + static_cast<size_t>(offset);
        const array prefix_chunk(chunk_ptr, Shape{1, take_count}, int32);
        pending_hidden = forward_hidden(ctx, prefix_chunk, false);
        has_pending_hidden = true;
        offset += take_count;
        chunks_since_sync += 1;
        if (chunks_since_sync >= sync_interval_chunks) {
            eval(pending_hidden);
            synchronize();
            has_pending_hidden = false;
            chunks_since_sync = 0;
        }
    }
    if (has_pending_hidden) {
        eval(pending_hidden);
        synchronize();
    }
}

array prefill_seed_logits_chunked(mlx_ctx* ctx, const std::vector<int32_t>& prompt_vec, int32_t prompt_len) {
    if (!ctx) {
        throw std::runtime_error("prefill_seed_logits_chunked: ctx is null");
    }
    if (prompt_len <= 0) {
        throw std::runtime_error("prefill_seed_logits_chunked: prompt_len must be > 0");
    }
    const int chunk_size = resolve_prefill_chunk_size(ctx);
    if (chunk_size <= 0) {
        throw std::runtime_error("prefill_seed_logits_chunked: invalid chunk size");
    }

    const int tail_chunk_size = std::max(
        chunk_size,
        env_positive_int("TALU_METAL_PREFILL_TAIL_CHUNK_SIZE", chunk_size)
    );
    const int32_t tail_len = std::min<int32_t>(prompt_len, tail_chunk_size);
    const int32_t prefix_len = prompt_len - tail_len;
    prefill_prefix_chunks(ctx, prompt_vec, prefix_len);

    const auto* tail_ptr = prompt_vec.data() + static_cast<size_t>(prefix_len);
    const array tail_arr(tail_ptr, Shape{1, tail_len}, int32);
    return forward_logits(ctx, tail_arr);
}

void reset_runtime_state(mlx_ctx* ctx) {
    ctx->stream_ready = false;
    ctx->trace_decode_token = 1;
    ctx->kv_reserve_tokens = 0;
    ctx->sampling_context_counts.clear();
    ctx->sampling_context_len = 0;
    ctx->sampling_unique_ids.clear();
    ctx->sampling_repetition_scales.clear();
    ctx->sampling_additive_penalties.clear();
    for (auto& kv : ctx->kv_cache) {
        kv.keys.reset();
        kv.values.reset();
        kv.offset = 0;
    }
    for (auto& lc : ctx->linear_cache) {
        lc.conv_state.reset();
        lc.state.reset();
    }
}
