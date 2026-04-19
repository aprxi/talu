// bridge_moe.inc — Mixture of Experts: slice_expert_matrix, run_grouped_affine_moe_reference.

array slice_expert_matrix(const array& value, int expert_index, const std::string& name) {
    if (value.ndim() != 3) {
        throw std::runtime_error("expected rank-3 expert tensor for " + name);
    }
    if (expert_index < 0 || expert_index >= value.shape(0)) {
        throw std::runtime_error("expert index out of bounds for " + name);
    }
    return reshape(
        slice(
            value,
            {expert_index, 0, 0},
            {expert_index + 1, value.shape(1), value.shape(2)}
        ),
        {value.shape(1), value.shape(2)}
    );
}

std::optional<array> maybe_slice_expert_matrix(
    const array& value,
    int expert_index,
    const std::string& name,
    bool allow_scalar = false
) {
    if (value.size() <= 1) {
        if (allow_scalar) return value;
        return std::nullopt;
    }
    return slice_expert_matrix(value, expert_index, name);
}

array apply_moe_nvfp4_post_scale(
    const array& value,
    const array& post_scale,
    const array& top_k_indices
) {
    if (post_scale.size() <= 0) return value;
    if (post_scale.size() == 1) return apply_nvfp4_post_scale(value, post_scale);
    if (value.ndim() <= 0 || top_k_indices.ndim() <= 0) return value;

    const int out_cols = value.shape(value.ndim() - 1);
    if (out_cols <= 0) return value;

    if (post_scale.ndim() == 1) {
        if (post_scale.shape(0) == out_cols) {
            return apply_nvfp4_post_scale(value, post_scale);
        }
        if (post_scale.dtype() != float32) return value;
        const array flat_indices = flatten(astype(top_k_indices, int32), 0, -1);
        const array selected = take(post_scale, flat_indices, 0);
        Shape scale_shape(static_cast<size_t>(value.ndim()), 1);
        for (int axis = 0; axis < top_k_indices.ndim(); ++axis) {
            scale_shape[static_cast<size_t>(axis)] = top_k_indices.shape(axis);
        }
        const array broadcast = reshape(selected, scale_shape);
        return value * astype(broadcast, value.dtype());
    }

    array expert_scale = post_scale;
    if (expert_scale.ndim() == 3 && expert_scale.shape(2) == 1) {
        expert_scale = reshape(expert_scale, {expert_scale.shape(0), expert_scale.shape(1)});
    } else if (expert_scale.ndim() == 3 && expert_scale.shape(1) == 1) {
        expert_scale = reshape(expert_scale, {expert_scale.shape(0), expert_scale.shape(2)});
    }
    if (expert_scale.ndim() != 2 || expert_scale.dtype() != float32) {
        return apply_nvfp4_post_scale(value, post_scale);
    }

    const int rows = expert_scale.shape(1);
    if (rows != 1 && rows != out_cols) return value;

    const array flat_indices = flatten(astype(top_k_indices, int32), 0, -1);
    const array selected = take(expert_scale, flat_indices, 0);
    Shape scale_shape(static_cast<size_t>(value.ndim()), 1);
    for (int axis = 0; axis < top_k_indices.ndim(); ++axis) {
        scale_shape[static_cast<size_t>(axis)] = top_k_indices.shape(axis);
    }
    scale_shape.back() = rows;
    const array broadcast = reshape(selected, scale_shape);
    return value * astype(broadcast, value.dtype());
}

array run_native_nvfp4_moe_gather(
    mlx_ctx* ctx,
    const LayerWeights& lw,
    const array& mlp_in
) {
    if (!lw.moe_has_nvfp4) {
        throw std::runtime_error("run_native_nvfp4_moe_gather requires native NVFP4 MoE weights");
    }

    array router_logits = array(0.0f);
    if (lw.moe_router_has_q) {
        router_logits = quantized_matmul(
            mlp_in,
            lw.moe_router_q_w,
            lw.moe_router_q_scales,
            std::nullopt,
            true,
            std::nullopt,
            std::nullopt,
            "nvfp4"
        );
        router_logits = apply_nvfp4_post_scale(router_logits, lw.moe_router_q_biases);
    } else {
        router_logits = matmul(mlp_in, transpose(lw.moe_router_w, {1, 0}));
    }

    const int k = lw.moe_experts_per_token;
    auto partitioned_indices = argpartition(router_logits, -k, -1);
    const int last_dim = router_logits.ndim() - 1;
    const int total_experts = lw.moe_num_experts;
    Shape start(router_logits.ndim(), 0);
    Shape stop = router_logits.shape();
    start[last_dim] = total_experts - k;
    auto top_k_indices = slice(partitioned_indices, start, stop);
    auto top_k_logits = take_along_axis(router_logits, top_k_indices, last_dim);
    auto expert_weights = softmax(top_k_logits, -1, true);

    const auto x_expanded = expand_dims(mlp_in, {-2, -3});
    const bool use_fused_gate_up =
        lw.moe_gate_up_w.size() > 1 &&
        lw.moe_gate_up_scales.size() > 1;
    array gate_out = array(0.0f);
    array up_out = array(0.0f);
    if (use_fused_gate_up) {
        auto gate_up_out = gather_qmm(
            x_expanded,
            lw.moe_gate_up_w,
            lw.moe_gate_up_scales,
            std::nullopt,
            std::nullopt,
            top_k_indices,
            true,
            16,
            4,
            "nvfp4",
            false
        );
        gate_up_out = apply_moe_nvfp4_post_scale(gate_up_out, lw.moe_gate_up_biases, top_k_indices);
        const int last_dim = gate_up_out.ndim() - 1;
        const int half_width = gate_up_out.shape(last_dim) / 2;
        Shape gate_start(gate_up_out.ndim(), 0);
        Shape gate_stop = gate_up_out.shape();
        gate_stop[last_dim] = half_width;
        Shape up_start(gate_up_out.ndim(), 0);
        up_start[last_dim] = half_width;
        gate_out = slice(gate_up_out, gate_start, gate_stop);
        up_out = slice(gate_up_out, up_start, gate_up_out.shape());
    } else {
        gate_out = gather_qmm(
            x_expanded,
            lw.moe_gate_w,
            lw.moe_gate_scales,
            std::nullopt,
            std::nullopt,
            top_k_indices,
            true,
            16,
            4,
            "nvfp4",
            false
        );
        gate_out = apply_moe_nvfp4_post_scale(gate_out, lw.moe_gate_biases, top_k_indices);

        up_out = gather_qmm(
            x_expanded,
            lw.moe_up_w,
            lw.moe_up_scales,
            std::nullopt,
            std::nullopt,
            top_k_indices,
            true,
            16,
            4,
            "nvfp4",
            false
        );
        up_out = apply_moe_nvfp4_post_scale(up_out, lw.moe_up_biases, top_k_indices);
    }

    auto mid = ctx->cfg.use_gelu
        ? (gelu_approx(gate_out) * up_out)
        : ((gate_out * sigmoid(gate_out)) * up_out);
    auto down_out = gather_qmm(
        mid,
        lw.moe_down_w,
        lw.moe_down_scales,
        std::nullopt,
        std::nullopt,
        top_k_indices,
        true,
        16,
        4,
        "nvfp4",
        false
    );
    down_out = apply_moe_nvfp4_post_scale(down_out, lw.moe_down_biases, top_k_indices);

    down_out = squeeze(down_out, -2);
    auto weights_expanded = expand_dims(expert_weights, -1);
    return sum(down_out * weights_expanded, -2);
}

// Reference grouped-affine MoE implementation used for bridge self-tests.
array run_grouped_affine_moe_reference(mlx_ctx* ctx, const LayerWeights& lw, const array& mlp_in) {
    const int batch = mlp_in.shape(0);
    const int seq = mlp_in.shape(1);
    const int hidden = mlp_in.shape(2);
    const int experts_per_token = lw.moe_experts_per_token;
    const int num_experts = lw.moe_num_experts;
    if (experts_per_token <= 0 || num_experts <= 0) {
        throw std::runtime_error("invalid MoE routing geometry");
    }

    array router_logits = array(0.0f);
    if (lw.moe_router_has_q) {
        router_logits = quantized_matmul(
            mlp_in,
            lw.moe_router_q_w,
            lw.moe_router_q_scales,
            std::nullopt,
            true,
            std::nullopt,
            std::nullopt,
            "nvfp4"
        );
        router_logits = apply_nvfp4_post_scale(router_logits, lw.moe_router_q_biases);
    } else {
        router_logits = matmul(mlp_in, transpose(lw.moe_router_w, {1, 0}));
    }
    const array router_logits_f32 = astype(router_logits, float32);
    eval(router_logits_f32);
    synchronize();
    const float* router_ptr = router_logits_f32.data<float>();
    if (!router_ptr) {
        throw std::runtime_error("failed to materialize MoE router logits");
    }

    std::vector<array> token_outputs;
    token_outputs.reserve(static_cast<size_t>(batch) * static_cast<size_t>(seq));
    std::vector<float> token_logits(static_cast<size_t>(num_experts), 0.0f);
    std::vector<int32_t> token_ids(static_cast<size_t>(num_experts), 0);

    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq; ++s) {
            const size_t token_offset =
                (static_cast<size_t>(b) * static_cast<size_t>(seq) + static_cast<size_t>(s)) *
                static_cast<size_t>(num_experts);
            std::memcpy(
                token_logits.data(),
                router_ptr + token_offset,
                static_cast<size_t>(num_experts) * sizeof(float)
            );
            for (int e = 0; e < num_experts; ++e) {
                token_ids[static_cast<size_t>(e)] = e;
            }
            stable_sort_topk_pairs(token_logits.data(), token_ids.data(), num_experts);

            float max_logit = -std::numeric_limits<float>::infinity();
            for (int k = 0; k < experts_per_token; ++k) {
                max_logit = std::max(max_logit, token_logits[static_cast<size_t>(k)]);
            }
            std::vector<float> expert_weights(static_cast<size_t>(experts_per_token), 0.0f);
            float weight_sum = 0.0f;
            for (int k = 0; k < experts_per_token; ++k) {
                const float weight = std::exp(token_logits[static_cast<size_t>(k)] - max_logit);
                expert_weights[static_cast<size_t>(k)] = weight;
                weight_sum += weight;
            }
            if (weight_sum <= 0.0f) {
                throw std::runtime_error("invalid MoE expert weight sum");
            }
            const float inv_weight_sum = 1.0f / weight_sum;

            const array token_input = slice(mlp_in, {b, s, 0}, {b + 1, s + 1, hidden});
            array token_output = zeros({1, 1, hidden}, mlp_in.dtype());

            for (int k = 0; k < experts_per_token; ++k) {
                const int expert_id = token_ids[static_cast<size_t>(k)];
                const float expert_weight = expert_weights[static_cast<size_t>(k)] * inv_weight_sum;
                if (expert_id < 0 || expert_id >= num_experts) continue;

                array gate = array(0.0f);
                array up = array(0.0f);
                if (lw.moe_has_nvfp4) {
                    const auto gate_post_scale = maybe_slice_expert_matrix(
                        lw.moe_gate_biases,
                        expert_id,
                        "moe_gate_biases",
                        true
                    );
                    gate = quantized_matmul(
                        token_input,
                        slice_expert_matrix(lw.moe_gate_w, expert_id, "moe_gate_w"),
                        slice_expert_matrix(lw.moe_gate_scales, expert_id, "moe_gate_scales"),
                        std::nullopt,
                        true,
                        std::nullopt,
                        std::nullopt,
                        "nvfp4"
                    );
                    if (gate_post_scale) gate = apply_nvfp4_post_scale(gate, *gate_post_scale);

                    const auto up_post_scale = maybe_slice_expert_matrix(
                        lw.moe_up_biases,
                        expert_id,
                        "moe_up_biases",
                        true
                    );
                    up = quantized_matmul(
                        token_input,
                        slice_expert_matrix(lw.moe_up_w, expert_id, "moe_up_w"),
                        slice_expert_matrix(lw.moe_up_scales, expert_id, "moe_up_scales"),
                        std::nullopt,
                        true,
                        std::nullopt,
                        std::nullopt,
                        "nvfp4"
                    );
                    if (up_post_scale) up = apply_nvfp4_post_scale(up, *up_post_scale);
                } else {
                    gate = quantized_matmul(
                        token_input,
                        slice_expert_matrix(lw.moe_gate_w, expert_id, "moe_gate_w"),
                        slice_expert_matrix(lw.moe_gate_scales, expert_id, "moe_gate_scales"),
                        maybe_slice_expert_matrix(lw.moe_gate_biases, expert_id, "moe_gate_biases"),
                        true,
                        lw.moe_expert_group_size,
                        ctx->cfg.quant_bits,
                        "affine"
                    );
                    up = quantized_matmul(
                        token_input,
                        slice_expert_matrix(lw.moe_up_w, expert_id, "moe_up_w"),
                        slice_expert_matrix(lw.moe_up_scales, expert_id, "moe_up_scales"),
                        maybe_slice_expert_matrix(lw.moe_up_biases, expert_id, "moe_up_biases"),
                        true,
                        lw.moe_expert_group_size,
                        ctx->cfg.quant_bits,
                        "affine"
                    );
                }
                array ff = array(0.0f);
                {
                    static const auto compiled_silu_gate_mul = compile(
                        [](const std::vector<array>& inputs) {
                            return std::vector<array>{silu(inputs[0]) * inputs[1]};
                        },
                        true
                    );
                    static const auto compiled_gelu_gate_mul = compile(
                        [](const std::vector<array>& inputs) {
                            return std::vector<array>{gelu_approx(inputs[0]) * inputs[1]};
                        },
                        true
                    );
                    ff = ctx->cfg.use_gelu
                        ? compiled_gelu_gate_mul({gate, up}).at(0)
                        : compiled_silu_gate_mul({gate, up}).at(0);
                }
                array down = array(0.0f);
                if (lw.moe_has_nvfp4) {
                    const auto down_post_scale = maybe_slice_expert_matrix(
                        lw.moe_down_biases,
                        expert_id,
                        "moe_down_biases",
                        true
                    );
                    down = quantized_matmul(
                        ff,
                        slice_expert_matrix(lw.moe_down_w, expert_id, "moe_down_w"),
                        slice_expert_matrix(lw.moe_down_scales, expert_id, "moe_down_scales"),
                        std::nullopt,
                        true,
                        std::nullopt,
                        std::nullopt,
                        "nvfp4"
                    );
                    if (down_post_scale) down = apply_nvfp4_post_scale(down, *down_post_scale);
                } else {
                    down = quantized_matmul(
                        ff,
                        slice_expert_matrix(lw.moe_down_w, expert_id, "moe_down_w"),
                        slice_expert_matrix(lw.moe_down_scales, expert_id, "moe_down_scales"),
                        maybe_slice_expert_matrix(lw.moe_down_biases, expert_id, "moe_down_biases"),
                        true,
                        lw.moe_expert_group_size,
                        ctx->cfg.quant_bits,
                        "affine"
                    );
                }
                token_output = token_output + down * array(expert_weight, down.dtype());
            }

            token_outputs.push_back(token_output);
        }
    }

    array rows = token_outputs.empty() ? zeros({0, hidden}, mlp_in.dtype()) : reshape(token_outputs[0], {1, hidden});
    for (size_t i = 1; i < token_outputs.size(); ++i) {
        rows = concatenate({rows, reshape(token_outputs[i], {1, hidden})}, 0);
    }
    return reshape(rows, {batch, seq, hidden});
}

array run_layer(
    mlx_ctx* ctx,
    int layer_idx,
    const array& hidden,
    const array& source_embeddings,
    const array& input_ids,
    const TraceFrame* trace_frame,
    const array* precomputed_x_norm = nullptr
) {
    LayerWeights& lw = ctx->layers[static_cast<size_t>(layer_idx)];
    if (trace_frame) {
        xray_emit_array_f32(
            XRAY_POINT_LAYER_INPUT,
            static_cast<uint16_t>(layer_idx),
            trace_frame->token,
            trace_frame->layer_position,
            hidden,
            "mlx_layer_input_host"
        );
    }

    const array x_norm = precomputed_x_norm
        ? *precomputed_x_norm
        : fast::rms_norm(hidden, std::optional<array>(lw.input_layernorm_w), ctx->cfg.rms_norm_eps);

    const array residual_branch = lw.is_linear
        ? run_linear_attention_layer(ctx, layer_idx, x_norm)
        : (lw.is_shortconv ? run_shortconv_layer(ctx, layer_idx, x_norm)
                           : run_full_attention_layer(ctx, layer_idx, x_norm));
    if (trace_frame && !lw.is_linear && !lw.is_shortconv) {
        xray_emit_array_f32(
            XRAY_POINT_ATTN_OUT,
            static_cast<uint16_t>(layer_idx),
            trace_frame->token,
            trace_frame->layer_position,
            residual_branch,
            "mlx_attn_out_host"
        );
    }

    const bool has_gemma_four_norm =
        lw.pre_feedforward_layernorm_w.size() > 1 &&
        lw.post_feedforward_layernorm_w.size() > 1;

    array h = array(0.0f);
    array mlp_in = array(0.0f);
    if (has_gemma_four_norm) {
        h = hidden + fast::rms_norm(residual_branch, std::optional<array>(lw.post_attention_layernorm_w), ctx->cfg.rms_norm_eps);
        mlp_in = fast::rms_norm(h, std::optional<array>(lw.pre_feedforward_layernorm_w), ctx->cfg.rms_norm_eps);
    } else {
        h = hidden + residual_branch;
        mlp_in = fast::rms_norm(h, std::optional<array>(lw.post_attention_layernorm_w), ctx->cfg.rms_norm_eps);
    }
    array down = array(0.0f);
    if (lw.mlp_is_moe) {
        if (lw.moe_has_nvfp4) {
            // Native NVFP4 experts use gather_qmm in NVFP4 mode with expert-aware
            // post-scale application; keeps execution on GPU and avoids host loops.
            down = run_native_nvfp4_moe_gather(ctx, lw, mlp_in);
        } else {
            const bool use_mxfp4_moe =
                lw.moe_gate_scales.dtype() == uint8 &&
                lw.moe_up_scales.dtype() == uint8 &&
                lw.moe_down_scales.dtype() == uint8;
            const bool use_fused_affine_gate_up =
                !use_mxfp4_moe &&
                lw.moe_gate_up_w.size() > 1 &&
                lw.moe_gate_up_scales.size() > 1;

        // Decode-specialized path: split routing from expert compute.
        // For B=1 S=1 with affine fused_gate_up, use the separate routing
        // helper + expert compute function to enable future custom kernel.
        const int B = mlp_in.shape(0);
        const int S = mlp_in.shape(1);
        const bool use_decode_expert_path =
            B == 1 && S == 1 &&
            use_fused_affine_gate_up &&
            !use_mxfp4_moe &&
            env_truthy("TALU_METAL_MOE_DECODE_GEMV", true);

            if (use_decode_expert_path) {
            // Routing: same logic as fused kernel but separated for custom expert compute.
            array router_logits = matmul(mlp_in, transpose(lw.moe_router_w, {1, 0}));
            const int k = lw.moe_experts_per_token;
            auto partitioned = argpartition(router_logits, -k, -1);
            const int rd = router_logits.ndim() - 1;
            Shape rstart(router_logits.ndim(), 0);
            Shape rstop = router_logits.shape();
            rstart[rd] = lw.moe_num_experts - k;
            auto tk_indices = slice(partitioned, rstart, rstop);
            auto tk_logits = take_along_axis(router_logits, tk_indices, rd);
            auto tk_weights = softmax(tk_logits, -1, true);

            struct { array top_k_indices; array expert_weights; } route = {
                std::move(tk_indices), std::move(tk_weights)
            };
            const auto gate_up_biases = lw.moe_gate_up_biases.size() > 1
                ? std::optional<array>(lw.moe_gate_up_biases)
                : std::nullopt;
            const auto down_biases = lw.moe_down_biases.size() > 1
                ? std::optional<array>(lw.moe_down_biases)
                : std::nullopt;
            down = moe_expert_compute_decode(
                mlp_in,
                route.top_k_indices,
                route.expert_weights,
                lw.moe_gate_up_w,
                lw.moe_gate_up_scales,
                gate_up_biases,
                lw.moe_down_w,
                lw.moe_down_scales,
                down_biases,
                lw.moe_expert_group_size,
                ctx->cfg.quant_bits,
                ctx->cfg.use_gelu
            );
            } else {

        down = use_mxfp4_moe
            ? *static_cast<array*>(mlx_lazy_fused_expert_mix_ffn_mxfp4(
                &mlp_in,
                &lw.moe_router_w,
                nullptr,
                nullptr,
                nullptr,
                &lw.moe_gate_w,
                &lw.moe_gate_scales,
                &lw.moe_up_w,
                &lw.moe_up_scales,
                &lw.moe_down_w,
                &lw.moe_down_scales,
                lw.moe_gate_biases.size() > 1 ? &lw.moe_gate_biases : nullptr,
                lw.moe_up_biases.size() > 1 ? &lw.moe_up_biases : nullptr,
                lw.moe_down_biases.size() > 1 ? &lw.moe_down_biases : nullptr,
                static_cast<size_t>(lw.moe_num_experts),
                static_cast<size_t>(lw.moe_experts_per_token),
                static_cast<size_t>(lw.moe_router_group_size),
                static_cast<size_t>(lw.moe_expert_group_size)
            ))
            : (use_fused_affine_gate_up
                ? *static_cast<array*>(mlx_lazy_fused_expert_mix_ffn_affine_fused_gate_up(
                    &mlp_in,
                    &lw.moe_router_w,
                    nullptr,
                    &lw.moe_gate_up_w,
                    &lw.moe_gate_up_scales,
                    lw.moe_gate_up_biases.size() > 1 ? &lw.moe_gate_up_biases : nullptr,
                    &lw.moe_down_w,
                    &lw.moe_down_scales,
                    lw.moe_down_biases.size() > 1 ? &lw.moe_down_biases : nullptr,
                    static_cast<size_t>(lw.moe_num_experts),
                    static_cast<size_t>(lw.moe_experts_per_token),
                    static_cast<size_t>(lw.moe_expert_group_size),
                    ctx->cfg.quant_bits,
                    ctx->cfg.use_gelu
                ))
            : *static_cast<array*>(mlx_lazy_fused_expert_mix_ffn_affine(
                &mlp_in,
                &lw.moe_router_w,
                nullptr,
                &lw.moe_gate_w,
                &lw.moe_gate_scales,
                lw.moe_gate_biases.size() > 1 ? &lw.moe_gate_biases : nullptr,
                &lw.moe_up_w,
                &lw.moe_up_scales,
                lw.moe_up_biases.size() > 1 ? &lw.moe_up_biases : nullptr,
                &lw.moe_down_w,
                &lw.moe_down_scales,
                lw.moe_down_biases.size() > 1 ? &lw.moe_down_biases : nullptr,
                static_cast<size_t>(lw.moe_num_experts),
                static_cast<size_t>(lw.moe_experts_per_token),
                static_cast<size_t>(lw.moe_expert_group_size),
                ctx->cfg.quant_bits,
                ctx->cfg.use_gelu
            )));
            } // end else (non-decode or non-eligible path)
        } // end non-nvfp4 branch

        if (lw.moe_shared_gate_rhs.size() > 1 || lw.moe_shared_gate_has_q) {
            const bool use_fused_shared_gate_up =
                lw.moe_shared_gate_up_rhs.size() > 1 || lw.moe_shared_gate_up_has_q;
            const int shared_gate_width = lw.moe_shared_gate_has_q
                ? lw.moe_shared_gate_q_w.shape(0)
                : lw.moe_shared_gate_rhs.shape(1);
            array shared_gate = array(0.0f);
            array shared_up = array(0.0f);
            if (use_fused_shared_gate_up && shared_gate_width > 0) {
                const array shared_gate_up = linear_decode_maybe_quantized(
                    ctx->fp8_decode_qmm_enabled,
                    ctx->cfg.quant_group_size,
                    ctx->cfg.quant_bits,
                    mlp_in,
                    lw.moe_shared_gate_up_rhs,
                    lw.moe_shared_gate_up_has_q,
                    lw.moe_shared_gate_up_q_w,
                    lw.moe_shared_gate_up_q_scales,
                    lw.moe_shared_gate_up_q_biases,
                    false
                );
                shared_gate = slice(
                    shared_gate_up,
                    {0, 0, 0},
                    {shared_gate_up.shape(0), shared_gate_up.shape(1), shared_gate_width}
                );
                shared_up = slice(
                    shared_gate_up,
                    {0, 0, shared_gate_width},
                    {shared_gate_up.shape(0), shared_gate_up.shape(1), 2 * shared_gate_width}
                );
            } else {
                shared_gate = linear_decode_maybe_quantized(
                    ctx->fp8_decode_qmm_enabled,
                    ctx->cfg.quant_group_size,
                    ctx->cfg.quant_bits,
                    mlp_in,
                    lw.moe_shared_gate_rhs,
                    lw.moe_shared_gate_has_q,
                    lw.moe_shared_gate_q_w,
                    lw.moe_shared_gate_q_scales,
                    lw.moe_shared_gate_q_biases,
                    false
                );
                shared_up = linear_decode_maybe_quantized(
                    ctx->fp8_decode_qmm_enabled,
                    ctx->cfg.quant_group_size,
                    ctx->cfg.quant_bits,
                    mlp_in,
                    lw.moe_shared_up_rhs,
                    lw.moe_shared_up_has_q,
                    lw.moe_shared_up_q_w,
                    lw.moe_shared_up_q_scales,
                    lw.moe_shared_up_q_biases,
                    false
                );
            }
            array shared_ff = array(0.0f);
            {
                static const auto compiled_silu_gate_mul = compile(
                    [](const std::vector<array>& inputs) {
                        return std::vector<array>{silu(inputs[0]) * inputs[1]};
                    },
                    true
                );
                static const auto compiled_gelu_gate_mul = compile(
                    [](const std::vector<array>& inputs) {
                        return std::vector<array>{gelu_approx(inputs[0]) * inputs[1]};
                    },
                    true
                );
                shared_ff = ctx->cfg.use_gelu
                    ? compiled_gelu_gate_mul({shared_gate, shared_up}).at(0)
                    : compiled_silu_gate_mul({shared_gate, shared_up}).at(0);
            }
            array shared_down = linear_decode_maybe_quantized(
                ctx->fp8_decode_qmm_enabled,
                ctx->cfg.quant_group_size,
                ctx->cfg.quant_bits,
                shared_ff,
                lw.moe_shared_down_rhs,
                lw.moe_shared_down_has_q,
                lw.moe_shared_down_q_w,
                lw.moe_shared_down_q_scales,
                lw.moe_shared_down_q_biases,
                false
            );

            if (lw.moe_shared_expert_gate_has_q) {
                array gate_logits = quantized_matmul(
                    mlp_in,
                    lw.moe_shared_expert_gate_q_w,
                    lw.moe_shared_expert_gate_q_scales,
                    std::nullopt,
                    true,
                    std::nullopt,
                    std::nullopt,
                    "nvfp4"
                );
                gate_logits = apply_nvfp4_post_scale(gate_logits, lw.moe_shared_expert_gate_q_biases);
                shared_down = shared_down * sigmoid(gate_logits);
            } else if (lw.moe_shared_expert_gate_w.size() > 1) {
                const array shared_gate_weight = lw.moe_shared_expert_gate_w.dtype() == mlp_in.dtype()
                    ? lw.moe_shared_expert_gate_w
                    : astype(lw.moe_shared_expert_gate_w, mlp_in.dtype());
                const array gate_logits = sum(
                    mlp_in * reshape(shared_gate_weight, {1, 1, shared_gate_weight.shape(0)}),
                    -1
                );
                shared_down = shared_down * expand_dims(sigmoid(gate_logits), -1);
            }

            down = down + shared_down;
        }
    } else {
        const int mlp_gate_width = lw.mlp_gate_width;
        const bool use_fused_gate_up =
            (lw.mlp_gate_up_rhs.size() > 1 || lw.mlp_gate_up_has_q) &&
            mlp_gate_width > 0;
        array ff = array(0.0f);
        if (use_fused_gate_up) {
            const array gate_up = linear_decode_maybe_quantized(
                ctx->fp8_decode_qmm_enabled,
                ctx->cfg.quant_group_size,
                ctx->cfg.quant_bits,
                mlp_in,
                lw.mlp_gate_up_rhs,
                lw.mlp_gate_up_has_q,
                lw.mlp_gate_up_q_w,
                lw.mlp_gate_up_q_scales,
                lw.mlp_gate_up_q_biases,
                false
            );
            const array gate = slice(
                gate_up,
                {0, 0, 0},
                {gate_up.shape(0), gate_up.shape(1), mlp_gate_width}
            );
            const array up = slice(
                gate_up,
                {0, 0, mlp_gate_width},
                {gate_up.shape(0), gate_up.shape(1), 2 * mlp_gate_width}
            );
            {
                static const auto compiled_silu_gate_mul = compile(
                    [](const std::vector<array>& inputs) {
                        return std::vector<array>{silu(inputs[0]) * inputs[1]};
                    },
                    true
                );
                static const auto compiled_gelu_gate_mul = compile(
                    [](const std::vector<array>& inputs) {
                        return std::vector<array>{gelu_approx(inputs[0]) * inputs[1]};
                    },
                    true
                );
                ff = ctx->cfg.use_gelu
                    ? compiled_gelu_gate_mul({gate, up}).at(0)
                    : compiled_silu_gate_mul({gate, up}).at(0);
            }
        } else {
            const array gate = linear_decode_maybe_quantized(
                ctx->fp8_decode_qmm_enabled,
                ctx->cfg.quant_group_size,
                ctx->cfg.quant_bits,
                mlp_in,
                lw.mlp_gate_rhs,
                lw.mlp_gate_has_q,
                lw.mlp_gate_q_w,
                lw.mlp_gate_q_scales,
                lw.mlp_gate_q_biases,
                false
            );
            const array up = linear_decode_maybe_quantized(
                ctx->fp8_decode_qmm_enabled,
                ctx->cfg.quant_group_size,
                ctx->cfg.quant_bits,
                mlp_in,
                lw.mlp_up_rhs,
                lw.mlp_up_has_q,
                lw.mlp_up_q_w,
                lw.mlp_up_q_scales,
                lw.mlp_up_q_biases,
                false
            );
            {
                static const auto compiled_silu_gate_mul = compile(
                    [](const std::vector<array>& inputs) {
                        return std::vector<array>{silu(inputs[0]) * inputs[1]};
                    },
                    true
                );
                static const auto compiled_gelu_gate_mul = compile(
                    [](const std::vector<array>& inputs) {
                        return std::vector<array>{gelu_approx(inputs[0]) * inputs[1]};
                    },
                    true
                );
                ff = ctx->cfg.use_gelu
                    ? compiled_gelu_gate_mul({gate, up}).at(0)
                    : compiled_silu_gate_mul({gate, up}).at(0);
            }
        }
        down = linear_decode_maybe_quantized(
            ctx->fp8_decode_qmm_enabled,
            ctx->cfg.quant_group_size,
            ctx->cfg.quant_bits,
            ff,
            lw.mlp_down_rhs,
            lw.mlp_down_has_q,
            lw.mlp_down_q_w,
            lw.mlp_down_q_scales,
            lw.mlp_down_q_biases,
            false
        );
    }

    const array mlp_branch = has_gemma_four_norm
        ? fast::rms_norm(down, std::optional<array>(lw.post_feedforward_layernorm_w), ctx->cfg.rms_norm_eps)
        : down;
    array out = h + mlp_branch;
    if (ctx->per_layer_input_enabled && lw.has_per_layer_input) {
        const int hpl = ctx->per_layer_hidden_size;
        const int layer_offset = layer_idx * hpl;

        // Per-layer input branch:
        // proj(source_embed) -> norm -> + per-layer token embed -> gate(hidden)
        // -> proj -> norm -> residual + scale
        const array source_projection = linear(source_embeddings, lw.per_layer_model_projection_rhs);
        array per_layer_input = source_projection * array(ctx->per_layer_model_projection_scale, source_projection.dtype());
        per_layer_input = fast::rms_norm(per_layer_input, std::optional<array>(ctx->per_layer_projection_norm_w), ctx->cfg.rms_norm_eps);

        if (ctx->per_layer_embedding.ndim() == 2 && ctx->per_layer_embedding.shape(1) >= (layer_offset + hpl)) {
            const array token_embed_all = take(ctx->per_layer_embedding, input_ids, 0); // [B,S,total_width]
            const array token_embed_layer = slice(
                token_embed_all,
                {0, 0, layer_offset},
                {token_embed_all.shape(0), token_embed_all.shape(1), layer_offset + hpl}
            );
            per_layer_input = per_layer_input + token_embed_layer * array(ctx->per_layer_embed_scale, token_embed_layer.dtype());
        }

        const array gate_branch = linear(out, lw.per_layer_input_gate_rhs);
        array gated = ctx->cfg.use_gelu ? gelu_approx(gate_branch) : silu(gate_branch);
        gated = gated * (per_layer_input * array(ctx->per_layer_input_scale, per_layer_input.dtype()));
        array branch = linear(gated, lw.per_layer_projection_rhs);
        branch = fast::rms_norm(branch, std::optional<array>(lw.post_per_layer_input_norm_w), ctx->cfg.rms_norm_eps);
        out = (out + branch) * array(lw.layer_scalar, out.dtype());
    } else if (std::fabs(lw.layer_scalar - 1.0f) > 1.0e-6f) {
        out = out * array(lw.layer_scalar, out.dtype());
    }
    if (trace_frame) {
        xray_emit_array_f32(
            XRAY_POINT_BLOCK_OUT,
            static_cast<uint16_t>(layer_idx),
            trace_frame->token,
            trace_frame->layer_position,
            out,
            "mlx_block_out_host"
        );
    }
    return out;
}

// ---------------------------------------------------------------------------
// Compiled decode core: a pure function over all transformer layers for S=1.
//
// Inputs (all fixed shapes for decode):
//   [0]           hidden      [1, 1, H]
//   [1]           offset_arr  [1]  (int32, current position)
//   [2]           attn_mask   [1, 1, 1, capacity+1]  (0 or -inf)
//   [3 + i*2]     kv_keys_i   [1, num_kv_heads, capacity, head_dim]
//   [3 + i*2 + 1] kv_vals_i   [1, num_kv_heads, capacity, head_dim]
//
// Outputs:
//   [0]           hidden_out  [1, 1, H]
//   [1 + i*2]     new_k_i     [1, num_kv_heads, 1, head_dim]
//   [1 + i*2 + 1] new_v_i     [1, num_kv_heads, 1, head_dim]
//
// Weights are accessed through the captured ctx pointer (constant across
// decode steps). KV caches and offset flow through explicit inputs so
// compile() sees fixed shapes and traces once.
// ---------------------------------------------------------------------------
