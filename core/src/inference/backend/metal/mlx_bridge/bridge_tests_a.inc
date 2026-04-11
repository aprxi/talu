// bridge_tests_a.inc — Tests: moe_gpu, conv_decode, attention_sdpa, kv_cache_reserve,
// shared_expert_fusion, dense_mlp_fusion, full_attention_qkv_fusion.

int32_t mlx_test_grouped_affine_moe_gpu_path(void) {
    try {
        mlx_ctx ctx{};
        ctx.cfg.quant_bits = 4;
        ctx.cfg.quant_group_size = 32;
        ctx.cfg.use_gelu = false;

        LayerWeights lw;
        lw.mlp_is_moe = true;
        lw.moe_num_experts = 4;
        lw.moe_experts_per_token = 2;
        lw.moe_expert_group_size = 32;

        constexpr int batch = 1;
        constexpr int seq = 2;
        constexpr int hidden = 32;
        constexpr int d_ff = 32;
        constexpr int num_experts = 4;

        std::vector<float> input_values(static_cast<size_t>(batch * seq * hidden));
        for (int i = 0; i < batch * seq * hidden; ++i) {
            input_values[static_cast<size_t>(i)] = static_cast<float>((i % 11) - 5) * 0.0625f;
        }
        const array mlp_in(input_values.data(), Shape{batch, seq, hidden}, float32);

        std::vector<float> router_values(static_cast<size_t>(num_experts * hidden));
        for (int expert = 0; expert < num_experts; ++expert) {
            for (int col = 0; col < hidden; ++col) {
                router_values[static_cast<size_t>(expert * hidden + col)] =
                    static_cast<float>((expert + 1) * ((col % 7) + 1)) * 0.01f;
            }
        }
        lw.moe_router_w = array(router_values.data(), Shape{num_experts, hidden}, float32);

        std::vector<float> gate_dense_values(static_cast<size_t>(num_experts * d_ff * hidden));
        std::vector<float> up_dense_values(static_cast<size_t>(num_experts * d_ff * hidden));
        std::vector<float> down_dense_values(static_cast<size_t>(num_experts * hidden * d_ff));
        for (int expert = 0; expert < num_experts; ++expert) {
            for (int row = 0; row < d_ff; ++row) {
                for (int col = 0; col < hidden; ++col) {
                    const size_t idx = static_cast<size_t>((expert * d_ff + row) * hidden + col);
                    gate_dense_values[idx] =
                        static_cast<float>(((expert + row + col) % 9) - 4) * 0.03125f;
                    up_dense_values[idx] =
                        static_cast<float>(((expert * 3 + row + col) % 13) - 6) * 0.0275f;
                }
            }
            for (int row = 0; row < hidden; ++row) {
                for (int col = 0; col < d_ff; ++col) {
                    const size_t idx = static_cast<size_t>((expert * hidden + row) * d_ff + col);
                    down_dense_values[idx] =
                        static_cast<float>(((expert * 5 + row + col) % 15) - 7) * 0.0225f;
                }
            }
        }

        const array gate_dense(gate_dense_values.data(), Shape{num_experts, d_ff, hidden}, float32);
        const array up_dense(up_dense_values.data(), Shape{num_experts, d_ff, hidden}, float32);
        const array down_dense(down_dense_values.data(), Shape{num_experts, hidden, d_ff}, float32);

        const auto gate_q = quantize(gate_dense, 32, 4, "affine");
        const auto up_q = quantize(up_dense, 32, 4, "affine");
        const auto down_q = quantize(down_dense, 32, 4, "affine");
        const array gate_up_dense = contiguous(concatenate({gate_dense, up_dense}, 1));
        const auto gate_up_q = quantize(gate_up_dense, 32, 4, "affine");
        if (gate_q.size() != 3 || up_q.size() != 3 || down_q.size() != 3 || gate_up_q.size() != 3) {
            throw std::runtime_error("mlx grouped-affine quantize returned unexpected output count");
        }

        lw.moe_gate_up_w = gate_up_q[0];
        lw.moe_gate_up_scales = gate_up_q[1];
        lw.moe_gate_up_biases = gate_up_q[2];
        lw.moe_gate_w = gate_q[0];
        lw.moe_gate_scales = gate_q[1];
        lw.moe_gate_biases = gate_q[2];
        lw.moe_up_w = up_q[0];
        lw.moe_up_scales = up_q[1];
        lw.moe_up_biases = up_q[2];
        lw.moe_down_w = down_q[0];
        lw.moe_down_scales = down_q[1];
        lw.moe_down_biases = down_q[2];

        const array expected = run_grouped_affine_moe_reference(&ctx, lw, mlp_in);
        const array actual_split = *static_cast<array*>(mlx_lazy_fused_expert_mix_ffn_affine(
            &mlp_in,
            &lw.moe_router_w,
            nullptr,
            &lw.moe_gate_w,
            &lw.moe_gate_scales,
            &lw.moe_gate_biases,
            &lw.moe_up_w,
            &lw.moe_up_scales,
            &lw.moe_up_biases,
            &lw.moe_down_w,
            &lw.moe_down_scales,
            &lw.moe_down_biases,
            static_cast<size_t>(lw.moe_num_experts),
            static_cast<size_t>(lw.moe_experts_per_token),
            static_cast<size_t>(lw.moe_expert_group_size),
            ctx.cfg.quant_bits,
            ctx.cfg.use_gelu
        ));
        const array actual_fused = *static_cast<array*>(mlx_lazy_fused_expert_mix_ffn_affine_fused_gate_up(
            &mlp_in,
            &lw.moe_router_w,
            nullptr,
            &lw.moe_gate_up_w,
            &lw.moe_gate_up_scales,
            &lw.moe_gate_up_biases,
            &lw.moe_down_w,
            &lw.moe_down_scales,
            &lw.moe_down_biases,
            static_cast<size_t>(lw.moe_num_experts),
            static_cast<size_t>(lw.moe_experts_per_token),
            static_cast<size_t>(lw.moe_expert_group_size),
            ctx.cfg.quant_bits,
            ctx.cfg.use_gelu
        ));

        const array split_diff_max = max(reshape(abs(astype(expected, float32) - astype(actual_split, float32)), {static_cast<int>(actual_split.size())}));
        const array fused_diff_max = max(reshape(abs(astype(expected, float32) - astype(actual_fused, float32)), {static_cast<int>(actual_fused.size())}));
        eval({split_diff_max, fused_diff_max});
        synchronize();
        const float* split_diff_ptr = split_diff_max.data<float>();
        const float* fused_diff_ptr = fused_diff_max.data<float>();
        if (!split_diff_ptr || !fused_diff_ptr) {
            g_last_error = "mlx_test_grouped_affine_moe_gpu_path: failed to materialize diffs";
            return 0;
        }
        if (split_diff_ptr[0] > 1e-4f) {
            g_last_error = "mlx_test_grouped_affine_moe_gpu_path: split grouped-affine MoE mismatch";
            return 0;
        }
        if (fused_diff_ptr[0] > 1e-4f) {
            g_last_error = "mlx_test_grouped_affine_moe_gpu_path: fused grouped-affine MoE mismatch";
            return 0;
        }
        return 1;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 0;
    } catch (...) {
        g_last_error = "unknown error in mlx_test_grouped_affine_moe_gpu_path";
        return 0;
    }
}

int32_t mlx_test_depthwise_conv_decode_step(void) {
    try {
        constexpr int batch = 2;
        constexpr int kernel = 4;
        constexpr int conv_dim = 6;

        std::vector<float> state_values(static_cast<size_t>(batch * (kernel - 1) * conv_dim));
        std::vector<float> token_values(static_cast<size_t>(batch * conv_dim));
        std::vector<float> weight_values(static_cast<size_t>(conv_dim * kernel));

        for (int i = 0; i < static_cast<int>(state_values.size()); ++i) {
            state_values[static_cast<size_t>(i)] = static_cast<float>(((i * 5) % 17) - 8) * 0.03125f;
        }
        for (int i = 0; i < static_cast<int>(token_values.size()); ++i) {
            token_values[static_cast<size_t>(i)] = static_cast<float>(((i * 7) % 11) - 5) * 0.0625f;
        }
        for (int i = 0; i < static_cast<int>(weight_values.size()); ++i) {
            weight_values[static_cast<size_t>(i)] = static_cast<float>(((i * 3) % 13) - 6) * 0.05f;
        }

        const array cache_state(state_values.data(), Shape{batch, kernel - 1, conv_dim}, float32);
        const array token_input(token_values.data(), Shape{batch, 1, conv_dim}, float32);
        const array conv_weight(weight_values.data(), Shape{conv_dim, kernel, 1}, float32);
        const array decode_weight = prepare_depthwise_conv1d_decode_weight(
            conv_weight,
            conv_dim,
            kernel,
            "mlx_test_depthwise_conv_decode_step"
        );

        const array conv_window = concatenate({cache_state, token_input}, 1);
        const array expected = conv1d(conv_window, conv_weight, 1, 0, 1, conv_dim);
        const array expected_next_state = slice(conv_window, {0, 1, 0}, {batch, kernel, conv_dim});

        auto [actual, next_state_opt] = depthwise_conv1d_decode_step(
            std::optional<array>(cache_state),
            token_input,
            decode_weight,
            kernel,
            conv_dim,
            "mlx_test_depthwise_conv_decode_step"
        );
        if (!next_state_opt.has_value()) {
            g_last_error = "mlx_test_depthwise_conv_decode_step: missing next state";
            return 0;
        }

        const array actual_next_state = *next_state_opt;
        const array conv_diff = max(reshape(abs(astype(expected, float32) - astype(actual, float32)), {static_cast<int>(expected.size())}));
        const array state_diff = max(reshape(abs(astype(expected_next_state, float32) - astype(actual_next_state, float32)), {static_cast<int>(expected_next_state.size())}));
        eval({conv_diff, state_diff});
        synchronize();

        const float* conv_diff_ptr = conv_diff.data<float>();
        const float* state_diff_ptr = state_diff.data<float>();
        if (!conv_diff_ptr || !state_diff_ptr) {
            g_last_error = "mlx_test_depthwise_conv_decode_step: failed to materialize diffs";
            return 0;
        }
        if (conv_diff_ptr[0] > 1e-5f) {
            g_last_error = "mlx_test_depthwise_conv_decode_step: conv output mismatch";
            return 0;
        }
        if (state_diff_ptr[0] > 1e-6f) {
            g_last_error = "mlx_test_depthwise_conv_decode_step: next state mismatch";
            return 0;
        }
        return 1;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 0;
    } catch (...) {
        g_last_error = "unknown error in mlx_test_depthwise_conv_decode_step";
        return 0;
    }
}

int32_t mlx_test_single_query_attention_matches_sdpa(void) {
    try {
        constexpr int batch = 1;
        constexpr int num_heads = 4;
        constexpr int num_kv_heads = 2;
        constexpr int seq_len = 7;
        constexpr int head_dim = 8;

        std::vector<float> q_values(static_cast<size_t>(batch * num_heads * head_dim));
        std::vector<float> k_values(static_cast<size_t>(batch * num_kv_heads * seq_len * head_dim));
        std::vector<float> v_values(static_cast<size_t>(batch * num_kv_heads * seq_len * head_dim));

        for (int i = 0; i < static_cast<int>(q_values.size()); ++i) {
            q_values[static_cast<size_t>(i)] = static_cast<float>(((i * 5) % 19) - 9) * 0.0625f;
        }
        for (int i = 0; i < static_cast<int>(k_values.size()); ++i) {
            k_values[static_cast<size_t>(i)] = static_cast<float>(((i * 7) % 23) - 11) * 0.03125f;
            v_values[static_cast<size_t>(i)] = static_cast<float>(((i * 3) % 17) - 8) * 0.04125f;
        }

        const array queries(q_values.data(), Shape{batch, num_heads, 1, head_dim}, float32);
        const array keys(k_values.data(), Shape{batch, num_kv_heads, seq_len, head_dim}, float32);
        const array values(v_values.data(), Shape{batch, num_kv_heads, seq_len, head_dim}, float32);

        const array expected = fast::scaled_dot_product_attention(
            queries,
            keys,
            values,
            1.0f / std::sqrt(static_cast<float>(head_dim)),
            ""
        );
        const array actual = single_query_grouped_attention(
            queries,
            keys,
            values,
            num_heads,
            num_kv_heads,
            1.0f / std::sqrt(static_cast<float>(head_dim)),
            false
        );

        const array diff_max = max(reshape(abs(astype(expected, float32) - astype(actual, float32)), {static_cast<int>(expected.size())}));
        eval(diff_max);
        synchronize();

        const float* diff_ptr = diff_max.data<float>();
        if (!diff_ptr) {
            g_last_error = "mlx_test_single_query_attention_matches_sdpa: failed to materialize diff";
            return 0;
        }
        if (diff_ptr[0] > 1e-5f) {
            g_last_error = "mlx_test_single_query_attention_matches_sdpa: attention mismatch";
            return 0;
        }
        return 1;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 0;
    } catch (...) {
        g_last_error = "unknown error in mlx_test_single_query_attention_matches_sdpa";
        return 0;
    }
}

int32_t mlx_test_kv_cache_reserve_preserves_prefix(void) {
    try {
        constexpr int batch = 1;
        constexpr int num_kv_heads = 2;
        constexpr int preserved_tokens = 3;
        constexpr int head_dim = 5;
        constexpr int step = 4;
        constexpr int reserve_tokens = 6;
        constexpr int expected_capacity = 8;

        mlx_ctx ctx{};
        ctx.kv_cache.resize(1);
        ctx.kv_reserve_tokens = reserve_tokens;

        KVCacheState& cache = ctx.kv_cache[0];
        cache.step = step;
        cache.offset = preserved_tokens;

        std::vector<float> key_values(static_cast<size_t>(batch * num_kv_heads * preserved_tokens * head_dim));
        std::vector<float> value_values(static_cast<size_t>(batch * num_kv_heads * preserved_tokens * head_dim));
        for (int i = 0; i < static_cast<int>(key_values.size()); ++i) {
            key_values[static_cast<size_t>(i)] = static_cast<float>(((i * 5) % 17) - 8) * 0.0625f;
            value_values[static_cast<size_t>(i)] = static_cast<float>(((i * 7) % 19) - 9) * 0.04125f;
        }

        cache.keys = array(key_values.data(), Shape{batch, num_kv_heads, preserved_tokens, head_dim}, float32);
        cache.values = array(value_values.data(), Shape{batch, num_kv_heads, preserved_tokens, head_dim}, float32);

        const array original_keys = *cache.keys;
        const array original_values = *cache.values;
        ensure_kv_cache_capacity(
            &ctx,
            &cache,
            batch,
            num_kv_heads,
            head_dim,
            preserved_tokens,
            float32,
            float32
        );

        if (!cache.keys.has_value() || !cache.values.has_value()) {
            g_last_error = "mlx_test_kv_cache_reserve_preserves_prefix: missing cache tensors";
            return 0;
        }
        if (cache.keys->shape(2) != expected_capacity || cache.values->shape(2) != expected_capacity) {
            g_last_error = "mlx_test_kv_cache_reserve_preserves_prefix: unexpected reserved capacity";
            return 0;
        }

        const array key_prefix = slice(*cache.keys, {0, 0, 0, 0}, {batch, num_kv_heads, preserved_tokens, head_dim});
        const array value_prefix = slice(*cache.values, {0, 0, 0, 0}, {batch, num_kv_heads, preserved_tokens, head_dim});
        const array key_tail = slice(*cache.keys, {0, 0, preserved_tokens, 0}, {batch, num_kv_heads, expected_capacity, head_dim});
        const array value_tail = slice(*cache.values, {0, 0, preserved_tokens, 0}, {batch, num_kv_heads, expected_capacity, head_dim});

        const array key_prefix_diff = max(reshape(abs(astype(key_prefix, float32) - astype(original_keys, float32)), {static_cast<int>(original_keys.size())}));
        const array value_prefix_diff = max(reshape(abs(astype(value_prefix, float32) - astype(original_values, float32)), {static_cast<int>(original_values.size())}));
        const array key_tail_max = max(reshape(abs(astype(key_tail, float32)), {static_cast<int>(key_tail.size())}));
        const array value_tail_max = max(reshape(abs(astype(value_tail, float32)), {static_cast<int>(value_tail.size())}));
        eval({key_prefix_diff, value_prefix_diff, key_tail_max, value_tail_max});
        synchronize();

        const float* key_prefix_diff_ptr = key_prefix_diff.data<float>();
        const float* value_prefix_diff_ptr = value_prefix_diff.data<float>();
        const float* key_tail_max_ptr = key_tail_max.data<float>();
        const float* value_tail_max_ptr = value_tail_max.data<float>();
        if (!key_prefix_diff_ptr || !value_prefix_diff_ptr || !key_tail_max_ptr || !value_tail_max_ptr) {
            g_last_error = "mlx_test_kv_cache_reserve_preserves_prefix: failed to materialize outputs";
            return 0;
        }
        if (key_prefix_diff_ptr[0] > 1e-6f) {
            g_last_error = "mlx_test_kv_cache_reserve_preserves_prefix: key prefix mismatch";
            return 0;
        }
        if (value_prefix_diff_ptr[0] > 1e-6f) {
            g_last_error = "mlx_test_kv_cache_reserve_preserves_prefix: value prefix mismatch";
            return 0;
        }
        if (key_tail_max_ptr[0] > 1e-6f) {
            g_last_error = "mlx_test_kv_cache_reserve_preserves_prefix: key tail not zeroed";
            return 0;
        }
        if (value_tail_max_ptr[0] > 1e-6f) {
            g_last_error = "mlx_test_kv_cache_reserve_preserves_prefix: value tail not zeroed";
            return 0;
        }
        return 1;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 0;
    } catch (...) {
        g_last_error = "unknown error in mlx_test_kv_cache_reserve_preserves_prefix";
        return 0;
    }
}

int32_t mlx_test_shared_expert_gate_up_fusion(void) {
    try {
        constexpr int hidden = 6;
        constexpr int seq = 2;
        constexpr int ff = 8;

        std::vector<float> input_values(static_cast<size_t>(seq * hidden));
        std::vector<float> gate_weight_values(static_cast<size_t>(ff * hidden));
        std::vector<float> up_weight_values(static_cast<size_t>(ff * hidden));

        for (int i = 0; i < static_cast<int>(input_values.size()); ++i) {
            input_values[static_cast<size_t>(i)] = static_cast<float>(((i * 5) % 13) - 6) * 0.0625f;
        }
        for (int i = 0; i < static_cast<int>(gate_weight_values.size()); ++i) {
            gate_weight_values[static_cast<size_t>(i)] = static_cast<float>(((i * 7) % 17) - 8) * 0.03125f;
            up_weight_values[static_cast<size_t>(i)] = static_cast<float>(((i * 3) % 19) - 9) * 0.04125f;
        }

        mlx_ctx ctx{};
        ctx.cfg.quant_group_size = 32;
        ctx.cfg.quant_bits = 4;
        ctx.cfg.rms_norm_eps = 1.0e-6f;

        LayerWeights lw{};
        const array gate_weight(gate_weight_values.data(), Shape{ff, hidden}, float32);
        const array up_weight(up_weight_values.data(), Shape{ff, hidden}, float32);
        lw.moe_shared_gate_rhs = to_rhs(gate_weight, "mlx_test_shared_expert_gate_up_fusion.gate");
        lw.moe_shared_up_rhs = to_rhs(up_weight, "mlx_test_shared_expert_gate_up_fusion.up");
        if (!combine_shared_expert_gate_up_weights(&lw)) {
            g_last_error = "mlx_test_shared_expert_gate_up_fusion: failed to combine shared gate/up weights";
            return 0;
        }

        const array mlp_in(input_values.data(), Shape{1, seq, hidden}, float32);
        const array gate_split = linear_decode_maybe_quantized(
            false,
            ctx.cfg.quant_group_size,
            ctx.cfg.quant_bits,
            mlp_in,
            lw.moe_shared_gate_rhs,
            false,
            lw.moe_shared_gate_q_w,
            lw.moe_shared_gate_q_scales,
            lw.moe_shared_gate_q_biases,
            false
        );
        const array up_split = linear_decode_maybe_quantized(
            false,
            ctx.cfg.quant_group_size,
            ctx.cfg.quant_bits,
            mlp_in,
            lw.moe_shared_up_rhs,
            false,
            lw.moe_shared_up_q_w,
            lw.moe_shared_up_q_scales,
            lw.moe_shared_up_q_biases,
            false
        );
        const array gate_up_fused = linear_decode_maybe_quantized(
            false,
            ctx.cfg.quant_group_size,
            ctx.cfg.quant_bits,
            mlp_in,
            lw.moe_shared_gate_up_rhs,
            false,
            lw.moe_shared_gate_up_q_w,
            lw.moe_shared_gate_up_q_scales,
            lw.moe_shared_gate_up_q_biases,
            false
        );
        const array gate_fused = slice(gate_up_fused, {0, 0, 0}, {1, seq, ff});
        const array up_fused = slice(gate_up_fused, {0, 0, ff}, {1, seq, 2 * ff});

        const array gate_diff = max(reshape(abs(astype(gate_split, float32) - astype(gate_fused, float32)), {static_cast<int>(gate_split.size())}));
        const array up_diff = max(reshape(abs(astype(up_split, float32) - astype(up_fused, float32)), {static_cast<int>(up_split.size())}));
        eval({gate_diff, up_diff});
        synchronize();

        const float* gate_diff_ptr = gate_diff.data<float>();
        const float* up_diff_ptr = up_diff.data<float>();
        if (!gate_diff_ptr || !up_diff_ptr) {
            g_last_error = "mlx_test_shared_expert_gate_up_fusion: failed to materialize diffs";
            return 0;
        }
        if (gate_diff_ptr[0] > 1e-6f) {
            g_last_error = "mlx_test_shared_expert_gate_up_fusion: gate output mismatch";
            return 0;
        }
        if (up_diff_ptr[0] > 1e-6f) {
            g_last_error = "mlx_test_shared_expert_gate_up_fusion: up output mismatch";
            return 0;
        }
        return 1;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 0;
    } catch (...) {
        g_last_error = "unknown error in mlx_test_shared_expert_gate_up_fusion";
        return 0;
    }
}

int32_t mlx_test_dense_mlp_gate_up_fusion(void) {
    try {
        constexpr int hidden = 6;
        constexpr int seq = 2;
        constexpr int ff = 8;

        std::vector<float> input_values(static_cast<size_t>(seq * hidden));
        std::vector<float> gate_weight_values(static_cast<size_t>(ff * hidden));
        std::vector<float> up_weight_values(static_cast<size_t>(ff * hidden));

        for (int i = 0; i < static_cast<int>(input_values.size()); ++i) {
            input_values[static_cast<size_t>(i)] = static_cast<float>(((i * 11) % 23) - 11) * 0.0625f;
        }
        for (int i = 0; i < static_cast<int>(gate_weight_values.size()); ++i) {
            gate_weight_values[static_cast<size_t>(i)] = static_cast<float>(((i * 5) % 17) - 8) * 0.03125f;
            up_weight_values[static_cast<size_t>(i)] = static_cast<float>(((i * 7) % 19) - 9) * 0.04125f;
        }

        mlx_ctx ctx{};
        ctx.cfg.quant_group_size = 32;
        ctx.cfg.quant_bits = 4;
        ctx.cfg.rms_norm_eps = 1.0e-6f;

        LayerWeights lw{};
        const array gate_weight(gate_weight_values.data(), Shape{ff, hidden}, float32);
        const array up_weight(up_weight_values.data(), Shape{ff, hidden}, float32);
        lw.mlp_gate_rhs = to_rhs(gate_weight, "mlx_test_dense_mlp_gate_up_fusion.gate");
        lw.mlp_up_rhs = to_rhs(up_weight, "mlx_test_dense_mlp_gate_up_fusion.up");
        if (!combine_dense_mlp_gate_up_weights(&lw)) {
            g_last_error = "mlx_test_dense_mlp_gate_up_fusion: failed to combine dense gate/up weights";
            return 0;
        }

        const array mlp_in(input_values.data(), Shape{1, seq, hidden}, float32);
        const array gate_split = linear_decode_maybe_quantized(
            false,
            ctx.cfg.quant_group_size,
            ctx.cfg.quant_bits,
            mlp_in,
            lw.mlp_gate_rhs,
            false,
            lw.mlp_gate_q_w,
            lw.mlp_gate_q_scales,
            lw.mlp_gate_q_biases,
            false
        );
        const array up_split = linear_decode_maybe_quantized(
            false,
            ctx.cfg.quant_group_size,
            ctx.cfg.quant_bits,
            mlp_in,
            lw.mlp_up_rhs,
            false,
            lw.mlp_up_q_w,
            lw.mlp_up_q_scales,
            lw.mlp_up_q_biases,
            false
        );
        const array gate_up_fused = linear_decode_maybe_quantized(
            false,
            ctx.cfg.quant_group_size,
            ctx.cfg.quant_bits,
            mlp_in,
            lw.mlp_gate_up_rhs,
            false,
            lw.mlp_gate_up_q_w,
            lw.mlp_gate_up_q_scales,
            lw.mlp_gate_up_q_biases,
            false
        );
        const array gate_fused = slice(gate_up_fused, {0, 0, 0}, {1, seq, ff});
        const array up_fused = slice(gate_up_fused, {0, 0, ff}, {1, seq, 2 * ff});

        const array gate_diff = max(reshape(abs(astype(gate_split, float32) - astype(gate_fused, float32)), {static_cast<int>(gate_split.size())}));
        const array up_diff = max(reshape(abs(astype(up_split, float32) - astype(up_fused, float32)), {static_cast<int>(up_split.size())}));
        eval({gate_diff, up_diff});
        synchronize();

        const float* gate_diff_ptr = gate_diff.data<float>();
        const float* up_diff_ptr = up_diff.data<float>();
        if (!gate_diff_ptr || !up_diff_ptr) {
            g_last_error = "mlx_test_dense_mlp_gate_up_fusion: failed to materialize diffs";
            return 0;
        }
        if (gate_diff_ptr[0] > 1e-6f) {
            g_last_error = "mlx_test_dense_mlp_gate_up_fusion: gate output mismatch";
            return 0;
        }
        if (up_diff_ptr[0] > 1e-6f) {
            g_last_error = "mlx_test_dense_mlp_gate_up_fusion: up output mismatch";
            return 0;
        }

        compact_dense_mlp_gate_up_weights(&lw);
        if (lw.mlp_gate_rhs.size() > 1 || lw.mlp_up_rhs.size() > 1 || lw.mlp_gate_has_q || lw.mlp_up_has_q) {
            g_last_error = "mlx_test_dense_mlp_gate_up_fusion: compaction did not clear split dense MLP projections";
            return 0;
        }
        const array gate_up_compacted = linear_decode_maybe_quantized(
            false,
            ctx.cfg.quant_group_size,
            ctx.cfg.quant_bits,
            mlp_in,
            lw.mlp_gate_up_rhs,
            false,
            lw.mlp_gate_up_q_w,
            lw.mlp_gate_up_q_scales,
            lw.mlp_gate_up_q_biases,
            false
        );
        const array compact_diff = max(reshape(
            abs(astype(gate_up_fused, float32) - astype(gate_up_compacted, float32)),
            {static_cast<int>(gate_up_fused.size())}
        ));
        eval(compact_diff);
        synchronize();
        const float* compact_diff_ptr = compact_diff.data<float>();
        if (!compact_diff_ptr) {
            g_last_error = "mlx_test_dense_mlp_gate_up_fusion: failed to materialize compact diff";
            return 0;
        }
        if (compact_diff_ptr[0] > 1e-6f) {
            g_last_error = "mlx_test_dense_mlp_gate_up_fusion: compacted fused output mismatch";
            return 0;
        }
        return 1;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 0;
    } catch (...) {
        g_last_error = "unknown error in mlx_test_dense_mlp_gate_up_fusion";
        return 0;
    }
}

int32_t mlx_test_full_attention_qkv_fusion(void) {
    try {
        constexpr int hidden = 6;
        constexpr int seq = 2;
        constexpr int q_width = 12;
        constexpr int k_width = 4;
        constexpr int v_width = 4;

        std::vector<float> input_values(static_cast<size_t>(seq * hidden));
        std::vector<float> q_weight_values(static_cast<size_t>(q_width * hidden));
        std::vector<float> k_weight_values(static_cast<size_t>(k_width * hidden));
        std::vector<float> v_weight_values(static_cast<size_t>(v_width * hidden));

        for (int i = 0; i < static_cast<int>(input_values.size()); ++i) {
            input_values[static_cast<size_t>(i)] = static_cast<float>(((i * 5) % 11) - 5) * 0.0625f;
        }
        for (int i = 0; i < static_cast<int>(q_weight_values.size()); ++i) {
            q_weight_values[static_cast<size_t>(i)] = static_cast<float>(((i * 7) % 17) - 8) * 0.03125f;
        }
        for (int i = 0; i < static_cast<int>(k_weight_values.size()); ++i) {
            k_weight_values[static_cast<size_t>(i)] = static_cast<float>(((i * 3) % 13) - 6) * 0.04125f;
        }
        for (int i = 0; i < static_cast<int>(v_weight_values.size()); ++i) {
            v_weight_values[static_cast<size_t>(i)] = static_cast<float>(((i * 9) % 19) - 9) * 0.0375f;
        }

        mlx_ctx ctx{};
        ctx.cfg.quant_group_size = 32;
        ctx.cfg.quant_bits = 4;

        LayerWeights lw{};
        const array q_weight(q_weight_values.data(), Shape{q_width, hidden}, float32);
        const array k_weight(k_weight_values.data(), Shape{k_width, hidden}, float32);
        const array v_weight(v_weight_values.data(), Shape{v_width, hidden}, float32);
        lw.attn_q_rhs = to_rhs(q_weight, "mlx_test_full_attention_qkv_fusion.q");
        lw.attn_k_rhs = to_rhs(k_weight, "mlx_test_full_attention_qkv_fusion.k");
        lw.attn_v_rhs = to_rhs(v_weight, "mlx_test_full_attention_qkv_fusion.v");
        if (!combine_full_attention_qkv_weights(&lw)) {
            g_last_error = "mlx_test_full_attention_qkv_fusion: failed to combine qkv weights";
            return 0;
        }

        const array x_norm(input_values.data(), Shape{1, seq, hidden}, float32);
        const array q_split = linear_decode_maybe_quantized(
            false,
            ctx.cfg.quant_group_size,
            ctx.cfg.quant_bits,
            x_norm,
            lw.attn_q_rhs,
            false,
            lw.attn_q_q_w,
            lw.attn_q_q_scales,
            lw.attn_q_q_biases,
            false
        );
        const array k_split = linear_decode_maybe_quantized(
            false,
            ctx.cfg.quant_group_size,
            ctx.cfg.quant_bits,
            x_norm,
            lw.attn_k_rhs,
            false,
            lw.attn_k_q_w,
            lw.attn_k_q_scales,
            lw.attn_k_q_biases,
            false
        );
        const array v_split = linear_decode_maybe_quantized(
            false,
            ctx.cfg.quant_group_size,
            ctx.cfg.quant_bits,
            x_norm,
            lw.attn_v_rhs,
            false,
            lw.attn_v_q_w,
            lw.attn_v_q_scales,
            lw.attn_v_q_biases,
            false
        );
        const array qkv_fused = linear_decode_maybe_quantized(
            false,
            ctx.cfg.quant_group_size,
            ctx.cfg.quant_bits,
            x_norm,
            lw.attn_qkv_rhs,
            false,
            lw.attn_qkv_q_w,
            lw.attn_qkv_q_scales,
            lw.attn_qkv_q_biases,
            false
        );
        const array q_fused = slice(qkv_fused, {0, 0, 0}, {1, seq, q_width});
        const array k_fused = slice(qkv_fused, {0, 0, q_width}, {1, seq, q_width + k_width});
        const array v_fused = slice(qkv_fused, {0, 0, q_width + k_width}, {1, seq, q_width + k_width + v_width});

        const array q_diff = max(reshape(abs(astype(q_split, float32) - astype(q_fused, float32)), {static_cast<int>(q_split.size())}));
        const array k_diff = max(reshape(abs(astype(k_split, float32) - astype(k_fused, float32)), {static_cast<int>(k_split.size())}));
        const array v_diff = max(reshape(abs(astype(v_split, float32) - astype(v_fused, float32)), {static_cast<int>(v_split.size())}));
        eval({q_diff, k_diff, v_diff});
        synchronize();

        const float* q_diff_ptr = q_diff.data<float>();
        const float* k_diff_ptr = k_diff.data<float>();
        const float* v_diff_ptr = v_diff.data<float>();
        if (!q_diff_ptr || !k_diff_ptr || !v_diff_ptr) {
            g_last_error = "mlx_test_full_attention_qkv_fusion: failed to materialize diffs";
            return 0;
        }
        if (q_diff_ptr[0] > 1e-6f) {
            g_last_error = "mlx_test_full_attention_qkv_fusion: q output mismatch";
            return 0;
        }
        if (k_diff_ptr[0] > 1e-6f) {
            g_last_error = "mlx_test_full_attention_qkv_fusion: k output mismatch";
            return 0;
        }
        if (v_diff_ptr[0] > 1e-6f) {
            g_last_error = "mlx_test_full_attention_qkv_fusion: v output mismatch";
            return 0;
        }

        compact_full_attention_qkv_weights(&lw);
        if (lw.attn_q_rhs.size() > 1 || lw.attn_k_rhs.size() > 1 || lw.attn_v_rhs.size() > 1 ||
            lw.attn_q_has_q || lw.attn_k_has_q || lw.attn_v_has_q)
        {
            g_last_error = "mlx_test_full_attention_qkv_fusion: compaction did not clear split attention projections";
            return 0;
        }
        const array qkv_compacted = linear_decode_maybe_quantized(
            false,
            ctx.cfg.quant_group_size,
            ctx.cfg.quant_bits,
            x_norm,
            lw.attn_qkv_rhs,
            false,
            lw.attn_qkv_q_w,
            lw.attn_qkv_q_scales,
            lw.attn_qkv_q_biases,
            false
        );
        const array compact_diff = max(reshape(
            abs(astype(qkv_fused, float32) - astype(qkv_compacted, float32)),
            {static_cast<int>(qkv_fused.size())}
        ));
        eval(compact_diff);
        synchronize();
        const float* compact_diff_ptr = compact_diff.data<float>();
        if (!compact_diff_ptr) {
            g_last_error = "mlx_test_full_attention_qkv_fusion: failed to materialize compact diff";
            return 0;
        }
        if (compact_diff_ptr[0] > 1e-6f) {
            g_last_error = "mlx_test_full_attention_qkv_fusion: compacted fused output mismatch";
            return 0;
        }
        return 1;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 0;
    } catch (...) {
        g_last_error = "unknown error in mlx_test_full_attention_qkv_fusion";
        return 0;
    }
}

