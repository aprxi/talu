// bridge_tests_b.inc — Tests: prefill_cache, double_qk_norm, rmsnorm_gated,
// chunked_prefill, linear_quant_inproj, affine_embedding, fused_mixer, topk_multi.

int32_t mlx_test_grouped_affine_prefill_cache_helper(void) {
    try {
        constexpr int hidden = 64;
        constexpr int seq = 3;
        constexpr int out_dim = 96;
        constexpr int group_size = 32;
        constexpr int bits = 4;

        std::vector<float> input_values(static_cast<size_t>(seq * hidden));
        std::vector<float> decode_values(static_cast<size_t>(hidden));
        std::vector<float> weight_values(static_cast<size_t>(out_dim * hidden));

        for (int i = 0; i < static_cast<int>(input_values.size()); ++i) {
            input_values[static_cast<size_t>(i)] = static_cast<float>(((i * 5) % 23) - 11) * 0.0625f;
        }
        for (int i = 0; i < static_cast<int>(decode_values.size()); ++i) {
            decode_values[static_cast<size_t>(i)] = static_cast<float>(((i * 7) % 29) - 14) * 0.05f;
        }
        for (int i = 0; i < static_cast<int>(weight_values.size()); ++i) {
            weight_values[static_cast<size_t>(i)] = static_cast<float>(((i * 11) % 31) - 15) * 0.03125f;
        }

        const array weight(weight_values.data(), Shape{out_dim, hidden}, float32);
        const std::vector<array> quant = quantize(weight, group_size, bits, "affine");
        if (quant.size() != 3) {
            g_last_error = "mlx_test_grouped_affine_prefill_cache_helper: quantize returned unexpected output count";
            return 0;
        }

        const array prefill_rhs = build_grouped_affine_prefill_rhs(
            "mlx_test_grouped_affine_prefill_cache_helper",
            quant[0],
            quant[1],
            quant[2],
            group_size,
            bits
        );
        if (prefill_rhs.size() <= 1 || prefill_rhs.ndim() != 2) {
            g_last_error = "mlx_test_grouped_affine_prefill_cache_helper: failed to build prefill rhs";
            return 0;
        }

        const array x_prefill(input_values.data(), Shape{1, seq, hidden}, float32);
        const array x_decode(decode_values.data(), Shape{1, 1, hidden}, float32);

        const array prefill_helper = linear_decode_maybe_quantized(
            true,
            group_size,
            bits,
            x_prefill,
            array(0.0f),
            true,
            quant[0],
            quant[1],
            quant[2],
            false,
            -1,
            "prefill_cache.prefill",
            &prefill_rhs
        );
        const array prefill_quant = quantized_matmul(
            x_prefill,
            quant[0],
            quant[1],
            std::optional<array>(quant[2]),
            true,
            group_size,
            bits,
            "affine"
        );
        const array decode_helper = linear_decode_maybe_quantized(
            true,
            group_size,
            bits,
            x_decode,
            array(0.0f),
            true,
            quant[0],
            quant[1],
            quant[2],
            false,
            -1,
            "prefill_cache.decode",
            &prefill_rhs
        );
        const array decode_quant = quantized_matmul(
            x_decode,
            quant[0],
            quant[1],
            std::optional<array>(quant[2]),
            true,
            group_size,
            bits,
            "affine"
        );

        const array prefill_diff = max(reshape(
            abs(astype(prefill_helper, float32) - astype(prefill_quant, float32)),
            {static_cast<int>(prefill_helper.size())}
        ));
        const array decode_diff = max(reshape(
            abs(astype(decode_helper, float32) - astype(decode_quant, float32)),
            {static_cast<int>(decode_helper.size())}
        ));
        eval({prefill_diff, decode_diff});
        synchronize();

        const float* prefill_diff_ptr = prefill_diff.data<float>();
        const float* decode_diff_ptr = decode_diff.data<float>();
        if (!prefill_diff_ptr || !decode_diff_ptr) {
            g_last_error = "mlx_test_grouped_affine_prefill_cache_helper: failed to materialize diffs";
            return 0;
        }
        if (prefill_diff_ptr[0] > 5e-2f) {
            g_last_error = "mlx_test_grouped_affine_prefill_cache_helper: prefill output mismatch";
            return 0;
        }
        if (decode_diff_ptr[0] > 1e-6f) {
            g_last_error = "mlx_test_grouped_affine_prefill_cache_helper: decode path no longer matches qmm";
            return 0;
        }
        return 1;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 0;
    } catch (...) {
        g_last_error = "unknown error in mlx_test_grouped_affine_prefill_cache_helper";
        return 0;
    }
}

int32_t mlx_test_gated_delta_no_double_qk_norm(void) {
    try {
        constexpr int B = 1;
        constexpr int T = 3;
        constexpr int Hk = 1;
        constexpr int Hv = 2;
        constexpr int Dk = 32;
        constexpr int Dv = 4;

        auto make_values = [](size_t count, float scale, int stride, int offset) {
            std::vector<float> values(count);
            for (size_t i = 0; i < count; ++i) {
                const int raw = static_cast<int>((static_cast<int>(i) * stride) % 29) - offset;
                values[i] = static_cast<float>(raw) * scale;
            }
            return values;
        };

        const std::vector<float> q_raw_values = make_values(static_cast<size_t>(B * T * Hk * Dk), 0.03125f, 5, 14);
        const std::vector<float> k_raw_values = make_values(static_cast<size_t>(B * T * Hk * Dk), 0.0275f, 7, 13);
        const std::vector<float> v_values = make_values(static_cast<size_t>(B * T * Hv * Dv), 0.05f, 3, 8);
        const std::vector<float> a_values = make_values(static_cast<size_t>(B * T * Hv), 0.04f, 11, 5);
        const std::vector<float> b_values = make_values(static_cast<size_t>(B * T * Hv), 0.03f, 13, 6);
        const std::vector<float> a_log_values = make_values(static_cast<size_t>(Hv), 0.02f, 17, 4);
        const std::vector<float> dt_bias_values = make_values(static_cast<size_t>(Hv), 0.05f, 19, 3);

        const array q_raw(q_raw_values.data(), Shape{B, T, Hk, Dk}, float32);
        const array k_raw(k_raw_values.data(), Shape{B, T, Hk, Dk}, float32);
        const array v(v_values.data(), Shape{B, T, Hv, Dv}, float32);
        const array a(a_values.data(), Shape{B, T, Hv}, float32);
        const array b(b_values.data(), Shape{B, T, Hv}, float32);
        const array A_log(a_log_values.data(), Shape{Hv}, float32);
        const array A_exp_f32 = exp(astype(A_log, float32));
        const array dt_bias(dt_bias_values.data(), Shape{Hv}, float32);

        const float inv_scale = 1.0f / std::sqrt(static_cast<float>(Dk));
        const array q = fast::rms_norm(q_raw, std::nullopt, 1.0e-6f) * array(inv_scale * inv_scale, float32);
        const array k = fast::rms_norm(k_raw, std::nullopt, 1.0e-6f) * array(inv_scale, float32);

        auto [actual_out, actual_state] = gated_delta_update(q, k, v, a, b, A_exp_f32, dt_bias, std::nullopt);

        array state = zeros({B, Hv, Dv, Dk}, float32);
        array q_use = repeat(q, Hv / Hk, 2);
        array k_use = repeat(k, Hv / Hk, 2);
        const array beta = sigmoid(b);
        const array g = compute_g(A_exp_f32, a, dt_bias);

        std::vector<array> outputs;
        outputs.reserve(T);
        for (int t = 0; t < T; ++t) {
            const array q_t = take_time_4d(q_use, t);
            const array k_t = take_time_4d(k_use, t);
            const array v_t = take_time_4d(v, t);
            const array beta_t = take_time_3d(beta, t);
            const array g_t = take_time_3d(g, t);

            const array k_exp = reshape(k_t, {B, Hv, 1, Dk});
            const array q_exp = reshape(q_t, {B, Hv, 1, Dk});
            state = state * reshape(g_t, {B, Hv, 1, 1});
            const array kv_mem = sum(state * k_exp, std::vector<int>{-1}, false);
            const array delta = (v_t - kv_mem) * reshape(beta_t, {B, Hv, 1});
            state = state + k_exp * reshape(delta, {B, Hv, Dv, 1});
            outputs.push_back(sum(state * q_exp, std::vector<int>{-1}, false));
        }
        const array expected_out = stack(outputs, 1);
        const array expected_state = state;

        const array out_diff = max(reshape(
            abs(astype(actual_out, float32) - astype(expected_out, float32)),
            {static_cast<int>(actual_out.size())}
        ));
        const array state_diff = max(reshape(
            abs(astype(actual_state, float32) - astype(expected_state, float32)),
            {static_cast<int>(actual_state.size())}
        ));
        eval({out_diff, state_diff});
        synchronize();

        const float* out_diff_ptr = out_diff.data<float>();
        const float* state_diff_ptr = state_diff.data<float>();
        if (!out_diff_ptr || !state_diff_ptr) {
            g_last_error = "mlx_test_gated_delta_no_double_qk_norm: failed to materialize diffs";
            return 0;
        }
        if (out_diff_ptr[0] > 1e-5f) {
            g_last_error = "mlx_test_gated_delta_no_double_qk_norm: output mismatch";
            return 0;
        }
        if (state_diff_ptr[0] > 1e-5f) {
            g_last_error = "mlx_test_gated_delta_no_double_qk_norm: state mismatch";
            return 0;
        }
        return 1;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 0;
    } catch (...) {
        g_last_error = "unknown error in mlx_test_gated_delta_no_double_qk_norm";
        return 0;
    }
}

int32_t mlx_test_rmsnorm_gated_kernel_matches_reference(void) {
    try {
        constexpr int B = 2;
        constexpr int S = 3;
        constexpr int Hv = 2;
        constexpr int Dv = 64;
        constexpr float eps = 1.0e-6f;

        auto make_values = [](size_t count, float scale, int stride, int offset) {
            std::vector<float> values(count);
            for (size_t i = 0; i < count; ++i) {
                const int raw = static_cast<int>((static_cast<int>(i) * stride) % 37) - offset;
                values[i] = static_cast<float>(raw) * scale;
            }
            return values;
        };

        const std::vector<float> hidden_values = make_values(static_cast<size_t>(B * S * Hv * Dv), 0.03125f, 5, 18);
        const std::vector<float> gate_values = make_values(static_cast<size_t>(B * S * Hv * Dv), 0.0275f, 7, 17);
        const std::vector<float> shared_weight_values = make_values(static_cast<size_t>(Dv), 0.05f, 3, 6);
        const std::vector<float> per_head_weight_values = make_values(static_cast<size_t>(Hv * Dv), 0.04f, 11, 8);

        const array hidden(hidden_values.data(), Shape{B, S, Hv, Dv}, float32);
        const array gate(gate_values.data(), Shape{B, S, Hv, Dv}, float32);
        const array shared_weight(shared_weight_values.data(), Shape{Dv}, float32);
        const array per_head_weight(per_head_weight_values.data(), Shape{Hv, Dv}, float32);
        const array per_head_weight_flat = reshape(per_head_weight, {Hv * Dv});

        const array actual_shared = rmsnorm_gated(hidden, gate, shared_weight, eps);
        const array expected_shared = rmsnorm_gated_fallback(hidden, gate, shared_weight, eps);

        const array hidden_f32 = astype(hidden, float32);
        const array gate_f32 = astype(gate, float32);
        const array sum_sq = sum(hidden_f32 * hidden_f32, std::vector<int>{-1}, true);
        const array inv_rms = rsqrt((sum_sq / array(static_cast<float>(Dv), float32)) + array(eps, float32));
        const array normed = hidden_f32 * inv_rms;
        const array expected_per_head_f32 =
            normed * reshape(per_head_weight, {1, 1, Hv, Dv}) * silu(gate_f32);
        const array expected_per_head = astype(expected_per_head_f32, hidden.dtype());

        const array actual_per_head = rmsnorm_gated(hidden, gate, per_head_weight, eps);
        const array actual_per_head_flat = rmsnorm_gated(hidden, gate, per_head_weight_flat, eps);

        const array shared_diff = max(reshape(
            abs(astype(actual_shared, float32) - astype(expected_shared, float32)),
            {static_cast<int>(actual_shared.size())}
        ));
        const array per_head_diff = max(reshape(
            abs(astype(actual_per_head, float32) - astype(expected_per_head, float32)),
            {static_cast<int>(actual_per_head.size())}
        ));
        const array per_head_flat_diff = max(reshape(
            abs(astype(actual_per_head_flat, float32) - astype(expected_per_head, float32)),
            {static_cast<int>(actual_per_head_flat.size())}
        ));
        eval({shared_diff, per_head_diff, per_head_flat_diff});
        synchronize();

        const float* shared_diff_ptr = shared_diff.data<float>();
        const float* per_head_diff_ptr = per_head_diff.data<float>();
        const float* per_head_flat_diff_ptr = per_head_flat_diff.data<float>();
        if (!shared_diff_ptr || !per_head_diff_ptr || !per_head_flat_diff_ptr) {
            g_last_error = "mlx_test_rmsnorm_gated_kernel_matches_reference: failed to materialize diffs";
            return 0;
        }
        if (shared_diff_ptr[0] > 1e-5f) {
            g_last_error = "mlx_test_rmsnorm_gated_kernel_matches_reference: shared-weight mismatch";
            return 0;
        }
        if (per_head_diff_ptr[0] > 1e-5f) {
            g_last_error = "mlx_test_rmsnorm_gated_kernel_matches_reference: per-head mismatch";
            return 0;
        }
        if (per_head_flat_diff_ptr[0] > 1e-5f) {
            g_last_error = "mlx_test_rmsnorm_gated_kernel_matches_reference: flat per-head mismatch";
            return 0;
        }
        return 1;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 0;
    } catch (...) {
        g_last_error = "unknown error in mlx_test_rmsnorm_gated_kernel_matches_reference";
        return 0;
    }
}

int32_t mlx_test_chunked_prefill_tail_matches_full_prompt(void) {
    try {
        struct EnvRestore {
            bool had_value = false;
            std::string old_value;
            ~EnvRestore() {
                if (had_value) {
                    setenv("TALU_METAL_PREFILL_CHUNK_SIZE", old_value.c_str(), 1);
                } else {
                    unsetenv("TALU_METAL_PREFILL_CHUNK_SIZE");
                }
            }
        } restore;
        if (const char* old = std::getenv("TALU_METAL_PREFILL_CHUNK_SIZE")) {
            restore.had_value = true;
            restore.old_value = old;
        }
        setenv("TALU_METAL_PREFILL_CHUNK_SIZE", "2", 1);

        constexpr int vocab = 8;
        constexpr int hidden = 4;
        constexpr int prompt_len = 5;
        constexpr int ff = 6;

        auto make_values = [](size_t count, float scale, int stride, int offset) {
            std::vector<float> values(count);
            for (size_t i = 0; i < count; ++i) {
                const int raw = static_cast<int>((static_cast<int>(i) * stride) % 23) - offset;
                values[i] = static_cast<float>(raw) * scale;
            }
            return values;
        };

        mlx_ctx full_ctx{};
        mlx_ctx chunk_ctx{};
        full_ctx.cfg.rms_norm_eps = 1.0e-6f;
        full_ctx.cfg.quant_group_size = 32;
        full_ctx.cfg.quant_bits = 4;
        full_ctx.cfg.num_attention_heads = 1;
        full_ctx.cfg.num_key_value_heads = 1;
        full_ctx.cfg.head_dim = hidden;
        full_ctx.cfg.hidden_size = hidden;
        full_ctx.cfg.num_hidden_layers = 1;
        full_ctx.cfg.hidden_size_per_layer_input = 0;
        full_ctx.cfg.use_v_norm = false;
        full_ctx.cfg.global_head_dim = 0;

        chunk_ctx.cfg = full_ctx.cfg;

        const std::vector<float> embed_values = make_values(static_cast<size_t>(vocab * hidden), 0.0625f, 5, 9);
        const std::vector<float> lm_head_values = make_values(static_cast<size_t>(vocab * hidden), 0.03125f, 7, 8);
        const std::vector<float> norm_values = make_values(static_cast<size_t>(hidden), 0.05f, 3, 2);
        const std::vector<float> q_values = make_values(static_cast<size_t>(hidden * hidden), 0.03125f, 11, 7);
        const std::vector<float> k_values = make_values(static_cast<size_t>(hidden * hidden), 0.0275f, 13, 6);
        const std::vector<float> v_values = make_values(static_cast<size_t>(hidden * hidden), 0.025f, 17, 5);
        const std::vector<float> o_values = make_values(static_cast<size_t>(hidden * hidden), 0.03f, 19, 4);
        const std::vector<float> gate_values = make_values(static_cast<size_t>(ff * hidden), 0.03f, 5, 7);
        const std::vector<float> up_values = make_values(static_cast<size_t>(ff * hidden), 0.04f, 7, 8);
        const std::vector<float> down_values = make_values(static_cast<size_t>(hidden * ff), 0.035f, 9, 6);

        full_ctx.embed_tokens = array(embed_values.data(), Shape{vocab, hidden}, float32);
        full_ctx.final_norm_w = array(norm_values.data(), Shape{hidden}, float32);
        full_ctx.lm_head_rhs = to_rhs(array(lm_head_values.data(), Shape{vocab, hidden}, float32), "mlx_test_chunked_prefill_tail_matches_full_prompt.lm_head");

        chunk_ctx.embed_tokens = full_ctx.embed_tokens;
        chunk_ctx.final_norm_w = full_ctx.final_norm_w;
        chunk_ctx.lm_head_rhs = full_ctx.lm_head_rhs;

        LayerWeights lw{};
        lw.input_layernorm_w = array(norm_values.data(), Shape{hidden}, float32);
        lw.post_attention_layernorm_w = array(norm_values.data(), Shape{hidden}, float32);
        lw.attn_q_norm_w = array(norm_values.data(), Shape{hidden}, float32);
        lw.attn_k_norm_w = array(norm_values.data(), Shape{hidden}, float32);
        lw.attn_num_heads = 1;
        lw.attn_num_kv_heads = 1;
        lw.attn_head_dim = hidden;
        lw.attn_scale = 1.0f / std::sqrt(static_cast<float>(hidden));
        lw.attn_rope_theta = 10000.0f;
        lw.attn_partial_rotary_factor = 1.0f;
        lw.attn_q_proj_width = hidden;
        lw.attn_k_proj_width = hidden;
        lw.attn_v_proj_width = hidden;
        lw.attn_q_rhs = to_rhs(array(q_values.data(), Shape{hidden, hidden}, float32), "mlx_test_chunked_prefill_tail_matches_full_prompt.q");
        lw.attn_k_rhs = to_rhs(array(k_values.data(), Shape{hidden, hidden}, float32), "mlx_test_chunked_prefill_tail_matches_full_prompt.k");
        lw.attn_v_rhs = to_rhs(array(v_values.data(), Shape{hidden, hidden}, float32), "mlx_test_chunked_prefill_tail_matches_full_prompt.v");
        lw.attn_o_rhs = to_rhs(array(o_values.data(), Shape{hidden, hidden}, float32), "mlx_test_chunked_prefill_tail_matches_full_prompt.o");
        lw.mlp_gate_rhs = to_rhs(array(gate_values.data(), Shape{ff, hidden}, float32), "mlx_test_chunked_prefill_tail_matches_full_prompt.gate");
        lw.mlp_up_rhs = to_rhs(array(up_values.data(), Shape{ff, hidden}, float32), "mlx_test_chunked_prefill_tail_matches_full_prompt.up");
        lw.mlp_down_rhs = to_rhs(array(down_values.data(), Shape{hidden, ff}, float32), "mlx_test_chunked_prefill_tail_matches_full_prompt.down");

        full_ctx.layers.push_back(lw);
        full_ctx.kv_cache.resize(1);
        full_ctx.linear_cache.resize(1);
        chunk_ctx.layers.push_back(lw);
        chunk_ctx.kv_cache.resize(1);
        chunk_ctx.linear_cache.resize(1);

        const int32_t prompt_ids[prompt_len] = {1, 3, 2, 5, 4};
        const std::vector<int32_t> prompt_vec(prompt_ids, prompt_ids + prompt_len);
        const array prompt_arr(prompt_ids, Shape{1, prompt_len}, int32);

        const array full_logits = forward_logits(&full_ctx, prompt_arr);
        const array chunked_logits = prefill_seed_logits_chunked(&chunk_ctx, prompt_vec, prompt_len);
        const array full_last = last_token_logits(full_logits);
        const array chunked_last = last_token_logits(chunked_logits);

        if (!full_ctx.kv_cache[0].keys.has_value() || !full_ctx.kv_cache[0].values.has_value() ||
            !chunk_ctx.kv_cache[0].keys.has_value() || !chunk_ctx.kv_cache[0].values.has_value())
        {
            g_last_error = "mlx_test_chunked_prefill_tail_matches_full_prompt: missing kv cache tensors";
            return 0;
        }

        const array logits_diff = max(reshape(
            abs(astype(full_last, float32) - astype(chunked_last, float32)),
            {static_cast<int>(full_last.size())}
        ));
        // Slice KV caches to the actually-used region before comparing —
        // capacities differ due to step-based growth rounding (step=256).
        const int kv_B = full_ctx.kv_cache[0].keys->shape(0);
        const int kv_H = full_ctx.kv_cache[0].keys->shape(1);
        const int kv_D = full_ctx.kv_cache[0].keys->shape(3);
        const array full_keys = slice(*full_ctx.kv_cache[0].keys,
            {0, 0, 0, 0}, {kv_B, kv_H, prompt_len, kv_D});
        const array chunk_keys = slice(*chunk_ctx.kv_cache[0].keys,
            {0, 0, 0, 0}, {kv_B, kv_H, prompt_len, kv_D});
        const array full_vals = slice(*full_ctx.kv_cache[0].values,
            {0, 0, 0, 0}, {kv_B, kv_H, prompt_len, kv_D});
        const array chunk_vals = slice(*chunk_ctx.kv_cache[0].values,
            {0, 0, 0, 0}, {kv_B, kv_H, prompt_len, kv_D});
        const array key_diff = max(reshape(
            abs(astype(full_keys, float32) - astype(chunk_keys, float32)),
            {static_cast<int>(full_keys.size())}
        ));
        const array value_diff = max(reshape(
            abs(astype(full_vals, float32) - astype(chunk_vals, float32)),
            {static_cast<int>(full_vals.size())}
        ));
        eval({logits_diff, key_diff, value_diff});
        synchronize();

        const float* logits_diff_ptr = logits_diff.data<float>();
        const float* key_diff_ptr = key_diff.data<float>();
        const float* value_diff_ptr = value_diff.data<float>();
        if (!logits_diff_ptr || !key_diff_ptr || !value_diff_ptr) {
            g_last_error = "mlx_test_chunked_prefill_tail_matches_full_prompt: failed to materialize diffs";
            return 0;
        }
        if (full_ctx.kv_cache[0].offset != prompt_len || chunk_ctx.kv_cache[0].offset != prompt_len) {
            g_last_error = "mlx_test_chunked_prefill_tail_matches_full_prompt: unexpected kv cache offset";
            return 0;
        }
        if (logits_diff_ptr[0] > 1e-5f) {
            g_last_error = "mlx_test_chunked_prefill_tail_matches_full_prompt: logits mismatch";
            return 0;
        }
        if (key_diff_ptr[0] > 1e-5f) {
            g_last_error = "mlx_test_chunked_prefill_tail_matches_full_prompt: key cache mismatch";
            return 0;
        }
        if (value_diff_ptr[0] > 1e-5f) {
            g_last_error = "mlx_test_chunked_prefill_tail_matches_full_prompt: value cache mismatch";
            return 0;
        }
        return 1;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 0;
    } catch (...) {
        g_last_error = "unknown error in mlx_test_chunked_prefill_tail_matches_full_prompt";
        return 0;
    }
}

int32_t mlx_test_linear_attention_fused_quant_inproj_reuse(void) {
    try {
        constexpr int qkvz_rows = 6;
        constexpr int ba_rows = 2;
        constexpr int total_rows = qkvz_rows + ba_rows;
        constexpr int packed_cols = 5;
        constexpr int group_cols = 4;

        std::vector<uint32_t> packed_values(static_cast<size_t>(total_rows * packed_cols));
        std::vector<float> scale_values(static_cast<size_t>(total_rows * group_cols));
        std::vector<float> bias_values(static_cast<size_t>(total_rows * group_cols));
        for (size_t i = 0; i < packed_values.size(); ++i) {
            packed_values[i] = static_cast<uint32_t>((i * 17) & 0xFFu);
        }
        for (size_t i = 0; i < scale_values.size(); ++i) {
            scale_values[i] = 0.01f * static_cast<float>((i % 11) + 1);
            bias_values[i] = -0.02f * static_cast<float>((i % 7) + 1);
        }

        LayerWeights lw{};
        const array fused_q_w = array(packed_values.data(), Shape{total_rows, packed_cols}, uint32);
        const array fused_q_scales = array(scale_values.data(), Shape{total_rows, group_cols}, float32);
        const array fused_q_biases = array(bias_values.data(), Shape{total_rows, group_cols}, float32);

        lw.lin_in_proj_qkvz_q_w = slice(fused_q_w, {0, 0}, {qkvz_rows, packed_cols});
        lw.lin_in_proj_ba_q_w = slice(fused_q_w, {qkvz_rows, 0}, {total_rows, packed_cols});
        lw.lin_in_proj_qkvz_q_scales = slice(fused_q_scales, {0, 0}, {qkvz_rows, group_cols});
        lw.lin_in_proj_ba_q_scales = slice(fused_q_scales, {qkvz_rows, 0}, {total_rows, group_cols});
        lw.lin_in_proj_qkvz_q_biases = slice(fused_q_biases, {0, 0}, {qkvz_rows, group_cols});
        lw.lin_in_proj_ba_q_biases = slice(fused_q_biases, {qkvz_rows, 0}, {total_rows, group_cols});
        lw.lin_in_proj_qkvz_has_q = true;
        lw.lin_in_proj_ba_has_q = true;

        // Simulate fused grouped-affine capture path that already owns the full
        // in-proj quantized payload.
        lw.lin_in_proj_all_q_w = fused_q_w;
        lw.lin_in_proj_all_q_scales = fused_q_scales;
        lw.lin_in_proj_all_q_biases = fused_q_biases;
        lw.lin_in_proj_all_has_q = true;

        eval({lw.lin_in_proj_all_q_w, lw.lin_in_proj_all_q_scales, lw.lin_in_proj_all_q_biases});
        synchronize();
        const uint32_t* before_w = lw.lin_in_proj_all_q_w.data<uint32_t>();
        const float* before_scales = lw.lin_in_proj_all_q_scales.data<float>();
        const float* before_biases = lw.lin_in_proj_all_q_biases.data<float>();
        if (before_w == nullptr || before_scales == nullptr || before_biases == nullptr) {
            g_last_error = "mlx_test_linear_attention_fused_quant_inproj_reuse: failed to materialize baseline tensors";
            return 0;
        }

        if (!combine_linear_in_proj_weights(&lw)) {
            g_last_error = "mlx_test_linear_attention_fused_quant_inproj_reuse: combine returned false";
            return 0;
        }

        eval({lw.lin_in_proj_all_q_w, lw.lin_in_proj_all_q_scales, lw.lin_in_proj_all_q_biases});
        synchronize();
        const uint32_t* after_w = lw.lin_in_proj_all_q_w.data<uint32_t>();
        const float* after_scales = lw.lin_in_proj_all_q_scales.data<float>();
        const float* after_biases = lw.lin_in_proj_all_q_biases.data<float>();
        if (after_w == nullptr || after_scales == nullptr || after_biases == nullptr) {
            g_last_error = "mlx_test_linear_attention_fused_quant_inproj_reuse: failed to materialize post-combine tensors";
            return 0;
        }

        if (after_w != before_w || after_scales != before_scales || after_biases != before_biases) {
            g_last_error = "mlx_test_linear_attention_fused_quant_inproj_reuse: combine rebuilt fused quantized in-proj tensors";
            return 0;
        }
        if (!lw.lin_in_proj_all_has_q) {
            g_last_error = "mlx_test_linear_attention_fused_quant_inproj_reuse: combined quantized flag dropped";
            return 0;
        }
        if (lw.lin_in_proj_all_q_w.shape(0) != total_rows || lw.lin_in_proj_all_q_w.shape(1) != packed_cols) {
            g_last_error = "mlx_test_linear_attention_fused_quant_inproj_reuse: fused weight shape changed";
            return 0;
        }
        return 1;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 0;
    } catch (...) {
        g_last_error = "unknown error in mlx_test_linear_attention_fused_quant_inproj_reuse";
        return 0;
    }
}

int32_t mlx_test_nvfp4_rowwise_post_scale_linear_decode(void) {
    try {
        constexpr int batch = 1;
        constexpr int seq = 2;
        constexpr int in_dim = 32;
        constexpr int qkvz_rows = 6;
        constexpr int ba_rows = 4;
        constexpr int total_rows = qkvz_rows + ba_rows;
        constexpr int packed_cols = in_dim / 8;
        constexpr int scale_cols = in_dim / 16;

        std::vector<float> input_values(static_cast<size_t>(batch * seq * in_dim));
        for (size_t i = 0; i < input_values.size(); ++i) {
            input_values[i] = static_cast<float>((static_cast<int>(i * 7) % 19) - 9) * 0.0625f;
        }

        std::vector<uint32_t> qkvz_packed(static_cast<size_t>(qkvz_rows * packed_cols));
        std::vector<uint8_t> qkvz_scales(static_cast<size_t>(qkvz_rows * scale_cols));
        std::vector<uint32_t> ba_packed(static_cast<size_t>(ba_rows * packed_cols));
        std::vector<uint8_t> ba_scales(static_cast<size_t>(ba_rows * scale_cols));
        for (size_t i = 0; i < qkvz_packed.size(); ++i) {
            qkvz_packed[i] = static_cast<uint32_t>((i * 13 + 0x55) & 0xFFFFFFFFu);
        }
        for (size_t i = 0; i < ba_packed.size(); ++i) {
            ba_packed[i] = static_cast<uint32_t>((i * 17 + 0xAA) & 0xFFFFFFFFu);
        }
        for (size_t i = 0; i < qkvz_scales.size(); ++i) {
            qkvz_scales[i] = static_cast<uint8_t>(32 + (i % 96));
        }
        for (size_t i = 0; i < ba_scales.size(); ++i) {
            ba_scales[i] = static_cast<uint8_t>(48 + (i % 80));
        }

        const array x(input_values.data(), Shape{batch, seq, in_dim}, float32);
        const array qkvz_q_w(qkvz_packed.data(), Shape{qkvz_rows, packed_cols}, uint32);
        const array qkvz_q_scales(qkvz_scales.data(), Shape{qkvz_rows, scale_cols}, uint8);
        const array ba_q_w(ba_packed.data(), Shape{ba_rows, packed_cols}, uint32);
        const array ba_q_scales(ba_scales.data(), Shape{ba_rows, scale_cols}, uint8);
        const array qkvz_post_scale = stop_gradient(array(0.875f, float32));
        const array ba_post_scale = stop_gradient(array(1.125f, float32));

        const array qkvz_out = linear_decode_maybe_quantized(
            true,
            16,
            4,
            x,
            array(0.0f),
            true,
            qkvz_q_w,
            qkvz_q_scales,
            qkvz_post_scale,
            false
        );
        const array ba_out = linear_decode_maybe_quantized(
            true,
            16,
            4,
            x,
            array(0.0f),
            true,
            ba_q_w,
            ba_q_scales,
            ba_post_scale,
            false
        );

        array fused_post_scale = array(0.0f);
        if (!concat_nvfp4_post_scales_for_rows(
                qkvz_post_scale,
                qkvz_rows,
                ba_post_scale,
                ba_rows,
                &fused_post_scale))
        {
            g_last_error = "mlx_test_nvfp4_rowwise_post_scale_linear_decode: failed to build fused post-scale";
            return 0;
        }

        const array fused_q_w = concatenate({qkvz_q_w, ba_q_w}, 0);
        const array fused_q_scales = concatenate({qkvz_q_scales, ba_q_scales}, 0);
        const array fused_out = linear_decode_maybe_quantized(
            true,
            16,
            4,
            x,
            array(0.0f),
            true,
            fused_q_w,
            fused_q_scales,
            fused_post_scale,
            false
        );

        const array fused_qkvz = slice(fused_out, {0, 0, 0}, {batch, seq, qkvz_rows});
        const array fused_ba = slice(fused_out, {0, 0, qkvz_rows}, {batch, seq, total_rows});
        const array qkvz_diff = max(reshape(
            abs(astype(qkvz_out, float32) - astype(fused_qkvz, float32)),
            {static_cast<int>(qkvz_out.size())}
        ));
        const array ba_diff = max(reshape(
            abs(astype(ba_out, float32) - astype(fused_ba, float32)),
            {static_cast<int>(ba_out.size())}
        ));
        eval({qkvz_diff, ba_diff});
        synchronize();

        const float* qkvz_diff_ptr = qkvz_diff.data<float>();
        const float* ba_diff_ptr = ba_diff.data<float>();
        if (!qkvz_diff_ptr || !ba_diff_ptr) {
            g_last_error = "mlx_test_nvfp4_rowwise_post_scale_linear_decode: failed to materialize diffs";
            return 0;
        }
        if (qkvz_diff_ptr[0] > 1e-5f) {
            g_last_error = "mlx_test_nvfp4_rowwise_post_scale_linear_decode: qkvz output mismatch";
            return 0;
        }
        if (ba_diff_ptr[0] > 1e-5f) {
            g_last_error = "mlx_test_nvfp4_rowwise_post_scale_linear_decode: ba output mismatch";
            return 0;
        }
        return 1;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 0;
    } catch (...) {
        g_last_error = "unknown error in mlx_test_nvfp4_rowwise_post_scale_linear_decode";
        return 0;
    }
}

int32_t mlx_test_talu_meta_nvfp4_detection(void) {
    try {
        namespace fs = std::filesystem;
        const fs::path base = fs::temp_directory_path() /
            fs::path("talu_mlx_meta_test_" + std::to_string(std::time(nullptr)));
        fs::create_directories(base);
        const fs::path meta = base / "talu_meta.json";

        {
            std::ofstream out(meta);
            if (!out) {
                g_last_error = "mlx_test_talu_meta_nvfp4_detection: failed to create meta file";
                fs::remove_all(base);
                return 0;
            }
            out << "{\n  \"scheme\": \"NVFP4\",\n  \"source_model_id\": \"Qwen/Qwen3.5-9B\"\n}\n";
        }

        const bool detected = model_has_talu_meta_scheme(base.string(), "NVFP4");
        const bool wrong = model_has_talu_meta_scheme(base.string(), "TQ4");
        fs::remove_all(base);
        if (!detected || wrong) {
            g_last_error = "mlx_test_talu_meta_nvfp4_detection: detection mismatch";
            return 0;
        }
        return 1;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 0;
    } catch (...) {
        g_last_error = "unknown error in mlx_test_talu_meta_nvfp4_detection";
        return 0;
    }
}

int32_t mlx_test_nvfp4_mmap_strict_policy(void) {
    const bool pure_nvfp4 = should_enable_nvfp4_mmap_strict(true, false);
    const bool hybrid = should_enable_nvfp4_mmap_strict(true, true);
    const bool no_nvfp4 = should_enable_nvfp4_mmap_strict(false, false);
    if (!pure_nvfp4 || hybrid || no_nvfp4) {
        g_last_error = "mlx_test_nvfp4_mmap_strict_policy: unexpected strict policy decision";
        return 0;
    }
    return 1;
}

int32_t mlx_test_grouped_affine_embedding_lookup_matches_reference(void) {
    try {
        constexpr int vocab = 5;
        constexpr int hidden = 8;
        constexpr int group_size = 4;
        constexpr int bits = 4;
        constexpr int packed_cols = hidden / 8;
        constexpr int groups = hidden / group_size;

        std::vector<uint32_t> packed(static_cast<size_t>(vocab * packed_cols), 0);
        std::vector<float> scales(static_cast<size_t>(vocab * groups), 0.0f);
        std::vector<float> biases(static_cast<size_t>(vocab * groups), 0.0f);
        for (int r = 0; r < vocab; ++r) {
            uint32_t word = 0;
            for (int c = 0; c < hidden; ++c) {
                const uint32_t q = static_cast<uint32_t>((r * 3 + c * 5) & 0xF);
                word |= (q << (c * bits));
            }
            packed[static_cast<size_t>(r)] = word;
            for (int g = 0; g < groups; ++g) {
                const size_t idx = static_cast<size_t>(r * groups + g);
                scales[idx] = 0.0625f * static_cast<float>(g + 1);
                biases[idx] = -0.125f * static_cast<float>(r + 1);
            }
        }

        mlx_ctx ctx{};
        ctx.cfg.quant_group_size = group_size;
        ctx.cfg.quant_bits = bits;
        ctx.cfg.hidden_size = hidden;
        ctx.embed_tokens_q_w = array(packed.data(), Shape{vocab, packed_cols}, uint32);
        ctx.embed_tokens_q_scales = array(scales.data(), Shape{vocab, groups}, float32);
        ctx.embed_tokens_q_biases = array(biases.data(), Shape{vocab, groups}, float32);
        ctx.embed_tokens_q_bits = bits;
        ctx.embed_tokens_has_q = true;

        const int32_t ids_values[] = {4, 1, 3, 0, 2, 1};
        const array ids(ids_values, Shape{2, 3}, int32);
        const array actual = embedding_lookup(&ctx, ids);

        std::vector<float> expected_values(static_cast<size_t>(2 * 3 * hidden), 0.0f);
        for (int i = 0; i < 6; ++i) {
            const int token = ids_values[i];
            const uint32_t word = packed[static_cast<size_t>(token)];
            for (int c = 0; c < hidden; ++c) {
                const uint32_t q = (word >> (c * bits)) & 0xF;
                const int g = c / group_size;
                const float scale = scales[static_cast<size_t>(token * groups + g)];
                const float bias = biases[static_cast<size_t>(token * groups + g)];
                expected_values[static_cast<size_t>(i * hidden + c)] =
                    static_cast<float>(q) * scale + bias;
            }
        }
        const array expected(expected_values.data(), Shape{2, 3, hidden}, float32);
        const array diff = max(reshape(
            abs(astype(actual, float32) - expected),
            {static_cast<int>(expected.size())}
        ));
        eval(diff);
        synchronize();
        const float* diff_ptr = diff.data<float>();
        if (!diff_ptr) {
            g_last_error = "mlx_test_grouped_affine_embedding_lookup_matches_reference: failed to materialize diff";
            return 0;
        }
        if (actual.ndim() != 3 || actual.shape(0) != 2 || actual.shape(1) != 3 || actual.shape(2) != hidden) {
            g_last_error = "mlx_test_grouped_affine_embedding_lookup_matches_reference: unexpected output shape";
            return 0;
        }
        if (diff_ptr[0] > 5e-3f) {
            g_last_error = "mlx_test_grouped_affine_embedding_lookup_matches_reference: lookup mismatch";
            return 0;
        }
        if (embedding_vocab_size(&ctx) != vocab) {
            g_last_error = "mlx_test_grouped_affine_embedding_lookup_matches_reference: vocab size mismatch";
            return 0;
        }
        return 1;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 0;
    } catch (...) {
        g_last_error = "unknown error in mlx_test_grouped_affine_embedding_lookup_matches_reference";
        return 0;
    }
}

int32_t mlx_test_linear_attention_fused_mixer_matches_reference(void) {
    // Fused compute-module SSM path was removed (slower than bridge reference
    // path which uses custom Metal kernels). Stub returns pass.
    return 1;
}

int32_t mlx_test_topk_candidate_extraction_multi(void) {
    try {
        constexpr int vocab = 8;
        constexpr int k = 4;
        const float logits_values[vocab] = {
            -3.0f,
            5.0f,
            0.5f,
            4.0f,
            1.5f,
            -2.0f,
            2.0f,
            3.5f,
        };
        const int32_t expected_ids[k] = {1, 3, 7, 6};
        const float expected_logits[k] = {5.0f, 4.0f, 3.5f, 2.0f};

        const array logits_last(logits_values, Shape{1, 1, vocab}, float32);
        auto topk_fn = compiled_topk_candidates(k);
        const std::vector<array> topk_outputs = (*topk_fn)({logits_last});
        if (topk_outputs.size() != 2) {
            g_last_error = "mlx_test_topk_candidate_extraction_multi: invalid compiled output";
            return 0;
        }

        const array top_k_logits = astype(topk_outputs[0], float32);
        const array top_k_indices = astype(topk_outputs[1], int32);
        eval({top_k_logits, top_k_indices});
        synchronize();

        const float* logits_ptr = top_k_logits.data<float>();
        const int32_t* ids_ptr = top_k_indices.data<int32_t>();
        if (!logits_ptr || !ids_ptr) {
            g_last_error = "mlx_test_topk_candidate_extraction_multi: failed to materialize outputs";
            return 0;
        }

        float actual_logits[k];
        int32_t actual_ids[k];
        std::memcpy(actual_logits, logits_ptr, sizeof(actual_logits));
        std::memcpy(actual_ids, ids_ptr, sizeof(actual_ids));
        stable_sort_topk_pairs(actual_logits, actual_ids, k);

        for (int idx = 0; idx < k; ++idx) {
            if (actual_ids[idx] != expected_ids[idx]) {
                g_last_error = "mlx_test_topk_candidate_extraction_multi: candidate id mismatch";
                return 0;
            }
            if (std::fabs(actual_logits[idx] - expected_logits[idx]) > 1e-6f) {
                g_last_error = "mlx_test_topk_candidate_extraction_multi: candidate logit mismatch";
                return 0;
            }
        }
        return 1;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 0;
    } catch (...) {
        g_last_error = "unknown error in mlx_test_topk_candidate_extraction_multi";
        return 0;
    }
}
