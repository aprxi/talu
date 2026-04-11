// bridge_api_run.inc — Streaming decode loop and consolidated run:
// mlx_decode_topk_stream, mlx_run.

int32_t mlx_decode_topk_stream(
    mlx_ctx* ctx,
    int32_t first_token,
    int32_t decode_tokens,
    const int32_t* eos_ids,
    int32_t eos_len,
    float temperature,
    int32_t top_k,
    float top_p,
    float min_p,
    float repetition_penalty,
    float presence_penalty,
    float frequency_penalty,
    int32_t** out_generated_ids,
    int32_t* out_generated_len
) {
    if (!ctx || !out_generated_ids || !out_generated_len) {
        g_last_error = "mlx_decode_topk_stream: invalid arguments";
        return 0;
    }
    if (!ctx->stream_ready) {
        g_last_error = "mlx_decode_topk_stream: prefill not ready";
        return 0;
    }
    if (decode_tokens < 0) {
        g_last_error = "mlx_decode_topk_stream: decode_tokens must be >= 0";
        return 0;
    }
    if (first_token < 0) {
        g_last_error = "mlx_decode_topk_stream: first_token must be >= 0";
        return 0;
    }
    if (top_k <= 0) {
        g_last_error = "mlx_decode_topk_stream: top_k must be > 0";
        return 0;
    }
    if (temperature < 0.0f) {
        g_last_error = "mlx_decode_topk_stream: temperature must be >= 0";
        return 0;
    }
    if (top_p < 0.0f || top_p > 1.0f) {
        g_last_error = "mlx_decode_topk_stream: top_p must be in [0,1]";
        return 0;
    }
    if (min_p < 0.0f || min_p > 1.0f) {
        g_last_error = "mlx_decode_topk_stream: min_p must be in [0,1]";
        return 0;
    }
    if (repetition_penalty <= 0.0f) {
        g_last_error = "mlx_decode_topk_stream: repetition_penalty must be > 0";
        return 0;
    }

    try {
        *out_generated_ids = nullptr;
        *out_generated_len = 0;
        if (decode_tokens == 0) return 1;
        const bool fused_lm_head_profile = env_truthy("TALU_METAL_FUSED_LM_HEAD_PROFILE", false);
        const size_t fused_hits_start = g_fused_lm_head_argmax_hits;
        const bool topk_stream_profile = []() {
            if (const char* raw = std::getenv("TALU_METAL_TOPK_STREAM_PROFILE")) {
                return std::string(raw) == "1";
            }
            return false;
        }();
        uint64_t profile_forward_ns = 0;
        uint64_t profile_penalty_ns = 0;
        uint64_t profile_sample_ns = 0;
        uint64_t profile_materialize_ns = 0;
        uint64_t profile_total_ns = 0;

        const StreamSamplingConfig sampling_cfg{
            .temperature = temperature,
            .top_k = top_k,
            .top_p = top_p,
            .min_p = min_p,
        };
        const bool has_penalties =
            repetition_penalty != 1.0f || presence_penalty != 0.0f || frequency_penalty != 0.0f;
        const bool use_fused_greedy_decode =
            !has_penalties &&
            (sampling_cfg.temperature <= 0.0f || sampling_cfg.top_k <= 1) &&
            can_use_fused_lm_head_argmax(ctx);
        const bool use_fused_topk_decode =
            !has_penalties &&
            !use_fused_greedy_decode &&
            can_use_fused_lm_head_topk(ctx, sampling_cfg.top_k);
        const size_t output_capacity = static_cast<size_t>(decode_tokens);
        std::unique_ptr<int32_t, decltype(&std::free)> generated_host(
            static_cast<int32_t*>(std::malloc(sizeof(int32_t) * output_capacity)),
            &std::free
        );
        if (!generated_host) {
            g_last_error = "mlx_decode_topk_stream: failed to allocate token buffer";
            return 0;
        }
        int32_t produced = 0;

        // Build compiled decode core if the model qualifies.
        const bool use_compiled_decode =
            !env_truthy("TALU_METAL_COMPILED_DECODE_DISABLE", false) &&
            can_use_compiled_decode(ctx);
        std::optional<CompiledDecodeFn> compiled_core;
        if (use_compiled_decode) {
            compiled_core = build_compiled_decode_core(ctx);
        }

        if (!has_penalties) {
            // Fast path: chunk decode to avoid very large token graphs and avoid
            // per-token async scheduling overhead.
            const int32_t topk_chunk_size = []() -> int32_t {
                if (const char* raw = std::getenv("TALU_METAL_TOPK_CHUNK")) {
                    char* end = nullptr;
                    const long parsed = std::strtol(raw, &end, 10);
                    if (end != raw && parsed > 0 && parsed <= std::numeric_limits<int32_t>::max()) {
                        return static_cast<int32_t>(parsed);
                    }
                }
                return 32;
            }();
            const int32_t effective_topk_chunk_size = std::min<int32_t>(topk_chunk_size, 128);
            const int32_t token_scalar = first_token;
            array current_token = array(&token_scalar, Shape{1, 1}, int32);
            std::vector<array> chunk_tokens;
            chunk_tokens.reserve(static_cast<size_t>(effective_topk_chunk_size));
            bool saw_eos = false;
            // Pre-compile sampling when not using fused decode paths.
            const auto compiled_sample = (!use_fused_greedy_decode && !use_fused_topk_decode)
                ? compiled_sampling_step(sampling_cfg)
                : nullptr;
            const auto total_t0 = std::chrono::steady_clock::now();
            while (produced < decode_tokens && !saw_eos) {
                const int32_t remaining = decode_tokens - produced;
                const int32_t chunk_target = std::min(effective_topk_chunk_size, remaining);
                chunk_tokens.clear();

                const auto t_build0 = std::chrono::steady_clock::now();
                for (int32_t i = 0; i < chunk_target; ++i) {
                    array next_token = array(0, int32);
                    if (use_fused_greedy_decode) {
                        next_token = fused_lm_head_argmax_token(ctx, current_token);
                    } else if (use_fused_topk_decode) {
                        next_token = sample_token_from_topk_candidates(
                            fused_lm_head_topk_candidates(ctx, current_token, sampling_cfg.top_k),
                            sampling_cfg);
                    } else if (compiled_core.has_value()) {
                        // --- Compiled decode path ---
                        // 1. Embed token.
                        array hidden = embedding_lookup(ctx, current_token);
                        if (std::fabs(ctx->cfg.embedding_multiplier - 1.0f) > 1.0e-6f) {
                            hidden = hidden * array(ctx->cfg.embedding_multiplier, hidden.dtype());
                        }
                        // 2. Build attention mask.
                        const int num_layers = static_cast<int>(ctx->layers.size());
                        const int capacity = ctx->kv_cache[0].keys->shape(2);
                        const int32_t cur_offset = ctx->kv_cache[0].offset;
                        const array offset_arr = array(&cur_offset, {1}, int32);
                        const array positions = astype(arange(0, capacity + 1, 1, int32), float32);
                        const array offset_f = array(static_cast<float>(cur_offset), float32);
                        const array cap_f = array(static_cast<float>(capacity), float32);
                        const array attn_mask = reshape(
                            where((positions < offset_f) | (positions == cap_f),
                                  array(0.0f),
                                  array(-std::numeric_limits<float>::infinity())),
                            {1, 1, 1, capacity + 1});
                        // 3. Collect inputs: [hidden, offset, mask, kv_k_0, kv_v_0, ...]
                        std::vector<array> core_inputs;
                        core_inputs.reserve(3 + static_cast<size_t>(num_layers) * 2);
                        core_inputs.push_back(hidden);
                        core_inputs.push_back(offset_arr);
                        core_inputs.push_back(attn_mask);
                        for (int li = 0; li < num_layers; ++li) {
                            core_inputs.push_back(*ctx->kv_cache[static_cast<size_t>(li)].keys);
                            core_inputs.push_back(*ctx->kv_cache[static_cast<size_t>(li)].values);
                        }
                        // 4. Call compiled core.
                        auto core_out = (*compiled_core)(core_inputs);
                        const array normed_hidden = core_out[0];
                        // 5. Commit new K/V to caches (host-side slice_update).
                        for (int li = 0; li < num_layers; ++li) {
                            const array& new_k = core_out[static_cast<size_t>(1 + li * 2)];
                            const array& new_v = core_out[static_cast<size_t>(1 + li * 2 + 1)];
                            KVCacheState& kv = ctx->kv_cache[static_cast<size_t>(li)];
                            const int prev = kv.offset;
                            const int nh = ctx->layers[static_cast<size_t>(li)].attn_num_kv_heads;
                            const int hd = ctx->layers[static_cast<size_t>(li)].attn_head_dim;
                            kv.keys = slice_update(*kv.keys, new_k,
                                {0, 0, prev, 0}, {1, nh, prev + 1, hd});
                            kv.values = slice_update(*kv.values, new_v,
                                {0, 0, prev, 0}, {1, nh, prev + 1, hd});
                            kv.offset = prev + 1;
                        }
                        // 6. LM head.
                        const array logits = forward_logits_from_hidden(ctx, normed_hidden);
                        next_token = (*compiled_sample)({last_token_logits(logits)}).at(0);
                    } else {
                        next_token = (*compiled_sample)({last_token_logits(forward_logits(ctx, current_token))}).at(0);
                    }
                    async_eval(next_token);
                    chunk_tokens.push_back(next_token);
                    current_token = next_token;
                }
                const auto t_build1 = std::chrono::steady_clock::now();

                const auto t_materialize0 = std::chrono::steady_clock::now();
                eval(chunk_tokens);
                const auto t_materialize1 = std::chrono::steady_clock::now();

                for (int32_t i = 0; i < chunk_target; ++i) {
                    const int32_t* token_ptr = chunk_tokens[static_cast<size_t>(i)].data<int32_t>();
                    if (!token_ptr) {
                        g_last_error = "mlx_decode_topk_stream: failed to materialize sampled token";
                        return 0;
                    }
                    const int32_t tok = token_ptr[0];
                    generated_host.get()[static_cast<size_t>(produced)] = tok;
                    produced += 1;
                    if (token_is_eos(tok, eos_ids, eos_len)) {
                        saw_eos = true;
                        break;
                    }
                }

                if (topk_stream_profile) {
                    profile_forward_ns += static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(t_build1 - t_build0).count()
                    );
                    profile_materialize_ns += static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(t_materialize1 - t_materialize0).count()
                    );
                }
            }
            if (topk_stream_profile) {
                const auto total_t1 = std::chrono::steady_clock::now();
                profile_total_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(total_t1 - total_t0).count()
                );
            }
        } else {
            // Penalized path.
            ctx->sampling_context_counts.clear();
            ctx->sampling_context_len = 0;

            const int vocab = embedding_vocab_size(ctx);
            if (vocab <= 0) {
                g_last_error = "mlx_decode_topk_stream: invalid vocab for penalties";
                return 0;
            }

            const bool presence_only_penalty =
                repetition_penalty == 1.0f &&
                frequency_penalty == 0.0f &&
                presence_penalty != 0.0f;

            const int32_t penalty_chunk_size = []() -> int32_t {
                if (const char* raw = std::getenv("TALU_METAL_TOPK_PENALTY_CHUNK")) {
                    char* end = nullptr;
                    const long parsed = std::strtol(raw, &end, 10);
                    if (end != raw && parsed > 0 && parsed <= std::numeric_limits<int32_t>::max()) {
                        return static_cast<int32_t>(parsed);
                    }
                }
                return 16;
            }();
            const int32_t effective_penalty_chunk_size = std::min<int32_t>(penalty_chunk_size, 64);

            if (presence_only_penalty) {
                // Chunked GPU path for the common presence-only configuration.
                // Keeps penalty state on-device and synchronizes once per chunk.
                const float penalty_value = presence_penalty;
                const array penalty_arr(&penalty_value, Shape{1, 1, 1}, float32);
                array presence_mask = zeros(Shape{1, 1, vocab}, float32);

                const int32_t first_token_scalar = first_token;
                array current_token = array(&first_token_scalar, Shape{1, 1}, int32);
                const array first_token_idx(&first_token_scalar, Shape{1, 1, 1}, int32);
                presence_mask = put_along_axis(presence_mask, first_token_idx, penalty_arr, -1);
                ctx->sampling_context_len = 1;
                std::vector<array> chunk_tokens;
                chunk_tokens.reserve(static_cast<size_t>(effective_penalty_chunk_size));

                const auto total_t0 = std::chrono::steady_clock::now();
                bool saw_eos = false;
                while (produced < decode_tokens && !saw_eos) {
                    const int32_t remaining = decode_tokens - produced;
                    const int32_t chunk_target = std::min(effective_penalty_chunk_size, remaining);
                    chunk_tokens.clear();

                    const auto t_build0 = std::chrono::steady_clock::now();
                    for (int32_t i = 0; i < chunk_target; ++i) {
                        const array logits_last = last_token_logits(forward_logits(ctx, current_token));
                        const array sampling_logits = logits_last - presence_mask;
                        const array next_token = sample_token_from_logits(sampling_logits, sampling_cfg);
                        chunk_tokens.push_back(next_token);

                        const array token_idx = reshape(astype(next_token, int32), {1, 1, 1});
                        presence_mask = put_along_axis(presence_mask, token_idx, penalty_arr, -1);
                        current_token = next_token;
                    }
                    const auto t_build1 = std::chrono::steady_clock::now();

                    const auto t_materialize0 = std::chrono::steady_clock::now();
                    const array chunk_rows = concatenate(chunk_tokens, 0); // [chunk,1]
                    const array chunk_flat = reshape(chunk_rows, {chunk_target}); // [chunk]
                    eval(chunk_flat);
                    const auto t_materialize1 = std::chrono::steady_clock::now();

                    const int32_t* chunk_ptr = chunk_flat.data<int32_t>();
                    for (int32_t i = 0; i < chunk_target; ++i) {
                        const int32_t tok = chunk_ptr[static_cast<size_t>(i)];
                        generated_host.get()[static_cast<size_t>(produced)] = tok;
                        produced += 1;
                        ctx->sampling_context_len += 1;
                        if (token_is_eos(tok, eos_ids, eos_len)) {
                            saw_eos = true;
                            break;
                        }
                    }

                    if (topk_stream_profile) {
                        profile_forward_ns += static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(t_build1 - t_build0).count()
                        );
                        profile_materialize_ns += static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(t_materialize1 - t_materialize0).count()
                        );
                    }
                }

                if (topk_stream_profile) {
                    const auto total_t1 = std::chrono::steady_clock::now();
                    profile_total_ns = static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(total_t1 - total_t0).count()
                    );
                }
            } else {
                // General penalized path: keep token counts on-device and decode
                // in bounded chunks to avoid host sync per token.
                const array zero_counts = zeros(Shape{1, 1, vocab}, float32);
                array token_counts = zero_counts;
                const array one_count = array(1.0f, float32);
                auto penalized_sample_fn = compiled_penalized_sampling_step(
                    sampling_cfg,
                    repetition_penalty,
                    presence_penalty,
                    frequency_penalty
                );

                const int32_t first_token_scalar = first_token;
                array current_token = array(&first_token_scalar, Shape{1, 1}, int32);
                const array first_token_idx(&first_token_scalar, Shape{1, 1, 1}, int32);
                token_counts = put_along_axis(token_counts, first_token_idx, one_count, -1);
                ctx->sampling_context_len = 1;
                std::vector<array> chunk_tokens;
                chunk_tokens.reserve(static_cast<size_t>(effective_penalty_chunk_size));

                const auto total_t0 = std::chrono::steady_clock::now();
                bool saw_eos = false;
                while (produced < decode_tokens && !saw_eos) {
                    const int32_t remaining = decode_tokens - produced;
                    const int32_t chunk_target = std::min(effective_penalty_chunk_size, remaining);
                    chunk_tokens.clear();

                    const auto t_build0 = std::chrono::steady_clock::now();
                    for (int32_t i = 0; i < chunk_target; ++i) {
                        const array logits_last = last_token_logits(forward_logits(ctx, current_token));
                        const std::vector<array> sampled_outputs = (*penalized_sample_fn)({logits_last, token_counts});
                        if (sampled_outputs.size() != 2) {
                            throw std::runtime_error("mlx_decode_topk_stream: invalid penalized sample output");
                        }
                        const array next_token = sampled_outputs[0];
                        token_counts = sampled_outputs[1];
                        chunk_tokens.push_back(next_token);
                        current_token = next_token;
                    }
                    const auto t_build1 = std::chrono::steady_clock::now();

                    const auto t_materialize0 = std::chrono::steady_clock::now();
                    const array chunk_rows = concatenate(chunk_tokens, 0); // [chunk,1]
                    const array chunk_flat = reshape(chunk_rows, {chunk_target}); // [chunk]
                    eval(chunk_flat);
                    const auto t_materialize1 = std::chrono::steady_clock::now();

                    const int32_t* chunk_ptr = chunk_flat.data<int32_t>();
                    for (int32_t i = 0; i < chunk_target; ++i) {
                        const int32_t tok = chunk_ptr[static_cast<size_t>(i)];
                        if (tok < 0 || tok >= vocab) {
                            throw std::runtime_error("mlx_decode_topk_stream: sampled token out of range");
                        }
                        generated_host.get()[static_cast<size_t>(produced)] = tok;
                        produced += 1;
                        ctx->sampling_context_len += 1;
                        if (token_is_eos(tok, eos_ids, eos_len)) {
                            saw_eos = true;
                            break;
                        }
                    }

                    if (topk_stream_profile) {
                        profile_forward_ns += static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(t_build1 - t_build0).count()
                        );
                        profile_materialize_ns += static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(t_materialize1 - t_materialize0).count()
                        );
                    }
                }

                if (topk_stream_profile) {
                    const auto total_t1 = std::chrono::steady_clock::now();
                    profile_total_ns = static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(total_t1 - total_t0).count()
                    );
                }
            }
        }

        if (topk_stream_profile) {
            std::fprintf(
                stderr,
                "[mlx][topk_stream_profile] tokens=%zu total_ms=%.3f forward_ms=%.3f penalty_ms=%.3f sample_ms=%.3f materialize_ms=%.3f\n",
                static_cast<size_t>(produced),
                static_cast<double>(profile_total_ns) / 1e6,
                static_cast<double>(profile_forward_ns) / 1e6,
                static_cast<double>(profile_penalty_ns) / 1e6,
                static_cast<double>(profile_sample_ns) / 1e6,
                static_cast<double>(profile_materialize_ns) / 1e6
            );
        }
        if (fused_lm_head_profile) {
            const size_t fused_hits = g_fused_lm_head_argmax_hits - fused_hits_start;
            std::fprintf(
                stderr,
                "[mlx][fused_lm_head_argmax] used=%s hits=%zu mode=topk_stream top_k=%d temperature=%.4f\n",
                (use_fused_greedy_decode || use_fused_topk_decode) ? "1" : "0",
                fused_hits,
                top_k,
                static_cast<double>(temperature)
            );
        }
        *out_generated_ids = generated_host.release();
        *out_generated_len = produced;
        return 1;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 0;
    } catch (...) {
        g_last_error = "unknown error in mlx_decode_topk_stream";
        return 0;
    }
}

int32_t mlx_run(
    mlx_ctx* ctx,
    const int32_t* prompt_ids,
    int32_t prompt_len,
    int32_t decode_tokens,
    int32_t warmup,
    mlx_result* out_result,
    int32_t capture_generated_tokens,
    int32_t** out_generated_ids,
    int32_t* out_generated_len
) {
    if (!ctx || !out_result || !out_generated_ids || !out_generated_len) {
        g_last_error = "mlx_run: null ctx/out_result/out_generated_ids/out_generated_len";
        return 0;
    }
    if (!prompt_ids || prompt_len <= 0) {
        g_last_error = "mlx_run: prompt_ids is null or prompt_len <= 0";
        return 0;
    }
    if (decode_tokens <= 0 || warmup < 0) {
        g_last_error = "mlx_run: invalid token/warmup arguments";
        return 0;
    }

    try {
        *out_generated_ids = nullptr;
        *out_generated_len = 0;
        const bool capture_tokens = capture_generated_tokens != 0;
        const bool fused_lm_head_profile = env_truthy("TALU_METAL_FUSED_LM_HEAD_PROFILE", false);
        const size_t fused_hits_start = g_fused_lm_head_argmax_hits;

        std::vector<int32_t> prompt_vec(prompt_ids, prompt_ids + prompt_len);
        std::vector<int32_t> captured_tokens;
        if (capture_tokens) {
            captured_tokens.resize(static_cast<size_t>(decode_tokens));
        }

        mlx_result best{};

        for (int run = 0; run < warmup + 1; ++run) {
            const bool is_warmup = run < warmup;
            reset_runtime_state(ctx);
            ctx->kv_reserve_tokens = resolve_kv_reserve_target_tokens(ctx, prompt_len, decode_tokens);

            auto t0 = std::chrono::steady_clock::now();
            array token_ids = array(0.0f);
            if (prompt_len > 1) {
                prefill_prefix_chunks(ctx, prompt_vec, prompt_len - 1);

                const int32_t last_token_scalar = prompt_vec[static_cast<size_t>(prompt_len - 1)];
                const array last_token_arr(&last_token_scalar, Shape{1, 1}, int32);
                token_ids = can_use_fused_lm_head_argmax(ctx)
                    ? fused_lm_head_argmax_token(ctx, last_token_arr)
                    : next_token_greedy(last_token_logits(forward_logits(ctx, last_token_arr)));
            } else {
                const array prompt_arr(prompt_vec.begin(), Shape{1, prompt_len}, int32);
                const array seed_logits = forward_logits(ctx, prompt_arr);
                token_ids = next_token_greedy(last_token_logits(seed_logits));
            }
            eval(token_ids);
            synchronize();
            ctx->stream_ready = true;
            auto t1 = std::chrono::steady_clock::now();

            auto t2 = std::chrono::steady_clock::now();
            const int32_t seeded_token = token_ids.item<int32_t>();
            int32_t* generated_ids = nullptr;
            int32_t generated_len = 0;
            const int32_t status = mlx_decode_topk_stream(
                ctx,
                seeded_token,
                decode_tokens,
                nullptr,
                0,
                0.0f,
                1,
                1.0f,
                0.0f,
                1.0f,
                0.0f,
                0.0f,
                &generated_ids,
                &generated_len
            );
            auto t3 = std::chrono::steady_clock::now();
            std::unique_ptr<int32_t, decltype(&std::free)> generated_guard(generated_ids, &std::free);
            if (status == 0 || generated_len < 0) {
                g_last_error = "mlx_run: decode_topk_stream failed";
                return 0;
            }
            if (capture_tokens && !is_warmup) {
                if (generated_len != decode_tokens) {
                    g_last_error = "mlx_run: generated token count mismatch";
                    return 0;
                }
                std::memcpy(
                    captured_tokens.data(),
                    generated_guard.get(),
                    sizeof(int32_t) * static_cast<size_t>(decode_tokens)
                );
            }

            const double prefill_s = std::chrono::duration<double>(t1 - t0).count();
            const double decode_s = std::chrono::duration<double>(t3 - t2).count();

            mlx_result current{};
            current.prompt_tokens = prompt_len;
            current.decode_tokens = decode_tokens;
            current.prefill_ms = prefill_s * 1000.0;
            current.prefill_tps = prefill_s > 0 ? (static_cast<double>(prompt_len) / prefill_s) : 0.0;
            current.decode_tps = decode_s > 0 ? (static_cast<double>(decode_tokens) / decode_s) : 0.0;

            if (!is_warmup) best = current;
        }

        if (capture_tokens && !captured_tokens.empty()) {
            const size_t bytes = sizeof(int32_t) * captured_tokens.size();
            int32_t* copied = static_cast<int32_t*>(std::malloc(bytes));
            if (!copied) {
                g_last_error = "mlx_run: failed to allocate generated token buffer";
                return 0;
            }
            std::memcpy(copied, captured_tokens.data(), bytes);
            *out_generated_ids = copied;
            *out_generated_len = static_cast<int32_t>(captured_tokens.size());
        }
        if (fused_lm_head_profile) {
            const size_t fused_hits = g_fused_lm_head_argmax_hits - fused_hits_start;
            std::fprintf(
                stderr,
                "[mlx][fused_lm_head_argmax] used=%s hits=%zu mode=run\n",
                can_use_fused_lm_head_argmax(ctx) ? "1" : "0",
                fused_hits
            );
        }

        *out_result = best;
        return 1;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 0;
    } catch (...) {
        g_last_error = "unknown error in mlx_run";
        return 0;
    }
}
