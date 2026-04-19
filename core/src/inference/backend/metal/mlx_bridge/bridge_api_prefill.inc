// bridge_api_prefill.inc — Prefill and embedding: mlx_prefill_first, mlx_prefill_logits,
// mlx_prefill_logits_batch, mlx_embed.

int32_t mlx_prefill_first(
    mlx_ctx* ctx,
    const int32_t* prompt_ids,
    int32_t prompt_len,
    int32_t* out_first_token
) {
    if (!ctx || !prompt_ids || prompt_len <= 0 || !out_first_token) {
        g_last_error = "mlx_prefill_first: invalid arguments";
        return 0;
    }

    try {
        if (logits_sanity_enabled()) {
            std::fprintf(stderr, "[mlx][logits_sanity] enter=mlx_prefill_first prompt_len=%d\n", prompt_len);
        }
        std::vector<int32_t> prompt_vec(prompt_ids, prompt_ids + prompt_len);
        reset_runtime_state(ctx);
        ctx->xray_enabled = (talu_metal_xray_is_enabled() != 0);
        ctx->kv_reserve_tokens = resolve_kv_reserve_target_tokens(ctx, prompt_len, 0);
        const int full_prompt_threshold = resolve_prefill_full_prompt_threshold(ctx);
        const auto select_first_token = [&](const array& seed_logits, const char* stage) -> array {
            const array logits_last = last_token_logits(seed_logits);
            if (logits_sanity_enabled()) {
                const int flat_size = static_cast<int>(logits_last.size());
                const array logits_flat = reshape(astype(logits_last, float32), {flat_size});
                eval(logits_flat);
                synchronize();
                emit_logits_sanity(stage, 0, logits_flat.data<float>(), flat_size);
            }
            return next_token_greedy(logits_last);
        };

        array first_token = array(0.0f);
        if (prompt_len > 1) {
            if (should_use_full_prompt_prefill(ctx, prompt_len, full_prompt_threshold)) {
                const array prompt_arr(prompt_vec.begin(), Shape{1, prompt_len}, int32);
                const array seed_logits = forward_logits(ctx, prompt_arr);
                first_token = select_first_token(seed_logits, "prefill_first_full");
            } else {
                const array seed_logits = prefill_seed_logits_chunked(ctx, prompt_vec, prompt_len);
                first_token = select_first_token(seed_logits, "prefill_first_chunked");
            }
        } else {
            const array prompt_arr(prompt_vec.begin(), Shape{1, prompt_len}, int32);
            const array seed_logits = forward_logits(ctx, prompt_arr);
            first_token = select_first_token(seed_logits, "prefill_first_single");
        }

        eval(first_token);
        synchronize();

        *out_first_token = first_token.item<int32_t>();
        ctx->stream_ready = true;
        return 1;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 0;
    } catch (...) {
        g_last_error = "unknown error in mlx_prefill_first";
        return 0;
    }
}

int32_t mlx_prefill_logits(
    mlx_ctx* ctx,
    const int32_t* prompt_ids,
    int32_t prompt_len,
    float* out_logits,
    int32_t logits_len
) {
    if (!ctx || !prompt_ids || prompt_len <= 0 || !out_logits || logits_len <= 0) {
        g_last_error = "mlx_prefill_logits: invalid arguments";
        return 0;
    }

    try {
        if (logits_sanity_enabled()) {
            std::fprintf(stderr, "[mlx][logits_sanity] enter=mlx_prefill_logits prompt_len=%d logits_len=%d\n", prompt_len, logits_len);
        }
        clear_fused_decode_ctx_cache();
        std::vector<int32_t> prompt_vec(prompt_ids, prompt_ids + prompt_len);
        reset_runtime_state(ctx);
        ctx->xray_enabled = (talu_metal_xray_is_enabled() != 0);
        ctx->kv_reserve_tokens = resolve_kv_reserve_target_tokens(ctx, prompt_len, 0);
        const int full_prompt_threshold = resolve_prefill_full_prompt_threshold(ctx);

        array first_logits = array(0.0f);
        const bool prefill_trace_enabled = ctx->xray_enabled &&
            (xray_should_emit(XRAY_POINT_EMBED, XRAY_GLOBAL_LAYER, static_cast<uint32_t>(prompt_len)) ||
                xray_should_emit(XRAY_POINT_LAYER_INPUT, 0, static_cast<uint32_t>(prompt_len)) ||
                xray_should_emit(XRAY_POINT_ATTN_OUT, 0, static_cast<uint32_t>(prompt_len)) ||
                xray_should_emit(XRAY_POINT_BLOCK_OUT, 0, static_cast<uint32_t>(prompt_len)) ||
                xray_should_emit(XRAY_POINT_FINAL_NORM, XRAY_GLOBAL_LAYER, 1) ||
                xray_should_emit(XRAY_POINT_LM_HEAD, XRAY_GLOBAL_LAYER, 0));
        if (prompt_len > 1) {
            if (should_use_full_prompt_prefill(ctx, prompt_len, full_prompt_threshold)) {
                const array prompt_arr(prompt_vec.begin(), Shape{1, prompt_len}, int32);
                const TraceFrame trace_frame{
                    .token = 0,
                    .layer_position = static_cast<uint32_t>(prompt_len),
                    .emit_embed = true,
                };
                const array seed_logits = prefill_trace_enabled
                    ? forward_logits(ctx, prompt_arr, &trace_frame)
                    : forward_logits(ctx, prompt_arr);
                first_logits = last_token_logits(seed_logits);
            } else if (prefill_trace_enabled) {
                // HF tensor references store prefill checkpoints from a full-prompt
                // pass. Emit those from full prompt, then rebuild stream state via
                // the incremental route to preserve decode behavior.
                const TraceFrame trace_frame{
                    .token = 0,
                    .layer_position = static_cast<uint32_t>(prompt_len),
                    .emit_embed = true,
                };
                const array prompt_arr(prompt_vec.begin(), Shape{1, prompt_len}, int32);
                const array seed_logits = forward_logits(ctx, prompt_arr, &trace_frame);
                first_logits = last_token_logits(seed_logits);

                reset_runtime_state(ctx);
                const array rebuild_logits = prefill_seed_logits_chunked(ctx, prompt_vec, prompt_len);
                (void)rebuild_logits;
            } else {
                const array seed_logits = prefill_seed_logits_chunked(ctx, prompt_vec, prompt_len);
                first_logits = last_token_logits(seed_logits);
            }
        } else {
            const array prompt_arr(prompt_vec.begin(), Shape{1, prompt_len}, int32);
            const TraceFrame trace_frame{
                .token = 0,
                .layer_position = static_cast<uint32_t>(prompt_len),
                .emit_embed = true,
            };
            const array seed_logits = prefill_trace_enabled
                ? forward_logits(ctx, prompt_arr, &trace_frame)
                : forward_logits(ctx, prompt_arr);
            first_logits = last_token_logits(seed_logits);
        }

        const int flat_size = static_cast<int>(first_logits.size());
        const array first_logits_flat = reshape(astype(first_logits, float32), {flat_size});
        if (static_cast<int32_t>(first_logits_flat.size()) != logits_len) {
            g_last_error = "mlx_prefill_logits: logits length mismatch";
            return 0;
        }

        eval(first_logits_flat);
        synchronize();
        if (logits_sanity_enabled()) {
            emit_logits_sanity("prefill_single", 0, first_logits_flat.data<float>(), logits_len);
        }
        std::memcpy(out_logits, first_logits_flat.data<float>(), static_cast<size_t>(logits_len) * sizeof(float));
        ctx->stream_ready = true;
        return 1;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 0;
    } catch (...) {
        g_last_error = "unknown error in mlx_prefill_logits";
        return 0;
    }
}

int32_t mlx_prefill_logits_with_vision(
    mlx_ctx* ctx,
    const int32_t* prompt_ids,
    int32_t prompt_len,
    const mlx_prefill_vision_input* vision_input,
    float* out_logits,
    int32_t logits_len
) {
    if (!vision_input) {
        g_last_error = "mlx_prefill_logits_with_vision: vision_input is null";
        return 0;
    }
    if (!ctx || !prompt_ids || prompt_len <= 0 || !out_logits || logits_len <= 0) {
        g_last_error = "mlx_prefill_logits_with_vision: invalid arguments";
        return 0;
    }
    if (!vision_input->merged_embeddings || vision_input->merged_value_count == 0) {
        g_last_error = "mlx_prefill_logits_with_vision: merged_embeddings must be non-empty";
        return 0;
    }
    if (vision_input->image_token_id == 0) {
        g_last_error = "mlx_prefill_logits_with_vision: image_token_id must be non-zero";
        return 0;
    }

    try {
        clear_fused_decode_ctx_cache();
        std::vector<int32_t> prompt_vec(prompt_ids, prompt_ids + prompt_len);
        reset_runtime_state(ctx);
        ctx->xray_enabled = (talu_metal_xray_is_enabled() != 0);
        ctx->kv_reserve_tokens = resolve_kv_reserve_target_tokens(ctx, prompt_len, 0);

        const array prompt_arr(prompt_vec.data(), Shape{1, prompt_len}, int32);
        array hidden = embedding_lookup(ctx, prompt_arr);
        if (std::fabs(ctx->cfg.embedding_multiplier - 1.0f) > 1.0e-6f) {
            hidden = hidden * array(ctx->cfg.embedding_multiplier, hidden.dtype());
        }

        if (hidden.ndim() != 3 || hidden.shape(0) != 1 || hidden.shape(1) != prompt_len) {
            g_last_error = "mlx_prefill_logits_with_vision: unexpected hidden embedding shape";
            return 0;
        }
        const int hidden_size = hidden.shape(2);
        if (hidden_size <= 0) {
            g_last_error = "mlx_prefill_logits_with_vision: invalid hidden size";
            return 0;
        }
        if (vision_input->merged_value_count % static_cast<uint32_t>(hidden_size) != 0) {
            g_last_error = "mlx_prefill_logits_with_vision: merged embedding values do not align to hidden size";
            return 0;
        }

        const size_t merged_rows = static_cast<size_t>(vision_input->merged_value_count) / static_cast<size_t>(hidden_size);
        const array merged_embeddings_f32(
            vision_input->merged_embeddings,
            Shape{static_cast<int>(merged_rows), hidden_size},
            float32
        );
        const array merged_embeddings = astype(merged_embeddings_f32, hidden.dtype());

        std::vector<int32_t> image_row_positions;
        image_row_positions.reserve(merged_rows);
        for (int32_t row = 0; row < prompt_len; ++row) {
            if (prompt_vec[static_cast<size_t>(row)] == static_cast<int32_t>(vision_input->image_token_id)) {
                image_row_positions.push_back(row);
            }
        }
        if (image_row_positions.size() != merged_rows) {
            if (image_row_positions.size() > merged_rows) {
                g_last_error = "mlx_prefill_logits_with_vision: prompt image placeholders exceed provided merged embeddings";
            } else {
                g_last_error = "mlx_prefill_logits_with_vision: unused merged embeddings do not match prompt image placeholder count";
            }
            return 0;
        }

        const array hidden_2d = reshape(hidden, {prompt_len, hidden_size});
        const array row_indices_base(
            image_row_positions.data(),
            Shape{static_cast<int>(merged_rows), 1},
            int32
        );
        const array row_indices = repeat(row_indices_base, hidden_size, 1);
        const array hidden_scattered_2d = put_along_axis(hidden_2d, row_indices, merged_embeddings, 0);
        const array hidden_scattered = reshape(hidden_scattered_2d, {1, prompt_len, hidden_size});

        const int full_prompt_threshold = resolve_prefill_full_prompt_threshold(ctx);
        const bool use_full_prompt =
            prompt_len <= 1 || should_use_full_prompt_prefill(ctx, prompt_len, full_prompt_threshold);

        array normed_hidden = array(0.0f);
        if (use_full_prompt) {
            normed_hidden = forward_hidden_from_embeddings(ctx, prompt_arr, hidden_scattered, true);
        } else {
            const int chunk_size = resolve_prefill_chunk_size(ctx);
            const int tail_chunk_size = std::max(
                chunk_size,
                env_positive_int("TALU_METAL_PREFILL_TAIL_CHUNK_SIZE", chunk_size)
            );
            const int32_t tail_len = std::min<int32_t>(prompt_len, tail_chunk_size);
            const int32_t prefix_len = prompt_len - tail_len;

            int32_t offset = 0;
            while (offset < prefix_len) {
                const int32_t take_count = std::min<int32_t>(prefix_len - offset, chunk_size);
                const array chunk_ids(prompt_vec.data() + static_cast<size_t>(offset), Shape{1, take_count}, int32);
                const array chunk_hidden = slice(
                    hidden_scattered,
                    {0, offset, 0},
                    {1, offset + take_count, hidden_size}
                );
                const array pending_hidden = forward_hidden_from_embeddings(ctx, chunk_ids, chunk_hidden, false);
                eval(pending_hidden);
                synchronize();
                offset += take_count;
            }

            const auto* tail_ids_ptr = prompt_vec.data() + static_cast<size_t>(prefix_len);
            const array tail_ids(tail_ids_ptr, Shape{1, tail_len}, int32);
            const array tail_hidden = slice(
                hidden_scattered,
                {0, prefix_len, 0},
                {1, prefix_len + tail_len, hidden_size}
            );
            normed_hidden = forward_hidden_from_embeddings(ctx, tail_ids, tail_hidden, true);
        }

        if (ctx->lm_head_q_decode_enabled &&
            normed_hidden.ndim() == 3 &&
            normed_hidden.shape(0) == 1 &&
            normed_hidden.shape(1) > 1 &&
            ctx->cfg.hidden_size_per_layer_input <= 0)
        {
            const int b = normed_hidden.shape(0);
            const int t = normed_hidden.shape(1);
            const int h = normed_hidden.shape(2);
            normed_hidden = slice(normed_hidden, {0, t - 1, 0}, {b, t, h});
        }

        array logits = forward_logits_from_hidden(ctx, normed_hidden);
        if (ctx->cfg.final_logit_softcapping > 0.0f) {
            const array softcap = array(ctx->cfg.final_logit_softcapping, logits.dtype());
            logits = tanh(logits / softcap) * softcap;
        }
        const array first_logits = last_token_logits(logits);
        const int flat_size = static_cast<int>(first_logits.size());
        const array first_logits_flat = reshape(astype(first_logits, float32), {flat_size});
        if (static_cast<int32_t>(first_logits_flat.size()) != logits_len) {
            g_last_error = "mlx_prefill_logits_with_vision: logits length mismatch";
            return 0;
        }

        eval(first_logits_flat);
        synchronize();
        std::memcpy(out_logits, first_logits_flat.data<float>(), static_cast<size_t>(logits_len) * sizeof(float));
        ctx->stream_ready = true;
        return 1;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 0;
    } catch (...) {
        g_last_error = "unknown error in mlx_prefill_logits_with_vision";
        return 0;
    }
}

int32_t mlx_prefill_logits_batch(
    mlx_ctx* const* ctxs,
    const int32_t* const* prompt_ids_ptrs,
    const int32_t* prompt_lens,
    float* const* out_logits_ptrs,
    int32_t batch_size,
    int32_t logits_len
) {
    if (!ctxs || !prompt_ids_ptrs || !prompt_lens || !out_logits_ptrs ||
        batch_size <= 0 || logits_len <= 0) {
        g_last_error = "mlx_prefill_logits_batch: invalid arguments";
        return 0;
    }

    if (batch_size == 1) {
        return mlx_prefill_logits(ctxs[0], prompt_ids_ptrs[0], prompt_lens[0], out_logits_ptrs[0], logits_len);
    }
    if (logits_sanity_enabled()) {
        std::fprintf(stderr, "[mlx][logits_sanity] enter=mlx_prefill_logits_batch batch=%d logits_len=%d\n", batch_size, logits_len);
    }

    for (int32_t idx = 0; idx < batch_size; ++idx) {
        if (!ctxs[idx] || !prompt_ids_ptrs[idx] || prompt_lens[idx] <= 0 || !out_logits_ptrs[idx]) {
            g_last_error = "mlx_prefill_logits_batch: null ctx/prompt/logits entry";
            return 0;
        }
    }

    // Per-layer input projection routes are currently unsafe under fused
    // prefill batching; preserve correctness by running per-row prefill.
    bool requires_serial_prefill = false;
    for (int32_t idx = 0; idx < batch_size and !requires_serial_prefill; ++idx) {
        if (ctxs[idx]->cfg.hidden_size_per_layer_input > 0) {
            requires_serial_prefill = true;
            break;
        }
        for (const LayerWeights& lw : ctxs[idx]->layers) {
            if (lw.has_per_layer_input) {
                requires_serial_prefill = true;
                break;
            }
        }
    }
    if (requires_serial_prefill) {
        for (int32_t idx = 0; idx < batch_size; ++idx) {
            const int32_t ok = mlx_prefill_logits(
                ctxs[idx],
                prompt_ids_ptrs[idx],
                prompt_lens[idx],
                out_logits_ptrs[idx],
                logits_len
            );
            if (ok == 0) return 0;
        }
        return 1;
    }

    try {
        clear_fused_decode_ctx_cache();
        const int full_prompt_threshold = resolve_prefill_batch_full_prompt_threshold();
        const int len_bucket = resolve_prefill_batch_len_bucket();
        std::vector<uint8_t> prefill_done(static_cast<size_t>(batch_size), 0);

        // Grouped fused prefill:
        // - T| topology groups: full-attention variable-length fused prefill.
        // - R| topology groups: staged ragged fused prefill (safe for hybrid).
        // - P| prompt+topology groups: fixed-length fused prefill.
        std::unordered_map<std::string, std::vector<int32_t>> prefill_groups;
        prefill_groups.reserve(static_cast<size_t>(batch_size));
        for (int32_t idx = 0; idx < batch_size; ++idx) {
            if (prompt_lens[idx] > full_prompt_threshold) {
                continue;
            }

            std::string key;
            if (supports_variable_prefill_fusion(ctxs[idx])) {
                if (!build_prefill_topology_key(ctxs[idx], &key)) continue;
                key = "T|" + key;
                if (len_bucket > 0) {
                    const int32_t bucket_id = prompt_lens[idx] / len_bucket;
                    key.push_back('|');
                    key.append(std::to_string(bucket_id));
                }
            } else if (supports_ragged_prefill_fusion(ctxs[idx])) {
                if (!build_prefill_topology_key(ctxs[idx], &key)) continue;
                key = "R|" + key;
            } else {
                if (!build_prefill_fusion_key(ctxs[idx], prompt_lens[idx], full_prompt_threshold, &key)) continue;
                key = "P|" + key;
            }
            prefill_groups[key].push_back(idx);
        }

        for (auto& entry : prefill_groups) {
            const std::string& group_key = entry.first;
            std::vector<int32_t>& rows = entry.second;
            const int32_t group_size = static_cast<int32_t>(rows.size());
            if (group_size < 2) continue;

            const bool ragged_group = group_key.rfind("R|", 0) == 0;
            if (ragged_group) {
                (void)run_ragged_prefill_group(
                    rows,
                    ctxs,
                    prompt_ids_ptrs,
                    prompt_lens,
                    out_logits_ptrs,
                    logits_len,
                    &prefill_done
                );
                continue;
            }

            const bool variable_len_capable = supports_variable_prefill_fusion(ctxs[rows[0]]);
            const int32_t first_prompt_len = prompt_lens[rows[0]];
            int32_t max_prompt_len = first_prompt_len;
            bool all_same_prompt_len = true;
            for (int32_t row_idx : rows) {
                const int32_t row_prompt_len = prompt_lens[row_idx];
                max_prompt_len = std::max(max_prompt_len, row_prompt_len);
                if (row_prompt_len != first_prompt_len) {
                    all_same_prompt_len = false;
                }
            }
            const bool variable_prompt_group = variable_len_capable && !all_same_prompt_len;
            const int32_t group_prompt_len = variable_prompt_group ? max_prompt_len : first_prompt_len;

            for (int32_t row_idx : rows) {
                reset_runtime_state(ctxs[row_idx]);
                ctxs[row_idx]->xray_enabled = false;
            }

            std::vector<int32_t> prompt_matrix(
                static_cast<size_t>(group_size) * static_cast<size_t>(group_prompt_len)
            );
            for (int32_t row = 0; row < group_size; ++row) {
                const int32_t src_row = rows[static_cast<size_t>(row)];
                const int32_t* src = prompt_ids_ptrs[src_row];
                const int32_t src_prompt_len = prompt_lens[src_row];
                int32_t* dst = prompt_matrix.data() + static_cast<size_t>(row) * static_cast<size_t>(group_prompt_len);
                std::memcpy(
                    dst,
                    src,
                    static_cast<size_t>(src_prompt_len) * sizeof(int32_t)
                );
                if (src_prompt_len < group_prompt_len) {
                    const int32_t pad_token = src[src_prompt_len - 1];
                    std::fill(
                        dst + src_prompt_len,
                        dst + group_prompt_len,
                        pad_token
                    );
                }
            }

            mlx_ctx fused_ctx = *ctxs[rows[0]];
            reset_runtime_state(&fused_ctx);
            fused_ctx.xray_enabled = false;

            const array prompt_arr(
                prompt_matrix.data(),
                Shape{group_size, group_prompt_len},
                int32
            );
            const array full_logits = forward_logits(&fused_ctx, prompt_arr); // [B,S,V]
            const int vocab_size = full_logits.shape(2);
            if (vocab_size != logits_len) {
                g_last_error = "mlx_prefill_logits_batch: logits length mismatch";
                return 0;
            }

            array logits_rows = array(0.0f);
            if (variable_prompt_group) {
                std::vector<int32_t> gather_indices(static_cast<size_t>(group_size));
                for (int32_t row = 0; row < group_size; ++row) {
                    const int32_t src_row = rows[static_cast<size_t>(row)];
                    gather_indices[static_cast<size_t>(row)] = prompt_lens[src_row] - 1;
                }
                const array gather_idx_base(
                    gather_indices.data(),
                    Shape{group_size, 1, 1},
                    int32
                );
                const array gather_idx = repeat(gather_idx_base, vocab_size, 2); // [B,1,V]
                const array gathered_logits = take_along_axis(full_logits, gather_idx, 1); // [B,1,V]
                logits_rows = reshape(astype(gathered_logits, float32), {group_size, logits_len});
            } else {
                const array logits_last = last_token_logits(full_logits);
                logits_rows = reshape(astype(logits_last, float32), {group_size, logits_len});
            }
            eval(logits_rows);
            synchronize();

            std::vector<mlx_ctx*> group_ctxs(static_cast<size_t>(group_size));
            const float* rows_ptr = logits_rows.data<float>();
            const size_t row_bytes = static_cast<size_t>(logits_len) * sizeof(float);
            for (int32_t row = 0; row < group_size; ++row) {
                const int32_t src_row = rows[static_cast<size_t>(row)];
                const float* row_logits =
                    rows_ptr + static_cast<size_t>(row) * static_cast<size_t>(logits_len);
                if (logits_sanity_enabled()) {
                    emit_logits_sanity("prefill_batch_fused", src_row, row_logits, logits_len);
                }
                std::memcpy(
                    out_logits_ptrs[src_row],
                    row_logits,
                    row_bytes
                );
                ctxs[src_row]->trace_decode_token = 1;
                prefill_done[static_cast<size_t>(src_row)] = 1;
                group_ctxs[static_cast<size_t>(row)] = ctxs[src_row];
            }
            if (variable_prompt_group) {
                std::vector<int32_t> group_prompt_lens(static_cast<size_t>(group_size));
                for (int32_t row = 0; row < group_size; ++row) {
                    const int32_t src_row = rows[static_cast<size_t>(row)];
                    group_prompt_lens[static_cast<size_t>(row)] = prompt_lens[src_row];
                }
                scatter_fused_prefill_runtime_state_to_ctxs(
                    fused_ctx,
                    group_ctxs.data(),
                    group_prompt_lens.data(),
                    group_size
                );
            } else {
                scatter_fused_runtime_state_to_ctxs(fused_ctx, group_ctxs.data(), group_size);
            }
        }

        std::vector<int32_t> fallback_rows;
        fallback_rows.reserve(static_cast<size_t>(batch_size));
        for (int32_t idx = 0; idx < batch_size; ++idx) {
            if (prefill_done[static_cast<size_t>(idx)] == 0) {
                fallback_rows.push_back(idx);
            }
        }
        if (fallback_rows.empty()) {
            return 1;
        }

        std::vector<std::vector<int32_t>> prompt_storage;
        if (prompt_storage.capacity() < fallback_rows.size()) {
            prompt_storage.reserve(fallback_rows.size());
        }

        for (int32_t idx : fallback_rows) {
            mlx_ctx* ctx = ctxs[idx];
            const int32_t* prompt_ids = prompt_ids_ptrs[idx];
            const int32_t prompt_len = prompt_lens[idx];
            prompt_storage.emplace_back(prompt_ids, prompt_ids + prompt_len);
            const std::vector<int32_t>& prompt_vec = prompt_storage.back();

            reset_runtime_state(ctx);
            ctx->xray_enabled = (talu_metal_xray_is_enabled() != 0);
            ctx->kv_reserve_tokens = resolve_kv_reserve_target_tokens(ctx, prompt_len, 0);

            array first_logits = array(0.0f);
            if (prompt_len > 1) {
                if (prompt_len <= full_prompt_threshold) {
                    const array prompt_arr(prompt_vec.begin(), Shape{1, prompt_len}, int32);
                    const array seed_logits = forward_logits(ctx, prompt_arr);
                    first_logits = last_token_logits(seed_logits);
                } else {
                    const array seed_logits = prefill_seed_logits_chunked(ctx, prompt_vec, prompt_len);
                    first_logits = last_token_logits(seed_logits);
                }
            } else {
                const array prompt_arr(prompt_vec.begin(), Shape{1, prompt_len}, int32);
                const array seed_logits = forward_logits(ctx, prompt_arr);
                first_logits = last_token_logits(seed_logits);
            }

            const int flat_size = static_cast<int>(first_logits.size());
            const array first_logits_flat = reshape(astype(first_logits, float32), {flat_size});
            if (static_cast<int32_t>(first_logits_flat.size()) != logits_len) {
                g_last_error = "mlx_prefill_logits_batch: logits length mismatch";
                return 0;
            }
            eval(first_logits_flat);
            synchronize();
            if (logits_sanity_enabled()) {
                emit_logits_sanity("prefill_batch_fallback", idx, first_logits_flat.data<float>(), logits_len);
            }
            std::memcpy(
                out_logits_ptrs[idx],
                first_logits_flat.data<float>(),
                static_cast<size_t>(logits_len) * sizeof(float)
            );
            ctxs[idx]->stream_ready = true;
            ctxs[idx]->trace_decode_token = 1;
        }
        return 1;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 0;
    } catch (...) {
        g_last_error = "unknown error in mlx_prefill_logits_batch";
        return 0;
    }
}

int32_t mlx_embed(
    mlx_ctx* ctx,
    const int32_t* token_ids,
    int32_t token_len,
    int32_t pooling,
    int32_t normalize,
    float* out_embedding,
    int32_t embedding_len
) {
    if (!ctx || !token_ids || token_len <= 0 || !out_embedding || embedding_len <= 0) {
        g_last_error = "mlx_embed: invalid arguments";
        return 0;
    }

    try {
        std::vector<int32_t> token_vec(token_ids, token_ids + token_len);
        reset_runtime_state(ctx);
        ctx->xray_enabled = (talu_metal_xray_is_enabled() != 0);

        const array token_arr(token_vec.begin(), Shape{1, token_len}, int32);
        const array hidden = forward_hidden(ctx, token_arr, true);
        const array hidden_f32 = astype(hidden, float32);
        const int hidden_size = hidden_f32.shape(2);
        if (embedding_len < hidden_size) {
            g_last_error = "mlx_embed: embedding buffer too small";
            return 0;
        }

        const int flat_size = static_cast<int>(hidden_f32.size());
        const array hidden_flat = reshape(hidden_f32, {flat_size});
        eval(hidden_flat);
        synchronize();

        const float* hidden_ptr = hidden_flat.data<float>();
        if (!hidden_ptr) {
            g_last_error = "mlx_embed: hidden data unavailable";
            return 0;
        }

        switch (pooling) {
            case 0: { // last
                const size_t offset = static_cast<size_t>(token_len - 1) * static_cast<size_t>(hidden_size);
                std::memcpy(out_embedding, hidden_ptr + offset, static_cast<size_t>(hidden_size) * sizeof(float));
                break;
            }
            case 1: { // mean
                std::fill(out_embedding, out_embedding + hidden_size, 0.0f);
                for (int32_t row = 0; row < token_len; ++row) {
                    const size_t row_offset = static_cast<size_t>(row) * static_cast<size_t>(hidden_size);
                    for (int col = 0; col < hidden_size; ++col) {
                        out_embedding[col] += hidden_ptr[row_offset + static_cast<size_t>(col)];
                    }
                }
                const float inv_count = 1.0f / static_cast<float>(token_len);
                for (int col = 0; col < hidden_size; ++col) {
                    out_embedding[col] *= inv_count;
                }
                break;
            }
            case 2: { // first
                std::memcpy(out_embedding, hidden_ptr, static_cast<size_t>(hidden_size) * sizeof(float));
                break;
            }
            default:
                g_last_error = "mlx_embed: invalid pooling";
                return 0;
        }

        if (normalize != 0) {
            double sum_sq = 0.0;
            for (int col = 0; col < hidden_size; ++col) {
                const double v = static_cast<double>(out_embedding[col]);
                sum_sq += v * v;
            }
            if (sum_sq > 0.0) {
                const float inv_norm = static_cast<float>(1.0 / std::sqrt(sum_sq));
                for (int col = 0; col < hidden_size; ++col) {
                    out_embedding[col] *= inv_norm;
                }
            }
        }

        return 1;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 0;
    } catch (...) {
        g_last_error = "unknown error in mlx_embed";
        return 0;
    }
}
