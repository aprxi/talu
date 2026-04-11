// bridge_api_stream.inc — Batch decode: mlx_decode_topk_candidates_batch,
// mlx_decode_topk_candidates_with_sampling.

int32_t mlx_decode_topk_candidates_batch(
    mlx_ctx* const* ctxs,
    const int32_t* tokens,
    int32_t top_k,
    float* const* out_candidate_logits_ptrs,
    int32_t* const* out_candidate_ids_ptrs,
    int32_t* out_candidate_counts,
    int32_t batch_size
) {
    if (!ctxs || !tokens || !out_candidate_logits_ptrs || !out_candidate_ids_ptrs ||
        !out_candidate_counts || batch_size <= 0) {
        g_last_error = "mlx_decode_topk_candidates_batch: invalid arguments";
        return 0;
    }
    if (top_k <= 0) {
        g_last_error = "mlx_decode_topk_candidates_batch: top_k must be > 0";
        return 0;
    }

    for (int32_t i = 0; i < batch_size; ++i) {
        if (!ctxs[i] || !out_candidate_logits_ptrs[i] || !out_candidate_ids_ptrs[i]) {
            g_last_error = "mlx_decode_topk_candidates_batch: null ctx or output entry";
            return 0;
        }
        if (!ctxs[i]->stream_ready) {
            g_last_error = "mlx_decode_topk_candidates_batch: prefill not ready";
            return 0;
        }
        if (tokens[i] < 0) {
            g_last_error = "mlx_decode_topk_candidates_batch: token must be >= 0";
            return 0;
        }
    }

    try {
        if (logits_sanity_enabled()) {
            std::fprintf(stderr, "[mlx][logits_sanity] enter=mlx_decode_topk_candidates_batch batch=%d top_k=%d\n", batch_size, top_k);
        }
        const int k = top_k;
        std::vector<mlx_ctx*> current_rows(static_cast<size_t>(batch_size));
        for (int32_t i = 0; i < batch_size; ++i) {
            current_rows[static_cast<size_t>(i)] = ctxs[i];
        }
        if (g_fused_decode_ctx_cache_dirty &&
            !same_ctx_row_set(g_fused_decode_ctx_cache_rows, current_rows))
        {
            clear_fused_decode_ctx_cache();
        }

        mlx_ctx* cached_fused_ctx = nullptr;
        if (try_get_cached_fused_decode_ctx_by_rows(current_rows, &cached_fused_ctx)) {
            std::vector<int32_t> token_vec(static_cast<size_t>(batch_size));
            for (int32_t i = 0; i < batch_size; ++i) {
                token_vec[static_cast<size_t>(i)] = tokens[i];
            }

            const array token_arr(token_vec.data(), Shape{batch_size, 1}, int32);
            const array logits_last = last_token_logits(forward_logits(cached_fused_ctx, token_arr)); // [B,1,V]
            emit_logits_sanity_rows("decode_topk_batch_cached", logits_last, nullptr, batch_size);
            if (logits_last.ndim() != 3) {
                g_last_error = "mlx_decode_topk_candidates_batch: invalid cached fused logits rank";
                return 0;
            }
            const int vocab = logits_last.shape(2);
            if (vocab <= 0 || vocab < k) {
                g_last_error = "mlx_decode_topk_candidates_batch: invalid cached fused vocab size";
                return 0;
            }

            if (k == 1) {
                const array top_ids = reshape(astype(argmax(logits_last, -1), int32), {batch_size}); // [B]
                eval(top_ids);
                synchronize();
                const int32_t* ids_ptr = top_ids.data<int32_t>();
                for (int32_t i = 0; i < batch_size; ++i) {
                    out_candidate_ids_ptrs[i][0] = ids_ptr[static_cast<size_t>(i)];
                    out_candidate_logits_ptrs[i][0] = 0.0f;
                    out_candidate_counts[i] = 1;
                    ctxs[i]->trace_decode_token += 1;
                }
            } else {
                auto topk_rows_fn = compiled_topk_candidates_rows(k);
                const std::vector<array> topk_outputs = (*topk_rows_fn)({logits_last});
                if (topk_outputs.size() != 2) {
                    g_last_error = "mlx_decode_topk_candidates_batch: invalid cached fused topk output";
                    return 0;
                }
                const array topk_logits_rows = astype(topk_outputs[0], float32);
                const array topk_ids_rows = astype(topk_outputs[1], int32);
                eval({topk_logits_rows, topk_ids_rows});
                synchronize();

                const float* logits_ptr = topk_logits_rows.data<float>();
                const int32_t* ids_ptr = topk_ids_rows.data<int32_t>();
                const size_t row_floats = static_cast<size_t>(k);
                for (int32_t i = 0; i < batch_size; ++i) {
                    std::memcpy(
                        out_candidate_logits_ptrs[i],
                        logits_ptr + static_cast<size_t>(i) * row_floats,
                        row_floats * sizeof(float)
                    );
                    std::memcpy(
                        out_candidate_ids_ptrs[i],
                        ids_ptr + static_cast<size_t>(i) * row_floats,
                        row_floats * sizeof(int32_t)
                    );
                    stable_sort_topk_pairs(out_candidate_logits_ptrs[i], out_candidate_ids_ptrs[i], k);
                    out_candidate_counts[i] = k;
                    ctxs[i]->trace_decode_token += 1;
                }
            }
            g_fused_decode_ctx_cache_dirty = true;
            return 1;
        }

        // Fast path: handle fully compatible full-batch decode without
        // per-token grouping/key construction.
        mlx_ctx fused_ctx_full_batch;
        int32_t full_batch_common_offset = -1;
        if (prepare_fused_decode_ctx(
                ctxs,
                batch_size,
                &fused_ctx_full_batch,
                &full_batch_common_offset
            ) &&
            full_batch_common_offset >= 0)
        {
            std::vector<int32_t> token_vec(static_cast<size_t>(batch_size));
            for (int32_t i = 0; i < batch_size; ++i) {
                token_vec[static_cast<size_t>(i)] = tokens[i];
            }

            const array token_arr(token_vec.data(), Shape{batch_size, 1}, int32);
            const array logits_last = last_token_logits(forward_logits(&fused_ctx_full_batch, token_arr)); // [B,1,V]
            emit_logits_sanity_rows("decode_topk_batch_full", logits_last, nullptr, batch_size);
            if (logits_last.ndim() != 3) {
                g_last_error = "mlx_decode_topk_candidates_batch: invalid full-batch fused logits rank";
                return 0;
            }
            const int vocab = logits_last.shape(2);
            if (vocab <= 0 || vocab < k) {
                g_last_error = "mlx_decode_topk_candidates_batch: invalid full-batch fused vocab size";
                return 0;
            }

            if (k == 1) {
                const array top_ids = reshape(astype(argmax(logits_last, -1), int32), {batch_size});
                eval(top_ids);
                synchronize();
                const int32_t* ids_ptr = top_ids.data<int32_t>();
                for (int32_t i = 0; i < batch_size; ++i) {
                    out_candidate_ids_ptrs[i][0] = ids_ptr[static_cast<size_t>(i)];
                    out_candidate_logits_ptrs[i][0] = 0.0f;
                    out_candidate_counts[i] = 1;
                    ctxs[i]->trace_decode_token += 1;
                }
            } else {
                auto topk_rows_fn = compiled_topk_candidates_rows(k);
                const std::vector<array> topk_outputs = (*topk_rows_fn)({logits_last});
                if (topk_outputs.size() != 2) {
                    g_last_error = "mlx_decode_topk_candidates_batch: invalid full-batch fused topk output";
                    return 0;
                }
                const array topk_logits_rows = astype(topk_outputs[0], float32);
                const array topk_ids_rows = astype(topk_outputs[1], int32);
                eval({topk_logits_rows, topk_ids_rows});
                synchronize();

                const float* logits_ptr = topk_logits_rows.data<float>();
                const int32_t* ids_ptr = topk_ids_rows.data<int32_t>();
                const size_t row_floats = static_cast<size_t>(k);
                for (int32_t i = 0; i < batch_size; ++i) {
                    std::memcpy(
                        out_candidate_logits_ptrs[i],
                        logits_ptr + static_cast<size_t>(i) * row_floats,
                        row_floats * sizeof(float)
                    );
                    std::memcpy(
                        out_candidate_ids_ptrs[i],
                        ids_ptr + static_cast<size_t>(i) * row_floats,
                        row_floats * sizeof(int32_t)
                    );
                    stable_sort_topk_pairs(out_candidate_logits_ptrs[i], out_candidate_ids_ptrs[i], k);
                    out_candidate_counts[i] = k;
                    ctxs[i]->trace_decode_token += 1;
                }
            }
            scatter_fused_runtime_state_to_ctxs(fused_ctx_full_batch, ctxs, batch_size);

            std::string next_key;
            if (build_decode_fusion_key(ctxs[0], &next_key)) {
                store_cached_fused_decode_ctx(current_rows, next_key, fused_ctx_full_batch);
            } else {
                clear_fused_decode_ctx_cache();
            }
            return 1;
        }

        std::vector<uint8_t> decode_done(static_cast<size_t>(batch_size), 0);

        std::unordered_map<std::string, std::vector<int32_t>> decode_groups;
        decode_groups.reserve(static_cast<size_t>(batch_size));
        for (int32_t row = 0; row < batch_size; ++row) {
            std::string key;
            if (!build_decode_fusion_key(ctxs[row], &key)) continue;
            decode_groups[key].push_back(row);
        }

        for (auto& entry : decode_groups) {
            std::vector<int32_t>& rows = entry.second;
            const int32_t group_size = static_cast<int32_t>(rows.size());
            if (group_size < 2) continue;
            const std::string& group_key = entry.first;

            std::vector<mlx_ctx*> group_ctxs(static_cast<size_t>(group_size));
            for (int32_t i = 0; i < group_size; ++i) {
                group_ctxs[static_cast<size_t>(i)] = ctxs[rows[static_cast<size_t>(i)]];
            }

            mlx_ctx fused_ctx_local;
            mlx_ctx* fused_ctx = nullptr;
            const bool cache_hit = try_get_cached_fused_decode_ctx(group_ctxs, group_key, &fused_ctx);
            if (!cache_hit) {
                int32_t common_offset = -1;
                if (!prepare_fused_decode_ctx(group_ctxs.data(), group_size, &fused_ctx_local, &common_offset) ||
                    common_offset < 0)
                {
                    continue;
                }
                fused_ctx = &fused_ctx_local;
            }

            std::vector<int32_t> token_vec(static_cast<size_t>(group_size));
            for (int32_t i = 0; i < group_size; ++i) {
                token_vec[static_cast<size_t>(i)] = tokens[rows[static_cast<size_t>(i)]];
            }

            const array token_arr(token_vec.data(), Shape{group_size, 1}, int32);
            const array logits_last = last_token_logits(forward_logits(fused_ctx, token_arr)); // [B,1,V]
            emit_logits_sanity_rows("decode_topk_batch_group", logits_last, rows.data(), group_size);
            if (logits_last.ndim() != 3) {
                g_last_error = "mlx_decode_topk_candidates_batch: invalid fused logits rank";
                return 0;
            }
            const int vocab = logits_last.shape(2);
            if (vocab <= 0 || vocab < k) {
                g_last_error = "mlx_decode_topk_candidates_batch: invalid fused vocab size";
                return 0;
            }

            if (k == 1) {
                const array top_ids = reshape(astype(argmax(logits_last, -1), int32), {group_size});
                eval(top_ids);
                synchronize();
                const int32_t* ids_ptr = top_ids.data<int32_t>();
                for (int32_t i = 0; i < group_size; ++i) {
                    const int32_t row = rows[static_cast<size_t>(i)];
                    out_candidate_ids_ptrs[row][0] = ids_ptr[static_cast<size_t>(i)];
                    out_candidate_logits_ptrs[row][0] = 0.0f;
                    out_candidate_counts[row] = 1;
                    ctxs[row]->trace_decode_token += 1;
                    decode_done[static_cast<size_t>(row)] = 1;
                }
            } else {
                auto topk_rows_fn = compiled_topk_candidates_rows(k);
                const std::vector<array> topk_outputs = (*topk_rows_fn)({logits_last});
                if (topk_outputs.size() != 2) {
                    g_last_error = "mlx_decode_topk_candidates_batch: invalid fused topk output";
                    return 0;
                }
                const array topk_logits_rows = astype(topk_outputs[0], float32); // [B,k]
                const array topk_ids_rows = astype(topk_outputs[1], int32);       // [B,k]
                eval({topk_logits_rows, topk_ids_rows});
                synchronize();

                const float* logits_ptr = topk_logits_rows.data<float>();
                const int32_t* ids_ptr = topk_ids_rows.data<int32_t>();
                const size_t row_floats = static_cast<size_t>(k);
                for (int32_t i = 0; i < group_size; ++i) {
                    const int32_t row = rows[static_cast<size_t>(i)];
                    std::memcpy(
                        out_candidate_logits_ptrs[row],
                        logits_ptr + static_cast<size_t>(i) * row_floats,
                        row_floats * sizeof(float)
                    );
                    std::memcpy(
                        out_candidate_ids_ptrs[row],
                        ids_ptr + static_cast<size_t>(i) * row_floats,
                        row_floats * sizeof(int32_t)
                    );
                    stable_sort_topk_pairs(out_candidate_logits_ptrs[row], out_candidate_ids_ptrs[row], k);
                    out_candidate_counts[row] = k;
                    ctxs[row]->trace_decode_token += 1;
                    decode_done[static_cast<size_t>(row)] = 1;
                }
            }
            scatter_fused_runtime_state_to_ctxs(*fused_ctx, group_ctxs.data(), group_size);

            std::string next_key;
            if (build_decode_fusion_key(group_ctxs[0], &next_key)) {
                if (!cache_hit) {
                    store_cached_fused_decode_ctx(group_ctxs, next_key, *fused_ctx);
                } else {
                    g_fused_decode_ctx_cache_rows = group_ctxs;
                    g_fused_decode_ctx_cache_key = next_key;
                    g_fused_decode_ctx_cache_dirty = false;
                }
            } else {
                clear_fused_decode_ctx_cache();
            }
        }

        std::vector<int32_t> fallback_rows;
        fallback_rows.reserve(static_cast<size_t>(batch_size));
        for (int32_t i = 0; i < batch_size; ++i) {
            if (decode_done[static_cast<size_t>(i)] == 0) {
                fallback_rows.push_back(i);
            }
        }
        if (fallback_rows.empty()) {
            return 1;
        }

        auto topk_fn = compiled_topk_candidates(k);

        auto& logits_flat_batch = g_decode_topk_batch_logits_flat_scratch;
        auto& ids_flat_batch = g_decode_topk_batch_ids_flat_scratch;
        logits_flat_batch.clear();
        ids_flat_batch.clear();
        if (logits_flat_batch.capacity() < fallback_rows.size()) {
            logits_flat_batch.reserve(fallback_rows.size());
        }
        if (ids_flat_batch.capacity() < fallback_rows.size()) {
            ids_flat_batch.reserve(fallback_rows.size());
        }

        std::vector<array> eval_targets;
        eval_targets.reserve(fallback_rows.size() * 2);

        for (int32_t i : fallback_rows) {
            mlx_ctx* ctx = ctxs[i];
            const int32_t token_scalar = tokens[i];
            const array token_arr(&token_scalar, Shape{1, 1}, int32);
            const array logits_last = last_token_logits(forward_logits(ctx, token_arr)); // [1,1,V]
            emit_logits_sanity_rows("decode_topk_batch_fallback", logits_last, &i, 1);
            if (logits_last.ndim() != 3) {
                g_last_error = "mlx_decode_topk_candidates_batch: invalid logits rank";
                return 0;
            }
            const int vocab = logits_last.shape(2);
            if (vocab <= 0 || vocab < k) {
                g_last_error = "mlx_decode_topk_candidates_batch: invalid vocab size";
                return 0;
            }

            if (k == 1) {
                const array top_id = reshape(astype(argmax(logits_last, -1), int32), {1});
                ids_flat_batch.push_back(top_id);
                eval_targets.push_back(ids_flat_batch.back());
            } else {
                const std::vector<array> topk_outputs = (*topk_fn)({logits_last});
                if (topk_outputs.size() != 2) {
                    g_last_error = "mlx_decode_topk_candidates_batch: invalid compiled output";
                    return 0;
                }

                logits_flat_batch.push_back(astype(topk_outputs[0], float32));
                ids_flat_batch.push_back(astype(topk_outputs[1], int32));
                eval_targets.push_back(logits_flat_batch.back());
                eval_targets.push_back(ids_flat_batch.back());
            }
        }
        if (!eval_targets.empty()) {
            eval(eval_targets);
            synchronize();
        }

        for (size_t row_idx = 0; row_idx < fallback_rows.size(); ++row_idx) {
            const int32_t i = fallback_rows[row_idx];
            if (k == 1) {
                out_candidate_ids_ptrs[i][0] = ids_flat_batch[row_idx].data<int32_t>()[0];
                out_candidate_logits_ptrs[i][0] = 0.0f;
                out_candidate_counts[i] = 1;
            } else {
                std::memcpy(
                    out_candidate_logits_ptrs[i],
                    logits_flat_batch[row_idx].data<float>(),
                    static_cast<size_t>(k) * sizeof(float)
                );
                std::memcpy(
                    out_candidate_ids_ptrs[i],
                    ids_flat_batch[row_idx].data<int32_t>(),
                    static_cast<size_t>(k) * sizeof(int32_t)
                );
                stable_sort_topk_pairs(out_candidate_logits_ptrs[i], out_candidate_ids_ptrs[i], k);
                out_candidate_counts[i] = k;
            }
            ctxs[i]->trace_decode_token += 1;
        }
        return 1;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 0;
    } catch (...) {
        g_last_error = "unknown error in mlx_decode_topk_candidates_batch";
        return 0;
    }
}

int32_t mlx_decode_topk_candidates_with_sampling(
    mlx_ctx* ctx,
    int32_t token,
    int32_t top_k,
    float repetition_penalty,
    float presence_penalty,
    float frequency_penalty,
    const int32_t* context_ids,
    int32_t context_len,
    float* out_candidate_logits,
    int32_t* out_candidate_ids,
    int32_t* out_candidate_count
) {
    if (!ctx || !out_candidate_logits || !out_candidate_ids || !out_candidate_count) {
        g_last_error = "mlx_decode_topk_candidates_with_sampling: invalid arguments";
        return 0;
    }
    if (!ctx->stream_ready) {
        g_last_error = "mlx_decode_topk_candidates_with_sampling: prefill not ready";
        return 0;
    }
    if (top_k <= 0) {
        g_last_error = "mlx_decode_topk_candidates_with_sampling: top_k must be > 0";
        return 0;
    }
    if (token < 0) {
        g_last_error = "mlx_decode_topk_candidates_with_sampling: token must be >= 0";
        return 0;
    }
    if (context_len < 0 || (context_len > 0 && context_ids == nullptr)) {
        g_last_error = "mlx_decode_topk_candidates_with_sampling: invalid context";
        return 0;
    }

    try {
        if (logits_sanity_enabled()) {
            std::fprintf(
                stderr,
                "[mlx][logits_sanity] enter=mlx_decode_topk_candidates_with_sampling token=%d top_k=%d context_len=%d\n",
                token,
                top_k,
                context_len
            );
        }
        const int32_t token_scalar = token;
        const array token_arr(&token_scalar, Shape{1, 1}, int32);
        const array logits_last = last_token_logits(forward_logits(ctx, token_arr)); // [1,1,V]
        emit_logits_sanity_rows("decode_topk_with_sampling_raw", logits_last, nullptr, 1);
        if (context_len == 0) {
            ctx->sampling_context_counts.clear();
            ctx->sampling_context_len = 0;
        } else if (context_len < ctx->sampling_context_len) {
            ctx->sampling_context_counts.clear();
            ctx->sampling_context_counts.reserve(static_cast<size_t>(context_len));
            for (int32_t i = 0; i < context_len; ++i) {
                ctx->sampling_context_counts[context_ids[i]] += 1;
            }
            ctx->sampling_context_len = context_len;
        } else if (context_len > ctx->sampling_context_len) {
            for (int32_t i = ctx->sampling_context_len; i < context_len; ++i) {
                ctx->sampling_context_counts[context_ids[i]] += 1;
            }
            ctx->sampling_context_len = context_len;
        }

        const array penalized_logits = apply_sampling_penalties_to_logits(
            ctx,
            logits_last,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
            ctx->sampling_context_counts
        );
        emit_logits_sanity_rows("decode_topk_with_sampling_penalized", penalized_logits, nullptr, 1);

        if (penalized_logits.ndim() != 3) {
            g_last_error = "mlx_decode_topk_candidates_with_sampling: invalid logits rank";
            return 0;
        }
        const int vocab = penalized_logits.shape(2);
        if (vocab <= 0) {
            g_last_error = "mlx_decode_topk_candidates_with_sampling: invalid vocab size";
            return 0;
        }

        const int k = std::min(top_k, vocab);
        auto topk_fn = compiled_topk_candidates(k);
        const std::vector<array> topk_outputs = (*topk_fn)({penalized_logits});
        if (topk_outputs.size() != 2) {
            g_last_error = "mlx_decode_topk_candidates_with_sampling: invalid compiled output";
            return 0;
        }
        const array top_k_logits_flat = astype(topk_outputs[0], float32);
        const array top_k_indices_flat = astype(topk_outputs[1], int32);

        eval({top_k_logits_flat, top_k_indices_flat});

        const float* logits_ptr = top_k_logits_flat.data<float>();
        const int32_t* ids_ptr = top_k_indices_flat.data<int32_t>();
        std::memcpy(out_candidate_logits, logits_ptr, static_cast<size_t>(k) * sizeof(float));
        std::memcpy(out_candidate_ids, ids_ptr, static_cast<size_t>(k) * sizeof(int32_t));
        stable_sort_topk_pairs(out_candidate_logits, out_candidate_ids, k);
        *out_candidate_count = k;
        return 1;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 0;
    } catch (...) {
        g_last_error = "unknown error in mlx_decode_topk_candidates_with_sampling";
        return 0;
    }
}

