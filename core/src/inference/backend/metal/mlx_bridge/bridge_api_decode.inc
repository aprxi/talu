// bridge_api_decode.inc — Decode operations: mlx_decode_logits, mlx_decode_logits_batch,
// mlx_decode_stream, apply_sampling_penalties (C++ ext), mlx_decode_topk_candidates.

int32_t mlx_decode_logits(
    mlx_ctx* ctx,
    int32_t token,
    float* out_logits,
    int32_t logits_len
) {
    if (!ctx || !out_logits || logits_len <= 0) {
        g_last_error = "mlx_decode_logits: invalid arguments";
        return 0;
    }
    if (!ctx->stream_ready) {
        g_last_error = "mlx_decode_logits: prefill not ready";
        return 0;
    }
    if (token < 0) {
        g_last_error = "mlx_decode_logits: token must be >= 0";
        return 0;
    }

    try {
        clear_fused_decode_ctx_cache();
        const int32_t token_scalar = token;
        const array token_arr(&token_scalar, Shape{1, 1}, int32);
        const bool decode_trace_enabled = ctx->xray_enabled &&
            (xray_should_emit(XRAY_POINT_LAYER_INPUT, 0, 1) ||
                xray_should_emit(XRAY_POINT_ATTN_OUT, 0, 1) ||
                xray_should_emit(XRAY_POINT_BLOCK_OUT, 0, 1) ||
                xray_should_emit(XRAY_POINT_FINAL_NORM, XRAY_GLOBAL_LAYER, 1) ||
                xray_should_emit(XRAY_POINT_LM_HEAD, XRAY_GLOBAL_LAYER, 0));
        const TraceFrame trace_frame{
            .token = ctx->trace_decode_token,
            .layer_position = 1,
            .emit_embed = false,
        };
        const array logits = decode_trace_enabled
            ? forward_logits(ctx, token_arr, &trace_frame)
            : forward_logits(ctx, token_arr);
        const array next_logits = last_token_logits(logits);
        const int flat_size = static_cast<int>(next_logits.size());
        const array next_logits_flat = reshape(astype(next_logits, float32), {flat_size});
        if (static_cast<int32_t>(next_logits_flat.size()) != logits_len) {
            g_last_error = "mlx_decode_logits: logits length mismatch";
            return 0;
        }

        eval(next_logits_flat);
        synchronize();
        std::memcpy(out_logits, next_logits_flat.data<float>(), static_cast<size_t>(logits_len) * sizeof(float));
        ctx->trace_decode_token += 1;
        return 1;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 0;
    } catch (...) {
        g_last_error = "unknown error in mlx_decode_logits";
        return 0;
    }
}

int32_t mlx_decode_logits_batch(
    mlx_ctx* const* ctxs,
    const int32_t* tokens,
    float* const* out_logits_ptrs,
    int32_t batch_size,
    int32_t logits_len
) {
    if (!ctxs || !tokens || !out_logits_ptrs || batch_size <= 0 || logits_len <= 0) {
        g_last_error = "mlx_decode_logits_batch: invalid arguments";
        return 0;
    }

    if (batch_size == 1) {
        return mlx_decode_logits(ctxs[0], tokens[0], out_logits_ptrs[0], logits_len);
    }

    for (int32_t i = 0; i < batch_size; ++i) {
        if (!ctxs[i] || !out_logits_ptrs[i]) {
            g_last_error = "mlx_decode_logits_batch: null ctx or logits pointer";
            return 0;
        }
        if (!ctxs[i]->stream_ready) {
            g_last_error = "mlx_decode_logits_batch: prefill not ready";
            return 0;
        }
        if (tokens[i] < 0) {
            g_last_error = "mlx_decode_logits_batch: token must be >= 0";
            return 0;
        }
    }

    try {
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
            const array logits_rows = reshape(astype(logits_last, float32), {batch_size, logits_len});
            eval(logits_rows);
            synchronize();

            const float* rows_ptr = logits_rows.data<float>();
            const size_t row_bytes = static_cast<size_t>(logits_len) * sizeof(float);
            for (int32_t i = 0; i < batch_size; ++i) {
                std::memcpy(
                    out_logits_ptrs[i],
                    rows_ptr + static_cast<size_t>(i) * static_cast<size_t>(logits_len),
                    row_bytes
                );
                ctxs[i]->trace_decode_token += 1;
            }
            g_fused_decode_ctx_cache_dirty = true;
            return 1;
        }

        // Fast path: if the entire batch is fuse-compatible, run one fused
        // decode directly and skip per-token grouping/key construction.
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
            const array logits_rows = reshape(astype(logits_last, float32), {batch_size, logits_len});
            eval(logits_rows);
            synchronize();

            const float* rows_ptr = logits_rows.data<float>();
            const size_t row_bytes = static_cast<size_t>(logits_len) * sizeof(float);
            for (int32_t i = 0; i < batch_size; ++i) {
                std::memcpy(
                    out_logits_ptrs[i],
                    rows_ptr + static_cast<size_t>(i) * static_cast<size_t>(logits_len),
                    row_bytes
                );
                ctxs[i]->trace_decode_token += 1;
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

        // Group rows by compatible decode runtime state, then fuse each group.
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
            const array logits_rows = reshape(astype(logits_last, float32), {group_size, logits_len});
            eval(logits_rows);
            synchronize();

            const float* rows_ptr = logits_rows.data<float>();
            const size_t row_bytes = static_cast<size_t>(logits_len) * sizeof(float);
            for (int32_t i = 0; i < group_size; ++i) {
                const int32_t row = rows[static_cast<size_t>(i)];
                std::memcpy(
                    out_logits_ptrs[row],
                    rows_ptr + static_cast<size_t>(i) * static_cast<size_t>(logits_len),
                    row_bytes
                );
                ctxs[row]->trace_decode_token += 1;
                decode_done[static_cast<size_t>(row)] = 1;
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

        auto& next_logits_flat_batch = g_decode_batch_logits_flat_scratch;
        next_logits_flat_batch.clear();
        if (next_logits_flat_batch.capacity() < fallback_rows.size()) {
            next_logits_flat_batch.reserve(fallback_rows.size());
        }

        for (int32_t i : fallback_rows) {
            mlx_ctx* ctx = ctxs[i];
            const int32_t token_scalar = tokens[i];
            const array token_arr(&token_scalar, Shape{1, 1}, int32);
            const bool decode_trace_enabled = ctx->xray_enabled &&
                (xray_should_emit(XRAY_POINT_LAYER_INPUT, 0, 1) ||
                    xray_should_emit(XRAY_POINT_ATTN_OUT, 0, 1) ||
                    xray_should_emit(XRAY_POINT_BLOCK_OUT, 0, 1) ||
                    xray_should_emit(XRAY_POINT_FINAL_NORM, XRAY_GLOBAL_LAYER, 1) ||
                    xray_should_emit(XRAY_POINT_LM_HEAD, XRAY_GLOBAL_LAYER, 0));
            const TraceFrame trace_frame{
                .token = ctx->trace_decode_token,
                .layer_position = 1,
                .emit_embed = false,
            };
            const array logits = decode_trace_enabled
                ? forward_logits(ctx, token_arr, &trace_frame)
                : forward_logits(ctx, token_arr);
            const array next_logits = last_token_logits(logits);
            const int flat_size = static_cast<int>(next_logits.size());
            const array next_logits_flat = reshape(astype(next_logits, float32), {flat_size});
            if (static_cast<int32_t>(next_logits_flat.size()) != logits_len) {
                g_last_error = "mlx_decode_logits_batch: logits length mismatch";
                return 0;
            }
            next_logits_flat_batch.push_back(next_logits_flat);
        }

        if (!next_logits_flat_batch.empty()) {
            eval(next_logits_flat_batch);
            synchronize();
        }
        const size_t bytes = static_cast<size_t>(logits_len) * sizeof(float);
        for (size_t row_idx = 0; row_idx < fallback_rows.size(); ++row_idx) {
            const int32_t i = fallback_rows[row_idx];
            std::memcpy(
                out_logits_ptrs[i],
                next_logits_flat_batch[row_idx].data<float>(),
                bytes
            );
            ctxs[i]->trace_decode_token += 1;
        }
        return 1;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 0;
    } catch (...) {
        g_last_error = "unknown error in mlx_decode_logits_batch";
        return 0;
    }
}

int32_t mlx_decode_stream(
    mlx_ctx* ctx,
    int32_t first_token,
    int32_t decode_tokens,
    const int32_t* eos_ids,
    int32_t eos_len,
    int32_t** out_generated_ids,
    int32_t* out_generated_len
) {
    if (!ctx || !out_generated_ids || !out_generated_len) {
        g_last_error = "mlx_decode_stream: invalid arguments";
        return 0;
    }
    if (!ctx->stream_ready) {
        g_last_error = "mlx_decode_stream: prefill not ready";
        return 0;
    }
    if (decode_tokens < 0) {
        g_last_error = "mlx_decode_stream: decode_tokens must be >= 0";
        return 0;
    }
    if (first_token < 0) {
        g_last_error = "mlx_decode_stream: first_token must be >= 0";
        return 0;
    }

    return mlx_decode_topk_stream(
        ctx,
        first_token,
        decode_tokens,
        eos_ids,
        eos_len,
        0.0f,
        1,
        1.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f,
        out_generated_ids,
        out_generated_len
    );
}

#ifdef __cplusplus
extern "C++" {
#endif

array apply_sampling_penalties_to_logits(
    mlx_ctx* ctx,
    const array& logits_last,
    float repetition_penalty,
    float presence_penalty,
    float frequency_penalty,
    const std::unordered_map<int32_t, int32_t>& token_counts
) {
    const bool has_repetition = repetition_penalty != 1.0f;
    const bool has_additive = presence_penalty != 0.0f || frequency_penalty != 0.0f;
    if (!has_repetition && !has_additive) return logits_last;
    if (token_counts.empty()) return logits_last;

    if (logits_last.ndim() != 3) {
        throw std::runtime_error("apply_sampling_penalties_to_logits: invalid logits rank");
    }
    const int vocab = logits_last.shape(2);
    if (vocab <= 0) {
        throw std::runtime_error("apply_sampling_penalties_to_logits: invalid vocab size");
    }

    auto& token_ids = ctx->sampling_unique_ids;
    auto& repetition_scales = ctx->sampling_repetition_scales;
    auto& additive_penalties = ctx->sampling_additive_penalties;
    token_ids.clear();
    if (has_repetition) repetition_scales.clear();
    if (has_additive) additive_penalties.clear();
    if (token_ids.capacity() < token_counts.size()) token_ids.reserve(token_counts.size());
    if (has_repetition && repetition_scales.capacity() < token_counts.size()) {
        repetition_scales.reserve(token_counts.size());
    }
    if (has_additive && additive_penalties.capacity() < token_counts.size()) {
        additive_penalties.reserve(token_counts.size());
    }

    for (const auto& entry : token_counts) {
        const int32_t token_id = entry.first;
        const int32_t count = entry.second;
        if (token_id < 0 || token_id >= vocab) {
            throw std::runtime_error("apply_sampling_penalties_to_logits: context token out of range");
        }
        token_ids.push_back(token_id);
        if (has_repetition) {
            repetition_scales.push_back(std::pow(repetition_penalty, static_cast<float>(count)));
        }
        if (has_additive) {
            additive_penalties.push_back(
                presence_penalty + (frequency_penalty * static_cast<float>(count))
            );
        }
    }

    const int unique_count = static_cast<int>(token_ids.size());
    if (unique_count == 0) return logits_last;

    const array token_ids_arr(token_ids.begin(), Shape{1, 1, unique_count}, int32);
    array selected_logits = take_along_axis(logits_last, token_ids_arr, -1);

    if (has_repetition) {
        const array repetition_scales_arr(
            repetition_scales.begin(),
            Shape{1, 1, unique_count},
            float32
        );
        selected_logits = where(
            selected_logits > 0.0f,
            selected_logits / repetition_scales_arr,
            selected_logits * repetition_scales_arr
        );
    }

    if (has_additive) {
        const array additive_penalties_arr(
            additive_penalties.begin(),
            Shape{1, 1, unique_count},
            float32
        );
        selected_logits = selected_logits - additive_penalties_arr;
    }

    return put_along_axis(logits_last, token_ids_arr, selected_logits, -1);
}

#ifdef __cplusplus
}
#endif

int32_t mlx_decode_topk_candidates(
    mlx_ctx* ctx,
    int32_t token,
    int32_t top_k,
    float* out_candidate_logits,
    int32_t* out_candidate_ids,
    int32_t* out_candidate_count
) {
    if (!ctx || !out_candidate_logits || !out_candidate_ids || !out_candidate_count) {
        g_last_error = "mlx_decode_topk_candidates: invalid arguments";
        return 0;
    }
    if (!ctx->stream_ready) {
        g_last_error = "mlx_decode_topk_candidates: prefill not ready";
        return 0;
    }
    if (top_k <= 0) {
        g_last_error = "mlx_decode_topk_candidates: top_k must be > 0";
        return 0;
    }
    if (token < 0) {
        g_last_error = "mlx_decode_topk_candidates: token must be >= 0";
        return 0;
    }

    try {
        const int32_t token_scalar = token;
        const array token_arr(&token_scalar, Shape{1, 1}, int32);
        TopKCandidateBatch topk{
            .logits = array(0.0f),
            .indices = array(0.0f),
            .k = 0,
        };
        int k = top_k;

        if (can_use_fused_lm_head_topk(ctx, k)) {
            auto [candidate_logits_flat, candidate_ids_flat] =
                fused_lm_head_topk_candidate_arrays(ctx, token_arr, k);
            eval({candidate_logits_flat, candidate_ids_flat});
            synchronize();

            const int candidate_count = candidate_logits_flat.shape(0);
            const array candidate_logits_f32 = astype(candidate_logits_flat, float32);
            const array candidate_ids_u32 = astype(candidate_ids_flat, uint32);
            eval({candidate_logits_f32, candidate_ids_u32});
            synchronize();
            const float* candidate_logits_ptr = candidate_logits_f32.data<float>();
            const uint32_t* candidate_ids_ptr = candidate_ids_u32.data<uint32_t>();
            if (!candidate_logits_ptr || !candidate_ids_ptr) {
                g_last_error = "mlx_decode_topk_candidates: fused candidate materialization failed";
                return 0;
            }

            std::fill_n(out_candidate_logits, static_cast<size_t>(k), -std::numeric_limits<float>::infinity());
            std::fill_n(out_candidate_ids, static_cast<size_t>(k), -1);
            int filled = 0;
            for (int idx = 0; idx < candidate_count; ++idx) {
                const uint32_t candidate_id_u32 = candidate_ids_ptr[idx];
                if (candidate_id_u32 == 0xFFFFFFFFu) continue;
                if (candidate_id_u32 > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
                    continue;
                }
                const int32_t candidate_id = static_cast<int32_t>(candidate_id_u32);
                const float candidate_logit = candidate_logits_ptr[idx];

                int insert_at = filled < k ? filled : k;
                for (int rank = 0; rank < filled; ++rank) {
                    const float existing_logit = out_candidate_logits[rank];
                    const int32_t existing_id = out_candidate_ids[rank];
                    if (candidate_logit > existing_logit ||
                        (candidate_logit == existing_logit && candidate_id < existing_id)) {
                        insert_at = rank;
                        break;
                    }
                }
                if (insert_at >= k) continue;

                const int upper = filled < k ? filled : (k - 1);
                for (int shift = upper; shift > insert_at; --shift) {
                    out_candidate_logits[shift] = out_candidate_logits[shift - 1];
                    out_candidate_ids[shift] = out_candidate_ids[shift - 1];
                }
                out_candidate_logits[insert_at] = candidate_logit;
                out_candidate_ids[insert_at] = candidate_id;
                if (filled < k) ++filled;
            }
            *out_candidate_count = filled;
            return 1;
        } else {
            const array logits_last = last_token_logits(forward_logits(ctx, token_arr)); // [1,1,V]

            if (logits_last.ndim() != 3) {
                g_last_error = "mlx_decode_topk_candidates: invalid logits rank";
                return 0;
            }
            const int vocab = logits_last.shape(2);
            if (vocab <= 0) {
                g_last_error = "mlx_decode_topk_candidates: invalid vocab size";
                return 0;
            }

            k = std::min(top_k, vocab);
            if (k == 1) {
                const array top_idx = reshape(astype(argmax(logits_last, -1), int32), {1});
                eval(top_idx);
                synchronize();
                out_candidate_ids[0] = top_idx.data<int32_t>()[0];
                out_candidate_logits[0] = 0.0f;
                *out_candidate_count = 1;
                return 1;
            }

            auto topk_fn = compiled_topk_candidates(k);
            const std::vector<array> topk_outputs = (*topk_fn)({logits_last});
            if (topk_outputs.size() != 2) {
                g_last_error = "mlx_decode_topk_candidates: invalid compiled output";
                return 0;
            }
            topk = TopKCandidateBatch{
                .logits = reshape(astype(topk_outputs[0], float32), {1, 1, k}),
                .indices = reshape(astype(topk_outputs[1], int32), {1, 1, k}),
                .k = k,
            };
            eval({topk.logits, topk.indices});
        }

        const array logits_flat = reshape(topk.logits, {k});
        const array ids_flat = reshape(topk.indices, {k});
        eval({logits_flat, ids_flat});
        synchronize();

        const float* logits_ptr = logits_flat.data<float>();
        const int32_t* ids_ptr = ids_flat.data<int32_t>();
        std::memcpy(out_candidate_logits, logits_ptr, static_cast<size_t>(k) * sizeof(float));
        std::memcpy(out_candidate_ids, ids_ptr, static_cast<size_t>(k) * sizeof(int32_t));
        stable_sort_topk_pairs(out_candidate_logits, out_candidate_ids, k);
        *out_candidate_count = k;
        return 1;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 0;
    } catch (...) {
        g_last_error = "unknown error in mlx_decode_topk_candidates";
        return 0;
    }
}
