// bridge_batch.inc — Batch fusion: topology keys, fusion helpers, fused batch
// decode/prefill dispatch.

array slice_batch_row(const array& src, int row) {
    if (src.ndim() <= 0) {
        throw std::runtime_error("slice_batch_row expects rank >= 1");
    }
    if (row < 0 || row >= src.shape(0)) {
        throw std::runtime_error("slice_batch_row row out of range");
    }
    Shape start(static_cast<size_t>(src.ndim()), 0);
    Shape end(static_cast<size_t>(src.ndim()), 0);
    for (int axis = 0; axis < src.ndim(); ++axis) {
        end[static_cast<size_t>(axis)] = src.shape(axis);
    }
    start[0] = row;
    end[0] = row + 1;
    return slice(src, start, end);
}

void scatter_fused_runtime_state_to_ctxs(const mlx_ctx& fused_ctx, mlx_ctx* const* ctxs, int32_t batch_size) {
    for (int32_t i = 0; i < batch_size; ++i) {
        mlx_ctx* ctx = ctxs[i];
        if (!ctx) continue;
        ctx->stream_ready = true;
    }

    const size_t layer_count = fused_ctx.kv_cache.size();
    for (size_t layer_idx = 0; layer_idx < layer_count; ++layer_idx) {
        const KVCacheState& fused_kv = fused_ctx.kv_cache[layer_idx];
        for (int32_t row = 0; row < batch_size; ++row) {
            mlx_ctx* dst = ctxs[row];
            if (!dst) continue;
            KVCacheState& dst_kv = dst->kv_cache[layer_idx];
            if (fused_kv.keys.has_value() && fused_kv.values.has_value()) {
                dst_kv.keys = slice_batch_row(*fused_kv.keys, row);
                dst_kv.values = slice_batch_row(*fused_kv.values, row);
                dst_kv.offset = fused_kv.offset;
            } else {
                dst_kv.keys.reset();
                dst_kv.values.reset();
                dst_kv.offset = 0;
            }
        }
    }

    const size_t linear_layer_count = fused_ctx.linear_cache.size();
    for (size_t layer_idx = 0; layer_idx < linear_layer_count; ++layer_idx) {
        const LinearCacheState& fused_lc = fused_ctx.linear_cache[layer_idx];
        for (int32_t row = 0; row < batch_size; ++row) {
            mlx_ctx* dst = ctxs[row];
            if (!dst) continue;
            LinearCacheState& dst_lc = dst->linear_cache[layer_idx];
            if (fused_lc.conv_state.has_value()) {
                dst_lc.conv_state = slice_batch_row(*fused_lc.conv_state, row);
            } else {
                dst_lc.conv_state.reset();
            }
            if (fused_lc.state.has_value()) {
                dst_lc.state = slice_batch_row(*fused_lc.state, row);
            } else {
                dst_lc.state.reset();
            }
        }
    }
}

void scatter_fused_prefill_runtime_state_to_ctxs(
    const mlx_ctx& fused_ctx,
    mlx_ctx* const* ctxs,
    const int32_t* prompt_lens,
    int32_t batch_size
) {
    for (int32_t i = 0; i < batch_size; ++i) {
        mlx_ctx* ctx = ctxs[i];
        if (!ctx) continue;
        ctx->stream_ready = true;
    }

    const size_t layer_count = fused_ctx.kv_cache.size();
    for (size_t layer_idx = 0; layer_idx < layer_count; ++layer_idx) {
        const KVCacheState& fused_kv = fused_ctx.kv_cache[layer_idx];
        for (int32_t row = 0; row < batch_size; ++row) {
            mlx_ctx* dst = ctxs[row];
            if (!dst) continue;
            const int32_t keep_len = prompt_lens ? prompt_lens[row] : 0;
            KVCacheState& dst_kv = dst->kv_cache[layer_idx];
            if (fused_kv.keys.has_value() && fused_kv.values.has_value()) {
                array row_keys = slice_batch_row(*fused_kv.keys, row);
                array row_values = slice_batch_row(*fused_kv.values, row);
                const int seq_len = row_keys.shape(2);
                const int clipped_len = std::max(0, std::min(keep_len, seq_len));
                if (clipped_len < seq_len) {
                    row_keys = slice(
                        row_keys,
                        {0, 0, 0, 0},
                        {1, row_keys.shape(1), clipped_len, row_keys.shape(3)}
                    );
                    row_values = slice(
                        row_values,
                        {0, 0, 0, 0},
                        {1, row_values.shape(1), clipped_len, row_values.shape(3)}
                    );
                }
                dst_kv.keys = row_keys;
                dst_kv.values = row_values;
                dst_kv.offset = clipped_len;
            } else {
                dst_kv.keys.reset();
                dst_kv.values.reset();
                dst_kv.offset = 0;
            }
        }
    }

    const size_t linear_layer_count = fused_ctx.linear_cache.size();
    for (size_t layer_idx = 0; layer_idx < linear_layer_count; ++layer_idx) {
        const LinearCacheState& fused_lc = fused_ctx.linear_cache[layer_idx];
        for (int32_t row = 0; row < batch_size; ++row) {
            mlx_ctx* dst = ctxs[row];
            if (!dst) continue;
            LinearCacheState& dst_lc = dst->linear_cache[layer_idx];
            if (fused_lc.conv_state.has_value()) {
                dst_lc.conv_state = slice_batch_row(*fused_lc.conv_state, row);
            } else {
                dst_lc.conv_state.reset();
            }
            if (fused_lc.state.has_value()) {
                dst_lc.state = slice_batch_row(*fused_lc.state, row);
            } else {
                dst_lc.state.reset();
            }
        }
    }
}

bool build_prefill_fusion_key(
    const mlx_ctx* ctx,
    int32_t prompt_len,
    int full_prompt_threshold,
    std::string* out_key
) {
    if (!ctx || !out_key) return false;
    if (ctx->xray_enabled) return false;
    // Models with per-layer input projections carry additional per-layer
    // hidden-state transforms that the current fixed-length fused prefill
    // route does not preserve correctly across row fusion.
    bool has_per_layer_input = ctx->cfg.hidden_size_per_layer_input > 0;
    if (!has_per_layer_input) {
        for (const LayerWeights& lw : ctx->layers) {
            if (lw.has_per_layer_input) {
                has_per_layer_input = true;
                break;
            }
        }
    }
    if (has_per_layer_input) return false;
    if (prompt_len <= 0 || prompt_len > full_prompt_threshold) return false;

    // Prefill starts from a reset runtime state; rows are compatible when they
    // share static model topology and prompt length.
    out_key->clear();
    out_key->append(std::to_string(prompt_len));
    out_key->push_back('|');
    out_key->append(std::to_string(ctx->cfg.num_hidden_layers));
    out_key->push_back('|');
    out_key->append(std::to_string(ctx->layers.size()));
    out_key->push_back('|');
    out_key->append(std::to_string(ctx->kv_cache.size()));
    out_key->push_back('|');
    out_key->append(std::to_string(ctx->linear_cache.size()));
    return true;
}

bool build_prefill_topology_key(const mlx_ctx* ctx, std::string* out_key) {
    if (!ctx || !out_key) return false;
    if (ctx->xray_enabled) return false;
    out_key->clear();
    out_key->append(std::to_string(ctx->cfg.num_hidden_layers));
    out_key->push_back('|');
    out_key->append(std::to_string(ctx->layers.size()));
    out_key->push_back('|');
    out_key->append(std::to_string(ctx->kv_cache.size()));
    out_key->push_back('|');
    out_key->append(std::to_string(ctx->linear_cache.size()));
    return true;
}

bool supports_variable_prefill_fusion(const mlx_ctx* ctx) {
    if (!ctx) return false;
    if (ctx->has_fp8_meta || ctx->has_grouped_affine_meta) return false;
    for (const LayerWeights& lw : ctx->layers) {
        // Variable-length fused prefill is only safe for pure full-attention
        // stacks where runtime state is fully represented by KV cache.
        if (lw.is_linear || lw.is_shortconv) return false;
    }
    return true;
}

bool supports_ragged_prefill_fusion(const mlx_ctx* ctx) {
    if (!ctx) return false;
    if (ctx->xray_enabled) return false;
    // Keep FP8 path conservative until ragged fused prefill is validated there.
    if (ctx->has_fp8_meta || ctx->has_grouped_affine_meta) return false;
    return true;
}

bool build_decode_fusion_key(const mlx_ctx* ctx, std::string* out_key) {
    if (!ctx || !out_key) return false;
    if (!ctx->stream_ready || ctx->xray_enabled) return false;
    if (ctx->cfg.hidden_size_per_layer_input > 0) return false;
    for (const LayerWeights& lw : ctx->layers) {
        if (lw.has_per_layer_input) return false;
    }

    out_key->clear();
    out_key->append(std::to_string(ctx->cfg.num_hidden_layers));
    out_key->push_back('|');
    out_key->append(std::to_string(ctx->kv_cache.size()));
    out_key->push_back('|');
    out_key->append(std::to_string(ctx->linear_cache.size()));
    out_key->push_back('|');

    // Decode fusion requires identical KV-cache occupancy/offset layout.
    for (const KVCacheState& kv : ctx->kv_cache) {
        const bool has_keys = kv.keys.has_value();
        const bool has_values = kv.values.has_value();
        if (has_keys != has_values) return false;
        if (!has_keys) {
            out_key->append("N;");
            continue;
        }
        out_key->push_back('K');
        out_key->append(std::to_string(kv.offset));
        out_key->push_back(':');
        out_key->append(std::to_string(kv.keys->shape(2)));
        out_key->push_back(';');
    }
    return true;
}

void flush_fused_decode_ctx_cache() {
    if (!g_fused_decode_ctx_cache.has_value()) return;
    if (!g_fused_decode_ctx_cache_dirty) return;
    if (g_fused_decode_ctx_cache_rows.empty()) return;
    scatter_fused_runtime_state_to_ctxs(
        g_fused_decode_ctx_cache.value(),
        g_fused_decode_ctx_cache_rows.data(),
        static_cast<int32_t>(g_fused_decode_ctx_cache_rows.size())
    );
    g_fused_decode_ctx_cache_dirty = false;
}

void clear_fused_decode_ctx_cache() {
    flush_fused_decode_ctx_cache();
    g_fused_decode_ctx_cache.reset();
    g_fused_decode_ctx_cache_rows.clear();
    g_fused_decode_ctx_cache_key.clear();
    g_fused_decode_ctx_cache_dirty = false;
}

bool same_ctx_row_set(const std::vector<mlx_ctx*>& lhs, const std::vector<mlx_ctx*>& rhs) {
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i] != rhs[i]) return false;
    }
    return true;
}

bool try_get_cached_fused_decode_ctx(
    const std::vector<mlx_ctx*>& rows,
    const std::string& key,
    mlx_ctx** out_ctx
) {
    if (!out_ctx) return false;
    *out_ctx = nullptr;
    if (!g_fused_decode_ctx_cache.has_value()) return false;
    if (g_fused_decode_ctx_cache_key != key) return false;
    if (!same_ctx_row_set(g_fused_decode_ctx_cache_rows, rows)) return false;
    *out_ctx = &g_fused_decode_ctx_cache.value();
    return true;
}

bool try_get_cached_fused_decode_ctx_by_rows(
    const std::vector<mlx_ctx*>& rows,
    mlx_ctx** out_ctx
) {
    if (!out_ctx) return false;
    *out_ctx = nullptr;
    if (!g_fused_decode_ctx_cache.has_value()) return false;
    if (!same_ctx_row_set(g_fused_decode_ctx_cache_rows, rows)) return false;
    if (g_fused_decode_ctx_cache_key.empty()) return false;
    for (mlx_ctx* row_ctx : rows) {
        if (!row_ctx) return false;
        std::string row_key;
        if (!build_decode_fusion_key(row_ctx, &row_key)) return false;
        if (row_key != g_fused_decode_ctx_cache_key) return false;
    }
    *out_ctx = &g_fused_decode_ctx_cache.value();
    return true;
}

void store_cached_fused_decode_ctx(
    const std::vector<mlx_ctx*>& rows,
    const std::string& key,
    const mlx_ctx& fused_ctx
) {
    g_fused_decode_ctx_cache = fused_ctx;
    g_fused_decode_ctx_cache_rows = rows;
    g_fused_decode_ctx_cache_key = key;
    g_fused_decode_ctx_cache_dirty = false;
}

bool prepare_fused_decode_ctx(
    mlx_ctx* const* ctxs,
    int32_t batch_size,
    mlx_ctx* fused_ctx,
    int32_t* out_common_offset
) {
    if (!ctxs || batch_size <= 1 || !fused_ctx || !out_common_offset) return false;
    mlx_ctx* first = ctxs[0];
    if (!first) return false;
    if (first->xray_enabled) return false;
    if (!first->stream_ready) return false;

    for (int32_t i = 1; i < batch_size; ++i) {
        mlx_ctx* ctx = ctxs[i];
        if (!ctx) return false;
        if (!ctx->stream_ready) return false;
        if (ctx->xray_enabled) return false;
        if (ctx->cfg.num_hidden_layers != first->cfg.num_hidden_layers) return false;
    }

    *fused_ctx = *first;
    fused_ctx->xray_enabled = false;
    fused_ctx->stream_ready = true;
    fused_ctx->sampling_context_counts.clear();
    fused_ctx->sampling_context_len = 0;
    fused_ctx->sampling_unique_ids.clear();
    fused_ctx->sampling_repetition_scales.clear();
    fused_ctx->sampling_additive_penalties.clear();

    int32_t common_offset = -1;
    for (size_t layer_idx = 0; layer_idx < fused_ctx->kv_cache.size(); ++layer_idx) {
        KVCacheState& fused_kv = fused_ctx->kv_cache[layer_idx];
        const KVCacheState& first_kv = ctxs[0]->kv_cache[layer_idx];
        const bool has_layer_kv = first_kv.keys.has_value() && first_kv.values.has_value();

        if (!has_layer_kv) {
            for (int32_t row = 0; row < batch_size; ++row) {
                const KVCacheState& row_kv = ctxs[row]->kv_cache[layer_idx];
                const bool row_has_kv = row_kv.keys.has_value() && row_kv.values.has_value();
                if (row_has_kv) return false;
            }
            fused_kv.keys.reset();
            fused_kv.values.reset();
            fused_kv.offset = 0;
            continue;
        }

        std::vector<array> keys_rows;
        std::vector<array> values_rows;
        keys_rows.reserve(static_cast<size_t>(batch_size));
        values_rows.reserve(static_cast<size_t>(batch_size));

        int layer_offset = -1;
        for (int32_t row = 0; row < batch_size; ++row) {
            const KVCacheState& row_kv = ctxs[row]->kv_cache[layer_idx];
            if (!row_kv.keys.has_value() || !row_kv.values.has_value()) return false;
            if (layer_offset < 0) {
                layer_offset = row_kv.offset;
            } else if (row_kv.offset != layer_offset) {
                return false;
            }
            keys_rows.push_back(*row_kv.keys);
            values_rows.push_back(*row_kv.values);
        }
        if (common_offset < 0) {
            common_offset = layer_offset;
        } else if (layer_offset != common_offset) {
            return false;
        }

        fused_kv.keys = concatenate(keys_rows, 0);
        fused_kv.values = concatenate(values_rows, 0);
        fused_kv.offset = layer_offset;
    }

    for (size_t layer_idx = 0; layer_idx < fused_ctx->linear_cache.size(); ++layer_idx) {
        std::vector<array> conv_rows;
        std::vector<array> state_rows;

        const LinearCacheState& first_lc = ctxs[0]->linear_cache[layer_idx];
        const bool has_conv = first_lc.conv_state.has_value();
        const bool has_state = first_lc.state.has_value();
        if (has_conv) conv_rows.reserve(static_cast<size_t>(batch_size));
        if (has_state) state_rows.reserve(static_cast<size_t>(batch_size));

        for (int32_t row = 0; row < batch_size; ++row) {
            const LinearCacheState& row_lc = ctxs[row]->linear_cache[layer_idx];
            if (row_lc.conv_state.has_value() != has_conv) return false;
            if (row_lc.state.has_value() != has_state) return false;
            if (has_conv) conv_rows.push_back(*row_lc.conv_state);
            if (has_state) state_rows.push_back(*row_lc.state);
        }

        LinearCacheState& fused_lc = fused_ctx->linear_cache[layer_idx];
        if (has_conv) {
            fused_lc.conv_state = concatenate(conv_rows, 0);
        } else {
            fused_lc.conv_state.reset();
        }
        if (has_state) {
            fused_lc.state = concatenate(state_rows, 0);
        } else {
            fused_lc.state.reset();
        }
    }

    *out_common_offset = common_offset >= 0 ? common_offset : 0;
    return true;
}

bool run_ragged_prefill_group(
    const std::vector<int32_t>& group_rows,
    mlx_ctx* const* ctxs,
    const int32_t* const* prompt_ids_ptrs,
    const int32_t* prompt_lens,
    float* const* out_logits_ptrs,
    int32_t logits_len,
    std::vector<uint8_t>* prefill_done
) {
    if (!ctxs || !prompt_ids_ptrs || !prompt_lens || !out_logits_ptrs || !prefill_done) {
        return false;
    }
    if (group_rows.size() < 2) return false;

    std::vector<int32_t> active_rows = group_rows;
    std::stable_sort(
        active_rows.begin(),
        active_rows.end(),
        [&](int32_t lhs, int32_t rhs) {
            return prompt_lens[lhs] < prompt_lens[rhs];
        }
    );

    for (int32_t row_idx : active_rows) {
        reset_runtime_state(ctxs[row_idx]);
        ctxs[row_idx]->xray_enabled = false;
    }

    int32_t consumed_len = 0;
    while (!active_rows.empty()) {
        const int32_t stage_target_len = prompt_lens[active_rows[0]];
        if (stage_target_len <= consumed_len) {
            throw std::runtime_error("run_ragged_prefill_group: invalid stage length");
        }
        const int32_t stage_len = stage_target_len - consumed_len;
        const int32_t stage_batch = static_cast<int32_t>(active_rows.size());

        std::vector<mlx_ctx*> stage_ctxs(active_rows.size());
        for (size_t i = 0; i < active_rows.size(); ++i) {
            stage_ctxs[i] = ctxs[active_rows[i]];
        }

        array logits_rows = array(0.0f);
        if (stage_batch == 1) {
            const int32_t row_idx = active_rows[0];
            const int32_t* stage_tokens = prompt_ids_ptrs[row_idx] + consumed_len;
            const array stage_arr(stage_tokens, Shape{1, stage_len}, int32);
            const array full_logits = forward_logits(ctxs[row_idx], stage_arr);
            if (full_logits.ndim() != 3 || full_logits.shape(2) != logits_len) {
                throw std::runtime_error("run_ragged_prefill_group: logits shape mismatch");
            }
            logits_rows = reshape(astype(last_token_logits(full_logits), float32), {1, logits_len});
            eval(logits_rows);
            synchronize();
            ctxs[row_idx]->stream_ready = true;
        } else {
            mlx_ctx fused_ctx;
            if (consumed_len == 0) {
                fused_ctx = *ctxs[active_rows[0]];
                reset_runtime_state(&fused_ctx);
                fused_ctx.xray_enabled = false;
            } else {
                int32_t common_offset = 0;
                const bool prepared = prepare_fused_decode_ctx(
                    stage_ctxs.data(),
                    stage_batch,
                    &fused_ctx,
                    &common_offset
                );
                if (!prepared || common_offset != consumed_len) {
                    return false;
                }
            }

            std::vector<int32_t> stage_tokens(
                static_cast<size_t>(stage_batch) * static_cast<size_t>(stage_len)
            );
            for (int32_t row = 0; row < stage_batch; ++row) {
                const int32_t src_row = active_rows[static_cast<size_t>(row)];
                const int32_t* src = prompt_ids_ptrs[src_row] + consumed_len;
                int32_t* dst = stage_tokens.data() +
                    static_cast<size_t>(row) * static_cast<size_t>(stage_len);
                std::memcpy(dst, src, static_cast<size_t>(stage_len) * sizeof(int32_t));
            }

            const array stage_arr(stage_tokens.data(), Shape{stage_batch, stage_len}, int32);
            const array full_logits = forward_logits(&fused_ctx, stage_arr);
            if (full_logits.ndim() != 3 || full_logits.shape(2) != logits_len) {
                throw std::runtime_error("run_ragged_prefill_group: fused logits shape mismatch");
            }
            logits_rows = reshape(astype(last_token_logits(full_logits), float32), {stage_batch, logits_len});
            eval(logits_rows);
            synchronize();
            scatter_fused_runtime_state_to_ctxs(fused_ctx, stage_ctxs.data(), stage_batch);
        }

        const float* rows_ptr = logits_rows.data<float>();
        const size_t row_bytes = static_cast<size_t>(logits_len) * sizeof(float);
        std::vector<int32_t> next_active;
        next_active.reserve(active_rows.size());
        for (size_t row = 0; row < active_rows.size(); ++row) {
            const int32_t src_row = active_rows[row];
            if (prompt_lens[src_row] == stage_target_len) {
                const float* row_logits = rows_ptr + row * static_cast<size_t>(logits_len);
                if (logits_sanity_enabled()) {
                    emit_logits_sanity("prefill_batch", src_row, row_logits, logits_len);
                }
                std::memcpy(
                    out_logits_ptrs[src_row],
                    row_logits,
                    row_bytes
                );
                ctxs[src_row]->trace_decode_token = 1;
                (*prefill_done)[static_cast<size_t>(src_row)] = 1;
            } else {
                next_active.push_back(src_row);
            }
        }
        active_rows.swap(next_active);
        consumed_len = stage_target_len;
    }

    return true;
}
