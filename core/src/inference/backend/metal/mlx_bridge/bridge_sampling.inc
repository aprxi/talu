// bridge_sampling.inc — Sampling infrastructure: StreamSamplingConfig, TopKCandidateBatch,
// top-k extraction, filtering, temperature/top-p/min-p, compiled sampling steps.

using CompiledTopKFn = std::function<std::vector<array>(const std::vector<array>&)>;
using CompiledPenalizedSampleFn = std::function<std::vector<array>(const std::vector<array>&)>;
using CompiledSampleFn = std::function<std::vector<array>(const std::vector<array>&)>;
struct StreamSamplingConfig {
    float temperature = 1.0f;
    int top_k = 20;
    float top_p = 0.95f;
    float min_p = 0.0f;
};
struct TopKCandidateBatch {
    array logits; // [1,1,k]
    array indices; // [1,1,k]
    int k = 0;
};

TopKCandidateBatch select_topk_candidates_rank3(const array& logits_last, int requested_top_k) {
    if (logits_last.ndim() != 3) {
        throw std::runtime_error("select_topk_candidates_rank3 expects rank-3 logits");
    }

    const int batch = logits_last.shape(0);
    const int vocab = logits_last.shape(2);
    const int k = std::min(requested_top_k, vocab);
    if (k <= 0) {
        throw std::runtime_error("select_topk_candidates_rank3 invalid top_k");
    }

    array working_logits = astype(logits_last, float32);
    std::vector<array> logits_parts;
    std::vector<array> index_parts;
    logits_parts.reserve(static_cast<size_t>(k));
    index_parts.reserve(static_cast<size_t>(k));

    for (int rank = 0; rank < k; ++rank) {
        const array best_idx = reshape(astype(argmax(working_logits, -1), int32), {batch, 1, 1});
        const array best_logit = take_along_axis(working_logits, best_idx, -1);
        logits_parts.push_back(best_logit);
        index_parts.push_back(best_idx);

        const array neg_inf = full(best_idx.shape(), -std::numeric_limits<float>::infinity(), float32);
        working_logits = put_along_axis(working_logits, best_idx, neg_inf, -1);
    }

    const array topk_logits = k == 1 ? logits_parts.front() : concatenate(logits_parts, -1);
    const array topk_indices = k == 1 ? index_parts.front() : concatenate(index_parts, -1);
    return TopKCandidateBatch{
        .logits = topk_logits,
        .indices = astype(topk_indices, int32),
        .k = k,
    };
}

void stable_sort_topk_pairs(float* logits, int32_t* ids, int count) {
    if (!logits || !ids || count <= 1) return;
    std::vector<int> order(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) order[static_cast<size_t>(i)] = i;
    std::stable_sort(
        order.begin(),
        order.end(),
        [&](int a, int b) {
            const float la = logits[a];
            const float lb = logits[b];
            if (la > lb) return true;
            if (la < lb) return false;
            return ids[a] < ids[b];
        }
    );
    std::vector<float> logits_sorted(static_cast<size_t>(count));
    std::vector<int32_t> ids_sorted(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        const int src = order[static_cast<size_t>(i)];
        logits_sorted[static_cast<size_t>(i)] = logits[src];
        ids_sorted[static_cast<size_t>(i)] = ids[src];
    }
    std::memcpy(logits, logits_sorted.data(), static_cast<size_t>(count) * sizeof(float));
    std::memcpy(ids, ids_sorted.data(), static_cast<size_t>(count) * sizeof(int32_t));
}
array apply_top_p_filter(const array& logits, float top_p);
bool env_truthy(const char* name, bool fallback);

std::shared_ptr<const CompiledTopKFn> compiled_topk_candidates(int k) {
    static std::unordered_map<int, std::shared_ptr<const CompiledTopKFn>> cache;
    static std::mutex cache_mutex;
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache.find(k);
    if (it != cache.end()) return it->second;

    auto fn = compile(
        [k](const std::vector<array>& inputs) {
            const array logits_last = astype(inputs.at(0), float32); // [1,1,V]
            if (logits_last.ndim() != 3) {
                throw std::runtime_error("compiled_topk_candidates expects rank-3 logits");
            }
            if (k <= 0) {
                throw std::runtime_error("compiled_topk_candidates invalid k");
            }
            const int vocab = logits_last.shape(2);
            if (vocab < k) {
                throw std::runtime_error("compiled_topk_candidates k exceeds vocab");
            }
            const TopKCandidateBatch topk = select_topk_candidates_rank3(logits_last, k);
            const array top_k_logits_flat = reshape(topk.logits, {k});
            const array top_k_indices_flat = reshape(topk.indices, {k});
            return std::vector<array>{
                top_k_logits_flat,
                top_k_indices_flat,
            };
        },
        true
    );
    auto inserted = cache.emplace(
        k,
        std::make_shared<const CompiledTopKFn>(std::move(fn))
    );
    return inserted.first->second;
}

std::shared_ptr<const CompiledTopKFn> compiled_topk_candidates_rows(int k) {
    static std::unordered_map<int, std::shared_ptr<const CompiledTopKFn>> cache;
    static std::mutex cache_mutex;
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache.find(k);
    if (it != cache.end()) return it->second;

    auto fn = compile(
        [k](const std::vector<array>& inputs) {
            const array logits_last = astype(inputs.at(0), float32); // [B,1,V]
            if (logits_last.ndim() != 3) {
                throw std::runtime_error("compiled_topk_candidates_rows expects rank-3 logits");
            }
            if (k <= 0) {
                throw std::runtime_error("compiled_topk_candidates_rows invalid k");
            }
            const int batch = logits_last.shape(0);
            const int vocab = logits_last.shape(2);
            if (vocab < k) {
                throw std::runtime_error("compiled_topk_candidates_rows k exceeds vocab");
            }
            const TopKCandidateBatch topk = select_topk_candidates_rank3(logits_last, k);
            const array top_k_logits_rows = reshape(topk.logits, {batch, k});
            const array top_k_indices_rows = reshape(topk.indices, {batch, k});
            return std::vector<array>{
                top_k_logits_rows,
                top_k_indices_rows,
            };
        },
        true
    );
    auto inserted = cache.emplace(
        k,
        std::make_shared<const CompiledTopKFn>(std::move(fn))
    );
    return inserted.first->second;
}

array apply_top_p_filter(const array& logits, float top_p) {
    if (logits.ndim() != 3) {
        throw std::runtime_error("apply_top_p_filter expects rank-3 logits");
    }
    if (!(top_p > 0.0f && top_p < 1.0f)) {
        return logits;
    }
    const int vocab = logits.shape(2);
    const array probs = exp(logits);
    const array sorted_indices = argsort(logits, -1); // ascending
    const array sorted_probs = take_along_axis(probs, sorted_indices, -1);
    const array cumulative_probs_sorted = cumsum(sorted_probs, -1);
    const array inverse_indices = put_along_axis(
        zeros_like(sorted_indices),
        sorted_indices,
        arange(vocab),
        -1
    );
    const array cumulative_probs = take_along_axis(cumulative_probs_sorted, inverse_indices, -1);
    const array neg_inf = array(-std::numeric_limits<float>::infinity(), logits.dtype());
    return where(cumulative_probs > (1.0f - top_p), logits, neg_inf);
}

TopKCandidateBatch extract_topk_candidates_from_last_logits(const array& logits_last, int requested_top_k) {
    if (logits_last.ndim() != 3) {
        throw std::runtime_error("extract_topk_candidates_from_last_logits expects rank-3 logits");
    }

    const int vocab = logits_last.shape(2);
    const int k = std::min(requested_top_k, vocab);
    if (k <= 0) {
        throw std::runtime_error("extract_topk_candidates_from_last_logits invalid top_k");
    }

    // This is the exact insertion point for a future fused lm_head+top_k path.
    // Today it still uses the existing compiled full-logits top-k extraction so
    // decode semantics remain unchanged.
    auto topk_fn = compiled_topk_candidates(k);
    const std::vector<array> topk_outputs = (*topk_fn)({logits_last});
    if (topk_outputs.size() != 2) {
        throw std::runtime_error("extract_topk_candidates_from_last_logits: invalid top-k output");
    }

    return TopKCandidateBatch{
        .logits = reshape(topk_outputs[0], {1, 1, k}),
        .indices = reshape(topk_outputs[1], {1, 1, k}),
        .k = k,
    };
}

array sample_token_from_topk_candidates(const TopKCandidateBatch& candidates, const StreamSamplingConfig& cfg) {
    if (candidates.k <= 0) {
        throw std::runtime_error("sample_token_from_topk_candidates invalid top_k");
    }
    if (candidates.logits.ndim() != 3 || candidates.indices.ndim() != 3) {
        throw std::runtime_error("sample_token_from_topk_candidates expects rank-3 candidates");
    }

    array sampled_logits = candidates.logits;
    if (cfg.top_p > 0.0f && cfg.top_p < 1.0f) {
        sampled_logits = apply_top_p_filter(sampled_logits, cfg.top_p);
    }

    if (std::fabs(cfg.temperature - 1.0f) > 1.0e-6f) {
        sampled_logits = sampled_logits * array(1.0f / cfg.temperature, sampled_logits.dtype());
    }
    const array sampled_local_idx = random::categorical(sampled_logits, -1); // [1,1]
    const array sampled_local_idx_3d = reshape(sampled_local_idx, {1, 1, 1});
    const array sampled_token_3d = take_along_axis(candidates.indices, sampled_local_idx_3d, -1); // [1,1,1]
    return reshape(sampled_token_3d, {1, 1});
}

array sample_token_from_logits(const array& logits_last, const StreamSamplingConfig& cfg) {
    if (logits_last.ndim() != 3) {
        throw std::runtime_error("sample_token_from_logits expects rank-3 logits");
    }
    if (cfg.temperature <= 0.0f || cfg.top_k <= 1) {
        return argmax(logits_last, -1);
    }
    if (cfg.min_p != 0.0f) {
        throw std::runtime_error("min_p != 0 is not yet supported by decode_topk_stream");
    }
    return sample_token_from_topk_candidates(
        extract_topk_candidates_from_last_logits(logits_last, cfg.top_k),
        cfg
    );
}

std::shared_ptr<const CompiledSampleFn> compiled_sampling_step(const StreamSamplingConfig& cfg) {
    auto format_float = [](float value) -> std::string {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.8g", value);
        return std::string(buf);
    };
    const std::string signature =
        "k=" + std::to_string(cfg.top_k) +
        "|temp=" + format_float(cfg.temperature) +
        "|top_p=" + format_float(cfg.top_p);

    static std::unordered_map<std::string, std::shared_ptr<const CompiledSampleFn>> cache;
    static std::mutex cache_mutex;
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache.find(signature);
    if (it != cache.end()) return it->second;

    auto fn = compile(
        [cfg](const std::vector<array>& inputs) {
            if (inputs.size() != 1) {
                throw std::runtime_error("compiled_sampling_step expects logits input");
            }
            const array logits_last = astype(inputs.at(0), float32); // [1,1,V]
            if (logits_last.ndim() != 3) {
                throw std::runtime_error("compiled_sampling_step expects rank-3 logits");
            }

            array next_token = array(0.0f);
            const int vocab = logits_last.shape(2);
            const int k = std::min(cfg.top_k, vocab);
            if (cfg.temperature <= 0.0f || k <= 1) {
                next_token = argmax(logits_last, -1); // [1,1]
            } else {
                const TopKCandidateBatch topk = select_topk_candidates_rank3(logits_last, k);
                array sampled_logits = topk.logits; // [1,1,k]
                const array topk_indices = topk.indices; // [1,1,k]
                if (cfg.top_p > 0.0f && cfg.top_p < 1.0f) {
                    sampled_logits = apply_top_p_filter(sampled_logits, cfg.top_p);
                }
                if (std::fabs(cfg.temperature - 1.0f) > 1.0e-6f) {
                    sampled_logits = sampled_logits * array(1.0f / cfg.temperature, sampled_logits.dtype());
                }
                const array sampled_local_idx = random::categorical(sampled_logits, -1); // [1,1]
                const array sampled_local_idx_3d = reshape(sampled_local_idx, {1, 1, 1});
                const array sampled_token_3d = take_along_axis(topk_indices, sampled_local_idx_3d, -1); // [1,1,1]
                next_token = reshape(sampled_token_3d, {1, 1});
            }
            return std::vector<array>{astype(next_token, int32)};
        },
        true
    );

    auto inserted = cache.emplace(
        signature,
        std::make_shared<const CompiledSampleFn>(std::move(fn))
    );
    return inserted.first->second;
}

std::shared_ptr<const CompiledPenalizedSampleFn> compiled_penalized_sampling_step(
    const StreamSamplingConfig& cfg,
    float repetition_penalty,
    float presence_penalty,
    float frequency_penalty
) {
    if (cfg.min_p != 0.0f) {
        throw std::runtime_error("min_p != 0 is not yet supported by decode_topk_stream");
    }

    char key_buf[256];
    std::snprintf(
        key_buf,
        sizeof(key_buf),
        "k=%d|temp=%.8g|top_p=%.8g|rep=%.8g|pres=%.8g|freq=%.8g",
        cfg.top_k,
        cfg.temperature,
        cfg.top_p,
        repetition_penalty,
        presence_penalty,
        frequency_penalty
    );
    const std::string cache_key(key_buf);

    static std::unordered_map<std::string, std::shared_ptr<const CompiledPenalizedSampleFn>> cache;
    static std::mutex cache_mutex;
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache.find(cache_key);
    if (it != cache.end()) return it->second;

    auto fn = compile(
        [cfg, repetition_penalty, presence_penalty, frequency_penalty](const std::vector<array>& inputs) {
            if (inputs.size() < 2) {
                throw std::runtime_error("compiled_penalized_sampling_step expects logits and token counts");
            }
            const array& logits_last = inputs.at(0); // [1,1,V]
            const array& token_counts = inputs.at(1); // [1,1,V]
            if (logits_last.ndim() != 3 || token_counts.ndim() != 3) {
                throw std::runtime_error("compiled_penalized_sampling_step expects rank-3 inputs");
            }
            if (logits_last.shape(0) != 1 || logits_last.shape(1) != 1) {
                throw std::runtime_error("compiled_penalized_sampling_step expects logits shape [1,1,V]");
            }
            if (token_counts.shape(0) != 1 || token_counts.shape(1) != 1) {
                throw std::runtime_error("compiled_penalized_sampling_step expects counts shape [1,1,V]");
            }
            if (token_counts.shape(2) != logits_last.shape(2)) {
                throw std::runtime_error("compiled_penalized_sampling_step counts/logits vocab mismatch");
            }

            const bool has_repetition = repetition_penalty != 1.0f;
            const bool has_additive = presence_penalty != 0.0f || frequency_penalty != 0.0f;
            const array zero_f = array(0.0f, float32);
            array sampling_logits = logits_last;
            const array seen_mask = token_counts > zero_f;

            if (has_repetition) {
                const array rep_log = array(std::log(repetition_penalty), float32);
                const array repetition_scales = exp(token_counts * rep_log);
                const array repetition_adjusted = where(
                    logits_last > zero_f,
                    logits_last / repetition_scales,
                    logits_last * repetition_scales
                );
                sampling_logits = where(seen_mask, repetition_adjusted, sampling_logits);
            }

            if (has_additive) {
                array additive_penalties = zeros_like(token_counts);
                if (presence_penalty != 0.0f) {
                    additive_penalties = additive_penalties +
                        (astype(seen_mask, float32) * array(presence_penalty, float32));
                }
                if (frequency_penalty != 0.0f) {
                    additive_penalties = additive_penalties +
                        (token_counts * array(frequency_penalty, float32));
                }
                sampling_logits = sampling_logits - additive_penalties;
            }

            array next_token = array(0.0f);
            const int vocab = sampling_logits.shape(2);
            const int k = std::min(cfg.top_k, vocab);
            if (k <= 0) {
                throw std::runtime_error("compiled_penalized_sampling_step invalid top_k");
            }

            if (cfg.temperature <= 0.0f || k <= 1) {
                next_token = argmax(sampling_logits, -1); // [1,1]
            } else {
                const TopKCandidateBatch topk = select_topk_candidates_rank3(sampling_logits, k);
                array sampled_logits = topk.logits; // [1,1,k]
                const array topk_indices = topk.indices; // [1,1,k]

                if (cfg.top_p > 0.0f && cfg.top_p < 1.0f) {
                    sampled_logits = apply_top_p_filter(sampled_logits, cfg.top_p);
                }
                if (std::fabs(cfg.temperature - 1.0f) > 1.0e-6f) {
                    sampled_logits = sampled_logits * array(1.0f / cfg.temperature, sampled_logits.dtype());
                }
                const array sampled_local_idx = random::categorical(sampled_logits, -1); // [1,1]
                const array sampled_local_idx_3d = reshape(sampled_local_idx, {1, 1, 1});
                const array sampled_token_3d = take_along_axis(topk_indices, sampled_local_idx_3d, -1); // [1,1,1]
                next_token = reshape(sampled_token_3d, {1, 1});
            }

            const array token_idx = reshape(astype(next_token, int32), {1, 1, 1});
            const array prev_count = take_along_axis(token_counts, token_idx, -1);
            const array updated_counts = put_along_axis(
                token_counts,
                token_idx,
                prev_count + array(1.0f, float32),
                -1
            );
            return std::vector<array>{ next_token, updated_counts };
        },
        true
    );
    auto inserted = cache.emplace(
        cache_key,
        std::make_shared<const CompiledPenalizedSampleFn>(std::move(fn))
    );
    return inserted.first->second;
}
