#include "bridge.h"

#include "mlx/backend/metal/metal.h"
#include "mlx/compile.h"
#include "mlx/fast.h"
#include "mlx/io.h"
#include "mlx/mlx.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <tuple>
#include <functional>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace mlx::core;

namespace {

thread_local std::string g_last_error;
std::mutex g_wired_limit_mutex;
size_t g_wired_limit_refcount = 0;
size_t g_wired_limit_previous = 0;
bool g_wired_limit_active = false;
std::mutex g_rng_mutex;
size_t g_active_ctx_count = 0;
thread_local size_t g_fused_lm_head_argmax_hits = 0;
thread_local std::vector<array> g_decode_batch_logits_flat_scratch;
thread_local std::vector<array> g_prefill_batch_logits_flat_scratch;
thread_local std::vector<array> g_decode_topk_batch_logits_flat_scratch;
thread_local std::vector<array> g_decode_topk_batch_ids_flat_scratch;
size_t g_qmm_attempts = 0;
size_t g_qmm_success = 0;

void enter_rng_epoch(int32_t seed) {
    std::lock_guard<std::mutex> lock(g_rng_mutex);
    if (g_active_ctx_count == 0) {
        random::seed(static_cast<uint64_t>(seed));
    }
    g_active_ctx_count += 1;
}

void leave_rng_epoch() {
    std::lock_guard<std::mutex> lock(g_rng_mutex);
    if (g_active_ctx_count == 0) return;
    g_active_ctx_count -= 1;
}

bool acquire_wired_limit() {
    const auto& info = device_info(Device::gpu);
    auto it = info.find("max_recommended_working_set_size");
    if (it == info.end() || !std::holds_alternative<size_t>(it->second)) {
        return false;
    }
    const size_t target_wired = std::get<size_t>(it->second);

    std::lock_guard<std::mutex> lock(g_wired_limit_mutex);
    if (g_wired_limit_refcount == 0) {
        g_wired_limit_previous = set_wired_limit(target_wired);
        g_wired_limit_active = true;
    }
    g_wired_limit_refcount += 1;
    return true;
}

void release_wired_limit() {
    std::lock_guard<std::mutex> lock(g_wired_limit_mutex);
    if (g_wired_limit_refcount == 0) return;
    g_wired_limit_refcount -= 1;
    if (g_wired_limit_refcount == 0 && g_wired_limit_active) {
        set_wired_limit(g_wired_limit_previous);
        g_wired_limit_active = false;
        g_wired_limit_previous = 0;
    }
}

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
            array top_k_indices = array(0.0f);
            array top_k_logits = array(0.0f);
            if (k == 1) {
                const array top_idx = argmax(logits_last, -1); // [1,1]
                top_k_indices = reshape(astype(top_idx, int32), {1, 1, 1});
                top_k_logits = take_along_axis(logits_last, top_k_indices, -1); // [1,1,1]
            } else {
                // Match metal backend candidate extraction: partition on logits
                // directly and gather the final K bucket.
                const array partitioned_indices = argpartition(logits_last, -k, -1);
                const array top_k_selector = arange(vocab - k, vocab, 1, int32);
                top_k_indices = take(partitioned_indices, top_k_selector, -1);
                top_k_logits = take_along_axis(logits_last, top_k_indices, -1);
            }
            const array top_k_logits_flat = reshape(top_k_logits, {k});
            const array top_k_indices_flat = reshape(astype(top_k_indices, int32), {k});
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

            array top_k_indices = array(0.0f);
            array top_k_logits = array(0.0f);
            if (k == 1) {
                const array top_idx = argmax(logits_last, -1); // [B,1]
                top_k_indices = reshape(astype(top_idx, int32), {batch, 1, 1});
                top_k_logits = take_along_axis(logits_last, top_k_indices, -1); // [B,1,1]
            } else {
                const array partitioned_indices = argpartition(logits_last, -k, -1);
                const array top_k_selector = arange(vocab - k, vocab, 1, int32);
                top_k_indices = take(partitioned_indices, top_k_selector, -1); // [B,1,k]
                top_k_logits = take_along_axis(logits_last, top_k_indices, -1); // [B,1,k]
            }
            const array top_k_logits_rows = reshape(top_k_logits, {batch, k});
            const array top_k_indices_rows = reshape(astype(top_k_indices, int32), {batch, k});
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
                array top_k_indices = array(0.0f);
                array top_k_logits = array(0.0f);
                if (k == 1) {
                    const array top_idx = argmax(logits_last, -1); // [1,1]
                    top_k_indices = reshape(astype(top_idx, int32), {1, 1, 1});
                    top_k_logits = take_along_axis(logits_last, top_k_indices, -1); // [1,1,1]
                } else {
                    const array partitioned_indices = argpartition(logits_last, -k, -1);
                    const array top_k_selector = arange(vocab - k, vocab, 1, int32);
                    top_k_indices = take(partitioned_indices, top_k_selector, -1); // [1,1,k]
                    top_k_logits = take_along_axis(logits_last, top_k_indices, -1); // [1,1,k]
                }

                array sampled_logits = reshape(top_k_logits, {1, 1, k}); // [1,1,k]
                const array topk_indices = reshape(astype(top_k_indices, int32), {1, 1, k}); // [1,1,k]
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
                array top_k_indices = array(0.0f);
                array top_k_logits = array(0.0f);
                if (k == 1) {
                    const array top_idx = argmax(sampling_logits, -1); // [1,1]
                    top_k_indices = reshape(astype(top_idx, int32), {1, 1, 1});
                    top_k_logits = take_along_axis(sampling_logits, top_k_indices, -1); // [1,1,1]
                } else {
                    const array partitioned_indices = argpartition(sampling_logits, -k, -1);
                    const array top_k_selector = arange(vocab - k, vocab, 1, int32);
                    top_k_indices = take(partitioned_indices, top_k_selector, -1); // [1,1,k]
                    top_k_logits = take_along_axis(sampling_logits, top_k_indices, -1); // [1,1,k]
                }

                array sampled_logits = reshape(top_k_logits, {1, 1, k}); // [1,1,k]
                const array topk_indices = reshape(astype(top_k_indices, int32), {1, 1, k}); // [1,1,k]

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

struct Qwen35Config {
    int hidden_size = 0;
    int num_hidden_layers = 0;
    int intermediate_size = 0;

    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;
    int global_head_dim = 0;

    int linear_num_value_heads = 0;
    int linear_num_key_heads = 0;
    int linear_key_head_dim = 0;
    int linear_value_head_dim = 0;
    int linear_conv_kernel_dim = 0;

    int full_attention_interval = 1;
    int sliding_window = 0;
    int hidden_size_per_layer_input = 0;
    int num_kv_shared_layers = 0;
    std::vector<std::string> layer_types = {};

    float rms_norm_eps = 1.0e-6f;
    float rope_theta = 10000000.0f;
    float partial_rotary_factor = 1.0f;
    float rope_theta_full = 1000000.0f;
    float rope_theta_local = 10000.0f;
    float partial_rotary_full = 0.25f;
    float partial_rotary_local = 1.0f;
    float embedding_multiplier = 1.0f;
    float attention_multiplier = 0.0f;
    float final_logit_softcapping = 0.0f;
    int quant_bits = 0;
    int quant_group_size = 0;

    bool tie_word_embeddings = true;
    bool use_layer_q_norm_head_dim = false;
    bool use_gelu = false;
    bool use_v_norm = false;
    bool has_nested_rope_parameters = false;
};

float adjust_proportional_global_rope_theta(float base_theta, int rope_dim, int global_head_dim) {
    if (base_theta <= 0.0f || rope_dim <= 0 || global_head_dim <= 0 || global_head_dim < rope_dim) {
        return base_theta;
    }
    const float exponent =
        static_cast<float>(rope_dim) / static_cast<float>(global_head_dim);
    return std::pow(base_theta, exponent);
}

struct ParsedModelConfig {
    Qwen35Config cfg;
    bool allow_qwen_norm_shift = true;
};

struct KVCacheState {
    std::optional<array> keys;
    std::optional<array> values;
    int offset = 0;
    int step = 256;
};

struct LinearCacheState {
    std::optional<array> conv_state;
    std::optional<array> state;
};

struct LayerWeights {
    bool is_linear = false;
    bool is_shortconv = false;

    array input_layernorm_w = array(0.0f);
    array post_attention_layernorm_w = array(0.0f);
    array pre_feedforward_layernorm_w = array(0.0f);
    array post_feedforward_layernorm_w = array(0.0f);
    bool has_per_layer_input = false;
    array per_layer_model_projection_rhs = array(0.0f); // [hidden, hpl]
    array per_layer_input_gate_rhs = array(0.0f); // [hidden, hpl]
    array per_layer_projection_rhs = array(0.0f); // [hpl, hidden]
    array post_per_layer_input_norm_w = array(0.0f); // [hidden]
    float layer_scalar = 1.0f;
    int kv_shared_source_layer = -1;
    bool attn_is_sliding = false;

    array mlp_gate_rhs = array(0.0f);
    array mlp_up_rhs = array(0.0f);
    array mlp_down_rhs = array(0.0f);
    array mlp_gate_q_w = array(0.0f);
    array mlp_gate_q_scales = array(0.0f);
    array mlp_gate_q_biases = array(0.0f);
    array mlp_up_q_w = array(0.0f);
    array mlp_up_q_scales = array(0.0f);
    array mlp_up_q_biases = array(0.0f);
    array mlp_down_q_w = array(0.0f);
    array mlp_down_q_scales = array(0.0f);
    array mlp_down_q_biases = array(0.0f);
    bool mlp_gate_has_q = false;
    bool mlp_up_has_q = false;
    bool mlp_down_has_q = false;

    // Full-attention branch.
    array attn_q_rhs = array(0.0f);
    array attn_k_rhs = array(0.0f);
    array attn_v_rhs = array(0.0f);
    array attn_o_rhs = array(0.0f);
    array attn_q_q_w = array(0.0f);
    array attn_q_q_scales = array(0.0f);
    array attn_q_q_biases = array(0.0f);
    array attn_k_q_w = array(0.0f);
    array attn_k_q_scales = array(0.0f);
    array attn_k_q_biases = array(0.0f);
    array attn_v_q_w = array(0.0f);
    array attn_v_q_scales = array(0.0f);
    array attn_v_q_biases = array(0.0f);
    array attn_o_q_w = array(0.0f);
    array attn_o_q_scales = array(0.0f);
    array attn_o_q_biases = array(0.0f);
    bool attn_q_has_q = false;
    bool attn_k_has_q = false;
    bool attn_v_has_q = false;
    bool attn_o_has_q = false;
    array attn_q_norm_w = array(0.0f);
    array attn_k_norm_w = array(0.0f);
    int attn_num_heads = 0;
    int attn_num_kv_heads = 0;
    int attn_head_dim = 0;
    float attn_scale = 1.0f;
    float attn_rope_theta = 10000000.0f;
    float attn_partial_rotary_factor = 1.0f;

    // Linear-attention (gated-delta) branch.
    array lin_conv1d_w = array(0.0f); // [conv_dim, kernel, 1]
    array lin_in_proj_qkv_rhs = array(0.0f);
    array lin_in_proj_z_rhs = array(0.0f);
    array lin_in_proj_a_rhs = array(0.0f);
    array lin_in_proj_b_rhs = array(0.0f);
    array lin_in_proj_qkvz_rhs = array(0.0f);
    array lin_in_proj_ba_rhs = array(0.0f);
    array lin_in_proj_qkvz_q_w = array(0.0f);
    array lin_in_proj_qkvz_q_scales = array(0.0f);
    array lin_in_proj_qkvz_q_biases = array(0.0f);
    array lin_in_proj_ba_q_w = array(0.0f);
    array lin_in_proj_ba_q_scales = array(0.0f);
    array lin_in_proj_ba_q_biases = array(0.0f);
    bool lin_in_proj_qkvz_has_q = false;
    bool lin_in_proj_ba_has_q = false;
    array lin_A_log = array(0.0f);
    array lin_A_exp_f32 = array(0.0f);
    array lin_dt_bias = array(0.0f);
    array lin_norm_w = array(0.0f);
    array lin_out_proj_rhs = array(0.0f);
    array lin_out_proj_q_w = array(0.0f);
    array lin_out_proj_q_scales = array(0.0f);
    array lin_out_proj_q_biases = array(0.0f);
    bool lin_out_proj_has_q = false;

    int lin_num_v_heads = 0;
    int lin_num_k_heads = 0;
    int lin_head_k_dim = 0;
    int lin_head_v_dim = 0;
    int lin_key_dim = 0;
    int lin_value_dim = 0;
    int lin_conv_dim = 0;
    int lin_conv_kernel = 0;

    // ShortConv branch (LFM2 family).
    array sc_in_proj_rhs = array(0.0f);
    array sc_in_proj_q_w = array(0.0f);
    array sc_in_proj_q_scales = array(0.0f);
    array sc_in_proj_q_biases = array(0.0f);
    array sc_conv1d_w = array(0.0f); // [conv_dim, kernel, 1]
    array sc_conv_bias = array(0.0f);
    array sc_out_proj_rhs = array(0.0f);
    array sc_out_proj_q_w = array(0.0f);
    array sc_out_proj_q_scales = array(0.0f);
    array sc_out_proj_q_biases = array(0.0f);
    bool sc_in_proj_has_q = false;
    bool sc_out_proj_has_q = false;
    bool sc_has_bias = false;
    int sc_conv_dim = 0;
    int sc_conv_kernel = 0;
};

array next_token_greedy(const array& logits_last) {
    return astype(argmax(logits_last, -1), int32);
}

bool ends_with(const std::string& s, const std::string& suffix) {
    if (suffix.size() > s.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open file: " + path.string());
    }
    std::string text;
    in.seekg(0, std::ios::end);
    text.resize(static_cast<size_t>(in.tellg()));
    in.seekg(0, std::ios::beg);
    in.read(text.data(), static_cast<std::streamsize>(text.size()));
    if (!in.good() && !in.eof()) {
        throw std::runtime_error("failed to read file: " + path.string());
    }
    return text;
}

std::vector<std::string> discover_weight_files(const std::string& model_path) {
    namespace fs = std::filesystem;
    const fs::path model_dir(model_path);

    std::unordered_set<std::string> unique;

    const fs::path direct = model_dir / "model.safetensors";
    if (fs::exists(direct)) unique.insert("model.safetensors");

    const fs::path index_file = model_dir / "model.safetensors.index.json";
    if (fs::exists(index_file)) {
        const std::string json = read_file(index_file);
        static const std::regex re("\\\"([^\\\"]+\\.safetensors)\\\"");
        for (std::sregex_iterator it(json.begin(), json.end(), re), end; it != end; ++it) {
            const std::string rel = (*it)[1].str();
            const fs::path abs = model_dir / rel;
            if (fs::exists(abs)) unique.insert(rel);
        }
    }

    if (unique.empty()) {
        for (const auto& entry : fs::directory_iterator(model_dir)) {
            if (!entry.is_regular_file()) continue;
            const std::string name = entry.path().filename().string();
            if (name.size() >= 12 && name.substr(name.size() - 12) == ".safetensors") {
                unique.insert(name);
            }
        }
    }

    std::vector<std::string> files(unique.begin(), unique.end());
    std::sort(files.begin(), files.end());
    if (files.empty()) {
        throw std::runtime_error("no safetensors weights found in model directory: " + model_path);
    }
    return files;
}

std::unordered_map<std::string, array> load_weight_tensors(const std::string& model_path) {
    const std::vector<std::string> weight_files = discover_weight_files(model_path);

    std::unordered_map<std::string, array> tensors;
    for (const std::string& rel : weight_files) {
        const std::string abs = (std::filesystem::path(model_path) / rel).string();
        auto loaded = load_safetensors(abs);
        for (auto& kv : loaded.first) {
            auto it = tensors.find(kv.first);
            if (it == tensors.end()) {
                tensors.emplace(kv.first, std::move(kv.second));
            } else {
                it->second = std::move(kv.second);
            }
        }
    }

    if (tensors.empty()) throw std::runtime_error("no tensors were loaded from safetensors files");
    return tensors;
}

std::string extract_named_object(const std::string& json, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const size_t key_pos = json.find(needle);
    if (key_pos == std::string::npos) throw std::runtime_error("missing config object: " + key);

    const size_t open = json.find('{', key_pos);
    if (open == std::string::npos) throw std::runtime_error("invalid config object: " + key);

    int depth = 0;
    bool in_string = false;
    bool escape = false;
    for (size_t i = open; i < json.size(); ++i) {
        const char ch = json[i];
        if (in_string) {
            if (escape) {
                escape = false;
            } else if (ch == '\\') {
                escape = true;
            } else if (ch == '"') {
                in_string = false;
            }
            continue;
        }
        if (ch == '"') {
            in_string = true;
            continue;
        }
        if (ch == '{') depth++;
        if (ch == '}') {
            depth--;
            if (depth == 0) {
                return json.substr(open, i - open + 1);
            }
        }
    }

    throw std::runtime_error("unterminated config object: " + key);
}

bool find_bool_value(const std::string& text, const std::string& key, bool* out) {
    const std::regex re("\\\"" + key + "\\\"\\s*:\\s*(true|false)");
    std::smatch m;
    if (!std::regex_search(text, m, re)) return false;
    *out = (m[1].str() == "true");
    return true;
}

bool find_int_value(const std::string& text, const std::string& key, int* out) {
    const std::regex re("\\\"" + key + "\\\"\\s*:\\s*(-?[0-9]+)");
    std::smatch m;
    if (!std::regex_search(text, m, re)) return false;
    *out = std::stoi(m[1].str());
    return true;
}

bool find_float_value(const std::string& text, const std::string& key, float* out) {
    const std::regex re("\\\"" + key + "\\\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)");
    std::smatch m;
    if (!std::regex_search(text, m, re)) return false;
    *out = std::stof(m[1].str());
    return true;
}

bool find_string_value(const std::string& text, const std::string& key, std::string* out) {
    const std::regex re("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
    std::smatch m;
    if (!std::regex_search(text, m, re)) return false;
    *out = m[1].str();
    return true;
}

bool find_string_array_value(const std::string& text, const std::string& key, std::vector<std::string>* out) {
    // Match across newlines so pretty-printed JSON arrays are parsed too.
    const std::regex arr_re("\\\"" + key + "\\\"\\s*:\\s*\\[([\\s\\S]*?)\\]");
    std::smatch m;
    if (!std::regex_search(text, m, arr_re)) return false;
    const std::string body = m[1].str();
    const std::regex item_re("\\\"([^\\\"]*)\\\"");
    out->clear();
    for (auto it = std::sregex_iterator(body.begin(), body.end(), item_re);
         it != std::sregex_iterator();
         ++it)
    {
        out->push_back((*it)[1].str());
    }
    return true;
}

std::vector<int> resolve_shared_kv_source_layers(const Qwen35Config& cfg) {
    const int n_layers = cfg.num_hidden_layers;
    std::vector<int> sources(static_cast<size_t>(std::max(0, n_layers)), -1);
    if (cfg.num_kv_shared_layers <= 0 || n_layers <= 0) return sources;
    if (static_cast<int>(cfg.layer_types.size()) != n_layers) return sources;

    const int shared_count = std::min(cfg.num_kv_shared_layers, n_layers);
    if (shared_count <= 0 || shared_count >= n_layers) return sources;
    const int first_shared_layer = n_layers - shared_count;
    if (first_shared_layer <= 0) return sources;

    for (int layer_idx = first_shared_layer; layer_idx < n_layers; ++layer_idx) {
        const std::string& target_layer_type = cfg.layer_types[static_cast<size_t>(layer_idx)];
        for (int src = first_shared_layer - 1; src >= 0; --src) {
            if (cfg.layer_types[static_cast<size_t>(src)] == target_layer_type) {
                sources[static_cast<size_t>(layer_idx)] = src;
                break;
            }
        }
    }
    return sources;
}

ParsedModelConfig parse_qwen35_config(const std::string& model_path) {
    const std::string config_text = read_file(std::filesystem::path(model_path) / "config.json");
    std::string text_cfg;
    try {
        text_cfg = extract_named_object(config_text, "text_config");
    } catch (const std::runtime_error&) {
        // Qwen3 uses a flat config; Qwen3.5 uses nested text_config.
        text_cfg = config_text;
    }

    Qwen35Config cfg;
    ParsedModelConfig parsed;

    if (!find_int_value(text_cfg, "hidden_size", &cfg.hidden_size)) throw std::runtime_error("missing text_config.hidden_size");
    if (!find_int_value(text_cfg, "num_hidden_layers", &cfg.num_hidden_layers)) throw std::runtime_error("missing text_config.num_hidden_layers");
    if (!find_int_value(text_cfg, "intermediate_size", &cfg.intermediate_size)) {
        if (!find_int_value(text_cfg, "d_ff", &cfg.intermediate_size)) {
            if (!find_int_value(text_cfg, "block_ff_dim", &cfg.intermediate_size)) {
                throw std::runtime_error("missing text_config.intermediate_size/d_ff/block_ff_dim");
            }
        }
    }

    if (!find_int_value(text_cfg, "num_attention_heads", &cfg.num_attention_heads)) {
        if (!find_int_value(text_cfg, "num_heads", &cfg.num_attention_heads)) {
            throw std::runtime_error("missing text_config.num_attention_heads/num_heads");
        }
    }
    if (!find_int_value(text_cfg, "num_key_value_heads", &cfg.num_key_value_heads)) {
        cfg.num_key_value_heads = cfg.num_attention_heads;
    }

    if (!find_int_value(text_cfg, "head_dim", &cfg.head_dim)) {
        cfg.head_dim = cfg.hidden_size / std::max(1, cfg.num_attention_heads);
    }
    find_int_value(text_cfg, "global_head_dim", &cfg.global_head_dim);

    // Linear-attention fields are model-family specific (Qwen3.5 hybrid).
    // Keep defaults when absent so pure-attention families (Qwen3) load.
    find_int_value(text_cfg, "linear_num_value_heads", &cfg.linear_num_value_heads);
    find_int_value(text_cfg, "linear_num_key_heads", &cfg.linear_num_key_heads);
    find_int_value(text_cfg, "linear_key_head_dim", &cfg.linear_key_head_dim);
    find_int_value(text_cfg, "linear_value_head_dim", &cfg.linear_value_head_dim);
    find_int_value(text_cfg, "linear_conv_kernel_dim", &cfg.linear_conv_kernel_dim);

    find_int_value(text_cfg, "full_attention_interval", &cfg.full_attention_interval);
    find_int_value(text_cfg, "sliding_window", &cfg.sliding_window);
    find_int_value(text_cfg, "hidden_size_per_layer_input", &cfg.hidden_size_per_layer_input);
    find_int_value(text_cfg, "num_kv_shared_layers", &cfg.num_kv_shared_layers);
    find_string_array_value(text_cfg, "layer_types", &cfg.layer_types);
    if (!find_float_value(text_cfg, "rms_norm_eps", &cfg.rms_norm_eps)) {
        if (!find_float_value(text_cfg, "norm_eps", &cfg.rms_norm_eps)) {
            find_float_value(text_cfg, "block_norm_eps", &cfg.rms_norm_eps);
        }
    }

    // Qwen3.5 stores rope settings inside rope_parameters.
    find_float_value(text_cfg, "rope_theta", &cfg.rope_theta);
    find_float_value(text_cfg, "partial_rotary_factor", &cfg.partial_rotary_factor);
    try {
        const std::string rope_params = extract_named_object(text_cfg, "rope_parameters");
        bool has_nested = false;
        try {
            const std::string full_rope = extract_named_object(rope_params, "full_attention");
            if (find_float_value(full_rope, "rope_theta", &cfg.rope_theta_full)) has_nested = true;
            if (find_float_value(full_rope, "partial_rotary_factor", &cfg.partial_rotary_full)) has_nested = true;
        } catch (const std::runtime_error&) {
        }
        try {
            const std::string sliding_rope = extract_named_object(rope_params, "sliding_attention");
            if (find_float_value(sliding_rope, "rope_theta", &cfg.rope_theta_local)) has_nested = true;
            if (find_float_value(sliding_rope, "partial_rotary_factor", &cfg.partial_rotary_local)) has_nested = true;
        } catch (const std::runtime_error&) {
        }
        cfg.has_nested_rope_parameters = has_nested;
    } catch (const std::runtime_error&) {
    }
    find_float_value(text_cfg, "final_logit_softcapping", &cfg.final_logit_softcapping);
    std::string hidden_activation;
    if (!find_string_value(text_cfg, "hidden_activation", &hidden_activation)) {
        find_string_value(text_cfg, "hidden_act", &hidden_activation);
    }
    if (!hidden_activation.empty()) {
        cfg.use_gelu = hidden_activation.find("gelu") != std::string::npos;
    }

    bool tie_embeddings = true;
    if (find_bool_value(config_text, "tie_word_embeddings", &tie_embeddings)) {
        cfg.tie_word_embeddings = tie_embeddings;
    }
    try {
        const std::string quant_cfg = extract_named_object(config_text, "quantization");
        find_int_value(quant_cfg, "bits", &cfg.quant_bits);
        find_int_value(quant_cfg, "group_size", &cfg.quant_group_size);
    } catch (const std::runtime_error&) {
        // Non-quantized checkpoints have no quantization object.
    }

    const std::regex lfm2_re("\\\"model_type\\\"\\s*:\\s*\\\"lfm2(?:_vl|_5)?\\\"");
    const bool is_lfm2_family = std::regex_search(config_text, lfm2_re) || std::regex_search(text_cfg, lfm2_re);
    const std::regex gemma4_re("\\\"model_type\\\"\\s*:\\s*\\\"gemma4(?:_text)?\\\"");
    const bool is_gemma4_family = std::regex_search(config_text, gemma4_re) || std::regex_search(text_cfg, gemma4_re);
    cfg.use_layer_q_norm_head_dim = is_gemma4_family;
    if (is_gemma4_family) {
        if (cfg.num_hidden_layers > 0 && static_cast<int>(cfg.layer_types.size()) != cfg.num_hidden_layers) {
            throw std::runtime_error(
                "gemma4 config parse error: layer_types length mismatch (got=" +
                std::to_string(cfg.layer_types.size()) +
                ", expected=" + std::to_string(cfg.num_hidden_layers) + ")"
            );
        }
        cfg.embedding_multiplier = std::sqrt(static_cast<float>(cfg.hidden_size));
        cfg.attention_multiplier = 1.0f;
        cfg.use_v_norm = true;
    }
    parsed.cfg = cfg;
    parsed.allow_qwen_norm_shift = !is_lfm2_family && !is_gemma4_family;
    return parsed;
}

void sanitize_qwen35_tensors(
    std::unordered_map<std::string, array>& tensors,
    bool tie_word_embeddings,
    bool allow_qwen_norm_shift
) {
    bool has_mtp_weights = false;
    bool has_unsanitized_conv1d = false;

    std::unordered_map<std::string, array> filtered;
    filtered.reserve(tensors.size());

    for (auto& kv : tensors) {
        const std::string& k = kv.first;
        if (k.find("mtp.") != std::string::npos) {
            has_mtp_weights = true;
            continue;
        }
        if (k.rfind("vision_tower", 0) == 0 ||
            k.rfind("model.visual", 0) == 0 ||
            k.rfind("model.vision_tower", 0) == 0 ||
            k.rfind("model.multi_modal_projector", 0) == 0) {
            continue;
        }
        if ((k.find("conv1d.weight") != std::string::npos || k.find("conv.conv.weight") != std::string::npos) &&
            kv.second.ndim() == 3 && kv.second.shape(2) != 1) {
            has_unsanitized_conv1d = true;
        }
        filtered.emplace(k, kv.second);
    }

    if (tie_word_embeddings) {
        filtered.erase("lm_head.weight");
        filtered.erase("model.language_model.lm_head.weight");
        filtered.erase("model.lm_head.weight");
    }

    const bool should_shift_norm = allow_qwen_norm_shift && (has_mtp_weights || has_unsanitized_conv1d);

    const std::vector<std::string> norm_suffixes = {
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        "model.language_model.norm.weight",
        "model.norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    };

    for (auto& kv : filtered) {
        const std::string& key = kv.first;
        array v = kv.second;

        if ((key.find("conv1d.weight") != std::string::npos || key.find("conv.conv.weight") != std::string::npos) &&
            v.ndim() == 3 && v.shape(2) != 1) {
            // model.sanitize: v = v.moveaxis(2, 1)
            v = transpose(v, {0, 2, 1});
        }

        if (should_shift_norm && v.ndim() == 1) {
            bool match = false;
            for (const auto& sfx : norm_suffixes) {
                if (ends_with(key, sfx)) {
                    match = true;
                    break;
                }
            }
            if (match) v = v + array(1.0f, v.dtype());
        }

        kv.second = v;
    }

    tensors = std::move(filtered);
}

const array& require_tensor(const std::unordered_map<std::string, array>& tensors, const std::string& key) {
    auto it = tensors.find(key);
    if (it == tensors.end() && ends_with(key, ".weight")) {
        const std::string packed_key = key.substr(0, key.size() - std::string(".weight").size()) + ".weight_packed";
        it = tensors.find(packed_key);
    }
    if (it == tensors.end()) throw std::runtime_error("missing required tensor: " + key);
    return it->second;
}

const array& require_any_tensor(
    const std::unordered_map<std::string, array>& tensors,
    const std::vector<std::string>& keys,
    const std::string& context
) {
    for (const auto& key : keys) {
        auto it = tensors.find(key);
        if (it != tensors.end()) return it->second;
    }
    throw std::runtime_error("missing required tensor for " + context);
}

bool has_tensor(const std::unordered_map<std::string, array>& tensors, const std::string& key) {
    if (tensors.find(key) != tensors.end()) return true;
    if (ends_with(key, ".weight")) {
        const std::string packed_key = key.substr(0, key.size() - std::string(".weight").size()) + ".weight_packed";
        return tensors.find(packed_key) != tensors.end();
    }
    return false;
}

std::string detect_text_prefix(const std::unordered_map<std::string, array>& tensors) {
    if (has_tensor(tensors, "model.language_model.embed_tokens.weight")) {
        return "model.language_model.";
    }
    if (has_tensor(tensors, "model.embed_tokens.weight")) {
        return "model.";
    }

    // Fallback for converted checkpoints that wrap text tensors under repeated
    // namespaces (for example model.language_model.language_model.*).
    const std::string embed_suffix = "embed_tokens.weight";
    std::string best_prefix;
    size_t embed_candidate_count = 0;
    std::string first_embed_key;
    for (const auto& kv : tensors) {
        const std::string& key = kv.first;
        if (!ends_with(key, embed_suffix)) continue;
        embed_candidate_count += 1;
        if (first_embed_key.empty()) first_embed_key = key;
        const std::string prefix = key.substr(0, key.size() - embed_suffix.size());
        if (prefix.empty()) continue;
        if (!has_tensor(tensors, prefix + "layers.0.input_layernorm.weight")) continue;
        if (prefix.size() > best_prefix.size()) {
            best_prefix = prefix;
        }
    }
    if (!best_prefix.empty()) {
        return best_prefix;
    }

    throw std::runtime_error(
        "missing embed_tokens.weight for known text prefixes"
        " (embed_candidates=" + std::to_string(embed_candidate_count) +
        (first_embed_key.empty() ? "" : ", first_embed_key=" + first_embed_key) +
        ")"
    );
}

array to_rhs(const array& weight_2d, const std::string& name) {
    if (weight_2d.ndim() != 2) {
        throw std::runtime_error("expected rank-2 weight for " + name);
    }
    // MLX-LM nn.Linear uses [out, in] weights. Convert once to [in, out] RHS.
    // Materialize a contiguous RHS tensor once at load-time so decode matmuls
    // do not pay strided-view penalties on every token step.
    return stop_gradient(copy(transpose(weight_2d, {1, 0})));
}

array expand_block_scales(const array& scales_2d_f32, int rows, int cols, const std::string& name) {
    if (scales_2d_f32.ndim() != 2) {
        throw std::runtime_error("expected rank-2 FP8 scale tensor for " + name);
    }
    const int scale_rows = scales_2d_f32.shape(0);
    const int scale_cols = scales_2d_f32.shape(1);
    if (scale_rows <= 0 || scale_cols <= 0) {
        throw std::runtime_error("invalid FP8 scale shape for " + name);
    }

    const int block_row = (rows + scale_rows - 1) / scale_rows;
    const int block_col = (cols + scale_cols - 1) / scale_cols;
    array expanded = repeat(scales_2d_f32, block_row, 0);
    expanded = repeat(expanded, block_col, 1);
    return slice(expanded, {0, 0}, {rows, cols});
}

array maybe_dequantize_fp8_weight(
    const std::unordered_map<std::string, array>& tensors,
    const std::string& weight_key,
    const array& weight_2d
) {
    if (!ends_with(weight_key, ".weight")) {
        return weight_2d;
    }
    if (weight_2d.ndim() != 2) {
        throw std::runtime_error("expected rank-2 weight for " + weight_key);
    }

    const std::string base = weight_key.substr(0, weight_key.size() - std::string(".weight").size());
    const std::string scale_inv_key = base + ".weight_scale_inv";
    const std::string block_scale_key = base + ".weight_block_scale";
    const std::string weight_scale_key = base + ".weight_scale";
    const bool has_nvfp4_packed = tensors.find(base + ".weight_packed") != tensors.end();
    const bool has_grouped_affine_scales = tensors.find(base + ".scales") != tensors.end();
    const bool has_grouped_affine_biases =
        tensors.find(base + ".biases") != tensors.end() ||
        tensors.find(base + ".weight_bias") != tensors.end();

    auto scale_inv_it = tensors.find(scale_inv_key);
    auto block_scale_it = tensors.find(block_scale_key);
    if (block_scale_it == tensors.end() &&
        !has_nvfp4_packed &&
        !(has_grouped_affine_scales && has_grouped_affine_biases)) {
        block_scale_it = tensors.find(weight_scale_key);
    }
    if (scale_inv_it == tensors.end() && block_scale_it == tensors.end()) {
        return weight_2d;
    }

    const int rows = weight_2d.shape(0);
    const int cols = weight_2d.shape(1);
    array scales_f32 = array(1.0f);

    if (block_scale_it != tensors.end()) {
        const array& block_scales = block_scale_it->second;
        if (block_scales.ndim() != 2) {
            throw std::runtime_error("mxfp8 block scales must be rank-2 for " + weight_key);
        }
        if (block_scales.shape(0) != rows) {
            throw std::runtime_error("mxfp8 block scales row mismatch for " + weight_key);
        }
        const int expected_scale_cols = (cols + 31) / 32;
        if (block_scales.shape(1) != expected_scale_cols) {
            throw std::runtime_error(
                "mxfp8 block scales col mismatch for " + weight_key +
                " expected=" + std::to_string(expected_scale_cols) +
                " got=" + std::to_string(block_scales.shape(1))
            );
        }

        if (block_scales.dtype() == uint8 || block_scales.dtype() == int8) {
            const array block_scales_u8 = astype(block_scales, uint8);
            const int flat_size = static_cast<int>(block_scales_u8.size());
            const array flat_u8 = reshape(block_scales_u8, {flat_size});
            eval(flat_u8);

            const uint8_t* scale_bytes = flat_u8.data<uint8_t>();
            std::vector<float> decoded(static_cast<size_t>(flat_size));
            for (int idx = 0; idx < flat_size; ++idx) {
                const uint32_t exp_bits = static_cast<uint32_t>(scale_bytes[idx]) << 23;
                float scale = 0.0f;
                std::memcpy(&scale, &exp_bits, sizeof(float));
                decoded[static_cast<size_t>(idx)] = scale;
            }
            scales_f32 = array(decoded.begin(), Shape{block_scales.shape(0), block_scales.shape(1)}, float32);
        } else {
            scales_f32 = astype(block_scales, float32);
        }
    } else {
        const array& scale_inv = scale_inv_it->second;
        if (scale_inv.ndim() != 2) {
            throw std::runtime_error("fp8 scale_inv must be rank-2 for " + weight_key);
        }
        scales_f32 = astype(scale_inv, float32);
    }

    const array expanded_scales = expand_block_scales(scales_f32, rows, cols, weight_key);
    array weight_as_f32 = array(0.0f);
    if (weight_2d.dtype() == uint8) {
        weight_as_f32 = from_fp8(weight_2d, float32);
    } else {
        weight_as_f32 = astype(weight_2d, float32);
    }
    const array dequant_f32 = weight_as_f32 * expanded_scales;
    array dequant_bf16 = astype(dequant_f32, bfloat16);
    dequant_bf16 = stop_gradient(copy(dequant_bf16));
    // Optional eager materialization for debugging/benchmark parity. Disabled
    // by default to reduce init-time memory pressure on larger checkpoints.
    if (const char* eager = std::getenv("TALU_METAL_EAGER_DEQUANT")) {
        if (std::string(eager) == "1") {
            eval(dequant_bf16);
            synchronize();
        }
    }
    return dequant_bf16;
}

float fp4_e2m1_nibble_to_f32(uint8_t nibble) {
    static constexpr float codebook[16] = {
        0.0f, 0.5f, 1.0f, 1.5f,
        2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f,
        -2.0f, -3.0f, -4.0f, -6.0f,
    };
    return codebook[nibble & 0x0F];
}

bool has_nvfp4_metadata_for_key(
    const std::unordered_map<std::string, array>& tensors,
    const std::string& weight_key
) {
    if (!ends_with(weight_key, ".weight")) return false;
    const std::string base = weight_key.substr(0, weight_key.size() - std::string(".weight").size());
    return tensors.find(base + ".weight_packed") != tensors.end() &&
        tensors.find(base + ".weight_scale") != tensors.end();
}

const array& nvfp4_unit_global_scale() {
    static thread_local std::optional<array> scale;
    if (!scale.has_value()) {
        scale = stop_gradient(copy(array(1.0f, float32)));
    }
    return *scale;
}

array dequantize_nvfp4_weight(
    const std::unordered_map<std::string, array>& tensors,
    const std::string& weight_key
) {
    const std::string base = weight_key.substr(0, weight_key.size() - std::string(".weight").size());
    const array packed_u8 = astype(require_tensor(tensors, base + ".weight_packed"), uint8);
    const array scales_f32 = astype(require_tensor(tensors, base + ".weight_scale"), float32);
    eval(packed_u8);
    eval(scales_f32);

    if (packed_u8.ndim() != 2 || scales_f32.ndim() != 2) {
        throw std::runtime_error("nvfp4 packed/scales must be rank-2 for " + weight_key);
    }

    const int rows = packed_u8.shape(0);
    const int packed_cols = packed_u8.shape(1);
    const int cols = packed_cols * 2;
    if (scales_f32.shape(0) != rows) {
        throw std::runtime_error("nvfp4 scales row mismatch for " + weight_key);
    }
    const int scale_cols = scales_f32.shape(1);
    if (scale_cols <= 0 || cols % scale_cols != 0) {
        throw std::runtime_error("invalid nvfp4 scales shape for " + weight_key);
    }
    const int group_size = cols / scale_cols;

    float global_scale = 1.0f;
    if (has_tensor(tensors, base + ".weight_global_scale")) {
        const array global_f32 = astype(require_tensor(tensors, base + ".weight_global_scale"), float32);
        eval(global_f32);
        if (global_f32.size() > 0) global_scale = global_f32.item<float>();
    }
    if (global_scale == 0.0f) {
        throw std::runtime_error("invalid nvfp4 global scale for " + weight_key);
    }

    const uint8_t* packed_ptr = packed_u8.data<uint8_t>();
    const float* scale_ptr = scales_f32.data<float>();
    std::vector<float> decoded(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    for (int r = 0; r < rows; ++r) {
        const size_t packed_row_off = static_cast<size_t>(r) * static_cast<size_t>(packed_cols);
        const size_t scale_row_off = static_cast<size_t>(r) * static_cast<size_t>(scale_cols);
        const size_t out_row_off = static_cast<size_t>(r) * static_cast<size_t>(cols);
        for (int c = 0; c < cols; ++c) {
            const uint8_t packed = packed_ptr[packed_row_off + static_cast<size_t>(c / 2)];
            const uint8_t nibble = (c & 1) == 0 ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
            const float local_scale = scale_ptr[scale_row_off + static_cast<size_t>(c / group_size)];
            decoded[out_row_off + static_cast<size_t>(c)] = fp4_e2m1_nibble_to_f32(nibble) * (local_scale / global_scale);
        }
    }

    array dequant_bf16 = astype(array(decoded.begin(), Shape{rows, cols}, float32), bfloat16);
    dequant_bf16 = stop_gradient(copy(dequant_bf16));
    return dequant_bf16;
}

bool transcode_nvfp4_to_affine_qmm(
    const std::unordered_map<std::string, array>& tensors,
    const std::string& weight_key,
    const array& lhs_weight,
    int target_group_size,
    array* out_q_w,
    array* out_q_scales,
    array* out_q_biases
) {
    if (!out_q_w || !out_q_scales || !out_q_biases) return false;
    if (target_group_size <= 0 || !has_nvfp4_metadata_for_key(tensors, weight_key)) return false;

    const array dense_bf16 = dequantize_nvfp4_weight(tensors, weight_key);
    const array dense_f32 = astype(dense_bf16, float32);
    eval(dense_f32);
    if (dense_f32.ndim() != 2 || lhs_weight.ndim() != 2) return false;

    const int rows = dense_f32.shape(0);
    const int cols = dense_f32.shape(1);
    const int packed_cols = cols / 8;
    if (rows <= 0 || cols <= 0 || (cols % target_group_size) != 0 || (cols % 8) != 0) return false;
    const bool lhs_matches_unpacked =
        lhs_weight.shape(0) == rows && lhs_weight.shape(1) == cols;
    const bool lhs_matches_packed =
        lhs_weight.shape(0) == rows && lhs_weight.shape(1) == packed_cols;
    if (!lhs_matches_unpacked && !lhs_matches_packed) {
        // Synthetic fused decode weights (for example linear-attention qkvz/ba)
        // must quantize the full concatenated tensor, not a single source slice.
        return false;
    }

    const int groups = cols / target_group_size;
    const float* src = dense_f32.data<float>();

    std::vector<uint32_t> packed(static_cast<size_t>(rows) * static_cast<size_t>(packed_cols), 0);
    std::vector<float> scales(static_cast<size_t>(rows) * static_cast<size_t>(groups), 0.0f);
    std::vector<float> biases(static_cast<size_t>(rows) * static_cast<size_t>(groups), 0.0f);

    for (int r = 0; r < rows; ++r) {
        const size_t row_off = static_cast<size_t>(r) * static_cast<size_t>(cols);
        const size_t packed_row_off = static_cast<size_t>(r) * static_cast<size_t>(packed_cols);
        const size_t group_row_off = static_cast<size_t>(r) * static_cast<size_t>(groups);
        for (int g = 0; g < groups; ++g) {
            const int start = g * target_group_size;
            float min_v = src[row_off + static_cast<size_t>(start)];
            float max_v = min_v;
            for (int i = 1; i < target_group_size; ++i) {
                const float v = src[row_off + static_cast<size_t>(start + i)];
                min_v = std::min(min_v, v);
                max_v = std::max(max_v, v);
            }

            float scale = (max_v - min_v) / 15.0f;
            float bias = min_v;
            if (!(scale > 0.0f) || !std::isfinite(scale)) {
                scale = 1.0f;
                bias = min_v;
            }
            scales[group_row_off + static_cast<size_t>(g)] = scale;
            biases[group_row_off + static_cast<size_t>(g)] = bias;

            for (int i = 0; i < target_group_size; ++i) {
                const int c = start + i;
                const float v = src[row_off + static_cast<size_t>(c)];
                int q = static_cast<int>(std::lrint((v - bias) / scale));
                q = std::max(0, std::min(15, q));
                const size_t packed_idx = packed_row_off + static_cast<size_t>(c / 8);
                const int shift = (c % 8) * 4;
                packed[packed_idx] |= (static_cast<uint32_t>(q) << shift);
            }
        }
    }

    *out_q_w = stop_gradient(copy(array(packed.begin(), Shape{rows, packed_cols}, uint32)));
    *out_q_scales = stop_gradient(copy(astype(array(scales.begin(), Shape{rows, groups}, float32), bfloat16)));
    *out_q_biases = stop_gradient(copy(astype(array(biases.begin(), Shape{rows, groups}, float32), bfloat16)));
    eval(*out_q_w);
    eval(*out_q_scales);
    eval(*out_q_biases);
    return true;
}

bool capture_native_nvfp4_quant(
    const std::unordered_map<std::string, array>& tensors,
    const std::string& weight_key,
    array* out_q_w,
    array* out_q_scales,
    array* out_q_biases
) {
    if (!out_q_w || !out_q_scales || !out_q_biases) return false;
    if (!has_nvfp4_metadata_for_key(tensors, weight_key)) return false;
    if (!ends_with(weight_key, ".weight")) return false;

    const std::string base = weight_key.substr(0, weight_key.size() - std::string(".weight").size());
    const array packed_src = astype(require_tensor(tensors, base + ".weight_packed"), uint8);
    const array scale_src = view(require_tensor(tensors, base + ".weight_scale"), uint8);
    eval(packed_src);
    eval(scale_src);

    if (packed_src.ndim() != 2 || scale_src.ndim() != 2) return false;
    const int rows = packed_src.shape(0);
    const int packed_byte_cols = packed_src.shape(1);
    const int cols = packed_byte_cols * 2;
    if (packed_byte_cols <= 0 || (packed_byte_cols % 4) != 0) return false;
    if ((cols % 16) != 0) return false;
    if (scale_src.shape(0) != rows || scale_src.shape(1) != (cols / 16)) return false;

    array global_scale = array(1.0f, float32);
    if (has_tensor(tensors, base + ".weight_global_scale")) {
        global_scale = astype(require_tensor(tensors, base + ".weight_global_scale"), float32);
    }

    *out_q_w = stop_gradient(copy(reshape(view(packed_src, uint32), {rows, packed_byte_cols / 4})));
    *out_q_scales = stop_gradient(copy(scale_src));
    *out_q_biases = stop_gradient(copy(global_scale));
    return true;
}

void materialize_nvfp4_affine_execution_tensors(
    std::unordered_map<std::string, array>* tensors,
    int target_group_size
) {
    if (!tensors || target_group_size <= 0) return;

    std::vector<std::string> packed_keys;
    packed_keys.reserve(tensors->size());
    for (const auto& entry : *tensors) {
        if (ends_with(entry.first, ".weight_packed")) {
            packed_keys.push_back(entry.first);
        }
    }

    for (const std::string& packed_key : packed_keys) {
        const std::string base = packed_key.substr(0, packed_key.size() - std::string(".weight_packed").size());
        const std::string weight_key = base + ".weight";
        if (tensors->find(weight_key) != tensors->end()) continue;

        auto packed_it = tensors->find(packed_key);
        if (packed_it == tensors->end()) continue;

        array q_w = array(0.0f);
        array q_scales = array(0.0f);
        array q_biases = array(0.0f);
        if (!transcode_nvfp4_to_affine_qmm(
                *tensors,
                weight_key,
                packed_it->second,
                target_group_size,
                &q_w,
                &q_scales,
                &q_biases))
        {
            continue;
        }

        tensors->insert_or_assign(weight_key, q_w);
        tensors->insert_or_assign(base + ".scales", q_scales);
        tensors->insert_or_assign(base + ".biases", q_biases);
    }
}

array maybe_dequantize_grouped_affine_weight(
    const std::unordered_map<std::string, array>& tensors,
    const Qwen35Config& cfg,
    const std::string& weight_key,
    const array& weight_2d
) {
    if (!ends_with(weight_key, ".weight")) {
        return weight_2d;
    }
    if (weight_2d.ndim() != 2) {
        throw std::runtime_error("expected rank-2 weight for " + weight_key);
    }

    const std::string base = weight_key.substr(0, weight_key.size() - std::string(".weight").size());
    const std::string scales_key = base + ".scales";
    const std::string biases_key = base + ".biases";
    const std::string weight_scale_key = base + ".weight_scale";
    const std::string weight_bias_key = base + ".weight_bias";

    auto scales_it = tensors.find(scales_key);
    if (scales_it == tensors.end()) scales_it = tensors.find(weight_scale_key);
    auto biases_it = tensors.find(biases_key);
    if (biases_it == tensors.end()) biases_it = tensors.find(weight_bias_key);
    if (scales_it == tensors.end() || biases_it == tensors.end()) {
        return weight_2d;
    }

    if (cfg.quant_group_size <= 0) {
        throw std::runtime_error("grouped-affine metadata found but quantization.group_size is missing/invalid");
    }

    const int rows = weight_2d.shape(0);
    const int packed_cols = weight_2d.shape(1);

    const array scales_f32 = astype(scales_it->second, float32);
    const array biases_f32 = astype(biases_it->second, float32);
    if (scales_f32.ndim() != 2 || biases_f32.ndim() != 2) {
        throw std::runtime_error("grouped-affine scales/biases must be rank-2 for " + weight_key);
    }
    if (scales_f32.shape(0) != rows || biases_f32.shape(0) != rows) {
        throw std::runtime_error("grouped-affine scales/biases row mismatch for " + weight_key);
    }
    if (scales_f32.shape(1) != biases_f32.shape(1)) {
        throw std::runtime_error("grouped-affine scales/biases group mismatch for " + weight_key);
    }

    // Auto-detect bits from scales shape. Mixed-precision models may use
    // different bits for embedding vs linear weights (e.g. U8 embed + U4 linear).
    const int scale_cols = scales_f32.shape(1);
    const int gs = cfg.quant_group_size;
    int bits = cfg.quant_bits;
    if (bits != 4 && bits != 8) bits = 4;
    // Verify the config hint matches; if not, try the other interpretation.
    const int cols_from_hint = packed_cols * (bits == 4 ? 8 : 4);
    const int expected_groups_hint = (cols_from_hint + gs - 1) / gs;
    if (expected_groups_hint != scale_cols) {
        const int alt_bits = bits == 4 ? 8 : 4;
        const int cols_alt = packed_cols * (alt_bits == 4 ? 8 : 4);
        const int expected_groups_alt = (cols_alt + gs - 1) / gs;
        if (expected_groups_alt == scale_cols) {
            bits = alt_bits;
        } else {
            throw std::runtime_error(
                "grouped-affine scales cols mismatch for " + weight_key +
                " (packed_cols=" + std::to_string(packed_cols) +
                ", scale_cols=" + std::to_string(scale_cols) +
                ", group_size=" + std::to_string(gs) + ")"
            );
        }
    }
    const int values_per_word = bits == 4 ? 8 : 4;
    const uint32_t mask = bits == 4 ? 0xF : 0xFF;
    const int cols = packed_cols * values_per_word;

    const array packed_u32 = astype(weight_2d, uint32);
    eval(packed_u32);
    eval(scales_f32);
    eval(biases_f32);

    const uint32_t* packed_ptr = packed_u32.data<uint32_t>();
    const float* scales_ptr = scales_f32.data<float>();
    const float* biases_ptr = biases_f32.data<float>();
    const int group_count = scales_f32.shape(1);

    std::vector<float> out(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    for (int r = 0; r < rows; ++r) {
        const size_t packed_row_off = static_cast<size_t>(r) * static_cast<size_t>(packed_cols);
        const size_t group_row_off = static_cast<size_t>(r) * static_cast<size_t>(group_count);
        const size_t out_row_off = static_cast<size_t>(r) * static_cast<size_t>(cols);
        for (int c = 0; c < cols; ++c) {
            const int word_idx = c / values_per_word;
            const int lane = c % values_per_word;
            const int shift = lane * bits;
            const uint32_t q = (packed_ptr[packed_row_off + static_cast<size_t>(word_idx)] >> shift) & mask;
            const int g = c / cfg.quant_group_size;
            const float scale = scales_ptr[group_row_off + static_cast<size_t>(g)];
            const float bias = biases_ptr[group_row_off + static_cast<size_t>(g)];
            out[out_row_off + static_cast<size_t>(c)] = static_cast<float>(q) * scale + bias;
        }
    }

    const array dequant_f32 = array(out.begin(), Shape{rows, cols}, float32);
    array dequant_dense = astype(dequant_f32, bfloat16);
    dequant_dense = stop_gradient(copy(dequant_dense));
    return dequant_dense;
}

bool has_grouped_affine_metadata_for_key(
    const std::unordered_map<std::string, array>& tensors,
    const std::string& weight_key
) {
    if (!ends_with(weight_key, ".weight")) return false;
    const std::string base = weight_key.substr(0, weight_key.size() - std::string(".weight").size());
    const bool has_scales =
        tensors.find(base + ".scales") != tensors.end() ||
        tensors.find(base + ".weight_scale") != tensors.end();
    const bool has_biases =
        tensors.find(base + ".biases") != tensors.end() ||
        tensors.find(base + ".weight_bias") != tensors.end();
    return has_scales && has_biases;
}

bool maybe_capture_grouped_affine_quant(
    const std::unordered_map<std::string, array>& tensors,
    const Qwen35Config& cfg,
    const std::string& weight_key,
    const array& lhs_weight,
    array* out_q_w,
    array* out_q_scales,
    array* out_q_biases,
    int* out_detected_bits = nullptr
) {
    if (!has_grouped_affine_metadata_for_key(tensors, weight_key)) return false;
    if (!ends_with(weight_key, ".weight")) return false;
    if (lhs_weight.ndim() != 2) return false;
    if (cfg.quant_bits != 4 && cfg.quant_bits != 8) return false;
    if (cfg.quant_group_size <= 0) return false;

    const std::string base = weight_key.substr(0, weight_key.size() - std::string(".weight").size());
    auto raw_it = tensors.find(weight_key);
    if (raw_it == tensors.end()) return false;

    auto scales_it = tensors.find(base + ".scales");
    if (scales_it == tensors.end()) scales_it = tensors.find(base + ".weight_scale");
    auto biases_it = tensors.find(base + ".biases");
    if (biases_it == tensors.end()) biases_it = tensors.find(base + ".weight_bias");
    if (scales_it == tensors.end() || biases_it == tensors.end()) return false;

    const array& raw_weight = raw_it->second;
    if (raw_weight.ndim() != 2) return false;
    const int rows = raw_weight.shape(0);
    const int packed_cols = raw_weight.shape(1);

    const array& scales_src = scales_it->second;
    const array& biases_src = biases_it->second;
    const Dtype scales_dt = scales_src.dtype();
    const Dtype biases_dt = biases_src.dtype();
    const bool scales_float = scales_dt == float16 || scales_dt == bfloat16 || scales_dt == float32 || scales_dt == float64;
    const bool biases_float = biases_dt == float16 || biases_dt == bfloat16 || biases_dt == float32 || biases_dt == float64;
    if (!scales_float || !biases_float) return false;
    if (scales_src.ndim() != 2 || biases_src.ndim() != 2) return false;
    if (scales_src.shape(0) != rows || biases_src.shape(0) != rows) return false;
    if (scales_src.shape(1) != biases_src.shape(1)) return false;

    // Auto-detect bits from scales shape for mixed-precision models.
    const int gs = cfg.quant_group_size;
    const int scale_cols = scales_src.shape(1);
    int bits = cfg.quant_bits;
    if (bits != 4 && bits != 8) bits = 4;
    int cols = packed_cols * (bits == 4 ? 8 : 4);
    int expected_groups = (cols + gs - 1) / gs;
    if (expected_groups != scale_cols) {
        const int alt_bits = bits == 4 ? 8 : 4;
        const int alt_cols = packed_cols * (alt_bits == 4 ? 8 : 4);
        const int alt_groups = (alt_cols + gs - 1) / gs;
        if (alt_groups == scale_cols) {
            bits = alt_bits;
            cols = alt_cols;
            expected_groups = alt_groups;
        } else {
            return false;
        }
    }

    const int values_per_word = bits == 4 ? 8 : 4;
    const bool lhs_matches_unpacked =
        lhs_weight.shape(0) == rows && lhs_weight.shape(1) == cols;
    const bool lhs_matches_packed =
        lhs_weight.shape(0) == rows && lhs_weight.shape(1) == packed_cols;
    if (!lhs_matches_unpacked && !lhs_matches_packed) {
        return false;
    }

    *out_q_w = astype(raw_weight, uint32);
    *out_q_scales = scales_src;
    *out_q_biases = biases_src;
    if (out_detected_bits) *out_detected_bits = bits;
    return true;
}

array load_linear_weight(
    const std::unordered_map<std::string, array>& tensors,
    const Qwen35Config& cfg,
    const std::string& weight_key
) {
    const array& raw_weight = require_tensor(tensors, weight_key);
    const array grouped = maybe_dequantize_grouped_affine_weight(tensors, cfg, weight_key, raw_weight);
    array out = array(0.0f);
    if (grouped.size() != raw_weight.size() || grouped.dtype() != raw_weight.dtype() || grouped.ndim() != raw_weight.ndim()) {
        out = maybe_dequantize_fp8_weight(tensors, weight_key, grouped);
    } else if (has_nvfp4_metadata_for_key(tensors, weight_key)) {
        out = dequantize_nvfp4_weight(tensors, weight_key);
    } else {
        out = maybe_dequantize_fp8_weight(tensors, weight_key, grouped);
    }

    return out;
}

void maybe_quantize_mxfp8_matrix(
    const std::unordered_map<std::string, array>& tensors,
    const Qwen35Config& cfg,
    const std::string& weight_key,
    const array& lhs_weight,
    bool enabled,
    array* out_q_w,
    array* out_q_scales,
    array* out_q_biases,
    bool* out_has_q,
    int* out_detected_bits = nullptr
) {
    if (!enabled) {
        *out_has_q = false;
        return;
    }
    if (out_q_biases) {
        *out_q_biases = array(0.0f);
    }
    g_qmm_attempts += 1;
    // Fast path for native MXFP8 checkpoints: reuse packed FP8 payload and
    // block scales directly for decode-time quantized matmul.
    if (ends_with(weight_key, ".weight")) {
        const std::string base = weight_key.substr(0, weight_key.size() - std::string(".weight").size());
        const std::string block_scale_key = base + ".weight_block_scale";
        const std::string weight_scale_key = base + ".weight_scale";
        auto raw_it = tensors.find(weight_key);
        auto scales_it = tensors.find(block_scale_key);
        if (scales_it == tensors.end()) {
            scales_it = tensors.find(weight_scale_key);
        }
        if (raw_it != tensors.end() && scales_it != tensors.end()) {
            const array& raw_weight = raw_it->second;
            const array& raw_scales = scales_it->second;
            if (raw_weight.ndim() == 2 && raw_scales.ndim() == 2) {
                const int rows = raw_weight.shape(0);
                const int cols = raw_weight.shape(1);
                const int expected_scale_cols = (cols + 31) / 32;
                if (raw_scales.shape(0) == rows && raw_scales.shape(1) == expected_scale_cols && cols % 4 == 0) {
                    try {
                        const array packed = reshape(view(raw_weight, uint32), {rows, cols / 4});
                        *out_q_w = packed;
                        *out_q_scales = astype(raw_scales, uint8);
                        *out_has_q = true;
                        g_qmm_success += 1;
                        return;
                    } catch (const std::exception&) {
                        // Fall through to generic quantize path.
                    }
                }
            }
        }
    }

    if (out_q_biases != nullptr &&
        has_nvfp4_metadata_for_key(tensors, weight_key) &&
        !has_grouped_affine_metadata_for_key(tensors, weight_key))
    {
        if (capture_native_nvfp4_quant(
                tensors,
                weight_key,
                out_q_w,
                out_q_scales,
                out_q_biases))
        {
            *out_has_q = true;
            g_qmm_success += 1;
            return;
        }
    }

    const bool enforce_grouped_affine_lossless = env_truthy("TALU_METAL_GAFFINE_LOSSLESS", true);
    if (enforce_grouped_affine_lossless && out_q_biases != nullptr) {
        if (maybe_capture_grouped_affine_quant(
                tensors,
                cfg,
                weight_key,
                lhs_weight,
                out_q_w,
                out_q_scales,
                out_q_biases,
                out_detected_bits))
        {
            *out_has_q = true;
            g_qmm_success += 1;
            return;
        }
        // If grouped-affine metadata exists but cannot be mapped directly
        // (for example synthetic fused tensors), avoid lossy re-quantization.
        if (has_grouped_affine_metadata_for_key(tensors, weight_key)) {
            *out_has_q = false;
            return;
        }
    }

    array quant_input = lhs_weight;
    const Dtype dt = quant_input.dtype();
    const bool is_float_input = (dt == float16 || dt == bfloat16 || dt == float32 || dt == float64);
    if (!is_float_input && ends_with(weight_key, ".weight")) {
        // Some converted checkpoints keep FP8 payload tensors as uint8 and
        // expose scales via metadata. Dequantize first when needed so MLX
        // quantize() receives a real floating tensor.
        if (has_nvfp4_metadata_for_key(tensors, weight_key)) {
            quant_input = dequantize_nvfp4_weight(tensors, weight_key);
        } else {
            quant_input = maybe_dequantize_grouped_affine_weight(tensors, cfg, weight_key, quant_input);
            quant_input = maybe_dequantize_fp8_weight(tensors, weight_key, quant_input);
        }
    }
    const Dtype qdt = quant_input.dtype();
    const bool quant_input_is_float = (qdt == float16 || qdt == bfloat16 || qdt == float32 || qdt == float64);
    if (!quant_input_is_float) {
        *out_has_q = false;
        return;
    }

    std::vector<array> q;
    try {
        q = quantize(
            quant_input,
            std::nullopt,
            std::nullopt,
            "mxfp8"
        );
    } catch (const std::exception&) {
        *out_has_q = false;
        return;
    }
    if (q.size() != 2) {
        throw std::runtime_error("mlx quantize(mxfp8) returned unexpected output count");
    }
    *out_q_w = q[0];
    *out_q_scales = q[1];
    *out_has_q = true;
    g_qmm_success += 1;
}

array linear_decode_maybe_quantized(
    bool decode_qmm_enabled,
    int quant_group_size,
    int quant_bits,
    const array& x,
    const array& rhs,
    bool has_q,
    const array& q_w,
    const array& q_scales,
    const array& q_biases,
    bool prefill_only_qmm = false,
    int layer_idx = -1,
    const char* op_tag = nullptr
) {
    const auto quant_compare_enabled = []() -> bool {
        return env_truthy("TALU_METAL_QMM_COMPARE", false);
    };
    const auto quant_compare_layer = []() -> int {
        if (const char* raw = std::getenv("TALU_METAL_QMM_COMPARE_LAYER")) {
            const int parsed = std::atoi(raw);
            return parsed > 0 ? parsed - 1 : parsed;
        }
        return 0;
    };
    const auto quant_compare_op = []() -> std::string {
        if (const char* raw = std::getenv("TALU_METAL_QMM_COMPARE_OP")) return std::string(raw);
        return std::string();
    };
    const auto should_compare_quant = [&](bool q_path_active) -> bool {
        if (!q_path_active || !quant_compare_enabled()) return false;
        if (rhs.size() <= 1 || rhs.ndim() < 2) return false;
        const int target_layer = quant_compare_layer();
        if (layer_idx >= 0 && target_layer >= 0 && layer_idx != target_layer) return false;
        const std::string target_op = quant_compare_op();
        if (target_op.empty()) return true;
        if (op_tag == nullptr) return false;
        return target_op == op_tag;
    };
    const auto emit_quant_compare = [&](const array& quant_out) {
        const array quant_f32 = astype(quant_out, float32);
        const array dense_f32 = astype(matmul(x, rhs), float32);
        const array diff_f32 = abs(quant_f32 - dense_f32);
        eval({quant_f32, dense_f32, diff_f32});
        synchronize();
        const float* diff_ptr = diff_f32.data<float>();
        const size_t n = static_cast<size_t>(diff_f32.size());
        if (diff_ptr == nullptr || n == 0) return;
        double sum = 0.0;
        float max_abs = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            const float v = diff_ptr[i];
            sum += static_cast<double>(v);
            if (v > max_abs) max_abs = v;
        }
        const double mean_abs = sum / static_cast<double>(n);
        std::fprintf(
            stderr,
            "[mlx][qmm_compare] layer=%d op=%s elems=%zu max_abs=%.6f mean_abs=%.6f q_dtype=%d rhs_dtype=%d x_shape=[%d,%d,%d]\n",
            layer_idx,
            op_tag ? op_tag : "(null)",
            n,
            static_cast<double>(max_abs),
            mean_abs,
            static_cast<int>(static_cast<Dtype::Val>(quant_out.dtype())),
            static_cast<int>(static_cast<Dtype::Val>(rhs.dtype())),
            x.ndim() > 0 ? x.shape(0) : 0,
            x.ndim() > 1 ? x.shape(1) : 0,
            x.ndim() > 2 ? x.shape(2) : 0
        );
    };

    const bool is_prefill_shape = x.ndim() >= 3 && x.shape(1) > 1;
    const bool allow_qmm = !prefill_only_qmm || is_prefill_shape;
    const bool q_ready = has_q && decode_qmm_enabled && allow_qmm && q_w.size() > 0 && q_scales.size() > 0;
    if (q_ready) {
        const bool affine_ready = q_biases.ndim() == 2 && q_biases.size() > 1 &&
            quant_group_size > 0 && (quant_bits == 4 || quant_bits == 8);
        const bool nvfp4_ready =
            !affine_ready &&
            q_w.dtype() == uint32 &&
            q_scales.dtype() == uint8 &&
            q_biases.size() == 1 &&
            q_biases.dtype() == float32 &&
            quant_group_size == 16 &&
            quant_bits == 4;
        if (affine_ready) {
            const array out = quantized_matmul(
                x,
                q_w,
                q_scales,
                std::optional<array>(q_biases),
                true,
                quant_group_size,
                quant_bits,
                "affine"
            );
            if (should_compare_quant(true)) emit_quant_compare(out);
            return out;
        }
        if (nvfp4_ready) {
            const bool reshape_back = x.ndim() == 3;
            const int batch = x.shape(0);
            const int seq = reshape_back ? x.shape(1) : 1;
            const array x2d = reshape_back ? reshape(x, {batch * seq, x.shape(2)}) : x;
            const array out2d = qqmm(
                x2d,
                q_w,
                std::optional<array>(q_scales),
                quant_group_size,
                quant_bits,
                "nvfp4",
                std::optional<array>(nvfp4_unit_global_scale()),
                std::optional<array>(q_biases)
            );
            const array out = reshape_back ? reshape(out2d, {batch, seq, out2d.shape(1)}) : out2d;
            if (should_compare_quant(true)) emit_quant_compare(out);
            return out;
        }
        const array out = quantized_matmul(
            x,
            q_w,
            q_scales,
            std::nullopt,
            true,
            std::nullopt,
            std::nullopt,
            "mxfp8"
        );
        if (should_compare_quant(true)) emit_quant_compare(out);
        return out;
    }
    if (rhs.size() <= 1 || rhs.ndim() < 2) {
        throw std::runtime_error("linear_decode_maybe_quantized: dense rhs is missing while quantized path is unavailable");
    }
    return matmul(x, rhs);
}

array linear(const array& x, const array& rhs) {
    return matmul(x, rhs);
}

array silu(const array& x) {
    return x * sigmoid(x);
}

array gelu_approx(const array& x) {
    const array coeff = array(0.044715f, x.dtype());
    const array sqrt_two_over_pi = array(0.7978845608f, x.dtype());
    const array half = array(0.5f, x.dtype());
    const array one = array(1.0f, x.dtype());
    const array x3 = x * x * x;
    const array inner = sqrt_two_over_pi * (x + coeff * x3);
    return half * x * (one + tanh(inner));
}

array softplus(const array& x) {
    const array ax = abs(x);
    const array maxv = maximum(x, array(0.0f));
    return log1p(exp(-ax)) + maxv;
}

array compute_g(const array& A_exp_f32, const array& a, const array& dt_bias) {
    static const auto compiled_compute_g = compile(
        [](const std::vector<array>& inputs) {
            const array& a_exp = inputs[0];
            const array& a_in = inputs[1];
            const array& dt_in = inputs[2];
            const array a_plus = astype(a_in + dt_in, float32);
            const array decay = exp(-(a_exp * softplus(a_plus)));
            return std::vector<array>{astype(decay, a_in.dtype())};
        },
        true
    );
    return compiled_compute_g({A_exp_f32, a, dt_bias}).at(0);
}

float scalar_to_f32(const array& value, const std::string& name) {
    if (value.size() <= 0) {
        throw std::runtime_error("invalid scalar tensor: " + name);
    }
    array flat = reshape(astype(value, float32), {1});
    eval(flat);
    const float* ptr = flat.data<float>();
    if (ptr == nullptr) {
        throw std::runtime_error("failed to read scalar tensor: " + name);
    }
    return ptr[0];
}

array last_token_logits(const array& logits) {
    if (logits.ndim() != 3) throw std::runtime_error("expected logits rank-3 [B,T,V]");
    const int b = logits.shape(0);
    const int t = logits.shape(1);
    const int v = logits.shape(2);
    if (t <= 0) throw std::runtime_error("logits have zero sequence length");
    return slice(logits, {0, t - 1, 0}, {b, t, v});
}

array take_time_4d(const array& a, int t) {
    const int b = a.shape(0);
    const int h = a.shape(2);
    const int d = a.shape(3);
    return reshape(slice(a, {0, t, 0, 0}, {b, t + 1, h, d}), {b, h, d});
}

array take_time_3d(const array& a, int t) {
    const int b = a.shape(0);
    const int h = a.shape(2);
    return reshape(slice(a, {0, t, 0}, {b, t + 1, h}), {b, h});
}

std::pair<array, array> gated_delta_update_fallback(
    const array& q,
    const array& k,
    const array& v,
    const array& a,
    const array& b,
    const array& A_exp_f32,
    const array& dt_bias,
    const std::optional<array>& state_opt
) {
    const int B = q.shape(0);
    const int T = q.shape(1);
    const int Hk = q.shape(2);
    const int Dk = q.shape(3);
    const int Hv = v.shape(2);
    const int Dv = v.shape(3);

    array state = state_opt.has_value() ? *state_opt : zeros({B, Hv, Dv, Dk}, q.dtype());

    array q_use = q;
    array k_use = k;
    if (Hv % Hk != 0) {
        throw std::runtime_error("gated-delta head mismatch: Hv must be divisible by Hk");
    }
    const int repeat_factor = Hv / Hk;
    if (repeat_factor > 1) {
        q_use = repeat(q_use, repeat_factor, 2);
        k_use = repeat(k_use, repeat_factor, 2);
    }

    const array beta = sigmoid(b);
    const array g = compute_g(A_exp_f32, a, dt_bias);

    std::vector<array> ys;
    ys.reserve(static_cast<size_t>(T));

    for (int t = 0; t < T; ++t) {
        const array q_t = take_time_4d(q_use, t);     // [B, Hv, Dk]
        const array k_t = take_time_4d(k_use, t);     // [B, Hv, Dk]
        const array v_t = take_time_4d(v, t);         // [B, Hv, Dv]
        const array beta_t = take_time_3d(beta, t);   // [B, Hv]
        const array g_t = take_time_3d(g, t);         // [B, Hv]

        const array k_exp = reshape(k_t, {B, Hv, 1, Dk});
        const array q_exp = reshape(q_t, {B, Hv, 1, Dk});

        state = state * reshape(g_t, {B, Hv, 1, 1});
        const array kv_mem = sum(state * k_exp, std::vector<int>{-1}, false); // [B, Hv, Dv]
        const array delta = (v_t - kv_mem) * reshape(beta_t, {B, Hv, 1});
        state = state + k_exp * reshape(delta, {B, Hv, Dv, 1});

        const array y_t = sum(state * q_exp, std::vector<int>{-1}, false); // [B, Hv, Dv]
        ys.push_back(y_t);
    }

    const array y = stack(ys, 1); // [B, T, Hv, Dv]
    return {y, state};
}

const std::string& gated_delta_kernel_source() {
    static const std::string source = R"(
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hqk, Dk]
        auto q_ = q + b_idx * T * Hqk * Dk + hv_idx * Dk;
        auto k_ = k + b_idx * T * Hqk * Dk + hv_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        // g: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_[hv_idx];
            kv_mem += state[i] * k_[s_idx];
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_[s_idx] * delta;
            out += state[i] * q_[s_idx];
          }
          out = simd_sum(out);
          if (thread_index_in_simdgroup == 0) {
            y[dv_idx] = static_cast<InT>(out);
          }

          // increment pointers to next timestep
          q_ += Hqk * Dk;
          k_ += Hqk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<InT>(state[i]);
        }
    )";
    return source;
}

std::pair<array, array> gated_delta_update_kernel(
    const array& q,
    const array& k,
    const array& v,
    const array& a,
    const array& b,
    const array& A_exp_f32,
    const array& dt_bias,
    const std::optional<array>& state_opt
) {
    const int B = q.shape(0);
    const int T = q.shape(1);
    const int Hk = q.shape(2);
    const int Dk = q.shape(3);
    const int Hv = v.shape(2);
    const int Dv = v.shape(3);

    if (Dk % 32 != 0) {
        throw std::runtime_error("gated-delta kernel requires Dk divisible by 32");
    }

    array state = state_opt.has_value() ? *state_opt : zeros({B, Hv, Dv, Dk}, q.dtype());
    array q_use = q;
    array k_use = k;

    if (Hv % Hk != 0) {
        throw std::runtime_error("gated-delta head mismatch: Hv must be divisible by Hk");
    }
    const int repeat_factor = Hv / Hk;
    if (repeat_factor > 1) {
        q_use = repeat(q_use, repeat_factor, 2);
        k_use = repeat(k_use, repeat_factor, 2);
    }

    const array beta = sigmoid(b);
    const array g = compute_g(A_exp_f32, a, dt_bias);
    const array t_scalar(static_cast<int32_t>(T), int32);

    static const fast::CustomKernelFunction kernel = fast::metal_kernel(
        "gated_delta_step",
        {"q", "k", "v", "g", "beta", "state_in", "T"},
        {"y", "state_out"},
        gated_delta_kernel_source()
    );

    const std::vector<array> outputs = kernel(
        {q_use, k_use, v, g, beta, state, t_scalar},
        {Shape{B, T, Hv, Dv}, state.shape()},
        {q.dtype(), q.dtype()},
        std::make_tuple(32, Dv, B * Hv),
        std::make_tuple(32, 4, 1),
        {
            {"InT", q.dtype()},
            {"Dk", Dk},
            {"Dv", Dv},
            {"Hqk", q_use.shape(2)},
            {"Hv", Hv},
        },
        std::nullopt,
        false,
        {}
    );
    return {outputs.at(0), outputs.at(1)};
}

std::pair<array, array> gated_delta_update(
    const array& q,
    const array& k,
    const array& v,
    const array& a,
    const array& b,
    const array& A_exp_f32,
    const array& dt_bias,
    const std::optional<array>& state_opt
) {
    if (metal::is_available() && default_device() == Device::gpu && q.shape(3) % 32 == 0) {
        return gated_delta_update_kernel(q, k, v, a, b, A_exp_f32, dt_bias, state_opt);
    }
    return gated_delta_update_fallback(q, k, v, a, b, A_exp_f32, dt_bias, state_opt);
}

array rmsnorm_gated(const array& hidden_states, const array& gate, const array& weight, float eps) {
    const array normed = fast::rms_norm(hidden_states, std::optional<array>(weight), eps);
    const array mixed_f32 = silu(astype(gate, float32)) * astype(normed, float32);
    return astype(mixed_f32, hidden_states.dtype());
}

} // namespace

struct mlx_ctx {
    std::string model_id;
    std::string model_path;

    Qwen35Config cfg;

    array embed_tokens = array(0.0f); // [vocab, hidden]
    array lm_head_rhs = array(0.0f);  // [hidden, vocab]
    array lm_head_q_w = array(0.0f);  // [vocab, hidden * bits / 32] packed
    array lm_head_q_scales = array(0.0f); // [vocab, hidden / group_size]
    array lm_head_q_biases = array(0.0f); // [vocab, hidden / group_size] affine-only
    int lm_head_q_bits = 0; // detected quantization bits for lm_head (may differ from cfg)
    array final_norm_w = array(0.0f); // [hidden]
    bool has_fp8_meta = false;
    bool has_mxfp8_meta = false;
    bool has_grouped_affine_meta = false;
    bool lm_head_q_decode_enabled = false;
    bool fp8_decode_qmm_enabled = false;
    bool per_layer_input_enabled = false;
    int per_layer_hidden_size = 0;
    float per_layer_input_scale = 0.70710677f; // 2^-0.5
    float per_layer_embed_scale = 1.0f;
    float per_layer_model_projection_scale = 1.0f;
    array per_layer_embedding = array(0.0f); // [vocab, layers*hpl]
    array per_layer_projection_norm_w = array(0.0f); // [hpl]

    std::vector<LayerWeights> layers;
    std::vector<KVCacheState> kv_cache;
    std::vector<LinearCacheState> linear_cache;
    bool profile_layers = false;
    bool stream_ready = false;
    uint32_t trace_decode_token = 1;
    bool xray_enabled = false;
    std::unordered_map<int32_t, int32_t> sampling_context_counts;
    int32_t sampling_context_len = 0;
    std::vector<int32_t> sampling_unique_ids;
    std::vector<float> sampling_repetition_scales;
    std::vector<float> sampling_additive_penalties;
};

namespace {

constexpr uint8_t XRAY_POINT_EMBED = 0;
constexpr uint8_t XRAY_POINT_LAYER_INPUT = 2;
constexpr uint8_t XRAY_POINT_ATTN_OUT = 15;
constexpr uint8_t XRAY_POINT_BLOCK_OUT = 21;
constexpr uint8_t XRAY_POINT_FINAL_NORM = 26;
constexpr uint8_t XRAY_POINT_LM_HEAD = 27;
constexpr uint16_t XRAY_GLOBAL_LAYER = 0xFFFF;

struct TraceFrame {
    uint32_t token = 0;
    uint32_t layer_position = 0;
    bool emit_embed = false;
};

thread_local std::optional<mlx_ctx> g_fused_decode_ctx_cache;
thread_local std::vector<mlx_ctx*> g_fused_decode_ctx_cache_rows;
thread_local std::string g_fused_decode_ctx_cache_key;
thread_local bool g_fused_decode_ctx_cache_dirty = false;

bool xray_should_emit(uint8_t point_id, uint16_t layer, uint32_t position) {
    return talu_metal_xray_should_emit(point_id, layer, position) != 0;
}

void xray_emit_array_f32(
    uint8_t point_id,
    uint16_t layer,
    uint32_t token,
    uint32_t position,
    const array& value,
    const char* kernel_name
) {
    if (!xray_should_emit(point_id, layer, position)) return;
    const int ndim = value.ndim();
    if (ndim <= 0 || ndim > 4) return;

    const array as_f32 = astype(value, float32);
    const int flat_size = static_cast<int>(as_f32.size());
    const array flat = reshape(as_f32, {flat_size});
    eval(flat);
    synchronize();

    uint32_t dims[4] = {0, 0, 0, 0};
    for (int i = 0; i < ndim; ++i) {
        dims[i] = static_cast<uint32_t>(as_f32.shape(i));
    }
    talu_metal_xray_emit_f32(
        point_id,
        layer,
        token,
        position,
        flat.data<float>(),
        dims[0],
        dims[1],
        dims[2],
        dims[3],
        static_cast<uint8_t>(ndim),
        kernel_name
    );
}

bool token_is_eos(int32_t token, const int32_t* eos_ids, int32_t eos_len) {
    if (!eos_ids || eos_len <= 0) return false;
    for (int32_t i = 0; i < eos_len; ++i) {
        if (token == eos_ids[i]) return true;
    }
    return false;
}

bool has_fp8_quant_metadata(const std::unordered_map<std::string, array>& tensors) {
    for (const auto& kv : tensors) {
        if (ends_with(kv.first, ".weight_scale_inv") ||
            ends_with(kv.first, ".weight_block_scale") ||
            ends_with(kv.first, ".weight_scale")) {
            return true;
        }
    }
    return false;
}

bool has_mxfp8_quant_metadata(const std::unordered_map<std::string, array>& tensors) {
    for (const auto& kv : tensors) {
        if (ends_with(kv.first, ".weight_block_scale")) {
            return true;
        }
    }
    return false;
}

bool has_grouped_affine_quant_metadata(const std::unordered_map<std::string, array>& tensors) {
    for (const auto& kv : tensors) {
        if (!ends_with(kv.first, ".weight")) continue;
        const std::string base = kv.first.substr(0, kv.first.size() - std::string(".weight").size());
        if (tensors.find(base + ".scales") != tensors.end() &&
            tensors.find(base + ".biases") != tensors.end()) {
            return true;
        }
    }
    return false;
}

bool has_nvfp4_quant_metadata(const std::unordered_map<std::string, array>& tensors) {
    for (const auto& kv : tensors) {
        if (!ends_with(kv.first, ".weight_packed")) continue;
        const std::string base = kv.first.substr(0, kv.first.size() - std::string(".weight_packed").size());
        if (tensors.find(base + ".weight_scale") != tensors.end()) {
            return true;
        }
    }
    return false;
}

bool env_truthy(const char* name, bool fallback) {
    const char* raw = std::getenv(name);
    if (!raw) return fallback;
    return std::string(raw) == "1";
}

bool logits_sanity_enabled() {
    return env_truthy("TALU_METAL_LOGITS_SANITY", false);
}

void emit_logits_sanity(const char* stage, int row_idx, const float* logits, int logits_len) {
    if (!logits || logits_len <= 0) return;

    int top_idx = 0;
    float top_val = logits[0];
    size_t nan_count = 0;
    size_t inf_count = 0;
    size_t neg_inf_count = 0;

    for (int i = 0; i < logits_len; ++i) {
        const float v = logits[i];
        if (std::isnan(v)) {
            nan_count += 1;
            continue;
        }
        if (!std::isfinite(v)) {
            inf_count += 1;
            if (v < 0.0f) neg_inf_count += 1;
            continue;
        }
        if (v > top_val) {
            top_val = v;
            top_idx = i;
        }
    }

    std::fprintf(
        stderr,
        "[mlx][logits_sanity] stage=%s row=%d top0_id=%d top0=%.6f nan=%zu inf=%zu neg_inf=%zu len=%d\n",
        stage ? stage : "(null)",
        row_idx,
        top_idx,
        static_cast<double>(top_val),
        nan_count,
        inf_count,
        neg_inf_count,
        logits_len
    );
    std::fflush(stderr);
}

void emit_logits_sanity_rows(
    const char* stage,
    const array& logits_last,
    const int32_t* logical_rows,
    int32_t logical_row_count
) {
    if (!logits_sanity_enabled()) return;
    if (logits_last.ndim() != 3) return;
    const int rows = logits_last.shape(0);
    const int vocab = logits_last.shape(2);
    if (rows <= 0 || vocab <= 0) return;

    const array logits_rows = reshape(astype(logits_last, float32), {rows, vocab});
    eval(logits_rows);
    synchronize();

    const float* ptr = logits_rows.data<float>();
    if (!ptr) return;

    const int limit = std::min(rows, std::max(0, logical_row_count));
    for (int r = 0; r < limit; ++r) {
        const int row_id = logical_rows ? logical_rows[r] : r;
        emit_logits_sanity(stage, row_id, ptr + static_cast<size_t>(r) * static_cast<size_t>(vocab), vocab);
    }
}

int env_positive_int(const char* name, int fallback) {
    const char* raw = std::getenv(name);
    if (!raw) return fallback;
    try {
        const int parsed = std::stoi(raw);
        if (parsed > 0) return parsed;
    } catch (...) {
    }
    return fallback;
}

int resolve_prefill_chunk_size() {
    // Keep prefill graphs bounded by default to avoid Metal OOM on larger
    // models; callers can override for benchmarking.
    return env_positive_int("TALU_METAL_PREFILL_CHUNK_SIZE", 64);
}

int resolve_prefill_full_prompt_threshold() {
    // For short prompts, one full-prompt forward avoids extra chunk-loop
    // dispatch/sync overhead in the prefill + seed-token path.
    return env_positive_int("TALU_METAL_PREFILL_FULL_PROMPT_THRESHOLD", 128);
}

bool should_use_full_prompt_prefill(const mlx_ctx* ctx, int32_t prompt_len, int full_prompt_threshold) {
    if (!ctx) return false;
    if (prompt_len <= 0) return false;
    if (prompt_len > full_prompt_threshold) return false;
    if (ctx->cfg.hidden_size_per_layer_input > 0) return false;
    return true;
}

int resolve_prefill_batch_full_prompt_threshold() {
    // Batched eval traffic commonly uses ~120-200 token prompts. A higher
    // batch-only threshold keeps these requests on fused prefill paths instead
    // of falling back to per-row serial prefill.
    return env_positive_int(
        "TALU_METAL_PREFILL_BATCH_FULL_PROMPT_THRESHOLD",
        env_positive_int("TALU_METAL_PREFILL_FULL_PROMPT_THRESHOLD", 512)
    );
}

int resolve_prefill_batch_len_bucket() {
    // Variable-length fused prefill pads rows to the group's max prompt
    // length. Bucketing avoids oversized padding when one long row would
    // otherwise stretch the whole batch.
    return env_positive_int("TALU_METAL_PREFILL_BATCH_LEN_BUCKET", 64);
}

struct MemoryBudgetEstimate {
    size_t memory_size_bytes = 0;
    size_t max_recommended_bytes = 0;
    size_t effective_budget_bytes = 0;
    size_t weight_bytes = 0;
    size_t kv_cache_bytes = 0;
    size_t linear_state_bytes = 0;
    size_t overhead_bytes = 0;
    size_t total_required_bytes = 0;
    int context_tokens = 0;
    int active_slots = 1;
};

size_t checked_add_size_t(size_t a, size_t b, const char* context) {
    if (b > std::numeric_limits<size_t>::max() - a) {
        throw std::runtime_error(std::string("size overflow while computing ") + context);
    }
    return a + b;
}

std::optional<size_t> recommended_working_set_bytes() {
    if (!metal::is_available()) return std::nullopt;
    const auto& info = device_info(Device::gpu);
    auto it = info.find("max_recommended_working_set_size");
    if (it == info.end() || !std::holds_alternative<size_t>(it->second)) {
        return std::nullopt;
    }
    const size_t value = std::get<size_t>(it->second);
    if (value == 0) return std::nullopt;
    return value;
}

std::optional<size_t> device_memory_size_bytes() {
    if (!metal::is_available()) return std::nullopt;
    const auto& info = device_info(Device::gpu);
    auto it = info.find("memory_size");
    if (it != info.end() && std::holds_alternative<size_t>(it->second)) {
        const size_t value = std::get<size_t>(it->second);
        if (value > 0) return value;
    }
    it = info.find("total_memory");
    if (it != info.end() && std::holds_alternative<size_t>(it->second)) {
        const size_t value = std::get<size_t>(it->second);
        if (value > 0) return value;
    }
    return std::nullopt;
}

size_t model_weight_bytes_on_disk(const std::string& model_path) {
    size_t total = 0;
    for (const std::string& rel : discover_weight_files(model_path)) {
        const std::filesystem::path abs = std::filesystem::path(model_path) / rel;
        const auto raw_size = std::filesystem::file_size(abs);
        if (raw_size > std::numeric_limits<size_t>::max()) {
            throw std::runtime_error("weight shard too large for host size_t: " + abs.string());
        }
        total = checked_add_size_t(total, static_cast<size_t>(raw_size), "weight bytes");
    }
    return total;
}

int estimate_full_attention_layer_count(const Qwen35Config& cfg) {
    const bool hybrid =
        cfg.linear_num_value_heads > 0 &&
        cfg.linear_num_key_heads > 0 &&
        cfg.linear_key_head_dim > 0 &&
        cfg.linear_value_head_dim > 0 &&
        cfg.linear_conv_kernel_dim > 0 &&
        cfg.full_attention_interval > 1;
    if (!hybrid) return cfg.num_hidden_layers;
    const int interval = std::max(1, cfg.full_attention_interval);
    return (cfg.num_hidden_layers + interval - 1) / interval;
}

size_t estimate_kv_cache_bytes(const Qwen35Config& cfg, int context_tokens, int active_slots) {
    if (context_tokens <= 0 || active_slots <= 0) return 0;
    const int full_layers = estimate_full_attention_layer_count(cfg);
    const size_t elems =
        static_cast<size_t>(full_layers) *
        static_cast<size_t>(active_slots) *
        2 *
        static_cast<size_t>(std::max(0, cfg.num_key_value_heads)) *
        static_cast<size_t>(context_tokens) *
        static_cast<size_t>(std::max(0, cfg.head_dim));
    return elems * sizeof(uint16_t); // bf16
}

size_t estimate_linear_state_bytes(const Qwen35Config& cfg) {
    const bool has_linear =
        cfg.linear_num_value_heads > 0 &&
        cfg.linear_num_key_heads > 0 &&
        cfg.linear_key_head_dim > 0 &&
        cfg.linear_value_head_dim > 0 &&
        cfg.linear_conv_kernel_dim > 0;
    if (!has_linear) return 0;

    const int full_layers = estimate_full_attention_layer_count(cfg);
    const int linear_layers = std::max(0, cfg.num_hidden_layers - full_layers);
    if (linear_layers == 0) return 0;

    const size_t lin_key_dim = static_cast<size_t>(cfg.linear_num_key_heads) * static_cast<size_t>(cfg.linear_key_head_dim);
    const size_t lin_value_dim = static_cast<size_t>(cfg.linear_num_value_heads) * static_cast<size_t>(cfg.linear_value_head_dim);
    const size_t lin_conv_dim = (lin_key_dim * 2) + lin_value_dim;

    const size_t conv_elems =
        static_cast<size_t>(std::max(0, cfg.linear_conv_kernel_dim - 1)) *
        lin_conv_dim;
    const size_t ssm_elems =
        static_cast<size_t>(cfg.linear_num_value_heads) *
        static_cast<size_t>(cfg.linear_value_head_dim) *
        static_cast<size_t>(cfg.linear_key_head_dim);
    const size_t per_layer_elems = checked_add_size_t(conv_elems, ssm_elems, "linear state bytes");
    const size_t total_elems = per_layer_elems * static_cast<size_t>(linear_layers);
    return total_elems * sizeof(uint16_t); // bf16
}

std::optional<MemoryBudgetEstimate> estimate_memory_budget(
    const std::string& model_path,
    const Qwen35Config& cfg
) {
    const std::optional<size_t> memory_size = device_memory_size_bytes();
    if (!memory_size.has_value()) return std::nullopt;
    const std::optional<size_t> recommended = recommended_working_set_bytes();
    const int context_tokens = env_positive_int("TALU_METAL_MEMORY_CONTEXT_TOKENS", env_positive_int("TOKENS", 200));
    const int active_slots = env_positive_int(
        "TALU_METAL_MAX_BATCH_SIZE",
        env_positive_int("TALU_CUDA_MAX_BATCH_SIZE", 8)
    );

    MemoryBudgetEstimate estimate{};
    estimate.memory_size_bytes = *memory_size;
    estimate.max_recommended_bytes = recommended.has_value() ? *recommended : 0;
    estimate.effective_budget_bytes = (estimate.memory_size_bytes * 8) / 10; // 0.8 * total unified memory
    estimate.context_tokens = std::max(1, context_tokens);
    estimate.active_slots = std::max(1, active_slots);
    estimate.weight_bytes = model_weight_bytes_on_disk(model_path);
    estimate.kv_cache_bytes = estimate_kv_cache_bytes(cfg, estimate.context_tokens, estimate.active_slots);
    estimate.linear_state_bytes = estimate_linear_state_bytes(cfg);
    estimate.overhead_bytes = estimate.weight_bytes / 10; // 10% weight overhead
    estimate.total_required_bytes = estimate.weight_bytes;
    estimate.total_required_bytes = checked_add_size_t(estimate.total_required_bytes, estimate.kv_cache_bytes, "total memory budget");
    estimate.total_required_bytes = checked_add_size_t(estimate.total_required_bytes, estimate.linear_state_bytes, "total memory budget");
    estimate.total_required_bytes = checked_add_size_t(estimate.total_required_bytes, estimate.overhead_bytes, "total memory budget");
    return estimate;
}

bool validate_memory_budget_or_set_error(const std::string& model_path, const Qwen35Config& cfg) {
    const std::optional<MemoryBudgetEstimate> estimate_opt = estimate_memory_budget(model_path, cfg);
    if (!estimate_opt.has_value()) return true;

    const MemoryBudgetEstimate& estimate = *estimate_opt;
    if (estimate.total_required_bytes <= estimate.effective_budget_bytes) return true;

    auto to_gib = [](size_t bytes) {
        return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    };
    char buf[1024];
    std::snprintf(
        buf,
        sizeof(buf),
        "metal memory budget exceeded: required=%.2f GiB (weights=%.2f, kv=%.2f, linear=%.2f, overhead=%.2f, context_tokens=%d, slots=%d) > budget=%.2f GiB (memory_size=%.2f GiB, max_recommended=%.2f GiB)",
        to_gib(estimate.total_required_bytes),
        to_gib(estimate.weight_bytes),
        to_gib(estimate.kv_cache_bytes),
        to_gib(estimate.linear_state_bytes),
        to_gib(estimate.overhead_bytes),
        estimate.context_tokens,
        estimate.active_slots,
        to_gib(estimate.effective_budget_bytes),
        to_gib(estimate.memory_size_bytes),
        to_gib(estimate.max_recommended_bytes)
    );
    g_last_error = buf;
    return false;
}

array run_full_attention_layer(mlx_ctx* ctx, int layer_idx, const array& x_norm) {
    LayerWeights& lw = ctx->layers[static_cast<size_t>(layer_idx)];
    KVCacheState& cache = ctx->kv_cache[static_cast<size_t>(layer_idx)];

    const int B = x_norm.shape(0);
    const int S = x_norm.shape(1);

    const bool prefill_only_qmm = false;
    const array q_proj_out = linear_decode_maybe_quantized(
        ctx->fp8_decode_qmm_enabled,
        ctx->cfg.quant_group_size,
        ctx->cfg.quant_bits,
        x_norm,
        lw.attn_q_rhs,
        lw.attn_q_has_q,
        lw.attn_q_q_w,
        lw.attn_q_q_scales,
        lw.attn_q_q_biases,
        prefill_only_qmm,
        layer_idx,
        "attn.q_proj"
    );
    const int q_last_dim = q_proj_out.shape(2);
    const int q_base_dim = lw.attn_num_heads * lw.attn_head_dim;
    array queries_raw = array(0.0f);
    array gate = array(0.0f);
    bool has_query_gate = false;
    if (q_last_dim == 2 * q_base_dim) {
        // Qwen3.5 full-attention branch: q projection carries both q and gate.
        has_query_gate = true;
        const array q_reshaped = reshape(q_proj_out, {B, S, lw.attn_num_heads, 2 * lw.attn_head_dim});
        queries_raw = slice(q_reshaped, {0, 0, 0, 0}, {B, S, lw.attn_num_heads, lw.attn_head_dim});
        const array gate_raw = slice(
            q_reshaped,
            {0, 0, 0, lw.attn_head_dim},
            {B, S, lw.attn_num_heads, 2 * lw.attn_head_dim}
        );
        gate = reshape(gate_raw, {B, S, q_base_dim});
    } else if (q_last_dim == q_base_dim) {
        // Qwen3 attention branch: q projection has no gating half.
        queries_raw = reshape(q_proj_out, {B, S, lw.attn_num_heads, lw.attn_head_dim});
    } else {
        throw std::runtime_error("unexpected self_attn.q_proj output shape");
    }

    const array k_proj_out = linear_decode_maybe_quantized(
        ctx->fp8_decode_qmm_enabled,
        ctx->cfg.quant_group_size,
        ctx->cfg.quant_bits,
        x_norm,
        lw.attn_k_rhs,
        lw.attn_k_has_q,
        lw.attn_k_q_w,
        lw.attn_k_q_scales,
        lw.attn_k_q_biases,
        prefill_only_qmm
    );
    const array v_proj_out = linear_decode_maybe_quantized(
        ctx->fp8_decode_qmm_enabled,
        ctx->cfg.quant_group_size,
        ctx->cfg.quant_bits,
        x_norm,
        lw.attn_v_rhs,
        lw.attn_v_has_q,
        lw.attn_v_q_w,
        lw.attn_v_q_scales,
        lw.attn_v_q_biases,
        prefill_only_qmm
    );

    array queries = fast::rms_norm(queries_raw, std::optional<array>(lw.attn_q_norm_w), ctx->cfg.rms_norm_eps);
    array keys = reshape(k_proj_out, {B, S, lw.attn_num_kv_heads, lw.attn_head_dim});
    keys = fast::rms_norm(keys, std::optional<array>(lw.attn_k_norm_w), ctx->cfg.rms_norm_eps);
    array values = reshape(v_proj_out, {B, S, lw.attn_num_kv_heads, lw.attn_head_dim});
    if (ctx->cfg.use_v_norm) {
        values = fast::rms_norm(values, std::nullopt, ctx->cfg.rms_norm_eps);
    }

    queries = transpose(queries, {0, 2, 1, 3});
    keys = transpose(keys, {0, 2, 1, 3});
    values = transpose(values, {0, 2, 1, 3});

    KVCacheState* read_cache = &cache;
    bool use_shared_read_cache = false;
    if (lw.kv_shared_source_layer >= 0) {
        const int src = lw.kv_shared_source_layer;
        if (src >= 0 && src < static_cast<int>(ctx->kv_cache.size())) {
            KVCacheState* src_cache = &ctx->kv_cache[static_cast<size_t>(src)];
            if (src_cache->keys.has_value() && src_cache->values.has_value() && src_cache->offset > 0) {
                read_cache = src_cache;
                use_shared_read_cache = true;
            }
        }
    }
    int rope_offset = cache.offset;
    if (use_shared_read_cache) {
        // Shared-KV readers use the source cache offset before the current
        // token block update. The caller passes the destination layer's token
        // block size separately, so reconstruct the source RoPE offset here.
        rope_offset = std::max(0, read_cache->offset - S);
    }
    const int rope_dims = static_cast<int>(std::round(lw.attn_head_dim * lw.attn_partial_rotary_factor));
    // Some architectures rotate only a leading subspace inside a larger
    // logical pairing domain. In that case:
    // - `lw.attn_head_dim` defines the half-split pairing topology
    // - `rope_dims` defines how many leading pairs are active
    //
    // A plain RoPE call with `dim = rope_dims` would rotate x[i] with
    // x[i + rope_dims/2], which is a different operator. The correct mapping
    // keeps the full-head pairing stride x[i] <-> x[i + head_dim/2], but only
    // for the first `rope_dims / 2` pairs. Implement that by gathering the
    // active first-half and matching second-half slices, applying standard RoPE
    // on the gathered view, then scattering the rotated values back.
    const bool use_proportional_full_head_rope =
        ctx->cfg.global_head_dim > 0 &&
        lw.attn_head_dim == ctx->cfg.global_head_dim &&
        rope_dims > 0 && rope_dims < lw.attn_head_dim;
    if (use_proportional_full_head_rope) {
        const int hd = lw.attn_head_dim;
        const int half_rope = rope_dims / 2;
        const int half_hd = hd / 2;
        // queries/keys shape here: [B, n_heads, S, hd] (already transposed)
        auto apply_rope_full_head = [&](array t) -> array {
            const int nh = t.shape(1);
            array t_first = slice(t, {0,0,0,0},               {B,nh,S,half_rope});
            array t_pass1 = slice(t, {0,0,0,half_rope},        {B,nh,S,half_hd});
            array t_third = slice(t, {0,0,0,half_hd},          {B,nh,S,half_hd+half_rope});
            array t_pass2 = slice(t, {0,0,0,half_hd+half_rope},{B,nh,S,hd});
            array rope_in = concatenate({t_first, t_third}, 3);
            rope_in = fast::rope(rope_in, rope_dims, false,
                                 std::optional<float>(lw.attn_rope_theta), 1.0f, rope_offset);
            array first_rot = slice(rope_in, {0,0,0,0},        {B,nh,S,half_rope});
            array third_rot = slice(rope_in, {0,0,0,half_rope},{B,nh,S,rope_dims});
            return concatenate({first_rot, t_pass1, third_rot, t_pass2}, 3);
        };
        queries = apply_rope_full_head(queries);
        keys = apply_rope_full_head(keys);
    } else {
        queries = fast::rope(queries, rope_dims, false,
                             std::optional<float>(lw.attn_rope_theta), 1.0f, rope_offset);
        keys = fast::rope(keys, rope_dims, false,
                          std::optional<float>(lw.attn_rope_theta), 1.0f, rope_offset);
    }

    const int prev = cache.offset;
    if (!use_shared_read_cache) {
        if (!cache.keys.has_value()) {
            cache.keys = keys;
            cache.values = values;
            cache.offset = S;
        } else if ((prev + S) > cache.keys->shape(2)) {
            const int n_steps = (cache.step + S - 1) / cache.step;
            const int extra = n_steps * cache.step;
            const array new_k = zeros({B, lw.attn_num_kv_heads, extra, lw.attn_head_dim}, keys.dtype());
            const array new_v = zeros({B, lw.attn_num_kv_heads, extra, lw.attn_head_dim}, values.dtype());

            if (cache.keys.has_value()) {
                if (prev % cache.step != 0) {
                    cache.keys = slice(*cache.keys, {0, 0, 0, 0}, {B, lw.attn_num_kv_heads, prev, lw.attn_head_dim});
                    cache.values = slice(*cache.values, {0, 0, 0, 0}, {B, lw.attn_num_kv_heads, prev, lw.attn_head_dim});
                }
                cache.keys = concatenate({*cache.keys, new_k}, 2);
                cache.values = concatenate({*cache.values, new_v}, 2);
            } else {
                cache.keys = new_k;
                cache.values = new_v;
            }
            cache.keys = slice_update(
                *cache.keys,
                keys,
                {0, 0, prev, 0},
                {B, lw.attn_num_kv_heads, prev + S, lw.attn_head_dim}
            );
            cache.values = slice_update(
                *cache.values,
                values,
                {0, 0, prev, 0},
                {B, lw.attn_num_kv_heads, prev + S, lw.attn_head_dim}
            );
            cache.offset = prev + S;
        } else {
            cache.keys = slice_update(
                *cache.keys,
                keys,
                {0, 0, prev, 0},
                {B, lw.attn_num_kv_heads, prev + S, lw.attn_head_dim}
            );
            cache.values = slice_update(
                *cache.values,
                values,
                {0, 0, prev, 0},
                {B, lw.attn_num_kv_heads, prev + S, lw.attn_head_dim}
            );
            cache.offset = prev + S;
        }
    } else {
        cache.offset = read_cache->offset;
    }

    if (!read_cache->keys.has_value() || !read_cache->values.has_value() || read_cache->offset <= 0) {
        throw std::runtime_error("invalid KV read cache state");
    }
    const int read_offset = read_cache->offset;
    const int window = (lw.attn_is_sliding && ctx->cfg.sliding_window > 0) ? ctx->cfg.sliding_window : 0;
    const int read_start = (window > 0 && read_offset > window) ? (read_offset - window) : 0;
    const array keys_view = slice(
        *read_cache->keys,
        {0, 0, read_start, 0},
        {B, lw.attn_num_kv_heads, read_offset, lw.attn_head_dim}
    );
    const array values_view = slice(
        *read_cache->values,
        {0, 0, read_start, 0},
        {B, lw.attn_num_kv_heads, read_offset, lw.attn_head_dim}
    );

    const std::string mask_mode = (S > 1) ? "causal" : "";
    const bool force_f32_attn =
        env_truthy("TALU_METAL_GEMMA4_FORCE_F32_ATTN", false) &&
        ctx->cfg.hidden_size_per_layer_input > 0;
    array attn_out = array(0.0f);
    if (force_f32_attn) {
        const array q_f32 = astype(queries, float32);
        const array k_f32 = astype(keys_view, float32);
        const array v_f32 = astype(values_view, float32);
        const array attn_f32 = fast::scaled_dot_product_attention(
            q_f32,
            k_f32,
            v_f32,
            lw.attn_scale,
            mask_mode
        );
        attn_out = astype(attn_f32, queries.dtype());
    } else {
        attn_out = fast::scaled_dot_product_attention(
            queries,
            keys_view,
            values_view,
            lw.attn_scale,
            mask_mode
        );
    }

    attn_out = transpose(attn_out, {0, 2, 1, 3});
    attn_out = reshape(attn_out, {B, S, lw.attn_num_heads * lw.attn_head_dim});
    if (has_query_gate) {
        attn_out = attn_out * sigmoid(gate);
    }

    return linear_decode_maybe_quantized(
        ctx->fp8_decode_qmm_enabled,
        ctx->cfg.quant_group_size,
        ctx->cfg.quant_bits,
        attn_out,
        lw.attn_o_rhs,
        lw.attn_o_has_q,
        lw.attn_o_q_w,
        lw.attn_o_q_scales,
        lw.attn_o_q_biases,
        prefill_only_qmm
    );
}

array run_linear_attention_layer(mlx_ctx* ctx, int layer_idx, const array& x_norm) {
    LayerWeights& lw = ctx->layers[static_cast<size_t>(layer_idx)];
    LinearCacheState& cache = ctx->linear_cache[static_cast<size_t>(layer_idx)];

    const int B = x_norm.shape(0);
    const int S = x_norm.shape(1);

    const bool prefill_only_qmm = false;
    const array qkvz = linear_decode_maybe_quantized(
        ctx->fp8_decode_qmm_enabled,
        ctx->cfg.quant_group_size,
        ctx->cfg.quant_bits,
        x_norm,
        lw.lin_in_proj_qkvz_rhs,
        lw.lin_in_proj_qkvz_has_q,
        lw.lin_in_proj_qkvz_q_w,
        lw.lin_in_proj_qkvz_q_scales,
        lw.lin_in_proj_qkvz_q_biases,
        prefill_only_qmm
    ); // [B,S,conv_dim+value_dim]
    const array qkv = slice(qkvz, {0, 0, 0}, {B, S, lw.lin_conv_dim});
    const array z_flat = slice(qkvz, {0, 0, lw.lin_conv_dim}, {B, S, lw.lin_conv_dim + lw.lin_value_dim});
    const array z = reshape(z_flat, {B, S, lw.lin_num_v_heads, lw.lin_head_v_dim});

    const array ba = linear_decode_maybe_quantized(
        ctx->fp8_decode_qmm_enabled,
        ctx->cfg.quant_group_size,
        ctx->cfg.quant_bits,
        x_norm,
        lw.lin_in_proj_ba_rhs,
        lw.lin_in_proj_ba_has_q,
        lw.lin_in_proj_ba_q_w,
        lw.lin_in_proj_ba_q_scales,
        lw.lin_in_proj_ba_q_biases,
        prefill_only_qmm
    ); // [B,S,2*Hv]
    const array b = slice(ba, {0, 0, 0}, {B, S, lw.lin_num_v_heads});
    const array a = slice(ba, {0, 0, lw.lin_num_v_heads}, {B, S, 2 * lw.lin_num_v_heads});

    if (!cache.conv_state.has_value()) {
        cache.conv_state = zeros({B, lw.lin_conv_kernel - 1, lw.lin_conv_dim}, x_norm.dtype());
    }
    if (!cache.state.has_value()) {
        cache.state = zeros({B, lw.lin_num_v_heads, lw.lin_head_v_dim, lw.lin_head_k_dim}, x_norm.dtype());
    }

    const array conv_input = concatenate({*cache.conv_state, qkv}, 1);
    cache.conv_state = slice(
        conv_input,
        {0, S, 0},
        {B, S + lw.lin_conv_kernel - 1, lw.lin_conv_dim}
    );

    array conv_out = conv1d(conv_input, lw.lin_conv1d_w, 1, 0, 1, lw.lin_conv_dim);
    conv_out = silu(conv_out);

    const array q_flat = slice(conv_out, {0, 0, 0}, {B, S, lw.lin_key_dim});
    const array k_flat = slice(conv_out, {0, 0, lw.lin_key_dim}, {B, S, 2 * lw.lin_key_dim});
    const array v_flat = slice(conv_out, {0, 0, 2 * lw.lin_key_dim}, {B, S, 2 * lw.lin_key_dim + lw.lin_value_dim});

    array q = reshape(q_flat, {B, S, lw.lin_num_k_heads, lw.lin_head_k_dim});
    array k = reshape(k_flat, {B, S, lw.lin_num_k_heads, lw.lin_head_k_dim});
    array v = reshape(v_flat, {B, S, lw.lin_num_v_heads, lw.lin_head_v_dim});

    const float inv_scale = 1.0f / std::sqrt(static_cast<float>(lw.lin_head_k_dim));
    q = fast::rms_norm(q, std::nullopt, 1.0e-6f) * array(inv_scale * inv_scale, q.dtype());
    k = fast::rms_norm(k, std::nullopt, 1.0e-6f) * array(inv_scale, k.dtype());

    auto [out, new_state] = gated_delta_update(q, k, v, a, b, lw.lin_A_exp_f32, lw.lin_dt_bias, cache.state);
    cache.state = new_state;

    const array normed = rmsnorm_gated(out, z, lw.lin_norm_w, ctx->cfg.rms_norm_eps);
    const array merged = reshape(normed, {B, S, lw.lin_value_dim});
    return linear_decode_maybe_quantized(
        ctx->fp8_decode_qmm_enabled,
        ctx->cfg.quant_group_size,
        ctx->cfg.quant_bits,
        merged,
        lw.lin_out_proj_rhs,
        lw.lin_out_proj_has_q,
        lw.lin_out_proj_q_w,
        lw.lin_out_proj_q_scales,
        lw.lin_out_proj_q_biases,
        prefill_only_qmm
    );
}

array run_shortconv_layer(mlx_ctx* ctx, int layer_idx, const array& x_norm) {
    LayerWeights& lw = ctx->layers[static_cast<size_t>(layer_idx)];
    LinearCacheState& cache = ctx->linear_cache[static_cast<size_t>(layer_idx)];

    const int B = x_norm.shape(0);
    const int S = x_norm.shape(1);

    const bool prefill_only_qmm = false;
    const array proj = linear_decode_maybe_quantized(
        ctx->fp8_decode_qmm_enabled,
        ctx->cfg.quant_group_size,
        ctx->cfg.quant_bits,
        x_norm,
        lw.sc_in_proj_rhs,
        lw.sc_in_proj_has_q,
        lw.sc_in_proj_q_w,
        lw.sc_in_proj_q_scales,
        lw.sc_in_proj_q_biases,
        prefill_only_qmm
    ); // [B,S,3*conv_dim]
    const array b_gate = slice(proj, {0, 0, 0}, {B, S, lw.sc_conv_dim});
    const array c_gate = slice(proj, {0, 0, lw.sc_conv_dim}, {B, S, 2 * lw.sc_conv_dim});
    const array x_proj = slice(proj, {0, 0, 2 * lw.sc_conv_dim}, {B, S, 3 * lw.sc_conv_dim});
    const array bx = b_gate * x_proj;

    if (!cache.conv_state.has_value()) {
        cache.conv_state = zeros({B, lw.sc_conv_kernel - 1, lw.sc_conv_dim}, x_norm.dtype());
    }

    const array conv_input = concatenate({*cache.conv_state, bx}, 1);
    cache.conv_state = slice(
        conv_input,
        {0, S, 0},
        {B, S + lw.sc_conv_kernel - 1, lw.sc_conv_dim}
    );

    array conv_out = conv1d(conv_input, lw.sc_conv1d_w, 1, 0, 1, lw.sc_conv_dim);
    if (lw.sc_has_bias) {
        conv_out = conv_out + reshape(lw.sc_conv_bias, {1, 1, lw.sc_conv_dim});
    }

    const array gated = c_gate * conv_out;
    return linear_decode_maybe_quantized(
        ctx->fp8_decode_qmm_enabled,
        ctx->cfg.quant_group_size,
        ctx->cfg.quant_bits,
        gated,
        lw.sc_out_proj_rhs,
        lw.sc_out_proj_has_q,
        lw.sc_out_proj_q_w,
        lw.sc_out_proj_q_scales,
        lw.sc_out_proj_q_biases,
        prefill_only_qmm
    );
}

array run_layer(
    mlx_ctx* ctx,
    int layer_idx,
    const array& hidden,
    const array& source_embeddings,
    const array& input_ids,
    const TraceFrame* trace_frame
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

    const array x_norm = fast::rms_norm(hidden, std::optional<array>(lw.input_layernorm_w), ctx->cfg.rms_norm_eps);

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
    const array h = has_gemma_four_norm
        ? (hidden + fast::rms_norm(residual_branch, std::optional<array>(lw.post_attention_layernorm_w), ctx->cfg.rms_norm_eps))
        : (hidden + residual_branch);

    const array mlp_in = has_gemma_four_norm
        ? fast::rms_norm(h, std::optional<array>(lw.pre_feedforward_layernorm_w), ctx->cfg.rms_norm_eps)
        : fast::rms_norm(h, std::optional<array>(lw.post_attention_layernorm_w), ctx->cfg.rms_norm_eps);
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
    const array ff = (ctx->cfg.use_gelu ? gelu_approx(gate) : silu(gate)) * up;
    const array down = linear_decode_maybe_quantized(
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

array forward_hidden(
    mlx_ctx* ctx,
    const array& input_ids,
    bool apply_final_norm,
    const TraceFrame* trace_frame = nullptr
) {
    array hidden = take(ctx->embed_tokens, input_ids, 0); // [B,S,H]
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
        for (int i = 0; i < static_cast<int>(ctx->layers.size()); ++i) {
            hidden = run_layer(ctx, i, hidden, source_embeddings, input_ids, trace_frame);
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

    const array hidden = forward_hidden(ctx, input_ids, true);
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
        const std::vector<array> topk_outputs = (*topk_fn)({logits_chunk});
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
    return {
        reshape(merged_logits, {merged_logits.shape(2)}),
        reshape(astype(merged_ids_i32, uint32), {merged_ids_i32.shape(2)})
    };
}

void prefill_prefix_chunks(mlx_ctx* ctx, const std::vector<int32_t>& prompt_vec, int32_t prefix_len) {
    if (prefix_len <= 0) return;
    const int chunk_size = resolve_prefill_chunk_size();
    int32_t offset = 0;
    while (offset < prefix_len) {
        const int32_t remaining = prefix_len - offset;
        const int32_t take_count = std::min<int32_t>(remaining, chunk_size);
        const auto* chunk_ptr = prompt_vec.data() + static_cast<size_t>(offset);
        const array prefix_chunk(chunk_ptr, Shape{1, take_count}, int32);
        const array prefill_hidden = forward_hidden(ctx, prefix_chunk, false);
        eval(prefill_hidden);
        synchronize();
        offset += take_count;
    }
}

void reset_runtime_state(mlx_ctx* ctx) {
    ctx->stream_ready = false;
    ctx->trace_decode_token = 1;
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

} // namespace

extern "C" {

int32_t mlx_is_available(void) {
    return metal::is_available() ? 1 : 0;
}

int32_t mlx_validate_config(const char* model_path) {
    if (!model_path) {
        g_last_error = "mlx_validate_config: model_path is null";
        return 0;
    }
    try {
        if (!std::filesystem::exists(model_path)) {
            throw std::runtime_error(std::string("resolved model_path does not exist: ") + model_path);
        }
        const ParsedModelConfig parsed = parse_qwen35_config(model_path);
        if (!validate_memory_budget_or_set_error(model_path, parsed.cfg)) {
            return 0;
        }
        return 1;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 0;
    } catch (...) {
        g_last_error = "unknown error in mlx_validate_config";
        return 0;
    }
}

mlx_ctx* mlx_create(const char* model_id, const char* model_path, int32_t seed) {
    bool wired_limit_acquired = false;
    bool rng_entered = false;
    const bool qmm_debug = []() {
        if (const char* value = std::getenv("TALU_METAL_QMM_DEBUG")) {
            return std::string(value) == "1";
        }
        return false;
    }();
    const bool nvfp4_debug = []() {
        if (const char* value = std::getenv("TALU_METAL_NVFP4_DEBUG")) {
            return std::string(value) == "1";
        }
        return false;
    }();
    const size_t qmm_attempts_before = g_qmm_attempts;
    const size_t qmm_success_before = g_qmm_success;
    try {
        if (!metal::is_available()) {
            g_last_error = "Metal backend is not available";
            return nullptr;
        }
        if (!model_id || !model_path) {
            g_last_error = "mlx_create: model_id/model_path is null";
            return nullptr;
        }

        if (!std::filesystem::exists(model_path)) {
            throw std::runtime_error(std::string("resolved model_path does not exist: ") + model_path);
        }

        set_default_device(Device::gpu);
        enter_rng_epoch(seed);
        rng_entered = true;

        auto ctx = std::make_unique<mlx_ctx>();
        ctx->model_id = std::string(model_id);
        ctx->model_path = std::string(model_path);
        if (const char* env = std::getenv("TALU_METAL_PROFILE_LAYERS")) {
            ctx->profile_layers = std::string(env) == "1";
        }

        // Cap Metal wired memory only when explicitly requested. Large local
        // models can exceed the recommended working-set ceiling during init.
        const bool enable_wired_limit = env_truthy("TALU_METAL_WIRED_LIMIT", false);
        if (enable_wired_limit) {
            wired_limit_acquired = acquire_wired_limit();
        }

        const ParsedModelConfig parsed_cfg = parse_qwen35_config(ctx->model_path);
        ctx->cfg = parsed_cfg.cfg;
        if (!validate_memory_budget_or_set_error(ctx->model_path, ctx->cfg)) {
            return nullptr;
        }

        auto tensors = load_weight_tensors(ctx->model_path);
        sanitize_qwen35_tensors(tensors, ctx->cfg.tie_word_embeddings, parsed_cfg.allow_qwen_norm_shift);
        const bool has_fp8_meta = has_fp8_quant_metadata(tensors);
        ctx->has_fp8_meta = has_fp8_meta;
        const bool has_mxfp8_meta = has_mxfp8_quant_metadata(tensors);
        ctx->has_mxfp8_meta = has_mxfp8_meta;
        bool has_grouped_affine_meta = has_grouped_affine_quant_metadata(tensors);
        const bool has_nvfp4_meta = has_nvfp4_quant_metadata(tensors);
        if (has_grouped_affine_meta && has_nvfp4_meta) {
            // Hybrid artifacts preserve grouped-affine execution tensors.
            ctx->cfg.quant_bits = 4;
            ctx->cfg.quant_group_size = 32;
        } else if (has_nvfp4_meta && !has_grouped_affine_meta) {
            // Pure NVFP4 must stay on the native MLX NVFP4 path.
            ctx->cfg.quant_bits = 4;
            ctx->cfg.quant_group_size = 16;
        }
        ctx->has_grouped_affine_meta = has_grouped_affine_meta;
        // Decode-time QMM defaults on for quantized checkpoints:
        // - native MXFP8 metadata
        // - grouped-affine metadata
        // - nvfp4 packed metadata
        // Callers can still force on/off via TALU_METAL_FP8_DECODE_QMM.
        const bool default_decode_qmm = has_mxfp8_meta || has_grouped_affine_meta || has_nvfp4_meta;
        ctx->fp8_decode_qmm_enabled = env_truthy("TALU_METAL_FP8_DECODE_QMM", default_decode_qmm);
        const bool enable_mlp_qmm = ctx->fp8_decode_qmm_enabled;
        // v3 cache/runtime policy: quantized path is allowed for prefill too,
        // so dense RHS is only retained when explicitly requested.
        const bool default_keep_dense_rhs_for_prefill = ctx->cfg.hidden_size_per_layer_input > 0;
        const bool keep_dense_rhs_for_prefill = env_truthy(
            "TALU_METAL_KEEP_DENSE_RHS_FOR_PREFILL",
            default_keep_dense_rhs_for_prefill
        );
        auto load_embed_weight = [&](const std::string& weight_key) -> array {
            return load_linear_weight(tensors, ctx->cfg, weight_key);
        };

        auto load_rhs_weight = [&](const std::string& weight_key) -> array {
            const array dense = load_linear_weight(tensors, ctx->cfg, weight_key);
            return to_rhs(dense, weight_key);
        };

        auto maybe_quantize_mxfp8_matrix = [&](
            const std::unordered_map<std::string, array>& tensors_ref,
            const Qwen35Config& cfg_ref,
            const std::string& weight_key,
            const array& lhs_weight,
            bool enabled,
            array* out_q_w,
            array* out_q_scales,
            array* out_q_biases,
            bool* out_has_q,
            int* out_detected_bits = nullptr
        ) {
            ::maybe_quantize_mxfp8_matrix(
                tensors_ref,
                cfg_ref,
                weight_key,
                lhs_weight,
                enabled,
                out_q_w,
                out_q_scales,
                out_q_biases,
                out_has_q,
                out_detected_bits
            );
        };

        const std::string text_prefix = detect_text_prefix(tensors);
        const std::string layer_prefix = text_prefix + "layers.";
        ctx->embed_tokens = load_embed_weight(text_prefix + "embed_tokens.weight");
        ctx->final_norm_w = require_any_tensor(
            tensors,
            {
                text_prefix + "norm.weight",
                text_prefix + "embedding_norm.weight",
                "model.norm.weight",
                "model.embedding_norm.weight",
            },
            "final_norm"
        );

        array lm_head_weight = array(0.0f); // [vocab, hidden]
        std::string lm_head_key;
        if (ctx->cfg.tie_word_embeddings) {
            lm_head_weight = ctx->embed_tokens;
            lm_head_key = text_prefix + "embed_tokens.weight";
            if (!has_tensor(tensors, lm_head_key)) {
                lm_head_key = "model.embed_tokens.weight";
            }
        } else {
            if (has_tensor(tensors, text_prefix + "lm_head.weight")) {
                lm_head_key = text_prefix + "lm_head.weight";
                lm_head_weight = load_linear_weight(tensors, ctx->cfg, text_prefix + "lm_head.weight");
            } else if (has_tensor(tensors, "lm_head.weight")) {
                lm_head_key = "lm_head.weight";
                lm_head_weight = load_linear_weight(tensors, ctx->cfg, "lm_head.weight");
            } else {
                throw std::runtime_error("missing lm_head weight (tie_word_embeddings=false)");
            }
        }

        const bool enable_lm_head_qmm = env_truthy("TALU_METAL_FP8_LM_HEAD_QMM", has_mxfp8_meta || has_grouped_affine_meta || has_nvfp4_meta);
        if (enable_lm_head_qmm) {
            if (lm_head_weight.size() == 1) {
                // Tie embeddings can point at raw FP8 tensors. Load dequantized copy only
                // if direct packed MXFP8 path is unavailable.
                lm_head_weight = load_linear_weight(tensors, ctx->cfg, lm_head_key);
            }
            bool lm_head_has_q = false;
            int lm_head_detected_bits = 0;
            maybe_quantize_mxfp8_matrix(
                tensors,
                ctx->cfg,
                lm_head_key,
                lm_head_weight,
                true,
                &ctx->lm_head_q_w,
                &ctx->lm_head_q_scales,
                &ctx->lm_head_q_biases,
                &lm_head_has_q,
                &lm_head_detected_bits
            );
            ctx->lm_head_q_decode_enabled = lm_head_has_q;
            ctx->lm_head_q_bits = lm_head_detected_bits > 0 ? lm_head_detected_bits : ctx->cfg.quant_bits;
        }

        // Keep dense lm_head RHS only when no quantized matmul format is
        // available.  Grouped-affine and NVFP4 checkpoints support QMM for both
        // prefill and decode, so the dense transpose is unnecessary.
        if (keep_dense_rhs_for_prefill || !ctx->lm_head_q_decode_enabled ||
            (!ctx->has_fp8_meta && !ctx->has_grouped_affine_meta && !has_nvfp4_meta)) {
            if (lm_head_weight.size() == 1) lm_head_weight = load_linear_weight(tensors, ctx->cfg, lm_head_key);
            {
                array rhs = to_rhs(lm_head_weight, lm_head_key);
                if (ctx->cfg.hidden_size_per_layer_input > 0 && !ctx->lm_head_q_decode_enabled) {
                    rhs = stop_gradient(copy(astype(rhs, float32)));
                }
                ctx->lm_head_rhs = rhs;
            }
        }

        array per_layer_model_projection_weight = array(0.0f);
        if (ctx->cfg.hidden_size_per_layer_input > 0) {
            const int hpl = ctx->cfg.hidden_size_per_layer_input;
            const std::string per_layer_embed_key = has_tensor(tensors, text_prefix + "embed_tokens_per_layer.weight")
                ? text_prefix + "embed_tokens_per_layer.weight"
                : "model.embed_tokens_per_layer.weight";
            const std::string per_layer_model_proj_key = has_tensor(tensors, text_prefix + "per_layer_model_projection.weight")
                ? text_prefix + "per_layer_model_projection.weight"
                : "model.per_layer_model_projection.weight";
            const std::string per_layer_proj_norm_key = has_tensor(tensors, text_prefix + "per_layer_projection_norm.weight")
                ? text_prefix + "per_layer_projection_norm.weight"
                : "model.per_layer_projection_norm.weight";

            if (has_tensor(tensors, per_layer_embed_key) &&
                has_tensor(tensors, per_layer_model_proj_key) &&
                has_tensor(tensors, per_layer_proj_norm_key))
            {
                const bool disable_per_layer_input =
                    env_truthy("TALU_METAL_DISABLE_PER_LAYER_INPUT", false);
                if (!disable_per_layer_input) {
                    ctx->per_layer_embedding = load_linear_weight(tensors, ctx->cfg, per_layer_embed_key);
                    per_layer_model_projection_weight = load_linear_weight(tensors, ctx->cfg, per_layer_model_proj_key);
                    ctx->per_layer_projection_norm_w = require_tensor(tensors, per_layer_proj_norm_key);
                    ctx->per_layer_hidden_size = hpl;
                    ctx->per_layer_embed_scale = std::sqrt(static_cast<float>(hpl));
                    ctx->per_layer_model_projection_scale = 1.0f / std::sqrt(static_cast<float>(ctx->cfg.hidden_size));
                    ctx->per_layer_input_enabled = true;
                }
            }
        }

        ctx->layers.reserve(static_cast<size_t>(ctx->cfg.num_hidden_layers));
        ctx->kv_cache.resize(static_cast<size_t>(ctx->cfg.num_hidden_layers));
        ctx->linear_cache.resize(static_cast<size_t>(ctx->cfg.num_hidden_layers));
        const std::vector<int> shared_kv_sources = resolve_shared_kv_source_layers(ctx->cfg);

        for (int i = 0; i < ctx->cfg.num_hidden_layers; ++i) {
            LayerWeights lw;
            if (i >= 0 && i < static_cast<int>(shared_kv_sources.size())) {
                lw.kv_shared_source_layer = shared_kv_sources[static_cast<size_t>(i)];
            }
            const std::string p = layer_prefix + std::to_string(i) + ".";

            lw.input_layernorm_w = require_any_tensor(
                tensors,
                {
                    p + "input_layernorm.weight",
                    p + "operator_norm.weight",
                },
                p + "input_layernorm/operator_norm"
            );
            lw.post_attention_layernorm_w = require_any_tensor(
                tensors,
                {
                    p + "post_attention_layernorm.weight",
                    p + "ffn_norm.weight",
                },
                p + "post_attention_layernorm/ffn_norm"
            );
            if (has_tensor(tensors, p + "pre_feedforward_layernorm.weight")) {
                lw.pre_feedforward_layernorm_w = require_tensor(tensors, p + "pre_feedforward_layernorm.weight");
            }
            if (has_tensor(tensors, p + "post_feedforward_layernorm.weight")) {
                lw.post_feedforward_layernorm_w = require_tensor(tensors, p + "post_feedforward_layernorm.weight");
            }
            if (ctx->per_layer_input_enabled) {
                const int hpl = ctx->per_layer_hidden_size;
                const int row_start = i * hpl;
                const int row_end = row_start + hpl;
                if (per_layer_model_projection_weight.ndim() == 2 &&
                    per_layer_model_projection_weight.shape(0) >= row_end &&
                    per_layer_model_projection_weight.shape(1) == ctx->cfg.hidden_size &&
                    has_tensor(tensors, p + "per_layer_input_gate.weight") &&
                    has_tensor(tensors, p + "per_layer_projection.weight") &&
                    has_tensor(tensors, p + "post_per_layer_input_norm.weight") &&
                    has_tensor(tensors, p + "layer_scalar"))
                {
                    const array per_layer_model_projection_slice = slice(
                        per_layer_model_projection_weight,
                        {row_start, 0},
                        {row_end, per_layer_model_projection_weight.shape(1)}
                    );
                    lw.per_layer_model_projection_rhs = to_rhs(
                        per_layer_model_projection_slice,
                        p + "per_layer_model_projection.weight[slice]"
                    );
                    lw.per_layer_input_gate_rhs = load_rhs_weight(p + "per_layer_input_gate.weight");
                    lw.per_layer_projection_rhs = load_rhs_weight(p + "per_layer_projection.weight");
                    lw.post_per_layer_input_norm_w = require_tensor(tensors, p + "post_per_layer_input_norm.weight");
                    lw.layer_scalar = scalar_to_f32(require_tensor(tensors, p + "layer_scalar"), p + "layer_scalar");
                    lw.has_per_layer_input = true;
                }
            }

            const std::string mlp_gate_key = has_tensor(tensors, p + "mlp.gate_proj.weight")
                ? p + "mlp.gate_proj.weight"
                : p + "feed_forward.w1.weight";
            const std::string mlp_up_key = has_tensor(tensors, p + "mlp.up_proj.weight")
                ? p + "mlp.up_proj.weight"
                : p + "feed_forward.w3.weight";
            const std::string mlp_down_key = has_tensor(tensors, p + "mlp.down_proj.weight")
                ? p + "mlp.down_proj.weight"
                : p + "feed_forward.w2.weight";

            const array& mlp_gate_weight = require_tensor(tensors, mlp_gate_key); // [out,in] raw
            const array& mlp_up_weight = require_tensor(tensors, mlp_up_key); // [out,in] raw
            const array& mlp_down_weight = require_tensor(tensors, mlp_down_key); // [out,in] raw

            maybe_quantize_mxfp8_matrix(
                tensors,
                ctx->cfg,
                mlp_gate_key,
                mlp_gate_weight,
                enable_mlp_qmm,
                &lw.mlp_gate_q_w,
                &lw.mlp_gate_q_scales,
                &lw.mlp_gate_q_biases,
                &lw.mlp_gate_has_q
            );
            maybe_quantize_mxfp8_matrix(
                tensors,
                ctx->cfg,
                mlp_up_key,
                mlp_up_weight,
                enable_mlp_qmm,
                &lw.mlp_up_q_w,
                &lw.mlp_up_q_scales,
                &lw.mlp_up_q_biases,
                &lw.mlp_up_has_q
            );
            maybe_quantize_mxfp8_matrix(
                tensors,
                ctx->cfg,
                mlp_down_key,
                mlp_down_weight,
                enable_mlp_qmm,
                &lw.mlp_down_q_w,
                &lw.mlp_down_q_scales,
                &lw.mlp_down_q_biases,
                &lw.mlp_down_has_q
            );
            if (keep_dense_rhs_for_prefill || !enable_mlp_qmm || !lw.mlp_gate_has_q) {
                lw.mlp_gate_rhs = load_rhs_weight(mlp_gate_key);
            }
            if (keep_dense_rhs_for_prefill || !enable_mlp_qmm || !lw.mlp_up_has_q) {
                lw.mlp_up_rhs = load_rhs_weight(mlp_up_key);
            }
            if (keep_dense_rhs_for_prefill || !enable_mlp_qmm || !lw.mlp_down_has_q) {
                lw.mlp_down_rhs = load_rhs_weight(mlp_down_key);
            }

            const bool has_linear_split = has_tensor(tensors, p + "linear_attn.in_proj_qkv.weight");
            const bool has_linear_fused = has_tensor(tensors, p + "mixer.in_proj.weight");
            const bool has_linear = has_linear_split || has_linear_fused;
            const bool has_shortconv = has_tensor(tensors, p + "conv.in_proj.weight");
            const bool has_attn = has_tensor(tensors, p + "self_attn.q_proj.weight");
            const int branch_count = (has_linear ? 1 : 0) + (has_shortconv ? 1 : 0) + (has_attn ? 1 : 0);
            if (branch_count == 0) {
                std::string present;
                for (const auto& kv : tensors) {
                    if (kv.first.rfind(p, 0) != 0) continue;
                    if (kv.first.find("linear_attn") == std::string::npos &&
                        kv.first.find("mixer") == std::string::npos &&
                        kv.first.find("conv") == std::string::npos &&
                        kv.first.find("self_attn") == std::string::npos) {
                        continue;
                    }
                    if (!present.empty()) present += ", ";
                    present += kv.first;
                    if (present.size() > 1200) {
                        present += ", ...";
                        break;
                    }
                }
                throw std::runtime_error(
                    "layer has neither linear_attn, conv, nor self_attn: " + std::to_string(i) +
                    " (prefix=" + p +
                    ", has_linear_split=" + std::to_string(has_linear_split) +
                    ", has_linear_fused=" + std::to_string(has_linear_fused) +
                    ", has_shortconv=" + std::to_string(has_shortconv) +
                    ", has_attn=" + std::to_string(has_attn) +
                    ", present=[" + present + "])"
                );
            }
            if (branch_count > 1) {
                throw std::runtime_error("layer has multiple mixer branches unexpectedly: " + std::to_string(i));
            }

            lw.is_linear = has_linear;
            lw.is_shortconv = has_shortconv;

            if (lw.is_linear) {
                if (ctx->cfg.linear_num_value_heads <= 0 ||
                    ctx->cfg.linear_num_key_heads <= 0 ||
                    ctx->cfg.linear_key_head_dim <= 0 ||
                    ctx->cfg.linear_value_head_dim <= 0 ||
                    ctx->cfg.linear_conv_kernel_dim <= 0) {
                    throw std::runtime_error("missing linear_attn config fields for hybrid layer");
                }
                lw.lin_num_v_heads = ctx->cfg.linear_num_value_heads;
                lw.lin_num_k_heads = ctx->cfg.linear_num_key_heads;
                lw.lin_head_k_dim = ctx->cfg.linear_key_head_dim;
                lw.lin_head_v_dim = ctx->cfg.linear_value_head_dim;
                lw.lin_key_dim = lw.lin_num_k_heads * lw.lin_head_k_dim;
                lw.lin_value_dim = lw.lin_num_v_heads * lw.lin_head_v_dim;
                lw.lin_conv_dim = lw.lin_key_dim * 2 + lw.lin_value_dim;
                lw.lin_conv_kernel = ctx->cfg.linear_conv_kernel_dim;

                lw.lin_conv1d_w = require_tensor(tensors, p + "linear_attn.conv1d.weight");
                if (has_linear_split) {
                    const std::string qkv_key = p + "linear_attn.in_proj_qkv.weight";
                    const std::string z_key = p + "linear_attn.in_proj_z.weight";
                    const std::string a_key = p + "linear_attn.in_proj_a.weight";
                    const std::string b_key = p + "linear_attn.in_proj_b.weight";
                    const array qkv_weight = load_linear_weight(tensors, ctx->cfg, qkv_key);
                    const array z_weight = load_linear_weight(tensors, ctx->cfg, z_key);
                    const array a_weight = load_linear_weight(tensors, ctx->cfg, a_key);
                    const array b_weight = load_linear_weight(tensors, ctx->cfg, b_key);
                    const array qkvz_weight = concatenate({qkv_weight, z_weight}, 0);
                    const array ba_weight = concatenate({b_weight, a_weight}, 0);
                    bool lin_qkvz_q_captured = false;
                    bool lin_ba_q_captured = false;
                    if (enable_mlp_qmm && env_truthy("TALU_METAL_GAFFINE_LOSSLESS", true)) {
                        // Split linear-attn checkpoints store grouped-affine metadata
                        // per source tensor. Capture each source tensor losslessly
                        // and concatenate packed payload + calibration metadata.
                        array qkv_q_w = array(0.0f);
                        array qkv_q_scales = array(0.0f);
                        array qkv_q_biases = array(0.0f);
                        array z_q_w = array(0.0f);
                        array z_q_scales = array(0.0f);
                        array z_q_biases = array(0.0f);
                        if (maybe_capture_grouped_affine_quant(
                                tensors,
                                ctx->cfg,
                                qkv_key,
                                qkv_weight,
                                &qkv_q_w,
                                &qkv_q_scales,
                                &qkv_q_biases) &&
                            maybe_capture_grouped_affine_quant(
                                tensors,
                                ctx->cfg,
                                z_key,
                                z_weight,
                                &z_q_w,
                                &z_q_scales,
                                &z_q_biases) &&
                            qkv_q_w.ndim() == 2 && z_q_w.ndim() == 2 &&
                            qkv_q_w.shape(1) == z_q_w.shape(1) &&
                            qkv_q_scales.ndim() == 2 && z_q_scales.ndim() == 2 &&
                            qkv_q_scales.shape(1) == z_q_scales.shape(1) &&
                            qkv_q_biases.ndim() == 2 && z_q_biases.ndim() == 2 &&
                            qkv_q_biases.shape(1) == z_q_biases.shape(1))
                        {
                            lw.lin_in_proj_qkvz_q_w = concatenate({qkv_q_w, z_q_w}, 0);
                            lw.lin_in_proj_qkvz_q_scales = concatenate({qkv_q_scales, z_q_scales}, 0);
                            lw.lin_in_proj_qkvz_q_biases = concatenate({qkv_q_biases, z_q_biases}, 0);
                            lw.lin_in_proj_qkvz_has_q = true;
                            lin_qkvz_q_captured = true;
                            g_qmm_attempts += 1;
                            g_qmm_success += 1;
                        }

                        array b_q_w = array(0.0f);
                        array b_q_scales = array(0.0f);
                        array b_q_biases = array(0.0f);
                        array a_q_w = array(0.0f);
                        array a_q_scales = array(0.0f);
                        array a_q_biases = array(0.0f);
                        if (maybe_capture_grouped_affine_quant(
                                tensors,
                                ctx->cfg,
                                b_key,
                                b_weight,
                                &b_q_w,
                                &b_q_scales,
                                &b_q_biases) &&
                            maybe_capture_grouped_affine_quant(
                                tensors,
                                ctx->cfg,
                                a_key,
                                a_weight,
                                &a_q_w,
                                &a_q_scales,
                                &a_q_biases) &&
                            b_q_w.ndim() == 2 && a_q_w.ndim() == 2 &&
                            b_q_w.shape(1) == a_q_w.shape(1) &&
                            b_q_scales.ndim() == 2 && a_q_scales.ndim() == 2 &&
                            b_q_scales.shape(1) == a_q_scales.shape(1) &&
                            b_q_biases.ndim() == 2 && a_q_biases.ndim() == 2 &&
                            b_q_biases.shape(1) == a_q_biases.shape(1))
                        {
                            lw.lin_in_proj_ba_q_w = concatenate({b_q_w, a_q_w}, 0);
                            lw.lin_in_proj_ba_q_scales = concatenate({b_q_scales, a_q_scales}, 0);
                            lw.lin_in_proj_ba_q_biases = concatenate({b_q_biases, a_q_biases}, 0);
                            lw.lin_in_proj_ba_has_q = true;
                            lin_ba_q_captured = true;
                            g_qmm_attempts += 1;
                            g_qmm_success += 1;
                        }
                    }
                    if (!lin_qkvz_q_captured && env_truthy("TALU_METAL_NVFP4_AFFINE_TRANSCODE", false)) {
                        array qkv_q_w = array(0.0f);
                        array qkv_q_scales = array(0.0f);
                        array qkv_q_biases = array(0.0f);
                        array z_q_w = array(0.0f);
                        array z_q_scales = array(0.0f);
                        array z_q_biases = array(0.0f);
                        if (transcode_nvfp4_to_affine_qmm(
                                tensors,
                                qkv_key,
                                qkv_weight,
                                ctx->cfg.quant_group_size > 0 ? ctx->cfg.quant_group_size : 32,
                                &qkv_q_w,
                                &qkv_q_scales,
                                &qkv_q_biases) &&
                            transcode_nvfp4_to_affine_qmm(
                                tensors,
                                z_key,
                                z_weight,
                                ctx->cfg.quant_group_size > 0 ? ctx->cfg.quant_group_size : 32,
                                &z_q_w,
                                &z_q_scales,
                                &z_q_biases) &&
                            qkv_q_w.ndim() == 2 && z_q_w.ndim() == 2 &&
                            qkv_q_w.shape(1) == z_q_w.shape(1) &&
                            qkv_q_scales.ndim() == 2 && z_q_scales.ndim() == 2 &&
                            qkv_q_scales.shape(1) == z_q_scales.shape(1) &&
                            qkv_q_biases.ndim() == 2 && z_q_biases.ndim() == 2 &&
                            qkv_q_biases.shape(1) == z_q_biases.shape(1))
                        {
                            lw.lin_in_proj_qkvz_q_w = concatenate({qkv_q_w, z_q_w}, 0);
                            lw.lin_in_proj_qkvz_q_scales = concatenate({qkv_q_scales, z_q_scales}, 0);
                            lw.lin_in_proj_qkvz_q_biases = concatenate({qkv_q_biases, z_q_biases}, 0);
                            lw.lin_in_proj_qkvz_has_q = true;
                            lin_qkvz_q_captured = true;
                            g_qmm_attempts += 1;
                            g_qmm_success += 1;
                        }
                    }
                    if (!lin_ba_q_captured && env_truthy("TALU_METAL_NVFP4_AFFINE_TRANSCODE", false)) {
                        array b_q_w = array(0.0f);
                        array b_q_scales = array(0.0f);
                        array b_q_biases = array(0.0f);
                        array a_q_w = array(0.0f);
                        array a_q_scales = array(0.0f);
                        array a_q_biases = array(0.0f);
                        if (transcode_nvfp4_to_affine_qmm(
                                tensors,
                                b_key,
                                b_weight,
                                ctx->cfg.quant_group_size > 0 ? ctx->cfg.quant_group_size : 32,
                                &b_q_w,
                                &b_q_scales,
                                &b_q_biases) &&
                            transcode_nvfp4_to_affine_qmm(
                                tensors,
                                a_key,
                                a_weight,
                                ctx->cfg.quant_group_size > 0 ? ctx->cfg.quant_group_size : 32,
                                &a_q_w,
                                &a_q_scales,
                                &a_q_biases) &&
                            b_q_w.ndim() == 2 && a_q_w.ndim() == 2 &&
                            b_q_w.shape(1) == a_q_w.shape(1) &&
                            b_q_scales.ndim() == 2 && a_q_scales.ndim() == 2 &&
                            b_q_scales.shape(1) == a_q_scales.shape(1) &&
                            b_q_biases.ndim() == 2 && a_q_biases.ndim() == 2 &&
                            b_q_biases.shape(1) == a_q_biases.shape(1))
                        {
                            lw.lin_in_proj_ba_q_w = concatenate({b_q_w, a_q_w}, 0);
                            lw.lin_in_proj_ba_q_scales = concatenate({b_q_scales, a_q_scales}, 0);
                            lw.lin_in_proj_ba_q_biases = concatenate({b_q_biases, a_q_biases}, 0);
                            lw.lin_in_proj_ba_has_q = true;
                            lin_ba_q_captured = true;
                            g_qmm_attempts += 1;
                            g_qmm_success += 1;
                        }
                    }
                    if (!lin_qkvz_q_captured) {
                        maybe_quantize_mxfp8_matrix(
                            tensors,
                            ctx->cfg,
                            p + "linear_attn.in_proj_qkvz",
                            qkvz_weight,
                            enable_mlp_qmm,
                            &lw.lin_in_proj_qkvz_q_w,
                            &lw.lin_in_proj_qkvz_q_scales,
                            &lw.lin_in_proj_qkvz_q_biases,
                            &lw.lin_in_proj_qkvz_has_q
                        );
                    }
                    if (!lin_ba_q_captured) {
                        maybe_quantize_mxfp8_matrix(
                            tensors,
                            ctx->cfg,
                            p + "linear_attn.in_proj_ba",
                            ba_weight,
                            enable_mlp_qmm,
                            &lw.lin_in_proj_ba_q_w,
                            &lw.lin_in_proj_ba_q_scales,
                            &lw.lin_in_proj_ba_q_biases,
                            &lw.lin_in_proj_ba_has_q
                        );
                    }
                    if (keep_dense_rhs_for_prefill || !enable_mlp_qmm || !lw.lin_in_proj_qkvz_has_q) {
                        const array qkv_rhs = to_rhs(qkv_weight, qkv_key);
                        const array z_rhs = to_rhs(z_weight, z_key);
                        lw.lin_in_proj_qkvz_rhs = concatenate({ qkv_rhs, z_rhs }, 1);
                    }
                    if (keep_dense_rhs_for_prefill || !enable_mlp_qmm || !lw.lin_in_proj_ba_has_q) {
                        const array b_rhs = to_rhs(b_weight, b_key);
                        const array a_rhs = to_rhs(a_weight, a_key);
                        lw.lin_in_proj_ba_rhs = concatenate({ b_rhs, a_rhs }, 1);
                    }
                } else {
                    const std::string fused_key = p + "mixer.in_proj.weight";
                    const array fused_in_proj_weight = load_linear_weight(tensors, ctx->cfg, fused_key); // [out, hidden]
                    if (fused_in_proj_weight.ndim() != 2) {
                        throw std::runtime_error("linear fused in_proj must be rank-2");
                    }
                    const int qkvz_cols = lw.lin_conv_dim + lw.lin_value_dim;
                    const int ba_cols = 2 * lw.lin_num_v_heads;
                    const int expected_cols = qkvz_cols + ba_cols;
                    const int fused_cols = fused_in_proj_weight.shape(0);
                    if (fused_cols != expected_cols) {
                        throw std::runtime_error(
                            "linear fused in_proj shape mismatch: expected cols=" +
                            std::to_string(expected_cols) + " got cols=" + std::to_string(fused_cols)
                        );
                    }
                    const array qkvz_weight = slice(
                        fused_in_proj_weight,
                        {0, 0},
                        {qkvz_cols, fused_in_proj_weight.shape(1)}
                    );
                    const array ba_weight = slice(
                        fused_in_proj_weight,
                        {qkvz_cols, 0},
                        {expected_cols, fused_in_proj_weight.shape(1)}
                    );
                    bool lin_qkvz_q_captured = false;
                    bool lin_ba_q_captured = false;
                    if (enable_mlp_qmm && env_truthy("TALU_METAL_GAFFINE_LOSSLESS", true)) {
                        // Fused linear-attn stores grouped-affine metadata for
                        // mixer.in_proj as one tensor. Slice packed payload +
                        // metadata rows losslessly into qkvz and ba branches.
                        array fused_q_w = array(0.0f);
                        array fused_q_scales = array(0.0f);
                        array fused_q_biases = array(0.0f);
                        if (maybe_capture_grouped_affine_quant(
                                tensors,
                                ctx->cfg,
                                fused_key,
                                fused_in_proj_weight,
                                &fused_q_w,
                                &fused_q_scales,
                                &fused_q_biases) &&
                            fused_q_w.ndim() == 2 &&
                            fused_q_scales.ndim() == 2 &&
                            fused_q_biases.ndim() == 2 &&
                            fused_q_w.shape(0) == expected_cols &&
                            fused_q_scales.shape(0) == expected_cols &&
                            fused_q_biases.shape(0) == expected_cols)
                        {
                            const int packed_cols = fused_q_w.shape(1);
                            const int group_cols = fused_q_scales.shape(1);
                            lw.lin_in_proj_qkvz_q_w = slice(fused_q_w, {0, 0}, {qkvz_cols, packed_cols});
                            lw.lin_in_proj_ba_q_w = slice(fused_q_w, {qkvz_cols, 0}, {expected_cols, packed_cols});
                            lw.lin_in_proj_qkvz_q_scales = slice(fused_q_scales, {0, 0}, {qkvz_cols, group_cols});
                            lw.lin_in_proj_ba_q_scales = slice(fused_q_scales, {qkvz_cols, 0}, {expected_cols, group_cols});
                            lw.lin_in_proj_qkvz_q_biases = slice(fused_q_biases, {0, 0}, {qkvz_cols, group_cols});
                            lw.lin_in_proj_ba_q_biases = slice(fused_q_biases, {qkvz_cols, 0}, {expected_cols, group_cols});
                            lw.lin_in_proj_qkvz_has_q = true;
                            lw.lin_in_proj_ba_has_q = true;
                            lin_qkvz_q_captured = true;
                            lin_ba_q_captured = true;
                            g_qmm_attempts += 2;
                            g_qmm_success += 2;
                        }
                    }
                    if (!lin_qkvz_q_captured) {
                        maybe_quantize_mxfp8_matrix(
                            tensors,
                            ctx->cfg,
                            p + "mixer.in_proj.qkvz_slice",
                            qkvz_weight,
                            enable_mlp_qmm,
                            &lw.lin_in_proj_qkvz_q_w,
                            &lw.lin_in_proj_qkvz_q_scales,
                            &lw.lin_in_proj_qkvz_q_biases,
                            &lw.lin_in_proj_qkvz_has_q
                        );
                    }
                    if (!lin_ba_q_captured) {
                        maybe_quantize_mxfp8_matrix(
                            tensors,
                            ctx->cfg,
                            p + "mixer.in_proj.ba_slice",
                            ba_weight,
                            enable_mlp_qmm,
                            &lw.lin_in_proj_ba_q_w,
                            &lw.lin_in_proj_ba_q_scales,
                            &lw.lin_in_proj_ba_q_biases,
                            &lw.lin_in_proj_ba_has_q
                        );
                    }
                    if ((keep_dense_rhs_for_prefill || !enable_mlp_qmm || !lw.lin_in_proj_qkvz_has_q) ||
                        (keep_dense_rhs_for_prefill || !enable_mlp_qmm || !lw.lin_in_proj_ba_has_q)) {
                        const array fused_in_proj_rhs = to_rhs(
                            fused_in_proj_weight,
                            fused_key
                        );
                        const int fused_rows = fused_in_proj_rhs.shape(0);
                        if (keep_dense_rhs_for_prefill || !enable_mlp_qmm || !lw.lin_in_proj_qkvz_has_q) {
                            lw.lin_in_proj_qkvz_rhs = slice(
                                fused_in_proj_rhs,
                                {0, 0},
                                {fused_rows, qkvz_cols}
                            );
                        }
                        if (keep_dense_rhs_for_prefill || !enable_mlp_qmm || !lw.lin_in_proj_ba_has_q) {
                            lw.lin_in_proj_ba_rhs = slice(
                                fused_in_proj_rhs,
                                {0, qkvz_cols},
                                {fused_rows, expected_cols}
                            );
                        }
                    }
                }
                lw.lin_A_log = require_tensor(tensors, p + "linear_attn.A_log");
                lw.lin_A_exp_f32 = exp(astype(lw.lin_A_log, float32));
                lw.lin_dt_bias = require_tensor(tensors, p + "linear_attn.dt_bias");
                lw.lin_norm_w = require_tensor(tensors, p + "linear_attn.norm.weight");
                const std::string out_proj_key = p + "linear_attn.out_proj.weight";
                const array& out_proj_weight = require_tensor(tensors, out_proj_key);
                maybe_quantize_mxfp8_matrix(
                    tensors,
                    ctx->cfg,
                    out_proj_key,
                    out_proj_weight,
                    enable_mlp_qmm,
                    &lw.lin_out_proj_q_w,
                    &lw.lin_out_proj_q_scales,
                    &lw.lin_out_proj_q_biases,
                    &lw.lin_out_proj_has_q
                );
                if (keep_dense_rhs_for_prefill || !enable_mlp_qmm || !lw.lin_out_proj_has_q) {
                    lw.lin_out_proj_rhs = load_rhs_weight(out_proj_key);
                }
            } else if (lw.is_shortconv) {
                const std::string sc_in_proj_key = p + "conv.in_proj.weight";
                const array& sc_in_proj_weight = require_tensor(tensors, sc_in_proj_key);
                maybe_quantize_mxfp8_matrix(
                    tensors,
                    ctx->cfg,
                    sc_in_proj_key,
                    sc_in_proj_weight,
                    enable_mlp_qmm,
                    &lw.sc_in_proj_q_w,
                    &lw.sc_in_proj_q_scales,
                    &lw.sc_in_proj_q_biases,
                    &lw.sc_in_proj_has_q
                );
                lw.sc_conv1d_w = require_tensor(tensors, p + "conv.conv.weight");
                const std::string sc_out_proj_key = p + "conv.out_proj.weight";
                const array& sc_out_proj_weight = require_tensor(tensors, sc_out_proj_key);
                maybe_quantize_mxfp8_matrix(
                    tensors,
                    ctx->cfg,
                    sc_out_proj_key,
                    sc_out_proj_weight,
                    enable_mlp_qmm,
                    &lw.sc_out_proj_q_w,
                    &lw.sc_out_proj_q_scales,
                    &lw.sc_out_proj_q_biases,
                    &lw.sc_out_proj_has_q
                );
                if (keep_dense_rhs_for_prefill || !enable_mlp_qmm || !lw.sc_in_proj_has_q) {
                    lw.sc_in_proj_rhs = load_rhs_weight(sc_in_proj_key);
                }
                if (keep_dense_rhs_for_prefill || !enable_mlp_qmm || !lw.sc_out_proj_has_q) {
                    lw.sc_out_proj_rhs = load_rhs_weight(sc_out_proj_key);
                }
                if (has_tensor(tensors, p + "conv.conv.bias")) {
                    lw.sc_conv_bias = require_tensor(tensors, p + "conv.conv.bias");
                    lw.sc_has_bias = true;
                }
                const int proj_dim = sc_in_proj_weight.shape(0);
                if (proj_dim % 3 != 0) {
                    throw std::runtime_error("shortconv in_proj output dim is not divisible by 3");
                }
                lw.sc_conv_dim = proj_dim / 3;
                lw.sc_conv_kernel = lw.sc_conv1d_w.shape(1);
            } else {
                const std::string q_proj_key = p + "self_attn.q_proj.weight";
                const std::string k_proj_key = p + "self_attn.k_proj.weight";
                const std::string v_proj_key = p + "self_attn.v_proj.weight";
                const std::string o_proj_key = has_tensor(tensors, p + "self_attn.o_proj.weight")
                    ? p + "self_attn.o_proj.weight"
                    : p + "self_attn.out_proj.weight";
                const array& q_proj_weight = require_tensor(tensors, q_proj_key);
                const array& k_proj_weight = require_tensor(tensors, k_proj_key);
                const array& v_proj_weight = require_tensor(tensors, v_proj_key);
                const array& o_proj_weight = require_tensor(tensors, o_proj_key);
                maybe_quantize_mxfp8_matrix(
                    tensors,
                    ctx->cfg,
                    q_proj_key,
                    q_proj_weight,
                    enable_mlp_qmm,
                    &lw.attn_q_q_w,
                    &lw.attn_q_q_scales,
                    &lw.attn_q_q_biases,
                    &lw.attn_q_has_q
                );
                maybe_quantize_mxfp8_matrix(
                    tensors,
                    ctx->cfg,
                    k_proj_key,
                    k_proj_weight,
                    enable_mlp_qmm,
                    &lw.attn_k_q_w,
                    &lw.attn_k_q_scales,
                    &lw.attn_k_q_biases,
                    &lw.attn_k_has_q
                );
                maybe_quantize_mxfp8_matrix(
                    tensors,
                    ctx->cfg,
                    v_proj_key,
                    v_proj_weight,
                    enable_mlp_qmm,
                    &lw.attn_v_q_w,
                    &lw.attn_v_q_scales,
                    &lw.attn_v_q_biases,
                    &lw.attn_v_has_q
                );
                maybe_quantize_mxfp8_matrix(
                    tensors,
                    ctx->cfg,
                    o_proj_key,
                    o_proj_weight,
                    enable_mlp_qmm,
                    &lw.attn_o_q_w,
                    &lw.attn_o_q_scales,
                    &lw.attn_o_q_biases,
                    &lw.attn_o_has_q
                );
                if (keep_dense_rhs_for_prefill || !enable_mlp_qmm || !lw.attn_q_has_q) {
                    lw.attn_q_rhs = load_rhs_weight(q_proj_key);
                }
                if (keep_dense_rhs_for_prefill || !enable_mlp_qmm || !lw.attn_k_has_q) {
                    lw.attn_k_rhs = load_rhs_weight(k_proj_key);
                }
                if (keep_dense_rhs_for_prefill || !enable_mlp_qmm || !lw.attn_v_has_q) {
                    lw.attn_v_rhs = load_rhs_weight(v_proj_key);
                }
                if (keep_dense_rhs_for_prefill || !enable_mlp_qmm || !lw.attn_o_has_q) {
                    lw.attn_o_rhs = load_rhs_weight(o_proj_key);
                }
                lw.attn_q_norm_w = require_any_tensor(
                    tensors,
                    { p + "self_attn.q_norm.weight", p + "self_attn.q_layernorm.weight" },
                    p + "self_attn.q_norm/q_layernorm"
                );
                lw.attn_k_norm_w = require_any_tensor(
                    tensors,
                    { p + "self_attn.k_norm.weight", p + "self_attn.k_layernorm.weight" },
                    p + "self_attn.k_norm/k_layernorm"
                );

                lw.attn_num_heads = ctx->cfg.num_attention_heads;
                if (lw.attn_num_heads <= 0) {
                    throw std::runtime_error("invalid num_attention_heads in config");
                }

                lw.attn_head_dim = ctx->cfg.head_dim;
                const int q_proj_out_dim = q_proj_weight.shape(0);
                if (q_proj_out_dim <= 0 || (q_proj_out_dim % lw.attn_num_heads) != 0) {
                    throw std::runtime_error(
                        "q_proj output dim is not divisible by num_attention_heads for layer " +
                        std::to_string(i)
                    );
                }

                if (ctx->cfg.use_layer_q_norm_head_dim) {
                    if (lw.attn_q_norm_w.ndim() != 1) {
                        throw std::runtime_error("q_norm must be rank-1 for layer " + std::to_string(i));
                    }
                    lw.attn_head_dim = lw.attn_q_norm_w.shape(0);
                    if (lw.attn_head_dim <= 0) {
                        throw std::runtime_error("derived attn head dim is invalid for layer " + std::to_string(i));
                    }
                }

                if (ctx->cfg.use_layer_q_norm_head_dim) {
                    const int k_proj_out_dim = k_proj_weight.shape(0);
                    if (k_proj_out_dim <= 0 || (k_proj_out_dim % lw.attn_head_dim) != 0) {
                        throw std::runtime_error(
                            "k_proj output dim is not divisible by derived head_dim for layer " +
                            std::to_string(i)
                        );
                    }
                    lw.attn_num_kv_heads = k_proj_out_dim / lw.attn_head_dim;
                    if (lw.attn_num_kv_heads <= 0) {
                        throw std::runtime_error("derived num_key_value_heads is invalid for layer " + std::to_string(i));
                    }
                } else {
                    lw.attn_num_kv_heads = ctx->cfg.num_key_value_heads;
                }
                lw.attn_scale = (ctx->cfg.attention_multiplier > 0.0f)
                    ? ctx->cfg.attention_multiplier
                    : (1.0f / std::sqrt(static_cast<float>(lw.attn_head_dim)));
                lw.attn_rope_theta = ctx->cfg.rope_theta;
                lw.attn_partial_rotary_factor = ctx->cfg.partial_rotary_factor;
                if (ctx->cfg.use_layer_q_norm_head_dim) {
                    bool is_full_attention_layer = lw.attn_head_dim > ctx->cfg.head_dim;
                    if (i >= 0 && i < static_cast<int>(ctx->cfg.layer_types.size())) {
                        is_full_attention_layer = (ctx->cfg.layer_types[static_cast<size_t>(i)] == "full_attention");
                    }
                    if (is_full_attention_layer) {
                        lw.attn_partial_rotary_factor = ctx->cfg.has_nested_rope_parameters
                            ? ctx->cfg.partial_rotary_full
                            : 0.25f;
                        lw.attn_rope_theta = ctx->cfg.has_nested_rope_parameters
                            ? ctx->cfg.rope_theta_full
                            : 1000000.0f;
                    } else {
                        lw.attn_partial_rotary_factor = ctx->cfg.has_nested_rope_parameters
                            ? ctx->cfg.partial_rotary_local
                            : 1.0f;
                        lw.attn_rope_theta = ctx->cfg.has_nested_rope_parameters
                            ? ctx->cfg.rope_theta_local
                            : 10000.0f;
                    }
                }
                const bool layer_type_marks_full =
                    i >= 0 &&
                    i < static_cast<int>(ctx->cfg.layer_types.size()) &&
                    ctx->cfg.layer_types[static_cast<size_t>(i)] == "full_attention";
                lw.attn_is_sliding =
                    i >= 0 &&
                    i < static_cast<int>(ctx->cfg.layer_types.size()) &&
                    ctx->cfg.layer_types[static_cast<size_t>(i)] == "sliding_attention";
                const bool inferred_full_from_head_dim = lw.attn_head_dim > ctx->cfg.head_dim;
                const bool is_full_attention_layer =
                    layer_type_marks_full || inferred_full_from_head_dim;
                const bool disable_global_rope_adjust =
                    env_truthy("TALU_METAL_DISABLE_GLOBAL_ROPE_ADJUST", false);
                if (!disable_global_rope_adjust &&
                    is_full_attention_layer &&
                    ctx->cfg.global_head_dim > 0)
                {
                    const int rope_dim = std::max(
                        0,
                        static_cast<int>(std::round(
                            lw.attn_head_dim * lw.attn_partial_rotary_factor
                        ))
                    );
                    lw.attn_rope_theta = adjust_proportional_global_rope_theta(
                        lw.attn_rope_theta,
                        rope_dim,
                        ctx->cfg.global_head_dim
                    );
                }
            }

            ctx->layers.push_back(std::move(lw));
        }

        const bool enable_init_warmup = env_truthy("TALU_METAL_INIT_WARMUP", true);
        if (enable_init_warmup) {
            // Pre-warm compiled compute_g path so WARMUP=0 benchmarks do not
            // include first-trace compilation overhead.
            for (const auto& lw : ctx->layers) {
                if (!lw.is_linear) continue;
                const array a_dummy = zeros({1, 1, lw.lin_num_v_heads}, lw.lin_dt_bias.dtype());
                const array g_dummy = compute_g(lw.lin_A_exp_f32, a_dummy, lw.lin_dt_bias);
                eval(g_dummy);
                synchronize();
                break;
            }

            // Pre-warm one tiny prefill+decode pass so WARMUP=0 timings don't
            // include first-use kernel/JIT materialization costs.
            reset_runtime_state(ctx.get());
            const std::vector<int32_t> warm_ids = { 0, 1 };
            const array warm_prompt(warm_ids.begin(), Shape{1, 2}, int32);
            const array warm_prefill_logits = forward_logits(ctx.get(), warm_prompt);
            const array warm_seed = next_token_greedy(last_token_logits(warm_prefill_logits));
            const array warm_decode_logits = forward_logits(ctx.get(), warm_seed);
            eval(warm_decode_logits);
            synchronize();
            reset_runtime_state(ctx.get());
        }

        {
            size_t layer_q_count = 0;
            size_t layer_q_total = 0;
            size_t layer_per_layer_count = 0;
            for (const LayerWeights& lw : ctx->layers) {
                const bool flags[] = {
                    lw.mlp_gate_has_q,
                    lw.mlp_up_has_q,
                    lw.mlp_down_has_q,
                    lw.lin_in_proj_qkvz_has_q,
                    lw.lin_in_proj_ba_has_q,
                    lw.lin_out_proj_has_q,
                    lw.sc_in_proj_has_q,
                    lw.sc_out_proj_has_q,
                    lw.attn_q_has_q,
                    lw.attn_k_has_q,
                    lw.attn_v_has_q,
                    lw.attn_o_has_q,
                };
                for (bool flag : flags) {
                    layer_q_total += 1;
                    if (flag) layer_q_count += 1;
                }
                if (lw.has_per_layer_input) layer_per_layer_count += 1;
            }
            std::fprintf(
                stderr,
                "[mlx][nvfp4] model=%s nvfp4=%d gaffine=%d decode_qmm=%d lm_head_q=%d layers_q=%zu/%zu group=%d bits=%d per_layer=%d/%zu hpl=%d gelu=%d\n",
                ctx->model_id.c_str(),
                has_nvfp4_meta ? 1 : 0,
                ctx->has_grouped_affine_meta ? 1 : 0,
                ctx->fp8_decode_qmm_enabled ? 1 : 0,
                ctx->lm_head_q_decode_enabled ? 1 : 0,
                layer_q_count,
                layer_q_total,
                ctx->cfg.quant_group_size,
                ctx->cfg.quant_bits,
                ctx->per_layer_input_enabled ? 1 : 0,
                layer_per_layer_count,
                ctx->per_layer_hidden_size,
                ctx->cfg.use_gelu ? 1 : 0
            );
            std::fflush(stderr);
        }

        if (qmm_debug) {
            const size_t attempts_delta = g_qmm_attempts - qmm_attempts_before;
            const size_t success_delta = g_qmm_success - qmm_success_before;
            std::fprintf(
                stderr,
                "[mlx][qmm] model=%s decode_qmm=%d attempts=%zu success=%zu\n",
                ctx->model_id.c_str(),
                ctx->fp8_decode_qmm_enabled ? 1 : 0,
                attempts_delta,
                success_delta
            );
        }

        return ctx.release();
    } catch (const std::exception& e) {
        if (wired_limit_acquired) release_wired_limit();
        if (rng_entered) leave_rng_epoch();
        g_last_error = e.what();
        return nullptr;
    } catch (...) {
        if (wired_limit_acquired) release_wired_limit();
        if (rng_entered) leave_rng_epoch();
        g_last_error = "unknown error in mlx_create";
        return nullptr;
    }
}

mlx_ctx* mlx_clone(mlx_ctx* source_ctx, int32_t seed) {
    bool wired_limit_acquired = false;
    bool rng_entered = false;
    try {
        if (!source_ctx) {
            g_last_error = "mlx_clone: source ctx is null";
            return nullptr;
        }

        const bool enable_wired_limit = env_truthy("TALU_METAL_WIRED_LIMIT", false);
        if (enable_wired_limit) {
            wired_limit_acquired = acquire_wired_limit();
        }

        enter_rng_epoch(seed);
        rng_entered = true;

        auto ctx = std::make_unique<mlx_ctx>();
        ctx->model_id = source_ctx->model_id;
        ctx->model_path = source_ctx->model_path;
        ctx->cfg = source_ctx->cfg;

        // Arrays are ref-counted in MLX. Copying these fields keeps one shared
        // immutable weight set while each context owns independent runtime state.
        ctx->embed_tokens = source_ctx->embed_tokens;
        ctx->lm_head_rhs = source_ctx->lm_head_rhs;
        ctx->lm_head_q_w = source_ctx->lm_head_q_w;
        ctx->lm_head_q_scales = source_ctx->lm_head_q_scales;
        ctx->lm_head_q_biases = source_ctx->lm_head_q_biases;
        ctx->lm_head_q_bits = source_ctx->lm_head_q_bits;
        ctx->final_norm_w = source_ctx->final_norm_w;
        ctx->has_fp8_meta = source_ctx->has_fp8_meta;
        ctx->has_mxfp8_meta = source_ctx->has_mxfp8_meta;
        ctx->has_grouped_affine_meta = source_ctx->has_grouped_affine_meta;
        ctx->lm_head_q_decode_enabled = source_ctx->lm_head_q_decode_enabled;
        ctx->fp8_decode_qmm_enabled = source_ctx->fp8_decode_qmm_enabled;
        ctx->per_layer_input_enabled = source_ctx->per_layer_input_enabled;
        ctx->per_layer_hidden_size = source_ctx->per_layer_hidden_size;
        ctx->per_layer_input_scale = source_ctx->per_layer_input_scale;
        ctx->per_layer_embed_scale = source_ctx->per_layer_embed_scale;
        ctx->per_layer_model_projection_scale = source_ctx->per_layer_model_projection_scale;
        ctx->per_layer_embedding = source_ctx->per_layer_embedding;
        ctx->per_layer_projection_norm_w = source_ctx->per_layer_projection_norm_w;
        ctx->layers = source_ctx->layers;
        ctx->profile_layers = source_ctx->profile_layers;

        // Runtime state starts clean for the cloned context.
        ctx->kv_cache.resize(source_ctx->kv_cache.size());
        ctx->linear_cache.resize(source_ctx->linear_cache.size());
        ctx->stream_ready = false;
        ctx->trace_decode_token = 1;
        ctx->xray_enabled = false;
        ctx->sampling_context_counts.clear();
        ctx->sampling_context_len = 0;
        ctx->sampling_unique_ids.clear();
        ctx->sampling_repetition_scales.clear();
        ctx->sampling_additive_penalties.clear();

        return ctx.release();
    } catch (const std::exception& e) {
        if (wired_limit_acquired) release_wired_limit();
        if (rng_entered) leave_rng_epoch();
        g_last_error = e.what();
        return nullptr;
    } catch (...) {
        if (wired_limit_acquired) release_wired_limit();
        if (rng_entered) leave_rng_epoch();
        g_last_error = "unknown error in mlx_clone";
        return nullptr;
    }
}

void mlx_destroy(mlx_ctx* ctx) {
    if (ctx) {
        release_wired_limit();
        leave_rng_epoch();
    }
    delete ctx;
}

int32_t mlx_reset(mlx_ctx* ctx) {
    if (!ctx) {
        g_last_error = "mlx_reset: invalid arguments";
        return 0;
    }
    try {
        reset_runtime_state(ctx);
        return 1;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 0;
    } catch (...) {
        g_last_error = "unknown error in mlx_reset";
        return 0;
    }
}

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
        const int full_prompt_threshold = resolve_prefill_full_prompt_threshold();
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
                prefill_prefix_chunks(ctx, prompt_vec, prompt_len - 1);

                const int32_t last_token_scalar = prompt_vec[static_cast<size_t>(prompt_len - 1)];
                const array last_token_arr(&last_token_scalar, Shape{1, 1}, int32);
                const array seed_logits = forward_logits(ctx, last_token_arr);
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
        const int full_prompt_threshold = resolve_prefill_full_prompt_threshold();

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
                prefill_prefix_chunks(ctx, prompt_vec, prompt_len - 1);

                const int32_t last_token_scalar = prompt_vec[static_cast<size_t>(prompt_len - 1)];
                const array last_token_arr(&last_token_scalar, Shape{1, 1}, int32);
                const array rebuild_logits = forward_logits(ctx, last_token_arr);
                (void)rebuild_logits;
            } else {
                prefill_prefix_chunks(ctx, prompt_vec, prompt_len - 1);

                const int32_t last_token_scalar = prompt_vec[static_cast<size_t>(prompt_len - 1)];
                const array last_token_arr(&last_token_scalar, Shape{1, 1}, int32);
                const array seed_logits = forward_logits(ctx, last_token_arr);
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

            array first_logits = array(0.0f);
            if (prompt_len > 1) {
                if (prompt_len <= full_prompt_threshold) {
                    const array prompt_arr(prompt_vec.begin(), Shape{1, prompt_len}, int32);
                    const array seed_logits = forward_logits(ctx, prompt_arr);
                    first_logits = last_token_logits(seed_logits);
                } else {
                    prefill_prefix_chunks(ctx, prompt_vec, prompt_len - 1);

                    const int32_t last_token_scalar = prompt_vec[static_cast<size_t>(prompt_len - 1)];
                    const array last_token_arr(&last_token_scalar, Shape{1, 1}, int32);
                    const array seed_logits = forward_logits(ctx, last_token_arr);
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

    try {
        *out_generated_ids = nullptr;
        *out_generated_len = 0;
        if (decode_tokens == 0) return 1;
        const bool fused_lm_head_profile = env_truthy("TALU_METAL_FUSED_LM_HEAD_PROFILE", false);
        const size_t fused_hits_start = g_fused_lm_head_argmax_hits;

        std::vector<array> generated_token_arrays;
        generated_token_arrays.reserve(static_cast<size_t>(decode_tokens));

        const int32_t first_token_scalar = first_token;
        const array first_token_arr(&first_token_scalar, Shape{1, 1}, int32);
        array current_token = can_use_fused_lm_head_argmax(ctx)
            ? fused_lm_head_argmax_token(ctx, first_token_arr)
            : next_token_greedy(last_token_logits(forward_logits(ctx, first_token_arr)));
        async_eval(current_token);

        for (int32_t i = 0; i < decode_tokens; ++i) {
            generated_token_arrays.push_back(current_token);
            if (i + 1 < decode_tokens) {
                const array next_token = can_use_fused_lm_head_argmax(ctx)
                    ? fused_lm_head_argmax_token(ctx, current_token)
                    : next_token_greedy(last_token_logits(forward_logits(ctx, current_token)));
                async_eval(next_token);
                current_token = next_token;
            }
        }

        const array generated_token_rows = concatenate(generated_token_arrays, 0); // [T,1]
        const array generated_token_flat = reshape(astype(generated_token_rows, int32), {decode_tokens}); // [T]
        eval(generated_token_flat);

        std::unique_ptr<int32_t, decltype(&std::free)> copied(
            static_cast<int32_t*>(std::malloc(sizeof(int32_t) * static_cast<size_t>(decode_tokens))),
            &std::free
        );
        if (!copied) {
            g_last_error = "mlx_decode_stream: failed to allocate token buffer";
            return 0;
        }

        const int32_t* token_ptr = generated_token_flat.data<int32_t>();
        int32_t produced = 0;
        for (int32_t i = 0; i < decode_tokens; ++i) {
            const int32_t tok = token_ptr[static_cast<size_t>(i)];
            copied.get()[static_cast<size_t>(produced)] = tok;
            produced += 1;
            if (token_is_eos(tok, eos_ids, eos_len)) break;
        }
        if (fused_lm_head_profile) {
            const size_t fused_hits = g_fused_lm_head_argmax_hits - fused_hits_start;
            std::fprintf(
                stderr,
                "[mlx][fused_lm_head_argmax] used=%s hits=%zu mode=decode_stream\n",
                can_use_fused_lm_head_argmax(ctx) ? "1" : "0",
                fused_hits
            );
        }

        *out_generated_ids = copied.release();
        *out_generated_len = produced;
        return 1;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 0;
    } catch (...) {
        g_last_error = "unknown error in mlx_decode_stream";
        return 0;
    }
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
            const auto total_t0 = std::chrono::steady_clock::now();
            while (produced < decode_tokens && !saw_eos) {
                const int32_t remaining = decode_tokens - produced;
                const int32_t chunk_target = std::min(effective_topk_chunk_size, remaining);
                chunk_tokens.clear();

                const auto t_build0 = std::chrono::steady_clock::now();
                for (int32_t i = 0; i < chunk_target; ++i) {
                    const array next_token = use_fused_greedy_decode
                        ? fused_lm_head_argmax_token(ctx, current_token)
                        : (use_fused_topk_decode
                            ? sample_token_from_topk_candidates(
                                fused_lm_head_topk_candidates(ctx, current_token, sampling_cfg.top_k),
                                sampling_cfg
                              )
                            : sample_token_from_logits(
                                last_token_logits(forward_logits(ctx, current_token)),
                                sampling_cfg
                              ));
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

            const int vocab = ctx->embed_tokens.ndim() == 2 ? ctx->embed_tokens.shape(0) : 0;
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
            auto t1 = std::chrono::steady_clock::now();

            auto t2 = std::chrono::steady_clock::now();
            // Match mlx-lm generation scheduling: seed first decode token,
            // then run one-step lookahead with async eval.
            array current_token = can_use_fused_lm_head_argmax(ctx)
                ? fused_lm_head_argmax_token(ctx, token_ids)
                : next_token_greedy(last_token_logits(forward_logits(ctx, token_ids)));
            async_eval(current_token);

            if (capture_tokens && !is_warmup) {
                // Preserve async decode scheduling while still capturing every
                // token by materializing all token arrays once at the end.
                std::vector<array> captured_token_arrays;
                captured_token_arrays.reserve(static_cast<size_t>(decode_tokens));

                for (int i = 0; i < decode_tokens; ++i) {
                    captured_token_arrays.push_back(current_token);
                    if (i + 1 < decode_tokens) {
                        const array next_token = can_use_fused_lm_head_argmax(ctx)
                            ? fused_lm_head_argmax_token(ctx, current_token)
                            : next_token_greedy(last_token_logits(forward_logits(ctx, current_token)));
                        async_eval(next_token);
                        current_token = next_token;
                    }
                }

                const array captured_rows = concatenate(captured_token_arrays, 0); // [T,1]
                const array captured_flat = reshape(astype(captured_rows, int32), {decode_tokens}); // [T]
                eval(captured_flat);
                std::memcpy(
                    captured_tokens.data(),
                    captured_flat.data<int32_t>(),
                    sizeof(int32_t) * static_cast<size_t>(decode_tokens)
                );
            } else {
                for (int i = 0; i < decode_tokens; ++i) {
                    if (i + 1 < decode_tokens) {
                        const array next_token = can_use_fused_lm_head_argmax(ctx)
                            ? fused_lm_head_argmax_token(ctx, current_token)
                            : next_token_greedy(last_token_logits(forward_logits(ctx, current_token)));
                        async_eval(next_token);
                        if (i == 0) {
                            eval(current_token);
                        }
                        current_token = next_token;
                    } else {
                        eval(current_token);
                    }
                }
            }
            synchronize();
            auto t3 = std::chrono::steady_clock::now();

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

void mlx_tokens_free(int32_t* ids) {
    std::free(ids);
}

const char* mlx_last_error(void) {
    return g_last_error.c_str();
}

} // extern "C"
