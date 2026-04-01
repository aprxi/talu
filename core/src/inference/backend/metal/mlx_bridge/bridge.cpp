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
thread_local std::vector<array> g_decode_batch_logits_flat_scratch;

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

std::shared_ptr<const CompiledTopKFn> compiled_topk_candidates(int k) {
    static std::unordered_map<int, std::shared_ptr<const CompiledTopKFn>> cache;
    static std::mutex cache_mutex;
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache.find(k);
    if (it != cache.end()) return it->second;

    auto fn = compile(
        [k](const std::vector<array>& inputs) {
            const array& logits_last = inputs.at(0); // [1,1,V]
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
            // Match metal backend candidate extraction: partition on logits
            // directly and gather the final K bucket.
            const array partitioned_indices = argpartition(logits_last, -k, -1);
            const array top_k_selector = arange(vocab - k, vocab, 1, int32);
            const array top_k_indices = take(partitioned_indices, top_k_selector, -1);
            const array top_k_logits = take_along_axis(logits_last, top_k_indices, -1);
            return std::vector<array>{
                top_k_logits,
                top_k_indices,
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

struct StreamSamplingConfig {
    float temperature = 1.0f;
    int top_k = 20;
    float top_p = 0.95f;
    float min_p = 0.0f;
};

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

    const int vocab = logits_last.shape(2);
    const int k = std::min(cfg.top_k, vocab);
    if (k <= 0) {
        throw std::runtime_error("sample_token_from_logits invalid top_k");
    }

    // Follow scheduler semantics: top-k candidate extraction first, then
    // top-p sampling over the K candidate logits.
    const array topk_part = argpartition(-logits_last, k - 1, -1);
    const array topk_select = arange(0, k, 1, int32);
    const array topk_indices = take(topk_part, topk_select, -1); // [1,1,k]
    array sampled_logits = take_along_axis(logits_last, topk_indices, -1); // [1,1,k]

    if (cfg.top_p > 0.0f && cfg.top_p < 1.0f) {
        sampled_logits = apply_top_p_filter(sampled_logits, cfg.top_p);
    }

    sampled_logits = sampled_logits * array(1.0f / cfg.temperature, sampled_logits.dtype());
    const array sampled_local_idx = random::categorical(sampled_logits, -1); // [1,1]
    const array sampled_local_idx_3d = reshape(sampled_local_idx, {1, 1, 1});
    const array sampled_token_3d = take_along_axis(topk_indices, sampled_local_idx_3d, -1); // [1,1,1]
    return reshape(sampled_token_3d, {1, 1});
}

struct Qwen35Config {
    int hidden_size = 0;
    int num_hidden_layers = 0;
    int intermediate_size = 0;

    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;

    int linear_num_value_heads = 0;
    int linear_num_key_heads = 0;
    int linear_key_head_dim = 0;
    int linear_value_head_dim = 0;
    int linear_conv_kernel_dim = 0;

    int full_attention_interval = 1;

    float rms_norm_eps = 1.0e-6f;
    float rope_theta = 10000000.0f;
    float partial_rotary_factor = 1.0f;

    bool tie_word_embeddings = true;
};

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

    array mlp_gate_rhs = array(0.0f);
    array mlp_up_rhs = array(0.0f);
    array mlp_down_rhs = array(0.0f);
    array mlp_gate_q_w = array(0.0f);
    array mlp_gate_q_scales = array(0.0f);
    array mlp_up_q_w = array(0.0f);
    array mlp_up_q_scales = array(0.0f);
    array mlp_down_q_w = array(0.0f);
    array mlp_down_q_scales = array(0.0f);
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
    array attn_k_q_w = array(0.0f);
    array attn_k_q_scales = array(0.0f);
    array attn_v_q_w = array(0.0f);
    array attn_v_q_scales = array(0.0f);
    array attn_o_q_w = array(0.0f);
    array attn_o_q_scales = array(0.0f);
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
    array lin_in_proj_ba_q_w = array(0.0f);
    array lin_in_proj_ba_q_scales = array(0.0f);
    bool lin_in_proj_qkvz_has_q = false;
    bool lin_in_proj_ba_has_q = false;
    array lin_A_log = array(0.0f);
    array lin_A_exp_f32 = array(0.0f);
    array lin_dt_bias = array(0.0f);
    array lin_norm_w = array(0.0f);
    array lin_out_proj_rhs = array(0.0f);
    array lin_out_proj_q_w = array(0.0f);
    array lin_out_proj_q_scales = array(0.0f);
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
    array sc_conv1d_w = array(0.0f); // [conv_dim, kernel, 1]
    array sc_conv_bias = array(0.0f);
    array sc_out_proj_rhs = array(0.0f);
    array sc_out_proj_q_w = array(0.0f);
    array sc_out_proj_q_scales = array(0.0f);
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
    if (!find_int_value(text_cfg, "intermediate_size", &cfg.intermediate_size)) throw std::runtime_error("missing text_config.intermediate_size");

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

    // Linear-attention fields are model-family specific (Qwen3.5 hybrid).
    // Keep defaults when absent so pure-attention families (Qwen3) load.
    find_int_value(text_cfg, "linear_num_value_heads", &cfg.linear_num_value_heads);
    find_int_value(text_cfg, "linear_num_key_heads", &cfg.linear_num_key_heads);
    find_int_value(text_cfg, "linear_key_head_dim", &cfg.linear_key_head_dim);
    find_int_value(text_cfg, "linear_value_head_dim", &cfg.linear_value_head_dim);
    find_int_value(text_cfg, "linear_conv_kernel_dim", &cfg.linear_conv_kernel_dim);

    find_int_value(text_cfg, "full_attention_interval", &cfg.full_attention_interval);
    if (!find_float_value(text_cfg, "rms_norm_eps", &cfg.rms_norm_eps)) {
        if (!find_float_value(text_cfg, "norm_eps", &cfg.rms_norm_eps)) {
            find_float_value(text_cfg, "block_norm_eps", &cfg.rms_norm_eps);
        }
    }

    // Qwen3.5 stores rope settings inside rope_parameters.
    find_float_value(text_cfg, "rope_theta", &cfg.rope_theta);
    find_float_value(text_cfg, "partial_rotary_factor", &cfg.partial_rotary_factor);

    bool tie_embeddings = true;
    if (find_bool_value(config_text, "tie_word_embeddings", &tie_embeddings)) {
        cfg.tie_word_embeddings = tie_embeddings;
    }

    const std::regex lfm2_re("\\\"model_type\\\"\\s*:\\s*\\\"lfm2(?:_vl|_5)?\\\"");
    const bool is_lfm2_family = std::regex_search(config_text, lfm2_re) || std::regex_search(text_cfg, lfm2_re);
    parsed.cfg = cfg;
    parsed.allow_qwen_norm_shift = !is_lfm2_family;
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
    return tensors.find(key) != tensors.end();
}

std::string detect_text_prefix(const std::unordered_map<std::string, array>& tensors) {
    if (has_tensor(tensors, "model.language_model.embed_tokens.weight")) {
        return "model.language_model.";
    }
    if (has_tensor(tensors, "model.embed_tokens.weight")) {
        return "model.";
    }
    throw std::runtime_error("missing embed_tokens.weight for known text prefixes");
}

array to_rhs(const array& weight_2d, const std::string& name) {
    if (weight_2d.ndim() != 2) {
        throw std::runtime_error("expected rank-2 weight for " + name);
    }
    // MLX-LM nn.Linear uses [out, in] weights. Convert once to [in, out] RHS.
    return transpose(weight_2d, {1, 0});
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

    auto scale_inv_it = tensors.find(scale_inv_key);
    auto block_scale_it = tensors.find(block_scale_key);
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
        const array& scale_inv = scale_inv_it->second;
        if (scale_inv.ndim() != 2) {
            throw std::runtime_error("fp8 scale_inv must be rank-2 for " + weight_key);
        }
        scales_f32 = astype(scale_inv, float32);
    }

    const array expanded_scales = expand_block_scales(scales_f32, rows, cols, weight_key);
    const array fp8_as_f32 = from_fp8(weight_2d, float32);
    const array dequant_f32 = fp8_as_f32 * expanded_scales;
    array dequant_bf16 = astype(dequant_f32, bfloat16);
    dequant_bf16 = stop_gradient(copy(dequant_bf16));
    // Materialize dequantized weights during load so large FP8 dequant graphs
    // do not all lower at first prefill command-buffer submit (can OOM on 4B).
    eval(dequant_bf16);
    synchronize();
    return dequant_bf16;
}

array load_linear_weight(
    const std::unordered_map<std::string, array>& tensors,
    const std::string& weight_key
) {
    const array& raw_weight = require_tensor(tensors, weight_key);
    return maybe_dequantize_fp8_weight(tensors, weight_key, raw_weight);
}

void maybe_quantize_mxfp8_matrix(
    const std::unordered_map<std::string, array>& tensors,
    const std::string& weight_key,
    const array& lhs_weight,
    bool enabled,
    array* out_q_w,
    array* out_q_scales,
    bool* out_has_q
) {
    if (!enabled) {
        *out_has_q = false;
        return;
    }
    // Fast path for native MXFP8 checkpoints: reuse packed FP8 payload and
    // block scales directly for decode-time quantized matmul.
    if (ends_with(weight_key, ".weight")) {
        const std::string base = weight_key.substr(0, weight_key.size() - std::string(".weight").size());
        const std::string block_scale_key = base + ".weight_block_scale";
        auto raw_it = tensors.find(weight_key);
        auto scales_it = tensors.find(block_scale_key);
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
                        return;
                    } catch (const std::exception&) {
                        // Fall through to generic quantize path.
                    }
                }
            }
        }
    }

    const std::vector<array> q = quantize(
        lhs_weight,
        std::nullopt,
        std::nullopt,
        "mxfp8"
    );
    if (q.size() != 2) {
        throw std::runtime_error("mlx quantize(mxfp8) returned unexpected output count");
    }
    *out_q_w = q[0];
    *out_q_scales = q[1];
    *out_has_q = true;
}

array linear_decode_maybe_quantized(
    bool decode_qmm_enabled,
    const array& x,
    const array& rhs,
    bool has_q,
    const array& q_w,
    const array& q_scales
) {
    const bool decode_step = x.ndim() == 3 && x.shape(1) == 1;
    if (has_q && decode_qmm_enabled && decode_step) {
        return quantized_matmul(
            x,
            q_w,
            q_scales,
            std::nullopt,
            true,
            std::nullopt,
            std::nullopt,
            "mxfp8"
        );
    }
    return matmul(x, rhs);
}

array linear(const array& x, const array& rhs) {
    return matmul(x, rhs);
}

array silu(const array& x) {
    return x * sigmoid(x);
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
    array final_norm_w = array(0.0f); // [hidden]
    bool has_fp8_meta = false;
    bool has_mxfp8_meta = false;
    bool lm_head_q_decode_enabled = false;
    bool fp8_decode_qmm_enabled = false;

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
        if (ends_with(kv.first, ".weight_scale_inv") || ends_with(kv.first, ".weight_block_scale")) {
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

bool env_truthy(const char* name, bool fallback) {
    const char* raw = std::getenv(name);
    if (!raw) return fallback;
    return std::string(raw) == "1";
}

array run_full_attention_layer(mlx_ctx* ctx, int layer_idx, const array& x_norm) {
    LayerWeights& lw = ctx->layers[static_cast<size_t>(layer_idx)];
    KVCacheState& cache = ctx->kv_cache[static_cast<size_t>(layer_idx)];

    const int B = x_norm.shape(0);
    const int S = x_norm.shape(1);

    const array q_proj_out = linear_decode_maybe_quantized(
        ctx->fp8_decode_qmm_enabled,
        x_norm,
        lw.attn_q_rhs,
        lw.attn_q_has_q,
        lw.attn_q_q_w,
        lw.attn_q_q_scales
    );
    const int q_last_dim = q_proj_out.shape(2);
    const int q_base_dim = lw.attn_num_heads * lw.attn_head_dim;
    array queries_raw = array(0.0f);
    array gate = array(0.0f);
    if (q_last_dim == 2 * q_base_dim) {
        // Qwen3.5 full-attention branch: q projection carries both q and gate.
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
        gate = ones({B, S, q_base_dim}, q_proj_out.dtype());
    } else {
        throw std::runtime_error("unexpected self_attn.q_proj output shape");
    }

    const array k_proj_out = linear_decode_maybe_quantized(
        ctx->fp8_decode_qmm_enabled,
        x_norm,
        lw.attn_k_rhs,
        lw.attn_k_has_q,
        lw.attn_k_q_w,
        lw.attn_k_q_scales
    );
    const array v_proj_out = linear_decode_maybe_quantized(
        ctx->fp8_decode_qmm_enabled,
        x_norm,
        lw.attn_v_rhs,
        lw.attn_v_has_q,
        lw.attn_v_q_w,
        lw.attn_v_q_scales
    );

    array queries = fast::rms_norm(queries_raw, std::optional<array>(lw.attn_q_norm_w), ctx->cfg.rms_norm_eps);
    array keys = reshape(k_proj_out, {B, S, lw.attn_num_kv_heads, lw.attn_head_dim});
    keys = fast::rms_norm(keys, std::optional<array>(lw.attn_k_norm_w), ctx->cfg.rms_norm_eps);
    array values = reshape(v_proj_out, {B, S, lw.attn_num_kv_heads, lw.attn_head_dim});

    queries = transpose(queries, {0, 2, 1, 3});
    keys = transpose(keys, {0, 2, 1, 3});
    values = transpose(values, {0, 2, 1, 3});

    const int rope_offset = cache.offset;
    queries = fast::rope(queries, static_cast<int>(std::round(ctx->cfg.head_dim * ctx->cfg.partial_rotary_factor)), false, std::optional<float>(ctx->cfg.rope_theta), 1.0f, rope_offset);
    keys = fast::rope(keys, static_cast<int>(std::round(ctx->cfg.head_dim * ctx->cfg.partial_rotary_factor)), false, std::optional<float>(ctx->cfg.rope_theta), 1.0f, rope_offset);

    const int prev = cache.offset;
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

    const array keys_view = slice(*cache.keys, {0, 0, 0, 0}, {B, lw.attn_num_kv_heads, cache.offset, lw.attn_head_dim});
    const array values_view = slice(*cache.values, {0, 0, 0, 0}, {B, lw.attn_num_kv_heads, cache.offset, lw.attn_head_dim});

    const std::string mask_mode = (S > 1) ? "causal" : "";
    array attn_out = fast::scaled_dot_product_attention(
        queries,
        keys_view,
        values_view,
        lw.attn_scale,
        mask_mode
    );

    attn_out = transpose(attn_out, {0, 2, 1, 3});
    attn_out = reshape(attn_out, {B, S, lw.attn_num_heads * lw.attn_head_dim});
    attn_out = attn_out * sigmoid(gate);

    return linear_decode_maybe_quantized(
        ctx->fp8_decode_qmm_enabled,
        attn_out,
        lw.attn_o_rhs,
        lw.attn_o_has_q,
        lw.attn_o_q_w,
        lw.attn_o_q_scales
    );
}

array run_linear_attention_layer(mlx_ctx* ctx, int layer_idx, const array& x_norm) {
    LayerWeights& lw = ctx->layers[static_cast<size_t>(layer_idx)];
    LinearCacheState& cache = ctx->linear_cache[static_cast<size_t>(layer_idx)];

    const int B = x_norm.shape(0);
    const int S = x_norm.shape(1);

    const array qkvz = linear_decode_maybe_quantized(
        ctx->fp8_decode_qmm_enabled,
        x_norm,
        lw.lin_in_proj_qkvz_rhs,
        lw.lin_in_proj_qkvz_has_q,
        lw.lin_in_proj_qkvz_q_w,
        lw.lin_in_proj_qkvz_q_scales
    ); // [B,S,conv_dim+value_dim]
    const array qkv = slice(qkvz, {0, 0, 0}, {B, S, lw.lin_conv_dim});
    const array z_flat = slice(qkvz, {0, 0, lw.lin_conv_dim}, {B, S, lw.lin_conv_dim + lw.lin_value_dim});
    const array z = reshape(z_flat, {B, S, lw.lin_num_v_heads, lw.lin_head_v_dim});

    const array ba = linear_decode_maybe_quantized(
        ctx->fp8_decode_qmm_enabled,
        x_norm,
        lw.lin_in_proj_ba_rhs,
        lw.lin_in_proj_ba_has_q,
        lw.lin_in_proj_ba_q_w,
        lw.lin_in_proj_ba_q_scales
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
        merged,
        lw.lin_out_proj_rhs,
        lw.lin_out_proj_has_q,
        lw.lin_out_proj_q_w,
        lw.lin_out_proj_q_scales
    );
}

array run_shortconv_layer(mlx_ctx* ctx, int layer_idx, const array& x_norm) {
    LayerWeights& lw = ctx->layers[static_cast<size_t>(layer_idx)];
    LinearCacheState& cache = ctx->linear_cache[static_cast<size_t>(layer_idx)];

    const int B = x_norm.shape(0);
    const int S = x_norm.shape(1);

    const array proj = linear_decode_maybe_quantized(
        ctx->fp8_decode_qmm_enabled,
        x_norm,
        lw.sc_in_proj_rhs,
        lw.sc_in_proj_has_q,
        lw.sc_in_proj_q_w,
        lw.sc_in_proj_q_scales
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
        gated,
        lw.sc_out_proj_rhs,
        lw.sc_out_proj_has_q,
        lw.sc_out_proj_q_w,
        lw.sc_out_proj_q_scales
    );
}

array run_layer(mlx_ctx* ctx, int layer_idx, const array& hidden, const TraceFrame* trace_frame) {
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

    const array h = hidden + residual_branch;

    const array mlp_in = fast::rms_norm(h, std::optional<array>(lw.post_attention_layernorm_w), ctx->cfg.rms_norm_eps);
    const array gate = linear_decode_maybe_quantized(
        ctx->fp8_decode_qmm_enabled,
        mlp_in,
        lw.mlp_gate_rhs,
        lw.mlp_gate_has_q,
        lw.mlp_gate_q_w,
        lw.mlp_gate_q_scales
    );
    const array up = linear_decode_maybe_quantized(
        ctx->fp8_decode_qmm_enabled,
        mlp_in,
        lw.mlp_up_rhs,
        lw.mlp_up_has_q,
        lw.mlp_up_q_w,
        lw.mlp_up_q_scales
    );
    const array ff = silu(gate) * up;
    const array down = linear_decode_maybe_quantized(
        ctx->fp8_decode_qmm_enabled,
        ff,
        lw.mlp_down_rhs,
        lw.mlp_down_has_q,
        lw.mlp_down_q_w,
        lw.mlp_down_q_scales
    );

    const array out = h + down;
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
            hidden = run_layer(ctx, i, hidden, trace_frame);
        }
    } else {
        for (int i = 0; i < static_cast<int>(ctx->layers.size()); ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            hidden = run_layer(ctx, i, hidden, trace_frame);
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
    if (ctx->has_fp8_meta && hidden.ndim() == 3 && hidden.shape(1) > 1) {
        // Prefill only needs next-token logits; avoid full-sequence lm_head.
        const int b = hidden.shape(0);
        const int t = hidden.shape(1);
        const int h = hidden.shape(2);
        lm_input = slice(hidden, {0, t - 1, 0}, {b, t, h});
    }
    const bool decode_step = lm_input.ndim() == 3 && lm_input.shape(1) == 1;
    const array logits = (ctx->lm_head_q_decode_enabled && decode_step)
        ? quantized_matmul(
              lm_input,
              ctx->lm_head_q_w,
              ctx->lm_head_q_scales,
              std::nullopt,
              true,
              std::nullopt,
              std::nullopt,
              "mxfp8"
          )
        : matmul(lm_input, ctx->lm_head_rhs);
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

} // namespace

extern "C" {

int32_t mlx_is_available(void) {
    return metal::is_available() ? 1 : 0;
}

mlx_ctx* mlx_create(const char* model_id, const char* model_path, int32_t seed) {
    bool wired_limit_acquired = false;
    bool rng_entered = false;
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

        // Match mlx-lm generation behavior on Metal by setting wired limit to
        // max recommended working set size before inference begins.
        wired_limit_acquired = acquire_wired_limit();

        const ParsedModelConfig parsed_cfg = parse_qwen35_config(ctx->model_path);
        ctx->cfg = parsed_cfg.cfg;

        auto tensors = load_weight_tensors(ctx->model_path);
        sanitize_qwen35_tensors(tensors, ctx->cfg.tie_word_embeddings, parsed_cfg.allow_qwen_norm_shift);
        const bool has_fp8_meta = has_fp8_quant_metadata(tensors);
        ctx->has_fp8_meta = has_fp8_meta;
        const bool has_mxfp8_meta = has_mxfp8_quant_metadata(tensors);
        ctx->has_mxfp8_meta = has_mxfp8_meta;
        ctx->fp8_decode_qmm_enabled = has_mxfp8_meta && env_truthy("TALU_METAL_FP8_DECODE_QMM", true);
        const bool enable_mlp_qmm = ctx->fp8_decode_qmm_enabled;

        const std::string text_prefix = detect_text_prefix(tensors);
        const std::string layer_prefix = text_prefix + "layers.";
        ctx->embed_tokens = require_tensor(tensors, text_prefix + "embed_tokens.weight");
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
                lm_head_weight = load_linear_weight(tensors, text_prefix + "lm_head.weight");
            } else if (has_tensor(tensors, "lm_head.weight")) {
                lm_head_key = "lm_head.weight";
                lm_head_weight = load_linear_weight(tensors, "lm_head.weight");
            } else {
                throw std::runtime_error("missing lm_head weight (tie_word_embeddings=false)");
            }
        }

        const bool enable_lm_head_qmm = has_mxfp8_meta && env_truthy("TALU_METAL_FP8_LM_HEAD_QMM", true);
        if (enable_lm_head_qmm) {
            if (lm_head_weight.size() == 1) {
                // Tie embeddings can point at raw FP8 tensors. Load dequantized copy only
                // if direct packed MXFP8 path is unavailable.
                lm_head_weight = load_linear_weight(tensors, lm_head_key);
            }
            bool lm_head_has_q = false;
            maybe_quantize_mxfp8_matrix(
                tensors,
                lm_head_key,
                lm_head_weight,
                true,
                &ctx->lm_head_q_w,
                &ctx->lm_head_q_scales,
                &lm_head_has_q
            );
            ctx->lm_head_q_decode_enabled = lm_head_has_q;
        }

        if (!ctx->lm_head_q_decode_enabled) {
            if (lm_head_weight.size() == 1) {
                lm_head_weight = load_linear_weight(tensors, lm_head_key);
            }
            ctx->lm_head_rhs = transpose(lm_head_weight, {1, 0});
        }

        ctx->layers.reserve(static_cast<size_t>(ctx->cfg.num_hidden_layers));
        ctx->kv_cache.resize(static_cast<size_t>(ctx->cfg.num_hidden_layers));
        ctx->linear_cache.resize(static_cast<size_t>(ctx->cfg.num_hidden_layers));

        for (int i = 0; i < ctx->cfg.num_hidden_layers; ++i) {
            LayerWeights lw;
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

            const std::string mlp_gate_key = has_tensor(tensors, p + "mlp.gate_proj.weight")
                ? p + "mlp.gate_proj.weight"
                : p + "feed_forward.w1.weight";
            const std::string mlp_up_key = has_tensor(tensors, p + "mlp.up_proj.weight")
                ? p + "mlp.up_proj.weight"
                : p + "feed_forward.w3.weight";
            const std::string mlp_down_key = has_tensor(tensors, p + "mlp.down_proj.weight")
                ? p + "mlp.down_proj.weight"
                : p + "feed_forward.w2.weight";

            const array mlp_gate_weight = load_linear_weight(tensors, mlp_gate_key); // [out,in]
            const array mlp_up_weight = load_linear_weight(tensors, mlp_up_key); // [out,in]
            const array mlp_down_weight = load_linear_weight(tensors, mlp_down_key); // [out,in]

            lw.mlp_gate_rhs = to_rhs(
                mlp_gate_weight,
                p + "mlp_gate"
            );
            lw.mlp_up_rhs = to_rhs(
                mlp_up_weight,
                p + "mlp_up"
            );
            lw.mlp_down_rhs = to_rhs(
                mlp_down_weight,
                p + "mlp_down"
            );
            maybe_quantize_mxfp8_matrix(
                tensors,
                mlp_gate_key,
                mlp_gate_weight,
                enable_mlp_qmm,
                &lw.mlp_gate_q_w,
                &lw.mlp_gate_q_scales,
                &lw.mlp_gate_has_q
            );
            maybe_quantize_mxfp8_matrix(
                tensors,
                mlp_up_key,
                mlp_up_weight,
                enable_mlp_qmm,
                &lw.mlp_up_q_w,
                &lw.mlp_up_q_scales,
                &lw.mlp_up_has_q
            );
            maybe_quantize_mxfp8_matrix(
                tensors,
                mlp_down_key,
                mlp_down_weight,
                enable_mlp_qmm,
                &lw.mlp_down_q_w,
                &lw.mlp_down_q_scales,
                &lw.mlp_down_has_q
            );

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
                    const array qkv_weight = load_linear_weight(tensors, qkv_key);
                    const array z_weight = load_linear_weight(tensors, z_key);
                    const array a_weight = load_linear_weight(tensors, a_key);
                    const array b_weight = load_linear_weight(tensors, b_key);
                    const array qkvz_weight = concatenate({qkv_weight, z_weight}, 0);
                    const array ba_weight = concatenate({b_weight, a_weight}, 0);
                    lw.lin_in_proj_qkv_rhs = to_rhs(qkv_weight, qkv_key);
                    lw.lin_in_proj_z_rhs = to_rhs(z_weight, z_key);
                    lw.lin_in_proj_a_rhs = to_rhs(a_weight, a_key);
                    lw.lin_in_proj_b_rhs = to_rhs(b_weight, b_key);
                    lw.lin_in_proj_qkvz_rhs = concatenate({lw.lin_in_proj_qkv_rhs, lw.lin_in_proj_z_rhs}, 1);
                    lw.lin_in_proj_ba_rhs = concatenate({lw.lin_in_proj_b_rhs, lw.lin_in_proj_a_rhs}, 1);
                    maybe_quantize_mxfp8_matrix(
                        tensors,
                        qkv_key,
                        qkvz_weight,
                        enable_mlp_qmm,
                        &lw.lin_in_proj_qkvz_q_w,
                        &lw.lin_in_proj_qkvz_q_scales,
                        &lw.lin_in_proj_qkvz_has_q
                    );
                    maybe_quantize_mxfp8_matrix(
                        tensors,
                        b_key,
                        ba_weight,
                        enable_mlp_qmm,
                        &lw.lin_in_proj_ba_q_w,
                        &lw.lin_in_proj_ba_q_scales,
                        &lw.lin_in_proj_ba_has_q
                    );
                } else {
                    const std::string fused_key = p + "mixer.in_proj.weight";
                    const array fused_in_proj_weight = load_linear_weight(tensors, fused_key); // [out, hidden]
                    const array fused_in_proj_rhs = to_rhs(
                        fused_in_proj_weight,
                        fused_key
                    );
                    if (fused_in_proj_rhs.ndim() != 2) {
                        throw std::runtime_error("linear fused in_proj must be rank-2");
                    }
                    const int qkvz_cols = lw.lin_conv_dim + lw.lin_value_dim;
                    const int ba_cols = 2 * lw.lin_num_v_heads;
                    const int expected_cols = qkvz_cols + ba_cols;
                    const int fused_rows = fused_in_proj_rhs.shape(0);
                    const int fused_cols = fused_in_proj_rhs.shape(1);
                    if (fused_cols != expected_cols) {
                        throw std::runtime_error(
                            "linear fused in_proj shape mismatch: expected cols=" +
                            std::to_string(expected_cols) + " got cols=" + std::to_string(fused_cols)
                        );
                    }
                    lw.lin_in_proj_qkvz_rhs = slice(
                        fused_in_proj_rhs,
                        {0, 0},
                        {fused_rows, qkvz_cols}
                    );
                    lw.lin_in_proj_ba_rhs = slice(
                        fused_in_proj_rhs,
                        {0, qkvz_cols},
                        {fused_rows, expected_cols}
                    );
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
                    maybe_quantize_mxfp8_matrix(
                        tensors,
                        p + "mixer.in_proj.qkvz_slice",
                        qkvz_weight,
                        enable_mlp_qmm,
                        &lw.lin_in_proj_qkvz_q_w,
                        &lw.lin_in_proj_qkvz_q_scales,
                        &lw.lin_in_proj_qkvz_has_q
                    );
                    maybe_quantize_mxfp8_matrix(
                        tensors,
                        p + "mixer.in_proj.ba_slice",
                        ba_weight,
                        enable_mlp_qmm,
                        &lw.lin_in_proj_ba_q_w,
                        &lw.lin_in_proj_ba_q_scales,
                        &lw.lin_in_proj_ba_has_q
                    );
                }
                lw.lin_A_log = require_tensor(tensors, p + "linear_attn.A_log");
                lw.lin_A_exp_f32 = exp(astype(lw.lin_A_log, float32));
                lw.lin_dt_bias = require_tensor(tensors, p + "linear_attn.dt_bias");
                lw.lin_norm_w = require_tensor(tensors, p + "linear_attn.norm.weight");
                const std::string out_proj_key = p + "linear_attn.out_proj.weight";
                const array out_proj_weight = load_linear_weight(tensors, out_proj_key);
                lw.lin_out_proj_rhs = to_rhs(out_proj_weight, out_proj_key);
                maybe_quantize_mxfp8_matrix(
                    tensors,
                    out_proj_key,
                    out_proj_weight,
                    enable_mlp_qmm,
                    &lw.lin_out_proj_q_w,
                    &lw.lin_out_proj_q_scales,
                    &lw.lin_out_proj_has_q
                );
            } else if (lw.is_shortconv) {
                const std::string sc_in_proj_key = p + "conv.in_proj.weight";
                const array sc_in_proj_weight = load_linear_weight(tensors, sc_in_proj_key);
                lw.sc_in_proj_rhs = to_rhs(sc_in_proj_weight, sc_in_proj_key);
                maybe_quantize_mxfp8_matrix(
                    tensors,
                    sc_in_proj_key,
                    sc_in_proj_weight,
                    enable_mlp_qmm,
                    &lw.sc_in_proj_q_w,
                    &lw.sc_in_proj_q_scales,
                    &lw.sc_in_proj_has_q
                );
                lw.sc_conv1d_w = require_tensor(tensors, p + "conv.conv.weight");
                const std::string sc_out_proj_key = p + "conv.out_proj.weight";
                const array sc_out_proj_weight = load_linear_weight(tensors, sc_out_proj_key);
                lw.sc_out_proj_rhs = to_rhs(sc_out_proj_weight, sc_out_proj_key);
                maybe_quantize_mxfp8_matrix(
                    tensors,
                    sc_out_proj_key,
                    sc_out_proj_weight,
                    enable_mlp_qmm,
                    &lw.sc_out_proj_q_w,
                    &lw.sc_out_proj_q_scales,
                    &lw.sc_out_proj_has_q
                );
                if (has_tensor(tensors, p + "conv.conv.bias")) {
                    lw.sc_conv_bias = require_tensor(tensors, p + "conv.conv.bias");
                    lw.sc_has_bias = true;
                }
                const int proj_dim = lw.sc_in_proj_rhs.shape(1);
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
                const array q_proj_weight = load_linear_weight(tensors, q_proj_key);
                const array k_proj_weight = load_linear_weight(tensors, k_proj_key);
                const array v_proj_weight = load_linear_weight(tensors, v_proj_key);
                const array o_proj_weight = load_linear_weight(tensors, o_proj_key);

                lw.attn_q_rhs = to_rhs(q_proj_weight, q_proj_key);
                lw.attn_k_rhs = to_rhs(k_proj_weight, k_proj_key);
                lw.attn_v_rhs = to_rhs(v_proj_weight, v_proj_key);
                lw.attn_o_rhs = to_rhs(o_proj_weight, p + "self_attn_o_proj");
                maybe_quantize_mxfp8_matrix(
                    tensors,
                    q_proj_key,
                    q_proj_weight,
                    enable_mlp_qmm,
                    &lw.attn_q_q_w,
                    &lw.attn_q_q_scales,
                    &lw.attn_q_has_q
                );
                maybe_quantize_mxfp8_matrix(
                    tensors,
                    k_proj_key,
                    k_proj_weight,
                    enable_mlp_qmm,
                    &lw.attn_k_q_w,
                    &lw.attn_k_q_scales,
                    &lw.attn_k_has_q
                );
                maybe_quantize_mxfp8_matrix(
                    tensors,
                    v_proj_key,
                    v_proj_weight,
                    enable_mlp_qmm,
                    &lw.attn_v_q_w,
                    &lw.attn_v_q_scales,
                    &lw.attn_v_has_q
                );
                maybe_quantize_mxfp8_matrix(
                    tensors,
                    o_proj_key,
                    o_proj_weight,
                    enable_mlp_qmm,
                    &lw.attn_o_q_w,
                    &lw.attn_o_q_scales,
                    &lw.attn_o_has_q
                );
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
                lw.attn_num_kv_heads = ctx->cfg.num_key_value_heads;
                lw.attn_head_dim = ctx->cfg.head_dim;
                lw.attn_scale = 1.0f / std::sqrt(static_cast<float>(lw.attn_head_dim));
            }

            ctx->layers.push_back(std::move(lw));
        }

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
        {
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
        std::vector<int32_t> prompt_vec(prompt_ids, prompt_ids + prompt_len);
        reset_runtime_state(ctx);
        ctx->xray_enabled = (talu_metal_xray_is_enabled() != 0);

        array first_token = array(0.0f);
        if (prompt_len > 1) {
            const array prefix_arr(prompt_vec.begin(), Shape{1, prompt_len - 1}, int32);
            const array prefill_hidden = forward_hidden(ctx, prefix_arr, false);
            (void)prefill_hidden;

            const int32_t last_token_scalar = prompt_vec[static_cast<size_t>(prompt_len - 1)];
            const array last_token_arr(&last_token_scalar, Shape{1, 1}, int32);
            const array seed_logits = forward_logits(ctx, last_token_arr);
            first_token = next_token_greedy(last_token_logits(seed_logits));
        } else {
            const array prompt_arr(prompt_vec.begin(), Shape{1, prompt_len}, int32);
            const array seed_logits = forward_logits(ctx, prompt_arr);
            first_token = next_token_greedy(last_token_logits(seed_logits));
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
        std::vector<int32_t> prompt_vec(prompt_ids, prompt_ids + prompt_len);
        reset_runtime_state(ctx);
        ctx->xray_enabled = (talu_metal_xray_is_enabled() != 0);

        array first_logits = array(0.0f);
        const bool prefill_trace_enabled = ctx->xray_enabled &&
            (xray_should_emit(XRAY_POINT_EMBED, XRAY_GLOBAL_LAYER, static_cast<uint32_t>(prompt_len)) ||
                xray_should_emit(XRAY_POINT_LAYER_INPUT, 0, static_cast<uint32_t>(prompt_len)) ||
                xray_should_emit(XRAY_POINT_ATTN_OUT, 0, static_cast<uint32_t>(prompt_len)) ||
                xray_should_emit(XRAY_POINT_BLOCK_OUT, 0, static_cast<uint32_t>(prompt_len)) ||
                xray_should_emit(XRAY_POINT_FINAL_NORM, XRAY_GLOBAL_LAYER, 1) ||
                xray_should_emit(XRAY_POINT_LM_HEAD, XRAY_GLOBAL_LAYER, 0));
        if (prompt_len > 1 && prefill_trace_enabled) {
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
            const array prefix_arr(prompt_vec.begin(), Shape{1, prompt_len - 1}, int32);
            const array prefill_hidden = forward_hidden(ctx, prefix_arr, false);
            (void)prefill_hidden;

            const int32_t last_token_scalar = prompt_vec[static_cast<size_t>(prompt_len - 1)];
            const array last_token_arr(&last_token_scalar, Shape{1, 1}, int32);
            const array rebuild_logits = forward_logits(ctx, last_token_arr);
            (void)rebuild_logits;
        } else if (prompt_len > 1) {
            const array prefix_arr(prompt_vec.begin(), Shape{1, prompt_len - 1}, int32);
            const array prefill_hidden = forward_hidden(ctx, prefix_arr, false);
            (void)prefill_hidden;

            const int32_t last_token_scalar = prompt_vec[static_cast<size_t>(prompt_len - 1)];
            const array last_token_arr(&last_token_scalar, Shape{1, 1}, int32);
            const array seed_logits = forward_logits(ctx, last_token_arr);
            first_logits = last_token_logits(seed_logits);
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
        auto& next_logits_flat_batch = g_decode_batch_logits_flat_scratch;
        next_logits_flat_batch.clear();
        if (next_logits_flat_batch.capacity() < static_cast<size_t>(batch_size)) {
            next_logits_flat_batch.reserve(static_cast<size_t>(batch_size));
        }

        for (int32_t i = 0; i < batch_size; ++i) {
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

        eval(next_logits_flat_batch);
        synchronize();
        const size_t bytes = static_cast<size_t>(logits_len) * sizeof(float);
        for (int32_t i = 0; i < batch_size; ++i) {
            std::memcpy(
                out_logits_ptrs[i],
                next_logits_flat_batch[static_cast<size_t>(i)].data<float>(),
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

        std::vector<array> generated_token_arrays;
        generated_token_arrays.reserve(static_cast<size_t>(decode_tokens));

        const int32_t first_token_scalar = first_token;
        const array first_token_arr(&first_token_scalar, Shape{1, 1}, int32);
        array current_token = next_token_greedy(last_token_logits(forward_logits(ctx, first_token_arr)));
        async_eval(current_token);

        for (int32_t i = 0; i < decode_tokens; ++i) {
            generated_token_arrays.push_back(current_token);
            if (i + 1 < decode_tokens) {
                const array next_token = next_token_greedy(last_token_logits(forward_logits(ctx, current_token)));
                async_eval(next_token);
                current_token = next_token;
            }
        }

        eval(generated_token_arrays);
        synchronize();

        const size_t capacity = static_cast<size_t>(decode_tokens);
        const size_t bytes = sizeof(int32_t) * capacity;
        int32_t* copied = static_cast<int32_t*>(std::malloc(bytes));
        if (!copied) {
            g_last_error = "mlx_decode_stream: failed to allocate token buffer";
            return 0;
        }

        int32_t produced = 0;
        for (int32_t i = 0; i < decode_tokens; ++i) {
            const int32_t tok = generated_token_arrays[static_cast<size_t>(i)].item<int32_t>();
            copied[static_cast<size_t>(produced)] = tok;
            produced += 1;
            if (token_is_eos(tok, eos_ids, eos_len)) break;
        }

        *out_generated_ids = copied;
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
    repetition_scales.clear();
    additive_penalties.clear();
    if (token_ids.capacity() < token_counts.size()) token_ids.reserve(token_counts.size());
    if (repetition_scales.capacity() < token_counts.size()) repetition_scales.reserve(token_counts.size());
    if (additive_penalties.capacity() < token_counts.size()) additive_penalties.reserve(token_counts.size());

    for (const auto& entry : token_counts) {
        const int32_t token_id = entry.first;
        const int32_t count = entry.second;
        if (token_id < 0 || token_id >= vocab) {
            throw std::runtime_error("apply_sampling_penalties_to_logits: context token out of range");
        }
        token_ids.push_back(token_id);
        repetition_scales.push_back(std::pow(repetition_penalty, static_cast<float>(count)));
        additive_penalties.push_back(
            presence_penalty + (frequency_penalty * static_cast<float>(count))
        );
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

        const int k = std::min(top_k, vocab);
        auto topk_fn = compiled_topk_candidates(k);
        const std::vector<array> topk_outputs = (*topk_fn)({logits_last});
        if (topk_outputs.size() != 2) {
            g_last_error = "mlx_decode_topk_candidates: invalid compiled output";
            return 0;
        }
        const array top_k_logits_flat = topk_outputs[0];
        const array top_k_indices_flat = topk_outputs[1];

        eval(topk_outputs);

        const float* logits_ptr = top_k_logits_flat.data<float>();
        const int32_t* ids_ptr = top_k_indices_flat.data<int32_t>();
        std::memcpy(out_candidate_logits, logits_ptr, static_cast<size_t>(k) * sizeof(float));
        std::memcpy(out_candidate_ids, ids_ptr, static_cast<size_t>(k) * sizeof(int32_t));
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
        const int32_t token_scalar = token;
        const array token_arr(&token_scalar, Shape{1, 1}, int32);
        const array logits_last = last_token_logits(forward_logits(ctx, token_arr)); // [1,1,V]
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
        const array top_k_logits_flat = topk_outputs[0];
        const array top_k_indices_flat = topk_outputs[1];

        eval(topk_outputs);

        const float* logits_ptr = top_k_logits_flat.data<float>();
        const int32_t* ids_ptr = top_k_indices_flat.data<int32_t>();
        std::memcpy(out_candidate_logits, logits_ptr, static_cast<size_t>(k) * sizeof(float));
        std::memcpy(out_candidate_ids, ids_ptr, static_cast<size_t>(k) * sizeof(int32_t));
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

    try {
        *out_generated_ids = nullptr;
        *out_generated_len = 0;
        if (decode_tokens == 0) return 1;

        const StreamSamplingConfig sampling_cfg{
            .temperature = temperature,
            .top_k = top_k,
            .top_p = top_p,
            .min_p = min_p,
        };

        std::vector<array> generated_token_arrays;
        generated_token_arrays.reserve(static_cast<size_t>(decode_tokens));

        const int32_t token_scalar = first_token;
        array current_token = array(&token_scalar, Shape{1, 1}, int32);
        array next_token = sample_token_from_logits(last_token_logits(forward_logits(ctx, current_token)), sampling_cfg);
        async_eval(next_token);

        for (int32_t i = 0; i < decode_tokens; ++i) {
            generated_token_arrays.push_back(next_token);
            if (i + 1 < decode_tokens) {
                next_token = sample_token_from_logits(last_token_logits(forward_logits(ctx, next_token)), sampling_cfg);
                async_eval(next_token);
            }
        }

        eval(generated_token_arrays);

        const size_t capacity = static_cast<size_t>(decode_tokens);
        const size_t bytes = sizeof(int32_t) * capacity;
        int32_t* copied = static_cast<int32_t*>(std::malloc(bytes));
        if (!copied) {
            g_last_error = "mlx_decode_topk_stream: failed to allocate token buffer";
            return 0;
        }

        int32_t produced = 0;
        for (int32_t i = 0; i < decode_tokens; ++i) {
            const int32_t tok = *generated_token_arrays[static_cast<size_t>(i)].data<int32_t>();
            copied[static_cast<size_t>(produced)] = tok;
            produced += 1;
            if (token_is_eos(tok, eos_ids, eos_len)) break;
        }

        *out_generated_ids = copied;
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
                const array prefix_arr(prompt_vec.begin(), Shape{1, prompt_len - 1}, int32);
                const array prefill_hidden = forward_hidden(ctx, prefix_arr, false);
                (void)prefill_hidden;

                const int32_t last_token_scalar = prompt_vec[static_cast<size_t>(prompt_len - 1)];
                const array last_token_arr(&last_token_scalar, Shape{1, 1}, int32);
                const array seed_logits = forward_logits(ctx, last_token_arr);
                token_ids = next_token_greedy(last_token_logits(seed_logits));
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
            array current_token = next_token_greedy(last_token_logits(forward_logits(ctx, token_ids)));
            async_eval(current_token);

            if (capture_tokens && !is_warmup) {
                // Preserve async decode scheduling while still capturing every
                // token by materializing all token arrays once at the end.
                std::vector<array> captured_token_arrays;
                captured_token_arrays.reserve(static_cast<size_t>(decode_tokens));

                for (int i = 0; i < decode_tokens; ++i) {
                    captured_token_arrays.push_back(current_token);
                    if (i + 1 < decode_tokens) {
                        const array next_token = next_token_greedy(last_token_logits(forward_logits(ctx, current_token)));
                        async_eval(next_token);
                        current_token = next_token;
                    }
                }

                eval(captured_token_arrays);
                for (int i = 0; i < decode_tokens; ++i) {
                    captured_tokens[static_cast<size_t>(i)] =
                        captured_token_arrays[static_cast<size_t>(i)].item<int32_t>();
                }
            } else {
                for (int i = 0; i < decode_tokens; ++i) {
                    if (i + 1 < decode_tokens) {
                        const array next_token = next_token_greedy(last_token_logits(forward_logits(ctx, current_token)));
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
