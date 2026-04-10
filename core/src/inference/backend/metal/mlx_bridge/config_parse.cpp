#include "config_parse.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>

namespace {

std::string read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("unable to open " + path.string());
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
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
            if (depth == 0) return json.substr(open, i - open + 1);
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
    const std::regex arr_re("\\\"" + key + "\\\"\\s*:\\s*\\[([\\s\\S]*?)\\]");
    std::smatch m;
    if (!std::regex_search(text, m, arr_re)) return false;
    const std::string body = m[1].str();
    const std::regex item_re("\\\"([^\\\"]*)\\\"");
    out->clear();
    for (auto it = std::sregex_iterator(body.begin(), body.end(), item_re);
         it != std::sregex_iterator();
         ++it) {
        out->push_back((*it)[1].str());
    }
    return true;
}

} // namespace

ParsedModelConfig parse_qwen35_config(const std::string& model_path) {
    const std::string config_text = read_file(std::filesystem::path(model_path) / "config.json");
    std::string text_cfg;
    try {
        text_cfg = extract_named_object(config_text, "text_config");
    } catch (const std::runtime_error&) {
        text_cfg = config_text;
    }

    Qwen35Config cfg;
    ParsedModelConfig parsed;

    if (!find_int_value(text_cfg, "hidden_size", &cfg.hidden_size)) throw std::runtime_error("missing text_config.hidden_size");
    if (!find_int_value(text_cfg, "num_hidden_layers", &cfg.num_hidden_layers)) throw std::runtime_error("missing text_config.num_hidden_layers");
    const bool has_intermediate_size = find_int_value(text_cfg, "intermediate_size", &cfg.intermediate_size);
    const bool has_moe_intermediate_size = find_int_value(text_cfg, "moe_intermediate_size", &cfg.moe_intermediate_size);
    const bool has_shared_expert_intermediate_size =
        find_int_value(text_cfg, "shared_expert_intermediate_size", &cfg.shared_expert_intermediate_size);
    if (!has_intermediate_size &&
        !has_moe_intermediate_size &&
        !has_shared_expert_intermediate_size &&
        !find_int_value(text_cfg, "d_ff", &cfg.intermediate_size) &&
        !find_int_value(text_cfg, "block_ff_dim", &cfg.intermediate_size)) {
        throw std::runtime_error(
            "missing text_config.intermediate_size/moe_intermediate_size/"
            "shared_expert_intermediate_size/d_ff/block_ff_dim"
        );
    }
    if (!has_intermediate_size) {
        if (has_moe_intermediate_size) {
            cfg.intermediate_size = cfg.moe_intermediate_size;
        } else if (has_shared_expert_intermediate_size) {
            cfg.intermediate_size = cfg.shared_expert_intermediate_size;
        }
    }
    find_int_value(text_cfg, "num_experts", &cfg.num_experts);
    find_int_value(text_cfg, "num_experts_per_tok", &cfg.num_experts_per_tok);

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
