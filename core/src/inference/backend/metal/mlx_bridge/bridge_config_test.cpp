#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "bridge.h"
#include "config_parse.h"

namespace {

std::filesystem::path write_config_dir(const std::string& dirname, const std::string& config_text) {
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / dirname;
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    std::ofstream out(dir / "config.json", std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open temporary config.json for writing");
    }
    out << config_text;
    out.close();
    return dir;
}

void expect_eq(int actual, int expected, const char* label) {
    if (actual != expected) {
        throw std::runtime_error(
            std::string(label) + " mismatch: expected " + std::to_string(expected) +
            ", got " + std::to_string(actual)
        );
    }
}

void test_parse_model_config_accepts_moe_intermediate_size() {
    const auto dir = write_config_dir(
        "talu-mlx-bridge-config-moe",
        R"JSON({
  "architectures": ["Qwen3_5MoeForConditionalGeneration"],
  "model_type": "qwen3_5_moe",
  "text_config": {
    "hidden_size": 2048,
    "num_hidden_layers": 40,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 512
  }
})JSON"
    );
    const ParsedModelConfig parsed = parse_model_config(dir.string());
    expect_eq(parsed.cfg.hidden_size, 2048, "hidden_size");
    expect_eq(parsed.cfg.num_hidden_layers, 40, "num_hidden_layers");
    expect_eq(parsed.cfg.num_attention_heads, 16, "num_attention_heads");
    expect_eq(parsed.cfg.intermediate_size, 512, "intermediate_size");
    expect_eq(parsed.cfg.moe_intermediate_size, 512, "moe_intermediate_size");
    expect_eq(parsed.cfg.shared_expert_intermediate_size, 512, "shared_expert_intermediate_size");
    std::filesystem::remove_all(dir);
}

void test_parse_model_config_accepts_shared_expert_intermediate_size() {
    const auto dir = write_config_dir(
        "talu-mlx-bridge-config-shared-expert",
        R"JSON({
  "architectures": ["Qwen3_5MoeForConditionalGeneration"],
  "model_type": "qwen3_5_moe",
  "text_config": {
    "hidden_size": 2048,
    "num_hidden_layers": 40,
    "num_attention_heads": 16,
    "shared_expert_intermediate_size": 768
  }
})JSON"
    );
    const ParsedModelConfig parsed = parse_model_config(dir.string());
    expect_eq(parsed.cfg.intermediate_size, 768, "shared_expert_intermediate_size");
    expect_eq(parsed.cfg.num_key_value_heads, 16, "default_num_key_value_heads");
    expect_eq(parsed.cfg.head_dim, 128, "derived_head_dim");
    std::filesystem::remove_all(dir);
}

void expect_approx(float actual, float expected, const char* label) {
    if (std::abs(actual - expected) > 1e-6f) {
        throw std::runtime_error(
            std::string(label) + " mismatch: expected " + std::to_string(expected) +
            ", got " + std::to_string(actual)
        );
    }
}

void expect_bool(bool actual, bool expected, const char* label) {
    if (actual != expected) {
        throw std::runtime_error(
            std::string(label) + " mismatch: expected " + std::to_string(expected) +
            ", got " + std::to_string(actual)
        );
    }
}

void test_parse_model_config_flags_override_replaces_defaults() {
    const auto dir = write_config_dir(
        "talu-mlx-bridge-config-flags",
        R"JSON({
  "text_config": {
    "hidden_size": 2048,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "intermediate_size": 8192
  }
})JSON"
    );
    // Parse with explicit flags — should override config-derived values.
    // Note: use_layer_q_norm_head_dim=1 requires layer_types to match
    // num_hidden_layers, so we keep it off for this test and test it
    // separately when layer_types is provided.
    mlx_model_flags flags = {};
    flags.use_layer_q_norm_head_dim = 0;
    flags.allow_norm_shift = 0;
    flags.use_v_norm = 1;
    flags.embedding_multiplier = 45.254f;
    flags.attention_multiplier = 1.0f;

    const ParsedModelConfig parsed = parse_model_config(dir.string(), &flags);
    expect_bool(parsed.cfg.use_layer_q_norm_head_dim, false, "use_layer_q_norm_head_dim");
    expect_bool(parsed.cfg.use_v_norm, true, "use_v_norm");
    expect_bool(parsed.allow_norm_shift, false, "allow_norm_shift");
    expect_approx(parsed.cfg.embedding_multiplier, 45.254f, "embedding_multiplier");
    expect_approx(parsed.cfg.attention_multiplier, 1.0f, "attention_multiplier");
    std::filesystem::remove_all(dir);
}

void test_parse_model_config_flags_apply_identity_values() {
    const auto dir = write_config_dir(
        "talu-mlx-bridge-config-flags-identity",
        R"JSON({
  "text_config": {
    "hidden_size": 2048,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "intermediate_size": 8192,
    "embedding_multiplier": 99.0
  }
})JSON"
    );
    // Flags with identity values (1.0 / 0.0) must still override config.
    mlx_model_flags flags = {};
    flags.use_layer_q_norm_head_dim = 0;
    flags.allow_norm_shift = 1;
    flags.use_v_norm = 0;
    flags.embedding_multiplier = 1.0f;
    flags.attention_multiplier = 0.0f;

    const ParsedModelConfig parsed = parse_model_config(dir.string(), &flags);
    expect_approx(parsed.cfg.embedding_multiplier, 1.0f, "embedding_multiplier_identity");
    expect_approx(parsed.cfg.attention_multiplier, 0.0f, "attention_multiplier_identity");
    expect_bool(parsed.allow_norm_shift, true, "allow_norm_shift_identity");
    std::filesystem::remove_all(dir);
}

void test_parse_model_config_reads_moe_routing_fields() {
    const auto dir = write_config_dir(
        "talu-mlx-bridge-config-routing",
        R"JSON({
  "architectures": ["Qwen3_5MoeForConditionalGeneration"],
  "model_type": "qwen3_5_moe_text",
  "text_config": {
    "hidden_size": 2048,
    "num_hidden_layers": 40,
    "num_attention_heads": 16,
    "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 512,
    "num_experts": 256,
    "num_experts_per_tok": 8
  },
  "quantization": {
    "bits": 4,
    "group_size": 32
  }
})JSON"
    );
    const ParsedModelConfig parsed = parse_model_config(dir.string());
    expect_eq(parsed.cfg.num_experts, 256, "num_experts");
    expect_eq(parsed.cfg.num_experts_per_tok, 8, "num_experts_per_tok");
    expect_eq(parsed.cfg.quant_bits, 4, "quant_bits");
    expect_eq(parsed.cfg.quant_group_size, 32, "quant_group_size");
    std::filesystem::remove_all(dir);
}

} // namespace

int main() {
    try {
        test_parse_model_config_accepts_moe_intermediate_size();
        test_parse_model_config_accepts_shared_expert_intermediate_size();
        test_parse_model_config_reads_moe_routing_fields();
        test_parse_model_config_flags_override_replaces_defaults();
        test_parse_model_config_flags_apply_identity_values();
        std::cout << "bridge_config_test: ok\n";
        return 0;
    } catch (const std::exception& err) {
        std::cerr << "bridge_config_test: " << err.what() << "\n";
        return 1;
    }
}
