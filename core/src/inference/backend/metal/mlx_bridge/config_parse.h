#pragma once

#include <string>
#include <vector>

struct BridgeModelConfig {
    int hidden_size = 0;
    int num_hidden_layers = 0;
    int intermediate_size = 0;
    int moe_intermediate_size = 0;
    int shared_expert_intermediate_size = 0;
    int num_experts = 0;
    int num_experts_per_tok = 0;

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
    std::string quant_mode = "affine";

    bool tie_word_embeddings = true;
    bool use_layer_q_norm_head_dim = false;
    bool use_gelu = false;
    bool use_v_norm = false;
    bool has_nested_rope_parameters = false;
};

struct ParsedModelConfig {
    BridgeModelConfig cfg;
    bool allow_norm_shift = true;
};

struct mlx_model_flags;
ParsedModelConfig parse_model_config(const std::string& model_path, const mlx_model_flags* flags = nullptr);
