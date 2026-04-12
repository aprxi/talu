// bridge_types.inc — Core data structures: KVCacheState, LinearCacheState, LayerWeights.

float adjust_proportional_global_rope_theta(float base_theta, int rope_dim, int global_head_dim) {
    if (base_theta <= 0.0f || rope_dim <= 0 || global_head_dim <= 0 || global_head_dim < rope_dim) {
        return base_theta;
    }
    const float exponent =
        static_cast<float>(rope_dim) / static_cast<float>(global_head_dim);
    return std::pow(base_theta, exponent);
}

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
    array mlp_gate_up_rhs = array(0.0f);
    array mlp_gate_q_w = array(0.0f);
    array mlp_gate_q_scales = array(0.0f);
    array mlp_gate_q_biases = array(0.0f);
    array mlp_up_q_w = array(0.0f);
    array mlp_up_q_scales = array(0.0f);
    array mlp_up_q_biases = array(0.0f);
    array mlp_gate_up_q_w = array(0.0f);
    array mlp_gate_up_q_scales = array(0.0f);
    array mlp_gate_up_q_biases = array(0.0f);
    array mlp_down_q_w = array(0.0f);
    array mlp_down_q_scales = array(0.0f);
    array mlp_down_q_biases = array(0.0f);
    bool mlp_gate_has_q = false;
    bool mlp_up_has_q = false;
    bool mlp_gate_up_has_q = false;
    bool mlp_down_has_q = false;
    int mlp_gate_width = 0;
    bool mlp_is_moe = false;
    int moe_num_experts = 0;
    int moe_experts_per_token = 0;
    int moe_router_group_size = 0;
    int moe_expert_group_size = 0;
    array moe_router_w = array(0.0f);
    array moe_gate_up_w = array(0.0f);
    array moe_gate_up_scales = array(0.0f);
    array moe_gate_up_biases = array(0.0f);
    array moe_gate_w = array(0.0f);
    array moe_gate_scales = array(0.0f);
    array moe_gate_biases = array(0.0f);
    array moe_up_w = array(0.0f);
    array moe_up_scales = array(0.0f);
    array moe_up_biases = array(0.0f);
    array moe_down_w = array(0.0f);
    array moe_down_scales = array(0.0f);
    array moe_down_biases = array(0.0f);
    array moe_shared_gate_rhs = array(0.0f);
    array moe_shared_gate_q_w = array(0.0f);
    array moe_shared_gate_q_scales = array(0.0f);
    array moe_shared_gate_q_biases = array(0.0f);
    array moe_shared_up_rhs = array(0.0f);
    array moe_shared_up_q_w = array(0.0f);
    array moe_shared_up_q_scales = array(0.0f);
    array moe_shared_up_q_biases = array(0.0f);
    array moe_shared_down_rhs = array(0.0f);
    array moe_shared_down_q_w = array(0.0f);
    array moe_shared_down_q_scales = array(0.0f);
    array moe_shared_down_q_biases = array(0.0f);
    array moe_shared_gate_up_rhs = array(0.0f);
    array moe_shared_gate_up_q_w = array(0.0f);
    array moe_shared_gate_up_q_scales = array(0.0f);
    array moe_shared_gate_up_q_biases = array(0.0f);
    bool moe_shared_gate_has_q = false;
    bool moe_shared_up_has_q = false;
    bool moe_shared_down_has_q = false;
    bool moe_shared_gate_up_has_q = false;
    array moe_shared_expert_gate_w = array(0.0f);

    // Full-attention branch.
    array attn_q_rhs = array(0.0f);
    array attn_k_rhs = array(0.0f);
    array attn_v_rhs = array(0.0f);
    array attn_qkv_rhs = array(0.0f);
    array attn_qkv_prefill_rhs = array(0.0f);
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
    array attn_qkv_q_w = array(0.0f);
    array attn_qkv_q_scales = array(0.0f);
    array attn_qkv_q_biases = array(0.0f);
    array attn_o_q_w = array(0.0f);
    array attn_o_q_scales = array(0.0f);
    array attn_o_q_biases = array(0.0f);
    bool attn_q_has_q = false;
    bool attn_k_has_q = false;
    bool attn_v_has_q = false;
    bool attn_qkv_has_q = false;
    bool attn_o_has_q = false;
    array attn_q_norm_w = array(0.0f);
    array attn_k_norm_w = array(0.0f);
    int attn_q_proj_width = 0;
    int attn_k_proj_width = 0;
    int attn_v_proj_width = 0;
    int attn_num_heads = 0;
    int attn_num_kv_heads = 0;
    int attn_head_dim = 0;
    float attn_scale = 1.0f;
    float attn_rope_theta = 10000000.0f;
    float attn_partial_rotary_factor = 1.0f;
    int attn_rope_dims = 0;
    array rope_freq = array(0.0f); // [rope_dims/2] pre-computed frequencies

    // Linear-attention (gated-delta) branch.
    array lin_conv1d_w = array(0.0f); // [conv_dim, kernel, 1]
    array lin_conv1d_decode_w = array(0.0f); // [1, kernel, conv_dim]
    array lin_in_proj_qkv_rhs = array(0.0f);
    array lin_in_proj_z_rhs = array(0.0f);
    array lin_in_proj_a_rhs = array(0.0f);
    array lin_in_proj_b_rhs = array(0.0f);
    array lin_in_proj_qkvz_rhs = array(0.0f);
    array lin_in_proj_qkvz_prefill_rhs = array(0.0f);
    array lin_in_proj_ba_rhs = array(0.0f);
    array lin_in_proj_all_rhs = array(0.0f);
    array lin_in_proj_qkvz_q_w = array(0.0f);
    array lin_in_proj_qkvz_q_scales = array(0.0f);
    array lin_in_proj_qkvz_q_biases = array(0.0f);
    array lin_in_proj_ba_q_w = array(0.0f);
    array lin_in_proj_ba_q_scales = array(0.0f);
    array lin_in_proj_ba_q_biases = array(0.0f);
    array lin_in_proj_all_q_w = array(0.0f);
    array lin_in_proj_all_q_scales = array(0.0f);
    array lin_in_proj_all_q_biases = array(0.0f);
    bool lin_in_proj_qkvz_has_q = false;
    bool lin_in_proj_ba_has_q = false;
    bool lin_in_proj_all_has_q = false;
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
