// ============================================================================
// Quantized Model Structure
// ============================================================================

struct FusedModelWeights {
    struct Layer {
        enum class LayerKind : int {
            attention_mlp = 0,
            shortconv = 1,
            mamba = 2,
        };
        LayerKind kind = LayerKind::attention_mlp;

        array ln1_w = array(0.0f, float32);  // attention norm
        array q_w = array(0.0f, float32), q_s = array(0.0f, float32), q_b = array(0.0f, float32);
        array k_w = array(0.0f, float32), k_s = array(0.0f, float32), k_b = array(0.0f, float32);
        array v_w = array(0.0f, float32), v_s = array(0.0f, float32), v_b = array(0.0f, float32);
        array o_w = array(0.0f, float32), o_s = array(0.0f, float32), o_b = array(0.0f, float32);
        // MLA attention (quantized)
        bool use_mla = false;
        int mla_n_heads = 0;
        int mla_q_lora_rank = 0;
        int mla_kv_lora_rank = 0;
        int mla_qk_head_dim = 0;
        int mla_qk_rope_head_dim = 0;
        int mla_qk_nope_head_dim = 0;
        int mla_v_head_dim = 0;
        array mla_q_a_w = array(0.0f, float32), mla_q_a_s = array(0.0f, float32), mla_q_a_b = array(0.0f, float32);
        array mla_q_b_w = array(0.0f, float32), mla_q_b_s = array(0.0f, float32), mla_q_b_b = array(0.0f, float32);
        array mla_kv_a_w = array(0.0f, float32), mla_kv_a_s = array(0.0f, float32), mla_kv_a_b = array(0.0f, float32);
        array mla_kv_b_w = array(0.0f, float32), mla_kv_b_s = array(0.0f, float32), mla_kv_b_b = array(0.0f, float32);
        array mla_q_a_norm = array(0.0f, float32), mla_kv_a_norm = array(0.0f, float32);
        array mla_o_w = array(0.0f, float32), mla_o_s = array(0.0f, float32), mla_o_b = array(0.0f, float32);
        // ShortConv mixer projections (quantized) and convolution weights.
        int shortconv_d_conv = 0;
        int shortconv_conv_dim = 0;
        array shortconv_in_w = array(0.0f, float32), shortconv_in_s = array(0.0f, float32), shortconv_in_b = array(0.0f, float32);
        array shortconv_out_w = array(0.0f, float32), shortconv_out_s = array(0.0f, float32), shortconv_out_b = array(0.0f, float32);
        array shortconv_conv_w = array(0.0f, float32);
        std::optional<array> shortconv_conv_b;
        std::optional<array> shortconv_kernel_broadcast;
        std::optional<array> shortconv_bias_row;
        // Mamba mixer (quantized core; optional quantized FFN)
        int mamba_d_state = 0;
        int mamba_d_conv = 0;
        int mamba_n_heads = 0;
        int mamba_d_head = 0;
        int mamba_n_groups = 1;
        uint8_t mamba_gate_up_layout = 0;
        array mamba_conv_w = array(0.0f, float32);
        std::optional<array> mamba_conv_b;
        array mamba_a_log = array(0.0f, float32);
        array mamba_d_skip = array(0.0f, float32);
        std::optional<array> mamba_dt_bias;
        std::optional<array> mamba_norm_weight;
        array mamba_in_w = array(0.0f, float32), mamba_in_s = array(0.0f, float32), mamba_in_b = array(0.0f, float32);
        array mamba_out_w = array(0.0f, float32), mamba_out_s = array(0.0f, float32), mamba_out_b = array(0.0f, float32);
        std::optional<array> mamba_gate_up_w, mamba_gate_up_s, mamba_gate_up_b;
        std::optional<array> mamba_down_w, mamba_down_s, mamba_down_b;
        array ln2_w = array(0.0f, float32);  // ffn norm (or post_attention_layernorm)
        array gate_w = array(0.0f, float32), gate_s = array(0.0f, float32), gate_b = array(0.0f, float32);
        array up_w = array(0.0f, float32), up_s = array(0.0f, float32), up_b = array(0.0f, float32);
        array down_w = array(0.0f, float32), down_s = array(0.0f, float32), down_b = array(0.0f, float32);
        bool use_moe = false;
        array moe_router_w = array(0.0f, float32);
        std::optional<array> moe_router_s;
        std::optional<array> moe_router_b;
        std::optional<array> moe_router_bias;
        array moe_gate_w = array(0.0f, float32), moe_gate_s = array(0.0f, float32);
        array moe_up_w = array(0.0f, float32), moe_up_s = array(0.0f, float32);
        array moe_down_w = array(0.0f, float32), moe_down_s = array(0.0f, float32);
        std::optional<array> moe_gate_bias;
        std::optional<array> moe_up_bias;
        std::optional<array> moe_down_bias;
        int moe_num_experts = 0;
        int moe_experts_per_token = 0;
        int moe_router_group_size = 0;
        int moe_expert_group_size = 0;
        std::optional<array> q_norm;
        std::optional<array> k_norm;
        // Optional extra FFN norms (4 norms per block)
        std::optional<array> pre_ffn_norm;
        std::optional<array> post_ffn_norm;
        // Fused weights (created at model setup time for efficiency)
        // QKV fused: [hidden_dim, (n_heads + 2*n_kv_heads) * head_dim]
        std::optional<array> qkv_w, qkv_s, qkv_b;
        // Gate+Up fused: [hidden_dim, 2 * d_ff]
        std::optional<array> gate_up_w, gate_up_s, gate_up_b;
        bool use_fused_qkv = false;
        bool use_fused_gate_up = false;
    };
    std::vector<Layer> layers;

    array ln_final = array(0.0f, float32);
    array lm_head_w = array(0.0f, float32), lm_head_s = array(0.0f, float32), lm_head_b = array(0.0f, float32);
    array embed_w = array(0.0f, float32), embed_s = array(0.0f, float32), embed_b = array(0.0f, float32);

    int n_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int hidden_dim = 0;
    int group_size = 0;
    int bits = 4;
    float rope_theta = 0.0f;
    float rms_eps = 0.0f;

    // Custom RoPE frequencies for scaled positional encoding
    std::optional<array> rope_freqs;

    // Architecture-specific config
    bool has_norm_weight_offset = false;  // (1+w) RMSNorm formulation
    bool use_gelu = false;  // GELU instead of SiLU
    float query_pre_attn_scalar = 0.0f;  // Custom attention scale

    // Custom scaling multipliers (data-driven from config.json)
    float embedding_multiplier = 1.0f;  // Scales embedding output
    float attention_multiplier = 0.0f;  // Custom attention scale (0 = use 1/sqrt(head_dim))
    float residual_multiplier = 1.0f;   // Scales residual connections
    float logits_scaling = 1.0f;        // Scales output logits

    bool topology_initialized = false;
};

static FusedModelWeights* g_fused_weights = nullptr;

static constexpr uint8_t kLayerKindAttentionMlp = 0;
static constexpr uint8_t kLayerKindShortConv = 1;
static constexpr uint8_t kLayerKindMamba = 2;

static FusedModelWeights::Layer::LayerKind decode_layer_kind(uint8_t kind_id) {
    switch (kind_id) {
        case kLayerKindAttentionMlp:
            return FusedModelWeights::Layer::LayerKind::attention_mlp;
        case kLayerKindShortConv:
            return FusedModelWeights::Layer::LayerKind::shortconv;
        case kLayerKindMamba:
            return FusedModelWeights::Layer::LayerKind::mamba;
        default:
            throw std::invalid_argument("Unsupported fused layer kind id");
    }
}

struct QuantizedPipelineState {
    std::optional<array> current_token;
};

static std::mutex g_quant_pipeline_mu;
static std::unordered_map<void*, QuantizedPipelineState> g_quant_pipeline_states;

static inline void* quant_decode_state_key(void* cache_ptr, void* model_ptr) {
    return cache_ptr != nullptr ? cache_ptr : model_ptr;
}

// Async decode API keeps a single thread-local pending token. This path is not
// used by the scheduler pipeline; scheduler state is keyed by cache/model above.
static thread_local std::optional<array> g_pending_token;

extern "C" void* mlx_lazy_fused_moe_ffn_mxfp4(
    const void* input,
    const void* router_w,
    const void* router_s,
    const void* router_b,
    const void* router_bias,
    const void* gate_w,
    const void* gate_s,
    const void* up_w,
    const void* up_s,
    const void* down_w,
    const void* down_s,
    const void* gate_bias,
    const void* up_bias,
    const void* down_bias,
    size_t num_experts,
    size_t experts_per_token,
    size_t router_group_size,
    size_t expert_group_size);

extern "C" void* mlx_lazy_mla_attention_quantized(
    const void* input,
    const void* q_a_w, const void* q_a_s, const void* q_a_b, const void* q_a_norm_w,
    const void* q_b_w, const void* q_b_s, const void* q_b_b,
    const void* kv_a_w, const void* kv_a_s, const void* kv_a_b, const void* kv_a_norm_w,
    const void* kv_b_w, const void* kv_b_s, const void* kv_b_b,
    const void* o_w, const void* o_s, const void* o_b,
    void* cache_ptr, size_t layer_idx,
    size_t n_heads,
    size_t q_lora_rank, size_t kv_lora_rank,
    size_t qk_head_dim, size_t qk_rope_head_dim, size_t qk_nope_head_dim, size_t v_head_dim,
    size_t pos_offset, float rope_theta,
    const void* runtime_rope_cos, const void* runtime_rope_sin, size_t runtime_rope_dim,
    float rms_eps,
    size_t group_size, size_t bits
);

extern "C" void* mlx_lazy_mamba_block_quantized(
    const void* input,
    const void* ln1_weight,
    const void* in_w,
    const void* in_s,
    const void* in_b,
    const void* conv_weight,
    const void* conv_bias,
    const void* a_log,
    const void* d_skip,
    const void* dt_bias,
    const void* norm_weight,
    const void* out_w,
    const void* out_s,
    const void* out_b,
    const void* ln2_weight,
    const void* gate_up_w,
    const void* gate_up_s,
    const void* gate_up_b,
    const void* down_w,
    const void* down_s,
    const void* down_b,
    size_t group_size,
    size_t bits,
    bool use_gelu,
    float residual_multiplier,
    float norm_eps,
    void* mamba_cache_ptr,
    size_t layer_idx,
    size_t d_state,
    size_t d_conv,
    size_t n_heads,
    size_t d_head,
    size_t n_groups,
    uint8_t gate_up_layout
);
