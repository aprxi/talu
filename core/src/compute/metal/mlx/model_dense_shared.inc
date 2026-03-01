// ============================================================================
// Dense Model Structure
// ============================================================================

struct FusedDenseModel {
    struct Layer {
        enum class LayerKind : int {
            attention_mlp = 0,
            shortconv = 1,
            mamba = 2,
        };
        LayerKind kind = LayerKind::attention_mlp;

        array ln1_w = array(0.0f, float32);      // attention norm
        array qkv_proj = array(0.0f, bfloat16);  // Pre-concatenated Q+K+V
        array o_proj = array(0.0f, bfloat16);
        // MLA attention (dense)
        bool use_mla = false;
        int mla_n_heads = 0;
        int mla_q_lora_rank = 0;
        int mla_kv_lora_rank = 0;
        int mla_qk_head_dim = 0;
        int mla_qk_rope_head_dim = 0;
        int mla_qk_nope_head_dim = 0;
        int mla_v_head_dim = 0;
        array mla_q_a_w = array(0.0f, bfloat16);
        array mla_q_b_w = array(0.0f, bfloat16);
        array mla_kv_a_w = array(0.0f, bfloat16);
        array mla_kv_b_w = array(0.0f, bfloat16);
        array mla_q_a_norm = array(0.0f, float32);
        array mla_kv_a_norm = array(0.0f, float32);
        array mla_o_w = array(0.0f, bfloat16);
        // ShortConv mixer (dense): in-proj/out-proj are pre-transposed.
        int shortconv_d_conv = 0;
        int shortconv_conv_dim = 0;
        array shortconv_in_proj_t = array(0.0f, bfloat16);
        array shortconv_out_proj_t = array(0.0f, bfloat16);
        std::optional<array> shortconv_kernel_broadcast;
        std::optional<array> shortconv_bias_row;
        // Mamba mixer (dense core; optional dense FFN)
        int mamba_d_state = 0;
        int mamba_d_conv = 0;
        int mamba_n_heads = 0;
        int mamba_d_head = 0;
        int mamba_n_groups = 1;
        uint8_t mamba_gate_up_layout = 0;
        array mamba_conv_weight = array(0.0f, float32);
        std::optional<array> mamba_conv_bias;
        array mamba_a_log = array(0.0f, float32);
        array mamba_d_skip = array(0.0f, float32);
        std::optional<array> mamba_dt_bias;
        std::optional<array> mamba_norm_weight;
        array mamba_in_proj = array(0.0f, bfloat16);
        array mamba_out_proj = array(0.0f, bfloat16);
        std::optional<array> mamba_gate_up;
        std::optional<array> mamba_down_proj;
        array ln2_w = array(0.0f, float32);      // ffn norm
        array gate_up_proj = array(0.0f, bfloat16);  // Pre-concatenated gate+up
        array down_proj = array(0.0f, bfloat16);
        std::optional<array> q_norm;
        std::optional<array> k_norm;
        int q_size = 0;    // For splitting QKV result
        int kv_size = 0;
    };
    std::vector<Layer> layers;

    array ln_final = array(0.0f, float32);
    array lm_head = array(0.0f, bfloat16);  // Pre-transposed
    array embed_tokens = array(0.0f, bfloat16);

    int n_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int hidden_dim = 0;
    float rope_theta = 0.0f;
    float rms_eps = 0.0f;
    bool topology_initialized = false;
    bool has_norm_weight_offset = false;
    bool use_gelu = false;
    float query_pre_attn_scalar = 0.0f;
    float embedding_multiplier = 1.0f;
    float attention_multiplier = 0.0f;
    float residual_multiplier = 1.0f;
    float logits_scaling = 1.0f;

    // Per-model decode constants (avoid thread-local cross-thread/model drift).
    bool decode_shapes_initialized = false;
    Shape q_shape = {1, 1, 1, 1};
    Shape kv_shape = {1, 1, 1, 1};
    Shape attn_out_shape = {1, 1, 1};
    Shape token_shape = {1, 1};
    float attn_scale = 1.0f;
};

// Global model pointer (for pipeline access)
static FusedDenseModel* g_fused_dense = nullptr;

struct DensePipelineState {
    std::optional<array> current_token;
};

static std::mutex g_dense_pipeline_mu;
static std::unordered_map<void*, DensePipelineState> g_dense_pipeline_states;

static inline void* dense_decode_state_key(void* cache_ptr, void* model_ptr) {
    return cache_ptr != nullptr ? cache_ptr : model_ptr;
}

static constexpr uint8_t kLayerKindAttentionMlp = 0;
static constexpr uint8_t kLayerKindShortConv = 1;
static constexpr uint8_t kLayerKindMamba = 2;

static FusedDenseModel::Layer::LayerKind decode_layer_kind(uint8_t kind_id) {
    switch (kind_id) {
        case kLayerKindAttentionMlp:
            return FusedDenseModel::Layer::LayerKind::attention_mlp;
        case kLayerKindShortConv:
            return FusedDenseModel::Layer::LayerKind::shortconv;
        case kLayerKindMamba:
            return FusedDenseModel::Layer::LayerKind::mamba;
        default:
            throw std::invalid_argument("Unsupported dense layer kind id");
    }
}

// Keep fused dense kernels architecture-agnostic: weight orientation is inferred
// from tensor shapes, never from parameter names or model-family heuristics.
static array orient_matmul_rhs(
    const array& weight,
    int in_features,
    std::optional<int> out_features = std::nullopt,
    bool transpose_when_ambiguous = true
) {
    if (weight.ndim() != 2) {
        throw std::invalid_argument("[dense] expected 2D weight for matmul rhs");
    }
    const int rows = weight.shape(0);
    const int cols = weight.shape(1);

    const bool direct = rows == in_features && (!out_features.has_value() || cols == out_features.value());
    const bool transposed = cols == in_features && (!out_features.has_value() || rows == out_features.value());

    if (direct && !transposed) {
        return weight;
    }
    if (transposed && !direct) {
        return transpose(weight);
    }
    if (direct && transposed) {
        // Square matrix is ambiguous. Match the non-fused path convention
        // (incoming linear weights are treated as [out, in]) unless caller opts out.
        return transpose_when_ambiguous ? transpose(weight) : weight;
    }
    throw std::invalid_argument("[dense] weight orientation incompatible with matmul shape");
}

static array to_fast_metal_dtype(const array& arr) {
    // Keep dense decode matmuls on float16 to stay on the fast Metal kernels.
    // Reintroducing bf16 activations/cache here forces mixed-precision casts in
    // the hot path and substantially reduces per-token throughput.
    return astype(arr, float16);
}

static inline array ensure_float16(const array& arr) {
    return arr.dtype() == float16 ? arr : astype(arr, float16);
}

static inline array ensure_float32(const array& arr) {
    return arr.dtype() == float32 ? arr : astype(arr, float32);
}

extern "C" void* mlx_lazy_mla_attention_bf16(
    const void* input,
    const void* q_a_w, const void* q_a_norm_w, const void* q_b_w,
    const void* kv_a_w, const void* kv_a_norm_w, const void* kv_b_w,
    const void* o_w,
    void* cache_ptr, size_t layer_idx,
    size_t n_heads,
    size_t q_lora_rank, size_t kv_lora_rank,
    size_t qk_head_dim, size_t qk_rope_head_dim, size_t qk_nope_head_dim, size_t v_head_dim,
    size_t pos_offset, float rope_theta,
    const void* runtime_rope_cos, const void* runtime_rope_sin, size_t runtime_rope_dim,
    float rms_eps
);

extern "C" void* mlx_lazy_mamba_block_bf16(
    const void* input,
    const void* ln1_weight,
    const void* in_proj,
    const void* conv_weight,
    const void* conv_bias,
    const void* a_log,
    const void* d_skip,
    const void* dt_bias,
    const void* norm_weight,
    const void* out_proj,
    const void* ln2_weight,
    const void* gate_up,
    const void* down_proj,
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
