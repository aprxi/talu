//! Model configuration contract types.
//!
//! These types describe model architecture/runtime metadata after config.json
//! parsing. They are owned by `models/`; compute code should only consume the
//! tensor and dtype primitives it needs.

pub const QuantMethod = enum(c_int) {
    none = 0,
    gaffine = 1,
    mxfp4 = 2,
    native = 3, // reserved, not currently used
    fp8 = 4,
    mxfp8 = 5,
};

pub const RopeScaling = struct {
    rope_type: enum { none, llama3, linear, yarn } = .none,
    factor: f32 = 1.0,
    low_freq_factor: f32 = 1.0,
    high_freq_factor: f32 = 4.0,
    // YaRN parameters (defaults match reference implementations)
    beta_slow: f32 = 1.0,
    beta_fast: f32 = 32.0,
    attention_factor: f32 = 0.0,
    mscale: f32 = 0.0,
    mscale_all_dim: f32 = 0.0,
    truncate: bool = true,
    original_max_position_embeddings: i32 = 8192,
    /// Optional multimodal RoPE section sizes (model-defined).
    mrope_section: [3]u32 = .{ 0, 0, 0 },
    mrope_interleaved: bool = false,
};

pub const ModelArch = enum {
    custom,
};

pub const ModelRuntime = struct {
    /// Canonical architecture id resolved at load time (e.g. "llama3", "granite_hybrid").
    architecture_id: ?[]const u8 = null,
    /// Architecture capability flags copied from static model metadata.
    has_moe: bool = false,
    has_mamba: bool = false,
    has_gated_delta: bool = false,
    has_shortconv: bool = false,
    has_mla: bool = false,
    has_per_layer_branch: bool = false,
    weight_offset: f32 = 0.0,
    qk_norm_weight_offset: f32 = 0.0,
    explicit_qk_norm_ops: bool = false,
    use_swiglu_variant: bool = false, // SwiGLU variant: alpha=1.702, clipping, (up+1) formulation
    use_transposed_mxfp4: bool = false,
    norm_weights_pre_shifted: bool = false, // Weights already include norm offset; skip runtime sanitization
};

pub const ModelConfig = struct {
    vocab_size: i32,
    d_model: i32,
    n_layers: i32,
    n_heads: i32,
    n_kv_groups: i32,
    d_ff: i32,
    max_seq_len: i32,
    head_dim: i32,
    global_head_dim: i32 = 0,
    rope_dim: i32 = 0,
    rope_theta: f32,
    norm_eps: f32,
    gaffine_group_size: i32,
    gaffine_bits: i32 = 4,
    tie_word_embeddings: bool = true,
    num_experts: i32 = 0,
    experts_per_token: i32 = 0,
    attention_bias: bool = false,
    quant_method: QuantMethod = .none,
    rope_scaling: RopeScaling = .{},
    model_arch: ModelArch = .custom,
    use_gelu: bool = false,
    use_qk_norm: bool = false,
    query_pre_attn_scalar: f32 = 0,
    rope_local_theta: f32 = 0,
    sliding_window: i32 = 0,
    sliding_window_pattern: i32 = 0,
    embedding_multiplier: f32 = 1.0,
    attention_multiplier: f32 = 0,
    residual_multiplier: f32 = 1.0,
    logits_scaling: f32 = 1.0,
    final_logit_softcapping: f32 = 0.0,
    hidden_size_per_layer_input: i32 = 0,
    vocab_size_per_layer_input: i32 = 0,
    num_kv_shared_layers: i32 = 0,
    attention_k_eq_v: bool = false, // V projection shares K projection weights
    use_raw_rms_norm: bool = false, // Norms use w*x instead of (1+w)*x
    use_v_norm: bool = false, // Apply weightless RMSNorm to values before attention
    bos_token_id: ?i32 = null,
    // Mamba/SSM config (for heterogeneous models like Granite Hybrid)
    mamba_d_state: i32 = 0, // SSM state dimension (e.g., 128)
    mamba_d_conv: i32 = 0, // Convolution kernel size (e.g., 4)
    mamba_n_heads: i32 = 0, // Number of SSM heads
    mamba_d_head: i32 = 0, // Head dimension for Mamba
    mamba_n_groups: i32 = 1, // Groups for B/C projection
    mamba_expand: i32 = 2, // Expansion factor (d_inner = d_model * expand)
    // ShortConv config (for heterogeneous models)
    shortconv_d_conv: i32 = 0, // Convolution kernel size (L_cache, e.g., 3)
    shortconv_conv_dim: i32 = 0, // Intermediate dimension
    shortconv_conv_dim_out: i32 = 0, // Output dimension (usually = d_model)
    shortconv_has_bias: bool = false, // Whether conv has bias
    // Linear-attention / gated-delta config (for heterogeneous models like Qwen3.5)
    linear_num_key_heads: i32 = 0,
    linear_num_value_heads: i32 = 0,
    linear_key_head_dim: i32 = 0,
    linear_value_head_dim: i32 = 0,
    // Vision encoder config (for multimodal models)
    vision_hidden_size: i32 = 0,
    vision_depth: i32 = 0,
    vision_num_heads: i32 = 0,
    vision_intermediate_size: i32 = 0,
    projector_hidden_size: i32 = 0,
    vision_out_hidden_size: i32 = 0,
    vision_patch_size: i32 = 0,
    vision_spatial_merge_size: i32 = 0,
    vision_temporal_patch_size: i32 = 0,
    vision_num_position_embeddings: i32 = 0,
    vision_max_num_patches: i32 = 0,
    // Vision special token IDs (0 means "unset")
    image_token_id: i32 = 0,
    vision_start_token_id: i32 = 0,
    vision_end_token_id: i32 = 0,
    // Optional vision probe layer indexes (for model-specific deepstack-style injection).
    vision_probe_layer_count: u8 = 0,
    vision_probe_layers: [8]u16 = [_]u16{0} ** 8,
    /// Whether Flash Attention is compatible with this model's head_dim.
    flash_attn_compatible: bool = false,

    /// Layer types for heterogeneous models (e.g., [0, 0, 1, 0, ...] for mamba=0, attention=1).
    /// Parsed from config.json's `layer_types` array and mapped to variant indices.
    /// When present, this overrides the graph's hardcoded layer_map.
    layer_types: ?[]const u8 = null,

    pub fn initFlashAttnCompat(self: *ModelConfig) void {
        self.flash_attn_compatible = switch (self.head_dim) {
            64, 128 => true,
            else => false,
        };
    }
};
