//! Model Architecture Types
//!
//! Static model metadata contracts shared by model definitions, loader logic,
//! and inference runtime wiring.

/// Input to an operation - either a tensor reference or a scalar value.
pub const OpInput = union(enum) {
    tensor: []const u8, // Tensor name (e.g., "weight", "x")
    scalar: f32, // Scalar value (e.g., 1.0)
};

/// Operation types that can appear in model block metadata.
/// High-level ops (norm, multihead_attention, mlp) are preferred.
/// Low-level ops (linear, split, etc.) are used with TALU_PRIMITIVES_ONLY=1.
pub const OpType = enum {
    // High-level fused ops (preferred)
    norm,
    multihead_attention,
    mlp,
    moe,
    mamba_mixer, // Mamba2 SSM layer (monolithic)
    shortconv, // Gated short convolution (monolithic)

    // Residual connection
    add,

    // Low-level primitive ops (for debugging/custom architectures)
    mul,
    mean,
    pow,
    rsqrt,
    matmul,
    split,
    transpose,
    reshape,
    softmax,
    silu,
    gelu,
    embedding,
    linear,
    rope,
    triu,
    scaled_dot_product_attention,
};

/// A single operation in static model metadata.
pub const Op = struct {
    op_type: OpType,
    name: ?[]const u8 = null, // For norm: "input_layernorm", etc.
    inputs: []const OpInput = &.{}, // Inputs to the operation
    outputs: []const []const u8 = &.{}, // Output tensor names for dataflow tracking

    // Op-specific parameters
    weight_offset: f32 = 0.0, // For norm: add to weight before scaling (e.g., (1+w) formulation)
    qk_norm: bool = false, // For attention: apply QK normalization
    fused_qkv: bool = false, // For attention: weights are fused [Q,K,V]
    fused_gate_up: bool = false, // For mlp: weights are fused [gate,up]
    sliding_window: ?i32 = null, // For attention

    // MLA (Multi-Latent Attention) specific fields
    mla: bool = false, // For attention: use MLA (compressed Q/KV projections)
    q_lora_rank: ?u32 = null, // MLA: Q compression rank
    kv_lora_rank: ?u32 = null, // MLA: KV compression rank
    qk_head_dim: ?u32 = null, // MLA: total Q/K head dimension (rope + nope)
    qk_rope_head_dim: ?u32 = null, // MLA: dimension with RoPE applied
    qk_nope_head_dim: ?u32 = null, // MLA: dimension without RoPE
    v_head_dim: ?u32 = null, // MLA: value head dimension
    rope_interleave: bool = true, // MLA: use interleaved RoPE layout
    activation: ?[]const u8 = null, // For ffn: "silu", "gelu", "relu"
    num_experts: i32 = 0, // For moe
    experts_per_token: i32 = 0, // For moe
    scale: f32 = 1.0, // For residual_add
    num_outputs: i32 = 0, // For split: number of output tensors
    dim: i32 = -1, // For split/softmax: dimension to operate on
    dim0: i32 = -1, // For transpose: first dimension
    dim1: i32 = -1, // For transpose: second dimension
    keepdim: bool = false, // For mean
    exponent: f32 = 1.0, // For pow
    shape: []const i32 = &.{}, // For reshape
    split_sizes: []const i32 = &.{}, // For split: sizes of each output
    is_causal: bool = true, // For sdpa: apply causal mask
    sdpa_scale: ?f32 = null, // For sdpa: explicit scale (null = use 1/sqrt(head_dim))

    // Mamba-specific fields (for mamba_mixer op)
    d_state: ?u32 = null, // SSM state dimension (e.g., 128)
    d_conv: ?u32 = null, // Convolution kernel size (e.g., 4) - also used by shortconv
    n_heads: ?u32 = null, // Number of SSM heads (e.g., 48)
    d_head: ?u32 = null, // Head dimension (e.g., 32)
    n_groups: ?u32 = null, // Groups for B/C projection (e.g., 1)
    d_inner: ?u32 = null, // Inner dimension (n_heads * d_head)

    // ShortConv-specific fields (for shortconv op)
    conv_dim: ?u32 = null, // Intermediate dimension
    conv_dim_out: ?u32 = null, // Output dimension
    conv_bias: bool = false, // Whether conv uses bias
};

/// Weight layout hints for loading and transformation.
pub const WeightLayout = enum {
    none,
    linear,
    conv1d_depthwise,
    embedding,
    gaffine,
};

/// Weight transforms applied after loading.
pub const WeightTransform = enum {
    transpose,
    maybe_transpose,
    quantize_gaffine,
    quantize_fp8,
    dtype_f32,
};

/// Declarative weight specification for architecture-driven loading.
pub const WeightSpec = struct {
    /// Internal ID for lookup (relative path from block).
    id: []const u8,
    /// Candidate name templates to try (expand {d} to layer index).
    candidates: []const []const u8,
    /// PyTorch module type name (Linear, Conv1d, Embedding, etc.).
    module_type: []const u8,
    /// Weight layout for transform hints.
    layout: WeightLayout,
    /// Expected dtype (for validation).
    dtype: []const u8,
    /// Whether weight is required (error if not found).
    required: bool,
    /// Expected shape (optional).
    expected_shape: ?[]const usize = null,
    /// Optional transforms to apply.
    transforms: []const WeightTransform = &.{},
};

/// A block variant for heterogeneous models (e.g., Mamba + Attention).
/// Maps an alternate config.json layer_type string to a variant index.
pub const VariantAlias = struct {
    alias: []const u8,
    variant_index: u8,
};

pub const BlockVariant = struct {
    name: []const u8,
    ops: []const Op,
    weights: []const WeightSpec = &.{},
};

/// A registered architecture definition.
/// Contains static operation metadata and weight contracts.
pub const Architecture = struct {
    name: []const u8,
    model_types: []const []const u8,

    // Static op metadata (homogeneous models use block_ops, heterogeneous use block_variants)
    block_ops: []const Op = &.{},
    pre_block_ops: []const Op = &.{},
    post_block_ops: []const Op = &.{},

    // Heterogeneous model support (e.g., Granite Hybrid with Mamba + Attention)
    // If block_variants is non-null, this is a heterogeneous model
    block_variants: ?[]BlockVariant = null,
    /// Maps layer index to variant index (e.g., [0,0,0,1,0,0,1,...] for Mamba=0, Attention=1)
    layer_map: ?[]const u8 = null,
    /// Maps alternate config.json layer_type strings to variant indices.
    /// E.g., config uses "conv" but architecture variant is "shortconv" at index 0.
    variant_aliases: ?[]const VariantAlias = null,

    // Weight specs for homogeneous blocks.
    block_weights: []const WeightSpec = &.{},
    // Global weights (embeddings, final norm, lm_head).
    global_weights: []const WeightSpec = &.{},
    // Common prefixes for block weight names (for candidate generation).
    // E.g., ["model.layers.{d}.", "layers.{d}."] - weights specify suffix only.
    weight_prefixes: []const []const u8 = &.{},

    // Flags derived from analyzing block_ops
    has_qk_norm: bool = false,
    has_moe: bool = false,
    has_mamba: bool = false, // Has mamba_mixer ops (heterogeneous)
    has_shortconv: bool = false, // Has shortconv ops (heterogeneous)
    has_mla: bool = false, // Has MLA (Multi-Latent Attention) ops
    has_fused_qkv: bool = false, // Fused QKV projection (single weight for Q+K+V)
    has_fused_gate_up: bool = false, // Fused gate_up projection (single weight for gate+up)
    num_norms_per_block: u8 = 2,
    use_gelu: bool = false,
    use_swiglu_oss: bool = false, // SwiGLU variant: alpha=1.702, ±7 clipping, (up+1)
    norm_weight_offset: f32 = 0.0,
    explicit_qk_norm_ops: bool = false,

    // Pre-block flags
    embedding_multiplier: f32 = 1.0, // Scaling factor after embedding (e.g., sqrt(hidden_size))

    /// Check if this is a heterogeneous model (multiple block variants)
    pub fn isHeterogeneous(self: *const Architecture) bool {
        return self.block_variants != null;
    }

    /// Get the variant index for a given layer (returns 0 for homogeneous models).
    /// If layer_types_override is provided (from model's config.json), it takes precedence
    /// over the architecture's layer_map.
    pub fn getVariantIndex(self: *const Architecture, layer_idx: usize) u8 {
        return self.getVariantIndexWithOverride(layer_idx, null);
    }

    /// Get the variant index with an optional override from model config.
    /// The override allows different-sized models of the same architecture to have
    /// different layer arrangements (e.g., granite-4.0-h-350m vs granite-4.0-h-1b).
    pub fn getVariantIndexWithOverride(self: *const Architecture, layer_idx: usize, layer_types_override: ?[]const u8) u8 {
        // Priority 1: Use model-specific layer_types override if provided
        if (layer_types_override) |override| {
            if (layer_idx < override.len) {
                return override[layer_idx];
            }
        }
        // Priority 2: Fall back to architecture layer_map
        if (self.layer_map) |map| {
            if (layer_idx < map.len) {
                return map[layer_idx];
            }
        }
        return 0;
    }

    /// Get the block variant for a given layer
    pub fn getVariant(self: *const Architecture, layer_idx: usize) ?*BlockVariant {
        return self.getVariantWithOverride(layer_idx, null);
    }

    /// Get the block variant with an optional layer_types override
    pub fn getVariantWithOverride(self: *const Architecture, layer_idx: usize, layer_types_override: ?[]const u8) ?*BlockVariant {
        if (self.block_variants) |variants| {
            const idx = self.getVariantIndexWithOverride(layer_idx, layer_types_override);
            if (idx < variants.len) {
                return &variants[idx];
            }
        }
        return null;
    }
};
