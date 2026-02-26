//! Model Architecture Types
//!
//! Static model metadata contracts shared by model definitions, loader logic,
//! and inference runtime wiring.

const std = @import("std");
const tensor = @import("../tensor.zig");

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

    // Vision pipeline ops
    patch_embed,
    spatial_merge,
    deepstack_extract,
    scatter,
};

pub const MLAConfig = struct {
    q_lora_rank: u32 = 0,
    kv_lora_rank: u32 = 0,
    qk_head_dim: u32 = 0,
    qk_rope_head_dim: u32 = 0,
    qk_nope_head_dim: u32 = 0,
    v_head_dim: u32 = 0,
    rope_interleave: bool = true,
};

pub const MambaConfig = struct {
    d_state: u32 = 0,
    d_conv: u32 = 0,
    n_heads: u32 = 0,
    d_head: u32 = 0,
    n_groups: u32 = 1,
    d_inner: u32 = 0,
};

pub const ShortConvConfig = struct {
    d_conv: u32 = 0,
    conv_dim: u32 = 0,
    conv_dim_out: u32 = 0,
    has_bias: bool = false,
};

/// Loader-only metadata paired with a block program.
/// LayerOp remains the execution bytecode; KernelMeta carries static loader hints.
pub const KernelMeta = struct {
    is_causal: bool = true,
    mla_config: ?MLAConfig = null,
    mamba_config: ?MambaConfig = null,
    shortconv_config: ?ShortConvConfig = null,
};

/// Architecture-owned config parsing hook.
/// Receives both parsed text-config object and root config object.
pub const ConfigParseHook = *const fn (
    config_obj: std.json.ObjectMap,
    root_obj: std.json.ObjectMap,
    config: *tensor.ModelConfig,
) void;

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
    /// Primary tensor suffix for block weights, or full tensor name for globals.
    suffix: []const u8,
    /// Additional suffix/name aliases to probe if primary lookup misses.
    aliases: []const []const u8 = &.{},
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
    /// Force conversion to f32 regardless of module-type preservation policy.
    force_f32: bool = false,
};

/// A block variant for heterogeneous models (e.g., Mamba + Attention).
/// Maps an alternate config.json layer_type string to a variant index.
pub const VariantAlias = struct {
    alias: []const u8,
    variant_index: u8,
};

pub const BlockVariant = struct {
    name: []const u8,
    meta: ?KernelMeta = null,
    weights: []const WeightSpec = &.{},
};

/// Vision probing metadata consumed by inference vision runtime.
/// This keeps tensor-name conventions in the models contract instead of
/// inference-owned hardcoded string templates.
pub const VisionMetadata = struct {
    fused_qkv_probe_candidates: []const []const u8 = &.{},
    split_qkv_probe_candidates: []const []const u8 = &.{},

    patch_embed_candidates: []const []const u8 = &.{},
    patch_embed_bias_candidates: []const []const u8 = &.{},
    position_embed_candidates: []const []const u8 = &.{},
    post_norm_weight_candidates: []const []const u8 = &.{},
    post_norm_bias_candidates: []const []const u8 = &.{},
    merger_norm_weight_candidates: []const []const u8 = &.{},
    merger_norm_bias_candidates: []const []const u8 = &.{},
    merger_fc1_candidates: []const []const u8 = &.{},
    merger_fc1_bias_candidates: []const []const u8 = &.{},
    merger_fc2_candidates: []const []const u8 = &.{},
    merger_fc2_bias_candidates: []const []const u8 = &.{},

    ln1_weight_templates: []const []const u8 = &.{},
    ln1_bias_templates: []const []const u8 = &.{},
    ln2_weight_templates: []const []const u8 = &.{},
    ln2_bias_templates: []const []const u8 = &.{},
    fused_qkv_weight_templates: []const []const u8 = &.{},
    fused_qkv_bias_templates: []const []const u8 = &.{},
    split_q_weight_templates: []const []const u8 = &.{},
    split_q_bias_templates: []const []const u8 = &.{},
    split_k_weight_templates: []const []const u8 = &.{},
    split_k_bias_templates: []const []const u8 = &.{},
    split_v_weight_templates: []const []const u8 = &.{},
    split_v_bias_templates: []const []const u8 = &.{},
    out_proj_weight_templates: []const []const u8 = &.{},
    out_proj_bias_templates: []const []const u8 = &.{},
    fc1_weight_templates: []const []const u8 = &.{},
    fc1_bias_templates: []const []const u8 = &.{},
    fc2_weight_templates: []const []const u8 = &.{},
    fc2_bias_templates: []const []const u8 = &.{},

    deepstack_norm_weight_templates: []const []const u8 = &.{},
    deepstack_norm_bias_templates: []const []const u8 = &.{},
    deepstack_fc1_weight_templates: []const []const u8 = &.{},
    deepstack_fc1_bias_templates: []const []const u8 = &.{},
    deepstack_fc2_weight_templates: []const []const u8 = &.{},
    deepstack_fc2_bias_templates: []const []const u8 = &.{},

    depth_split_qproj_templates: []const []const u8 = &.{},
    depth_fused_qkv_templates: []const []const u8 = &.{},
    intermediate_fc1_templates: []const []const u8 = &.{},
};

/// Canonical block kinds for heterogeneous model topologies.
pub const BlockKind = enum {
    /// Standard transformer block with attention and FFN.
    attention_mlp,
    /// Mamba2 state-space mixer block.
    mamba,
    /// ShortConv gated convolution block.
    shortconv,

    /// Convert a variant name string to canonical block kind.
    pub fn fromVariantName(name: []const u8) ?BlockKind {
        const known = std.StaticStringMap(BlockKind).initComptime(.{
            .{ "attention", .attention_mlp },
            .{ "attention_mlp", .attention_mlp },
            .{ "transformer", .attention_mlp },
            .{ "full_attention", .attention_mlp },
            .{ "sliding_attention", .attention_mlp },
            .{ "linear_attention", .mamba },
            .{ "mamba", .mamba },
            .{ "mamba2", .mamba },
            .{ "ssm", .mamba },
            .{ "shortconv", .shortconv },
            .{ "conv", .shortconv },
        });
        return known.get(name);
    }
};

/// Loader/backend contract: maximum grouped-affine groups supported by runtime kernels.
pub const MAX_SUPPORTED_GAFFINE_GROUPS: usize = 1024;

/// Fused-model layer kind identifiers for Metal compute bindings.
pub const FusedLayerKindId = enum(u8) {
    attention_mlp = 0,
    shortconv = 1,
    mamba = 2,
};

/// Returns the fused-model layer id for kinds supported by fused Metal decode.
/// `null` means the kind is not supported by fused Metal execution.
pub fn fusedLayerKindId(kind: BlockKind) ?FusedLayerKindId {
    return switch (kind) {
        .attention_mlp => .attention_mlp,
        .shortconv => .shortconv,
        .mamba => .mamba,
    };
}

/// A registered architecture definition.
/// Contains static operation metadata and weight contracts.
pub const Architecture = struct {
    name: []const u8,
    model_types: []const []const u8,
    kernel_meta: KernelMeta = .{},
    parse_config_hook: ?ConfigParseHook = null,

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
    // Weight IDs used to infer d_ff from actual loaded tensors when config
    // fields are inconsistent with checkpoint shapes.
    d_ff_source_weight_ids: []const []const u8 = &.{},
    // Explicit opt-in for weight-driven d_ff resolution during loading.
    resolve_d_ff_from_weights: bool = false,
    // Optional shortconv depthwise weight id used for shape-driven dimension
    // inference when config omits conv_dim/conv_dim_out/d_conv.
    shortconv_dims_source_weight_id: ?[]const u8 = null,
    // Explicit opt-in for weight-driven shortconv dim resolution during loading.
    resolve_shortconv_dims_from_weights: bool = false,
    // Optional weight IDs used to probe original checkpoint dtype. When empty,
    // loader falls back to d_ff_source_weight_ids, then metadata order.
    weight_dtype_source_weight_ids: []const []const u8 = &.{},
    // Explicit opt-in for loader-side synthetic fused weights (QKV/GateUp).
    // Keep true for existing architectures; new families can disable to enforce
    // strict checkpoint-native weight contracts.
    enable_loader_fusions: bool = true,

    // Runtime flags declared on architecture metadata.
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

    // Vision metadata contract for runtime probing.
    vision: VisionMetadata = .{},

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
