//! CPU Block Kernel Containers + Scratch
//!
//! This file defines the CPU-side kernel structs and scratch buffers used by the
//! transformer engine (`src/model/*`).
//!
//! The *execution order* (the transformer "graph") lives in `src/model/`;
//! this file provides the concrete CPU kernels (attention/ffn/norm) and the per-layer
//! containers (`TransformerBlock`) that the engine references.

const std = @import("std");
const tensor = @import("../../../../tensor.zig");
const compute = @import("../../../../compute/root.zig");
const matmul = compute.cpu.matmul;
const registry = compute.registry;
const cpu_rowwise = compute.cpu.rowwise;
const cpu_copy = compute.cpu.tensor_copy;
const cpu_layout = compute.cpu.layout_transform;
const graph_runtime = @import("../graph.zig");
const ops = graph_runtime.layer_ops;
const fmt = @import("../kernels/describe_fmt.zig");
const runtime_mod = @import("runtime.zig");
const inspect = @import("../../../../xray/root.zig");
const trace = inspect.trace;
const log = @import("../../../../log.zig");
const progress_mod = @import("../../../../capi/progress.zig");
const topology = @import("../../topology.zig");

pub const BufferId = ops.BufferId;

// Import CPU kernels
const attn = @import("../kernels/attention.zig");
const ffn = @import("../kernels/ffn.zig");
const moe = @import("../kernels/moe.zig");
const norm = @import("../kernels/norm.zig");
const rope = @import("../kernels/rope.zig");
const embedding = @import("../kernels/embedding.zig");
const mamba = @import("../kernels/mamba.zig");
const shortconv = @import("../kernels/shortconv.zig");
const mla = @import("../kernels/mla_attention.zig");

const Tensor = tensor.Tensor;
const ModelConfig = tensor.ModelConfig;

// Re-export kernel types for external use
pub const AttnTemp = attn.AttnTemp;
pub const AttnCache = attn.AttnCache;
pub const FfnScratch = ffn.FfnScratch;
pub const MultiHeadAttention = attn.MultiHeadAttention;
pub const SwiGLU = ffn.SwiGLU;
pub const GateUpLayout = ffn.GateUpLayout;
pub const RMSNorm = norm.RMSNorm;
pub const RoPE = rope.RoPE;
pub const rmsnormForward = norm.rmsnormForward;
pub const gatherEmbeddings = embedding.gatherEmbeddings;
// MLA (Multi-Latent Attention) types
pub const MLAConfig = mla.MLAConfig;
pub const MLACache = mla.MLACache;
pub const MLATemp = mla.MLATemp;
pub const MLAttention = mla.MLAttention;

// Mamba kernel types (for heterogeneous models)
pub const MambaKernel = mamba.MambaKernel;
pub const MambaConfig = mamba.MambaConfig;
pub const MambaWeights = mamba.MambaWeights;
pub const MambaState = mamba.MambaState;
pub const MambaScratch = mamba.MambaScratch;

// ShortConv kernel types (for heterogeneous models)
pub const ShortConvKernel = shortconv.ShortConvKernel;
pub const ShortConvConfig = shortconv.ShortConvConfig;
pub const ShortConvWeights = shortconv.ShortConvWeights;
pub const ShortConvState = shortconv.ShortConvState;
pub const ShortConvScratch = shortconv.ShortConvScratch;

pub const CpuKernel = runtime_mod.CpuKernel;

pub const NUM_TMP_BUFFERS = runtime_mod.NUM_TMP_BUFFERS;
pub const ScratchBuffer = runtime_mod.ScratchBuffer;

/// Pre-concatenated weights for fused kernels (optional per block).
pub const FusedBlockWeights = struct {
    qkv_proj: ?Tensor = null,
    gate_up: ?Tensor = null,
    gate_up_layout: ffn.GateUpLayout = .concat,
};

/// MoE (Mixture of Experts) weights for a layer
pub const MoEWeights = struct {
    router_weight: Tensor,
    router_bias: ?[]const f32 = null,
    experts: []moe.ExpertWeights,
    num_experts: usize,
    experts_per_token: usize,
    use_mxfp4: bool = false,
};

/// Weights for an attention + MLP transformer block.
pub const AttentionMlpWeights = struct {
    ln1_weight: *const Tensor, // input_layernorm
    ln2_weight: *const Tensor, // post_attention_layernorm
    ln1_bias: ?*const Tensor = null, // input_layernorm bias (LayerNorm models)
    ln2_bias: ?*const Tensor = null, // post_attention_layernorm bias (LayerNorm models)
    // Q/K/V projections - optional when using native fused QKV
    q_proj: ?*const Tensor = null,
    k_proj: ?*const Tensor = null,
    v_proj: ?*const Tensor = null,
    o_proj: *const Tensor,
    // Dense FFN weights (null if MoE)
    w1: ?*const Tensor = null,
    w2: ?*const Tensor = null,
    w3: ?*const Tensor = null,
    // MLP biases (optional, for models like BERT with biased dense layers)
    w1_bias: ?*const Tensor = null,
    w2_bias: ?*const Tensor = null,
    rope: ?*rope.RoPE = null,
    /// Sliding window attention size for this layer (0 = disabled/global attention).
    /// Non-global layers may use a window (e.g., 512/1024).
    sliding_window: usize = 0,
    fused: FusedBlockWeights = .{},
    // QKNorm weights (optional per-head normalization)
    q_norm: ?*const Tensor = null,
    k_norm: ?*const Tensor = null,
    // Optional extra FFN norms (4 norms per block instead of 2)
    pre_ffn_norm: ?*const Tensor = null, // pre_feedforward_layernorm
    post_ffn_norm: ?*const Tensor = null, // post_feedforward_layernorm
    // Attention biases (optional, for models with biased attention)
    q_bias: ?[]const f32 = null,
    k_bias: ?[]const f32 = null,
    v_bias: ?[]const f32 = null,
    o_bias: ?[]const f32 = null,
    // MoE weights (null if dense FFN)
    moe_weights: ?*MoEWeights = null,
    // Attention sinks - per-head extra logit prepended to the score vector before softmax.
    sinks: ?[]const f32 = null,
    /// Whether attention is causal (default true). Set to false for encoder models (BERT).
    is_causal: bool = true,
    /// Graph block ops for this layer (determines kernel list ordering).
    /// Empty means default pre-norm order: [norm, attention, norm, mlp].
    block_ops: []const graph_runtime.Op = &.{},

    // === MLA (Multi-Latent Attention) weights ===
    // Used when mla_config is set. MLA uses compressed Q/KV projections.
    mla_config: ?mla.MLAConfig = null,
    // Q projection: two-stage compression (hidden -> q_lora_rank -> n_heads * qk_head_dim)
    q_a_proj: ?*const Tensor = null, // [q_lora_rank, hidden]
    q_a_norm: ?*const Tensor = null, // [q_lora_rank]
    q_b_proj: ?*const Tensor = null, // [n_heads * qk_head_dim, q_lora_rank]
    // KV projection: compressed with shared RoPE key
    kv_a_proj: ?*const Tensor = null, // [kv_lora_rank + qk_rope_head_dim, hidden]
    kv_a_norm: ?*const Tensor = null, // [kv_lora_rank]
    kv_b_proj: ?*const Tensor = null, // [n_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]

    /// Check if this block uses MLA attention.
    pub fn isMLA(self: *const AttentionMlpWeights) bool {
        return self.mla_config != null;
    }
};

/// Weights for a Mamba2 SSM block.
/// Granite Hybrid Mamba blocks have: norm -> mamba -> norm -> mlp
pub const MambaBlockWeights = struct {
    ln1_weight: *const Tensor, // input_layernorm (norm before mixer)
    ln2_weight: ?*const Tensor = null, // post_attention_layernorm (norm before MLP)
    config: mamba.MambaConfig,
    weights: mamba.MambaWeights,
    // FFN/MLP weights (shared_mlp for Granite Hybrid)
    fused_gate_up: ?FusedBlockWeights = null, // gate+up projection (input_linear)
    down_proj: ?*const Tensor = null, // down projection (output_linear)
};

/// Weights for a ShortConv block.
/// ShortConv blocks have: norm -> shortconv -> norm -> mlp
pub const ShortConvBlockWeights = struct {
    ln1_weight: *const Tensor, // operator_norm (norm before mixer)
    ln2_weight: ?*const Tensor = null, // ffn_norm (norm before MLP)
    config: shortconv.ShortConvConfig,
    weights: shortconv.ShortConvWeights,
    // FFN/MLP weights (either fused or separate w1/w2/w3)
    fused_gate_up: ?FusedBlockWeights = null, // gate+up projection (fused)
    w1: ?*const Tensor = null, // gate projection (separate)
    w2: ?*const Tensor = null, // down projection (always present)
    w3: ?*const Tensor = null, // up projection (separate)
};

/// Tagged union for block weights - supports heterogeneous models.
pub const BlockWeights = union(BlockType) {
    attention_mlp: AttentionMlpWeights,
    mamba: MambaBlockWeights,
    shortconv: ShortConvBlockWeights,
};

/// Map of weight IDs to tensor pointers (used by graph-driven loaders).
pub const WeightMap = std.StringHashMapUnmanaged(*const Tensor);

/// Context required to map weight IDs to block weight structs.
pub const BlockMapContext = struct {
    rope: ?*rope.RoPE = null,
    sliding_window: usize = 0,
    is_causal: bool = true,
    block_ops: []const graph_runtime.Op = &.{},
    mamba_config: ?mamba.MambaConfig = null,
    shortconv_config: ?shortconv.ShortConvConfig = null,
    mla_config: ?mla.MLAConfig = null,
    // MoE configuration (from model config)
    num_experts: usize = 0,
    experts_per_token: usize = 0,
    allocator: ?std.mem.Allocator = null,
};

/// Initialization context for building TransformerBlock instances from a weight map.
pub const BlockInitContext = struct {
    allocator: std.mem.Allocator,
    d_model: usize,
    d_ff: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    norm_eps: f32,
    runtime: tensor.ModelRuntime,
    residual_multiplier: f32,
    attention_scale: f32,
    use_gelu: bool,
    block_idx: usize,
    map_context: BlockMapContext,
};

fn getRequiredWeight(map: *const WeightMap, name: []const u8) !*const Tensor {
    if (map.get(name)) |weight| return weight;
    return error.MissingWeight;
}

/// Fuse separate gate (w1) and up (w3) projection weights into a single tensor.
/// This converts 2 separate matmuls into 1 fused matmul for the FFN forward pass.
///
/// Input:
///   w1 (gate): [d_model, d_ff]
///   w3 (up):   [d_model, d_ff]
///
/// Output:
///   fused: [d_model, 2*d_ff] with concat layout (gate columns, then up columns)
///
/// Memory layout: For each row i, fused[i, 0..d_ff] = w1[i, :], fused[i, d_ff..2*d_ff] = w3[i, :]
///
/// The returned tensor is owned by the caller and must be freed.
/// Returns null for quantized types (which have more complex layouts with scales/biases).
/// Fuse gate (w1) and up (w3) projection weights into a single tensor.
/// Works for both F32 [d_model, d_ff] and BF16/F16 [d_ff, d_model] layouts.
///
/// Layout detection:
/// - F32 weights: [d_model, d_ff] (in, out) -> fuse columns to [d_model, 2*d_ff]
/// - BF16/F16 weights: [d_ff, d_model] (out, in) -> fuse rows to [2*d_ff, d_model]
///
/// The FFN kernel detects the layout at runtime by checking shape against d_model/d_ff.
/// Returns null for quantized types (which have complex layouts with scales/biases).
fn fuseGateUpWeights(allocator: std.mem.Allocator, w1: *const Tensor, w3: *const Tensor) !?Tensor {
    return cpu_layout.fuseTwoProjectionWeights(allocator, w1, w3);
}

fn getOptionalWeight(map: *const WeightMap, name: []const u8) ?*const Tensor {
    return map.get(name);
}

fn getRequiredWeightAlias(map: *const WeightMap, name: []const u8, alias: []const u8) !*const Tensor {
    if (map.get(name)) |weight| return weight;
    if (map.get(alias)) |weight| return weight;
    return error.MissingWeight;
}

fn getOptionalWeightAlias(map: *const WeightMap, name: []const u8, alias: []const u8) ?*const Tensor {
    return map.get(name) orelse map.get(alias);
}

fn getBiasSlice(map: *const WeightMap, name: []const u8) !?[]const f32 {
    if (map.get(name)) |weight| {
        if (weight.dtype != .f32) return error.InvalidBiasDType;
        return weight.asSlice(f32);
    }
    return null;
}

fn getBiasSliceAlias(map: *const WeightMap, name: []const u8, alias: []const u8) !?[]const f32 {
    if (map.get(name)) |weight| {
        if (weight.dtype != .f32) return error.InvalidBiasDType;
        return weight.asSlice(f32);
    }
    if (map.get(alias)) |weight| {
        if (weight.dtype != .f32) return error.InvalidBiasDType;
        return weight.asSlice(f32);
    }
    return null;
}

/// Build MoE weights from weight map (graph-driven loading).
/// Returns null if no MoE weights are present in the map.
/// Supports two formats:
/// 1. Indexed format (Qwen3 MoE): mlp.experts.{N}.{gate_proj,up_proj,down_proj}.weight
/// 2. MXFP4 stacked format (GPT-OSS): mlp.experts.{gate_up_proj,down_proj}_{blocks,scales}
fn buildMoEWeightsFromMap(
    allocator: std.mem.Allocator,
    map: *const WeightMap,
    num_experts: usize,
    experts_per_token: usize,
) !?*MoEWeights {
    // Detect MoE by checking for router weight
    const router_weight_ptr = map.get("mlp.gate.weight") orelse
        map.get("mlp.router.weight") orelse
        return null;

    if (num_experts == 0) return null;

    // Try MXFP4 stacked format first (GPT-OSS)
    if (map.get("mlp.experts.gate_up_proj_blocks")) |gate_up_blocks| {
        return try buildMxfp4MoEWeights(allocator, map, router_weight_ptr, gate_up_blocks, num_experts, experts_per_token);
    }

    // Try indexed format (Qwen3 MoE)
    return try buildIndexedMoEWeights(allocator, map, router_weight_ptr, num_experts, experts_per_token);
}

/// Build MoE weights from MXFP4 stacked format (GPT-OSS style).
fn buildMxfp4MoEWeights(
    allocator: std.mem.Allocator,
    map: *const WeightMap,
    router_weight_ptr: *const Tensor,
    gate_up_blocks: *const Tensor,
    num_experts: usize,
    experts_per_token: usize,
) !*MoEWeights {
    // Get required MXFP4 tensors
    const gate_up_scales = map.get("mlp.experts.gate_up_proj_scales") orelse return error.MissingWeight;
    const down_blocks = map.get("mlp.experts.down_proj_blocks") orelse return error.MissingWeight;
    const down_scales = map.get("mlp.experts.down_proj_scales") orelse return error.MissingWeight;

    // Get optional biases
    const gate_up_bias_tensor = map.get("mlp.experts.gate_up_proj_bias");
    const down_bias_tensor = map.get("mlp.experts.down_proj_bias");
    const gate_up_bias_values: ?[]const f32 = if (gate_up_bias_tensor) |t| t.asSlice(f32) else null;
    const down_bias_values: ?[]const f32 = if (down_bias_tensor) |t| t.asSlice(f32) else null;

    // Get router bias
    const router_bias_tensor = map.get("mlp.router.bias");
    const router_bias: ?[]const f32 = if (router_bias_tensor) |t| t.asSlice(f32) else null;

    // Validate shapes
    if (gate_up_blocks.data_size % num_experts != 0) return error.InvalidShape;
    if (down_blocks.data_size % num_experts != 0) return error.InvalidShape;
    if (gate_up_scales.data_size % num_experts != 0) return error.InvalidShape;
    if (down_scales.data_size % num_experts != 0) return error.InvalidShape;
    if (gate_up_bias_values) |b| if (b.len % num_experts != 0) return error.InvalidShape;
    if (down_bias_values) |b| if (b.len % num_experts != 0) return error.InvalidShape;

    const gate_up_expert_bytes = gate_up_blocks.data_size / num_experts;
    const down_expert_bytes = down_blocks.data_size / num_experts;
    const gate_up_scale_expert_size = gate_up_scales.data_size / num_experts;
    const down_scale_expert_size = down_scales.data_size / num_experts;
    const gate_up_bias_expert_size: usize = if (gate_up_bias_values) |b| b.len / num_experts else 0;
    const down_bias_expert_size: usize = if (down_bias_values) |b| b.len / num_experts else 0;

    // Get output dimensions from tensor shapes
    // gate_up_blocks shape: [num_experts, d_ff*2, blocks_per_row, 16]
    const gate_up_out_dim: usize = if (gate_up_blocks.n_dims >= 2) @intCast(gate_up_blocks.shape[1]) else return error.InvalidShape;
    const down_out_dim: usize = if (down_blocks.n_dims >= 2) @intCast(down_blocks.shape[1]) else return error.InvalidShape;

    const gate_up_bytes_per_row: usize = gate_up_expert_bytes / gate_up_out_dim;
    const down_bytes_per_row: usize = down_expert_bytes / down_out_dim;

    // Build expert weights array
    var experts = try allocator.alloc(moe.ExpertWeights, num_experts);
    errdefer allocator.free(experts);

    for (0..num_experts) |e| {
        const gate_up_bytes = gate_up_blocks.data()[e * gate_up_expert_bytes ..][0..gate_up_expert_bytes];
        const down_bytes = down_blocks.data()[e * down_expert_bytes ..][0..down_expert_bytes];

        experts[e] = .{
            .gate_up_proj = Tensor.view(gate_up_bytes.ptr, &.{ @intCast(gate_up_out_dim), gate_up_bytes_per_row }, .mxfp4, gate_up_expert_bytes),
            .gate_up_scales = gate_up_scales.data()[e * gate_up_scale_expert_size ..][0..gate_up_scale_expert_size],
            .gate_up_bias = if (gate_up_bias_values) |b| b[e * gate_up_bias_expert_size ..][0..gate_up_bias_expert_size] else null,
            .down_proj = Tensor.view(down_bytes.ptr, &.{ @intCast(down_out_dim), down_bytes_per_row }, .mxfp4, down_expert_bytes),
            .down_scales = down_scales.data()[e * down_scale_expert_size ..][0..down_scale_expert_size],
            .down_bias = if (down_bias_values) |b| b[e * down_bias_expert_size ..][0..down_bias_expert_size] else null,
        };
    }

    // Allocate and populate MoEWeights
    const moe_weights = try allocator.create(MoEWeights);
    moe_weights.* = .{
        .router_weight = router_weight_ptr.*,
        .router_bias = router_bias,
        .experts = experts,
        .num_experts = num_experts,
        .experts_per_token = experts_per_token,
        .use_mxfp4 = true,
    };

    return moe_weights;
}

/// Build MoE weights from indexed format (Qwen3 MoE style).
fn buildIndexedMoEWeights(
    allocator: std.mem.Allocator,
    map: *const WeightMap,
    router_weight_ptr: *const Tensor,
    num_experts: usize,
    experts_per_token: usize,
) !*MoEWeights {
    // Build expert weights array
    var experts = try allocator.alloc(moe.ExpertWeights, num_experts);
    errdefer allocator.free(experts);

    var name_buf: [128]u8 = undefined;
    for (0..num_experts) |e| {
        // Get expert gate_proj
        const gate_name = std.fmt.bufPrint(&name_buf, "mlp.experts.{d}.gate_proj.weight", .{e}) catch return error.BufferOverflow;
        const gate_proj = map.get(gate_name) orelse return error.MissingWeight;

        // Get expert up_proj
        const up_name = std.fmt.bufPrint(&name_buf, "mlp.experts.{d}.up_proj.weight", .{e}) catch return error.BufferOverflow;
        const up_proj = map.get(up_name) orelse return error.MissingWeight;

        // Get expert down_proj
        const down_name = std.fmt.bufPrint(&name_buf, "mlp.experts.{d}.down_proj.weight", .{e}) catch return error.BufferOverflow;
        const down_proj = map.get(down_name) orelse return error.MissingWeight;

        experts[e] = .{
            .gate_proj = gate_proj.*,
            .up_proj = up_proj.*,
            .down_proj = down_proj.*,
        };
    }

    // Allocate and populate MoEWeights
    const moe_weights = try allocator.create(MoEWeights);
    moe_weights.* = .{
        .router_weight = router_weight_ptr.*,
        .router_bias = null, // Router bias not yet supported in weight loader
        .experts = experts,
        .num_experts = num_experts,
        .experts_per_token = experts_per_token,
        .use_mxfp4 = false,
    };

    return moe_weights;
}

/// Create a heap-allocated NormKernel, choosing LayerNorm when bias is present.
fn createNormKernel(
    allocator: std.mem.Allocator,
    weight: *const Tensor,
    bias: ?*const Tensor,
    dim: usize,
    eps: f32,
    weight_offset: f32,
    layer_idx: u16,
    trace_point: trace.TracePoint,
) !*norm.NormKernel {
    const ptr = try allocator.create(norm.NormKernel);
    errdefer allocator.destroy(ptr);
    if (bias != null) {
        ptr.* = .{ .layer = .{
            .weight = weight,
            .bias = bias,
            .dim = dim,
            .eps = eps,
            .layer_idx = layer_idx,
            .trace_point = trace_point,
        } };
    } else {
        ptr.* = .{ .rms = .{
            .weight = weight,
            .dim = dim,
            .eps = eps,
            .weight_offset = weight_offset,
            .layer_idx = layer_idx,
            .trace_point = trace_point,
        } };
    }
    return ptr;
}

/// Build block weights from a weight map (graph-driven loader).
pub fn blockWeightsFromMap(
    map: *const WeightMap,
    block_type: BlockType,
    context: BlockMapContext,
) !BlockWeights {
    switch (block_type) {
        .attention_mlp => {
            // Norm weights (try standard names, then alternate names)
            const ln1_weight = getOptionalWeight(map, "input_layernorm.weight") orelse
                try getRequiredWeight(map, "operator_norm.weight");
            const ln2_weight = getOptionalWeight(map, "post_attention_layernorm.weight") orelse
                try getRequiredWeight(map, "ffn_norm.weight");
            // Norm biases (present in LayerNorm models like BERT, absent in RMSNorm models)
            const ln1_bias = getOptionalWeight(map, "input_layernorm.bias");
            const ln2_bias = getOptionalWeight(map, "post_attention_layernorm.bias");

            // Output projection (try standard names, then alternate names)
            const o_proj = getOptionalWeightAlias(map, "self_attn.o_proj.weight", "mixer.o_proj.weight") orelse
                try getRequiredWeight(map, "self_attn.out_proj.weight");

            // Check for MLA weights first (takes precedence over standard Q/K/V)
            const q_a_proj = getOptionalWeight(map, "self_attn.q_a_proj.weight");
            const is_mla = q_a_proj != null;

            var q_a_norm: ?*const Tensor = null;
            var q_b_proj: ?*const Tensor = null;
            var kv_a_proj: ?*const Tensor = null;
            var kv_a_norm: ?*const Tensor = null;
            var kv_b_proj: ?*const Tensor = null;
            var q_proj: ?*const Tensor = null;
            var k_proj: ?*const Tensor = null;
            var v_proj: ?*const Tensor = null;
            var fused_qkv: ?*const Tensor = null;

            if (is_mla) {
                // MLA attention weights
                q_a_norm = try getRequiredWeight(map, "self_attn.q_a_layernorm.weight");
                q_b_proj = try getRequiredWeight(map, "self_attn.q_b_proj.weight");
                kv_a_proj = try getRequiredWeight(map, "self_attn.kv_a_proj_with_mqa.weight");
                kv_a_norm = try getRequiredWeight(map, "self_attn.kv_a_layernorm.weight");
                kv_b_proj = try getRequiredWeight(map, "self_attn.kv_b_proj.weight");
            } else {
                // Standard attention weights
                fused_qkv = getOptionalWeightAlias(map, "self_attn.qkv_proj.weight", "mixer.qkv_proj.weight");
                q_proj = if (fused_qkv == null) try getRequiredWeightAlias(map, "self_attn.q_proj.weight", "mixer.q_proj.weight") else null;
                k_proj = if (fused_qkv == null) try getRequiredWeightAlias(map, "self_attn.k_proj.weight", "mixer.k_proj.weight") else null;
                v_proj = if (fused_qkv == null) try getRequiredWeightAlias(map, "self_attn.v_proj.weight", "mixer.v_proj.weight") else null;
            }

            const pre_ffn_norm = getOptionalWeight(map, "pre_feedforward_layernorm.weight");
            const post_ffn_norm = getOptionalWeight(map, "post_feedforward_layernorm.weight");
            // QK norm (try standard names, then alternate names)
            const q_norm = getOptionalWeightAlias(map, "self_attn.q_norm.weight", "mixer.q_norm.weight") orelse
                getOptionalWeight(map, "self_attn.q_layernorm.weight");
            const k_norm = getOptionalWeightAlias(map, "self_attn.k_norm.weight", "mixer.k_norm.weight") orelse
                getOptionalWeight(map, "self_attn.k_layernorm.weight");

            const q_bias = try getBiasSliceAlias(map, "self_attn.q_proj.bias", "mixer.q_proj.bias");
            const k_bias = try getBiasSliceAlias(map, "self_attn.k_proj.bias", "mixer.k_proj.bias");
            const v_bias = try getBiasSliceAlias(map, "self_attn.v_proj.bias", "mixer.v_proj.bias");
            const o_bias = try getBiasSliceAlias(map, "self_attn.o_proj.bias", "mixer.o_proj.bias");
            // Attention sinks - per-head learnable logit prepended before softmax
            const sinks = try getBiasSlice(map, "self_attn.sinks");

            var fused = FusedBlockWeights{};
            if (fused_qkv) |fq| fused.qkv_proj = fq.*;

            // Get MLA config from context if this is an MLA model
            const mla_config = if (is_mla) context.mla_config else null;

            // Build MoE weights from map if this is an MoE layer
            const moe_weights: ?*MoEWeights = if (context.allocator) |alloc|
                try buildMoEWeightsFromMap(alloc, map, context.num_experts, context.experts_per_token)
            else
                null;

            var w1: ?*const Tensor = null;
            var w2: ?*const Tensor = null;
            var w3: ?*const Tensor = null;
            var w1_bias: ?*const Tensor = null;
            var w2_bias: ?*const Tensor = null;
            if (moe_weights == null) {
                const fused_gate_up = getOptionalWeight(map, "mlp.gate_up_proj.weight") orelse
                    getOptionalWeight(map, "mlp.input_linear.weight");
                if (fused_gate_up) |fg| {
                    fused.gate_up = fg.*;
                } else {
                    // FFN weights (try standard names, then alternate names, then dense MLP names)
                    w1 = getOptionalWeight(map, "mlp.gate_proj.weight") orelse
                        getOptionalWeight(map, "feed_forward.w1.weight") orelse
                        getOptionalWeight(map, "mlp.dense_in.weight");
                    // Up projection is optional — absent in dense-only MLPs (BERT)
                    w3 = getOptionalWeight(map, "mlp.up_proj.weight") orelse
                        getOptionalWeight(map, "feed_forward.w3.weight");
                    if (w1 == null) return error.MissingWeight;
                }
                // Down projection (try standard names, then alternate names)
                w2 = getOptionalWeight(map, "mlp.down_proj.weight") orelse
                    getOptionalWeight(map, "mlp.output_linear.weight") orelse
                    getOptionalWeight(map, "feed_forward.w2.weight") orelse
                    getOptionalWeight(map, "mlp.dense_out.weight") orelse
                    return error.MissingWeight;
                // MLP biases (dense-only models like BERT)
                w1_bias = getOptionalWeight(map, "mlp.gate_proj.bias") orelse
                    getOptionalWeight(map, "mlp.dense_in.bias");
                w2_bias = getOptionalWeight(map, "mlp.down_proj.bias") orelse
                    getOptionalWeight(map, "mlp.dense_out.bias");
            }

            return BlockWeights{
                .attention_mlp = .{
                    .ln1_weight = ln1_weight,
                    .ln2_weight = ln2_weight,
                    .ln1_bias = ln1_bias,
                    .ln2_bias = ln2_bias,
                    .q_proj = q_proj,
                    .k_proj = k_proj,
                    .v_proj = v_proj,
                    .o_proj = o_proj,
                    .w1 = w1,
                    .w2 = w2,
                    .w3 = w3,
                    .w1_bias = w1_bias,
                    .w2_bias = w2_bias,
                    .rope = context.rope,
                    .sliding_window = context.sliding_window,
                    .fused = fused,
                    .q_norm = q_norm,
                    .k_norm = k_norm,
                    .pre_ffn_norm = pre_ffn_norm,
                    .post_ffn_norm = post_ffn_norm,
                    .q_bias = q_bias,
                    .k_bias = k_bias,
                    .v_bias = v_bias,
                    .o_bias = o_bias,
                    .moe_weights = moe_weights,
                    .sinks = sinks,
                    .is_causal = context.is_causal,
                    .block_ops = context.block_ops,
                    // MLA weights (null if standard attention)
                    .mla_config = mla_config,
                    .q_a_proj = q_a_proj,
                    .q_a_norm = q_a_norm,
                    .q_b_proj = q_b_proj,
                    .kv_a_proj = kv_a_proj,
                    .kv_a_norm = kv_a_norm,
                    .kv_b_proj = kv_b_proj,
                },
            };
        },
        .mamba => {
            const mamba_config = context.mamba_config orelse return error.MissingMambaConfig;
            const ln1_weight = try getRequiredWeight(map, "input_layernorm.weight");
            const ln2_weight = getOptionalWeight(map, "post_attention_layernorm.weight");

            const in_proj = try getRequiredWeight(map, "mixer.in_proj.weight");
            const out_proj = try getRequiredWeight(map, "mixer.out_proj.weight");
            const conv1d_weight = try getRequiredWeight(map, "mixer.conv1d.weight");
            const conv1d_bias = getOptionalWeight(map, "mixer.conv1d.bias");
            const A_log = try getRequiredWeight(map, "mixer.A_log");
            const D = try getRequiredWeight(map, "mixer.D");
            const dt_bias = getOptionalWeight(map, "mixer.dt_bias");
            const norm_weight = getOptionalWeight(map, "mixer.norm.weight");

            const fused_gate_up = getOptionalWeight(map, "mlp.input_linear.weight") orelse
                getOptionalWeight(map, "mlp.gate_up_proj.weight");
            const down_proj = getOptionalWeight(map, "mlp.output_linear.weight") orelse
                getOptionalWeight(map, "mlp.down_proj.weight");

            var fused_gate_up_weights: ?FusedBlockWeights = null;
            if (fused_gate_up) |fg| {
                fused_gate_up_weights = .{ .gate_up = fg.*, .gate_up_layout = .concat };
            }

            return BlockWeights{ .mamba = .{
                .ln1_weight = ln1_weight,
                .ln2_weight = ln2_weight,
                .config = mamba_config,
                .weights = .{
                    .in_proj = in_proj,
                    .out_proj = out_proj,
                    .conv1d_weight = conv1d_weight,
                    .conv1d_bias = conv1d_bias,
                    .A_log = A_log,
                    .D = D,
                    .dt_bias = dt_bias,
                    .norm_weight = norm_weight,
                },
                .fused_gate_up = fused_gate_up_weights,
                .down_proj = down_proj,
            } };
        },
        .shortconv => {
            // ShortConv block weights
            const shortconv_config = context.shortconv_config orelse return error.MissingShortConvConfig;
            const ln1_weight = try getRequiredWeight(map, "operator_norm.weight");
            const ln2_weight = getOptionalWeight(map, "ffn_norm.weight");

            const in_proj = try getRequiredWeight(map, "conv.in_proj.weight");
            const out_proj = try getRequiredWeight(map, "conv.out_proj.weight");
            const conv1d_weight = try getRequiredWeight(map, "conv.conv.weight");
            // Conv bias is typically not used, but check for it anyway
            const conv1d_bias: ?*const Tensor = null;

            // FFN weights: check for fused first, then separate w1/w2/w3
            const fused_gate_up = getOptionalWeight(map, "feed_forward.gate_up_proj.weight") orelse
                getOptionalWeight(map, "mlp.gate_up_proj.weight");

            var fused_gate_up_weights: ?FusedBlockWeights = null;
            var w1: ?*const Tensor = null;
            var w2: ?*const Tensor = null;
            var w3: ?*const Tensor = null;

            if (fused_gate_up) |fg| {
                fused_gate_up_weights = .{ .gate_up = fg.*, .gate_up_layout = .concat };
            } else {
                // Separate FFN weights (try alternate names first, then standard)
                w1 = getOptionalWeight(map, "feed_forward.w1.weight") orelse
                    getOptionalWeight(map, "mlp.gate_proj.weight");
                w3 = getOptionalWeight(map, "feed_forward.w3.weight") orelse
                    getOptionalWeight(map, "mlp.up_proj.weight");
            }
            w2 = getOptionalWeight(map, "feed_forward.w2.weight") orelse
                getOptionalWeight(map, "mlp.down_proj.weight") orelse
                getOptionalWeight(map, "mlp.output_linear.weight");

            return BlockWeights{ .shortconv = .{
                .ln1_weight = ln1_weight,
                .ln2_weight = ln2_weight,
                .config = shortconv_config,
                .weights = .{
                    .in_proj = in_proj,
                    .out_proj = out_proj,
                    .conv1d_weight = conv1d_weight,
                    .conv1d_bias = conv1d_bias,
                },
                .fused_gate_up = fused_gate_up_weights,
                .w1 = w1,
                .w2 = w2,
                .w3 = w3,
            } };
        },
    }
}

/// FFN layer type - either dense SwiGLU or Mixture of Experts
pub const FfnLayer = union(enum) {
    swiglu: ffn.SwiGLU,
    moe_ffn: moe.MoEFFN,

    pub fn forward(self: *const FfnLayer, input_tensor: *const Tensor, output_tensor: *Tensor, scratch: *ScratchBuffer) !void {
        switch (self.*) {
            .swiglu => |*s| try s.forward(input_tensor, output_tensor, &scratch.ffn_scratch, &scratch.matmul_scratch),
            .moe_ffn => |*m| try m.forward(input_tensor, output_tensor, &scratch.moe_scratch, &scratch.matmul_scratch),
        }
    }

    pub fn getDModel(self: *const FfnLayer) usize {
        return switch (self.*) {
            .swiglu => |s| s.d_model,
            .moe_ffn => |m| m.d_model,
        };
    }

    /// Describe this FFN module for introspection/debugging.
    pub fn describe(self: *const FfnLayer, writer: anytype, indent: usize, show_kernels: bool) !void {
        switch (self.*) {
            .swiglu => |*layer| try describeSwiglu(layer, writer, indent, show_kernels),
            .moe_ffn => |*layer| try describeMoe(layer, writer, indent, show_kernels),
        }
    }

    /// Forward pass with performance tracing.
    pub fn forwardTraced(self: *const FfnLayer, input_tensor: *const Tensor, output_tensor: *Tensor, scratch: *ScratchBuffer) !void {
        try self.forward(input_tensor, output_tensor, scratch);
    }

    fn describeSwiglu(layer: *const ffn.SwiGLU, writer: anytype, indent: usize, show_kernels: bool) !void {
        try fmt.writeIndent(writer, indent);

        const activation: []const u8 = if (layer.use_gelu) "GELU" else "SiLU";
        try writer.print("MLP(intermediate_size={}, activation={s})\n", .{ layer.d_ff, activation });

        if (layer.fused_gate_up) |*fused| {
            try fmt.describeLinearLine(writer, indent + 2, "gate_up_proj", fused, null, layer.d_model, layer.d_ff * 2);
        } else {
            const gate_weight = layer.w1 orelse return error.MissingWeight;
            const up_weight = layer.w3 orelse return error.MissingWeight;

            try fmt.describeLinearLine(writer, indent + 2, "gate_proj", gate_weight, null, layer.d_model, layer.d_ff);
            try fmt.describeLinearLine(writer, indent + 2, "up_proj", up_weight, null, layer.d_model, layer.d_ff);
        }

        try fmt.describeLinearLine(writer, indent + 2, "down_proj", layer.w2, null, layer.d_ff, layer.d_model);

        if (show_kernels) {
            try fmt.writeIndent(writer, indent + 2);
            try writer.writeAll("Kernels:\n");
            try formatSwigluKernels(layer, writer, indent + 4);
        }
    }

    fn formatSwigluKernels(layer: *const ffn.SwiGLU, writer: anytype, indent: usize) !void {
        if (layer.fused_gate_up) |*fused| {
            try fmt.formatSeqMatmulOp(writer, indent, layer.d_model, layer.d_ff * 2, fused.dtype);
        } else {
            const gate_weight = layer.w1 orelse return;
            const up_weight = layer.w3 orelse return;

            try fmt.formatSeqMatmulOp(writer, indent, layer.d_model, layer.d_ff, gate_weight.dtype);
            try fmt.formatSeqMatmulOp(writer, indent, layer.d_model, layer.d_ff, up_weight.dtype);
        }

        // Activation
        if (layer.use_gelu) {
            const gelu_op = fmt.KernelOp{ .gelu = .{} };
            try gelu_op.format(writer, indent);
        } else {
            const silu_op = fmt.KernelOp{ .silu = .{} };
            try silu_op.format(writer, indent);
        }

        // Multiply gate * up
        const mul_op = fmt.KernelOp{ .mul = .{} };
        try mul_op.format(writer, indent);

        // Down projection
        try fmt.formatSeqMatmulOp(writer, indent, layer.d_ff, layer.d_model, layer.w2.dtype);
    }

    fn describeMoe(layer: *const moe.MoEFFN, writer: anytype, indent: usize, show_kernels: bool) !void {
        try fmt.writeIndent(writer, indent);
        try writer.print("MoE(num_experts={}, experts_per_token={}, d_ff={})\n", .{
            layer.num_experts,
            layer.experts_per_token,
            layer.d_ff,
        });

        try fmt.writeIndent(writer, indent + 2);
        try writer.print("(router): Linear(in={}, out={})\n", .{ layer.d_model, layer.num_experts });

        try fmt.writeIndent(writer, indent + 2);
        try writer.print("(experts): {}× FFN(d_model={}, d_ff={})\n", .{
            layer.num_experts,
            layer.d_model,
            layer.d_ff,
        });

        if (show_kernels) {
            try fmt.writeIndent(writer, indent + 2);
            try writer.writeAll("Kernels:\n");
            try formatMoeKernels(layer, writer, indent + 4);
        }
    }

    fn formatMoeKernels(layer: *const moe.MoEFFN, writer: anytype, indent: usize) !void {
        const route_op = fmt.KernelOp{ .moe_route = .{
            .num_experts = layer.num_experts,
            .experts_per_token = layer.experts_per_token,
        } };
        try route_op.format(writer, indent);

        try fmt.writeIndent(writer, indent);
        try writer.print("└─ expert_gate(x[seq, {}]) → [seq, {}] (×{} experts)\n", .{
            layer.d_model,
            layer.d_ff,
            layer.experts_per_token,
        });

        try fmt.writeIndent(writer, indent);
        try writer.print("└─ expert_up(x[seq, {}]) → [seq, {}]\n", .{ layer.d_model, layer.d_ff });

        const silu_op = fmt.KernelOp{ .silu = .{} };
        try silu_op.format(writer, indent);

        const mul_op = fmt.KernelOp{ .mul = .{} };
        try mul_op.format(writer, indent);

        try fmt.writeIndent(writer, indent);
        try writer.print("└─ expert_down(x[seq, {}]) → [seq, {}]\n", .{ layer.d_ff, layer.d_model });

        try fmt.writeIndent(writer, indent);
        try writer.writeAll("└─ weighted_sum(expert_outputs, routing_weights)\n");
    }
};

// =============================================================================
// Block Type System for Heterogeneous Models
// =============================================================================

/// Canonical block kinds are shared across backends.
pub const BlockType = topology.BlockKind;

/// Kernel container for a single transformer block.
/// Holds the kernel structs (attention, FFN) that the transformer engine references.
/// Note: The forward() logic is in `src/model/root.zig`, not here.
pub const TransformerBlock = struct {
    /// Ordered kernel list (ids align with graph compiler emission order).
    kernels: []CpuKernel,

    /// Common fields for all block types.
    residual_multiplier: f32 = 1.0,
    block_idx: usize = 0,

    /// Weight registry for primitive op execution.
    /// Maps weight names (e.g., "qkv_proj", "o_proj") to tensor pointers.
    /// Enables the `linear` primitive op to look up weights by name.
    weight_registry: std.StringHashMapUnmanaged(*const tensor.Tensor) = .{},

    /// Storage for fused weights (copied from BlockWeights to extend lifetime).
    /// These are referenced by weight_registry entries.
    fused_qkv_storage: ?tensor.Tensor = null,
    fused_gate_up_storage: ?tensor.Tensor = null,
    /// True if fused_gate_up_storage data was allocated by us (needs freeing)
    fused_gate_up_owned: bool = false,

    // =========================================================================
    // Kernel ownership (heap-allocated for pointer stability)
    // =========================================================================

    _ln1: *norm.NormKernel,
    _ln2: ?*norm.NormKernel = null,
    _pre_ffn_norm: ?*norm.NormKernel = null,
    _post_ffn_norm: ?*norm.NormKernel = null,
    _attn: ?*attn.MultiHeadAttention = null,
    _mla_attn: ?*mla.MLAttention = null,
    _mamba: ?*mamba.MambaKernel = null,
    _shortconv: ?*shortconv.ShortConvKernel = null,
    _ffn_layer: ?*FfnLayer = null,

    // =========================================================================
    // Type accessors
    // =========================================================================

    /// Check if this block is a Mamba block.
    pub fn isMamba(self: *const TransformerBlock) bool {
        return self._mamba != null;
    }

    /// Check if this block is a ShortConv block.
    pub fn isShortConv(self: *const TransformerBlock) bool {
        return self._shortconv != null;
    }

    /// Check if this block is an attention + MLP block (standard or MLA).
    pub fn isAttentionMlp(self: *const TransformerBlock) bool {
        return self._attn != null or self._mla_attn != null;
    }

    /// Check if this block uses MLA (Multi-Latent Attention).
    pub fn isMLA(self: *const TransformerBlock) bool {
        return self._mla_attn != null;
    }

    /// Get the Mamba kernel, or null if this is not a Mamba block.
    pub fn getMambaKernel(self: *const TransformerBlock) ?*const mamba.MambaKernel {
        return self._mamba;
    }

    /// Get the ShortConv kernel, or null if this is not a ShortConv block.
    pub fn getShortConvKernel(self: *const TransformerBlock) ?*const shortconv.ShortConvKernel {
        return self._shortconv;
    }

    /// Get the attention struct (only valid for standard attention_mlp blocks).
    pub fn getAttention(self: *const TransformerBlock) ?*const attn.MultiHeadAttention {
        return self._attn;
    }

    /// Get a mutable reference to the attention struct (only valid for standard attention_mlp blocks).
    pub fn getAttentionMut(self: *TransformerBlock) ?*attn.MultiHeadAttention {
        return self._attn;
    }

    /// Get the MLA attention struct (only valid for MLA attention_mlp blocks).
    pub fn getMLAAttention(self: *const TransformerBlock) ?*const mla.MLAttention {
        return self._mla_attn;
    }

    /// Get a mutable reference to the MLA attention struct (only valid for MLA attention_mlp blocks).
    pub fn getMLAAttentionMut(self: *TransformerBlock) ?*mla.MLAttention {
        return self._mla_attn;
    }

    /// Get the FFN layer (valid for both attention_mlp and mamba blocks with MLP).
    /// Returns a pointer to heap-allocated FfnLayer for kernel list stability.
    pub fn getFfnLayer(self: *const TransformerBlock) ?*const FfnLayer {
        return self._ffn_layer;
    }

    /// Get ln1 (input layernorm) - available for both block types.
    pub fn getLn1(self: *const TransformerBlock) *const norm.NormKernel {
        return self._ln1;
    }

    /// Get ln2 (post-attention layernorm for attention_mlp, optional post-mixer for mamba).
    pub fn getLn2(self: *const TransformerBlock) ?*const norm.NormKernel {
        return self._ln2;
    }

    /// Get pre_ffn_norm (only for attention_mlp blocks with this norm).
    pub fn getPreFfnNorm(self: *const TransformerBlock) ?*const norm.NormKernel {
        return self._pre_ffn_norm;
    }

    /// Get post_ffn_norm (only for attention_mlp blocks with this norm).
    pub fn getPostFfnNorm(self: *const TransformerBlock) ?*const norm.NormKernel {
        return self._post_ffn_norm;
    }

    pub fn deinit(self: *TransformerBlock, allocator: std.mem.Allocator) void {
        allocator.destroy(self._ln1);
        if (self._ln2) |p| allocator.destroy(p);
        if (self._pre_ffn_norm) |p| allocator.destroy(p);
        if (self._post_ffn_norm) |p| allocator.destroy(p);
        if (self._attn) |p| allocator.destroy(p);
        if (self._mamba) |p| allocator.destroy(p);
        if (self._shortconv) |p| {
            // Free transposed weight buffer before destroying kernel
            p.deinit();
            allocator.destroy(p);
        }
        if (self._ffn_layer) |p| allocator.destroy(p);

        // Free fused gate_up storage if we allocated it
        if (self.fused_gate_up_owned) {
            if (self.fused_gate_up_storage) |fg| {
                const data_ptr = fg.data_ptr orelse unreachable;
                const elem_size = fg.dtype.elementSize();
                const total_size = @as(usize, @intCast(fg.shape[0])) * @as(usize, @intCast(fg.shape[1])) * elem_size;
                allocator.free(@as([*]u8, @ptrCast(data_ptr))[0..total_size]);
            }
        }

        if (self.kernels.len > 0) allocator.free(self.kernels);
        self.weight_registry.deinit(allocator);
    }

    /// Initialize a block from a weight map.
    /// The caller must invoke initWeightRegistry() after storing the block
    /// in a stable location.
    pub fn fromMap(
        context: BlockInitContext,
        block_type: BlockType,
        weight_map: *const WeightMap,
    ) !TransformerBlock {
        const weights = try blockWeightsFromMap(weight_map, block_type, context.map_context);
        return TransformerBlock.init(
            context.allocator,
            context.d_model,
            context.d_ff,
            context.n_heads,
            context.n_kv_heads,
            context.head_dim,
            context.max_seq_len,
            weights,
            context.norm_eps,
            context.runtime,
            context.residual_multiplier,
            context.attention_scale,
            context.use_gelu,
            context.block_idx,
        );
    }

    pub fn init(
        allocator: std.mem.Allocator,
        d_model: usize,
        d_ff: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        weights: BlockWeights,
        norm_eps: f32,
        runtime: tensor.ModelRuntime,
        residual_multiplier: f32,
        attention_scale: f32,
        use_gelu: bool,
        block_idx: usize,
    ) !TransformerBlock {
        return switch (weights) {
            .attention_mlp => |attn_weights| try initAttentionMlp(
                allocator,
                d_model,
                d_ff,
                n_heads,
                n_kv_heads,
                head_dim,
                max_seq_len,
                attn_weights,
                norm_eps,
                runtime,
                residual_multiplier,
                attention_scale,
                use_gelu,
                block_idx,
            ),
            .mamba => |mamba_weights| try initMamba(
                allocator,
                d_model,
                d_ff,
                mamba_weights,
                norm_eps,
                runtime,
                residual_multiplier,
                use_gelu,
                block_idx,
            ),
            .shortconv => |shortconv_weights| try initShortConv(
                allocator,
                d_model,
                d_ff,
                shortconv_weights,
                norm_eps,
                runtime,
                residual_multiplier,
                use_gelu,
                block_idx,
            ),
        };
    }

    /// Initialize an attention + MLP block (standard transformer or MLA).
    fn initAttentionMlp(
        allocator: std.mem.Allocator,
        d_model: usize,
        d_ff: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        weights: AttentionMlpWeights,
        norm_eps: f32,
        runtime: tensor.ModelRuntime,
        residual_multiplier: f32,
        attention_scale: f32,
        use_gelu: bool,
        block_idx: usize,
    ) !TransformerBlock {
        // Check if this is an MLA (Multi-Latent Attention) block
        const is_mla = weights.isMLA();

        // Resolve matmul kernels at load time based on weight dtypes
        // For native fused QKV, q/k/v_proj are null
        const has_split_qkv = weights.q_proj != null and weights.k_proj != null and weights.v_proj != null;

        // Get attention matmul kernel dtype - use fused QKV if available, else separate Q
        // For MLA, use q_a_proj dtype
        const qkv_weight_dtype = if (is_mla)
            weights.q_a_proj.?.dtype
        else if (weights.fused.qkv_proj) |fq|
            fq.dtype
        else if (weights.q_proj) |q|
            q.dtype
        else
            return error.MissingAttentionWeights;

        const dk_qkv = try matmul.matmulKernel(qkv_weight_dtype);
        const dk_o = try matmul.matmulKernel(weights.o_proj.dtype);

        // Check if K/V have different dtypes than Q (some models mix different dtypes)
        // Only relevant when using separate projections
        const dk_k: ?matmul.DispatchedKernel = if (has_split_qkv and weights.k_proj.?.dtype != weights.q_proj.?.dtype)
            try matmul.matmulKernel(weights.k_proj.?.dtype)
        else
            null;
        const dk_v: ?matmul.DispatchedKernel = if (has_split_qkv and weights.v_proj.?.dtype != weights.q_proj.?.dtype)
            try matmul.matmulKernel(weights.v_proj.?.dtype)
        else
            null;

        const dk_qkv_fused: ?matmul.DispatchedKernel = if (weights.fused.qkv_proj) |fq|
            try matmul.matmulKernel(fq.dtype)
        else
            null;

        const kernel_name_k: ?[]const u8 = if (dk_k) |dk| dk.name else null;
        const kernel_name_v: ?[]const u8 = if (dk_v) |dk| dk.name else null;
        const kernel_name_qkv_fused: ?[]const u8 = if (dk_qkv_fused) |dk| dk.name else null;

        const flash_attention_fn: ?compute.cpu.simd.flash_attention.FlashAttentionFn = blk: {
            const kernel_val = registry.selectFlashAttentionForHeadDim(head_dim) catch |err| switch (err) {
                error.UnsupportedHeadDim => break :blk null,
                else => return err,
            };
            switch (kernel_val) {
                .flash_attention => |fn_ptr| break :blk fn_ptr,
                else => return error.InvalidKernel,
            }
        };

        const ln1_ptr = try createNormKernel(allocator, weights.ln1_weight, weights.ln1_bias, d_model, norm_eps, runtime.weight_offset, @intCast(block_idx), .layer_attn_norm);
        errdefer allocator.destroy(ln1_ptr);

        const ln2_ptr = try createNormKernel(allocator, weights.ln2_weight, weights.ln2_bias, d_model, norm_eps, runtime.weight_offset, @intCast(block_idx), .layer_ffn_norm);
        errdefer allocator.destroy(ln2_ptr);

        var pre_ffn_norm_ptr: ?*norm.NormKernel = null;
        if (weights.pre_ffn_norm) |w| {
            pre_ffn_norm_ptr = try createNormKernel(allocator, w, null, d_model, norm_eps, runtime.weight_offset, @intCast(block_idx), .layer_ffn_norm);
        }
        errdefer if (pre_ffn_norm_ptr) |p| allocator.destroy(p);

        var post_ffn_norm_ptr: ?*norm.NormKernel = null;
        if (weights.post_ffn_norm) |w| {
            post_ffn_norm_ptr = try createNormKernel(allocator, w, null, d_model, norm_eps, runtime.weight_offset, @intCast(block_idx), .layer_ffn_norm);
        }
        errdefer if (post_ffn_norm_ptr) |p| allocator.destroy(p);

        // Create either MLA or standard attention kernel
        var attn_kernel: ?*attn.MultiHeadAttention = null;
        var mla_kernel: ?*mla.MLAttention = null;

        if (is_mla) {
            // MLA attention kernel
            const mla_cfg = weights.mla_config.?;
            const mla_ptr = try allocator.create(mla.MLAttention);
            errdefer allocator.destroy(mla_ptr);
            // MLA uses qk_head_dim for attention scaling, not standard head_dim
            const mla_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(mla_cfg.qk_head_dim)));
            mla_ptr.* = .{
                .d_model = d_model,
                .n_heads = n_heads,
                .max_seq_len = max_seq_len,
                .config = mla_cfg,
                .allocator = allocator,
                .q_a_proj = weights.q_a_proj.?,
                .q_a_norm = weights.q_a_norm.?,
                .q_b_proj = weights.q_b_proj.?,
                .kv_a_proj = weights.kv_a_proj.?,
                .kv_a_norm = weights.kv_a_norm.?,
                .kv_b_proj = weights.kv_b_proj.?,
                .o_proj = weights.o_proj,
                .rope = weights.rope,
                .norm_eps = norm_eps,
                .scale = mla_scale,
                .matmul_fn = dk_qkv.func,
                .layer_idx = @intCast(block_idx),
            };
            mla_kernel = mla_ptr;
        } else {
            // Standard attention kernel
            const attn_ptr = try allocator.create(attn.MultiHeadAttention);
            errdefer allocator.destroy(attn_ptr);
            attn_ptr.* = .{
                .d_model = d_model,
                .n_heads = n_heads,
                .n_kv_heads = n_kv_heads,
                .head_dim = head_dim,
                .max_seq_len = max_seq_len,
                .scale = attention_scale,
                .qk_norm_weight_offset = runtime.qk_norm_weight_offset,
                .sliding_window = weights.sliding_window,
                .layer_idx = @intCast(block_idx),
                .q_proj = weights.q_proj,
                .k_proj = weights.k_proj,
                .v_proj = weights.v_proj,
                .o_proj = weights.o_proj,
                .fused_qkv = weights.fused.qkv_proj,
                .rope = weights.rope,
                .q_norm = weights.q_norm,
                .k_norm = weights.k_norm,
                .norm_eps = norm_eps,
                .allocator = allocator,
                .matmul_qkv = dk_qkv.func,
                .matmul_k = if (dk_k) |dk| dk.func else null,
                .matmul_v = if (dk_v) |dk| dk.func else null,
                .matmul_qkv_fused = if (dk_qkv_fused) |dk| dk.func else null,
                .matmul_o = dk_o.func,
                .kernel_name_qkv = dk_qkv.name,
                .kernel_name_k = kernel_name_k,
                .kernel_name_v = kernel_name_v,
                .kernel_name_qkv_fused = kernel_name_qkv_fused,
                .kernel_name_o = dk_o.name,
                .q_bias = weights.q_bias,
                .k_bias = weights.k_bias,
                .v_bias = weights.v_bias,
                .o_bias = weights.o_bias,
                .sinks = weights.sinks,
                .is_causal = weights.is_causal,
                .flash_attention_fn = flash_attention_fn,
            };
            attn_kernel = attn_ptr;
        }

        const ffn_ptr = try allocator.create(FfnLayer);
        errdefer allocator.destroy(ffn_ptr);
        const is_moe = weights.moe_weights != null;

        // Track fused gate_up ownership at function scope (used in block return)
        var owned_fused_gate_up: ?Tensor = null;
        errdefer if (owned_fused_gate_up) |fg| {
            const data_ptr = fg.data_ptr orelse unreachable;
            const elem_size = fg.dtype.elementSize();
            const total_size = @as(usize, @intCast(fg.shape[0])) * @as(usize, @intCast(fg.shape[1])) * elem_size;
            allocator.free(@as([*]u8, @ptrCast(data_ptr))[0..total_size]);
        };

        if (weights.moe_weights) |moe_w| {
            log.debug("load", "MoE kernel config", .{
                .block = block_idx,
                .use_mxfp4 = moe_w.use_mxfp4,
                .use_swiglu_variant = runtime.use_swiglu_variant,
                .use_transposed_weights = runtime.use_transposed_mxfp4,
            }, @src());
            ffn_ptr.* = .{ .moe_ffn = .{
                .allocator = allocator,
                .d_model = d_model,
                .d_ff = d_ff,
                .num_experts = moe_w.num_experts,
                .experts_per_token = moe_w.experts_per_token,
                .router_weight = moe_w.router_weight,
                .router_bias = moe_w.router_bias,
                .experts = moe_w.experts,
                .use_mxfp4 = moe_w.use_mxfp4,
                .use_swiglu_variant = runtime.use_swiglu_variant,
                .use_transposed_weights = runtime.use_transposed_mxfp4,
                .layer_idx = @intCast(block_idx),
                .kernel_name = if (moe_w.use_mxfp4) "moe_mxfp4" else "moe_f32",
            } };
        } else {
            const w2 = weights.w2 orelse return error.MissingFFNWeights;
            var fused_gate_up = weights.fused.gate_up;

            // Fuse w1 and w3 at load time if they're separate (halves matmuls in FFN)
            // Only for non-quantized types - quantized have complex layouts with scales/biases.
            if (fused_gate_up == null and weights.w1 != null and weights.w3 != null) {
                if (try fuseGateUpWeights(allocator, weights.w1.?, weights.w3.?)) |fused_tensor| {
                    fused_gate_up = fused_tensor;
                    owned_fused_gate_up = fused_tensor; // Track for ownership
                }
            }

            const matmul_gate_dtype = if (weights.w1) |w1| w1.dtype else if (fused_gate_up) |fg| fg.dtype else return error.MissingFFNWeights;
            const dk_gate = try matmul.matmulKernel(matmul_gate_dtype);
            const dk_down = try matmul.matmulKernel(w2.dtype);
            const dk_gate_up: ?matmul.DispatchedKernel = if (fused_gate_up) |fg|
                try matmul.matmulKernel(fg.dtype)
            else
                null;
            const w1_bias_slice: ?[]const f32 = if (weights.w1_bias) |b| blk: {
                if (b.dtype != .f32) return error.InvalidBiasDType;
                break :blk b.asSlice(f32);
            } else null;
            const w2_bias_slice: ?[]const f32 = if (weights.w2_bias) |b| blk: {
                if (b.dtype != .f32) return error.InvalidBiasDType;
                break :blk b.asSlice(f32);
            } else null;

            ffn_ptr.* = .{
                .swiglu = .{
                    .d_model = d_model,
                    .d_ff = d_ff,
                    .use_gelu = use_gelu,
                    .use_swiglu_variant = runtime.use_swiglu_variant,
                    .layer_idx = @intCast(block_idx),
                    .w1 = if (fused_gate_up != null) null else weights.w1, // Clear if fused
                    .w2 = w2,
                    .w3 = if (fused_gate_up != null) null else weights.w3, // Clear if fused
                    .w1_bias = w1_bias_slice,
                    .w2_bias = w2_bias_slice,
                    .fused_gate_up = fused_gate_up,
                    .fused_gate_up_layout = weights.fused.gate_up_layout,
                    .allocator = allocator,
                    .matmul_gate = dk_gate.func,
                    .matmul_gate_up = if (dk_gate_up) |dk| dk.func else null,
                    .matmul_down = dk_down.func,
                    .kernel_name_gate = dk_gate.name,
                    .kernel_name_gate_up = if (dk_gate_up) |dk| dk.name else null,
                    .kernel_name_down = dk_down.name,
                },
            };
        }

        var kernel_list = std.ArrayListUnmanaged(CpuKernel){};
        errdefer kernel_list.deinit(allocator);

        // Build kernel list in graph op order (supports pre-norm and post-norm architectures).
        // Kernel-producing ops are: norm, multihead_attention, mlp/moe.
        // Non-kernel ops (add) are skipped here — they're handled by the compiled program.
        if (weights.block_ops.len > 0) {
            // All available norms in order: ln1, ln2, pre_ffn_norm, post_ffn_norm
            const norm_pool = [_]?*norm.NormKernel{ ln1_ptr, ln2_ptr, pre_ffn_norm_ptr, post_ffn_norm_ptr };
            var norm_idx: usize = 0;

            for (weights.block_ops) |op| {
                switch (op.op_type) {
                    .norm => {
                        // Skip norms that don't produce kernels (e.g., QK norm handled inside attention)
                        while (norm_idx < norm_pool.len) : (norm_idx += 1) {
                            if (norm_pool[norm_idx] != null) break;
                        }
                        if (norm_idx < norm_pool.len) {
                            try kernel_list.append(allocator, .{ .norm = norm_pool[norm_idx].? });
                            norm_idx += 1;
                        }
                    },
                    .multihead_attention => {
                        if (mla_kernel) |mla_k| {
                            try kernel_list.append(allocator, .{ .mla_attention = mla_k });
                        } else {
                            try kernel_list.append(allocator, .{ .attention = attn_kernel.? });
                        }
                    },
                    .mlp, .moe => {
                        if (is_moe) {
                            try kernel_list.append(allocator, .{ .moe = &ffn_ptr.moe_ffn });
                        } else {
                            try kernel_list.append(allocator, .{ .swiglu = &ffn_ptr.swiglu });
                        }
                    },
                    else => {}, // add, etc. — not kernels
                }
            }
        } else {
            // Default pre-norm order (backward compat when no block_ops provided)
            try kernel_list.append(allocator, .{ .norm = ln1_ptr });
            if (mla_kernel) |mla_k| {
                try kernel_list.append(allocator, .{ .mla_attention = mla_k });
            } else {
                try kernel_list.append(allocator, .{ .attention = attn_kernel.? });
            }
            try kernel_list.append(allocator, .{ .norm = ln2_ptr });

            if (pre_ffn_norm_ptr) |pre| {
                try kernel_list.append(allocator, .{ .norm = pre });
            }

            if (is_moe) {
                try kernel_list.append(allocator, .{ .moe = &ffn_ptr.moe_ffn });
            } else {
                try kernel_list.append(allocator, .{ .swiglu = &ffn_ptr.swiglu });
            }

            if (post_ffn_norm_ptr) |post| {
                try kernel_list.append(allocator, .{ .norm = post });
            }
        }

        const kernels = try kernel_list.toOwnedSlice(allocator);
        errdefer allocator.free(kernels);

        return TransformerBlock{
            .kernels = kernels,
            .residual_multiplier = residual_multiplier,
            .block_idx = block_idx,
            .fused_qkv_storage = weights.fused.qkv_proj,
            .fused_gate_up_storage = if (owned_fused_gate_up) |fg| fg else weights.fused.gate_up,
            .fused_gate_up_owned = owned_fused_gate_up != null,
            ._ln1 = ln1_ptr,
            ._ln2 = ln2_ptr,
            ._pre_ffn_norm = pre_ffn_norm_ptr,
            ._post_ffn_norm = post_ffn_norm_ptr,
            ._attn = attn_kernel,
            ._mla_attn = mla_kernel,
            ._mamba = null,
            ._ffn_layer = ffn_ptr,
        };
    }

    /// Initialize a Mamba2 SSM block.
    /// Granite Hybrid Mamba blocks have: norm -> mamba -> norm -> mlp
    fn initMamba(
        allocator: std.mem.Allocator,
        d_model: usize,
        d_ff: usize,
        weights: MambaBlockWeights,
        norm_eps: f32,
        runtime: tensor.ModelRuntime,
        residual_multiplier: f32,
        use_gelu: bool,
        block_idx: usize,
    ) !TransformerBlock {
        const in_proj_kernel = try registry.selectKernel(.matmul, .cpu, weights.weights.in_proj.dtype);
        const out_proj_kernel = try registry.selectKernel(.matmul, .cpu, weights.weights.out_proj.dtype);
        const ssm_kernel = try registry.selectKernel(.ssm_scan, .cpu, .f32);

        const matmul_in_proj = switch (in_proj_kernel) {
            .matmul => |fn_ptr| fn_ptr,
            else => return error.InvalidKernel,
        };
        const matmul_out_proj = switch (out_proj_kernel) {
            .matmul => |fn_ptr| fn_ptr,
            else => return error.InvalidKernel,
        };
        const ssm_scan = switch (ssm_kernel) {
            .ssm_scan => |fn_ptr| fn_ptr,
            else => return error.InvalidKernel,
        };

        const ln1_ptr = try createNormKernel(allocator, weights.ln1_weight, null, d_model, norm_eps, runtime.weight_offset, @intCast(block_idx), .layer_attn_norm);
        errdefer allocator.destroy(ln1_ptr);

        var ln2_ptr: ?*norm.NormKernel = null;
        if (weights.ln2_weight) |ln2_w| {
            ln2_ptr = try createNormKernel(allocator, ln2_w, null, d_model, norm_eps, runtime.weight_offset, @intCast(block_idx), .layer_ffn_norm);
        }
        errdefer if (ln2_ptr) |p| allocator.destroy(p);

        const mamba_ptr = try allocator.create(mamba.MambaKernel);
        errdefer allocator.destroy(mamba_ptr);
        mamba_ptr.* = mamba.MambaKernel.init(
            weights.config,
            weights.weights,
            matmul_in_proj,
            matmul_out_proj,
            ssm_scan,
        );
        mamba_ptr.layer_idx = @intCast(block_idx);

        // Build FFN layer if weights are present (Granite Hybrid has shared_mlp for all layers)
        var ffn_ptr: ?*FfnLayer = null;
        var fused_gate_up_storage: ?tensor.Tensor = null;

        if (weights.fused_gate_up) |fused| {
            // Fused gate+up projection from shared_mlp.input_linear
            if (fused.gate_up) |gate_up| {
                fused_gate_up_storage = gate_up;
                const down_proj = weights.down_proj orelse return error.InvalidConfiguration;
                const dk_gate_up = try matmul.matmulKernel(gate_up.dtype);
                const dk_down = try matmul.matmulKernel(down_proj.dtype);
                const ffn_layer_ptr = try allocator.create(FfnLayer);
                errdefer allocator.destroy(ffn_layer_ptr);
                ffn_layer_ptr.* = .{
                    .swiglu = .{
                        .d_model = d_model,
                        .d_ff = d_ff,
                        .use_gelu = use_gelu,
                        .layer_idx = @intCast(block_idx),
                        .w1 = null, // gate_proj
                        .w2 = down_proj, // down_proj (required)
                        .w3 = null, // up_proj
                        .fused_gate_up = gate_up,
                        .fused_gate_up_layout = fused.gate_up_layout,
                        .allocator = allocator,
                        .matmul_gate = dk_gate_up.func, // Reuse for both gate/up when fused
                        .matmul_gate_up = dk_gate_up.func,
                        .matmul_down = dk_down.func,
                        .kernel_name_gate = null,
                        .kernel_name_gate_up = dk_gate_up.name,
                        .kernel_name_down = dk_down.name,
                    },
                };
                ffn_ptr = ffn_layer_ptr;
            }
        }

        var kernel_list = std.ArrayListUnmanaged(CpuKernel){};
        errdefer kernel_list.deinit(allocator);

        try kernel_list.append(allocator, .{ .norm = ln1_ptr });
        try kernel_list.append(allocator, .{ .mamba = mamba_ptr });

        if (ln2_ptr) |n| {
            try kernel_list.append(allocator, .{ .norm = n });
        }

        if (ffn_ptr) |ffn_ptr_local| {
            try kernel_list.append(allocator, .{ .swiglu = &ffn_ptr_local.swiglu });
        }

        const kernels = try kernel_list.toOwnedSlice(allocator);
        errdefer allocator.free(kernels);

        return TransformerBlock{
            .kernels = kernels,
            .residual_multiplier = residual_multiplier,
            .block_idx = block_idx,
            .fused_gate_up_storage = fused_gate_up_storage,
            ._ln1 = ln1_ptr,
            ._ln2 = ln2_ptr,
            ._pre_ffn_norm = null,
            ._post_ffn_norm = null,
            ._attn = null,
            ._mamba = mamba_ptr,
            ._shortconv = null,
            ._ffn_layer = ffn_ptr,
        };
    }

    /// Initialize a ShortConv block.
    /// ShortConv blocks have: norm -> shortconv -> norm -> mlp
    fn initShortConv(
        allocator: std.mem.Allocator,
        d_model: usize,
        d_ff: usize,
        weights: ShortConvBlockWeights,
        norm_eps: f32,
        runtime: tensor.ModelRuntime,
        residual_multiplier: f32,
        use_gelu: bool,
        block_idx: usize,
    ) !TransformerBlock {
        const dk_in_proj = try matmul.matmulKernel(weights.weights.in_proj.dtype);
        const dk_out_proj = try matmul.matmulKernel(weights.weights.out_proj.dtype);

        const ln1_ptr = try createNormKernel(allocator, weights.ln1_weight, null, d_model, norm_eps, runtime.weight_offset, @intCast(block_idx), .layer_attn_norm);
        errdefer allocator.destroy(ln1_ptr);

        var ln2_ptr: ?*norm.NormKernel = null;
        if (weights.ln2_weight) |ln2_w| {
            ln2_ptr = try createNormKernel(allocator, ln2_w, null, d_model, norm_eps, runtime.weight_offset, @intCast(block_idx), .layer_ffn_norm);
        }
        errdefer if (ln2_ptr) |p| allocator.destroy(p);

        const shortconv_ptr = try allocator.create(shortconv.ShortConvKernel);
        errdefer allocator.destroy(shortconv_ptr);
        shortconv_ptr.* = shortconv.ShortConvKernel.init(
            weights.config,
            weights.weights,
            dk_in_proj.func,
            dk_out_proj.func,
            dk_in_proj.name,
            dk_out_proj.name,
        );
        shortconv_ptr.layer_idx = @intCast(block_idx);

        // Pre-transpose conv weights for SIMD-optimized depthwise convolution
        // This eliminates strided gather operations in the hot loop
        try shortconv_ptr.initTransposedWeights(allocator);

        // Build FFN layer if weights are present
        var ffn_ptr: ?*FfnLayer = null;
        var fused_gate_up_storage: ?tensor.Tensor = null;
        var fused_gate_up_owned: bool = false;

        if (weights.fused_gate_up) |fused| {
            if (fused.gate_up) |gate_up| {
                fused_gate_up_storage = gate_up;
                const down_proj = weights.w2 orelse return error.InvalidConfiguration;
                const dk_gate_up = try matmul.matmulKernel(gate_up.dtype);
                const dk_down = try matmul.matmulKernel(down_proj.dtype);
                const ffn_layer_ptr = try allocator.create(FfnLayer);
                errdefer allocator.destroy(ffn_layer_ptr);
                ffn_layer_ptr.* = .{ .swiglu = .{
                    .d_model = d_model,
                    .d_ff = d_ff,
                    .use_gelu = use_gelu,
                    .layer_idx = @intCast(block_idx),
                    .w1 = null,
                    .w2 = down_proj,
                    .w3 = null,
                    .fused_gate_up = gate_up,
                    .fused_gate_up_layout = fused.gate_up_layout,
                    .allocator = allocator,
                    .matmul_gate = dk_gate_up.func,
                    .matmul_gate_up = dk_gate_up.func,
                    .matmul_down = dk_down.func,
                    .kernel_name_gate = null,
                    .kernel_name_gate_up = dk_gate_up.name,
                    .kernel_name_down = dk_down.name,
                } };
                ffn_ptr = ffn_layer_ptr;
            }
        } else if (weights.w1 != null and weights.w2 != null and weights.w3 != null) {
            // Separate w1/w2/w3 weights - try to fuse w1 (gate) and w3 (up) at load time
            const w1 = weights.w1.?;
            const w2 = weights.w2.?;
            const w3 = weights.w3.?;

            // Try to fuse for non-quantized types. Quantized types need separate matmuls.
            const maybe_fused = try fuseGateUpWeights(allocator, w1, w3);
            if (maybe_fused) |fused_tensor| {
                fused_gate_up_storage = fused_tensor;
                fused_gate_up_owned = true; // We allocated this, need to free it

                const dk_gate_up = try matmul.matmulKernel(w1.dtype);
                const dk_down = try matmul.matmulKernel(w2.dtype);
                const ffn_layer_ptr = try allocator.create(FfnLayer);
                errdefer allocator.destroy(ffn_layer_ptr);
                ffn_layer_ptr.* = .{ .swiglu = .{
                    .d_model = d_model,
                    .d_ff = d_ff,
                    .use_gelu = use_gelu,
                    .layer_idx = @intCast(block_idx),
                    .w1 = null,
                    .w2 = w2,
                    .w3 = null,
                    .fused_gate_up = fused_tensor,
                    .fused_gate_up_layout = .concat,
                    .allocator = allocator,
                    .matmul_gate = dk_gate_up.func,
                    .matmul_gate_up = dk_gate_up.func,
                    .matmul_down = dk_down.func,
                    .kernel_name_gate = null,
                    .kernel_name_gate_up = dk_gate_up.name,
                    .kernel_name_down = dk_down.name,
                } };
                ffn_ptr = ffn_layer_ptr;
            } else {
                // Quantized types - use separate w1/w3 matmuls
                const dk_gate = try matmul.matmulKernel(w1.dtype);
                const dk_down = try matmul.matmulKernel(w2.dtype);
                const ffn_layer_ptr = try allocator.create(FfnLayer);
                errdefer allocator.destroy(ffn_layer_ptr);
                ffn_layer_ptr.* = .{ .swiglu = .{
                    .d_model = d_model,
                    .d_ff = d_ff,
                    .use_gelu = use_gelu,
                    .layer_idx = @intCast(block_idx),
                    .w1 = w1,
                    .w2 = w2,
                    .w3 = w3,
                    .fused_gate_up = null,
                    .fused_gate_up_layout = .concat,
                    .allocator = allocator,
                    .matmul_gate = dk_gate.func,
                    .matmul_gate_up = null,
                    .matmul_down = dk_down.func,
                    .kernel_name_gate = dk_gate.name,
                    .kernel_name_gate_up = null,
                    .kernel_name_down = dk_down.name,
                } };
                ffn_ptr = ffn_layer_ptr;
            }
        }

        var kernel_list = std.ArrayListUnmanaged(CpuKernel){};
        errdefer kernel_list.deinit(allocator);

        try kernel_list.append(allocator, .{ .norm = ln1_ptr });
        try kernel_list.append(allocator, .{ .shortconv = shortconv_ptr });

        if (ln2_ptr) |n| {
            try kernel_list.append(allocator, .{ .norm = n });
        }

        if (ffn_ptr) |ffn_ptr_local| {
            try kernel_list.append(allocator, .{ .swiglu = &ffn_ptr_local.swiglu });
        }

        const kernels = try kernel_list.toOwnedSlice(allocator);
        errdefer allocator.free(kernels);

        return TransformerBlock{
            .kernels = kernels,
            .residual_multiplier = residual_multiplier,
            .block_idx = block_idx,
            .fused_gate_up_storage = fused_gate_up_storage,
            .fused_gate_up_owned = fused_gate_up_owned,
            ._ln1 = ln1_ptr,
            ._ln2 = ln2_ptr,
            ._pre_ffn_norm = null,
            ._post_ffn_norm = null,
            ._attn = null,
            ._mamba = null,
            ._shortconv = shortconv_ptr,
            ._ffn_layer = ffn_ptr,
        };
    }

    /// Initialize the weight registry after the struct is at its final location.
    /// Must be called after TransformerBlock is stored in a stable heap location.
    pub fn initWeightRegistry(self: *TransformerBlock, allocator: std.mem.Allocator, weights: BlockWeights) !void {
        switch (weights) {
            .attention_mlp => |attn_weights| try self.initAttentionMlpRegistry(allocator, attn_weights),
            .mamba => |mamba_weights| try self.initMambaRegistry(allocator, mamba_weights),
            .shortconv => |shortconv_weights| try self.initShortConvRegistry(allocator, shortconv_weights),
        }
    }

    /// Initialize weight registry for attention + MLP blocks.
    fn initAttentionMlpRegistry(self: *TransformerBlock, allocator: std.mem.Allocator, weights: AttentionMlpWeights) !void {
        const register_weight_alias = struct {
            fn putIfMissing(map: *std.StringHashMapUnmanaged(*const tensor.Tensor), alloc: std.mem.Allocator, name: []const u8, weight: *const tensor.Tensor) !void {
                if (map.get(name) == null) {
                    try map.put(alloc, name, weight);
                }
            }
        }.putIfMissing;

        // Now we can safely store pointers to self.fused_*_storage fields
        if (self.fused_qkv_storage) |*fq| {
            try self.weight_registry.put(allocator, "qkv_proj", fq);
            try self.weight_registry.put(allocator, "self_attn.qkv_proj", fq);
        } else {
            if (weights.q_proj) |qp| try self.weight_registry.put(allocator, "q_proj", qp);
            if (weights.k_proj) |kp| try self.weight_registry.put(allocator, "k_proj", kp);
            if (weights.v_proj) |vp| try self.weight_registry.put(allocator, "v_proj", vp);
            if (weights.q_proj) |qp| try self.weight_registry.put(allocator, "self_attn.q_proj", qp);
            if (weights.k_proj) |kp| try self.weight_registry.put(allocator, "self_attn.k_proj", kp);
            if (weights.v_proj) |vp| try self.weight_registry.put(allocator, "self_attn.v_proj", vp);
        }
        try self.weight_registry.put(allocator, "o_proj", weights.o_proj);
        try self.weight_registry.put(allocator, "self_attn.o_proj", weights.o_proj);

        if (self.fused_gate_up_storage) |*fg| {
            try self.weight_registry.put(allocator, "gate_up_proj", fg);
            try self.weight_registry.put(allocator, "mlp.gate_up_proj", fg);
        } else {
            if (weights.w1) |w1| try self.weight_registry.put(allocator, "gate_proj", w1);
            if (weights.w3) |w3| try self.weight_registry.put(allocator, "up_proj", w3);
            if (weights.w1) |w1| try self.weight_registry.put(allocator, "mlp.gate_proj", w1);
            if (weights.w3) |w3| try self.weight_registry.put(allocator, "mlp.up_proj", w3);
        }
        if (weights.w2) |w2| try self.weight_registry.put(allocator, "down_proj", w2);
        if (weights.w2) |w2| try self.weight_registry.put(allocator, "mlp.down_proj", w2);

        try register_weight_alias(&self.weight_registry, allocator, "input_layernorm.weight", weights.ln1_weight);
        try register_weight_alias(&self.weight_registry, allocator, "post_attention_layernorm.weight", weights.ln2_weight);
        try register_weight_alias(&self.weight_registry, allocator, "ln1.weight", weights.ln1_weight);
        try register_weight_alias(&self.weight_registry, allocator, "ln2.weight", weights.ln2_weight);

        if (weights.pre_ffn_norm) |pre| {
            try register_weight_alias(&self.weight_registry, allocator, "pre_feedforward_layernorm.weight", pre);
            try register_weight_alias(&self.weight_registry, allocator, "pre_ffn_norm.weight", pre);
        }
        if (weights.post_ffn_norm) |post| {
            try register_weight_alias(&self.weight_registry, allocator, "post_feedforward_layernorm.weight", post);
            try register_weight_alias(&self.weight_registry, allocator, "post_ffn_norm.weight", post);
        }
        if (weights.q_norm) |q_norm| {
            try register_weight_alias(&self.weight_registry, allocator, "q_norm.weight", q_norm);
            try register_weight_alias(&self.weight_registry, allocator, "self_attn.q_norm.weight", q_norm);
        }
        if (weights.k_norm) |k_norm| {
            try register_weight_alias(&self.weight_registry, allocator, "k_norm.weight", k_norm);
            try register_weight_alias(&self.weight_registry, allocator, "self_attn.k_norm.weight", k_norm);
        }

        // MLA weights (if present)
        if (weights.q_a_proj) |w| {
            try register_weight_alias(&self.weight_registry, allocator, "self_attn.q_a_proj.weight", w);
        }
        if (weights.q_a_norm) |w| {
            try register_weight_alias(&self.weight_registry, allocator, "self_attn.q_a_layernorm.weight", w);
        }
        if (weights.q_b_proj) |w| {
            try register_weight_alias(&self.weight_registry, allocator, "self_attn.q_b_proj.weight", w);
        }
        if (weights.kv_a_proj) |w| {
            try register_weight_alias(&self.weight_registry, allocator, "self_attn.kv_a_proj_with_mqa.weight", w);
        }
        if (weights.kv_a_norm) |w| {
            try register_weight_alias(&self.weight_registry, allocator, "self_attn.kv_a_layernorm.weight", w);
        }
        if (weights.kv_b_proj) |w| {
            try register_weight_alias(&self.weight_registry, allocator, "self_attn.kv_b_proj.weight", w);
        }
    }

    /// Initialize weight registry for Mamba blocks.
    fn initMambaRegistry(self: *TransformerBlock, allocator: std.mem.Allocator, weights: MambaBlockWeights) !void {
        // Register Mamba-specific weight names
        try self.weight_registry.put(allocator, "input_layernorm.weight", weights.ln1_weight);
        try self.weight_registry.put(allocator, "norm.weight", weights.ln1_weight);
        if (weights.ln2_weight) |ln2_w| {
            try self.weight_registry.put(allocator, "post_mixer_layernorm.weight", ln2_w);
        }
        // Mamba mixer weights are accessed through the MambaKernel, not the registry
    }

    /// Initialize weight registry for ShortConv blocks.
    fn initShortConvRegistry(self: *TransformerBlock, allocator: std.mem.Allocator, weights: ShortConvBlockWeights) !void {
        // Register ShortConv-specific weight names
        try self.weight_registry.put(allocator, "operator_norm.weight", weights.ln1_weight);
        try self.weight_registry.put(allocator, "input_layernorm.weight", weights.ln1_weight);
        if (weights.ln2_weight) |ln2_w| {
            try self.weight_registry.put(allocator, "ffn_norm.weight", ln2_w);
            try self.weight_registry.put(allocator, "post_mixer_layernorm.weight", ln2_w);
        }
        // ShortConv mixer weights are accessed through the ShortConvKernel, not the registry
    }
};

/// Build CPU kernel blocks from loader-stage `BlockWeights`.
/// This preserves existing behavior from the prior `TransformerModel.init` path,
/// but returns the blocks directly so callers can store them in `LoadedModel`.
pub fn buildBlocks(
    allocator: std.mem.Allocator,
    config: ModelConfig,
    runtime: tensor.ModelRuntime,
    block_weights: []const BlockWeights,
    progress: progress_mod.ProgressContext,
) ![]TransformerBlock {
    if (block_weights.len != config.n_layers) return error.InvalidLayerCount;

    const block_array = try allocator.alloc(TransformerBlock, @intCast(config.n_layers));
    errdefer allocator.free(block_array);

    const default_attention_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(config.head_dim)));
    const attention_scale: f32 = if (config.attention_multiplier > 0)
        config.attention_multiplier
    else if (config.query_pre_attn_scalar > 0)
        1.0 / @sqrt(config.query_pre_attn_scalar)
    else
        default_attention_scale;

    const use_gelu_activation = config.use_gelu;
    for (block_array, block_weights, 0..) |*block_slot, block_weight, layer_idx| {
        block_slot.* = try TransformerBlock.init(
            allocator,
            @intCast(config.d_model),
            @intCast(config.d_ff),
            @intCast(config.n_heads),
            @intCast(config.n_kv_groups),
            @intCast(config.head_dim),
            @intCast(config.max_seq_len),
            block_weight,
            config.norm_eps,
            runtime,
            config.residual_multiplier,
            attention_scale,
            use_gelu_activation,
            layer_idx,
        );
        // Initialize weight registry now that block is in its final heap location
        try block_slot.initWeightRegistry(allocator, block_weight);
        progress.updateLine(1, @intCast(layer_idx + 1), null);
    }

    return block_array;
}

fn addInto(a: *const Tensor, b: *const Tensor, out: *Tensor) void {
    // Internal invariants: tensors must be f32 with matching sizes
    std.debug.assert(a.dtype == .f32 and b.dtype == .f32 and out.dtype == .f32);
    std.debug.assert(a.numel == b.numel and a.numel == out.numel);
    cpu_rowwise.addInto(a.asSlice(f32), b.asSlice(f32), out.asSlice(f32));
}

pub fn addIntoScaled(a: *const Tensor, b: *const Tensor, out: *Tensor, scale: f32) void {
    std.debug.assert(a.dtype == .f32 and b.dtype == .f32 and out.dtype == .f32);
    std.debug.assert(a.numel == b.numel and a.numel == out.numel);
    cpu_rowwise.addIntoScaled(a.asSlice(f32), b.asSlice(f32), out.asSlice(f32), scale);
}

pub fn copyTensor(src: *const Tensor, dst: *Tensor) void {
    // Internal invariant: tensors must have matching data size
    std.debug.assert(src.data_size == dst.data_size);
    cpu_copy.copyContiguous(dst.data(), src.data());
}

// =============================================================================
// Comprehensive Unit Tests
// =============================================================================

test "ScratchBuffer: init and deinit basic lifecycle" {
    const allocator = std.testing.allocator;
    const d_model = 128;
    const d_ff = 512;
    const n_layers = 4;

    var scratch = try ScratchBuffer.init(allocator, d_model, d_ff, n_layers);
    defer scratch.deinit();

    try std.testing.expectEqual(d_model, scratch.d_model);
    try std.testing.expectEqual(d_ff, scratch.d_ff);
    try std.testing.expectEqual(n_layers, scratch.attn_caches.len);

    // All tmp buffers should be initially empty
    for (scratch.tmp) |temp_slice| {
        try std.testing.expectEqual(@as(usize, 0), temp_slice.len);
    }
}

test "ScratchBuffer: ensure allocates all buffers correctly" {
    const allocator = std.testing.allocator;
    const d_model = 64;
    const d_ff = 256;
    const n_layers = 2;
    const seq_len = 4;

    var scratch = try ScratchBuffer.init(allocator, d_model, d_ff, n_layers);
    defer scratch.deinit();

    try scratch.ensure(seq_len);

    // Expected buffer length: seq_len * max(d_model, d_ff * 2)
    const max_dim = @max(d_model, d_ff * 2);
    const expected_buffer_len = seq_len * max_dim;

    // All tmp buffers should be allocated to the same size
    for (scratch.tmp) |temp_slice| {
        try std.testing.expectEqual(expected_buffer_len, temp_slice.len);
    }
}

test "ScratchBuffer: ensure handles fused projection size (2x d_ff)" {
    const allocator = std.testing.allocator;
    const d_model = 128;
    const d_ff = 512;
    const n_layers = 1;
    const seq_len = 2;

    var scratch = try ScratchBuffer.init(allocator, d_model, d_ff, n_layers);
    defer scratch.deinit();

    try scratch.ensure(seq_len);

    // With d_ff * 2 > d_model, buffer should be sized for fused gate_up
    const expected_buffer_len = seq_len * (d_ff * 2);
    try std.testing.expectEqual(expected_buffer_len, scratch.tmp[0].len);
}

test "ScratchBuffer: ensure is idempotent" {
    const allocator = std.testing.allocator;
    var scratch = try ScratchBuffer.init(allocator, 64, 256, 1);
    defer scratch.deinit();

    try scratch.ensure(4);
    const initial_ptr = scratch.tmp[0].ptr;
    const initial_len = scratch.tmp[0].len;

    // Calling ensure again with same seq_len should not reallocate
    try scratch.ensure(4);
    try std.testing.expectEqual(initial_ptr, scratch.tmp[0].ptr);
    try std.testing.expectEqual(initial_len, scratch.tmp[0].len);
}

test "ScratchBuffer: ensure grows buffer when needed" {
    const allocator = std.testing.allocator;
    var scratch = try ScratchBuffer.init(allocator, 64, 256, 1);
    defer scratch.deinit();

    try scratch.ensure(2);
    const initial_len = scratch.tmp[0].len;

    // Ensure with larger seq_len should reallocate
    try scratch.ensure(8);
    const expanded_len = scratch.tmp[0].len;

    try std.testing.expect(expanded_len > initial_len);
}

test "ScratchBuffer: getTmp returns correct buffer slice" {
    const allocator = std.testing.allocator;
    var scratch = try ScratchBuffer.init(allocator, 64, 256, 1);
    defer scratch.deinit();

    const seq_len = 4;
    try scratch.ensure(seq_len);

    // Test norm_out buffer (BufferId = 1)
    const norm_out = scratch.getTmp(.norm_out, 100);
    try std.testing.expectEqual(@as(usize, 100), norm_out.len);

    // Test branch_out buffer (BufferId = 2)
    const branch_out = scratch.getTmp(.branch_out, 200);
    try std.testing.expectEqual(@as(usize, 200), branch_out.len);

    // Test tmp3 buffer (BufferId = 3)
    const tmp3 = scratch.getTmp(.tmp3, 50);
    try std.testing.expectEqual(@as(usize, 50), tmp3.len);
}

test "ScratchBuffer: getTmp with different BufferIds returns different buffers" {
    const allocator = std.testing.allocator;
    var scratch = try ScratchBuffer.init(allocator, 64, 256, 1);
    defer scratch.deinit();

    try scratch.ensure(4);

    const norm_out = scratch.getTmp(.norm_out, 10);
    const branch_out = scratch.getTmp(.branch_out, 10);

    // Write different values to each buffer
    norm_out[0] = 1.0;
    branch_out[0] = 2.0;

    // Verify they are independent
    try std.testing.expectEqual(@as(f32, 1.0), norm_out[0]);
    try std.testing.expectEqual(@as(f32, 2.0), branch_out[0]);
}

test "ScratchBuffer: getLayerTmp returns layer_tmp buffer" {
    const allocator = std.testing.allocator;
    var scratch = try ScratchBuffer.init(allocator, 64, 256, 1);
    defer scratch.deinit();

    try scratch.ensure(4);

    const layer_tmp = scratch.getLayerTmp(100);
    try std.testing.expectEqual(@as(usize, 100), layer_tmp.len);

    // layer_tmp should be index 0, independent from other BufferIds
    layer_tmp[0] = 42.0;
    const norm_out = scratch.getTmp(.norm_out, 10);
    norm_out[0] = 99.0;

    try std.testing.expectEqual(@as(f32, 42.0), layer_tmp[0]);
}

test "ScratchBuffer: resetCaches clears all attention caches" {
    const allocator = std.testing.allocator;
    const n_layers = 3;
    var scratch = try ScratchBuffer.init(allocator, 64, 256, n_layers);
    defer scratch.deinit();

    // Simulate setting up caches (in real usage, these would be populated during forward pass)
    for (scratch.attn_caches) |*cache| {
        cache.* = .{}; // Initialize with default values
    }

    // Reset should work without error
    scratch.resetCaches();

    // After reset, caches should be in clean state
    for (scratch.attn_caches) |cache| {
        try std.testing.expectEqual(@as(usize, 0), cache.cache_position);
    }
}

test "ScratchBuffer: multiple ensure calls with decreasing size" {
    const allocator = std.testing.allocator;
    var scratch = try ScratchBuffer.init(allocator, 64, 256, 1);
    defer scratch.deinit();

    // Allocate large buffer
    try scratch.ensure(10);
    const allocated_len = scratch.tmp[0].len;

    // Ensure with smaller size should not shrink
    try scratch.ensure(2);
    try std.testing.expectEqual(allocated_len, scratch.tmp[0].len);
}

test "addIntoScaled: basic addition with scale=1.0" {
    const allocator = std.testing.allocator;

    const shape = [_]usize{4};
    var a_owned = try tensor.OwnedTensor.init(allocator, .f32, &shape);
    defer a_owned.deinit();
    var b_owned = try tensor.OwnedTensor.init(allocator, .f32, &shape);
    defer b_owned.deinit();
    var out_owned = try tensor.OwnedTensor.init(allocator, .f32, &shape);
    defer out_owned.deinit();

    const a_data = a_owned.asSlice(f32);
    const b_data = b_owned.asSlice(f32);
    a_data[0] = 1.0;
    a_data[1] = 2.0;
    a_data[2] = 3.0;
    a_data[3] = 4.0;
    b_data[0] = 5.0;
    b_data[1] = 6.0;
    b_data[2] = 7.0;
    b_data[3] = 8.0;

    const a = a_owned.toTensor();
    const b = b_owned.toTensor();
    var out = out_owned.toTensor();
    addIntoScaled(&a, &b, &out, 1.0);

    const out_data = out_owned.asSlice(f32);
    try std.testing.expectEqual(@as(f32, 6.0), out_data[0]);
    try std.testing.expectEqual(@as(f32, 8.0), out_data[1]);
    try std.testing.expectEqual(@as(f32, 10.0), out_data[2]);
    try std.testing.expectEqual(@as(f32, 12.0), out_data[3]);
}

test "addIntoScaled: addition with scale=2.0" {
    const allocator = std.testing.allocator;

    const shape = [_]usize{4};
    var a_owned = try tensor.OwnedTensor.init(allocator, .f32, &shape);
    defer a_owned.deinit();
    var b_owned = try tensor.OwnedTensor.init(allocator, .f32, &shape);
    defer b_owned.deinit();
    var out_owned = try tensor.OwnedTensor.init(allocator, .f32, &shape);
    defer out_owned.deinit();

    const a_data = a_owned.asSlice(f32);
    const b_data = b_owned.asSlice(f32);
    a_data[0] = 1.0;
    a_data[1] = 2.0;
    a_data[2] = 3.0;
    a_data[3] = 4.0;
    b_data[0] = 5.0;
    b_data[1] = 6.0;
    b_data[2] = 7.0;
    b_data[3] = 8.0;

    const a = a_owned.toTensor();
    const b = b_owned.toTensor();
    var out = out_owned.toTensor();
    addIntoScaled(&a, &b, &out, 2.0);

    const out_data = out_owned.asSlice(f32);
    try std.testing.expectEqual(@as(f32, 11.0), out_data[0]); // 1 + 5*2
    try std.testing.expectEqual(@as(f32, 14.0), out_data[1]); // 2 + 6*2
    try std.testing.expectEqual(@as(f32, 17.0), out_data[2]); // 3 + 7*2
    try std.testing.expectEqual(@as(f32, 20.0), out_data[3]); // 4 + 8*2
}

test "addIntoScaled: addition with scale=0.5" {
    const allocator = std.testing.allocator;

    const shape = [_]usize{4};
    var a_owned = try tensor.OwnedTensor.init(allocator, .f32, &shape);
    defer a_owned.deinit();
    var b_owned = try tensor.OwnedTensor.init(allocator, .f32, &shape);
    defer b_owned.deinit();
    var out_owned = try tensor.OwnedTensor.init(allocator, .f32, &shape);
    defer out_owned.deinit();

    const a_data = a_owned.asSlice(f32);
    const b_data = b_owned.asSlice(f32);
    a_data[0] = 10.0;
    a_data[1] = 20.0;
    a_data[2] = 30.0;
    a_data[3] = 40.0;
    b_data[0] = 4.0;
    b_data[1] = 8.0;
    b_data[2] = 12.0;
    b_data[3] = 16.0;

    const a = a_owned.toTensor();
    const b = b_owned.toTensor();
    var out = out_owned.toTensor();
    addIntoScaled(&a, &b, &out, 0.5);

    const out_data = out_owned.asSlice(f32);
    try std.testing.expectEqual(@as(f32, 12.0), out_data[0]); // 10 + 4*0.5
    try std.testing.expectEqual(@as(f32, 24.0), out_data[1]); // 20 + 8*0.5
    try std.testing.expectEqual(@as(f32, 36.0), out_data[2]); // 30 + 12*0.5
    try std.testing.expectEqual(@as(f32, 48.0), out_data[3]); // 40 + 16*0.5
}

test "addIntoScaled: large vector with SIMD" {
    const allocator = std.testing.allocator;

    // Use a size that's definitely larger than SIMD vector length
    const shape = [_]usize{128};
    var a_owned = try tensor.OwnedTensor.init(allocator, .f32, &shape);
    defer a_owned.deinit();
    var b_owned = try tensor.OwnedTensor.init(allocator, .f32, &shape);
    defer b_owned.deinit();
    var out_owned = try tensor.OwnedTensor.init(allocator, .f32, &shape);
    defer out_owned.deinit();

    const a_data = a_owned.asSlice(f32);
    const b_data = b_owned.asSlice(f32);

    // Fill with predictable pattern
    for (0..128) |i| {
        a_data[i] = @floatFromInt(i);
        b_data[i] = @floatFromInt(i * 2);
    }

    const a = a_owned.toTensor();
    const b = b_owned.toTensor();
    var out = out_owned.toTensor();
    addIntoScaled(&a, &b, &out, 1.5);

    const out_data = out_owned.asSlice(f32);
    for (0..128) |i| {
        const expected = @as(f32, @floatFromInt(i)) + @as(f32, @floatFromInt(i * 2)) * 1.5;
        try std.testing.expectEqual(expected, out_data[i]);
    }
}

test "addIntoScaled: non-SIMD-aligned size" {
    const allocator = std.testing.allocator;

    // Use an odd size that won't align perfectly with SIMD vectors
    const shape = [_]usize{13};
    var a_owned = try tensor.OwnedTensor.init(allocator, .f32, &shape);
    defer a_owned.deinit();
    var b_owned = try tensor.OwnedTensor.init(allocator, .f32, &shape);
    defer b_owned.deinit();
    var out_owned = try tensor.OwnedTensor.init(allocator, .f32, &shape);
    defer out_owned.deinit();

    const a_data = a_owned.asSlice(f32);
    const b_data = b_owned.asSlice(f32);

    for (0..13) |i| {
        a_data[i] = @floatFromInt(i);
        b_data[i] = @floatFromInt(i + 1);
    }

    const a = a_owned.toTensor();
    const b = b_owned.toTensor();
    var out = out_owned.toTensor();
    addIntoScaled(&a, &b, &out, 2.0);

    const out_data = out_owned.asSlice(f32);
    for (0..13) |i| {
        const expected = @as(f32, @floatFromInt(i)) + @as(f32, @floatFromInt(i + 1)) * 2.0;
        try std.testing.expectEqual(expected, out_data[i]);
    }
}

test "copyTensor: basic copy" {
    const allocator = std.testing.allocator;

    const shape = [_]usize{4};
    var src_owned = try tensor.OwnedTensor.init(allocator, .f32, &shape);
    defer src_owned.deinit();
    var dst_owned = try tensor.OwnedTensor.init(allocator, .f32, &shape);
    defer dst_owned.deinit();

    const src_data = src_owned.asSlice(f32);
    src_data[0] = 1.5;
    src_data[1] = 2.5;
    src_data[2] = 3.5;
    src_data[3] = 4.5;

    const src = src_owned.toTensor();
    var dst = dst_owned.toTensor();
    copyTensor(&src, &dst);

    const dst_data = dst_owned.asSlice(f32);
    try std.testing.expectEqual(@as(f32, 1.5), dst_data[0]);
    try std.testing.expectEqual(@as(f32, 2.5), dst_data[1]);
    try std.testing.expectEqual(@as(f32, 3.5), dst_data[2]);
    try std.testing.expectEqual(@as(f32, 4.5), dst_data[3]);
}

test "copyTensor: large tensor" {
    const allocator = std.testing.allocator;

    const shape = [_]usize{256};
    var src_owned = try tensor.OwnedTensor.init(allocator, .f32, &shape);
    defer src_owned.deinit();
    var dst_owned = try tensor.OwnedTensor.init(allocator, .f32, &shape);
    defer dst_owned.deinit();

    const src_data = src_owned.asSlice(f32);
    for (0..256) |i| {
        src_data[i] = @floatFromInt(i);
    }

    const src = src_owned.toTensor();
    var dst = dst_owned.toTensor();
    copyTensor(&src, &dst);

    const dst_data = dst_owned.asSlice(f32);
    for (0..256) |i| {
        try std.testing.expectEqual(@as(f32, @floatFromInt(i)), dst_data[i]);
    }
}

test "copyTensor: multi-dimensional tensor" {
    const allocator = std.testing.allocator;

    const shape = [_]usize{ 4, 8 };
    var src_owned = try tensor.OwnedTensor.init(allocator, .f32, &shape);
    defer src_owned.deinit();
    var dst_owned = try tensor.OwnedTensor.init(allocator, .f32, &shape);
    defer dst_owned.deinit();

    const src_data = src_owned.asSlice(f32);
    for (0..32) |i| {
        src_data[i] = @floatFromInt(i * 3);
    }

    const src = src_owned.toTensor();
    var dst = dst_owned.toTensor();
    copyTensor(&src, &dst);

    const dst_data = dst_owned.asSlice(f32);
    for (0..32) |i| {
        try std.testing.expectEqual(@as(f32, @floatFromInt(i * 3)), dst_data[i]);
    }
}

test "TransformerBlock: getDModel from SwiGLU FFN" {
    const allocator = std.testing.allocator;
    const d_model = 128;
    var w2_owned = try tensor.OwnedTensor.init(allocator, .f32, &.{ 1, 1 });
    defer w2_owned.deinit();
    var w2_tensor = w2_owned.view();
    const dk = try matmul.matmulKernel(.f32);

    var ln1_data = [_]f32{0};
    var ln2_data = [_]f32{0};
    var ln1_tensor = Tensor.view2DSlice(ln1_data[0..], 1, 1);
    var ln2_tensor = Tensor.view2DSlice(ln2_data[0..], 1, 1);

    const ln1 = try allocator.create(norm.NormKernel);
    defer allocator.destroy(ln1);
    ln1.* = .{ .rms = .{ .weight = &ln1_tensor, .dim = d_model, .eps = 1e-5, .weight_offset = 0.0 } };

    const ln2 = try allocator.create(norm.NormKernel);
    defer allocator.destroy(ln2);
    ln2.* = .{ .rms = .{ .weight = &ln2_tensor, .dim = d_model, .eps = 1e-5, .weight_offset = 0.0 } };

    const ffn_ptr = try allocator.create(FfnLayer);
    defer allocator.destroy(ffn_ptr);
    ffn_ptr.* = .{ .swiglu = .{
        .d_model = d_model,
        .d_ff = 512,
        .use_gelu = false,
        .use_swiglu_variant = false,
        .w1 = null,
        .w2 = &w2_tensor,
        .w3 = null,
        .fused_gate_up = null,
        .fused_gate_up_layout = .concat,
        .allocator = allocator,
        .matmul_gate = dk.func,
        .matmul_gate_up = null,
        .matmul_down = dk.func,
    } };

    const block = TransformerBlock{
        .kernels = &.{},
        .residual_multiplier = 1.0,
        .block_idx = 0,
        ._ln1 = ln1,
        ._ln2 = ln2,
        ._pre_ffn_norm = null,
        ._post_ffn_norm = null,
        ._attn = null,
        ._mamba = null,
        ._ffn_layer = ffn_ptr,
    };

    const ffn_layer_ptr = block.getFfnLayer() orelse unreachable;
    const result = ffn_layer_ptr.getDModel();
    try std.testing.expectEqual(@as(usize, d_model), result);
}

test "buildBlocks: creates correct number of blocks" {
    const allocator = std.testing.allocator;

    const d_model = 64;
    const d_ff = 256;
    const n_heads = 4;
    const n_kv_heads = 4;
    const head_dim = 16;
    const n_layers = 3;

    const config = ModelConfig{
        .vocab_size = 1000,
        .d_model = d_model,
        .n_layers = n_layers,
        .n_heads = n_heads,
        .n_kv_groups = n_kv_heads,
        .d_ff = d_ff,
        .max_seq_len = 512,
        .head_dim = head_dim,
        .rope_theta = 10000.0,
        .norm_eps = 1e-5,
        .gaffine_group_size = 128,
    };

    const runtime = tensor.ModelRuntime{};

    // Create dummy weights for testing
    var ln1_weights: [n_layers]tensor.OwnedTensor = undefined; // initialized in loop below
    var ln2_weights: [n_layers]tensor.OwnedTensor = undefined; // initialized in loop below
    var q_weights: [n_layers]tensor.OwnedTensor = undefined; // initialized in loop below
    var k_weights: [n_layers]tensor.OwnedTensor = undefined; // initialized in loop below
    var v_weights: [n_layers]tensor.OwnedTensor = undefined; // initialized in loop below
    var o_weights: [n_layers]tensor.OwnedTensor = undefined; // initialized in loop below
    var w1_weights: [n_layers]tensor.OwnedTensor = undefined; // initialized in loop below
    var w2_weights: [n_layers]tensor.OwnedTensor = undefined; // initialized in loop below
    var w3_weights: [n_layers]tensor.OwnedTensor = undefined; // initialized in loop below

    for (0..n_layers) |i| {
        ln1_weights[i] = try tensor.OwnedTensor.init(allocator, .f32, &[_]usize{d_model});
        ln2_weights[i] = try tensor.OwnedTensor.init(allocator, .f32, &[_]usize{d_model});
        q_weights[i] = try tensor.OwnedTensor.init(allocator, .f32, &[_]usize{ d_model, d_model });
        k_weights[i] = try tensor.OwnedTensor.init(allocator, .f32, &[_]usize{ d_model, d_model });
        v_weights[i] = try tensor.OwnedTensor.init(allocator, .f32, &[_]usize{ d_model, d_model });
        o_weights[i] = try tensor.OwnedTensor.init(allocator, .f32, &[_]usize{ d_model, d_model });
        w1_weights[i] = try tensor.OwnedTensor.init(allocator, .f32, &[_]usize{ d_ff, d_model });
        w2_weights[i] = try tensor.OwnedTensor.init(allocator, .f32, &[_]usize{ d_model, d_ff });
        w3_weights[i] = try tensor.OwnedTensor.init(allocator, .f32, &[_]usize{ d_ff, d_model });
    }
    defer for (0..n_layers) |i| {
        ln1_weights[i].deinit();
        ln2_weights[i].deinit();
        q_weights[i].deinit();
        k_weights[i].deinit();
        v_weights[i].deinit();
        o_weights[i].deinit();
        w1_weights[i].deinit();
        w2_weights[i].deinit();
        w3_weights[i].deinit();
    };

    // Tensor views
    var ln1_tensors: [n_layers]Tensor = undefined; // initialized in loop below
    var ln2_tensors: [n_layers]Tensor = undefined; // initialized in loop below
    var q_tensors: [n_layers]Tensor = undefined; // initialized in loop below
    var k_tensors: [n_layers]Tensor = undefined; // initialized in loop below
    var v_tensors: [n_layers]Tensor = undefined; // initialized in loop below
    var o_tensors: [n_layers]Tensor = undefined; // initialized in loop below
    var w1_tensors: [n_layers]Tensor = undefined; // initialized in loop below
    var w2_tensors: [n_layers]Tensor = undefined; // initialized in loop below
    var w3_tensors: [n_layers]Tensor = undefined; // initialized in loop below

    for (0..n_layers) |i| {
        ln1_tensors[i] = ln1_weights[i].toTensor();
        ln2_tensors[i] = ln2_weights[i].toTensor();
        q_tensors[i] = q_weights[i].toTensor();
        k_tensors[i] = k_weights[i].toTensor();
        v_tensors[i] = v_weights[i].toTensor();
        o_tensors[i] = o_weights[i].toTensor();
        w1_tensors[i] = w1_weights[i].toTensor();
        w2_tensors[i] = w2_weights[i].toTensor();
        w3_tensors[i] = w3_weights[i].toTensor();
    }

    var block_weights: [n_layers]BlockWeights = undefined; // initialized in loop below
    for (0..n_layers) |i| {
        block_weights[i] = .{ .attention_mlp = .{
            .ln1_weight = &ln1_tensors[i],
            .ln2_weight = &ln2_tensors[i],
            .q_proj = &q_tensors[i],
            .k_proj = &k_tensors[i],
            .v_proj = &v_tensors[i],
            .o_proj = &o_tensors[i],
            .w1 = &w1_tensors[i],
            .w2 = &w2_tensors[i],
            .w3 = &w3_tensors[i],
        } };
    }

    const blocks = try buildBlocks(allocator, config, runtime, &block_weights, progress_mod.ProgressContext.NONE);
    defer {
        for (blocks) |*block| block.deinit(allocator);
        allocator.free(blocks);
    }

    try std.testing.expectEqual(@as(usize, n_layers), blocks.len);

    // Verify each block has correct index
    for (blocks, 0..) |block, i| {
        try std.testing.expectEqual(i, block.block_idx);
    }
}

test "buildBlocks handles heterogeneous blocks" {
    const allocator = std.testing.allocator;

    const d_model: usize = 8;
    const d_ff: usize = 16;
    const n_heads: usize = 2;
    const n_kv_heads: usize = 2;
    const head_dim: usize = 4;
    const n_layers: usize = 2;

    const config = ModelConfig{
        .vocab_size = 128,
        .d_model = d_model,
        .n_layers = n_layers,
        .n_heads = n_heads,
        .n_kv_groups = n_kv_heads,
        .d_ff = d_ff,
        .max_seq_len = 32,
        .head_dim = head_dim,
        .rope_theta = 10000.0,
        .norm_eps = 1e-5,
        .gaffine_group_size = 128,
    };

    const runtime = tensor.ModelRuntime{};

    var ln1_attn = try tensor.OwnedTensor.init(allocator, .f32, &.{d_model});
    defer ln1_attn.deinit();
    var ln2_attn = try tensor.OwnedTensor.init(allocator, .f32, &.{d_model});
    defer ln2_attn.deinit();
    var q = try tensor.OwnedTensor.init(allocator, .f32, &.{ d_model, d_model });
    defer q.deinit();
    var k = try tensor.OwnedTensor.init(allocator, .f32, &.{ d_model, d_model });
    defer k.deinit();
    var v = try tensor.OwnedTensor.init(allocator, .f32, &.{ d_model, d_model });
    defer v.deinit();
    var o = try tensor.OwnedTensor.init(allocator, .f32, &.{ d_model, d_model });
    defer o.deinit();
    var w1 = try tensor.OwnedTensor.init(allocator, .f32, &.{ d_ff, d_model });
    defer w1.deinit();
    var w2 = try tensor.OwnedTensor.init(allocator, .f32, &.{ d_model, d_ff });
    defer w2.deinit();
    var w3 = try tensor.OwnedTensor.init(allocator, .f32, &.{ d_ff, d_model });
    defer w3.deinit();

    var ln1_attn_tensor = ln1_attn.view();
    var ln2_attn_tensor = ln2_attn.view();
    var q_tensor = q.view();
    var k_tensor = k.view();
    var v_tensor = v.view();
    var o_tensor = o.view();
    var w1_tensor = w1.view();
    var w2_tensor = w2.view();
    var w3_tensor = w3.view();

    const attn_weights = AttentionMlpWeights{
        .ln1_weight = &ln1_attn_tensor,
        .ln2_weight = &ln2_attn_tensor,
        .q_proj = &q_tensor,
        .k_proj = &k_tensor,
        .v_proj = &v_tensor,
        .o_proj = &o_tensor,
        .w1 = &w1_tensor,
        .w2 = &w2_tensor,
        .w3 = &w3_tensor,
    };

    const d_state: usize = 4;
    const d_conv: usize = 2;
    const n_groups: usize = 1;
    const d_inner: usize = n_heads * head_dim;
    const proj_len: usize = 2 * d_inner + 2 * n_groups * d_state + n_heads;
    const xbc_len: usize = d_inner + 2 * n_groups * d_state;

    var ln1_mamba = try tensor.OwnedTensor.init(allocator, .f32, &.{d_model});
    defer ln1_mamba.deinit();
    var ln2_mamba = try tensor.OwnedTensor.init(allocator, .f32, &.{d_model});
    defer ln2_mamba.deinit();
    var in_proj = try tensor.OwnedTensor.init(allocator, .f32, &.{ d_model, proj_len });
    defer in_proj.deinit();
    var conv1d_weight = try tensor.OwnedTensor.init(allocator, .f32, &.{ xbc_len, d_conv });
    defer conv1d_weight.deinit();
    var a_log = try tensor.OwnedTensor.init(allocator, .f32, &.{ d_inner, d_state });
    defer a_log.deinit();
    var d_skip = try tensor.OwnedTensor.init(allocator, .f32, &.{d_inner});
    defer d_skip.deinit();
    var out_proj = try tensor.OwnedTensor.init(allocator, .f32, &.{ d_inner, d_model });
    defer out_proj.deinit();
    var gate_up = try tensor.OwnedTensor.init(allocator, .f32, &.{ d_model, d_ff * 2 });
    defer gate_up.deinit();
    var down_proj = try tensor.OwnedTensor.init(allocator, .f32, &.{ d_model, d_ff });
    defer down_proj.deinit();

    var ln1_mamba_tensor = ln1_mamba.view();
    var ln2_mamba_tensor = ln2_mamba.view();
    var in_proj_tensor = in_proj.view();
    var conv1d_tensor = conv1d_weight.view();
    var a_log_tensor = a_log.view();
    var d_skip_tensor = d_skip.view();
    var out_proj_tensor = out_proj.view();
    const gate_up_tensor = gate_up.view();
    var down_proj_tensor = down_proj.view();

    const mamba_weights = MambaBlockWeights{
        .ln1_weight = &ln1_mamba_tensor,
        .ln2_weight = &ln2_mamba_tensor,
        .config = .{
            .d_model = @intCast(d_model),
            .d_state = @intCast(d_state),
            .d_conv = @intCast(d_conv),
            .n_heads = @intCast(n_heads),
            .d_head = @intCast(head_dim),
            .n_groups = @intCast(n_groups),
        },
        .weights = .{
            .in_proj = &in_proj_tensor,
            .conv1d_weight = &conv1d_tensor,
            .A_log = &a_log_tensor,
            .D = &d_skip_tensor,
            .out_proj = &out_proj_tensor,
        },
        .fused_gate_up = .{
            .gate_up = gate_up_tensor,
            .gate_up_layout = .concat,
        },
        .down_proj = &down_proj_tensor,
    };

    const block_weights = [_]BlockWeights{
        .{ .attention_mlp = attn_weights },
        .{ .mamba = mamba_weights },
    };

    const blocks = try buildBlocks(allocator, config, runtime, &block_weights, progress_mod.ProgressContext.NONE);
    defer {
        for (blocks) |*block| block.deinit(allocator);
        allocator.free(blocks);
    }

    try std.testing.expectEqual(@as(usize, 2), blocks.len);
    try std.testing.expect(blocks[0].isAttentionMlp());
    try std.testing.expect(blocks[1].isMamba());
    try std.testing.expect(blocks[0].getAttention() != null);
    try std.testing.expect(blocks[1].getMambaKernel() != null);
    try std.testing.expectEqual(@as(usize, 4), blocks[0].kernels.len);
    try std.testing.expectEqual(@as(usize, 4), blocks[1].kernels.len);
}

test "buildBlocks: validates layer count mismatch" {
    const allocator = std.testing.allocator;

    const config = ModelConfig{
        .vocab_size = 1000,
        .d_model = 64,
        .n_layers = 3,
        .n_heads = 4,
        .n_kv_groups = 4,
        .d_ff = 256,
        .max_seq_len = 512,
        .head_dim = 16,
        .rope_theta = 10000.0,
        .norm_eps = 1e-5,
        .gaffine_group_size = 128,
    };

    const runtime = tensor.ModelRuntime{};

    // Create only 2 block weights when config expects 3
    const empty_block_weights = [_]BlockWeights{};

    const result = buildBlocks(allocator, config, runtime, &empty_block_weights, progress_mod.ProgressContext.NONE);
    try std.testing.expectError(error.InvalidLayerCount, result);
}

test "TransformerBlock.deinit frees all memory" {
    const allocator = std.testing.allocator;

    const d_model = 8;
    const d_ff = 16;
    const n_heads = 2;
    const n_kv_heads = 2;
    const head_dim = 4;

    var ln1_weight = try tensor.OwnedTensor.init(allocator, .f32, &.{d_model});
    defer ln1_weight.deinit();
    var ln2_weight = try tensor.OwnedTensor.init(allocator, .f32, &.{d_model});
    defer ln2_weight.deinit();
    var q_weight = try tensor.OwnedTensor.init(allocator, .f32, &.{ d_model, d_model });
    defer q_weight.deinit();
    var k_weight = try tensor.OwnedTensor.init(allocator, .f32, &.{ d_model, d_model });
    defer k_weight.deinit();
    var v_weight = try tensor.OwnedTensor.init(allocator, .f32, &.{ d_model, d_model });
    defer v_weight.deinit();
    var o_weight = try tensor.OwnedTensor.init(allocator, .f32, &.{ d_model, d_model });
    defer o_weight.deinit();
    var w1_weight = try tensor.OwnedTensor.init(allocator, .f32, &.{ d_ff, d_model });
    defer w1_weight.deinit();
    var w2_weight = try tensor.OwnedTensor.init(allocator, .f32, &.{ d_model, d_ff });
    defer w2_weight.deinit();
    var w3_weight = try tensor.OwnedTensor.init(allocator, .f32, &.{ d_ff, d_model });
    defer w3_weight.deinit();

    var ln1_tensor = ln1_weight.toTensor();
    var ln2_tensor = ln2_weight.toTensor();
    var q_tensor = q_weight.toTensor();
    var k_tensor = k_weight.toTensor();
    var v_tensor = v_weight.toTensor();
    var o_tensor = o_weight.toTensor();
    var w1_tensor = w1_weight.toTensor();
    var w2_tensor = w2_weight.toTensor();
    var w3_tensor = w3_weight.toTensor();

    const weights = BlockWeights{ .attention_mlp = .{
        .ln1_weight = &ln1_tensor,
        .ln2_weight = &ln2_tensor,
        .q_proj = &q_tensor,
        .k_proj = &k_tensor,
        .v_proj = &v_tensor,
        .o_proj = &o_tensor,
        .w1 = &w1_tensor,
        .w2 = &w2_tensor,
        .w3 = &w3_tensor,
    } };

    var block = try TransformerBlock.init(
        allocator,
        d_model,
        d_ff,
        n_heads,
        n_kv_heads,
        head_dim,
        32,
        weights,
        1e-5,
        .{},
        1.0,
        1.0,
        false,
        0,
    );
    defer block.deinit(allocator);
}

test "init ScratchBuffer attn_caches" {
    const allocator = std.testing.allocator;
    const n_layers = 5;

    var scratch = try ScratchBuffer.init(allocator, 128, 512, n_layers);
    defer scratch.deinit();

    // All caches should be initialized
    try std.testing.expectEqual(n_layers, scratch.attn_caches.len);

    // Each cache should have default initialization
    for (scratch.attn_caches) |cache| {
        try std.testing.expectEqual(@as(usize, 0), cache.cache_position);
    }
}

test "getTmp getLayerTmp BufferId coverage" {
    const allocator = std.testing.allocator;
    var scratch = try ScratchBuffer.init(allocator, 64, 256, 1);
    defer scratch.deinit();

    try scratch.ensure(2);

    // Test various BufferId values
    _ = scratch.getTmp(.norm_out, 10);
    _ = scratch.getTmp(.branch_out, 10);
    _ = scratch.getTmp(.tmp3, 10);
    _ = scratch.getTmp(.tmp4, 10);
    _ = scratch.getTmp(.tmp5, 10);

    // All should succeed without panic
}

test "addIntoScaled: handles single-element tensors" {
    const allocator = std.testing.allocator;

    const shape = [_]usize{1};
    var a_owned = try tensor.OwnedTensor.init(allocator, .f32, &shape);
    defer a_owned.deinit();
    var b_owned = try tensor.OwnedTensor.init(allocator, .f32, &shape);
    defer b_owned.deinit();
    var out_owned = try tensor.OwnedTensor.init(allocator, .f32, &shape);
    defer out_owned.deinit();

    a_owned.asSlice(f32)[0] = 3.0;
    b_owned.asSlice(f32)[0] = 4.0;

    const a = a_owned.toTensor();
    const b = b_owned.toTensor();
    var out = out_owned.toTensor();
    addIntoScaled(&a, &b, &out, 2.0);

    try std.testing.expectEqual(@as(f32, 11.0), out_owned.asSlice(f32)[0]); // 3 + 4*2
}

test "copyTensor: handles single-element tensors" {
    const allocator = std.testing.allocator;

    const shape = [_]usize{1};
    var src_owned = try tensor.OwnedTensor.init(allocator, .f32, &shape);
    defer src_owned.deinit();
    var dst_owned = try tensor.OwnedTensor.init(allocator, .f32, &shape);
    defer dst_owned.deinit();

    src_owned.asSlice(f32)[0] = 42.5;

    const src = src_owned.toTensor();
    var dst = dst_owned.toTensor();
    copyTensor(&src, &dst);

    try std.testing.expectEqual(@as(f32, 42.5), dst_owned.asSlice(f32)[0]);
}

// =============================================================================
// Heterogeneous Model Tests (Phase 6)
// =============================================================================

test "TransformerBlock: attention_mlp block type accessors" {
    const allocator = std.testing.allocator;
    const d_model = 64;
    var w2_owned = try tensor.OwnedTensor.init(allocator, .f32, &.{ 1, 1 });
    defer w2_owned.deinit();
    var w2_tensor = w2_owned.view();
    const dk = try matmul.matmulKernel(.f32);

    var ln1_data = [_]f32{0};
    var ln2_data = [_]f32{0};
    var ln1_tensor = Tensor.view2DSlice(ln1_data[0..], 1, 1);
    var ln2_tensor = Tensor.view2DSlice(ln2_data[0..], 1, 1);

    const ln1 = try allocator.create(norm.NormKernel);
    defer allocator.destroy(ln1);
    ln1.* = .{ .rms = .{ .weight = &ln1_tensor, .dim = d_model, .eps = 1e-5, .weight_offset = 0.0 } };

    const ln2 = try allocator.create(norm.NormKernel);
    defer allocator.destroy(ln2);
    ln2.* = .{ .rms = .{ .weight = &ln2_tensor, .dim = d_model, .eps = 1e-5, .weight_offset = 0.0 } };

    var o_data = [_]f32{0};
    var o_tensor = Tensor.view2DSlice(o_data[0..], 1, 1);
    const attn_ptr = try allocator.create(attn.MultiHeadAttention);
    defer allocator.destroy(attn_ptr);
    attn_ptr.* = .{
        .d_model = d_model,
        .n_heads = 2,
        .n_kv_heads = 2,
        .head_dim = 32,
        .max_seq_len = 16,
        .scale = 1.0,
        .o_proj = &o_tensor,
        .allocator = allocator,
        .matmul_qkv = dk.func,
        .matmul_o = dk.func,
    };

    const ffn_ptr = try allocator.create(FfnLayer);
    defer allocator.destroy(ffn_ptr);
    ffn_ptr.* = .{ .swiglu = .{
        .d_model = d_model,
        .d_ff = 256,
        .use_gelu = false,
        .use_swiglu_variant = false,
        .w1 = null,
        .w2 = &w2_tensor,
        .w3 = null,
        .fused_gate_up = null,
        .fused_gate_up_layout = .concat,
        .allocator = allocator,
        .matmul_gate = dk.func,
        .matmul_gate_up = null,
        .matmul_down = dk.func,
    } };

    const block = TransformerBlock{
        .kernels = &.{},
        ._ln1 = ln1,
        ._ln2 = ln2,
        ._pre_ffn_norm = null,
        ._post_ffn_norm = null,
        ._attn = attn_ptr,
        ._mamba = null,
        ._ffn_layer = ffn_ptr,
    };

    // Test type accessors
    try std.testing.expect(block.isAttentionMlp());
    try std.testing.expect(!block.isMamba());

    // Test component accessors
    try std.testing.expect(block.getAttention() != null);
    try std.testing.expect(block.getFfnLayer() != null);
    try std.testing.expect(block.getMambaKernel() == null);
    try std.testing.expect(block.getLn1().dim() == d_model);
    try std.testing.expect(block.getLn2() != null);
}

test "TransformerBlock: mamba block type accessors" {
    const allocator = std.testing.allocator;
    const d_model = 64;
    const mamba_config = mamba.MambaConfig{
        .d_model = d_model,
        .d_state = 16,
        .d_conv = 4,
        .n_heads = 4,
        .d_head = 32,
        .n_groups = 1,
    };
    var in_proj_data = [_]f32{0};
    var out_proj_data = [_]f32{0};
    var conv1d_data = [_]f32{0};
    var A_log_data = [_]f32{0};
    var D_data = [_]f32{0};
    var ln1_data = [_]f32{0};
    const in_proj = Tensor.view2DSlice(in_proj_data[0..], 1, 1);
    const out_proj = Tensor.view2DSlice(out_proj_data[0..], 1, 1);
    const conv1d_weight = Tensor.view2DSlice(conv1d_data[0..], 1, 1);
    const A_log = Tensor.view2DSlice(A_log_data[0..], 1, 1);
    const D = Tensor.view2DSlice(D_data[0..], 1, 1);
    var ln1_weight = Tensor.view2DSlice(ln1_data[0..], 1, 1);
    const mamba_weights = mamba.MambaWeights{
        .in_proj = &in_proj,
        .out_proj = &out_proj,
        .conv1d_weight = &conv1d_weight,
        .A_log = &A_log,
        .D = &D,
    };
    const in_kernel = try registry.selectKernel(.matmul, .cpu, .f32);
    const out_kernel = try registry.selectKernel(.matmul, .cpu, .f32);
    const ssm_kernel = try registry.selectKernel(.ssm_scan, .cpu, .f32);
    const matmul_in_proj = switch (in_kernel) {
        .matmul => |fn_ptr| fn_ptr,
        else => return error.InvalidKernel,
    };
    const matmul_out_proj = switch (out_kernel) {
        .matmul => |fn_ptr| fn_ptr,
        else => return error.InvalidKernel,
    };
    const ssm_scan = switch (ssm_kernel) {
        .ssm_scan => |fn_ptr| fn_ptr,
        else => return error.InvalidKernel,
    };
    const mamba_kernel = mamba.MambaKernel.init(
        mamba_config,
        mamba_weights,
        matmul_in_proj,
        matmul_out_proj,
        ssm_scan,
    );

    const ln1 = try allocator.create(norm.NormKernel);
    defer allocator.destroy(ln1);
    ln1.* = .{ .rms = .{ .weight = &ln1_weight, .dim = d_model, .eps = 1e-5, .weight_offset = 0.0 } };

    const mamba_ptr = try allocator.create(mamba.MambaKernel);
    defer allocator.destroy(mamba_ptr);
    mamba_ptr.* = mamba_kernel;

    const block = TransformerBlock{
        .kernels = &.{},
        ._ln1 = ln1,
        ._ln2 = null,
        ._pre_ffn_norm = null,
        ._post_ffn_norm = null,
        ._attn = null,
        ._mamba = mamba_ptr,
        ._ffn_layer = null,
    };

    // Test type accessors
    try std.testing.expect(block.isMamba());
    try std.testing.expect(!block.isAttentionMlp());

    // Test component accessors
    try std.testing.expect(block.getAttention() == null);
    try std.testing.expect(block.getFfnLayer() == null);
    try std.testing.expect(block.getMambaKernel() != null);
    try std.testing.expect(block.getLn1().dim() == d_model);
    try std.testing.expect(block.getLn2() == null); // Mamba block without ln2
}

test "ScratchBuffer: initMamba allocates state for mamba layers" {
    const allocator = std.testing.allocator;
    var scratch = try ScratchBuffer.init(allocator, 64, 256, 4);
    defer scratch.deinit();

    // Before initMamba, no mamba state
    try std.testing.expect(scratch.mamba_states == null);
    try std.testing.expect(scratch.mamba_scratch == null);
    try std.testing.expect(scratch.getMambaState(0) == null);

    // Initialize Mamba state for 2 Mamba layers
    const mamba_config = mamba.MambaConfig{
        .d_model = 64,
        .d_state = 16,
        .d_conv = 4,
        .n_heads = 4,
        .d_head = 32,
        .n_groups = 1,
    };
    try scratch.initMamba(2, mamba_config);

    // After initMamba, mamba state should be available
    try std.testing.expect(scratch.mamba_states != null);
    try std.testing.expect(scratch.mamba_scratch != null);
    try std.testing.expect(scratch.mamba_states.?.len == 2);

    // Can get state for valid indices
    try std.testing.expect(scratch.getMambaState(0) != null);
    try std.testing.expect(scratch.getMambaState(1) != null);
    try std.testing.expect(scratch.getMambaState(2) == null); // Out of bounds

    // Scratch buffer available
    try std.testing.expect(scratch.getMambaScratch() != null);
}

test "ScratchBuffer: resetCaches resets both attention and mamba state" {
    const allocator = std.testing.allocator;
    var scratch = try ScratchBuffer.init(allocator, 64, 256, 2);
    defer scratch.deinit();

    // Initialize Mamba state
    const mamba_config = mamba.MambaConfig{
        .d_model = 64,
        .d_state = 16,
        .d_conv = 4,
        .n_heads = 4,
        .d_head = 32,
        .n_groups = 1,
    };
    try scratch.initMamba(1, mamba_config);

    // Modify state to non-zero values
    if (scratch.getMambaState(0)) |state| {
        for (state.conv_state) |*v| v.* = 1.0;
        for (state.ssm_state) |*v| v.* = 1.0;
    }

    // Reset caches
    scratch.resetCaches();

    // Mamba state should be zeroed
    if (scratch.getMambaState(0)) |state| {
        for (state.conv_state) |v| {
            try std.testing.expectEqual(@as(f32, 0.0), v);
        }
        for (state.ssm_state) |v| {
            try std.testing.expectEqual(@as(f32, 0.0), v);
        }
    }
}

test "blockWeightsFromMap maps attention weights" {
    const allocator = std.testing.allocator;

    var map = WeightMap{};
    defer map.deinit(allocator);

    var ln1_data = [_]f32{ 1.0, 2.0 };
    var ln2_data = [_]f32{ 3.0, 4.0 };
    var q_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var k_data = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    var v_data = [_]f32{ 9.0, 10.0, 11.0, 12.0 };
    var o_data = [_]f32{ 13.0, 14.0, 15.0, 16.0 };
    var w1_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var w2_data = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    var w3_data = [_]f32{ 9.0, 10.0, 11.0, 12.0 };

    var ln1 = Tensor.view2DSlice(ln1_data[0..], 1, ln1_data.len);
    ln1.n_dims = 1;
    ln1.shape[0] = @intCast(ln1_data.len);
    var ln2 = Tensor.view2DSlice(ln2_data[0..], 1, ln2_data.len);
    ln2.n_dims = 1;
    ln2.shape[0] = @intCast(ln2_data.len);

    const q = Tensor.view2DSlice(q_data[0..], 2, 2);
    const k = Tensor.view2DSlice(k_data[0..], 2, 2);
    const v = Tensor.view2DSlice(v_data[0..], 2, 2);
    const o = Tensor.view2DSlice(o_data[0..], 2, 2);
    const w1 = Tensor.view2DSlice(w1_data[0..], 2, 2);
    const w2 = Tensor.view2DSlice(w2_data[0..], 2, 2);
    const w3 = Tensor.view2DSlice(w3_data[0..], 2, 2);

    try map.put(allocator, "input_layernorm.weight", &ln1);
    try map.put(allocator, "post_attention_layernorm.weight", &ln2);
    try map.put(allocator, "self_attn.q_proj.weight", &q);
    try map.put(allocator, "self_attn.k_proj.weight", &k);
    try map.put(allocator, "self_attn.v_proj.weight", &v);
    try map.put(allocator, "self_attn.o_proj.weight", &o);
    try map.put(allocator, "mlp.gate_proj.weight", &w1);
    try map.put(allocator, "mlp.up_proj.weight", &w3);
    try map.put(allocator, "mlp.down_proj.weight", &w2);

    const ctx = BlockMapContext{ .rope = null, .sliding_window = 0 };
    const weights = try blockWeightsFromMap(&map, .attention_mlp, ctx);
    switch (weights) {
        .attention_mlp => |w| {
            try std.testing.expect(w.q_proj != null);
            try std.testing.expect(w.k_proj != null);
            try std.testing.expect(w.v_proj != null);
            try std.testing.expect(w.fused.qkv_proj == null);
            try std.testing.expect(w.w1 != null);
            try std.testing.expect(w.w2 != null);
            try std.testing.expect(w.w3 != null);
        },
        else => try std.testing.expect(false),
    }
}

test "blockWeightsFromMap maps attention weights from mixer alias" {
    const allocator = std.testing.allocator;

    var map = WeightMap{};
    defer map.deinit(allocator);

    var ln1_data = [_]f32{ 1.0, 2.0 };
    var ln2_data = [_]f32{ 3.0, 4.0 };
    var q_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var k_data = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    var v_data = [_]f32{ 9.0, 10.0, 11.0, 12.0 };
    var o_data = [_]f32{ 13.0, 14.0, 15.0, 16.0 };
    var w1_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var w2_data = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    var w3_data = [_]f32{ 9.0, 10.0, 11.0, 12.0 };

    var ln1 = Tensor.view2DSlice(ln1_data[0..], 1, ln1_data.len);
    ln1.n_dims = 1;
    ln1.shape[0] = @intCast(ln1_data.len);
    var ln2 = Tensor.view2DSlice(ln2_data[0..], 1, ln2_data.len);
    ln2.n_dims = 1;
    ln2.shape[0] = @intCast(ln2_data.len);

    const q = Tensor.view2DSlice(q_data[0..], 2, 2);
    const k = Tensor.view2DSlice(k_data[0..], 2, 2);
    const v = Tensor.view2DSlice(v_data[0..], 2, 2);
    const o = Tensor.view2DSlice(o_data[0..], 2, 2);
    const w1 = Tensor.view2DSlice(w1_data[0..], 2, 2);
    const w2 = Tensor.view2DSlice(w2_data[0..], 2, 2);
    const w3 = Tensor.view2DSlice(w3_data[0..], 2, 2);

    try map.put(allocator, "input_layernorm.weight", &ln1);
    try map.put(allocator, "post_attention_layernorm.weight", &ln2);
    try map.put(allocator, "mixer.q_proj.weight", &q);
    try map.put(allocator, "mixer.k_proj.weight", &k);
    try map.put(allocator, "mixer.v_proj.weight", &v);
    try map.put(allocator, "mixer.o_proj.weight", &o);
    try map.put(allocator, "mlp.gate_proj.weight", &w1);
    try map.put(allocator, "mlp.up_proj.weight", &w3);
    try map.put(allocator, "mlp.down_proj.weight", &w2);

    const ctx = BlockMapContext{ .rope = null, .sliding_window = 0 };
    const weights = try blockWeightsFromMap(&map, .attention_mlp, ctx);
    switch (weights) {
        .attention_mlp => |w| {
            try std.testing.expect(w.q_proj != null);
            try std.testing.expect(w.k_proj != null);
            try std.testing.expect(w.v_proj != null);
            try std.testing.expect(w.fused.qkv_proj == null);
            try std.testing.expect(w.w1 != null);
            try std.testing.expect(w.w2 != null);
            try std.testing.expect(w.w3 != null);
        },
        else => try std.testing.expect(false),
    }
}

test "TransformerBlock.fromMap builds attention block" {
    const allocator = std.testing.allocator;

    var map = WeightMap{};
    defer map.deinit(allocator);

    var ln1_data = [_]f32{ 1.0, 2.0 };
    var ln2_data = [_]f32{ 3.0, 4.0 };
    var q_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var k_data = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    var v_data = [_]f32{ 9.0, 10.0, 11.0, 12.0 };
    var o_data = [_]f32{ 13.0, 14.0, 15.0, 16.0 };
    var w1_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var w2_data = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    var w3_data = [_]f32{ 9.0, 10.0, 11.0, 12.0 };

    var ln1 = Tensor.view2DSlice(ln1_data[0..], 1, ln1_data.len);
    ln1.n_dims = 1;
    ln1.shape[0] = @intCast(ln1_data.len);
    var ln2 = Tensor.view2DSlice(ln2_data[0..], 1, ln2_data.len);
    ln2.n_dims = 1;
    ln2.shape[0] = @intCast(ln2_data.len);

    const q = Tensor.view2DSlice(q_data[0..], 2, 2);
    const k = Tensor.view2DSlice(k_data[0..], 2, 2);
    const v = Tensor.view2DSlice(v_data[0..], 2, 2);
    const o = Tensor.view2DSlice(o_data[0..], 2, 2);
    const w1 = Tensor.view2DSlice(w1_data[0..], 2, 2);
    const w2 = Tensor.view2DSlice(w2_data[0..], 2, 2);
    const w3 = Tensor.view2DSlice(w3_data[0..], 2, 2);

    try map.put(allocator, "input_layernorm.weight", &ln1);
    try map.put(allocator, "post_attention_layernorm.weight", &ln2);
    try map.put(allocator, "self_attn.q_proj.weight", &q);
    try map.put(allocator, "self_attn.k_proj.weight", &k);
    try map.put(allocator, "self_attn.v_proj.weight", &v);
    try map.put(allocator, "self_attn.o_proj.weight", &o);
    try map.put(allocator, "mlp.gate_proj.weight", &w1);
    try map.put(allocator, "mlp.up_proj.weight", &w3);
    try map.put(allocator, "mlp.down_proj.weight", &w2);

    const context = BlockInitContext{
        .allocator = allocator,
        .d_model = 2,
        .d_ff = 2,
        .n_heads = 1,
        .n_kv_heads = 1,
        .head_dim = 2,
        .max_seq_len = 4,
        .norm_eps = 1e-5,
        .runtime = .{},
        .residual_multiplier = 1.0,
        .attention_scale = 1.0,
        .use_gelu = false,
        .block_idx = 0,
        .map_context = .{ .rope = null, .sliding_window = 0 },
    };

    var block = try TransformerBlock.fromMap(context, .attention_mlp, &map);
    defer block.deinit(allocator);
    try std.testing.expect(block.isAttentionMlp());
}
