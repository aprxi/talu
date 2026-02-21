//! Models-owned runtime block weight contracts.
//!
//! This module defines the loader/runtime data contract used by `models/load`.
//! It intentionally stays data-only and does not depend on inference executors.

const std = @import("std");
const tensor = @import("../../tensor.zig");
const graph_types = @import("../op_types.zig");

const Tensor = tensor.Tensor;

pub const GateUpLayout = enum {
    concat,
    interleaved,
};

pub const FusedBlockWeights = struct {
    qkv_proj: ?Tensor = null,
    gate_up: ?Tensor = null,
    gate_up_layout: GateUpLayout = .concat,
};

pub const ExpertWeights = struct {
    gate_proj: ?Tensor = null,
    gate_scales: ?[]const u8 = null,
    gate_bias: ?[]const f32 = null,
    up_proj: ?Tensor = null,
    up_scales: ?[]const u8 = null,
    up_bias: ?[]const f32 = null,
    gate_up_proj: ?Tensor = null,
    gate_up_scales: ?[]const u8 = null,
    gate_up_bias: ?[]const f32 = null,
    down_proj: Tensor,
    down_scales: ?[]const u8 = null,
    down_bias: ?[]const f32 = null,
};

pub const MoEWeights = struct {
    router_weight: Tensor,
    router_bias: ?[]const f32 = null,
    experts: []ExpertWeights,
    num_experts: usize,
    experts_per_token: usize,
    use_mxfp4: bool = false,
};

pub const MLAConfig = struct {
    q_lora_rank: usize,
    kv_lora_rank: usize,
    qk_head_dim: usize,
    qk_rope_head_dim: usize,
    qk_nope_head_dim: usize,
    v_head_dim: usize,
    rope_interleave: bool,
};

pub const MambaConfig = struct {
    d_model: u32,
    d_state: u32,
    d_conv: u32,
    n_heads: u32,
    d_head: u32,
    n_groups: u32 = 1,
};

pub const MambaWeights = struct {
    in_proj: *const Tensor,
    conv1d_weight: *const Tensor,
    conv1d_bias: ?*const Tensor = null,
    A_log: *const Tensor,
    D: *const Tensor,
    dt_bias: ?*const Tensor = null,
    norm_weight: ?*const Tensor = null,
    out_proj: *const Tensor,
};

pub const ShortConvConfig = struct {
    d_model: u32,
    d_conv: u32,
    conv_dim: u32,
    conv_dim_out: u32,
    has_bias: bool = false,
};

pub const ShortConvWeights = struct {
    in_proj: *const Tensor,
    conv1d_weight: *const Tensor,
    conv1d_bias: ?*const Tensor = null,
    out_proj: *const Tensor,
};

pub const AttentionMlpWeights = struct {
    ln1_weight: *const Tensor,
    ln2_weight: *const Tensor,
    ln1_bias: ?*const Tensor = null,
    ln2_bias: ?*const Tensor = null,
    q_proj: ?*const Tensor = null,
    k_proj: ?*const Tensor = null,
    v_proj: ?*const Tensor = null,
    o_proj: *const Tensor,
    w1: ?*const Tensor = null,
    w2: ?*const Tensor = null,
    w3: ?*const Tensor = null,
    w1_bias: ?*const Tensor = null,
    w2_bias: ?*const Tensor = null,
    // Opaque runtime pointer owned by loader/runtime init.
    rope: ?*anyopaque = null,
    sliding_window: usize = 0,
    fused: FusedBlockWeights = .{},
    q_norm: ?*const Tensor = null,
    k_norm: ?*const Tensor = null,
    pre_ffn_norm: ?*const Tensor = null,
    post_ffn_norm: ?*const Tensor = null,
    q_bias: ?[]const f32 = null,
    k_bias: ?[]const f32 = null,
    v_bias: ?[]const f32 = null,
    o_bias: ?[]const f32 = null,
    moe_weights: ?*MoEWeights = null,
    sinks: ?[]const f32 = null,
    is_causal: bool = true,
    block_ops: []const graph_types.Op = &.{},
    mla_config: ?MLAConfig = null,
    q_a_proj: ?*const Tensor = null,
    q_a_norm: ?*const Tensor = null,
    q_b_proj: ?*const Tensor = null,
    kv_a_proj: ?*const Tensor = null,
    kv_a_norm: ?*const Tensor = null,
    kv_b_proj: ?*const Tensor = null,

    pub fn isMLA(self: *const AttentionMlpWeights) bool {
        return self.mla_config != null;
    }
};

pub const MambaBlockWeights = struct {
    ln1_weight: *const Tensor,
    ln2_weight: ?*const Tensor = null,
    config: MambaConfig,
    weights: MambaWeights,
    fused_gate_up: ?FusedBlockWeights = null,
    down_proj: ?*const Tensor = null,
};

pub const ShortConvBlockWeights = struct {
    ln1_weight: *const Tensor,
    ln2_weight: ?*const Tensor = null,
    config: ShortConvConfig,
    weights: ShortConvWeights,
    fused_gate_up: ?FusedBlockWeights = null,
    w1: ?*const Tensor = null,
    w2: ?*const Tensor = null,
    w3: ?*const Tensor = null,
};

pub const BlockKind = graph_types.BlockKind;
pub const BlockWeights = union(BlockKind) {
    attention_mlp: AttentionMlpWeights,
    mamba: MambaBlockWeights,
    shortconv: ShortConvBlockWeights,
};

pub const WeightMap = std.StringHashMapUnmanaged(*const Tensor);

pub const BlockMapContext = struct {
    rope: ?*anyopaque = null,
    sliding_window: usize = 0,
    is_causal: bool = true,
    block_ops: []const graph_types.Op = &.{},
    mamba_config: ?MambaConfig = null,
    shortconv_config: ?ShortConvConfig = null,
    mla_config: ?MLAConfig = null,
    num_experts: usize = 0,
    experts_per_token: usize = 0,
    allocator: ?std.mem.Allocator = null,
};

fn getRequiredWeight(map: *const WeightMap, name: []const u8) !*const Tensor {
    if (map.get(name)) |weight| return weight;
    return error.MissingWeight;
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

fn buildMxfp4MoEWeights(
    allocator: std.mem.Allocator,
    map: *const WeightMap,
    router_weight_ptr: *const Tensor,
    gate_up_blocks: *const Tensor,
    num_experts: usize,
    experts_per_token: usize,
) !*MoEWeights {
    const gate_up_scales = map.get("mlp.experts.gate_up_proj_scales") orelse return error.MissingWeight;
    const down_blocks = map.get("mlp.experts.down_proj_blocks") orelse return error.MissingWeight;
    const down_scales = map.get("mlp.experts.down_proj_scales") orelse return error.MissingWeight;

    const gate_up_bias_tensor = map.get("mlp.experts.gate_up_proj_bias");
    const down_bias_tensor = map.get("mlp.experts.down_proj_bias");
    const gate_up_bias_values: ?[]const f32 = if (gate_up_bias_tensor) |t| t.asSlice(f32) else null;
    const down_bias_values: ?[]const f32 = if (down_bias_tensor) |t| t.asSlice(f32) else null;

    const router_bias_tensor = map.get("mlp.router.bias");
    const router_bias: ?[]const f32 = if (router_bias_tensor) |t| t.asSlice(f32) else null;

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

    const gate_up_out_dim: usize = if (gate_up_blocks.n_dims >= 2) @intCast(gate_up_blocks.shape[1]) else return error.InvalidShape;
    const down_out_dim: usize = if (down_blocks.n_dims >= 2) @intCast(down_blocks.shape[1]) else return error.InvalidShape;

    const gate_up_bytes_per_row: usize = gate_up_expert_bytes / gate_up_out_dim;
    const down_bytes_per_row: usize = down_expert_bytes / down_out_dim;

    var experts = try allocator.alloc(ExpertWeights, num_experts);
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

fn buildIndexedMoEWeights(
    allocator: std.mem.Allocator,
    map: *const WeightMap,
    router_weight_ptr: *const Tensor,
    num_experts: usize,
    experts_per_token: usize,
) !*MoEWeights {
    var experts = try allocator.alloc(ExpertWeights, num_experts);
    errdefer allocator.free(experts);

    var name_buf: [128]u8 = undefined;
    for (0..num_experts) |e| {
        const gate_name = std.fmt.bufPrint(&name_buf, "mlp.experts.{d}.gate_proj.weight", .{e}) catch return error.BufferOverflow;
        const gate_proj = map.get(gate_name) orelse return error.MissingWeight;

        const up_name = std.fmt.bufPrint(&name_buf, "mlp.experts.{d}.up_proj.weight", .{e}) catch return error.BufferOverflow;
        const up_proj = map.get(up_name) orelse return error.MissingWeight;

        const down_name = std.fmt.bufPrint(&name_buf, "mlp.experts.{d}.down_proj.weight", .{e}) catch return error.BufferOverflow;
        const down_proj = map.get(down_name) orelse return error.MissingWeight;

        experts[e] = .{
            .gate_proj = gate_proj.*,
            .up_proj = up_proj.*,
            .down_proj = down_proj.*,
        };
    }

    const moe_weights = try allocator.create(MoEWeights);
    moe_weights.* = .{
        .router_weight = router_weight_ptr.*,
        .router_bias = null,
        .experts = experts,
        .num_experts = num_experts,
        .experts_per_token = experts_per_token,
        .use_mxfp4 = false,
    };
    return moe_weights;
}

fn buildMoEWeightsFromMap(
    allocator: std.mem.Allocator,
    map: *const WeightMap,
    num_experts: usize,
    experts_per_token: usize,
) !?*MoEWeights {
    const router_weight_ptr = map.get("mlp.gate.weight") orelse
        map.get("mlp.router.weight") orelse
        return null;

    if (num_experts == 0) return null;

    if (map.get("mlp.experts.gate_up_proj_blocks")) |gate_up_blocks| {
        return try buildMxfp4MoEWeights(allocator, map, router_weight_ptr, gate_up_blocks, num_experts, experts_per_token);
    }
    return try buildIndexedMoEWeights(allocator, map, router_weight_ptr, num_experts, experts_per_token);
}

pub fn blockWeightsFromMap(
    map: *const WeightMap,
    block_type: BlockKind,
    context: BlockMapContext,
) !BlockWeights {
    switch (block_type) {
        .attention_mlp => {
            const ln1_weight = getOptionalWeight(map, "input_layernorm.weight") orelse
                try getRequiredWeight(map, "operator_norm.weight");
            const ln2_weight = getOptionalWeight(map, "post_attention_layernorm.weight") orelse
                try getRequiredWeight(map, "ffn_norm.weight");
            const ln1_bias = getOptionalWeight(map, "input_layernorm.bias");
            const ln2_bias = getOptionalWeight(map, "post_attention_layernorm.bias");

            const o_proj = getOptionalWeightAlias(map, "self_attn.o_proj.weight", "mixer.o_proj.weight") orelse
                try getRequiredWeight(map, "self_attn.out_proj.weight");

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
                q_a_norm = try getRequiredWeight(map, "self_attn.q_a_layernorm.weight");
                q_b_proj = try getRequiredWeight(map, "self_attn.q_b_proj.weight");
                kv_a_proj = try getRequiredWeight(map, "self_attn.kv_a_proj_with_mqa.weight");
                kv_a_norm = try getRequiredWeight(map, "self_attn.kv_a_layernorm.weight");
                kv_b_proj = try getRequiredWeight(map, "self_attn.kv_b_proj.weight");
            } else {
                fused_qkv = getOptionalWeightAlias(map, "self_attn.qkv_proj.weight", "mixer.qkv_proj.weight");
                q_proj = if (fused_qkv == null) try getRequiredWeightAlias(map, "self_attn.q_proj.weight", "mixer.q_proj.weight") else null;
                k_proj = if (fused_qkv == null) try getRequiredWeightAlias(map, "self_attn.k_proj.weight", "mixer.k_proj.weight") else null;
                v_proj = if (fused_qkv == null) try getRequiredWeightAlias(map, "self_attn.v_proj.weight", "mixer.v_proj.weight") else null;
            }

            const pre_ffn_norm = getOptionalWeight(map, "pre_feedforward_layernorm.weight");
            const post_ffn_norm = getOptionalWeight(map, "post_feedforward_layernorm.weight");
            const q_norm = getOptionalWeightAlias(map, "self_attn.q_norm.weight", "mixer.q_norm.weight") orelse
                getOptionalWeight(map, "self_attn.q_layernorm.weight");
            const k_norm = getOptionalWeightAlias(map, "self_attn.k_norm.weight", "mixer.k_norm.weight") orelse
                getOptionalWeight(map, "self_attn.k_layernorm.weight");

            const q_bias = try getBiasSliceAlias(map, "self_attn.q_proj.bias", "mixer.q_proj.bias");
            const k_bias = try getBiasSliceAlias(map, "self_attn.k_proj.bias", "mixer.k_proj.bias");
            const v_bias = try getBiasSliceAlias(map, "self_attn.v_proj.bias", "mixer.v_proj.bias");
            const o_bias = try getBiasSliceAlias(map, "self_attn.o_proj.bias", "mixer.o_proj.bias");
            const sinks = try getBiasSlice(map, "self_attn.sinks");

            var fused = FusedBlockWeights{};
            if (fused_qkv) |fq| fused.qkv_proj = fq.*;

            const mla_config = if (is_mla) context.mla_config else null;
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
                    w1 = getOptionalWeight(map, "mlp.gate_proj.weight") orelse
                        getOptionalWeight(map, "feed_forward.w1.weight") orelse
                        getOptionalWeight(map, "mlp.dense_in.weight");
                    w3 = getOptionalWeight(map, "mlp.up_proj.weight") orelse
                        getOptionalWeight(map, "feed_forward.w3.weight");
                    if (w1 == null) return error.MissingWeight;
                }
                w2 = getOptionalWeight(map, "mlp.down_proj.weight") orelse
                    getOptionalWeight(map, "mlp.output_linear.weight") orelse
                    getOptionalWeight(map, "feed_forward.w2.weight") orelse
                    getOptionalWeight(map, "mlp.dense_out.weight") orelse
                    return error.MissingWeight;
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
            const shortconv_config = context.shortconv_config orelse return error.MissingShortConvConfig;
            const ln1_weight = try getRequiredWeight(map, "operator_norm.weight");
            const ln2_weight = getOptionalWeight(map, "ffn_norm.weight");
            const in_proj = try getRequiredWeight(map, "conv.in_proj.weight");
            const out_proj = try getRequiredWeight(map, "conv.out_proj.weight");
            const conv1d_weight = try getRequiredWeight(map, "conv.conv.weight");
            const conv1d_bias: ?*const Tensor = null;

            const fused_gate_up = getOptionalWeight(map, "feed_forward.gate_up_proj.weight") orelse
                getOptionalWeight(map, "mlp.gate_up_proj.weight");

            var fused_gate_up_weights: ?FusedBlockWeights = null;
            var w1: ?*const Tensor = null;
            var w2: ?*const Tensor = null;
            var w3: ?*const Tensor = null;
            if (fused_gate_up) |fg| {
                fused_gate_up_weights = .{ .gate_up = fg.*, .gate_up_layout = .concat };
            } else {
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
