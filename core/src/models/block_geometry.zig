//! Models-owned block geometry inference.
//!
//! These helpers derive architecture facts from model config and loaded weight
//! shapes. Backends consume the resolved facts; they do not own the model
//! policy that decides them.

const std = @import("std");
const tensor = @import("compute_pkg").tensor;
const config_types = @import("config/types.zig");
const runtime_blocks = @import("runtime_blocks.zig");

const Tensor = tensor.Tensor;
const ModelConfig = config_types.ModelConfig;
const AttentionMlpWeights = runtime_blocks.AttentionMlpWeights;

pub const AttentionShape = struct {
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
};

pub const InstructionAttentionShape = struct {
    head_dim_u32: u32,
    rope_dim_u32: u32,
    n_heads_u32: u32,
    n_kv_heads_u32: u32,
};

pub const InstructionAttentionShapeInput = struct {
    q_dim: usize,
    kv_dim: usize,
    sliding_window: usize,
    default_head_dim: u32,
    default_rope_dim: u32,
    default_n_heads: u32,
    default_n_kv_heads: u32,
    global_head_dim: i32,
    q_norm_rows: ?usize = null,
    k_norm_rows: ?usize = null,
};

fn tensorVectorDim(t: *const Tensor) ?usize {
    if (t.n_dims == 1) {
        return @intCast(t.shape[0]);
    }
    if (t.n_dims == 2) {
        if (t.shape[0] == 1 and t.shape[1] > 0) return @intCast(t.shape[1]);
        if (t.shape[1] == 1 and t.shape[0] > 0) return @intCast(t.shape[0]);
    }
    return null;
}

fn projectedOutDim(weight: *const Tensor, d_model: usize) ?usize {
    if (weight.n_dims != 2) return null;
    const a: usize = @intCast(weight.shape[0]);
    const b: usize = @intCast(weight.shape[1]);
    if (a == d_model and b > 0) return b;
    if (b == d_model and a > 0) return a;
    return null;
}

pub fn inferAttentionShape(config: ModelConfig, weights: AttentionMlpWeights) AttentionShape {
    const d_model: usize = @intCast(config.d_model);
    var shape = AttentionShape{
        .n_heads = @intCast(config.n_heads),
        .n_kv_heads = @intCast(config.n_kv_groups),
        .head_dim = @intCast(config.head_dim),
    };

    if (weights.q_norm) |q_norm| {
        if (tensorVectorDim(q_norm)) |dim| shape.head_dim = dim;
    } else if (weights.k_norm) |k_norm| {
        if (tensorVectorDim(k_norm)) |dim| shape.head_dim = dim;
    }

    const q_out_opt = projectedOutDim(weights.o_proj, d_model);
    if (q_out_opt) |q_out| {
        if (shape.head_dim > 0 and (q_out % shape.head_dim) == 0) {
            shape.n_heads = q_out / shape.head_dim;
        }
    }

    if (weights.k_proj) |k_proj| {
        if (projectedOutDim(k_proj, d_model)) |kv_out| {
            if (shape.head_dim > 0 and (kv_out % shape.head_dim) == 0) {
                shape.n_kv_heads = kv_out / shape.head_dim;
            }
        }
    } else if (weights.fused.qkv_proj) |qkv_proj| {
        if (q_out_opt) |q_out| {
            if (projectedOutDim(&qkv_proj, d_model)) |qkv_out| {
                const effective_q_out = if (weights.attention_config.query_gate) q_out * 2 else q_out;
                if (qkv_out >= effective_q_out) {
                    const kv_pair_out = qkv_out - effective_q_out;
                    if ((kv_pair_out % 2) == 0) {
                        const kv_out = kv_pair_out / 2;
                        if (shape.head_dim > 0 and (kv_out % shape.head_dim) == 0) {
                            shape.n_kv_heads = kv_out / shape.head_dim;
                        }
                    }
                }
            }
        }
    }

    if (shape.n_heads == 0) shape.n_heads = 1;
    if (shape.n_kv_heads == 0) shape.n_kv_heads = 1;
    if (shape.head_dim == 0) shape.head_dim = 1;
    return shape;
}

pub fn inferAttentionDff(config: ModelConfig, weights: AttentionMlpWeights) usize {
    const d_model: usize = @intCast(config.d_model);
    const default_d_ff: usize = @intCast(config.d_ff);

    if (weights.fused.gate_up) |gate_up| {
        if (projectedOutDim(&gate_up, d_model)) |gate_up_out| {
            if ((gate_up_out % 2) == 0 and gate_up_out > 0) return gate_up_out / 2;
        }
    }
    if (weights.w1) |w1| {
        if (projectedOutDim(w1, d_model)) |w1_out| {
            if (w1_out > 0) return w1_out;
        }
    }
    if (weights.w2) |w2| {
        if (projectedOutDim(w2, d_model)) |w2_out| {
            if (w2_out > 0) return w2_out;
        }
    }
    return default_d_ff;
}

pub fn resolveAttentionScale(config: ModelConfig, head_dim: usize) f32 {
    if (config.attention_multiplier > 0) return config.attention_multiplier;
    if (config.query_pre_attn_scalar > 0) return 1.0 / @sqrt(config.query_pre_attn_scalar);
    return 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
}

pub fn resolveRuntimeAttentionScale(config: ModelConfig, base_attention_scale: f32, head_dim: usize) f32 {
    if (config.attention_multiplier > 0 or config.query_pre_attn_scalar > 0) return base_attention_scale;
    if (head_dim == 0) return base_attention_scale;
    return 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
}

pub fn resolveAttentionScaleOverride(config: ModelConfig, default_head_dim: usize) f32 {
    if (config.attention_multiplier <= 0 and config.query_pre_attn_scalar <= 0) return 0.0;
    return resolveAttentionScale(config, default_head_dim);
}

pub fn resolveSharedKvSourceLayer(config: ModelConfig, layer_idx: usize) ?usize {
    if (config.num_kv_shared_layers <= 0) return null;
    const layer_types = config.layer_types orelse return null;
    const n_layers: usize = @intCast(config.n_layers);
    if (layer_types.len != n_layers) return null;
    if (layer_idx >= n_layers) return null;

    const shared_count: usize = @min(@as(usize, @intCast(config.num_kv_shared_layers)), n_layers);
    if (shared_count == 0 or shared_count == n_layers) return null;
    const first_shared_layer = n_layers - shared_count;
    if (layer_idx < first_shared_layer or first_shared_layer == 0) return null;

    const target_layer_type = layer_types[layer_idx];
    var src = first_shared_layer;
    while (src > 0) {
        src -= 1;
        if (layer_types[src] == target_layer_type) return src;
    }
    return null;
}

pub fn resolveMinSharedKvSourceLayer(config: ModelConfig) ?usize {
    const n_layers: usize = @intCast(config.n_layers);
    var min_source: ?usize = null;
    var layer_idx: usize = 0;
    while (layer_idx < n_layers) : (layer_idx += 1) {
        if (resolveSharedKvSourceLayer(config, layer_idx)) |source| {
            min_source = if (min_source) |current| @min(current, source) else source;
        }
    }
    return min_source;
}

pub fn resolveInstructionAttentionShape(input: InstructionAttentionShapeInput) !InstructionAttentionShape {
    var head_dim_u32 = input.default_head_dim;
    if (input.q_norm_rows) |rows| {
        head_dim_u32 = @intCast(rows);
    } else if (input.k_norm_rows) |rows| {
        head_dim_u32 = @intCast(rows);
    }
    if (head_dim_u32 == 0) return error.InvalidAttentionGeometry;
    const head_dim_usize: usize = @intCast(head_dim_u32);

    var n_heads_u32 = input.default_n_heads;
    if ((input.q_dim % head_dim_usize) == 0) {
        n_heads_u32 = @intCast(input.q_dim / head_dim_usize);
    } else if (input.q_norm_rows != null) {
        return error.InvalidAttentionGeometry;
    }

    var n_kv_heads_u32 = input.default_n_kv_heads;
    if ((input.kv_dim % head_dim_usize) == 0) {
        n_kv_heads_u32 = @intCast(input.kv_dim / head_dim_usize);
    } else if (input.k_norm_rows != null) {
        return error.InvalidAttentionGeometry;
    }
    if (n_heads_u32 == 0 or n_kv_heads_u32 == 0 or (n_heads_u32 % n_kv_heads_u32) != 0) {
        return error.InvalidAttentionGeometry;
    }

    var rope_dim_u32 = @min(input.default_rope_dim, head_dim_u32);
    if (input.sliding_window > 0 and input.global_head_dim > 0) {
        rope_dim_u32 = head_dim_u32;
    }

    return .{
        .head_dim_u32 = head_dim_u32,
        .rope_dim_u32 = rope_dim_u32,
        .n_heads_u32 = n_heads_u32,
        .n_kv_heads_u32 = n_kv_heads_u32,
    };
}

fn vectorTensor(comptime len: usize, data: *[len]f32) Tensor {
    var result = Tensor.view2DSlice(data[0..], 1, len);
    result.n_dims = 1;
    result.shape[0] = @intCast(len);
    return result;
}

pub const testing = struct {
    pub fn runContractTests() !void {
        try expectInferAttentionShapeDerivesPerLayerHeadDimensions();
        try expectInferAttentionShapeDerivesHeadCounts();
        try expectInferAttentionShapeDerivesFusedQkvWithoutQueryGate();
        try expectInferAttentionShapeDerivesFusedGatedQkv();
        try expectInferAttentionDffDerivesFromFusedGateUp();
        try expectInferAttentionDffDerivesFromLayerWeights();
        try expectResolveAttentionScaleHonorsOverridePrecedence();
        try expectResolveRuntimeAttentionScalePreservesOverride();
        try expectResolveRuntimeAttentionScaleDerivesPerLayerScale();
        try expectResolveAttentionScaleOverrideReturnsZeroWithoutOverride();
        try expectResolveSharedKvSourceLayerSelectsPreviousMatchingLayerType();
        try expectResolveMinSharedKvSourceLayerSelectsLowestSource();
        try expectResolveInstructionAttentionShapeDerivesNormalizedShape();
        try expectResolveInstructionAttentionShapeRejectsInvalidShape();
    }
};

fn expectInferAttentionShapeDerivesPerLayerHeadDimensions() !void {
    var ln_data = [_]f32{1} ** 6;
    var o_proj_data = [_]f32{1} ** 36;
    var k_proj_data = [_]f32{1} ** 12;
    var q_norm_data = [_]f32{1} ** 3;
    var k_norm_data = [_]f32{1} ** 3;

    var ln = vectorTensor(6, &ln_data);
    const o_proj = Tensor.view2DSlice(o_proj_data[0..], 6, 6);
    const k_proj = Tensor.view2DSlice(k_proj_data[0..], 6, 2);
    var q_norm = vectorTensor(3, &q_norm_data);
    var k_norm = vectorTensor(3, &k_norm_data);

    var config = std.mem.zeroes(ModelConfig);
    config.d_model = 6;
    config.n_heads = 2;
    config.n_kv_groups = 1;
    config.head_dim = 2;

    const weights = AttentionMlpWeights{
        .ln1_weight = &ln,
        .ln2_weight = &ln,
        .k_proj = &k_proj,
        .o_proj = &o_proj,
        .q_norm = &q_norm,
        .k_norm = &k_norm,
    };
    const inferred = inferAttentionShape(config, weights);
    try std.testing.expectEqual(@as(usize, 2), inferred.n_heads);
    try std.testing.expectEqual(@as(usize, 1), inferred.n_kv_heads);
    try std.testing.expectEqual(@as(usize, 3), inferred.head_dim);
}

test "inferAttentionShape derives per-layer head dimensions from attention weights" {
    try expectInferAttentionShapeDerivesPerLayerHeadDimensions();
}

fn expectInferAttentionShapeDerivesHeadCounts() !void {
    var ln_data = [_]f32{1} ** 6;
    var o_proj_data = [_]f32{1} ** 48;
    var k_proj_data = [_]f32{1} ** 24;

    var ln = vectorTensor(6, &ln_data);
    const o_proj = Tensor.view2DSlice(o_proj_data[0..], 6, 8);
    const k_proj = Tensor.view2DSlice(k_proj_data[0..], 6, 4);

    var config = std.mem.zeroes(ModelConfig);
    config.d_model = 6;
    config.n_heads = 1;
    config.n_kv_groups = 1;
    config.head_dim = 2;

    const weights = AttentionMlpWeights{
        .ln1_weight = &ln,
        .ln2_weight = &ln,
        .k_proj = &k_proj,
        .o_proj = &o_proj,
    };
    const inferred = inferAttentionShape(config, weights);
    try std.testing.expectEqual(@as(usize, 4), inferred.n_heads);
    try std.testing.expectEqual(@as(usize, 2), inferred.n_kv_heads);
    try std.testing.expectEqual(@as(usize, 2), inferred.head_dim);
}

test "inferAttentionShape derives head counts from projection widths" {
    try expectInferAttentionShapeDerivesHeadCounts();
}

fn expectInferAttentionShapeDerivesFusedQkvWithoutQueryGate() !void {
    var ln_data = [_]f32{1} ** 4;
    var o_proj_data = [_]f32{1} ** 16;
    var qkv_proj_data = [_]f32{1} ** 48;

    var ln = vectorTensor(4, &ln_data);
    const o_proj = Tensor.view2DSlice(o_proj_data[0..], 4, 4);
    const qkv_proj = Tensor.view2DSlice(qkv_proj_data[0..], 4, 12);

    var config = std.mem.zeroes(ModelConfig);
    config.d_model = 4;
    config.n_heads = 2;
    config.n_kv_groups = 2;
    config.head_dim = 2;

    const weights = AttentionMlpWeights{
        .ln1_weight = &ln,
        .ln2_weight = &ln,
        .o_proj = &o_proj,
        .fused = .{ .qkv_proj = qkv_proj },
    };
    const inferred = inferAttentionShape(config, weights);
    try std.testing.expectEqual(@as(usize, 2), inferred.n_heads);
    try std.testing.expectEqual(@as(usize, 2), inferred.n_kv_heads);
    try std.testing.expectEqual(@as(usize, 2), inferred.head_dim);
}

test "inferAttentionShape derives kv heads from fused qkv without query gate" {
    try expectInferAttentionShapeDerivesFusedQkvWithoutQueryGate();
}

fn expectInferAttentionShapeDerivesFusedGatedQkv() !void {
    var ln_data = [_]f32{1} ** 4;
    var o_proj_data = [_]f32{1} ** 16;
    var qkv_proj_data = [_]f32{1} ** 48;

    var ln = vectorTensor(4, &ln_data);
    const o_proj = Tensor.view2DSlice(o_proj_data[0..], 4, 4);
    const qkv_proj = Tensor.view2DSlice(qkv_proj_data[0..], 4, 12);

    var config = std.mem.zeroes(ModelConfig);
    config.d_model = 4;
    config.n_heads = 2;
    config.n_kv_groups = 1;
    config.head_dim = 2;

    const weights = AttentionMlpWeights{
        .ln1_weight = &ln,
        .ln2_weight = &ln,
        .o_proj = &o_proj,
        .fused = .{ .qkv_proj = qkv_proj },
        .attention_config = .{ .query_gate = true },
    };
    const inferred = inferAttentionShape(config, weights);
    try std.testing.expectEqual(@as(usize, 2), inferred.n_heads);
    try std.testing.expectEqual(@as(usize, 1), inferred.n_kv_heads);
    try std.testing.expectEqual(@as(usize, 2), inferred.head_dim);
}

test "inferAttentionShape derives kv heads from fused gated qkv" {
    try expectInferAttentionShapeDerivesFusedGatedQkv();
}

fn expectInferAttentionDffDerivesFromFusedGateUp() !void {
    var ln_data = [_]f32{1} ** 6;
    var o_proj_data = [_]f32{1} ** 24;
    var gate_up_data = [_]f32{1} ** 144;

    var ln = vectorTensor(6, &ln_data);
    const o_proj = Tensor.view2DSlice(o_proj_data[0..], 6, 4);
    const gate_up = Tensor.view2DSlice(gate_up_data[0..], 6, 24);

    var config = std.mem.zeroes(ModelConfig);
    config.d_model = 6;
    config.d_ff = 4;
    config.n_heads = 2;
    config.n_kv_groups = 1;
    config.head_dim = 2;

    const weights = AttentionMlpWeights{
        .ln1_weight = &ln,
        .ln2_weight = &ln,
        .o_proj = &o_proj,
        .fused = .{ .gate_up = gate_up },
    };
    const inferred = inferAttentionDff(config, weights);
    try std.testing.expectEqual(@as(usize, 12), inferred);
}

test "inferAttentionDff derives per-layer FFN width from fused gate_up" {
    try expectInferAttentionDffDerivesFromFusedGateUp();
}

fn expectInferAttentionDffDerivesFromLayerWeights() !void {
    var ln_data = [_]f32{1} ** 6;
    var o_proj_data = [_]f32{1} ** 24;
    var w1_data = [_]f32{1} ** 72;
    var w2_data = [_]f32{1} ** 72;

    var ln = vectorTensor(6, &ln_data);
    const o_proj = Tensor.view2DSlice(o_proj_data[0..], 6, 4);
    const w1 = Tensor.view2DSlice(w1_data[0..], 12, 6);
    const w2 = Tensor.view2DSlice(w2_data[0..], 6, 12);

    var config = std.mem.zeroes(ModelConfig);
    config.d_model = 6;
    config.d_ff = 4;
    config.n_heads = 2;
    config.n_kv_groups = 1;
    config.head_dim = 2;

    const weights = AttentionMlpWeights{
        .ln1_weight = &ln,
        .ln2_weight = &ln,
        .o_proj = &o_proj,
        .w1 = &w1,
        .w2 = &w2,
    };
    const inferred = inferAttentionDff(config, weights);
    try std.testing.expectEqual(@as(usize, 12), inferred);
}

test "inferAttentionDff derives per-layer FFN width from layer weights" {
    try expectInferAttentionDffDerivesFromLayerWeights();
}

fn expectResolveAttentionScaleHonorsOverridePrecedence() !void {
    var config = std.mem.zeroes(ModelConfig);
    config.head_dim = 64;

    try std.testing.expectApproxEqAbs(@as(f32, 0.125), resolveAttentionScale(config, 64), 0.000001);

    config.query_pre_attn_scalar = 256.0;
    try std.testing.expectApproxEqAbs(@as(f32, 0.0625), resolveAttentionScale(config, 64), 0.000001);

    config.attention_multiplier = 0.5;
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), resolveAttentionScale(config, 64), 0.000001);
}

test "resolveAttentionScale honors config override precedence" {
    try expectResolveAttentionScaleHonorsOverridePrecedence();
}

fn expectResolveRuntimeAttentionScalePreservesOverride() !void {
    var config = std.mem.zeroes(ModelConfig);
    const base_attention_scale: f32 = 0.375;

    config.query_pre_attn_scalar = 256.0;
    try std.testing.expectApproxEqAbs(base_attention_scale, resolveRuntimeAttentionScale(config, base_attention_scale, 32), 0.000001);

    config.query_pre_attn_scalar = 0.0;
    config.attention_multiplier = 0.5;
    try std.testing.expectApproxEqAbs(base_attention_scale, resolveRuntimeAttentionScale(config, base_attention_scale, 32), 0.000001);
}

test "resolveRuntimeAttentionScale preserves explicit override scale" {
    try expectResolveRuntimeAttentionScalePreservesOverride();
}

fn expectResolveRuntimeAttentionScaleDerivesPerLayerScale() !void {
    const config = std.mem.zeroes(ModelConfig);
    const base_attention_scale: f32 = 0.25;

    try std.testing.expectApproxEqAbs(@as(f32, 0.125), resolveRuntimeAttentionScale(config, base_attention_scale, 64), 0.000001);
    try std.testing.expectApproxEqAbs(base_attention_scale, resolveRuntimeAttentionScale(config, base_attention_scale, 0), 0.000001);
}

test "resolveRuntimeAttentionScale derives per-layer scale without override" {
    try expectResolveRuntimeAttentionScaleDerivesPerLayerScale();
}

fn expectResolveAttentionScaleOverrideReturnsZeroWithoutOverride() !void {
    var config = std.mem.zeroes(ModelConfig);
    config.head_dim = 64;

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), resolveAttentionScaleOverride(config, 64), 0.000001);

    config.query_pre_attn_scalar = 256.0;
    try std.testing.expectApproxEqAbs(@as(f32, 0.0625), resolveAttentionScaleOverride(config, 64), 0.000001);

    config.attention_multiplier = 0.5;
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), resolveAttentionScaleOverride(config, 64), 0.000001);
}

test "resolveAttentionScaleOverride returns zero without explicit override" {
    try expectResolveAttentionScaleOverrideReturnsZeroWithoutOverride();
}

fn expectResolveSharedKvSourceLayerSelectsPreviousMatchingLayerType() !void {
    const layer_types = [_]u8{ 1, 2, 1, 2 };
    var config = std.mem.zeroes(ModelConfig);
    config.n_layers = @intCast(layer_types.len);
    config.num_kv_shared_layers = 2;
    config.layer_types = &layer_types;

    try std.testing.expectEqual(@as(?usize, 0), resolveSharedKvSourceLayer(config, 2));
    try std.testing.expectEqual(@as(?usize, 1), resolveSharedKvSourceLayer(config, 3));
    try std.testing.expectEqual(@as(?usize, null), resolveSharedKvSourceLayer(config, 1));
}

test "resolveSharedKvSourceLayer selects previous matching layer type" {
    try expectResolveSharedKvSourceLayerSelectsPreviousMatchingLayerType();
}

fn expectResolveMinSharedKvSourceLayerSelectsLowestSource() !void {
    const layer_types = [_]u8{ 1, 2, 1, 2 };
    var config = std.mem.zeroes(ModelConfig);
    config.n_layers = @intCast(layer_types.len);
    config.num_kv_shared_layers = 2;
    config.layer_types = &layer_types;

    try std.testing.expectEqual(@as(?usize, 0), resolveMinSharedKvSourceLayer(config));

    const unmatched_layer_types = [_]u8{ 1, 2, 3, 4 };
    config.layer_types = &unmatched_layer_types;
    try std.testing.expectEqual(@as(?usize, null), resolveMinSharedKvSourceLayer(config));
}

test "resolveMinSharedKvSourceLayer selects lowest shared source layer" {
    try expectResolveMinSharedKvSourceLayerSelectsLowestSource();
}

fn expectResolveInstructionAttentionShapeDerivesNormalizedShape() !void {
    const shape = try resolveInstructionAttentionShape(.{
        .q_dim = 8,
        .kv_dim = 4,
        .sliding_window = 128,
        .default_head_dim = 2,
        .default_rope_dim = 1,
        .default_n_heads = 1,
        .default_n_kv_heads = 1,
        .global_head_dim = 4,
        .q_norm_rows = 4,
        .k_norm_rows = null,
    });
    try std.testing.expectEqual(@as(u32, 4), shape.head_dim_u32);
    try std.testing.expectEqual(@as(u32, 4), shape.rope_dim_u32);
    try std.testing.expectEqual(@as(u32, 2), shape.n_heads_u32);
    try std.testing.expectEqual(@as(u32, 1), shape.n_kv_heads_u32);
}

test "resolveInstructionAttentionShape derives normalized instruction shape" {
    try expectResolveInstructionAttentionShapeDerivesNormalizedShape();
}

fn expectResolveInstructionAttentionShapeRejectsInvalidShape() !void {
    try std.testing.expectError(error.InvalidAttentionGeometry, resolveInstructionAttentionShape(.{
        .q_dim = 10,
        .kv_dim = 4,
        .sliding_window = 0,
        .default_head_dim = 4,
        .default_rope_dim = 4,
        .default_n_heads = 1,
        .default_n_kv_heads = 1,
        .global_head_dim = 0,
        .q_norm_rows = 4,
        .k_norm_rows = null,
    }));

    try std.testing.expectError(error.InvalidAttentionGeometry, resolveInstructionAttentionShape(.{
        .q_dim = 6,
        .kv_dim = 4,
        .sliding_window = 0,
        .default_head_dim = 2,
        .default_rope_dim = 2,
        .default_n_heads = 3,
        .default_n_kv_heads = 2,
        .global_head_dim = 0,
        .q_norm_rows = null,
        .k_norm_rows = null,
    }));
}

test "resolveInstructionAttentionShape rejects invalid normalized instruction shape" {
    try expectResolveInstructionAttentionShapeRejectsInvalidShape();
}
