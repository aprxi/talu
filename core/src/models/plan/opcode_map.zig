//! Single source of truth: mapping between model op metadata and runtime opcodes.

const std = @import("std");
const layer_ops = @import("../layer_ops.zig");
const op_types = @import("../op_types.zig");
const opcode_mod = @import("opcode.zig");

pub const Opcode = opcode_mod.Opcode;

/// In v1 all OpType variants are executable.
pub fn isExecutableOpType(op: op_types.OpType) bool {
    _ = op;
    return true;
}

pub fn opcodeForOpType(op: op_types.OpType) Opcode {
    return switch (op) {
        .norm => .rmsnorm,
        .multihead_attention => .multihead_attention,
        .mlp => .swiglu,
        .moe => .moe,
        .mamba_mixer => .mamba_mixer,
        .shortconv => .shortconv,
        .add => .residual_add,
        .mul => .mul,
        .mean => .mean,
        .pow => .pow,
        .rsqrt => .rsqrt,
        .matmul => .matmul,
        .split => .split,
        .transpose => .transpose,
        .reshape => .reshape,
        .softmax => .softmax,
        .silu => .silu,
        .gelu => .gelu,
        .embedding => .embedding,
        .linear => .linear,
        .rope => .rope,
        .triu => .triu,
        .scaled_dot_product_attention => .scaled_dot_product_attention,
        .patch_embed => .vision_patch_embed,
        .spatial_merge => .vision_spatial_merge,
        .deepstack_extract => .vision_deepstack_extract,
        .scatter => .vision_scatter,
    };
}

pub fn opTypeForOpcode(opcode: Opcode) ?op_types.OpType {
    return switch (opcode) {
        .rmsnorm => .norm,
        .multihead_attention => .multihead_attention,
        .swiglu => .mlp,
        .moe => .moe,
        .mamba_mixer => .mamba_mixer,
        .shortconv => .shortconv,
        .residual_add => .add,
        .mul => .mul,
        .mean => .mean,
        .pow => .pow,
        .rsqrt => .rsqrt,
        .matmul => .matmul,
        .split => .split,
        .transpose => .transpose,
        .reshape => .reshape,
        .softmax => .softmax,
        .silu => .silu,
        .gelu => .gelu,
        .embedding => .embedding,
        .linear => .linear,
        .rope => .rope,
        .triu => .triu,
        .scaled_dot_product_attention => .scaled_dot_product_attention,
        .vision_patch_embed => .patch_embed,
        .vision_spatial_merge => .spatial_merge,
        .vision_deepstack_extract => .deepstack_extract,
        .vision_scatter => .scatter,
        else => null,
    };
}

pub fn opcodeForLayerOp(op: layer_ops.LayerOp) Opcode {
    return switch (op) {
        .kernel => |kernel_op| opcodeForOpType(kernel_op.debug_type),
        .add => .residual_add,
        .linear => .linear,
        .matmul => .matmul,
        .split => .split,
        .softmax => .softmax,
        .silu => .silu,
        .gelu => .gelu,
        .mul => .mul,
        .add_tensor => .add_tensor,
        .add_scalar => .add_scalar,
        .mul_scalar => .mul_scalar,
        .mean => .mean,
        .pow => .pow,
        .rsqrt => .rsqrt,
        .add_param => .add_param,
        .add_param_scalar => .add_param_scalar,
        .mul_param => .mul_param,
        .reshape => .reshape,
        .transpose => .transpose,
        .rope => .rope,
        .triu => .triu,
        .sdpa => .scaled_dot_product_attention,
        .patch_embed => .vision_patch_embed,
        .spatial_merge => .vision_spatial_merge,
        .deepstack_extract => .vision_deepstack_extract,
        .scatter => .vision_scatter,
    };
}

test "every executable OpType maps to exactly one opcode and round-trips" {
    inline for (std.meta.tags(op_types.OpType)) |tag| {
        if (!isExecutableOpType(tag)) continue;
        const opcode = opcodeForOpType(tag);
        const roundtrip = opTypeForOpcode(opcode) orelse return error.TestUnexpectedResult;
        try std.testing.expectEqual(tag, roundtrip);
    }
}

test "opcodeForLayerOp maps kernel debug type and residual add correctly" {
    const kernel_op: layer_ops.LayerOp = .{ .kernel = .{
        .id = 0,
        .in = .residual,
        .out = .norm_out,
        .debug_type = .norm,
    } };
    try std.testing.expectEqual(Opcode.rmsnorm, opcodeForLayerOp(kernel_op));

    const add_op: layer_ops.LayerOp = .{ .add = .{
        .branch = .branch_out,
        .scale = .one,
    } };
    try std.testing.expectEqual(Opcode.residual_add, opcodeForLayerOp(add_op));
}

test "opcodeForLayerOp maps vision operations" {
    const patch: layer_ops.LayerOp = .{ .patch_embed = .{
        .in = .residual,
        .out = .norm_out,
    } };
    try std.testing.expectEqual(Opcode.vision_patch_embed, opcodeForLayerOp(patch));

    const scatter: layer_ops.LayerOp = .{ .scatter = .{
        .text_in = .residual,
        .vision_in = .norm_out,
        .out = .branch_out,
        .image_token_id = 151655,
    } };
    try std.testing.expectEqual(Opcode.vision_scatter, opcodeForLayerOp(scatter));
}
