//! Qwen3-MoE model-version metadata.

const layer_ops = @import("../layer_ops.zig");

pub const id: []const u8 = "qwen3_moe";
pub const family: []const u8 = "qwen";
pub const version: []const u8 = "qwen3_moe";
pub const model_types: []const []const u8 = &.{
    "qwen3_moe",
};

/// Static block topology for Qwen3-MoE decoder blocks.
/// Sequence: norm -> attention -> add -> norm -> moe -> add.
pub const attention_mlp_program: []const layer_ops.LayerOp = &.{
    .{ .kernel = .{
        .id = 0,
        .in = .residual,
        .out = .norm_out,
        .debug_type = .norm,
    } },
    .{ .kernel = .{
        .id = 1,
        .in = .norm_out,
        .out = .branch_out,
        .debug_type = .multihead_attention,
    } },
    .{ .add = .{
        .branch = .branch_out,
        .scale = .residual_multiplier,
    } },
    .{ .kernel = .{
        .id = 2,
        .in = .residual,
        .out = .norm_out,
        .debug_type = .norm,
    } },
    .{ .kernel = .{
        .id = 3,
        .in = .norm_out,
        .out = .branch_out,
        .debug_type = .moe,
    } },
    .{ .add = .{
        .branch = .branch_out,
        .scale = .residual_multiplier,
    } },
};
