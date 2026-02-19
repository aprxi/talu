//! Granite3 model-version metadata.

const layer_ops = @import("../layer_ops.zig");

pub const id: []const u8 = "granite3";
pub const family: []const u8 = "granite";
pub const version: []const u8 = "granite3";
pub const model_types: []const []const u8 = &.{
    "granite",
};

/// Static block topology for Granite3 decoder blocks.
/// Graph uses explicit mul before add; executor-level add scale handles this via
/// residual multiplier.
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
        .debug_type = .mlp,
    } },
    .{ .add = .{
        .branch = .branch_out,
        .scale = .residual_multiplier,
    } },
};
