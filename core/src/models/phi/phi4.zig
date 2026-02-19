//! Phi4 model-version metadata.

const layer_ops = @import("../layer_ops.zig");

/// Graph architecture id is "phi" for the phi family.
pub const id: []const u8 = "phi";
pub const family: []const u8 = "phi";
pub const version: []const u8 = "phi4";
pub const model_types: []const []const u8 = &.{
    "phi3",
    "phi4",
    "phi",
};

/// Static block topology for Phi decoder blocks.
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
