//! Llama3 family model-version metadata.

const layer_ops = @import("../layer_ops.zig");

pub const id: []const u8 = "llama3";
pub const family: []const u8 = "llama";
pub const version: []const u8 = "llama3";
pub const model_types: []const []const u8 = &.{
    "llama",
    "llama3",
    "llama3.1",
    "llama3.2",
    "idefics3",
};

/// Static block topology for Llama3-style decoder blocks.
/// Sequence: norm -> attention -> add -> norm -> ffn -> add.
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
