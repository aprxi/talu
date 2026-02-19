//! MiniLM (BERT-family) model-version metadata.

const layer_ops = @import("../layer_ops.zig");

pub const id: []const u8 = "minilm";
pub const family: []const u8 = "bert";
pub const version: []const u8 = "minilm";
pub const model_types: []const []const u8 = &.{
    "bert",
    "minilm",
};

/// Static block topology for MiniLM encoder blocks.
/// Sequence: attention -> add -> norm -> mlp -> add -> norm.
pub const attention_mlp_program: []const layer_ops.LayerOp = &.{
    .{ .kernel = .{
        .id = 0,
        .in = .residual,
        .out = .branch_out,
        .debug_type = .multihead_attention,
    } },
    .{ .add = .{
        .branch = .branch_out,
        .scale = .residual_multiplier,
    } },
    .{ .kernel = .{
        .id = 1,
        .in = .residual,
        .out = .norm_out,
        .debug_type = .norm,
    } },
    .{ .kernel = .{
        .id = 2,
        .in = .norm_out,
        .out = .branch_out,
        .debug_type = .mlp,
    } },
    .{ .add = .{
        .branch = .branch_out,
        .scale = .residual_multiplier,
    } },
    .{ .kernel = .{
        .id = 3,
        .in = .residual,
        .out = .norm_out,
        .debug_type = .norm,
    } },
};
