//! Gemma3 model-version metadata.

const layer_ops = @import("../layer_ops.zig");

pub const id: []const u8 = "gemma3";
pub const family: []const u8 = "gemma";
pub const version: []const u8 = "gemma3";
pub const model_types: []const []const u8 = &.{
    "gemma3",
    "gemma3_text",
    "gemma2",
    "gemma",
};

/// Static block topology for Gemma3 decoder blocks.
/// Gemma3 graph includes extra norm steps on attention and MLP branches:
/// norm -> attn -> norm -> add -> norm -> mlp -> norm -> add.
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
    .{ .kernel = .{
        .id = 2,
        .in = .branch_out,
        .out = .norm_out,
        .debug_type = .norm,
    } },
    .{ .add = .{
        .branch = .norm_out,
        .scale = .residual_multiplier,
    } },
    .{ .kernel = .{
        .id = 3,
        .in = .residual,
        .out = .norm_out,
        .debug_type = .norm,
    } },
    .{ .kernel = .{
        .id = 4,
        .in = .norm_out,
        .out = .branch_out,
        .debug_type = .mlp,
    } },
    .{ .kernel = .{
        .id = 5,
        .in = .branch_out,
        .out = .norm_out,
        .debug_type = .norm,
    } },
    .{ .add = .{
        .branch = .norm_out,
        .scale = .residual_multiplier,
    } },
};
