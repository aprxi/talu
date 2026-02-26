//! Vision LayerOp program parsing helpers.
//!
//! Keeps backend vision runtime initialization aligned with model-declared
//! vision pipeline metadata.

const std = @import("std");
const layer_ops = @import("../models/layer_ops.zig");

pub const ParsedVisionProgram = struct {
    spatial_merge_size: usize,
    deepstack_visual_layers: [8]usize,
    deepstack_layer_count: usize,
};

/// Parse model-declared vision pipeline ops and resolve runtime parameters.
/// The program must declare at least one `.patch_embed` and `.spatial_merge` op.
pub fn parseVisionProgram(
    program: []const layer_ops.LayerOp,
    default_spatial_merge_size: usize,
    default_deepstack_visual_layers: [8]usize,
    default_deepstack_layer_count: usize,
) !ParsedVisionProgram {
    if (default_spatial_merge_size == 0) return error.InvalidSpatialMergeSize;
    if (default_deepstack_layer_count > default_deepstack_visual_layers.len) {
        return error.InvalidDeepstackConfig;
    }

    var has_patch_embed = false;
    var has_spatial_merge = false;
    var spatial_merge_size = default_spatial_merge_size;

    var deepstack_visual_layers: [8]usize = [_]usize{0} ** 8;
    var deepstack_layer_count: usize = 0;

    for (program) |op| {
        switch (op) {
            .patch_embed => has_patch_embed = true,
            .spatial_merge => |merge_op| {
                has_spatial_merge = true;
                const merge_size = std.math.cast(usize, merge_op.merge_size) orelse return error.InvalidSpatialMergeSize;
                if (merge_size == 0) return error.InvalidSpatialMergeSize;
                spatial_merge_size = merge_size;
            },
            .deepstack_extract => |extract_op| {
                if (deepstack_layer_count >= deepstack_visual_layers.len) return error.TooManyDeepstackLayers;
                const layer_idx = std.math.cast(usize, extract_op.layer_index) orelse return error.InvalidDeepstackLayer;
                deepstack_visual_layers[deepstack_layer_count] = layer_idx;
                deepstack_layer_count += 1;
            },
            else => {},
        }
    }

    if (!has_patch_embed) return error.MissingPatchEmbedOp;
    if (!has_spatial_merge) return error.MissingSpatialMergeOp;

    if (deepstack_layer_count == 0) {
        deepstack_layer_count = default_deepstack_layer_count;
        if (deepstack_layer_count > deepstack_visual_layers.len) {
            return error.InvalidDeepstackConfig;
        }
        for (0..deepstack_layer_count) |idx| {
            deepstack_visual_layers[idx] = default_deepstack_visual_layers[idx];
        }
    }

    return .{
        .spatial_merge_size = spatial_merge_size,
        .deepstack_visual_layers = deepstack_visual_layers,
        .deepstack_layer_count = deepstack_layer_count,
    };
}

test "parseVisionProgram enforces required patch and merge ops" {
    const program = [_]layer_ops.LayerOp{
        .{ .deepstack_extract = .{ .in = .norm_out, .out = .tmp3, .layer_index = 1 } },
    };
    try std.testing.expectError(
        error.MissingPatchEmbedOp,
        parseVisionProgram(&program, 2, [_]usize{0} ** 8, 0),
    );
}

test "parseVisionProgram applies merge and deepstack overrides from program" {
    const program = [_]layer_ops.LayerOp{
        .{ .patch_embed = .{ .in = .residual, .out = .norm_out } },
        .{ .deepstack_extract = .{ .in = .norm_out, .out = .tmp3, .layer_index = 4 } },
        .{ .deepstack_extract = .{ .in = .norm_out, .out = .tmp4, .layer_index = 9 } },
        .{ .spatial_merge = .{ .in = .norm_out, .out = .branch_out, .merge_size = 3 } },
    };
    const parsed = try parseVisionProgram(&program, 2, [_]usize{0} ** 8, 0);
    try std.testing.expectEqual(@as(usize, 3), parsed.spatial_merge_size);
    try std.testing.expectEqual(@as(usize, 2), parsed.deepstack_layer_count);
    try std.testing.expectEqual(@as(usize, 4), parsed.deepstack_visual_layers[0]);
    try std.testing.expectEqual(@as(usize, 9), parsed.deepstack_visual_layers[1]);
}
