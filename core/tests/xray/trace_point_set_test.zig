//! Integration tests for xray.TracePointSet
//!
//! TracePointSet is a packed struct bitset for selecting which trace points
//! to capture. Provides all(), none(), and contains() methods.

const std = @import("std");
const main = @import("main");
const xray = main.xray;

const TracePointSet = xray.TracePointSet;
const TracePoint = xray.TracePoint;

test "TracePointSet: all() contains all points" {
    const all = TracePointSet.all();

    try std.testing.expect(all.contains(.embed));
    try std.testing.expect(all.contains(.embed_pos));
    try std.testing.expect(all.contains(.layer_input));
    try std.testing.expect(all.contains(.layer_attn_out));
    try std.testing.expect(all.contains(.layer_ffn_down));
    try std.testing.expect(all.contains(.logits));
    try std.testing.expect(all.contains(.logits_scaled));
}

test "TracePointSet: none() contains no points" {
    const none = TracePointSet.none();

    try std.testing.expect(!none.contains(.embed));
    try std.testing.expect(!none.contains(.logits));
    try std.testing.expect(!none.contains(.layer_attn_out));
    try std.testing.expect(!none.contains(.layer_ffn_down));
}

test "TracePointSet: individual field access" {
    var points = TracePointSet.none();
    points.embed = true;
    points.logits = true;

    try std.testing.expect(points.contains(.embed));
    try std.testing.expect(points.contains(.logits));
    try std.testing.expect(!points.contains(.layer_attn_out));
    try std.testing.expect(!points.contains(.layer_input));
}

test "TracePointSet: contains checks specific point" {
    var points = TracePointSet.none();
    points.layer_attn_out = true;
    points.layer_ffn_down = true;

    try std.testing.expect(points.contains(.layer_attn_out));
    try std.testing.expect(points.contains(.layer_ffn_down));
    try std.testing.expect(!points.contains(.embed));
    try std.testing.expect(!points.contains(.logits));
}

test "TracePointSet: is packed struct" {
    // Verify it's a packed struct (important for memory layout)
    const info = @typeInfo(TracePointSet);
    try std.testing.expect(info == .@"struct");
    try std.testing.expect(info.@"struct".layout == .@"packed");
}
