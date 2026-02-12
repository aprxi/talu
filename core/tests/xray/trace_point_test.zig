//! Integration tests for xray.TracePoint
//!
//! TracePoint is an enum representing locations in the inference pipeline
//! where tensor values can be captured for inspection.

const std = @import("std");
const main = @import("main");
const xray = main.xray;

const TracePoint = xray.TracePoint;

test "TracePoint: all trace points are enumerable" {
    const points = [_]TracePoint{
        .embed,
        .embed_pos,
        .layer_input,
        .layer_attn_norm,
        .layer_q,
        .layer_k,
        .layer_v,
        .layer_qk,
        .layer_attn_weights,
        .layer_attn_out,
        .layer_ffn_norm,
        .layer_ffn_gate,
        .layer_ffn_up,
        .layer_ffn_act,
        .layer_ffn_down,
        .layer_residual,
        .final_norm,
        .logits,
        .logits_scaled,
    };

    for (points) |point| {
        const value = @intFromEnum(point);
        const reconstructed: TracePoint = @enumFromInt(value);
        try std.testing.expectEqual(point, reconstructed);
    }
}

test "TracePoint: can be used as map key" {
    var counts = std.AutoHashMap(TracePoint, usize).init(std.testing.allocator);
    defer counts.deinit();

    try counts.put(.embed, 10);
    try counts.put(.logits, 20);

    try std.testing.expectEqual(@as(usize, 10), counts.get(.embed).?);
    try std.testing.expectEqual(@as(usize, 20), counts.get(.logits).?);
}

test "TracePoint: name returns string representation" {
    try std.testing.expectEqualStrings("embed", TracePoint.embed.name());
    try std.testing.expectEqualStrings("logits", TracePoint.logits.name());
    try std.testing.expectEqualStrings("layer_attn_out", TracePoint.layer_attn_out.name());
}
