//! Integration tests for xray.AttentionType
//!
//! AttentionType selects the attention implementation (MHA, GQA, MQA).
//! Provides name() for display.

const std = @import("std");
const main = @import("main");
const xray = main.xray;

const AttentionType = xray.AttentionType;
const analyze = xray.execution_plan.analyze;

// ===== name =====

test "AttentionType: name returns unique string for each variant" {
    const names = [_][:0]const u8{
        AttentionType.multi_head.name(),
        AttentionType.grouped_query.name(),
        AttentionType.multi_query.name(),
    };

    for (names, 0..) |a, i| {
        try std.testing.expect(a.len > 0);
        for (names[i + 1 ..]) |b| {
            try std.testing.expect(!std.mem.eql(u8, a, b));
        }
    }
}

test "AttentionType: name matches expected display strings" {
    try std.testing.expectEqualStrings("MultiHeadAttention", AttentionType.multi_head.name());
    try std.testing.expectEqualStrings("GroupedQueryAttention", AttentionType.grouped_query.name());
    try std.testing.expectEqualStrings("MultiQueryAttention", AttentionType.multi_query.name());
}

test "AttentionType: analyze selects correct type from head configuration" {
    // MHA: num_kv_heads == num_heads
    const mha = analyze(.{ .num_heads = 32, .num_kv_heads = 32 });
    try std.testing.expectEqual(AttentionType.multi_head, mha.attention_type);

    // GQA: num_kv_heads < num_heads (but > 1)
    const gqa = analyze(.{ .num_heads = 32, .num_kv_heads = 8 });
    try std.testing.expectEqual(AttentionType.grouped_query, gqa.attention_type);

    // MQA: num_kv_heads == 1
    const mqa = analyze(.{ .num_heads = 32, .num_kv_heads = 1 });
    try std.testing.expectEqual(AttentionType.multi_query, mqa.attention_type);
}
