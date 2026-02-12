//! Integration tests for xray.FfnType
//!
//! FfnType selects the feed-forward network implementation.
//! Provides name() for display and isMoe() to distinguish expert routing.

const std = @import("std");
const main = @import("main");
const xray = main.xray;

const FfnType = xray.FfnType;
const analyze = xray.execution_plan.analyze;

// ===== name =====

test "FfnType: name returns unique string for each variant" {
    const names = [_][:0]const u8{
        FfnType.swiglu_silu.name(),
        FfnType.swiglu_gelu.name(),
        FfnType.moe_silu.name(),
        FfnType.moe_gelu.name(),
    };

    for (names, 0..) |a, i| {
        try std.testing.expect(a.len > 0);
        for (names[i + 1 ..]) |b| {
            try std.testing.expect(!std.mem.eql(u8, a, b));
        }
    }
}

test "FfnType: name matches expected display strings" {
    try std.testing.expectEqualStrings("SwiGLU(SiLU)", FfnType.swiglu_silu.name());
    try std.testing.expectEqualStrings("SwiGLU(GELU)", FfnType.swiglu_gelu.name());
    try std.testing.expectEqualStrings("MoE(SiLU)", FfnType.moe_silu.name());
    try std.testing.expectEqualStrings("MoE(GELU)", FfnType.moe_gelu.name());
}

// ===== isMoe =====

test "FfnType: isMoe distinguishes expert-routed from dense FFN" {
    try std.testing.expect(!FfnType.swiglu_silu.isMoe());
    try std.testing.expect(!FfnType.swiglu_gelu.isMoe());
    try std.testing.expect(FfnType.moe_silu.isMoe());
    try std.testing.expect(FfnType.moe_gelu.isMoe());
}

test "FfnType: analyze selects correct type from model config" {
    // Dense SiLU (default)
    const dense = analyze(.{ .num_experts = 0 });
    try std.testing.expectEqual(FfnType.swiglu_silu, dense.ffn_type);
    try std.testing.expect(!dense.ffn_type.isMoe());

    // Dense GELU
    const gelu = analyze(.{ .num_experts = 0, .use_gelu = true });
    try std.testing.expectEqual(FfnType.swiglu_gelu, gelu.ffn_type);

    // MoE SiLU
    const moe = analyze(.{ .num_experts = 8, .experts_per_token = 2 });
    try std.testing.expectEqual(FfnType.moe_silu, moe.ffn_type);
    try std.testing.expect(moe.ffn_type.isMoe());

    // MoE GELU
    const moe_gelu = analyze(.{ .num_experts = 8, .experts_per_token = 2, .use_gelu = true });
    try std.testing.expectEqual(FfnType.moe_gelu, moe_gelu.ffn_type);
}
