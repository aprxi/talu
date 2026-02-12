//! Integration tests for inference.FfnLayer
//!
//! Tests the FfnLayer union which can be either SwiGLU or MoE FFN.
//! Note: Full forward pass tests require loaded weights.

const std = @import("std");
const main = @import("main");

const FfnLayer = main.inference.backend.FfnLayer;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "FfnLayer type is accessible" {
    const T = FfnLayer;
    _ = T;
}

test "FfnLayer is a union" {
    const info = @typeInfo(FfnLayer);
    try std.testing.expect(info == .@"union");
}

test "FfnLayer has swiglu and moe_ffn variants" {
    const info = @typeInfo(FfnLayer);
    const fields = info.@"union".fields;

    var has_swiglu = false;
    var has_moe_ffn = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "swiglu")) has_swiglu = true;
        if (comptime std.mem.eql(u8, field.name, "moe_ffn")) has_moe_ffn = true;
    }

    try std.testing.expect(has_swiglu);
    try std.testing.expect(has_moe_ffn);
}
