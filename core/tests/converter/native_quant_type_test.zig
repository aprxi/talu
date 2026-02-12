//! Integration tests for converter.NativeQuantType
//!
//! NativeQuantType represents native K-quant quantization types.

const std = @import("std");
const main = @import("main");
const NativeQuantType = main.converter.NativeQuantType;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "NativeQuantType type is accessible" {
    const T = NativeQuantType;
    _ = T;
}

test "NativeQuantType is an enum" {
    const info = @typeInfo(NativeQuantType);
    try std.testing.expect(info == .@"enum");
}

test "NativeQuantType has expected variants" {
    // Check for expected quantization types
    _ = NativeQuantType.q4_0;
    _ = NativeQuantType.q4_k_m;
    _ = NativeQuantType.q5_k;
    _ = NativeQuantType.q6_k;
    _ = NativeQuantType.q8_0;
    _ = NativeQuantType.f16;
}

// =============================================================================
// Method Tests
// =============================================================================

test "NativeQuantType has toString method" {
    try std.testing.expect(@hasDecl(NativeQuantType, "toString"));
}

test "NativeQuantType has fromString method" {
    try std.testing.expect(@hasDecl(NativeQuantType, "fromString"));
}

test "NativeQuantType has toDType method" {
    try std.testing.expect(@hasDecl(NativeQuantType, "toDType"));
}

// =============================================================================
// String Conversion Tests
// =============================================================================

test "NativeQuantType toString returns expected strings" {
    try std.testing.expectEqualStrings("Q4_0", NativeQuantType.q4_0.toString());
    try std.testing.expectEqualStrings("Q4_K_M", NativeQuantType.q4_k_m.toString());
    try std.testing.expectEqualStrings("Q6_K", NativeQuantType.q6_k.toString());
    try std.testing.expectEqualStrings("Q8_0", NativeQuantType.q8_0.toString());
    try std.testing.expectEqualStrings("F16", NativeQuantType.f16.toString());
}

test "NativeQuantType fromString parses valid strings" {
    try std.testing.expectEqual(NativeQuantType.q4_0, NativeQuantType.fromString("q4_0").?);
    try std.testing.expectEqual(NativeQuantType.q4_k_m, NativeQuantType.fromString("q4_k_m").?);
    try std.testing.expectEqual(NativeQuantType.q6_k, NativeQuantType.fromString("q6_k").?);
    try std.testing.expectEqual(NativeQuantType.f16, NativeQuantType.fromString("f16").?);
}

test "NativeQuantType fromString returns null for invalid strings" {
    try std.testing.expect(NativeQuantType.fromString("invalid") == null);
    try std.testing.expect(NativeQuantType.fromString("") == null);
}
