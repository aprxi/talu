//! Integration tests for io.safetensors.UnifiedSafeTensors
//!
//! UnifiedSafeTensors provides unified access to both single and sharded models.

const std = @import("std");
const main = @import("main");
const UnifiedSafeTensors = main.io.safetensors.root.UnifiedSafeTensors;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "UnifiedSafeTensors type is accessible" {
    const T = UnifiedSafeTensors;
    _ = T;
}

test "UnifiedSafeTensors is a union" {
    const info = @typeInfo(UnifiedSafeTensors);
    try std.testing.expect(info == .@"union");
}

test "UnifiedSafeTensors has single variant" {
    const info = @typeInfo(UnifiedSafeTensors);
    const fields = info.@"union".fields;

    var has_single = false;
    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "single")) has_single = true;
    }

    try std.testing.expect(has_single);
}

test "UnifiedSafeTensors has sharded variant" {
    const info = @typeInfo(UnifiedSafeTensors);
    const fields = info.@"union".fields;

    var has_sharded = false;
    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "sharded")) has_sharded = true;
    }

    try std.testing.expect(has_sharded);
}

// =============================================================================
// Method Tests
// =============================================================================

test "UnifiedSafeTensors has load method" {
    try std.testing.expect(@hasDecl(UnifiedSafeTensors, "load"));
}

test "UnifiedSafeTensors has deinit method" {
    try std.testing.expect(@hasDecl(UnifiedSafeTensors, "deinit"));
}

test "UnifiedSafeTensors has getTensor method" {
    try std.testing.expect(@hasDecl(UnifiedSafeTensors, "getTensor"));
}
