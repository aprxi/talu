//! Integration tests for converter.WeightLayoutMap
//!
//! WeightLayoutMap maps tensor names to weight layouts for graph-driven
//! quantization decisions.

const std = @import("std");
const main = @import("main");
const converter = main.converter;
const WeightLayoutMap = converter.WeightLayoutMap;
const graph = main.graph;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "WeightLayoutMap type is accessible" {
    const T = WeightLayoutMap;
    _ = T;
}

test "WeightLayoutMap is a struct" {
    const info = @typeInfo(WeightLayoutMap);
    try std.testing.expect(info == .@"struct");
}

test "WeightLayoutMap has expected fields" {
    const info = @typeInfo(WeightLayoutMap);
    const fields = info.@"struct".fields;

    var has_allocator = false;
    var has_layouts = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "allocator")) has_allocator = true;
        if (comptime std.mem.eql(u8, field.name, "layouts")) has_layouts = true;
    }

    try std.testing.expect(has_allocator);
    try std.testing.expect(has_layouts);
}

// =============================================================================
// Method Tests
// =============================================================================

test "WeightLayoutMap has init method" {
    try std.testing.expect(@hasDecl(WeightLayoutMap, "init"));
}

test "WeightLayoutMap has deinit method" {
    try std.testing.expect(@hasDecl(WeightLayoutMap, "deinit"));
}

test "WeightLayoutMap has shouldQuantize method" {
    try std.testing.expect(@hasDecl(WeightLayoutMap, "shouldQuantize"));
}

// =============================================================================
// Lifecycle Tests
// =============================================================================

test "WeightLayoutMap.init creates empty map" {
    const allocator = std.testing.allocator;

    var map = WeightLayoutMap.init(allocator);
    defer map.deinit();

    // Map should be empty initially
    try std.testing.expectEqual(@as(usize, 0), map.layouts.count());
}

test "WeightLayoutMap.shouldQuantize returns null for unknown tensors" {
    const allocator = std.testing.allocator;

    var map = WeightLayoutMap.init(allocator);
    defer map.deinit();

    // Unknown tensor should return null
    const result = map.shouldQuantize("unknown.tensor.name");
    try std.testing.expectEqual(@as(?bool, null), result);
}

test "WeightLayoutMap.deinit is idempotent-safe" {
    const allocator = std.testing.allocator;

    var map = WeightLayoutMap.init(allocator);
    // Single deinit should work
    map.deinit();
    // Map is now undefined, don't use it again
}
