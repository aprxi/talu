//! Integration tests for io.safetensors.ShardedSafeTensors
//!
//! ShardedSafeTensors handles models split across multiple safetensors files.

const std = @import("std");
const main = @import("main");
const ShardedSafeTensors = main.io.safetensors.root.ShardedSafeTensors;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "ShardedSafeTensors type is accessible" {
    const T = ShardedSafeTensors;
    _ = T;
}

test "ShardedSafeTensors is a struct" {
    const info = @typeInfo(ShardedSafeTensors);
    try std.testing.expect(info == .@"struct");
}

test "ShardedSafeTensors has expected fields" {
    const info = @typeInfo(ShardedSafeTensors);
    const fields = info.@"struct".fields;

    var has_allocator = false;
    var has_weight_map = false;
    var has_shards = false;
    var has_base_dir = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "allocator")) has_allocator = true;
        if (comptime std.mem.eql(u8, field.name, "weight_map")) has_weight_map = true;
        if (comptime std.mem.eql(u8, field.name, "shards")) has_shards = true;
        if (comptime std.mem.eql(u8, field.name, "base_dir")) has_base_dir = true;
    }

    try std.testing.expect(has_allocator);
    try std.testing.expect(has_weight_map);
    try std.testing.expect(has_shards);
    try std.testing.expect(has_base_dir);
}

// =============================================================================
// Method Tests
// =============================================================================

test "ShardedSafeTensors has load method" {
    try std.testing.expect(@hasDecl(ShardedSafeTensors, "load"));
}

test "ShardedSafeTensors has deinit method" {
    try std.testing.expect(@hasDecl(ShardedSafeTensors, "deinit"));
}

test "ShardedSafeTensors has getTensor method" {
    try std.testing.expect(@hasDecl(ShardedSafeTensors, "getTensor"));
}
