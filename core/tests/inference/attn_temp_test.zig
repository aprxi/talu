//! Integration tests for inference.AttnTemp
//!
//! Tests the AttnTemp struct which provides temporary buffers for attention computation.

const std = @import("std");
const main = @import("main");

const AttnTemp = main.inference.backend.AttnTemp;

// =============================================================================
// Initialization Tests
// =============================================================================

test "AttnTemp initializes with empty slices" {
    const temp = AttnTemp{};

    try std.testing.expectEqual(@as(usize, 0), temp.q.len);
    try std.testing.expectEqual(@as(usize, 0), temp.k.len);
    try std.testing.expectEqual(@as(usize, 0), temp.v.len);
    try std.testing.expectEqual(@as(usize, 0), temp.qkv.len);
    try std.testing.expectEqual(@as(usize, 0), temp.scores.len);
    try std.testing.expectEqual(@as(usize, 0), temp.context_buffer.len);
}

// =============================================================================
// Memory Management Tests
// =============================================================================

test "AttnTemp.deinit frees allocated buffers" {
    const allocator = std.testing.allocator;

    var temp = AttnTemp{};
    temp.q = try allocator.alloc(f32, 100);
    temp.k = try allocator.alloc(f32, 100);
    temp.v = try allocator.alloc(f32, 100);

    temp.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), temp.q.len);
    try std.testing.expectEqual(@as(usize, 0), temp.k.len);
    try std.testing.expectEqual(@as(usize, 0), temp.v.len);
}

test "AttnTemp.deinit is safe to call on empty struct" {
    const allocator = std.testing.allocator;

    var temp = AttnTemp{};
    temp.deinit(allocator);
}

test "AttnTemp.deinit handles partial allocation" {
    const allocator = std.testing.allocator;

    var temp = AttnTemp{};
    temp.q = try allocator.alloc(f32, 50);
    temp.scores = try allocator.alloc(f32, 200);

    temp.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), temp.q.len);
    try std.testing.expectEqual(@as(usize, 0), temp.scores.len);
}

test "AttnTemp buffers can be allocated independently" {
    const allocator = std.testing.allocator;

    var temp = AttnTemp{};
    temp.qkv = try allocator.alloc(f32, 512);
    temp.context_buffer = try allocator.alloc(f32, 256);

    try std.testing.expectEqual(@as(usize, 512), temp.qkv.len);
    try std.testing.expectEqual(@as(usize, 256), temp.context_buffer.len);

    temp.deinit(allocator);
}
