//! Integration tests for inference.AttnCache
//!
//! Tests the AttnCache struct which stores key-value pairs for attention.

const std = @import("std");
const main = @import("main");

const AttnCache = main.inference.backend.AttnCache;

// =============================================================================
// Initialization Tests
// =============================================================================

test "AttnCache initializes with empty slices and zero position" {
    const cache = AttnCache{};

    try std.testing.expectEqual(@as(usize, 0), cache.kv_k.len);
    try std.testing.expectEqual(@as(usize, 0), cache.kv_v.len);
    try std.testing.expectEqual(@as(usize, 0), cache.kv_capacity);
    try std.testing.expectEqual(@as(usize, 0), cache.cache_pos);
}

// =============================================================================
// Position Management Tests
// =============================================================================

test "AttnCache.resetCache sets position to zero" {
    var cache = AttnCache{};
    cache.cache_pos = 100;

    cache.resetCache();

    try std.testing.expectEqual(@as(usize, 0), cache.cache_pos);
}

test "AttnCache can track cache position" {
    var cache = AttnCache{};

    cache.cache_pos = 0;
    cache.cache_pos += 10;
    try std.testing.expectEqual(@as(usize, 10), cache.cache_pos);

    cache.cache_pos += 5;
    try std.testing.expectEqual(@as(usize, 15), cache.cache_pos);

    cache.resetCache();
    try std.testing.expectEqual(@as(usize, 0), cache.cache_pos);
}

// =============================================================================
// Memory Management Tests
// =============================================================================

test "AttnCache.deinit frees allocated KV buffers" {
    const allocator = std.testing.allocator;

    var cache = AttnCache{};
    cache.kv_k = try allocator.alloc(f32, 1024);
    cache.kv_v = try allocator.alloc(f32, 1024);
    cache.kv_capacity = 128;
    cache.cache_pos = 64;

    cache.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), cache.kv_k.len);
    try std.testing.expectEqual(@as(usize, 0), cache.kv_v.len);
}

test "AttnCache.deinit is safe on empty struct" {
    const allocator = std.testing.allocator;

    var cache = AttnCache{};
    cache.deinit(allocator);
}

test "AttnCache capacity tracks allocated size" {
    const allocator = std.testing.allocator;
    const capacity = 256;
    const kv_size = capacity * 64; // capacity * head_dim

    var cache = AttnCache{};
    cache.kv_k = try allocator.alloc(f32, kv_size);
    cache.kv_v = try allocator.alloc(f32, kv_size);
    cache.kv_capacity = capacity;

    try std.testing.expectEqual(capacity, cache.kv_capacity);
    try std.testing.expectEqual(kv_size, cache.kv_k.len);

    cache.deinit(allocator);
}
