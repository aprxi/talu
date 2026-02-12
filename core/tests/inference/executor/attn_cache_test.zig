//! Integration tests for inference.executor.AttnCache
//!
//! Tests the AttnCache type from the executor module.

const std = @import("std");
const main = @import("main");

const AttnCache = main.inference.executor.AttnCache;

test "AttnCache type is accessible" {
    const T = AttnCache;
    _ = T;
}

test "AttnCache initializes with empty slices" {
    const cache = AttnCache{};
    try std.testing.expectEqual(@as(usize, 0), cache.kv_k.len);
    try std.testing.expectEqual(@as(usize, 0), cache.kv_v.len);
    try std.testing.expectEqual(@as(usize, 0), cache.cache_pos);
}

test "AttnCache.resetCache sets position to zero" {
    var cache = AttnCache{};
    cache.cache_pos = 100;
    cache.resetCache();
    try std.testing.expectEqual(@as(usize, 0), cache.cache_pos);
}
