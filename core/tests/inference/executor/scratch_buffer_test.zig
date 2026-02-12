//! Integration tests for inference.executor.ScratchBuffer
//!
//! Tests the ScratchBuffer type from the executor module.

const std = @import("std");
const main = @import("main");

const ScratchBuffer = main.inference.executor.ScratchBuffer;

test "ScratchBuffer type is accessible" {
    const T = ScratchBuffer;
    _ = T;
}

test "ScratchBuffer.init creates buffer with correct dimensions" {
    const allocator = std.testing.allocator;
    const d_model = 64;
    const d_ff = 256;
    const n_layers = 2;

    var scratch = try ScratchBuffer.init(allocator, d_model, d_ff, n_layers);
    defer scratch.deinit();

    try std.testing.expectEqual(d_model, scratch.d_model);
    try std.testing.expectEqual(d_ff, scratch.d_ff);
    try std.testing.expectEqual(n_layers, scratch.attn_caches.len);
}

test "ScratchBuffer.resetCaches resets all cache positions" {
    const allocator = std.testing.allocator;

    var scratch = try ScratchBuffer.init(allocator, 64, 256, 3);
    defer scratch.deinit();

    for (scratch.attn_caches) |*cache| {
        cache.cache_pos = 42;
    }

    scratch.resetCaches();

    for (scratch.attn_caches) |cache| {
        try std.testing.expectEqual(@as(usize, 0), cache.cache_pos);
    }
}
