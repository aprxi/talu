//! Integration tests for inference.backend.cpu.kernels LayeredBatchedKVCache

const std = @import("std");
const main = @import("main");

const LayeredBatchedKVCache = main.inference.backend.kernels.kv_cache.LayeredBatchedKVCache;

test "LayeredBatchedKVCache.init creates caches for all layers" {
    const allocator = std.testing.allocator;
    const n_layers = 4;

    var layered = try LayeredBatchedKVCache.init(allocator, n_layers, 2, 8, 64, 128);
    defer layered.deinit();

    try std.testing.expectEqual(n_layers, layered.n_layers);
    try std.testing.expectEqual(n_layers, layered.layers.len);
}

test "LayeredBatchedKVCache.getLayer returns correct layer" {
    const allocator = std.testing.allocator;
    var layered = try LayeredBatchedKVCache.init(allocator, 4, 2, 4, 32, 64);
    defer layered.deinit();

    const layer0 = layered.getLayer(0);
    const layer2 = layered.getLayer(2);

    try std.testing.expect(layer0 != layer2);
    try std.testing.expectEqual(@as(usize, 4), layer0.n_kv_heads);
}

test "LayeredBatchedKVCache position sync across layers" {
    const allocator = std.testing.allocator;
    const n_layers = 4;
    var layered = try LayeredBatchedKVCache.init(allocator, n_layers, 2, 4, 32, 64);
    defer layered.deinit();

    const slot = layered.allocSlot().?;
    layered.setPosition(slot, 42);

    for (0..n_layers) |layer_idx| {
        try std.testing.expectEqual(@as(usize, 42), layered.getLayer(layer_idx).getPosition(slot));
    }
}

test "LayeredBatchedKVCache.freeSlot works across layers" {
    const allocator = std.testing.allocator;
    var layered = try LayeredBatchedKVCache.init(allocator, 2, 2, 4, 32, 64);
    defer layered.deinit();

    const slot = layered.allocSlot().?;
    layered.freeSlot(slot);

    for (0..2) |layer_idx| {
        try std.testing.expect(!layered.getLayer(layer_idx).isActive(slot));
    }
}
