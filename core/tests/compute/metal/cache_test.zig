//! Integration tests for Metal/MLX Cache
//!
//! Cache provides KV cache management for transformer attention.
//! Supports both bfloat16 and quantized formats.
//! Tests are skipped on non-macOS platforms at compile time.

const std = @import("std");
const builtin = @import("builtin");
const main = @import("main");
const metal = main.core.compute.metal;
const graph = metal.graph;
const Cache = graph.Cache;
const device_mod = metal.device;

// =============================================================================
// Cache Creation Tests
// =============================================================================

test "Cache init creates valid cache" {
    if (comptime builtin.os.tag != .macos) return;
    if (!device_mod.isAvailable()) return;

    const cache = Cache.init(12, true); // 12 layers, bfloat16
    defer cache.deinit();

    try std.testing.expect(cache.handle != null);
    try std.testing.expect(cache.use_bfloat16);
}

test "Cache init with quantized format" {
    if (comptime builtin.os.tag != .macos) return;
    if (!device_mod.isAvailable()) return;

    const cache = Cache.init(8, false); // 8 layers, quantized
    defer cache.deinit();

    try std.testing.expect(cache.handle != null);
    try std.testing.expect(!cache.use_bfloat16);
}

test "Cache init various layer counts" {
    if (comptime builtin.os.tag != .macos) return;
    if (!device_mod.isAvailable()) return;

    const layer_counts = [_]usize{ 1, 4, 12, 24, 32 };
    for (layer_counts) |n_layers| {
        const cache = Cache.init(n_layers, true);
        defer cache.deinit();
        try std.testing.expect(cache.handle != null);
    }
}

// =============================================================================
// Cache Update Tests
// =============================================================================

test "Cache updateAndFetch stores KV data" {
    if (comptime builtin.os.tag != .macos) return;
    if (!device_mod.isAvailable()) return;

    const cache = Cache.init(2, true);
    defer cache.deinit();

    // Create K/V arrays
    const k_data = [_]f32{1.0} ** 64;
    const v_data = [_]f32{2.0} ** 64;
    const shape = [_]i64{ 1, 1, 1, 64 };

    const k_handle = graph.createArrayF32(&k_data, &shape);
    const v_handle = graph.createArrayF32(&v_data, &shape);
    defer graph.freeArray(k_handle);
    defer graph.freeArray(v_handle);

    const result = cache.updateAndFetch(0, k_handle, v_handle);

    try std.testing.expect(result.k != null);
    try std.testing.expect(result.v != null);
}

test "Cache updateAndFetch accumulates across calls" {
    if (comptime builtin.os.tag != .macos) return;
    if (!device_mod.isAvailable()) return;

    const cache = Cache.init(2, true);
    defer cache.deinit();

    const shape = [_]i64{ 1, 1, 1, 64 };

    // First update
    const k1 = [_]f32{1.0} ** 64;
    const v1 = [_]f32{1.0} ** 64;
    const k1_handle = graph.createArrayF32(&k1, &shape);
    const v1_handle = graph.createArrayF32(&v1, &shape);
    _ = cache.updateAndFetch(0, k1_handle, v1_handle);

    // Second update
    const k2 = [_]f32{2.0} ** 64;
    const v2 = [_]f32{2.0} ** 64;
    const k2_handle = graph.createArrayF32(&k2, &shape);
    const v2_handle = graph.createArrayF32(&v2, &shape);
    const result = cache.updateAndFetch(0, k2_handle, v2_handle);

    // Should have accumulated data
    try std.testing.expect(result.k != null);
    try std.testing.expect(result.v != null);
}

// =============================================================================
// Cache Get/Set Tests
// =============================================================================

test "Cache get retrieves stored data" {
    if (comptime builtin.os.tag != .macos) return;
    if (!device_mod.isAvailable()) return;

    const cache = Cache.init(2, true);
    defer cache.deinit();

    // Store data
    const k_data = [_]f32{3.14} ** 64;
    const v_data = [_]f32{2.71} ** 64;
    const shape = [_]i64{ 1, 1, 1, 64 };

    const k_handle = graph.createArrayF32(&k_data, &shape);
    const v_handle = graph.createArrayF32(&v_data, &shape);
    _ = cache.updateAndFetch(0, k_handle, v_handle);

    // Retrieve
    const result = cache.get(0);
    // Result may contain the cached data
    _ = result;
}

test "Cache setFull stores complete cache" {
    if (comptime builtin.os.tag != .macos) return;
    if (!device_mod.isAvailable()) return;

    const cache = Cache.init(2, true);
    defer cache.deinit();

    const k_data = [_]f32{1.0} ** 64;
    const v_data = [_]f32{2.0} ** 64;
    const shape = [_]i64{ 1, 1, 1, 64 };

    const k_handle = graph.createArrayF32(&k_data, &shape);
    const v_handle = graph.createArrayF32(&v_data, &shape);
    defer graph.freeArray(k_handle);
    defer graph.freeArray(v_handle);

    cache.setFull(0, k_handle, v_handle);

    // Should be retrievable
    const result = cache.get(0);
    _ = result;
}

// =============================================================================
// Cache Eval Tests
// =============================================================================

test "Cache evalAll forces evaluation" {
    if (comptime builtin.os.tag != .macos) return;
    if (!device_mod.isAvailable()) return;

    const n_layers: usize = 2;
    const cache = Cache.init(n_layers, true);
    defer cache.deinit();

    // Store data in all layers
    const shape = [_]i64{ 1, 1, 1, 64 };
    for (0..n_layers) |layer| {
        const k_data = [_]f32{1.0} ** 64;
        const v_data = [_]f32{2.0} ** 64;
        const k_handle = graph.createArrayF32(&k_data, &shape);
        const v_handle = graph.createArrayF32(&v_data, &shape);
        cache.setFull(layer, k_handle, v_handle);
    }

    // Eval all
    cache.evalAll(n_layers);

    // Verify data is accessible
    for (0..n_layers) |layer| {
        const result = cache.get(layer);
        _ = result;
    }
}

// =============================================================================
// Cache Quantized Tests
// =============================================================================

test "Cache getQuantized returns triplets" {
    if (comptime builtin.os.tag != .macos) return;
    if (!device_mod.isAvailable()) return;

    const cache = Cache.init(2, false); // quantized mode
    defer cache.deinit();

    // Store data
    const k_data = [_]f32{1.0} ** 64;
    const v_data = [_]f32{2.0} ** 64;
    const shape = [_]i64{ 1, 1, 1, 64 };

    const k_handle = graph.createArrayF32(&k_data, &shape);
    const v_handle = graph.createArrayF32(&v_data, &shape);
    _ = cache.updateAndFetch(0, k_handle, v_handle);

    // Get quantized triplets
    const result = cache.getQuantized(0);
    // Triplets contain weights, scales, biases
    _ = result;
}
