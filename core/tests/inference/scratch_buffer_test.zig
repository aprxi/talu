//! Integration tests for inference scratch buffer types
//!
//! Tests AttnTemp, AttnCache, FfnScratch, and ScratchBuffer which provide
//! pre-allocated working memory for inference operations.

const std = @import("std");
const main = @import("main");

const AttnTemp = main.inference.backend.AttnTemp;
const AttnCache = main.inference.backend.AttnCache;
const FfnScratch = main.inference.backend.FfnScratch;
const ScratchBuffer = main.inference.backend.ScratchBuffer;

// =============================================================================
// AttnTemp Tests
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

test "AttnTemp.deinit frees allocated buffers" {
    const allocator = std.testing.allocator;

    var temp = AttnTemp{};

    // Allocate some buffers
    temp.q = try allocator.alloc(f32, 100);
    temp.k = try allocator.alloc(f32, 100);
    temp.v = try allocator.alloc(f32, 100);

    // deinit should free all
    temp.deinit(allocator);

    // After deinit, all slices should be empty
    try std.testing.expectEqual(@as(usize, 0), temp.q.len);
    try std.testing.expectEqual(@as(usize, 0), temp.k.len);
    try std.testing.expectEqual(@as(usize, 0), temp.v.len);
}

test "AttnTemp.deinit is safe to call on empty struct" {
    const allocator = std.testing.allocator;

    var temp = AttnTemp{};
    temp.deinit(allocator); // Should not crash
}

test "AttnTemp.deinit handles partial allocation" {
    const allocator = std.testing.allocator;

    var temp = AttnTemp{};

    // Only allocate some buffers
    temp.q = try allocator.alloc(f32, 50);
    temp.scores = try allocator.alloc(f32, 200);

    temp.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), temp.q.len);
    try std.testing.expectEqual(@as(usize, 0), temp.scores.len);
}

// =============================================================================
// AttnCache Tests
// =============================================================================

test "AttnCache initializes with empty slices and zero position" {
    const cache = AttnCache{};

    try std.testing.expectEqual(@as(usize, 0), cache.kv_k.len);
    try std.testing.expectEqual(@as(usize, 0), cache.kv_v.len);
    try std.testing.expectEqual(@as(usize, 0), cache.kv_capacity);
    try std.testing.expectEqual(@as(usize, 0), cache.cache_pos);
}

test "AttnCache.resetCache sets position to zero" {
    var cache = AttnCache{};
    cache.cache_pos = 100;

    cache.resetCache();

    try std.testing.expectEqual(@as(usize, 0), cache.cache_pos);
}

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

test "AttnCache can track cache position" {
    var cache = AttnCache{};

    // Simulate filling cache
    cache.cache_pos = 0;
    cache.cache_pos += 10; // Add 10 tokens
    try std.testing.expectEqual(@as(usize, 10), cache.cache_pos);

    cache.cache_pos += 5; // Add 5 more
    try std.testing.expectEqual(@as(usize, 15), cache.cache_pos);

    // Reset for new sequence
    cache.resetCache();
    try std.testing.expectEqual(@as(usize, 0), cache.cache_pos);
}

// =============================================================================
// FfnScratch Tests
// =============================================================================

test "FfnScratch initializes with empty slices" {
    const scratch = FfnScratch{};

    try std.testing.expectEqual(@as(usize, 0), scratch.gate.len);
    try std.testing.expectEqual(@as(usize, 0), scratch.gate_act.len);
    try std.testing.expectEqual(@as(usize, 0), scratch.up.len);
    try std.testing.expectEqual(@as(usize, 0), scratch.hidden.len);
}

test "FfnScratch.deinit frees all buffers" {
    const allocator = std.testing.allocator;

    var scratch = FfnScratch{};
    scratch.gate = try allocator.alloc(f32, 256);
    scratch.gate_act = try allocator.alloc(f32, 256);
    scratch.up = try allocator.alloc(f32, 256);
    scratch.hidden = try allocator.alloc(f32, 256);

    scratch.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), scratch.gate.len);
    try std.testing.expectEqual(@as(usize, 0), scratch.gate_act.len);
    try std.testing.expectEqual(@as(usize, 0), scratch.up.len);
    try std.testing.expectEqual(@as(usize, 0), scratch.hidden.len);
}

test "FfnScratch.deinit is safe on empty struct" {
    const allocator = std.testing.allocator;

    var scratch = FfnScratch{};
    scratch.deinit(allocator); // Should not crash
}

// =============================================================================
// ScratchBuffer Tests
// =============================================================================

test "ScratchBuffer.init creates buffer with correct dimensions" {
    const allocator = std.testing.allocator;
    const d_model = 512;
    const d_ff = 2048;
    const n_layers = 4;

    var scratch = try ScratchBuffer.init(allocator, d_model, d_ff, n_layers);
    defer scratch.deinit();

    try std.testing.expectEqual(d_model, scratch.d_model);
    try std.testing.expectEqual(d_ff, scratch.d_ff);
    try std.testing.expectEqual(n_layers, scratch.attn_caches.len);
}

test "ScratchBuffer.ensure allocates temporary buffers" {
    const allocator = std.testing.allocator;
    const d_model = 64;
    const d_ff = 256;
    const n_layers = 2;

    var scratch = try ScratchBuffer.init(allocator, d_model, d_ff, n_layers);
    defer scratch.deinit();

    // Initially tmp buffers are empty
    try std.testing.expectEqual(@as(usize, 0), scratch.tmp[0].len);

    // After ensure, they should be allocated
    try scratch.ensure(16); // seq_len = 16

    // tmp buffers should now have size
    try std.testing.expect(scratch.tmp[0].len > 0);
}

test "ScratchBuffer.getTmp returns correct slice" {
    const allocator = std.testing.allocator;
    const d_model = 64;
    const d_ff = 256;
    const n_layers = 2;
    const seq_len = 8;

    var scratch = try ScratchBuffer.init(allocator, d_model, d_ff, n_layers);
    defer scratch.deinit();

    try scratch.ensure(seq_len);

    // BufferId.norm_out = 1
    const BufferId = main.inference.executor.BufferId;
    const norm_out_slice = scratch.getTmp(BufferId.norm_out, 100);
    try std.testing.expectEqual(@as(usize, 100), norm_out_slice.len);

    // BufferId.branch_out = 2
    const branch_out_slice = scratch.getTmp(BufferId.branch_out, 50);
    try std.testing.expectEqual(@as(usize, 50), branch_out_slice.len);
}

test "ScratchBuffer.getLayerTmp returns layer temp buffer" {
    const allocator = std.testing.allocator;

    var scratch = try ScratchBuffer.init(allocator, 64, 256, 2);
    defer scratch.deinit();

    try scratch.ensure(8);

    const layer_tmp = scratch.getLayerTmp(100);
    try std.testing.expectEqual(@as(usize, 100), layer_tmp.len);
}

test "ScratchBuffer.resetCaches resets all cache positions" {
    const allocator = std.testing.allocator;

    var scratch = try ScratchBuffer.init(allocator, 64, 256, 3);
    defer scratch.deinit();

    // Set some cache positions
    for (scratch.attn_caches) |*cache| {
        cache.cache_pos = 42;
    }

    scratch.resetCaches();

    // All should be reset
    for (scratch.attn_caches) |cache| {
        try std.testing.expectEqual(@as(usize, 0), cache.cache_pos);
    }
}

test "ScratchBuffer.deinit frees all memory" {
    const allocator = std.testing.allocator;

    var scratch = try ScratchBuffer.init(allocator, 128, 512, 4);
    try scratch.ensure(32);

    scratch.deinit();

    // Test passes if no memory leak detected
}

test "ScratchBuffer works with small model config" {
    const allocator = std.testing.allocator;

    // Minimal config (tiny model)
    var scratch = try ScratchBuffer.init(allocator, 32, 64, 1);
    defer scratch.deinit();

    try scratch.ensure(1); // Single token
}

test "ScratchBuffer works with large model config" {
    const allocator = std.testing.allocator;

    // Large config (similar to 7B model)
    var scratch = try ScratchBuffer.init(allocator, 4096, 11008, 32);
    defer scratch.deinit();

    try scratch.ensure(1); // Single token decode
}
