//! Integration tests for inference.backend.cpu.kernels BatchedKVCache

const std = @import("std");
const main = @import("main");

const BatchedKVCache = main.inference.backend.kernels.kv_cache.BatchedKVCache;
const SlotState = main.inference.backend.kernels.kv_cache.SlotState;

// =============================================================================
// SlotState Tests
// =============================================================================

test "SlotState has correct defaults" {
    const slot = SlotState{};
    try std.testing.expectEqual(@as(usize, 0), slot.position);
    try std.testing.expectEqual(false, slot.active);
    try std.testing.expectEqual(@as(u64, 0), slot.sequence_id);
}

// =============================================================================
// BatchedKVCache Tests
// =============================================================================

test "BatchedKVCache.init creates cache with correct dimensions" {
    const allocator = std.testing.allocator;
    const max_batch = 4;
    const n_kv_heads = 8;
    const head_dim = 64;
    const max_seq_len = 2048;

    var cache = try BatchedKVCache.init(allocator, max_batch, n_kv_heads, head_dim, max_seq_len);
    defer cache.deinit();

    try std.testing.expectEqual(@as(usize, max_batch), cache.max_batch_size);
    try std.testing.expectEqual(@as(usize, n_kv_heads), cache.n_kv_heads);
    try std.testing.expectEqual(@as(usize, head_dim), cache.head_dim);
    try std.testing.expectEqual(@as(usize, max_seq_len), cache.max_seq_len);
}

test "BatchedKVCache slot allocation" {
    const allocator = std.testing.allocator;
    var cache = try BatchedKVCache.init(allocator, 2, 2, 8, 16);
    defer cache.deinit();

    const s0 = cache.allocSlot();
    const s1 = cache.allocSlot();
    const s2 = cache.allocSlot();

    try std.testing.expectEqual(@as(?usize, 0), s0);
    try std.testing.expectEqual(@as(?usize, 1), s1);
    try std.testing.expectEqual(@as(?usize, null), s2);
}

test "BatchedKVCache.activeCount tracks active slots" {
    const allocator = std.testing.allocator;
    var cache = try BatchedKVCache.init(allocator, 4, 2, 8, 16);
    defer cache.deinit();

    try std.testing.expectEqual(@as(usize, 0), cache.activeCount());

    _ = cache.allocSlot();
    try std.testing.expectEqual(@as(usize, 1), cache.activeCount());

    const s1 = cache.allocSlot().?;
    _ = cache.allocSlot();
    try std.testing.expectEqual(@as(usize, 3), cache.activeCount());

    cache.freeSlot(s1);
    try std.testing.expectEqual(@as(usize, 2), cache.activeCount());
}

test "BatchedKVCache position management" {
    const allocator = std.testing.allocator;
    var cache = try BatchedKVCache.init(allocator, 2, 2, 8, 16);
    defer cache.deinit();

    const slot = cache.allocSlot().?;
    try std.testing.expectEqual(@as(usize, 0), cache.getPosition(slot));

    cache.setPosition(slot, 10);
    try std.testing.expectEqual(@as(usize, 10), cache.getPosition(slot));

    cache.incrementPosition(slot);
    try std.testing.expectEqual(@as(usize, 11), cache.getPosition(slot));

    cache.resetSlot(slot);
    try std.testing.expectEqual(@as(usize, 0), cache.getPosition(slot));
    try std.testing.expect(cache.isActive(slot));
}
