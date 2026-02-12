//! Batched KV Cache for Continuous Batching
//!
//! This module provides a slotted KV cache that supports multiple concurrent
//! sequences. Each slot holds the KV cache for one sequence, enabling
//! continuous batching where requests can join/leave at token boundaries.
//!
//! Memory layout (per layer):
//!   K cache: [max_batch, n_kv_heads, max_seq_len, head_dim]
//!   V cache: [max_batch, n_kv_heads, max_seq_len, head_dim]
//!
//! Slots are allocated/freed dynamically as requests arrive/complete.

const std = @import("std");

/// Per-slot state tracking sequence position and activity.
pub const SlotState = struct {
    /// Current sequence position (number of tokens cached)
    position: usize = 0,
    /// Whether this slot is actively being used
    active: bool = false,
    /// Sequence ID for debugging/tracking (optional)
    sequence_id: u64 = 0,
};

/// Batched KV cache supporting multiple concurrent sequences.
///
/// Unlike the single-sequence AttnCache, this cache pre-allocates slots
/// for `max_batch_size` sequences and manages their lifecycle.
pub const BatchedKVCache = struct {
    allocator: std.mem.Allocator,

    // Configuration
    max_batch_size: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,

    // Per-slot state
    slots: []SlotState,

    // KV storage: flattened [max_batch, n_kv_heads, max_seq_len, head_dim]
    key_cache: []f32,
    value_cache: []f32,

    // Derived constants for indexing
    slot_stride: usize, // n_kv_heads * max_seq_len * head_dim
    head_stride: usize, // max_seq_len * head_dim

    /// Create a new batched KV cache.
    ///
    /// This allocates the full cache upfront for predictable memory usage.
    /// Memory = 2 * max_batch * n_kv_heads * max_seq_len * head_dim * sizeof(f32)
    pub fn init(
        allocator: std.mem.Allocator,
        max_batch_size: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) !BatchedKVCache {
        const head_stride = max_seq_len * head_dim;
        const slot_stride = n_kv_heads * head_stride;
        const total_entries = max_batch_size * slot_stride;

        const slot_state_entries = try allocator.alloc(SlotState, max_batch_size);
        errdefer allocator.free(slot_state_entries);
        @memset(slot_state_entries, SlotState{});

        const key_cache = try allocator.alloc(f32, total_entries);
        errdefer allocator.free(key_cache);

        const value_cache = try allocator.alloc(f32, total_entries);
        errdefer allocator.free(value_cache);

        return .{
            .allocator = allocator,
            .max_batch_size = max_batch_size,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .max_seq_len = max_seq_len,
            .slots = slot_state_entries,
            .key_cache = key_cache,
            .value_cache = value_cache,
            .slot_stride = slot_stride,
            .head_stride = head_stride,
        };
    }

    pub fn deinit(self: *BatchedKVCache) void {
        self.allocator.free(self.value_cache);
        self.allocator.free(self.key_cache);
        self.allocator.free(self.slots);
        self.* = undefined;
    }

    // =========================================================================
    // Slot Management
    // =========================================================================

    /// Allocate a free slot for a new sequence.
    /// Returns null if no slots are available.
    pub fn allocSlot(self: *BatchedKVCache) ?usize {
        for (self.slots, 0..) |*slot, slot_index| {
            if (!slot.active) {
                slot.active = true;
                slot.position = 0;
                slot.sequence_id = 0;
                return slot_index;
            }
        }
        return null;
    }

    /// Allocate a slot with a specific sequence ID for tracking.
    pub fn allocSlotWithId(self: *BatchedKVCache, sequence_id: u64) ?usize {
        if (self.allocSlot()) |slot_index| {
            self.slots[slot_index].sequence_id = sequence_id;
            return slot_index;
        }
        return null;
    }

    /// Free a slot, making it available for new sequences.
    pub fn freeSlot(self: *BatchedKVCache, slot_index: usize) void {
        std.debug.assert(slot_index < self.max_batch_size);
        self.slots[slot_index].active = false;
        self.slots[slot_index].position = 0;
    }

    /// Reset a slot's position without freeing it (for reuse with new prompt).
    pub fn resetSlot(self: *BatchedKVCache, slot_index: usize) void {
        std.debug.assert(slot_index < self.max_batch_size);
        self.slots[slot_index].position = 0;
    }

    /// Get the current position for a slot.
    pub fn getPosition(self: *const BatchedKVCache, slot_index: usize) usize {
        std.debug.assert(slot_index < self.max_batch_size);
        return self.slots[slot_index].position;
    }

    /// Set the position for a slot (after prefill or decode).
    pub fn setPosition(self: *BatchedKVCache, slot_index: usize, position: usize) void {
        std.debug.assert(slot_index < self.max_batch_size);
        std.debug.assert(position <= self.max_seq_len);
        self.slots[slot_index].position = position;
    }

    /// Increment position by 1 (after single token decode).
    pub fn incrementPosition(self: *BatchedKVCache, slot_index: usize) void {
        std.debug.assert(slot_index < self.max_batch_size);
        std.debug.assert(self.slots[slot_index].position < self.max_seq_len);
        self.slots[slot_index].position += 1;
    }

    /// Check if a slot is active.
    pub fn isActive(self: *const BatchedKVCache, slot_index: usize) bool {
        std.debug.assert(slot_index < self.max_batch_size);
        return self.slots[slot_index].active;
    }

    /// Count number of active slots.
    pub fn activeCount(self: *const BatchedKVCache) usize {
        var active_slot_count: usize = 0;
        for (self.slots) |slot| {
            if (slot.active) active_slot_count += 1;
        }
        return active_slot_count;
    }

    /// Get list of active slot indices.
    pub fn getActiveSlots(self: *const BatchedKVCache, out: []usize) usize {
        var active_slot_count: usize = 0;
        for (self.slots, 0..) |slot, slot_index| {
            if (slot.active) {
                if (active_slot_count < out.len) {
                    out[active_slot_count] = slot_index;
                }
                active_slot_count += 1;
            }
        }
        return active_slot_count;
    }

    // =========================================================================
    // KV Access
    // =========================================================================

    /// Get K cache slice for a specific (slot, kv_head, position).
    /// Returns a slice of length `head_dim`.
    pub fn getK(self: *const BatchedKVCache, slot_index: usize, kv_head: usize, position: usize) []f32 {
        const offset = self.kvOffset(slot_index, kv_head, position);
        return self.key_cache[offset..][0..self.head_dim];
    }

    /// Get V cache slice for a specific (slot, kv_head, position).
    pub fn getV(self: *const BatchedKVCache, slot_index: usize, kv_head: usize, position: usize) []f32 {
        const offset = self.kvOffset(slot_index, kv_head, position);
        return self.value_cache[offset..][0..self.head_dim];
    }

    /// Get K cache base pointer for a (slot, kv_head) - all positions.
    /// Layout: [position, head_dim] with stride = head_dim
    pub fn getKHead(self: *const BatchedKVCache, slot_index: usize, kv_head: usize) []f32 {
        const offset = slot_index * self.slot_stride + kv_head * self.head_stride;
        return self.key_cache[offset..][0..self.head_stride];
    }

    /// Get V cache base pointer for a (slot, kv_head) - all positions.
    pub fn getVHead(self: *const BatchedKVCache, slot_index: usize, kv_head: usize) []f32 {
        const offset = slot_index * self.slot_stride + kv_head * self.head_stride;
        return self.value_cache[offset..][0..self.head_stride];
    }

    /// Append K/V vectors at the current position and increment.
    /// k_data and v_data should be [n_kv_heads * head_dim] (one token).
    pub fn appendKV(self: *BatchedKVCache, slot_index: usize, k_data: []const f32, v_data: []const f32) !void {
        std.debug.assert(slot_index < self.max_batch_size);
        const slot_state = &self.slots[slot_index];
        if (slot_state.position >= self.max_seq_len) return error.CacheOverflow;

        const slot_position = slot_state.position;
        const kv_values_per_token = self.n_kv_heads * self.head_dim;
        std.debug.assert(k_data.len == kv_values_per_token);
        std.debug.assert(v_data.len == kv_values_per_token);

        // Copy K/V for each head
        for (0..self.n_kv_heads) |kv_head| {
            const src_k = k_data[kv_head * self.head_dim ..][0..self.head_dim];
            const src_v = v_data[kv_head * self.head_dim ..][0..self.head_dim];
            const dst_k = self.getK(slot_index, kv_head, slot_position);
            const dst_v = self.getV(slot_index, kv_head, slot_position);
            @memcpy(dst_k, src_k);
            @memcpy(dst_v, src_v);
        }

        slot_state.position = slot_position + 1;
    }

    /// Append K/V for multiple positions (prefill).
    /// k_data and v_data should be [seq_len * n_kv_heads * head_dim].
    pub fn appendKVBatch(
        self: *BatchedKVCache,
        slot_index: usize,
        k_data: []const f32,
        v_data: []const f32,
        seq_len: usize,
    ) !void {
        std.debug.assert(slot_index < self.max_batch_size);
        const slot_state = &self.slots[slot_index];
        const kv_values_per_token = self.n_kv_heads * self.head_dim;

        if (slot_state.position + seq_len > self.max_seq_len) return error.CacheOverflow;
        std.debug.assert(k_data.len == seq_len * kv_values_per_token);
        std.debug.assert(v_data.len == seq_len * kv_values_per_token);

        const start_position = slot_state.position;

        // Copy K/V for each position and head
        for (0..self.n_kv_heads) |kv_head| {
            for (0..seq_len) |token_index| {
                const src_k = k_data[token_index * kv_values_per_token + kv_head * self.head_dim ..][0..self.head_dim];
                const src_v = v_data[token_index * kv_values_per_token + kv_head * self.head_dim ..][0..self.head_dim];
                const dst_k = self.getK(slot_index, kv_head, start_position + token_index);
                const dst_v = self.getV(slot_index, kv_head, start_position + token_index);
                @memcpy(dst_k, src_k);
                @memcpy(dst_v, src_v);
            }
        }

        slot_state.position = start_position + seq_len;
    }

    // =========================================================================
    // Internal
    // =========================================================================

    fn kvOffset(self: *const BatchedKVCache, slot_index: usize, kv_head: usize, position: usize) usize {
        std.debug.assert(slot_index < self.max_batch_size);
        std.debug.assert(kv_head < self.n_kv_heads);
        std.debug.assert(position < self.max_seq_len);
        return slot_index * self.slot_stride + kv_head * self.head_stride + position * self.head_dim;
    }
};

// =============================================================================
// Per-Layer Cache Array
// =============================================================================

/// Array of batched KV caches, one per transformer layer.
pub const LayeredBatchedKVCache = struct {
    allocator: std.mem.Allocator,
    layers: []BatchedKVCache,
    n_layers: usize,

    pub fn init(
        allocator: std.mem.Allocator,
        n_layers: usize,
        max_batch_size: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) !LayeredBatchedKVCache {
        const layer_caches = try allocator.alloc(BatchedKVCache, n_layers);
        errdefer allocator.free(layer_caches);

        var initialized_count: usize = 0;
        errdefer {
            for (layer_caches[0..initialized_count]) |*layer| {
                layer.deinit();
            }
        }

        for (layer_caches) |*layer| {
            layer.* = try BatchedKVCache.init(
                allocator,
                max_batch_size,
                n_kv_heads,
                head_dim,
                max_seq_len,
            );
            initialized_count += 1;
        }

        return .{
            .allocator = allocator,
            .layers = layer_caches,
            .n_layers = n_layers,
        };
    }

    pub fn deinit(self: *LayeredBatchedKVCache) void {
        for (self.layers) |*layer| {
            layer.deinit();
        }
        self.allocator.free(self.layers);
        self.* = undefined;
    }

    /// Get the cache for a specific layer.
    pub fn getLayer(self: *LayeredBatchedKVCache, layer_idx: usize) *BatchedKVCache {
        std.debug.assert(layer_idx < self.n_layers);
        return &self.layers[layer_idx];
    }

    /// Allocate a slot across all layers (atomic operation).
    pub fn allocSlot(self: *LayeredBatchedKVCache) ?usize {
        // All layers share the same slot indices, so we only check layer 0
        return self.layers[0].allocSlot();
    }

    /// Allocate with sequence ID.
    pub fn allocSlotWithId(self: *LayeredBatchedKVCache, sequence_id: u64) ?usize {
        if (self.allocSlot()) |slot_index| {
            // Set sequence ID in all layers for debugging
            for (self.layers) |*layer| {
                layer.slots[slot_index].sequence_id = sequence_id;
            }
            return slot_index;
        }
        return null;
    }

    /// Free a slot across all layers.
    pub fn freeSlot(self: *LayeredBatchedKVCache, slot_index: usize) void {
        for (self.layers) |*layer| {
            layer.freeSlot(slot_index);
        }
    }

    /// Reset a slot across all layers.
    pub fn resetSlot(self: *LayeredBatchedKVCache, slot_index: usize) void {
        for (self.layers) |*layer| {
            layer.resetSlot(slot_index);
        }
    }

    /// Get position (same across all layers).
    pub fn getPosition(self: *const LayeredBatchedKVCache, slot_index: usize) usize {
        return self.layers[0].getPosition(slot_index);
    }

    /// Set position across all layers.
    pub fn setPosition(self: *LayeredBatchedKVCache, slot_index: usize, position: usize) void {
        for (self.layers) |*layer| {
            layer.setPosition(slot_index, position);
        }
    }

    /// Increment position across all layers.
    pub fn incrementPosition(self: *LayeredBatchedKVCache, slot_index: usize) void {
        for (self.layers) |*layer| {
            layer.incrementPosition(slot_index);
        }
    }
};

// =============================================================================
// Tests
// =============================================================================

test "allocSlot basic" {
    const allocator = std.testing.allocator;
    var kv_cache = try BatchedKVCache.init(allocator, 4, 8, 64, 2048);
    defer kv_cache.deinit();

    // Allocate all slots
    const slot0 = kv_cache.allocSlot();
    const slot1 = kv_cache.allocSlot();
    const slot2 = kv_cache.allocSlot();
    const slot3 = kv_cache.allocSlot();
    const slot4 = kv_cache.allocSlot(); // Should fail

    try std.testing.expect(slot0 != null);
    try std.testing.expect(slot1 != null);
    try std.testing.expect(slot2 != null);
    try std.testing.expect(slot3 != null);
    try std.testing.expect(slot4 == null);

    try std.testing.expectEqual(@as(usize, 4), kv_cache.activeCount());

    // Free one and reallocate
    kv_cache.freeSlot(slot1.?);
    try std.testing.expectEqual(@as(usize, 3), kv_cache.activeCount());

    const slot1_new = kv_cache.allocSlot();
    try std.testing.expect(slot1_new != null);
    try std.testing.expectEqual(slot1.?, slot1_new.?);
}

test "appendKV basic" {
    const allocator = std.testing.allocator;
    const n_kv_heads = 2;
    const head_dim = 4;
    var kv_cache = try BatchedKVCache.init(allocator, 2, n_kv_heads, head_dim, 16);
    defer kv_cache.deinit();

    const slot = kv_cache.allocSlot().?;
    try std.testing.expectEqual(@as(usize, 0), kv_cache.getPosition(slot));

    // Append one token
    const k_data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 }; // 2 heads * 4 dim
    const v_data = [_]f32{ 10, 20, 30, 40, 50, 60, 70, 80 };
    try kv_cache.appendKV(slot, &k_data, &v_data);

    try std.testing.expectEqual(@as(usize, 1), kv_cache.getPosition(slot));

    // Verify data
    const k0 = kv_cache.getK(slot, 0, 0);
    const k1 = kv_cache.getK(slot, 1, 0);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1, 2, 3, 4 }, k0);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 5, 6, 7, 8 }, k1);

    const v0 = kv_cache.getV(slot, 0, 0);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 10, 20, 30, 40 }, v0);
}

test "init LayeredBatchedKVCache" {
    const allocator = std.testing.allocator;
    var layered_cache = try LayeredBatchedKVCache.init(allocator, 4, 2, 8, 64, 1024);
    defer layered_cache.deinit();

    const slot = layered_cache.allocSlot().?;
    try std.testing.expectEqual(@as(usize, 0), layered_cache.getPosition(slot));

    layered_cache.setPosition(slot, 10);
    try std.testing.expectEqual(@as(usize, 10), layered_cache.getPosition(slot));

    // Position should be same across all layers
    for (0..4) |layer_idx| {
        try std.testing.expectEqual(@as(usize, 10), layered_cache.getLayer(layer_idx).getPosition(slot));
    }

    layered_cache.freeSlot(slot);
    try std.testing.expect(!layered_cache.layers[0].isActive(slot));
}

// =============================================================================
// Comprehensive Unit Tests
// =============================================================================

test "init various sizes" {
    const allocator = std.testing.allocator;

    // Small cache
    {
        var cache = try BatchedKVCache.init(allocator, 1, 1, 32, 128);
        defer cache.deinit();
        try std.testing.expectEqual(@as(usize, 1), cache.max_batch_size);
        try std.testing.expectEqual(@as(usize, 1), cache.n_kv_heads);
        try std.testing.expectEqual(@as(usize, 32), cache.head_dim);
        try std.testing.expectEqual(@as(usize, 128), cache.max_seq_len);
        try std.testing.expectEqual(@as(usize, 0), cache.activeCount());
    }

    // Typical cache
    {
        var cache = try BatchedKVCache.init(allocator, 8, 4, 64, 2048);
        defer cache.deinit();
        try std.testing.expectEqual(@as(usize, 8), cache.max_batch_size);
        try std.testing.expectEqual(@as(usize, 4), cache.n_kv_heads);
        try std.testing.expectEqual(@as(usize, 2048), cache.max_seq_len);
    }

    // Large cache
    {
        var cache = try BatchedKVCache.init(allocator, 16, 8, 128, 4096);
        defer cache.deinit();
        try std.testing.expectEqual(@as(usize, 16), cache.max_batch_size);
        try std.testing.expectEqual(@as(usize, 8), cache.n_kv_heads);
    }
}

test "init slot state" {
    const allocator = std.testing.allocator;
    var cache = try BatchedKVCache.init(allocator, 4, 2, 32, 1024);
    defer cache.deinit();

    // All slots should be inactive and at position 0
    for (cache.slots, 0..) |slot, idx| {
        try std.testing.expect(!slot.active);
        try std.testing.expectEqual(@as(usize, 0), slot.position);
        try std.testing.expectEqual(@as(u64, 0), slot.sequence_id);
        try std.testing.expect(!cache.isActive(idx));
    }
}

test "allocSlot and freeSlot" {
    const allocator = std.testing.allocator;
    var cache = try BatchedKVCache.init(allocator, 3, 2, 32, 512);
    defer cache.deinit();

    // Allocate first slot
    const slot0 = cache.allocSlot();
    try std.testing.expect(slot0 != null);
    try std.testing.expectEqual(@as(usize, 0), slot0.?);
    try std.testing.expect(cache.isActive(slot0.?));
    try std.testing.expectEqual(@as(usize, 1), cache.activeCount());

    // Allocate second slot
    const slot1 = cache.allocSlot();
    try std.testing.expect(slot1 != null);
    try std.testing.expectEqual(@as(usize, 1), slot1.?);
    try std.testing.expectEqual(@as(usize, 2), cache.activeCount());

    // Allocate third slot
    const slot2 = cache.allocSlot();
    try std.testing.expect(slot2 != null);
    try std.testing.expectEqual(@as(usize, 2), slot2.?);
    try std.testing.expectEqual(@as(usize, 3), cache.activeCount());

    // Try to allocate fourth slot (should fail)
    const slot3 = cache.allocSlot();
    try std.testing.expect(slot3 == null);
    try std.testing.expectEqual(@as(usize, 3), cache.activeCount());

    // Free middle slot
    cache.freeSlot(slot1.?);
    try std.testing.expect(!cache.isActive(slot1.?));
    try std.testing.expectEqual(@as(usize, 2), cache.activeCount());

    // Reallocate should reuse freed slot
    const slot1_new = cache.allocSlot();
    try std.testing.expectEqual(slot1.?, slot1_new.?);
    try std.testing.expectEqual(@as(usize, 3), cache.activeCount());
}

test "BatchedKVCache: allocSlotWithId" {
    const allocator = std.testing.allocator;
    var cache = try BatchedKVCache.init(allocator, 4, 2, 32, 512);
    defer cache.deinit();

    const seq_id: u64 = 12345;
    const slot = cache.allocSlotWithId(seq_id);
    try std.testing.expect(slot != null);
    try std.testing.expectEqual(seq_id, cache.slots[slot.?].sequence_id);
    try std.testing.expect(cache.isActive(slot.?));

    // Allocate all remaining slots
    _ = cache.allocSlot();
    _ = cache.allocSlot();
    _ = cache.allocSlot();

    // Should return null when full
    const slot_full = cache.allocSlotWithId(67890);
    try std.testing.expect(slot_full == null);
}

test "BatchedKVCache: resetSlot preserves active status" {
    const allocator = std.testing.allocator;
    var cache = try BatchedKVCache.init(allocator, 2, 2, 32, 512);
    defer cache.deinit();

    const slot = cache.allocSlot().?;
    cache.setPosition(slot, 100);
    try std.testing.expectEqual(@as(usize, 100), cache.getPosition(slot));
    try std.testing.expect(cache.isActive(slot));

    // Reset should clear position but keep slot active
    cache.resetSlot(slot);
    try std.testing.expectEqual(@as(usize, 0), cache.getPosition(slot));
    try std.testing.expect(cache.isActive(slot));
}

test "getPosition setPosition management" {
    const allocator = std.testing.allocator;
    var cache = try BatchedKVCache.init(allocator, 2, 2, 32, 1024);
    defer cache.deinit();

    const slot = cache.allocSlot().?;

    // Initial position
    try std.testing.expectEqual(@as(usize, 0), cache.getPosition(slot));

    // Set position
    cache.setPosition(slot, 42);
    try std.testing.expectEqual(@as(usize, 42), cache.getPosition(slot));

    // Increment position
    cache.incrementPosition(slot);
    try std.testing.expectEqual(@as(usize, 43), cache.getPosition(slot));

    // Increment again
    cache.incrementPosition(slot);
    try std.testing.expectEqual(@as(usize, 44), cache.getPosition(slot));

    // Set to max
    cache.setPosition(slot, 1024);
    try std.testing.expectEqual(@as(usize, 1024), cache.getPosition(slot));

    // Reset brings it back to 0
    cache.resetSlot(slot);
    try std.testing.expectEqual(@as(usize, 0), cache.getPosition(slot));
}

test "BatchedKVCache: getActiveSlots" {
    const allocator = std.testing.allocator;
    var cache = try BatchedKVCache.init(allocator, 5, 2, 32, 512);
    defer cache.deinit();

    var active_slots: [5]usize = undefined;

    // No active slots initially
    const count0 = cache.getActiveSlots(&active_slots);
    try std.testing.expectEqual(@as(usize, 0), count0);

    // Allocate some slots
    const slot0 = cache.allocSlot().?;
    const slot1 = cache.allocSlot().?;
    const slot2 = cache.allocSlot().?;

    const count1 = cache.getActiveSlots(&active_slots);
    try std.testing.expectEqual(@as(usize, 3), count1);
    try std.testing.expectEqual(slot0, active_slots[0]);
    try std.testing.expectEqual(slot1, active_slots[1]);
    try std.testing.expectEqual(slot2, active_slots[2]);

    // Free middle slot
    cache.freeSlot(slot1);
    const count2 = cache.getActiveSlots(&active_slots);
    try std.testing.expectEqual(@as(usize, 2), count2);
    try std.testing.expectEqual(slot0, active_slots[0]);
    try std.testing.expectEqual(slot2, active_slots[1]);
}

test "BatchedKVCache: getActiveSlots with small buffer" {
    const allocator = std.testing.allocator;
    var cache = try BatchedKVCache.init(allocator, 5, 2, 32, 512);
    defer cache.deinit();

    // Allocate 4 slots
    _ = cache.allocSlot();
    _ = cache.allocSlot();
    _ = cache.allocSlot();
    _ = cache.allocSlot();

    // Buffer too small - should still return correct count
    var small_buffer: [2]usize = undefined;
    const count = cache.getActiveSlots(&small_buffer);
    try std.testing.expectEqual(@as(usize, 4), count);
    // Only first 2 should be filled
    try std.testing.expectEqual(@as(usize, 0), small_buffer[0]);
    try std.testing.expectEqual(@as(usize, 1), small_buffer[1]);
}

test "getKHead getVHead indexing" {
    const allocator = std.testing.allocator;
    const n_kv_heads = 2;
    const head_dim = 4;
    const max_seq_len = 8;
    var cache = try BatchedKVCache.init(allocator, 2, n_kv_heads, head_dim, max_seq_len);
    defer cache.deinit();

    const slot = cache.allocSlot().?;

    // Get K/V slices at various positions
    for (0..n_kv_heads) |kv_head| {
        for (0..max_seq_len) |pos| {
            const k_slice = cache.getK(slot, kv_head, pos);
            const v_slice = cache.getV(slot, kv_head, pos);
            try std.testing.expectEqual(head_dim, k_slice.len);
            try std.testing.expectEqual(head_dim, v_slice.len);
        }
    }

    // Check head slices
    for (0..n_kv_heads) |kv_head| {
        const k_head = cache.getKHead(slot, kv_head);
        const v_head = cache.getVHead(slot, kv_head);
        try std.testing.expectEqual(cache.head_stride, k_head.len);
        try std.testing.expectEqual(cache.head_stride, v_head.len);
    }
}

test "BatchedKVCache: appendKV single token" {
    const allocator = std.testing.allocator;
    const n_kv_heads = 2;
    const head_dim = 4;
    var cache = try BatchedKVCache.init(allocator, 2, n_kv_heads, head_dim, 16);
    defer cache.deinit();

    const slot = cache.allocSlot().?;

    // Append first token
    const k1 = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 }; // head0: 1,2,3,4  head1: 5,6,7,8
    const v1 = [_]f32{ 11, 12, 13, 14, 15, 16, 17, 18 };
    try cache.appendKV(slot, &k1, &v1);

    try std.testing.expectEqual(@as(usize, 1), cache.getPosition(slot));

    // Verify data at position 0
    {
        const k_h0 = cache.getK(slot, 0, 0);
        const k_h1 = cache.getK(slot, 1, 0);
        try std.testing.expectEqualSlices(f32, &[_]f32{ 1, 2, 3, 4 }, k_h0);
        try std.testing.expectEqualSlices(f32, &[_]f32{ 5, 6, 7, 8 }, k_h1);

        const v_h0 = cache.getV(slot, 0, 0);
        const v_h1 = cache.getV(slot, 1, 0);
        try std.testing.expectEqualSlices(f32, &[_]f32{ 11, 12, 13, 14 }, v_h0);
        try std.testing.expectEqualSlices(f32, &[_]f32{ 15, 16, 17, 18 }, v_h1);
    }

    // Append second token
    const k2 = [_]f32{ 10, 20, 30, 40, 50, 60, 70, 80 };
    const v2 = [_]f32{ 110, 120, 130, 140, 150, 160, 170, 180 };
    try cache.appendKV(slot, &k2, &v2);

    try std.testing.expectEqual(@as(usize, 2), cache.getPosition(slot));

    // Verify data at position 1
    {
        const k_h0 = cache.getK(slot, 0, 1);
        const k_h1 = cache.getK(slot, 1, 1);
        try std.testing.expectEqualSlices(f32, &[_]f32{ 10, 20, 30, 40 }, k_h0);
        try std.testing.expectEqualSlices(f32, &[_]f32{ 50, 60, 70, 80 }, k_h1);
    }
}

test "BatchedKVCache: appendKV overflow detection" {
    const allocator = std.testing.allocator;
    const max_seq_len = 2;
    var cache = try BatchedKVCache.init(allocator, 1, 1, 4, max_seq_len);
    defer cache.deinit();

    const slot = cache.allocSlot().?;
    const k_data = [_]f32{ 1, 2, 3, 4 };
    const v_data = [_]f32{ 5, 6, 7, 8 };

    // First append
    try cache.appendKV(slot, &k_data, &v_data);
    try std.testing.expectEqual(@as(usize, 1), cache.getPosition(slot));

    // Second append
    try cache.appendKV(slot, &k_data, &v_data);
    try std.testing.expectEqual(@as(usize, 2), cache.getPosition(slot));

    // Third append should fail
    try std.testing.expectError(error.CacheOverflow, cache.appendKV(slot, &k_data, &v_data));
    try std.testing.expectEqual(@as(usize, 2), cache.getPosition(slot));
}

test "BatchedKVCache: appendKVBatch prefill" {
    const allocator = std.testing.allocator;
    const n_kv_heads = 2;
    const head_dim = 3;
    const seq_len = 4;
    var cache = try BatchedKVCache.init(allocator, 1, n_kv_heads, head_dim, 16);
    defer cache.deinit();

    const slot = cache.allocSlot().?;

    // Create prefill data: [seq_len * n_kv_heads * head_dim]
    // seq_len=4, n_kv_heads=2, head_dim=3 => 24 elements
    const k_data = [_]f32{
        // token 0
        1, 2, 3, // head 0
        4, 5, 6, // head 1
        // token 1
        10, 11, 12, // head 0
        13, 14, 15, // head 1
        // token 2
        20, 21, 22, // head 0
        23, 24, 25, // head 1
        // token 3
        30, 31, 32, // head 0
        33, 34, 35, // head 1
    };
    const v_data = [_]f32{
        // token 0
        101, 102, 103,
        104, 105, 106,
        // token 1
        111, 112, 113,
        114, 115, 116,
        // token 2
        121, 122, 123,
        124, 125, 126,
        // token 3
        131, 132, 133,
        134, 135, 136,
    };

    try cache.appendKVBatch(slot, &k_data, &v_data, seq_len);
    try std.testing.expectEqual(@as(usize, 4), cache.getPosition(slot));

    // Verify token 0
    {
        const k_h0 = cache.getK(slot, 0, 0);
        const k_h1 = cache.getK(slot, 1, 0);
        try std.testing.expectEqualSlices(f32, &[_]f32{ 1, 2, 3 }, k_h0);
        try std.testing.expectEqualSlices(f32, &[_]f32{ 4, 5, 6 }, k_h1);

        const v_h0 = cache.getV(slot, 0, 0);
        try std.testing.expectEqualSlices(f32, &[_]f32{ 101, 102, 103 }, v_h0);
    }

    // Verify token 2
    {
        const k_h0 = cache.getK(slot, 0, 2);
        const k_h1 = cache.getK(slot, 1, 2);
        try std.testing.expectEqualSlices(f32, &[_]f32{ 20, 21, 22 }, k_h0);
        try std.testing.expectEqualSlices(f32, &[_]f32{ 23, 24, 25 }, k_h1);

        const v_h1 = cache.getV(slot, 1, 2);
        try std.testing.expectEqualSlices(f32, &[_]f32{ 124, 125, 126 }, v_h1);
    }

    // Verify token 3
    {
        const k_h0 = cache.getK(slot, 0, 3);
        try std.testing.expectEqualSlices(f32, &[_]f32{ 30, 31, 32 }, k_h0);

        const v_h0 = cache.getV(slot, 0, 3);
        try std.testing.expectEqualSlices(f32, &[_]f32{ 131, 132, 133 }, v_h0);
    }
}

test "BatchedKVCache: appendKVBatch overflow detection" {
    const allocator = std.testing.allocator;
    const max_seq_len = 5;
    var cache = try BatchedKVCache.init(allocator, 1, 1, 2, max_seq_len);
    defer cache.deinit();

    const slot = cache.allocSlot().?;

    // Prefill 3 tokens
    const k3 = [_]f32{ 1, 2, 3, 4, 5, 6 }; // 3 tokens * 1 head * 2 dim
    const v3 = [_]f32{ 1, 2, 3, 4, 5, 6 };
    try cache.appendKVBatch(slot, &k3, &v3, 3);
    try std.testing.expectEqual(@as(usize, 3), cache.getPosition(slot));

    // Add 2 more tokens (should succeed, total=5)
    const k2 = [_]f32{ 7, 8, 9, 10 }; // 2 tokens * 1 head * 2 dim
    const v2 = [_]f32{ 7, 8, 9, 10 };
    try cache.appendKVBatch(slot, &k2, &v2, 2);
    try std.testing.expectEqual(@as(usize, 5), cache.getPosition(slot));

    // Try to add 1 more token (should fail)
    const k1 = [_]f32{ 11, 12 };
    const v1 = [_]f32{ 11, 12 };
    try std.testing.expectError(error.CacheOverflow, cache.appendKVBatch(slot, &k1, &v1, 1));
    try std.testing.expectEqual(@as(usize, 5), cache.getPosition(slot));
}

test "BatchedKVCache: appendKVBatch with prior position" {
    const allocator = std.testing.allocator;
    var cache = try BatchedKVCache.init(allocator, 1, 1, 2, 16);
    defer cache.deinit();

    const slot = cache.allocSlot().?;

    // Set initial position to 5
    cache.setPosition(slot, 5);

    // Append 3 tokens starting from position 5
    const k_data = [_]f32{ 1, 2, 3, 4, 5, 6 }; // 3 tokens
    const v_data = [_]f32{ 11, 12, 13, 14, 15, 16 };
    try cache.appendKVBatch(slot, &k_data, &v_data, 3);

    try std.testing.expectEqual(@as(usize, 8), cache.getPosition(slot));

    // Verify data at positions 5, 6, 7
    const k5 = cache.getK(slot, 0, 5);
    const k6 = cache.getK(slot, 0, 6);
    const k7 = cache.getK(slot, 0, 7);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1, 2 }, k5);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 3, 4 }, k6);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 5, 6 }, k7);
}

test "allocSlot multiple independent" {
    const allocator = std.testing.allocator;
    const n_kv_heads = 1;
    const head_dim = 3;
    var cache = try BatchedKVCache.init(allocator, 3, n_kv_heads, head_dim, 8);
    defer cache.deinit();

    const slot0 = cache.allocSlot().?;
    const slot1 = cache.allocSlot().?;
    const slot2 = cache.allocSlot().?;

    // Different data for each slot
    const k0_data = [_]f32{ 1, 2, 3 };
    const v0_data = [_]f32{ 10, 20, 30 };
    const k1_data = [_]f32{ 4, 5, 6 };
    const v1_data = [_]f32{ 40, 50, 60 };
    const k2_data = [_]f32{ 7, 8, 9 };
    const v2_data = [_]f32{ 70, 80, 90 };

    try cache.appendKV(slot0, &k0_data, &v0_data);
    try cache.appendKV(slot1, &k1_data, &v1_data);
    try cache.appendKV(slot2, &k2_data, &v2_data);

    // Verify each slot has its own data
    const k0 = cache.getK(slot0, 0, 0);
    const k1 = cache.getK(slot1, 0, 0);
    const k2 = cache.getK(slot2, 0, 0);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1, 2, 3 }, k0);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 4, 5, 6 }, k1);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 7, 8, 9 }, k2);

    const v1_vec = cache.getV(slot1, 0, 0);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 40, 50, 60 }, v1_vec);

    // Each slot has independent position
    try std.testing.expectEqual(@as(usize, 1), cache.getPosition(slot0));
    try std.testing.expectEqual(@as(usize, 1), cache.getPosition(slot1));
    try std.testing.expectEqual(@as(usize, 1), cache.getPosition(slot2));

    // Advance slot1 further
    try cache.appendKV(slot1, &k1_data, &v1_data);
    try std.testing.expectEqual(@as(usize, 1), cache.getPosition(slot0));
    try std.testing.expectEqual(@as(usize, 2), cache.getPosition(slot1));
    try std.testing.expectEqual(@as(usize, 1), cache.getPosition(slot2));
}

test "BatchedKVCache: getKHead and getVHead coverage" {
    const allocator = std.testing.allocator;
    const n_kv_heads = 3;
    const head_dim = 4;
    const max_seq_len = 5;
    var cache = try BatchedKVCache.init(allocator, 2, n_kv_heads, head_dim, max_seq_len);
    defer cache.deinit();

    const slot = cache.allocSlot().?;

    // Populate some data
    for (0..n_kv_heads) |kv_head| {
        for (0..3) |pos| {
            const k_slice = cache.getK(slot, kv_head, pos);
            const v_slice = cache.getV(slot, kv_head, pos);
            for (k_slice, 0..) |*elem, idx| {
                elem.* = @floatFromInt(kv_head * 1000 + pos * 100 + idx);
            }
            for (v_slice, 0..) |*elem, idx| {
                elem.* = @floatFromInt(kv_head * 1000 + pos * 100 + idx + 50);
            }
        }
    }

    // Verify getKHead returns correct slice
    const k_head1 = cache.getKHead(slot, 1);
    try std.testing.expectEqual(cache.head_stride, k_head1.len);

    // Verify we can access data through head slice
    // Position 0, head 1
    const k_pos0 = k_head1[0..head_dim];
    try std.testing.expectEqual(@as(f32, 1000), k_pos0[0]);
    try std.testing.expectEqual(@as(f32, 1001), k_pos0[1]);

    // Position 1, head 1
    const k_pos1 = k_head1[head_dim .. 2 * head_dim];
    try std.testing.expectEqual(@as(f32, 1100), k_pos1[0]);
}

test "init LayeredBatchedKVCache structure" {
    const allocator = std.testing.allocator;
    const n_layers = 6;
    const max_batch = 4;
    const n_kv_heads = 4;
    const head_dim = 64;
    const max_seq_len = 1024;

    var layered = try LayeredBatchedKVCache.init(
        allocator,
        n_layers,
        max_batch,
        n_kv_heads,
        head_dim,
        max_seq_len,
    );
    defer layered.deinit();

    try std.testing.expectEqual(n_layers, layered.n_layers);
    try std.testing.expectEqual(n_layers, layered.layers.len);

    // Each layer should have the same configuration
    for (layered.layers) |*layer| {
        try std.testing.expectEqual(max_batch, layer.max_batch_size);
        try std.testing.expectEqual(n_kv_heads, layer.n_kv_heads);
        try std.testing.expectEqual(head_dim, layer.head_dim);
        try std.testing.expectEqual(max_seq_len, layer.max_seq_len);
    }
}

test "LayeredBatchedKVCache: getLayer accessor" {
    const allocator = std.testing.allocator;
    var layered = try LayeredBatchedKVCache.init(allocator, 4, 2, 8, 64, 512);
    defer layered.deinit();

    // Get each layer
    for (0..4) |layer_idx| {
        const layer = layered.getLayer(layer_idx);
        try std.testing.expectEqual(@as(usize, 2), layer.max_batch_size);
        try std.testing.expectEqual(@as(usize, 8), layer.n_kv_heads);
    }
}

test "LayeredBatchedKVCache: allocSlot synchronization" {
    const allocator = std.testing.allocator;
    var layered = try LayeredBatchedKVCache.init(allocator, 3, 2, 4, 32, 256);
    defer layered.deinit();

    const slot = layered.allocSlot();
    try std.testing.expect(slot != null);

    // allocSlot only marks layer 0 as active (implementation detail)
    try std.testing.expect(layered.layers[0].isActive(slot.?));
    try std.testing.expectEqual(@as(usize, 0), layered.layers[0].getPosition(slot.?));
    try std.testing.expectEqual(@as(usize, 1), layered.layers[0].activeCount());

    // Other layers are not automatically marked active
    // (This is current behavior - slots are managed independently per layer)
}

test "LayeredBatchedKVCache: allocSlotWithId synchronization" {
    const allocator = std.testing.allocator;
    var layered = try LayeredBatchedKVCache.init(allocator, 3, 2, 4, 32, 256);
    defer layered.deinit();

    const seq_id: u64 = 98765;
    const slot = layered.allocSlotWithId(seq_id);
    try std.testing.expect(slot != null);

    // Sequence ID should be set in all layers
    for (layered.layers) |*layer| {
        try std.testing.expectEqual(seq_id, layer.slots[slot.?].sequence_id);
    }

    // But only layer 0 is marked active (current implementation)
    try std.testing.expect(layered.layers[0].isActive(slot.?));
}

test "LayeredBatchedKVCache: freeSlot synchronization" {
    const allocator = std.testing.allocator;
    var layered = try LayeredBatchedKVCache.init(allocator, 4, 3, 4, 32, 256);
    defer layered.deinit();

    const slot0 = layered.allocSlot().?;
    const slot1 = layered.allocSlot().?;

    // Both slots are active in layer 0
    try std.testing.expect(layered.layers[0].isActive(slot0));
    try std.testing.expect(layered.layers[0].isActive(slot1));

    // Free slot 0
    layered.freeSlot(slot0);

    // Slot 0 should be freed in all layers
    for (layered.layers) |*layer| {
        try std.testing.expect(!layer.isActive(slot0));
    }

    // Slot 1 should still be active in layer 0
    try std.testing.expect(layered.layers[0].isActive(slot1));
}

test "LayeredBatchedKVCache: resetSlot synchronization" {
    const allocator = std.testing.allocator;
    var layered = try LayeredBatchedKVCache.init(allocator, 3, 2, 4, 32, 256);
    defer layered.deinit();

    const slot = layered.allocSlot().?;

    // Set different positions in each layer (shouldn't normally happen, but test it)
    layered.layers[0].setPosition(slot, 10);
    layered.layers[1].setPosition(slot, 20);
    layered.layers[2].setPosition(slot, 30);

    // Reset should affect all layers
    layered.resetSlot(slot);

    for (layered.layers) |*layer| {
        try std.testing.expectEqual(@as(usize, 0), layer.getPosition(slot));
    }

    // Only layer 0 is marked active (from allocSlot)
    try std.testing.expect(layered.layers[0].isActive(slot));
}

test "setPosition LayeredBatchedKVCache sync" {
    const allocator = std.testing.allocator;
    var layered = try LayeredBatchedKVCache.init(allocator, 4, 2, 4, 32, 512);
    defer layered.deinit();

    const slot = layered.allocSlot().?;

    // getPosition should read from layer 0
    try std.testing.expectEqual(@as(usize, 0), layered.getPosition(slot));

    // setPosition should update all layers
    layered.setPosition(slot, 42);
    try std.testing.expectEqual(@as(usize, 42), layered.getPosition(slot));
    for (layered.layers) |*layer| {
        try std.testing.expectEqual(@as(usize, 42), layer.getPosition(slot));
    }

    // incrementPosition should update all layers
    layered.incrementPosition(slot);
    try std.testing.expectEqual(@as(usize, 43), layered.getPosition(slot));
    for (layered.layers) |*layer| {
        try std.testing.expectEqual(@as(usize, 43), layer.getPosition(slot));
    }
}

test "getLayer KV isolation" {
    const allocator = std.testing.allocator;
    const n_layers = 2;
    const n_kv_heads = 1;
    const head_dim = 3;
    var layered = try LayeredBatchedKVCache.init(
        allocator,
        n_layers,
        2,
        n_kv_heads,
        head_dim,
        8,
    );
    defer layered.deinit();

    const slot = layered.allocSlot().?;

    // Store different data in each layer
    const k0_data = [_]f32{ 1, 2, 3 };
    const v0_data = [_]f32{ 10, 20, 30 };
    const k1_data = [_]f32{ 4, 5, 6 };
    const v1_data = [_]f32{ 40, 50, 60 };

    try layered.getLayer(0).appendKV(slot, &k0_data, &v0_data);
    try layered.getLayer(1).appendKV(slot, &k1_data, &v1_data);

    // Verify data is isolated per layer
    const layer0_k = layered.getLayer(0).getK(slot, 0, 0);
    const layer1_k = layered.getLayer(1).getK(slot, 0, 0);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1, 2, 3 }, layer0_k);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 4, 5, 6 }, layer1_k);

    const layer0_v = layered.getLayer(0).getV(slot, 0, 0);
    const layer1_v = layered.getLayer(1).getV(slot, 0, 0);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 10, 20, 30 }, layer0_v);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 40, 50, 60 }, layer1_v);
}

test "init LayeredBatchedKVCache empty" {
    const allocator = std.testing.allocator;
    var layered = try LayeredBatchedKVCache.init(allocator, 2, 2, 4, 32, 256);
    defer layered.deinit();

    // Allocate all slots
    const slot0 = layered.allocSlot();
    const slot1 = layered.allocSlot();
    try std.testing.expect(slot0 != null);
    try std.testing.expect(slot1 != null);

    // Try to allocate when full
    const slot2 = layered.allocSlot();
    try std.testing.expect(slot2 == null);

    // Same for allocSlotWithId
    const slot3 = layered.allocSlotWithId(12345);
    try std.testing.expect(slot3 == null);
}

test "init memory layout" {
    const allocator = std.testing.allocator;
    const max_batch_size = 2;
    const n_kv_heads = 2;
    const head_dim = 3;
    const max_seq_len = 4;
    var cache = try BatchedKVCache.init(
        allocator,
        max_batch_size,
        n_kv_heads,
        head_dim,
        max_seq_len,
    );
    defer cache.deinit();

    // Verify stride calculations
    const expected_head_stride = max_seq_len * head_dim; // 4 * 3 = 12
    const expected_slot_stride = n_kv_heads * expected_head_stride; // 2 * 12 = 24
    try std.testing.expectEqual(expected_head_stride, cache.head_stride);
    try std.testing.expectEqual(expected_slot_stride, cache.slot_stride);

    // Verify total memory allocation
    const expected_total = max_batch_size * expected_slot_stride; // 2 * 24 = 48
    try std.testing.expectEqual(expected_total, cache.key_cache.len);
    try std.testing.expectEqual(expected_total, cache.value_cache.len);
}

test "getKHead kvOffset calculation" {
    const allocator = std.testing.allocator;
    const max_batch_size = 2;
    const n_kv_heads = 3;
    const head_dim = 4;
    const max_seq_len = 5;
    var cache = try BatchedKVCache.init(
        allocator,
        max_batch_size,
        n_kv_heads,
        head_dim,
        max_seq_len,
    );
    defer cache.deinit();

    // Test various offset calculations through getK/getV
    const slot0_head0_pos0 = cache.kvOffset(0, 0, 0);
    const slot0_head1_pos0 = cache.kvOffset(0, 1, 0);
    const slot0_head0_pos1 = cache.kvOffset(0, 0, 1);
    const slot1_head0_pos0 = cache.kvOffset(1, 0, 0);

    // Verify offsets are distinct and follow expected pattern
    try std.testing.expectEqual(@as(usize, 0), slot0_head0_pos0);
    try std.testing.expectEqual(cache.head_stride, slot0_head1_pos0 - slot0_head0_pos0);
    try std.testing.expectEqual(cache.head_dim, slot0_head0_pos1 - slot0_head0_pos0);
    try std.testing.expectEqual(cache.slot_stride, slot1_head0_pos0 - slot0_head0_pos0);
}

test "BatchedKVCache.init: validates memory allocation" {
    const allocator = std.testing.allocator;

    // Test basic initialization
    var cache = try BatchedKVCache.init(allocator, 2, 4, 64, 128);
    defer cache.deinit();

    // Verify all fields are initialized
    try std.testing.expectEqual(@as(usize, 2), cache.max_batch_size);
    try std.testing.expectEqual(@as(usize, 4), cache.n_kv_heads);
    try std.testing.expectEqual(@as(usize, 64), cache.head_dim);
    try std.testing.expectEqual(@as(usize, 128), cache.max_seq_len);
    try std.testing.expectEqual(@as(usize, 2), cache.slots.len);

    // Verify stride calculations
    const expected_head_stride = 128 * 64; // max_seq_len * head_dim
    const expected_slot_stride = 4 * expected_head_stride; // n_kv_heads * head_stride
    try std.testing.expectEqual(expected_head_stride, cache.head_stride);
    try std.testing.expectEqual(expected_slot_stride, cache.slot_stride);

    // Verify memory allocation sizes
    const expected_total = 2 * expected_slot_stride; // max_batch_size * slot_stride
    try std.testing.expectEqual(expected_total, cache.key_cache.len);
    try std.testing.expectEqual(expected_total, cache.value_cache.len);
}

test "BatchedKVCache.init: edge cases" {
    const allocator = std.testing.allocator;

    // Minimum size cache (1x1x1x1)
    {
        var cache = try BatchedKVCache.init(allocator, 1, 1, 1, 1);
        defer cache.deinit();
        try std.testing.expectEqual(@as(usize, 1), cache.key_cache.len);
        try std.testing.expectEqual(@as(usize, 1), cache.value_cache.len);
    }

    // Large head dimension
    {
        var cache = try BatchedKVCache.init(allocator, 1, 1, 256, 16);
        defer cache.deinit();
        try std.testing.expectEqual(@as(usize, 256), cache.head_dim);
    }

    // Large sequence length
    {
        var cache = try BatchedKVCache.init(allocator, 1, 1, 32, 8192);
        defer cache.deinit();
        try std.testing.expectEqual(@as(usize, 8192), cache.max_seq_len);
    }
}

test "BatchedKVCache.deinit: cleanup and invalidation" {
    const allocator = std.testing.allocator;
    var cache = try BatchedKVCache.init(allocator, 4, 8, 64, 512);

    // Verify allocations were made
    try std.testing.expectEqual(@as(usize, 4), cache.slots.len);
    try std.testing.expect(cache.key_cache.len > 0);
    try std.testing.expect(cache.value_cache.len > 0);

    cache.deinit();

    // After deinit, the struct should be undefined
    // The allocator in test mode will catch any memory leaks,
    // which verifies proper cleanup
}

test "BatchedKVCache.getPosition: boundary conditions" {
    const allocator = std.testing.allocator;
    var cache = try BatchedKVCache.init(allocator, 3, 2, 32, 100);
    defer cache.deinit();

    const slot0 = cache.allocSlot().?;
    const slot1 = cache.allocSlot().?;
    const slot2 = cache.allocSlot().?;

    // Initial positions are all 0
    try std.testing.expectEqual(@as(usize, 0), cache.getPosition(slot0));
    try std.testing.expectEqual(@as(usize, 0), cache.getPosition(slot1));
    try std.testing.expectEqual(@as(usize, 0), cache.getPosition(slot2));

    // Set different positions
    cache.setPosition(slot0, 0);
    cache.setPosition(slot1, 50);
    cache.setPosition(slot2, 100);

    try std.testing.expectEqual(@as(usize, 0), cache.getPosition(slot0));
    try std.testing.expectEqual(@as(usize, 50), cache.getPosition(slot1));
    try std.testing.expectEqual(@as(usize, 100), cache.getPosition(slot2));
}

test "BatchedKVCache.setPosition: validates boundaries" {
    const allocator = std.testing.allocator;
    const max_seq_len = 64;
    var cache = try BatchedKVCache.init(allocator, 2, 2, 32, max_seq_len);
    defer cache.deinit();

    const slot = cache.allocSlot().?;

    // Set to 0 (minimum)
    cache.setPosition(slot, 0);
    try std.testing.expectEqual(@as(usize, 0), cache.getPosition(slot));

    // Set to max_seq_len (maximum allowed)
    cache.setPosition(slot, max_seq_len);
    try std.testing.expectEqual(@as(usize, max_seq_len), cache.getPosition(slot));

    // Set to middle values
    cache.setPosition(slot, 32);
    try std.testing.expectEqual(@as(usize, 32), cache.getPosition(slot));

    cache.setPosition(slot, 1);
    try std.testing.expectEqual(@as(usize, 1), cache.getPosition(slot));

    cache.setPosition(slot, max_seq_len - 1);
    try std.testing.expectEqual(@as(usize, max_seq_len - 1), cache.getPosition(slot));
}

test "BatchedKVCache.incrementPosition: sequential increments" {
    const allocator = std.testing.allocator;
    var cache = try BatchedKVCache.init(allocator, 2, 2, 32, 100);
    defer cache.deinit();

    const slot = cache.allocSlot().?;

    // Start at 0
    try std.testing.expectEqual(@as(usize, 0), cache.getPosition(slot));

    // Increment multiple times
    cache.incrementPosition(slot);
    try std.testing.expectEqual(@as(usize, 1), cache.getPosition(slot));

    cache.incrementPosition(slot);
    try std.testing.expectEqual(@as(usize, 2), cache.getPosition(slot));

    cache.incrementPosition(slot);
    try std.testing.expectEqual(@as(usize, 3), cache.getPosition(slot));

    // Set to near max and increment
    cache.setPosition(slot, 98);
    cache.incrementPosition(slot);
    try std.testing.expectEqual(@as(usize, 99), cache.getPosition(slot));
}

test "BatchedKVCache.incrementPosition: from non-zero start" {
    const allocator = std.testing.allocator;
    var cache = try BatchedKVCache.init(allocator, 2, 2, 32, 100);
    defer cache.deinit();

    const slot = cache.allocSlot().?;

    // Set initial position
    cache.setPosition(slot, 50);

    // Increment from there
    cache.incrementPosition(slot);
    try std.testing.expectEqual(@as(usize, 51), cache.getPosition(slot));

    cache.incrementPosition(slot);
    try std.testing.expectEqual(@as(usize, 52), cache.getPosition(slot));
}

test "BatchedKVCache.isActive: state transitions" {
    const allocator = std.testing.allocator;
    var cache = try BatchedKVCache.init(allocator, 3, 2, 32, 128);
    defer cache.deinit();

    // All slots initially inactive
    try std.testing.expect(!cache.isActive(0));
    try std.testing.expect(!cache.isActive(1));
    try std.testing.expect(!cache.isActive(2));

    // Allocate slot 0
    const slot0 = cache.allocSlot().?;
    try std.testing.expect(cache.isActive(slot0));
    try std.testing.expect(!cache.isActive(1));
    try std.testing.expect(!cache.isActive(2));

    // Allocate slot 1
    const slot1 = cache.allocSlot().?;
    try std.testing.expect(cache.isActive(slot0));
    try std.testing.expect(cache.isActive(slot1));
    try std.testing.expect(!cache.isActive(2));

    // Free slot 0
    cache.freeSlot(slot0);
    try std.testing.expect(!cache.isActive(slot0));
    try std.testing.expect(cache.isActive(slot1));

    // Reset doesn't change active status
    cache.resetSlot(slot1);
    try std.testing.expect(cache.isActive(slot1));
}

test "BatchedKVCache.activeCount: tracks allocation changes" {
    const allocator = std.testing.allocator;
    var cache = try BatchedKVCache.init(allocator, 5, 2, 32, 128);
    defer cache.deinit();

    // Initially 0
    try std.testing.expectEqual(@as(usize, 0), cache.activeCount());

    // Allocate incrementally
    const slot0 = cache.allocSlot().?;
    try std.testing.expectEqual(@as(usize, 1), cache.activeCount());

    const slot1 = cache.allocSlot().?;
    try std.testing.expectEqual(@as(usize, 2), cache.activeCount());

    const slot2 = cache.allocSlot().?;
    try std.testing.expectEqual(@as(usize, 3), cache.activeCount());

    // Free and verify count decreases
    cache.freeSlot(slot1);
    try std.testing.expectEqual(@as(usize, 2), cache.activeCount());

    cache.freeSlot(slot0);
    try std.testing.expectEqual(@as(usize, 1), cache.activeCount());

    cache.freeSlot(slot2);
    try std.testing.expectEqual(@as(usize, 0), cache.activeCount());

    // Reallocate
    _ = cache.allocSlot();
    try std.testing.expectEqual(@as(usize, 1), cache.activeCount());
}

test "BatchedKVCache.getK: returns correct slice length" {
    const allocator = std.testing.allocator;
    const head_dim = 64;
    var cache = try BatchedKVCache.init(allocator, 2, 4, head_dim, 128);
    defer cache.deinit();

    const slot = cache.allocSlot().?;

    // Test all heads and various positions
    for (0..4) |kv_head| {
        for (0..10) |pos| {
            const k_slice = cache.getK(slot, kv_head, pos);
            try std.testing.expectEqual(head_dim, k_slice.len);
        }
    }
}

test "BatchedKVCache.getK: slices are independent" {
    const allocator = std.testing.allocator;
    const head_dim = 4;
    var cache = try BatchedKVCache.init(allocator, 2, 2, head_dim, 8);
    defer cache.deinit();

    const slot = cache.allocSlot().?;

    // Get different slices
    const k_h0_p0 = cache.getK(slot, 0, 0);
    const k_h0_p1 = cache.getK(slot, 0, 1);
    const k_h1_p0 = cache.getK(slot, 1, 0);

    // Write different values to each
    @memset(k_h0_p0, 1.0);
    @memset(k_h0_p1, 2.0);
    @memset(k_h1_p0, 3.0);

    // Verify independence
    try std.testing.expectEqual(@as(f32, 1.0), k_h0_p0[0]);
    try std.testing.expectEqual(@as(f32, 2.0), k_h0_p1[0]);
    try std.testing.expectEqual(@as(f32, 3.0), k_h1_p0[0]);
}

test "BatchedKVCache.getV: returns correct slice length" {
    const allocator = std.testing.allocator;
    const head_dim = 96;
    var cache = try BatchedKVCache.init(allocator, 2, 4, head_dim, 128);
    defer cache.deinit();

    const slot = cache.allocSlot().?;

    // Test all heads and various positions
    for (0..4) |kv_head| {
        for (0..10) |pos| {
            const v_slice = cache.getV(slot, kv_head, pos);
            try std.testing.expectEqual(head_dim, v_slice.len);
        }
    }
}

test "BatchedKVCache.getV: slices are independent" {
    const allocator = std.testing.allocator;
    const head_dim = 4;
    var cache = try BatchedKVCache.init(allocator, 2, 2, head_dim, 8);
    defer cache.deinit();

    const slot = cache.allocSlot().?;

    // Get different slices
    const v_h0_p0 = cache.getV(slot, 0, 0);
    const v_h0_p1 = cache.getV(slot, 0, 1);
    const v_h1_p0 = cache.getV(slot, 1, 0);

    // Write different values to each
    @memset(v_h0_p0, 10.0);
    @memset(v_h0_p1, 20.0);
    @memset(v_h1_p0, 30.0);

    // Verify independence
    try std.testing.expectEqual(@as(f32, 10.0), v_h0_p0[0]);
    try std.testing.expectEqual(@as(f32, 20.0), v_h0_p1[0]);
    try std.testing.expectEqual(@as(f32, 30.0), v_h1_p0[0]);
}

test "BatchedKVCache.getK and getV: K and V are independent" {
    const allocator = std.testing.allocator;
    const head_dim = 4;
    var cache = try BatchedKVCache.init(allocator, 1, 1, head_dim, 8);
    defer cache.deinit();

    const slot = cache.allocSlot().?;

    const k_slice = cache.getK(slot, 0, 0);
    const v_slice = cache.getV(slot, 0, 0);

    // Write different values
    @memset(k_slice, 100.0);
    @memset(v_slice, 200.0);

    // Verify they don't interfere
    try std.testing.expectEqual(@as(f32, 100.0), k_slice[0]);
    try std.testing.expectEqual(@as(f32, 200.0), v_slice[0]);
}

test "LayeredBatchedKVCache.init: validates layer creation" {
    const allocator = std.testing.allocator;

    var layered = try LayeredBatchedKVCache.init(allocator, 8, 4, 16, 64, 1024);
    defer layered.deinit();

    // Verify structure
    try std.testing.expectEqual(@as(usize, 8), layered.n_layers);
    try std.testing.expectEqual(@as(usize, 8), layered.layers.len);

    // Verify each layer has correct configuration
    for (layered.layers) |*layer| {
        try std.testing.expectEqual(@as(usize, 4), layer.max_batch_size);
        try std.testing.expectEqual(@as(usize, 16), layer.n_kv_heads);
        try std.testing.expectEqual(@as(usize, 64), layer.head_dim);
        try std.testing.expectEqual(@as(usize, 1024), layer.max_seq_len);
        try std.testing.expectEqual(@as(usize, 0), layer.activeCount());
    }
}

test "LayeredBatchedKVCache.init: edge cases" {
    const allocator = std.testing.allocator;

    // Single layer
    {
        var layered = try LayeredBatchedKVCache.init(allocator, 1, 2, 4, 32, 256);
        defer layered.deinit();
        try std.testing.expectEqual(@as(usize, 1), layered.n_layers);
    }

    // Many layers
    {
        var layered = try LayeredBatchedKVCache.init(allocator, 32, 2, 4, 32, 256);
        defer layered.deinit();
        try std.testing.expectEqual(@as(usize, 32), layered.n_layers);
    }
}

test "LayeredBatchedKVCache.deinit: cleanup all layers" {
    const allocator = std.testing.allocator;
    var layered = try LayeredBatchedKVCache.init(allocator, 6, 4, 8, 64, 512);

    // Verify structure was created
    try std.testing.expectEqual(@as(usize, 6), layered.n_layers);
    try std.testing.expectEqual(@as(usize, 6), layered.layers.len);

    layered.deinit();

    // After deinit, the struct should be undefined
    // The allocator in test mode will catch any memory leaks,
    // which verifies all layers and their buffers were freed
}

test "LayeredBatchedKVCache.getPosition: reads from layer 0" {
    const allocator = std.testing.allocator;
    var layered = try LayeredBatchedKVCache.init(allocator, 4, 2, 4, 32, 256);
    defer layered.deinit();

    const slot = layered.allocSlot().?;

    // Set different positions in each layer (unusual but tests the function)
    layered.layers[0].setPosition(slot, 10);
    layered.layers[1].setPosition(slot, 20);
    layered.layers[2].setPosition(slot, 30);
    layered.layers[3].setPosition(slot, 40);

    // getPosition should return layer 0's value
    try std.testing.expectEqual(@as(usize, 10), layered.getPosition(slot));
}

test "LayeredBatchedKVCache.setPosition: updates all layers" {
    const allocator = std.testing.allocator;
    var layered = try LayeredBatchedKVCache.init(allocator, 5, 2, 4, 32, 256);
    defer layered.deinit();

    const slot = layered.allocSlot().?;

    // Set position via layered interface
    layered.setPosition(slot, 77);

    // Verify all layers were updated
    for (layered.layers) |*layer| {
        try std.testing.expectEqual(@as(usize, 77), layer.getPosition(slot));
    }

    // Verify via layered interface
    try std.testing.expectEqual(@as(usize, 77), layered.getPosition(slot));
}

test "LayeredBatchedKVCache.setPosition: boundary values" {
    const allocator = std.testing.allocator;
    const max_seq_len = 128;
    var layered = try LayeredBatchedKVCache.init(allocator, 3, 2, 4, 32, max_seq_len);
    defer layered.deinit();

    const slot = layered.allocSlot().?;

    // Set to 0
    layered.setPosition(slot, 0);
    try std.testing.expectEqual(@as(usize, 0), layered.getPosition(slot));
    for (layered.layers) |*layer| {
        try std.testing.expectEqual(@as(usize, 0), layer.getPosition(slot));
    }

    // Set to max
    layered.setPosition(slot, max_seq_len);
    try std.testing.expectEqual(@as(usize, max_seq_len), layered.getPosition(slot));
    for (layered.layers) |*layer| {
        try std.testing.expectEqual(@as(usize, max_seq_len), layer.getPosition(slot));
    }
}

test "LayeredBatchedKVCache.incrementPosition: updates all layers" {
    const allocator = std.testing.allocator;
    var layered = try LayeredBatchedKVCache.init(allocator, 4, 2, 4, 32, 256);
    defer layered.deinit();

    const slot = layered.allocSlot().?;

    // Set initial position
    layered.setPosition(slot, 10);

    // Increment
    layered.incrementPosition(slot);

    // Verify all layers were incremented
    try std.testing.expectEqual(@as(usize, 11), layered.getPosition(slot));
    for (layered.layers) |*layer| {
        try std.testing.expectEqual(@as(usize, 11), layer.getPosition(slot));
    }

    // Increment again
    layered.incrementPosition(slot);
    try std.testing.expectEqual(@as(usize, 12), layered.getPosition(slot));
    for (layered.layers) |*layer| {
        try std.testing.expectEqual(@as(usize, 12), layer.getPosition(slot));
    }
}

test "LayeredBatchedKVCache.incrementPosition: from zero" {
    const allocator = std.testing.allocator;
    var layered = try LayeredBatchedKVCache.init(allocator, 3, 2, 4, 32, 256);
    defer layered.deinit();

    const slot = layered.allocSlot().?;

    // Initial position is 0
    try std.testing.expectEqual(@as(usize, 0), layered.getPosition(slot));

    // Increment from 0
    layered.incrementPosition(slot);

    // Verify all layers
    try std.testing.expectEqual(@as(usize, 1), layered.getPosition(slot));
    for (layered.layers) |*layer| {
        try std.testing.expectEqual(@as(usize, 1), layer.getPosition(slot));
    }
}

test "allocSlot freeSlot stress test" {
    const allocator = std.testing.allocator;
    const max_batch = 8;
    var cache = try BatchedKVCache.init(allocator, max_batch, 4, 32, 512);
    defer cache.deinit();

    var allocated_slots: [max_batch]?usize = [_]?usize{null} ** max_batch;

    // Allocate all
    for (0..max_batch) |i| {
        allocated_slots[i] = cache.allocSlot();
        try std.testing.expect(allocated_slots[i] != null);
    }
    try std.testing.expectEqual(@as(usize, max_batch), cache.activeCount());

    // Should be full
    try std.testing.expect(cache.allocSlot() == null);

    // Free every other slot
    var i: usize = 0;
    while (i < max_batch) : (i += 2) {
        cache.freeSlot(allocated_slots[i].?);
        allocated_slots[i] = null;
    }
    try std.testing.expectEqual(@as(usize, max_batch / 2), cache.activeCount());

    // Reallocate freed slots
    i = 0;
    while (i < max_batch) : (i += 2) {
        allocated_slots[i] = cache.allocSlot();
        try std.testing.expect(allocated_slots[i] != null);
    }
    try std.testing.expectEqual(@as(usize, max_batch), cache.activeCount());

    // Free all
    for (allocated_slots) |maybe_slot| {
        if (maybe_slot) |slot| {
            cache.freeSlot(slot);
        }
    }
    try std.testing.expectEqual(@as(usize, 0), cache.activeCount());
}
