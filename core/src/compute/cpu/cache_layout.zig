//! Cache indexing primitives for CPU kernels.

const std = @import("std");

/// Compute flat offset for `[slot, kv_head, position, head_dim]` layout.
pub fn kvOffset(
    slot_index: usize,
    kv_head: usize,
    position: usize,
    max_batch_size: usize,
    n_kv_heads: usize,
    max_seq_len: usize,
    slot_stride: usize,
    head_stride: usize,
    head_dim: usize,
) usize {
    std.debug.assert(slot_index < max_batch_size);
    std.debug.assert(kv_head < n_kv_heads);
    std.debug.assert(position < max_seq_len);
    return slot_index * slot_stride + kv_head * head_stride + position * head_dim;
}

/// Apply signed delta to an unsigned position.
pub fn offsetPosition(base_pos: usize, delta: isize) !usize {
    if (delta == 0) return base_pos;
    const shifted: i64 = @as(i64, @intCast(base_pos)) + @as(i64, delta);
    if (shifted < 0) return error.InvalidShape;
    return @as(usize, @intCast(shifted));
}

test "kvOffset computes expected flat index" {
    const off = kvOffset(
        2, // slot
        3, // head
        5, // pos
        8, // max_batch
        4, // n_kv_heads
        16, // max_seq
        256, // slot_stride
        64, // head_stride
        4, // head_dim
    );
    try std.testing.expectEqual(@as(usize, 2 * 256 + 3 * 64 + 5 * 4), off);
}

test "offsetPosition rejects negative target" {
    try std.testing.expectError(error.InvalidShape, offsetPosition(3, -4));
}
