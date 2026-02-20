//! Generic indexing primitives.

const std = @import("std");

/// Compute flat offset for a 4D logical layout using explicit strides.
pub fn flatOffset4D(
    index0: usize,
    index1: usize,
    index2: usize,
    stride0: usize,
    stride1: usize,
    stride2: usize,
) usize {
    return index0 * stride0 + index1 * stride1 + index2 * stride2;
}

/// Apply signed delta to an unsigned base index.
pub fn offsetSigned(base: usize, delta: isize) !usize {
    if (delta == 0) return base;
    const shifted: i64 = @as(i64, @intCast(base)) + @as(i64, delta);
    if (shifted < 0) return error.InvalidShape;
    return @as(usize, @intCast(shifted));
}

test "flatOffset4D computes expected flat index" {
    const off = flatOffset4D(2, 3, 5, 256, 64, 4);
    try std.testing.expectEqual(@as(usize, 2 * 256 + 3 * 64 + 5 * 4), off);
}

test "offsetSigned rejects negative target" {
    try std.testing.expectError(error.InvalidShape, offsetSigned(3, -4));
}
