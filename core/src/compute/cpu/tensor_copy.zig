//! Copy primitives for contiguous CPU buffers.

const std = @import("std");

/// Copy contiguous bytes from `src` into `dst`.
pub fn copyContiguous(dst: []u8, src: []const u8) void {
    std.debug.assert(dst.len == src.len);
    std.mem.copyForwards(u8, dst, src);
}

test "copyContiguous copies full slice" {
    var dst = [_]u8{ 0, 0, 0, 0 };
    const src = [_]u8{ 1, 2, 3, 4 };
    copyContiguous(&dst, &src);
    try std.testing.expectEqualSlices(u8, &src, &dst);
}
