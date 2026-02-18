//! Tensor gather/scatter primitives for CPU kernels.

const std = @import("std");

/// Gather rows from a contiguous `[row_count, row_width]` f32 matrix.
pub fn gatherRowsF32(
    src_rows: []const f32,
    row_count: usize,
    row_width: usize,
    row_indices: []const u32,
    dst_rows: []f32,
) !void {
    if (src_rows.len < row_count * row_width) return error.InvalidShape;
    if (dst_rows.len < row_indices.len * row_width) return error.InvalidShape;

    for (row_indices, 0..) |row_idx_u32, out_row_idx| {
        const row_idx: usize = @intCast(row_idx_u32);
        if (row_idx >= row_count) return error.InvalidTokenId;
        const src_row = src_rows[row_idx * row_width ..][0..row_width];
        const dst_row = dst_rows[out_row_idx * row_width ..][0..row_width];
        @memcpy(dst_row, src_row);
    }
}

/// Add feature rows into destination rows selected by token positions.
pub fn scatterAddRowsByPositions(
    dst_rows: []f32,
    dst_row_count: usize,
    row_width: usize,
    positions: []const usize,
    src_rows: []const f32,
) !void {
    if (dst_rows.len < dst_row_count * row_width) return error.InvalidShape;
    if (src_rows.len % row_width != 0) return error.InvalidShape;

    const available_rows = src_rows.len / row_width;
    const row_count = @min(positions.len, available_rows);
    for (0..row_count) |row_idx| {
        const token_pos = positions[row_idx];
        if (token_pos >= dst_row_count) continue;
        const dst = dst_rows[token_pos * row_width ..][0..row_width];
        const src = src_rows[row_idx * row_width ..][0..row_width];
        for (dst, src) |*d, s| d.* += s;
    }
}

test "gatherRowsF32 copies rows by index" {
    const src = [_]f32{
        1,   2,   3,
        10,  20,  30,
        100, 200, 300,
    };
    const idx = [_]u32{ 2, 0 };
    var out = [_]f32{0} ** 6;

    try gatherRowsF32(&src, 3, 3, &idx, &out);

    try std.testing.expectEqualSlices(f32, &[_]f32{ 100, 200, 300, 1, 2, 3 }, &out);
}

test "gatherRowsF32 returns InvalidTokenId for OOB index" {
    const src = [_]f32{ 1, 2, 3, 4 };
    const idx = [_]u32{ 0, 3 };
    var out = [_]f32{0} ** 4;

    try std.testing.expectError(error.InvalidTokenId, gatherRowsF32(&src, 2, 2, &idx, &out));
}
