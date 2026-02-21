//! Tensor gather/scatter primitives for CPU compute.

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

/// Add feature rows into destination rows selected by positions.
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
        const dst_row_idx = positions[row_idx];
        if (dst_row_idx >= dst_row_count) continue;
        const dst = dst_rows[dst_row_idx * row_width ..][0..row_width];
        const src = src_rows[row_idx * row_width ..][0..row_width];
        for (dst, src) |*d, s| d.* += s;
    }
}

/// Collect positions in `values` matching `needle`.
///
/// Returns an owned slice; caller owns the result and must free it.
pub fn collectPositionsU32(
    allocator: std.mem.Allocator,
    values: []const u32,
    needle: u32,
) ![]usize {
    var count: usize = 0;
    for (values) |value| {
        if (value == needle) count += 1;
    }
    if (count == 0) return &.{};

    const positions = try allocator.alloc(usize, count);
    errdefer allocator.free(positions);

    var write_idx: usize = 0;
    for (values, 0..) |value, idx| {
        if (value != needle) continue;
        positions[write_idx] = idx;
        write_idx += 1;
    }
    std.debug.assert(write_idx == count);
    return positions;
}

/// Scatter source rows into destination rows selected by matched identifier values.
pub fn scatterRowsByMatchedId(
    hidden_states: []f32,
    seq_len: usize,
    row_width: usize,
    row_ids: []const u32,
    row_id_match: u32,
    embeddings: []const f32,
) !void {
    if (hidden_states.len != seq_len * row_width) return error.InvalidShape;
    if (row_ids.len != seq_len) return error.InvalidShape;
    if (embeddings.len % row_width != 0) return error.InvalidShape;

    const embed_tokens = embeddings.len / row_width;
    var embed_idx: usize = 0;

    for (row_ids, 0..) |row_id, pos| {
        if (row_id != row_id_match) continue;
        if (embed_idx >= embed_tokens) return error.InvalidShape;
        const src = embeddings[embed_idx * row_width ..][0..row_width];
        const dst = hidden_states[pos * row_width ..][0..row_width];
        @memcpy(dst, src);
        embed_idx += 1;
    }

    if (embed_idx != embed_tokens) return error.InvalidShape;
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

test "scatterAddRowsByPositions accumulates selected rows" {
    var dst = [_]f32{
        1, 1,
        2, 2,
        3, 3,
    };
    const positions = [_]usize{ 2, 0 };
    const src = [_]f32{
        10, 20,
        30, 40,
    };
    try scatterAddRowsByPositions(&dst, 3, 2, &positions, &src);
    try std.testing.expectEqualSlices(f32, &[_]f32{
        31, 41,
        2,  2,
        13, 23,
    }, &dst);
}

test "collectPositionsU32 returns matching indexes" {
    const allocator = std.testing.allocator;
    const tokens = [_]u32{ 5, 9, 5, 7, 5 };
    const positions = try collectPositionsU32(allocator, &tokens, 5);
    defer if (positions.len > 0) allocator.free(positions);
    try std.testing.expectEqual(@as(usize, 3), positions.len);
    try std.testing.expectEqual(@as(usize, 0), positions[0]);
    try std.testing.expectEqual(@as(usize, 2), positions[1]);
    try std.testing.expectEqual(@as(usize, 4), positions[2]);
}

test "scatterRowsByMatchedId copies embeddings for matched ids" {
    var hidden = [_]f32{
        0, 0,
        0, 0,
        0, 0,
    };
    const row_ids = [_]u32{ 7, 1, 7 };
    const embeds = [_]f32{
        10, 11,
        20, 21,
    };
    try scatterRowsByMatchedId(&hidden, 3, 2, &row_ids, 7, &embeds);
    try std.testing.expectEqualSlices(f32, &[_]f32{
        10, 11,
        0,  0,
        20, 21,
    }, &hidden);
}
