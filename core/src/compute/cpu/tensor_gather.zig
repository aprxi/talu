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

/// Collect token positions matching `needle`.
///
/// Returns an owned slice; caller owns the result and must free it.
pub fn collectPositionsU32(
    allocator: std.mem.Allocator,
    tokens: []const u32,
    needle: u32,
) ![]usize {
    var count: usize = 0;
    for (tokens) |token| {
        if (token == needle) count += 1;
    }
    if (count == 0) return &.{};

    const positions = try allocator.alloc(usize, count);
    errdefer allocator.free(positions);

    var write_idx: usize = 0;
    for (tokens, 0..) |token, idx| {
        if (token != needle) continue;
        positions[write_idx] = idx;
        write_idx += 1;
    }
    std.debug.assert(write_idx == count);
    return positions;
}

/// Scatter source rows into destination rows selected by matched identifier.
pub fn scatterRowsByMatchedId(
    hidden_states: []f32,
    seq_len: usize,
    row_width: usize,
    token_ids: []const u32,
    token_id_match: u32,
    embeddings: []const f32,
) !void {
    if (hidden_states.len != seq_len * row_width) return error.InvalidShape;
    if (token_ids.len != seq_len) return error.InvalidShape;
    if (embeddings.len % row_width != 0) return error.InvalidShape;

    const embed_tokens = embeddings.len / row_width;
    var embed_idx: usize = 0;

    for (token_ids, 0..) |token_id, pos| {
        if (token_id != token_id_match) continue;
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
