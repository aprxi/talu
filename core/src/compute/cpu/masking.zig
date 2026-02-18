//! Masking primitives for CPU compute path.

const std = @import("std");

/// Apply upper-triangular mask.
///
/// Writes `out_data` from `input_data`, setting elements below the selected
/// diagonal to `-inf`.
pub fn triu(
    input_data: []const f32,
    out_data: []f32,
    rows: usize,
    cols: usize,
    diagonal: i64,
) void {
    const neg_inf = -std.math.inf(f32);
    for (0..rows) |row| {
        for (0..cols) |col| {
            const elem_offset = row * cols + col;
            const signed_col: i64 = @intCast(col);
            const signed_row: i64 = @intCast(row);
            if (signed_col < signed_row + diagonal) {
                out_data[elem_offset] = neg_inf;
            } else {
                out_data[elem_offset] = input_data[elem_offset];
            }
        }
    }
}

