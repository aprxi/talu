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

test "triu masks lower triangle below diagonal" {
    const input = [_]f32{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    };
    var out = [_]f32{0} ** 9;
    triu(&input, &out, 3, 3, 0);
    try std.testing.expect(std.math.isInf(out[3]) and out[3] < 0.0);
    try std.testing.expect(std.math.isInf(out[6]) and out[6] < 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), out[8], 1e-6);
}
