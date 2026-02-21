//! Row-wise and element-wise CPU primitives for inference backends.

const std = @import("std");
const simd = @import("simd/arch/root.zig");

const VEC_LEN = simd.f32_vec_len;
const F32Vec = simd.F32Vec;

/// In-place row add: `dst += src * scale`.
pub fn addScaledInPlace(dst: []f32, src: []const f32, scale: f32) void {
    std.debug.assert(dst.len == src.len);
    if (dst.len == 0) return;

    if (scale == 1.0) {
        var index: usize = 0;
        while (index + VEC_LEN - 1 < dst.len) : (index += VEC_LEN) {
            const d: F32Vec = dst[index..][0..VEC_LEN].*;
            const s: F32Vec = src[index..][0..VEC_LEN].*;
            dst[index..][0..VEC_LEN].* = d + s;
        }
        while (index < dst.len) : (index += 1) {
            dst[index] += src[index];
        }
        return;
    }

    const scale_vec: F32Vec = @splat(scale);
    var index: usize = 0;
    while (index + VEC_LEN - 1 < dst.len) : (index += VEC_LEN) {
        const d: F32Vec = dst[index..][0..VEC_LEN].*;
        const s: F32Vec = src[index..][0..VEC_LEN].*;
        dst[index..][0..VEC_LEN].* = @mulAdd(F32Vec, s, scale_vec, d);
    }
    while (index < dst.len) : (index += 1) {
        dst[index] += src[index] * scale;
    }
}

/// In-place multiply: `values *= scale`.
pub fn scaleInPlace(values: []f32, scale: f32) void {
    if (scale == 1.0) return;
    const scale_vec: F32Vec = @splat(scale);
    var index: usize = 0;
    while (index + VEC_LEN - 1 < values.len) : (index += VEC_LEN) {
        const v: F32Vec = values[index..][0..VEC_LEN].*;
        values[index..][0..VEC_LEN].* = v * scale_vec;
    }
    while (index < values.len) : (index += 1) {
        values[index] *= scale;
    }
}

/// In-place divide using reciprocal multiply: `values /= divisor`.
pub fn scaleInPlaceReciprocal(values: []f32, divisor: f32) void {
    if (divisor == 1.0) return;
    const recip = 1.0 / divisor;
    scaleInPlace(values, recip);
}

/// Element-wise add into output: `out = a + b`.
pub fn addInto(a: []const f32, b: []const f32, out: []f32) void {
    std.debug.assert(a.len == b.len and a.len == out.len);

    var index: usize = 0;
    while (index + VEC_LEN - 1 < out.len) : (index += VEC_LEN) {
        const va: F32Vec = a[index..][0..VEC_LEN].*;
        const vb: F32Vec = b[index..][0..VEC_LEN].*;
        out[index..][0..VEC_LEN].* = va + vb;
    }
    while (index < out.len) : (index += 1) {
        out[index] = a[index] + b[index];
    }
}

/// Element-wise scaled add into output: `out = a + b * scale`.
pub fn addIntoScaled(a: []const f32, b: []const f32, out: []f32, scale: f32) void {
    std.debug.assert(a.len == b.len and a.len == out.len);
    if (scale == 1.0) return addInto(a, b, out);

    const scale_vec: F32Vec = @splat(scale);
    var index: usize = 0;
    while (index + VEC_LEN - 1 < out.len) : (index += VEC_LEN) {
        const va: F32Vec = a[index..][0..VEC_LEN].*;
        const vb: F32Vec = b[index..][0..VEC_LEN].*;
        out[index..][0..VEC_LEN].* = @mulAdd(F32Vec, vb, scale_vec, va);
    }
    while (index < out.len) : (index += 1) {
        out[index] = a[index] + b[index] * scale;
    }
}

/// Add rows from source matrix into destination matrix in-place.
///
/// Destination and source are contiguous row-major buffers.
pub fn addRowsInPlace(
    dst: []f32,
    src: []const f32,
    row_count: usize,
    row_width: usize,
    src_row_stride: usize,
) !void {
    if (dst.len < row_count * row_width) return error.InvalidShape;
    if (src.len < row_count * src_row_stride) return error.InvalidShape;

    for (0..row_count) |row_idx| {
        const dst_row = dst[row_idx * row_width ..][0..row_width];
        const src_row = src[row_idx * src_row_stride ..][0..row_width];
        addScaledInPlace(dst_row, src_row, 1.0);
    }
}

/// Add one source row to every destination row in-place.
pub fn addBroadcastRowInPlace(
    dst: []f32,
    row_count: usize,
    row_width: usize,
    src_row: []const f32,
) !void {
    if (dst.len < row_count * row_width) return error.InvalidShape;
    if (src_row.len < row_width) return error.InvalidShape;
    for (0..row_count) |row_idx| {
        const dst_row = dst[row_idx * row_width ..][0..row_width];
        addScaledInPlace(dst_row, src_row[0..row_width], 1.0);
    }
}

test "addScaledInPlace with scale 1.0" {
    var dst = [_]f32{ 1, 2, 3, 4 };
    const src = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    addScaledInPlace(&dst, &src, 1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), dst[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.5), dst[3], 1e-6);
}

test "addIntoScaled applies scale" {
    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 2, 2, 2, 2 };
    var out = [_]f32{ 0, 0, 0, 0 };
    addIntoScaled(&a, &b, &out, 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 2), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5), out[3], 1e-6);
}

test "scaleInPlaceReciprocal divides values" {
    var values = [_]f32{ 2, 4, 8 };
    scaleInPlaceReciprocal(&values, 2.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1), values[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4), values[2], 1e-6);
}

test "scaleInPlace multiplies in place" {
    var values = [_]f32{ 1.0, -2.0, 3.0 };
    scaleInPlace(&values, 0.5);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 0.5, -1.0, 1.5 }, &values);
}

test "addInto sums vectors into output" {
    const a = [_]f32{ 1, 2, 3 };
    const b = [_]f32{ 4, 5, 6 };
    var out = [_]f32{ 0, 0, 0 };
    addInto(&a, &b, &out);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 5, 7, 9 }, &out);
}

test "addRowsInPlace adds row-wise with source stride" {
    var dst = [_]f32{
        1, 2,
        3, 4,
    };
    const src = [_]f32{
        10, 20, 99,
        30, 40, 88,
    };
    try addRowsInPlace(&dst, &src, 2, 2, 3);
    try std.testing.expectEqualSlices(f32, &[_]f32{
        11, 22,
        33, 44,
    }, &dst);
}

test "addBroadcastRowInPlace adds one row to all rows" {
    var dst = [_]f32{
        1, 2, 3,
        4, 5, 6,
    };
    const src_row = [_]f32{ 10, 20, 30 };
    try addBroadcastRowInPlace(&dst, 2, 3, &src_row);
    try std.testing.expectEqualSlices(f32, &[_]f32{
        11, 22, 33,
        14, 25, 36,
    }, &dst);
}
