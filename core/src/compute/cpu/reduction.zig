//! Reduction primitives used by CPU kernels.

const std = @import("std");
const simd = @import("simd/arch/root.zig");

const VEC_LEN = simd.f32_vec_len;
const F32Vec = simd.F32Vec;

/// Dot product between equal-length f32 rows.
pub fn dotRow(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var sum0: F32Vec = @splat(0);
    var sum1: F32Vec = @splat(0);
    var idx: usize = 0;
    while (idx + 2 * VEC_LEN - 1 < a.len) : (idx += 2 * VEC_LEN) {
        const a0: F32Vec = a[idx..][0..VEC_LEN].*;
        const b0: F32Vec = b[idx..][0..VEC_LEN].*;
        const a1: F32Vec = a[idx + VEC_LEN ..][0..VEC_LEN].*;
        const b1: F32Vec = b[idx + VEC_LEN ..][0..VEC_LEN].*;
        sum0 = @mulAdd(F32Vec, a0, b0, sum0);
        sum1 = @mulAdd(F32Vec, a1, b1, sum1);
    }
    while (idx + VEC_LEN - 1 < a.len) : (idx += VEC_LEN) {
        const av: F32Vec = a[idx..][0..VEC_LEN].*;
        const bv: F32Vec = b[idx..][0..VEC_LEN].*;
        sum0 = @mulAdd(F32Vec, av, bv, sum0);
    }

    var sum = @reduce(.Add, sum0 + sum1);
    while (idx < a.len) : (idx += 1) {
        sum += a[idx] * b[idx];
    }
    return sum;
}

/// `out += weight * values` with SIMD acceleration.
pub fn weightedAccumulateRow(out: []f32, values: []const f32, weight: f32) void {
    std.debug.assert(out.len == values.len);

    const weight_vec: F32Vec = @splat(weight);
    var idx: usize = 0;
    while (idx + VEC_LEN - 1 < out.len) : (idx += VEC_LEN) {
        const vv: F32Vec = values[idx..][0..VEC_LEN].*;
        const dst = out[idx..][0..VEC_LEN];
        dst.* = @mulAdd(F32Vec, weight_vec, vv, dst.*);
    }
    while (idx < out.len) : (idx += 1) {
        out[idx] += weight * values[idx];
    }
}

/// Compute `(a0·b0 + a1·b1) * scale`.
pub fn dotPairScaled(
    a0: []const f32,
    b0: []const f32,
    a1: []const f32,
    b1: []const f32,
    scale: f32,
) f32 {
    return (dotRow(a0, b0) + dotRow(a1, b1)) * scale;
}

/// Maximum value in a non-empty vector.
pub fn maxValue(values: []const f32) f32 {
    std.debug.assert(values.len > 0);
    var max_vec: F32Vec = @splat(values[0]);
    var idx: usize = 0;
    while (idx + VEC_LEN - 1 < values.len) : (idx += VEC_LEN) {
        const vv: F32Vec = values[idx..][0..VEC_LEN].*;
        max_vec = @max(max_vec, vv);
    }
    var max_value = @reduce(.Max, max_vec);
    while (idx < values.len) : (idx += 1) {
        max_value = @max(max_value, values[idx]);
    }
    return max_value;
}

/// Argmax index for a non-empty vector.
pub fn argmaxIndex(values: []const f32) usize {
    std.debug.assert(values.len > 0);
    var best_idx: usize = 0;
    var best_val: f32 = values[0];
    for (values[1..], 1..) |v, i| {
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }
    return best_idx;
}

/// Mean pool rows from contiguous `[row_count, row_width]` into `out`.
pub fn meanPoolRows(values: []const f32, row_count: usize, row_width: usize, out: []f32) !void {
    if (values.len < row_count * row_width) return error.InvalidShape;
    if (out.len < row_width) return error.InvalidShape;
    @memset(out[0..row_width], 0.0);

    for (0..row_count) |row_idx| {
        const row = values[row_idx * row_width ..][0..row_width];
        for (0..row_width) |col_idx| {
            out[col_idx] += row[col_idx];
        }
    }

    const scale = 1.0 / @as(f32, @floatFromInt(row_count));
    for (0..row_width) |col_idx| {
        out[col_idx] *= scale;
    }
}

/// L2-normalize one vector in-place. No-op for zero norm.
pub fn l2NormalizeInPlace(values: []f32) void {
    var norm_sq: f32 = 0.0;
    for (values) |v| norm_sq += v * v;
    if (norm_sq <= 0.0) return;
    const inv_norm = 1.0 / @sqrt(norm_sq);
    for (values) |*v| v.* *= inv_norm;
}

/// Mean over last dim for shape `[seq_len, hidden_size]`.
pub fn meanLastDim3D(input_data: []const f32, seq_len: usize, hidden_size: usize, output: []f32) !void {
    if (input_data.len < seq_len * hidden_size) return error.InvalidShape;
    if (output.len < seq_len) return error.InvalidShape;
    for (0..seq_len) |t| {
        const base = t * hidden_size;
        var sum: f32 = 0.0;
        for (0..hidden_size) |h| sum += input_data[base + h];
        output[t] = sum / @as(f32, @floatFromInt(hidden_size));
    }
}

/// Mean over last dim for shape `[seq_len, head_count, hidden_size]`.
pub fn meanLastDim4D(input_data: []const f32, seq_len: usize, head_count: usize, hidden_size: usize, output: []f32) !void {
    if (input_data.len < seq_len * head_count * hidden_size) return error.InvalidShape;
    if (output.len < seq_len * head_count) return error.InvalidShape;
    for (0..seq_len) |t| {
        for (0..head_count) |h| {
            const base = (t * head_count + h) * hidden_size;
            var sum: f32 = 0.0;
            for (0..hidden_size) |d| sum += input_data[base + d];
            output[t * head_count + h] = sum / @as(f32, @floatFromInt(hidden_size));
        }
    }
}

test "dotRow computes expected value" {
    const a = [_]f32{ 1, 2, 3 };
    const b = [_]f32{ 4, 5, 6 };
    try std.testing.expectApproxEqAbs(@as(f32, 32.0), dotRow(&a, &b), 1e-6);
}

test "weightedAccumulateRow updates output in place" {
    var out = [_]f32{ 1, 2, 3, 4 };
    const values = [_]f32{ 10, 20, 30, 40 };
    weightedAccumulateRow(&out, &values, 0.5);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 6, 12, 18, 24 }, &out);
}

test "dotPairScaled combines two dot products and scale" {
    const a0 = [_]f32{ 1, 2 };
    const b0 = [_]f32{ 3, 4 };
    const a1 = [_]f32{ 5, 6 };
    const b1 = [_]f32{ 7, 8 };
    const got = dotPairScaled(&a0, &b0, &a1, &b1, 0.5);
    // (11 + 83) * 0.5 = 47
    try std.testing.expectApproxEqAbs(@as(f32, 47.0), got, 1e-6);
}
