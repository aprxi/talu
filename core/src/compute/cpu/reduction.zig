//! Reduction primitives used by CPU kernels.

const std = @import("std");
const simd = @import("../simd/root.zig");

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
