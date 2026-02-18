//! SIMD-friendly dot-product helpers for f32 vectors.
//!
//! Complexity: O(n) time, O(1) extra space.
//! Alignment: SIMD path requires both slices to be aligned to @alignOf(@Vector(8, f32)).

const std = @import("std");

/// Compute dot product of two f32 slices.
pub fn dotProductF32(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);

    const lane_count: usize = 8;
    const Vec = @Vector(lane_count, f32);
    const vec_align = @alignOf(Vec);

    var sum_vec: Vec = @splat(0.0);
    var idx: usize = 0;
    const len = a.len;
    const vec_len = len / lane_count;

    const a_aligned = (@intFromPtr(a.ptr) % vec_align) == 0;
    const b_aligned = (@intFromPtr(b.ptr) % vec_align) == 0;

    if (a_aligned and b_aligned and vec_len > 0) {
        const a_vec: [*]const Vec = @ptrCast(@alignCast(a.ptr));
        const b_vec: [*]const Vec = @ptrCast(@alignCast(b.ptr));
        var vec_idx: usize = 0;
        while (vec_idx < vec_len) : (vec_idx += 1) {
            sum_vec += a_vec[vec_idx] * b_vec[vec_idx];
        }
        idx = vec_len * lane_count;
    }

    var sum: f32 = @reduce(.Add, sum_vec);
    while (idx < len) : (idx += 1) {
        sum += a[idx] * b[idx];
    }
    return sum;
}

/// Compute dot product for four query vectors against the same input vector.
pub fn dotProductF32Batch4(q0: []const f32, q1: []const f32, q2: []const f32, q3: []const f32, b: []const f32) [4]f32 {
    std.debug.assert(q0.len == b.len);
    std.debug.assert(q1.len == b.len);
    std.debug.assert(q2.len == b.len);
    std.debug.assert(q3.len == b.len);

    const lane_count: usize = 8;
    const Vec = @Vector(lane_count, f32);
    const vec_align = @alignOf(Vec);

    var acc0: Vec = @splat(0.0);
    var acc1: Vec = @splat(0.0);
    var acc2: Vec = @splat(0.0);
    var acc3: Vec = @splat(0.0);

    var idx: usize = 0;
    const len = b.len;
    const vec_len = len / lane_count;

    const aligned = (@intFromPtr(b.ptr) % vec_align) == 0 and
        (@intFromPtr(q0.ptr) % vec_align) == 0 and
        (@intFromPtr(q1.ptr) % vec_align) == 0 and
        (@intFromPtr(q2.ptr) % vec_align) == 0 and
        (@intFromPtr(q3.ptr) % vec_align) == 0;

    if (aligned and vec_len > 0) {
        const b_vec: [*]const Vec = @ptrCast(@alignCast(b.ptr));
        const q0_vec: [*]const Vec = @ptrCast(@alignCast(q0.ptr));
        const q1_vec: [*]const Vec = @ptrCast(@alignCast(q1.ptr));
        const q2_vec: [*]const Vec = @ptrCast(@alignCast(q2.ptr));
        const q3_vec: [*]const Vec = @ptrCast(@alignCast(q3.ptr));
        var vec_idx: usize = 0;
        while (vec_idx < vec_len) : (vec_idx += 1) {
            const b_vals = b_vec[vec_idx];
            acc0 += q0_vec[vec_idx] * b_vals;
            acc1 += q1_vec[vec_idx] * b_vals;
            acc2 += q2_vec[vec_idx] * b_vals;
            acc3 += q3_vec[vec_idx] * b_vals;
        }
        idx = vec_len * lane_count;
    }

    var s0: f32 = @reduce(.Add, acc0);
    var s1: f32 = @reduce(.Add, acc1);
    var s2: f32 = @reduce(.Add, acc2);
    var s3: f32 = @reduce(.Add, acc3);
    while (idx < len) : (idx += 1) {
        const b_val = b[idx];
        s0 += q0[idx] * b_val;
        s1 += q1[idx] * b_val;
        s2 += q2[idx] * b_val;
        s3 += q3[idx] * b_val;
    }

    return .{ s0, s1, s2, s3 };
}

fn dotProductScalar(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var sum: f32 = 0.0;
    for (a, 0..) |value, idx| {
        sum += value * b[idx];
    }
    return sum;
}

test "dotProductF32 matches scalar reference" {
    const input = [_]f32{ 0.5, -1.0, 2.0, 3.5, -0.25, 1.25, 0.0, 4.0, 1.5 };
    const weights = [_]f32{ 1.0, 0.5, -2.0, 0.25, 4.0, -0.75, 1.0, 0.1, -3.0 };

    const scalar = dotProductScalar(&input, &weights);
    const simd = dotProductF32(&input, &weights);

    try std.testing.expectApproxEqAbs(scalar, simd, 1e-5);
}

test "dotProductF32Batch4 matches scalar reference" {
    const input = [_]f32{ 0.25, -0.5, 1.0, 2.0, 0.0, -1.5, 3.0, 0.75, 2.25 };
    const q0 = [_]f32{ 1.0, 0.5, -2.0, 0.25, 4.0, -0.75, 1.0, 0.1, -3.0 };
    const q1 = [_]f32{ -0.25, 1.0, 0.5, 0.0, -1.0, 2.0, 0.5, 1.5, -0.5 };
    const q2 = [_]f32{ 0.0, 0.0, 0.25, -1.0, 2.0, 0.5, 1.0, -2.0, 3.0 };
    const q3 = [_]f32{ 2.0, -1.0, 0.0, 0.75, 1.5, -0.25, -1.0, 0.0, 0.5 };

    const ref0 = dotProductScalar(&q0, &input);
    const ref1 = dotProductScalar(&q1, &input);
    const ref2 = dotProductScalar(&q2, &input);
    const ref3 = dotProductScalar(&q3, &input);

    const simd = dotProductF32Batch4(&q0, &q1, &q2, &q3, &input);

    try std.testing.expectApproxEqAbs(ref0, simd[0], 1e-5);
    try std.testing.expectApproxEqAbs(ref1, simd[1], 1e-5);
    try std.testing.expectApproxEqAbs(ref2, simd[2], 1e-5);
    try std.testing.expectApproxEqAbs(ref3, simd[3], 1e-5);
}
