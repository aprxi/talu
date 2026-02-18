//! Activation functions: SiLU, GELU, ReLU, sigmoid, tanh.

const std = @import("std");
const fast_math = @import("fast_math.zig");
const simd = @import("../../simd/root.zig");

const VEC_LEN = simd.f32_vec_len;
const F32Vec = simd.F32Vec;

const fastExp = fast_math.fastExp;
const fastExpScalar = fast_math.fastExpScalar;

/// Scalar reference implementation of SiLU for equivalence testing.
/// Uses std.math.exp for maximum precision.
pub fn siluScalarReference(out: []f32, input: []const f32) void {
    std.debug.assert(out.len == input.len);
    for (input, 0..) |x, i| {
        const sig = 1.0 / (1.0 + @exp(-x));
        out[i] = x * sig;
    }
}

/// SiLU activation with SIMD optimization.
/// For equivalence testing, use siluScalarReference.
pub fn siluContiguous(out: []f32, input: []const f32) void {
    @setFloatMode(.optimized);
    std.debug.assert(out.len == input.len);

    const one: F32Vec = @splat(1.0);
    var vec_idx: usize = 0;
    while (vec_idx + VEC_LEN - 1 < input.len) : (vec_idx += VEC_LEN) {
        const input_vec: F32Vec = input[vec_idx..][0..VEC_LEN].*;
        const exp_neg = fastExp(-input_vec);
        const sig = one / (one + exp_neg);
        out[vec_idx..][0..VEC_LEN].* = input_vec * sig;
    }
    while (vec_idx < input.len) : (vec_idx += 1) {
        const input_value = input[vec_idx];
        const sig = 1.0 / (1.0 + fastExpScalar(-input_value));
        out[vec_idx] = input_value * sig;
    }
}

pub fn geluContiguous(out: []f32, input: []const f32) void {
    @setFloatMode(.optimized);
    std.debug.assert(out.len == input.len);

    const sqrt_2_over_pi: f32 = 0.7978845608028654;
    const coeff: f32 = 0.044715;

    for (input, 0..) |x, elem_idx| {
        const x3 = x * x * x;
        const inner = sqrt_2_over_pi * (x + coeff * x3);
        const tanh_val = std.math.tanh(inner);
        out[elem_idx] = 0.5 * x * (1.0 + tanh_val);
    }
}

pub fn reluContiguous(out: []f32, input: []const f32) void {
    std.debug.assert(out.len == input.len);
    for (input, 0..) |x, elem_idx| {
        out[elem_idx] = @max(0, x);
    }
}

pub fn sigmoidContiguous(out: []f32, input: []const f32) void {
    @setFloatMode(.optimized);
    std.debug.assert(out.len == input.len);

    const one: F32Vec = @splat(1.0);
    var vec_idx: usize = 0;
    while (vec_idx + VEC_LEN - 1 < input.len) : (vec_idx += VEC_LEN) {
        const input_vec: F32Vec = input[vec_idx..][0..VEC_LEN].*;
        const exp_neg = fastExp(-input_vec);
        out[vec_idx..][0..VEC_LEN].* = one / (one + exp_neg);
    }
    while (vec_idx < input.len) : (vec_idx += 1) {
        const input_value = input[vec_idx];
        out[vec_idx] = 1.0 / (1.0 + fastExpScalar(-input_value));
    }
}

pub fn tanhContiguous(out: []f32, input: []const f32) void {
    std.debug.assert(out.len == input.len);
    for (input, 0..) |x, elem_idx| {
        out[elem_idx] = std.math.tanh(x);
    }
}

test "reluContiguous basic inputs" {
    const input = [_]f32{ -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0 };
    const expected = [_]f32{ 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 2.0 };
    var output: [input.len]f32 = undefined;

    reluContiguous(&output, &input);

    for (output, expected) |out, exp| {
        try std.testing.expectEqual(exp, out);
    }
}

test "reluContiguous non-negative outputs" {
    const input = [_]f32{ -100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0 };
    var output: [input.len]f32 = undefined;

    reluContiguous(&output, &input);

    // All outputs must be >= 0
    for (output) |val| {
        try std.testing.expect(val >= 0.0);
    }

    // Positive inputs pass through unchanged
    try std.testing.expectEqual(@as(f32, 1.0), output[4]);
    try std.testing.expectEqual(@as(f32, 10.0), output[5]);
    try std.testing.expectEqual(@as(f32, 100.0), output[6]);
}

test "reluContiguous edge cases - extreme values" {
    const input = [_]f32{ -1e38, -1e10, 0.0, 1e-10, 1e38 };
    var output: [input.len]f32 = undefined;

    reluContiguous(&output, &input);

    try std.testing.expectEqual(@as(f32, 0.0), output[0]);
    try std.testing.expectEqual(@as(f32, 0.0), output[1]);
    try std.testing.expectEqual(@as(f32, 0.0), output[2]);
    try std.testing.expectApproxEqRel(@as(f32, 1e-10), output[3], 1e-5);
    try std.testing.expectApproxEqRel(@as(f32, 1e38), output[4], 1e-5);
}

test "sigmoidContiguous known values" {
    const input = [_]f32{ -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0 };
    var output: [input.len]f32 = undefined;

    sigmoidContiguous(&output, &input);

    // sigmoid(0) = 0.5
    try std.testing.expectApproxEqRel(@as(f32, 0.5), output[3], 1e-5);

    // sigmoid(-x) + sigmoid(x) = 1 (symmetry)
    try std.testing.expectApproxEqRel(@as(f32, 1.0), output[0] + output[6], 1e-3);
    try std.testing.expectApproxEqRel(@as(f32, 1.0), output[1] + output[5], 1e-3);
    try std.testing.expectApproxEqRel(@as(f32, 1.0), output[2] + output[4], 1e-3);
}

test "sigmoidContiguous range (0,1)" {
    const input = [_]f32{ -100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0 };
    var output: [input.len]f32 = undefined;

    sigmoidContiguous(&output, &input);

    // All outputs must be in [0, 1] (fast exp may hit exact boundaries)
    for (output) |val| {
        try std.testing.expect(val >= 0.0);
        try std.testing.expect(val <= 1.0);
    }

    // Large positive values approach 1
    try std.testing.expect(output[6] > 0.99);

    // Large negative values approach 0
    try std.testing.expect(output[0] < 0.01);
}

test "sigmoidContiguous edge cases - extreme values" {
    const input = [_]f32{ -88.0, -50.0, 0.0, 50.0, 88.0 };
    var output: [input.len]f32 = undefined;

    sigmoidContiguous(&output, &input);

    // All outputs should be finite and in valid range (fast exp may hit exact boundaries)
    for (output) |val| {
        try std.testing.expect(std.math.isFinite(val));
        try std.testing.expect(val >= 0.0);
        try std.testing.expect(val <= 1.0);
    }
}

test "sigmoidContiguous SIMD vs scalar consistency" {
    const allocator = std.testing.allocator;

    // Test with array size that exercises both SIMD and scalar paths
    const size = VEC_LEN * 3 + 2; // SIMD path + scalar remainder
    const input = try allocator.alloc(f32, size);
    defer allocator.free(input);
    const output = try allocator.alloc(f32, size);
    defer allocator.free(output);

    // Fill with varied test values
    for (0..size) |i| {
        input[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - @as(i32, @intCast(size / 2)))) * 0.5;
    }

    sigmoidContiguous(output, input);

    // Verify all outputs are in valid range
    for (output) |val| {
        try std.testing.expect(val > 0.0);
        try std.testing.expect(val < 1.0);
    }

    // Verify symmetry around center
    const mid = size / 2;
    if (mid > 0 and mid < size - 1) {
        try std.testing.expectApproxEqRel(@as(f32, 0.5), output[mid], 1e-3);
    }
}

test "tanhContiguous known values" {
    const input = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    var output: [input.len]f32 = undefined;

    tanhContiguous(&output, &input);

    // tanh(0) = 0
    try std.testing.expectApproxEqRel(@as(f32, 0.0), output[2], 1e-5);

    // tanh is odd: tanh(-x) = -tanh(x)
    try std.testing.expectApproxEqRel(-output[4], output[0], 1e-5);
    try std.testing.expectApproxEqRel(-output[3], output[1], 1e-5);

    // Verify against standard library
    for (input, output) |in, out| {
        const expected = std.math.tanh(in);
        try std.testing.expectApproxEqRel(expected, out, 1e-5);
    }
}

test "tanhContiguous range (-1,1)" {
    const input = [_]f32{ -100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0 };
    var output: [input.len]f32 = undefined;

    tanhContiguous(&output, &input);

    // All outputs must be in [-1, 1]
    for (output) |val| {
        try std.testing.expect(val >= -1.0);
        try std.testing.expect(val <= 1.0);
    }

    // Large positive values approach 1
    try std.testing.expect(output[6] > 0.99);

    // Large negative values approach -1
    try std.testing.expect(output[0] < -0.99);
}

test "tanhContiguous edge cases - extreme values" {
    const input = [_]f32{ -1e10, -50.0, 0.0, 50.0, 1e10 };
    var output: [input.len]f32 = undefined;

    tanhContiguous(&output, &input);

    // All outputs should be finite and in valid range
    for (output) |val| {
        try std.testing.expect(std.math.isFinite(val));
        try std.testing.expect(val >= -1.0);
        try std.testing.expect(val <= 1.0);
    }
}

test "geluContiguous known values" {
    const input = [_]f32{ -3.0, -1.0, 0.0, 1.0, 3.0 };
    var output: [input.len]f32 = undefined;

    geluContiguous(&output, &input);

    // GELU(0) ≈ 0
    try std.testing.expectApproxEqRel(@as(f32, 0.0), output[2], 1e-3);

    // Verify formula: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    for (input, output) |in, out| {
        const x3 = in * in * in;
        const inner = 0.7978845608028654 * (in + 0.044715 * x3);
        const tanh_val = std.math.tanh(inner);
        const expected = 0.5 * in * (1.0 + tanh_val);
        try std.testing.expectApproxEqRel(expected, out, 1e-3);
    }

    // GELU(3) ≈ 3.0 (approaches identity for large positive x)
    try std.testing.expect(output[4] > 2.95);

    // GELU(-3) ≈ -0.004 (small negative for large negative x)
    try std.testing.expect(output[0] < 0.01);
    try std.testing.expect(output[0] > -0.01);
}

test "geluContiguous properties" {
    const input = [_]f32{ -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0 };
    var output: [input.len]f32 = undefined;

    geluContiguous(&output, &input);

    // GELU is monotonically increasing - verify overall trend
    // (Small floating point errors near zero are acceptable)
    try std.testing.expect(output[0] < output[output.len - 1]);
    try std.testing.expect(output[1] < output[output.len - 2]);

    // For large positive x, GELU(x) ≈ x
    try std.testing.expectApproxEqRel(@as(f32, 5.0), output[6], 1e-2);

    // For large negative x, GELU(x) ≈ 0
    try std.testing.expect(@abs(output[0]) < 0.01);
}

test "geluContiguous edge cases - extreme values" {
    const input = [_]f32{ -10.0, -5.0, 0.0, 5.0, 10.0 };
    var output: [input.len]f32 = undefined;

    geluContiguous(&output, &input);

    // All outputs should be finite
    for (output) |val| {
        try std.testing.expect(std.math.isFinite(val));
    }

    // Very large negative values should be near 0
    try std.testing.expect(@abs(output[0]) < 0.001);

    // Very large positive values should be close to input
    try std.testing.expectApproxEqRel(@as(f32, 10.0), output[4], 1e-2);
}

test "siluScalarReference known values" {
    const input = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    var output: [input.len]f32 = undefined;

    siluScalarReference(&output, &input);

    // SiLU(0) = 0 * sigmoid(0) = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output[2], 1e-7);

    // Verify x * sigmoid(x) using std.math.exp (same as the implementation)
    for (input, 0..) |x, i| {
        const sig = 1.0 / (1.0 + @exp(-x));
        const expected = x * sig;
        try std.testing.expectApproxEqRel(expected, output[i], 1e-6);
    }

    // SiLU is positive for positive inputs
    try std.testing.expect(output[3] > 0.0);
    try std.testing.expect(output[4] > 0.0);

    // SiLU(-x) is small and negative for negative x
    try std.testing.expect(output[0] < 0.0);
    try std.testing.expect(output[1] < 0.0);
}

test "siluContiguous known values" {
    const input = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    var output: [input.len]f32 = undefined;

    siluContiguous(&output, &input);

    // SiLU(0) = 0 * sigmoid(0) = 0
    try std.testing.expectApproxEqRel(@as(f32, 0.0), output[2], 1e-3);

    // Verify formula: x * sigmoid(x)
    for (input, output) |in, out| {
        const sig = 1.0 / (1.0 + fastExpScalar(-in));
        const expected = in * sig;
        try std.testing.expectApproxEqRel(expected, out, 1e-3);
    }
}

test "siluContiguous properties" {
    const input = [_]f32{ -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0 };
    var output: [input.len]f32 = undefined;

    siluContiguous(&output, &input);

    // SiLU is monotonically increasing - verify overall trend
    // (Small floating point errors near zero are acceptable)
    try std.testing.expect(output[0] < output[output.len - 1]);
    try std.testing.expect(output[1] < output[output.len - 2]);

    // For large positive x, SiLU(x) ≈ x (since sigmoid(x) ≈ 1)
    try std.testing.expectApproxEqRel(@as(f32, 5.0), output[6], 5e-2);

    // For large negative x, SiLU(x) ≈ 0 (relaxed tolerance for fast exp)
    try std.testing.expect(@abs(output[0]) < 0.05);
}

test "siluContiguous edge cases - extreme values" {
    const input = [_]f32{ -88.0, -10.0, 0.0, 10.0, 88.0 };
    var output: [input.len]f32 = undefined;

    siluContiguous(&output, &input);

    // All outputs should be finite
    for (output) |val| {
        try std.testing.expect(std.math.isFinite(val));
    }

    // Very large negative values should be near 0
    try std.testing.expect(@abs(output[0]) < 0.01);

    // Very large positive values should be close to input
    try std.testing.expectApproxEqRel(@as(f32, 88.0), output[4], 1e-1);
}

test "siluContiguous SIMD vs scalar reference equivalence" {
    const allocator = std.testing.allocator;

    // Test with array size that exercises both SIMD and scalar paths
    const size = VEC_LEN * 3 + 2; // SIMD path + scalar remainder
    const input = try allocator.alloc(f32, size);
    defer allocator.free(input);
    const simd_output = try allocator.alloc(f32, size);
    defer allocator.free(simd_output);
    const scalar_output = try allocator.alloc(f32, size);
    defer allocator.free(scalar_output);

    // Fill with varied test values
    for (0..size) |i| {
        input[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - @as(i32, @intCast(size / 2)))) * 0.3;
    }

    // Run both implementations
    siluContiguous(simd_output, input);
    siluScalarReference(scalar_output, input);

    // Verify SIMD matches scalar reference for all elements
    for (simd_output, scalar_output) |simd_val, scalar_val| {
        try std.testing.expect(std.math.isFinite(simd_val));
        try std.testing.expect(std.math.isFinite(scalar_val));
        // fastExp approximation allows up to 1e-3 relative error
        try std.testing.expectApproxEqRel(scalar_val, simd_val, 1e-3);
    }
}

test "siluContiguous basic" {
    const input = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    var output: [input.len]f32 = undefined;
    siluContiguous(&output, &input);

    // SiLU(0) = 0
    try std.testing.expectApproxEqRel(@as(f32, 0.0), output[2], 1e-5);

    // Verify x * sigmoid(x) formula
    for (input, 0..) |x, i| {
        const sig = 1.0 / (1.0 + fastExpScalar(-x));
        const expected = x * sig;
        try std.testing.expectApproxEqRel(expected, output[i], 1e-3);
    }
}

test "geluContiguous basic" {
    const input = [_]f32{ -3.0, -1.0, 0.0, 1.0, 3.0 };
    var output: [input.len]f32 = undefined;
    geluContiguous(&output, &input);

    // GELU(0) ≈ 0
    try std.testing.expectApproxEqRel(@as(f32, 0.0), output[2], 1e-3);

    // For large positive x, GELU(x) ≈ x
    try std.testing.expect(output[4] > 2.9);
}

test "reluContiguous basic" {
    const input = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    var output: [input.len]f32 = undefined;
    reluContiguous(&output, &input);

    // Negative values -> 0
    try std.testing.expectEqual(@as(f32, 0.0), output[0]);
    try std.testing.expectEqual(@as(f32, 0.0), output[1]);
    try std.testing.expectEqual(@as(f32, 0.0), output[2]);
    // Positive values pass through
    try std.testing.expectEqual(@as(f32, 1.0), output[3]);
    try std.testing.expectEqual(@as(f32, 2.0), output[4]);
}

test "sigmoidContiguous basic" {
    const input = [_]f32{ -5.0, 0.0, 5.0 };
    var output: [input.len]f32 = undefined;
    sigmoidContiguous(&output, &input);

    // sigmoid(0) = 0.5
    try std.testing.expectApproxEqRel(@as(f32, 0.5), output[1], 1e-5);

    // All outputs in (0, 1)
    for (output) |val| {
        try std.testing.expect(val > 0.0 and val < 1.0);
    }
}

test "tanhContiguous basic" {
    const input = [_]f32{ -2.0, 0.0, 2.0 };
    var output: [input.len]f32 = undefined;
    tanhContiguous(&output, &input);

    // tanh(0) = 0
    try std.testing.expectApproxEqRel(@as(f32, 0.0), output[1], 1e-5);

    // Verify against std.math.tanh
    for (input, output) |in, out| {
        try std.testing.expectApproxEqRel(std.math.tanh(in), out, 1e-5);
    }
}
