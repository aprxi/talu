//! Fast exp approximations with SIMD acceleration.

const std = @import("std");
const simd = @import("simd/arch/root.zig");

const VEC_LEN = simd.f32_vec_len;
const F32Vec = simd.F32Vec;

/// Comptime-derived math constants for exp approximation.
/// Using std.math ensures full precision and documents the derivation.
const exp_constants = struct {
    /// log2(e) = 1/ln(2)
    const log2e: f32 = 1.0 / std.math.ln2;
    /// ln(2) split into high and low parts for range reduction accuracy
    const ln2_hi: f32 = 0.693359375; // Exact in float
    const ln2_lo: f32 = -2.12194440e-4; // Correction term
    /// Overflow/underflow bounds: exp(88.72) ≈ FLT_MAX
    const exp_hi: f32 = 88.3762626647949;
    const exp_lo: f32 = -88.3762626647949;
    /// Polynomial coefficients for 2^x approximation on [-0.5, 0.5]
    /// These are minimax coefficients, not easily derived from std.math
    const p0: f32 = 1.9875691500e-4;
    const p1: f32 = 1.3981999507e-3;
    const p2: f32 = 8.3334519073e-3;
    const p3: f32 = 4.1665795894e-2;
    const p4: f32 = 1.6666665459e-1;
    const p5: f32 = 5.0000001201e-1;
};

/// Fast vectorized exp approximation using Schraudolph's method.
/// Accurate to ~1% for |x| < 10, suitable for normalized-score transforms.
/// Based on: exp(x) ≈ 2^(x/ln2) with polynomial approximation for fractional part.
/// Uses comptime-detected SIMD width for optimal performance on any architecture.
pub inline fn fastExp(x: F32Vec) F32Vec {
    const exp_consts = exp_constants;
    const I32Vec = @Vector(VEC_LEN, i32);

    // Clamp x to avoid overflow/underflow
    var x_clamped = @max(@min(x, @as(F32Vec, @splat(exp_consts.exp_hi))), @as(F32Vec, @splat(exp_consts.exp_lo)));

    // Compute fx = floor(x * log2e + 0.5)
    const fx = @floor(x_clamped * @as(F32Vec, @splat(exp_consts.log2e)) + @as(F32Vec, @splat(0.5)));
    const fxi: I32Vec = @intFromFloat(fx);

    // x = x - fx * ln2 (range reduction using hi/lo split)
    x_clamped = x_clamped - fx * @as(F32Vec, @splat(exp_consts.ln2_hi));
    x_clamped = x_clamped - fx * @as(F32Vec, @splat(exp_consts.ln2_lo));

    // Polynomial approximation of 2^frac using Horner's method
    var y: F32Vec = @splat(exp_consts.p0);
    y = y * x_clamped + @as(F32Vec, @splat(exp_consts.p1));
    y = y * x_clamped + @as(F32Vec, @splat(exp_consts.p2));
    y = y * x_clamped + @as(F32Vec, @splat(exp_consts.p3));
    y = y * x_clamped + @as(F32Vec, @splat(exp_consts.p4));
    y = y * x_clamped + @as(F32Vec, @splat(exp_consts.p5));
    y = y * x_clamped * x_clamped + x_clamped + @as(F32Vec, @splat(1.0));

    // Build 2^n by manipulating float exponent bits
    const exponent_bits = (fxi + @as(I32Vec, @splat(127))) << @as(@Vector(VEC_LEN, u5), @splat(23));
    const pow2n: F32Vec = @bitCast(exponent_bits);

    return y * pow2n;
}

/// Scalar fast exp for remainder elements
pub inline fn fastExpScalar(x: f32) f32 {
    const exp_consts = exp_constants;

    var x_clamped = @max(@min(x, exp_consts.exp_hi), exp_consts.exp_lo);
    const fx = @floor(x_clamped * exp_consts.log2e + 0.5);
    const fxi: i32 = @intFromFloat(fx);
    x_clamped = x_clamped - fx * exp_consts.ln2_hi - fx * exp_consts.ln2_lo;

    var y = exp_consts.p0;
    y = y * x_clamped + exp_consts.p1;
    y = y * x_clamped + exp_consts.p2;
    y = y * x_clamped + exp_consts.p3;
    y = y * x_clamped + exp_consts.p4;
    y = y * x_clamped + exp_consts.p5;
    y = y * x_clamped * x_clamped + x_clamped + 1.0;

    const exponent_bits: u32 = @bitCast((fxi + 127) << 23);
    const pow2n: f32 = @bitCast(exponent_bits);
    return y * pow2n;
}

test "fastExp accuracy - compare against std.math.exp" {
    // Test fastExp accuracy compared to standard library exp
    // Using representative score/logit ranges.
    const test_values = [_]f32{ -5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0 };

    for (test_values) |x| {
        const fast_result = fastExpScalar(x);
        const std_result = @exp(x);
        const rel_error = @abs(fast_result - std_result) / std_result;

        // Fast exp should be accurate to within ~1% for typical values
        try std.testing.expect(rel_error < 0.01);
    }
}

test "fastExp range - typical score range (-10 to 10)" {
    // Test fastExp across a common score range.
    var x: f32 = -10.0;
    while (x <= 10.0) : (x += 0.5) {
        const fast_result = fastExpScalar(x);
        const std_result = @exp(x);

        // Verify the result is finite and positive
        try std.testing.expect(std.math.isFinite(fast_result));
        try std.testing.expect(fast_result > 0);

        // Check accuracy (within 1% for this range)
        const rel_error = @abs(fast_result - std_result) / std_result;
        try std.testing.expect(rel_error < 0.01);
    }
}

test "fastExp edge cases - very negative, zero, overflow boundaries" {
    // Test fastExp(0) = 1
    {
        const result = fastExpScalar(0.0);
        try std.testing.expectApproxEqRel(1.0, result, 1e-5);
    }

    // Test fastExp for very negative values (should approach 0)
    {
        const result = fastExpScalar(-50.0);
        try std.testing.expect(result < 1e-20);
        try std.testing.expect(result > 0);
    }

    // Test fastExp for values near the clamping boundary
    {
        const result_hi = fastExpScalar(88.0); // Near exp_hi
        try std.testing.expect(std.math.isFinite(result_hi));
        try std.testing.expect(result_hi > 0);

        const result_lo = fastExpScalar(-88.0); // Near exp_lo
        try std.testing.expect(std.math.isFinite(result_lo));
        try std.testing.expect(result_lo >= 0);
    }

    // Test clamping prevents overflow/underflow
    {
        const result_overflow = fastExpScalar(1000.0); // Should clamp
        try std.testing.expect(std.math.isFinite(result_overflow));

        const result_underflow = fastExpScalar(-1000.0); // Should clamp
        try std.testing.expect(std.math.isFinite(result_underflow));
        try std.testing.expect(result_underflow >= 0);
    }
}

test "fastExp SIMD vs scalar consistency" {
    // Test that vector and scalar paths produce consistent results
    // This ensures the SIMD optimization doesn't introduce errors

    const test_values = [_]f32{ -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0 };

    // Test each element through scalar path
    for (test_values) |x| {
        const scalar_result = fastExpScalar(x);

        // Build a vector with the same value in all lanes
        const vec_input: F32Vec = @splat(x);
        const vec_result = fastExp(vec_input);

        // All lanes should match the scalar result
        for (0..VEC_LEN) |lane| {
            const rel_error = @abs(vec_result[lane] - scalar_result) / @max(scalar_result, 1e-10);
            try std.testing.expect(rel_error < 1e-5);
        }
    }
}

test "fastExpScalar matches std.math.exp" {
    // Test basic accuracy
    const values = [_]f32{ -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0 };
    for (values) |x| {
        const result = fastExpScalar(x);
        const expected = @exp(x);
        const rel_error = @abs(result - expected) / expected;
        try std.testing.expect(rel_error < 0.01);
    }
}
