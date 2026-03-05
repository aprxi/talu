//! Fast math approximations with SIMD acceleration.
//!
//! - fastExp / fastExpScalar: exp(x) approximation (~1e-7 relative error)
//! - fastSinCos / fastSinCosScalar: simultaneous sin(x) and cos(x) (~5e-6 max error)

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

// ============================================================================
// Fast sin/cos (simultaneous) with SIMD acceleration
// ============================================================================

/// Return type for simultaneous sin/cos computation.
pub const SinCosResult = struct { sin: f32, cos: f32 };
pub const SinCosVecResult = struct { sin: F32Vec, cos: F32Vec };

/// Constants for sin/cos Cody-Waite range reduction and polynomial evaluation.
const sincos_constants = struct {
    /// 2/π for range reduction: k = round(x * two_over_pi)
    const two_over_pi: f32 = 0.63661977236758134;
    /// π/2 split into high + low for Cody-Waite range reduction.
    /// pio2_hi is the closest f32 to π/2; pio2_lo is the correction.
    const pio2_hi: f32 = 1.5707963705062866;
    const pio2_lo: f32 = -4.3711388286737929e-08;
    /// Taylor polynomial coefficients for sin(r) on [-π/4, π/4]:
    /// sin(r) = r * (1 + r² * (s1 + r² * (s2 + r² * (s3 + r² * s4))))
    const s1: f32 = -1.6666667e-01; // ≈ -1/6
    const s2: f32 = 8.3333340e-03; // ≈ 1/120
    const s3: f32 = -1.9841270e-04; // ≈ -1/5040
    const s4: f32 = 2.7557319e-06; // ≈ 1/362880
    /// Taylor polynomial coefficients for cos(r) on [-π/4, π/4]:
    /// cos(r) = 1 + r² * (c1 + r² * (c2 + r² * (c3 + r² * c4)))
    const c1: f32 = -5.0000000e-01; // ≈ -1/2
    const c2: f32 = 4.1666668e-02; // ≈ 1/24
    const c3: f32 = -1.3888889e-03; // ≈ -1/720
    const c4: f32 = 2.4801587e-05; // ≈ 1/40320
};

/// Fast scalar sincos: computes both sin(x) and cos(x).
/// Uses Cody-Waite range reduction + degree-9/8 Taylor polynomials.
/// Max error ~5e-6 (adequate for f32 RoPE rotations).
pub inline fn fastSinCosScalar(x: f32) SinCosResult {
    const sc = sincos_constants;

    // Range reduce: x = k * π/2 + r, where |r| ≤ π/4
    const k = @round(x * sc.two_over_pi);
    const r = (x - k * sc.pio2_hi) - k * sc.pio2_lo;
    const ki = @as(i32, @intFromFloat(k));
    const q: u32 = @as(u32, @bitCast(ki)) & 3;

    // Polynomial evaluation on reduced argument
    const r2 = r * r;
    const sin_r = r * (1.0 + r2 * (sc.s1 + r2 * (sc.s2 + r2 * (sc.s3 + r2 * sc.s4))));
    const cos_r = 1.0 + r2 * (sc.c1 + r2 * (sc.c2 + r2 * (sc.c3 + r2 * sc.c4)));

    // Select and negate based on quadrant
    var s = if (q & 1 == 0) sin_r else cos_r;
    var c = if (q & 1 == 0) cos_r else sin_r;
    if (q & 2 != 0) s = -s;
    if ((q + 1) & 2 != 0) c = -c;

    return .{ .sin = s, .cos = c };
}

/// Fast vectorized sincos: computes sin(x) and cos(x) for all SIMD lanes.
/// Uses Cody-Waite range reduction + Taylor polynomials with SIMD quadrant selection.
pub inline fn fastSinCos(x: F32Vec) SinCosVecResult {
    const sc = sincos_constants;
    const I32Vec = @Vector(VEC_LEN, i32);
    const U32Vec = @Vector(VEC_LEN, u32);

    // Range reduce: x = k * π/2 + r
    const k = @round(x * @as(F32Vec, @splat(sc.two_over_pi)));
    const r = (x - k * @as(F32Vec, @splat(sc.pio2_hi))) - k * @as(F32Vec, @splat(sc.pio2_lo));

    // Quadrant as integer
    const ki: I32Vec = @intFromFloat(k);
    const ku: U32Vec = @bitCast(ki);
    const q = ku & @as(U32Vec, @splat(3));

    // Polynomial evaluation
    const r2 = r * r;
    const sin_r = r * (@as(F32Vec, @splat(1.0)) + r2 * (@as(F32Vec, @splat(sc.s1)) + r2 * (@as(F32Vec, @splat(sc.s2)) + r2 * (@as(F32Vec, @splat(sc.s3)) + r2 * @as(F32Vec, @splat(sc.s4))))));
    const cos_r = @as(F32Vec, @splat(1.0)) + r2 * (@as(F32Vec, @splat(sc.c1)) + r2 * (@as(F32Vec, @splat(sc.c2)) + r2 * (@as(F32Vec, @splat(sc.c3)) + r2 * @as(F32Vec, @splat(sc.c4)))));

    // Quadrant-based selection via SIMD masks
    const zero_u: U32Vec = @splat(0);
    const one_u: U32Vec = @splat(1);
    const two_u: U32Vec = @splat(2);
    const swap = (q & one_u) != zero_u;
    const neg_s = (q & two_u) != zero_u;
    const neg_c = ((q +% one_u) & two_u) != zero_u;

    // Select sin/cos based on quadrant, then apply sign
    const s_raw = @select(f32, swap, cos_r, sin_r);
    const c_raw = @select(f32, swap, sin_r, cos_r);

    // Negate via sign-bit XOR (branchless)
    const sign_bit: U32Vec = @splat(0x80000000);
    const s_bits: U32Vec = @bitCast(s_raw);
    const c_bits: U32Vec = @bitCast(c_raw);
    const s: F32Vec = @bitCast(s_bits ^ @select(u32, neg_s, sign_bit, zero_u));
    const c: F32Vec = @bitCast(c_bits ^ @select(u32, neg_c, sign_bit, zero_u));

    return .{ .sin = s, .cos = c };
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

// ============================================================================
// fastSinCos tests
// ============================================================================

test "fastSinCosScalar basic values" {
    // Test at well-known angles
    const cases = [_]struct { x: f32, sin: f32, cos: f32 }{
        .{ .x = 0.0, .sin = 0.0, .cos = 1.0 },
        .{ .x = std.math.pi / 2.0, .sin = 1.0, .cos = 0.0 },
        .{ .x = std.math.pi, .sin = 0.0, .cos = -1.0 },
        .{ .x = -std.math.pi / 2.0, .sin = -1.0, .cos = 0.0 },
        .{ .x = std.math.pi / 4.0, .sin = 0.7071068, .cos = 0.7071068 },
        .{ .x = -std.math.pi / 4.0, .sin = -0.7071068, .cos = 0.7071068 },
    };
    for (cases) |c| {
        const result = fastSinCosScalar(c.x);
        try std.testing.expectApproxEqAbs(c.sin, result.sin, 1e-4);
        try std.testing.expectApproxEqAbs(c.cos, result.cos, 1e-4);
    }
}

test "fastSinCosScalar large angles (RoPE range)" {
    // RoPE angles: pos * inv_freq, up to pos=255
    const angles = [_]f32{ 10.0, 50.0, 100.0, 200.0, 255.0, -30.0 };
    for (angles) |x| {
        const result = fastSinCosScalar(x);
        const std_sin = @sin(x);
        const std_cos = @cos(x);
        try std.testing.expectApproxEqAbs(std_sin, result.sin, 1e-3);
        try std.testing.expectApproxEqAbs(std_cos, result.cos, 1e-3);
    }
}

test "fastSinCosScalar identity sin²+cos²=1" {
    var x: f32 = -20.0;
    while (x <= 20.0) : (x += 0.7) {
        const result = fastSinCosScalar(x);
        const sum_sq = result.sin * result.sin + result.cos * result.cos;
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum_sq, 1e-4);
    }
}

test "fastSinCos SIMD vs scalar consistency" {
    const test_values = [_]f32{ -10.0, -3.14, -1.0, 0.0, 1.0, 3.14, 10.0, 100.0 };
    for (test_values) |x| {
        const scalar = fastSinCosScalar(x);
        const vec_input: F32Vec = @splat(x);
        const vec_result = fastSinCos(vec_input);
        for (0..VEC_LEN) |lane| {
            try std.testing.expectApproxEqAbs(scalar.sin, vec_result.sin[lane], 1e-6);
            try std.testing.expectApproxEqAbs(scalar.cos, vec_result.cos[lane], 1e-6);
        }
    }
}
