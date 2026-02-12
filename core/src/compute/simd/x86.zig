//! x86/AVX2 intrinsics for quantized dot products.
//!
//! Uses pmaddubsw and pmaddwd for optimal int8 dot products on x86_64.
//! Provides scalar fallback for non-AVX2 targets.

const std = @import("std");
const builtin = @import("builtin");

/// pmaddubsw: Multiply unsigned×signed bytes, add adjacent pairs to i16
/// This is the critical instruction for quantized dot products.
/// Input: a = unsigned bytes [0..255], b = signed bytes [-128..127]
/// Output: i16 pairs where out[i] = a[2i]*b[2i] + a[2i+1]*b[2i+1]
pub inline fn maddubsw(a: @Vector(32, u8), b: @Vector(32, i8)) @Vector(16, i16) {
    if (comptime builtin.cpu.arch == .x86_64) {
        // Use vpmaddubsw instruction directly via inline assembly
        return asm ("vpmaddubsw %[b], %[a], %[result]"
            : [result] "=x" (-> @Vector(16, i16)),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else {
        // Portable implementation for non-x86
        const a_i16: @Vector(32, i16) = a;
        const b_i16: @Vector(32, i16) = b;
        const prod = a_i16 * b_i16;
        // Sum adjacent pairs
        const evens: @Vector(16, i16) = @shuffle(i16, prod, undefined, @Vector(16, i32){
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
        });
        const odds: @Vector(16, i16) = @shuffle(i16, prod, undefined, @Vector(16, i32){
            1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
        });
        return evens +% odds;
    }
}

/// pmaddwd: Multiply i16 pairs, add adjacent pairs to i32
/// Input: 16 i16 values
/// Output: 8 i32 values where out[i] = a[2i]*b[2i] + a[2i+1]*b[2i+1]
pub inline fn pmaddwd(a: @Vector(16, i16), ones: @Vector(16, i16)) @Vector(8, i32) {
    if (comptime builtin.cpu.arch == .x86_64) {
        return asm ("vpmaddwd %[ones], %[a], %[result]"
            : [result] "=x" (-> @Vector(8, i32)),
            : [a] "x" (a),
              [ones] "x" (ones),
        );
    } else {
        // Portable implementation
        const a_i32: @Vector(16, i32) = a;
        const ones_i32: @Vector(16, i32) = ones;
        const prod = a_i32 * ones_i32;
        const evens: @Vector(8, i32) = @shuffle(i32, prod, undefined, @Vector(8, i32){
            0, 2, 4, 6, 8, 10, 12, 14,
        });
        const odds: @Vector(8, i32) = @shuffle(i32, prod, undefined, @Vector(8, i32){
            1, 3, 5, 7, 9, 11, 13, 15,
        });
        return evens + odds;
    }
}

/// Absolute value of i8 vector (returns as u8 for pmaddubsw)
pub inline fn absI8(x: @Vector(32, i8)) @Vector(32, u8) {
    if (comptime builtin.cpu.arch == .x86_64) {
        // pabsb instruction
        const result = asm ("vpabsb %[x], %[result]"
            : [result] "=x" (-> @Vector(32, i8)),
            : [x] "x" (x),
        );
        return @bitCast(result);
    } else {
        // Manual abs: (x ^ (x >> 7)) - (x >> 7)
        const mask = x >> @as(@Vector(32, u3), @splat(7)); // sign bit extended to all bits
        const abs_val = (x ^ mask) -% mask;
        return @bitCast(abs_val);
    }
}

/// Apply sign of 'sign' to 'x': if sign[i] < 0, negate x[i]
pub inline fn signI8(x: @Vector(32, i8), sign: @Vector(32, i8)) @Vector(32, i8) {
    if (comptime builtin.cpu.arch == .x86_64) {
        // psignb instruction
        return asm ("vpsignb %[sign], %[x], %[result]"
            : [result] "=x" (-> @Vector(32, i8)),
            : [x] "x" (x),
              [sign] "x" (sign),
        );
    } else {
        const sign_mask = sign >> @as(@Vector(32, u3), @splat(7));
        const negated = (x ^ sign_mask) -% sign_mask;
        const zero_mask: @Vector(32, i8) = @select(i8, sign == @as(@Vector(32, i8), @splat(0)), @as(@Vector(32, i8), @splat(0)), @as(@Vector(32, i8), @splat(-1)));
        return negated & zero_mask;
    }
}

// =============================================================================
// F16 Conversion Intrinsics (F16C extension, part of AVX2)
// =============================================================================

/// vcvtph2ps: Convert 8 FP16 values to 8 FP32 values using hardware instruction.
/// Available on x86_64 with F16C extension (included in AVX2).
/// This is ~10x faster than bit manipulation for F16→F32 conversion.
pub inline fn cvtph2ps(fp16_vec: @Vector(8, u16)) @Vector(8, f32) {
    if (comptime builtin.cpu.arch == .x86_64 and
        std.Target.x86.featureSetHas(builtin.cpu.features, .f16c))
    {
        // Use vcvtph2ps instruction directly
        // Input: 128-bit register with 8 FP16 values
        // Output: 256-bit register with 8 FP32 values
        return asm ("vcvtph2ps %[src], %[dst]"
            : [dst] "=x" (-> @Vector(8, f32)),
            : [src] "x" (fp16_vec),
        );
    } else {
        // Portable fallback using bit manipulation
        return fp16ToF32Portable(fp16_vec);
    }
}

/// Portable F16→F32 conversion via bit manipulation.
/// Handles all IEEE 754 cases: normal, subnormal, zero, infinity, NaN.
fn fp16ToF32Portable(h: @Vector(8, u16)) @Vector(8, f32) {
    const V32 = @Vector(8, u32);
    const h32: V32 = h;

    // Extract components
    const sign: V32 = (h32 >> @splat(15)) << @splat(31);
    const exp: V32 = (h32 >> @splat(10)) & @as(V32, @splat(0x1F));
    const mant: V32 = h32 & @as(V32, @splat(0x3FF));

    // Check for special cases using masks
    const is_zero_or_subnormal = exp == @as(V32, @splat(0));
    const is_inf_or_nan = exp == @as(V32, @splat(0x1F));
    const is_subnormal = is_zero_or_subnormal and (mant != @as(V32, @splat(0)));

    // Normal case: exp_f32 = exp_f16 - 15 + 127 = exp_f16 + 112
    const normal_exp = (exp + @as(V32, @splat(112))) << @splat(23);
    const normal_mant = mant << @splat(13);
    const normal_result = sign | normal_exp | normal_mant;

    // Zero case
    const zero_result = sign;

    // Infinity/NaN case: exp = 0xFF, mantissa preserved (shifted)
    const inf_nan_result = sign | @as(V32, @splat(0x7F800000)) | (mant << @splat(13));

    // Subnormal case: normalize by finding leading bit
    // For simplicity, use scalar conversion for subnormals (rare in practice)
    var subnormal_result: V32 = undefined;
    inline for (0..8) |i| {
        if (is_subnormal[i] != 0) {
            const m = mant[i];
            var shift: u5 = 0;
            var normalized_mant = m;
            // Find position of leading 1 bit
            while (normalized_mant < 0x200 and shift < 10) : (shift += 1) {
                normalized_mant <<= 1;
            }
            normalized_mant = (normalized_mant & 0x1FF) << 14; // Remove implicit 1, shift to f32 position
            const sub_exp: u32 = @as(u32, 113) - @as(u32, shift); // 127 - 15 + 1 - shift
            subnormal_result[i] = sign[i] | (sub_exp << 23) | normalized_mant;
        } else {
            subnormal_result[i] = 0;
        }
    }

    // Select appropriate result based on case
    var result = @select(u32, is_zero_or_subnormal, zero_result, normal_result);
    result = @select(u32, is_inf_or_nan, inf_nan_result, result);
    result = @select(u32, is_subnormal, subnormal_result, result);

    return @bitCast(result);
}

// =============================================================================
// Quantized Dot Product Intrinsics
// =============================================================================

/// Combined multiply-sum for i8×i8 → i32 using pmaddubsw + pmaddwd
/// This matches C's mul_sum_i8_pairs_float but returns i32 before float conversion
pub inline fn mulSumI8Pairs(x: @Vector(32, i8), y: @Vector(32, i8)) @Vector(8, i32) {
    // pmaddubsw requires unsigned × signed
    // Use sign trick: abs(x) × sign(y, x) = x × y
    const abs_x = absI8(x); // abs(x) as unsigned
    const signed_y = signI8(y, x); // y with sign of x applied

    const dot = maddubsw(abs_x, signed_y); // u8 × i8 → i16 pairs
    const ones: @Vector(16, i16) = @splat(1);
    return pmaddwd(dot, ones); // i16 pairs → i32
}

/// Fast Q4×Q8 dot product using unsigned nibbles directly
/// Uses algebraic identity: sum((q-8)*y) = sum(q*y) - 8*sum(y)
/// This avoids vpabsb/vpsignb overhead by keeping nibbles as u8 [0..15]
/// Returns: {dot_product, sum_of_y} where final = dot_product - 8*sum_of_y
pub inline fn mulSumU8I8WithYSum(q4: @Vector(32, u8), y: @Vector(32, i8)) struct { dot: @Vector(8, i32), sum_y: i32 } {
    // Direct u8 × i8 multiplication using maddubsw - no sign trick needed!
    const dot = maddubsw(q4, y); // u8 × i8 → i16 pairs
    const ones: @Vector(16, i16) = @splat(1);
    const dot_i32 = pmaddwd(dot, ones); // i16 pairs → i32

    // Sum all y values for the correction term
    // Use psadbw trick: sad(y, 0) gives sum of absolute values, but we need signed sum
    // Instead, widen to i16 and sum
    const y_lo: @Vector(16, i8) = @shuffle(i8, y, undefined, @Vector(16, i32){
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    });
    const y_hi: @Vector(16, i8) = @shuffle(i8, y, undefined, @Vector(16, i32){
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    });
    const y_lo_i16: @Vector(16, i16) = y_lo;
    const y_hi_i16: @Vector(16, i16) = y_hi;
    const y_sum_lanes = y_lo_i16 + y_hi_i16;
    const sum_y: i32 = @reduce(.Add, y_sum_lanes);

    return .{ .dot = dot_i32, .sum_y = sum_y };
}

// =============================================================================
// Tests
// =============================================================================

test "maddubsw: basic multiplication and horizontal add" {
    // [1,2] × [1,1] = 1×1 + 2×1 = 3
    // [3,4] × [1,1] = 3×1 + 4×1 = 7
    const a: @Vector(32, u8) = @splat(0);
    var a_mut = a;
    a_mut[0] = 1;
    a_mut[1] = 2;
    a_mut[2] = 3;
    a_mut[3] = 4;

    const b: @Vector(32, i8) = @splat(1);

    const result = maddubsw(a_mut, b);

    try std.testing.expectEqual(@as(i16, 3), result[0]);
    try std.testing.expectEqual(@as(i16, 7), result[1]);
}

test "maddubsw: unsigned × signed with negative multiplier" {
    // [10,20] × [1,-1] = 10×1 + 20×(-1) = 10 - 20 = -10
    const a: @Vector(32, u8) = @splat(0);
    var a_mut = a;
    a_mut[0] = 10;
    a_mut[1] = 20;

    const b: @Vector(32, i8) = @splat(0);
    var b_mut = b;
    b_mut[0] = 1;
    b_mut[1] = -1;

    const result = maddubsw(a_mut, b_mut);

    try std.testing.expectEqual(@as(i16, -10), result[0]);
}

test "maddubsw: overflow behavior to i16" {
    // [255,255] × [127,127] = 255×127 + 255×127 = 32385 + 32385 = 64770
    const a: @Vector(32, u8) = @splat(0);
    var a_mut = a;
    a_mut[0] = 255;
    a_mut[1] = 255;

    const b: @Vector(32, i8) = @splat(0);
    var b_mut = b;
    b_mut[0] = 127;
    b_mut[1] = 127;

    const result = maddubsw(a_mut, b_mut);

    // vpmaddubsw saturates to i16 range: [-32768, 32767]
    // 255 * 127 * 2 = 64770 saturates to 32767
    try std.testing.expectEqual(@as(i16, 32767), result[0]);
}

test "maddubsw: all zeros" {
    const a: @Vector(32, u8) = @splat(0);
    const b: @Vector(32, i8) = @splat(0);

    const result = maddubsw(a, b);

    for (0..16) |i| {
        try std.testing.expectEqual(@as(i16, 0), result[i]);
    }
}

test "maddubsw: alternating pattern" {
    // [1,0,2,0,3,0,4,0...] × [1,1,1,1,1,1,1,1...]
    // Result: [1, 2, 3, 4, ...]
    const a: @Vector(32, u8) = @splat(0);
    var a_mut = a;
    for (0..16) |i| {
        a_mut[i * 2] = @intCast(i + 1);
        a_mut[i * 2 + 1] = 0;
    }

    const b: @Vector(32, i8) = @splat(1);

    const result = maddubsw(a_mut, b);

    for (0..16) |i| {
        try std.testing.expectEqual(@as(i16, @intCast(i + 1)), result[i]);
    }
}

test "pmaddwd: basic multiplication and horizontal add" {
    // [1,2] × [1,1] = 1×1 + 2×1 = 3
    // [3,4] × [1,1] = 3×1 + 4×1 = 7
    const a: @Vector(16, i16) = @splat(0);
    var a_mut = a;
    a_mut[0] = 1;
    a_mut[1] = 2;
    a_mut[2] = 3;
    a_mut[3] = 4;

    const ones: @Vector(16, i16) = @splat(1);

    const result = pmaddwd(a_mut, ones);

    try std.testing.expectEqual(@as(i32, 3), result[0]);
    try std.testing.expectEqual(@as(i32, 7), result[1]);
}

test "pmaddwd: overflow to i32" {
    // [32767,32767] × [1,1] = 32767 + 32767 = 65534 (fits in i32)
    const a: @Vector(16, i16) = @splat(0);
    var a_mut = a;
    a_mut[0] = 32767;
    a_mut[1] = 32767;

    const ones: @Vector(16, i16) = @splat(1);

    const result = pmaddwd(a_mut, ones);

    try std.testing.expectEqual(@as(i32, 65534), result[0]);
}

test "pmaddwd: negative values" {
    // [-10,20] × [1,1] = -10 + 20 = 10
    const a: @Vector(16, i16) = @splat(0);
    var a_mut = a;
    a_mut[0] = -10;
    a_mut[1] = 20;

    const ones: @Vector(16, i16) = @splat(1);

    const result = pmaddwd(a_mut, ones);

    try std.testing.expectEqual(@as(i32, 10), result[0]);
}

test "pmaddwd: all zeros" {
    const a: @Vector(16, i16) = @splat(0);
    const ones: @Vector(16, i16) = @splat(1);

    const result = pmaddwd(a, ones);

    for (0..8) |i| {
        try std.testing.expectEqual(@as(i32, 0), result[i]);
    }
}

test "absI8: positive values unchanged" {
    const x: @Vector(32, i8) = @splat(0);
    var x_mut = x;
    x_mut[0] = 10;
    x_mut[1] = 127;
    x_mut[2] = 1;

    const result = absI8(x_mut);

    try std.testing.expectEqual(@as(u8, 10), result[0]);
    try std.testing.expectEqual(@as(u8, 127), result[1]);
    try std.testing.expectEqual(@as(u8, 1), result[2]);
}

test "absI8: negative values negated" {
    const x: @Vector(32, i8) = @splat(0);
    var x_mut = x;
    x_mut[0] = -10;
    x_mut[1] = -127;
    x_mut[2] = -1;

    const result = absI8(x_mut);

    try std.testing.expectEqual(@as(u8, 10), result[0]);
    try std.testing.expectEqual(@as(u8, 127), result[1]);
    try std.testing.expectEqual(@as(u8, 1), result[2]);
}

test "absI8: edge case -128" {
    // abs(-128) is implementation defined for i8
    // Typically wraps to -128 or becomes 128 (as u8)
    const x: @Vector(32, i8) = @splat(0);
    var x_mut = x;
    x_mut[0] = -128;

    const result = absI8(x_mut);

    // On most architectures, abs(-128) = 128 (as u8)
    // vpabsb returns 0x80 which is 128 as u8
    try std.testing.expectEqual(@as(u8, 128), result[0]);
}

test "absI8: all zeros" {
    const x: @Vector(32, i8) = @splat(0);

    const result = absI8(x);

    for (0..32) |i| {
        try std.testing.expectEqual(@as(u8, 0), result[i]);
    }
}

test "absI8: alternating signs" {
    const x: @Vector(32, i8) = @splat(0);
    var x_mut = x;
    for (0..16) |i| {
        x_mut[i * 2] = @intCast(i + 1);
        x_mut[i * 2 + 1] = -@as(i8, @intCast(i + 1));
    }

    const result = absI8(x_mut);

    for (0..16) |i| {
        try std.testing.expectEqual(@as(u8, @intCast(i + 1)), result[i * 2]);
        try std.testing.expectEqual(@as(u8, @intCast(i + 1)), result[i * 2 + 1]);
    }
}

test "signI8: positive selector keeps value" {
    const x: @Vector(32, i8) = @splat(0);
    var x_mut = x;
    x_mut[0] = 10;
    x_mut[1] = -5;

    const sign: @Vector(32, i8) = @splat(1); // positive selector

    const result = signI8(x_mut, sign);

    try std.testing.expectEqual(@as(i8, 10), result[0]);
    try std.testing.expectEqual(@as(i8, -5), result[1]);
}

test "signI8: negative selector negates" {
    const x: @Vector(32, i8) = @splat(0);
    var x_mut = x;
    x_mut[0] = 10;
    x_mut[1] = -5;

    const sign: @Vector(32, i8) = @splat(-1); // negative selector

    const result = signI8(x_mut, sign);

    try std.testing.expectEqual(@as(i8, -10), result[0]);
    try std.testing.expectEqual(@as(i8, 5), result[1]);
}

test "signI8: zero selector gives zero" {
    const x: @Vector(32, i8) = @splat(0);
    var x_mut = x;
    x_mut[0] = 10;
    x_mut[1] = -5;
    x_mut[2] = 127;

    const sign: @Vector(32, i8) = @splat(0); // zero selector

    const result = signI8(x_mut, sign);

    for (0..32) |i| {
        try std.testing.expectEqual(@as(i8, 0), result[i]);
    }
}

test "signI8: mixed selectors" {
    const x: @Vector(32, i8) = @splat(0);
    var x_mut = x;
    x_mut[0] = 10;
    x_mut[1] = 20;
    x_mut[2] = 30;

    const sign: @Vector(32, i8) = @splat(0);
    var sign_mut = sign;
    sign_mut[0] = 1; // keep
    sign_mut[1] = -1; // negate
    sign_mut[2] = 0; // zero

    const result = signI8(x_mut, sign_mut);

    try std.testing.expectEqual(@as(i8, 10), result[0]);
    try std.testing.expectEqual(@as(i8, -20), result[1]);
    try std.testing.expectEqual(@as(i8, 0), result[2]);
}

test "mulSumI8Pairs: basic positive values" {
    // [1,2] × [3,4] = 1×3 + 2×4 = 3 + 8 = 11
    const x: @Vector(32, i8) = @splat(0);
    var x_mut = x;
    x_mut[0] = 1;
    x_mut[1] = 2;

    const y: @Vector(32, i8) = @splat(0);
    var y_mut = y;
    y_mut[0] = 3;
    y_mut[1] = 4;

    const result = mulSumI8Pairs(x_mut, y_mut);

    try std.testing.expectEqual(@as(i32, 11), result[0]);
}

test "mulSumI8Pairs: negative values" {
    // [-1,2] × [3,-4] = -1×3 + 2×(-4) = -3 - 8 = -11
    const x: @Vector(32, i8) = @splat(0);
    var x_mut = x;
    x_mut[0] = -1;
    x_mut[1] = 2;

    const y: @Vector(32, i8) = @splat(0);
    var y_mut = y;
    y_mut[0] = 3;
    y_mut[1] = -4;

    const result = mulSumI8Pairs(x_mut, y_mut);

    try std.testing.expectEqual(@as(i32, -11), result[0]);
}

test "mulSumI8Pairs: all zeros" {
    const x: @Vector(32, i8) = @splat(0);
    const y: @Vector(32, i8) = @splat(0);

    const result = mulSumI8Pairs(x, y);

    for (0..8) |i| {
        try std.testing.expectEqual(@as(i32, 0), result[i]);
    }
}

test "mulSumI8Pairs: accumulation across all lanes" {
    // Test that all 8 output lanes are computed correctly
    const x: @Vector(32, i8) = @splat(0);
    var x_mut = x;
    for (0..32) |i| {
        x_mut[i] = @intCast(@as(i8, @intCast(i + 1)));
    }

    const y: @Vector(32, i8) = @splat(1);

    const result = mulSumI8Pairs(x_mut, y);

    // Lane 0: x[0]×y[0] + x[1]×y[1] + x[2]×y[2] + x[3]×y[3] = 1 + 2 + 3 + 4 = 10
    // But wait, mulSumI8Pairs does pairs, so:
    // Lane 0: (x[0]×y[0] + x[1]×y[1]) + (x[2]×y[2] + x[3]×y[3]) = (1+2) + (3+4) = 10
    // Actually, maddubsw creates 16 i16s, then pmaddwd creates 8 i32s
    // So lane 0 = (x[0]×y[0] + x[1]×y[1]) + (x[2]×y[2] + x[3]×y[3])
    try std.testing.expectEqual(@as(i32, 10), result[0]);
    // Lane 1: (x[4]×y[4] + x[5]×y[5]) + (x[6]×y[6] + x[7]×y[7]) = (5+6) + (7+8) = 26
    try std.testing.expectEqual(@as(i32, 26), result[1]);
}

test "mulSumU8I8WithYSum: basic calculation" {
    // q4=[1,2,3,4], y=[1,1,1,1]
    // dot: (1×1 + 2×1) + (3×1 + 4×1) = 3 + 7 = 10
    // sum_y: 1+1+1+1 = 4
    const q4: @Vector(32, u8) = @splat(0);
    var q4_mut = q4;
    q4_mut[0] = 1;
    q4_mut[1] = 2;
    q4_mut[2] = 3;
    q4_mut[3] = 4;

    const y: @Vector(32, i8) = @splat(1);

    const result = mulSumU8I8WithYSum(q4_mut, y);

    try std.testing.expectEqual(@as(i32, 10), result.dot[0]);
    try std.testing.expectEqual(@as(i32, 32), result.sum_y); // 32 ones
}

test "mulSumU8I8WithYSum: negative y values" {
    // q4=[8,8], y=[1,-1]
    // dot: 8×1 + 8×(-1) = 8 - 8 = 0
    // sum_y: needs to sum all 32 y values
    const q4: @Vector(32, u8) = @splat(8);

    const y: @Vector(32, i8) = @splat(0);
    var y_mut = y;
    y_mut[0] = 1;
    y_mut[1] = -1;

    const result = mulSumU8I8WithYSum(q4, y_mut);

    try std.testing.expectEqual(@as(i32, 0), result.dot[0]);
    try std.testing.expectEqual(@as(i32, 0), result.sum_y); // 1 + (-1) + 0 + ... = 0
}

test "mulSumU8I8WithYSum: algebraic identity verification" {
    // Test the identity: (q-8)×y = q×y - 8×sum(y)
    // Use simple values to avoid overflow
    const q4: @Vector(32, u8) = @splat(8); // Neutral value (q-8 = 0)

    const y: @Vector(32, i8) = @splat(2);

    const result = mulSumU8I8WithYSum(q4, y);

    // q=8, y=2 for all 32 elements
    // Each pair: 8×2 + 8×2 = 32
    // Each i32 accumulates 2 pairs: 32 + 32 = 64
    try std.testing.expectEqual(@as(i32, 64), result.dot[0]);
    try std.testing.expectEqual(@as(i32, 64), result.sum_y); // 32 × 2 = 64
}

// =============================================================================
// F16 Conversion Tests
// =============================================================================

test "cvtph2ps: basic positive values" {
    // FP16 encoding for common values:
    // 1.0 = 0x3C00, 2.0 = 0x4000, 0.5 = 0x3800
    const fp16_input: @Vector(8, u16) = .{
        0x3C00, // 1.0
        0x4000, // 2.0
        0x3800, // 0.5
        0x4200, // 3.0
        0x4400, // 4.0
        0x4500, // 5.0
        0x4600, // 6.0
        0x4700, // 7.0
    };

    const result = cvtph2ps(fp16_input);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), result[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), result[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), result[3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), result[4], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), result[5], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), result[6], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), result[7], 1e-6);
}

test "cvtph2ps: negative values" {
    // Negative FP16: sign bit (0x8000) | magnitude
    const fp16_input: @Vector(8, u16) = .{
        0xBC00, // -1.0
        0xC000, // -2.0
        0xB800, // -0.5
        0xC200, // -3.0
        0x0000, // 0.0
        0x8000, // -0.0
        0x3C00, // 1.0
        0xC400, // -4.0
    };

    const result = cvtph2ps(fp16_input);

    try std.testing.expectApproxEqAbs(@as(f32, -1.0), result[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), result[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), result[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -3.0), result[3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[4], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -0.0), result[5], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result[6], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -4.0), result[7], 1e-6);
}

test "cvtph2ps: zero values" {
    const fp16_input: @Vector(8, u16) = .{
        0x0000, // +0.0
        0x8000, // -0.0
        0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    };

    const result = cvtph2ps(fp16_input);

    try std.testing.expectEqual(@as(f32, 0.0), result[0]);
    try std.testing.expectEqual(@as(f32, -0.0), result[1]);
    // Check sign bit for -0.0
    try std.testing.expect(@as(u32, @bitCast(result[1])) == 0x80000000);
}

test "cvtph2ps: infinity" {
    const fp16_input: @Vector(8, u16) = .{
        0x7C00, // +Inf
        0xFC00, // -Inf
        0x3C00, 0x3C00, 0x3C00, 0x3C00, 0x3C00, 0x3C00,
    };

    const result = cvtph2ps(fp16_input);

    try std.testing.expect(std.math.isPositiveInf(result[0]));
    try std.testing.expect(std.math.isNegativeInf(result[1]));
}

test "cvtph2ps: NaN" {
    // FP16 NaN: exp=0x1F (all ones), mantissa != 0
    const fp16_input: @Vector(8, u16) = .{
        0x7E00, // NaN (quiet)
        0x7C01, // NaN (signaling)
        0xFE00, // -NaN
        0x3C00, 0x3C00, 0x3C00, 0x3C00, 0x3C00,
    };

    const result = cvtph2ps(fp16_input);

    try std.testing.expect(std.math.isNan(result[0]));
    try std.testing.expect(std.math.isNan(result[1]));
    try std.testing.expect(std.math.isNan(result[2]));
}

test "cvtph2ps: small normal values" {
    // Test values near the edge of normal/subnormal range
    // Smallest normal FP16: 2^-14 ≈ 6.1e-5, encoded as exp=1, mant=0 = 0x0400
    const fp16_input: @Vector(8, u16) = .{
        0x0400, // Smallest positive normal: 2^-14
        0x0401, // Slightly larger
        0x3555, // ~0.333 (1/3 approximation)
        0x3C00, // 1.0
        0x0000, 0x0000, 0x0000, 0x0000,
    };

    const result = cvtph2ps(fp16_input);

    // 2^-14 ≈ 6.103515625e-5
    try std.testing.expectApproxEqRel(@as(f32, 6.103515625e-5), result[0], 1e-4);
    try std.testing.expect(result[1] > result[0]); // Slightly larger
}
