//! Data Types and Quantization Formats
//!
//! Unified dtype for tensor operations supporting both standard types (for FFI)
//! and quantized formats (internal). Also provides FP16/BF16/FP8 conversion utilities.

const std = @import("std");

/// Unified dtype for tensor operations.
/// Supports both standard types (for FFI) and quantized formats (internal).
///
/// FFI values (0-11) match numpy/DLPack conventions for external interop.
/// Quantized types (20+) are internal only.
pub const DType = enum(u8) {
    // Standard types - FFI compatible (values 0-11 match external APIs)
    f32 = 0,
    f64 = 1,
    i32 = 2,
    i64 = 3,
    f16 = 4,
    bf16 = 5,
    i8 = 6,
    i16 = 7,
    u8 = 8,
    u16 = 9,
    u32 = 10,
    u64 = 11,

    // Quantized types - internal only (values 20+)
    grouped_affine_u4 = 25,
    grouped_affine_u8 = 26,
    mxfp4 = 27,
    f8_e4m3 = 28,

    // =========================================================================
    // FFI conversion (for Python/C/DLPack boundaries)
    // =========================================================================

    /// Convert to FFI-compatible u32 value for external APIs.
    /// Quantized types return u8 representation (they're byte arrays externally).
    pub fn toFFI(self: DType) u32 {
        return switch (self) {
            .f32 => 0,
            .f64 => 1,
            .i32 => 2,
            .i64 => 3,
            .f16 => 4,
            .bf16 => 5,
            .i8 => 6,
            .i16 => 7,
            .u8 => 8,
            .u16 => 9,
            .u32 => 10,
            .u64 => 11,
            // Quantized types appear as u8 arrays externally
            .grouped_affine_u4, .grouped_affine_u8, .mxfp4, .f8_e4m3 => 8,
        };
    }

    /// Create DType from FFI u32 value. Returns null for invalid values.
    pub fn fromFFI(val: u32) ?DType {
        return switch (val) {
            0 => .f32,
            1 => .f64,
            2 => .i32,
            3 => .i64,
            4 => .f16,
            5 => .bf16,
            6 => .i8,
            7 => .i16,
            8 => .u8,
            9 => .u16,
            10 => .u32,
            11 => .u64,
            else => null,
        };
    }

    /// Convert to numpy typestr format (e.g., "<f4")
    pub fn toTypeStr(self: DType) [*:0]const u8 {
        return switch (self) {
            .f32 => "<f4",
            .f64 => "<f8",
            .f16 => "<f2",
            .bf16 => "<V2", // bfloat16 has no numpy typestr, use void
            .i8 => "<i1",
            .i16 => "<i2",
            .i32 => "<i4",
            .i64 => "<i8",
            .u8 => "<u1",
            .u16 => "<u2",
            .u32 => "<u4",
            .u64 => "<u8",
            // Quantized types appear as u8 arrays
            .grouped_affine_u4, .grouped_affine_u8, .mxfp4, .f8_e4m3 => "<u1",
        };
    }

    // =========================================================================
    // Size and properties
    // =========================================================================

    /// Element size in bytes for non-quantized types.
    /// Quantized types return 0 (they require block-based size calculations).
    pub fn elementSize(self: DType) usize {
        return switch (self) {
            .f32 => 4,
            .f64 => 8,
            .f16 => 2,
            .bf16 => 2,
            .i8 => 1,
            .i16 => 2,
            .i32 => 4,
            .i64 => 8,
            .u8 => 1,
            .u16 => 2,
            .u32 => 4,
            .u64 => 8,
            .f8_e4m3 => 1,
            .grouped_affine_u4, .grouped_affine_u8, .mxfp4 => 0,
        };
    }

    /// Check if this is a quantized block type
    pub fn isQuantized(self: DType) bool {
        return switch (self) {
            .grouped_affine_u4, .grouped_affine_u8, .mxfp4 => true,
            else => false,
        };
    }

    /// Check if this is a standard (non-quantized) type
    pub fn isStandard(self: DType) bool {
        return @intFromEnum(self) < 20;
    }
};

/// GGML FP16 storage type
pub const GGMLFp16 = u16;

/// Grouped affine quantization metadata (u4/u8 packed weights + per-group scale/bias)
pub const GroupedAffineMeta = struct {
    scales: []u8,
    biases: []u8,
    group_size: usize,
    scales_dtype: DType = .bf16, // F16 or BF16
};

/// MXFP4 quantization metadata (Microsoft Microscaling)
/// Format: 4-bit values with E8M0 scales (32 values per scale)
pub const MXFP4Meta = struct {
    scales: []u8, // E8M0 scales (one per 32 values)
    block_size: usize, // Usually 32
};

// =============================================================================
// FP16/BF16 Conversion Utilities
// =============================================================================

pub fn f32ToFp16(value: f32) u16 {
    return @bitCast(@as(f16, @floatCast(value)));
}

/// Hardware-accelerated FP16 to F32 conversion.
/// Uses F16C instruction set when available.
pub inline fn fp16ToF32(fp16_bits: u16) f32 {
    return @floatCast(@as(f16, @bitCast(fp16_bits)));
}

/// Vectorized FP16 to F32 conversion - 8 values at once
pub inline fn fp16x8ToF32(fp16_bits: @Vector(8, u16)) @Vector(8, f32) {
    const h_f16: @Vector(8, f16) = @bitCast(fp16_bits);
    return @floatCast(h_f16);
}

/// Vectorized FP16 to F32 conversion.
/// Uses hardware vcvtph2ps instruction on x86 with F16C (part of AVX2).
/// Falls back to bit manipulation on other platforms.
///
/// F16 format: 1 sign + 5 exp (bias 15) + 10 mantissa
/// F32 format: 1 sign + 8 exp (bias 127) + 23 mantissa
pub inline fn fp16x8ToF32Bits(comptime VEC: comptime_int, h: @Vector(VEC, u16)) @Vector(VEC, f32) {
    const simd = @import("compute/cpu/simd/arch/root.zig");

    // For 8-element vectors on x86 with F16C, use hardware instruction
    if (comptime VEC == 8) {
        return simd.cvtph2ps(h);
    }

    // For other vector widths, process in 8-element chunks or use portable fallback
    if (comptime VEC > 8 and VEC % 8 == 0) {
        var result: @Vector(VEC, f32) = undefined; // filled element-by-element in comptime loop below
        comptime var i: usize = 0;
        inline while (i < VEC) : (i += 8) {
            const chunk: @Vector(8, u16) = @shuffle(u16, h, undefined, blk: {
                var indices: [8]i32 = undefined;
                for (0..8) |j| indices[j] = @intCast(i + j);
                break :blk indices;
            });
            const converted = simd.cvtph2ps(chunk);
            inline for (0..8) |j| {
                result[i + j] = converted[j];
            }
        }
        return result;
    }

    // Portable fallback for non-8 vector widths
    return fp16ToF32Portable(VEC, h);
}

/// Portable FP16 to F32 conversion via bit manipulation.
/// Used as fallback when hardware F16C is not available.
fn fp16ToF32Portable(comptime VEC: comptime_int, h: @Vector(VEC, u16)) @Vector(VEC, f32) {
    const V32 = @Vector(VEC, u32);

    // Extend to 32-bit for manipulation
    const h32: V32 = h;

    // Extract components
    const sign: V32 = (h32 >> @splat(15)) << @splat(31);
    const exp: V32 = (h32 >> @splat(10)) & @as(V32, @splat(0x1F));
    const mant: V32 = h32 & @as(V32, @splat(0x3FF));

    // Check for special cases using masks
    const exp_zero: @Vector(VEC, bool) = exp == @as(V32, @splat(0));
    const exp_max: @Vector(VEC, bool) = exp == @as(V32, @splat(0x1F));
    const mant_zero: @Vector(VEC, bool) = mant == @as(V32, @splat(0));

    // Normal case: rebias exponent (15 -> 127, add 112) and shift mantissa
    const normal_exp: V32 = (exp + @as(V32, @splat(112))) << @splat(23);
    const normal_mant: V32 = mant << @splat(13);
    const normal_result: V32 = sign | normal_exp | normal_mant;

    // Zero case: just the sign bit
    const zero_result: V32 = sign;

    // Infinity case: F32 infinity exponent (0xFF << 23)
    const inf_result: V32 = sign | @as(V32, @splat(0x7F800000));

    // NaN case: F32 NaN (exponent all 1s, mantissa non-zero, preserve mantissa bits)
    const nan_result: V32 = sign | @as(V32, @splat(0x7F800000)) | (mant << @splat(13));

    // Subnormal case: needs normalization
    // F16 subnormal: value = 2^-14 * (mant/1024) = 2^-24 * mant
    // We use the multiplication approach which is still faster than full floatCast
    const subnorm_scale: @Vector(VEC, f32) = @splat(5.9604644775390625e-8); // 2^-24
    const mant_f32: @Vector(VEC, f32) = @floatFromInt(mant);
    const subnorm_unsigned: @Vector(VEC, f32) = mant_f32 * subnorm_scale;
    // Apply sign (negate if sign bit set)
    const sign_mask: @Vector(VEC, bool) = (h32 >> @splat(15)) != @as(V32, @splat(0));
    const subnorm_result: @Vector(VEC, f32) = @select(f32, sign_mask, -subnorm_unsigned, subnorm_unsigned);

    // Build result by selecting based on conditions
    // Priority: zero > subnormal > inf > nan > normal
    var result: V32 = normal_result;

    // Apply infinity (exp == 31, mant == 0)
    const is_inf = @select(bool, mant_zero, exp_max, @as(@Vector(VEC, bool), @splat(false)));
    result = @select(u32, is_inf, inf_result, result);

    // Apply NaN (exp == 31, mant != 0)
    const is_nan = @select(bool, exp_max, @select(bool, mant_zero, @as(@Vector(VEC, bool), @splat(false)), @as(@Vector(VEC, bool), @splat(true))), @as(@Vector(VEC, bool), @splat(false)));
    result = @select(u32, is_nan, nan_result, result);

    // Apply subnormal (exp == 0, mant != 0) - need to use f32 result
    const is_subnorm = @select(bool, exp_zero, @select(bool, mant_zero, @as(@Vector(VEC, bool), @splat(false)), @as(@Vector(VEC, bool), @splat(true))), @as(@Vector(VEC, bool), @splat(false)));
    const result_f32: @Vector(VEC, f32) = @bitCast(result);
    const with_subnorm: @Vector(VEC, f32) = @select(f32, is_subnorm, subnorm_result, result_f32);

    // Apply zero (exp == 0, mant == 0)
    const is_zero = @select(bool, exp_zero, mant_zero, @as(@Vector(VEC, bool), @splat(false)));
    const zero_f32: @Vector(VEC, f32) = @bitCast(zero_result);
    return @select(f32, is_zero, zero_f32, with_subnorm);
}

/// Convert BFloat16 to Float32.
/// BF16 is the upper 16 bits of an IEEE 754 float32.
pub fn bf16ToF32(bf16_bits: u16) f32 {
    const bits = @as(u32, bf16_bits) << 16;
    return @bitCast(bits);
}

// ============================================================================
// FP8 E4M3 Support
// ============================================================================

/// Convert a single FP8 E4M3 value to f32
/// FP8 E4M3 format: 1 sign bit, 4 exponent bits (bias=7), 3 mantissa bits
/// Range: ±448, smallest subnormal: 2^-9
pub inline fn fp8e4m3ToF32(value: u8) f32 {
    const sign: u32 = @as(u32, value >> 7) << 31;
    const exp: u32 = (value >> 3) & 0x0F;
    const mant: u32 = value & 0x07;

    if (exp == 0) {
        // Subnormal: value = (-1)^sign * 2^-6 * (0.mantissa)
        // mantissa bits are 0.m2 m1 m0 = mant/8
        if (mant == 0) {
            // Zero (preserve sign)
            return @bitCast(sign);
        }
        // Subnormal: 2^-6 * (mant/8) = mant * 2^-9
        const f: f32 = @floatFromInt(mant);
        const result = f * (1.0 / 512.0); // 2^-9
        return if (sign != 0) -result else result;
    } else if (exp == 0x0F and mant == 0x07) {
        // NaN (E4M3 uses all 1s as NaN, no infinity)
        return std.math.nan(f32);
    } else {
        // Normal: value = (-1)^sign * 2^(exp-7) * (1 + mantissa/8)
        // Convert to f32: exp_f32 = exp - 7 + 127 = exp + 120
        // mantissa needs to be shifted: 3 bits -> 23 bits = shift left by 20
        const exp_f32: u32 = (exp + 120) << 23;
        const mant_f32: u32 = mant << 20;
        return @bitCast(sign | exp_f32 | mant_f32);
    }
}

/// Dequantize FP8 E4M3 tensor to f32 with scale factor
/// scale_inv is the inverse scale (weight_scale_inv from safetensors)
/// Dequantization: output[i] = fp8_to_f32(input[i]) * scale_inv
fn dequantizeFp8E4M3(
    input: []const u8,
    scale_inv: f32,
    output: []f32,
) void {
    for (input, 0..) |value, idx| {
        output[idx] = fp8e4m3ToF32(value) * scale_inv;
    }
}

/// Dequantize FP8 E4M3 tensor to bf16 with scale factor
pub fn dequantizeFp8E4M3ToBf16(
    input: []const u8,
    scale_inv: f32,
    output: []u16,
) void {
    for (input, 0..) |value, idx| {
        const scaled_value = fp8e4m3ToF32(value) * scale_inv;
        output[idx] = f32ToBf16(scaled_value);
    }
}

/// Convert f32 to bf16 (truncate lower 16 bits)
pub inline fn f32ToBf16(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    return @truncate(bits >> 16);
}

// =============================================================================
// Tests
// =============================================================================

test "DType.elementSize - standard types" {
    try std.testing.expectEqual(@as(usize, 4), DType.f32.elementSize());
    try std.testing.expectEqual(@as(usize, 8), DType.f64.elementSize());
    try std.testing.expectEqual(@as(usize, 2), DType.f16.elementSize());
    try std.testing.expectEqual(@as(usize, 2), DType.bf16.elementSize());
    try std.testing.expectEqual(@as(usize, 1), DType.i8.elementSize());
    try std.testing.expectEqual(@as(usize, 2), DType.i16.elementSize());
    try std.testing.expectEqual(@as(usize, 4), DType.i32.elementSize());
    try std.testing.expectEqual(@as(usize, 8), DType.i64.elementSize());
    try std.testing.expectEqual(@as(usize, 1), DType.u8.elementSize());
    try std.testing.expectEqual(@as(usize, 2), DType.u16.elementSize());
    try std.testing.expectEqual(@as(usize, 4), DType.u32.elementSize());
    try std.testing.expectEqual(@as(usize, 8), DType.u64.elementSize());
    try std.testing.expectEqual(@as(usize, 1), DType.f8_e4m3.elementSize());
}

test "DType.elementSize - quantized types return 0" {
    try std.testing.expectEqual(@as(usize, 0), DType.grouped_affine_u4.elementSize());
    try std.testing.expectEqual(@as(usize, 0), DType.grouped_affine_u8.elementSize());
    try std.testing.expectEqual(@as(usize, 0), DType.mxfp4.elementSize());
}

test "DType.isQuantized" {
    // Standard types should not be quantized
    try std.testing.expect(!DType.f32.isQuantized());
    try std.testing.expect(!DType.f64.isQuantized());
    try std.testing.expect(!DType.f16.isQuantized());
    try std.testing.expect(!DType.bf16.isQuantized());
    try std.testing.expect(!DType.i8.isQuantized());
    try std.testing.expect(!DType.i16.isQuantized());
    try std.testing.expect(!DType.i32.isQuantized());
    try std.testing.expect(!DType.i64.isQuantized());
    try std.testing.expect(!DType.u8.isQuantized());
    try std.testing.expect(!DType.u16.isQuantized());
    try std.testing.expect(!DType.u32.isQuantized());
    try std.testing.expect(!DType.u64.isQuantized());
    try std.testing.expect(!DType.f8_e4m3.isQuantized());

    // Quantized types should be quantized
    try std.testing.expect(DType.grouped_affine_u4.isQuantized());
    try std.testing.expect(DType.grouped_affine_u8.isQuantized());
    try std.testing.expect(DType.mxfp4.isQuantized());
}

test "DType.isStandard" {
    // Standard types (enum value < 20)
    try std.testing.expect(DType.f32.isStandard());
    try std.testing.expect(DType.f64.isStandard());
    try std.testing.expect(DType.f16.isStandard());
    try std.testing.expect(DType.bf16.isStandard());
    try std.testing.expect(DType.i8.isStandard());
    try std.testing.expect(DType.i16.isStandard());
    try std.testing.expect(DType.i32.isStandard());
    try std.testing.expect(DType.i64.isStandard());
    try std.testing.expect(DType.u8.isStandard());
    try std.testing.expect(DType.u16.isStandard());
    try std.testing.expect(DType.u32.isStandard());
    try std.testing.expect(DType.u64.isStandard());

    // Quantized types are not standard (enum value >= 20)
    try std.testing.expect(!DType.grouped_affine_u4.isStandard());
    try std.testing.expect(!DType.grouped_affine_u8.isStandard());
    try std.testing.expect(!DType.mxfp4.isStandard());
}

test "DType.toFFI - standard types" {
    try std.testing.expectEqual(@as(u32, 0), DType.f32.toFFI());
    try std.testing.expectEqual(@as(u32, 1), DType.f64.toFFI());
    try std.testing.expectEqual(@as(u32, 2), DType.i32.toFFI());
    try std.testing.expectEqual(@as(u32, 3), DType.i64.toFFI());
    try std.testing.expectEqual(@as(u32, 4), DType.f16.toFFI());
    try std.testing.expectEqual(@as(u32, 5), DType.bf16.toFFI());
    try std.testing.expectEqual(@as(u32, 6), DType.i8.toFFI());
    try std.testing.expectEqual(@as(u32, 7), DType.i16.toFFI());
    try std.testing.expectEqual(@as(u32, 8), DType.u8.toFFI());
    try std.testing.expectEqual(@as(u32, 9), DType.u16.toFFI());
    try std.testing.expectEqual(@as(u32, 10), DType.u32.toFFI());
    try std.testing.expectEqual(@as(u32, 11), DType.u64.toFFI());
}

test "DType.toFFI - quantized types map to u8" {
    try std.testing.expectEqual(@as(u32, 8), DType.grouped_affine_u4.toFFI());
    try std.testing.expectEqual(@as(u32, 8), DType.grouped_affine_u8.toFFI());
    try std.testing.expectEqual(@as(u32, 8), DType.mxfp4.toFFI());
    try std.testing.expectEqual(@as(u32, 8), DType.f8_e4m3.toFFI());
}

test "DType.fromFFI - valid values" {
    try std.testing.expectEqual(DType.f32, DType.fromFFI(0).?);
    try std.testing.expectEqual(DType.f64, DType.fromFFI(1).?);
    try std.testing.expectEqual(DType.i32, DType.fromFFI(2).?);
    try std.testing.expectEqual(DType.i64, DType.fromFFI(3).?);
    try std.testing.expectEqual(DType.f16, DType.fromFFI(4).?);
    try std.testing.expectEqual(DType.bf16, DType.fromFFI(5).?);
    try std.testing.expectEqual(DType.i8, DType.fromFFI(6).?);
    try std.testing.expectEqual(DType.i16, DType.fromFFI(7).?);
    try std.testing.expectEqual(DType.u8, DType.fromFFI(8).?);
    try std.testing.expectEqual(DType.u16, DType.fromFFI(9).?);
    try std.testing.expectEqual(DType.u32, DType.fromFFI(10).?);
    try std.testing.expectEqual(DType.u64, DType.fromFFI(11).?);
}

test "DType.fromFFI - invalid values return null" {
    try std.testing.expect(DType.fromFFI(12) == null);
    try std.testing.expect(DType.fromFFI(20) == null);
    try std.testing.expect(DType.fromFFI(100) == null);
    try std.testing.expect(DType.fromFFI(255) == null);
}

test "DType.toTypeStr - standard types" {
    try std.testing.expectEqualStrings("<f4", std.mem.span(DType.f32.toTypeStr()));
    try std.testing.expectEqualStrings("<f8", std.mem.span(DType.f64.toTypeStr()));
    try std.testing.expectEqualStrings("<f2", std.mem.span(DType.f16.toTypeStr()));
    try std.testing.expectEqualStrings("<V2", std.mem.span(DType.bf16.toTypeStr()));
    try std.testing.expectEqualStrings("<i1", std.mem.span(DType.i8.toTypeStr()));
    try std.testing.expectEqualStrings("<i2", std.mem.span(DType.i16.toTypeStr()));
    try std.testing.expectEqualStrings("<i4", std.mem.span(DType.i32.toTypeStr()));
    try std.testing.expectEqualStrings("<i8", std.mem.span(DType.i64.toTypeStr()));
    try std.testing.expectEqualStrings("<u1", std.mem.span(DType.u8.toTypeStr()));
    try std.testing.expectEqualStrings("<u2", std.mem.span(DType.u16.toTypeStr()));
    try std.testing.expectEqualStrings("<u4", std.mem.span(DType.u32.toTypeStr()));
    try std.testing.expectEqualStrings("<u8", std.mem.span(DType.u64.toTypeStr()));
}

test "DType.toTypeStr - quantized types map to u1" {
    try std.testing.expectEqualStrings("<u1", std.mem.span(DType.grouped_affine_u4.toTypeStr()));
    try std.testing.expectEqualStrings("<u1", std.mem.span(DType.grouped_affine_u8.toTypeStr()));
    try std.testing.expectEqualStrings("<u1", std.mem.span(DType.mxfp4.toTypeStr()));
}

test "f32ToBf16 - basic conversions" {
    // 1.0
    try std.testing.expectEqual(@as(u16, 0x3F80), f32ToBf16(1.0));
    // -1.0
    try std.testing.expectEqual(@as(u16, 0xBF80), f32ToBf16(-1.0));
    // 0.0
    try std.testing.expectEqual(@as(u16, 0x0000), f32ToBf16(0.0));
    // -0.0
    try std.testing.expectEqual(@as(u16, 0x8000), f32ToBf16(-0.0));
    // 2.0
    try std.testing.expectEqual(@as(u16, 0x4000), f32ToBf16(2.0));
    // 0.5
    try std.testing.expectEqual(@as(u16, 0x3F00), f32ToBf16(0.5));
}

test "f32ToBf16 - special values" {
    const inf = std.math.inf(f32);
    const ninf = -std.math.inf(f32);
    const nan_val = std.math.nan(f32);

    // Infinity
    const inf_bf16 = f32ToBf16(inf);
    try std.testing.expectEqual(@as(u16, 0x7F80), inf_bf16);

    // Negative infinity
    const ninf_bf16 = f32ToBf16(ninf);
    try std.testing.expectEqual(@as(u16, 0xFF80), ninf_bf16);

    // NaN (should preserve NaN-ness, though exact bits may vary)
    const nan_bf16 = f32ToBf16(nan_val);
    // Check that exponent is all 1s (0x7F80 masked) and mantissa is non-zero
    try std.testing.expect((nan_bf16 & 0x7F80) == 0x7F80);
    try std.testing.expect((nan_bf16 & 0x007F) != 0);
}

test "bf16ToF32 - basic conversions" {
    // 1.0
    try std.testing.expectEqual(@as(f32, 1.0), bf16ToF32(0x3F80));
    // -1.0
    try std.testing.expectEqual(@as(f32, -1.0), bf16ToF32(0xBF80));
    // 0.0
    try std.testing.expectEqual(@as(f32, 0.0), bf16ToF32(0x0000));
    // -0.0
    try std.testing.expectEqual(@as(f32, -0.0), bf16ToF32(0x8000));
    // 2.0
    try std.testing.expectEqual(@as(f32, 2.0), bf16ToF32(0x4000));
    // 0.5
    try std.testing.expectEqual(@as(f32, 0.5), bf16ToF32(0x3F00));
}

test "bf16ToF32 - special values" {
    // Infinity
    try std.testing.expectEqual(std.math.inf(f32), bf16ToF32(0x7F80));
    // Negative infinity
    try std.testing.expectEqual(-std.math.inf(f32), bf16ToF32(0xFF80));
    // NaN
    const nan_result = bf16ToF32(0x7FC0);
    try std.testing.expect(std.math.isNan(nan_result));
}

test "f32 to bf16 round-trip - precision loss" {
    // BF16 has 7 bits of mantissa (vs 23 for F32), so we expect precision loss
    const original: f32 = 3.14159265359;
    const bf16_bits = f32ToBf16(original);
    const restored = bf16ToF32(bf16_bits);

    // Should be approximately equal, but not exact
    const diff = @abs(original - restored);
    // BF16 precision is about 2^-7 ≈ 0.0078 relative to value
    try std.testing.expect(diff < 0.03); // Allow some error

    // Should preserve sign and rough magnitude
    try std.testing.expect(restored > 3.0 and restored < 3.3);
}

test "f32 to bf16 round-trip - exact values" {
    // Values that can be represented exactly in bf16 (7 mantissa bits)
    const exact_values = [_]f32{ 0.0, -0.0, 1.0, -1.0, 2.0, 4.0, 0.5, 0.25, -2.0, 16.0 };

    for (exact_values) |val| {
        const bf16_bits = f32ToBf16(val);
        const restored = bf16ToF32(bf16_bits);
        try std.testing.expectEqual(val, restored);
    }
}

test "f32ToFp16 and fp16ToF32 - basic conversions" {
    // 1.0
    const fp16_1 = f32ToFp16(1.0);
    try std.testing.expectEqual(@as(f32, 1.0), fp16ToF32(fp16_1));

    // -1.0
    const fp16_neg1 = f32ToFp16(-1.0);
    try std.testing.expectEqual(@as(f32, -1.0), fp16ToF32(fp16_neg1));

    // 0.0
    const fp16_0 = f32ToFp16(0.0);
    try std.testing.expectEqual(@as(f32, 0.0), fp16ToF32(fp16_0));

    // 2.0
    const fp16_2 = f32ToFp16(2.0);
    try std.testing.expectEqual(@as(f32, 2.0), fp16ToF32(fp16_2));
}

test "f32 to fp16 round-trip" {
    const test_values = [_]f32{ 1.0, -1.0, 0.0, 2.0, 0.5, -2.0, 4.0, 0.25 };

    for (test_values) |val| {
        const fp16_bits = f32ToFp16(val);
        const restored = fp16ToF32(fp16_bits);
        try std.testing.expectEqual(val, restored);
    }
}

test "fp16ToF32 - special values" {
    // FP16 infinity: sign=0, exp=11111, mantissa=0 = 0x7C00
    try std.testing.expectEqual(std.math.inf(f32), fp16ToF32(0x7C00));

    // FP16 negative infinity: sign=1, exp=11111, mantissa=0 = 0xFC00
    try std.testing.expectEqual(-std.math.inf(f32), fp16ToF32(0xFC00));

    // FP16 NaN: sign=0, exp=11111, mantissa!=0 = 0x7E00
    const nan_result = fp16ToF32(0x7E00);
    try std.testing.expect(std.math.isNan(nan_result));
}

test "fp16x8ToF32 - vectorized conversion" {
    // Create a vector of fp16 values
    const fp16_vec = @Vector(8, u16){
        f32ToFp16(1.0),
        f32ToFp16(2.0),
        f32ToFp16(3.0),
        f32ToFp16(4.0),
        f32ToFp16(0.5),
        f32ToFp16(-1.0),
        f32ToFp16(0.0),
        f32ToFp16(8.0),
    };

    const f32_vec = fp16x8ToF32(fp16_vec);

    try std.testing.expectEqual(@as(f32, 1.0), f32_vec[0]);
    try std.testing.expectEqual(@as(f32, 2.0), f32_vec[1]);
    try std.testing.expectEqual(@as(f32, 3.0), f32_vec[2]);
    try std.testing.expectEqual(@as(f32, 4.0), f32_vec[3]);
    try std.testing.expectEqual(@as(f32, 0.5), f32_vec[4]);
    try std.testing.expectEqual(@as(f32, -1.0), f32_vec[5]);
    try std.testing.expectEqual(@as(f32, 0.0), f32_vec[6]);
    try std.testing.expectEqual(@as(f32, 8.0), f32_vec[7]);
}

test "fp16x8ToF32Bits - vectorized bit manipulation conversion" {
    if (comptime @import("builtin").cpu.arch != .x86_64) return;

    // Test with the same values as fp16x8ToF32 to verify equivalence
    const fp16_vec = @Vector(8, u16){
        f32ToFp16(1.0),
        f32ToFp16(2.0),
        f32ToFp16(3.0),
        f32ToFp16(4.0),
        f32ToFp16(0.5),
        f32ToFp16(-1.0),
        f32ToFp16(0.0),
        f32ToFp16(8.0),
    };

    const f32_vec = fp16x8ToF32Bits(8, fp16_vec);

    try std.testing.expectEqual(@as(f32, 1.0), f32_vec[0]);
    try std.testing.expectEqual(@as(f32, 2.0), f32_vec[1]);
    try std.testing.expectEqual(@as(f32, 3.0), f32_vec[2]);
    try std.testing.expectEqual(@as(f32, 4.0), f32_vec[3]);
    try std.testing.expectEqual(@as(f32, 0.5), f32_vec[4]);
    try std.testing.expectEqual(@as(f32, -1.0), f32_vec[5]);
    try std.testing.expectEqual(@as(f32, 0.0), f32_vec[6]);
    try std.testing.expectEqual(@as(f32, 8.0), f32_vec[7]);
}

test "fp16x8ToF32Bits - special values" {
    if (comptime @import("builtin").cpu.arch != .x86_64) return;

    // Test infinity, negative infinity, NaN, negative zero
    const fp16_vec = @Vector(8, u16){
        0x7C00, // +infinity
        0xFC00, // -infinity
        0x7E00, // NaN
        0x8000, // -0.0
        0x0000, // +0.0
        f32ToFp16(0.25),
        f32ToFp16(-0.125),
        f32ToFp16(16.0),
    };

    const f32_vec = fp16x8ToF32Bits(8, fp16_vec);

    try std.testing.expectEqual(std.math.inf(f32), f32_vec[0]);
    try std.testing.expectEqual(-std.math.inf(f32), f32_vec[1]);
    try std.testing.expect(std.math.isNan(f32_vec[2]));
    try std.testing.expectEqual(@as(f32, -0.0), f32_vec[3]);
    try std.testing.expectEqual(@as(f32, 0.0), f32_vec[4]);
    try std.testing.expectEqual(@as(f32, 0.25), f32_vec[5]);
    try std.testing.expectEqual(@as(f32, -0.125), f32_vec[6]);
    try std.testing.expectEqual(@as(f32, 16.0), f32_vec[7]);
}

test "fp16x8ToF32Bits - subnormal values" {
    if (comptime @import("builtin").cpu.arch != .x86_64) return;

    // FP16 subnormals: exp=0, mant!=0
    // Smallest positive subnormal: 0x0001 = 2^-24
    // Largest subnormal: 0x03FF = 1023 * 2^-24
    const fp16_vec = @Vector(8, u16){
        0x0001, // smallest positive subnormal
        0x0002, // 2 * 2^-24
        0x03FF, // largest subnormal (1023 * 2^-24)
        0x8001, // smallest negative subnormal
        0x0100, // 256 * 2^-24
        0x0000, // zero (not subnormal)
        f32ToFp16(1.0), // normal value
        0x0010, // 16 * 2^-24
    };

    const f32_vec = fp16x8ToF32Bits(8, fp16_vec);

    const eps = 1e-10;
    const scale = 5.9604644775390625e-8; // 2^-24

    try std.testing.expectApproxEqAbs(1.0 * scale, f32_vec[0], eps);
    try std.testing.expectApproxEqAbs(2.0 * scale, f32_vec[1], eps);
    try std.testing.expectApproxEqAbs(1023.0 * scale, f32_vec[2], eps);
    try std.testing.expectApproxEqAbs(-1.0 * scale, f32_vec[3], eps);
    try std.testing.expectApproxEqAbs(256.0 * scale, f32_vec[4], eps);
    try std.testing.expectEqual(@as(f32, 0.0), f32_vec[5]);
    try std.testing.expectEqual(@as(f32, 1.0), f32_vec[6]);
    try std.testing.expectApproxEqAbs(16.0 * scale, f32_vec[7], eps);
}

test "fp16x8ToF32Bits matches fp16x8ToF32 for normal values" {
    if (comptime @import("builtin").cpu.arch != .x86_64) return;

    // Exhaustive test for a range of normal values
    const test_values = [_]f32{
        1.0,  -1.0,  2.0,   -2.0,   0.5,   -0.5,   4.0,    -4.0,
        0.25, -0.25, 8.0,   -8.0,   0.125, -0.125, 16.0,   -16.0,
        3.14, -2.71, 100.0, -100.0, 0.001, -0.001, 1000.0, -1000.0,
    };

    var i: usize = 0;
    while (i + 8 <= test_values.len) : (i += 8) {
        var fp16_vec: @Vector(8, u16) = undefined;
        inline for (0..8) |j| {
            fp16_vec[j] = f32ToFp16(test_values[i + j]);
        }

        const bits_result = fp16x8ToF32Bits(8, fp16_vec);
        const cast_result = fp16x8ToF32(fp16_vec);

        inline for (0..8) |j| {
            // Both methods should produce identical results for normal values
            try std.testing.expectEqual(cast_result[j], bits_result[j]);
        }
    }
}

test "fp8e4m3ToF32 - zero" {
    // Positive zero: sign=0, exp=0, mant=0
    try std.testing.expectEqual(@as(f32, 0.0), fp8e4m3ToF32(0b00000000));

    // Negative zero: sign=1, exp=0, mant=0
    try std.testing.expectEqual(@as(f32, -0.0), fp8e4m3ToF32(0b10000000));
}

test "fp8e4m3ToF32 - normal values" {
    // Test some known conversions
    // 1.0 in E4M3: sign=0, exp=7 (biased, unbias to 0), mant=0 -> 2^0 * 1.0 = 1.0
    // Binary: 0 0111 000 = 0x38
    const one = fp8e4m3ToF32(0b00111000);
    try std.testing.expect(@abs(one - 1.0) < 0.001);

    // 2.0 in E4M3: sign=0, exp=8, mant=0 -> 2^1 * 1.0 = 2.0
    // Binary: 0 1000 000 = 0x40
    const two = fp8e4m3ToF32(0b01000000);
    try std.testing.expect(@abs(two - 2.0) < 0.001);
}

test "fp8e4m3ToF32 - subnormal values" {
    // Smallest subnormal: exp=0, mant=1 -> 2^-9 * 1 = 0.001953125
    // Binary: 0 0000 001 = 0x01
    const tiny = fp8e4m3ToF32(0b00000001);
    const expected = 1.0 / 512.0; // 2^-9
    try std.testing.expect(@abs(tiny - expected) < 0.00001);
}

test "fp8e4m3ToF32 - NaN" {
    // NaN in E4M3: exp=15, mant=7 (all 1s)
    // Binary: 0 1111 111 = 0x7F
    const nan_val = fp8e4m3ToF32(0b01111111);
    try std.testing.expect(std.math.isNan(nan_val));

    // Negative NaN
    const neg_nan = fp8e4m3ToF32(0b11111111);
    try std.testing.expect(std.math.isNan(neg_nan));
}

test "fp8e4m3ToF32 - negative values" {
    // -1.0: sign=1, exp=7, mant=0
    // Binary: 1 0111 000 = 0xB8
    const neg_one = fp8e4m3ToF32(0b10111000);
    try std.testing.expect(@abs(neg_one - (-1.0)) < 0.001);

    // -2.0: sign=1, exp=8, mant=0
    // Binary: 1 1000 000 = 0xC0
    const neg_two = fp8e4m3ToF32(0b11000000);
    try std.testing.expect(@abs(neg_two - (-2.0)) < 0.001);
}

test "dequantizeFp8E4M3ToBf16 - basic functionality" {
    const input = [_]u8{
        0b00111000, // 1.0
        0b01000000, // 2.0
        0b00000000, // 0.0
        0b10111000, // -1.0
    };
    var output: [4]u16 = undefined;

    dequantizeFp8E4M3ToBf16(&input, 1.0, &output);

    // Check that values were converted and scaled
    const restored = [_]f32{
        bf16ToF32(output[0]),
        bf16ToF32(output[1]),
        bf16ToF32(output[2]),
        bf16ToF32(output[3]),
    };

    try std.testing.expect(@abs(restored[0] - 1.0) < 0.01);
    try std.testing.expect(@abs(restored[1] - 2.0) < 0.01);
    try std.testing.expect(@abs(restored[2] - 0.0) < 0.01);
    try std.testing.expect(@abs(restored[3] - (-1.0)) < 0.01);
}

test "dequantizeFp8E4M3ToBf16 - with scaling" {
    const input = [_]u8{
        0b00111000, // 1.0
        0b01000000, // 2.0
    };
    var output: [2]u16 = undefined;

    // Scale by 2.0
    dequantizeFp8E4M3ToBf16(&input, 2.0, &output);

    const restored = [_]f32{
        bf16ToF32(output[0]),
        bf16ToF32(output[1]),
    };

    try std.testing.expect(@abs(restored[0] - 2.0) < 0.01);
    try std.testing.expect(@abs(restored[1] - 4.0) < 0.02);
}

test "bf16ToF32 - denormal values" {
    // Smallest positive denormal in bf16: 0x0001
    // This represents 2^-126 * 2^-7 = 2^-133 in f32 terms
    // When shifted to f32, it becomes 2^-126 (smallest f32 denormal range start)
    const denorm = bf16ToF32(0x0001);
    try std.testing.expect(denorm > 0.0);
    try std.testing.expect(denorm < 1e-30);
}

test "f32ToBf16 - denormal values" {
    // Very small denormal f32 value
    const tiny_f32: f32 = 1e-40;
    const bf16_bits = f32ToBf16(tiny_f32);

    // Should be flushed to zero or preserved as tiny denormal
    const restored = bf16ToF32(bf16_bits);
    try std.testing.expect(restored >= 0.0);
    try std.testing.expect(restored < 1e-30);
}

test "DType enum values - FFI compatibility" {
    // Verify that standard types have the expected enum values for FFI
    try std.testing.expectEqual(@as(u8, 0), @intFromEnum(DType.f32));
    try std.testing.expectEqual(@as(u8, 1), @intFromEnum(DType.f64));
    try std.testing.expectEqual(@as(u8, 2), @intFromEnum(DType.i32));
    try std.testing.expectEqual(@as(u8, 3), @intFromEnum(DType.i64));
    try std.testing.expectEqual(@as(u8, 4), @intFromEnum(DType.f16));
    try std.testing.expectEqual(@as(u8, 5), @intFromEnum(DType.bf16));

    // Verify quantized types start at 25
    try std.testing.expectEqual(@as(u8, 25), @intFromEnum(DType.grouped_affine_u4));
    try std.testing.expectEqual(@as(u8, 26), @intFromEnum(DType.grouped_affine_u8));
}

test "f32 and bf16 edge case - very large numbers" {
    const large: f32 = 3.4e38; // Near f32 max
    const bf16_bits = f32ToBf16(large);
    const restored = bf16ToF32(bf16_bits);

    // Should preserve large magnitude (though may lose precision)
    try std.testing.expect(restored > 1e38);
}

test "f32 and bf16 edge case - very small positive numbers" {
    const small: f32 = 1.4e-45; // Near f32 min positive
    const bf16_bits = f32ToBf16(small);
    const restored = bf16ToF32(bf16_bits);

    // May flush to zero or preserve as denormal
    try std.testing.expect(restored >= 0.0);
}
