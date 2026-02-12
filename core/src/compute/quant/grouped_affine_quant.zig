//! Shared grouped-affine quantization helpers for 4-bit and 8-bit decode paths.

const std = @import("std");
const dtype = @import("../../dtype.zig");
const simd = @import("../simd/root.zig");

pub const DType = dtype.DType;

const fp16ToF32 = dtype.fp16ToF32;
const bf16ToF32 = dtype.bf16ToF32;

pub inline fn scaleBiasToF32(dtype_tag: DType, v: u16) f32 {
    // Loader validation ensures scales are stored as F16/BF16; keep a debug assert for invariants.
    std.debug.assert(dtype_tag == .f16 or dtype_tag == .bf16);
    return switch (dtype_tag) {
        .f16 => fp16ToF32(v),
        .bf16 => bf16ToF32(v),
        else => unreachable,
    };
}

/// Extract 8 nibbles from a single u32 (for remainder handling).
/// Interleaved format: [lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3]
pub inline fn extractNibbles(word: u32) @Vector(8, f32) {
    const bytes: @Vector(4, u8) = @bitCast(word);
    const mask: @Vector(4, u8) = @splat(0x0F);
    const low_nibbles = bytes & mask;
    const high_nibbles = (bytes >> @as(@Vector(4, u8), @splat(4))) & mask;
    const nib: @Vector(8, u32) = .{
        low_nibbles[0], high_nibbles[0], low_nibbles[1], high_nibbles[1],
        low_nibbles[2], high_nibbles[2], low_nibbles[3], high_nibbles[3],
    };
    return @floatFromInt(nib);
}

/// Extract 32 nibbles from 4 U32s (16 bytes).
pub inline fn extract32NibblesToFloat(w_ptr: [*]align(1) const u32) struct {
    n0: @Vector(8, f32),
    n1: @Vector(8, f32),
    n2: @Vector(8, f32),
    n3: @Vector(8, f32),
} {
    const bytes16: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(w_ptr)).*;
    const nibbles32: @Vector(32, u8) = simd.extract32Nibbles(bytes16);

    const n0_u8: @Vector(8, u8) = @shuffle(u8, nibbles32, undefined, [8]i32{ 0, 1, 2, 3, 4, 5, 6, 7 });
    const n1_u8: @Vector(8, u8) = @shuffle(u8, nibbles32, undefined, [8]i32{ 8, 9, 10, 11, 12, 13, 14, 15 });
    const n2_u8: @Vector(8, u8) = @shuffle(u8, nibbles32, undefined, [8]i32{ 16, 17, 18, 19, 20, 21, 22, 23 });
    const n3_u8: @Vector(8, u8) = @shuffle(u8, nibbles32, undefined, [8]i32{ 24, 25, 26, 27, 28, 29, 30, 31 });

    return .{
        .n0 = @floatFromInt(@as(@Vector(8, u32), n0_u8)),
        .n1 = @floatFromInt(@as(@Vector(8, u32), n1_u8)),
        .n2 = @floatFromInt(@as(@Vector(8, u32), n2_u8)),
        .n3 = @floatFromInt(@as(@Vector(8, u32), n3_u8)),
    };
}

/// Extract 4 bytes from a u32 into f32 vector.
pub inline fn extractBytes(word: u32) @Vector(4, f32) {
    const bytes_u8: @Vector(4, u8) = .{
        @truncate((word >> 0) & 0xFF),
        @truncate((word >> 8) & 0xFF),
        @truncate((word >> 16) & 0xFF),
        @truncate((word >> 24) & 0xFF),
    };
    return @floatFromInt(bytes_u8);
}

// =============================================================================
// Unit Tests
// =============================================================================

test "extractNibbles: basic nibble extraction from u32" {
    // Test word 0x12345678
    // Byte layout (little-endian): [0x78, 0x56, 0x34, 0x12]
    // Each byte produces [low_nibble, high_nibble]
    // 0x78 → [0x8, 0x7]
    // 0x56 → [0x6, 0x5]
    // 0x34 → [0x4, 0x3]
    // 0x12 → [0x2, 0x1]
    // Interleaved format: [8, 7, 6, 5, 4, 3, 2, 1]
    const word: u32 = 0x12345678;
    const result = extractNibbles(word);

    try std.testing.expectEqual(@as(f32, 0x8), result[0]); // low nibble of 0x78
    try std.testing.expectEqual(@as(f32, 0x7), result[1]); // high nibble of 0x78
    try std.testing.expectEqual(@as(f32, 0x6), result[2]); // low nibble of 0x56
    try std.testing.expectEqual(@as(f32, 0x5), result[3]); // high nibble of 0x56
    try std.testing.expectEqual(@as(f32, 0x4), result[4]); // low nibble of 0x34
    try std.testing.expectEqual(@as(f32, 0x3), result[5]); // high nibble of 0x34
    try std.testing.expectEqual(@as(f32, 0x2), result[6]); // low nibble of 0x12
    try std.testing.expectEqual(@as(f32, 0x1), result[7]); // high nibble of 0x12
}

test "extractNibbles: all zeros" {
    // All nibbles should be 0
    const word: u32 = 0x00000000;
    const result = extractNibbles(word);

    for (0..8) |i| {
        try std.testing.expectEqual(@as(f32, 0), result[i]);
    }
}

test "extractNibbles: all ones (max nibble value)" {
    // 0xFFFFFFFF has all nibbles = 0xF (15 in decimal)
    const word: u32 = 0xFFFFFFFF;
    const result = extractNibbles(word);

    for (0..8) |i| {
        try std.testing.expectEqual(@as(f32, 0xF), result[i]);
    }
}

test "extractNibbles: alternating pattern" {
    // 0xAAAAAAAA = 0b10101010... → low nibbles = 0xA, high nibbles = 0xA
    const word: u32 = 0xAAAAAAAA;
    const result = extractNibbles(word);

    for (0..8) |i| {
        try std.testing.expectEqual(@as(f32, 0xA), result[i]);
    }
}

test "extractNibbles: single byte pattern" {
    // 0x00000001 tests least significant byte
    const word: u32 = 0x00000001;
    const result = extractNibbles(word);

    try std.testing.expectEqual(@as(f32, 0x1), result[0]); // low nibble of 0x01
    try std.testing.expectEqual(@as(f32, 0x0), result[1]); // high nibble of 0x01
    // Rest should be zero
    for (2..8) |i| {
        try std.testing.expectEqual(@as(f32, 0), result[i]);
    }
}

test "extract32NibblesToFloat: basic extraction" {
    // Create aligned test data: 4 u32 words = 16 bytes
    var data = [_]u32{ 0x10, 0x32, 0x54, 0x76 };
    const w_ptr: [*]align(1) const u32 = &data;

    const result = extract32NibblesToFloat(w_ptr);

    // Verify all four vectors are returned with correct length (8 f32 elements each)
    try std.testing.expectEqual(@as(usize, 8), @typeInfo(@TypeOf(result.n0)).vector.len);
    try std.testing.expectEqual(@as(usize, 8), @typeInfo(@TypeOf(result.n1)).vector.len);
    try std.testing.expectEqual(@as(usize, 8), @typeInfo(@TypeOf(result.n2)).vector.len);
    try std.testing.expectEqual(@as(usize, 8), @typeInfo(@TypeOf(result.n3)).vector.len);
}

test "extract32NibblesToFloat: all zeros" {
    var data = [_]u32{ 0, 0, 0, 0 };
    const w_ptr: [*]align(1) const u32 = &data;

    const result = extract32NibblesToFloat(w_ptr);

    // All nibbles should be 0
    for (0..8) |i| {
        try std.testing.expectEqual(@as(f32, 0), result.n0[i]);
        try std.testing.expectEqual(@as(f32, 0), result.n1[i]);
        try std.testing.expectEqual(@as(f32, 0), result.n2[i]);
        try std.testing.expectEqual(@as(f32, 0), result.n3[i]);
    }
}

test "extract32NibblesToFloat: max nibbles" {
    // All F nibbles (max 4-bit value)
    var data = [_]u32{ 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF };
    const w_ptr: [*]align(1) const u32 = &data;

    const result = extract32NibblesToFloat(w_ptr);

    // All nibbles should be 0xF (15)
    for (0..8) |i| {
        try std.testing.expectEqual(@as(f32, 0xF), result.n0[i]);
        try std.testing.expectEqual(@as(f32, 0xF), result.n1[i]);
        try std.testing.expectEqual(@as(f32, 0xF), result.n2[i]);
        try std.testing.expectEqual(@as(f32, 0xF), result.n3[i]);
    }
}

test "extractBytes: basic byte extraction" {
    // Word 0x12345678 in little-endian byte order: [0x78, 0x56, 0x34, 0x12]
    const word: u32 = 0x12345678;
    const result = extractBytes(word);

    try std.testing.expectEqual(@as(f32, 0x78), result[0]);
    try std.testing.expectEqual(@as(f32, 0x56), result[1]);
    try std.testing.expectEqual(@as(f32, 0x34), result[2]);
    try std.testing.expectEqual(@as(f32, 0x12), result[3]);
}

test "extractBytes: all zeros" {
    const word: u32 = 0x00000000;
    const result = extractBytes(word);

    for (0..4) |i| {
        try std.testing.expectEqual(@as(f32, 0), result[i]);
    }
}

test "extractBytes: all max values" {
    // All bytes should be 0xFF (255)
    const word: u32 = 0xFFFFFFFF;
    const result = extractBytes(word);

    for (0..4) |i| {
        try std.testing.expectEqual(@as(f32, 0xFF), result[i]);
    }
}

test "extractBytes: sequential pattern" {
    // Test that byte order is preserved correctly
    const word: u32 = 0x03020100;
    const result = extractBytes(word);

    try std.testing.expectEqual(@as(f32, 0x00), result[0]);
    try std.testing.expectEqual(@as(f32, 0x01), result[1]);
    try std.testing.expectEqual(@as(f32, 0x02), result[2]);
    try std.testing.expectEqual(@as(f32, 0x03), result[3]);
}

test "scaleBiasToF32: fp16 conversion accuracy" {
    // Test fp16 conversion
    // FP16 value 1.0: sign=0, exp=15 (0x0F), mantissa=0
    // Bit pattern: 0b0011110000000000 = 0x3C00
    const fp16_one: u16 = 0x3C00;
    const result = scaleBiasToF32(.f16, fp16_one);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result, 0.0001);

    // FP16 value 2.0: exp=16 (0x10), mantissa=0
    // Bit pattern: 0b0100000000000000 = 0x4000
    const fp16_two: u16 = 0x4000;
    const result2 = scaleBiasToF32(.f16, fp16_two);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), result2, 0.0001);

    // FP16 value 0.5: exp=14 (0x0E), mantissa=0
    // Bit pattern: 0b0011100000000000 = 0x3800
    const fp16_half: u16 = 0x3800;
    const result3 = scaleBiasToF32(.f16, fp16_half);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), result3, 0.0001);
}

test "scaleBiasToF32: bf16 conversion accuracy" {
    // BF16 is upper 16 bits of FP32
    // BF16 value 1.0: FP32 0x3F800000 >> 16 = 0x3F80
    const bf16_one: u16 = 0x3F80;
    const result = scaleBiasToF32(.bf16, bf16_one);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result, 0.0001);

    // BF16 value 2.0: FP32 0x40000000 >> 16 = 0x4000
    const bf16_two: u16 = 0x4000;
    const result2 = scaleBiasToF32(.bf16, bf16_two);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), result2, 0.0001);

    // BF16 value 0.5: FP32 0x3F000000 >> 16 = 0x3F00
    const bf16_half: u16 = 0x3F00;
    const result3 = scaleBiasToF32(.bf16, bf16_half);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), result3, 0.0001);
}

test "scaleBiasToF32: zero values" {
    // FP16 zero
    const fp16_zero: u16 = 0x0000;
    const result1 = scaleBiasToF32(.f16, fp16_zero);
    try std.testing.expectEqual(@as(f32, 0.0), result1);

    // BF16 zero
    const bf16_zero: u16 = 0x0000;
    const result2 = scaleBiasToF32(.bf16, bf16_zero);
    try std.testing.expectEqual(@as(f32, 0.0), result2);
}

test "scaleBiasToF32: negative values" {
    // FP16 negative 1.0: sign bit set, exp=15, mantissa=0
    // Bit pattern: 0b1011110000000000 = 0xBC00
    const fp16_neg_one: u16 = 0xBC00;
    const result1 = scaleBiasToF32(.f16, fp16_neg_one);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), result1, 0.0001);

    // BF16 negative 1.0: FP32 0xBF800000 >> 16 = 0xBF80
    const bf16_neg_one: u16 = 0xBF80;
    const result2 = scaleBiasToF32(.bf16, bf16_neg_one);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), result2, 0.0001);
}

test "scaleBiasToF32 quantized dequantization" {
    // Test the full dequantization formula: output = quantized_value * scale + bias
    // This simulates how extractNibbles output would be used

    // Extract nibbles from a word
    const word: u32 = 0x00000001; // nibbles: [1, 0, 0, 0, 0, 0, 0, 0]
    const nibbles = extractNibbles(word);

    // Apply scale and bias (using simple f32 values for testing)
    const scale: f32 = 2.0;
    const bias: f32 = 1.0;

    // First nibble is 1, so result should be: 1 * 2.0 + 1.0 = 3.0
    const result = nibbles[0] * scale + bias;
    try std.testing.expectEqual(@as(f32, 3.0), result);

    // Second nibble is 0, so result should be: 0 * 2.0 + 1.0 = 1.0
    const result2 = nibbles[1] * scale + bias;
    try std.testing.expectEqual(@as(f32, 1.0), result2);
}

test "scaleBiasToF32 fp16 scale bias" {
    // Full integration test: nibble extraction + fp16 scale/bias conversion + dequantization
    const word: u32 = 0x00000005; // nibbles: [5, 0, 0, 0, 0, 0, 0, 0]
    const nibbles = extractNibbles(word);

    // FP16 scale = 0.5 (0x3800)
    const scale_fp16: u16 = 0x3800;
    const scale = scaleBiasToF32(.f16, scale_fp16);

    // FP16 bias = 1.0 (0x3C00)
    const bias_fp16: u16 = 0x3C00;
    const bias = scaleBiasToF32(.f16, bias_fp16);

    // First nibble is 5, result should be: 5 * 0.5 + 1.0 = 3.5
    const result = nibbles[0] * scale + bias;
    try std.testing.expectApproxEqAbs(@as(f32, 3.5), result, 0.01);
}

test "scaleBiasToF32 bf16 negative bias" {
    // Test with bf16 and negative bias
    const word: u32 = 0x000000FF; // nibbles: [F, F, 0, 0, 0, 0, 0, 0]
    const nibbles = extractNibbles(word);

    // BF16 scale = 0.5 (0x3F00)
    const scale_bf16: u16 = 0x3F00;
    const scale = scaleBiasToF32(.bf16, scale_bf16);

    // BF16 bias = -1.0 (0xBF80)
    const bias_bf16: u16 = 0xBF80;
    const bias = scaleBiasToF32(.bf16, bias_bf16);

    // First nibble is F (15), result should be: 15 * 0.5 + (-1.0) = 6.5
    const result = nibbles[0] * scale + bias;
    try std.testing.expectApproxEqAbs(@as(f32, 6.5), result, 0.01);
}

test "extractBytes byte extraction" {
    // Test 8-bit quantization path (grouped_affine_u8)
    const word: u32 = 0x12345678;
    const bytes = extractBytes(word);

    // Apply scale and bias to first byte (0x78 = 120)
    const scale: f32 = 0.1;
    const bias: f32 = -5.0;

    // Result should be: 120 * 0.1 + (-5.0) = 7.0
    const result = bytes[0] * scale + bias;
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), result, 0.001);
}

test "extractNibbles max scale max nibble" {
    // Test edge case: largest nibble value (15) with large scale
    const word: u32 = 0x000000FF; // nibbles: [F, F, ...]
    const nibbles = extractNibbles(word);

    // Large scale to test numerical stability
    // FP16 max normal value is ~65504, use something reasonable like 100.0
    // FP16 100.0: exp=21 (0x15), mantissa ~0x240 → 0x5640
    const scale: f32 = 100.0;
    const bias: f32 = 0.0;

    // Nibble F (15) * 100.0 = 1500.0
    const result = nibbles[0] * scale + bias;
    try std.testing.expectApproxEqAbs(@as(f32, 1500.0), result, 0.1);
}

test "scaleBiasToF32 zero scale" {
    // Test with zero scale (should produce bias regardless of nibble value)
    const word: u32 = 0x000000FF; // nibbles: [F, F, ...]
    const nibbles = extractNibbles(word);

    const scale: f32 = 0.0;
    const bias: f32 = 42.0;

    // Result should be bias only
    const result = nibbles[0] * scale + bias;
    try std.testing.expectEqual(@as(f32, 42.0), result);
}
