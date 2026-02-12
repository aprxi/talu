//! SIMD vector types and width detection.
//!
//! Provides portable vector types that adapt to AVX2, SSE, or NEON at comptime.
//! Re-exports platform-specific intrinsics from x86.zig and arm.zig.

const std = @import("std");
const builtin = @import("builtin");

/// Preferred vector width in bits for the target CPU.
/// We cap at 256-bit (AVX2) because:
/// 1. AVX-512 causes frequency throttling on many CPUs
/// 2. Small head_dim (64-128) doesn't benefit from wider vectors
/// 3. AVX2 is the sweet spot for LLM inference workloads
pub const vector_bit_width: comptime_int = blk: {
    const arch = builtin.cpu.arch;
    if (arch == .x86_64 or arch == .x86) {
        // Cap at AVX2 (256-bit) - AVX-512 often slower due to throttling
        if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx2) or
            std.Target.x86.featureSetHas(builtin.cpu.features, .avx))
        {
            break :blk 256;
        }
        // SSE (128-bit vectors)
        break :blk 128;
    } else if (arch == .aarch64 or arch == .arm) {
        // NEON is 128-bit
        break :blk 128;
    } else {
        // Default for other architectures
        break :blk 128;
    }
};

/// Preferred f32 vector length (number of elements)
pub const f32_vec_len: comptime_int = vector_bit_width / 32;

/// Type alias for f32 SIMD vector at the detected width
pub const F32Vec = @Vector(f32_vec_len, f32);

// =============================================================================
// Architecture-specific intrinsics
// =============================================================================
// Re-export from arch-specific modules for quantized dot products.
// Each architecture module provides the same interface:
// - mulSumI8Pairs: i8×i8 → i32 dot product for Q8 quantization
// - mulSumU8I8WithYSum: u8×i8 dot product with y-sum for Q4 quantization

// Architecture-specific implementations
pub const arm = @import("arm.zig");
pub const x86 = @import("x86.zig");

const impl = if (builtin.cpu.arch == .aarch64) arm else x86;

// Re-export intrinsics used by quantized matmul
pub const mulSumI8Pairs = impl.mulSumI8Pairs;
pub const mulSumU8I8WithYSum = impl.mulSumU8I8WithYSum;

// Re-export F16 conversion intrinsic (hardware vcvtph2ps on x86 with F16C, portable fallback otherwise)
pub const cvtph2ps = x86.cvtph2ps;

// Re-export nibble extraction for grouped-affine u4 matmul
pub const extract32Nibbles = if (builtin.cpu.arch == .aarch64)
    arm.extract32Nibbles
else
    // Portable implementation - interleaved format [lo0, hi0, lo1, hi1, ...]
    struct {
        pub inline fn extract32Nibbles(bytes: @Vector(16, u8)) @Vector(32, u8) {
            const lo = bytes & @as(@Vector(16, u8), @splat(0x0F));
            const hi = bytes >> @as(@Vector(16, u8), @splat(4));
            return @shuffle(u8, lo, hi, @Vector(32, i32){
                0,  ~@as(i32, 0),  1,  ~@as(i32, 1),  2,  ~@as(i32, 2),  3,  ~@as(i32, 3),
                4,  ~@as(i32, 4),  5,  ~@as(i32, 5),  6,  ~@as(i32, 6),  7,  ~@as(i32, 7),
                8,  ~@as(i32, 8),  9,  ~@as(i32, 9),  10, ~@as(i32, 10), 11, ~@as(i32, 11),
                12, ~@as(i32, 12), 13, ~@as(i32, 13), 14, ~@as(i32, 14), 15, ~@as(i32, 15),
            });
        }
    }.extract32Nibbles;

// =============================================================================
// Tests
// =============================================================================

test "simd width detection" {
    const width = vector_bit_width;
    try std.testing.expect(width == 128 or width == 256);
    try std.testing.expectEqual(width / 32, f32_vec_len);
    try std.testing.expectEqual(f32_vec_len, @typeInfo(F32Vec).vector.len);
}

test "extract32Nibbles extracts low and high nibbles" {
    // Input: 16 bytes where each byte has distinct low and high nibbles
    // Byte 0x12 has low=2, high=1
    const input: @Vector(16, u8) = .{
        0x10, 0x32, 0x54, 0x76, // bytes 0-3
        0x98, 0xBA, 0xDC, 0xFE, // bytes 4-7
        0x01, 0x23, 0x45, 0x67, // bytes 8-11
        0x89, 0xAB, 0xCD, 0xEF, // bytes 12-15
    };

    const result = extract32Nibbles(input);

    // Output format is interleaved: [lo0, hi0, lo1, hi1, ...]
    // For byte 0x10: lo=0, hi=1
    try std.testing.expectEqual(@as(u8, 0x0), result[0]); // lo nibble of byte 0
    try std.testing.expectEqual(@as(u8, 0x1), result[1]); // hi nibble of byte 0

    // For byte 0x32: lo=2, hi=3
    try std.testing.expectEqual(@as(u8, 0x2), result[2]); // lo nibble of byte 1
    try std.testing.expectEqual(@as(u8, 0x3), result[3]); // hi nibble of byte 1

    // For byte 0x54: lo=4, hi=5
    try std.testing.expectEqual(@as(u8, 0x4), result[4]);
    try std.testing.expectEqual(@as(u8, 0x5), result[5]);

    // For byte 0xFE: lo=E, hi=F
    try std.testing.expectEqual(@as(u8, 0xE), result[14]);
    try std.testing.expectEqual(@as(u8, 0xF), result[15]);
}

test "extract32Nibbles handles zero input" {
    const input: @Vector(16, u8) = @splat(0x00);
    const result = extract32Nibbles(input);

    // All nibbles should be 0
    for (0..32) |idx| {
        try std.testing.expectEqual(@as(u8, 0), result[idx]);
    }
}

test "extract32Nibbles handles max nibbles" {
    // 0xFF has both nibbles = 0xF
    const input: @Vector(16, u8) = @splat(0xFF);
    const result = extract32Nibbles(input);

    // All nibbles should be 0xF
    for (0..32) |idx| {
        try std.testing.expectEqual(@as(u8, 0xF), result[idx]);
    }
}

test "extract32Nibbles produces 32 elements from 16 bytes" {
    const input: @Vector(16, u8) = .{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    const result = extract32Nibbles(input);

    // Result should have exactly 32 elements
    try std.testing.expectEqual(@as(usize, 32), @typeInfo(@TypeOf(result)).vector.len);
}

// =============================================================================
// Additional Comprehensive Tests
// =============================================================================

test "mulSumI8Pairs: basic signed×signed multiplication" {
    const x: @Vector(32, i8) = .{
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    };
    const y: @Vector(32, i8) = @splat(1);

    const result = mulSumI8Pairs(x, y);

    // Each group of 4 consecutive pairs sums to a specific value
    // This tests the basic accumulation pattern
    const total: i32 = @reduce(.Add, result);
    try std.testing.expectEqual(@as(i32, 272), total); // 2*(1+2+...+16) = 2*136 = 272
}

test "mulSumI8Pairs: negative values" {
    const x: @Vector(32, i8) = @splat(-5);
    const y: @Vector(32, i8) = @splat(3);

    const result = mulSumI8Pairs(x, y);

    // Each i32 accumulates 4 products: -5*3*4 = -60
    inline for (0..8) |i| {
        try std.testing.expectEqual(@as(i32, -60), result[i]);
    }
}

test "mulSumI8Pairs: mixed positive and negative" {
    const x: @Vector(32, i8) = @splat(0);
    var x_mut = x;
    for (0..32) |i| {
        x_mut[i] = if (i % 2 == 0) 5 else -5;
    }
    const y: @Vector(32, i8) = @splat(2);

    const result = mulSumI8Pairs(x_mut, y);

    // Each group: (5*2 + -5*2) * 2 pairs = 0
    inline for (0..8) |i| {
        try std.testing.expectEqual(@as(i32, 0), result[i]);
    }
}

test "mulSumI8Pairs: zero input" {
    const x: @Vector(32, i8) = @splat(0);
    const y: @Vector(32, i8) = @splat(100);

    const result = mulSumI8Pairs(x, y);

    inline for (0..8) |i| {
        try std.testing.expectEqual(@as(i32, 0), result[i]);
    }
}

test "mulSumI8Pairs: max positive values" {
    const x: @Vector(32, i8) = @splat(127);
    const y: @Vector(32, i8) = @splat(1);

    const result = mulSumI8Pairs(x, y);

    // Each group: 127*4 = 508
    inline for (0..8) |i| {
        try std.testing.expectEqual(@as(i32, 508), result[i]);
    }
}

test "mulSumI8Pairs: min negative values" {
    const x: @Vector(32, i8) = @splat(-128);
    const y: @Vector(32, i8) = @splat(1);

    const result = mulSumI8Pairs(x, y);

    // Each group: -128*4 = -512
    inline for (0..8) |i| {
        try std.testing.expectEqual(@as(i32, -512), result[i]);
    }
}

test "mulSumI8Pairs: alternating pattern" {
    const x: @Vector(32, i8) = @splat(0);
    var x_mut = x;
    for (0..32) |i| {
        x_mut[i] = @intCast((i % 8) + 1);
    }
    const y: @Vector(32, i8) = @splat(1);

    const result = mulSumI8Pairs(x_mut, y);

    // Pattern repeats every 8 elements: 1,2,3,4,5,6,7,8
    // Each i32 accumulates 4 consecutive values
    // First group (0-3): 1+2+3+4 = 10
    // Second group (4-7): 5+6+7+8 = 26
    try std.testing.expectEqual(@as(i32, 10), result[0]);
    try std.testing.expectEqual(@as(i32, 26), result[1]);
    try std.testing.expectEqual(@as(i32, 10), result[2]);
    try std.testing.expectEqual(@as(i32, 26), result[3]);
}

test "mulSumU8I8WithYSum: basic unsigned×signed" {
    const q4: @Vector(32, u8) = @splat(8);
    const y: @Vector(32, i8) = @splat(2);

    const result = mulSumU8I8WithYSum(q4, y);

    // dot: 8*2*32 = 512
    const total_dot: i32 = @reduce(.Add, result.dot);
    try std.testing.expectEqual(@as(i32, 512), total_dot);

    // sum_y: 2*32 = 64
    try std.testing.expectEqual(@as(i32, 64), result.sum_y);
}

test "mulSumU8I8WithYSum: Q4 range [0..15]" {
    const q4: @Vector(32, u8) = @splat(0);
    var q4_mut = q4;
    for (0..32) |i| {
        q4_mut[i] = @intCast(i % 16);
    }
    const y: @Vector(32, i8) = @splat(1);

    const result = mulSumU8I8WithYSum(q4_mut, y);

    // sum of nibbles: 2*(0+1+2+...+15) = 2*120 = 240
    const total_dot: i32 = @reduce(.Add, result.dot);
    try std.testing.expectEqual(@as(i32, 240), total_dot);

    // sum_y: 32
    try std.testing.expectEqual(@as(i32, 32), result.sum_y);
}

test "mulSumU8I8WithYSum: negative weights" {
    const q4: @Vector(32, u8) = @splat(15); // Max nibble
    const y: @Vector(32, i8) = @splat(-2);

    const result = mulSumU8I8WithYSum(q4, y);

    // dot: 15*(-2)*32 = -960
    const total_dot: i32 = @reduce(.Add, result.dot);
    try std.testing.expectEqual(@as(i32, -960), total_dot);

    // sum_y: -2*32 = -64
    try std.testing.expectEqual(@as(i32, -64), result.sum_y);
}

test "mulSumU8I8WithYSum: zero nibbles" {
    const q4: @Vector(32, u8) = @splat(0);
    const y: @Vector(32, i8) = @splat(10);

    const result = mulSumU8I8WithYSum(q4, y);

    // dot should be 0
    const total_dot: i32 = @reduce(.Add, result.dot);
    try std.testing.expectEqual(@as(i32, 0), total_dot);

    // sum_y: 10*32 = 320
    try std.testing.expectEqual(@as(i32, 320), result.sum_y);
}

test "mulSumU8I8WithYSum: zero weights" {
    const q4: @Vector(32, u8) = @splat(15);
    const y: @Vector(32, i8) = @splat(0);

    const result = mulSumU8I8WithYSum(q4, y);

    // dot should be 0
    const total_dot: i32 = @reduce(.Add, result.dot);
    try std.testing.expectEqual(@as(i32, 0), total_dot);

    // sum_y: 0
    try std.testing.expectEqual(@as(i32, 0), result.sum_y);
}

test "mulSumU8I8WithYSum: Q4 offset correction identity" {
    // Test: (q-8)*y = q*y - 8*sum(y)
    const q4: @Vector(32, u8) = @splat(0);
    var q4_mut = q4;
    for (0..32) |i| {
        q4_mut[i] = @intCast((i % 16));
    }
    const y: @Vector(32, i8) = @splat(3);

    const result = mulSumU8I8WithYSum(q4_mut, y);

    const total_dot: i32 = @reduce(.Add, result.dot);
    const corrected = total_dot - 8 * result.sum_y;

    // Manual calculation: sum((q-8)*y) where q cycles through 0..15 twice
    // = 3 * sum(q-8) = 3 * 2 * ((0-8)+(1-8)+...+(15-8))
    // = 3 * 2 * (-8-7-6-5-4-3-2-1+0+1+2+3+4+5+6+7)
    // = 3 * 2 * (-8) = -48
    try std.testing.expectEqual(@as(i32, -48), corrected);
}

test "mulSumU8I8WithYSum: large values" {
    const q4: @Vector(32, u8) = @splat(15);
    const y: @Vector(32, i8) = @splat(127);

    const result = mulSumU8I8WithYSum(q4, y);

    // dot: 15*127*32 = 60960
    const total_dot: i32 = @reduce(.Add, result.dot);
    try std.testing.expectEqual(@as(i32, 60960), total_dot);

    // sum_y: 127*32 = 4064
    try std.testing.expectEqual(@as(i32, 4064), result.sum_y);
}

test "mulSumU8I8WithYSum: mixed nibbles and negative weights" {
    const q4: @Vector(32, u8) = @splat(0);
    var q4_mut = q4;
    for (0..32) |i| {
        q4_mut[i] = @intCast(i % 2 * 15); // Alternating 0, 15, 0, 15...
    }
    const y: @Vector(32, i8) = @splat(-1);

    const result = mulSumU8I8WithYSum(q4_mut, y);

    // 16 zeros and 16 fifteens: 16*0*(-1) + 16*15*(-1) = -240
    const total_dot: i32 = @reduce(.Add, result.dot);
    try std.testing.expectEqual(@as(i32, -240), total_dot);

    // sum_y: -1*32 = -32
    try std.testing.expectEqual(@as(i32, -32), result.sum_y);
}

test "F32Vec: correct type and size" {
    const vec_type_info = @typeInfo(F32Vec);
    try std.testing.expect(vec_type_info == .vector);
    try std.testing.expectEqual(f32, vec_type_info.vector.child);
    try std.testing.expectEqual(f32_vec_len, vec_type_info.vector.len);
}

test "vector width constants: relationship" {
    // Verify the mathematical relationship
    try std.testing.expectEqual(vector_bit_width / 32, f32_vec_len);

    // Verify reasonable bounds
    try std.testing.expect(vector_bit_width >= 128);
    try std.testing.expect(vector_bit_width <= 256);
    try std.testing.expect(f32_vec_len >= 4);
    try std.testing.expect(f32_vec_len <= 8);
}

test "extract32Nibbles: sequential bytes" {
    const input: @Vector(16, u8) = .{
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
        0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF,
    };

    const result = extract32Nibbles(input);

    // 0x00: lo=0, hi=0
    try std.testing.expectEqual(@as(u8, 0x0), result[0]);
    try std.testing.expectEqual(@as(u8, 0x0), result[1]);

    // 0x11: lo=1, hi=1
    try std.testing.expectEqual(@as(u8, 0x1), result[2]);
    try std.testing.expectEqual(@as(u8, 0x1), result[3]);

    // 0xFF: lo=F, hi=F
    try std.testing.expectEqual(@as(u8, 0xF), result[30]);
    try std.testing.expectEqual(@as(u8, 0xF), result[31]);
}

test "extract32Nibbles: alternating nibbles" {
    // Create pattern where low and high nibbles differ
    const input: @Vector(16, u8) = @splat(0xA5); // 1010 0101

    const result = extract32Nibbles(input);

    // All lo nibbles should be 5
    // All hi nibbles should be A (10)
    for (0..16) |i| {
        try std.testing.expectEqual(@as(u8, 0x5), result[i * 2]); // lo
        try std.testing.expectEqual(@as(u8, 0xA), result[i * 2 + 1]); // hi
    }
}

test "extract32Nibbles: boundary nibbles" {
    // Test nibbles at min (0x0) and max (0xF) values
    const input: @Vector(16, u8) = @splat(0);
    var input_mut = input;
    input_mut[0] = 0x0F; // lo=F, hi=0
    input_mut[1] = 0xF0; // lo=0, hi=F

    const result = extract32Nibbles(input_mut);

    // First byte (0x0F)
    try std.testing.expectEqual(@as(u8, 0xF), result[0]); // lo
    try std.testing.expectEqual(@as(u8, 0x0), result[1]); // hi

    // Second byte (0xF0)
    try std.testing.expectEqual(@as(u8, 0x0), result[2]); // lo
    try std.testing.expectEqual(@as(u8, 0xF), result[3]); // hi
}

test "extract32Nibbles: preserves nibble ordering" {
    // Test that the interleaved format is correct for MLX uint32 packing
    // Format: [lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3]
    // When packed into uint32: hi3<<28 | lo3<<24 | hi2<<20 | lo2<<16 | hi1<<12 | lo1<<8 | hi0<<4 | lo0
    const input: @Vector(16, u8) = .{
        0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    };

    const result = extract32Nibbles(input);

    // First 8 bytes produce 16 nibbles
    const expected_first_16: [16]u8 = .{
        1, 0, 3, 2, 5, 4, 7, 6, // bytes 0-3
        9, 8, 11, 10, 13, 12, 15, 14, // bytes 4-7
    };

    for (expected_first_16, 0..) |expected, i| {
        try std.testing.expectEqual(expected, result[i]);
    }
}

test "mulSumI8Pairs and mulSumU8I8WithYSum: consistency for valid range" {
    // For values in [0..127], both signed and unsigned interpretations should work
    const data: @Vector(32, u8) = @splat(0);
    var data_mut = data;
    for (0..32) |i| {
        data_mut[i] = @intCast(i % 100); // Keep in valid i8 range
    }
    const weights: @Vector(32, i8) = @splat(2);

    // Convert to signed for mulSumI8Pairs
    const data_signed: @Vector(32, i8) = @bitCast(data_mut);

    const result_i8 = mulSumI8Pairs(data_signed, weights);
    const result_u8 = mulSumU8I8WithYSum(data_mut, weights);

    // The dot products should match
    const total_i8: i32 = @reduce(.Add, result_i8);
    const total_u8: i32 = @reduce(.Add, result_u8.dot);

    try std.testing.expectEqual(total_i8, total_u8);
}

test "vector operations: reduction and broadcasting" {
    // Test that SIMD results can be properly reduced
    const x: @Vector(32, i8) = @splat(5);
    const y: @Vector(32, i8) = @splat(3);

    const result = mulSumI8Pairs(x, y);

    // Each lane should be 5*3*4 = 60
    inline for (0..8) |i| {
        try std.testing.expectEqual(@as(i32, 60), result[i]);
    }

    // Total reduction
    const total: i32 = @reduce(.Add, result);
    try std.testing.expectEqual(@as(i32, 480), total); // 60 * 8
}

test "mulSumI8Pairs: accumulation precision" {
    // Test that products don't overflow during accumulation
    const x: @Vector(32, i8) = @splat(100);
    const y: @Vector(32, i8) = @splat(100);

    const result = mulSumI8Pairs(x, y);

    // Each group: 100*100*4 = 40000 (fits in i32)
    inline for (0..8) |i| {
        try std.testing.expectEqual(@as(i32, 40000), result[i]);
    }
}

test "mulSumU8I8WithYSum: sum_y computation" {
    // Test that sum_y is computed correctly for various patterns
    const q4: @Vector(32, u8) = @splat(1);

    // Alternating positive/negative
    const y: @Vector(32, i8) = @splat(0);
    var y_mut = y;
    for (0..32) |i| {
        y_mut[i] = if (i % 2 == 0) 10 else -10;
    }

    const result = mulSumU8I8WithYSum(q4, y_mut);

    // sum_y should be 0 (16*10 + 16*(-10) = 0)
    try std.testing.expectEqual(@as(i32, 0), result.sum_y);
}

test "extract32Nibbles: all nibble values 0-15" {
    // Create a byte with each nibble value from 0-15
    const input: @Vector(16, u8) = @splat(0);
    var input_mut = input;
    for (0..8) |i| {
        const lo: u8 = @intCast(i * 2);
        const hi: u8 = @intCast(i * 2 + 1);
        input_mut[i] = (hi << 4) | lo;
    }

    const result = extract32Nibbles(input_mut);

    // Verify all nibble values 0-15 appear correctly
    for (0..8) |i| {
        const expected_lo: u8 = @intCast(i * 2);
        const expected_hi: u8 = @intCast(i * 2 + 1);
        try std.testing.expectEqual(expected_lo, result[i * 2]);
        try std.testing.expectEqual(expected_hi, result[i * 2 + 1]);
    }
}
