//! ARM/NEON intrinsics for quantized dot products and bf16 operations.
//!
//! Uses SDOT instruction on ARMv8.2-A+ (M1/M2/M3/M4, A76+) for optimal
//! performance, with scalar fallback for older architectures.
//!
//! Uses BFDOT instruction on ARMv8.6-A+ (M2/M3/M4) for bf16 dot products.
//! Runtime detection is used to ensure binaries work across all Apple Silicon.

const std = @import("std");
const builtin = @import("builtin");

// Detect dotprod feature (available on M1/M2/M3/M4, A76+)
const has_dotprod = blk: {
    if (builtin.cpu.arch != .aarch64) break :blk false;
    break :blk std.Target.aarch64.featureSetHas(builtin.cpu.features, .dotprod);
};

/// Runtime bf16 feature detection (cached).
/// On macOS aarch64, queries hw.optional.arm.FEAT_BF16 via sysctlbyname.
/// Result is cached after first call for zero overhead in hot paths.
pub var has_bf16: bool = false;
var bf16_detected: bool = false;

pub fn detectBf16() bool {
    if (bf16_detected) return has_bf16;

    if (comptime builtin.cpu.arch != .aarch64) {
        bf16_detected = true;
        has_bf16 = false;
        return false;
    }

    if (comptime builtin.os.tag == .macos) {
        const c = @cImport(@cInclude("sys/sysctl.h"));
        var value: c_int = 0;
        var size: usize = @sizeOf(c_int);
        const result = c.sysctlbyname("hw.optional.arm.FEAT_BF16", &value, &size, null, 0);
        has_bf16 = (result == 0 and value != 0);
    } else {
        // Linux/other: use compile-time detection
        // TODO: Add Linux runtime detection via getauxval(AT_HWCAP2) & HWCAP2_BF16
        has_bf16 = std.Target.aarch64.featureSetHas(builtin.cpu.features, .bf16);
    }

    bf16_detected = true;
    return has_bf16;
}

/// Calculate dot product of two 128-bit vectors (16x i8) accumulating into 4x i32.
/// Uses SDOT instruction if available (M1/M2/M3/M4), falls back to manual calculation.
inline fn dot128(a: @Vector(16, i8), b: @Vector(16, i8)) @Vector(4, i32) {
    if (comptime has_dotprod and builtin.cpu.arch == .aarch64) {
        // SDOT: signed 8-bit integer dot product
        // sdot Vd.4S, Vn.16B, Vm.16B
        // Each of the 4 output i32 lanes gets the sum of 4 adjacent i8*i8 products
        var acc: @Vector(4, i32) = @splat(0);
        asm ("sdot %[acc].4s, %[a].16b, %[b].16b"
            : [acc] "+w" (acc),
            : [a] "w" (a),
              [b] "w" (b),
        );
        return acc;
    } else {
        // Portable implementation for targets without SDOT
        // 1. Widen to i16 and multiply
        const a_lo: @Vector(8, i16) = @shuffle(i8, a, undefined, @Vector(8, i32){ 0, 1, 2, 3, 4, 5, 6, 7 });
        const a_hi: @Vector(8, i16) = @shuffle(i8, a, undefined, @Vector(8, i32){ 8, 9, 10, 11, 12, 13, 14, 15 });
        const b_lo: @Vector(8, i16) = @shuffle(i8, b, undefined, @Vector(8, i32){ 0, 1, 2, 3, 4, 5, 6, 7 });
        const b_hi: @Vector(8, i16) = @shuffle(i8, b, undefined, @Vector(8, i32){ 8, 9, 10, 11, 12, 13, 14, 15 });

        const prod_lo = a_lo * b_lo;
        const prod_hi = a_hi * b_hi;

        // 2. Sum groups of 4 adjacent products into i32
        var res: @Vector(4, i32) = undefined;
        comptime var lane_idx = 0;
        inline while (lane_idx < 4) : (lane_idx += 1) {
            // Each output is sum of 4 products: lo[i*2..i*2+1] + hi[i*2..i*2+1]
            res[lane_idx] = @as(i32, prod_lo[lane_idx * 2]) + @as(i32, prod_lo[lane_idx * 2 + 1]) +
                @as(i32, prod_hi[lane_idx * 2]) + @as(i32, prod_hi[lane_idx * 2 + 1]);
        }
        return res;
    }
}

/// Calculate unsigned×signed dot product of two 128-bit vectors.
/// This matches the behavior of x86 maddubsw + pmaddwd for Q4 quantization.
inline fn dotU8I8_128(a: @Vector(16, u8), b: @Vector(16, i8)) @Vector(4, i32) {
    if (comptime has_dotprod and builtin.cpu.arch == .aarch64) {
        // For unsigned×signed, we can still use SDOT if values fit in signed range
        // Q4 nibbles are [0..15] which fits in i8, so cast and use signed dot
        const a_signed: @Vector(16, i8) = @bitCast(a);
        var acc: @Vector(4, i32) = @splat(0);
        asm ("sdot %[acc].4s, %[a].16b, %[b].16b"
            : [acc] "+w" (acc),
            : [a] "w" (a_signed),
              [b] "w" (b),
        );
        return acc;
    } else {
        // Portable implementation: widen and multiply
        const a_lo: @Vector(8, i16) = @intCast(@shuffle(u8, a, undefined, @Vector(8, i32){ 0, 1, 2, 3, 4, 5, 6, 7 }));
        const a_hi: @Vector(8, i16) = @intCast(@shuffle(u8, a, undefined, @Vector(8, i32){ 8, 9, 10, 11, 12, 13, 14, 15 }));
        const b_lo: @Vector(8, i16) = @shuffle(i8, b, undefined, @Vector(8, i32){ 0, 1, 2, 3, 4, 5, 6, 7 });
        const b_hi: @Vector(8, i16) = @shuffle(i8, b, undefined, @Vector(8, i32){ 8, 9, 10, 11, 12, 13, 14, 15 });

        const prod_lo = a_lo * b_lo;
        const prod_hi = a_hi * b_hi;

        var res: @Vector(4, i32) = undefined;
        comptime var lane_idx = 0;
        inline while (lane_idx < 4) : (lane_idx += 1) {
            res[lane_idx] = @as(i32, prod_lo[lane_idx * 2]) + @as(i32, prod_lo[lane_idx * 2 + 1]) +
                @as(i32, prod_hi[lane_idx * 2]) + @as(i32, prod_hi[lane_idx * 2 + 1]);
        }
        return res;
    }
}

/// Combined multiply-sum for i8×i8 → i32
/// Input: 32 bytes (matching x86 256-bit block size)
/// Output: 8x i32 sums (each is sum of 4 adjacent i8×i8 products)
///
/// This matches the interface of the x86 version which uses pmaddubsw + pmaddwd.
pub inline fn mulSumI8Pairs(x: @Vector(32, i8), y: @Vector(32, i8)) @Vector(8, i32) {
    // Split 256-bit input into two 128-bit NEON operations
    const x_lo: @Vector(16, i8) = @shuffle(i8, x, undefined, @Vector(16, i32){
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    });
    const x_hi: @Vector(16, i8) = @shuffle(i8, x, undefined, @Vector(16, i32){
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    });
    const y_lo: @Vector(16, i8) = @shuffle(i8, y, undefined, @Vector(16, i32){
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    });
    const y_hi: @Vector(16, i8) = @shuffle(i8, y, undefined, @Vector(16, i32){
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    });

    const res_lo = dot128(x_lo, y_lo);
    const res_hi = dot128(x_hi, y_hi);

    // Join the two 4xi32 results into 8xi32
    return @shuffle(i32, res_lo, res_hi, @Vector(8, i32){ 0, 1, 2, 3, -1, -2, -3, -4 });
}

/// Fast nibble extraction using ARM NEON TBL (table lookup) instruction.
/// Extracts 32 nibbles (4 bits each) from 16 bytes into 32 bytes.
/// Format: interleaved [lo0, hi0, lo1, hi1, ...] matching MLX's uint32 packing
/// On Apple Silicon this is ~10x faster than scalar extraction.
pub inline fn extract32Nibbles(bytes: @Vector(16, u8)) @Vector(32, u8) {
    const lo = bytes & @as(@Vector(16, u8), @splat(0x0F));
    const hi = bytes >> @as(@Vector(16, u8), @splat(4));

    // Interleaved format: [lo0, hi0, lo1, hi1, lo2, hi2, ...]
    return @shuffle(u8, lo, hi, @Vector(32, i32){
        0,  ~@as(i32, 0),  1,  ~@as(i32, 1),  2,  ~@as(i32, 2),  3,  ~@as(i32, 3),
        4,  ~@as(i32, 4),  5,  ~@as(i32, 5),  6,  ~@as(i32, 6),  7,  ~@as(i32, 7),
        8,  ~@as(i32, 8),  9,  ~@as(i32, 9),  10, ~@as(i32, 10), 11, ~@as(i32, 11),
        12, ~@as(i32, 12), 13, ~@as(i32, 13), 14, ~@as(i32, 14), 15, ~@as(i32, 15),
    });
}

/// Fast Q4×Q8 dot product using unsigned nibbles directly.
/// Uses algebraic identity: sum((q-8)*y) = sum(q*y) - 8*sum(y)
///
/// Input: q4 = unsigned nibbles [0..15], y = signed i8
/// Returns: {dot_product, sum_of_y} for offset correction
pub inline fn mulSumU8I8WithYSum(q4: @Vector(32, u8), y: @Vector(32, i8)) struct { dot: @Vector(8, i32), sum_y: i32 } {
    // 1. Calculate sum of y (needed for Q4 offset correction: -8 * sum_y)
    const y_i16: @Vector(32, i16) = y;
    const sum_y: i32 = @reduce(.Add, y_i16);

    // 2. Calculate dot product using NEON
    const q4_lo: @Vector(16, u8) = @shuffle(u8, q4, undefined, @Vector(16, i32){
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    });
    const q4_hi: @Vector(16, u8) = @shuffle(u8, q4, undefined, @Vector(16, i32){
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    });
    const y_lo: @Vector(16, i8) = @shuffle(i8, y, undefined, @Vector(16, i32){
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    });
    const y_hi: @Vector(16, i8) = @shuffle(i8, y, undefined, @Vector(16, i32){
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    });

    const res_lo = dotU8I8_128(q4_lo, y_lo);
    const res_hi = dotU8I8_128(q4_hi, y_hi);

    const dot = @shuffle(i32, res_lo, res_hi, @Vector(8, i32){ 0, 1, 2, 3, -1, -2, -3, -4 });

    return .{ .dot = dot, .sum_y = sum_y };
}

// =============================================================================
// BF16 Operations (ARMv8.6-A+, M2/M3/M4)
// =============================================================================

/// BFDOT: bf16 dot product instruction.
/// Computes dot product of 8 bf16 pairs, accumulating into 4 f32 lanes.
/// Each output lane gets sum of 2 adjacent bf16×bf16 products.
/// Available on M2/M3/M4 (ARMv8.6-A with FEAT_BF16).
/// Uses runtime detection for cross-generation binary compatibility.
pub inline fn bfdot8(acc: @Vector(4, f32), a: @Vector(8, u16), b: @Vector(8, u16)) @Vector(4, f32) {
    // On aarch64, use native BFDOT if available (runtime-detected).
    // The assembly is always compiled but only executed if bf16 is detected.
    if (comptime builtin.cpu.arch == .aarch64) {
        if (has_bf16) {
            // BFDOT: bfloat16 dot product
            // bfdot Vd.4S, Vn.8H, Vm.8H
            var result = acc;
            asm ("bfdot %[acc].4s, %[a].8h, %[b].8h"
                : [acc] "+w" (result),
                : [a] "w" (a),
                  [b] "w" (b),
            );
            return result;
        }
    }

    // Fallback: convert bf16 to f32 and use FMA
    var result = acc;
    inline for (0..4) |lane| {
        const a0 = @as(f32, @bitCast(@as(u32, a[lane * 2]) << 16));
        const a1 = @as(f32, @bitCast(@as(u32, a[lane * 2 + 1]) << 16));
        const b0 = @as(f32, @bitCast(@as(u32, b[lane * 2]) << 16));
        const b1 = @as(f32, @bitCast(@as(u32, b[lane * 2 + 1]) << 16));
        result[lane] += a0 * b0 + a1 * b1;
    }
    return result;
}

/// Convert 8 f32 values to 8 bf16 values (truncation).
/// Uses BFCVTN instruction on ARMv8.6-A+ for optimal performance.
/// Uses runtime detection for cross-generation binary compatibility.
pub inline fn f32ToBf16x8(lo: @Vector(4, f32), hi: @Vector(4, f32)) @Vector(8, u16) {
    if (comptime builtin.cpu.arch == .aarch64) {
        if (has_bf16) {
            // BFCVTN/BFCVTN2: convert f32 to bf16
            var result: @Vector(8, u16) = undefined;
            // First convert low 4 floats to low 4 bf16
            asm ("bfcvtn %[out].4h, %[in].4s"
                : [out] "=w" (result),
                : [in] "w" (lo),
            );
            // Then convert high 4 floats to high 4 bf16
            asm ("bfcvtn2 %[out].8h, %[in].4s"
                : [out] "+w" (result),
                : [in] "w" (hi),
            );
            return result;
        }
    }

    // Fallback: manual truncation (drop lower 16 bits)
    var result: @Vector(8, u16) = undefined;
    inline for (0..4) |i| {
        result[i] = @truncate(@as(u32, @bitCast(lo[i])) >> 16);
        result[i + 4] = @truncate(@as(u32, @bitCast(hi[i])) >> 16);
    }
    return result;
}

// =============================================================================
// Tests
// =============================================================================

test "mulSumI8Pairs basic" {
    const a: @Vector(16, i8) = .{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    const b: @Vector(16, i8) = .{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

    const result = dot128(a, b);
    // SDOT groups 4 adjacent elements per lane:
    // Lane 0: a[0..3]*b[0..3] = 1+2+3+4 = 10
    // Lane 1: a[4..7]*b[4..7] = 5+6+7+8 = 26
    // Lane 2: a[8..11]*b[8..11] = 9+10+11+12 = 42
    // Lane 3: a[12..15]*b[12..15] = 13+14+15+16 = 58
    try std.testing.expectEqual(@Vector(4, i32){ 10, 26, 42, 58 }, result);
}

test "mulSumI8Pairs with zeros" {
    const a: @Vector(16, i8) = @splat(0);
    const b: @Vector(16, i8) = .{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };

    const result = dot128(a, b);
    try std.testing.expectEqual(@Vector(4, i32){ 0, 0, 0, 0 }, result);
}

test "mulSumI8Pairs negative values" {
    const a: @Vector(16, i8) = .{ -1, -2, -3, -4, -5, -6, -7, -8, -1, -2, -3, -4, -5, -6, -7, -8 };
    const b: @Vector(16, i8) = .{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

    const result = dot128(a, b);
    // SDOT groups 4 adjacent elements per lane:
    // Lane 0: -1-2-3-4 = -10
    // Lane 1: -5-6-7-8 = -26
    // Lane 2: -1-2-3-4 = -10
    // Lane 3: -5-6-7-8 = -26
    try std.testing.expectEqual(@Vector(4, i32){ -10, -26, -10, -26 }, result);
}

test "mulSumI8Pairs signed multiplication" {
    const a: @Vector(16, i8) = .{ 2, -3, 4, -5, 1, -2, 3, -4, 5, -6, 7, -8, 1, 2, 3, 4 };
    const b: @Vector(16, i8) = .{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };

    const result = dot128(a, b);
    // SDOT groups 4 adjacent elements per lane:
    // Lane 0: 2*1 + -3*2 + 4*3 + -5*4 = 2-6+12-20 = -12
    // Lane 1: 1*5 + -2*6 + 3*7 + -4*8 = 5-12+21-32 = -18
    // Lane 2: 5*9 + -6*10 + 7*11 + -8*12 = 45-60+77-96 = -34
    // Lane 3: 1*13 + 2*14 + 3*15 + 4*16 = 13+28+45+64 = 150
    try std.testing.expectEqual(@Vector(4, i32){ -12, -18, -34, 150 }, result);
}

test "mulSumI8Pairs max values" {
    const a: @Vector(16, i8) = @splat(127);
    const b: @Vector(16, i8) = @splat(1);

    const result = dot128(a, b);
    // Each group: 127*4 = 508
    try std.testing.expectEqual(@Vector(4, i32){ 508, 508, 508, 508 }, result);
}

test "mulSumI8Pairs min values" {
    const a: @Vector(16, i8) = @splat(-128);
    const b: @Vector(16, i8) = @splat(1);

    const result = dot128(a, b);
    // Each group: -128*4 = -512
    try std.testing.expectEqual(@Vector(4, i32){ -512, -512, -512, -512 }, result);
}

test "mulSumI8Pairs accumulation i32" {
    // Test that products accumulate correctly without overflow
    const a: @Vector(16, i8) = @splat(100);
    const b: @Vector(16, i8) = @splat(100);

    const result = dot128(a, b);
    // Each group: 100*100*4 = 40000
    try std.testing.expectEqual(@Vector(4, i32){ 40000, 40000, 40000, 40000 }, result);
}

test "mulSumU8I8WithYSum dotU8I8_128 basic" {
    const a: @Vector(16, u8) = .{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    const b: @Vector(16, i8) = @splat(1);

    const result = dotU8I8_128(a, b);
    // SDOT groups 4 adjacent elements per lane:
    // Lane 0: 1+2+3+4 = 10
    // Lane 1: 5+6+7+8 = 26
    // Lane 2: 9+10+11+12 = 42
    // Lane 3: 13+14+15+16 = 58
    try std.testing.expectEqual(@Vector(4, i32){ 10, 26, 42, 58 }, result);
}

test "mulSumU8I8WithYSum dotU8I8_128 mixed signedness" {
    const a: @Vector(16, u8) = @splat(10);
    const b: @Vector(16, i8) = .{ 1, -1, 2, -2, 1, -1, 2, -2, 1, -1, 2, -2, 1, -1, 2, -2 };

    const result = dotU8I8_128(a, b);
    // Lane 0: 10*(1-1+2-2) = 0
    // Lane 1: 10*(1-1+2-2) = 0
    // Lane 2: 10*(1-1+2-2) = 0
    // Lane 3: 10*(1-1+2-2) = 0
    try std.testing.expectEqual(@Vector(4, i32){ 0, 0, 0, 0 }, result);
}

test "mulSumU8I8WithYSum dotU8I8_128 unsigned range" {
    // Test Q4 nibble range [0..15]
    const a: @Vector(16, u8) = .{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    const b: @Vector(16, i8) = @splat(2);

    const result = dotU8I8_128(a, b);
    // SDOT groups 4 adjacent elements per lane:
    // Lane 0: 2*(0+1+2+3) = 12
    // Lane 1: 2*(4+5+6+7) = 44
    // Lane 2: 2*(8+9+10+11) = 76
    // Lane 3: 2*(12+13+14+15) = 108
    try std.testing.expectEqual(@Vector(4, i32){ 12, 44, 76, 108 }, result);
}

test "mulSumU8I8WithYSum dotU8I8_128 negative weights" {
    const a: @Vector(16, u8) = @splat(15); // Max nibble value
    const b: @Vector(16, i8) = @splat(-1);

    const result = dotU8I8_128(a, b);
    // Each group: 15*(-1)*4 = -60
    try std.testing.expectEqual(@Vector(4, i32){ -60, -60, -60, -60 }, result);
}

test "mulSumU8I8WithYSum dotU8I8_128 max unsigned" {
    // Note: This function is designed for Q4 nibble values [0..15]
    // Test with max Q4 nibble value (15) instead of full u8 range
    const a: @Vector(16, u8) = @splat(15);
    const b: @Vector(16, i8) = @splat(1);

    const result = dotU8I8_128(a, b);
    // Each group: 15*4 = 60
    try std.testing.expectEqual(@Vector(4, i32){ 60, 60, 60, 60 }, result);
}

test "mulSumI8Pairs matches expected" {
    const x: @Vector(32, i8) = .{
        1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    };
    const y: @Vector(32, i8) = @splat(1);

    const result = mulSumI8Pairs(x, y);
    // SDOT groups 4 adjacent elements per lane (using dot128 twice):
    // First 16 elements:
    // Lane 0: 1+2+3+4 = 10
    // Lane 1: 5+6+7+8 = 26
    // Lane 2: 9+10+11+12 = 42
    // Lane 3: 13+14+15+16 = 58
    // Second 16 elements:
    // Lane 4: 17+18+19+20 = 74
    // Lane 5: 21+22+23+24 = 90
    // Lane 6: 25+26+27+28 = 106
    // Lane 7: 29+30+31+32 = 122
    try std.testing.expectEqual(@Vector(8, i32){ 10, 26, 42, 58, 74, 90, 106, 122 }, result);
}

test "mulSumI8Pairs zeros vec32" {
    const x: @Vector(32, i8) = @splat(0);
    const y: @Vector(32, i8) = @splat(5);

    const result = mulSumI8Pairs(x, y);
    try std.testing.expectEqual(@Vector(8, i32){ 0, 0, 0, 0, 0, 0, 0, 0 }, result);
}

test "mulSumI8Pairs negative products" {
    const x: @Vector(32, i8) = @splat(-10);
    const y: @Vector(32, i8) = @splat(5);

    const result = mulSumI8Pairs(x, y);
    // Each group: -10*5*4 = -200
    try std.testing.expectEqual(@Vector(8, i32){ -200, -200, -200, -200, -200, -200, -200, -200 }, result);
}

test "extract32Nibbles basic" {
    const bytes: @Vector(16, u8) = .{
        0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
        0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE,
    };

    const result = extract32Nibbles(bytes);

    // Interleaved [lo0, hi0, lo1, hi1, ...]
    const expected: @Vector(32, u8) = .{
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    };
    try std.testing.expectEqual(expected, result);
}

test "extract32Nibbles all zeros" {
    const bytes: @Vector(16, u8) = @splat(0x00);
    const result = extract32Nibbles(bytes);
    try std.testing.expectEqual(@as(@Vector(32, u8), @splat(0)), result);
}

test "extract32Nibbles all ones" {
    const bytes: @Vector(16, u8) = @splat(0xFF);
    const result = extract32Nibbles(bytes);
    // All nibbles should be 15
    try std.testing.expectEqual(@as(@Vector(32, u8), @splat(15)), result);
}

test "extract32Nibbles interleaved format" {
    // Test that nibbles are properly interleaved
    const bytes: @Vector(16, u8) = .{
        0x12, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    };

    const result = extract32Nibbles(bytes);

    // First two nibbles should be 2 (lo) and 1 (hi)
    try std.testing.expectEqual(@as(u8, 2), result[0]);
    try std.testing.expectEqual(@as(u8, 1), result[1]);
}

test "mulSumU8I8WithYSum basic" {
    const q4: @Vector(32, u8) = @splat(8); // All 8s (midpoint of 0-15)
    const y: @Vector(32, i8) = @splat(1); // All 1s

    const result = mulSumU8I8WithYSum(q4, y);

    // dot = 8 * 32 = 256, grouped into 8 results of 32 each
    try std.testing.expectEqual(@as(i32, 32), @divExact(@reduce(.Add, result.dot), 8));
    // sum_y = 32
    try std.testing.expectEqual(@as(i32, 32), result.sum_y);
}

test "mulSumU8I8WithYSum with scale correction" {
    // Test Q4 offset correction: (q-8)*y = q*y - 8*sum(y)
    const q4: @Vector(32, u8) = @splat(15); // Max nibble
    const y: @Vector(32, i8) = @splat(2);

    const result = mulSumU8I8WithYSum(q4, y);

    // dot = 15*2*32 = 960
    const total_dot: i32 = @reduce(.Add, result.dot);
    try std.testing.expectEqual(@as(i32, 960), total_dot);

    // sum_y = 2*32 = 64
    try std.testing.expectEqual(@as(i32, 64), result.sum_y);

    // After offset correction: 960 - 8*64 = 960 - 512 = 448
    const corrected = total_dot - 8 * result.sum_y;
    try std.testing.expectEqual(@as(i32, 448), corrected);
}

test "mulSumU8I8WithYSum negative weights" {
    const q4: @Vector(32, u8) = @splat(10);
    const y: @Vector(32, i8) = @splat(-3);

    const result = mulSumU8I8WithYSum(q4, y);

    // dot = 10*(-3)*32 = -960
    try std.testing.expectEqual(@as(i32, -960), @reduce(.Add, result.dot));

    // sum_y = -3*32 = -96
    try std.testing.expectEqual(@as(i32, -96), result.sum_y);
}

test "mulSumU8I8WithYSum zero nibbles" {
    const q4: @Vector(32, u8) = @splat(0);
    const y: @Vector(32, i8) = @splat(5);

    const result = mulSumU8I8WithYSum(q4, y);

    // dot should be 0
    try std.testing.expectEqual(@as(i32, 0), @reduce(.Add, result.dot));

    // sum_y = 5*32 = 160
    try std.testing.expectEqual(@as(i32, 160), result.sum_y);
}

test "mulSumU8I8WithYSum mixed values" {
    const q4: @Vector(32, u8) = .{
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    };
    const y: @Vector(32, i8) = @splat(1);

    const result = mulSumU8I8WithYSum(q4, y);

    // sum of q4: 2*(0+1+2+...+15) = 2*120 = 240
    try std.testing.expectEqual(@as(i32, 240), @reduce(.Add, result.dot));

    // sum_y = 32
    try std.testing.expectEqual(@as(i32, 32), result.sum_y);
}

test "mulSumU8I8WithYSum accumulator precision" {
    // Test that accumulator handles large values correctly
    const q4: @Vector(32, u8) = @splat(15);
    const y: @Vector(32, i8) = @splat(127);

    const result = mulSumU8I8WithYSum(q4, y);

    // dot = 15*127*32 = 60960
    try std.testing.expectEqual(@as(i32, 60960), @reduce(.Add, result.dot));

    // sum_y = 127*32 = 4064
    try std.testing.expectEqual(@as(i32, 4064), result.sum_y);
}
