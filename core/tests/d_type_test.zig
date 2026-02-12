//! Integration tests for DType
//!
//! DType is the unified data type enum for tensor operations.
//! Supports both standard types (for FFI) and quantized formats (internal).

const std = @import("std");
const main = @import("main");
const DType = main.DType;

// =============================================================================
// FFI Conversion Tests
// =============================================================================

test "DType.toFFI returns correct values for standard types" {
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

test "DType.toFFI returns u8 for quantized types" {
    // All quantized types appear as u8 arrays externally
    try std.testing.expectEqual(@as(u32, 8), DType.q8_0.toFFI());
    try std.testing.expectEqual(@as(u32, 8), DType.q4_0.toFFI());
    try std.testing.expectEqual(@as(u32, 8), DType.q4_1.toFFI());
    try std.testing.expectEqual(@as(u32, 8), DType.q6_k.toFFI());
    try std.testing.expectEqual(@as(u32, 8), DType.grouped_affine_u4.toFFI());
    try std.testing.expectEqual(@as(u32, 8), DType.mxfp4.toFFI());
}

test "DType.fromFFI creates correct types" {
    try std.testing.expectEqual(DType.f32, DType.fromFFI(0).?);
    try std.testing.expectEqual(DType.f64, DType.fromFFI(1).?);
    try std.testing.expectEqual(DType.i32, DType.fromFFI(2).?);
    try std.testing.expectEqual(DType.i64, DType.fromFFI(3).?);
    try std.testing.expectEqual(DType.f16, DType.fromFFI(4).?);
    try std.testing.expectEqual(DType.bf16, DType.fromFFI(5).?);
    try std.testing.expectEqual(DType.u8, DType.fromFFI(8).?);
}

test "DType.fromFFI returns null for invalid values" {
    try std.testing.expect(DType.fromFFI(12) == null);
    try std.testing.expect(DType.fromFFI(100) == null);
    try std.testing.expect(DType.fromFFI(255) == null);
}

// =============================================================================
// Element Size Tests
// =============================================================================

test "DType.elementSize returns correct sizes for standard types" {
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
}

test "DType.elementSize returns 0 for quantized types" {
    // Quantized types require block-based size calculations
    try std.testing.expectEqual(@as(usize, 0), DType.q8_0.elementSize());
    try std.testing.expectEqual(@as(usize, 0), DType.q4_0.elementSize());
    try std.testing.expectEqual(@as(usize, 0), DType.q4_1.elementSize());
    try std.testing.expectEqual(@as(usize, 0), DType.q6_k.elementSize());
    try std.testing.expectEqual(@as(usize, 0), DType.grouped_affine_u4.elementSize());
    try std.testing.expectEqual(@as(usize, 0), DType.mxfp4.elementSize());
}

test "DType.elementSize returns 1 for f8_e4m3" {
    try std.testing.expectEqual(@as(usize, 1), DType.f8_e4m3.elementSize());
}

// =============================================================================
// Type Classification Tests
// =============================================================================

test "DType.isQuantized returns true for quantized types" {
    try std.testing.expect(DType.q8_0.isQuantized());
    try std.testing.expect(DType.q4_0.isQuantized());
    try std.testing.expect(DType.q4_1.isQuantized());
    try std.testing.expect(DType.q5_0.isQuantized());
    try std.testing.expect(DType.q6_k.isQuantized());
    try std.testing.expect(DType.q4_k.isQuantized());
    try std.testing.expect(DType.q5_k.isQuantized());
    try std.testing.expect(DType.grouped_affine_u4.isQuantized());
    try std.testing.expect(DType.grouped_affine_u8.isQuantized());
    try std.testing.expect(DType.mxfp4.isQuantized());
}

test "DType.isQuantized returns false for standard types" {
    try std.testing.expect(!DType.f32.isQuantized());
    try std.testing.expect(!DType.f64.isQuantized());
    try std.testing.expect(!DType.f16.isQuantized());
    try std.testing.expect(!DType.bf16.isQuantized());
    try std.testing.expect(!DType.i32.isQuantized());
    try std.testing.expect(!DType.u8.isQuantized());
    try std.testing.expect(!DType.f8_e4m3.isQuantized());
}

test "DType.isStandard returns true for standard types" {
    try std.testing.expect(DType.f32.isStandard());
    try std.testing.expect(DType.f64.isStandard());
    try std.testing.expect(DType.f16.isStandard());
    try std.testing.expect(DType.bf16.isStandard());
    try std.testing.expect(DType.i8.isStandard());
    try std.testing.expect(DType.u8.isStandard());
}

test "DType.isStandard returns false for quantized types" {
    try std.testing.expect(!DType.q8_0.isStandard());
    try std.testing.expect(!DType.q4_0.isStandard());
    try std.testing.expect(!DType.mxfp4.isStandard());
}

// =============================================================================
// Type String Tests
// =============================================================================

test "DType.toTypeStr returns numpy format strings" {
    try std.testing.expectEqualStrings("<f4", std.mem.span(DType.f32.toTypeStr()));
    try std.testing.expectEqualStrings("<f8", std.mem.span(DType.f64.toTypeStr()));
    try std.testing.expectEqualStrings("<f2", std.mem.span(DType.f16.toTypeStr()));
    try std.testing.expectEqualStrings("<i4", std.mem.span(DType.i32.toTypeStr()));
    try std.testing.expectEqualStrings("<i8", std.mem.span(DType.i64.toTypeStr()));
    try std.testing.expectEqualStrings("<u1", std.mem.span(DType.u8.toTypeStr()));
}

test "DType.toTypeStr returns u1 for quantized types" {
    // Quantized types appear as u8 arrays
    try std.testing.expectEqualStrings("<u1", std.mem.span(DType.q8_0.toTypeStr()));
    try std.testing.expectEqualStrings("<u1", std.mem.span(DType.q4_0.toTypeStr()));
    try std.testing.expectEqualStrings("<u1", std.mem.span(DType.mxfp4.toTypeStr()));
}

// =============================================================================
// Enum Value Tests
// =============================================================================

test "DType standard types have values 0-11" {
    try std.testing.expectEqual(@as(u8, 0), @intFromEnum(DType.f32));
    try std.testing.expectEqual(@as(u8, 1), @intFromEnum(DType.f64));
    try std.testing.expectEqual(@as(u8, 11), @intFromEnum(DType.u64));
}

test "DType quantized types have values 20+" {
    try std.testing.expectEqual(@as(u8, 20), @intFromEnum(DType.q8_0));
    try std.testing.expectEqual(@as(u8, 21), @intFromEnum(DType.q4_0));
    try std.testing.expectEqual(@as(u8, 27), @intFromEnum(DType.mxfp4));
}

// =============================================================================
// Roundtrip Tests
// =============================================================================

test "DType FFI roundtrip for standard types" {
    const types = [_]DType{ .f32, .f64, .i32, .i64, .f16, .bf16, .i8, .i16, .u8, .u16, .u32, .u64 };

    for (types) |dt| {
        const ffi_val = dt.toFFI();
        const roundtrip = DType.fromFFI(ffi_val);
        try std.testing.expect(roundtrip != null);
        try std.testing.expectEqual(dt, roundtrip.?);
    }
}
