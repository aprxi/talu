//! Normalization weight loader utilities.
//!
//! Handles loading and conversion of RMSNorm/QKNorm weights with
//! optional format conversion for Metal native norms (BF16/F16 to F32).

const std = @import("std");
const builtin = @import("builtin");

const tensor = @import("../../tensor.zig");
const dtype = @import("../../dtype.zig");
const st_loader = @import("root.zig");
const st_names = @import("names.zig");

fn shouldUseMetalNativeNorms(allocator: std.mem.Allocator) bool {
    const force_cpu_backend = if (std.posix.getenv("BACKEND")) |b| std.mem.eql(u8, b, "cpu") else false;
    _ = allocator;
    return builtin.os.tag == .macos and !force_cpu_backend;
}

/// Try to load a 1D RMSNorm/QKNorm weight vector.
/// Returns a heap-allocated `*tensor.Tensor` owned by the caller's allocator.
///
/// - If `use_metal_norms` is true, returns the safetensors view (bf16/f16) directly.
/// - Otherwise, converts bf16/f16 to f32 into an owned buffer for CPU kernels.
fn tryLoadNormWeightLayer(
    allocator: std.mem.Allocator,
    safetensors: *st_loader.UnifiedSafeTensors,
    layer_idx: usize,
    comptime options: anytype,
    use_metal_norms: bool,
) ?*tensor.Tensor {
    var name_buffer: [128]u8 = undefined;
    const tensor_name = st_names.selectNameLayer(safetensors, name_buffer[0..], layer_idx, options) catch return null;
    const tensor_view = safetensors.getTensor(tensor_name, null) catch return null;

    const tensor_ptr = allocator.create(tensor.Tensor) catch return null;
    if (use_metal_norms) {
        tensor_ptr.* = tensor_view;
        return tensor_ptr;
    }

    if (tensor_view.dtype == .f32) {
        tensor_ptr.* = tensor_view;
        return tensor_ptr;
    }

    if (tensor_view.dtype == .bf16 or tensor_view.dtype == .f16) {
        const element_count: usize = @intCast(tensor_view.numElements());
        var owned_tensor = tensor.OwnedTensor.init(allocator, .f32, &.{element_count}) catch return null;
        const dst_f32 = owned_tensor.asSlice(f32);
        const src_u16 = @as([*]align(1) const u16, @ptrCast(tensor_view.data.ptr))[0 .. tensor_view.data.len / @sizeOf(u16)];
        if (tensor_view.dtype == .bf16) {
            for (0..element_count) |elem_idx| dst_f32[elem_idx] = dtype.bf16ToF32(src_u16[elem_idx]);
        } else {
            for (0..element_count) |elem_idx| dst_f32[elem_idx] = dtype.fp16ToF32(src_u16[elem_idx]);
        }
        tensor_ptr.* = owned_tensor.view();
        return tensor_ptr;
    }

    return null;
}

// ============================================================================
// Unit Tests
// ============================================================================

test "bf16ToF32 conversion - basic values" {
    // Test known bf16 bit patterns and verify f32 results
    // bf16 0x3F80 = 1.0f
    const val_1_0 = dtype.bf16ToF32(0x3F80);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), val_1_0, 0.0001);

    // bf16 0x4000 = 2.0f
    const val_2_0 = dtype.bf16ToF32(0x4000);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), val_2_0, 0.0001);

    // bf16 0x0000 = 0.0f
    const val_0_0 = dtype.bf16ToF32(0x0000);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), val_0_0, 0.0001);

    // bf16 0xBF80 = -1.0f (negative)
    const val_neg_1_0 = dtype.bf16ToF32(0xBF80);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), val_neg_1_0, 0.0001);

    // bf16 0x3F00 = 0.5f
    const val_0_5 = dtype.bf16ToF32(0x3F00);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), val_0_5, 0.0001);
}

test "bf16ToF32 conversion - special values" {
    // Test edge cases: very small values, very large values, infinity

    // Very small positive value (denormal)
    // bf16 0x0001 represents a very small denormal number
    const small_val = dtype.bf16ToF32(0x0001);
    try std.testing.expect(small_val > 0.0);
    try std.testing.expect(small_val < 0.0001);

    // Very large value
    // bf16 0x7F00 = approximately 1.7014e38 (close to infinity threshold)
    const large_val = dtype.bf16ToF32(0x7F00);
    try std.testing.expect(large_val > 1.0e38);

    // Positive infinity
    // bf16 0x7F80 = +inf
    const pos_inf = dtype.bf16ToF32(0x7F80);
    try std.testing.expect(std.math.isInf(pos_inf));
    try std.testing.expect(pos_inf > 0);

    // Negative infinity
    // bf16 0xFF80 = -inf
    const neg_inf = dtype.bf16ToF32(0xFF80);
    try std.testing.expect(std.math.isInf(neg_inf));
    try std.testing.expect(neg_inf < 0);

    // NaN
    // bf16 0x7FC0 = NaN
    const nan_val = dtype.bf16ToF32(0x7FC0);
    try std.testing.expect(std.math.isNan(nan_val));
}

test "fp16ToF32 conversion - basic values" {
    // Test known f16 bit patterns and verify f32 results
    // f16 0x3C00 = 1.0f
    const val_1_0 = dtype.fp16ToF32(0x3C00);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), val_1_0, 0.0001);

    // f16 0x4000 = 2.0f
    const val_2_0 = dtype.fp16ToF32(0x4000);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), val_2_0, 0.0001);

    // f16 0x0000 = 0.0f
    const val_0_0 = dtype.fp16ToF32(0x0000);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), val_0_0, 0.0001);

    // f16 0xBC00 = -1.0f (negative)
    const val_neg_1_0 = dtype.fp16ToF32(0xBC00);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), val_neg_1_0, 0.0001);

    // f16 0x3800 = 0.5f
    const val_0_5 = dtype.fp16ToF32(0x3800);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), val_0_5, 0.0001);

    // f16 0x4200 = 3.0f
    const val_3_0 = dtype.fp16ToF32(0x4200);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), val_3_0, 0.0001);
}

test "fp16ToF32 conversion - special values" {
    // Test edge cases for f16

    // Very small positive value (denormal)
    // f16 0x0001 represents a very small denormal number
    const small_val = dtype.fp16ToF32(0x0001);
    try std.testing.expect(small_val > 0.0);
    try std.testing.expect(small_val < 0.00001);

    // Very large value
    // f16 0x7BFF = 65504 (max normal f16 value)
    const large_val = dtype.fp16ToF32(0x7BFF);
    try std.testing.expect(large_val > 65000.0);
    try std.testing.expect(large_val < 66000.0);

    // Positive infinity
    // f16 0x7C00 = +inf
    const pos_inf = dtype.fp16ToF32(0x7C00);
    try std.testing.expect(std.math.isInf(pos_inf));
    try std.testing.expect(pos_inf > 0);

    // Negative infinity
    // f16 0xFC00 = -inf
    const neg_inf = dtype.fp16ToF32(0xFC00);
    try std.testing.expect(std.math.isInf(neg_inf));
    try std.testing.expect(neg_inf < 0);

    // NaN
    // f16 0x7E00 = NaN
    const nan_val = dtype.fp16ToF32(0x7E00);
    try std.testing.expect(std.math.isNan(nan_val));
}

test "bf16 array conversion - memory correctness" {
    // Test converting an array of bf16 values to f32
    // This simulates what happens in tryLoadNormWeightLayer
    const allocator = std.testing.allocator;

    // Create test bf16 data: [1.0, 2.0, 0.5, -1.0, 0.0]
    const bf16_data = [_]u16{ 0x3F80, 0x4000, 0x3F00, 0xBF80, 0x0000 };
    const expected_f32 = [_]f32{ 1.0, 2.0, 0.5, -1.0, 0.0 };

    // Allocate and convert
    const element_count = bf16_data.len;
    var owned_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{element_count});
    defer owned_tensor.deinit();

    const dst_f32 = owned_tensor.asSlice(f32);

    // Convert bf16 to f32
    for (0..element_count) |i| {
        dst_f32[i] = dtype.bf16ToF32(bf16_data[i]);
    }

    // Verify results
    for (0..element_count) |i| {
        try std.testing.expectApproxEqAbs(expected_f32[i], dst_f32[i], 0.0001);
    }
}

test "fp16 array conversion - memory correctness" {
    // Test converting an array of f16 values to f32
    // This simulates what happens in tryLoadNormWeightLayer
    const allocator = std.testing.allocator;

    // Create test f16 data: [1.0, 2.0, 0.5, -1.0, 0.0, 3.0]
    const fp16_data = [_]u16{ 0x3C00, 0x4000, 0x3800, 0xBC00, 0x0000, 0x4200 };
    const expected_f32 = [_]f32{ 1.0, 2.0, 0.5, -1.0, 0.0, 3.0 };

    // Allocate and convert
    const element_count = fp16_data.len;
    var owned_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{element_count});
    defer owned_tensor.deinit();

    const dst_f32 = owned_tensor.asSlice(f32);

    // Convert f16 to f32
    for (0..element_count) |i| {
        dst_f32[i] = dtype.fp16ToF32(fp16_data[i]);
    }

    // Verify results
    for (0..element_count) |i| {
        try std.testing.expectApproxEqAbs(expected_f32[i], dst_f32[i], 0.0001);
    }
}

test "bf16 array conversion - large array no memory leak" {
    // Test that large array conversions don't leak memory
    const allocator = std.testing.allocator;

    const element_count = 4096; // Typical layer norm size
    var owned_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{element_count});
    defer owned_tensor.deinit();

    const dst_f32 = owned_tensor.asSlice(f32);

    // Create pattern of bf16 values
    var bf16_data = try allocator.alloc(u16, element_count);
    defer allocator.free(bf16_data);

    // Fill with alternating pattern
    for (0..element_count) |i| {
        bf16_data[i] = if (i % 2 == 0) 0x3F80 else 0x4000; // 1.0 or 2.0
    }

    // Convert
    for (0..element_count) |i| {
        dst_f32[i] = dtype.bf16ToF32(bf16_data[i]);
    }

    // Verify pattern
    for (0..element_count) |i| {
        const expected: f32 = if (i % 2 == 0) 1.0 else 2.0;
        try std.testing.expectApproxEqAbs(expected, dst_f32[i], 0.0001);
    }
}

test "fp16 array conversion - large array no memory leak" {
    // Test that large array conversions don't leak memory
    const allocator = std.testing.allocator;

    const element_count = 4096; // Typical layer norm size
    var owned_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{element_count});
    defer owned_tensor.deinit();

    const dst_f32 = owned_tensor.asSlice(f32);

    // Create pattern of f16 values
    var fp16_data = try allocator.alloc(u16, element_count);
    defer allocator.free(fp16_data);

    // Fill with alternating pattern
    for (0..element_count) |i| {
        fp16_data[i] = if (i % 2 == 0) 0x3C00 else 0x4000; // 1.0 or 2.0
    }

    // Convert
    for (0..element_count) |i| {
        dst_f32[i] = dtype.fp16ToF32(fp16_data[i]);
    }

    // Verify pattern
    for (0..element_count) |i| {
        const expected: f32 = if (i % 2 == 0) 1.0 else 2.0;
        try std.testing.expectApproxEqAbs(expected, dst_f32[i], 0.0001);
    }
}

test "mixed precision array - bf16 and f16 mixed conversion" {
    // Test that we can handle both bf16 and f16 conversions in same test context
    const allocator = std.testing.allocator;

    // Test bf16
    const bf16_val = dtype.bf16ToF32(0x3F80); // 1.0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), bf16_val, 0.0001);

    // Test f16
    const fp16_val = dtype.fp16ToF32(0x3C00); // 1.0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), fp16_val, 0.0001);

    // Create tensors for both
    var bf16_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{3});
    defer bf16_tensor.deinit();

    var fp16_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{3});
    defer fp16_tensor.deinit();

    const bf16_dst = bf16_tensor.asSlice(f32);
    const fp16_dst = fp16_tensor.asSlice(f32);

    // Convert bf16 values
    const bf16_src = [_]u16{ 0x3F80, 0x4000, 0x3F00 }; // 1.0, 2.0, 0.5
    for (0..3) |i| {
        bf16_dst[i] = dtype.bf16ToF32(bf16_src[i]);
    }

    // Convert f16 values
    const fp16_src = [_]u16{ 0x3C00, 0x4000, 0x3800 }; // 1.0, 2.0, 0.5
    for (0..3) |i| {
        fp16_dst[i] = dtype.fp16ToF32(fp16_src[i]);
    }

    // Both should produce same results
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(bf16_dst[i], fp16_dst[i], 0.0001);
    }
}

test "bf16 conversion - precision and rounding" {
    // bf16 has lower precision than f16, test that conversion preserves what precision it has
    // bf16 can represent powers of 2 exactly
    const powers_of_2_bf16 = [_]u16{
        0x3F80, // 1.0 = 2^0
        0x4000, // 2.0 = 2^1
        0x4080, // 4.0 = 2^2
        0x4100, // 8.0 = 2^3
        0x4180, // 16.0 = 2^4
    };

    const expected_powers = [_]f32{ 1.0, 2.0, 4.0, 8.0, 16.0 };

    for (0..powers_of_2_bf16.len) |i| {
        const result = dtype.bf16ToF32(powers_of_2_bf16[i]);
        try std.testing.expectApproxEqAbs(expected_powers[i], result, 0.0001);
    }
}

test "fp16 conversion - precision and rounding" {
    // f16 has higher precision than bf16 in the mantissa
    // Test various fractional values
    const fp16_fractional = [_]u16{
        0x3C00, // 1.0
        0x3E00, // 1.5
        0x4000, // 2.0
        0x4200, // 3.0
        0x4400, // 4.0
        0x4500, // 5.0
        0x3555, // ~0.333 (approximately 1/3)
    };

    const expected_values = [_]f32{ 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 0.333 };
    const tolerances = [_]f32{ 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.001 }; // 1/3 needs more tolerance

    for (0..fp16_fractional.len) |i| {
        const result = dtype.fp16ToF32(fp16_fractional[i]);
        try std.testing.expectApproxEqAbs(expected_values[i], result, tolerances[i]);
    }
}

test "conversion - zero handling" {
    // Both bf16 and f16 should handle positive and negative zero
    // Positive zero
    const bf16_pos_zero = dtype.bf16ToF32(0x0000);
    const fp16_pos_zero = dtype.fp16ToF32(0x0000);
    try std.testing.expectEqual(@as(f32, 0.0), bf16_pos_zero);
    try std.testing.expectEqual(@as(f32, 0.0), fp16_pos_zero);

    // Negative zero
    const bf16_neg_zero = dtype.bf16ToF32(0x8000);
    const fp16_neg_zero = dtype.fp16ToF32(0x8000);
    try std.testing.expectEqual(@as(f32, -0.0), bf16_neg_zero);
    try std.testing.expectEqual(@as(f32, -0.0), fp16_neg_zero);
}

test "conversion - denormal numbers" {
    // Test that denormal numbers (very small) are handled correctly
    // Smallest positive denormal bf16: 0x0001
    const bf16_denorm = dtype.bf16ToF32(0x0001);
    try std.testing.expect(bf16_denorm > 0.0);
    try std.testing.expect(bf16_denorm < 1.0e-38); // Very small

    // Smallest positive denormal f16: 0x0001
    const fp16_denorm = dtype.fp16ToF32(0x0001);
    try std.testing.expect(fp16_denorm > 0.0);
    try std.testing.expect(fp16_denorm < 1.0e-4); // Very small for f16
}

test "memory allocation - single element tensor" {
    // Test that we can allocate and convert a single element correctly
    const allocator = std.testing.allocator;

    var owned_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{1});
    defer owned_tensor.deinit();

    const dst_f32 = owned_tensor.asSlice(f32);
    try std.testing.expectEqual(@as(usize, 1), dst_f32.len);

    // Convert a single bf16 value
    dst_f32[0] = dtype.bf16ToF32(0x4000); // 2.0
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), dst_f32[0], 0.0001);
}

test "memory allocation - empty tensor" {
    // Test that we can handle zero-element tensors gracefully
    const allocator = std.testing.allocator;

    var owned_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{0});
    defer owned_tensor.deinit();

    const dst_f32 = owned_tensor.asSlice(f32);
    try std.testing.expectEqual(@as(usize, 0), dst_f32.len);
}
