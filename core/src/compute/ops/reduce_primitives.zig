//! Reduction operations with stride-aware, dtype-generic implementations.
//!
//! All ops work with TensorView and handle both contiguous and strided tensors.
//! Supports reduction along specific axes or across entire tensors.

const std = @import("std");
const tv = @import("tensor_view.zig");
const math = @import("math_primitives/root.zig");

const TensorView = tv.TensorView;
const DType = tv.DType;

// Use existing SIMD infrastructure
const simd = math.simd;
const VEC_LEN = simd.f32_vec_len;
const F32Vec = simd.F32Vec;

/// Dtype conversion helpers
const dtype_mod = @import("../../dtype.zig");

fn fp16ToF32(x: u16) f32 {
    return dtype_mod.fp16ToF32(x);
}

fn f32ToFp16(x: f32) u16 {
    return dtype_mod.f32ToFp16(x);
}

fn bf16ToF32(x: u16) f32 {
    return dtype_mod.bf16ToF32(x);
}

fn f32ToBf16(x: f32) u16 {
    return dtype_mod.f32ToBf16(x);
}

fn f32Identity(x: f32) f32 {
    return x;
}

/// Sum reduction along a specific axis.
/// Output shape is input shape with the specified axis dimension = 1.
pub fn sum(out: TensorView, input: TensorView, axis: usize) void {
    std.debug.assert(axis < input.ndim);
    std.debug.assert(out.ndim == input.ndim);
    std.debug.assert(out.shape[axis] == 1);

    // Verify other dimensions match
    for (0..input.ndim) |i| {
        if (i != axis) {
            std.debug.assert(out.shape[i] == input.shape[i]);
        }
    }

    switch (input.dtype) {
        .f32 => sumGeneric(f32, f32Identity, f32Identity, out, input, axis),
        .f16 => sumGeneric(u16, fp16ToF32, f32ToFp16, out, input, axis),
        .bf16 => sumGeneric(u16, bf16ToF32, f32ToBf16, out, input, axis),
        else => unreachable,
    }
}

fn sumGeneric(
    comptime T: type,
    comptime toF32: fn (T) f32,
    comptime fromF32: fn (f32) T,
    out: TensorView,
    input: TensorView,
    axis: usize,
) void {
    const out_data = @as([*]T, @ptrCast(@alignCast(out.data)));
    const in_data = @as([*]const T, @ptrCast(@alignCast(input.data)));

    // Initialize output to zero
    for (0..out.numel) |i| {
        out_data[i] = fromF32(0.0);
    }

    var coords: [tv.MAX_NDIM]usize = undefined;
    for (0..input.numel) |elem_idx| {
        input.indexToCoords(elem_idx, &coords);
        const in_offset = input.coordsToOffset(coords[0..input.ndim]);

        // For output, set the reduced axis to 0
        coords[axis] = 0;
        const out_offset = out.coordsToOffset(coords[0..out.ndim]);

        const current_sum = toF32(out_data[out_offset]);
        const value = toF32(in_data[in_offset]);
        out_data[out_offset] = fromF32(current_sum + value);
    }
}

/// Mean reduction along a specific axis.
/// Output shape is input shape with the specified axis dimension = 1.
pub fn mean(out: TensorView, input: TensorView, axis: usize) void {
    std.debug.assert(axis < input.ndim);
    std.debug.assert(out.ndim == input.ndim);
    std.debug.assert(out.shape[axis] == 1);

    // First compute sum
    sum(out, input, axis);

    // Then divide by count
    const count = @as(f32, @floatFromInt(input.shape[axis]));

    switch (out.dtype) {
        .f32 => meanDivideGeneric(f32, f32Identity, f32Identity, out, count),
        .f16 => meanDivideGeneric(u16, fp16ToF32, f32ToFp16, out, count),
        .bf16 => meanDivideGeneric(u16, bf16ToF32, f32ToBf16, out, count),
        else => unreachable,
    }
}

fn meanDivideGeneric(
    comptime T: type,
    comptime toF32: fn (T) f32,
    comptime fromF32: fn (f32) T,
    out: TensorView,
    count: f32,
) void {
    const out_data = @as([*]T, @ptrCast(@alignCast(out.data)));

    for (0..out.numel) |i| {
        const value = toF32(out_data[i]);
        out_data[i] = fromF32(value / count);
    }
}

/// Max reduction along a specific axis.
/// Output shape is input shape with the specified axis dimension = 1.
pub fn max(out: TensorView, input: TensorView, axis: usize) void {
    std.debug.assert(axis < input.ndim);
    std.debug.assert(out.ndim == input.ndim);
    std.debug.assert(out.shape[axis] == 1);

    switch (input.dtype) {
        .f32 => maxGeneric(f32, f32Identity, f32Identity, out, input, axis),
        .f16 => maxGeneric(u16, fp16ToF32, f32ToFp16, out, input, axis),
        .bf16 => maxGeneric(u16, bf16ToF32, f32ToBf16, out, input, axis),
        else => unreachable,
    }
}

fn maxGeneric(
    comptime T: type,
    comptime toF32: fn (T) f32,
    comptime fromF32: fn (f32) T,
    out: TensorView,
    input: TensorView,
    axis: usize,
) void {
    const out_data = @as([*]T, @ptrCast(@alignCast(out.data)));
    const in_data = @as([*]const T, @ptrCast(@alignCast(input.data)));

    // Initialize output to -inf
    for (0..out.numel) |i| {
        out_data[i] = fromF32(-std.math.inf(f32));
    }

    var coords: [tv.MAX_NDIM]usize = undefined;
    for (0..input.numel) |elem_idx| {
        input.indexToCoords(elem_idx, &coords);
        const in_offset = input.coordsToOffset(coords[0..input.ndim]);

        coords[axis] = 0;
        const out_offset = out.coordsToOffset(coords[0..out.ndim]);

        const current_max = toF32(out_data[out_offset]);
        const value = toF32(in_data[in_offset]);
        out_data[out_offset] = fromF32(@max(current_max, value));
    }
}

/// Min reduction along a specific axis.
/// Output shape is input shape with the specified axis dimension = 1.
pub fn min(out: TensorView, input: TensorView, axis: usize) void {
    std.debug.assert(axis < input.ndim);
    std.debug.assert(out.ndim == input.ndim);
    std.debug.assert(out.shape[axis] == 1);

    switch (input.dtype) {
        .f32 => minGeneric(f32, f32Identity, f32Identity, out, input, axis),
        .f16 => minGeneric(u16, fp16ToF32, f32ToFp16, out, input, axis),
        .bf16 => minGeneric(u16, bf16ToF32, f32ToBf16, out, input, axis),
        else => unreachable,
    }
}

fn minGeneric(
    comptime T: type,
    comptime toF32: fn (T) f32,
    comptime fromF32: fn (f32) T,
    out: TensorView,
    input: TensorView,
    axis: usize,
) void {
    const out_data = @as([*]T, @ptrCast(@alignCast(out.data)));
    const in_data = @as([*]const T, @ptrCast(@alignCast(input.data)));

    // Initialize output to +inf
    for (0..out.numel) |i| {
        out_data[i] = fromF32(std.math.inf(f32));
    }

    var coords: [tv.MAX_NDIM]usize = undefined;
    for (0..input.numel) |elem_idx| {
        input.indexToCoords(elem_idx, &coords);
        const in_offset = input.coordsToOffset(coords[0..input.ndim]);

        coords[axis] = 0;
        const out_offset = out.coordsToOffset(coords[0..out.ndim]);

        const current_min = toF32(out_data[out_offset]);
        const value = toF32(in_data[in_offset]);
        out_data[out_offset] = fromF32(@min(current_min, value));
    }
}

/// ArgMax reduction along a specific axis - returns indices of maximum values.
/// Output shape is input shape with the specified axis dimension = 1.
/// Output dtype is always u32 (indices).
pub fn argmax(out: TensorView, input: TensorView, axis: usize) void {
    std.debug.assert(axis < input.ndim);
    std.debug.assert(out.ndim == input.ndim);
    std.debug.assert(out.shape[axis] == 1);
    std.debug.assert(out.dtype == .u32);

    switch (input.dtype) {
        .f32 => argmaxGeneric(f32, f32Identity, out, input, axis),
        .f16 => argmaxGeneric(u16, fp16ToF32, out, input, axis),
        .bf16 => argmaxGeneric(u16, bf16ToF32, out, input, axis),
        else => unreachable,
    }
}

fn argmaxGeneric(
    comptime T: type,
    comptime toF32: fn (T) f32,
    out: TensorView,
    input: TensorView,
    axis: usize,
) void {
    const out_data = @as([*]u32, @ptrCast(@alignCast(out.data)));
    const in_data = @as([*]const T, @ptrCast(@alignCast(input.data)));

    // Initialize tracking arrays
    const max_values = std.heap.page_allocator.alloc(f32, out.numel) catch unreachable;
    defer std.heap.page_allocator.free(max_values);

    for (0..out.numel) |i| {
        max_values[i] = -std.math.inf(f32);
        out_data[i] = 0;
    }

    var coords: [tv.MAX_NDIM]usize = undefined;
    for (0..input.numel) |elem_idx| {
        input.indexToCoords(elem_idx, &coords);
        const in_offset = input.coordsToOffset(coords[0..input.ndim]);
        const axis_index = coords[axis];

        coords[axis] = 0;
        const out_offset = out.coordsToOffset(coords[0..out.ndim]);

        const value = toF32(in_data[in_offset]);
        if (value > max_values[out_offset]) {
            max_values[out_offset] = value;
            out_data[out_offset] = @intCast(axis_index);
        }
    }
}

/// ArgMin reduction along a specific axis - returns indices of minimum values.
/// Output shape is input shape with the specified axis dimension = 1.
/// Output dtype is always u32 (indices).
pub fn argmin(out: TensorView, input: TensorView, axis: usize) void {
    std.debug.assert(axis < input.ndim);
    std.debug.assert(out.ndim == input.ndim);
    std.debug.assert(out.shape[axis] == 1);
    std.debug.assert(out.dtype == .u32);

    switch (input.dtype) {
        .f32 => argminGeneric(f32, f32Identity, out, input, axis),
        .f16 => argminGeneric(u16, fp16ToF32, out, input, axis),
        .bf16 => argminGeneric(u16, bf16ToF32, out, input, axis),
        else => unreachable,
    }
}

fn argminGeneric(
    comptime T: type,
    comptime toF32: fn (T) f32,
    out: TensorView,
    input: TensorView,
    axis: usize,
) void {
    const out_data = @as([*]u32, @ptrCast(@alignCast(out.data)));
    const in_data = @as([*]const T, @ptrCast(@alignCast(input.data)));

    // Initialize tracking arrays
    const min_values = std.heap.page_allocator.alloc(f32, out.numel) catch unreachable;
    defer std.heap.page_allocator.free(min_values);

    for (0..out.numel) |i| {
        min_values[i] = std.math.inf(f32);
        out_data[i] = 0;
    }

    var coords: [tv.MAX_NDIM]usize = undefined;
    for (0..input.numel) |elem_idx| {
        input.indexToCoords(elem_idx, &coords);
        const in_offset = input.coordsToOffset(coords[0..input.ndim]);
        const axis_index = coords[axis];

        coords[axis] = 0;
        const out_offset = out.coordsToOffset(coords[0..out.ndim]);

        const value = toF32(in_data[in_offset]);
        if (value < min_values[out_offset]) {
            min_values[out_offset] = value;
            out_data[out_offset] = @intCast(axis_index);
        }
    }
}

/// Variance reduction along a specific axis.
/// Output shape is input shape with the specified axis dimension = 1.
pub fn variance(out: TensorView, input: TensorView, axis: usize, correction: f32) void {
    std.debug.assert(axis < input.ndim);
    std.debug.assert(out.ndim == input.ndim);
    std.debug.assert(out.shape[axis] == 1);

    switch (input.dtype) {
        .f32 => varianceGeneric(f32, f32Identity, f32Identity, out, input, axis, correction),
        .f16 => varianceGeneric(u16, fp16ToF32, f32ToFp16, out, input, axis, correction),
        .bf16 => varianceGeneric(u16, bf16ToF32, f32ToBf16, out, input, axis, correction),
        else => unreachable,
    }
}

fn varianceGeneric(
    comptime T: type,
    comptime toF32: fn (T) f32,
    comptime fromF32: fn (f32) T,
    out: TensorView,
    input: TensorView,
    axis: usize,
    correction: f32,
) void {
    const out_data = @as([*]T, @ptrCast(@alignCast(out.data)));
    const in_data = @as([*]const T, @ptrCast(@alignCast(input.data)));

    // Allocate temporary storage for means
    const means = std.heap.page_allocator.alloc(f32, out.numel) catch unreachable;
    defer std.heap.page_allocator.free(means);

    // Initialize
    for (0..out.numel) |i| {
        means[i] = 0.0;
        out_data[i] = fromF32(0.0);
    }

    // First pass: compute means
    var coords: [tv.MAX_NDIM]usize = undefined;
    for (0..input.numel) |elem_idx| {
        input.indexToCoords(elem_idx, &coords);
        const in_offset = input.coordsToOffset(coords[0..input.ndim]);

        coords[axis] = 0;
        const out_offset = out.coordsToOffset(coords[0..out.ndim]);

        const value = toF32(in_data[in_offset]);
        means[out_offset] += value;
    }

    const count = @as(f32, @floatFromInt(input.shape[axis]));
    for (0..out.numel) |i| {
        means[i] /= count;
    }

    // Second pass: compute variance
    for (0..input.numel) |elem_idx| {
        input.indexToCoords(elem_idx, &coords);
        const in_offset = input.coordsToOffset(coords[0..input.ndim]);

        coords[axis] = 0;
        const out_offset = out.coordsToOffset(coords[0..out.ndim]);

        const value = toF32(in_data[in_offset]);
        const diff = value - means[out_offset];
        const current_var = toF32(out_data[out_offset]);
        out_data[out_offset] = fromF32(current_var + diff * diff);
    }

    // Divide by (n - correction)
    const divisor = count - correction;
    for (0..out.numel) |i| {
        const var_sum = toF32(out_data[i]);
        out_data[i] = fromF32(var_sum / divisor);
    }
}

/// Standard deviation reduction along a specific axis.
/// Output shape is input shape with the specified axis dimension = 1.
pub fn std_dev(out: TensorView, input: TensorView, axis: usize, correction: f32) void {
    // First compute variance
    variance(out, input, axis, correction);

    // Then take sqrt
    switch (out.dtype) {
        .f32 => stdSqrtGeneric(f32, f32Identity, f32Identity, out),
        .f16 => stdSqrtGeneric(u16, fp16ToF32, f32ToFp16, out),
        .bf16 => stdSqrtGeneric(u16, bf16ToF32, f32ToBf16, out),
        else => unreachable,
    }
}

fn stdSqrtGeneric(
    comptime T: type,
    comptime toF32: fn (T) f32,
    comptime fromF32: fn (f32) T,
    out: TensorView,
) void {
    const out_data = @as([*]T, @ptrCast(@alignCast(out.data)));

    for (0..out.numel) |i| {
        const value = toF32(out_data[i]);
        out_data[i] = fromF32(@sqrt(value));
    }
}

// ============================================================================
// Tests
// ============================================================================

test "sum 1D" {
    var in_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var out_data = [_]f32{0.0};

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{5}, .f32);
    const output = TensorView.initContiguous(@ptrCast(&out_data), &.{1}, .f32);

    sum(output, input, 0);

    try std.testing.expectApproxEqAbs(@as(f32, 15.0), out_data[0], 1e-6);
}

test "sum 2D along axis 0" {
    var in_data = [_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };
    var out_data = [_]f32{ 0.0, 0.0, 0.0 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{ 2, 3 }, .f32);
    const output = TensorView.initContiguous(@ptrCast(&out_data), &.{ 1, 3 }, .f32);

    sum(output, input, 0);

    try std.testing.expectApproxEqAbs(@as(f32, 5.0), out_data[0], 1e-6); // 1+4
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), out_data[1], 1e-6); // 2+5
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), out_data[2], 1e-6); // 3+6
}

test "sum 2D along axis 1" {
    var in_data = [_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };
    var out_data = [_]f32{ 0.0, 0.0 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{ 2, 3 }, .f32);
    const output = TensorView.initContiguous(@ptrCast(&out_data), &.{ 2, 1 }, .f32);

    sum(output, input, 1);

    try std.testing.expectApproxEqAbs(@as(f32, 6.0), out_data[0], 1e-6);  // 1+2+3
    try std.testing.expectApproxEqAbs(@as(f32, 15.0), out_data[1], 1e-6); // 4+5+6
}

test "sum 3D along axis 1" {
    var in_data = [_]f32{
        1.0, 2.0,  // [0,0,:]
        3.0, 4.0,  // [0,1,:]
        5.0, 6.0,  // [1,0,:]
        7.0, 8.0,  // [1,1,:]
    };
    var out_data = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{ 2, 2, 2 }, .f32);
    const output = TensorView.initContiguous(@ptrCast(&out_data), &.{ 2, 1, 2 }, .f32);

    sum(output, input, 1);

    try std.testing.expectApproxEqAbs(@as(f32, 4.0), out_data[0], 1e-6);  // 1+3
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), out_data[1], 1e-6);  // 2+4
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), out_data[2], 1e-6); // 5+7
    try std.testing.expectApproxEqAbs(@as(f32, 14.0), out_data[3], 1e-6); // 6+8
}

test "mean 1D" {
    var in_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var out_data = [_]f32{0.0};

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{5}, .f32);
    const output = TensorView.initContiguous(@ptrCast(&out_data), &.{1}, .f32);

    mean(output, input, 0);

    try std.testing.expectApproxEqAbs(@as(f32, 3.0), out_data[0], 1e-6);
}

test "mean 2D along axis 0" {
    var in_data = [_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };
    var out_data = [_]f32{ 0.0, 0.0, 0.0 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{ 2, 3 }, .f32);
    const output = TensorView.initContiguous(@ptrCast(&out_data), &.{ 1, 3 }, .f32);

    mean(output, input, 0);

    try std.testing.expectApproxEqAbs(@as(f32, 2.5), out_data[0], 1e-6); // (1+4)/2
    try std.testing.expectApproxEqAbs(@as(f32, 3.5), out_data[1], 1e-6); // (2+5)/2
    try std.testing.expectApproxEqAbs(@as(f32, 4.5), out_data[2], 1e-6); // (3+6)/2
}

test "mean 2D along axis 1" {
    var in_data = [_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };
    var out_data = [_]f32{ 0.0, 0.0 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{ 2, 3 }, .f32);
    const output = TensorView.initContiguous(@ptrCast(&out_data), &.{ 2, 1 }, .f32);

    mean(output, input, 1);

    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out_data[0], 1e-6); // (1+2+3)/3
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), out_data[1], 1e-6); // (4+5+6)/3
}

test "max 1D" {
    var in_data = [_]f32{ 3.0, 1.0, 5.0, 2.0, 4.0 };
    var out_data = [_]f32{0.0};

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{5}, .f32);
    const output = TensorView.initContiguous(@ptrCast(&out_data), &.{1}, .f32);

    max(output, input, 0);

    try std.testing.expectApproxEqAbs(@as(f32, 5.0), out_data[0], 1e-6);
}

test "max 2D along axis 0" {
    var in_data = [_]f32{
        1.0, 5.0, 3.0,
        4.0, 2.0, 6.0,
    };
    var out_data = [_]f32{ 0.0, 0.0, 0.0 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{ 2, 3 }, .f32);
    const output = TensorView.initContiguous(@ptrCast(&out_data), &.{ 1, 3 }, .f32);

    max(output, input, 0);

    try std.testing.expectApproxEqAbs(@as(f32, 4.0), out_data[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), out_data[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), out_data[2], 1e-6);
}

test "max 2D along axis 1" {
    var in_data = [_]f32{
        1.0, 5.0, 3.0,
        4.0, 2.0, 6.0,
    };
    var out_data = [_]f32{ 0.0, 0.0 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{ 2, 3 }, .f32);
    const output = TensorView.initContiguous(@ptrCast(&out_data), &.{ 2, 1 }, .f32);

    max(output, input, 1);

    try std.testing.expectApproxEqAbs(@as(f32, 5.0), out_data[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), out_data[1], 1e-6);
}

test "min 1D" {
    var in_data = [_]f32{ 3.0, 1.0, 5.0, 2.0, 4.0 };
    var out_data = [_]f32{0.0};

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{5}, .f32);
    const output = TensorView.initContiguous(@ptrCast(&out_data), &.{1}, .f32);

    min(output, input, 0);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out_data[0], 1e-6);
}

test "min 2D along axis 0" {
    var in_data = [_]f32{
        1.0, 5.0, 3.0,
        4.0, 2.0, 6.0,
    };
    var out_data = [_]f32{ 0.0, 0.0, 0.0 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{ 2, 3 }, .f32);
    const output = TensorView.initContiguous(@ptrCast(&out_data), &.{ 1, 3 }, .f32);

    min(output, input, 0);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out_data[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out_data[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), out_data[2], 1e-6);
}

test "min 2D along axis 1" {
    var in_data = [_]f32{
        1.0, 5.0, 3.0,
        4.0, 2.0, 6.0,
    };
    var out_data = [_]f32{ 0.0, 0.0 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{ 2, 3 }, .f32);
    const output = TensorView.initContiguous(@ptrCast(&out_data), &.{ 2, 1 }, .f32);

    min(output, input, 1);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out_data[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out_data[1], 1e-6);
}

test "argmax 1D" {
    var in_data = [_]f32{ 3.0, 1.0, 5.0, 2.0, 4.0 };
    var out_data = [_]u32{0};

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{5}, .f32);
    const output = TensorView.initContiguous(@ptrCast(&out_data), &.{1}, .u32);

    argmax(output, input, 0);

    try std.testing.expectEqual(@as(u32, 2), out_data[0]); // Index of 5.0
}

test "argmax 2D along axis 0" {
    var in_data = [_]f32{
        1.0, 5.0, 3.0,
        4.0, 2.0, 6.0,
    };
    var out_data = [_]u32{ 0, 0, 0 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{ 2, 3 }, .f32);
    const output = TensorView.initContiguous(@ptrCast(&out_data), &.{ 1, 3 }, .u32);

    argmax(output, input, 0);

    try std.testing.expectEqual(@as(u32, 1), out_data[0]); // 4.0 is at index 1
    try std.testing.expectEqual(@as(u32, 0), out_data[1]); // 5.0 is at index 0
    try std.testing.expectEqual(@as(u32, 1), out_data[2]); // 6.0 is at index 1
}

test "argmax 2D along axis 1" {
    var in_data = [_]f32{
        1.0, 5.0, 3.0,
        4.0, 2.0, 6.0,
    };
    var out_data = [_]u32{ 0, 0 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{ 2, 3 }, .f32);
    const output = TensorView.initContiguous(@ptrCast(&out_data), &.{ 2, 1 }, .u32);

    argmax(output, input, 1);

    try std.testing.expectEqual(@as(u32, 1), out_data[0]); // 5.0 is at index 1
    try std.testing.expectEqual(@as(u32, 2), out_data[1]); // 6.0 is at index 2
}

test "argmin 1D" {
    var in_data = [_]f32{ 3.0, 1.0, 5.0, 2.0, 4.0 };
    var out_data = [_]u32{0};

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{5}, .f32);
    const output = TensorView.initContiguous(@ptrCast(&out_data), &.{1}, .u32);

    argmin(output, input, 0);

    try std.testing.expectEqual(@as(u32, 1), out_data[0]); // Index of 1.0
}

test "argmin 2D along axis 0" {
    var in_data = [_]f32{
        1.0, 5.0, 3.0,
        4.0, 2.0, 6.0,
    };
    var out_data = [_]u32{ 0, 0, 0 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{ 2, 3 }, .f32);
    const output = TensorView.initContiguous(@ptrCast(&out_data), &.{ 1, 3 }, .u32);

    argmin(output, input, 0);

    try std.testing.expectEqual(@as(u32, 0), out_data[0]); // 1.0 is at index 0
    try std.testing.expectEqual(@as(u32, 1), out_data[1]); // 2.0 is at index 1
    try std.testing.expectEqual(@as(u32, 0), out_data[2]); // 3.0 is at index 0
}

test "argmin 2D along axis 1" {
    var in_data = [_]f32{
        1.0, 5.0, 3.0,
        4.0, 2.0, 6.0,
    };
    var out_data = [_]u32{ 0, 0 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{ 2, 3 }, .f32);
    const output = TensorView.initContiguous(@ptrCast(&out_data), &.{ 2, 1 }, .u32);

    argmin(output, input, 1);

    try std.testing.expectEqual(@as(u32, 0), out_data[0]); // 1.0 is at index 0
    try std.testing.expectEqual(@as(u32, 1), out_data[1]); // 2.0 is at index 1
}

test "variance 1D" {
    var in_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var out_data = [_]f32{0.0};

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{5}, .f32);
    const output = TensorView.initContiguous(@ptrCast(&out_data), &.{1}, .f32);

    variance(output, input, 0, 1.0); // Bessel's correction

    // Mean = 3.0, variance with correction = ((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 4
    // = (4 + 1 + 0 + 1 + 4) / 4 = 10 / 4 = 2.5
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), out_data[0], 1e-6);
}

test "variance 2D along axis 0" {
    var in_data = [_]f32{
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    };
    var out_data = [_]f32{ 0.0, 0.0 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{ 3, 2 }, .f32);
    const output = TensorView.initContiguous(@ptrCast(&out_data), &.{ 1, 2 }, .f32);

    variance(output, input, 0, 0.0); // No correction

    // Column 0: mean = 3.0, var = ((1-3)^2 + (3-3)^2 + (5-3)^2) / 3 = (4+0+4)/3 = 8/3
    // Column 1: mean = 4.0, var = ((2-4)^2 + (4-4)^2 + (6-4)^2) / 3 = (4+0+4)/3 = 8/3
    try std.testing.expectApproxEqAbs(@as(f32, 8.0 / 3.0), out_data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0 / 3.0), out_data[1], 1e-5);
}

test "variance 2D along axis 1" {
    var in_data = [_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };
    var out_data = [_]f32{ 0.0, 0.0 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{ 2, 3 }, .f32);
    const output = TensorView.initContiguous(@ptrCast(&out_data), &.{ 2, 1 }, .f32);

    variance(output, input, 1, 0.0); // No correction

    // Row 0: mean = 2.0, var = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = (1+0+1)/3 = 2/3
    // Row 1: mean = 5.0, var = ((4-5)^2 + (5-5)^2 + (6-5)^2) / 3 = (1+0+1)/3 = 2/3
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 / 3.0), out_data[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 / 3.0), out_data[1], 1e-6);
}

test "std_dev 1D" {
    var in_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var out_data = [_]f32{0.0};

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{5}, .f32);
    const output = TensorView.initContiguous(@ptrCast(&out_data), &.{1}, .f32);

    std_dev(output, input, 0, 1.0); // Bessel's correction

    // Variance = 2.5, std = sqrt(2.5) ≈ 1.5811
    try std.testing.expectApproxEqAbs(@as(f32, 1.5811), out_data[0], 1e-4);
}

test "std_dev 2D along axis 0" {
    var in_data = [_]f32{
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    };
    var out_data = [_]f32{ 0.0, 0.0 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{ 3, 2 }, .f32);
    const output = TensorView.initContiguous(@ptrCast(&out_data), &.{ 1, 2 }, .f32);

    std_dev(output, input, 0, 0.0); // No correction

    // std = sqrt(8/3) ≈ 1.6330
    try std.testing.expectApproxEqAbs(@as(f32, 1.6330), out_data[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.6330), out_data[1], 1e-4);
}

test "sum mean max min argmax argmin single element" {
    var in_data = [_]f32{42.0};
    var out_data = [_]f32{0.0};

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{1}, .f32);
    const output = TensorView.initContiguous(@ptrCast(&out_data), &.{1}, .f32);

    sum(output, input, 0);
    try std.testing.expectApproxEqAbs(@as(f32, 42.0), out_data[0], 1e-6);

    mean(output, input, 0);
    try std.testing.expectApproxEqAbs(@as(f32, 42.0), out_data[0], 1e-6);

    max(output, input, 0);
    try std.testing.expectApproxEqAbs(@as(f32, 42.0), out_data[0], 1e-6);

    min(output, input, 0);
    try std.testing.expectApproxEqAbs(@as(f32, 42.0), out_data[0], 1e-6);
}

test "sum mean max min negative values" {
    var in_data = [_]f32{ -3.0, -1.0, -2.0, -5.0, -4.0 };
    var out_data = [_]f32{0.0};

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{5}, .f32);
    const output = TensorView.initContiguous(@ptrCast(&out_data), &.{1}, .f32);

    sum(output, input, 0);
    try std.testing.expectApproxEqAbs(@as(f32, -15.0), out_data[0], 1e-6);

    mean(output, input, 0);
    try std.testing.expectApproxEqAbs(@as(f32, -3.0), out_data[0], 1e-6);

    max(output, input, 0);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), out_data[0], 1e-6);

    min(output, input, 0);
    try std.testing.expectApproxEqAbs(@as(f32, -5.0), out_data[0], 1e-6);
}

test "sum mean max min mixed positive negative" {
    var in_data = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    var out_data = [_]f32{0.0};

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{5}, .f32);
    const output = TensorView.initContiguous(@ptrCast(&out_data), &.{1}, .f32);

    sum(output, input, 0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out_data[0], 1e-6);

    mean(output, input, 0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out_data[0], 1e-6);
}

test "multi-dtype support: sum with f16" {
    const allocator = std.testing.allocator;

    const in_data_f32 = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const in_data_f16 = try allocator.alloc(u16, 4);
    defer allocator.free(in_data_f16);
    const out_data_f16 = try allocator.alloc(u16, 1);
    defer allocator.free(out_data_f16);

    for (in_data_f32, 0..) |val, i| {
        in_data_f16[i] = f32ToFp16(val);
    }

    const input = TensorView.initContiguous(@ptrCast(in_data_f16.ptr), &.{4}, .f16);
    const output = TensorView.initContiguous(@ptrCast(out_data_f16.ptr), &.{1}, .f16);

    sum(output, input, 0);

    const result = fp16ToF32(out_data_f16[0]);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), result, 1e-2);
}

test "multi-dtype support: mean with bf16" {
    const allocator = std.testing.allocator;

    const in_data_f32 = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const in_data_bf16 = try allocator.alloc(u16, 4);
    defer allocator.free(in_data_bf16);
    const out_data_bf16 = try allocator.alloc(u16, 1);
    defer allocator.free(out_data_bf16);

    for (in_data_f32, 0..) |val, i| {
        in_data_bf16[i] = f32ToBf16(val);
    }

    const input = TensorView.initContiguous(@ptrCast(in_data_bf16.ptr), &.{4}, .bf16);
    const output = TensorView.initContiguous(@ptrCast(out_data_bf16.ptr), &.{1}, .bf16);

    mean(output, input, 0);

    const result = bf16ToF32(out_data_bf16[0]);
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), result, 1e-2);
}

test "3D tensor: sum along each axis" {
    var in_data = [_]f32{
        1.0,  2.0,  3.0,  4.0,  // [0,0,:] and [0,1,:]
        5.0,  6.0,  7.0,  8.0,  // [1,0,:] and [1,1,:]
        9.0,  10.0, 11.0, 12.0, // [2,0,:] and [2,1,:]
    };

    // Shape: [3, 2, 2]
    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{ 3, 2, 2 }, .f32);

    // Test axis 0: reduce first dimension
    {
        var out_data = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
        const output = TensorView.initContiguous(@ptrCast(&out_data), &.{ 1, 2, 2 }, .f32);
        sum(output, input, 0);

        // [0,:,:] = [1,2,3,4], [1,:,:] = [5,6,7,8], [2,:,:] = [9,10,11,12]
        // Sum: [15, 18, 21, 24]
        try std.testing.expectApproxEqAbs(@as(f32, 15.0), out_data[0], 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, 18.0), out_data[1], 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, 21.0), out_data[2], 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, 24.0), out_data[3], 1e-6);
    }

    // Test axis 1: reduce second dimension
    {
        var out_data = [_]f32{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
        const output = TensorView.initContiguous(@ptrCast(&out_data), &.{ 3, 1, 2 }, .f32);
        sum(output, input, 1);

        // [:,0,:] = [[1,2], [5,6], [9,10]]
        // [:,1,:] = [[3,4], [7,8], [11,12]]
        // Sum along axis 1: [[4,6], [12,14], [20,22]]
        try std.testing.expectApproxEqAbs(@as(f32, 4.0), out_data[0], 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, 6.0), out_data[1], 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, 12.0), out_data[2], 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, 14.0), out_data[3], 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, 20.0), out_data[4], 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, 22.0), out_data[5], 1e-6);
    }

    // Test axis 2: reduce last dimension
    {
        var out_data = [_]f32{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
        const output = TensorView.initContiguous(@ptrCast(&out_data), &.{ 3, 2, 1 }, .f32);
        sum(output, input, 2);

        // [:,:,0] = [[1,3], [5,7], [9,11]]
        // [:,:,1] = [[2,4], [6,8], [10,12]]
        // Sum along axis 2: [[3,7], [11,15], [19,23]]
        try std.testing.expectApproxEqAbs(@as(f32, 3.0), out_data[0], 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, 7.0), out_data[1], 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, 11.0), out_data[2], 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, 15.0), out_data[3], 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, 19.0), out_data[4], 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, 23.0), out_data[5], 1e-6);
    }
}

test "sum mean variance numerical stability large" {
    var in_data = [_]f32{ 1e6, 2e6, 3e6, 4e6, 5e6 };
    var out_data = [_]f32{0.0};

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{5}, .f32);
    const output = TensorView.initContiguous(@ptrCast(&out_data), &.{1}, .f32);

    sum(output, input, 0);
    try std.testing.expectApproxEqRel(@as(f32, 15e6), out_data[0], 1e-5);

    mean(output, input, 0);
    try std.testing.expectApproxEqRel(@as(f32, 3e6), out_data[0], 1e-5);
}

test "sum mean variance numerical stability small" {
    var in_data = [_]f32{ 1e-6, 2e-6, 3e-6, 4e-6, 5e-6 };
    var out_data = [_]f32{0.0};

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{5}, .f32);
    const output = TensorView.initContiguous(@ptrCast(&out_data), &.{1}, .f32);

    sum(output, input, 0);
    try std.testing.expectApproxEqRel(@as(f32, 15e-6), out_data[0], 1e-4);

    mean(output, input, 0);
    try std.testing.expectApproxEqRel(@as(f32, 3e-6), out_data[0], 1e-4);
}

test "argmax with duplicate max values - returns first occurrence" {
    var in_data = [_]f32{ 1.0, 5.0, 3.0, 5.0, 2.0 };
    var out_data = [_]u32{0};

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{5}, .f32);
    const output = TensorView.initContiguous(@ptrCast(&out_data), &.{1}, .u32);

    argmax(output, input, 0);

    try std.testing.expectEqual(@as(u32, 1), out_data[0]); // First occurrence of 5.0
}

test "argmin with duplicate min values - returns first occurrence" {
    var in_data = [_]f32{ 3.0, 1.0, 5.0, 1.0, 2.0 };
    var out_data = [_]u32{0};

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{5}, .f32);
    const output = TensorView.initContiguous(@ptrCast(&out_data), &.{1}, .u32);

    argmin(output, input, 0);

    try std.testing.expectEqual(@as(u32, 1), out_data[0]); // First occurrence of 1.0
}

test "variance with Bessel's correction vs no correction" {
    var in_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var out_data_no_corr = [_]f32{0.0};
    var out_data_bessel = [_]f32{0.0};

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{5}, .f32);
    const output_no_corr = TensorView.initContiguous(@ptrCast(&out_data_no_corr), &.{1}, .f32);
    const output_bessel = TensorView.initContiguous(@ptrCast(&out_data_bessel), &.{1}, .f32);

    variance(output_no_corr, input, 0, 0.0); // No correction
    variance(output_bessel, input, 0, 1.0); // Bessel's correction

    // No correction: sum((x - mean)^2) / n = 10 / 5 = 2.0
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out_data_no_corr[0], 1e-6);

    // Bessel's correction: sum((x - mean)^2) / (n - 1) = 10 / 4 = 2.5
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), out_data_bessel[0], 1e-6);
}
