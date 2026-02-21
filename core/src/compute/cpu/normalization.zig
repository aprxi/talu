//! Normalization primitives for CPU compute path.

const std = @import("std");
const tensor = @import("../../tensor.zig");
const math_ops = @import("math.zig");
const tv = @import("tensor_view.zig");
const norm_view = @import("norm_primitives.zig");

const Tensor = tensor.Tensor;

/// TensorView RMSNorm wrapper for strided/typed inputs.
pub fn rmsNorm(out: tv.TensorView, input: tv.TensorView, weight: tv.TensorView, eps: f32) void {
    norm_view.rmsNorm(out, input, weight, eps);
}

/// TensorView LayerNorm wrapper for strided/typed inputs.
pub fn layerNorm(out: tv.TensorView, input: tv.TensorView, weight: tv.TensorView, bias: ?tv.TensorView, eps: f32) void {
    norm_view.layerNorm(out, input, weight, bias, eps);
}

/// RMSNorm in-place on f32 slices with f32 weights.
pub fn rmsnormInPlace(x: []f32, weight: []const f32, eps: f32) void {
    std.debug.assert(x.len == weight.len);
    if (x.len == 0) return;

    var sum_sq: f32 = 0;
    for (x) |value| {
        sum_sq += value * value;
    }
    const rms = @sqrt(sum_sq / @as(f32, @floatFromInt(x.len)) + eps);
    const inv_rms = 1.0 / rms;
    for (x, 0..) |*value, index| {
        value.* = value.* * inv_rms * weight[index];
    }
}

/// RMSNorm in-place with tensor-backed weights (f32/f16/bf16).
pub fn rmsnormInPlaceWeightTensor(vec: []f32, weight_tensor: *const Tensor, eps: f32, weight_offset: f32) void {
    math_ops.rmsnormInPlaceWeightTensor(vec, weight_tensor, eps, weight_offset);
}

/// Apply RMSNorm over a `[token_count, head_count, head_dim]` view packed with
/// row stride `row_stride`.
pub fn rmsnormHeadsInPlace(
    values: []f32,
    token_count: usize,
    head_count: usize,
    head_dim: usize,
    row_stride: usize,
    weight_tensor: *const Tensor,
    eps: f32,
    weight_offset: f32,
) void {
    std.debug.assert(row_stride >= head_count * head_dim);
    std.debug.assert(values.len >= token_count * row_stride);
    for (0..token_count) |token_idx| {
        for (0..head_count) |head_idx| {
            const offset = token_idx * row_stride + head_idx * head_dim;
            rmsnormInPlaceWeightTensor(values[offset .. offset + head_dim], weight_tensor, eps, weight_offset);
        }
    }
}

/// Apply tensor-backed RMSNorm over contiguous rows `[row_count, row_dim]`.
pub fn rmsnormContiguousWeightTensor(
    values: []f32,
    row_count: usize,
    row_dim: usize,
    weight_tensor: *const Tensor,
    eps: f32,
    weight_offset: f32,
) !void {
    if (row_dim == 0) return error.InvalidShape;
    if (values.len < row_count * row_dim) return error.InvalidShape;
    for (0..row_count) |row_idx| {
        const offset = row_idx * row_dim;
        rmsnormInPlaceWeightTensor(values[offset .. offset + row_dim], weight_tensor, eps, weight_offset);
    }
}

/// LayerNorm over one contiguous row.
pub fn layerNormRow(
    input: []const f32,
    output: []f32,
    weight: []const f32,
    bias: []const f32,
    eps: f32,
) void {
    std.debug.assert(input.len == output.len);
    std.debug.assert(weight.len == input.len);
    std.debug.assert(bias.len == input.len);

    var mean: f32 = 0.0;
    for (input) |v| mean += v;
    mean /= @as(f32, @floatFromInt(input.len));

    var var_sum: f32 = 0.0;
    for (input) |v| {
        const d = v - mean;
        var_sum += d * d;
    }
    const variance = var_sum / @as(f32, @floatFromInt(input.len));
    const inv_std = 1.0 / @sqrt(variance + eps);

    for (0..input.len) |i| {
        output[i] = (input[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

/// LayerNorm over contiguous rows `[row_count, row_width]`.
pub fn layerNormRows(
    input: []const f32,
    output: []f32,
    row_count: usize,
    row_width: usize,
    weight: []const f32,
    bias: []const f32,
    eps: f32,
) !void {
    if (weight.len != row_width or bias.len != row_width) return error.InvalidShape;
    if (input.len < row_count * row_width) return error.InvalidShape;
    if (output.len < row_count * row_width) return error.InvalidShape;

    for (0..row_count) |row_idx| {
        const in_row = input[row_idx * row_width ..][0..row_width];
        const out_row = output[row_idx * row_width ..][0..row_width];
        layerNormRow(in_row, out_row, weight, bias, eps);
    }
}

test "rmsnormInPlace keeps unit scale for ones" {
    var x = [_]f32{ 1, 1, 1, 1 };
    const w = [_]f32{ 1, 1, 1, 1 };
    rmsnormInPlace(&x, &w, 1e-5);
    for (x) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), v, 1e-3);
    }
}

test "rmsnormInPlaceWeightTensor handles f32 weights" {
    const allocator = std.testing.allocator;
    var w = try tensor.OwnedTensor.init(allocator, .f32, &.{4});
    defer w.deinit();
    const ws = w.asSlice(f32);
    @memcpy(ws, &[_]f32{ 1, 1, 1, 1 });

    var x = [_]f32{ 1, 2, 3, 4 };
    var w_view = w.view();
    rmsnormInPlaceWeightTensor(&x, &w_view, 1e-6, 0.0);

    var sq_sum: f32 = 0;
    for (x) |v| sq_sum += v * v;
    const rms = @sqrt(sq_sum / 4.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), rms, 1e-3);
}

test "rmsnormHeadsInPlace normalizes each head slice" {
    const allocator = std.testing.allocator;
    var w = try tensor.OwnedTensor.init(allocator, .f32, &.{2});
    defer w.deinit();
    const ws = w.asSlice(f32);
    @memcpy(ws, &[_]f32{ 1, 1 });

    var x = [_]f32{
        1, 2, // token0 head0
        3, 4, // token0 head1
        5, 6, // token1 head0
        7, 8, // token1 head1
    };
    var wv = w.view();
    rmsnormHeadsInPlace(&x, 2, 2, 2, 4, &wv, 1e-6, 0.0);

    // Spot-check first head scaled to finite normalized values.
    try std.testing.expect(std.math.isFinite(x[0]));
    try std.testing.expect(std.math.isFinite(x[1]));
    try std.testing.expect(std.math.isFinite(x[2]));
    try std.testing.expect(std.math.isFinite(x[3]));
}

test "rmsnormContiguousWeightTensor normalizes each row" {
    const allocator = std.testing.allocator;
    var w = try tensor.OwnedTensor.init(allocator, .f32, &.{2});
    defer w.deinit();
    @memcpy(w.asSlice(f32), &[_]f32{ 1.0, 1.0 });
    var wv = w.view();

    var values = [_]f32{
        1.0, 2.0,
        3.0, 4.0,
    };
    try rmsnormContiguousWeightTensor(&values, 2, 2, &wv, 1e-6, 0.0);

    try std.testing.expect(std.math.isFinite(values[0]));
    try std.testing.expect(std.math.isFinite(values[1]));
    try std.testing.expect(std.math.isFinite(values[2]));
    try std.testing.expect(std.math.isFinite(values[3]));
}

test "rmsNorm wrapper runs TensorView RMSNorm" {
    var input_data = [_]f32{ 1.0, 2.0 };
    var out_data = [_]f32{ 0.0, 0.0 };
    var weight_data = [_]f32{ 1.0, 1.0 };

    const input = tv.TensorView.initContiguous(@ptrCast(&input_data), &.{ 1, 2 }, .f32);
    const out = tv.TensorView.initContiguous(@ptrCast(&out_data), &.{ 1, 2 }, .f32);
    const weight = tv.TensorView.initContiguous(@ptrCast(&weight_data), &.{2}, .f32);
    rmsNorm(out, input, weight, 1e-6);

    try std.testing.expect(std.math.isFinite(out_data[0]));
    try std.testing.expect(std.math.isFinite(out_data[1]));
}

test "layerNorm wrapper runs TensorView LayerNorm with bias" {
    var input_data = [_]f32{ 1.0, 2.0 };
    var out_data = [_]f32{ 0.0, 0.0 };
    var weight_data = [_]f32{ 1.0, 1.0 };
    var bias_data = [_]f32{ 0.0, 0.0 };

    const input = tv.TensorView.initContiguous(@ptrCast(&input_data), &.{ 1, 2 }, .f32);
    const out = tv.TensorView.initContiguous(@ptrCast(&out_data), &.{ 1, 2 }, .f32);
    const weight = tv.TensorView.initContiguous(@ptrCast(&weight_data), &.{2}, .f32);
    const bias = tv.TensorView.initContiguous(@ptrCast(&bias_data), &.{2}, .f32);
    layerNorm(out, input, weight, bias, 1e-6);

    try std.testing.expectApproxEqAbs(@as(f32, -1.0), out_data[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out_data[1], 0.01);
}

test "layerNormRow normalizes one row" {
    const input = [_]f32{ 1.0, 2.0, 3.0 };
    var out = [_]f32{ 0.0, 0.0, 0.0 };
    const weight = [_]f32{ 1.0, 1.0, 1.0 };
    const bias = [_]f32{ 0.0, 0.0, 0.0 };
    layerNormRow(&input, &out, &weight, &bias, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[1], 1e-5);
    try std.testing.expect(out[0] < 0.0);
    try std.testing.expect(out[2] > 0.0);
}

test "layerNormRows normalizes each row independently" {
    const input = [_]f32{
        1.0, 2.0,
        3.0, 4.0,
    };
    var out = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    const weight = [_]f32{ 1.0, 1.0 };
    const bias = [_]f32{ 0.0, 0.0 };
    try layerNormRows(&input, &out, 2, 2, &weight, &bias, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[0] + out[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[2] + out[3], 1e-4);
}
