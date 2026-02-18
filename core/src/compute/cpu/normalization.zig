//! Normalization primitives for CPU compute path.

const std = @import("std");
const tensor = @import("../../tensor.zig");
const math_ops = @import("../ops/math_primitives/root.zig");

const Tensor = tensor.Tensor;

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
