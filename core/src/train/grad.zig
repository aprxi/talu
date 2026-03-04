//! Gradient tensor types for training.
//!
//! All gradients are stored as f32 regardless of forward-pass dtype.
//! This matches PyTorch's mixed-precision pattern where forward is bf16/f16
//! but gradients and optimizer state are always f32.

const std = @import("std");
const tensor_mod = @import("../tensor.zig");

const Tensor = tensor_mod.Tensor;
const OwnedTensor = tensor_mod.OwnedTensor;
const DType = tensor_mod.DType;
const Allocator = std.mem.Allocator;

/// f32 gradient tensor with SIMD-aligned memory.
///
/// Wraps an OwnedTensor that is always f32, providing gradient-specific
/// operations (zeroing, accumulation, 2D views for matmul).
pub const GradTensor = struct {
    buffer: OwnedTensor,

    /// Create a zero-initialized gradient tensor with the given shape.
    /// Shape is specified as a slice of dimension sizes (e.g., &.{rows, cols}).
    pub fn init(allocator: Allocator, shape: []const usize) !GradTensor {
        const buffer = try OwnedTensor.init(allocator, .f32, shape);
        return .{ .buffer = buffer };
    }

    /// Zero all gradient values. Called between training steps.
    pub fn zero(self: *GradTensor) void {
        @memset(self.buffer.data, 0);
    }

    /// Get f32 slice for reading gradient values.
    pub fn asSlice(self: *const GradTensor) []const f32 {
        return self.buffer.asSlice(f32);
    }

    /// Get mutable f32 slice for writing gradient values.
    pub fn asSliceMut(self: *GradTensor) []f32 {
        return self.buffer.asSlice(f32);
    }

    /// Accumulate: self += other (element-wise addition).
    /// `other` must have the same length as self.
    pub fn accumulate(self: *GradTensor, other: []const f32) void {
        const dst = self.asSliceMut();
        std.debug.assert(dst.len == other.len);
        for (dst, other) |*d, o| {
            d.* += o;
        }
    }

    /// Scale all gradient values by a scalar.
    pub fn scale(self: *GradTensor, factor: f32) void {
        const dst = self.asSliceMut();
        for (dst) |*d| {
            d.* *= factor;
        }
    }

    /// Compute L2 norm of gradient values.
    pub fn norm(self: *const GradTensor) f32 {
        const data = self.asSlice();
        var sum_sq: f32 = 0.0;
        for (data) |v| {
            sum_sq += v * v;
        }
        return @sqrt(sum_sq);
    }

    /// Number of elements in the gradient tensor.
    pub fn numElements(self: *const GradTensor) usize {
        return self.buffer.numElements();
    }

    /// Get a 2D Tensor view for use with matmul operations.
    /// The underlying data must have exactly rows * cols elements.
    pub fn view2D(self: *const GradTensor, rows: usize, cols: usize) Tensor {
        const data = @as([]f32, @constCast(self.asSlice()));
        return Tensor.view2DSlice(data, rows, cols);
    }

    pub fn deinit(self: *GradTensor) void {
        self.buffer.deinit();
        self.* = undefined;
    }
};

// =============================================================================
// Tests
// =============================================================================

test "GradTensor init creates zero-filled f32 tensor" {
    const allocator = std.testing.allocator;
    var grad = try GradTensor.init(allocator, &.{ 3, 4 });
    defer grad.deinit();

    const data = grad.asSlice();
    try std.testing.expectEqual(@as(usize, 12), data.len);
    for (data) |v| {
        try std.testing.expectEqual(@as(f32, 0.0), v);
    }
}

test "GradTensor zero resets values" {
    const allocator = std.testing.allocator;
    var grad = try GradTensor.init(allocator, &.{4});
    defer grad.deinit();

    const data = grad.asSliceMut();
    data[0] = 1.0;
    data[1] = 2.0;
    data[2] = 3.0;
    data[3] = 4.0;

    grad.zero();
    for (grad.asSlice()) |v| {
        try std.testing.expectEqual(@as(f32, 0.0), v);
    }
}

test "GradTensor accumulate adds element-wise" {
    const allocator = std.testing.allocator;
    var grad = try GradTensor.init(allocator, &.{3});
    defer grad.deinit();

    const data = grad.asSliceMut();
    data[0] = 1.0;
    data[1] = 2.0;
    data[2] = 3.0;

    const other = [_]f32{ 0.5, 1.5, 2.5 };
    grad.accumulate(&other);

    try std.testing.expectApproxEqAbs(@as(f32, 1.5), grad.asSlice()[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.5), grad.asSlice()[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.5), grad.asSlice()[2], 1e-6);
}

test "GradTensor accumulate twice doubles values" {
    const allocator = std.testing.allocator;
    var grad = try GradTensor.init(allocator, &.{2});
    defer grad.deinit();

    const vals = [_]f32{ 1.0, 2.0 };
    grad.accumulate(&vals);
    grad.accumulate(&vals);

    try std.testing.expectApproxEqAbs(@as(f32, 2.0), grad.asSlice()[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), grad.asSlice()[1], 1e-6);
}

test "GradTensor scale multiplies all values" {
    const allocator = std.testing.allocator;
    var grad = try GradTensor.init(allocator, &.{3});
    defer grad.deinit();

    const data = grad.asSliceMut();
    data[0] = 2.0;
    data[1] = 4.0;
    data[2] = 6.0;

    grad.scale(0.5);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), grad.asSlice()[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), grad.asSlice()[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), grad.asSlice()[2], 1e-6);
}

test "GradTensor norm computes L2 norm" {
    const allocator = std.testing.allocator;
    var grad = try GradTensor.init(allocator, &.{2});
    defer grad.deinit();

    const data = grad.asSliceMut();
    data[0] = 3.0;
    data[1] = 4.0;

    try std.testing.expectApproxEqAbs(@as(f32, 5.0), grad.norm(), 1e-6);
}

test "GradTensor view2D returns correct tensor shape" {
    const allocator = std.testing.allocator;
    var grad = try GradTensor.init(allocator, &.{ 2, 3 });
    defer grad.deinit();

    const data = grad.asSliceMut();
    data[0] = 1.0;
    data[3] = 4.0;

    const view = grad.view2D(2, 3);
    try std.testing.expectEqual(@as(i32, 2), view.n_dims);
    try std.testing.expectEqual(@as(usize, 2), @as(usize, @intCast(view.shape[0])));
    try std.testing.expectEqual(@as(usize, 3), @as(usize, @intCast(view.shape[1])));
}

test "GradTensor numElements matches shape" {
    const allocator = std.testing.allocator;
    var grad = try GradTensor.init(allocator, &.{ 5, 3 });
    defer grad.deinit();

    try std.testing.expectEqual(@as(usize, 15), grad.numElements());
}
