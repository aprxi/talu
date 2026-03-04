//! Integration tests for GradTensor.

const std = @import("std");
const lib = @import("main");
const train = lib.train;
const GradTensor = train.GradTensor;

test "GradTensor lifecycle: init, write, read, zero, deinit" {
    const allocator = std.testing.allocator;
    var grad = try GradTensor.init(allocator, &.{ 4, 8 });
    defer grad.deinit();

    // Verify zero-initialized
    try std.testing.expectEqual(@as(usize, 32), grad.numElements());
    for (grad.asSlice()) |v| {
        try std.testing.expectEqual(@as(f32, 0.0), v);
    }

    // Write values
    const data = grad.asSliceMut();
    for (data, 0..) |*v, i| {
        v.* = @floatFromInt(i);
    }

    // Verify values persist
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), grad.asSlice()[5], 1e-6);

    // Zero and verify
    grad.zero();
    for (grad.asSlice()) |v| {
        try std.testing.expectEqual(@as(f32, 0.0), v);
    }
}

test "GradTensor accumulate and scale compose correctly" {
    const allocator = std.testing.allocator;
    var grad = try GradTensor.init(allocator, &.{4});
    defer grad.deinit();

    const vals = [_]f32{ 1, 2, 3, 4 };
    grad.accumulate(&vals);
    grad.accumulate(&vals);
    // Now: [2, 4, 6, 8]

    grad.scale(0.5);
    // Now: [1, 2, 3, 4]

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), grad.asSlice()[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), grad.asSlice()[3], 1e-6);
}

test "GradTensor norm is L2 norm" {
    const allocator = std.testing.allocator;
    var grad = try GradTensor.init(allocator, &.{4});
    defer grad.deinit();

    // [3, 0, 4, 0] -> norm = 5
    const data = grad.asSliceMut();
    data[0] = 3.0;
    data[2] = 4.0;

    try std.testing.expectApproxEqAbs(@as(f32, 5.0), grad.norm(), 1e-6);
}
