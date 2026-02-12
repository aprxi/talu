//! Integration tests for xray.TracedTensor
//!
//! TracedTensor is a lightweight reference to tensor data during tracing.
//! It contains just enough info to read the tensor: pointer, dtype, shape, ndim.

const std = @import("std");
const main = @import("main");
const xray = main.xray;

const TracedTensor = xray.TracedTensor;

test "TracedTensor: element count calculation" {
    const tensor = TracedTensor{
        .ptr = undefined,
        .dtype = .f32,
        .shape = .{ 2, 3, 4, 0 },
        .ndim = 3,
    };

    try std.testing.expectEqual(@as(usize, 24), tensor.elementCount());
}

test "TracedTensor: byte size calculation for f32" {
    const tensor = TracedTensor{
        .ptr = undefined,
        .dtype = .f32,
        .shape = .{ 10, 0, 0, 0 },
        .ndim = 1,
    };

    // f32 = 4 bytes, 10 elements = 40 bytes
    try std.testing.expectEqual(@as(usize, 40), tensor.byteSize());
}

test "TracedTensor: byte size calculation for f16" {
    const tensor = TracedTensor{
        .ptr = undefined,
        .dtype = .f16,
        .shape = .{ 10, 0, 0, 0 },
        .ndim = 1,
    };

    // f16 = 2 bytes, 10 elements = 20 bytes
    try std.testing.expectEqual(@as(usize, 20), tensor.byteSize());
}

test "TracedTensor: handles zero dimensions (scalar)" {
    const tensor = TracedTensor{
        .ptr = undefined,
        .dtype = .f32,
        .shape = .{ 0, 0, 0, 0 },
        .ndim = 0,
    };

    // Scalar has 1 element
    try std.testing.expectEqual(@as(usize, 1), tensor.elementCount());
}

test "TracedTensor: multi-dimensional shape" {
    const tensor = TracedTensor{
        .ptr = undefined,
        .dtype = .f32,
        .shape = .{ 2, 3, 4, 5 },
        .ndim = 4,
    };

    // 2 * 3 * 4 * 5 = 120
    try std.testing.expectEqual(@as(usize, 120), tensor.elementCount());
}
