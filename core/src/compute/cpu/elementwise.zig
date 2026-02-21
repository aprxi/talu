//! Element-wise scalar primitives for CPU compute path.

const std = @import("std");

/// `out = input + scalar`
pub fn addScalar(input: []const f32, out: []f32, scalar: f32) void {
    std.debug.assert(input.len == out.len);
    for (0..input.len) |idx| {
        out[idx] = input[idx] + scalar;
    }
}

/// `out = input * scalar`
pub fn mulScalar(input: []const f32, out: []f32, scalar: f32) void {
    std.debug.assert(input.len == out.len);
    for (0..input.len) |idx| {
        out[idx] = input[idx] * scalar;
    }
}

/// `out = pow(input, exponent)`
pub fn powScalar(input: []const f32, out: []f32, exponent: f32) void {
    std.debug.assert(input.len == out.len);
    for (0..input.len) |idx| {
        out[idx] = std.math.pow(f32, input[idx], exponent);
    }
}

/// `out = 1 / sqrt(input)`
pub fn rsqrt(input: []const f32, out: []f32) void {
    std.debug.assert(input.len == out.len);
    for (0..input.len) |idx| {
        out[idx] = 1.0 / std.math.sqrt(input[idx]);
    }
}

test "addScalar adds scalar to each element" {
    const input = [_]f32{ 1.0, -2.0, 3.5 };
    var out = [_]f32{ 0.0, 0.0, 0.0 };
    addScalar(&input, &out, 2.0);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 3.0, 0.0, 5.5 }, &out);
}

test "mulScalar multiplies each element by scalar" {
    const input = [_]f32{ 1.0, -2.0, 3.0 };
    var out = [_]f32{ 0.0, 0.0, 0.0 };
    mulScalar(&input, &out, -2.0);
    try std.testing.expectEqualSlices(f32, &[_]f32{ -2.0, 4.0, -6.0 }, &out);
}

test "powScalar raises each element to exponent" {
    const input = [_]f32{ 1.0, 4.0, 9.0 };
    var out = [_]f32{ 0.0, 0.0, 0.0 };
    powScalar(&input, &out, 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), out[2], 1e-6);
}

test "rsqrt computes reciprocal square root" {
    const input = [_]f32{ 1.0, 4.0, 16.0 };
    var out = [_]f32{ 0.0, 0.0, 0.0 };
    rsqrt(&input, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), out[2], 1e-6);
}
