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

