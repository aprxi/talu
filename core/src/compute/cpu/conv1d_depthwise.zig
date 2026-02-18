//! Depthwise 1D convolution primitives for CPU shortconv-style kernels.

const std = @import("std");
const simd = @import("../simd/root.zig");

const VEC = simd.f32_vec_len;

/// SIMD element-wise multiply: `out = a * b`.
pub fn simdMul(a: []const f32, b: []const f32, out: []f32, len: usize) void {
    @setFloatMode(.optimized);
    var i: usize = 0;

    while (i + VEC <= len) : (i += VEC) {
        const va: @Vector(VEC, f32) = a[i..][0..VEC].*;
        const vb: @Vector(VEC, f32) = b[i..][0..VEC].*;
        out[i..][0..VEC].* = va * vb;
    }
    while (i < len) : (i += 1) {
        out[i] = a[i] * b[i];
    }
}

/// Depthwise 1D convolution with channel-major weights `[conv_dim, d_conv]`.
pub fn runChannelMajor(
    B_gate: []const f32,
    x_proj: []const f32,
    state: []f32,
    weight: []const f32,
    out: []f32,
    bias: ?[]const f32,
    conv_dim: usize,
    d_conv: usize,
) void {
    @setFloatMode(.optimized);

    if (d_conv > 1) {
        const shift_src = state[conv_dim..];
        const shift_dst = state[0 .. (d_conv - 1) * conv_dim];
        @memcpy(shift_dst, shift_src[0..shift_dst.len]);
    }

    const newest_row = state[(d_conv - 1) * conv_dim ..][0..conv_dim];
    simdMul(B_gate, x_proj, newest_row, conv_dim);

    @memset(out, 0);
    for (0..d_conv) |k| {
        const state_row = state[k * conv_dim ..][0..conv_dim];

        var i: usize = 0;
        while (i + 2 * VEC <= conv_dim) : (i += 2 * VEC) {
            const s0: @Vector(VEC, f32) = state_row[i..][0..VEC].*;
            const s1: @Vector(VEC, f32) = state_row[i + VEC ..][0..VEC].*;

            var w0: @Vector(VEC, f32) = undefined;
            var w1: @Vector(VEC, f32) = undefined;
            inline for (0..VEC) |j| {
                w0[j] = weight[(i + j) * d_conv + k];
                w1[j] = weight[(i + VEC + j) * d_conv + k];
            }

            const o0: @Vector(VEC, f32) = out[i..][0..VEC].*;
            const o1: @Vector(VEC, f32) = out[i + VEC ..][0..VEC].*;
            out[i..][0..VEC].* = @mulAdd(@Vector(VEC, f32), s0, w0, o0);
            out[i + VEC ..][0..VEC].* = @mulAdd(@Vector(VEC, f32), s1, w1, o1);
        }

        while (i + VEC <= conv_dim) : (i += VEC) {
            const s: @Vector(VEC, f32) = state_row[i..][0..VEC].*;
            var w: @Vector(VEC, f32) = undefined;
            inline for (0..VEC) |j| {
                w[j] = weight[(i + j) * d_conv + k];
            }
            const o: @Vector(VEC, f32) = out[i..][0..VEC].*;
            out[i..][0..VEC].* = @mulAdd(@Vector(VEC, f32), s, w, o);
        }

        while (i < conv_dim) : (i += 1) {
            out[i] += state_row[i] * weight[i * d_conv + k];
        }
    }

    if (bias) |b| {
        var i: usize = 0;
        while (i + VEC <= conv_dim) : (i += VEC) {
            const o: @Vector(VEC, f32) = out[i..][0..VEC].*;
            const bv: @Vector(VEC, f32) = b[i..][0..VEC].*;
            out[i..][0..VEC].* = o + bv;
        }
        while (i < conv_dim) : (i += 1) {
            out[i] += b[i];
        }
    }
}

/// Depthwise 1D convolution with time-major transposed weights `[d_conv, conv_dim]`.
pub fn runTimeMajor(
    B_gate: []const f32,
    x_proj: []const f32,
    state: []f32,
    weight_t: []const f32,
    out: []f32,
    bias: ?[]const f32,
    conv_dim: usize,
    d_conv: usize,
) void {
    @setFloatMode(.optimized);

    if (d_conv > 1) {
        const shift_src = state[conv_dim..];
        const shift_dst = state[0 .. (d_conv - 1) * conv_dim];
        @memcpy(shift_dst, shift_src[0..shift_dst.len]);
    }

    const newest_row = state[(d_conv - 1) * conv_dim ..][0..conv_dim];
    simdMul(B_gate, x_proj, newest_row, conv_dim);

    @memset(out, 0);

    var k: usize = 0;
    while (k + 4 <= d_conv) : (k += 4) {
        const s0 = state[k * conv_dim ..][0..conv_dim];
        const s1 = state[(k + 1) * conv_dim ..][0..conv_dim];
        const s2 = state[(k + 2) * conv_dim ..][0..conv_dim];
        const s3 = state[(k + 3) * conv_dim ..][0..conv_dim];
        const w0 = weight_t[k * conv_dim ..][0..conv_dim];
        const w1 = weight_t[(k + 1) * conv_dim ..][0..conv_dim];
        const w2 = weight_t[(k + 2) * conv_dim ..][0..conv_dim];
        const w3 = weight_t[(k + 3) * conv_dim ..][0..conv_dim];

        var i: usize = 0;
        while (i + VEC <= conv_dim) : (i += VEC) {
            var acc: @Vector(VEC, f32) = out[i..][0..VEC].*;
            acc = @mulAdd(@Vector(VEC, f32), s0[i..][0..VEC].*, w0[i..][0..VEC].*, acc);
            acc = @mulAdd(@Vector(VEC, f32), s1[i..][0..VEC].*, w1[i..][0..VEC].*, acc);
            acc = @mulAdd(@Vector(VEC, f32), s2[i..][0..VEC].*, w2[i..][0..VEC].*, acc);
            acc = @mulAdd(@Vector(VEC, f32), s3[i..][0..VEC].*, w3[i..][0..VEC].*, acc);
            out[i..][0..VEC].* = acc;
        }
        while (i < conv_dim) : (i += 1) {
            out[i] += s0[i] * w0[i] + s1[i] * w1[i] + s2[i] * w2[i] + s3[i] * w3[i];
        }
    }

    while (k + 2 <= d_conv) : (k += 2) {
        const s0 = state[k * conv_dim ..][0..conv_dim];
        const s1 = state[(k + 1) * conv_dim ..][0..conv_dim];
        const w0 = weight_t[k * conv_dim ..][0..conv_dim];
        const w1 = weight_t[(k + 1) * conv_dim ..][0..conv_dim];

        var i: usize = 0;
        while (i + VEC <= conv_dim) : (i += VEC) {
            var acc: @Vector(VEC, f32) = out[i..][0..VEC].*;
            acc = @mulAdd(@Vector(VEC, f32), s0[i..][0..VEC].*, w0[i..][0..VEC].*, acc);
            acc = @mulAdd(@Vector(VEC, f32), s1[i..][0..VEC].*, w1[i..][0..VEC].*, acc);
            out[i..][0..VEC].* = acc;
        }
        while (i < conv_dim) : (i += 1) {
            out[i] += s0[i] * w0[i] + s1[i] * w1[i];
        }
    }

    while (k < d_conv) : (k += 1) {
        const state_row = state[k * conv_dim ..][0..conv_dim];
        const weight_row = weight_t[k * conv_dim ..][0..conv_dim];

        var i: usize = 0;
        while (i + VEC <= conv_dim) : (i += VEC) {
            const s: @Vector(VEC, f32) = state_row[i..][0..VEC].*;
            const w: @Vector(VEC, f32) = weight_row[i..][0..VEC].*;
            const o: @Vector(VEC, f32) = out[i..][0..VEC].*;
            out[i..][0..VEC].* = @mulAdd(@Vector(VEC, f32), s, w, o);
        }
        while (i < conv_dim) : (i += 1) {
            out[i] += state_row[i] * weight_row[i];
        }
    }

    if (bias) |b| {
        var i: usize = 0;
        while (i + VEC <= conv_dim) : (i += VEC) {
            const o: @Vector(VEC, f32) = out[i..][0..VEC].*;
            const bv: @Vector(VEC, f32) = b[i..][0..VEC].*;
            out[i..][0..VEC].* = o + bv;
        }
        while (i < conv_dim) : (i += 1) {
            out[i] += b[i];
        }
    }
}

/// Transpose channel-major depthwise weights `[conv_dim, d_conv]` into
/// time-major layout `[d_conv, conv_dim]`.
pub fn transposeChannelMajorToTimeMajor(
    src: []const f32,
    dst: []f32,
    conv_dim: usize,
    d_conv: usize,
) !void {
    if (src.len < conv_dim * d_conv or dst.len < conv_dim * d_conv) return error.InvalidShape;
    for (0..conv_dim) |ch| {
        for (0..d_conv) |k| {
            dst[k * conv_dim + ch] = src[ch * d_conv + k];
        }
    }
}

/// Update depthwise state and apply one per-channel convolution step in place.
///
/// `values` is both input (newest sample per channel) and output (convolved value).
/// Weight layout is channel-major `[channels, kernel_size]`.
pub fn stepDepthwiseState(
    values: []f32,
    state: []f32,
    weight: []const f32,
    bias: ?[]const f32,
    channels: usize,
    kernel_size: usize,
) !void {
    if (kernel_size == 0) return error.InvalidShape;
    if (values.len < channels) return error.InvalidShape;
    if (state.len < channels * kernel_size) return error.InvalidShape;
    if (weight.len < channels * kernel_size) return error.InvalidShape;
    if (bias) |b| {
        if (b.len < channels) return error.InvalidShape;
    }

    for (0..channels) |ch| {
        const state_offset = ch * kernel_size;
        if (kernel_size > 1) {
            for (0..kernel_size - 1) |i| {
                state[state_offset + i] = state[state_offset + i + 1];
            }
        }
        state[state_offset + kernel_size - 1] = values[ch];

        var sum: f32 = 0;
        for (0..kernel_size) |k| {
            sum += state[state_offset + k] * weight[ch * kernel_size + k];
        }
        if (bias) |b| sum += b[ch];
        values[ch] = sum;
    }
}

test "simdMul computes elementwise product" {
    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 5, 6, 7, 8 };
    var out = [_]f32{ 0, 0, 0, 0 };
    simdMul(&a, &b, &out, out.len);
    try std.testing.expectApproxEqAbs(@as(f32, 5), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 32), out[3], 1e-6);
}

test "transposeChannelMajorToTimeMajor transposes expected layout" {
    const src = [_]f32{
        1, 2, 3, // ch0
        4, 5, 6, // ch1
    };
    var dst = [_]f32{0} ** 6;

    try transposeChannelMajorToTimeMajor(&src, &dst, 2, 3);

    try std.testing.expectEqualSlices(f32, &[_]f32{
        1, 4,
        2, 5,
        3, 6,
    }, &dst);
}

test "stepDepthwiseState updates state and writes output in place" {
    var values = [_]f32{ 3.0, 4.0 };
    var state = [_]f32{
        1.0, 2.0, // ch0
        0.5, 1.5, // ch1
    };
    const weight = [_]f32{
        1.0, 1.0, // ch0
        2.0, 0.0, // ch1
    };
    const bias = [_]f32{ 0.5, -1.0 };

    try stepDepthwiseState(&values, &state, &weight, &bias, 2, 2);

    // ch0: state => [2,3], dot => 5, +0.5 => 5.5
    // ch1: state => [1.5,4], dot => 3, -1 => 2
    try std.testing.expectApproxEqAbs(@as(f32, 5.5), values[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), values[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), state[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), state[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), state[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), state[3], 1e-6);
}
