//! Activation primitives for CPU compute path.

const std = @import("std");
const simd = @import("../simd/root.zig");
const math_ops = @import("math_primitives/root.zig");

const VEC_LEN = simd.f32_vec_len;
const F32Vec = simd.F32Vec;

/// SiLU activation: `x * sigmoid(x)`.
pub inline fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

/// Softplus activation: `log(1 + exp(x))` (numerically stable branch).
pub inline fn softplus(x: f32) f32 {
    if (x > 20.0) return x;
    return @log(1.0 + @exp(x));
}

/// GELU approximation used by CPU FFN paths.
pub fn geluApprox(x: f32) f32 {
    const sqrt_2_over_pi: f32 = 0.7978845608;
    const x3 = x * x * x;
    const inner = sqrt_2_over_pi * (x + 0.044715 * x3);
    return x * 0.5 * (1.0 + std.math.tanh(inner));
}

/// SwiGLU variant with alpha=1.702, clipping, and (up+1) formulation.
pub fn swigluVariantScalar(gate_value: f32, up_value: f32) f32 {
    const alpha: f32 = 1.702;
    const limit: f32 = 7.0;
    const gate_clipped = if (gate_value > limit) limit else if (gate_value < -limit) -limit else gate_value;
    const up_clipped = std.math.clamp(up_value, -limit, limit);
    const sigmoid_value = 1.0 / (1.0 + @exp(-alpha * gate_clipped));
    return (gate_clipped * sigmoid_value) * (up_clipped + 1.0);
}

/// Apply SiLU pointwise from `src` into `dst`.
pub fn siluMap(src: []const f32, dst: []f32) void {
    std.debug.assert(src.len == dst.len);
    for (src, dst) |in, *out| {
        out.* = silu(in);
    }
}

/// Apply GELU pointwise from `src` into `dst`.
pub fn geluMap(src: []const f32, dst: []f32) void {
    std.debug.assert(src.len == dst.len);
    for (src, dst) |in, *out| {
        out.* = geluApprox(in);
    }
}

/// Elementwise multiply `a * b` into `out`.
pub fn elementwiseMul(a: []const f32, b: []const f32, out: []f32) void {
    std.debug.assert(a.len == b.len and a.len == out.len);
    var idx: usize = 0;
    while (idx + VEC_LEN - 1 < out.len) : (idx += VEC_LEN) {
        const a_vec: F32Vec = a[idx..][0..VEC_LEN].*;
        const b_vec: F32Vec = b[idx..][0..VEC_LEN].*;
        out[idx..][0..VEC_LEN].* = a_vec * b_vec;
    }
    while (idx < out.len) : (idx += 1) {
        out[idx] = a[idx] * b[idx];
    }
}

/// Compute `gelu(gate) * up` over split buffers.
pub fn geluMulSplit(gate_values: []const f32, up_values: []const f32, out: []f32) void {
    std.debug.assert(gate_values.len == up_values.len and gate_values.len == out.len);
    for (0..out.len) |idx| {
        out[idx] = geluApprox(gate_values[idx]) * up_values[idx];
    }
}

/// Compute `silu(gate) * up` over split buffers.
pub fn siluMulSplit(gate_values: []const f32, up_values: []const f32, out: []f32) void {
    std.debug.assert(gate_values.len == up_values.len and gate_values.len == out.len);
    const one: F32Vec = @splat(1.0);
    var idx: usize = 0;
    while (idx + VEC_LEN - 1 < out.len) : (idx += VEC_LEN) {
        const gate_vec: F32Vec = gate_values[idx..][0..VEC_LEN].*;
        const up_vec: F32Vec = up_values[idx..][0..VEC_LEN].*;
        const exp_neg = math_ops.fastExp(-gate_vec);
        const sig = one / (one + exp_neg);
        out[idx..][0..VEC_LEN].* = (gate_vec * sig) * up_vec;
    }
    while (idx < out.len) : (idx += 1) {
        const gate = gate_values[idx];
        const sigmoid = 1.0 / (1.0 + math_ops.fastExpScalar(-gate));
        out[idx] = (gate * sigmoid) * up_values[idx];
    }
}

/// Apply SwiGLU variant over interleaved `[glu0, lin0, glu1, lin1, ...]`.
pub fn swigluVariantInterleaved(gate_up_values: []const f32, out: []f32) void {
    std.debug.assert(gate_up_values.len == out.len * 2);
    for (0..out.len) |idx| {
        out[idx] = swigluVariantScalar(gate_up_values[idx * 2], gate_up_values[idx * 2 + 1]);
    }
}

/// Apply SwiGLU variant over split gate/up buffers.
pub fn swigluVariantSplit(gate_values: []const f32, up_values: []const f32, out: []f32) void {
    std.debug.assert(gate_values.len == up_values.len and gate_values.len == out.len);
    for (0..out.len) |idx| {
        out[idx] = swigluVariantScalar(gate_values[idx], up_values[idx]);
    }
}

/// Compute `gelu(gate) * up` over interleaved `[gate, up]` rows.
pub fn geluMulInterleaved(gate_up_values: []const f32, out: []f32) void {
    std.debug.assert(gate_up_values.len == out.len * 2);
    for (0..out.len) |idx| {
        out[idx] = geluApprox(gate_up_values[idx * 2]) * gate_up_values[idx * 2 + 1];
    }
}

/// Compute `silu(gate) * up` over interleaved `[gate, up]` rows.
pub fn siluMulInterleaved(gate_up_values: []const f32, out: []f32) void {
    std.debug.assert(gate_up_values.len == out.len * 2);
    for (0..out.len) |idx| {
        const gate = gate_up_values[idx * 2];
        const up = gate_up_values[idx * 2 + 1];
        const sigmoid = 1.0 / (1.0 + math_ops.fastExpScalar(-gate));
        out[idx] = (gate * sigmoid) * up;
    }
}

test "silu activation" {
    try std.testing.expectApproxEqAbs(@as(f32, 0), silu(0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), silu(10.0), 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0), silu(-10.0), 0.01);
}

test "softplus activation" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.693147), softplus(0), 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 25.0), softplus(25.0), 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0), softplus(-10.0), 0.001);
}

test "siluMulSplit computes pairwise gated output" {
    const gate = [_]f32{ 1.0, -1.0, 0.5 };
    const up = [_]f32{ 2.0, 2.0, 2.0 };
    var out = [_]f32{ 0.0, 0.0, 0.0 };
    siluMulSplit(&gate, &up, &out);
    try std.testing.expect(out[0] > out[2]);
    try std.testing.expect(out[1] < 0.0);
}

test "swigluVariantInterleaved matches scalar helper" {
    const in = [_]f32{ 1.0, 2.0, -1.0, 0.5 };
    var out = [_]f32{ 0.0, 0.0 };
    swigluVariantInterleaved(&in, &out);
    try std.testing.expectApproxEqAbs(swigluVariantScalar(1.0, 2.0), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(swigluVariantScalar(-1.0, 0.5), out[1], 1e-6);
}
