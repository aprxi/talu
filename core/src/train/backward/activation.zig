//! Activation backward passes: SiLU, GELU, and fused SwiGLU.
//!
//! SiLU:    y = x * sigmoid(x)
//! GELU:    y = x * 0.5 * (1 + erf(x / sqrt(2)))
//! SwiGLU:  y = silu(gate) * up

const std = @import("std");
const compute = @import("../../compute/root.zig");

const simd = compute.cpu.simd.arch;
const VEC = simd.f32_vec_len;
const F32Vec = simd.F32Vec;
const math_fast = @import("../../compute/cpu/math_fast.zig");
const fastExp = math_fast.fastExp;
const fastExpScalar = math_fast.fastExpScalar;

/// Abramowitz & Stegun polynomial approximation of erf(x).
/// Max error < 1.5e-7 for all x.
fn erff(x: f32) f32 {
    const a = @abs(x);
    const t = 1.0 / (1.0 + 0.3275911 * a);
    const t2 = t * t;
    const t3 = t2 * t;
    const t4 = t3 * t;
    const t5 = t4 * t;
    const poly = 0.254829592 * t - 0.284496736 * t2 + 1.421413741 * t3 - 1.453152027 * t4 + 1.061405429 * t5;
    const result = 1.0 - poly * @exp(-a * a);
    return if (x < 0) -result else result;
}

/// SiLU backward: d_input = d_output * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))
///
/// Simplified: d_input = d_output * sigmoid(x) * (1 + x * (1 - sigmoid(x)))
///
/// input:      [count] — saved input from forward pass
/// grad_input: [count] — overwritten with gradient
/// grad_output:[count]
pub fn siluBackward(
    grad_input: []f32,
    grad_output: []const f32,
    input: []const f32,
) void {
    @setFloatMode(.optimized);
    std.debug.assert(grad_input.len == grad_output.len);
    std.debug.assert(grad_input.len == input.len);

    const one: F32Vec = @splat(1.0);
    const count = grad_input.len;
    var i: usize = 0;
    while (i + VEC <= count) : (i += VEC) {
        const x: F32Vec = input[i..][0..VEC].*;
        const dy: F32Vec = grad_output[i..][0..VEC].*;
        const sig = one / (one + fastExp(-x));
        grad_input[i..][0..VEC].* = dy * sig * (one + x * (one - sig));
    }
    while (i < count) : (i += 1) {
        const x = input[i];
        const sig = 1.0 / (1.0 + fastExpScalar(-x));
        grad_input[i] = grad_output[i] * sig * (1.0 + x * (1.0 - sig));
    }
}

/// GELU backward (exact): d_input = d_output * (0.5 * (1 + erf(x/sqrt2)) + x * exp(-x^2/2) / sqrt(2*pi))
///
/// input:      [count]
/// grad_input: [count] — overwritten
/// grad_output:[count]
pub fn geluBackward(
    grad_input: []f32,
    grad_output: []const f32,
    input: []const f32,
) void {
    std.debug.assert(grad_input.len == grad_output.len);
    std.debug.assert(grad_input.len == input.len);

    const sqrt_2: f32 = @sqrt(2.0);
    const inv_sqrt_2pi: f32 = 1.0 / @sqrt(2.0 * std.math.pi);

    for (grad_input, grad_output, input) |*dx, dy, x| {
        const cdf = 0.5 * (1.0 + erff(x / sqrt_2));
        const pdf = inv_sqrt_2pi * @exp(-0.5 * x * x);
        dx.* = dy * (cdf + x * pdf);
    }
}

/// SwiGLU backward: fused silu(gate) * up
///
/// Given d_output, compute:
///   d_gate = d_output * up * silu'(gate)
///   d_up   = d_output * silu(gate)
///
/// gate:        [count] — saved gate input from forward
/// up:          [count] — saved up input from forward
/// grad_gate:   [count] — overwritten
/// grad_up:     [count] — overwritten
/// grad_output: [count]
pub fn swigluBackward(
    grad_gate: []f32,
    grad_up: []f32,
    grad_output: []const f32,
    gate: []const f32,
    up: []const f32,
) void {
    @setFloatMode(.optimized);
    std.debug.assert(grad_gate.len == grad_output.len);
    std.debug.assert(grad_up.len == grad_output.len);
    std.debug.assert(gate.len == grad_output.len);
    std.debug.assert(up.len == grad_output.len);

    const one: F32Vec = @splat(1.0);
    const count = grad_gate.len;
    var i: usize = 0;
    while (i + VEC <= count) : (i += VEC) {
        const g: F32Vec = gate[i..][0..VEC].*;
        const u: F32Vec = up[i..][0..VEC].*;
        const dy: F32Vec = grad_output[i..][0..VEC].*;
        const sig = one / (one + fastExp(-g));
        const silu_g = g * sig;
        const silu_grad = sig * (one + g * (one - sig));
        grad_gate[i..][0..VEC].* = dy * u * silu_grad;
        grad_up[i..][0..VEC].* = dy * silu_g;
    }
    while (i < count) : (i += 1) {
        const g = gate[i];
        const u = up[i];
        const dy = grad_output[i];
        const sig = 1.0 / (1.0 + fastExpScalar(-g));
        const silu_g = g * sig;
        const silu_grad = sig * (1.0 + g * (1.0 - sig));
        grad_gate[i] = dy * u * silu_grad;
        grad_up[i] = dy * silu_g;
    }
}

// =============================================================================
// Tests
// =============================================================================

test "siluBackward at x=0 gives 0.5" {
    // silu(0) = 0 * sigmoid(0) = 0
    // silu'(0) = sigmoid(0) * (1 + 0*(1-sigmoid(0))) = 0.5
    var grad_input = [_]f32{0};
    const grad_output = [_]f32{1.0};
    const input = [_]f32{0.0};

    siluBackward(&grad_input, &grad_output, &input);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), grad_input[0], 1e-5);
}

test "siluBackward scales with grad_output" {
    var grad_1 = [_]f32{0};
    var grad_2 = [_]f32{0};
    const input = [_]f32{1.0};

    siluBackward(&grad_1, &[_]f32{1.0}, &input);
    siluBackward(&grad_2, &[_]f32{2.0}, &input);

    try std.testing.expectApproxEqAbs(grad_1[0] * 2.0, grad_2[0], 1e-4);
}

test "geluBackward at x=0 gives 0.5" {
    // gelu(0) = 0, gelu'(0) = 0.5*(1+erf(0)) + 0 = 0.5
    var grad_input = [_]f32{0};
    const grad_output = [_]f32{1.0};
    const input = [_]f32{0.0};

    geluBackward(&grad_input, &grad_output, &input);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), grad_input[0], 1e-5);
}

test "swigluBackward decomposes correctly" {
    // gate=0 => silu(gate)=0, silu'(gate)=0.5
    // up=2
    // d_gate = dy * up * silu'(gate) = 1 * 2 * 0.5 = 1.0
    // d_up = dy * silu(gate) = 1 * 0 = 0
    var dg = [_]f32{0};
    var du = [_]f32{0};
    const dy = [_]f32{1.0};
    const gate = [_]f32{0.0};
    const up_val = [_]f32{2.0};

    swigluBackward(&dg, &du, &dy, &gate, &up_val);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), dg[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), du[0], 1e-5);
}

test "swigluBackward with non-zero gate" {
    // gate=2 => sigmoid(2)≈0.8808, silu(2)≈1.7616
    // up=1, dy=1
    // d_up = dy * silu(gate) ≈ 1.7616
    var dg = [_]f32{0};
    var du = [_]f32{0};
    const dy = [_]f32{1.0};
    const gate = [_]f32{2.0};
    const up_val = [_]f32{1.0};

    swigluBackward(&dg, &du, &dy, &gate, &up_val);

    const sig: f32 = 1.0 / (1.0 + @exp(@as(f32, -2.0)));
    const silu_2 = 2.0 * sig;
    try std.testing.expectApproxEqAbs(silu_2, du[0], 1e-3);
    try std.testing.expect(dg[0] > 0.0);
}
