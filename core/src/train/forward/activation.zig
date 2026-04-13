//! SwiGLU activation forward pass for training.

const std = @import("std");
const compute = @import("compute_pkg");

const simd = compute.cpu.simd.arch;
const VEC = simd.f32_vec_len;
const F32Vec = simd.F32Vec;
const math_fast = @import("compute_pkg").cpu.math_fast;
const fastExp = math_fast.fastExp;
const fastExpScalar = math_fast.fastExpScalar;

/// SwiGLU forward: output[i] = silu(gate[i]) * up[i]
/// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
pub fn swigluForward(output: []f32, gate: []const f32, up: []const f32, len: usize) void {
    @setFloatMode(.optimized);

    const one: F32Vec = @splat(1.0);
    var i: usize = 0;
    while (i + VEC <= len) : (i += VEC) {
        const x: F32Vec = gate[i..][0..VEC].*;
        const exp_neg = fastExp(-x);
        const sig = one / (one + exp_neg);
        const u: F32Vec = up[i..][0..VEC].*;
        output[i..][0..VEC].* = x * sig * u;
    }
    // Scalar tail
    while (i < len) : (i += 1) {
        const x = gate[i];
        const sigmoid = 1.0 / (1.0 + fastExpScalar(-x));
        output[i] = x * sigmoid * up[i];
    }
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

test "swigluForward computes silu(gate) * up" {
    var output: [2]f32 = undefined;
    const gate = [_]f32{ 0.0, 1.0 };
    const up = [_]f32{ 2.0, 2.0 };

    swigluForward(&output, &gate, &up, 2);

    // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    try testing.expectApproxEqAbs(@as(f32, 0.0), output[0], 1e-5);
    // silu(1) = 1 * sigmoid(1) = 1 / (1 + exp(-1)) ≈ 0.7311
    try testing.expectApproxEqAbs(@as(f32, 0.7311), output[1] / 2.0, 1e-3);
}

test "swigluForward SIMD path matches scalar" {
    // Test with enough elements to exercise both SIMD and scalar paths
    const len = VEC * 3 + 2;
    var gate: [VEC * 3 + 2]f32 = undefined;
    var up_buf: [VEC * 3 + 2]f32 = undefined;
    var output: [VEC * 3 + 2]f32 = undefined;

    for (0..len) |i| {
        gate[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - @as(i32, @intCast(len / 2)))) * 0.3;
        up_buf[i] = 1.0 + @as(f32, @floatFromInt(i)) * 0.1;
    }

    swigluForward(&output, &gate, &up_buf, len);

    // Verify against scalar reference (using std @exp for high-precision comparison)
    for (0..len) |i| {
        const x = gate[i];
        const sig = 1.0 / (1.0 + @exp(-x));
        const expected = x * sig * up_buf[i];
        try testing.expectApproxEqRel(expected, output[i], 1e-3);
    }
}
