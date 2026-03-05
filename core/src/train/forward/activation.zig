//! SwiGLU activation forward pass for training.

const std = @import("std");

/// SwiGLU forward: output[i] = silu(gate[i]) * up[i]
/// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
pub fn swigluForward(output: []f32, gate: []const f32, up: []const f32, len: usize) void {
    for (0..len) |i| {
        const x = gate[i];
        const sigmoid = 1.0 / (1.0 + @exp(-x));
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
