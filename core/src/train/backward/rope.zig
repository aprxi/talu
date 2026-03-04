//! RoPE (Rotary Position Embedding) backward pass.
//!
//! Forward: applies complex rotation per dimension pair.
//!   x_lo' = x_lo * cos - x_hi * sin
//!   x_hi' = x_lo * sin + x_hi * cos
//!
//! Backward: apply inverse rotation (transpose of orthogonal rotation matrix).
//!   dx_lo = dy_lo * cos + dy_hi * sin
//!   dx_hi = -dy_lo * sin + dy_hi * cos

const std = @import("std");

/// Apply inverse RoPE rotation to gradients.
///
/// This is the backward pass for RoPE. Since the rotation is orthogonal,
/// the backward pass is the transpose (inverse) rotation.
///
/// grad_io:   [n_heads * head_dim] — in-place: gradient from upstream,
///            overwritten with gradient w.r.t. pre-RoPE values.
/// n_heads:   number of attention heads
/// head_dim:  dimension per head
/// rope_dim:  number of dimensions rotated (≤ head_dim, must be even)
/// position:  token position (for computing angles)
/// theta:     RoPE base frequency (typically 10000.0)
pub fn ropeBackward(
    grad_io: []f32,
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    position: usize,
    theta: f32,
) void {
    std.debug.assert(grad_io.len == n_heads * head_dim);
    std.debug.assert(rope_dim <= head_dim);
    std.debug.assert(rope_dim % 2 == 0);

    const half = rope_dim / 2;
    const pos_f: f32 = @floatFromInt(position);

    for (0..n_heads) |head| {
        const base = head * head_dim;
        for (0..half) |pair| {
            // Compute angle: position * theta^(-2*pair/rope_dim)
            const freq_exp = -2.0 * @as(f32, @floatFromInt(pair)) / @as(f32, @floatFromInt(rope_dim));
            const inv_freq = std.math.pow(f32, theta, freq_exp);
            const angle = pos_f * inv_freq;

            const cos_a = @cos(angle);
            const sin_a = @sin(angle);

            const lo_idx = base + pair;
            const hi_idx = base + half + pair;

            const dy_lo = grad_io[lo_idx];
            const dy_hi = grad_io[hi_idx];

            // Inverse rotation (transpose of rotation matrix)
            grad_io[lo_idx] = dy_lo * cos_a + dy_hi * sin_a;
            grad_io[hi_idx] = -dy_lo * sin_a + dy_hi * cos_a;
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

/// Apply forward RoPE (for testing round-trip)
fn ropeForward(io: []f32, n_heads: usize, head_dim: usize, rope_dim: usize, position: usize, theta: f32) void {
    const half = rope_dim / 2;
    const pos_f: f32 = @floatFromInt(position);

    for (0..n_heads) |head| {
        const base = head * head_dim;
        for (0..half) |pair| {
            const freq_exp = -2.0 * @as(f32, @floatFromInt(pair)) / @as(f32, @floatFromInt(rope_dim));
            const inv_freq = std.math.pow(f32, theta, freq_exp);
            const angle = pos_f * inv_freq;

            const cos_a = @cos(angle);
            const sin_a = @sin(angle);

            const lo_idx = base + pair;
            const hi_idx = base + half + pair;

            const x_lo = io[lo_idx];
            const x_hi = io[hi_idx];

            io[lo_idx] = x_lo * cos_a - x_hi * sin_a;
            io[hi_idx] = x_lo * sin_a + x_hi * cos_a;
        }
    }
}

test "ropeBackward inverts ropeForward" {
    // Apply forward then backward — should recover original values
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const original = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    ropeForward(&data, 1, 4, 4, 5, 10000.0);

    // After forward, values should have changed
    var changed = false;
    for (data, original) |d, o| {
        if (@abs(d - o) > 1e-6) {
            changed = true;
            break;
        }
    }
    try std.testing.expect(changed);

    // Apply backward (inverse rotation)
    ropeBackward(&data, 1, 4, 4, 5, 10000.0);

    // Should recover original
    for (data, original) |d, o| {
        try std.testing.expectApproxEqAbs(o, d, 1e-4);
    }
}

test "ropeBackward at position 0 is identity" {
    // At position 0, angle = 0, so cos=1, sin=0 => no rotation
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const original = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    ropeBackward(&data, 1, 4, 4, 0, 10000.0);

    for (data, original) |d, o| {
        try std.testing.expectApproxEqAbs(o, d, 1e-6);
    }
}

test "ropeBackward partial rope_dim" {
    // rope_dim < head_dim: only first rope_dim dimensions are rotated
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };

    ropeBackward(&data, 1, 8, 4, 5, 10000.0);

    // Dimensions 4..7 (outside rope_dim) should be unchanged
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), data[4], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), data[5], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), data[6], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), data[7], 1e-6);
}
