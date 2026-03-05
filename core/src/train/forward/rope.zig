//! Rotary position embeddings (RoPE) forward pass for training.

const std = @import("std");

/// RoPE forward: apply rotary position embeddings in-place.
/// Same algorithm as backward/rope.zig's test helper ropeForward.
pub fn ropeForward(io: []f32, n_heads: usize, head_dim: usize, rope_dim: usize, position: usize, theta: f32) void {
    std.debug.assert(io.len == n_heads * head_dim);
    std.debug.assert(rope_dim <= head_dim);
    std.debug.assert(rope_dim % 2 == 0);

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

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

test "ropeForward at position 0 is identity" {
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const original = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    ropeForward(&data, 1, 4, 4, 0, 10000.0);

    // At position 0, cos=1, sin=0, so no change
    for (data, original) |d, o| {
        try testing.expectApproxEqAbs(o, d, 1e-6);
    }
}
