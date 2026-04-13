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
const compute = @import("compute_pkg");

const simd_arch = compute.cpu.simd.arch;
const VEC = simd_arch.f32_vec_len;
const F32Vec = simd_arch.F32Vec;
const math_fast = @import("compute_pkg").cpu.math_fast;
const fastSinCos = math_fast.fastSinCos;
const fastSinCosScalar = math_fast.fastSinCosScalar;

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

/// Batch RoPE backward: apply inverse rotation to all positions at once.
///
/// Precomputes inv_freq table once (eliminates redundant pow() calls across positions).
/// grad_io layout: [batch * seq_len * n_heads * head_dim]
pub fn ropeBackwardBatch(
    grad_io: []f32,
    batch: usize,
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    theta: f32,
) void {
    std.debug.assert(rope_dim <= head_dim);
    std.debug.assert(rope_dim % 2 == 0);

    const half = rope_dim / 2;
    const stride = n_heads * head_dim;
    std.debug.assert(grad_io.len >= batch * seq_len * stride);

    // Precompute inv_freq table once (constant across all positions)
    var inv_freq_buf: [512]f32 = undefined;
    const inv_freq = inv_freq_buf[0..half];
    for (0..half) |pair| {
        const freq_exp = -2.0 * @as(f32, @floatFromInt(pair)) / @as(f32, @floatFromInt(rope_dim));
        inv_freq[pair] = std.math.pow(f32, theta, freq_exp);
    }

    for (0..batch) |bi| {
        for (0..seq_len) |pos| {
            const pos_f: f32 = @floatFromInt(pos);
            const pos_v: F32Vec = @splat(pos_f);
            const token_base = (bi * seq_len + pos) * stride;

            for (0..n_heads) |head| {
                const base = token_base + head * head_dim;

                // SIMD path: process VEC pairs at once
                var pair: usize = 0;
                while (pair + VEC <= half) : (pair += VEC) {
                    const inv_f: F32Vec = inv_freq[pair..][0..VEC].*;
                    const angles = pos_v * inv_f;
                    const sc = fastSinCos(angles);

                    const dy_lo: F32Vec = grad_io[base + pair ..][0..VEC].*;
                    const dy_hi: F32Vec = grad_io[base + half + pair ..][0..VEC].*;

                    // Inverse rotation (transpose)
                    grad_io[base + pair ..][0..VEC].* = dy_lo * sc.cos + dy_hi * sc.sin;
                    grad_io[base + half + pair ..][0..VEC].* = dy_hi * sc.cos - dy_lo * sc.sin;
                }
                // Scalar tail
                while (pair < half) : (pair += 1) {
                    const angle = pos_f * inv_freq[pair];
                    const sc = fastSinCosScalar(angle);

                    const dy_lo = grad_io[base + pair];
                    const dy_hi = grad_io[base + half + pair];

                    grad_io[base + pair] = dy_lo * sc.cos + dy_hi * sc.sin;
                    grad_io[base + half + pair] = dy_hi * sc.cos - dy_lo * sc.sin;
                }
            }
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

test "ropeBackwardBatch matches per-position ropeBackward" {
    // batch=2, seq=3, n_heads=2, head_dim=4, rope_dim=4
    const b: usize = 2;
    const s: usize = 3;
    const nh: usize = 2;
    const hd: usize = 4;
    const total = b * s * nh * hd;

    var data_batch: [total]f32 = undefined;
    var data_ref: [total]f32 = undefined;
    for (0..total) |i| {
        const v = @as(f32, @floatFromInt(i)) * 0.1 + 1.0;
        data_batch[i] = v;
        data_ref[i] = v;
    }

    // Reference: per-position calls
    for (0..b) |bi| {
        for (0..s) |pos| {
            const token_idx = bi * s + pos;
            ropeBackward(data_ref[token_idx * nh * hd ..][0 .. nh * hd], nh, hd, hd, pos, 10000.0);
        }
    }

    // Batch version
    ropeBackwardBatch(&data_batch, b, s, nh, hd, hd, 10000.0);

    for (0..total) |i| {
        try std.testing.expectApproxEqAbs(data_ref[i], data_batch[i], 1e-5);
    }
}
