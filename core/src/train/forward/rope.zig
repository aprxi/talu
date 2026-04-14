//! Rotary position embeddings (RoPE) forward pass for training.

const std = @import("std");
const compute = @import("compute_pkg");

const simd = compute.cpu.simd.arch;
const VEC = simd.f32_vec_len;
const F32Vec = simd.F32Vec;
const math_fast = @import("compute_pkg").cpu.math_fast;
const fastSinCos = math_fast.fastSinCos;
const fastSinCosScalar = math_fast.fastSinCosScalar;

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

/// Batch RoPE forward: apply rotary position embeddings to all positions at once.
///
/// Precomputes inv_freq table once (eliminates redundant pow() calls across positions).
/// io layout: [batch * seq_len * n_heads * head_dim]
pub fn ropeForwardBatch(
    io: []f32,
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
    std.debug.assert(io.len >= batch * seq_len * stride);

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

                    const x_lo: F32Vec = io[base + pair ..][0..VEC].*;
                    const x_hi: F32Vec = io[base + half + pair ..][0..VEC].*;

                    io[base + pair ..][0..VEC].* = x_lo * sc.cos - x_hi * sc.sin;
                    io[base + half + pair ..][0..VEC].* = x_lo * sc.sin + x_hi * sc.cos;
                }
                // Scalar tail
                while (pair < half) : (pair += 1) {
                    const angle = pos_f * inv_freq[pair];
                    const sc = fastSinCosScalar(angle);

                    const x_lo = io[base + pair];
                    const x_hi = io[base + half + pair];

                    io[base + pair] = x_lo * sc.cos - x_hi * sc.sin;
                    io[base + half + pair] = x_lo * sc.sin + x_hi * sc.cos;
                }
            }
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

test "ropeForwardBatch matches per-position ropeForward" {
    // batch=2, seq=3, n_heads=2, head_dim=4, rope_dim=4
    const b: usize = 2;
    const s: usize = 3;
    const nh: usize = 2;
    const hd: usize = 4;
    const total = b * s * nh * hd;

    // Fill with deterministic values
    var data_batch: [total]f32 = undefined;
    var data_ref: [total]f32 = undefined;
    for (0..total) |i| {
        const v = @as(f32, @floatFromInt(i)) * 0.1 + 1.0;
        data_batch[i] = v;
        data_ref[i] = v;
    }

    // Reference: per-position calls (matching pass.zig pattern)
    for (0..b) |bi| {
        for (0..s) |pos| {
            const token_idx = bi * s + pos;
            ropeForward(data_ref[token_idx * nh * hd ..][0 .. nh * hd], nh, hd, hd, pos, 10000.0);
        }
    }

    // Batch version
    ropeForwardBatch(&data_batch, b, s, nh, hd, hd, 10000.0);

    // Must match
    for (0..total) |i| {
        try testing.expectApproxEqAbs(data_ref[i], data_batch[i], 1e-5);
    }
}
