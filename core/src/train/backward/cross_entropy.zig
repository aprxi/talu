//! Cross-entropy loss backward pass.
//!
//! The fused softmax + cross-entropy gradient is:
//!   grad_logits[i] = softmax(logits)[i] - (i == target ? 1.0 : 0.0)
//!
//! This is numerically stable and avoids computing softmax separately.
//! For a batch, processes each row independently.

const std = @import("std");
const compute = @import("../../compute/root.zig");

const simd = compute.cpu.simd.arch;
const VEC = simd.f32_vec_len;
const F32Vec = simd.F32Vec;
const math_fast = @import("../../compute/cpu/math_fast.zig");
const fastExp = math_fast.fastExp;
const fastExpScalar = math_fast.fastExpScalar;

/// Compute cross-entropy gradient in-place.
///
/// After this call, grad_logits contains:
///   grad[b, i] = softmax(logits[b, :])[i] - (i == target[b] ? 1.0 : 0.0)
///
/// Normalized by batch_size (mean reduction).
///
/// grad_logits: [batch_size * vocab_size] — overwritten with gradient.
/// logits:      [batch_size * vocab_size] — forward pass logits.
/// targets:     [batch_size] — target token IDs.
pub fn crossEntropyBackward(
    grad_logits: []f32,
    logits: []const f32,
    targets: []const u32,
    batch_size: usize,
    vocab_size: usize,
) void {
    std.debug.assert(grad_logits.len == batch_size * vocab_size);
    std.debug.assert(logits.len == batch_size * vocab_size);
    std.debug.assert(targets.len == batch_size);

    @setFloatMode(.optimized);
    const scale = 1.0 / @as(f32, @floatFromInt(batch_size));

    for (0..batch_size) |b| {
        const offset = b * vocab_size;
        const row = logits[offset..][0..vocab_size];
        const grad_row = grad_logits[offset..][0..vocab_size];
        const target = targets[b];

        // SIMD max reduction
        var max_vec: F32Vec = @splat(-std.math.inf(f32));
        var i: usize = 0;
        while (i + VEC <= vocab_size) : (i += VEC) {
            const v: F32Vec = row[i..][0..VEC].*;
            max_vec = @max(max_vec, v);
        }
        var max_val = @reduce(.Max, max_vec);
        while (i < vocab_size) : (i += 1) {
            max_val = @max(max_val, row[i]);
        }

        // SIMD exp + store + accumulate
        const max_v: F32Vec = @splat(max_val);
        var sum_vec: F32Vec = @splat(0.0);
        i = 0;
        while (i + VEC <= vocab_size) : (i += VEC) {
            const v: F32Vec = row[i..][0..VEC].*;
            const exp_v = fastExp(v - max_v);
            grad_row[i..][0..VEC].* = exp_v;
            sum_vec += exp_v;
        }
        var sum_exp = @reduce(.Add, sum_vec);
        while (i < vocab_size) : (i += 1) {
            const exp_v = fastExpScalar(row[i] - max_val);
            grad_row[i] = exp_v;
            sum_exp += exp_v;
        }

        // SIMD normalize: softmax = exp / sum * scale
        const norm: F32Vec = @splat(1.0 / sum_exp * scale);
        i = 0;
        while (i + VEC <= vocab_size) : (i += VEC) {
            var g: F32Vec = grad_row[i..][0..VEC].*;
            g *= norm;
            grad_row[i..][0..VEC].* = g;
        }
        const norm_scalar = 1.0 / sum_exp * scale;
        while (i < vocab_size) : (i += 1) {
            grad_row[i] *= norm_scalar;
        }

        // Subtract 1/batch_size at the target index
        if (target < vocab_size) {
            grad_row[target] -= scale;
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

test "crossEntropyBackward single element batch" {
    // logits = [0, 0, 0, 0], target = 2
    // softmax = [0.25, 0.25, 0.25, 0.25]
    // grad = [0.25, 0.25, 0.25-1, 0.25] = [0.25, 0.25, -0.75, 0.25]
    var grad = [_]f32{ 0, 0, 0, 0 };
    const logits = [_]f32{ 0, 0, 0, 0 };
    const targets = [_]u32{2};

    crossEntropyBackward(&grad, &logits, &targets, 1, 4);

    try std.testing.expectApproxEqAbs(@as(f32, 0.25), grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), grad[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -0.75), grad[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), grad[3], 1e-5);
}

test "crossEntropyBackward gradient row sums to zero" {
    var grad = [_]f32{ 0, 0, 0 };
    const logits = [_]f32{ 1.0, 2.0, 3.0 };
    const targets = [_]u32{1};

    crossEntropyBackward(&grad, &logits, &targets, 1, 3);

    var sum: f32 = 0.0;
    for (grad) |g| sum += g;
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sum, 1e-5);
}

test "crossEntropyBackward batch mean reduction" {
    // batch=2, vocab=3
    // With mean reduction, gradients are scaled by 1/batch_size
    var grad = [_]f32{ 0, 0, 0, 0, 0, 0 };
    const logits = [_]f32{ 0, 0, 0, 0, 0, 0 };
    const targets = [_]u32{ 0, 0 };

    crossEntropyBackward(&grad, &logits, &targets, 2, 3);

    // Each row: softmax = [1/3, 1/3, 1/3], scaled by 1/2
    // grad[0] = 1/3 * 1/2 - 1/2 = 1/6 - 1/2 = -1/3
    // grad[1] = 1/3 * 1/2 = 1/6
    // grad[2] = 1/3 * 1/2 = 1/6
    const expected_target: f32 = 1.0 / 6.0 - 0.5;
    const expected_other: f32 = 1.0 / 6.0;

    try std.testing.expectApproxEqAbs(expected_target, grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(expected_other, grad[1], 1e-5);
    try std.testing.expectApproxEqAbs(expected_other, grad[2], 1e-5);
}

test "crossEntropyBackward numerical stability with large logits" {
    // Large logits should not cause overflow due to max subtraction
    var grad = [_]f32{ 0, 0, 0 };
    const logits = [_]f32{ 1000.0, 1001.0, 999.0 };
    const targets = [_]u32{1};

    crossEntropyBackward(&grad, &logits, &targets, 1, 3);

    // Should not be NaN or Inf
    for (grad) |g| {
        try std.testing.expect(!std.math.isNan(g));
        try std.testing.expect(!std.math.isInf(g));
    }

    // Row should sum to ~0
    var sum: f32 = 0.0;
    for (grad) |g| sum += g;
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sum, 1e-5);
}
