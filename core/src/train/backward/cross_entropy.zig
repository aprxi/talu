//! Cross-entropy loss backward pass.
//!
//! The fused softmax + cross-entropy gradient is:
//!   grad_logits[i] = softmax(logits)[i] - (i == target ? 1.0 : 0.0)
//!
//! This is numerically stable and avoids computing softmax separately.
//! For a batch, processes each row independently.

const std = @import("std");

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

    const scale = 1.0 / @as(f32, @floatFromInt(batch_size));

    for (0..batch_size) |b| {
        const offset = b * vocab_size;
        const row = logits[offset..][0..vocab_size];
        const grad_row = grad_logits[offset..][0..vocab_size];
        const target = targets[b];

        // Compute softmax for this row (numerically stable: subtract max first)
        var max_val: f32 = row[0];
        for (row[1..]) |v| {
            if (v > max_val) max_val = v;
        }

        var sum_exp: f32 = 0.0;
        for (row, grad_row) |v, *g| {
            const exp_v = @exp(v - max_val);
            g.* = exp_v;
            sum_exp += exp_v;
        }

        // Normalize to get softmax, then subtract 1 at target index
        const inv_sum = 1.0 / sum_exp;
        for (grad_row) |*g| {
            g.* *= inv_sum * scale;
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
