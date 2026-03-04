//! Loss computation for training.
//!
//! Cross-entropy loss with numerically stable log-softmax.

const std = @import("std");

/// Compute mean cross-entropy loss over a batch.
///
/// loss = -1/N * sum_b( log(softmax(logits[b])[target[b]]) )
///
/// Uses log-sum-exp for numerical stability.
///
/// logits:  [batch_size * vocab_size]
/// targets: [batch_size]
pub fn crossEntropyLoss(
    logits: []const f32,
    targets: []const u32,
    batch_size: usize,
    vocab_size: usize,
) f32 {
    std.debug.assert(logits.len == batch_size * vocab_size);
    std.debug.assert(targets.len == batch_size);

    var total_loss: f32 = 0.0;

    for (0..batch_size) |b| {
        const row = logits[b * vocab_size ..][0..vocab_size];
        const target = targets[b];

        // log_softmax = logit[target] - log(sum(exp(logits - max)))
        var max_val: f32 = row[0];
        for (row[1..]) |v| {
            if (v > max_val) max_val = v;
        }

        var sum_exp: f32 = 0.0;
        for (row) |v| {
            sum_exp += @exp(v - max_val);
        }

        const log_sum_exp = max_val + @log(sum_exp);
        const log_prob = if (target < vocab_size)
            row[target] - log_sum_exp
        else
            -log_sum_exp; // out-of-range target

        total_loss -= log_prob;
    }

    return total_loss / @as(f32, @floatFromInt(batch_size));
}

// =============================================================================
// Tests
// =============================================================================

test "crossEntropyLoss uniform logits" {
    // All logits equal => softmax = 1/vocab => loss = log(vocab)
    const logits = [_]f32{ 0, 0, 0, 0 };
    const targets = [_]u32{0};

    const loss = crossEntropyLoss(&logits, &targets, 1, 4);
    try std.testing.expectApproxEqAbs(@log(@as(f32, 4.0)), loss, 1e-5);
}

test "crossEntropyLoss confident prediction has low loss" {
    // One logit much larger than others
    const logits = [_]f32{ 100.0, 0.0, 0.0 };
    const targets = [_]u32{0};

    const loss = crossEntropyLoss(&logits, &targets, 1, 3);
    try std.testing.expect(loss < 0.01);
}

test "crossEntropyLoss wrong prediction has high loss" {
    // Target is not the max logit
    const logits = [_]f32{ 100.0, 0.0, 0.0 };
    const targets = [_]u32{1};

    const loss = crossEntropyLoss(&logits, &targets, 1, 3);
    try std.testing.expect(loss > 90.0);
}

test "crossEntropyLoss batch averaging" {
    // Two identical samples => same loss as one sample
    const logits_1 = [_]f32{ 1.0, 2.0, 3.0 };
    const targets_1 = [_]u32{2};
    const loss_1 = crossEntropyLoss(&logits_1, &targets_1, 1, 3);

    const logits_2 = [_]f32{ 1.0, 2.0, 3.0, 1.0, 2.0, 3.0 };
    const targets_2 = [_]u32{ 2, 2 };
    const loss_2 = crossEntropyLoss(&logits_2, &targets_2, 2, 3);

    try std.testing.expectApproxEqAbs(loss_1, loss_2, 1e-5);
}

test "crossEntropyLoss numerical stability with large logits" {
    const logits = [_]f32{ 1000.0, 1001.0, 999.0 };
    const targets = [_]u32{1};

    const loss = crossEntropyLoss(&logits, &targets, 1, 3);
    try std.testing.expect(!std.math.isNan(loss));
    try std.testing.expect(!std.math.isInf(loss));
    try std.testing.expect(loss >= 0.0);
}
