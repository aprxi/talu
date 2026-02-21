//! Index-space score/probability vector primitives.

const std = @import("std");

/// Zero all probabilities then write back a normalized subset.
pub fn renormalizeSubset(probabilities: []f32, sorted_entries: anytype, subset_sum: f32) void {
    const inverse_sum = 1.0 / subset_sum;
    @memset(probabilities, 0);
    for (sorted_entries) |entry| probabilities[entry.index] = entry.value * inverse_sum;
}

/// Apply min-p thresholding and renormalize the vector in-place.
pub fn applyMinP(probabilities: []f32, min_p: f32) void {
    if (min_p <= 0.0) return;

    var max_probability: f32 = 0.0;
    for (probabilities) |probability| max_probability = @max(max_probability, probability);

    const min_probability_threshold = min_p * max_probability;
    var filtered_probability_sum: f32 = 0.0;
    for (probabilities) |*probability| {
        if (probability.* < min_probability_threshold) {
            probability.* = 0.0;
        } else {
            filtered_probability_sum += probability.*;
        }
    }

    if (filtered_probability_sum > 0) {
        const inverse_filtered_sum = 1.0 / filtered_probability_sum;
        for (probabilities) |*probability| probability.* *= inverse_filtered_sum;
    }
}

/// Apply multiplicative penalty for selected indices in a score vector.
pub fn applyIndexPenalty(logits: []f32, selected_indices: []const u32, penalty: f32) void {
    if (penalty == 1.0) return;

    for (selected_indices) |idx| {
        if (idx < logits.len) {
            const score = logits[idx];
            if (score > 0) {
                logits[idx] = score / penalty;
            } else {
                logits[idx] = score * penalty;
            }
        }
    }
}

fn biasEntryIndex(bias_entry: anytype) ?usize {
    const BiasEntry = @TypeOf(bias_entry);
    if (@hasField(BiasEntry, "index")) return @intCast(@field(bias_entry, "index"));
    if (@hasField(BiasEntry, "token_id")) return @intCast(@field(bias_entry, "token_id"));
    return null;
}

/// Apply additive bias entries to a score vector.
pub fn applyIndexBias(logits: []f32, bias_entries: anytype) void {
    for (bias_entries) |bias_entry| {
        const idx = biasEntryIndex(bias_entry) orelse continue;
        if (idx < logits.len) {
            logits[idx] += bias_entry.bias;
        }
    }
}

test "renormalizeSubset zeroes non-selected probabilities" {
    const Entry = struct { index: usize, value: f32 };
    var probs = [_]f32{ 0.1, 0.2, 0.3, 0.4 };
    const entries = [_]Entry{
        .{ .index = 3, .value = 0.4 },
        .{ .index = 1, .value = 0.2 },
    };
    renormalizeSubset(&probs, entries[0..], 0.6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), probs[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), probs[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 / 3.0), probs[3], 1e-6);
}

test "applyMinP filters and renormalizes probabilities" {
    var probs = [_]f32{ 0.6, 0.25, 0.1, 0.05 };
    applyMinP(&probs, 0.5); // threshold=0.3 -> only first survives
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), probs[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), probs[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), probs[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), probs[3], 1e-6);
}

test "applyIndexPenalty applies multiplicative sign-aware penalty" {
    var logits = [_]f32{ 2.0, -2.0, 1.0, 0.5 };
    const indices = [_]u32{ 0, 1, 9 };
    applyIndexPenalty(&logits, &indices, 2.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), logits[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -4.0), logits[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), logits[2], 1e-6);
}

test "applyIndexBias supports index and token_id entry shapes" {
    const BiasByIndex = struct { index: usize, bias: f32 };
    const BiasByToken = struct { token_id: usize, bias: f32 };
    var logits = [_]f32{ 0.0, 0.0, 0.0 };
    const by_index = [_]BiasByIndex{.{ .index = 1, .bias = 0.75 }};
    const by_token = [_]BiasByToken{.{ .token_id = 2, .bias = -0.25 }};
    applyIndexBias(&logits, by_index[0..]);
    applyIndexBias(&logits, by_token[0..]);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), logits[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.75), logits[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -0.25), logits[2], 1e-6);
}
