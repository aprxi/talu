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

/// Apply additive presence and frequency penalties in a single pass.
/// For each unique token in context_tokens:
///   logit[token] -= presence_penalty + frequency_penalty * count(token)
/// Tokens with IDs >= logits.len are ignored.
pub fn applyAdditivePenalties(
    logits: []f32,
    context_tokens: []const u32,
    presence_penalty: f32,
    frequency_penalty: f32,
) void {
    if (context_tokens.len == 0) return;
    if (presence_penalty == 0.0 and frequency_penalty == 0.0) return;

    // Count occurrences: use logits as scratch isn't possible (they contain values),
    // so iterate context_tokens and apply penalties incrementally.
    // First pass: apply frequency_penalty for every occurrence.
    // Track which tokens we've seen for the presence_penalty (applied once per unique).
    // Since we can't allocate, use a second pass approach:
    //   - If only frequency_penalty: subtract penalty per occurrence
    //   - If only presence_penalty: subtract penalty once per unique token
    //   - If both: combine
    if (frequency_penalty != 0.0 and presence_penalty == 0.0) {
        // Simple case: subtract frequency_penalty for each occurrence
        for (context_tokens) |idx| {
            if (idx < logits.len) {
                logits[idx] -= frequency_penalty;
            }
        }
    } else if (frequency_penalty == 0.0) {
        // Presence-only: subtract once per unique token.
        // Iterate and skip duplicates by checking if we already applied.
        // Use a simple approach: for each token, check if it appeared earlier.
        for (context_tokens, 0..) |idx, i| {
            if (idx >= logits.len) continue;
            // Check if this token appeared earlier in context_tokens
            var seen_before = false;
            for (context_tokens[0..i]) |prev| {
                if (prev == idx) {
                    seen_before = true;
                    break;
                }
            }
            if (!seen_before) {
                logits[idx] -= presence_penalty;
            }
        }
    } else {
        // Both penalties: count occurrences, then apply once.
        // Two-pass: first count, then apply.
        // Pass 1: count occurrences using frequency_penalty per token as accumulator.
        // We iterate and for each token, apply frequency_penalty immediately.
        // Then do a second unique-pass for presence_penalty.
        for (context_tokens) |idx| {
            if (idx < logits.len) {
                logits[idx] -= frequency_penalty;
            }
        }
        // Pass 2: apply presence_penalty once per unique token.
        for (context_tokens, 0..) |idx, i| {
            if (idx >= logits.len) continue;
            var seen_before = false;
            for (context_tokens[0..i]) |prev| {
                if (prev == idx) {
                    seen_before = true;
                    break;
                }
            }
            if (!seen_before) {
                logits[idx] -= presence_penalty;
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

test "applyAdditivePenalties presence only subtracts once per unique token" {
    var logits = [_]f32{ 5.0, 5.0, 5.0, 5.0 };
    const context = [_]u32{ 0, 1, 0, 1, 1 }; // token 0 appears 2x, token 1 appears 3x
    applyAdditivePenalties(&logits, &context, 1.0, 0.0);
    // Presence: subtract 1.0 once per unique token
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), logits[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), logits[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), logits[2], 1e-6); // not in context
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), logits[3], 1e-6); // not in context
}

test "applyAdditivePenalties frequency only subtracts proportional to count" {
    var logits = [_]f32{ 10.0, 10.0, 10.0, 10.0 };
    const context = [_]u32{ 0, 1, 0, 1, 1 }; // token 0: 2x, token 1: 3x
    applyAdditivePenalties(&logits, &context, 0.0, 0.5);
    // Frequency: subtract 0.5 * count per token
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), logits[0], 1e-6); // 10 - 0.5*2
    try std.testing.expectApproxEqAbs(@as(f32, 8.5), logits[1], 1e-6); // 10 - 0.5*3
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), logits[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), logits[3], 1e-6);
}

test "applyAdditivePenalties both penalties combined" {
    var logits = [_]f32{ 10.0, 10.0, 10.0 };
    const context = [_]u32{ 0, 1, 0 }; // token 0: 2x, token 1: 1x
    applyAdditivePenalties(&logits, &context, 1.0, 0.5);
    // Token 0: 10 - 1.0 (presence) - 0.5*2 (frequency) = 8.0
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), logits[0], 1e-6);
    // Token 1: 10 - 1.0 (presence) - 0.5*1 (frequency) = 8.5
    try std.testing.expectApproxEqAbs(@as(f32, 8.5), logits[1], 1e-6);
    // Token 2: untouched
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), logits[2], 1e-6);
}

test "applyAdditivePenalties ignores out-of-bounds tokens" {
    var logits = [_]f32{ 5.0, 5.0 };
    const context = [_]u32{ 0, 100, 999 };
    applyAdditivePenalties(&logits, &context, 1.0, 0.5);
    // Only token 0 is in bounds: 5.0 - 1.0 - 0.5 = 3.5
    try std.testing.expectApproxEqAbs(@as(f32, 3.5), logits[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), logits[1], 1e-6);
}

test "applyAdditivePenalties empty context is noop" {
    var logits = [_]f32{ 5.0, 5.0, 5.0 };
    const context = [_]u32{};
    applyAdditivePenalties(&logits, &context, 2.0, 1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), logits[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), logits[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), logits[2], 1e-6);
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
