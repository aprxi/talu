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
