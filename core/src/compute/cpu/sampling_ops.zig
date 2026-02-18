//! Sampling vector primitives for CPU inference paths.

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

/// Apply repetition penalty to logits for tokens that appear in context.
pub fn applyRepetitionPenalty(logits: []f32, context_tokens: []const u32, penalty: f32) void {
    if (penalty == 1.0) return;

    for (context_tokens) |token_id| {
        if (token_id < logits.len) {
            const token_logit = logits[token_id];
            if (token_logit > 0) {
                logits[token_id] = token_logit / penalty;
            } else {
                logits[token_id] = token_logit * penalty;
            }
        }
    }
}

/// Apply additive token biases to logits.
pub fn applyLogitBias(logits: []f32, bias_entries: anytype) void {
    for (bias_entries) |bias_entry| {
        if (bias_entry.token_id < logits.len) {
            logits[bias_entry.token_id] += bias_entry.bias;
        }
    }
}

