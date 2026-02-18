//! Token sampling strategies for text generation.
//!
//! Implements greedy, top-k, and top-p (nucleus) sampling with
//! temperature scaling, repetition penalty, and logit biasing.

const std = @import("std");
const compute = @import("../../../compute/root.zig");
const ops = compute.ops.math;
const simd = compute.simd;
const validate = @import("../../../validate/root.zig");

// Use comptime-detected SIMD width for all vector operations
const VEC_LEN = simd.f32_vec_len;
const F32Vec = simd.F32Vec;

pub const SamplingStrategy = enum {
    greedy,
    top_k,
    top_p,
};

/// Entry for logit bias: token_id -> bias value to add to logit.
pub const LogitBiasEntry = struct {
    token_id: u32,
    bias: f32,
};

pub const SamplingConfig = struct {
    strategy: SamplingStrategy = .greedy,
    temperature: f32 = 1.0,
    top_k: usize = 1,
    top_p: f32 = 0.9,
    /// Minimum probability threshold (min_p sampling).
    /// Tokens with probability < min_p * max_prob are excluded.
    /// 0.0 = disabled (default).
    min_p: f32 = 0.0,
    /// Repetition penalty applied to tokens in the context.
    /// 1.0 = no penalty (default), >1.0 = discourage repetition.
    repetition_penalty: f32 = 1.0,
    /// Token IDs to apply repetition penalty to (typically recent context).
    /// If null, no penalty is applied.
    context_tokens: ?[]const u32 = null,
    /// Random seed for reproducibility.
    /// 0 = use default/time-based seed (non-deterministic).
    /// Non-zero = deterministic sampling.
    seed: u64 = 0,
    /// Logit bias entries: add bias values to specific token logits before sampling.
    /// Positive values increase probability, negative decrease it.
    /// Use -100 or lower to effectively ban a token.
    logit_bias: ?[]const LogitBiasEntry = null,
};

// Use u32 for index to match C's 8-byte struct size (int + float)
// This is critical for cache efficiency during sorting:
// - u32 + f32 = 8 bytes (fits more in L2 cache)
// - usize + f32 = 16 bytes with padding (2x memory traffic)
const IndexValue = extern struct {
    index: u32,
    value: f32,

    /// Create an IndexValue with zeroed padding (required for extern structs).
    fn init(index: u32, value: f32) IndexValue {
        var iv = std.mem.zeroes(IndexValue);
        iv.index = index;
        iv.value = value;
        return iv;
    }
};

/// Workspace for sampler - avoids allocation per sample while being thread-safe.
/// Create once per thread/inference session and reuse.
pub const Workspace = struct {
    allocator: std.mem.Allocator,
    probabilities: []f32,
    sorted_entries: []IndexValue,

    /// Initialize workspace from a backing allocator. Call deinit() when done.
    pub fn init(allocator: std.mem.Allocator, vocab_size: usize) !Workspace {
        const probabilities = try allocator.alloc(f32, vocab_size);
        errdefer allocator.free(probabilities);
        const sorted_entries = try allocator.alloc(IndexValue, vocab_size);
        return .{ .allocator = allocator, .probabilities = probabilities, .sorted_entries = sorted_entries };
    }

    pub fn deinit(self: *Workspace) void {
        self.allocator.free(self.probabilities);
        self.allocator.free(self.sorted_entries);
        self.* = undefined;
    }
};

pub const Sampler = struct {
    allocator: std.mem.Allocator,
    prng: std.Random.DefaultPrng,
    workspace: Workspace,
    grammar_sampler: ?*validate.sampler.ConstrainedSampler = null,

    /// Initialize sampler with seed and workspace for a given vocab size.
    pub fn init(allocator: std.mem.Allocator, seed: u64, vocab_size: usize) !Sampler {
        return .{
            .allocator = allocator,
            .prng = std.Random.DefaultPrng.init(seed),
            .workspace = try Workspace.init(allocator, vocab_size),
        };
    }

    pub fn deinit(self: *Sampler) void {
        self.workspace.deinit();
        self.* = undefined;
    }

    /// Reseed the random number generator for deterministic output.
    pub fn reseed(self: *Sampler, seed: u64) void {
        self.prng = std.Random.DefaultPrng.init(seed);
    }

    /// Sample from logits (read-only version, no repetition penalty).
    /// For repetition penalty support, use sampleMut with mutable logits.
    pub fn sample(self: *Sampler, logits: []const f32, config: SamplingConfig) !usize {
        return self.sampleImpl(logits, config);
    }

    /// Sample from mutable logits with repetition penalty and logit bias support.
    /// Modifies logits in-place to apply penalties and biases before sampling.
    pub fn sampleMut(self: *Sampler, logits: []f32, config: SamplingConfig) !usize {
        // Apply repetition penalty if configured
        if (config.repetition_penalty != 1.0) {
            if (config.context_tokens) |tokens| {
                applyRepetitionPenalty(logits, tokens, config.repetition_penalty);
            }
        }

        // Apply logit bias if configured
        if (config.logit_bias) |bias_entries| {
            applyLogitBias(logits, bias_entries);
        }

        return self.sampleImpl(logits, config);
    }

    pub fn sampleConstrained(
        self: *Sampler,
        logits: []f32,
        config: SamplingConfig,
        tokenizer: anytype,
    ) !usize {
        if (self.grammar_sampler) |gs| {
            try gs.applyConstraints(logits, tokenizer);
        }
        return self.sampleMut(logits, config);
    }

    pub fn acceptToken(self: *Sampler, token_id: u32, token_text: []const u8) !void {
        if (self.grammar_sampler) |gs| {
            try gs.acceptToken(token_id, token_text);
        }
    }

    fn sampleImpl(self: *Sampler, logits: []const f32, config: SamplingConfig) !usize {
        if (logits.len == 0) return error.InvalidInput;
        if (logits.len > self.workspace.probabilities.len) return error.InvalidInput;

        // Validate sampling parameters
        if (config.temperature < 0) return error.InvalidTemperature;
        if (config.top_p < 0 or config.top_p > 1.0) return error.InvalidTopP;
        if (config.min_p < 0 or config.min_p > 1.0) return error.InvalidMinP;

        if (config.strategy == .greedy) {
            var best_token_index: usize = 0;
            var best_logit_value: f32 = logits[0];
            for (logits[1..], 1..) |logit, logit_index| {
                if (logit > best_logit_value) {
                    best_logit_value = logit;
                    best_token_index = logit_index;
                }
            }
            return best_token_index;
        }

        // For non-greedy strategies, temperature must be positive
        if (config.temperature <= 0) return error.InvalidTemperature;

        // Use workspace buffers (no allocation per sample!)
        const probabilities = self.workspace.probabilities[0..logits.len];

        // Find max using SIMD
        var max_logit_vec: F32Vec = @splat(logits[0]);
        var vec_index: usize = 0;
        while (vec_index + VEC_LEN - 1 < logits.len) : (vec_index += VEC_LEN) {
            const logit_vec: F32Vec = logits[vec_index..][0..VEC_LEN].*;
            max_logit_vec = @max(max_logit_vec, logit_vec);
        }
        var max_logit_value = @reduce(.Max, max_logit_vec);
        for (logits[vec_index..]) |logit| max_logit_value = @max(max_logit_value, logit);

        // Compute exp and sum - SIMD version
        const inverse_temperature = 1.0 / config.temperature;
        const max_logit_splat: F32Vec = @splat(max_logit_value);
        const inverse_temperature_vec: F32Vec = @splat(inverse_temperature);
        var probability_sum_vec: F32Vec = @splat(0);

        vec_index = 0;
        while (vec_index + VEC_LEN - 1 < logits.len) : (vec_index += VEC_LEN) {
            const logit_vec: F32Vec = logits[vec_index..][0..VEC_LEN].*;
            const scaled = (logit_vec - max_logit_splat) * inverse_temperature_vec;
            const prob_vec = ops.fastExp(scaled);
            probabilities[vec_index..][0..VEC_LEN].* = prob_vec;
            probability_sum_vec += prob_vec;
        }
        var probability_sum = @reduce(.Add, probability_sum_vec);
        // Handle remainder
        for (logits[vec_index..], probabilities[vec_index..]) |logit, *probability| {
            probability.* = ops.fastExpScalar((logit - max_logit_value) * inverse_temperature);
            probability_sum += probability.*;
        }

        // Normalize - SIMD
        const inv_probability_sum = 1.0 / probability_sum;
        const inv_probability_sum_vec: F32Vec = @splat(inv_probability_sum);
        vec_index = 0;
        while (vec_index + VEC_LEN - 1 < probabilities.len) : (vec_index += VEC_LEN) {
            const prob_vec: F32Vec = probabilities[vec_index..][0..VEC_LEN].*;
            probabilities[vec_index..][0..VEC_LEN].* = prob_vec * inv_probability_sum_vec;
        }
        for (probabilities[vec_index..]) |*probability| probability.* *= inv_probability_sum;

        // Apply min_p filtering if configured
        // Tokens with probability < min_p * max_prob are zeroed out
        if (config.min_p > 0.0) {
            var max_probability: f32 = 0.0;
            for (probabilities) |probability| max_probability = @max(max_probability, probability);

            const min_probability_threshold = config.min_p * max_probability;
            var filtered_probability_sum: f32 = 0.0;
            for (probabilities) |*probability| {
                if (probability.* < min_probability_threshold) {
                    probability.* = 0.0;
                } else {
                    filtered_probability_sum += probability.*;
                }
            }

            // Renormalize after filtering
            if (filtered_probability_sum > 0) {
                const inverse_filtered_sum = 1.0 / filtered_probability_sum;
                for (probabilities) |*probability| probability.* *= inverse_filtered_sum;
            }
        }

        if (config.strategy == .top_k) {
            // For top_k: use quick select O(N) to find top K, then sort only those K
            const sorted_entries = self.workspace.sorted_entries[0..logits.len];
            for (probabilities, 0..) |probability, token_index| {
                sorted_entries[token_index] = IndexValue.init(@intCast(token_index), probability);
            }

            const top_k_count = @min(config.top_k, logits.len);

            // Quick select partitions so top K are in [0..k), rest are in [k..n)
            quickSelectTopK(sorted_entries, top_k_count);

            // Sort only the top K elements (typically 40 vs 152K)
            std.sort.pdq(IndexValue, sorted_entries[0..top_k_count], {}, byProbabilityDesc);

            // Compute sum of top-k
            var top_k_prob_sum: f32 = 0;
            for (sorted_entries[0..top_k_count]) |entry| {
                top_k_prob_sum += probabilities[entry.index];
            }
            if (top_k_prob_sum == 0) return error.InvalidInput;

            renormalizeSubset(probabilities, sorted_entries[0..top_k_count], top_k_prob_sum);
        } else if (config.strategy == .top_p) {
            // For top_p: we need full sort since we don't know cutoff ahead of time
            const sorted_entries = self.workspace.sorted_entries[0..logits.len];
            for (probabilities, 0..) |probability, token_index| {
                sorted_entries[token_index] = IndexValue.init(@intCast(token_index), probability);
            }
            std.sort.pdq(IndexValue, sorted_entries, {}, byProbabilityDesc);

            // Find cutoff and compute sum in one pass
            var cumulative_probability: f32 = 0;
            var cutoff_len: usize = logits.len;
            for (sorted_entries, 0..) |entry, sorted_index| {
                cumulative_probability += entry.value;
                if (cumulative_probability >= config.top_p) {
                    cutoff_len = sorted_index + 1;
                    break;
                }
            }
            if (cumulative_probability == 0) return error.InvalidInput;

            renormalizeSubset(probabilities, sorted_entries[0..cutoff_len], cumulative_probability);
        }

        // Sample from multinomial
        const random_draw = self.prng.random().float(f32);
        var cumulative_probability: f32 = 0;
        var sampled_index: usize = logits.len - 1;
        for (probabilities, 0..) |probability, token_index| {
            cumulative_probability += probability;
            if (random_draw < cumulative_probability) {
                sampled_index = token_index;
                break;
            }
        }
        return sampled_index;
    }
};

fn byProbabilityDesc(_: void, a: IndexValue, b: IndexValue) bool {
    return a.value > b.value;
}

/// Zero all probs then write back a subset with renormalization.
/// Used by both top_k and top_p after determining the cutoff set.
inline fn renormalizeSubset(probabilities: []f32, sorted_entries: []const IndexValue, subset_sum: f32) void {
    const inverse_sum = 1.0 / subset_sum;
    @memset(probabilities, 0);
    for (sorted_entries) |entry| probabilities[entry.index] = entry.value * inverse_sum;
}

/// Hoare partition (fewer swaps than Lomuto, descending order)
inline fn partition(items: []IndexValue, left_bound: usize, right_bound: usize) usize {
    // Median-of-three pivot selection
    const middle = left_bound + (right_bound - left_bound) / 2;
    const left_value = items[left_bound].value;
    const middle_value = items[middle].value;
    const right_value = items[right_bound].value;
    const pivot_index = if ((left_value >= middle_value) == (middle_value >= right_value)) middle else if ((left_value >= middle_value) == (right_value >= left_value)) left_bound else right_bound;
    const pivot_value = items[pivot_index].value;

    var left_index = left_bound;
    var right_index = right_bound;

    while (true) {
        // Find element smaller than pivot (wrong side for descending)
        while (items[left_index].value > pivot_value) left_index += 1;
        // Find element larger than pivot (wrong side for descending)
        while (items[right_index].value < pivot_value) right_index -= 1;

        if (left_index >= right_index) return right_index;

        // Swap
        const swap_value = items[left_index];
        items[left_index] = items[right_index];
        items[right_index] = swap_value;
        left_index += 1;
        right_index -= 1;
    }
}

/// Quick select to partially sort so that the top K elements are in positions [0..k)
/// Uses Hoare partition with median-of-three pivot selection
fn quickSelectTopK(items: []IndexValue, top_k: usize) void {
    if (items.len <= 1 or top_k == 0) return;
    const target_count = @min(top_k, items.len);

    var left_index: usize = 0;
    var right_index: usize = items.len - 1;

    while (left_index < right_index) {
        const pivot_index = partition(items, left_index, right_index);

        // Hoare partition: elements in [lo..p] are >= pivot, [p+1..hi] are <= pivot
        if (pivot_index + 1 >= target_count) {
            right_index = pivot_index;
        } else {
            left_index = pivot_index + 1;
        }
    }
}

/// Apply repetition penalty to logits for tokens that appear in context.
/// Penalty > 1.0 discourages repetition, < 1.0 encourages it.
/// Algorithm: if logit > 0, divide by penalty; if logit < 0, multiply by penalty.
/// This ensures positive logits become less positive and negative become more negative.
fn applyRepetitionPenalty(logits: []f32, context_tokens: []const u32, penalty: f32) void {
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

/// Apply logit bias to specific tokens.
/// Adds bias value to each token's logit, modifying sampling probabilities.
/// Positive bias increases probability, negative decreases it.
/// Use large negative values (-100) to effectively ban tokens.
fn applyLogitBias(logits: []f32, bias_entries: []const LogitBiasEntry) void {
    for (bias_entries) |bias_entry| {
        if (bias_entry.token_id < logits.len) {
            logits[bias_entry.token_id] += bias_entry.bias;
        }
    }
}

test "sample greedy picks max" {
    var sampler_state = try Sampler.init(std.testing.allocator, 1, 16);
    defer sampler_state.deinit();
    const config = SamplingConfig{ .strategy = .greedy };

    // Table-driven test cases: {logits, expected_index}
    const cases = .{
        .{ &[_]f32{ 0.1, 0.9, 0.2 }, 1 },
        .{ &[_]f32{ 5.0, 1.0, 2.0 }, 0 },
        .{ &[_]f32{ 0.0, 0.0, 1.0 }, 2 },
        .{ &[_]f32{ -1.0, -0.5, -2.0 }, 1 },
    };
    inline for (cases) |test_case| {
        try std.testing.expectEqual(test_case[1], try sampler_state.sample(test_case[0], config));
    }
}

test "sample top_k limits choices" {
    var sampler_state = try Sampler.init(std.testing.allocator, 123, 16);
    defer sampler_state.deinit();
    const logits = [_]f32{ 10.0, 9.0, 1.0 };
    const config = SamplingConfig{ .strategy = .top_k, .top_k = 1 };
    // Only index 0 should ever be chosen
    for (0..3) |_| {
        try std.testing.expectEqual(@as(usize, 0), try sampler_state.sample(&logits, config));
    }
}

// ============================================================================
// EXPANDED UNIT TESTS
// ============================================================================

// ----------------------------------------------------------------------------
// top_p sampling tests
// ----------------------------------------------------------------------------

// Test top_p sampling with cumulative probability cutoff at 0.9.
// Should only sample from tokens whose cumulative probability <= 0.9.
test "sample top_p cutoff" {
    var sampler_state = try Sampler.init(std.testing.allocator, 42, 16);
    defer sampler_state.deinit();

    // Logits that produce probabilities: [0.5, 0.3, 0.15, 0.05] after softmax
    // With top_p=0.9: should only sample from first 3 tokens (0.5 + 0.3 + 0.15 = 0.95)
    const logits = [_]f32{ 2.0, 1.4, 0.7, -0.5 };
    const config = SamplingConfig{ .strategy = .top_p, .top_p = 0.9 };

    // Run multiple samples to verify last token (index 3) is never chosen
    var counts = [_]usize{0} ** 4;
    for (0..100) |_| {
        const sampled_index = try sampler_state.sample(&logits, config);
        try std.testing.expect(sampled_index < 4);
        counts[sampled_index] += 1;
    }

    // Last token should never be sampled with top_p=0.9
    try std.testing.expectEqual(@as(usize, 0), counts[3]);
}

test "sample top_p with p=1.0 includes all" {
    var sampler_state = try Sampler.init(std.testing.allocator, 456, 16);
    defer sampler_state.deinit();

    const logits = [_]f32{ 1.0, 0.5, 0.2, -0.5 };
    const config = SamplingConfig{ .strategy = .top_p, .top_p = 1.0 };

    // With top_p=1.0, all tokens should be potentially sampled
    var counts = [_]usize{0} ** 4;
    for (0..200) |_| {
        const sampled_index = try sampler_state.sample(&logits, config);
        counts[sampled_index] += 1;
    }

    // All tokens should have been sampled at least once (probabilistically)
    for (counts) |count| {
        try std.testing.expect(count > 0);
    }
}

// Test top_p with very small p value (should sample only top token).
test "sample top_p with very small p" {
    var sampler_state = try Sampler.init(std.testing.allocator, 789, 16);
    defer sampler_state.deinit();

    const logits = [_]f32{ 10.0, 5.0, 2.0, 0.5 };
    const config = SamplingConfig{ .strategy = .top_p, .top_p = 0.01 };

    // With very small top_p, should only sample the highest probability token
    for (0..10) |_| {
        const sampled_index = try sampler_state.sample(&logits, config);
        try std.testing.expectEqual(@as(usize, 0), sampled_index);
    }
}

// ----------------------------------------------------------------------------
// Temperature scaling tests
// ----------------------------------------------------------------------------

test "sample high temperature uniform" {
    var sampler_state = try Sampler.init(std.testing.allocator, 111, 16);
    defer sampler_state.deinit();

    // Highly skewed logits
    const logits = [_]f32{ 10.0, 1.0, 0.5, 0.1 };

    // High temperature should flatten the distribution
    const config = SamplingConfig{ .strategy = .top_k, .top_k = 4, .temperature = 10.0 };

    var counts = [_]usize{0} ** 4;
    for (0..1000) |_| {
        const sampled_index = try sampler_state.sample(&logits, config);
        counts[sampled_index] += 1;
    }

    // With high temperature, distribution should be more uniform than without.
    // Very loose bounds: each token should be sampled at least once and no single
    // token should dominate completely (< 90% of samples).
    for (counts) |count| {
        try std.testing.expect(count > 0);
        try std.testing.expect(count < 900);
    }
}

// Test that low temperature makes distribution more peaked (greedy-like).
// At temperature → 0, should always pick the argmax.
test "sample low temperature peaked" {
    var sampler_state = try Sampler.init(std.testing.allocator, 222, 16);
    defer sampler_state.deinit();

    const logits = [_]f32{ 2.0, 1.9, 1.8, 0.5 };

    // Very low temperature should make it nearly greedy
    const config = SamplingConfig{ .strategy = .top_k, .top_k = 4, .temperature = 0.01 };

    var counts = [_]usize{0} ** 4;
    for (0..100) |_| {
        const sampled_index = try sampler_state.sample(&logits, config);
        counts[sampled_index] += 1;
    }

    // Should almost always pick index 0 (highest logit)
    try std.testing.expect(counts[0] > 90);
}

// Test temperature = 1.0 produces expected softmax distribution.
test "sample temperature 1.0 standard softmax" {
    var sampler_state = try Sampler.init(std.testing.allocator, 333, 16);
    defer sampler_state.deinit();

    const logits = [_]f32{ 1.0, 0.0, -1.0 };
    const config = SamplingConfig{ .strategy = .top_k, .top_k = 3, .temperature = 1.0 };

    var counts = [_]usize{0} ** 3;
    for (0..1000) |_| {
        const sampled_index = try sampler_state.sample(&logits, config);
        counts[sampled_index] += 1;
    }

    // With temp=1.0, probabilities should follow standard softmax
    // Token 0 should be sampled most frequently
    try std.testing.expect(counts[0] > counts[1]);
    try std.testing.expect(counts[1] > counts[2]);
}

// ----------------------------------------------------------------------------
// Repetition penalty tests
// ----------------------------------------------------------------------------

// Test that repetition penalty reduces probability of repeated tokens.
// Penalty > 1.0 should discourage tokens in context.
test "sample repetition penalty" {
    var sampler_state = try Sampler.init(std.testing.allocator, 444, 16);
    defer sampler_state.deinit();

    // Mutable logits for repetition penalty
    const logits = [_]f32{ 2.0, 2.0, 2.0, 2.0 };
    const context_tokens = [_]u32{ 0, 1 }; // Penalize tokens 0 and 1

    const config = SamplingConfig{
        .strategy = .top_k,
        .top_k = 4,
        .temperature = 1.0,
        .repetition_penalty = 2.0,
        .context_tokens = &context_tokens,
    };

    var counts = [_]usize{0} ** 4;
    for (0..1000) |_| {
        // Reset logits each iteration since sampleMut modifies them
        var logits_copy = logits;
        const sampled_index = try sampler_state.sampleMut(&logits_copy, config);
        counts[sampled_index] += 1;
    }

    // Tokens 2 and 3 (not in context) should be sampled more frequently
    try std.testing.expect(counts[2] + counts[3] > counts[0] + counts[1]);
}

// Test repetition penalty with positive and negative logits.
// Positive logits are divided by penalty, negative logits are multiplied.
test "sample repetition penalty positive negative" {
    var logits = [_]f32{ 2.0, -2.0, 1.0, -1.0 };
    const context_tokens = [_]u32{ 0, 1 };
    const penalty: f32 = 2.0;

    applyRepetitionPenalty(&logits, &context_tokens, penalty);

    // Token 0: positive logit divided by penalty: 2.0 / 2.0 = 1.0
    try std.testing.expectApproxEqAbs(1.0, logits[0], 0.001);

    // Token 1: negative logit multiplied by penalty: -2.0 * 2.0 = -4.0
    try std.testing.expectApproxEqAbs(-4.0, logits[1], 0.001);

    // Tokens 2 and 3 should be unchanged
    try std.testing.expectApproxEqAbs(1.0, logits[2], 0.001);
    try std.testing.expectApproxEqAbs(-1.0, logits[3], 0.001);
}

// Test that repetition penalty = 1.0 has no effect.
test "sample repetition penalty 1.0 noop" {
    var logits = [_]f32{ 2.0, 1.0, 0.5, -1.0 };
    const original_logits = logits;
    const context_tokens = [_]u32{ 0, 1, 2 };
    const penalty: f32 = 1.0;

    applyRepetitionPenalty(&logits, &context_tokens, penalty);

    // Logits should be unchanged
    for (logits, original_logits) |logit, original| {
        try std.testing.expectEqual(original, logit);
    }
}

test "sampleMut modifies logits in place" {
    var sampler_state = try Sampler.init(std.testing.allocator, 5555, 16);
    defer sampler_state.deinit();

    // Create logits and save original values
    var logits = [_]f32{ 6.0, 5.0, 2.0, 1.0 };
    const original_logits = logits;

    // Configure with repetition penalty to modify logits
    const context_tokens = [_]u32{ 0, 1 }; // Penalize tokens 0 and 1
    const config = SamplingConfig{
        .strategy = .greedy,
        .repetition_penalty = 4.0, // Strong penalty to ensure token 2 becomes max
        .context_tokens = &context_tokens,
    };

    // Call sampleMut - should modify logits in place
    const sampled_index = try sampler_state.sampleMut(&logits, config);

    // Verify logits were modified (tokens 0 and 1 should be penalized)
    // Token 0: positive logit divided by penalty: 6.0 / 4.0 = 1.5
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), logits[0], 0.001);
    // Token 1: positive logit divided by penalty: 5.0 / 4.0 = 1.25
    try std.testing.expectApproxEqAbs(@as(f32, 1.25), logits[1], 0.001);
    // Tokens 2 and 3 should be unchanged
    try std.testing.expectEqual(original_logits[2], logits[2]);
    try std.testing.expectEqual(original_logits[3], logits[3]);

    // Verify a valid token index was returned
    try std.testing.expect(sampled_index < 4);
    // With greedy strategy and penalty applied, token 2 should be selected (now has highest logit at 2.0)
    try std.testing.expectEqual(@as(usize, 2), sampled_index);
}

// ----------------------------------------------------------------------------
// Logit bias tests
// ----------------------------------------------------------------------------

// Test that logit bias modifies specific tokens.
test "sample logit bias modifies" {
    var logits = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const bias_entries = [_]LogitBiasEntry{
        .{ .token_id = 0, .bias = 10.0 }, // Boost token 0
        .{ .token_id = 2, .bias = -5.0 }, // Reduce token 2
    };

    applyLogitBias(&logits, &bias_entries);

    try std.testing.expectApproxEqAbs(@as(f32, 11.0), logits[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), logits[1], 0.001); // Unchanged
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), logits[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), logits[3], 0.001); // Unchanged
}

// Test that negative bias can effectively ban tokens.
test "sample logit bias large negative bans" {
    var sampler_state = try Sampler.init(std.testing.allocator, 999, 16);
    defer sampler_state.deinit();

    // Token 0 has highest logit initially
    var logits = [_]f32{ 10.0, 5.0, 2.0, 1.0 };
    const bias_entries = [_]LogitBiasEntry{
        .{ .token_id = 0, .bias = -100.0 }, // Ban token 0
    };

    const config = SamplingConfig{
        .strategy = .top_k,
        .top_k = 4,
        .temperature = 1.0,
        .logit_bias = &bias_entries,
    };

    // With bias applied, token 0 should never be selected
    for (0..50) |_| {
        const sampled_index = try sampler_state.sampleMut(&logits, config);
        try std.testing.expect(sampled_index != 0);
        // Reset logits for next iteration
        logits = [_]f32{ 10.0, 5.0, 2.0, 1.0 };
    }
}

// Test that positive bias can force token selection.
test "sample logit bias large positive forces" {
    var sampler_state = try Sampler.init(std.testing.allocator, 888, 16);
    defer sampler_state.deinit();

    // Token 3 has lowest logit initially
    var logits = [_]f32{ 5.0, 4.0, 3.0, 0.1 };
    const bias_entries = [_]LogitBiasEntry{
        .{ .token_id = 3, .bias = 100.0 }, // Strongly boost token 3
    };

    const config = SamplingConfig{
        .strategy = .greedy,
        .logit_bias = &bias_entries,
    };

    // With large bias, token 3 should be selected (greedy picks max)
    const sampled_index = try sampler_state.sampleMut(&logits, config);
    try std.testing.expectEqual(@as(usize, 3), sampled_index);
}

// Test that out-of-bounds token IDs are ignored safely.
test "sample logit bias ignores out-of-bounds" {
    var logits = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const original_logits = logits;
    const bias_entries = [_]LogitBiasEntry{
        .{ .token_id = 100, .bias = 50.0 }, // Out of bounds
        .{ .token_id = 999, .bias = -50.0 }, // Out of bounds
    };

    applyLogitBias(&logits, &bias_entries);

    // Logits should be unchanged (out-of-bounds ignored)
    for (logits, original_logits) |logit, original| {
        try std.testing.expectEqual(original, logit);
    }
}

// ----------------------------------------------------------------------------
// Numerical stability tests
// ----------------------------------------------------------------------------

// Test sampling with very large logits (overflow resistance).
// Should handle logits near f32 max without overflow.
test "sample numerical stability large" {
    var sampler_state = try Sampler.init(std.testing.allocator, 555, 16);
    defer sampler_state.deinit();

    // Very large logits
    const logits = [_]f32{ 1000.0, 999.0, 998.0, 500.0 };
    const config = SamplingConfig{ .strategy = .top_k, .top_k = 4, .temperature = 1.0 };

    // Should not crash or produce NaN
    const sampled_index = try sampler_state.sample(&logits, config);
    try std.testing.expect(sampled_index < 4);
}

// Test sampling with very small logits (underflow resistance).
// Should handle very negative logits without underflow issues.
test "sample numerical stability small" {
    var sampler_state = try Sampler.init(std.testing.allocator, 666, 16);
    defer sampler_state.deinit();

    // Very small (negative) logits
    const logits = [_]f32{ -100.0, -200.0, -300.0, -400.0 };
    const config = SamplingConfig{ .strategy = .top_k, .top_k = 4, .temperature = 1.0 };

    // Should pick the least negative (index 0)
    const sampled_index = try sampler_state.sample(&logits, config);
    try std.testing.expectEqual(@as(usize, 0), sampled_index);
}

// Test sampling when all logits are identical.
// Should produce uniform distribution.
test "sample numerical stability same" {
    var sampler_state = try Sampler.init(std.testing.allocator, 777, 16);
    defer sampler_state.deinit();

    // All logits identical
    const logits = [_]f32{ 5.0, 5.0, 5.0, 5.0 };
    const config = SamplingConfig{ .strategy = .top_k, .top_k = 4, .temperature = 1.0 };

    var counts = [_]usize{0} ** 4;
    for (0..1000) |_| {
        const sampled_index = try sampler_state.sample(&logits, config);
        counts[sampled_index] += 1;
    }

    // Should be roughly uniform (each token ~25%)
    for (counts) |count| {
        try std.testing.expect(count > 200);
        try std.testing.expect(count < 300);
    }
}

// Test that very large differences in logits don't cause numerical issues.
test "sample numerical stability large differences" {
    var sampler_state = try Sampler.init(std.testing.allocator, 888, 16);
    defer sampler_state.deinit();

    // Mix of very large and very small logits
    const logits = [_]f32{ 100.0, -100.0, 50.0, -50.0 };
    const config = SamplingConfig{ .strategy = .top_k, .top_k = 4, .temperature = 1.0 };

    // Should always pick index 0 (much larger than others)
    for (0..10) |_| {
        const sampled_index = try sampler_state.sample(&logits, config);
        try std.testing.expectEqual(@as(usize, 0), sampled_index);
    }
}

// ----------------------------------------------------------------------------
// Edge case tests
// ----------------------------------------------------------------------------

// Test sampling with single token vocabulary.
test "sample single token vocab" {
    var sampler_state = try Sampler.init(std.testing.allocator, 999, 16);
    defer sampler_state.deinit();

    const logits = [_]f32{42.0};

    // Test with different strategies
    const configs = [_]SamplingConfig{
        .{ .strategy = .greedy },
        .{ .strategy = .top_k, .top_k = 1 },
        .{ .strategy = .top_p, .top_p = 0.9 },
    };

    for (configs) |config| {
        const sampled_index = try sampler_state.sample(&logits, config);
        try std.testing.expectEqual(@as(usize, 0), sampled_index);
    }
}

// Test that empty logits array returns an error.
test "sample empty logits handling" {
    var sampler_state = try Sampler.init(std.testing.allocator, 1111, 16);
    defer sampler_state.deinit();

    const logits = [_]f32{};
    const config = SamplingConfig{ .strategy = .greedy };

    // Should return InvalidInput error
    try std.testing.expectError(error.InvalidInput, sampler_state.sample(&logits, config));
}

// Test two-token vocabulary edge case.
test "sample two token vocab" {
    var sampler_state = try Sampler.init(std.testing.allocator, 1212, 16);
    defer sampler_state.deinit();

    const logits = [_]f32{ 1.0, 0.0 };
    const config = SamplingConfig{ .strategy = .top_k, .top_k = 2, .temperature = 1.0 };

    var counts = [_]usize{0} ** 2;
    for (0..100) |_| {
        const sampled_index = try sampler_state.sample(&logits, config);
        counts[sampled_index] += 1;
    }

    // First token should be sampled more frequently
    try std.testing.expect(counts[0] > counts[1]);
}

// ----------------------------------------------------------------------------
// top_k edge case tests
// ----------------------------------------------------------------------------

// Test top_k = 1 (should behave like greedy sampling).
test "sample top_k k=1 is greedy" {
    var sampler_state = try Sampler.init(std.testing.allocator, 1313, 16);
    defer sampler_state.deinit();

    const logits = [_]f32{ 5.0, 10.0, 3.0, 7.0 };
    const config = SamplingConfig{ .strategy = .top_k, .top_k = 1 };

    // Should always pick index 1 (highest logit)
    for (0..10) |_| {
        const sampled_index = try sampler_state.sample(&logits, config);
        try std.testing.expectEqual(@as(usize, 1), sampled_index);
    }
}

// Test top_k > vocab_size (should include all tokens).
test "sample top_k k greater than vocab" {
    var sampler_state = try Sampler.init(std.testing.allocator, 1414, 16);
    defer sampler_state.deinit();

    const logits = [_]f32{ 1.0, 2.0, 3.0 };
    // Set k to 100, much larger than vocab size of 3
    const config = SamplingConfig{ .strategy = .top_k, .top_k = 100, .temperature = 1.0 };

    var counts = [_]usize{0} ** 3;
    for (0..300) |_| {
        const sampled_index = try sampler_state.sample(&logits, config);
        counts[sampled_index] += 1;
    }

    // All tokens should be sampled
    for (counts) |count| {
        try std.testing.expect(count > 0);
    }
}

// Test top_k = 0 edge case.
test "sample top_k k=0" {
    var sampler_state = try Sampler.init(std.testing.allocator, 1515, 16);
    defer sampler_state.deinit();

    const logits = [_]f32{ 1.0, 2.0, 3.0 };
    const config = SamplingConfig{ .strategy = .top_k, .top_k = 0 };

    // k=0 should result in error (no tokens to sample)
    try std.testing.expectError(error.InvalidInput, sampler_state.sample(&logits, config));
}

// Test top_k with equal probabilities.
test "sample top_k with equal probabilities" {
    var sampler_state = try Sampler.init(std.testing.allocator, 1616, 16);
    defer sampler_state.deinit();

    const logits = [_]f32{ 2.0, 2.0, 2.0, 2.0, 2.0 };
    const config = SamplingConfig{ .strategy = .top_k, .top_k = 3, .temperature = 1.0 };

    var sampled = [_]bool{false} ** 5;
    for (0..100) |_| {
        const sampled_index = try sampler_state.sample(&logits, config);
        sampled[sampled_index] = true;
    }

    // Should sample from at least some of the tokens
    var sampled_count: usize = 0;
    for (sampled) |was_sampled| {
        if (was_sampled) sampled_count += 1;
    }
    try std.testing.expect(sampled_count >= 2);
}

// ----------------------------------------------------------------------------
// Invalid parameter tests
// ----------------------------------------------------------------------------

// Test invalid temperature (negative) returns error.
test "sample invalid negative temp" {
    var sampler_state = try Sampler.init(std.testing.allocator, 1717, 16);
    defer sampler_state.deinit();

    const logits = [_]f32{ 1.0, 2.0, 3.0 };
    const config = SamplingConfig{ .strategy = .top_k, .top_k = 2, .temperature = -1.0 };

    try std.testing.expectError(error.InvalidTemperature, sampler_state.sample(&logits, config));
}

// Test invalid top_p (< 0) returns error.
test "sample invalid negative top_p" {
    var sampler_state = try Sampler.init(std.testing.allocator, 1818, 16);
    defer sampler_state.deinit();

    const logits = [_]f32{ 1.0, 2.0, 3.0 };
    const config = SamplingConfig{ .strategy = .top_p, .top_p = -0.5 };

    try std.testing.expectError(error.InvalidTopP, sampler_state.sample(&logits, config));
}

// Test invalid top_p (> 1.0) returns error.
test "sample invalid top_p greater than 1.0" {
    var sampler_state = try Sampler.init(std.testing.allocator, 1919, 16);
    defer sampler_state.deinit();

    const logits = [_]f32{ 1.0, 2.0, 3.0 };
    const config = SamplingConfig{ .strategy = .top_p, .top_p = 1.5 };

    try std.testing.expectError(error.InvalidTopP, sampler_state.sample(&logits, config));
}

// Test zero temperature with non-greedy strategy returns error.
test "sample invalid zero temp non-greedy" {
    var sampler_state = try Sampler.init(std.testing.allocator, 2020, 16);
    defer sampler_state.deinit();

    const logits = [_]f32{ 1.0, 2.0, 3.0 };
    const config = SamplingConfig{ .strategy = .top_k, .top_k = 2, .temperature = 0.0 };

    try std.testing.expectError(error.InvalidTemperature, sampler_state.sample(&logits, config));
}

// Test invalid min_p (< 0) returns error.
test "sample invalid negative min_p" {
    var sampler_state = try Sampler.init(std.testing.allocator, 2121, 16);
    defer sampler_state.deinit();

    const logits = [_]f32{ 1.0, 2.0, 3.0 };
    const config = SamplingConfig{ .strategy = .top_k, .top_k = 2, .min_p = -0.1 };

    try std.testing.expectError(error.InvalidMinP, sampler_state.sample(&logits, config));
}

// Test invalid min_p (> 1.0) returns error.
test "sample invalid min_p greater than 1.0" {
    var sampler_state = try Sampler.init(std.testing.allocator, 2222, 16);
    defer sampler_state.deinit();

    const logits = [_]f32{ 1.0, 2.0, 3.0 };
    const config = SamplingConfig{ .strategy = .top_k, .top_k = 2, .min_p = 1.5 };

    try std.testing.expectError(error.InvalidMinP, sampler_state.sample(&logits, config));
}

// Test logits larger than workspace returns error.
test "sample invalid logits exceed workspace" {
    var sampler_state = try Sampler.init(std.testing.allocator, 2323, 4);
    defer sampler_state.deinit();

    // Create logits array larger than workspace (4)
    const logits = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const config = SamplingConfig{ .strategy = .greedy };

    try std.testing.expectError(error.InvalidInput, sampler_state.sample(&logits, config));
}

// ----------------------------------------------------------------------------
// Workspace lifecycle tests
// ----------------------------------------------------------------------------

test "workspace init and deinit" {
    var workspace = try Workspace.init(std.testing.allocator, 100);
    defer workspace.deinit();

    try std.testing.expectEqual(@as(usize, 100), workspace.probabilities.len);
    try std.testing.expectEqual(@as(usize, 100), workspace.sorted_entries.len);
}

test "Workspace.init large vocab size" {
    var workspace = try Workspace.init(std.testing.allocator, 50000);
    defer workspace.deinit();

    try std.testing.expectEqual(@as(usize, 50000), workspace.probabilities.len);
    try std.testing.expectEqual(@as(usize, 50000), workspace.sorted_entries.len);
}

test "Workspace.init minimal vocab size" {
    var workspace = try Workspace.init(std.testing.allocator, 1);
    defer workspace.deinit();

    try std.testing.expectEqual(@as(usize, 1), workspace.probabilities.len);
    try std.testing.expectEqual(@as(usize, 1), workspace.sorted_entries.len);
}

// ----------------------------------------------------------------------------
// Sampler lifecycle tests
// ----------------------------------------------------------------------------

test "sampler init and deinit" {
    var sampler_state = try Sampler.init(std.testing.allocator, 42, 100);
    defer sampler_state.deinit();

    try std.testing.expectEqual(@as(usize, 100), sampler_state.workspace.probabilities.len);
    try std.testing.expectEqual(@as(usize, 100), sampler_state.workspace.sorted_entries.len);
}

test "Sampler.sample deterministic seed" {
    const seed: u64 = 12345;
    var sampler1 = try Sampler.init(std.testing.allocator, seed, 16);
    defer sampler1.deinit();

    var sampler2 = try Sampler.init(std.testing.allocator, seed, 16);
    defer sampler2.deinit();

    const logits = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const config = SamplingConfig{ .strategy = .top_k, .top_k = 4, .temperature = 1.0 };

    // Both samplers should produce identical sequences
    for (0..10) |_| {
        const idx1 = try sampler1.sample(&logits, config);
        const idx2 = try sampler2.sample(&logits, config);
        try std.testing.expectEqual(idx1, idx2);
    }
}

test "Sampler.sample different seeds" {
    var sampler1 = try Sampler.init(std.testing.allocator, 111, 16);
    defer sampler1.deinit();

    var sampler2 = try Sampler.init(std.testing.allocator, 999, 16);
    defer sampler2.deinit();

    const logits = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const config = SamplingConfig{ .strategy = .top_k, .top_k = 4, .temperature = 1.0 };

    // Collect samples from both
    var samples1 = [_]usize{0} ** 20;
    var samples2 = [_]usize{0} ** 20;

    for (0..20) |i| {
        samples1[i] = try sampler1.sample(&logits, config);
        samples2[i] = try sampler2.sample(&logits, config);
    }

    // Sequences should differ (probabilistically almost certain)
    var differences: usize = 0;
    for (samples1, samples2) |s1, s2| {
        if (s1 != s2) differences += 1;
    }
    try std.testing.expect(differences > 0);
}

// ----------------------------------------------------------------------------
// min_p filtering tests
// ----------------------------------------------------------------------------

test "sample min_p filters low probability" {
    var sampler_state = try Sampler.init(std.testing.allocator, 2424, 16);
    defer sampler_state.deinit();

    // Logits that create a clear hierarchy after softmax
    const logits = [_]f32{ 10.0, 5.0, 1.0, 0.1 };
    const config = SamplingConfig{ .strategy = .top_k, .top_k = 4, .temperature = 1.0, .min_p = 0.1 };

    // With min_p = 0.1, tokens with prob < 0.1 * max_prob should be filtered
    var counts = [_]usize{0} ** 4;
    for (0..200) |_| {
        const sampled_index = try sampler_state.sample(&logits, config);
        counts[sampled_index] += 1;
    }

    // Token 0 should be sampled most, token 3 should rarely if ever be sampled
    try std.testing.expect(counts[0] > counts[3]);
}

test "sample min_p with 0.0 has no effect" {
    var sampler_state = try Sampler.init(std.testing.allocator, 2525, 16);
    defer sampler_state.deinit();

    const logits = [_]f32{ 2.0, 1.0, 0.5, 0.1 };
    const config = SamplingConfig{ .strategy = .top_k, .top_k = 4, .temperature = 1.0, .min_p = 0.0 };

    var counts = [_]usize{0} ** 4;
    for (0..200) |_| {
        const sampled_index = try sampler_state.sample(&logits, config);
        counts[sampled_index] += 1;
    }

    // All tokens should be sampled (no filtering)
    for (counts) |count| {
        try std.testing.expect(count > 0);
    }
}

test "sample min_p with 1.0 only max" {
    var sampler_state = try Sampler.init(std.testing.allocator, 2626, 16);
    defer sampler_state.deinit();

    const logits = [_]f32{ 10.0, 5.0, 2.0, 1.0 };
    const config = SamplingConfig{ .strategy = .top_k, .top_k = 4, .temperature = 1.0, .min_p = 1.0 };

    // With min_p = 1.0, only tokens with prob >= max_prob can be sampled
    // This means only the max probability token
    for (0..10) |_| {
        const sampled_index = try sampler_state.sample(&logits, config);
        try std.testing.expectEqual(@as(usize, 0), sampled_index);
    }
}

test "sample min_p combined with top_p" {
    var sampler_state = try Sampler.init(std.testing.allocator, 2727, 16);
    defer sampler_state.deinit();

    const logits = [_]f32{ 5.0, 4.0, 3.0, 0.1, 0.05 };
    const config = SamplingConfig{
        .strategy = .top_p,
        .top_p = 0.9,
        .temperature = 1.0,
        .min_p = 0.05, // Filter very low probability tokens
    };

    var counts = [_]usize{0} ** 5;
    for (0..200) |_| {
        const sampled_index = try sampler_state.sample(&logits, config);
        counts[sampled_index] += 1;
    }

    // Last token should be filtered out or very rare
    try std.testing.expect(counts[4] < 10);
}

test "sample min_p combined with top_k" {
    var sampler_state = try Sampler.init(std.testing.allocator, 2828, 16);
    defer sampler_state.deinit();

    const logits = [_]f32{ 10.0, 9.0, 1.0, 0.5, 0.1 };
    const config = SamplingConfig{
        .strategy = .top_k,
        .top_k = 4,
        .temperature = 1.0,
        .min_p = 0.05, // Filter very low probability tokens first
    };

    var counts = [_]usize{0} ** 5;
    for (0..200) |_| {
        const sampled_index = try sampler_state.sample(&logits, config);
        counts[sampled_index] += 1;
    }

    // Last token should be very rare or never sampled
    try std.testing.expect(counts[4] < 10);
}

// ----------------------------------------------------------------------------
// Internal function tests: byProbabilityDesc
// ----------------------------------------------------------------------------

test "Sampler.sample byProbabilityDesc descending" {
    const a = IndexValue.init(0, 5.0);
    const b = IndexValue.init(1, 10.0);
    const c = IndexValue.init(2, 3.0);

    // b > a > c
    try std.testing.expect(byProbabilityDesc({}, b, a)); // 10.0 > 5.0
    try std.testing.expect(byProbabilityDesc({}, a, c)); // 5.0 > 3.0
    try std.testing.expect(byProbabilityDesc({}, b, c)); // 10.0 > 3.0

    // Not sorted: a < b
    try std.testing.expect(!byProbabilityDesc({}, a, b)); // 5.0 not > 10.0
}

test "Sampler.sample byProbabilityDesc equal values" {
    const a = IndexValue.init(0, 5.0);
    const b = IndexValue.init(1, 5.0);

    // Equal values should return false (not greater)
    try std.testing.expect(!byProbabilityDesc({}, a, b));
    try std.testing.expect(!byProbabilityDesc({}, b, a));
}

test "Sampler.sample byProbabilityDesc negative values" {
    const a = IndexValue.init(0, -1.0);
    const b = IndexValue.init(1, -5.0);

    // -1.0 > -5.0
    try std.testing.expect(byProbabilityDesc({}, a, b));
    try std.testing.expect(!byProbabilityDesc({}, b, a));
}

// ----------------------------------------------------------------------------
// Internal function tests: quickSelectTopK
// ----------------------------------------------------------------------------

/// Test helper: create IndexValue array from (index, value) pairs.
fn makeIndexValues(comptime pairs: anytype) [pairs.len]IndexValue {
    var result: [pairs.len]IndexValue = std.mem.zeroes([pairs.len]IndexValue);
    inline for (pairs, 0..) |pair, i| {
        result[i].index = pair[0];
        result[i].value = pair[1];
    }
    return result;
}

test "Sampler.sample quickSelectTopK partitions" {
    var items = makeIndexValues(.{
        .{ 0, 5.0 },
        .{ 1, 10.0 },
        .{ 2, 3.0 },
        .{ 3, 8.0 },
        .{ 4, 1.0 },
    });

    quickSelectTopK(&items, 3);

    // After quickSelect, top 3 elements should be in first 3 positions
    // (not necessarily sorted, but their values should be >= remaining elements)
    const top3_vals = [_]f32{ items[0].value, items[1].value, items[2].value };
    const rest_vals = [_]f32{ items[3].value, items[4].value };

    // Find min of top 3
    var min_top: f32 = top3_vals[0];
    for (top3_vals) |val| min_top = @min(min_top, val);

    // Find max of rest
    var max_rest: f32 = rest_vals[0];
    for (rest_vals) |val| max_rest = @max(max_rest, val);

    // Min of top 3 should be >= max of rest
    try std.testing.expect(min_top >= max_rest);
}

test "Sampler.sample quickSelectTopK k=1" {
    var items = makeIndexValues(.{ .{ 0, 3.0 }, .{ 1, 7.0 }, .{ 2, 1.0 }, .{ 3, 5.0 } });
    quickSelectTopK(&items, 1);
    try std.testing.expectEqual(@as(f32, 7.0), items[0].value);
}

test "Sampler.sample quickSelectTopK k=size" {
    var items = makeIndexValues(.{ .{ 0, 3.0 }, .{ 1, 7.0 }, .{ 2, 1.0 } });
    const original_items = items;
    quickSelectTopK(&items, 3);

    // All elements should still be present (just partitioned)
    var found = [_]bool{false} ** 3;
    for (items) |item| {
        for (original_items, 0..) |orig, orig_idx| {
            if (item.index == orig.index and item.value == orig.value) {
                found[orig_idx] = true;
            }
        }
    }
    for (found) |was_found| try std.testing.expect(was_found);
}

test "Sampler.sample quickSelectTopK k=0" {
    var items = makeIndexValues(.{ .{ 0, 3.0 }, .{ 1, 7.0 } });
    quickSelectTopK(&items, 0);
    try std.testing.expect(items.len == 2);
}

test "Sampler.sample quickSelectTopK single element" {
    var items = makeIndexValues(.{.{ 0, 5.0 }});
    quickSelectTopK(&items, 1);
    try std.testing.expectEqual(@as(f32, 5.0), items[0].value);
}

test "Sampler.sample quickSelectTopK duplicates" {
    var items = makeIndexValues(.{
        .{ 0, 5.0 },
        .{ 1, 5.0 },
        .{ 2, 5.0 },
        .{ 3, 1.0 },
        .{ 4, 1.0 },
    });

    quickSelectTopK(&items, 3);

    // Top 3 should all be value 5.0
    for (items[0..3]) |item| {
        try std.testing.expectEqual(@as(f32, 5.0), item.value);
    }
}

test "Sampler.sample quickSelectTopK descending input" {
    var items = makeIndexValues(.{ .{ 0, 10.0 }, .{ 1, 9.0 }, .{ 2, 8.0 }, .{ 3, 7.0 }, .{ 4, 6.0 } });
    quickSelectTopK(&items, 3);
    for (items[0..3]) |item| try std.testing.expect(item.value >= 8.0);
}

test "Sampler.sample quickSelectTopK ascending input" {
    var items = makeIndexValues(.{ .{ 0, 1.0 }, .{ 1, 2.0 }, .{ 2, 3.0 }, .{ 3, 4.0 }, .{ 4, 5.0 } });
    quickSelectTopK(&items, 2);
    for (items[0..2]) |item| try std.testing.expect(item.value >= 4.0);
}

test "Sampler.sample quickSelectTopK large k" {
    var items = makeIndexValues(.{ .{ 0, 3.0 }, .{ 1, 7.0 }, .{ 2, 1.0 } });
    quickSelectTopK(&items, 100);
    try std.testing.expect(items.len == 3);
}

// ----------------------------------------------------------------------------
// Internal function tests: renormalizeSubset
// ----------------------------------------------------------------------------

test "Sampler.sample renormalizeSubset basic" {
    var probabilities = [_]f32{ 0.5, 0.3, 0.2, 0.1 };
    const sorted_subset = makeIndexValues(.{ .{ 0, 0.5 }, .{ 1, 0.3 } });
    renormalizeSubset(&probabilities, &sorted_subset, 0.8);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5 / 0.8), probabilities[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.3 / 0.8), probabilities[1], 0.001);
    try std.testing.expectEqual(@as(f32, 0.0), probabilities[2]);
    try std.testing.expectEqual(@as(f32, 0.0), probabilities[3]);
}

test "Sampler.sample renormalizeSubset single" {
    var probabilities = [_]f32{ 0.5, 0.3, 0.2 };
    const sorted_subset = makeIndexValues(.{.{ 1, 0.3 }});
    renormalizeSubset(&probabilities, &sorted_subset, 0.3);
    try std.testing.expectEqual(@as(f32, 0.0), probabilities[0]);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), probabilities[1], 0.001);
    try std.testing.expectEqual(@as(f32, 0.0), probabilities[2]);
}

test "Sampler.sample renormalizeSubset all" {
    var probabilities = [_]f32{ 0.4, 0.3, 0.2, 0.1 };
    const sorted_subset = makeIndexValues(.{ .{ 0, 0.4 }, .{ 1, 0.3 }, .{ 2, 0.2 }, .{ 3, 0.1 } });
    renormalizeSubset(&probabilities, &sorted_subset, 1.0);
    var sum: f32 = 0.0;
    for (probabilities) |p| sum += p;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 0.001);
}

test "Sampler.sample renormalizeSubset proportions" {
    var probabilities = [_]f32{ 0.6, 0.4, 0.2, 0.1 };
    const sorted_subset = makeIndexValues(.{ .{ 0, 0.6 }, .{ 1, 0.4 } });
    renormalizeSubset(&probabilities, &sorted_subset, 1.0);
    const ratio = probabilities[0] / probabilities[1];
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), ratio, 0.001);
}

// ----------------------------------------------------------------------------
// Integration tests: Complex scenarios
// ----------------------------------------------------------------------------

test "Sampler.sample repetition penalty with bias" {
    var sampler_state = try Sampler.init(std.testing.allocator, 3030, 16);
    defer sampler_state.deinit();

    var logits = [_]f32{ 5.0, 5.0, 5.0, 5.0 };
    const context_tokens = [_]u32{ 0, 1 };
    const bias_entries = [_]LogitBiasEntry{
        .{ .token_id = 2, .bias = -10.0 },
    };

    const config = SamplingConfig{
        .strategy = .top_k,
        .top_k = 4,
        .temperature = 1.0,
        .repetition_penalty = 2.0,
        .context_tokens = &context_tokens,
        .logit_bias = &bias_entries,
    };

    var counts = [_]usize{0} ** 4;
    for (0..200) |_| {
        logits = [_]f32{ 5.0, 5.0, 5.0, 5.0 };
        const sampled_index = try sampler_state.sampleMut(&logits, config);
        counts[sampled_index] += 1;
    }

    // Token 3 should be sampled most (no penalty, no bias)
    // Tokens 0,1 have repetition penalty
    // Token 2 has negative bias
    try std.testing.expect(counts[3] > counts[0]);
    try std.testing.expect(counts[3] > counts[1]);
    try std.testing.expect(counts[3] > counts[2]);
}

test "Sampler.sample top_p min_p temperature" {
    var sampler_state = try Sampler.init(std.testing.allocator, 3131, 16);
    defer sampler_state.deinit();

    const logits = [_]f32{ 10.0, 8.0, 4.0, 2.0, 0.5, 0.1 };
    const config = SamplingConfig{
        .strategy = .top_p,
        .top_p = 0.9,
        .temperature = 0.5, // Lower temperature = more peaked
        .min_p = 0.01,
    };

    var counts = [_]usize{0} ** 6;
    for (0..300) |_| {
        const sampled_index = try sampler_state.sample(&logits, config);
        counts[sampled_index] += 1;
    }

    // With low temperature and filtering, first token should dominate
    try std.testing.expect(counts[0] > 100);

    // Last token should be very rare or never sampled
    try std.testing.expect(counts[5] < 10);
}

test "Sampler.sample greedy with repetition" {
    var sampler_state = try Sampler.init(std.testing.allocator, 3232, 16);
    defer sampler_state.deinit();

    // Token 0 has highest logit initially
    var logits = [_]f32{ 10.0, 5.0, 3.0, 2.0 };
    const context_tokens = [_]u32{0}; // Penalize token 0

    const config = SamplingConfig{
        .strategy = .greedy,
        .repetition_penalty = 3.0, // Strong penalty
        .context_tokens = &context_tokens,
    };

    const sampled_index = try sampler_state.sampleMut(&logits, config);

    // With strong penalty on token 0, token 1 should be selected
    try std.testing.expectEqual(@as(usize, 1), sampled_index);
}

test "Sampler.sample top_k all filters" {
    var sampler_state = try Sampler.init(std.testing.allocator, 3333, 16);
    defer sampler_state.deinit();

    var logits = [_]f32{ 10.0, 9.0, 8.0, 7.0, 1.0, 0.5 };
    const context_tokens = [_]u32{ 0, 1 };
    const bias_entries = [_]LogitBiasEntry{
        .{ .token_id = 5, .bias = -50.0 },
    };

    const config = SamplingConfig{
        .strategy = .top_k,
        .top_k = 4,
        .temperature = 0.8,
        .min_p = 0.05,
        .repetition_penalty = 1.5,
        .context_tokens = &context_tokens,
        .logit_bias = &bias_entries,
    };

    var counts = [_]usize{0} ** 6;
    for (0..200) |_| {
        logits = [_]f32{ 10.0, 9.0, 8.0, 7.0, 1.0, 0.5 };
        const sampled_index = try sampler_state.sampleMut(&logits, config);
        counts[sampled_index] += 1;
    }

    // Token 5 should never be sampled (large negative bias)
    try std.testing.expectEqual(@as(usize, 0), counts[5]);

    // Tokens 2 and 3 should be sampled more than 0 and 1 (repetition penalty)
    try std.testing.expect(counts[2] + counts[3] > counts[0] + counts[1]);
}

// ----------------------------------------------------------------------------
// Boundary and stress tests
// ----------------------------------------------------------------------------

test "Sampler.sample stress large vocab" {
    var sampler_state = try Sampler.init(std.testing.allocator, 4040, 10000);
    defer sampler_state.deinit();

    const allocator = std.testing.allocator;
    const logits = try allocator.alloc(f32, 10000);
    defer allocator.free(logits);

    // Initialize with random-ish values
    for (logits, 0..) |*logit, i| {
        logit.* = @as(f32, @floatFromInt(i % 100)) / 100.0;
    }

    const config = SamplingConfig{ .strategy = .top_k, .top_k = 50, .temperature = 1.0 };

    // Should complete without error
    const sampled_index = try sampler_state.sample(logits, config);
    try std.testing.expect(sampled_index < 10000);
}

test "Sampler.sample stress many samples" {
    var sampler_state = try Sampler.init(std.testing.allocator, 4141, 16);
    defer sampler_state.deinit();

    const logits = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const config = SamplingConfig{ .strategy = .top_k, .top_k = 4, .temperature = 1.0 };

    // Run many samples to test stability
    for (0..1000) |_| {
        const sampled_index = try sampler_state.sample(&logits, config);
        try std.testing.expect(sampled_index < 4);
    }
}

test "Sampler.sample boundary infinities" {
    var sampler_state = try Sampler.init(std.testing.allocator, 4242, 16);
    defer sampler_state.deinit();

    const logits = [_]f32{ 1.0, std.math.inf(f32), 2.0, 3.0 };
    const config = SamplingConfig{ .strategy = .greedy };

    // Should pick the infinity (index 1)
    const sampled_index = try sampler_state.sample(&logits, config);
    try std.testing.expectEqual(@as(usize, 1), sampled_index);
}

test "Sampler.sample boundary negative infinity" {
    var sampler_state = try Sampler.init(std.testing.allocator, 4343, 16);
    defer sampler_state.deinit();

    const logits = [_]f32{ 1.0, -std.math.inf(f32), 2.0, 3.0 };
    const config = SamplingConfig{ .strategy = .greedy };

    // Should not pick negative infinity (index 1), should pick max (index 3)
    const sampled_index = try sampler_state.sample(&logits, config);
    try std.testing.expectEqual(@as(usize, 3), sampled_index);
}

test "Sampler.sample boundary zero logits" {
    var sampler_state = try Sampler.init(std.testing.allocator, 4444, 16);
    defer sampler_state.deinit();

    const logits = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    const config = SamplingConfig{ .strategy = .top_k, .top_k = 4, .temperature = 1.0 };

    // Should produce uniform distribution
    var counts = [_]usize{0} ** 4;
    for (0..400) |_| {
        const sampled_index = try sampler_state.sample(&logits, config);
        counts[sampled_index] += 1;
    }

    // Each token should be sampled roughly equally
    for (counts) |count| {
        try std.testing.expect(count > 50);
        try std.testing.expect(count < 150);
    }
}

test "Sampler.sample boundary out-of-bounds" {
    var logits = [_]f32{ 2.0, 2.0, 2.0, 2.0 };
    const context_tokens = [_]u32{ 0, 1, 100, 999 }; // Some out of bounds
    const penalty: f32 = 2.0;

    applyRepetitionPenalty(&logits, &context_tokens, penalty);

    // Only tokens 0 and 1 should be penalized
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), logits[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), logits[1], 0.001);

    // Tokens 2 and 3 should be unchanged
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), logits[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), logits[3], 0.001);
}

// ----------------------------------------------------------------------------
// Internal function tests: partition
// ----------------------------------------------------------------------------

test "Sampler.sample partition equal values" {
    var items = makeIndexValues(.{ .{ 0, 5.0 }, .{ 1, 5.0 }, .{ 2, 5.0 }, .{ 3, 5.0 } });
    const pivot_idx = partition(&items, 0, items.len - 1);
    try std.testing.expect(pivot_idx < items.len);
    for (items) |item| try std.testing.expectEqual(@as(f32, 5.0), item.value);
}

test "Sampler.sample partition two elements" {
    var items = makeIndexValues(.{ .{ 0, 3.0 }, .{ 1, 7.0 } });
    const pivot_idx = partition(&items, 0, 1);
    try std.testing.expect(items[0].value >= items[1].value);
    try std.testing.expect(pivot_idx <= 1);
}

test "Sampler.sample partition sorted descending" {
    var items = makeIndexValues(.{ .{ 0, 10.0 }, .{ 1, 8.0 }, .{ 2, 6.0 }, .{ 3, 4.0 }, .{ 4, 2.0 } });
    const pivot_idx = partition(&items, 0, items.len - 1);
    const pivot_val = items[pivot_idx].value;
    for (items[0..pivot_idx]) |item| try std.testing.expect(item.value >= pivot_val);
    for (items[pivot_idx + 1 ..]) |item| try std.testing.expect(item.value <= pivot_val);
}

test "Sampler.sample partition sorted ascending" {
    var items = makeIndexValues(.{ .{ 0, 2.0 }, .{ 1, 4.0 }, .{ 2, 6.0 }, .{ 3, 8.0 }, .{ 4, 10.0 } });
    const pivot_idx = partition(&items, 0, items.len - 1);
    const pivot_val = items[pivot_idx].value;
    for (items[0..pivot_idx]) |item| try std.testing.expect(item.value >= pivot_val);
    for (items[pivot_idx + 1 ..]) |item| try std.testing.expect(item.value <= pivot_val);
}

test "Sampler.sample partition negative values" {
    var items = makeIndexValues(.{ .{ 0, -2.0 }, .{ 1, -8.0 }, .{ 2, -4.0 }, .{ 3, -1.0 } });
    const pivot_idx = partition(&items, 0, items.len - 1);
    const pivot_val = items[pivot_idx].value;
    for (items[0..pivot_idx]) |item| try std.testing.expect(item.value >= pivot_val);
    for (items[pivot_idx + 1 ..]) |item| try std.testing.expect(item.value <= pivot_val);
}

test "Sampler.sample partition mixed values" {
    var items = makeIndexValues(.{ .{ 0, 5.0 }, .{ 1, -3.0 }, .{ 2, 2.0 }, .{ 3, -7.0 }, .{ 4, 0.0 } });
    const pivot_idx = partition(&items, 0, items.len - 1);
    const pivot_val = items[pivot_idx].value;
    for (items[0..pivot_idx]) |item| try std.testing.expect(item.value >= pivot_val);
    for (items[pivot_idx + 1 ..]) |item| try std.testing.expect(item.value <= pivot_val);
}

test "Sampler.sample partition preserves elements" {
    var items = makeIndexValues(.{ .{ 0, 5.0 }, .{ 1, 10.0 }, .{ 2, 3.0 }, .{ 3, 8.0 } });
    const original_indices = [_]u32{ 0, 1, 2, 3 };
    const original_values = [_]f32{ 5.0, 10.0, 3.0, 8.0 };
    _ = partition(&items, 0, items.len - 1);
    for (original_indices, original_values) |orig_idx, orig_val| {
        var found = false;
        for (items) |item| {
            if (item.index == orig_idx and item.value == orig_val) {
                found = true;
                break;
            }
        }
        try std.testing.expect(found);
    }
}

test "Sampler.sample partition subrange" {
    var items = makeIndexValues(.{ .{ 0, 1.0 }, .{ 1, 9.0 }, .{ 2, 5.0 }, .{ 3, 3.0 }, .{ 4, 7.0 }, .{ 5, 2.0 } });

    // Partition only middle elements (indices 1-4)
    const pivot_idx = partition(&items, 1, 4);

    // Pivot should be within the specified range
    try std.testing.expect(pivot_idx >= 1 and pivot_idx <= 4);

    // First and last elements should be unchanged
    try std.testing.expectEqual(@as(f32, 1.0), items[0].value);
    try std.testing.expectEqual(@as(f32, 2.0), items[5].value);

    // Elements within partitioned range should satisfy invariant
    const pivot_val = items[pivot_idx].value;
    for (items[1..pivot_idx]) |item| {
        try std.testing.expect(item.value >= pivot_val);
    }
    for (items[pivot_idx + 1 .. 5]) |item| {
        try std.testing.expect(item.value <= pivot_val);
    }
}
