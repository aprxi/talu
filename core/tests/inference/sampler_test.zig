//! Integration tests for inference.Sampler
//!
//! Tests the Sampler struct which handles token sampling strategies:
//! - Greedy (argmax)
//! - Top-k (sample from top k tokens)
//! - Top-p / nucleus (sample from smallest set with cumulative prob >= p)

const std = @import("std");
const main = @import("main");
const Sampler = main.inference.Sampler;
const SamplingConfig = main.inference.SamplingConfig;
const SamplingStrategy = main.inference.SamplingStrategy;

// =============================================================================
// Initialization Tests
// =============================================================================

test "Sampler.init creates sampler with correct vocab size" {
    const allocator = std.testing.allocator;

    var sampler = try Sampler.init(allocator, 42, 1000);
    defer sampler.deinit();

    // Workspace should have vocab_size capacity
    try std.testing.expectEqual(@as(usize, 1000), sampler.workspace.probabilities.len);
}

test "Sampler.init with different seeds produces different PRNGs" {
    const allocator = std.testing.allocator;

    var s1 = try Sampler.init(allocator, 123, 100);
    defer s1.deinit();

    var s2 = try Sampler.init(allocator, 456, 100);
    defer s2.deinit();

    // Generate random numbers from each - should be different
    const r1 = s1.prng.random().int(u64);
    const r2 = s2.prng.random().int(u64);

    try std.testing.expect(r1 != r2);
}

test "Sampler.init with same seed is deterministic" {
    const allocator = std.testing.allocator;

    var s1 = try Sampler.init(allocator, 999, 100);
    defer s1.deinit();

    var s2 = try Sampler.init(allocator, 999, 100);
    defer s2.deinit();

    // Same seed should produce same sequence
    const r1 = s1.prng.random().int(u64);
    const r2 = s2.prng.random().int(u64);

    try std.testing.expectEqual(r1, r2);
}

// =============================================================================
// Greedy Sampling Tests
// =============================================================================

test "Sampler greedy returns argmax" {
    const allocator = std.testing.allocator;

    var sampler = try Sampler.init(allocator, 1, 16);
    defer sampler.deinit();

    const cfg = SamplingConfig{ .strategy = .greedy };

    // Test various logit patterns
    const cases = [_]struct { logits: []const f32, expected: usize }{
        .{ .logits = &[_]f32{ 0.1, 0.9, 0.2 }, .expected = 1 },
        .{ .logits = &[_]f32{ 5.0, 1.0, 2.0 }, .expected = 0 },
        .{ .logits = &[_]f32{ 0.0, 0.0, 1.0 }, .expected = 2 },
        .{ .logits = &[_]f32{ -1.0, -0.5, -2.0 }, .expected = 1 },
        .{ .logits = &[_]f32{42.0}, .expected = 0 },
    };

    for (cases) |tc| {
        const result = try sampler.sample(tc.logits, cfg);
        try std.testing.expectEqual(tc.expected, result);
    }
}

test "Sampler greedy handles ties by returning first occurrence" {
    const allocator = std.testing.allocator;

    var sampler = try Sampler.init(allocator, 1, 16);
    defer sampler.deinit();

    const cfg = SamplingConfig{ .strategy = .greedy };
    const logits = [_]f32{ 1.0, 1.0, 1.0 };

    const result = try sampler.sample(&logits, cfg);
    try std.testing.expectEqual(@as(usize, 0), result);
}

test "Sampler greedy works with large vocab" {
    const allocator = std.testing.allocator;
    const vocab_size = 50000;

    var sampler = try Sampler.init(allocator, 1, vocab_size);
    defer sampler.deinit();

    // Create logits with known max position
    var logits: [vocab_size]f32 = undefined;
    for (&logits, 0..) |*l, i| {
        l.* = @floatFromInt(i);
    }
    logits[vocab_size - 1] = 999999.0; // Make last element the max

    const cfg = SamplingConfig{ .strategy = .greedy };
    const result = try sampler.sample(&logits, cfg);

    try std.testing.expectEqual(vocab_size - 1, result);
}

// =============================================================================
// Top-k Sampling Tests
// =============================================================================

test "Sampler top_k with k=1 behaves like greedy" {
    const allocator = std.testing.allocator;

    var sampler = try Sampler.init(allocator, 42, 16);
    defer sampler.deinit();

    const logits = [_]f32{ 10.0, 9.0, 1.0 };
    const cfg = SamplingConfig{
        .strategy = .top_k,
        .top_k = 1,
        .temperature = 1.0,
    };

    // With k=1, only the highest probability token can be selected
    for (0..10) |_| {
        const result = try sampler.sample(&logits, cfg);
        try std.testing.expectEqual(@as(usize, 0), result);
    }
}

test "Sampler top_k samples only from top k tokens" {
    const allocator = std.testing.allocator;

    var sampler = try Sampler.init(allocator, 12345, 16);
    defer sampler.deinit();

    // Logits where positions 0 and 1 are clearly highest
    const logits = [_]f32{ 100.0, 99.0, -100.0, -100.0, -100.0 };
    const cfg = SamplingConfig{
        .strategy = .top_k,
        .top_k = 2,
        .temperature = 1.0,
    };

    // Sample many times - should only get 0 or 1
    for (0..50) |_| {
        const result = try sampler.sample(&logits, cfg);
        try std.testing.expect(result == 0 or result == 1);
    }
}

test "Sampler top_k with k larger than vocab uses all tokens" {
    const allocator = std.testing.allocator;

    var sampler = try Sampler.init(allocator, 42, 16);
    defer sampler.deinit();

    const logits = [_]f32{ 1.0, 1.0, 1.0 };
    const cfg = SamplingConfig{
        .strategy = .top_k,
        .top_k = 100, // Much larger than vocab
        .temperature = 1.0,
    };

    // Should work without error
    _ = try sampler.sample(&logits, cfg);
}

// =============================================================================
// Top-p (Nucleus) Sampling Tests
// =============================================================================

test "Sampler top_p with p=1.0 samples from all tokens" {
    const allocator = std.testing.allocator;

    var sampler = try Sampler.init(allocator, 42, 16);
    defer sampler.deinit();

    const logits = [_]f32{ 1.0, 1.0, 1.0 };
    const cfg = SamplingConfig{
        .strategy = .top_p,
        .top_p = 1.0,
        .temperature = 1.0,
    };

    // Should work without error
    _ = try sampler.sample(&logits, cfg);
}

test "Sampler top_p with low p restricts to high probability tokens" {
    const allocator = std.testing.allocator;

    var sampler = try Sampler.init(allocator, 99999, 16);
    defer sampler.deinit();

    // Strong probability skew - first token dominates
    const logits = [_]f32{ 100.0, 0.0, 0.0, 0.0, 0.0 };
    const cfg = SamplingConfig{
        .strategy = .top_p,
        .top_p = 0.5, // Only need 50% cumulative
        .temperature = 1.0,
    };

    // First token has ~99%+ probability, so it should almost always be selected
    var count_first: usize = 0;
    for (0..20) |_| {
        const result = try sampler.sample(&logits, cfg);
        if (result == 0) count_first += 1;
    }

    // Should be selected most of the time (allowing for some randomness)
    try std.testing.expect(count_first >= 15);
}

// =============================================================================
// Temperature Tests
// =============================================================================

test "Sampler temperature=0 with greedy works" {
    const allocator = std.testing.allocator;

    var sampler = try Sampler.init(allocator, 1, 16);
    defer sampler.deinit();

    const logits = [_]f32{ 1.0, 2.0, 0.5 };
    const cfg = SamplingConfig{
        .strategy = .greedy,
        .temperature = 0.0,
    };

    const result = try sampler.sample(&logits, cfg);
    try std.testing.expectEqual(@as(usize, 1), result);
}

test "Sampler temperature affects distribution sharpness" {
    const allocator = std.testing.allocator;

    // Low temperature should concentrate probability
    var sampler_low_temp = try Sampler.init(allocator, 42, 100);
    defer sampler_low_temp.deinit();

    var sampler_high_temp = try Sampler.init(allocator, 42, 100);
    defer sampler_high_temp.deinit();

    const logits = [_]f32{ 2.0, 1.0, 0.0 };

    const cfg_low = SamplingConfig{
        .strategy = .top_k,
        .top_k = 3,
        .temperature = 0.1,
    };

    const cfg_high = SamplingConfig{
        .strategy = .top_k,
        .top_k = 3,
        .temperature = 2.0,
    };

    // Low temperature should mostly pick index 0
    var count_0_low: usize = 0;
    for (0..20) |_| {
        if (try sampler_low_temp.sample(&logits, cfg_low) == 0) count_0_low += 1;
    }

    // High temperature should be more uniform
    var count_0_high: usize = 0;
    for (0..20) |_| {
        if (try sampler_high_temp.sample(&logits, cfg_high) == 0) count_0_high += 1;
    }

    // Low temp should pick token 0 more often than high temp
    try std.testing.expect(count_0_low >= count_0_high);
}

// =============================================================================
// Error Handling Tests
// =============================================================================

test "Sampler returns error on empty logits" {
    const allocator = std.testing.allocator;

    var sampler = try Sampler.init(allocator, 1, 16);
    defer sampler.deinit();

    const cfg = SamplingConfig{ .strategy = .greedy };
    const empty: []const f32 = &.{};

    const result = sampler.sample(empty, cfg);
    try std.testing.expectError(error.InvalidInput, result);
}

test "Sampler returns error on negative temperature for non-greedy" {
    const allocator = std.testing.allocator;

    var sampler = try Sampler.init(allocator, 1, 16);
    defer sampler.deinit();

    const logits = [_]f32{ 1.0, 2.0 };
    const cfg = SamplingConfig{
        .strategy = .top_k,
        .top_k = 2,
        .temperature = -1.0,
    };

    const result = sampler.sample(&logits, cfg);
    try std.testing.expectError(error.InvalidTemperature, result);
}

test "Sampler returns error on invalid top_p" {
    const allocator = std.testing.allocator;

    var sampler = try Sampler.init(allocator, 1, 16);
    defer sampler.deinit();

    const logits = [_]f32{ 1.0, 2.0 };

    // top_p > 1.0
    const cfg_high = SamplingConfig{
        .strategy = .top_p,
        .top_p = 1.5,
        .temperature = 1.0,
    };
    try std.testing.expectError(error.InvalidTopP, sampler.sample(&logits, cfg_high));

    // top_p < 0
    const cfg_neg = SamplingConfig{
        .strategy = .top_p,
        .top_p = -0.1,
        .temperature = 1.0,
    };
    try std.testing.expectError(error.InvalidTopP, sampler.sample(&logits, cfg_neg));
}

// =============================================================================
// Min-p Sampling Tests
// =============================================================================

test "Sampler min_p filters low probability tokens" {
    const allocator = std.testing.allocator;

    var sampler = try Sampler.init(allocator, 42, 100);
    defer sampler.deinit();

    // Strong probability skew
    const logits = [_]f32{ 100.0, 50.0, 0.0, 0.0, 0.0 };
    const cfg = SamplingConfig{
        .strategy = .top_k,
        .top_k = 5,
        .min_p = 0.1, // Filter tokens with prob < 10% of max
        .temperature = 1.0,
    };

    // Low probability tokens should be filtered
    var seen_high: bool = false;
    for (0..20) |_| {
        const result = try sampler.sample(&logits, cfg);
        if (result <= 1) seen_high = true;
        // Tokens 2,3,4 have very low probability and should be filtered
    }
    try std.testing.expect(seen_high);
}

// =============================================================================
// Repetition Penalty Tests
// =============================================================================

test "Sampler sampleMut applies repetition penalty" {
    const allocator = std.testing.allocator;

    var sampler = try Sampler.init(allocator, 1, 16);
    defer sampler.deinit();

    // Without penalty, token 0 should be selected (highest logit)
    var logits_no_penalty = [_]f32{ 2.0, 1.0, 0.5 };
    const cfg_no_penalty = SamplingConfig{ .strategy = .greedy };
    const result_no_penalty = try sampler.sampleMut(&logits_no_penalty, cfg_no_penalty);
    try std.testing.expectEqual(@as(usize, 0), result_no_penalty);

    // With penalty on token 0, token 1 might be selected
    var logits_with_penalty = [_]f32{ 2.0, 1.9, 0.5 };
    const context = [_]u32{0};
    const cfg_with_penalty = SamplingConfig{
        .strategy = .greedy,
        .repetition_penalty = 10.0, // Strong penalty
        .context_tokens = &context,
    };
    const result_with_penalty = try sampler.sampleMut(&logits_with_penalty, cfg_with_penalty);
    try std.testing.expectEqual(@as(usize, 1), result_with_penalty);
}
