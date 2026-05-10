//! Backend-neutral sampling request configuration.

const std = @import("std");

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
    /// Nucleus threshold. 1.0 = disabled (default).
    top_p: f32 = 1.0,
    /// Minimum probability threshold (min_p sampling).
    /// Tokens with probability < min_p * max_prob are excluded.
    /// 0.0 = disabled (default).
    min_p: f32 = 0.0,
    /// Repetition penalty applied to tokens in the context.
    /// 1.0 = no penalty (default), >1.0 = discourage repetition.
    repetition_penalty: f32 = 1.0,
    /// Additive presence penalty. Subtracts this value from logits of tokens
    /// that appear in context_tokens (once per unique token).
    /// 0.0 = disabled (default). Typical range: [-2.0, 2.0].
    presence_penalty: f32 = 0.0,
    /// Additive frequency penalty. Subtracts penalty * count from logits
    /// where count is the number of times the token appears in context_tokens.
    /// 0.0 = disabled (default). Typical range: [-2.0, 2.0].
    frequency_penalty: f32 = 0.0,
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

test "SamplingConfig defaults describe greedy sampling" {
    const config = SamplingConfig{};
    try std.testing.expectEqual(SamplingStrategy.greedy, config.strategy);
    try std.testing.expectEqual(@as(f32, 1.0), config.temperature);
    try std.testing.expectEqual(@as(usize, 1), config.top_k);
    try std.testing.expectEqual(@as(f32, 1.0), config.top_p);
    try std.testing.expectEqual(@as(u64, 0), config.seed);
}

test "LogitBiasEntry stores token bias contract" {
    const entry = LogitBiasEntry{ .token_id = 17, .bias = -2.5 };
    try std.testing.expectEqual(@as(u32, 17), entry.token_id);
    try std.testing.expectEqual(@as(f32, -2.5), entry.bias);
}
