//! Backend-neutral sampling policy helpers.

const std = @import("std");
const compute = @import("compute_pkg");
const contracts = @import("contracts.zig");

const ops = compute.cpu.sampling_ops;

pub const SamplingConfig = contracts.SamplingConfig;

pub fn validateSamplerConfigBounds(config: *const SamplingConfig) !void {
    if (config.temperature < 0.0) return error.InvalidTemperature;
    if (config.top_p < 0.0 or config.top_p > 1.0) return error.InvalidTopP;
    if (config.min_p < 0.0 or config.min_p > 1.0) return error.InvalidMinP;
}

pub fn hasLogitMutations(config: SamplingConfig) bool {
    return config.repetition_penalty != 1.0 or
        config.presence_penalty != 0.0 or
        config.frequency_penalty != 0.0 or
        config.logit_bias != null;
}

pub fn canUseDirectGreedyCandidate(config: SamplingConfig, candidate_count: usize) bool {
    return candidate_count != 0 and config.strategy == .greedy and !hasLogitMutations(config);
}

pub fn applyLogitMutations(logits: []f32, config: *const SamplingConfig) void {
    const context_tokens = config.context_tokens orelse &.{};
    if (config.repetition_penalty != 1.0 and context_tokens.len > 0) {
        ops.applyIndexPenalty(logits, context_tokens, config.repetition_penalty);
    }
    if (config.logit_bias) |bias_entries| {
        ops.applyIndexBias(logits, bias_entries);
    }
    if ((config.presence_penalty != 0.0 or config.frequency_penalty != 0.0) and context_tokens.len > 0) {
        ops.applyAdditivePenalties(logits, context_tokens, config.presence_penalty, config.frequency_penalty);
    }
}

pub fn applyCandidateLogitMutations(
    candidate_logits: []f32,
    candidate_ids: []const u32,
    config: *const SamplingConfig,
) !void {
    if (candidate_logits.len != candidate_ids.len) return error.InvalidArgument;
    const context_tokens = config.context_tokens orelse &.{};

    if (config.repetition_penalty != 1.0) {
        const penalty = config.repetition_penalty;
        for (context_tokens) |context_id| {
            for (candidate_ids, candidate_logits) |candidate_id, *logit| {
                if (candidate_id == context_id) {
                    logit.* = if (logit.* > 0.0) logit.* / penalty else logit.* * penalty;
                }
            }
        }
    }

    if (config.presence_penalty != 0.0 or config.frequency_penalty != 0.0) {
        for (candidate_ids, candidate_logits) |candidate_id, *logit| {
            var count: f32 = 0.0;
            for (context_tokens) |context_id| {
                if (context_id == candidate_id) count += 1.0;
            }
            if (count > 0.0) {
                logit.* -= config.presence_penalty + config.frequency_penalty * count;
            }
        }
    }

    if (config.logit_bias) |bias_entries| {
        for (bias_entries) |entry| {
            for (candidate_ids, candidate_logits) |candidate_id, *logit| {
                if (candidate_id == entry.token_id) {
                    logit.* += entry.bias;
                }
            }
        }
    }
}

fn hostLogitCandidateBetter(
    logit: f32,
    token_id: u32,
    existing_logit: f32,
    existing_token_id: u32,
) bool {
    return logit > existing_logit or (logit == existing_logit and token_id < existing_token_id);
}

fn insertHostTopKCandidate(
    logits: []f32,
    ids: []u32,
    count: *usize,
    max_count: usize,
    logit: f32,
    token_id: u32,
) void {
    if (max_count == 0) return;
    var insert_at = count.*;
    if (insert_at == max_count and !hostLogitCandidateBetter(logit, token_id, logits[max_count - 1], ids[max_count - 1])) {
        return;
    }
    if (insert_at < max_count) count.* += 1;
    if (insert_at >= max_count) insert_at = max_count - 1;
    while (insert_at > 0 and hostLogitCandidateBetter(logit, token_id, logits[insert_at - 1], ids[insert_at - 1])) {
        logits[insert_at] = logits[insert_at - 1];
        ids[insert_at] = ids[insert_at - 1];
        insert_at -= 1;
    }
    logits[insert_at] = logit;
    ids[insert_at] = token_id;
}

pub fn extractTopKFromHostLogitsRow(
    logits: []const f32,
    top_k: usize,
    candidate_logits_out: []f32,
    candidate_ids_out: []u32,
) !usize {
    if (top_k == 0 or top_k > 256) return error.InvalidArgument;
    if (candidate_logits_out.len < top_k or candidate_ids_out.len < top_k) return error.InvalidArgument;
    if (logits.len == 0 or logits.len > std.math.maxInt(u32)) return error.InvalidArgument;
    const count_limit = @min(top_k, logits.len);
    var count: usize = 0;
    for (logits, 0..) |logit, token_index| {
        insertHostTopKCandidate(
            candidate_logits_out,
            candidate_ids_out,
            &count,
            count_limit,
            logit,
            @intCast(token_index),
        );
    }
    if (top_k == 1 and count == 1) candidate_logits_out[0] = 0.0;
    return count;
}

pub fn isBoundedTopKRoute(config: *const SamplingConfig, top_k_capacity: usize) bool {
    return config.strategy == .top_k and
        config.top_k > 0 and
        config.top_k <= top_k_capacity;
}

pub fn isTopKOrGreedyCandidateRoute(config: *const SamplingConfig, top_k_capacity: usize) bool {
    return switch (config.strategy) {
        .top_k => isBoundedTopKRoute(config, top_k_capacity),
        .greedy => !hasLogitMutations(config.*),
        else => false,
    };
}

pub fn isTopKStreamingWithoutMutations(config: *const SamplingConfig, top_k_capacity: usize) bool {
    return isBoundedTopKRoute(config, top_k_capacity) and
        config.temperature > 0.0 and
        config.top_p >= 0.0 and
        config.top_p <= 1.0 and
        config.min_p >= 0.0 and
        config.min_p <= 1.0 and
        !hasLogitMutations(config.*);
}

pub fn isTopKStreamingWithPenaltyMutations(config: *const SamplingConfig, top_k_capacity: usize) bool {
    return isBoundedTopKRoute(config, top_k_capacity) and
        config.temperature >= 0.0 and
        config.top_p >= 0.0 and
        config.top_p <= 1.0 and
        config.min_p >= 0.0 and
        config.min_p <= 1.0 and
        config.repetition_penalty > 0.0 and
        config.logit_bias == null;
}

test "inference sampling policy validateSamplerConfigBounds and hasLogitMutations contracts" {
    try validateSamplerConfigBounds(&.{});
    try std.testing.expectError(error.InvalidTemperature, validateSamplerConfigBounds(&.{ .temperature = -0.1 }));
    try std.testing.expectError(error.InvalidTopP, validateSamplerConfigBounds(&.{ .top_p = 1.1 }));
    try std.testing.expectError(error.InvalidMinP, validateSamplerConfigBounds(&.{ .min_p = -0.1 }));

    try std.testing.expect(!hasLogitMutations(.{}));
    for ([_]SamplingConfig{
        .{ .repetition_penalty = 1.1 },
        .{ .presence_penalty = 0.1 },
        .{ .frequency_penalty = 0.1 },
        .{ .logit_bias = &.{.{ .token_id = 2, .bias = -1.0 }} },
    }) |config| {
        try std.testing.expect(hasLogitMutations(config));
    }
}

test "inference sampling policy applyLogitMutations and applyCandidateLogitMutations contracts" {
    var full_logits = [_]f32{ 2.0, -2.0, 10.0, 4.0 };
    var candidate_logits = full_logits;
    const candidate_ids = [_]u32{ 0, 1, 2, 3 };
    const context_tokens = [_]u32{ 0, 1, 0 };
    const bias_entries = [_]contracts.LogitBiasEntry{.{ .token_id = 2, .bias = -3.0 }};
    const config = SamplingConfig{
        .repetition_penalty = 2.0,
        .presence_penalty = 1.0,
        .frequency_penalty = 0.5,
        .context_tokens = context_tokens[0..],
        .logit_bias = bias_entries[0..],
    };

    applyLogitMutations(full_logits[0..], &config);
    try applyCandidateLogitMutations(candidate_logits[0..], candidate_ids[0..], &config);
    const expected = [_]f32{ -1.5, -5.5, 7.0, 4.0 };
    for (full_logits, candidate_logits, expected) |full, candidate, want| {
        try std.testing.expectApproxEqAbs(want, full, 1e-6);
        try std.testing.expectApproxEqAbs(want, candidate, 1e-6);
    }
    try std.testing.expectError(error.InvalidArgument, applyCandidateLogitMutations(candidate_logits[0..2], candidate_ids[0..1], &config));
}

test "inference sampling policy extractTopKFromHostLogitsRow returns deterministic candidates" {
    const logits = [_]f32{ 1.0, 8.0, 3.0, 8.0, -2.0, 5.0 };
    var candidate_logits: [3]f32 = undefined;
    var candidate_ids: [3]u32 = undefined;

    const count = try extractTopKFromHostLogitsRow(
        logits[0..],
        3,
        candidate_logits[0..],
        candidate_ids[0..],
    );

    try std.testing.expectEqual(@as(usize, 3), count);
    try std.testing.expectEqualSlices(u32, &.{ 1, 3, 5 }, candidate_ids[0..]);
    try std.testing.expectEqualSlices(f32, &.{ 8.0, 8.0, 5.0 }, candidate_logits[0..]);
}

test "inference sampling policy canUseDirectGreedyCandidate isBoundedTopKRoute isTopKOrGreedyCandidateRoute contracts" {
    try std.testing.expect(canUseDirectGreedyCandidate(.{ .strategy = .greedy }, 1));
    try std.testing.expect(!canUseDirectGreedyCandidate(.{ .strategy = .greedy }, 0));
    try std.testing.expect(!canUseDirectGreedyCandidate(.{ .strategy = .greedy, .presence_penalty = 0.5 }, 1));

    try std.testing.expect(isBoundedTopKRoute(&.{ .strategy = .top_k, .top_k = 64 }, 256));
    try std.testing.expect(!isBoundedTopKRoute(&.{ .strategy = .top_k, .top_k = 257 }, 256));

    try std.testing.expect(isTopKOrGreedyCandidateRoute(&.{ .strategy = .top_k, .top_k = 64 }, 256));
    try std.testing.expect(isTopKOrGreedyCandidateRoute(&.{ .strategy = .greedy }, 256));
    try std.testing.expect(!isTopKOrGreedyCandidateRoute(&.{ .strategy = .greedy, .logit_bias = &.{.{ .token_id = 42, .bias = 1.0 }} }, 256));
}

test "inference sampling policy isTopKStreamingWithoutMutations and isTopKStreamingWithPenaltyMutations contracts" {
    const topk = SamplingConfig{ .strategy = .top_k, .top_k = 64 };
    try std.testing.expect(isTopKStreamingWithoutMutations(&topk, 256));
    try std.testing.expect(!isTopKStreamingWithoutMutations(&.{ .strategy = .top_k, .top_k = 64, .repetition_penalty = 1.1 }, 256));

    var penalized = topk;
    penalized.temperature = 0.0;
    penalized.repetition_penalty = 1.15;
    penalized.presence_penalty = 0.75;
    penalized.frequency_penalty = 0.25;
    try std.testing.expect(isTopKStreamingWithPenaltyMutations(&penalized, 256));
    try std.testing.expect(!isTopKStreamingWithPenaltyMutations(&.{ .strategy = .top_k, .top_k = 64, .logit_bias = &.{.{ .token_id = 42, .bias = 1.5 }} }, 256));
}
