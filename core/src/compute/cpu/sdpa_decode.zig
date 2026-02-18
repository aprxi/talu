//! Decode-time SDPA primitives for CPU kernels.

const std = @import("std");
const reduction = @import("reduction.zig");
const softmax = @import("softmax.zig");

pub const DecodeHeadConfig = struct {
    start_kv_index: usize,
    kv_sequence_len: usize,
    head_dim: usize,
    scale: f32,
    sink_logit: ?f32 = null,
    exact_softmax: bool,
};

pub const PrefillHeadConfig = struct {
    start_kv_index: usize,
    end_kv_index: usize,
    sequence_len: usize,
    kv_head_idx: usize,
    head_dim: usize,
    kv_total_dim: usize,
    scale: f32,
    sink_logit: ?f32 = null,
    exact_softmax: bool,
    is_causal: bool,
};

/// Compute decode attention for one head against cached K/V and write context.
pub fn decodeHeadScoresAndContext(
    query_head: []const f32,
    k_cache_head: []const f32,
    v_cache_head: []const f32,
    scores_for_head: []f32,
    context_for_head: []f32,
    cfg: DecodeHeadConfig,
) void {
    std.debug.assert(query_head.len >= cfg.head_dim);
    std.debug.assert(scores_for_head.len >= cfg.kv_sequence_len);
    std.debug.assert(context_for_head.len >= cfg.head_dim);
    std.debug.assert(k_cache_head.len >= cfg.kv_sequence_len * cfg.head_dim);
    std.debug.assert(v_cache_head.len >= cfg.kv_sequence_len * cfg.head_dim);
    std.debug.assert(cfg.start_kv_index <= cfg.kv_sequence_len);

    var max_score: f32 = -std.math.inf(f32);
    var key_index: usize = 0;
    while (key_index < cfg.start_kv_index) : (key_index += 1) {
        scores_for_head[key_index] = -std.math.inf(f32);
    }
    while (key_index < cfg.kv_sequence_len) : (key_index += 1) {
        const k_row = k_cache_head[key_index * cfg.head_dim ..][0..cfg.head_dim];
        const dot = reduction.dotRow(query_head[0..cfg.head_dim], k_row) * cfg.scale;
        scores_for_head[key_index] = dot;
        if (dot > max_score) max_score = dot;
    }

    if (cfg.sink_logit) |sink| {
        if (sink > max_score) max_score = sink;
    }
    softmax.maskedInPlaceWithMax(
        scores_for_head,
        cfg.start_kv_index,
        cfg.kv_sequence_len,
        cfg.sink_logit,
        cfg.exact_softmax,
        max_score,
        null,
    );

    @memset(context_for_head[0..cfg.head_dim], 0);
    var kv_index: usize = 0;
    while (kv_index < cfg.kv_sequence_len) : (kv_index += 1) {
        const attn_weight = scores_for_head[kv_index];
        const v_row = v_cache_head[kv_index * cfg.head_dim ..][0..cfg.head_dim];
        reduction.weightedAccumulateRow(context_for_head[0..cfg.head_dim], v_row, attn_weight);
    }
}

/// Compute one prefill attention row for a single (query, head) pair.
pub fn prefillHeadScoresAndContext(
    query_head: []const f32,
    key_values: []const f32,
    value_values: []const f32,
    scores_for_query: []f32,
    context_for_head: []f32,
    cfg: PrefillHeadConfig,
) void {
    std.debug.assert(query_head.len >= cfg.head_dim);
    std.debug.assert(scores_for_query.len >= cfg.sequence_len);
    std.debug.assert(context_for_head.len >= cfg.head_dim);
    std.debug.assert(cfg.start_kv_index <= cfg.end_kv_index and cfg.end_kv_index <= cfg.sequence_len);
    std.debug.assert(key_values.len >= cfg.sequence_len * cfg.kv_total_dim);
    std.debug.assert(value_values.len >= cfg.sequence_len * cfg.kv_total_dim);
    std.debug.assert((cfg.kv_head_idx + 1) * cfg.head_dim <= cfg.kv_total_dim);

    var max_score: f32 = -std.math.inf(f32);

    for (0..cfg.start_kv_index) |key_index| {
        scores_for_query[key_index] = -std.math.inf(f32);
    }
    for (cfg.start_kv_index..cfg.end_kv_index) |key_index| {
        const key_head = key_values[key_index * cfg.kv_total_dim + cfg.kv_head_idx * cfg.head_dim ..][0..cfg.head_dim];
        const dot = reduction.dotRow(query_head[0..cfg.head_dim], key_head) * cfg.scale;
        scores_for_query[key_index] = dot;
        if (dot > max_score) max_score = dot;
    }

    if (cfg.is_causal) {
        for (cfg.end_kv_index..cfg.sequence_len) |key_index| {
            scores_for_query[key_index] = -std.math.inf(f32);
        }
    }

    if (cfg.sink_logit) |sink| {
        if (sink > max_score) max_score = sink;
    }
    softmax.maskedInPlaceWithMax(
        scores_for_query,
        cfg.start_kv_index,
        cfg.end_kv_index,
        cfg.sink_logit,
        cfg.exact_softmax,
        max_score,
        null,
    );

    @memset(context_for_head[0..cfg.head_dim], 0);
    for (0..cfg.end_kv_index) |key_index| {
        const attn_weight = scores_for_query[key_index];
        const value_head = value_values[key_index * cfg.kv_total_dim + cfg.kv_head_idx * cfg.head_dim ..][0..cfg.head_dim];
        reduction.weightedAccumulateRow(context_for_head[0..cfg.head_dim], value_head, attn_weight);
    }
}

test "decodeHeadScoresAndContext computes expected context" {
    const query = [_]f32{ 1.0, 0.0 };
    const k_cache = [_]f32{
        1.0, 0.0,
        0.0, 1.0,
    };
    const v_cache = [_]f32{
        10.0, 20.0,
        30.0, 40.0,
    };
    var scores = [_]f32{ 0.0, 0.0 };
    var context = [_]f32{ 0.0, 0.0 };

    decodeHeadScoresAndContext(
        &query,
        &k_cache,
        &v_cache,
        &scores,
        &context,
        .{
            .start_kv_index = 0,
            .kv_sequence_len = 2,
            .head_dim = 2,
            .scale = 1.0,
            .exact_softmax = false,
        },
    );

    try std.testing.expectApproxEqAbs(@as(f32, 15.379), context[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 25.379), context[1], 0.01);
}

test "prefillHeadScoresAndContext computes causal row" {
    const head_dim: usize = 2;
    const sequence_len: usize = 3;
    const kv_total_dim: usize = 2;
    const query = [_]f32{ 1.0, 0.0 };
    const key_values = [_]f32{
        1.0, 0.0, // k0
        0.0, 1.0, // k1
        1.0, 0.0, // k2
    };
    const value_values = [_]f32{
        10.0, 20.0,
        30.0, 40.0,
        50.0, 60.0,
    };
    var scores = [_]f32{ 0.0, 0.0, 0.0 };
    var ctx = [_]f32{ 0.0, 0.0 };

    prefillHeadScoresAndContext(
        &query,
        &key_values,
        &value_values,
        &scores,
        &ctx,
        .{
            .start_kv_index = 0,
            .end_kv_index = 2,
            .sequence_len = sequence_len,
            .kv_head_idx = 0,
            .head_dim = head_dim,
            .kv_total_dim = kv_total_dim,
            .scale = 1.0,
            .exact_softmax = false,
            .is_causal = true,
        },
    );

    try std.testing.expect(scores[0] > scores[1]);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), scores[2], 1e-6);
    try std.testing.expect(ctx[0] > 10.0 and ctx[0] < 20.0);
}
