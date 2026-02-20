//! Decode-time SDPA primitives for CPU kernels.

const std = @import("std");
const reduction = @import("reduction.zig");
const softmax = @import("softmax.zig");

/// Compute decode attention for one head against cached K/V and write context.
pub fn decodeHeadScoresAndContext(
    query_head: []const f32,
    k_cache_head: []const f32,
    v_cache_head: []const f32,
    scores_for_head: []f32,
    context_for_head: []f32,
    start_kv_index: usize,
    kv_sequence_len: usize,
    feature_width: usize,
    scale: f32,
    sink_logit: ?f32,
    exact_softmax: bool,
) void {
    std.debug.assert(query_head.len >= feature_width);
    std.debug.assert(scores_for_head.len >= kv_sequence_len);
    std.debug.assert(context_for_head.len >= feature_width);
    std.debug.assert(k_cache_head.len >= kv_sequence_len * feature_width);
    std.debug.assert(v_cache_head.len >= kv_sequence_len * feature_width);
    std.debug.assert(start_kv_index <= kv_sequence_len);

    var max_score: f32 = -std.math.inf(f32);
    var key_index: usize = 0;
    while (key_index < start_kv_index) : (key_index += 1) {
        scores_for_head[key_index] = -std.math.inf(f32);
    }
    while (key_index < kv_sequence_len) : (key_index += 1) {
        const k_row = k_cache_head[key_index * feature_width ..][0..feature_width];
        const dot = reduction.dotRow(query_head[0..feature_width], k_row) * scale;
        scores_for_head[key_index] = dot;
        if (dot > max_score) max_score = dot;
    }

    if (sink_logit) |sink| {
        if (sink > max_score) max_score = sink;
    }
    softmax.maskedInPlaceWithMax(
        scores_for_head,
        start_kv_index,
        kv_sequence_len,
        sink_logit,
        exact_softmax,
        max_score,
        null,
    );

    @memset(context_for_head[0..feature_width], 0);
    var kv_index: usize = 0;
    while (kv_index < kv_sequence_len) : (kv_index += 1) {
        const attn_weight = scores_for_head[kv_index];
        const v_row = v_cache_head[kv_index * feature_width ..][0..feature_width];
        reduction.weightedAccumulateRow(context_for_head[0..feature_width], v_row, attn_weight);
    }
}

/// Compute one prefill attention row for a single (query, head) pair.
pub fn prefillHeadScoresAndContext(
    query_head: []const f32,
    key_values: []const f32,
    value_values: []const f32,
    scores_for_query: []f32,
    context_for_head: []f32,
    start_kv_index: usize,
    end_kv_index: usize,
    sequence_len: usize,
    kv_head_index: usize,
    feature_width: usize,
    kv_total_width: usize,
    scale: f32,
    sink_logit: ?f32,
    exact_softmax: bool,
    causal_mask: bool,
) void {
    std.debug.assert(query_head.len >= feature_width);
    std.debug.assert(scores_for_query.len >= sequence_len);
    std.debug.assert(context_for_head.len >= feature_width);
    std.debug.assert(start_kv_index <= end_kv_index and end_kv_index <= sequence_len);
    std.debug.assert(key_values.len >= sequence_len * kv_total_width);
    std.debug.assert(value_values.len >= sequence_len * kv_total_width);
    std.debug.assert((kv_head_index + 1) * feature_width <= kv_total_width);

    var max_score: f32 = -std.math.inf(f32);

    for (0..start_kv_index) |key_index| {
        scores_for_query[key_index] = -std.math.inf(f32);
    }
    for (start_kv_index..end_kv_index) |key_index| {
        const key_head = key_values[key_index * kv_total_width + kv_head_index * feature_width ..][0..feature_width];
        const dot = reduction.dotRow(query_head[0..feature_width], key_head) * scale;
        scores_for_query[key_index] = dot;
        if (dot > max_score) max_score = dot;
    }

    if (causal_mask) {
        for (end_kv_index..sequence_len) |key_index| {
            scores_for_query[key_index] = -std.math.inf(f32);
        }
    }

    if (sink_logit) |sink| {
        if (sink > max_score) max_score = sink;
    }
    softmax.maskedInPlaceWithMax(
        scores_for_query,
        start_kv_index,
        end_kv_index,
        sink_logit,
        exact_softmax,
        max_score,
        null,
    );

    @memset(context_for_head[0..feature_width], 0);
    for (0..end_kv_index) |key_index| {
        const attn_weight = scores_for_query[key_index];
        const value_head = value_values[key_index * kv_total_width + kv_head_index * feature_width ..][0..feature_width];
        reduction.weightedAccumulateRow(context_for_head[0..feature_width], value_head, attn_weight);
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
        0,
        2,
        2,
        1.0,
        null,
        false,
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
        0,
        2,
        sequence_len,
        0,
        head_dim,
        kv_total_dim,
        1.0,
        null,
        false,
        true,
    );

    try std.testing.expect(scores[0] > scores[1]);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), scores[2], 1e-6);
    try std.testing.expect(ctx[0] > 10.0 and ctx[0] < 20.0);
}
