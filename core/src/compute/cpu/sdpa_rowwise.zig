//! Row-wise SDPA primitives for CPU kernels.

const std = @import("std");
const reduction = @import("reduction.zig");
const softmax = @import("softmax.zig");

/// Compute one masked SDPA row against a historical key/value matrix.
pub fn computeMaskedRowWeightedSum(
    query_row: []const f32,
    key_matrix_rows: []const f32,
    value_matrix_rows: []const f32,
    score_buffer: []f32,
    output_row: []f32,
    valid_start_index: usize,
    history_len: usize,
    feature_width: usize,
    scale: f32,
    extra_sink_score: ?f32,
    exact_softmax: bool,
) void {
    std.debug.assert(query_row.len >= feature_width);
    std.debug.assert(score_buffer.len >= history_len);
    std.debug.assert(output_row.len >= feature_width);
    std.debug.assert(key_matrix_rows.len >= history_len * feature_width);
    std.debug.assert(value_matrix_rows.len >= history_len * feature_width);
    std.debug.assert(valid_start_index <= history_len);

    var max_score: f32 = -std.math.inf(f32);
    var row_index: usize = 0;
    while (row_index < valid_start_index) : (row_index += 1) {
        score_buffer[row_index] = -std.math.inf(f32);
    }
    while (row_index < history_len) : (row_index += 1) {
        const key_row = key_matrix_rows[row_index * feature_width ..][0..feature_width];
        const dot = reduction.dotRow(query_row[0..feature_width], key_row) * scale;
        score_buffer[row_index] = dot;
        if (dot > max_score) max_score = dot;
    }

    if (extra_sink_score) |sink| {
        if (sink > max_score) max_score = sink;
    }
    softmax.maskedInPlaceWithMax(
        score_buffer,
        valid_start_index,
        history_len,
        extra_sink_score,
        exact_softmax,
        max_score,
        null,
    );

    @memset(output_row[0..feature_width], 0);
    var value_row_index: usize = 0;
    while (value_row_index < history_len) : (value_row_index += 1) {
        const weight = score_buffer[value_row_index];
        const value_row = value_matrix_rows[value_row_index * feature_width ..][0..feature_width];
        reduction.weightedAccumulateRow(output_row[0..feature_width], value_row, weight);
    }
}

/// Compute one bounded SDPA row using an explicit active key window.
pub fn computeBoundedRowWeightedSum(
    query_row: []const f32,
    key_rows: []const f32,
    value_rows: []const f32,
    score_buffer: []f32,
    output_row: []f32,
    active_start_index: usize,
    active_end_index: usize,
    sequence_len: usize,
    source_group_index: usize,
    feature_width: usize,
    source_row_width: usize,
    scale: f32,
    extra_sink_score: ?f32,
    exact_softmax: bool,
    causal_mask: bool,
) void {
    std.debug.assert(query_row.len >= feature_width);
    std.debug.assert(score_buffer.len >= sequence_len);
    std.debug.assert(output_row.len >= feature_width);
    std.debug.assert(active_start_index <= active_end_index and active_end_index <= sequence_len);
    std.debug.assert(key_rows.len >= sequence_len * source_row_width);
    std.debug.assert(value_rows.len >= sequence_len * source_row_width);
    std.debug.assert((source_group_index + 1) * feature_width <= source_row_width);

    var max_score: f32 = -std.math.inf(f32);

    for (0..active_start_index) |row_index| {
        score_buffer[row_index] = -std.math.inf(f32);
    }
    for (active_start_index..active_end_index) |row_index| {
        const key_row = key_rows[row_index * source_row_width + source_group_index * feature_width ..][0..feature_width];
        const dot = reduction.dotRow(query_row[0..feature_width], key_row) * scale;
        score_buffer[row_index] = dot;
        if (dot > max_score) max_score = dot;
    }

    if (causal_mask) {
        for (active_end_index..sequence_len) |row_index| {
            score_buffer[row_index] = -std.math.inf(f32);
        }
    }

    if (extra_sink_score) |sink| {
        if (sink > max_score) max_score = sink;
    }
    softmax.maskedInPlaceWithMax(
        score_buffer,
        active_start_index,
        active_end_index,
        extra_sink_score,
        exact_softmax,
        max_score,
        null,
    );

    @memset(output_row[0..feature_width], 0);
    for (0..active_end_index) |row_index| {
        const weight = score_buffer[row_index];
        const value_row = value_rows[row_index * source_row_width + source_group_index * feature_width ..][0..feature_width];
        reduction.weightedAccumulateRow(output_row[0..feature_width], value_row, weight);
    }
}

test "computeMaskedRowWeightedSum computes expected context" {
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

    computeMaskedRowWeightedSum(
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

test "computeBoundedRowWeightedSum computes causal row" {
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

    computeBoundedRowWeightedSum(
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
