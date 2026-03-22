//! Row-wise SDPA primitives for CPU kernels.

const std = @import("std");
const builtin = @import("builtin");
const reduction = @import("reduction.zig");
const softmax = @import("softmax.zig");
const accelerate = @import("accelerate.zig");

// SIMD types for optimized operations
const simd = @import("simd/arch/root.zig");
const VEC_LEN = simd.f32_vec_len;
const F32Vec = simd.F32Vec;

/// Threshold for using Accelerate BLAS (history_len * feature_width)
/// Lower threshold = use BLAS more aggressively
const ACCELERATE_THRESHOLD: usize = 512;

/// Flash Decoding is disabled - Accelerate BLAS is more efficient for our use case
const FLASH_DECODE_THRESHOLD: usize = 999999;

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

    const active_len = history_len - valid_start_index;

    // Use Flash Decoding for longer contexts (online softmax, no score buffer needed)
    // This scales better with context length due to improved cache locality
    if (active_len >= FLASH_DECODE_THRESHOLD and !exact_softmax) {
        computeFlashDecode(
            query_row,
            key_matrix_rows,
            value_matrix_rows,
            output_row,
            valid_start_index,
            history_len,
            feature_width,
            scale,
            extra_sink_score,
        );
        return;
    }

    // Use Accelerate BLAS for medium computations
    if (comptime accelerate.available) {
        if (active_len * feature_width >= ACCELERATE_THRESHOLD) {
            computeMaskedRowAccelerate(
                query_row,
                key_matrix_rows,
                value_matrix_rows,
                score_buffer,
                output_row,
                valid_start_index,
                history_len,
                feature_width,
                scale,
                extra_sink_score,
                exact_softmax,
            );
            return;
        }
    }

    // Fallback: SIMD-optimized per-row computation
    computeMaskedRowSimd(
        query_row,
        key_matrix_rows,
        value_matrix_rows,
        score_buffer,
        output_row,
        valid_start_index,
        history_len,
        feature_width,
        scale,
        extra_sink_score,
        exact_softmax,
    );
}

/// Flash Decoding using online softmax - no score buffer needed.
/// Uses the Flash Attention online softmax algorithm for better cache efficiency.
fn computeFlashDecode(
    query_row: []const f32,
    key_matrix_rows: []const f32,
    value_matrix_rows: []const f32,
    output_row: []f32,
    valid_start_index: usize,
    history_len: usize,
    feature_width: usize,
    scale: f32,
    extra_sink_score: ?f32,
) void {
    @setFloatMode(.optimized);

    // Online softmax state
    var m: f32 = -std.math.inf(f32); // Running max
    var l: f32 = 0; // Running sum of exp(score - m)

    // Initialize output accumulator
    @memset(output_row[0..feature_width], 0);

    // Handle sink token first if present
    if (extra_sink_score) |sink_score| {
        m = sink_score;
        l = 1.0; // exp(sink_score - sink_score) = 1
        // Sink contributes zero value (no v_sink), so acc stays zero
    }

    // Process each KV position with online softmax
    var k_idx: usize = valid_start_index;
    while (k_idx < history_len) : (k_idx += 1) {
        const k_ptr = key_matrix_rows[k_idx * feature_width ..][0..feature_width];
        const v_ptr = value_matrix_rows[k_idx * feature_width ..][0..feature_width];

        // Compute attention score: Q · K * scale
        const score = reduction.dotRow(query_row[0..feature_width], k_ptr) * scale;

        // Online softmax update
        const m_new = @max(m, score);
        const exp_m_diff = std.math.exp(m - m_new);
        const exp_score = std.math.exp(score - m_new);
        const l_new = l * exp_m_diff + exp_score;

        // Rescale existing accumulator and add new contribution
        // acc = acc * (l * exp_m_diff / l_new) + v * (exp_score / l_new)
        const scale_prev = if (l_new == 0) 0 else (l * exp_m_diff / l_new);
        const scale_curr = if (l_new == 0) 0 else (exp_score / l_new);

        // SIMD update of accumulator
        var idx: usize = 0;
        while (idx + VEC_LEN <= feature_width) : (idx += VEC_LEN) {
            const acc_vec: F32Vec = output_row[idx..][0..VEC_LEN].*;
            const v_vec: F32Vec = v_ptr[idx..][0..VEC_LEN].*;
            const scale_prev_vec: F32Vec = @splat(scale_prev);
            const scale_curr_vec: F32Vec = @splat(scale_curr);
            output_row[idx..][0..VEC_LEN].* = @mulAdd(F32Vec, scale_curr_vec, v_vec, acc_vec * scale_prev_vec);
        }
        // Scalar tail
        while (idx < feature_width) : (idx += 1) {
            output_row[idx] = output_row[idx] * scale_prev + v_ptr[idx] * scale_curr;
        }

        m = m_new;
        l = l_new;
    }
}

/// Flash Decoding for GQA using Accelerate BLAS.
/// Batches Q @ K^T for all queries sharing the same KV head.
pub fn computeFlashDecodeGQA(
    queries: []const f32, // [n_queries, feature_width]
    key_matrix_rows: []const f32, // [history_len, feature_width]
    value_matrix_rows: []const f32, // [history_len, feature_width]
    output_rows: []f32, // [n_queries, feature_width]
    n_queries: usize,
    valid_start_index: usize,
    history_len: usize,
    feature_width: usize,
    scale: f32,
    sinks: ?[]const f32, // [n_queries] sink scores or null
    head_offset: usize, // offset into sinks array
) void {
    const active_len = history_len - valid_start_index;
    if (n_queries == 0 or active_len == 0) return;

    // Use batched BLAS when Accelerate is available
    if (comptime accelerate.available) {
        computeFlashDecodeGQA_Accelerate(
            queries,
            key_matrix_rows,
            value_matrix_rows,
            output_rows,
            n_queries,
            valid_start_index,
            history_len,
            feature_width,
            scale,
            sinks,
            head_offset,
        );
        return;
    }

    // Fallback to per-query processing
    for (0..n_queries) |q| {
        const sink_score: ?f32 = if (sinks) |s| s[head_offset + q] else null;
        computeFlashDecode(
            queries[q * feature_width ..][0..feature_width],
            key_matrix_rows,
            value_matrix_rows,
            output_rows[q * feature_width ..][0..feature_width],
            valid_start_index,
            history_len,
            feature_width,
            scale,
            sink_score,
        );
    }
}

/// Accelerate BLAS implementation for GQA batched decode.
fn computeFlashDecodeGQA_Accelerate(
    queries: []const f32,
    key_matrix_rows: []const f32,
    value_matrix_rows: []const f32,
    output_rows: []f32,
    n_queries: usize,
    valid_start_index: usize,
    history_len: usize,
    feature_width: usize,
    scale: f32,
    sinks: ?[]const f32,
    head_offset: usize,
) void {
    const active_len = history_len - valid_start_index;

    // Stack-allocate score buffer for batched queries
    // Max 16 queries * 8K context = 128K floats = 512KB (fits in stack)
    const MAX_SCORES: usize = 16 * 8192;
    var scores_buf: [MAX_SCORES]f32 = undefined;

    if (n_queries * history_len > MAX_SCORES) {
        // Fallback for very large batches
        for (0..n_queries) |q| {
            const sink_score: ?f32 = if (sinks) |s| s[head_offset + q] else null;
            computeFlashDecode(
                queries[q * feature_width ..][0..feature_width],
                key_matrix_rows,
                value_matrix_rows,
                output_rows[q * feature_width ..][0..feature_width],
                valid_start_index,
                history_len,
                feature_width,
                scale,
                sink_score,
            );
        }
        return;
    }

    // Batched Q @ K^T: [n_queries, feature_width] @ [history_len, feature_width]^T
    // = [n_queries, history_len]
    const active_k = key_matrix_rows[valid_start_index * feature_width ..][0 .. active_len * feature_width];
    const scores = scores_buf[0 .. n_queries * active_len];

    accelerate.sgemmTransBScaled(
        queries[0 .. n_queries * feature_width],
        active_k,
        scores,
        n_queries, // M
        active_len, // N
        feature_width, // K
        scale,
    );

    // Per-query optimized softmax and output computation
    for (0..n_queries) |q| {
        const query_scores = scores[q * active_len ..][0..active_len];

        // Find max
        var max_score: f32 = -std.math.inf(f32);
        for (query_scores) |s| {
            max_score = @max(max_score, s);
        }
        if (sinks) |s| {
            max_score = @max(max_score, s[head_offset + q]);
        }

        // Softmax: exp(score - max) / sum
        var sum: f32 = 0;
        for (query_scores) |*s| {
            s.* = std.math.exp(s.* - max_score);
            sum += s.*;
        }
        if (sinks) |s| {
            sum += std.math.exp(s[head_offset + q] - max_score);
        }
        const inv_sum = if (sum > 0) 1.0 / sum else 0;
        for (query_scores) |*s| {
            s.* *= inv_sum;
        }

        // scores @ V: [1, active_len] @ [active_len, feature_width] = [1, feature_width]
        const active_v = value_matrix_rows[valid_start_index * feature_width ..][0 .. active_len * feature_width];
        const output = output_rows[q * feature_width ..][0..feature_width];

        accelerate.sgemm(
            query_scores,
            active_v,
            output,
            1, // M
            feature_width, // N
            active_len, // K
        );
    }
}

/// SIMD-optimized per-row computation (original algorithm with 4x unrolling)
fn computeMaskedRowSimd(
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

/// Accelerate BLAS-based attention computation
fn computeMaskedRowAccelerate(
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
    // Q @ K^T with fused scale: C = scale * A @ B^T
    // [1, feature_width] @ [history_len, feature_width]^T = [1, history_len]
    accelerate.sgemmTransBScaled(
        query_row[0..feature_width],
        key_matrix_rows[0 .. history_len * feature_width],
        score_buffer[0..history_len],
        1, // M = 1 (single query)
        history_len, // N = history_len
        feature_width, // K = feature_width (head_dim)
        scale, // Fused scaling
    );

    // Set invalid positions to -inf and find max
    var max_score: f32 = -std.math.inf(f32);

    for (0..valid_start_index) |i| {
        score_buffer[i] = -std.math.inf(f32);
    }

    // Find max with SIMD (no scaling needed - already fused)
    var idx: usize = valid_start_index;
    var max_vec: F32Vec = @splat(-std.math.inf(f32));

    while (idx + VEC_LEN <= history_len) : (idx += VEC_LEN) {
        const scores: F32Vec = score_buffer[idx..][0..VEC_LEN].*;
        max_vec = @max(max_vec, scores);
    }
    max_score = @reduce(.Max, max_vec);

    // Scalar tail
    while (idx < history_len) : (idx += 1) {
        max_score = @max(max_score, score_buffer[idx]);
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

    // scores @ V: [1, history_len] @ [history_len, feature_width] = [1, feature_width]
    accelerate.sgemm(
        score_buffer[0..history_len],
        value_matrix_rows[0 .. history_len * feature_width],
        output_row[0..feature_width],
        1, // M = 1
        feature_width, // N = feature_width
        history_len, // K = history_len
    );
}

/// Batched attention for multiple query heads sharing the same K/V (GQA).
/// Computes attention for n_queries heads at once using batched BLAS.
pub fn computeMaskedRowBatched(
    queries: []const f32, // [n_queries, feature_width]
    key_matrix_rows: []const f32, // [history_len, feature_width]
    value_matrix_rows: []const f32, // [history_len, feature_width]
    score_buffer: []f32, // [n_queries, history_len] (strided by score_stride)
    output_rows: []f32, // [n_queries, feature_width]
    n_queries: usize,
    valid_start_index: usize,
    history_len: usize,
    feature_width: usize,
    score_stride: usize,
    scale: f32,
    sinks: ?[]const f32, // [n_queries] or null
    head_offset: usize, // offset into sinks array
    exact_softmax: bool,
) void {
    if (comptime !accelerate.available) {
        // Fallback to per-head processing
        for (0..n_queries) |q| {
            const sink_logit: ?f32 = if (sinks) |s| s[head_offset + q] else null;
            computeMaskedRowWeightedSum(
                queries[q * feature_width ..][0..feature_width],
                key_matrix_rows,
                value_matrix_rows,
                score_buffer[q * score_stride ..][0..history_len],
                output_rows[q * feature_width ..][0..feature_width],
                valid_start_index,
                history_len,
                feature_width,
                scale,
                sink_logit,
                exact_softmax,
            );
        }
        return;
    }

    // Batched Q @ K^T: [n_queries, feature_width] @ [history_len, feature_width]^T
    // = [n_queries, history_len]
    // Note: score_buffer may have stride > history_len, so we need temp buffer
    var scores_temp: [16 * 4096]f32 = undefined; // Max 16 heads * 4K context on stack
    const use_temp = score_stride != history_len and n_queries * history_len <= scores_temp.len;

    if (use_temp) {
        accelerate.sgemmTransBScaled(
            queries[0 .. n_queries * feature_width],
            key_matrix_rows[0 .. history_len * feature_width],
            scores_temp[0 .. n_queries * history_len],
            n_queries,
            history_len,
            feature_width,
            scale,
        );
    } else {
        // Fall back to per-row BLAS for strided output
        for (0..n_queries) |q| {
            accelerate.sgemmTransBScaled(
                queries[q * feature_width ..][0..feature_width],
                key_matrix_rows[0 .. history_len * feature_width],
                score_buffer[q * score_stride ..][0..history_len],
                1,
                history_len,
                feature_width,
                scale,
            );
        }
    }

    // Per-head softmax and copy to strided buffer
    for (0..n_queries) |q| {
        const scores = if (use_temp)
            scores_temp[q * history_len ..][0..history_len]
        else
            score_buffer[q * score_stride ..][0..history_len];

        const out_scores = score_buffer[q * score_stride ..][0..history_len];

        // Set invalid positions to -inf and find max
        var max_score: f32 = -std.math.inf(f32);
        for (0..valid_start_index) |i| {
            scores[i] = -std.math.inf(f32);
        }
        for (valid_start_index..history_len) |i| {
            max_score = @max(max_score, scores[i]);
        }

        const sink_logit: ?f32 = if (sinks) |s| s[head_offset + q] else null;
        if (sink_logit) |sink| {
            max_score = @max(max_score, sink);
        }

        softmax.maskedInPlaceWithMax(
            scores,
            valid_start_index,
            history_len,
            sink_logit,
            exact_softmax,
            max_score,
            null,
        );

        // Copy to strided output if using temp
        if (use_temp) {
            @memcpy(out_scores, scores);
        }
    }

    // Batched scores @ V: [n_queries, history_len] @ [history_len, feature_width]
    // = [n_queries, feature_width]
    // Use per-head for now since scores are strided
    for (0..n_queries) |q| {
        accelerate.sgemm(
            score_buffer[q * score_stride ..][0..history_len],
            value_matrix_rows[0 .. history_len * feature_width],
            output_rows[q * feature_width ..][0..feature_width],
            1,
            feature_width,
            history_len,
        );
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
