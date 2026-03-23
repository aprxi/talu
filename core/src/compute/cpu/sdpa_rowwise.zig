//! Row-wise SDPA primitives for CPU kernels.

const std = @import("std");
const builtin = @import("builtin");
const reduction = @import("reduction.zig");
const softmax = @import("softmax.zig");
const accelerate = @import("accelerate.zig");
const metal_accel = @import("metal_accel.zig");

// SIMD types for optimized operations
const simd = @import("simd/arch/root.zig");
const VEC_LEN = simd.f32_vec_len;
const F32Vec = simd.F32Vec;

/// Threshold for using Accelerate BLAS (history_len * feature_width)
/// Lower threshold = use BLAS more aggressively
const ACCELERATE_THRESHOLD: usize = 512;

/// Threshold for using Metal acceleration (history_len).
/// Below this, CPU BLAS is faster due to Metal dispatch overhead.
const METAL_THRESHOLD: usize = 256;

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

    // Try Metal for large contexts, fall back to Accelerate BLAS
    const use_metal = active_len >= METAL_THRESHOLD and metal_accel.matmulTransBScaled(
        queries[0 .. n_queries * feature_width],
        active_k,
        scores,
        n_queries, // M
        active_len, // N
        feature_width, // K
        scale,
    );

    if (!use_metal) {
        accelerate.sgemmTransBScaled(
            queries[0 .. n_queries * feature_width],
            active_k,
            scores,
            n_queries, // M
            active_len, // N
            feature_width, // K
            scale,
        );
    }

    // Per-query softmax (must be done before batched matmul)
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
    }

    // Batched scores @ V: [n_queries, active_len] @ [active_len, feature_width] = [n_queries, feature_width]
    const active_v = value_matrix_rows[valid_start_index * feature_width ..][0 .. active_len * feature_width];
    accelerate.sgemm(
        scores,
        active_v,
        output_rows[0 .. n_queries * feature_width],
        n_queries, // M - batch all queries
        feature_width, // N
        active_len, // K
    );
}

/// Flash Decoding GQA for K-only INT8 (V stays f32).
/// Uses K-reuse optimization: load each K row once, compute for all Q heads.
/// Uses SDOT for Q @ K^T, batched BLAS for scores @ V (no V dequant needed).
pub fn computeFlashDecodeGQA_I8(
    queries: []const f32, // [n_queries, feature_width]
    key_matrix_rows_i8: []const i8, // [history_len, feature_width] as INT8
    key_scales: []const f32, // [history_len]
    value_matrix_rows: []const f32, // [history_len, feature_width] as f32
    output_rows: []f32, // [n_queries, feature_width]
    n_queries: usize,
    valid_start_index: usize,
    history_len: usize,
    feature_width: usize,
    scale: f32,
    sinks: ?[]const f32,
    head_offset: usize,
) void {
    const active_len = history_len - valid_start_index;
    if (n_queries == 0 or active_len == 0) return;

    // Maximum supported dimensions for stack allocation
    const MAX_HEAD_DIM: usize = 256;
    const MAX_CONTEXT: usize = 8192;
    const MAX_QUERIES: usize = 16;

    std.debug.assert(feature_width <= MAX_HEAD_DIM);
    std.debug.assert(active_len <= MAX_CONTEXT);
    std.debug.assert(n_queries <= MAX_QUERIES);

    // Score buffer for all queries
    var scores_buf: [MAX_QUERIES * MAX_CONTEXT]f32 = undefined;
    const scores = scores_buf[0 .. n_queries * active_len];

    // Active K/V slices
    const active_k_i8 = key_matrix_rows_i8[valid_start_index * feature_width ..][0 .. active_len * feature_width];
    const active_k_scales = key_scales[valid_start_index..history_len];
    const active_v = value_matrix_rows[valid_start_index * feature_width ..][0 .. active_len * feature_width];

    // Step 1: Quantize ALL Q heads to i8 first (K-reuse optimization)
    var q_i8_all: [MAX_QUERIES][MAX_HEAD_DIM]i8 = undefined;
    var q_scales_all: [MAX_QUERIES]f32 = undefined;

    for (0..n_queries) |q| {
        const query = queries[q * feature_width ..][0..feature_width];
        q_scales_all[q] = quantizeToI8(query, q_i8_all[q][0..feature_width]);
    }

    // Step 2: K-reuse - load each K row once, compute dot for ALL Q heads
    // This maximizes memory bandwidth utilization
    for (0..active_len) |k_idx| {
        const k_row_i8 = active_k_i8[k_idx * feature_width ..][0..feature_width];
        const k_scale = active_k_scales[k_idx];

        for (0..n_queries) |q| {
            const q_i8 = q_i8_all[q][0..feature_width];
            const dot = dotRowI8xI8(q_i8, k_row_i8, q_scales_all[q], k_scale) * scale;
            scores[q * active_len + k_idx] = dot;
        }
    }

    // Step 3: Per-query softmax
    for (0..n_queries) |q| {
        const query_scores = scores[q * active_len ..][0..active_len];
        var max_score: f32 = -std.math.inf(f32);
        for (query_scores) |s| max_score = @max(max_score, s);
        if (sinks) |s| max_score = @max(max_score, s[head_offset + q]);

        var sum: f32 = 0;
        for (query_scores) |*s| {
            s.* = std.math.exp(s.* - max_score);
            sum += s.*;
        }
        if (sinks) |s| sum += std.math.exp(s[head_offset + q] - max_score);
        const inv_sum = if (sum > 0) 1.0 / sum else 0;
        for (query_scores) |*s| s.* *= inv_sum;
    }

    // Step 4: Batched BLAS for scores @ V (V is already f32, no dequant needed!)
    accelerate.sgemmScaledStrided(
        scores,
        active_v,
        output_rows,
        n_queries, // M = all queries at once
        feature_width, // N
        active_len, // K
        active_len, // lda = stride between query rows in scores
        1.0, // alpha
        0.0, // beta: overwrite
    );
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

// =============================================================================
// INT8 Quantized KV Cache Support
// =============================================================================

/// Fast SDOT-style i8×i8→i32 dot product.
/// Query is pre-quantized to i8 for maximum performance.
/// Uses direct vector widening for optimal NEON codegen.
inline fn dotRowI8xI8(query_i8: []const i8, key_i8: []const i8, q_scale: f32, key_scale: f32) f32 {
    std.debug.assert(query_i8.len == key_i8.len);
    const len = query_i8.len;

    var acc: i32 = 0;
    var i: usize = 0;

    // SIMD: process 16 i8×i8 → accumulate to i32
    // Direct vector widening generates efficient NEON instructions
    while (i + 16 <= len) : (i += 16) {
        const qv: @Vector(16, i8) = query_i8[i..][0..16].*;
        const kv: @Vector(16, i8) = key_i8[i..][0..16].*;
        // Widen to i16, multiply, widen to i32, reduce
        acc += @reduce(.Add, @as(@Vector(16, i32), @as(@Vector(16, i16), qv) * @as(@Vector(16, i16), kv)));
    }

    // Scalar tail
    while (i < len) : (i += 1) {
        acc += @as(i32, query_i8[i]) * @as(i32, key_i8[i]);
    }

    // Apply both scales at the end
    return @as(f32, @floatFromInt(acc)) * q_scale * key_scale;
}

/// Quantize f32 vector to i8 with per-vector scaling.
/// Returns the scale factor used. SIMD optimized.
inline fn quantizeToI8(src: []const f32, dst: []i8) f32 {
    std.debug.assert(src.len == dst.len);
    const len = src.len;

    // Find max absolute value using SIMD
    var max_abs: f32 = 0;
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        const v: @Vector(4, f32) = src[i..][0..4].*;
        const abs_v = @abs(v);
        max_abs = @max(max_abs, @reduce(.Max, abs_v));
    }
    while (i < len) : (i += 1) {
        max_abs = @max(max_abs, @abs(src[i]));
    }

    // Compute scale (i8 range: -127 to 127, avoiding -128 for symmetry)
    const scale = if (max_abs > 0) max_abs / 127.0 else 1.0;
    const inv_scale = if (max_abs > 0) 127.0 / max_abs else 1.0;

    // Quantize using SIMD
    const inv_scale_vec: @Vector(4, f32) = @splat(inv_scale);
    const min_val: @Vector(4, f32) = @splat(-127.0);
    const max_val: @Vector(4, f32) = @splat(127.0);

    i = 0;
    while (i + 4 <= len) : (i += 4) {
        const v: @Vector(4, f32) = src[i..][0..4].*;
        const scaled = v * inv_scale_vec;
        const clamped = @min(max_val, @max(min_val, scaled));
        const rounded: @Vector(4, i32) = @intFromFloat(@round(clamped));
        dst[i] = @truncate(rounded[0]);
        dst[i + 1] = @truncate(rounded[1]);
        dst[i + 2] = @truncate(rounded[2]);
        dst[i + 3] = @truncate(rounded[3]);
    }
    while (i < len) : (i += 1) {
        const scaled = src[i] * inv_scale;
        dst[i] = @intFromFloat(@round(@max(-127.0, @min(127.0, scaled))));
    }

    return scale;
}

/// Compute dot product between f32 query and i8 key with inline dequantization.
/// Uses SIMD to process 16 i8 values at a time.
/// NOTE: This is the legacy slow path. Use dotRowI8xI8 with pre-quantized Q for 2-6x speedup.
fn dotRowI8(query: []const f32, key_i8: []const i8, key_scale: f32) f32 {
    std.debug.assert(query.len == key_i8.len);
    const len = query.len;

    var sum: f32 = 0;
    var i: usize = 0;

    // SIMD: process 16 i8 at a time, accumulate in f32
    while (i + 16 <= len) : (i += 16) {
        const i8_vec: @Vector(16, i8) = key_i8[i..][0..16].*;

        // Process in 4 groups of 4 to convert i8 → f32
        // Group 0
        const i8_0: @Vector(4, i8) = .{ i8_vec[0], i8_vec[1], i8_vec[2], i8_vec[3] };
        const f32_0: @Vector(4, f32) = @floatFromInt(@as(@Vector(4, i32), i8_0));
        const q_0: @Vector(4, f32) = query[i..][0..4].*;
        sum += @reduce(.Add, q_0 * f32_0);

        // Group 1
        const i8_1: @Vector(4, i8) = .{ i8_vec[4], i8_vec[5], i8_vec[6], i8_vec[7] };
        const f32_1: @Vector(4, f32) = @floatFromInt(@as(@Vector(4, i32), i8_1));
        const q_1: @Vector(4, f32) = query[i + 4 ..][0..4].*;
        sum += @reduce(.Add, q_1 * f32_1);

        // Group 2
        const i8_2: @Vector(4, i8) = .{ i8_vec[8], i8_vec[9], i8_vec[10], i8_vec[11] };
        const f32_2: @Vector(4, f32) = @floatFromInt(@as(@Vector(4, i32), i8_2));
        const q_2: @Vector(4, f32) = query[i + 8 ..][0..4].*;
        sum += @reduce(.Add, q_2 * f32_2);

        // Group 3
        const i8_3: @Vector(4, i8) = .{ i8_vec[12], i8_vec[13], i8_vec[14], i8_vec[15] };
        const f32_3: @Vector(4, f32) = @floatFromInt(@as(@Vector(4, i32), i8_3));
        const q_3: @Vector(4, f32) = query[i + 12 ..][0..4].*;
        sum += @reduce(.Add, q_3 * f32_3);
    }

    // Scalar tail
    while (i < len) : (i += 1) {
        sum += query[i] * @as(f32, @floatFromInt(key_i8[i]));
    }

    return sum * key_scale;
}

/// Weighted accumulate with inline i8 dequantization.
/// output += weight * (value_i8 * scale)
fn weightedAccumulateRowI8(output: []f32, value_i8: []const i8, value_scale: f32, weight: f32) void {
    std.debug.assert(output.len == value_i8.len);
    const len = output.len;
    const combined_scale = weight * value_scale;
    const scale_vec: @Vector(4, f32) = @splat(combined_scale);

    var i: usize = 0;

    // SIMD: process 16 i8 at a time
    while (i + 16 <= len) : (i += 16) {
        const i8_vec: @Vector(16, i8) = value_i8[i..][0..16].*;

        // Group 0
        const i8_0: @Vector(4, i8) = .{ i8_vec[0], i8_vec[1], i8_vec[2], i8_vec[3] };
        const f32_0: @Vector(4, f32) = @floatFromInt(@as(@Vector(4, i32), i8_0));
        const out_0: @Vector(4, f32) = output[i..][0..4].*;
        output[i..][0..4].* = @mulAdd(@Vector(4, f32), f32_0, scale_vec, out_0);

        // Group 1
        const i8_1: @Vector(4, i8) = .{ i8_vec[4], i8_vec[5], i8_vec[6], i8_vec[7] };
        const f32_1: @Vector(4, f32) = @floatFromInt(@as(@Vector(4, i32), i8_1));
        const out_1: @Vector(4, f32) = output[i + 4 ..][0..4].*;
        output[i + 4 ..][0..4].* = @mulAdd(@Vector(4, f32), f32_1, scale_vec, out_1);

        // Group 2
        const i8_2: @Vector(4, i8) = .{ i8_vec[8], i8_vec[9], i8_vec[10], i8_vec[11] };
        const f32_2: @Vector(4, f32) = @floatFromInt(@as(@Vector(4, i32), i8_2));
        const out_2: @Vector(4, f32) = output[i + 8 ..][0..4].*;
        output[i + 8 ..][0..4].* = @mulAdd(@Vector(4, f32), f32_2, scale_vec, out_2);

        // Group 3
        const i8_3: @Vector(4, i8) = .{ i8_vec[12], i8_vec[13], i8_vec[14], i8_vec[15] };
        const f32_3: @Vector(4, f32) = @floatFromInt(@as(@Vector(4, i32), i8_3));
        const out_3: @Vector(4, f32) = output[i + 12 ..][0..4].*;
        output[i + 12 ..][0..4].* = @mulAdd(@Vector(4, f32), f32_3, scale_vec, out_3);
    }

    // Scalar tail
    while (i < len) : (i += 1) {
        output[i] += @as(f32, @floatFromInt(value_i8[i])) * combined_scale;
    }
}

/// Compute masked SDPA with K-only INT8 (V stays f32).
/// Uses fast SDOT-style i8×i8 dot product for Q @ K^T (2-6x faster than f32×i8).
/// V is already f32, so we use BLAS for scores @ V (no dequant overhead).
pub fn computeMaskedRowWeightedSumI8(
    query_row: []const f32,
    key_matrix_rows_i8: []const i8,
    key_scales: []const f32, // [history_len] scales
    value_matrix_rows: []const f32, // V is f32
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
    std.debug.assert(key_matrix_rows_i8.len >= history_len * feature_width);
    std.debug.assert(value_matrix_rows.len >= history_len * feature_width);
    std.debug.assert(key_scales.len >= history_len);
    std.debug.assert(valid_start_index <= history_len);

    // Maximum supported head dimension (stack buffer for quantized Q)
    const MAX_HEAD_DIM = 256;
    std.debug.assert(feature_width <= MAX_HEAD_DIM);

    // Quantize Q to i8 once - this is fast (feature_width is small, e.g., 128)
    var query_i8_buf: [MAX_HEAD_DIM]i8 = undefined;
    const query_i8 = query_i8_buf[0..feature_width];
    const q_scale = quantizeToI8(query_row[0..feature_width], query_i8);

    // Compute Q @ K^T using fast SDOT-style i8×i8 dot product
    var max_score: f32 = -std.math.inf(f32);
    var row_index: usize = 0;

    // Set invalid positions to -inf
    while (row_index < valid_start_index) : (row_index += 1) {
        score_buffer[row_index] = -std.math.inf(f32);
    }

    // Compute dot products with i8×i8 SDOT (2-6x faster than f32×i8)
    while (row_index < history_len) : (row_index += 1) {
        const key_row_i8 = key_matrix_rows_i8[row_index * feature_width ..][0..feature_width];
        const key_scale = key_scales[row_index];
        const dot = dotRowI8xI8(query_i8, key_row_i8, q_scale, key_scale) * scale;
        score_buffer[row_index] = dot;
        if (dot > max_score) max_score = dot;
    }

    // Handle extra sink score
    if (extra_sink_score) |sink| {
        if (sink > max_score) max_score = sink;
    }

    // Softmax
    softmax.maskedInPlaceWithMax(
        score_buffer,
        valid_start_index,
        history_len,
        extra_sink_score,
        exact_softmax,
        max_score,
        null,
    );

    // Compute scores @ V using BLAS (V is already f32, no dequant needed!)
    const active_len = history_len - valid_start_index;
    const active_scores = score_buffer[valid_start_index..history_len];
    const active_v = value_matrix_rows[valid_start_index * feature_width ..][0 .. active_len * feature_width];

    accelerate.sgemm(
        active_scores,
        active_v,
        output_row[0..feature_width],
        1, // M = 1 query
        feature_width, // N
        active_len, // K
    );
}

/// Dequantize a chunk of V rows from i8 to f32
fn dequantizeVChunk(src: []const i8, scales: []const f32, dst: []f32, n_rows: usize, row_width: usize) void {
    std.debug.assert(src.len >= n_rows * row_width);
    std.debug.assert(scales.len >= n_rows);
    std.debug.assert(dst.len >= n_rows * row_width);

    for (0..n_rows) |row| {
        const scale = scales[row];
        const scale_vec: @Vector(4, f32) = @splat(scale);
        const src_row = src[row * row_width ..][0..row_width];
        const dst_row = dst[row * row_width ..][0..row_width];

        var i: usize = 0;
        while (i + 16 <= row_width) : (i += 16) {
            const i8_vec: @Vector(16, i8) = src_row[i..][0..16].*;

            // Group 0
            const i8_0: @Vector(4, i8) = .{ i8_vec[0], i8_vec[1], i8_vec[2], i8_vec[3] };
            dst_row[i..][0..4].* = @as(@Vector(4, f32), @floatFromInt(@as(@Vector(4, i32), i8_0))) * scale_vec;

            // Group 1
            const i8_1: @Vector(4, i8) = .{ i8_vec[4], i8_vec[5], i8_vec[6], i8_vec[7] };
            dst_row[i + 4 ..][0..4].* = @as(@Vector(4, f32), @floatFromInt(@as(@Vector(4, i32), i8_1))) * scale_vec;

            // Group 2
            const i8_2: @Vector(4, i8) = .{ i8_vec[8], i8_vec[9], i8_vec[10], i8_vec[11] };
            dst_row[i + 8 ..][0..4].* = @as(@Vector(4, f32), @floatFromInt(@as(@Vector(4, i32), i8_2))) * scale_vec;

            // Group 3
            const i8_3: @Vector(4, i8) = .{ i8_vec[12], i8_vec[13], i8_vec[14], i8_vec[15] };
            dst_row[i + 12 ..][0..4].* = @as(@Vector(4, f32), @floatFromInt(@as(@Vector(4, i32), i8_3))) * scale_vec;
        }

        while (i < row_width) : (i += 1) {
            dst_row[i] = @as(f32, @floatFromInt(src_row[i])) * scale;
        }
    }
}
