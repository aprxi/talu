//! Attention backward pass for scaled dot-product attention (SDPA).
//!
//! Forward:
//!   scores = Q @ K^T * scale            [n_heads, seq_len]
//!   probs  = softmax(scores, dim=-1)    [n_heads, seq_len]
//!   output = probs @ V                  [n_heads, head_dim]
//!
//! Backward:
//!   d_V     = probs^T @ d_output        [seq_len, n_heads, head_dim] -> accumulated per KV head
//!   d_probs = d_output @ V^T            [n_heads, seq_len]
//!   d_scores = d_probs * probs - probs * sum(d_probs * probs)  (softmax backward)
//!   d_Q     = d_scores @ K * scale      [n_heads, head_dim]
//!   d_K     = d_scores^T @ Q * scale    [seq_len, n_heads, head_dim] -> accumulated per KV head
//!
//! GQA: multiple query heads share one KV head. Gradients for shared KV heads
//! are accumulated across query heads.

const std = @import("std");
const compute = @import("compute_pkg");
const parallel = @import("compute_pkg").parallel;

const simd = compute.cpu.simd.arch;
const VEC = simd.f32_vec_len;
const F32Vec = simd.F32Vec;

/// SIMD-vectorized dot product.
fn simdDot(a: []const f32, b: []const f32) f32 {
    @setFloatMode(.optimized);
    const len = a.len;
    std.debug.assert(b.len >= len);

    var acc: F32Vec = @splat(0);
    var i: usize = 0;
    while (i + VEC <= len) : (i += VEC) {
        const av: F32Vec = a[i..][0..VEC].*;
        const bv: F32Vec = b[i..][0..VEC].*;
        acc = @mulAdd(F32Vec, av, bv, acc);
    }
    var result = @reduce(.Add, acc);
    while (i < len) : (i += 1) {
        result += a[i] * b[i];
    }
    return result;
}

/// SIMD-vectorized out[j] += scale * v[j].
fn simdScaleAdd(out: []f32, v: []const f32, scale: f32) void {
    @setFloatMode(.optimized);
    const len = out.len;
    std.debug.assert(v.len >= len);

    const s: F32Vec = @splat(scale);
    var i: usize = 0;
    while (i + VEC <= len) : (i += VEC) {
        var o: F32Vec = out[i..][0..VEC].*;
        const vi: F32Vec = v[i..][0..VEC].*;
        o = @mulAdd(F32Vec, s, vi, o);
        out[i..][0..VEC].* = o;
    }
    while (i < len) : (i += 1) {
        out[i] += scale * v[i];
    }
}

/// Compute attention backward for a single token position (training step).
///
/// This computes gradients for Q, K, V given the saved attention probabilities
/// and the gradient from upstream.
///
/// For GQA, d_K and d_V accumulate across query heads that share a KV head.
///
/// Params:
///   grad_q:      [n_heads * head_dim]           — overwritten
///   grad_k:      [seq_len * n_kv_heads * head_dim] — accumulated
///   grad_v:      [seq_len * n_kv_heads * head_dim] — accumulated
///   grad_output: [n_heads * head_dim]
///   query:       [n_heads * head_dim]            — saved from forward
///   key_cache:   [seq_len * n_kv_heads * head_dim]
///   value_cache: [seq_len * n_kv_heads * head_dim]
///   probs:       [n_heads * seq_len]             — saved softmax output
///   n_heads, n_kv_heads, head_dim, seq_len: shape params
///   scale: attention scale (typically 1/sqrt(head_dim))
pub fn attentionBackward(
    grad_q: []f32,
    grad_k: []f32,
    grad_v: []f32,
    grad_output: []const f32,
    query: []const f32,
    key_cache: []const f32,
    value_cache: []const f32,
    probs: []const f32,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) void {
    const kv_groups = n_heads / n_kv_heads;

    std.debug.assert(grad_q.len == n_heads * head_dim);
    std.debug.assert(grad_k.len == seq_len * n_kv_heads * head_dim);
    std.debug.assert(grad_v.len == seq_len * n_kv_heads * head_dim);
    std.debug.assert(grad_output.len == n_heads * head_dim);
    std.debug.assert(query.len == n_heads * head_dim);
    std.debug.assert(key_cache.len == seq_len * n_kv_heads * head_dim);
    std.debug.assert(value_cache.len == seq_len * n_kv_heads * head_dim);
    std.debug.assert(probs.len == n_heads * seq_len);

    for (0..n_heads) |h| {
        const kv_h = h / kv_groups;
        const q_offset = h * head_dim;
        const p_offset = h * seq_len;
        const do_offset = h * head_dim;

        const prob_row = probs[p_offset..][0..seq_len];
        const d_out = grad_output[do_offset..][0..head_dim];
        const q_row = query[q_offset..][0..head_dim];

        // Step 1: d_probs[t] = dot(d_output, V[t]) for each t
        // Step 2: softmax backward: d_scores[t] = probs[t] * (d_probs[t] - dot(probs, d_probs))
        // Compute d_probs and the dot product in one pass
        var d_probs_buf: [4096]f32 = undefined;
        const d_probs = d_probs_buf[0..seq_len];

        var prob_dot_dprob: f32 = 0.0;
        for (0..seq_len) |t| {
            const v_row = value_cache[(t * n_kv_heads + kv_h) * head_dim ..][0..head_dim];
            const dot_val = simdDot(d_out, v_row);
            d_probs[t] = dot_val;
            prob_dot_dprob += prob_row[t] * dot_val;
        }

        // d_scores[t] = probs[t] * (d_probs[t] - sum(probs * d_probs))
        // Then d_scores *= scale for the Q/K gradient computation
        // (scale was applied in forward: scores = Q@K^T * scale)

        // Step 3: d_Q = sum_t(d_scores[t] * K[t]) * scale
        const dq = grad_q[q_offset..][0..head_dim];
        @memset(dq, 0);

        for (0..seq_len) |t| {
            const d_score = prob_row[t] * (d_probs[t] - prob_dot_dprob) * scale;

            const k_row = key_cache[(t * n_kv_heads + kv_h) * head_dim ..][0..head_dim];

            // d_Q += d_score * K[t]
            simdScaleAdd(dq, k_row, d_score);

            // d_K[t] += d_score * Q  (accumulated across GQA heads)
            const dk_row = grad_k[(t * n_kv_heads + kv_h) * head_dim ..][0..head_dim];
            simdScaleAdd(dk_row, q_row, d_score);

            // d_V[t] += probs[t] * d_output  (accumulated across GQA heads)
            const dv_row = grad_v[(t * n_kv_heads + kv_h) * head_dim ..][0..head_dim];
            simdScaleAdd(dv_row, d_out, prob_row[t]);
        }
    }
}

/// Context for Phase 1: compute d_scores and d_Q.
/// Threaded over batch × n_heads × seq_len items.
const Phase1Ctx = struct {
    grad_q: []f32,
    d_scores: []f32,
    grad_output: []const f32,
    key_cache: []const f32,
    value_cache: []const f32,
    attn_probs: []const f32,
    seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    kv_groups: usize,
    heads_x_seq: usize,
};

/// Phase 1 worker: compute d_scores and d_Q for a range of (bi, h, qi) items.
fn phase1Task(start: usize, end: usize, ctx: *Phase1Ctx) void {
    @setFloatMode(.optimized);
    const s = ctx.seq_len;
    const nh = ctx.n_heads;
    const nkv = ctx.n_kv_heads;
    const hd = ctx.head_dim;
    const scale = ctx.scale;
    const kv_groups = ctx.kv_groups;
    const hxs = ctx.heads_x_seq;

    for (start..end) |item| {
        const bi = item / hxs;
        const remainder = item % hxs;
        const h = remainder / s;
        const qi = remainder % s;
        const kv_h = h / kv_groups;

        const do_vec = ctx.grad_output[(bi * s + qi) * nh * hd + h * hd ..][0..hd];
        const prob_row = ctx.attn_probs[(bi * nh + h) * s * s + qi * s ..][0..s];
        const ds_row = ctx.d_scores[(bi * nh + h) * s * s + qi * s ..][0..s];

        // d_probs[t] = dot(d_output, V[t]), stored in ds_row temporarily
        var prob_dot_dprob: f32 = 0.0;
        for (0..qi + 1) |t| {
            const v_row = ctx.value_cache[(bi * s + t) * nkv * hd + kv_h * hd ..][0..hd];
            const dp = simdDot(do_vec, v_row);
            ds_row[t] = dp;
            prob_dot_dprob += prob_row[t] * dp;
        }
        // Zero masked positions
        for (qi + 1..s) |t| {
            ds_row[t] = 0.0;
        }

        // Softmax backward + scale: d_scores[t] = probs[t] * (d_probs[t] - sum) * scale
        for (0..qi + 1) |t| {
            ds_row[t] = prob_row[t] * (ds_row[t] - prob_dot_dprob) * scale;
        }

        // d_Q = sum_t(d_scores[t] * K[t])
        const dq = ctx.grad_q[(bi * s + qi) * nh * hd + h * hd ..][0..hd];
        @memset(dq, 0);
        for (0..qi + 1) |t| {
            const k_row = ctx.key_cache[(bi * s + t) * nkv * hd + kv_h * hd ..][0..hd];
            simdScaleAdd(dq, k_row, ds_row[t]);
        }
    }
}

/// Context for Phase 2: compute d_K and d_V.
/// Threaded over batch × n_kv_heads × seq_len items.
const Phase2Ctx = struct {
    grad_k: []f32,
    grad_v: []f32,
    d_scores: []const f32,
    query: []const f32,
    grad_output: []const f32,
    attn_probs: []const f32,
    seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    kv_groups: usize,
    nkv_x_seq: usize,
};

/// Phase 2 worker: compute d_K[t] and d_V[t] for a range of (bi, kv_h, t) items.
fn phase2Task(start: usize, end: usize, ctx: *Phase2Ctx) void {
    @setFloatMode(.optimized);
    const s = ctx.seq_len;
    const nh = ctx.n_heads;
    const nkv = ctx.n_kv_heads;
    const hd = ctx.head_dim;
    const kv_groups = ctx.kv_groups;
    const nkv_x_s = ctx.nkv_x_seq;

    for (start..end) |item| {
        const bi = item / nkv_x_s;
        const remainder = item % nkv_x_s;
        const kv_h = remainder / s;
        const t = remainder % s;

        const dk = ctx.grad_k[(bi * s + t) * nkv * hd + kv_h * hd ..][0..hd];
        const dv = ctx.grad_v[(bi * s + t) * nkv * hd + kv_h * hd ..][0..hd];
        @memset(dk, 0);
        @memset(dv, 0);

        // Accumulate across query heads in this KV group and query positions
        const h_start = kv_h * kv_groups;
        const h_end = h_start + kv_groups;
        for (h_start..h_end) |h| {
            // Only qi >= t contribute (causal: positions before t don't attend to t)
            for (t..s) |qi| {
                const ds_idx = (bi * nh + h) * s * s + qi * s + t;
                const d_score = ctx.d_scores[ds_idx];
                const q_row = ctx.query[(bi * s + qi) * nh * hd + h * hd ..][0..hd];
                simdScaleAdd(dk, q_row, d_score);

                const prob = ctx.attn_probs[ds_idx];
                const do_row = ctx.grad_output[(bi * s + qi) * nh * hd + h * hd ..][0..hd];
                simdScaleAdd(dv, do_row, prob);
            }
        }
    }
}

/// Batch attention backward: threaded two-phase computation.
///
/// Phase 1 (threaded over batch × n_heads × seq_len):
///   Computes d_scores and d_Q. Each (bi, h, qi) is independent.
///
/// Phase 2 (threaded over batch × n_kv_heads × seq_len):
///   Computes d_K and d_V. Each (bi, kv_h, t) is independent.
///
/// d_scores_buf: [batch * n_heads * seq_len * seq_len] — scratch for intermediate d_scores.
pub fn attentionBackwardBatch(
    grad_q: []f32,
    grad_k: []f32,
    grad_v: []f32,
    grad_output: []const f32,
    query: []const f32,
    key_cache: []const f32,
    value_cache: []const f32,
    attn_probs: []const f32,
    d_scores_buf: []f32,
    batch: usize,
    seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f32,
) void {
    const kv_groups = n_heads / n_kv_heads;
    const bs = batch * seq_len;

    std.debug.assert(grad_q.len >= bs * n_heads * head_dim);
    std.debug.assert(grad_k.len >= bs * n_kv_heads * head_dim);
    std.debug.assert(grad_v.len >= bs * n_kv_heads * head_dim);
    std.debug.assert(grad_output.len >= bs * n_heads * head_dim);
    std.debug.assert(query.len >= bs * n_heads * head_dim);
    std.debug.assert(key_cache.len >= bs * n_kv_heads * head_dim);
    std.debug.assert(value_cache.len >= bs * n_kv_heads * head_dim);
    std.debug.assert(attn_probs.len >= batch * n_heads * seq_len * seq_len);
    std.debug.assert(d_scores_buf.len >= batch * n_heads * seq_len * seq_len);

    // Phase 1: d_Q and d_scores
    const total_phase1 = batch * n_heads * seq_len;
    var ctx1 = Phase1Ctx{
        .grad_q = grad_q,
        .d_scores = d_scores_buf,
        .grad_output = grad_output,
        .key_cache = key_cache,
        .value_cache = value_cache,
        .attn_probs = attn_probs,
        .seq_len = seq_len,
        .n_heads = n_heads,
        .n_kv_heads = n_kv_heads,
        .head_dim = head_dim,
        .scale = scale,
        .kv_groups = kv_groups,
        .heads_x_seq = n_heads * seq_len,
    };
    parallel.global().parallelFor(total_phase1, phase1Task, &ctx1);

    // Phase 2: d_K and d_V
    const total_phase2 = batch * n_kv_heads * seq_len;
    var ctx2 = Phase2Ctx{
        .grad_k = grad_k,
        .grad_v = grad_v,
        .d_scores = d_scores_buf,
        .query = query,
        .grad_output = grad_output,
        .attn_probs = attn_probs,
        .seq_len = seq_len,
        .n_heads = n_heads,
        .n_kv_heads = n_kv_heads,
        .head_dim = head_dim,
        .kv_groups = kv_groups,
        .nkv_x_seq = n_kv_heads * seq_len,
    };
    parallel.global().parallelFor(total_phase2, phase2Task, &ctx2);
}

// =============================================================================
// Tests
// =============================================================================

test "attentionBackward gradient shapes" {
    // Simple: 1 head, head_dim=2, seq_len=2
    const n_heads: usize = 1;
    const n_kv_heads: usize = 1;
    const head_dim: usize = 2;
    const seq_len: usize = 2;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    // Uniform attention probs
    const probs = [_]f32{ 0.5, 0.5 };
    const query = [_]f32{ 1.0, 0.0 };
    const key_cache = [_]f32{ 1.0, 0.0, 0.0, 1.0 }; // 2 tokens
    const value_cache = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    const grad_output = [_]f32{ 1.0, 1.0 };

    var grad_q = [_]f32{ 0, 0 };
    var grad_k = [_]f32{ 0, 0, 0, 0 };
    var grad_v = [_]f32{ 0, 0, 0, 0 };

    attentionBackward(
        &grad_q,
        &grad_k,
        &grad_v,
        &grad_output,
        &query,
        &key_cache,
        &value_cache,
        &probs,
        n_heads,
        n_kv_heads,
        head_dim,
        seq_len,
        scale,
    );

    // With uniform probs, d_V should be probs * d_output for each token
    // d_V[0] = 0.5 * [1,1] = [0.5, 0.5]
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), grad_v[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), grad_v[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), grad_v[2], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), grad_v[3], 1e-4);
}

test "attentionBackward GQA accumulates across heads" {
    // 2 query heads sharing 1 KV head
    const n_heads: usize = 2;
    const n_kv_heads: usize = 1;
    const head_dim: usize = 2;
    const seq_len: usize = 1;
    const scale: f32 = 1.0;

    // Both heads have prob=1.0 for the single token
    const probs = [_]f32{ 1.0, 1.0 };
    const query = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    const key_cache = [_]f32{ 1.0, 1.0 };
    const value_cache = [_]f32{ 1.0, 0.0 };
    const grad_output = [_]f32{ 1.0, 0.0, 0.0, 1.0 };

    var grad_q = [_]f32{ 0, 0, 0, 0 };
    var grad_k = [_]f32{ 0, 0 };
    var grad_v = [_]f32{ 0, 0 };

    attentionBackward(
        &grad_q,
        &grad_k,
        &grad_v,
        &grad_output,
        &query,
        &key_cache,
        &value_cache,
        &probs,
        n_heads,
        n_kv_heads,
        head_dim,
        seq_len,
        scale,
    );

    // d_V should accumulate from both heads
    // Head 0: probs[0]=1.0, d_out=[1,0] => d_V += [1, 0]
    // Head 1: probs[0]=1.0, d_out=[0,1] => d_V += [0, 1]
    // Total d_V = [1, 1]
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), grad_v[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), grad_v[1], 1e-4);
}

test "attentionBackwardBatch matches per-position attentionBackward" {
    // batch=2, seq=3, n_heads=2, n_kv_heads=2, head_dim=4
    const b: usize = 2;
    const s: usize = 3;
    const nh: usize = 2;
    const nkv: usize = 2;
    const hd: usize = 4;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

    const bs = b * s;
    const q_len = bs * nh * hd;
    const kv_len = bs * nkv * hd;
    const prob_len = b * nh * s * s;

    // Deterministic test data
    var query: [q_len]f32 = undefined;
    var key_cache: [kv_len]f32 = undefined;
    var value_cache: [kv_len]f32 = undefined;
    var grad_output: [q_len]f32 = undefined;

    for (0..q_len) |i| {
        query[i] = @as(f32, @floatFromInt(i % 7)) * 0.1 - 0.3;
        grad_output[i] = @as(f32, @floatFromInt(i % 11)) * 0.05 - 0.2;
    }
    for (0..kv_len) |i| {
        key_cache[i] = @as(f32, @floatFromInt(i % 13)) * 0.08 - 0.4;
        value_cache[i] = @as(f32, @floatFromInt(i % 9)) * 0.12 - 0.5;
    }

    // Build valid causal attention probs (softmax rows that sum to 1, zero after causal mask)
    var attn_probs: [prob_len]f32 = undefined;
    for (0..b) |bi| {
        for (0..nh) |h| {
            for (0..s) |qi| {
                const row_start = (bi * nh + h) * s * s + qi * s;
                var row_sum: f32 = 0.0;
                for (0..s) |t| {
                    if (t > qi) {
                        attn_probs[row_start + t] = 0.0;
                    } else {
                        const val = @as(f32, @floatFromInt((row_start + t) % 5)) * 0.1 + 0.1;
                        attn_probs[row_start + t] = val;
                        row_sum += val;
                    }
                }
                // Normalize to sum to 1
                for (0..qi + 1) |t| {
                    attn_probs[row_start + t] /= row_sum;
                }
            }
        }
    }

    // Reference: per-position attentionBackward
    var ref_grad_q: [q_len]f32 = undefined;
    var ref_grad_k: [kv_len]f32 = .{0} ** kv_len;
    var ref_grad_v: [kv_len]f32 = .{0} ** kv_len;

    for (0..b) |bi| {
        for (0..s) |pos| {
            const token_idx = bi * s + pos;
            const q_offset = token_idx * nh * hd;
            const kv_offset = bi * s * nkv * hd;

            // Gather prob rows for this position
            var prob_buf: [nh * s]f32 = undefined;
            for (0..nh) |h| {
                const src_off = (bi * nh + h) * s * s + pos * s;
                const dst_off = h * s;
                @memcpy(prob_buf[dst_off..][0..s], attn_probs[src_off..][0..s]);
            }

            attentionBackward(
                ref_grad_q[q_offset..][0 .. nh * hd],
                ref_grad_k[kv_offset..][0 .. s * nkv * hd],
                ref_grad_v[kv_offset..][0 .. s * nkv * hd],
                grad_output[q_offset..][0 .. nh * hd],
                query[q_offset..][0 .. nh * hd],
                key_cache[kv_offset..][0 .. s * nkv * hd],
                value_cache[kv_offset..][0 .. s * nkv * hd],
                &prob_buf,
                nh,
                nkv,
                hd,
                s,
                scale,
            );
        }
    }

    // Batch version
    var batch_grad_q: [q_len]f32 = undefined;
    var batch_grad_k: [kv_len]f32 = undefined;
    var batch_grad_v: [kv_len]f32 = undefined;
    var d_scores: [prob_len]f32 = undefined;

    attentionBackwardBatch(
        &batch_grad_q,
        &batch_grad_k,
        &batch_grad_v,
        &grad_output,
        &query,
        &key_cache,
        &value_cache,
        &attn_probs,
        &d_scores,
        b,
        s,
        nh,
        nkv,
        hd,
        scale,
    );

    // Compare grad_q
    for (0..q_len) |i| {
        try std.testing.expectApproxEqAbs(ref_grad_q[i], batch_grad_q[i], 1e-4);
    }
    // Compare grad_k
    for (0..kv_len) |i| {
        try std.testing.expectApproxEqAbs(ref_grad_k[i], batch_grad_k[i], 1e-4);
    }
    // Compare grad_v
    for (0..kv_len) |i| {
        try std.testing.expectApproxEqAbs(ref_grad_v[i], batch_grad_v[i], 1e-4);
    }
}
