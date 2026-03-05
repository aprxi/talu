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
const compute = @import("../../compute/root.zig");

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
