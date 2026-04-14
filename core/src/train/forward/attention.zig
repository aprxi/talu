//! Scaled dot-product attention forward pass for training.
//!
//! Full [seq × seq] attention matrices (training mode), not incremental
//! KV-cache style (inference mode). Saves attention probabilities for backward.

const std = @import("std");
const compute = @import("compute_pkg");
const parallel = @import("compute_pkg").parallel;

const simd = compute.cpu.simd.arch;
const VEC = simd.f32_vec_len;
const F32Vec = simd.F32Vec;
const math_fast = @import("compute_pkg").cpu.math_fast;
const fastExp = math_fast.fastExp;
const fastExpScalar = math_fast.fastExpScalar;

/// SIMD-vectorized dot product over `len` elements.
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

/// SIMD-vectorized out[j] += scale * v[j] for `len` elements.
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

/// Context for threaded attention forward.
const AttnCtx = struct {
    output: []f32,
    probs_out: []f32,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    kv_groups: usize,
    heads_x_seq: usize,
};

/// Process attention for a range of (batch, head, query_position) triples.
/// Items are ordered as: item = bi * n_heads * seq_len + h * seq_len + qi
/// so consecutive items share the same (bi, h) for cache locality.
fn attnTask(start: usize, end: usize, ctx: *AttnCtx) void {
    const seq_len = ctx.seq_len;
    const n_heads = ctx.n_heads;
    const n_kv_heads = ctx.n_kv_heads;
    const head_dim = ctx.head_dim;
    const scale = ctx.scale;
    const kv_groups = ctx.kv_groups;
    const heads_x_seq = ctx.heads_x_seq;

    for (start..end) |item| {
        const bi = item / heads_x_seq;
        const remainder = item % heads_x_seq;
        const h = remainder / seq_len;
        const qi = remainder % seq_len;
        const kv_h = h / kv_groups;

        const q_offset = (bi * seq_len + qi) * n_heads * head_dim + h * head_dim;
        const q_vec = ctx.q[q_offset..][0..head_dim];
        const prob_offset = (bi * n_heads + h) * seq_len * seq_len + qi * seq_len;
        const prob_row = ctx.probs_out[prob_offset..][0..seq_len];

        // Compute scores with causal mask
        var max_score: f32 = -std.math.inf(f32);
        for (0..seq_len) |ki| {
            if (ki > qi) {
                prob_row[ki] = -std.math.inf(f32);
            } else {
                const k_offset = (bi * seq_len + ki) * n_kv_heads * head_dim + kv_h * head_dim;
                const k_vec = ctx.k[k_offset..][0..head_dim];
                const dot = simdDot(q_vec, k_vec);
                prob_row[ki] = dot * scale;
                max_score = @max(max_score, prob_row[ki]);
            }
        }

        // SIMD softmax over valid positions [0..qi+1]
        const valid = qi + 1;
        const max_v: F32Vec = @splat(max_score);
        var sum_vec: F32Vec = @splat(0.0);
        var ki: usize = 0;
        while (ki + VEC <= valid) : (ki += VEC) {
            const v: F32Vec = prob_row[ki..][0..VEC].*;
            const e = fastExp(v - max_v);
            prob_row[ki..][0..VEC].* = e;
            sum_vec += e;
        }
        var sum_exp = @reduce(.Add, sum_vec);
        while (ki < valid) : (ki += 1) {
            const e = fastExpScalar(prob_row[ki] - max_score);
            prob_row[ki] = e;
            sum_exp += e;
        }
        // Zero masked positions
        for (valid..seq_len) |k| {
            prob_row[k] = 0.0;
        }
        // SIMD normalize
        const inv_sum = if (sum_exp > 0.0) 1.0 / sum_exp else 0.0;
        const inv_v: F32Vec = @splat(inv_sum);
        ki = 0;
        while (ki + VEC <= valid) : (ki += VEC) {
            var v: F32Vec = prob_row[ki..][0..VEC].*;
            v *= inv_v;
            prob_row[ki..][0..VEC].* = v;
        }
        while (ki < valid) : (ki += 1) {
            prob_row[ki] *= inv_sum;
        }

        // Weighted sum of values
        const out_offset = (bi * seq_len + qi) * n_heads * head_dim + h * head_dim;
        const out_vec = ctx.output[out_offset..][0..head_dim];
        @memset(out_vec, 0.0);
        for (0..seq_len) |vi| {
            if (prob_row[vi] == 0.0) continue;
            const v_offset = (bi * seq_len + vi) * n_kv_heads * head_dim + kv_h * head_dim;
            const v_vec = ctx.v[v_offset..][0..head_dim];
            simdScaleAdd(out_vec, v_vec, prob_row[vi]);
        }
    }
}

/// Scaled dot-product attention with causal masking (full sequence, training mode).
///
/// Q: [batch * seq * n_heads * head_dim]  (stored as [bs, n_heads * head_dim])
/// K: [batch * seq * n_kv_heads * head_dim]
/// V: [batch * seq * n_kv_heads * head_dim]
/// probs_out: [batch * n_heads * seq * seq]  — saved for backward
/// output: [batch * seq * n_heads * head_dim]
pub fn attentionForward(
    output: []f32,
    probs_out: []f32,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    batch: usize,
    seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) void {
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    const kv_groups = n_heads / n_kv_heads;
    const total_items = batch * n_heads * seq_len;

    var ctx = AttnCtx{
        .output = output,
        .probs_out = probs_out,
        .q = q,
        .k = k,
        .v = v,
        .seq_len = seq_len,
        .n_heads = n_heads,
        .n_kv_heads = n_kv_heads,
        .head_dim = head_dim,
        .scale = scale,
        .kv_groups = kv_groups,
        .heads_x_seq = n_heads * seq_len,
    };

    parallel.global().parallelFor(total_items, attnTask, &ctx);
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

test "attentionForward causal masking works" {
    // batch=1, seq=2, n_heads=1, n_kv_heads=1, head_dim=2
    // Q: [[1,0], [0,1]], K: [[1,0], [0,1]], V: [[1,2], [3,4]]
    const q_data = [_]f32{ 1, 0, 0, 1 }; // [2, 1*2]
    const k_data = [_]f32{ 1, 0, 0, 1 };
    const v_data = [_]f32{ 1, 2, 3, 4 };
    var output: [4]f32 = undefined;
    var probs: [4]f32 = undefined; // [1, 1, 2, 2]

    attentionForward(&output, &probs, &q_data, &k_data, &v_data, 1, 2, 1, 1, 2);

    // Position 0 can only attend to position 0
    // probs[0, :] should be [1.0, 0.0] (causal mask)
    try testing.expectApproxEqAbs(@as(f32, 1.0), probs[0], 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 0.0), probs[1], 1e-3);

    // Output at position 0 should be V[0] = [1, 2]
    try testing.expectApproxEqAbs(@as(f32, 1.0), output[0], 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 2.0), output[1], 1e-3);

    // Position 1 attends to both positions
    // Q[1]=[0,1], K[0]=[1,0], K[1]=[0,1]
    // scores: Q[1].K[0] = 0, Q[1].K[1] = 1/sqrt(2)
    // probs[1, :] should be softmax([0, 0.707]) ≈ [0.33, 0.67]
    try testing.expect(probs[3] > probs[2]); // position 1 gets more weight
}
