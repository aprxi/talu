//! Training forward pass for decoder-only transformers.
//!
//! Implements the standard pre-LN architecture:
//!   embed → [RMSNorm → Attention → Residual → RMSNorm → SwiGLU → Residual] × L → RMSNorm → LM Head
//!
//! Saves all intermediate activations needed by the backward pass into the
//! ActivationCache. Uses f32 throughout (no mixed precision).
//!
//! Attention is computed as full [seq × seq] matrices (training mode),
//! not incremental KV-cache style (inference mode).

const std = @import("std");
const tensor_mod = @import("../tensor.zig");
const compute = @import("../compute/root.zig");
const model_config = @import("model_config.zig");
const model_weights_mod = @import("model_weights.zig");
const activations_mod = @import("activations.zig");
const loss_mod = @import("loss.zig");

const Tensor = tensor_mod.Tensor;
const MatmulScratch = compute.cpu.linalg.MatmulScratch;
const matmulF32 = compute.cpu.linalg.matmulF32;
const softmaxContiguous = compute.cpu.math_softmax.softmaxContiguous;
const TransformerConfig = model_config.TransformerConfig;
const ModelWeights = model_weights_mod.ModelWeights;
const LayerWeights = model_weights_mod.LayerWeights;
const ActivationCache = activations_mod.ActivationCache;
const LayerActivations = activations_mod.LayerActivations;

/// Run the full forward pass and compute cross-entropy loss.
///
/// Writes activations into `cache` for the backward pass.
/// Returns the mean cross-entropy loss over all tokens.
///
/// tokens:  [batch_size * seq_len] — input token IDs
/// targets: [batch_size * seq_len] — target token IDs (shifted by 1)
pub fn forward(
    weights: *const ModelWeights,
    cache: *ActivationCache,
    tokens: []const u32,
    targets: []const u32,
    scratch: *MatmulScratch,
) f32 {
    const config = weights.config;
    const b: usize = cache.batch_size;
    const s: usize = config.seq_len;
    const d: usize = config.d_model;
    const bs = b * s;

    std.debug.assert(tokens.len == bs);
    std.debug.assert(targets.len == bs);

    // 1. Token embedding lookup: hidden[i] = embedding[tokens[i]]
    embeddingForward(cache.hidden, weights.token_embedding.asSlice(f32), tokens, d);

    // 2. Transformer layers
    for (cache.layers, weights.layers) |*la, *lw| {
        layerForward(cache.hidden, la, lw, config, bs, cache.scratch, scratch);
    }

    // 3. Final RMSNorm
    rmsnormForwardSave(
        cache.final_normed,
        cache.final_inv_rms,
        cache.hidden,
        weights.final_norm.asSlice(f32),
        config.norm_eps,
        bs,
        d,
    );

    // 4. LM Head: logits = final_normed @ lm_head^T
    //    final_normed: [bs, d], lm_head: [vocab, d] → logits: [bs, vocab]
    //    We need: logits = final_normed @ lm_head^T
    //    matmulF32 does: out = a @ b, so we need b = lm_head^T
    //    Instead: logits[i,v] = sum_d(final_normed[i,d] * lm_head[v,d])
    //    This is: out[bs, vocab] = a[bs, d] @ b[d, vocab]
    //    But lm_head is [vocab, d], so we do the transpose manually.
    lmHeadForward(
        cache.logits,
        cache.final_normed,
        weights.lm_head.asSlice(f32),
        bs,
        d,
        config.vocab_size,
    );

    // 5. Cross-entropy loss
    return loss_mod.crossEntropyLoss(cache.logits, targets, bs, config.vocab_size);
}

/// Token embedding lookup.
/// output[i * d .. (i+1) * d] = embedding[tokens[i] * d .. (tokens[i]+1) * d]
fn embeddingForward(output: []f32, embedding: []const f32, tokens: []const u32, d_model: usize) void {
    for (tokens, 0..) |token, i| {
        const src = embedding[token * d_model ..][0..d_model];
        const dst = output[i * d_model ..][0..d_model];
        @memcpy(dst, src);
    }
}

/// RMSNorm forward that saves inv_rms for the backward pass.
///
/// output[i] = input[i] * inv_rms[row] * weight
/// inv_rms[row] = 1 / sqrt(mean(input[row]^2) + eps)
fn rmsnormForwardSave(
    output: []f32,
    inv_rms: []f32,
    input: []const f32,
    weight: []const f32,
    eps: f32,
    rows: usize,
    cols: usize,
) void {
    std.debug.assert(output.len >= rows * cols);
    std.debug.assert(input.len >= rows * cols);
    std.debug.assert(inv_rms.len >= rows);
    std.debug.assert(weight.len == cols);

    const cols_f: f32 = @floatFromInt(cols);

    for (0..rows) |row| {
        const in_row = input[row * cols ..][0..cols];
        const out_row = output[row * cols ..][0..cols];

        // Compute mean squared value
        var sum_sq: f32 = 0.0;
        for (in_row) |v| {
            sum_sq += v * v;
        }
        const rms = @sqrt(sum_sq / cols_f + eps);
        const irms = 1.0 / rms;
        inv_rms[row] = irms;

        // Apply normalization and scale
        for (out_row, in_row, weight) |*o, x, w| {
            o.* = x * irms * w;
        }
    }
}

/// Forward pass for a single transformer layer.
///
/// Modifies `hidden` in-place: hidden = hidden + Attn(Norm(hidden)) + FFN(Norm(hidden + attn_out))
///
/// `ffn_scratch`: [bs * d_ff] temporary buffer for SwiGLU intermediate result.
///                The caller provides this from ActivationCache.scratch.
fn layerForward(
    hidden: []f32,
    la: *LayerActivations,
    lw: *const LayerWeights,
    config: TransformerConfig,
    bs: usize,
    ffn_scratch: []f32,
    scratch: *MatmulScratch,
) void {
    const d: usize = config.d_model;
    const nh: usize = config.num_heads;
    const nkv: usize = config.num_kv_heads;
    const hd: usize = config.headDim();
    const ff: usize = config.d_ff;
    const s: usize = config.seq_len;
    const b: usize = bs / s;

    // Save residual input for backward
    @memcpy(la.residual_pre_attn[0 .. bs * d], hidden[0 .. bs * d]);

    // Attention RMSNorm
    rmsnormForwardSave(la.normed_attn, la.inv_rms_attn, hidden, lw.attn_norm.asSlice(f32), config.norm_eps, bs, d);

    // Q/K/V projections: [bs, d] @ [out, d]^T → [bs, out]
    linearForward(la.q, la.normed_attn, lw.q_proj.asSlice(f32), bs, d, nh * hd, scratch);
    linearForward(la.k, la.normed_attn, lw.k_proj.asSlice(f32), bs, d, nkv * hd, scratch);
    linearForward(la.v, la.normed_attn, lw.v_proj.asSlice(f32), bs, d, nkv * hd, scratch);

    // Apply RoPE to Q and K per position
    for (0..b) |bi| {
        for (0..s) |pos| {
            const token_idx = bi * s + pos;
            ropeForward(la.q[token_idx * nh * hd ..][0 .. nh * hd], nh, hd, hd, pos, config.rope_theta);
            ropeForward(la.k[token_idx * nkv * hd ..][0 .. nkv * hd], nkv, hd, hd, pos, config.rope_theta);
        }
    }

    // Scaled dot-product attention
    attentionForward(la.attn_output, la.attn_probs, la.q, la.k, la.v, b, s, nh, nkv, hd);

    // Output projection: [bs, nh*hd] @ [d, nh*hd]^T → [bs, d]
    linearForward(hidden, la.attn_output, lw.o_proj.asSlice(f32), bs, nh * hd, d, scratch);

    // Add attention residual
    for (hidden[0 .. bs * d], la.residual_pre_attn[0 .. bs * d]) |*h, r| {
        h.* += r;
    }

    // Save residual for FFN
    @memcpy(la.residual_pre_ffn[0 .. bs * d], hidden[0 .. bs * d]);

    // FFN RMSNorm
    rmsnormForwardSave(la.normed_ffn, la.inv_rms_ffn, hidden, lw.ffn_norm.asSlice(f32), config.norm_eps, bs, d);

    // SwiGLU FFN: output = down_proj(silu(gate_proj(x)) * up_proj(x))
    linearForward(la.gate, la.normed_ffn, lw.gate_proj.asSlice(f32), bs, d, ff, scratch);
    linearForward(la.up, la.normed_ffn, lw.up_proj.asSlice(f32), bs, d, ff, scratch);

    // Compute silu(gate) * up into ffn_scratch [bs * ff]
    swigluForward(ffn_scratch[0 .. bs * ff], la.gate, la.up, bs * ff);

    // Down projection: [bs, ff] → [bs, d]
    linearForward(hidden, ffn_scratch, lw.down_proj.asSlice(f32), bs, ff, d, scratch);

    // Add FFN residual
    for (hidden[0 .. bs * d], la.residual_pre_ffn[0 .. bs * d]) |*h, r| {
        h.* += r;
    }
}

/// Linear layer forward: output = input @ weight^T
/// input: [rows, in_dim], weight: [out_dim, in_dim] → output: [rows, out_dim]
fn linearForward(output: []f32, input: []const f32, weight: []const f32, rows: usize, in_dim: usize, out_dim: usize, scratch: *MatmulScratch) void {
    // output[i, j] = sum_k(input[i, k] * weight[j, k])
    // This is: output = input @ weight^T
    // matmulF32 does: out = a @ b where a[m,k] @ b[k,n] → out[m,n]
    // So we need: a = input[rows, in_dim], b = weight^T[in_dim, out_dim]
    // But weight is stored as [out_dim, in_dim].
    // Manual implementation for now since we need the transpose.
    _ = scratch;
    std.debug.assert(output.len >= rows * out_dim);
    std.debug.assert(input.len >= rows * in_dim);
    std.debug.assert(weight.len >= out_dim * in_dim);

    for (0..rows) |i| {
        const in_row = input[i * in_dim ..][0..in_dim];
        const out_row = output[i * out_dim ..][0..out_dim];
        for (0..out_dim) |j| {
            const w_row = weight[j * in_dim ..][0..in_dim];
            var sum: f32 = 0.0;
            for (in_row, w_row) |a, b| {
                sum += a * b;
            }
            out_row[j] = sum;
        }
    }
}

/// LM head forward: logits[i, v] = sum_d(input[i, d] * weight[v, d])
/// Same as linearForward but separated for clarity in the main forward.
fn lmHeadForward(logits: []f32, input: []const f32, weight: []const f32, rows: usize, d_model: usize, vocab_size: usize) void {
    std.debug.assert(logits.len >= rows * vocab_size);
    std.debug.assert(input.len >= rows * d_model);
    std.debug.assert(weight.len >= vocab_size * d_model);

    for (0..rows) |i| {
        const in_row = input[i * d_model ..][0..d_model];
        const out_row = logits[i * vocab_size ..][0..vocab_size];
        for (0..vocab_size) |v| {
            const w_row = weight[v * d_model ..][0..d_model];
            var sum: f32 = 0.0;
            for (in_row, w_row) |a, b| {
                sum += a * b;
            }
            out_row[v] = sum;
        }
    }
}

/// SwiGLU forward: output[i] = silu(gate[i]) * up[i]
/// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
fn swigluForward(output: []f32, gate: []const f32, up: []const f32, len: usize) void {
    for (0..len) |i| {
        const x = gate[i];
        const sigmoid = 1.0 / (1.0 + @exp(-x));
        output[i] = x * sigmoid * up[i];
    }
}

/// Scaled dot-product attention with causal masking (full sequence, training mode).
///
/// Q: [batch * seq * n_heads * head_dim]  (stored as [bs, n_heads * head_dim])
/// K: [batch * seq * n_kv_heads * head_dim]
/// V: [batch * seq * n_kv_heads * head_dim]
/// probs_out: [batch * n_heads * seq * seq]  — saved for backward
/// output: [batch * seq * n_heads * head_dim]
fn attentionForward(
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

    for (0..batch) |bi| {
        for (0..n_heads) |h| {
            const kv_h = h / kv_groups;

            // Compute attention scores: scores[qi, ki] = Q[qi] . K[ki] * scale
            // Then apply causal mask and softmax
            for (0..seq_len) |qi| {
                const q_offset = (bi * seq_len + qi) * n_heads * head_dim + h * head_dim;
                const q_vec = q[q_offset..][0..head_dim];
                const prob_offset = (bi * n_heads + h) * seq_len * seq_len + qi * seq_len;
                const prob_row = probs_out[prob_offset..][0..seq_len];

                // Compute scores with causal mask
                var max_score: f32 = -std.math.inf(f32);
                for (0..seq_len) |ki| {
                    if (ki > qi) {
                        prob_row[ki] = -std.math.inf(f32);
                    } else {
                        const k_offset = (bi * seq_len + ki) * n_kv_heads * head_dim + kv_h * head_dim;
                        const k_vec = k[k_offset..][0..head_dim];
                        var dot: f32 = 0.0;
                        for (q_vec, k_vec) |qv, kv| {
                            dot += qv * kv;
                        }
                        prob_row[ki] = dot * scale;
                        max_score = @max(max_score, prob_row[ki]);
                    }
                }

                // Softmax over valid positions [0..qi+1]
                var sum_exp: f32 = 0.0;
                for (0..seq_len) |ki| {
                    if (ki > qi) {
                        prob_row[ki] = 0.0;
                    } else {
                        const e = @exp(prob_row[ki] - max_score);
                        prob_row[ki] = e;
                        sum_exp += e;
                    }
                }
                const inv_sum = if (sum_exp > 0.0) 1.0 / sum_exp else 0.0;
                for (0..seq_len) |ki| {
                    prob_row[ki] *= inv_sum;
                }

                // Weighted sum of values: output[qi, h] = sum_ki(probs[qi, ki] * V[ki])
                const out_offset = (bi * seq_len + qi) * n_heads * head_dim + h * head_dim;
                const out_vec = output[out_offset..][0..head_dim];
                @memset(out_vec, 0.0);
                for (0..seq_len) |ki| {
                    if (prob_row[ki] == 0.0) continue;
                    const v_offset = (bi * seq_len + ki) * n_kv_heads * head_dim + kv_h * head_dim;
                    const v_vec = v[v_offset..][0..head_dim];
                    const p = prob_row[ki];
                    for (out_vec, v_vec) |*o, vv| {
                        o.* += p * vv;
                    }
                }
            }
        }
    }
}

/// RoPE forward: apply rotary position embeddings in-place.
/// Same algorithm as backward/rope.zig's test helper ropeForward.
fn ropeForward(io: []f32, n_heads: usize, head_dim: usize, rope_dim: usize, position: usize, theta: f32) void {
    std.debug.assert(io.len == n_heads * head_dim);
    std.debug.assert(rope_dim <= head_dim);
    std.debug.assert(rope_dim % 2 == 0);

    const half = rope_dim / 2;
    const pos_f: f32 = @floatFromInt(position);

    for (0..n_heads) |head| {
        const base = head * head_dim;
        for (0..half) |pair| {
            const freq_exp = -2.0 * @as(f32, @floatFromInt(pair)) / @as(f32, @floatFromInt(rope_dim));
            const inv_freq = std.math.pow(f32, theta, freq_exp);
            const angle = pos_f * inv_freq;

            const cos_a = @cos(angle);
            const sin_a = @sin(angle);

            const lo_idx = base + pair;
            const hi_idx = base + half + pair;

            const x_lo = io[lo_idx];
            const x_hi = io[hi_idx];

            io[lo_idx] = x_lo * cos_a - x_hi * sin_a;
            io[hi_idx] = x_lo * sin_a + x_hi * cos_a;
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

test "embeddingForward looks up correct rows" {
    // Embedding: 4 tokens, d_model=3
    const embedding = [_]f32{
        0.1, 0.2, 0.3, // token 0
        0.4, 0.5, 0.6, // token 1
        0.7, 0.8, 0.9, // token 2
        1.0, 1.1, 1.2, // token 3
    };
    const tokens = [_]u32{ 2, 0, 3 };
    var output: [9]f32 = undefined;

    embeddingForward(&output, &embedding, &tokens, 3);

    // token 2 → [0.7, 0.8, 0.9]
    try testing.expectApproxEqAbs(@as(f32, 0.7), output[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.8), output[1], 1e-6);
    // token 0 → [0.1, 0.2, 0.3]
    try testing.expectApproxEqAbs(@as(f32, 0.1), output[3], 1e-6);
    // token 3 → [1.0, 1.1, 1.2]
    try testing.expectApproxEqAbs(@as(f32, 1.0), output[6], 1e-6);
}

test "rmsnormForwardSave produces normalized output and saves inv_rms" {
    const input = [_]f32{ 3.0, 4.0 }; // rms = sqrt((9+16)/2) = sqrt(12.5)
    var output: [2]f32 = undefined;
    var inv_rms: [1]f32 = undefined;
    const weight = [_]f32{ 1.0, 1.0 };

    rmsnormForwardSave(&output, &inv_rms, &input, &weight, 1e-5, 1, 2);

    const expected_rms = @sqrt(12.5 + 1e-5);
    const expected_inv = 1.0 / expected_rms;
    try testing.expectApproxEqAbs(expected_inv, inv_rms[0], 1e-5);
    try testing.expectApproxEqAbs(3.0 * expected_inv, output[0], 1e-5);
    try testing.expectApproxEqAbs(4.0 * expected_inv, output[1], 1e-5);
}

test "linearForward computes input @ weight^T" {
    // input: [2, 3], weight: [2, 3] (out_dim=2, in_dim=3)
    const input = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const weight = [_]f32{
        1, 0, 0, // row 0: select dim 0
        0, 1, 0, // row 1: select dim 1
    };
    var output: [4]f32 = undefined;
    var scratch = try MatmulScratch.init(testing.allocator);
    defer scratch.deinit();

    linearForward(&output, &input, &weight, 2, 3, 2, &scratch);

    // Row 0: [1,2,3] @ [[1,0],[0,1],[0,0]] = [1, 2]
    try testing.expectApproxEqAbs(@as(f32, 1.0), output[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 2.0), output[1], 1e-5);
    // Row 1: [4,5,6] @ ... = [4, 5]
    try testing.expectApproxEqAbs(@as(f32, 4.0), output[2], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 5.0), output[3], 1e-5);
}

test "swigluForward computes silu(gate) * up" {
    var output: [2]f32 = undefined;
    const gate = [_]f32{ 0.0, 1.0 };
    const up = [_]f32{ 2.0, 2.0 };

    swigluForward(&output, &gate, &up, 2);

    // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    try testing.expectApproxEqAbs(@as(f32, 0.0), output[0], 1e-5);
    // silu(1) = 1 * sigmoid(1) = 1 / (1 + exp(-1)) ≈ 0.7311
    try testing.expectApproxEqAbs(@as(f32, 0.7311), output[1] / 2.0, 1e-3);
}

test "attentionForward causal masking works" {
    // batch=1, seq=2, n_heads=1, n_kv_heads=1, head_dim=2
    // Q: [[1,0], [0,1]], K: [[1,0], [0,1]], V: [[1,2], [3,4]]
    const q = [_]f32{ 1, 0, 0, 1 }; // [2, 1*2]
    const k = [_]f32{ 1, 0, 0, 1 };
    const v = [_]f32{ 1, 2, 3, 4 };
    var output: [4]f32 = undefined;
    var probs: [4]f32 = undefined; // [1, 1, 2, 2]

    attentionForward(&output, &probs, &q, &k, &v, 1, 2, 1, 1, 2);

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

test "ropeForward at position 0 is identity" {
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const original = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    ropeForward(&data, 1, 4, 4, 0, 10000.0);

    // At position 0, cos=1, sin=0, so no change
    for (data, original) |d, o| {
        try testing.expectApproxEqAbs(o, d, 1e-6);
    }
}
