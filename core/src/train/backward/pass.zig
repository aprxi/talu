//! Backward pass orchestration for decoder-only transformers.
//!
//! Reverses the forward pass defined in forward/pass.zig, calling existing backward
//! kernels in the correct order. All gradients are accumulated into the ModelWeights
//! gradient buffers (which must be zeroed before calling).
//!
//! Order:
//!   1. Cross-entropy backward → grad_logits
//!   2. LM head backward → grad_hidden, grad_lm_head
//!   3. Final RMSNorm backward → grad_hidden, grad_final_norm
//!   4. Per-layer (reverse):
//!      a. FFN: down_proj → swiglu → gate/up_proj → ffn_norm → residual add
//!      b. Attn: o_proj → attention (per-position) → rope → q/k/v_proj → attn_norm → residual add
//!   5. Embedding backward → grad_token_embedding

const std = @import("std");
const compute = @import("../../compute/root.zig");
const model_config = @import("../model_config.zig");
const model_weights_mod = @import("../model_weights.zig");
const activations_mod = @import("../activations.zig");

// Backward kernels (sibling files in backward/)
const cross_entropy = @import("cross_entropy.zig");
const linear = @import("linear.zig");
const rmsnorm_bw = @import("rmsnorm.zig");
const activation_bw = @import("activation.zig");
const rope_bw = @import("rope.zig");
const embedding_bw = @import("embedding.zig");
const attention_bw = @import("attention.zig");

const simd = compute.cpu.simd.arch;
const VEC = simd.f32_vec_len;
const F32Vec = simd.F32Vec;

const MatmulScratch = compute.cpu.linalg.MatmulScratch;
const TransformerConfig = model_config.TransformerConfig;
const ModelWeights = model_weights_mod.ModelWeights;
const LayerWeights = model_weights_mod.LayerWeights;
const ActivationCache = activations_mod.ActivationCache;
const LayerActivations = activations_mod.LayerActivations;

/// Run the full backward pass, computing gradients for all model weights.
///
/// Assumes forward() has already been called and cache contains saved activations.
/// Gradients are accumulated into weights.grad_* buffers (must be zeroed beforehand).
///
/// tokens:  [batch_size * seq_len] — input token IDs (for embedding backward)
/// targets: [batch_size * seq_len] — target token IDs (for cross-entropy backward)
pub fn backward(
    weights: *ModelWeights,
    cache: *ActivationCache,
    tokens: []const u32,
    targets: []const u32,
    scratch: *MatmulScratch,
) void {
    const config = weights.config;
    const b: usize = cache.batch_size;
    const s: usize = config.seq_len;
    const d: usize = config.d_model;
    const v: usize = config.vocab_size;
    const bs = b * s;

    // --- 1. Cross-entropy backward → grad_logits (stored in cache.scratch) ---
    // We reuse cache.logits as grad_logits since we no longer need the logits.
    // crossEntropyBackward overwrites grad_logits with softmax(logits) - one_hot(target).
    // But we need logits as input too, so we must use a separate buffer.
    // Use cache.scratch[0..bs*v] for grad_logits.
    const grad_logits = cache.scratch[0 .. bs * v];
    cross_entropy.crossEntropyBackward(
        grad_logits,
        cache.logits,
        targets,
        bs,
        v,
    );

    // --- 2. LM head backward ---
    // Forward was: logits = final_normed @ lm_head^T
    // grad_lm_head += grad_logits^T @ final_normed  (accumulated)
    // grad_hidden = grad_logits @ lm_head            (overwritten)
    linear.gradWeight(
        weights.grad_lm_head.asSliceMut(),
        grad_logits,
        cache.final_normed,
        bs,
        v,
        d,
        scratch,
    );
    linear.gradInput(
        cache.grad_hidden,
        grad_logits,
        weights.lm_head.asSlice(f32),
        bs,
        v,
        d,
        scratch,
    );

    // --- 3. Final RMSNorm backward ---
    // Forward was: final_normed = rmsnorm(hidden, final_norm_weight)
    // grad_hidden is overwritten, grad_final_norm is accumulated.
    // Input to final rmsnorm was cache.hidden (the post-layer hidden state).
    // Copy grad_hidden to scratch to avoid aliasing grad_input/grad_output
    // (the kernel reads grad_output after writing grad_input on the same row).
    const grad_output_copy = cache.scratch[0 .. bs * d];
    @memcpy(grad_output_copy, cache.grad_hidden[0 .. bs * d]);
    rmsnorm_bw.rmsnormBackward(
        cache.grad_hidden,
        weights.grad_final_norm.asSliceMut(),
        grad_output_copy,
        cache.hidden, // input to rmsnorm (saved as final hidden state)
        cache.final_inv_rms,
        weights.final_norm.asSlice(f32),
        bs,
        d,
        0.0, // weight_offset
        null,
    );

    // --- 4. Per-layer backward (reverse order) ---
    const num_layers = config.num_layers;
    var layer_idx: usize = num_layers;
    while (layer_idx > 0) {
        layer_idx -= 1;
        layerBackward(
            cache.grad_hidden,
            &cache.layers[layer_idx],
            &weights.layers[layer_idx],
            config,
            bs,
            cache.scratch,
            scratch,
        );
    }

    // --- 5. Embedding backward ---
    // Forward was: hidden[i] = embedding[tokens[i]]
    // grad_token_embedding[token] += grad_hidden[i] (accumulated)
    embedding_bw.embeddingBackward(
        weights.grad_token_embedding.asSliceMut(),
        cache.grad_hidden,
        tokens,
        bs,
        d,
    );
}

/// Backward pass for a single transformer layer.
///
/// grad_hidden: [bs * d] — gradient flowing backward through layers.
///              On entry: gradient from the layer above (or final norm).
///              On exit: gradient to pass to the layer below.
fn layerBackward(
    grad_hidden: []f32,
    la: *const LayerActivations,
    lw: *LayerWeights,
    config: TransformerConfig,
    bs: usize,
    global_scratch: []f32,
    scratch: *MatmulScratch,
) void {
    const d: usize = config.d_model;
    const nh: usize = config.num_heads;
    const nkv: usize = config.num_kv_heads;
    const hd: usize = config.headDim();
    const ff: usize = config.d_ff;
    const s: usize = config.seq_len;
    const b: usize = bs / s;

    // =====================================================================
    // FFN backward
    // =====================================================================
    // Forward was:
    //   residual_pre_ffn = hidden (after attn block)
    //   normed_ffn = rmsnorm(hidden, ffn_norm)
    //   gate = normed_ffn @ gate_proj^T
    //   up = normed_ffn @ up_proj^T
    //   swiglu_out = silu(gate) * up
    //   hidden = down_proj(swiglu_out) + residual_pre_ffn

    // The grad_hidden we received is d(loss)/d(hidden_after_ffn_residual).
    // Save a copy for the residual path.
    // We'll use global_scratch[0..bs*d] for grad_residual_ffn.
    const grad_residual_ffn = global_scratch[0 .. bs * d];
    @memcpy(grad_residual_ffn, grad_hidden[0 .. bs * d]);

    // Down projection backward:
    // Forward: hidden_before_residual = swiglu_out @ down_proj^T
    // grad_down_proj += grad_hidden^T @ swiglu_out
    // grad_swiglu_out = grad_hidden @ down_proj

    // We need swiglu_out, which was computed as silu(gate)*up but not saved.
    // Recompute it into a scratch area. We need bs*ff space.
    const grad_swiglu = global_scratch[bs * d .. bs * d + bs * ff];
    recomputeSwiglu(grad_swiglu, la.gate, la.up, bs * ff);

    // Now grad_swiglu temporarily holds the forward swiglu output (for gradWeight).
    linear.gradWeight(
        lw.grad_down_proj.asSliceMut(),
        grad_hidden,
        grad_swiglu, // input to down_proj was swiglu_out
        bs,
        d,
        ff,
        scratch,
    );

    // grad_swiglu_out = grad_hidden @ down_proj (overwritten)
    linear.gradInput(
        grad_swiglu,
        grad_hidden,
        lw.down_proj.asSlice(f32),
        bs,
        d,
        ff,
        scratch,
    );

    // SwiGLU backward:
    // grad_gate and grad_up from swiglu backward (both overwritten)
    // We need separate buffers for grad_gate and grad_up.
    // Use two regions of global_scratch after grad_residual_ffn.
    const grad_gate = global_scratch[bs * d + bs * ff .. bs * d + 2 * bs * ff];
    const grad_up = global_scratch[bs * d + 2 * bs * ff .. bs * d + 3 * bs * ff];
    activation_bw.swigluBackward(
        grad_gate,
        grad_up,
        grad_swiglu,
        la.gate,
        la.up,
    );

    // Gate projection backward:
    // Forward: gate = normed_ffn @ gate_proj^T
    linear.gradWeight(
        lw.grad_gate_proj.asSliceMut(),
        grad_gate,
        la.normed_ffn,
        bs,
        ff,
        d,
        scratch,
    );
    // grad_normed_ffn from gate path = grad_gate @ gate_proj (overwritten into grad_hidden)
    linear.gradInput(
        grad_hidden,
        grad_gate,
        lw.gate_proj.asSlice(f32),
        bs,
        ff,
        d,
        scratch,
    );

    // Up projection backward:
    // Forward: up = normed_ffn @ up_proj^T
    linear.gradWeight(
        lw.grad_up_proj.asSliceMut(),
        grad_up,
        la.normed_ffn,
        bs,
        ff,
        d,
        scratch,
    );
    // grad_normed_ffn from up path = grad_up @ up_proj
    // Accumulate into grad_hidden (which already has gate path contribution).
    linear.gradInputAccum(
        grad_hidden,
        grad_up,
        lw.up_proj.asSlice(f32),
        bs,
        ff,
        d,
        scratch,
    );

    // FFN RMSNorm backward:
    // Forward: normed_ffn = rmsnorm(residual_pre_ffn, ffn_norm)
    // Copy grad_hidden to avoid aliasing grad_input/grad_output.
    const ffn_norm_grad_out = grad_swiglu[0 .. bs * d]; // reuse scratch region
    @memcpy(ffn_norm_grad_out, grad_hidden[0 .. bs * d]);
    rmsnorm_bw.rmsnormBackward(
        grad_hidden,
        lw.grad_ffn_norm.asSliceMut(),
        ffn_norm_grad_out,
        la.residual_pre_ffn, // input to this rmsnorm
        la.inv_rms_ffn,
        lw.ffn_norm.asSlice(f32),
        bs,
        d,
        0.0,
        grad_residual_ffn, // fused residual add
    );

    // =====================================================================
    // Attention backward
    // =====================================================================
    // Forward was:
    //   residual_pre_attn = hidden (input to this layer)
    //   normed_attn = rmsnorm(hidden, attn_norm)
    //   q = normed_attn @ q_proj^T, then RoPE
    //   k = normed_attn @ k_proj^T, then RoPE
    //   v = normed_attn @ v_proj^T
    //   attn_output = attention(q, k, v)
    //   hidden = attn_output @ o_proj^T + residual_pre_attn

    // Save residual gradient for attention block.
    // Place at start of scratch; attention temporaries go after it.
    const grad_residual_attn = global_scratch[0 .. bs * d];
    @memcpy(grad_residual_attn, grad_hidden[0 .. bs * d]);

    // Output projection backward:
    // Forward: o_proj_out = attn_output @ o_proj^T
    linear.gradWeight(
        lw.grad_o_proj.asSliceMut(),
        grad_hidden,
        la.attn_output,
        bs,
        d,
        nh * hd,
        scratch,
    );
    // grad_attn_output = grad_hidden @ o_proj (overwritten)
    // Place after grad_residual_attn to avoid overlap.
    const grad_attn_output = global_scratch[bs * d .. bs * d + bs * nh * hd];
    linear.gradInput(
        grad_attn_output,
        grad_hidden,
        lw.o_proj.asSlice(f32),
        bs,
        d,
        nh * hd,
        scratch,
    );

    // Attention backward (threaded batch version):
    // Phase 1: d_scores + d_Q (threaded over batch × n_heads × seq_len)
    // Phase 2: d_K + d_V (threaded over batch × n_kv_heads × seq_len)
    // Layout: [grad_residual_attn: bs*d] [grad_attn_output: bs*nh*hd] [grad_q: bs*nh*hd] [grad_k: bs*nkv*hd] [grad_v: bs*nkv*hd] [d_scores: b*nh*s*s]
    const qkv_base = bs * d + bs * nh * hd;
    const grad_q = global_scratch[qkv_base .. qkv_base + bs * nh * hd];
    const grad_k = global_scratch[qkv_base + bs * nh * hd .. qkv_base + bs * nh * hd + bs * nkv * hd];
    const grad_v = global_scratch[qkv_base + bs * nh * hd + bs * nkv * hd .. qkv_base + bs * nh * hd + 2 * bs * nkv * hd];
    const d_scores_base = qkv_base + bs * nh * hd + 2 * bs * nkv * hd;
    const d_scores = global_scratch[d_scores_base .. d_scores_base + b * nh * s * s];

    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

    attention_bw.attentionBackwardBatch(
        grad_q,
        grad_k,
        grad_v,
        grad_attn_output,
        la.q,
        la.k,
        la.v,
        la.attn_probs,
        d_scores,
        b,
        s,
        nh,
        nkv,
        hd,
        scale,
    );

    // RoPE backward on grad_q and grad_k (batch version precomputes inv_freq once)
    rope_bw.ropeBackwardBatch(grad_q[0 .. bs * nh * hd], b, s, nh, hd, hd, config.rope_theta);
    rope_bw.ropeBackwardBatch(grad_k[0 .. bs * nkv * hd], b, s, nkv, hd, hd, config.rope_theta);

    // Fused QKV projection backward.
    // grad_qkv = [grad_q | grad_k | grad_v] is already contiguous in global_scratch.
    const qkv_dim = nh * hd + 2 * nkv * hd;
    const grad_qkv = global_scratch[qkv_base .. qkv_base + bs * qkv_dim];

    // Fused gradWeight: grad_qkv_proj += grad_qkv^T @ normed_attn
    // Write into a contiguous scratch region, then scatter to individual grad tensors.
    const grad_qkv_weight_base = d_scores_base; // reuse d_scores region (no longer needed)
    const grad_qkv_weight = global_scratch[grad_qkv_weight_base .. grad_qkv_weight_base + qkv_dim * d];
    // Initialize from existing gradient accumulators (gradWeight accumulates, not overwrites).
    const q_grad_size = nh * hd * d;
    const kv_grad_size = nkv * hd * d;
    @memcpy(grad_qkv_weight[0..q_grad_size], lw.grad_q_proj.asSlice());
    @memcpy(grad_qkv_weight[q_grad_size .. q_grad_size + kv_grad_size], lw.grad_k_proj.asSlice());
    @memcpy(grad_qkv_weight[q_grad_size + kv_grad_size .. q_grad_size + 2 * kv_grad_size], lw.grad_v_proj.asSlice());

    linear.gradWeight(
        grad_qkv_weight,
        grad_qkv,
        la.normed_attn,
        bs,
        qkv_dim,
        d,
        scratch,
    );

    // Scatter back to individual grad tensors.
    @memcpy(lw.grad_q_proj.asSliceMut(), grad_qkv_weight[0..q_grad_size]);
    @memcpy(lw.grad_k_proj.asSliceMut(), grad_qkv_weight[q_grad_size .. q_grad_size + kv_grad_size]);
    @memcpy(lw.grad_v_proj.asSliceMut(), grad_qkv_weight[q_grad_size + kv_grad_size .. q_grad_size + 2 * kv_grad_size]);

    // Fused gradInput: grad_hidden = grad_qkv @ qkv_proj (single overwrite)
    linear.gradInput(
        grad_hidden,
        grad_qkv,
        lw.qkv_proj_buf,
        bs,
        qkv_dim,
        d,
        scratch,
    );

    // Attention RMSNorm backward:
    // Forward: normed_attn = rmsnorm(residual_pre_attn, attn_norm)
    // Copy grad_hidden to avoid aliasing grad_input/grad_output.
    const attn_norm_grad_out = global_scratch[bs * d .. 2 * bs * d];
    @memcpy(attn_norm_grad_out, grad_hidden[0 .. bs * d]);
    rmsnorm_bw.rmsnormBackward(
        grad_hidden,
        lw.grad_attn_norm.asSliceMut(),
        attn_norm_grad_out,
        la.residual_pre_attn,
        la.inv_rms_attn,
        lw.attn_norm.asSlice(f32),
        bs,
        d,
        0.0,
        grad_residual_attn, // fused residual add
    );
}

/// Recompute silu(gate) * up (SwiGLU forward) for use in down_proj gradWeight.
/// We didn't save this intermediate in the forward pass, so recompute it here.
fn recomputeSwiglu(output: []f32, gate: []const f32, up: []const f32, len: usize) void {
    @setFloatMode(.optimized);
    const fast = @import("../../compute/cpu/math_fast.zig");

    const one: F32Vec = @splat(1.0);
    var i: usize = 0;
    while (i + VEC <= len) : (i += VEC) {
        const x: F32Vec = gate[i..][0..VEC].*;
        const exp_neg = fast.fastExp(-x);
        const sig = one / (one + exp_neg);
        const u: F32Vec = up[i..][0..VEC].*;
        output[i..][0..VEC].* = x * sig * u;
    }
    while (i < len) : (i += 1) {
        const x = gate[i];
        const sigmoid = 1.0 / (1.0 + fast.fastExpScalar(-x));
        output[i] = x * sigmoid * up[i];
    }
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;
const forward_mod = @import("../forward/root.zig");

fn testConfig() TransformerConfig {
    return .{
        .vocab_size = 8,
        .d_model = 4,
        .num_layers = 1,
        .num_heads = 1,
        .num_kv_heads = 1,
        .d_ff = 8,
        .seq_len = 2,
    };
}

test "backward produces non-zero gradients" {
    const config = testConfig();
    var weights = try ModelWeights.init(testing.allocator, config);
    defer weights.deinit();
    weights.initRandom(42);

    var cache = try ActivationCache.init(testing.allocator, config, 1);
    defer cache.deinit();

    var mm_scratch = try MatmulScratch.init(testing.allocator);
    defer mm_scratch.deinit();

    const tokens = [_]u32{ 1, 3 };
    const targets = [_]u32{ 3, 5 };

    // Forward pass
    _ = forward_mod.forward(&weights, &cache, &tokens, &targets, &mm_scratch);

    // Zero grads then backward
    weights.zeroGrads();
    backward(&weights, &cache, &tokens, &targets, &mm_scratch);

    // Check that at least some gradients are non-zero
    var has_nonzero = false;
    for (weights.grad_lm_head.asSlice()) |v| {
        if (v != 0.0) {
            has_nonzero = true;
            break;
        }
    }
    try testing.expect(has_nonzero);

    has_nonzero = false;
    for (weights.grad_token_embedding.asSlice()) |v| {
        if (v != 0.0) {
            has_nonzero = true;
            break;
        }
    }
    try testing.expect(has_nonzero);

    has_nonzero = false;
    for (weights.layers[0].grad_q_proj.asSlice()) |v| {
        if (v != 0.0) {
            has_nonzero = true;
            break;
        }
    }
    try testing.expect(has_nonzero);
}

test "backward gradient finite check" {
    const config = testConfig();
    var weights = try ModelWeights.init(testing.allocator, config);
    defer weights.deinit();
    weights.initRandom(123);

    var cache = try ActivationCache.init(testing.allocator, config, 1);
    defer cache.deinit();

    var mm_scratch = try MatmulScratch.init(testing.allocator);
    defer mm_scratch.deinit();

    const tokens = [_]u32{ 0, 2 };
    const targets = [_]u32{ 2, 4 };

    _ = forward_mod.forward(&weights, &cache, &tokens, &targets, &mm_scratch);
    weights.zeroGrads();
    backward(&weights, &cache, &tokens, &targets, &mm_scratch);

    // All gradients should be finite (not NaN or Inf)
    for (weights.grad_lm_head.asSlice()) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
    for (weights.grad_final_norm.asSlice()) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
    for (weights.grad_token_embedding.asSlice()) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
    for (weights.layers[0].grad_q_proj.asSlice()) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
    for (weights.layers[0].grad_o_proj.asSlice()) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
    for (weights.layers[0].grad_gate_proj.asSlice()) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
    for (weights.layers[0].grad_down_proj.asSlice()) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

test "backward with two layers produces non-zero gradients in both" {
    const config: TransformerConfig = .{
        .vocab_size = 8,
        .d_model = 4,
        .num_layers = 2,
        .num_heads = 1,
        .num_kv_heads = 1,
        .d_ff = 8,
        .seq_len = 2,
    };

    var weights = try ModelWeights.init(testing.allocator, config);
    defer weights.deinit();
    weights.initRandom(99);

    var cache = try ActivationCache.init(testing.allocator, config, 1);
    defer cache.deinit();

    var mm_scratch = try MatmulScratch.init(testing.allocator);
    defer mm_scratch.deinit();

    const tokens = [_]u32{ 0, 1 };
    const targets = [_]u32{ 1, 2 };

    _ = forward_mod.forward(&weights, &cache, &tokens, &targets, &mm_scratch);
    weights.zeroGrads();
    backward(&weights, &cache, &tokens, &targets, &mm_scratch);

    // Both layers should have non-zero gradients
    for (0..2) |layer_idx| {
        var has_nonzero = false;
        for (weights.layers[layer_idx].grad_q_proj.asSlice()) |v| {
            if (v != 0.0) {
                has_nonzero = true;
                break;
            }
        }
        try testing.expect(has_nonzero);
    }
}
