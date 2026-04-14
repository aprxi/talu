//! Training forward pass orchestration for decoder-only transformers.
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
const compute = @import("compute_pkg");
const model_config = @import("../model_config.zig");
const model_weights_mod = @import("../model_weights.zig");
const activations_mod = @import("../activations.zig");

const MatmulScratch = compute.cpu.linalg.MatmulScratch;
const TransformerConfig = model_config.TransformerConfig;
const ModelWeights = model_weights_mod.ModelWeights;
const LayerWeights = model_weights_mod.LayerWeights;
const ActivationCache = activations_mod.ActivationCache;
const LayerActivations = activations_mod.LayerActivations;

// Forward kernels
const linear_fwd = @import("linear.zig");
const attention_fwd = @import("attention.zig");
const norm_fwd = @import("norm.zig");
const activation_fwd = @import("activation.zig");
const rope_fwd = @import("rope.zig");
const embedding_fwd = @import("embedding.zig");
const loss_fwd = @import("loss.zig");

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
    embedding_fwd.embeddingForward(cache.hidden, weights.token_embedding.asSlice(f32), tokens, d);

    // 2. Transformer layers
    for (cache.layers, weights.layers) |*la, *lw| {
        layerForward(cache.hidden, la, lw, config, bs, cache.scratch, scratch);
    }

    // 3. Final RMSNorm
    norm_fwd.rmsnormForwardSave(
        cache.final_normed,
        cache.final_inv_rms,
        cache.hidden,
        weights.final_norm.asSlice(f32),
        config.norm_eps,
        bs,
        d,
    );

    // 4. LM Head: logits = final_normed @ lm_head^T
    linear_fwd.lmHeadForward(
        cache.logits,
        cache.final_normed,
        weights.lm_head.asSlice(f32),
        bs,
        d,
        config.vocab_size,
        scratch,
    );

    // 5. Cross-entropy loss
    return loss_fwd.crossEntropyLoss(cache.logits, targets, bs, config.vocab_size);
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
    norm_fwd.rmsnormForwardSave(la.normed_attn, la.inv_rms_attn, hidden, lw.attn_norm.asSlice(f32), config.norm_eps, bs, d);

    // Fused QKV projection: [bs, d] @ [qkv_dim, d]^T → [bs, qkv_dim]
    // la.qkv is contiguous [q | k | v], lw.qkv_proj_buf is [q_proj; k_proj; v_proj].
    const qkv_dim = nh * hd + 2 * nkv * hd;
    linear_fwd.linearForward(la.qkv, la.normed_attn, lw.qkv_proj_buf, bs, d, qkv_dim, scratch);

    // Apply RoPE to Q and K (batch version precomputes inv_freq once)
    rope_fwd.ropeForwardBatch(la.q[0 .. bs * nh * hd], b, s, nh, hd, hd, config.rope_theta);
    rope_fwd.ropeForwardBatch(la.k[0 .. bs * nkv * hd], b, s, nkv, hd, hd, config.rope_theta);

    // Scaled dot-product attention
    attention_fwd.attentionForward(la.attn_output, la.attn_probs, la.q, la.k, la.v, b, s, nh, nkv, hd);

    // Output projection: [bs, nh*hd] @ [d, nh*hd]^T → [bs, d]
    linear_fwd.linearForward(hidden, la.attn_output, lw.o_proj.asSlice(f32), bs, nh * hd, d, scratch);

    // Add attention residual
    for (hidden[0 .. bs * d], la.residual_pre_attn[0 .. bs * d]) |*h, r| {
        h.* += r;
    }

    // Save residual for FFN
    @memcpy(la.residual_pre_ffn[0 .. bs * d], hidden[0 .. bs * d]);

    // FFN RMSNorm
    norm_fwd.rmsnormForwardSave(la.normed_ffn, la.inv_rms_ffn, hidden, lw.ffn_norm.asSlice(f32), config.norm_eps, bs, d);

    // SwiGLU FFN: output = down_proj(silu(gate_proj(x)) * up_proj(x))
    linear_fwd.linearForward(la.gate, la.normed_ffn, lw.gate_proj.asSlice(f32), bs, d, ff, scratch);
    linear_fwd.linearForward(la.up, la.normed_ffn, lw.up_proj.asSlice(f32), bs, d, ff, scratch);

    // Compute silu(gate) * up into ffn_scratch [bs * ff]
    activation_fwd.swigluForward(ffn_scratch[0 .. bs * ff], la.gate, la.up, bs * ff);

    // Down projection: [bs, ff] → [bs, d]
    linear_fwd.linearForward(hidden, ffn_scratch, lw.down_proj.asSlice(f32), bs, ff, d, scratch);

    // Add FFN residual
    for (hidden[0 .. bs * d], la.residual_pre_ffn[0 .. bs * d]) |*h, r| {
        h.* += r;
    }
}
