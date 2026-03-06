//! Activation cache for training backward pass.
//!
//! Pre-allocates all intermediate buffers needed by the backward pass.
//! Each forward pass writes into these buffers; the backward pass reads them.
//! Buffers are reused across training steps (no per-step allocation).
//!
//! Memory layout is flat f32 slices sized for [batch_size * seq_len * dim].

const std = @import("std");
const model_config = @import("model_config.zig");

const Allocator = std.mem.Allocator;
const TransformerConfig = model_config.TransformerConfig;

/// Saved activations for a single transformer layer's backward pass.
pub const LayerActivations = struct {
    /// Input to the layer before attention RMSNorm (for residual gradient).
    /// [batch * seq * d_model]
    residual_pre_attn: []f32,

    /// Output of attention RMSNorm (input to Q/K/V projections).
    /// [batch * seq * d_model]
    normed_attn: []f32,

    /// Inverse RMS values saved by RMSNorm for backward.
    /// [batch * seq]
    inv_rms_attn: []f32,

    /// Contiguous QKV buffer backing q, k, v slices.
    /// [batch * seq * (num_heads * head_dim + 2 * num_kv_heads * head_dim)]
    qkv: []f32,

    /// Query vectors after projection and RoPE (view into qkv).
    /// [batch * seq * num_heads * head_dim]
    q: []f32,

    /// Key vectors after projection and RoPE (view into qkv).
    /// [batch * seq * num_kv_heads * head_dim]
    k: []f32,

    /// Value vectors after projection (view into qkv).
    /// [batch * seq * num_kv_heads * head_dim]
    v: []f32,

    /// Attention output before output projection.
    /// [batch * seq * num_heads * head_dim]
    attn_output: []f32,

    /// Attention probabilities (post-softmax).
    /// [batch * num_heads * seq * seq]
    attn_probs: []f32,

    /// Input to the layer before FFN RMSNorm (for residual gradient).
    /// [batch * seq * d_model]
    residual_pre_ffn: []f32,

    /// Output of FFN RMSNorm (input to gate/up projections).
    /// [batch * seq * d_model]
    normed_ffn: []f32,

    /// Inverse RMS values saved by FFN RMSNorm.
    /// [batch * seq]
    inv_rms_ffn: []f32,

    /// Gate projection output (pre-activation).
    /// [batch * seq * d_ff]
    gate: []f32,

    /// Up projection output.
    /// [batch * seq * d_ff]
    up: []f32,
};

/// Pre-allocated activation cache for the entire model.
pub const ActivationCache = struct {
    allocator: Allocator,
    config: TransformerConfig,
    batch_size: u32,

    layers: []LayerActivations,

    /// Final RMSNorm inverse RMS values.
    /// [batch * seq]
    final_inv_rms: []f32,

    /// Output of final RMSNorm (input to LM head).
    /// [batch * seq * d_model]
    final_normed: []f32,

    /// Logits output from LM head.
    /// [batch * seq * vocab_size]
    logits: []f32,

    /// Scratch buffer for the hidden state flowing through layers.
    /// [batch * seq * d_model]
    hidden: []f32,

    /// Gradient scratch for hidden state during backward.
    /// [batch * seq * d_model]
    grad_hidden: []f32,

    /// General scratch buffer for intermediate backward computations.
    /// Sized to max(d_model, d_ff, num_heads * head_dim) * batch * seq.
    scratch: []f32,

    pub fn init(allocator: Allocator, config: TransformerConfig, batch_size: u32) !ActivationCache {
        const b: usize = batch_size;
        const s: usize = config.seq_len;
        const d: usize = config.d_model;
        const ff: usize = config.d_ff;
        const nh: usize = config.num_heads;
        const nkv: usize = config.num_kv_heads;
        const hd: usize = config.headDim();
        const v: usize = config.vocab_size;

        const bs = b * s;
        const bsd = bs * d;
        const bsff = bs * ff;

        var self: ActivationCache = undefined;
        self.allocator = allocator;
        self.config = config;
        self.batch_size = batch_size;

        self.final_inv_rms = try allocator.alloc(f32, bs);
        errdefer allocator.free(self.final_inv_rms);

        self.final_normed = try allocator.alloc(f32, bsd);
        errdefer allocator.free(self.final_normed);

        self.logits = try allocator.alloc(f32, bs * v);
        errdefer allocator.free(self.logits);

        self.hidden = try allocator.alloc(f32, bsd);
        errdefer allocator.free(self.hidden);

        self.grad_hidden = try allocator.alloc(f32, bsd);
        errdefer allocator.free(self.grad_hidden);

        // Scratch sized for backward pass peak usage (total elements, not per-token):
        //   FFN backward:  bs*(d + 3*ff)  (residual + swiglu recompute + grad_gate + grad_up)
        //   Attn backward: bs*(d + 2*nh*hd + 2*nkv*hd) + max(b*nh*s*s, qkv_dim*d)
        //                  (residual + attn_output + grad_q/k/v + d_scores OR grad_qkv_weight)
        //   CE backward:   bs*v  (grad_logits)
        const ffn_total = bs * (d + 3 * ff);
        const qkv_dim = nh * hd + 2 * nkv * hd;
        const attn_base = bs * (d + 2 * nh * hd + 2 * nkv * hd);
        const attn_tail = @max(b * nh * s * s, qkv_dim * d);
        const attn_total = attn_base + attn_tail;
        const ce_total = bs * v;
        self.scratch = try allocator.alloc(f32, @max(ce_total, @max(ffn_total, attn_total)));
        errdefer allocator.free(self.scratch);

        self.layers = try allocator.alloc(LayerActivations, config.num_layers);
        var initialized_layers: u32 = 0;
        errdefer {
            for (self.layers[0..initialized_layers]) |*la| {
                freeLayerActivations(allocator, la);
            }
            allocator.free(self.layers);
        }

        for (self.layers) |*la| {
            la.residual_pre_attn = try allocator.alloc(f32, bsd);
            errdefer allocator.free(la.residual_pre_attn);
            la.normed_attn = try allocator.alloc(f32, bsd);
            errdefer allocator.free(la.normed_attn);
            la.inv_rms_attn = try allocator.alloc(f32, bs);
            errdefer allocator.free(la.inv_rms_attn);
            const q_size = bs * nh * hd;
            const kv_size = bs * nkv * hd;
            la.qkv = try allocator.alloc(f32, q_size + 2 * kv_size);
            errdefer allocator.free(la.qkv);
            la.q = la.qkv[0..q_size];
            la.k = la.qkv[q_size .. q_size + kv_size];
            la.v = la.qkv[q_size + kv_size .. q_size + 2 * kv_size];
            la.attn_output = try allocator.alloc(f32, bs * nh * hd);
            errdefer allocator.free(la.attn_output);
            la.attn_probs = try allocator.alloc(f32, b * nh * s * s);
            errdefer allocator.free(la.attn_probs);
            la.residual_pre_ffn = try allocator.alloc(f32, bsd);
            errdefer allocator.free(la.residual_pre_ffn);
            la.normed_ffn = try allocator.alloc(f32, bsd);
            errdefer allocator.free(la.normed_ffn);
            la.inv_rms_ffn = try allocator.alloc(f32, bs);
            errdefer allocator.free(la.inv_rms_ffn);
            la.gate = try allocator.alloc(f32, bsff);
            errdefer allocator.free(la.gate);
            la.up = try allocator.alloc(f32, bsff);
            // No errdefer needed for last alloc — initialized_layers covers it.

            initialized_layers += 1;
        }

        return self;
    }

    pub fn deinit(self: *ActivationCache) void {
        for (self.layers) |*la| {
            freeLayerActivations(self.allocator, la);
        }
        self.allocator.free(self.layers);
        self.allocator.free(self.scratch);
        self.allocator.free(self.grad_hidden);
        self.allocator.free(self.hidden);
        self.allocator.free(self.logits);
        self.allocator.free(self.final_normed);
        self.allocator.free(self.final_inv_rms);
        self.* = undefined;
    }
};

fn freeLayerActivations(allocator: Allocator, la: *LayerActivations) void {
    allocator.free(la.up);
    allocator.free(la.gate);
    allocator.free(la.inv_rms_ffn);
    allocator.free(la.normed_ffn);
    allocator.free(la.residual_pre_ffn);
    allocator.free(la.attn_probs);
    allocator.free(la.attn_output);
    allocator.free(la.qkv);
    allocator.free(la.inv_rms_attn);
    allocator.free(la.normed_attn);
    allocator.free(la.residual_pre_attn);
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

fn testConfig() TransformerConfig {
    return .{
        .vocab_size = 32,
        .d_model = 16,
        .num_layers = 2,
        .num_heads = 2,
        .num_kv_heads = 2,
        .d_ff = 32,
        .seq_len = 8,
    };
}

test "ActivationCache init and deinit" {
    const config = testConfig();
    var cache = try ActivationCache.init(testing.allocator, config, 4);
    defer cache.deinit();

    try testing.expectEqual(@as(usize, 2), cache.layers.len);
    try testing.expectEqual(@as(u32, 4), cache.batch_size);
}

test "ActivationCache buffer sizes match config" {
    const config = testConfig();
    const batch: u32 = 4;
    var cache = try ActivationCache.init(testing.allocator, config, batch);
    defer cache.deinit();

    const bs: usize = 4 * 8; // batch * seq
    const bsd: usize = bs * 16; // batch * seq * d_model

    // Global buffers
    try testing.expectEqual(bs, cache.final_inv_rms.len);
    try testing.expectEqual(bsd, cache.final_normed.len);
    try testing.expectEqual(bs * 32, cache.logits.len); // batch * seq * vocab
    try testing.expectEqual(bsd, cache.hidden.len);
    try testing.expectEqual(bsd, cache.grad_hidden.len);

    // Per-layer buffers
    const la = &cache.layers[0];
    try testing.expectEqual(bsd, la.residual_pre_attn.len);
    try testing.expectEqual(bsd, la.normed_attn.len);
    try testing.expectEqual(bs, la.inv_rms_attn.len);
    // q: batch * seq * num_heads * head_dim = 4 * 8 * 2 * 8 = 512
    try testing.expectEqual(@as(usize, 4 * 8 * 2 * 8), la.q.len);
    // attn_probs: batch * num_heads * seq * seq = 4 * 2 * 8 * 8 = 512
    try testing.expectEqual(@as(usize, 4 * 2 * 8 * 8), la.attn_probs.len);
    // gate: batch * seq * d_ff = 4 * 8 * 32 = 1024
    try testing.expectEqual(@as(usize, 4 * 8 * 32), la.gate.len);
}

test "ActivationCache layer activations are writable" {
    const config = testConfig();
    var cache = try ActivationCache.init(testing.allocator, config, 2);
    defer cache.deinit();

    // Verify we can write to activation buffers
    cache.layers[0].residual_pre_attn[0] = 1.0;
    cache.layers[1].gate[0] = 2.0;
    cache.hidden[0] = 3.0;
    cache.logits[0] = 4.0;

    try testing.expectEqual(@as(f32, 1.0), cache.layers[0].residual_pre_attn[0]);
    try testing.expectEqual(@as(f32, 2.0), cache.layers[1].gate[0]);
    try testing.expectEqual(@as(f32, 3.0), cache.hidden[0]);
    try testing.expectEqual(@as(f32, 4.0), cache.logits[0]);
}
