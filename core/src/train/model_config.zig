//! Transformer architecture configuration for from-scratch training.
//!
//! Defines the model structure: vocabulary, dimensions, layers, heads, etc.
//! Used by ModelWeights to allocate weight tensors and by forward/backward
//! passes to determine tensor shapes.

/// Architecture configuration for a decoder-only transformer.
///
/// Matches the common Llama/Qwen/Gemma architecture:
///   embed → [RMSNorm → Attention → Residual → RMSNorm → SwiGLU → Residual] × L → RMSNorm → Linear
pub const TransformerConfig = struct {
    vocab_size: u32,
    d_model: u32,
    num_layers: u32,
    num_heads: u32,
    num_kv_heads: u32,
    d_ff: u32,
    seq_len: u32,
    rope_theta: f32 = 10000.0,
    norm_eps: f32 = 1e-5,

    /// Head dimension derived from d_model / num_heads.
    pub fn headDim(self: TransformerConfig) u32 {
        return self.d_model / self.num_heads;
    }

    /// Number of query heads per KV head (for grouped-query attention).
    pub fn headsPerGroup(self: TransformerConfig) u32 {
        return self.num_heads / self.num_kv_heads;
    }

    /// Total number of weight parameters (approximate, excludes optimizer state).
    pub fn totalParams(self: TransformerConfig) u64 {
        const d: u64 = self.d_model;
        const v: u64 = self.vocab_size;
        const ff: u64 = self.d_ff;
        const hd: u64 = self.headDim();
        const nh: u64 = self.num_heads;
        const nkv: u64 = self.num_kv_heads;
        const nl: u64 = self.num_layers;

        const embedding = v * d;
        const per_layer =
            d + // attn_norm
            nh * hd * d + // q_proj
            nkv * hd * d + // k_proj
            nkv * hd * d + // v_proj
            nh * hd * d + // o_proj (d_model × d_model for MHA)
            d + // ffn_norm
            ff * d + // gate_proj
            ff * d + // up_proj
            d * ff; // down_proj
        const final_norm = d;
        const lm_head = v * d;

        return embedding + nl * per_layer + final_norm + lm_head;
    }
};

test "TransformerConfig headDim" {
    const config = TransformerConfig{
        .vocab_size = 512,
        .d_model = 256,
        .num_layers = 4,
        .num_heads = 4,
        .num_kv_heads = 4,
        .d_ff = 1024,
        .seq_len = 256,
    };
    try @import("std").testing.expectEqual(@as(u32, 64), config.headDim());
}

test "TransformerConfig headsPerGroup MHA" {
    const config = TransformerConfig{
        .vocab_size = 512,
        .d_model = 256,
        .num_layers = 4,
        .num_heads = 4,
        .num_kv_heads = 4,
        .d_ff = 1024,
        .seq_len = 256,
    };
    try @import("std").testing.expectEqual(@as(u32, 1), config.headsPerGroup());
}

test "TransformerConfig headsPerGroup GQA" {
    const config = TransformerConfig{
        .vocab_size = 512,
        .d_model = 256,
        .num_layers = 4,
        .num_heads = 4,
        .num_kv_heads = 2,
        .d_ff = 1024,
        .seq_len = 256,
    };
    try @import("std").testing.expectEqual(@as(u32, 2), config.headsPerGroup());
}

test "TransformerConfig totalParams shakespeare" {
    const config = TransformerConfig{
        .vocab_size = 512,
        .d_model = 256,
        .num_layers = 4,
        .num_heads = 4,
        .num_kv_heads = 4,
        .d_ff = 1024,
        .seq_len = 256,
    };
    const params = config.totalParams();
    // Embedding: 512*256 = 131072
    // Per layer: 256 + 256*256 + 256*256 + 256*256 + 256*256 + 256 + 1024*256 + 1024*256 + 256*1024
    //          = 256 + 65536 + 65536 + 65536 + 65536 + 256 + 262144 + 262144 + 262144 = 1,048,588
    // ×4 layers = 4,194,352 (approximate, rounding might differ)
    // Final norm: 256
    // LM head: 512*256 = 131072
    // Total should be around 4.5M
    try @import("std").testing.expect(params > 4_000_000 and params < 5_000_000);
}
