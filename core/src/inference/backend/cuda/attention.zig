//! Attention path helpers for CUDA backend.

const std = @import("std");
const attention_policy = @import("attention_policy.zig");

pub fn effectiveSeqLen(config: attention_policy.Config, seq_len_u32: u32, sliding_window: usize, is_causal: bool) u32 {
    _ = config;
    return attention_policy.effectiveAttentionSeqLen(seq_len_u32, sliding_window, is_causal);
}

pub fn useFusedHeadsF16Kv(
    config: attention_policy.Config,
    seq_len_u32: u32,
    sliding_window: usize,
    is_causal: bool,
    head_dim_u32: u32,
    fused_kernel_available: bool,
) bool {
    const effective_seq_len_u32 = attention_policy.effectiveAttentionSeqLen(seq_len_u32, sliding_window, is_causal);
    return attention_policy.canUseFusedAttentionHeadsF16Kv(
        config,
        effective_seq_len_u32,
        head_dim_u32,
        fused_kernel_available,
    );
}

test "useFusedHeadsF16Kv respects sliding window and threshold" {
    const cfg = attention_policy.Config{
        .kv_cache_dtype_fp16 = true,
        .enable_fused_attention_f16_kv = true,
        .max_fused_attention_f16_kv_seq_len = 384,
        .max_supported_fused_f16_kv_head_dim = 512,
    };

    try std.testing.expect(useFusedHeadsF16Kv(cfg, 512, 256, true, 128, true));
    try std.testing.expect(!useFusedHeadsF16Kv(cfg, 512, 0, true, 128, true));
}
