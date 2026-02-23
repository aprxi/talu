//! Shared CUDA attention path policy.
//!
//! Keeps decode/prefill path selection consistent and testable.

const std = @import("std");

pub const Config = struct {
    kv_cache_dtype_fp16: bool,
    enable_fused_attention_f16_kv: bool,
    max_fused_attention_f16_kv_seq_len: u32,
    max_supported_fused_f16_kv_head_dim: u32,
};

pub fn needAttentionScoreBuffers(config: Config, max_seq_len: usize, head_dim: usize) bool {
    return !config.kv_cache_dtype_fp16 or
        !config.enable_fused_attention_f16_kv or
        head_dim > config.max_supported_fused_f16_kv_head_dim or
        max_seq_len > config.max_fused_attention_f16_kv_seq_len;
}

pub fn effectiveAttentionSeqLen(seq_len_u32: u32, sliding_window: usize, is_causal: bool) u32 {
    if (!(sliding_window > 0 and is_causal)) return seq_len_u32;
    const window_u32 = std.math.cast(u32, sliding_window) orelse std.math.maxInt(u32);
    return if (seq_len_u32 > window_u32) window_u32 else seq_len_u32;
}

pub fn canUseFusedAttentionHeadsF16Kv(
    config: Config,
    effective_seq_len_u32: u32,
    head_dim_u32: u32,
    fused_kernel_available: bool,
) bool {
    return config.kv_cache_dtype_fp16 and
        config.enable_fused_attention_f16_kv and
        effective_seq_len_u32 <= config.max_fused_attention_f16_kv_seq_len and
        head_dim_u32 <= config.max_supported_fused_f16_kv_head_dim and
        fused_kernel_available;
}

test "effectiveAttentionSeqLen respects sliding causal window" {
    try std.testing.expectEqual(@as(u32, 128), effectiveAttentionSeqLen(128, 0, true));
    try std.testing.expectEqual(@as(u32, 128), effectiveAttentionSeqLen(128, 64, false));
    try std.testing.expectEqual(@as(u32, 64), effectiveAttentionSeqLen(128, 64, true));
}

test "canUseFusedAttentionHeadsF16Kv enforces sequence threshold" {
    const cfg = Config{
        .kv_cache_dtype_fp16 = true,
        .enable_fused_attention_f16_kv = true,
        .max_fused_attention_f16_kv_seq_len = 384,
        .max_supported_fused_f16_kv_head_dim = 512,
    };
    try std.testing.expect(canUseFusedAttentionHeadsF16Kv(cfg, 256, 128, true));
    try std.testing.expect(!canUseFusedAttentionHeadsF16Kv(cfg, 385, 128, true));
    try std.testing.expect(!canUseFusedAttentionHeadsF16Kv(cfg, 256, 513, true));
    try std.testing.expect(!canUseFusedAttentionHeadsF16Kv(cfg, 256, 128, false));
}

test "needAttentionScoreBuffers follows fused policy envelope" {
    const cfg = Config{
        .kv_cache_dtype_fp16 = true,
        .enable_fused_attention_f16_kv = true,
        .max_fused_attention_f16_kv_seq_len = 384,
        .max_supported_fused_f16_kv_head_dim = 512,
    };
    try std.testing.expect(!needAttentionScoreBuffers(cfg, 256, 128));
    try std.testing.expect(needAttentionScoreBuffers(cfg, 1024, 128));
    try std.testing.expect(needAttentionScoreBuffers(cfg, 256, 1024));
}
