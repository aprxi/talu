//! Integration tests for xray.LayerGeometry
//!
//! LayerGeometry computes derived dimensions and parameter counts for a
//! transformer layer given attention and FFN configurations. Used for
//! performance estimation and model analysis.

const std = @import("std");
const main = @import("main");
const xray = main.xray;

const LayerGeometry = xray.LayerGeometry;
const AttnConfig = xray.AttnConfig;
const FfnConfig = xray.FfnConfig;

test "LayerGeometry: standard attention config" {
    const attn = AttnConfig{
        .n_heads = 32,
        .n_kv_heads = 8,
        .head_dim = 128,
        .d_model = 4096,
    };
    const ffn = FfnConfig{ .swiglu = .{
        .d_model = 4096,
        .d_ff = 14336,
    } };

    const geom = LayerGeometry.init(attn, ffn);

    // q_dim = n_heads * head_dim = 32 * 128 = 4096
    try std.testing.expectEqual(@as(usize, 4096), geom.q_dim);
    // kv_dim = n_kv_heads * head_dim = 8 * 128 = 1024
    try std.testing.expectEqual(@as(usize, 1024), geom.kv_dim);

    // QKV projections should be calculated
    try std.testing.expect(geom.qkv_proj_weights > 0);
    // FFN weights
    try std.testing.expect(geom.ffn_weights > 0);
    // No biases in this config
    try std.testing.expectEqual(@as(usize, 0), geom.attn_bias_params);
}

test "LayerGeometry: with biases" {
    const attn = AttnConfig{
        .n_heads = 16,
        .n_kv_heads = 4,
        .head_dim = 64,
        .d_model = 1024,
        .has_q_bias = true,
        .has_k_bias = true,
        .has_v_bias = true,
        .has_o_bias = true,
    };
    const ffn = FfnConfig{ .swiglu = .{
        .d_model = 1024,
        .d_ff = 4096,
    } };

    const geom = LayerGeometry.init(attn, ffn);

    // Should have bias params
    try std.testing.expect(geom.attn_bias_params > 0);
}

test "LayerGeometry: MoE config" {
    const attn = AttnConfig{
        .n_heads = 8,
        .n_kv_heads = 8,
        .head_dim = 128,
        .d_model = 1024,
    };
    const ffn = FfnConfig{ .moe_ffn = .{
        .d_model = 1024,
        .d_ff = 2816,
        .num_experts = 8,
        .experts_per_token = 2,
    } };

    const geom = LayerGeometry.init(attn, ffn);

    // Router: d_model * num_experts = 1024 * 8
    try std.testing.expectEqual(@as(usize, 1024 * 8), geom.router_weights);
    // Expert weights should be set
    try std.testing.expect(geom.expert_weights > 0);
}

test "LayerGeometry: GQA (grouped query attention)" {
    // Test with n_kv_heads < n_heads (GQA)
    const attn = AttnConfig{
        .n_heads = 32,
        .n_kv_heads = 4, // GQA with 8:1 ratio
        .head_dim = 128,
        .d_model = 4096,
    };
    const ffn = FfnConfig{ .swiglu = .{
        .d_model = 4096,
        .d_ff = 11008,
    } };

    const geom = LayerGeometry.init(attn, ffn);

    // q_dim = n_heads * head_dim = 32 * 128 = 4096
    try std.testing.expectEqual(@as(usize, 4096), geom.q_dim);
    // kv_dim = n_kv_heads * head_dim = 4 * 128 = 512
    try std.testing.expectEqual(@as(usize, 512), geom.kv_dim);
}

test "LayerGeometry: MHA (multi-head attention, n_kv_heads == n_heads)" {
    const attn = AttnConfig{
        .n_heads = 12,
        .n_kv_heads = 12, // MHA
        .head_dim = 64,
        .d_model = 768,
    };
    const ffn = FfnConfig{ .swiglu = .{
        .d_model = 768,
        .d_ff = 3072,
    } };

    const geom = LayerGeometry.init(attn, ffn);

    // q_dim == kv_dim for MHA
    try std.testing.expectEqual(geom.q_dim, geom.kv_dim);
}
