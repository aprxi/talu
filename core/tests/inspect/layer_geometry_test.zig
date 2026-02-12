//! Integration tests for LayerGeometry
//!
//! LayerGeometry calculates transformer layer parameter counts and geometry
//! from attention and FFN configurations.

const std = @import("std");
const main = @import("main");
const perf_estimate = main.inspect.perf_estimate;
const LayerGeometry = perf_estimate.LayerGeometry;
const AttnConfig = perf_estimate.AttnConfig;
const FfnConfig = perf_estimate.FfnConfig;

// =============================================================================
// Basic Initialization Tests
// =============================================================================

test "LayerGeometry.init calculates attention dimensions" {
    const attn = AttnConfig{
        .n_heads = 8,
        .n_kv_heads = 8,
        .head_dim = 64,
        .d_model = 512,
    };

    const ffn = FfnConfig{ .swiglu = .{
        .d_model = 512,
        .d_ff = 2048,
    } };

    const geom = LayerGeometry.init(attn, ffn);

    // q_dim = n_heads * head_dim = 8 * 64 = 512
    try std.testing.expectEqual(@as(usize, 512), geom.q_dim);
    // kv_dim = n_kv_heads * head_dim = 8 * 64 = 512
    try std.testing.expectEqual(@as(usize, 512), geom.kv_dim);
}

test "LayerGeometry.init with GQA (grouped query attention)" {
    const attn = AttnConfig{
        .n_heads = 32,
        .n_kv_heads = 8, // GQA: fewer KV heads
        .head_dim = 128,
        .d_model = 4096,
    };

    const ffn = FfnConfig{ .swiglu = .{
        .d_model = 4096,
        .d_ff = 11008,
    } };

    const geom = LayerGeometry.init(attn, ffn);

    // q_dim = 32 * 128 = 4096
    try std.testing.expectEqual(@as(usize, 4096), geom.q_dim);
    // kv_dim = 8 * 128 = 1024 (smaller due to GQA)
    try std.testing.expectEqual(@as(usize, 1024), geom.kv_dim);
}

// =============================================================================
// QKV Projection Weight Tests
// =============================================================================

test "LayerGeometry calculates QKV projection weights" {
    const attn = AttnConfig{
        .n_heads = 4,
        .n_kv_heads = 4,
        .head_dim = 32,
        .d_model = 128,
    };

    const ffn = FfnConfig{ .swiglu = .{
        .d_model = 128,
        .d_ff = 512,
    } };

    const geom = LayerGeometry.init(attn, ffn);

    // q_proj = d_model * q_dim = 128 * 128 = 16384
    // k_proj = d_model * kv_dim = 128 * 128 = 16384
    // v_proj = d_model * kv_dim = 128 * 128 = 16384
    // o_proj = q_dim * d_model = 128 * 128 = 16384
    // total = 65536
    try std.testing.expectEqual(@as(usize, 65536), geom.qkv_proj_weights);
}

test "LayerGeometry with GQA has fewer KV projection weights" {
    const attn = AttnConfig{
        .n_heads = 8,
        .n_kv_heads = 2, // 4x fewer KV heads
        .head_dim = 64,
        .d_model = 512,
    };

    const ffn = FfnConfig{ .swiglu = .{
        .d_model = 512,
        .d_ff = 2048,
    } };

    const geom = LayerGeometry.init(attn, ffn);

    // q_dim = 8 * 64 = 512
    // kv_dim = 2 * 64 = 128
    // q_proj = 512 * 512 = 262144
    // k_proj = 512 * 128 = 65536
    // v_proj = 512 * 128 = 65536
    // o_proj = 512 * 512 = 262144
    // total = 655360
    try std.testing.expectEqual(@as(usize, 655360), geom.qkv_proj_weights);
}

// =============================================================================
// Attention Bias Tests
// =============================================================================

test "LayerGeometry counts attention biases" {
    const attn = AttnConfig{
        .n_heads = 4,
        .n_kv_heads = 4,
        .head_dim = 32,
        .d_model = 128,
        .has_q_bias = true,
        .has_k_bias = true,
        .has_v_bias = true,
        .has_o_bias = true,
    };

    const ffn = FfnConfig{ .swiglu = .{
        .d_model = 128,
        .d_ff = 512,
    } };

    const geom = LayerGeometry.init(attn, ffn);

    // q_bias = q_dim = 128
    // k_bias = kv_dim = 128
    // v_bias = kv_dim = 128
    // o_bias = d_model = 128
    // total = 512
    try std.testing.expectEqual(@as(usize, 512), geom.attn_bias_params);
}

test "LayerGeometry with no biases has zero bias params" {
    const attn = AttnConfig{
        .n_heads = 4,
        .n_kv_heads = 4,
        .head_dim = 32,
        .d_model = 128,
    };

    const ffn = FfnConfig{ .swiglu = .{
        .d_model = 128,
        .d_ff = 512,
    } };

    const geom = LayerGeometry.init(attn, ffn);

    try std.testing.expectEqual(@as(usize, 0), geom.attn_bias_params);
}

// =============================================================================
// SwiGLU FFN Tests
// =============================================================================

test "LayerGeometry calculates SwiGLU FFN weights" {
    const attn = AttnConfig{
        .n_heads = 8,
        .n_kv_heads = 8,
        .head_dim = 64,
        .d_model = 512,
    };

    const ffn = FfnConfig{ .swiglu = .{
        .d_model = 512,
        .d_ff = 2048,
    } };

    const geom = LayerGeometry.init(attn, ffn);

    // SwiGLU: gate, up, down projections
    // ffn_weights = d_model * d_ff * 3 = 512 * 2048 * 3 = 3145728
    try std.testing.expectEqual(@as(usize, 3145728), geom.ffn_weights);
    try std.testing.expectEqual(@as(usize, 0), geom.router_weights);
    try std.testing.expectEqual(@as(usize, 0), geom.expert_weights);
}

// =============================================================================
// MoE FFN Tests
// =============================================================================

test "LayerGeometry calculates MoE FFN weights" {
    const attn = AttnConfig{
        .n_heads = 8,
        .n_kv_heads = 8,
        .head_dim = 64,
        .d_model = 512,
    };

    const ffn = FfnConfig{ .moe_ffn = .{
        .d_model = 512,
        .d_ff = 1024,
        .num_experts = 8,
        .experts_per_token = 2,
    } };

    const geom = LayerGeometry.init(attn, ffn);

    // router_weights = d_model * num_experts = 512 * 8 = 4096
    try std.testing.expectEqual(@as(usize, 4096), geom.router_weights);

    // expert_weights = num_experts * d_model * d_ff * 3 = 8 * 512 * 1024 * 3 = 12582912
    try std.testing.expectEqual(@as(usize, 12582912), geom.expert_weights);

    // ffn_weights = router + expert = 4096 + 12582912 = 12587008
    try std.testing.expectEqual(@as(usize, 12587008), geom.ffn_weights);
}

// =============================================================================
// Total Parameter Tests
// =============================================================================

test "LayerGeometry calculates total layer parameters" {
    const attn = AttnConfig{
        .n_heads = 4,
        .n_kv_heads = 4,
        .head_dim = 32,
        .d_model = 128,
        .has_q_bias = true,
    };

    const ffn = FfnConfig{ .swiglu = .{
        .d_model = 128,
        .d_ff = 512,
    } };

    const geom = LayerGeometry.init(attn, ffn);

    // qkv_proj_weights = 65536
    // attn_bias_params = 128 (only q_bias)
    // ffn_weights = 128 * 512 * 3 = 196608
    // total = 65536 + 128 + 196608 = 262272
    try std.testing.expectEqual(@as(usize, 262272), geom.total_layer_params);
}
