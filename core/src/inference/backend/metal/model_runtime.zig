//! Inference-owned fused/dense model runtime FFI surface for Metal backend.

const std = @import("std");
const compute = @import("../../../compute/root.zig");
const runtime_graph = @import("runtime_graph.zig");

pub const ArrayHandle = compute.metal.graph.ArrayHandle;
pub const CacheHandle = runtime_graph.CacheHandle;
pub const ShortConvCacheHandle = runtime_graph.ShortConvCacheHandle;
pub const MambaCacheHandle = runtime_graph.MambaCacheHandle;

pub const FusedModelHandle = ?*anyopaque;
pub const DenseModelHandle = ?*anyopaque;
pub const DecodeModel = struct {
    handle: *anyopaque,
};

pub fn decodeModelFromFused(handle: FusedModelHandle) ?DecodeModel {
    if (handle) |h| {
        const wrapped = mlx_decode_model_wrap_fused(h);
        if (wrapped) |w| return .{ .handle = w };
        mlx_fused_model_free(h);
    }
    return null;
}

pub fn decodeModelFromDense(handle: DenseModelHandle) ?DecodeModel {
    if (handle) |h| {
        const wrapped = mlx_decode_model_wrap_dense(h);
        if (wrapped) |w| return .{ .handle = w };
        mlx_dense_model_free(h);
    }
    return null;
}

pub fn decodeModelFree(model: DecodeModel) void {
    mlx_decode_model_free(model.handle);
}

pub fn decodeStepLogits(
    model: DecodeModel,
    cache: CacheHandle,
    shortconv_cache: ShortConvCacheHandle,
    mamba_cache: MambaCacheHandle,
    token_id: u32,
    pos_offset: usize,
) ArrayHandle {
    return mlx_decode_model_step_logits(model.handle, cache, shortconv_cache, mamba_cache, token_id, pos_offset);
}

pub fn decodeStepLogitsBatch(
    model: DecodeModel,
    caches: []const CacheHandle,
    shortconv_caches: []const ShortConvCacheHandle,
    mamba_caches: []const MambaCacheHandle,
    token_ids: []const u32,
    pos_offsets: []const usize,
    out_logits: []ArrayHandle,
) void {
    std.debug.assert(caches.len == shortconv_caches.len);
    std.debug.assert(caches.len == mamba_caches.len);
    std.debug.assert(caches.len == token_ids.len);
    std.debug.assert(caches.len == pos_offsets.len);
    std.debug.assert(caches.len == out_logits.len);
    mlx_decode_model_step_logits_batch(
        model.handle,
        caches.ptr,
        shortconv_caches.ptr,
        mamba_caches.ptr,
        token_ids.ptr,
        pos_offsets.ptr,
        out_logits.ptr,
        out_logits.len,
    );
}

pub fn decodeBatch(
    model: DecodeModel,
    cache: CacheHandle,
    shortconv_cache: ShortConvCacheHandle,
    mamba_cache: MambaCacheHandle,
    first_token: u32,
    start_pos: usize,
    out_tokens: [*]u32,
    max_tokens: usize,
    eos_ids: [*]const u32,
    n_eos_ids: usize,
) u32 {
    return mlx_decode_model_decode_batch(model.handle, cache, shortconv_cache, mamba_cache, first_token, start_pos, out_tokens, max_tokens, eos_ids, n_eos_ids);
}

pub fn pipelinePrime(
    model: DecodeModel,
    cache: CacheHandle,
    shortconv_cache: ShortConvCacheHandle,
    mamba_cache: MambaCacheHandle,
    first_token_id: u32,
    pos_offset: usize,
) void {
    mlx_decode_model_pipeline_prime(model.handle, cache, shortconv_cache, mamba_cache, first_token_id, pos_offset);
}

pub fn pipelineStep(
    model: DecodeModel,
    cache: CacheHandle,
    shortconv_cache: ShortConvCacheHandle,
    mamba_cache: MambaCacheHandle,
    pos_offset: usize,
) u32 {
    return mlx_decode_model_pipeline_step(model.handle, cache, shortconv_cache, mamba_cache, pos_offset);
}

pub fn pipelineFlushWithCache(
    model: DecodeModel,
    cache: CacheHandle,
    shortconv_cache: ShortConvCacheHandle,
    mamba_cache: MambaCacheHandle,
) u32 {
    return mlx_decode_model_pipeline_flush(model.handle, cache, shortconv_cache, mamba_cache);
}

pub extern fn mlx_decode_model_wrap_fused(model: *anyopaque) ?*anyopaque;
pub extern fn mlx_decode_model_wrap_dense(model: *anyopaque) ?*anyopaque;
pub extern fn mlx_decode_model_free(model: *anyopaque) void;
pub extern fn mlx_decode_model_step_logits(
    model: *anyopaque,
    cache: CacheHandle,
    shortconv_cache: ShortConvCacheHandle,
    mamba_cache: MambaCacheHandle,
    token_id: u32,
    pos_offset: usize,
) ArrayHandle;
pub extern fn mlx_decode_model_step_logits_batch(
    model: *anyopaque,
    caches: [*]const CacheHandle,
    shortconv_caches: [*]const ShortConvCacheHandle,
    mamba_caches: [*]const MambaCacheHandle,
    token_ids: [*]const u32,
    pos_offsets: [*]const usize,
    out_logits: [*]ArrayHandle,
    count: usize,
) void;
pub extern fn mlx_decode_model_decode_batch(
    model: *anyopaque,
    cache: CacheHandle,
    shortconv_cache: ShortConvCacheHandle,
    mamba_cache: MambaCacheHandle,
    first_token: u32,
    start_pos: usize,
    out_tokens: [*]u32,
    max_tokens: usize,
    eos_ids: [*]const u32,
    n_eos_ids: usize,
) u32;
pub extern fn mlx_decode_model_pipeline_prime(
    model: *anyopaque,
    cache: CacheHandle,
    shortconv_cache: ShortConvCacheHandle,
    mamba_cache: MambaCacheHandle,
    first_token_id: u32,
    pos_offset: usize,
) void;
pub extern fn mlx_decode_model_pipeline_step(
    model: *anyopaque,
    cache: CacheHandle,
    shortconv_cache: ShortConvCacheHandle,
    mamba_cache: MambaCacheHandle,
    pos_offset: usize,
) u32;
pub extern fn mlx_decode_model_pipeline_flush(
    model: *anyopaque,
    cache: CacheHandle,
    shortconv_cache: ShortConvCacheHandle,
    mamba_cache: MambaCacheHandle,
) u32;

pub extern fn mlx_fused_model_create(
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    hidden_dim: usize,
    group_size: usize,
    bits: usize,
    rope_theta: f32,
    rms_eps: f32,
) FusedModelHandle;
pub extern fn mlx_fused_model_set_embeddings(model: FusedModelHandle, w: ArrayHandle, s: ArrayHandle, b: ArrayHandle) void;
pub extern fn mlx_fused_model_set_final(model: FusedModelHandle, ln_w: ArrayHandle, lm_w: ArrayHandle, lm_s: ArrayHandle, lm_b: ArrayHandle) void;
pub extern fn mlx_fused_model_set_rope_freqs(model: FusedModelHandle, freqs: ArrayHandle) void;
pub extern fn mlx_fused_model_set_arch_config(model: FusedModelHandle, has_norm_weight_offset: bool, use_gelu: bool, query_pre_attn_scalar: f32) void;
pub extern fn mlx_fused_model_set_scaling_config(
    model: FusedModelHandle,
    embedding_multiplier: f32,
    attention_multiplier: f32,
    residual_multiplier: f32,
    logits_scaling: f32,
) void;
pub extern fn mlx_fused_model_set_topology(model: FusedModelHandle, layer_kinds: [*]const u8, n_layer_kinds: usize) void;
pub extern fn mlx_fused_model_set_layer(
    model: FusedModelHandle,
    layer_idx: usize,
    ln1_w: ArrayHandle,
    q_w: ArrayHandle,
    q_s: ArrayHandle,
    q_b: ArrayHandle,
    k_w: ArrayHandle,
    k_s: ArrayHandle,
    k_b: ArrayHandle,
    v_w: ArrayHandle,
    v_s: ArrayHandle,
    v_b: ArrayHandle,
    o_w: ArrayHandle,
    o_s: ArrayHandle,
    o_b: ArrayHandle,
    ln2_w: ArrayHandle,
    gate_w: ArrayHandle,
    gate_s: ArrayHandle,
    gate_b: ArrayHandle,
    up_w: ArrayHandle,
    up_s: ArrayHandle,
    up_b: ArrayHandle,
    down_w: ArrayHandle,
    down_s: ArrayHandle,
    down_b: ArrayHandle,
    q_norm: ArrayHandle,
    k_norm: ArrayHandle,
    pre_ffn_norm: ArrayHandle,
    post_ffn_norm: ArrayHandle,
    shortconv_d_conv: usize,
    shortconv_conv_dim: usize,
    shortconv_in_w: ArrayHandle,
    shortconv_in_s: ArrayHandle,
    shortconv_in_b: ArrayHandle,
    shortconv_out_w: ArrayHandle,
    shortconv_out_s: ArrayHandle,
    shortconv_out_b: ArrayHandle,
    shortconv_conv_w: ArrayHandle,
    shortconv_conv_b: ArrayHandle,
    moe_router_w: ArrayHandle,
    moe_router_s: ArrayHandle,
    moe_router_b: ArrayHandle,
    moe_router_bias: ArrayHandle,
    moe_gate_w: ArrayHandle,
    moe_gate_s: ArrayHandle,
    moe_up_w: ArrayHandle,
    moe_up_s: ArrayHandle,
    moe_down_w: ArrayHandle,
    moe_down_s: ArrayHandle,
    moe_gate_bias: ArrayHandle,
    moe_up_bias: ArrayHandle,
    moe_down_bias: ArrayHandle,
    moe_num_experts: usize,
    moe_experts_per_token: usize,
    moe_router_group_size: usize,
    moe_expert_group_size: usize,
) void;
pub extern fn mlx_fused_model_set_layer_mla_quantized(
    model: FusedModelHandle,
    layer_idx: usize,
    n_heads: usize,
    q_lora_rank: usize,
    kv_lora_rank: usize,
    qk_head_dim: usize,
    qk_rope_head_dim: usize,
    qk_nope_head_dim: usize,
    v_head_dim: usize,
    q_a_w: ArrayHandle,
    q_a_s: ArrayHandle,
    q_a_b: ArrayHandle,
    q_b_w: ArrayHandle,
    q_b_s: ArrayHandle,
    q_b_b: ArrayHandle,
    kv_a_w: ArrayHandle,
    kv_a_s: ArrayHandle,
    kv_a_b: ArrayHandle,
    kv_b_w: ArrayHandle,
    kv_b_s: ArrayHandle,
    kv_b_b: ArrayHandle,
    q_a_norm: ArrayHandle,
    kv_a_norm: ArrayHandle,
    o_w: ArrayHandle,
    o_s: ArrayHandle,
    o_b: ArrayHandle,
) void;
pub extern fn mlx_fused_model_set_layer_mamba_quantized(
    model: FusedModelHandle,
    layer_idx: usize,
    d_state: usize,
    d_conv: usize,
    n_heads: usize,
    d_head: usize,
    n_groups: usize,
    gate_up_layout: u8,
    ln1_w: ArrayHandle,
    conv_weight: ArrayHandle,
    conv_bias: ArrayHandle,
    a_log: ArrayHandle,
    d_skip: ArrayHandle,
    dt_bias: ArrayHandle,
    norm_weight: ArrayHandle,
    in_w: ArrayHandle,
    in_s: ArrayHandle,
    in_b: ArrayHandle,
    out_w: ArrayHandle,
    out_s: ArrayHandle,
    out_b: ArrayHandle,
    ln2_w: ArrayHandle,
    gate_up_w: ArrayHandle,
    gate_up_s: ArrayHandle,
    gate_up_b: ArrayHandle,
    down_w: ArrayHandle,
    down_s: ArrayHandle,
    down_b: ArrayHandle,
) void;
pub extern fn mlx_fused_model_optimize(model: FusedModelHandle) void;
pub extern fn mlx_fused_model_compile(model: FusedModelHandle) void;
pub extern fn mlx_fused_model_free(model: FusedModelHandle) void;

pub extern fn mlx_fused_decode_step_logits(
    model: FusedModelHandle,
    cache: CacheHandle,
    shortconv_cache: ShortConvCacheHandle,
    mamba_cache: MambaCacheHandle,
    token_id: u32,
    pos_offset: usize,
) ArrayHandle;
pub extern fn mlx_fused_decode_batch(
    model: FusedModelHandle,
    cache: CacheHandle,
    shortconv_cache: ShortConvCacheHandle,
    mamba_cache: MambaCacheHandle,
    first_token: u32,
    start_pos: usize,
    out_tokens: [*]u32,
    max_tokens: usize,
    eos_ids: [*]const u32,
    n_eos_ids: usize,
) u32;
pub extern fn mlx_pipeline_prime(
    model: FusedModelHandle,
    cache: CacheHandle,
    shortconv_cache: ShortConvCacheHandle,
    mamba_cache: MambaCacheHandle,
    first_token_id: u32,
    pos_offset: usize,
) void;
pub extern fn mlx_pipeline_step(
    model: FusedModelHandle,
    cache: CacheHandle,
    shortconv_cache: ShortConvCacheHandle,
    mamba_cache: MambaCacheHandle,
    pos_offset: usize,
) u32;
pub extern fn mlx_pipeline_flush(
    model: FusedModelHandle,
    cache: CacheHandle,
    shortconv_cache: ShortConvCacheHandle,
    mamba_cache: MambaCacheHandle,
) u32;

pub extern fn mlx_dense_model_create(
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    hidden_dim: usize,
    rope_theta: f32,
    rms_eps: f32,
) DenseModelHandle;
pub extern fn mlx_dense_model_set_embeddings(model: DenseModelHandle, embed: ArrayHandle) void;
pub extern fn mlx_dense_model_set_final(model: DenseModelHandle, ln_w: ArrayHandle, lm_head: ArrayHandle) void;
pub extern fn mlx_dense_model_set_arch_config(
    model: DenseModelHandle,
    has_norm_weight_offset: bool,
    use_gelu: bool,
    query_pre_attn_scalar: f32,
) void;
pub extern fn mlx_dense_model_set_scaling_config(
    model: DenseModelHandle,
    embedding_multiplier: f32,
    attention_multiplier: f32,
    residual_multiplier: f32,
    logits_scaling: f32,
) void;
pub extern fn mlx_dense_model_set_topology(model: DenseModelHandle, layer_kinds: [*]const u8, n_layer_kinds: usize) void;
pub extern fn mlx_dense_model_set_layer(
    model: DenseModelHandle,
    layer_idx: usize,
    ln1_w: ArrayHandle,
    q_proj: ArrayHandle,
    k_proj: ArrayHandle,
    v_proj: ArrayHandle,
    o_proj: ArrayHandle,
    ln2_w: ArrayHandle,
    gate_proj: ArrayHandle,
    up_proj: ArrayHandle,
    down_proj: ArrayHandle,
    q_norm: ArrayHandle,
    k_norm: ArrayHandle,
    shortconv_d_conv: usize,
    shortconv_conv_dim: usize,
    shortconv_in_proj: ArrayHandle,
    shortconv_conv_weight: ArrayHandle,
    shortconv_conv_bias: ArrayHandle,
    shortconv_out_proj: ArrayHandle,
) void;
pub extern fn mlx_dense_model_set_layer_mla_bf16(
    model: DenseModelHandle,
    layer_idx: usize,
    n_heads: usize,
    q_lora_rank: usize,
    kv_lora_rank: usize,
    qk_head_dim: usize,
    qk_rope_head_dim: usize,
    qk_nope_head_dim: usize,
    v_head_dim: usize,
    q_a_w: ArrayHandle,
    q_b_w: ArrayHandle,
    kv_a_w: ArrayHandle,
    kv_b_w: ArrayHandle,
    q_a_norm: ArrayHandle,
    kv_a_norm: ArrayHandle,
    o_w: ArrayHandle,
) void;
pub extern fn mlx_dense_model_set_layer_mamba_bf16(
    model: DenseModelHandle,
    layer_idx: usize,
    d_state: usize,
    d_conv: usize,
    n_heads: usize,
    d_head: usize,
    n_groups: usize,
    gate_up_layout: u8,
    ln1_w: ArrayHandle,
    conv_weight: ArrayHandle,
    conv_bias: ArrayHandle,
    a_log: ArrayHandle,
    d_skip: ArrayHandle,
    dt_bias: ArrayHandle,
    norm_weight: ArrayHandle,
    in_proj: ArrayHandle,
    out_proj: ArrayHandle,
    ln2_w: ArrayHandle,
    gate_up: ArrayHandle,
    down_proj: ArrayHandle,
) void;
pub extern fn mlx_dense_model_free(model: DenseModelHandle) void;

pub extern fn mlx_dense_pipeline_prime(
    model: DenseModelHandle,
    cache: CacheHandle,
    shortconv_cache: ShortConvCacheHandle,
    mamba_cache: MambaCacheHandle,
    first_token_id: u32,
    pos_offset: usize,
) void;
pub extern fn mlx_dense_pipeline_step(
    model: DenseModelHandle,
    cache: CacheHandle,
    shortconv_cache: ShortConvCacheHandle,
    mamba_cache: MambaCacheHandle,
    pos_offset: usize,
) u32;
pub extern fn mlx_dense_pipeline_flush(
    model: DenseModelHandle,
    cache: CacheHandle,
    shortconv_cache: ShortConvCacheHandle,
    mamba_cache: MambaCacheHandle,
) u32;
pub extern fn mlx_dense_decode_step_logits(
    model: DenseModelHandle,
    cache: CacheHandle,
    shortconv_cache: ShortConvCacheHandle,
    mamba_cache: MambaCacheHandle,
    token_id: u32,
    pos_offset: usize,
) ArrayHandle;
pub extern fn mlx_dense_decode_batch(
    model: DenseModelHandle,
    cache: CacheHandle,
    shortconv_cache: ShortConvCacheHandle,
    mamba_cache: MambaCacheHandle,
    first_token: u32,
    start_pos: usize,
    out_tokens: [*]u32,
    max_tokens: usize,
    eos_ids: [*]const u32,
    n_eos_ids: usize,
) u32;

test "pipelineFlushWithCache exposes stable callable signature" {
    const fn_info = @typeInfo(@TypeOf(pipelineFlushWithCache)).@"fn";
    try std.testing.expectEqual(@as(usize, 4), fn_info.params.len);
    const f = pipelineFlushWithCache;
    _ = f;
}

test "mlx_fused_model_compile exposes stable callable signature" {
    const fn_info = @typeInfo(@TypeOf(mlx_fused_model_compile)).@"fn";
    try std.testing.expectEqual(@as(usize, 1), fn_info.params.len);
    const f = mlx_fused_model_compile;
    _ = f;
}

test "mlx_dense_model_set_layer_mla_bf16 exposes stable callable signature" {
    const fn_info = @typeInfo(@TypeOf(mlx_dense_model_set_layer_mla_bf16)).@"fn";
    try std.testing.expectEqual(@as(usize, 16), fn_info.params.len);
    const f = mlx_dense_model_set_layer_mla_bf16;
    _ = f;
}
