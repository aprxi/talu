//! Inference-owned fused/dense model runtime FFI surface for Metal backend.

const compute = @import("../../../compute/root.zig");
const runtime_graph = @import("runtime_graph.zig");

pub const ArrayHandle = compute.metal.graph.ArrayHandle;
pub const CacheHandle = runtime_graph.CacheHandle;
pub const ShortConvCacheHandle = runtime_graph.ShortConvCacheHandle;

pub const FusedModelHandle = ?*anyopaque;
pub const DenseModelHandle = ?*anyopaque;
pub const CompiledLayerHandle = ?*anyopaque;

pub const CompiledLayer = struct {
    handle: CompiledLayerHandle,

    pub fn forward(
        self: CompiledLayer,
        hidden: ArrayHandle,
        cache_ptr: CacheHandle,
        layer_idx: usize,
        pos_offset: usize,
    ) ArrayHandle {
        if (self.handle == null) return null;
        return mlx_layer_forward(self.handle, hidden, cache_ptr, layer_idx, pos_offset);
    }
};

pub extern fn mlx_compile_layer(
    q_weight: ArrayHandle,
    q_scales: ArrayHandle,
    q_biases: ArrayHandle,
    k_weight: ArrayHandle,
    k_scales: ArrayHandle,
    k_biases: ArrayHandle,
    v_weight: ArrayHandle,
    v_scales: ArrayHandle,
    v_biases: ArrayHandle,
    o_weight: ArrayHandle,
    o_scales: ArrayHandle,
    o_biases: ArrayHandle,
    gate_weight: ArrayHandle,
    gate_scales: ArrayHandle,
    gate_biases: ArrayHandle,
    up_weight: ArrayHandle,
    up_scales: ArrayHandle,
    up_biases: ArrayHandle,
    down_weight: ArrayHandle,
    down_scales: ArrayHandle,
    down_biases: ArrayHandle,
    attn_norm: ArrayHandle,
    ffn_norm: ArrayHandle,
    q_norm: ArrayHandle,
    k_norm: ArrayHandle,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    hidden_dim: usize,
    group_size: usize,
    bits: usize,
    rope_theta: f32,
    rms_eps: f32,
) CompiledLayerHandle;

pub extern fn mlx_layer_forward(
    compiled_handle: CompiledLayerHandle,
    hidden: ArrayHandle,
    cache_ptr: CacheHandle,
    layer_idx: usize,
    pos_offset: usize,
) ArrayHandle;

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
) void;
pub extern fn mlx_fused_model_optimize(model: FusedModelHandle) void;
pub extern fn mlx_fused_model_free(model: FusedModelHandle) void;

pub extern fn mlx_fused_decode_step_logits(
    model: FusedModelHandle,
    cache: CacheHandle,
    shortconv_cache: ShortConvCacheHandle,
    token_id: u32,
    pos_offset: usize,
) ArrayHandle;
pub extern fn mlx_fused_decode_batch(
    model: FusedModelHandle,
    cache: CacheHandle,
    shortconv_cache: ShortConvCacheHandle,
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
    first_token_id: u32,
    pos_offset: usize,
) void;
pub extern fn mlx_pipeline_step(
    model: FusedModelHandle,
    cache: CacheHandle,
    shortconv_cache: ShortConvCacheHandle,
    pos_offset: usize,
) u32;
pub extern fn mlx_pipeline_flush() u32;

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
pub extern fn mlx_dense_model_free(model: DenseModelHandle) void;

pub extern fn mlx_dense_pipeline_prime(
    model: DenseModelHandle,
    cache: CacheHandle,
    shortconv_cache: ShortConvCacheHandle,
    first_token_id: u32,
    pos_offset: usize,
) void;
pub extern fn mlx_dense_pipeline_step(
    model: DenseModelHandle,
    cache: CacheHandle,
    shortconv_cache: ShortConvCacheHandle,
    pos_offset: usize,
) u32;
pub extern fn mlx_dense_pipeline_flush() u32;
pub extern fn mlx_dense_decode_step_logits(
    model: DenseModelHandle,
    cache: CacheHandle,
    shortconv_cache: ShortConvCacheHandle,
    token_id: u32,
    pos_offset: usize,
) ArrayHandle;
pub extern fn mlx_dense_decode_batch(
    model: DenseModelHandle,
    cache: CacheHandle,
    shortconv_cache: ShortConvCacheHandle,
    first_token: u32,
    start_pos: usize,
    out_tokens: [*]u32,
    max_tokens: usize,
    eos_ids: [*]const u32,
    n_eos_ids: usize,
) u32;
