//! Metal attention/forward kernel surface.

pub const supported = true;

const compute = @import("../../../../compute/root.zig");
const cache_executor = @import("../executor/runtime.zig");
const weights = @import("../executor/weights.zig");

const mlx_graph = compute.metal.graph;
const ArrayHandle = mlx_graph.ArrayHandle;

pub const Cache = cache_executor.Cache;

pub const AttnCache = struct {
    cache: ?Cache = null,
    layer_idx: usize = 0,
    pos_offset: usize = 0,
};

pub const AttnTemp = struct {
    runtime_rope_cos_handle: ArrayHandle = null,
    runtime_rope_sin_handle: ArrayHandle = null,
    runtime_rope_dim: usize = 0,
};

pub const MatmulScratch = struct {};

pub const MultiHeadAttention = struct {
    /// Canonical kernel-call contract for backend parity checks.
    pub const ForwardParams = struct {
        input_tensor: ArrayHandle,
        output_tensor: *ArrayHandle,
        cache: *AttnCache,
        scratch: *AttnTemp,
        matmul_scratch: *MatmulScratch,
        use_cache: bool,
    };

    const QuantizedWeight = weights.WeightHandles.QuantizedWeight;

    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rope_theta: f32,
    norm_eps: f32,
    query_pre_attn_scalar: f32 = 0.0,
    attention_multiplier: f32 = 0.0,

    q_proj: ?QuantizedWeight = null,
    k_proj: ?QuantizedWeight = null,
    v_proj: ?QuantizedWeight = null,
    o_proj: ?QuantizedWeight = null,

    q_proj_bf16: ?ArrayHandle = null,
    k_proj_bf16: ?ArrayHandle = null,
    v_proj_bf16: ?ArrayHandle = null,
    o_proj_bf16: ?ArrayHandle = null,

    q_norm: ?ArrayHandle = null,
    k_norm: ?ArrayHandle = null,

    q_bias: ?ArrayHandle = null,
    k_bias: ?ArrayHandle = null,
    v_bias: ?ArrayHandle = null,
    o_bias: ?ArrayHandle = null,
    attn_sinks: ?ArrayHandle = null,

    pub fn forward(
        self: *const MultiHeadAttention,
        input_tensor: ArrayHandle,
        output_tensor: *ArrayHandle,
        cache: *AttnCache,
        scratch: *AttnTemp,
        matmul_scratch: *MatmulScratch,
        use_cache: bool,
    ) !void {
        _ = matmul_scratch;
        const cache_handle = if (use_cache and cache.cache != null) cache.cache.?.handle else null;
        const q_norm = if (self.q_norm) |h| h else null;
        const k_norm = if (self.k_norm) |h| h else null;
        const q_bias = if (self.q_bias) |h| h else null;
        const k_bias = if (self.k_bias) |h| h else null;
        const v_bias = if (self.v_bias) |h| h else null;
        const o_bias = if (self.o_bias) |h| h else null;
        const attn_sinks = if (self.attn_sinks) |h| h else null;

        if (self.q_proj != null and self.k_proj != null and self.v_proj != null) {
            const q_proj = self.q_proj.?;
            const k_proj = self.k_proj.?;
            const v_proj = self.v_proj.?;
            if (self.o_proj) |o_proj| {
                output_tensor.* = mlx_graph.mlx_lazy_fused_attention(
                    input_tensor,
                    q_proj.weights,
                    q_proj.scales,
                    q_proj.biases,
                    k_proj.weights,
                    k_proj.scales,
                    k_proj.biases,
                    v_proj.weights,
                    v_proj.scales,
                    v_proj.biases,
                    o_proj.weights,
                    o_proj.scales,
                    o_proj.biases,
                    q_norm,
                    k_norm,
                    q_bias,
                    k_bias,
                    v_bias,
                    o_bias,
                    attn_sinks,
                    cache_handle,
                    cache.layer_idx,
                    self.n_heads,
                    self.n_kv_heads,
                    self.head_dim,
                    cache.pos_offset,
                    self.rope_theta,
                    scratch.runtime_rope_cos_handle,
                    scratch.runtime_rope_sin_handle,
                    scratch.runtime_rope_dim,
                    self.norm_eps,
                    q_proj.group_size,
                    q_proj.bits,
                    self.query_pre_attn_scalar,
                    self.attention_multiplier,
                );
                return;
            }
            const o_proj_bf16 = if (self.o_proj_bf16) |h| h else null;
            if (o_proj_bf16 == null) return error.MissingField;
            output_tensor.* = mlx_graph.mlx_lazy_fused_attention_qkv_quantized_o_dense(
                input_tensor,
                q_proj.weights,
                q_proj.scales,
                q_proj.biases,
                k_proj.weights,
                k_proj.scales,
                k_proj.biases,
                v_proj.weights,
                v_proj.scales,
                v_proj.biases,
                o_proj_bf16,
                q_norm,
                k_norm,
                q_bias,
                k_bias,
                v_bias,
                o_bias,
                attn_sinks,
                cache_handle,
                cache.layer_idx,
                self.n_heads,
                self.n_kv_heads,
                self.head_dim,
                cache.pos_offset,
                self.rope_theta,
                scratch.runtime_rope_cos_handle,
                scratch.runtime_rope_sin_handle,
                scratch.runtime_rope_dim,
                self.norm_eps,
                q_proj.group_size,
                q_proj.bits,
                self.query_pre_attn_scalar,
                self.attention_multiplier,
            );
            return;
        }

        const q_proj_bf16 = if (self.q_proj_bf16) |h| h else null;
        const k_proj_bf16 = if (self.k_proj_bf16) |h| h else null;
        const v_proj_bf16 = if (self.v_proj_bf16) |h| h else null;
        const o_proj_bf16 = if (self.o_proj_bf16) |h| h else null;
        if (q_proj_bf16 == null or k_proj_bf16 == null or v_proj_bf16 == null or o_proj_bf16 == null) {
            return error.MissingField;
        }

        output_tensor.* = mlx_graph.mlx_lazy_fused_attention_bf16(
            input_tensor,
            q_proj_bf16,
            k_proj_bf16,
            v_proj_bf16,
            o_proj_bf16,
            q_norm,
            k_norm,
            q_bias,
            k_bias,
            v_bias,
            o_bias,
            attn_sinks,
            cache_handle,
            cache.layer_idx,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            cache.pos_offset,
            self.rope_theta,
            scratch.runtime_rope_cos_handle,
            scratch.runtime_rope_sin_handle,
            scratch.runtime_rope_dim,
            self.norm_eps,
            self.query_pre_attn_scalar,
            self.attention_multiplier,
        );
    }
};
