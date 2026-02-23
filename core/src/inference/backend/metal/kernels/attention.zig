//! Metal attention/forward kernel surface.

pub const supported = true;

const std = @import("std");
const builtin = @import("builtin");
const compute = @import("../../../../compute/root.zig");
const runtime_graph = @import("../runtime_graph.zig");
const weights = @import("../executor/weights.zig");
const mlx_fused = @import("../mlx/ffi.zig");

const device_mod = compute.metal.device;
const graph = compute.metal.graph;
const ArrayHandle = mlx_fused.ArrayHandle;

pub const Cache = runtime_graph.Cache;

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
                output_tensor.* = mlx_fused.mlx_lazy_fused_attention(
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
            output_tensor.* = mlx_fused.mlx_lazy_fused_attention_qkv_quantized_o_dense(
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

        output_tensor.* = mlx_fused.mlx_lazy_fused_attention_bf16(
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

pub const FusedAttention = MultiHeadAttention;

test "MultiHeadAttention.forward preallocates cache to max_seq_len" {
    if (comptime builtin.os.tag != .macos) return;
    if (!device_mod.isAvailable()) return;

    const max_seq_len: usize = 1024;
    const cache = runtime_graph.Cache.init(1, true, max_seq_len);
    defer cache.deinit();

    const input_data = [_]f32{ 1.0, 2.0 };
    const input_shape = [_]i64{ 1, 1, 2 };
    const input_handle = graph.createArrayF32(&input_data, &input_shape);
    defer graph.freeArray(input_handle);

    const weight_data = [_]f32{
        1.0, 0.0,
        0.0, 1.0,
    };
    const weight_shape = [_]i64{ 2, 2 };
    const q_weight = graph.createArrayF32(&weight_data, &weight_shape);
    defer graph.freeArray(q_weight);
    const k_weight = graph.createArrayF32(&weight_data, &weight_shape);
    defer graph.freeArray(k_weight);
    const v_weight = graph.createArrayF32(&weight_data, &weight_shape);
    defer graph.freeArray(v_weight);
    const o_weight = graph.createArrayF32(&weight_data, &weight_shape);
    defer graph.freeArray(o_weight);

    const attention = MultiHeadAttention{
        .n_heads = 1,
        .n_kv_heads = 1,
        .head_dim = 2,
        .rope_theta = 10000.0,
        .norm_eps = 1e-5,
        .q_proj_bf16 = q_weight,
        .k_proj_bf16 = k_weight,
        .v_proj_bf16 = v_weight,
        .o_proj_bf16 = o_weight,
    };

    var output_handle: ArrayHandle = null;
    var attn_cache = AttnCache{
        .cache = cache,
        .layer_idx = 0,
        .pos_offset = 0,
    };
    var attn_temp = AttnTemp{};
    var matmul_scratch = MatmulScratch{};
    try attention.forward(
        input_handle,
        &output_handle,
        &attn_cache,
        &attn_temp,
        &matmul_scratch,
        true,
    );
    defer graph.freeArray(output_handle);

    graph.eval(&[_]ArrayHandle{output_handle});

    const cached = cache.get(0);
    try std.testing.expect(cached.k != null);

    var shape_buffer: [8]usize = undefined;
    const rank = graph.getShape(cached.k, &shape_buffer);
    try std.testing.expectEqual(@as(usize, 4), rank);
    try std.testing.expectEqual(max_seq_len, shape_buffer[2]);
}
