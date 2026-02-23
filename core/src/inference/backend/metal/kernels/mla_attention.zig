//! Metal MLA (Multi-Latent Attention) kernel surface.

pub const supported = true;

const runtime_graph = @import("../runtime_graph.zig");
const weights = @import("../executor/weights.zig");
const mlx_fused = @import("../mlx/ffi.zig");

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

pub const MLAttention = struct {
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
    rope_theta: f32,
    norm_eps: f32,

    q_lora_rank: usize,
    kv_lora_rank: usize,
    qk_head_dim: usize,
    qk_rope_head_dim: usize,
    qk_nope_head_dim: usize,
    v_head_dim: usize,

    q_a_proj: ?QuantizedWeight = null,
    q_b_proj: ?QuantizedWeight = null,
    kv_a_proj: ?QuantizedWeight = null,
    kv_b_proj: ?QuantizedWeight = null,

    q_a_proj_bf16: ?ArrayHandle = null,
    q_b_proj_bf16: ?ArrayHandle = null,
    kv_a_proj_bf16: ?ArrayHandle = null,
    kv_b_proj_bf16: ?ArrayHandle = null,

    q_a_norm: ?ArrayHandle = null,
    kv_a_norm: ?ArrayHandle = null,

    o_proj: ?QuantizedWeight = null,
    o_proj_bf16: ?ArrayHandle = null,

    pub fn forward(
        self: *const MLAttention,
        input_tensor: ArrayHandle,
        output_tensor: *ArrayHandle,
        cache: *AttnCache,
        scratch: *AttnTemp,
        matmul_scratch: *MatmulScratch,
        use_cache: bool,
    ) !void {
        _ = matmul_scratch;

        if (self.qk_head_dim != self.qk_nope_head_dim + self.qk_rope_head_dim) {
            return error.InvalidShape;
        }

        const cache_handle = if (use_cache and cache.cache != null) cache.cache.?.handle else null;
        const q_a_norm = self.q_a_norm orelse return error.MissingField;
        const kv_a_norm = self.kv_a_norm orelse return error.MissingField;

        const has_quantized_core = self.q_a_proj != null and
            self.q_b_proj != null and
            self.kv_a_proj != null and
            self.kv_b_proj != null;
        const has_dense_core = self.q_a_proj_bf16 != null and
            self.q_b_proj_bf16 != null and
            self.kv_a_proj_bf16 != null and
            self.kv_b_proj_bf16 != null;
        if (has_quantized_core == has_dense_core) return error.InvalidTensorType;

        if (has_quantized_core) {
            const q_a = self.q_a_proj.?;
            const q_b = self.q_b_proj.?;
            const kv_a = self.kv_a_proj.?;
            const kv_b = self.kv_b_proj.?;
            const out = self.o_proj orelse return error.InvalidTensorType;
            if (self.o_proj_bf16 != null) return error.InvalidTensorType;

            if (q_a.group_size != q_b.group_size or q_a.group_size != kv_a.group_size or q_a.group_size != kv_b.group_size or q_a.group_size != out.group_size) {
                return error.InvalidTensorType;
            }
            if (q_a.bits != q_b.bits or q_a.bits != kv_a.bits or q_a.bits != kv_b.bits or q_a.bits != out.bits) {
                return error.InvalidTensorType;
            }

            output_tensor.* = mlx_fused.mlx_lazy_mla_attention_quantized(
                input_tensor,
                q_a.weights,
                q_a.scales,
                q_a.biases,
                q_a_norm,
                q_b.weights,
                q_b.scales,
                q_b.biases,
                kv_a.weights,
                kv_a.scales,
                kv_a.biases,
                kv_a_norm,
                kv_b.weights,
                kv_b.scales,
                kv_b.biases,
                out.weights,
                out.scales,
                out.biases,
                cache_handle,
                cache.layer_idx,
                self.n_heads,
                self.q_lora_rank,
                self.kv_lora_rank,
                self.qk_head_dim,
                self.qk_rope_head_dim,
                self.qk_nope_head_dim,
                self.v_head_dim,
                cache.pos_offset,
                self.rope_theta,
                scratch.runtime_rope_cos_handle,
                scratch.runtime_rope_sin_handle,
                scratch.runtime_rope_dim,
                self.norm_eps,
                q_a.group_size,
                q_a.bits,
            );
            return;
        }

        if (self.o_proj != null) return error.InvalidTensorType;
        output_tensor.* = mlx_fused.mlx_lazy_mla_attention_bf16(
            input_tensor,
            self.q_a_proj_bf16.?,
            q_a_norm,
            self.q_b_proj_bf16.?,
            self.kv_a_proj_bf16.?,
            kv_a_norm,
            self.kv_b_proj_bf16.?,
            self.o_proj_bf16.?,
            cache_handle,
            cache.layer_idx,
            self.n_heads,
            self.q_lora_rank,
            self.kv_lora_rank,
            self.qk_head_dim,
            self.qk_rope_head_dim,
            self.qk_nope_head_dim,
            self.v_head_dim,
            cache.pos_offset,
            self.rope_theta,
            scratch.runtime_rope_cos_handle,
            scratch.runtime_rope_sin_handle,
            scratch.runtime_rope_dim,
            self.norm_eps,
        );
    }
};

test "MLAttention.forward rejects invalid MLA head dimensions" {
    const dummy = @as(ArrayHandle, @ptrFromInt(1));
    const layer = MLAttention{
        .n_heads = 1,
        .rope_theta = 10000.0,
        .norm_eps = 1e-6,
        .q_lora_rank = 8,
        .kv_lora_rank = 8,
        .qk_head_dim = 16,
        .qk_rope_head_dim = 4,
        .qk_nope_head_dim = 11,
        .v_head_dim = 8,
        .q_a_norm = dummy,
        .kv_a_norm = dummy,
        .q_a_proj_bf16 = dummy,
        .q_b_proj_bf16 = dummy,
        .kv_a_proj_bf16 = dummy,
        .kv_b_proj_bf16 = dummy,
        .o_proj_bf16 = dummy,
    };
    var out: ArrayHandle = null;
    var cache: AttnCache = .{};
    var scratch: AttnTemp = .{};
    var mm: MatmulScratch = .{};
    try @import("std").testing.expectError(
        error.InvalidShape,
        layer.forward(dummy, &out, &cache, &scratch, &mm, false),
    );
}
