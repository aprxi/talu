//! Metal Mamba kernel surface.

pub const supported = true;

const weights = @import("../executor/weights.zig");
const runtime_graph = @import("../runtime_graph.zig");
const mlx_fused = @import("../mlx/ffi.zig");

const ArrayHandle = mlx_fused.ArrayHandle;

pub const WeightHandles = weights.WeightHandles;
pub const MambaCache = runtime_graph.MambaCache;

pub const MambaState = struct {
    cache: ?MambaCache = null,
    layer_idx: usize = 0,
};

pub const MambaScratch = struct {};
pub const MatmulScratch = struct {};

pub const MambaKernel = struct {
    /// Canonical kernel-call contract for backend parity checks.
    pub const ForwardParams = struct {
        input_tensor: ArrayHandle,
        output_tensor: *ArrayHandle,
        state: *MambaState,
        scratch: *MambaScratch,
        matmul_scratch: *MatmulScratch,
    };

    const QuantizedWeight = WeightHandles.QuantizedWeight;

    d_state: usize,
    d_conv: usize,
    n_heads: usize,
    d_head: usize,
    n_groups: usize,
    use_gelu: bool = false,
    residual_multiplier: f32 = 1.0,
    norm_eps: f32,
    gate_up_layout: u8 = 0, // 0=concat, 1=interleaved

    ln1_weight: ArrayHandle,
    in_proj: ?QuantizedWeight = null,
    in_proj_bf16: ?ArrayHandle = null,
    conv_weight: ArrayHandle,
    conv_bias: ?ArrayHandle = null,
    a_log: ArrayHandle,
    d_skip: ArrayHandle,
    dt_bias: ?ArrayHandle = null,
    norm_weight: ?ArrayHandle = null,
    out_proj: ?QuantizedWeight = null,
    out_proj_bf16: ?ArrayHandle = null,

    ln2_weight: ?ArrayHandle = null,
    gate_up: ?QuantizedWeight = null,
    gate_up_bf16: ?ArrayHandle = null,
    down_proj: ?QuantizedWeight = null,
    down_proj_bf16: ?ArrayHandle = null,

    pub fn forward(
        self: *const MambaKernel,
        input_tensor: ArrayHandle,
        output_tensor: *ArrayHandle,
        state: *MambaState,
        scratch: *MambaScratch,
        matmul_scratch: *MatmulScratch,
    ) !void {
        _ = scratch;
        _ = matmul_scratch;

        const cache_handle = if (state.cache) |c| c.handle else null;

        const has_quantized_core = self.in_proj != null and self.out_proj != null;
        const has_dense_core = self.in_proj_bf16 != null and self.out_proj_bf16 != null;
        if (has_quantized_core == has_dense_core) return error.InvalidTensorType;

        const has_quantized_ffn = self.gate_up != null and self.down_proj != null and self.ln2_weight != null;
        const has_dense_ffn = self.gate_up_bf16 != null and self.down_proj_bf16 != null and self.ln2_weight != null;
        if (has_quantized_ffn and has_dense_ffn) return error.InvalidTensorType;

        if (has_quantized_core) {
            const in_proj = self.in_proj.?;
            const out_proj = self.out_proj.?;
            if (in_proj.group_size != out_proj.group_size or in_proj.bits != out_proj.bits) return error.InvalidTensorType;

            if (has_quantized_ffn) {
                const gate_up_q = self.gate_up.?;
                const down_proj_q = self.down_proj.?;
                if (gate_up_q.group_size != in_proj.group_size or gate_up_q.bits != in_proj.bits or down_proj_q.group_size != in_proj.group_size or down_proj_q.bits != in_proj.bits) {
                    return error.InvalidTensorType;
                }
            }

            output_tensor.* = mlx_fused.mlx_lazy_mamba_block_quantized(
                input_tensor,
                self.ln1_weight,
                in_proj.weights,
                in_proj.scales,
                in_proj.biases,
                self.conv_weight,
                if (self.conv_bias) |b| b else null,
                self.a_log,
                self.d_skip,
                if (self.dt_bias) |b| b else null,
                if (self.norm_weight) |w| w else null,
                out_proj.weights,
                out_proj.scales,
                out_proj.biases,
                if (self.ln2_weight) |w| w else null,
                if (has_quantized_ffn) self.gate_up.?.weights else null,
                if (has_quantized_ffn) self.gate_up.?.scales else null,
                if (has_quantized_ffn) self.gate_up.?.biases else null,
                if (has_quantized_ffn) self.down_proj.?.weights else null,
                if (has_quantized_ffn) self.down_proj.?.scales else null,
                if (has_quantized_ffn) self.down_proj.?.biases else null,
                in_proj.group_size,
                in_proj.bits,
                self.use_gelu,
                self.residual_multiplier,
                self.norm_eps,
                cache_handle,
                state.layer_idx,
                self.d_state,
                self.d_conv,
                self.n_heads,
                self.d_head,
                self.n_groups,
                self.gate_up_layout,
            );
            return;
        }

        output_tensor.* = mlx_fused.mlx_lazy_mamba_block_bf16(
            input_tensor,
            self.ln1_weight,
            self.in_proj_bf16.?,
            self.conv_weight,
            if (self.conv_bias) |b| b else null,
            self.a_log,
            self.d_skip,
            if (self.dt_bias) |b| b else null,
            if (self.norm_weight) |w| w else null,
            self.out_proj_bf16.?,
            if (self.ln2_weight) |w| w else null,
            if (self.gate_up_bf16) |w| w else null,
            if (self.down_proj_bf16) |w| w else null,
            self.use_gelu,
            self.residual_multiplier,
            self.norm_eps,
            cache_handle,
            state.layer_idx,
            self.d_state,
            self.d_conv,
            self.n_heads,
            self.d_head,
            self.n_groups,
            self.gate_up_layout,
        );
    }
};
