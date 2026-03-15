//! Metal gated-delta kernel surface.

pub const supported = true;

const weights = @import("../executor/weights.zig");
const runtime_graph = @import("../runtime_graph.zig");
const mlx_fused = @import("../mlx/ffi.zig");

const ArrayHandle = mlx_fused.ArrayHandle;

pub const WeightHandles = weights.WeightHandles;
pub const GatedDeltaCache = runtime_graph.GatedDeltaCache;

pub const GatedDeltaState = struct {
    cache: ?GatedDeltaCache = null,
    layer_idx: usize = 0,
    capture_enabled: bool = false,
    capture_in_proj: ArrayHandle = null,
    capture_conv: ArrayHandle = null,
    capture_ssm: ArrayHandle = null,
    capture_norm: ArrayHandle = null,
};

pub const GatedDeltaScratch = struct {};
pub const MatmulScratch = struct {};

pub const GatedDeltaKernel = struct {
    pub const ForwardParams = struct {
        input_tensor: ArrayHandle,
        output_tensor: *ArrayHandle,
        state: *GatedDeltaState,
        scratch: *GatedDeltaScratch,
        matmul_scratch: *MatmulScratch,
    };

    const QuantizedWeight = WeightHandles.QuantizedWeight;

    d_conv: usize,
    n_heads: usize,
    n_key_heads: usize,
    d_head: usize,
    in_proj: ?QuantizedWeight = null,
    in_proj_bf16: ?ArrayHandle = null,
    conv_weight: ArrayHandle,
    conv_bias: ?ArrayHandle = null,
    a_log: ArrayHandle,
    dt_bias: ?ArrayHandle = null,
    norm_weight: ?ArrayHandle = null,
    out_proj: ?QuantizedWeight = null,
    out_proj_bf16: ?ArrayHandle = null,

    pub fn forward(
        self: *const GatedDeltaKernel,
        input_tensor: ArrayHandle,
        output_tensor: *ArrayHandle,
        state: *GatedDeltaState,
        scratch: *GatedDeltaScratch,
        matmul_scratch: *MatmulScratch,
    ) !void {
        _ = scratch;
        _ = matmul_scratch;

        const cache_handle = if (state.cache) |cache| cache.handle else null;
        const has_quantized_core = self.in_proj != null and self.out_proj != null;
        const has_dense_core = self.in_proj_bf16 != null and self.out_proj_bf16 != null;
        if (has_quantized_core == has_dense_core) return error.InvalidTensorType;
        // XRAY CONTRACT:
        // Verification is observability-only. This kernel must stay on the
        // production path regardless of tracing/verification state.
        state.capture_in_proj = null;
        state.capture_conv = null;
        state.capture_ssm = null;
        state.capture_norm = null;

        if (has_quantized_core) {
            const in_proj = self.in_proj.?;
            const out_proj = self.out_proj.?;
            if (in_proj.group_size != out_proj.group_size or in_proj.bits != out_proj.bits) {
                return error.InvalidTensorType;
            }
            if (state.capture_enabled) {
                output_tensor.* = mlx_fused.mlx_lazy_gated_delta_mixer_quantized_capture(
                    input_tensor,
                    in_proj.weights,
                    in_proj.scales,
                    in_proj.biases,
                    self.conv_weight,
                    if (self.conv_bias) |bias| bias else null,
                    self.a_log,
                    if (self.dt_bias) |bias| bias else null,
                    if (self.norm_weight) |weight| weight else null,
                    out_proj.weights,
                    out_proj.scales,
                    out_proj.biases,
                    in_proj.group_size,
                    in_proj.bits,
                    cache_handle,
                    state.layer_idx,
                    self.d_conv,
                    self.n_heads,
                    self.n_key_heads,
                    self.d_head,
                    &state.capture_in_proj,
                    &state.capture_conv,
                    &state.capture_ssm,
                    &state.capture_norm,
                );
            } else {
                output_tensor.* = mlx_fused.mlx_lazy_gated_delta_mixer_quantized(
                    input_tensor,
                    in_proj.weights,
                    in_proj.scales,
                    in_proj.biases,
                    self.conv_weight,
                    if (self.conv_bias) |bias| bias else null,
                    self.a_log,
                    if (self.dt_bias) |bias| bias else null,
                    if (self.norm_weight) |weight| weight else null,
                    out_proj.weights,
                    out_proj.scales,
                    out_proj.biases,
                    in_proj.group_size,
                    in_proj.bits,
                    cache_handle,
                    state.layer_idx,
                    self.d_conv,
                    self.n_heads,
                    self.n_key_heads,
                    self.d_head,
                );
            }
            return;
        }

        if (state.capture_enabled) {
            output_tensor.* = mlx_fused.mlx_lazy_gated_delta_mixer_bf16_capture(
                input_tensor,
                self.in_proj_bf16.?,
                self.conv_weight,
                if (self.conv_bias) |bias| bias else null,
                self.a_log,
                if (self.dt_bias) |bias| bias else null,
                if (self.norm_weight) |weight| weight else null,
                self.out_proj_bf16.?,
                cache_handle,
                state.layer_idx,
                self.d_conv,
                self.n_heads,
                self.n_key_heads,
                self.d_head,
                &state.capture_in_proj,
                &state.capture_conv,
                &state.capture_ssm,
                &state.capture_norm,
            );
        } else {
            output_tensor.* = mlx_fused.mlx_lazy_gated_delta_mixer_bf16(
                input_tensor,
                self.in_proj_bf16.?,
                self.conv_weight,
                if (self.conv_bias) |bias| bias else null,
                self.a_log,
                if (self.dt_bias) |bias| bias else null,
                if (self.norm_weight) |weight| weight else null,
                self.out_proj_bf16.?,
                cache_handle,
                state.layer_idx,
                self.d_conv,
                self.n_heads,
                self.n_key_heads,
                self.d_head,
            );
        }
    }
};
