//! Metal ShortConv kernel surface.

pub const supported = true;

const weights = @import("../executor/weights.zig");
const cache_executor = @import("../executor/runtime.zig");
const mlx_fused = @import("../mlx/ffi.zig");

const ArrayHandle = mlx_fused.ArrayHandle;

pub const WeightHandles = weights.WeightHandles;
pub const ShortConvCache = cache_executor.ShortConvCache;

pub const ShortConvState = struct {
    cache: ?ShortConvCache = null,
    layer_idx: usize = 0,
};

pub const ShortConvScratch = struct {};
pub const MatmulScratch = struct {};

pub const ShortConvKernel = struct {
    /// Canonical kernel-call contract for backend parity checks.
    pub const ForwardParams = struct {
        input_tensor: ArrayHandle,
        output_tensor: *ArrayHandle,
        state: *ShortConvState,
        scratch: *ShortConvScratch,
        matmul_scratch: *MatmulScratch,
    };

    const QuantizedWeight = WeightHandles.QuantizedWeight;

    in_proj: ?QuantizedWeight = null,
    out_proj: ?QuantizedWeight = null,
    in_proj_bf16: ?ArrayHandle = null,
    out_proj_bf16: ?ArrayHandle = null,
    conv_weight: ArrayHandle,
    conv_bias: ?ArrayHandle = null,
    d_conv: usize,
    conv_dim: usize,

    pub fn forward(
        self: *const ShortConvKernel,
        input_tensor: ArrayHandle,
        output_tensor: *ArrayHandle,
        state: *ShortConvState,
        scratch: *ShortConvScratch,
        matmul_scratch: *MatmulScratch,
    ) !void {
        _ = scratch;
        _ = matmul_scratch;

        const cache_handle = if (state.cache) |sc| sc.handle else null;

        if (self.in_proj != null and self.out_proj != null) {
            const in_proj = self.in_proj.?;
            const out_proj = self.out_proj.?;
            output_tensor.* = mlx_fused.mlx_lazy_shortconv_mixer_quantized(
                input_tensor,
                in_proj.weights,
                in_proj.scales,
                in_proj.biases,
                self.conv_weight,
                if (self.conv_bias) |b| b else null,
                out_proj.weights,
                out_proj.scales,
                out_proj.biases,
                in_proj.group_size,
                in_proj.bits,
                cache_handle,
                state.layer_idx,
                self.d_conv,
                self.conv_dim,
            );
            return;
        }

        if (self.in_proj_bf16 != null and self.out_proj_bf16 != null) {
            output_tensor.* = mlx_fused.mlx_lazy_shortconv_mixer_bf16(
                input_tensor,
                self.in_proj_bf16.?,
                self.conv_weight,
                if (self.conv_bias) |b| b else null,
                self.out_proj_bf16.?,
                cache_handle,
                state.layer_idx,
                self.d_conv,
                self.conv_dim,
            );
            return;
        }

        return error.MissingField;
    }
};
