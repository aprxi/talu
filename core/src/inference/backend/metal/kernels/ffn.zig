//! Metal FFN kernel surface.

pub const supported = true;

const weights = @import("../executor/weights.zig");
const mlx_fused = @import("../mlx/ffi.zig");

const ArrayHandle = mlx_fused.ArrayHandle;

pub const WeightHandles = weights.WeightHandles;

pub const FfnScratch = struct {};
pub const MatmulScratch = struct {};

pub const SwiGLU = struct {
    /// Canonical kernel-call contract for backend parity checks.
    pub const ForwardParams = struct {
        input_tensor: ArrayHandle,
        output_tensor: *ArrayHandle,
        scratch: *FfnScratch,
        matmul_scratch: *MatmulScratch,
    };

    const QuantizedWeight = WeightHandles.QuantizedWeight;

    use_gelu: bool = false,
    w1: ?QuantizedWeight = null,
    w2: ?QuantizedWeight = null,
    w3: ?QuantizedWeight = null,
    w1_bf16: ?ArrayHandle = null,
    w2_bf16: ?ArrayHandle = null,
    w3_bf16: ?ArrayHandle = null,

    pub fn forward(
        self: *const SwiGLU,
        input_tensor: ArrayHandle,
        output_tensor: *ArrayHandle,
        scratch: *FfnScratch,
        matmul_scratch: *MatmulScratch,
    ) !void {
        _ = scratch;
        _ = matmul_scratch;
        if (self.w1 != null and self.w2 != null and self.w3 != null) {
            const w1 = self.w1.?;
            const w2 = self.w2.?;
            const w3 = self.w3.?;
            output_tensor.* = mlx_fused.mlx_lazy_fused_ffn(
                input_tensor,
                w1.weights,
                w1.scales,
                w1.biases,
                w3.weights,
                w3.scales,
                w3.biases,
                w2.weights,
                w2.scales,
                w2.biases,
                w1.group_size,
                w1.bits,
                self.use_gelu,
            );
            return;
        }
        if (self.w1_bf16 != null and self.w2_bf16 != null and self.w3_bf16 != null) {
            output_tensor.* = mlx_fused.mlx_lazy_fused_ffn_bf16(
                input_tensor,
                self.w1_bf16.?,
                self.w3_bf16.?,
                self.w2_bf16.?,
            );
            return;
        }
        return error.MissingField;
    }
};
