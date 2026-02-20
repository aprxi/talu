//! Metal MoE kernel surface.

pub const supported = true;

const weights = @import("../executor/weights.zig");
const mlx_fused = @import("../mlx/ffi.zig");

const ArrayHandle = mlx_fused.ArrayHandle;

pub const WeightHandles = weights.WeightHandles;

pub const MoEScratch = struct {};
pub const MatmulScratch = struct {};

pub const MoEFFN = struct {
    /// Canonical kernel-call contract for backend parity checks.
    pub const ForwardParams = struct {
        input_tensor: ArrayHandle,
        output_tensor: *ArrayHandle,
        scratch: *MoEScratch,
        matmul_scratch: *MatmulScratch,
    };

    weights: *const WeightHandles.MoEWeights,

    pub fn forward(
        self: *const MoEFFN,
        input_tensor: ArrayHandle,
        output_tensor: *ArrayHandle,
        scratch: *MoEScratch,
        matmul_scratch: *MatmulScratch,
    ) !void {
        _ = scratch;
        _ = matmul_scratch;
        const moe = self.weights;
        output_tensor.* = mlx_fused.mlx_lazy_fused_moe_ffn_mxfp4(
            input_tensor,
            moe.router_w,
            if (moe.router_s) |rs| rs else null,
            if (moe.router_b) |rb| rb else null,
            if (moe.router_bias) |rb| rb else null,
            moe.gate_w,
            moe.gate_s,
            moe.up_w,
            moe.up_s,
            moe.down_w,
            moe.down_s,
            if (moe.gate_bias) |gb| gb else null,
            if (moe.up_bias) |ub| ub else null,
            if (moe.down_bias) |db| db else null,
            moe.num_experts,
            moe.experts_per_token,
            moe.router_group_size,
            moe.expert_group_size,
        );
    }
};
