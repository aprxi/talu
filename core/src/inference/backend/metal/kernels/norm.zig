//! Metal normalization kernel surface.

pub const supported = true;

const compute = @import("../../../../compute/root.zig");
const weights = @import("../executor/weights.zig");

const mlx_graph = compute.metal.graph;
const ArrayHandle = mlx_graph.ArrayHandle;

pub const WeightHandles = weights.WeightHandles;

pub const RMSNorm = struct {
    /// Canonical kernel-call contract for backend parity checks.
    pub const ForwardParams = struct {
        input: ArrayHandle,
        output: *ArrayHandle,
    };

    weight: ArrayHandle,
    eps: f32,

    pub fn forward(self: *const RMSNorm, input: ArrayHandle, output: *ArrayHandle) void {
        output.* = mlx_graph.mlx_lazy_rms_norm(input, self.weight, self.eps);
    }
};
