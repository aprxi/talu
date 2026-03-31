//! Metal weight-loading and lifecycle kernel surface.

pub const supported = true;

const weights = @import("../executor/weights.zig");

pub const MLXError = weights.MLXError;
pub const WeightHandles = weights.WeightHandles;

pub const loadWeightsToGPU = weights.loadWeightsToGPU;
pub const freeWeights = weights.freeWeights;

pub const WeightAccess = struct {
    /// Canonical kernel-call contract for backend parity checks.
    pub const ForwardParams = struct {
        weight_index: usize,
        output_weight: **const WeightHandles.LayerWeights,
    };

    weight_handles: *const WeightHandles,

    pub fn forward(
        self: *const WeightAccess,
        weight_index: usize,
        output_weight: **const WeightHandles.LayerWeights,
    ) !void {
        if (weight_index >= self.weight_handles.layers.len) return error.InvalidArgument;
        output_weight.* = &self.weight_handles.layers[weight_index];
    }
};

test {
    _ = WeightAccess;
}
