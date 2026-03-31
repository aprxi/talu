//! Metal embedding kernel surface.

pub const supported = true;

const compute = @import("../../../../compute/root.zig");
const model = @import("../executor/model.zig");
const weights = @import("../executor/weights.zig");

const ArrayHandle = compute.metal.graph.ArrayHandle;

pub const WeightHandles = weights.WeightHandles;

pub const gatherTokenEmbeddingsLazy = model.gatherTokenEmbeddingsLazy;

pub const EmbeddingLookup = struct {
    /// Canonical kernel-call contract for backend parity checks.
    pub const ForwardParams = struct {
        token_ids: []const u32,
        output_tensor: *ArrayHandle,
    };

    weight_handles: *const WeightHandles,

    pub fn forward(
        self: *const EmbeddingLookup,
        token_ids: []const u32,
        output_tensor: *ArrayHandle,
    ) !void {
        output_tensor.* = try gatherTokenEmbeddingsLazy(self.weight_handles, token_ids);
    }
};

test {
    _ = EmbeddingLookup;
}
