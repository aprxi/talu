//! CPU weight/container kernel surface.

const impl = @import("../executor/weights.zig");

pub const TransformerBlock = impl.TransformerBlock;
pub const ScratchBuffer = impl.ScratchBuffer;
pub const BlockWeights = impl.BlockWeights;

pub const WeightAccess = struct {
    /// Canonical kernel-call contract for backend parity checks.
    pub const ForwardParams = struct {
        weight_index: usize,
        output_weight: **const TransformerBlock,
    };

    blocks: []const TransformerBlock,

    pub fn forward(
        self: *const WeightAccess,
        weight_index: usize,
        output_weight: **const TransformerBlock,
    ) !void {
        if (weight_index >= self.blocks.len) return error.InvalidArgument;
        output_weight.* = &self.blocks[weight_index];
    }
};

test "WeightAccess.forward rejects out-of-range index" {
    const empty: [0]TransformerBlock = .{};
    const access = WeightAccess{ .blocks = empty[0..] };
    var out: *const TransformerBlock = undefined;
    try @import("std").testing.expectError(error.InvalidArgument, access.forward(0, &out));
}
