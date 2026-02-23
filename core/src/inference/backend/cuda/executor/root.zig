//! CUDA executor contract module.
//!
//! This module currently provides contract-shape types only. Runtime execution
//! lives in `engine.zig`.

const topology = @import("../../../../models/op_types.zig");
const kernels = @import("../kernels/root.zig");
const std = @import("std");

pub const weights = struct {
    pub const BlockType = topology.BlockKind;
    pub const FfnLayer = struct {};
};

pub const runtime = struct {
    pub const AttnTemp = struct {};
    pub const AttnCache = struct {};
    pub const ScratchBuffer = struct {};
};

pub const model = struct {
    pub const Model = struct {
        pub fn forward(self: *@This()) !void {
            _ = self;
            return error.UnsupportedModel;
        }
    };
};

pub const block = struct {
    pub const TransformerBlock = struct {
        pub fn forward(self: *@This()) !void {
            _ = self;
            return error.UnsupportedModel;
        }
    };

    pub const Block = @This().TransformerBlock;
};

pub const Model = model.Model;
pub const Transformer = model.Model;
pub const Block = block.TransformerBlock;
pub const TransformerBlock = block.TransformerBlock;

pub const Attention = kernels.attention.MultiHeadAttention;
pub const RMSNorm = kernels.norm.RMSNorm;
pub const FFNLayer = weights.FfnLayer;
pub const BlockKind = weights.BlockType;
pub const AttnTemp = runtime.AttnTemp;
pub const AttnCache = runtime.AttnCache;
pub const ScratchBuffer = runtime.ScratchBuffer;

test "model.forward returns UnsupportedModel for CUDA contract shim" {
    var m: Model = .{};
    try std.testing.expectError(error.UnsupportedModel, m.forward());
}

test "block.forward returns UnsupportedModel for CUDA contract shim" {
    var b: TransformerBlock = .{};
    try std.testing.expectError(error.UnsupportedModel, b.forward());
}
