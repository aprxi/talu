//! CPU backend executor module root.
//!
//! Groups CPU execution-time orchestration helpers.

pub const weights = @import("weights.zig");
pub const runtime = @import("runtime.zig");
pub const model = @import("model.zig");
pub const block = @import("block.zig");
pub const layers = struct {
    pub const formatLinearLike = model.formatLinearLike;
    pub const formatRmsNormLike = model.formatRmsNormLike;
    pub const Linear = model.Linear;
    pub const Embedding = model.Embedding;
};

pub const Model = model.Model;
pub const Transformer = model.Model;
pub const Block = block.Block;
pub const TransformerBlock = block.TransformerBlock;

pub const Linear = layers.Linear;
pub const Embedding = layers.Embedding;

const ops = @import("../graph.zig").layer_ops;
pub const LayerOp = ops.LayerOp;
pub const BufferId = ops.BufferId;

const kernels = @import("../kernels/root.zig");
pub const Attention = kernels.attention.MultiHeadAttention;
pub const RMSNorm = kernels.norm.RMSNorm;
pub const FFNLayer = weights.FfnLayer;
pub const BlockKind = weights.BlockType;
pub const AttnTemp = runtime.AttnTemp;
pub const AttnCache = runtime.AttnCache;
pub const ScratchBuffer = runtime.ScratchBuffer;
