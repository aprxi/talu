//! Executor - transformer model execution.
//!
//! This module contains the execution logic that runs transformer models
//! using the compute kernels. It bridges model types (from model/) with
//! kernel implementations (from compute/backend/).
//!
//! Main types:
//! - Block: Transformer block with forward() execution
//! - Model: Complete transformer model

pub const common = @import("common.zig");

pub const Block = @import("block.zig").Block;
pub const Transformer = @import("model.zig").Transformer;

// Re-export model types for convenience
const ops = @import("../../graph/root.zig").layer_ops;
pub const LayerOp = ops.LayerOp;
pub const BufferId = ops.BufferId;

// Re-export kernel types needed by callers
pub const Attention = common.Attention;
pub const RMSNorm = common.RMSNorm;
pub const FFNLayer = common.FFNLayer;
pub const AttnTemp = common.AttnTemp;
pub const AttnCache = common.AttnCache;
pub const ScratchBuffer = common.ScratchBuffer;
pub const TransformerBlock = common.TransformerBlock;

// Re-export layer types
pub const layers = @import("layers.zig");
pub const Linear = layers.Linear;
pub const Embedding = layers.Embedding;
