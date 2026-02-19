//! Metal backend executor module root.
//!
//! Groups Metal execution-time orchestration helpers.

pub const weights = @import("weights.zig");
pub const runtime = @import("runtime.zig");
pub const model = @import("model.zig");
pub const block = @import("block.zig");
const kernels = @import("../kernels/root.zig");

// Canonical executor aliases (kept symmetric with CPU executor root).
pub const Model = model.Model;
pub const Transformer = model.Model;
pub const Block = block.TransformerBlock;
pub const TransformerBlock = block.TransformerBlock;
pub const BlockKind = weights.WeightHandles.LayerWeights.LayerKind;

// Kernel/scratch aliases for cross-backend discoverability.
pub const Attention = kernels.attention.MultiHeadAttention;
pub const RMSNorm = kernels.norm.RMSNorm;
pub const FFNLayer = weights.WeightHandles.LayerWeights;
pub const AttnTemp = kernels.attention.AttnTemp;
pub const AttnCache = kernels.attention.AttnCache;
pub const ScratchBuffer = struct {};
