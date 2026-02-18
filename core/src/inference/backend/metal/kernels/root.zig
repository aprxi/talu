//! Metal backend kernel surface.
//!
//! Backends expose architecture-specific kernel modules and explicit support flags.

pub const support = .{
    .attention = true,
    .describe_fmt = true,
    .embedding = true,
    .ffn = true,
    .fused_attention = true,
    .kv_cache = true,
    .mamba = false,
    .mla_attention = false,
    .moe = true,
    .norm = true,
    .rope = true,
    .shortconv = true,
    .weights = true,
};

const compute = @import("../../../../compute/root.zig");

pub const attention = @import("attention.zig");
pub const describe_fmt = @import("describe_fmt.zig");
pub const embedding = @import("embedding.zig");
pub const ffn = @import("ffn.zig");
pub const fused_attention = @import("fused_attention.zig");
pub const kv_cache = @import("kv_cache.zig");
pub const mamba = @import("mamba.zig");
pub const mla_attention = @import("mla_attention.zig");
pub const moe = @import("moe.zig");
pub const norm = @import("norm.zig");
pub const rope = @import("rope.zig");
pub const shortconv = @import("shortconv.zig");
pub const weights = @import("weights.zig");

pub const TransformerBlock = @import("../executor/block.zig").TransformerBlock;
pub const MultiHeadAttention = attention.MultiHeadAttention;
pub const SwiGLU = ffn.SwiGLU;
pub const RMSNorm = norm.RMSNorm;
pub const ShortConvKernel = shortconv.ShortConvKernel;
pub const MoEFFN = moe.MoEFFN;
pub const EmbeddingLookup = embedding.EmbeddingLookup;
pub const KVCache = kv_cache.KVCache;
pub const FusedAttention = fused_attention.FusedAttention;
pub const RotaryEmbedding = rope.RotaryEmbedding;
pub const WeightAccess = weights.WeightAccess;

pub const matmul = compute.metal.matmul;
pub const graph = compute.metal.graph;
pub const device = compute.metal.device;
