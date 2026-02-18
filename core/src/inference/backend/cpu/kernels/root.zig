//! CPU kernel exports
//!
//! This is a convenience entrypoint for CPU-only kernel types and scratch buffers.
//! It is intentionally separate from `src/compute/backend/cpu/root.zig` (the backend
//! object) to keep the backend API focused.

pub const support = .{
    .attention = true,
    .describe_fmt = true,
    .embedding = true,
    .ffn = true,
    .fused_attention = true,
    .kv_cache = true,
    .mamba = true,
    .mla_attention = true,
    .moe = true,
    .norm = true,
    .rope = true,
    .shortconv = true,
    .weights = true,
};

const block_kernels = @import("../executor/weights.zig");
const moe_kernels = @import("moe.zig");
pub const kv_cache = @import("kv_cache.zig");
pub const attention = @import("attention.zig");
pub const describe_fmt = @import("describe_fmt.zig");
pub const embedding = @import("embedding.zig");
pub const ffn = @import("ffn.zig");
pub const fused_attention = @import("fused_attention.zig");
pub const mamba = @import("mamba.zig");
pub const mla_attention = @import("mla_attention.zig");
pub const moe = @import("moe.zig");
pub const norm = @import("norm.zig");
pub const rope = @import("rope.zig");
pub const shortconv = @import("shortconv.zig");
pub const weights = @import("weights.zig");

// Block containers + scratch
pub const TransformerBlock = block_kernels.TransformerBlock;
pub const ScratchBuffer = block_kernels.ScratchBuffer;
pub const EmbeddingLookup = embedding.EmbeddingLookup;
pub const KVCache = kv_cache.KVCache;
pub const FusedAttention = fused_attention.FusedAttention;
pub const RotaryEmbedding = rope.RotaryEmbedding;
pub const WeightAccess = weights.WeightAccess;

// Attention / FFN kernel structs and scratch
pub const AttnTemp = block_kernels.AttnTemp;
pub const AttnCache = block_kernels.AttnCache;
pub const FfnScratch = block_kernels.FfnScratch;
pub const MultiHeadAttention = block_kernels.MultiHeadAttention;
pub const SwiGLU = block_kernels.SwiGLU;
pub const GateUpLayout = block_kernels.GateUpLayout;
pub const RMSNorm = block_kernels.RMSNorm;
pub const RoPE = block_kernels.RoPE;

// Common CPU kernel entrypoints
pub const rmsnormForward = block_kernels.rmsnormForward;
pub const gatherEmbeddings = block_kernels.gatherEmbeddings;

// MoE kernel exports
pub const MoEFFN = moe_kernels.MoEFFN;
pub const MoEScratch = moe_kernels.MoEScratch;
pub const ExpertWeights = moe_kernels.ExpertWeights;

// Mamba kernel exports (for heterogeneous models)
const mamba_kernels = @import("mamba.zig");
pub const MambaKernel = mamba_kernels.MambaKernel;
pub const MambaConfig = mamba_kernels.MambaConfig;
pub const MambaWeights = mamba_kernels.MambaWeights;
pub const MambaState = mamba_kernels.MambaState;
pub const MambaScratch = mamba_kernels.MambaScratch;

// ShortConv kernel exports (for heterogeneous models)
const shortconv_kernels = @import("shortconv.zig");
pub const ShortConvKernel = shortconv_kernels.ShortConvKernel;
pub const ShortConvConfig = shortconv_kernels.ShortConvConfig;
pub const ShortConvWeights = shortconv_kernels.ShortConvWeights;
pub const ShortConvState = shortconv_kernels.ShortConvState;
pub const ShortConvScratch = shortconv_kernels.ShortConvScratch;

// Batched KV cache for continuous batching
pub const BatchedKVCache = kv_cache.BatchedKVCache;
pub const LayeredBatchedKVCache = kv_cache.LayeredBatchedKVCache;
pub const SlotState = kv_cache.SlotState;

// Batched attention types
const attention_kernels = @import("attention.zig");
pub const BatchedDecodeRequest = attention_kernels.BatchedDecodeRequest;
pub const BatchedAttnTemp = attention_kernels.BatchedAttnTemp;
pub const forwardBatchedDecode = attention_kernels.forwardBatchedDecode;
