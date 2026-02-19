//! Shared inference backend topology primitives.
//!
//! This module defines backend-agnostic topology contracts used by CPU/Metal
//! executor layers so block-kind semantics stay aligned.

const std = @import("std");

/// Canonical block kinds for heterogeneous model topologies.
pub const BlockKind = enum {
    /// Standard transformer block with attention and FFN.
    attention_mlp,
    /// Mamba2 state-space mixer block.
    mamba,
    /// ShortConv gated convolution block.
    shortconv,

    /// Convert a variant name string to canonical block kind.
    pub fn fromVariantName(name: []const u8) ?BlockKind {
        const known = std.StaticStringMap(BlockKind).initComptime(.{
            .{ "attention", .attention_mlp },
            .{ "attention_mlp", .attention_mlp },
            .{ "transformer", .attention_mlp },
            .{ "full_attention", .attention_mlp },
            .{ "sliding_attention", .attention_mlp },
            .{ "linear_attention", .mamba },
            .{ "mamba", .mamba },
            .{ "mamba2", .mamba },
            .{ "ssm", .mamba },
            .{ "shortconv", .shortconv },
            .{ "conv", .shortconv },
        });
        return known.get(name);
    }
};

/// Fused-model layer kind identifiers for Metal compute bindings.
///
/// Keep this mapping centralized so backend-kind semantics do not drift
/// across Zig and fused-call sites.
pub const FusedLayerKindId = enum(u8) {
    attention_mlp = 0,
    shortconv = 1,
};

/// Returns the fused-model layer id for kinds supported by fused Metal decode.
/// `null` means the kind is not supported by fused Metal execution.
pub fn fusedLayerKindId(kind: BlockKind) ?FusedLayerKindId {
    return switch (kind) {
        .attention_mlp => .attention_mlp,
        .shortconv => .shortconv,
        .mamba => null,
    };
}
