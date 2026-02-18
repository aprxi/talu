//! Metal fused-attention kernel surface.

pub const supported = true;

const attention = @import("attention.zig");
const weights = @import("../executor/weights.zig");

pub const WeightHandles = weights.WeightHandles;
pub const FusedAttention = attention.MultiHeadAttention;

test {
    _ = FusedAttention;
}
