//! QKV operator family root.

pub const route = @import("route.zig");
pub const fused = @import("fused.zig");

pub const runQkvProjection = route.runQkvProjection;
pub const tryFusedQkvForward = fused.tryFusedQkvForward;
pub const tryFusedGaffineU4QkvForward = fused.tryFusedGaffineU4QkvForward;
pub const tryFusedGaffineU8QkvForward = fused.tryFusedGaffineU8QkvForward;
pub const tryFusedNvfp4QkvForward = fused.tryFusedNvfp4QkvForward;
pub const tryFusedNvfp4QkvLtForward = fused.tryFusedNvfp4QkvLtForward;
pub const tryFusedDenseU16QkvForward = fused.tryFusedDenseU16QkvForward;
pub const canFuseDenseU16QkvWeights = fused.canFuseDenseU16QkvWeights;
pub const canFuseGaffineQkvWeights = fused.canFuseGaffineQkvWeights;
