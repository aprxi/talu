//! FFN operator family root.

pub const route = @import("route.zig");
pub const fused_gate_up = @import("fused_gate_up.zig");
pub const step = @import("step.zig");

pub const runGateUpProjectionWithWeights = route.runGateUpProjectionWithWeights;
pub const runFfnActivationMul = route.runFfnActivationMul;
pub const canFuseGaffineGateUpWeights = fused_gate_up.canFuseGaffineGateUpWeights;
pub const tryFusedGateUpForward = fused_gate_up.tryFusedGateUpForward;
pub const tryFusedGaffineU8GateUpForward = fused_gate_up.tryFusedGaffineU8GateUpForward;
pub const tryFusedGaffineU8GateUpSiluForward = fused_gate_up.tryFusedGaffineU8GateUpSiluForward;
pub const tryFusedGaffineU4GateUpSiluForward = fused_gate_up.tryFusedGaffineU4GateUpSiluForward;
pub const tryFusedFp8GateUpSiluForward = fused_gate_up.tryFusedFp8GateUpSiluForward;
pub const tryFusedFp8GateUpForward = fused_gate_up.tryFusedFp8GateUpForward;
pub const tryFusedMxfp8GateUpSiluForward = fused_gate_up.tryFusedMxfp8GateUpSiluForward;
pub const tryFusedMxfp8GateUpForward = fused_gate_up.tryFusedMxfp8GateUpForward;
pub const tryFusedNvfp4GateUpSiluForward = fused_gate_up.tryFusedNvfp4GateUpSiluForward;
pub const tryFusedNvfp4GateUpGeluForward = fused_gate_up.tryFusedNvfp4GateUpGeluForward;
pub const tryFusedNvfp4GateUpForward = fused_gate_up.tryFusedNvfp4GateUpForward;
pub const tryFusedNvfp4GateUpLtForward = fused_gate_up.tryFusedNvfp4GateUpLtForward;
pub const tryFusedDenseU16GateUpForward = fused_gate_up.tryFusedDenseU16GateUpForward;
pub const tryFusedDenseU16GateUpSiluForward = fused_gate_up.tryFusedDenseU16GateUpSiluForward;
pub const applyBiasF32 = step.applyBiasF32;
pub const runFfnStep = step.runFfnStep;
