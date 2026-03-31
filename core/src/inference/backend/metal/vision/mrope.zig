//! Metal mRoPE helpers.
//!
//! mRoPE semantics are currently shared with CPU and reused explicitly via
//! this backend-local shim for import symmetry.

const cpu_mrope = @import("../../../vision_mrope.zig");

pub const resolveMropeSection = cpu_mrope.resolveMropeSection;
pub const applyPositionDelta = cpu_mrope.applyPositionDelta;
pub const buildMultimodalMropePositions = cpu_mrope.buildMultimodalMropePositions;
pub const computePositionDelta = cpu_mrope.computePositionDelta;
