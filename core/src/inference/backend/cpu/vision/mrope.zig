//! CPU vision mRoPE aliases.

const shared = @import("../../../vision_mrope.zig");

pub const PrefillVisionImage = shared.PrefillVisionImage;
pub const resolveMropeSection = shared.resolveMropeSection;
pub const applyPositionDelta = shared.applyPositionDelta;
pub const buildMultimodalMropePositions = shared.buildMultimodalMropePositions;
pub const computePositionDelta = shared.computePositionDelta;

