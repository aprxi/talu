//! CPU vision type aliases.
//!
//! Shared vision payload contracts are owned at `inference/vision_types.zig`.

const shared = @import("../../../vision_types.zig");

pub const PrefillVisionImage = shared.PrefillVisionImage;
pub const PrefillVisionInput = shared.PrefillVisionInput;
pub const EncodedVisionOutput = shared.EncodedVisionOutput;

