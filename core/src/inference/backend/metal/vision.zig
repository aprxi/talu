//! Metal backend vision module.
//!
//! Current implementation delegates vision encoding to the CPU vision runtime.
//! This keeps backend boundaries explicit while preserving behavior.

const common = @import("../cpu/vision/types.zig");
const cpu_vision = @import("../cpu/vision/root.zig");

pub const PrefillVisionImage = common.PrefillVisionImage;
pub const PrefillVisionInput = common.PrefillVisionInput;
pub const EncodedVisionOutput = common.EncodedVisionOutput;

pub const VisionRuntime = cpu_vision.VisionRuntime;
pub const scatterVisionEmbeddings = cpu_vision.scatterVisionEmbeddings;
