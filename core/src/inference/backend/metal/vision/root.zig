//! Metal backend vision module.
//!
//! Vision preprocessing/runtime currently delegates to CPU implementation.
//! Keeping this under `metal/vision/` makes delegation explicit and keeps
//! backend module layout symmetric.

const cpu_vision_types = @import("../../cpu/vision/types.zig");
const cpu_vision = @import("../../cpu/vision/root.zig");

pub const PrefillVisionImage = cpu_vision_types.PrefillVisionImage;
pub const PrefillVisionInput = cpu_vision_types.PrefillVisionInput;
pub const EncodedVisionOutput = cpu_vision_types.EncodedVisionOutput;

pub const VisionRuntime = cpu_vision.VisionRuntime;
pub const scatterVisionEmbeddings = cpu_vision.scatterVisionEmbeddings;
