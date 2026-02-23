//! CUDA vision runtime wiring.
//!
//! Vision preprocessing is shared host-side runtime logic and is reused by
//! CUDA backend for multimodal prefill.

const cpu_vision = @import("../../cpu/vision/root.zig");
const std = @import("std");

pub const PrefillVisionImage = cpu_vision.PrefillVisionImage;
pub const PrefillVisionInput = cpu_vision.PrefillVisionInput;
pub const EncodedVisionOutput = cpu_vision.EncodedVisionOutput;
pub const VisionRuntime = cpu_vision.VisionRuntime;
pub const maxPixels = cpu_vision.maxPixels;
pub const scatterVisionEmbeddings = cpu_vision.scatterVisionEmbeddings;

test "maxPixels is wired to shared vision runtime limit" {
    try std.testing.expect(maxPixels() > 0);
}
