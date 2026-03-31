//! Metal backend vision module.
//!
//! Metal-native vision runtime module.
//!
//! Keeps vision preprocessing/runtime under `metal/vision/` so backend
//! ownership is explicit. Runtime compute executes on Metal through MLX.

const std = @import("std");
const native = @import("native_split_qkv.zig");

pub const PrefillVisionImage = native.PrefillVisionImage;
pub const PrefillVisionInput = native.PrefillVisionInput;
pub const EncodedVisionOutput = native.EncodedVisionOutput;
pub const VisionRuntime = native.VisionRuntime;
pub const scatterVisionEmbeddings = native.scatterVisionEmbeddings;

pub fn maxPixels() u64 {
    return 512 * 512;
}

test "maxPixels returns metal vision limit" {
    try std.testing.expectEqual(@as(u64, 512 * 512), maxPixels());
}
