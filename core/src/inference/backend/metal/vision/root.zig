//! Metal vision boundary.
//!
//! This module deliberately exposes a Metal-owned runtime type even while
//! encode/scatter behavior is still delegated to the current shared
//! implementation. Keeping this boundary explicit prevents accidental
//! re-introduction of direct CPU type aliasing in the backend surface.

const std = @import("std");
const models = @import("models_pkg");
const shared = @import("../../../vision_types.zig");
const cpu_vision = @import("../../cpu/vision/root.zig");

const LoadedModel = models.LoadedModel;

pub const PrefillVisionImage = shared.PrefillVisionImage;
pub const PrefillVisionInput = shared.PrefillVisionInput;
pub const EncodedVisionOutput = shared.EncodedVisionOutput;
pub const VisionGrid = shared.VisionGrid;

pub const VisionRuntime = struct {
    impl: cpu_vision.VisionRuntime,

    pub fn init(allocator: std.mem.Allocator, loaded: *LoadedModel) !?VisionRuntime {
        const rt = (try cpu_vision.VisionRuntime.init(allocator, loaded)) orelse return null;
        return .{ .impl = rt };
    }

    pub fn deinit(self: *VisionRuntime) void {
        self.impl.deinit();
    }

    pub fn encodeImages(self: *VisionRuntime, images: []const PrefillVisionImage) !EncodedVisionOutput {
        return self.impl.encodeImages(images);
    }

    pub fn scatterIntoHidden(
        self: *VisionRuntime,
        hidden_states: []f32,
        seq_len: usize,
        d_model: usize,
        token_ids: []const u32,
        image_token_id: u32,
        embeddings: []const f32,
    ) !void {
        return self.impl.scatterIntoHidden(
            hidden_states,
            seq_len,
            d_model,
            token_ids,
            image_token_id,
            embeddings,
        );
    }
};

pub fn scatterVisionEmbeddings(
    hidden_states: []f32,
    seq_len: usize,
    d_model: usize,
    token_ids: []const u32,
    image_token_id: u32,
    embeddings: []const f32,
) !void {
    return cpu_vision.scatterVisionEmbeddings(
        hidden_states,
        seq_len,
        d_model,
        token_ids,
        image_token_id,
        embeddings,
    );
}

pub fn maxPixels() u64 {
    return cpu_vision.maxPixels();
}

test "maxPixels forwards to shared vision limit" {
    try std.testing.expectEqual(cpu_vision.maxPixels(), maxPixels());
}
