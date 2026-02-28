//! Vision runtime selector.
//!
//! Chooses between supported vision attention layouts:
//! - fused QKV attention weights
//! - split Q/K/V attention weights
//!
//! Selection is signature-based (tensor layout), not model-family based.

const std = @import("std");
const models = @import("../../../../models/root.zig");
const log = @import("../../../../log.zig");
const common_vision = @import("types.zig");

const fused_qkv_runtime = @import("fused_qkv.zig");
const split_qkv_runtime = @import("split_qkv.zig");

const LoadedModel = models.LoadedModel;

pub const PrefillVisionImage = common_vision.PrefillVisionImage;
pub const PrefillVisionInput = common_vision.PrefillVisionInput;
pub const EncodedVisionOutput = common_vision.EncodedVisionOutput;

/// CPU vision runtime uses serial O(n²) attention today.
pub const MAX_PIXELS: u64 = 512 * 512;

pub fn maxPixels() u64 {
    return MAX_PIXELS;
}

const VisionAttentionLayout = models.vision.AttentionLayout;

pub const VisionRuntime = struct {
    allocator: std.mem.Allocator,
    impl: union(enum) {
        fused_qkv: fused_qkv_runtime.VisionRuntime,
        split_qkv: split_qkv_runtime.VisionRuntime,
    },

    pub fn init(allocator: std.mem.Allocator, loaded: *LoadedModel) !?VisionRuntime {
        log.debug("inference", "Vision runtime init start", .{
            .vision_hidden_size = loaded.config.vision_hidden_size,
            .vision_depth = loaded.config.vision_depth,
            .vision_num_heads = loaded.config.vision_num_heads,
            .vision_intermediate_size = loaded.config.vision_intermediate_size,
            .projector_hidden_size = loaded.config.projector_hidden_size,
            .vision_patch_size = loaded.config.vision_patch_size,
            .vision_spatial_merge_size = loaded.config.vision_spatial_merge_size,
            .vision_temporal_patch_size = loaded.config.vision_temporal_patch_size,
            .vision_num_position_embeddings = loaded.config.vision_num_position_embeddings,
            .vision_max_num_patches = loaded.config.vision_max_num_patches,
            .has_safetensors = @as(u8, @intFromBool(loaded.st != null)),
        }, @src());
        try models.vision.hydrateVisionConfigFromWeights(loaded);
        log.debug("inference", "Vision runtime config hydrated", .{
            .vision_hidden_size = loaded.config.vision_hidden_size,
            .vision_depth = loaded.config.vision_depth,
            .vision_num_heads = loaded.config.vision_num_heads,
            .vision_intermediate_size = loaded.config.vision_intermediate_size,
            .projector_hidden_size = loaded.config.projector_hidden_size,
            .vision_patch_size = loaded.config.vision_patch_size,
            .vision_spatial_merge_size = loaded.config.vision_spatial_merge_size,
            .vision_temporal_patch_size = loaded.config.vision_temporal_patch_size,
            .vision_num_position_embeddings = loaded.config.vision_num_position_embeddings,
            .vision_max_num_patches = loaded.config.vision_max_num_patches,
        }, @src());
        const detected_layout = models.vision.detectVisionAttentionLayout(loaded);
        log.debug("inference", "Vision runtime attention layout detected", .{
            .layout = @tagName(detected_layout),
        }, @src());

        switch (detected_layout) {
            .fused_qkv => {
                if (try fused_qkv_runtime.VisionRuntime.init(allocator, loaded)) |rt| {
                    log.debug("inference", "Vision runtime selected", .{ .layout = "fused_qkv" }, @src());
                    return .{ .allocator = allocator, .impl = .{ .fused_qkv = rt } };
                }
                if (try split_qkv_runtime.VisionRuntime.init(allocator, loaded)) |rt| {
                    log.debug("inference", "Vision runtime selected", .{ .layout = "split_qkv" }, @src());
                    return .{ .allocator = allocator, .impl = .{ .split_qkv = rt } };
                }
            },
            .split_qkv => {
                if (try split_qkv_runtime.VisionRuntime.init(allocator, loaded)) |rt| {
                    log.debug("inference", "Vision runtime selected", .{ .layout = "split_qkv" }, @src());
                    return .{ .allocator = allocator, .impl = .{ .split_qkv = rt } };
                }
                if (try fused_qkv_runtime.VisionRuntime.init(allocator, loaded)) |rt| {
                    log.debug("inference", "Vision runtime selected", .{ .layout = "fused_qkv" }, @src());
                    return .{ .allocator = allocator, .impl = .{ .fused_qkv = rt } };
                }
            },
            .unknown => {
                if (try fused_qkv_runtime.VisionRuntime.init(allocator, loaded)) |rt| {
                    log.debug("inference", "Vision runtime selected", .{ .layout = "fused_qkv" }, @src());
                    return .{ .allocator = allocator, .impl = .{ .fused_qkv = rt } };
                }
                if (try split_qkv_runtime.VisionRuntime.init(allocator, loaded)) |rt| {
                    log.debug("inference", "Vision runtime selected", .{ .layout = "split_qkv" }, @src());
                    return .{ .allocator = allocator, .impl = .{ .split_qkv = rt } };
                }
            },
        }

        log.debug("inference", "Vision runtime unavailable", .{
            .vision_hidden_size = loaded.config.vision_hidden_size,
            .vision_depth = loaded.config.vision_depth,
            .vision_num_heads = loaded.config.vision_num_heads,
            .vision_intermediate_size = loaded.config.vision_intermediate_size,
        }, @src());
        return null;
    }

    pub fn deinit(self: *VisionRuntime) void {
        switch (self.impl) {
            .fused_qkv => |*rt| rt.deinit(),
            .split_qkv => |*rt| rt.deinit(),
        }
    }

    pub fn encodeImages(self: *VisionRuntime, images: []const PrefillVisionImage) !EncodedVisionOutput {
        switch (self.impl) {
            .fused_qkv => |*rt| return rt.encodeImages(images),
            .split_qkv => |*rt| {
                var split_images = try self.allocator.alloc(split_qkv_runtime.PrefillVisionImage, images.len);
                defer self.allocator.free(split_images);

                for (images, 0..) |img, idx| {
                    split_images[idx] = .{
                        .pixels = img.pixels,
                        .width = img.width,
                        .height = img.height,
                        .grid = img.grid,
                        .token_count = img.token_count,
                    };
                }

                const out = try rt.encodeImages(split_images);
                return .{
                    .merged_embeddings = out.merged_embeddings,
                    .deepstack_layer_embeddings = out.deepstack_layer_embeddings,
                };
            },
        }
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
        switch (self.impl) {
            .fused_qkv => |*rt| {
                return rt.scatterIntoHidden(
                    hidden_states,
                    seq_len,
                    d_model,
                    token_ids,
                    image_token_id,
                    embeddings,
                );
            },
            .split_qkv => |*rt| {
                return rt.scatterIntoHidden(
                    hidden_states,
                    seq_len,
                    d_model,
                    token_ids,
                    image_token_id,
                    embeddings,
                );
            },
        }
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
    return fused_qkv_runtime.scatterVisionEmbeddings(
        hidden_states,
        seq_len,
        d_model,
        token_ids,
        image_token_id,
        embeddings,
    );
}

test "maxPixels returns cpu vision limit" {
    try std.testing.expectEqual(@as(u64, 512 * 512), maxPixels());
}
