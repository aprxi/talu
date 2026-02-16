//! Vision runtime selector.
//!
//! Chooses between supported vision attention layouts:
//! - fused QKV attention weights
//! - split Q/K/V attention weights
//!
//! Selection is signature-based (tensor layout), not model-family based.

const std = @import("std");
const io = @import("../../../io/root.zig");

const fused_qkv_runtime = @import("vision_runtime_fused_qkv.zig");
const split_qkv_runtime = @import("vision_runtime_split_qkv.zig");

const LoadedModel = io.weights.LoadedModel;
const SafeTensors = io.safetensors.root.UnifiedSafeTensors;

pub const PrefillVisionImage = fused_qkv_runtime.PrefillVisionImage;
pub const PrefillVisionInput = fused_qkv_runtime.PrefillVisionInput;
pub const EncodedVisionOutput = fused_qkv_runtime.EncodedVisionOutput;

const VisionAttentionLayout = enum {
    fused_qkv,
    split_qkv,
    unknown,
};

pub const VisionRuntime = struct {
    allocator: std.mem.Allocator,
    impl: union(enum) {
        fused_qkv: fused_qkv_runtime.VisionRuntime,
        split_qkv: split_qkv_runtime.VisionRuntime,
    },

    pub fn init(allocator: std.mem.Allocator, loaded: *LoadedModel) !?VisionRuntime {
        const detected_layout = detectVisionAttentionLayout(loaded);

        switch (detected_layout) {
            .fused_qkv => {
                if (try fused_qkv_runtime.VisionRuntime.init(allocator, loaded)) |rt| {
                    return .{ .allocator = allocator, .impl = .{ .fused_qkv = rt } };
                }
                if (try split_qkv_runtime.VisionRuntime.init(allocator, loaded)) |rt| {
                    return .{ .allocator = allocator, .impl = .{ .split_qkv = rt } };
                }
            },
            .split_qkv => {
                if (try split_qkv_runtime.VisionRuntime.init(allocator, loaded)) |rt| {
                    return .{ .allocator = allocator, .impl = .{ .split_qkv = rt } };
                }
                if (try fused_qkv_runtime.VisionRuntime.init(allocator, loaded)) |rt| {
                    return .{ .allocator = allocator, .impl = .{ .fused_qkv = rt } };
                }
            },
            .unknown => {
                if (try fused_qkv_runtime.VisionRuntime.init(allocator, loaded)) |rt| {
                    return .{ .allocator = allocator, .impl = .{ .fused_qkv = rt } };
                }
                if (try split_qkv_runtime.VisionRuntime.init(allocator, loaded)) |rt| {
                    return .{ .allocator = allocator, .impl = .{ .split_qkv = rt } };
                }
            },
        }

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

fn detectVisionAttentionLayout(loaded: *LoadedModel) VisionAttentionLayout {
    if (loaded.st == null) return .unknown;

    const st = &loaded.st.?;
    if (hasAnyTensor(st, &.{
        "model.visual.blocks.0.attn.qkv.weight",
    })) {
        return .fused_qkv;
    }

    if (hasAnyTensor(st, &.{
        "model.visual.blocks.0.attn.q_proj.weight",
        "model.visual.blocks.0.self_attn.q_proj.weight",
        "model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight",
    })) {
        return .split_qkv;
    }

    return .unknown;
}

fn hasAnyTensor(st: *SafeTensors, candidates: []const []const u8) bool {
    for (candidates) |name| {
        _ = st.getTensor(name, null) catch continue;
        return true;
    }
    return false;
}
