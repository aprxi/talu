//! Vision runtime selector.
//!
//! Chooses between supported vision attention layouts:
//! - fused QKV attention weights
//! - split Q/K/V attention weights
//!
//! Selection is signature-based (tensor layout), not model-family based.

const std = @import("std");
const io = @import("../../../../io/root.zig");
const models = @import("../../../../models/root.zig");
const tensor = @import("../../../../tensor.zig");
const log = @import("../../../../log.zig");
const common_vision = @import("types.zig");

const fused_qkv_runtime = @import("fused_qkv.zig");
const split_qkv_runtime = @import("split_qkv.zig");

const LoadedModel = models.LoadedModel;
const SafeTensors = io.safetensors.root.UnifiedSafeTensors;

pub const PrefillVisionImage = common_vision.PrefillVisionImage;
pub const PrefillVisionInput = common_vision.PrefillVisionInput;
pub const EncodedVisionOutput = common_vision.EncodedVisionOutput;

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
        try hydrateVisionConfigFromWeights(loaded);
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
        const detected_layout = detectVisionAttentionLayout(loaded);
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
        "model.vision_model.encoder.layers.0.self_attn.q_proj.weight",
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

fn hydrateVisionConfigFromWeights(loaded: *LoadedModel) !void {
    if (loaded.st == null) return;

    var cfg = &loaded.config;
    const needs_hydration = cfg.vision_hidden_size <= 0 or
        cfg.vision_depth <= 0 or
        cfg.vision_num_heads <= 0 or
        cfg.vision_intermediate_size <= 0 or
        cfg.projector_hidden_size <= 0 or
        cfg.vision_patch_size <= 0 or
        cfg.vision_spatial_merge_size <= 0 or
        cfg.vision_temporal_patch_size <= 0 or
        cfg.vision_num_position_embeddings <= 0 or
        cfg.vision_max_num_patches <= 0;
    if (!needs_hydration) return;

    const st = &loaded.st.?;
    if (!hasAnyTensor(st, &.{
        "model.visual.patch_embed.proj.weight",
        "model.vision_tower.vision_model.embeddings.patch_embedding.weight",
        "model.vision_model.embeddings.patch_embedding.weight",
    })) return;

    const patch_tensor = getTensorByCandidates(st, &.{
        "model.visual.patch_embed.proj.weight",
        "model.vision_tower.vision_model.embeddings.patch_embedding.weight",
        "model.vision_model.embeddings.patch_embedding.weight",
    }) catch |err| switch (err) {
        error.NotFound => return,
        else => return err,
    };

    if (cfg.vision_hidden_size <= 0 and patch_tensor.n_dims > 0) {
        if (castPositiveI32(patch_tensor.shape[0])) |hidden| cfg.vision_hidden_size = hidden;
    }

    if (cfg.vision_temporal_patch_size <= 0) {
        if (inferTemporalPatchSize(patch_tensor)) |temporal| cfg.vision_temporal_patch_size = temporal;
    }
    if (cfg.vision_temporal_patch_size <= 0) cfg.vision_temporal_patch_size = 1;

    if (cfg.vision_patch_size <= 0) {
        const temporal_patch_size: usize = @intCast(cfg.vision_temporal_patch_size);
        if (inferPatchSize(patch_tensor, temporal_patch_size)) |patch| cfg.vision_patch_size = patch;
    }
    if (cfg.vision_patch_size <= 0) cfg.vision_patch_size = 16;

    const pos_tensor = getTensorByCandidates(st, &.{
        "model.visual.pos_embed.weight",
        "model.vision_tower.vision_model.embeddings.position_embedding.weight",
        "model.vision_model.embeddings.position_embedding.weight",
    }) catch |err| switch (err) {
        error.NotFound => null,
        else => return err,
    };
    if (cfg.vision_num_position_embeddings <= 0) {
        if (pos_tensor) |pos| {
            if (inferPositionEmbeddingCount(pos, cfg.vision_hidden_size)) |count| {
                cfg.vision_num_position_embeddings = count;
            }
        }
    }

    const merger_fc1_weight = getTensorByCandidates(st, &.{
        "model.visual.merger.linear_fc1.weight",
        "model.multi_modal_projector.linear_1.weight",
    }) catch |err| switch (err) {
        error.NotFound => null,
        else => return err,
    };

    if (cfg.vision_spatial_merge_size <= 0 and merger_fc1_weight != null and cfg.vision_hidden_size > 0) {
        if (inferSpatialMergeSize(merger_fc1_weight.?, @intCast(cfg.vision_hidden_size))) |merge_size| {
            cfg.vision_spatial_merge_size = merge_size;
        }
    }
    if (cfg.vision_spatial_merge_size <= 0) cfg.vision_spatial_merge_size = 1;

    if (cfg.projector_hidden_size <= 0 and merger_fc1_weight != null and cfg.vision_hidden_size > 0 and cfg.vision_spatial_merge_size > 0) {
        if (inferProjectorHiddenSize(
            merger_fc1_weight.?,
            @intCast(cfg.vision_hidden_size),
            @intCast(cfg.vision_spatial_merge_size),
        )) |projector_hidden_size| {
            cfg.projector_hidden_size = projector_hidden_size;
        }
    }

    if (cfg.vision_depth <= 0) {
        cfg.vision_depth = @intCast(try inferVisionDepth(st));
    }

    if (cfg.vision_num_heads <= 0 and cfg.vision_hidden_size > 0) {
        cfg.vision_num_heads = @intCast(inferVisionHeads(@intCast(cfg.vision_hidden_size)));
    }

    if (cfg.vision_intermediate_size <= 0 and cfg.vision_hidden_size > 0) {
        if (try inferVisionIntermediateSize(st, @intCast(cfg.vision_hidden_size))) |intermediate| {
            cfg.vision_intermediate_size = intermediate;
        }
    }

    if (cfg.vision_max_num_patches <= 0 and cfg.vision_num_position_embeddings > 0) {
        cfg.vision_max_num_patches = cfg.vision_num_position_embeddings;
    }
}

fn getTensorByCandidates(st: *SafeTensors, candidates: []const []const u8) !tensor.Tensor {
    for (candidates) |name| {
        const t = st.getTensor(name, null) catch |err| switch (err) {
            error.NotFound => continue,
            else => return err,
        };
        return t;
    }
    return error.NotFound;
}

fn getLayerTensorByTemplates(st: *SafeTensors, layer_idx: usize, templates: []const []const u8) !tensor.Tensor {
    var name_buf: [192]u8 = undefined;
    for (templates) |template| {
        const name = try expandLayerTemplate(name_buf[0..], template, layer_idx);
        const t = st.getTensor(name, null) catch |err| switch (err) {
            error.NotFound => continue,
            else => return err,
        };
        return t;
    }
    return error.NotFound;
}

fn expandLayerTemplate(buf: []u8, template: []const u8, layer_idx: usize) ![]const u8 {
    const marker = "{d}";
    if (std.mem.indexOf(u8, template, marker)) |idx| {
        return std.fmt.bufPrint(buf, "{s}{d}{s}", .{
            template[0..idx],
            layer_idx,
            template[idx + marker.len ..],
        });
    }
    return if (template.len <= buf.len) blk: {
        @memcpy(buf[0..template.len], template);
        break :blk buf[0..template.len];
    } else error.NoSpaceLeft;
}

fn castPositiveI32(value: i64) ?i32 {
    if (value <= 0) return null;
    return std.math.cast(i32, value);
}

fn inferTemporalPatchSize(patch_tensor: tensor.Tensor) ?i32 {
    if (patch_tensor.n_dims >= 5) return castPositiveI32(patch_tensor.shape[2]);
    if (patch_tensor.n_dims >= 4) return 1;
    if (patch_tensor.n_dims == 2) return 1;
    return null;
}

fn inferPatchSize(patch_tensor: tensor.Tensor, temporal_patch_size: usize) ?i32 {
    if (patch_tensor.n_dims >= 5) {
        return castPositiveI32(patch_tensor.shape[3]);
    }
    if (patch_tensor.n_dims >= 4) {
        return castPositiveI32(patch_tensor.shape[2]);
    }
    if (patch_tensor.n_dims == 2) {
        if (patch_tensor.shape[1] <= 0) return null;
        const patch_dim: usize = std.math.cast(usize, patch_tensor.shape[1]) orelse return null;
        const denom = std.math.mul(usize, 3, temporal_patch_size) catch return null;
        if (denom == 0 or patch_dim % denom != 0) return null;
        const area = patch_dim / denom;
        const side = std.math.sqrt(area);
        if (side * side != area) return null;
        return std.math.cast(i32, side);
    }
    return null;
}

fn inferPositionEmbeddingCount(pos_tensor: tensor.Tensor, vision_hidden_size: i32) ?i32 {
    if (pos_tensor.n_dims == 2 and vision_hidden_size > 0) {
        const hidden_i64: i64 = @intCast(vision_hidden_size);
        const dim0 = pos_tensor.shape[0];
        const dim1 = pos_tensor.shape[1];
        if (dim0 == hidden_i64 and dim1 > 0 and dim1 != hidden_i64) return std.math.cast(i32, dim1);
        if (dim1 == hidden_i64 and dim0 > 0 and dim0 != hidden_i64) return std.math.cast(i32, dim0);
    }
    if (pos_tensor.n_dims == 1 and vision_hidden_size > 0 and pos_tensor.shape[0] > 0) {
        const total: usize = std.math.cast(usize, pos_tensor.shape[0]) orelse return null;
        const hidden: usize = @intCast(vision_hidden_size);
        if (hidden == 0 or total % hidden != 0) return null;
        return std.math.cast(i32, total / hidden);
    }
    return null;
}

fn inferSpatialMergeSize(merger_fc1_weight: tensor.Tensor, vision_hidden_size: usize) ?i32 {
    if (merger_fc1_weight.n_dims != 2 or vision_hidden_size == 0) return null;

    const dim1 = std.math.cast(usize, merger_fc1_weight.shape[1]) orelse return null;
    const dim0 = std.math.cast(usize, merger_fc1_weight.shape[0]) orelse return null;
    const candidates = [_]usize{ dim1, dim0 };
    for (candidates) |dim| {
        if (dim == 0 or dim % vision_hidden_size != 0) continue;
        const merge_units = dim / vision_hidden_size;
        const side = std.math.sqrt(merge_units);
        if (side > 0 and side * side == merge_units) {
            return std.math.cast(i32, side);
        }
    }
    return null;
}

fn inferProjectorHiddenSize(
    merger_fc1_weight: tensor.Tensor,
    vision_hidden_size: usize,
    spatial_merge_size: usize,
) ?i32 {
    if (merger_fc1_weight.n_dims != 2 or vision_hidden_size == 0 or spatial_merge_size == 0) return null;

    const dim0 = std.math.cast(usize, merger_fc1_weight.shape[0]) orelse return null;
    const dim1 = std.math.cast(usize, merger_fc1_weight.shape[1]) orelse return null;
    const merge_units = std.math.mul(usize, spatial_merge_size, spatial_merge_size) catch return null;
    const merged_width = std.math.mul(usize, vision_hidden_size, merge_units) catch return null;

    if (dim0 == merged_width and dim1 > 0 and dim1 != merged_width) return std.math.cast(i32, dim1);
    if (dim1 == merged_width and dim0 > 0 and dim0 != merged_width) return std.math.cast(i32, dim0);
    return null;
}

fn inferVisionDepth(st: *SafeTensors) !usize {
    const split_depth = try countLayerDepth(st, &.{
        "model.visual.blocks.{d}.attn.q_proj.weight",
        "model.visual.blocks.{d}.self_attn.q_proj.weight",
        "model.vision_tower.vision_model.encoder.layers.{d}.self_attn.q_proj.weight",
        "model.vision_model.encoder.layers.{d}.self_attn.q_proj.weight",
    });
    if (split_depth > 0) return split_depth;

    const fused_depth = try countLayerDepth(st, &.{
        "model.visual.blocks.{d}.attn.qkv.weight",
    });
    return fused_depth;
}

fn countLayerDepth(st: *SafeTensors, templates: []const []const u8) !usize {
    var depth: usize = 0;
    var layer_idx: usize = 0;
    while (layer_idx < 512) : (layer_idx += 1) {
        _ = getLayerTensorByTemplates(st, layer_idx, templates) catch |err| switch (err) {
            error.NotFound => break,
            else => return err,
        };
        depth += 1;
    }
    return depth;
}

fn inferVisionHeads(vision_hidden_size: usize) usize {
    if (vision_hidden_size >= 64 and vision_hidden_size % 64 == 0) return vision_hidden_size / 64;
    if (vision_hidden_size >= 80 and vision_hidden_size % 80 == 0) return vision_hidden_size / 80;
    if (vision_hidden_size >= 128 and vision_hidden_size % 128 == 0) return vision_hidden_size / 128;
    if (vision_hidden_size >= 16 and vision_hidden_size % 16 == 0) return 16;
    return 1;
}

fn inferVisionIntermediateSize(st: *SafeTensors, vision_hidden_size: usize) !?i32 {
    const fc1 = getLayerTensorByTemplates(st, 0, &.{
        "model.visual.blocks.0.mlp.linear_fc1.weight",
        "model.vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight",
        "model.vision_model.encoder.layers.0.mlp.fc1.weight",
    }) catch |err| switch (err) {
        error.NotFound => return null,
        else => return err,
    };
    if (fc1.n_dims != 2) return null;
    const dim0 = std.math.cast(usize, fc1.shape[0]) orelse return null;
    const dim1 = std.math.cast(usize, fc1.shape[1]) orelse return null;

    if (dim0 == vision_hidden_size and dim1 > 0 and dim1 != vision_hidden_size) return std.math.cast(i32, dim1);
    if (dim1 == vision_hidden_size and dim0 > 0 and dim0 != vision_hidden_size) return std.math.cast(i32, dim0);
    return null;
}

test "inferVisionHeads prefers 64-dim heads" {
    try std.testing.expectEqual(@as(usize, 12), inferVisionHeads(768));
    try std.testing.expectEqual(@as(usize, 16), inferVisionHeads(1024));
}

test "inferSpatialMergeSize infers square merge units from merger input width" {
    var data: [8]f32 = [_]f32{0} ** 8;
    const weight = tensor.Tensor.view2DSlice(data[0..], 2, 4);
    try std.testing.expectEqual(@as(?i32, 2), inferSpatialMergeSize(weight, 1));
}
