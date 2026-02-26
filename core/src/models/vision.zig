//! Models-owned vision loading/probing helpers.
//!
//! This module keeps safetensors probing and config hydration out of inference
//! runtime code so backend vision execution stays model-agnostic.

const std = @import("std");
const tensor = @import("../tensor.zig");
const layer_ops = @import("layer_ops.zig");
const op_types = @import("op_types.zig");
const registry = @import("registry.zig");
const st_writer = @import("../io/safetensors/writer.zig");
const generic_weights = @import("load/generic_weights.zig");
const weights = @import("load/weights.zig");

pub const SafeTensors = @import("../io/safetensors/root.zig").UnifiedSafeTensors;
pub const LoadedModel = weights.LoadedModel;
pub const VisionMetadata = op_types.VisionMetadata;
pub const AttentionLayout = enum {
    fused_qkv,
    split_qkv,
    unknown,
};

pub fn resolveVisionMetadata(loaded: *const LoadedModel) VisionMetadata {
    if (loaded.runtime.architecture_id) |arch_id| {
        if (registry.runtimeArchitectureById(arch_id)) |arch| {
            return arch.vision;
        }
    }
    return .{};
}

pub fn resolveVisionProgram(loaded: *const LoadedModel) ?[]const layer_ops.LayerOp {
    const arch_id = loaded.runtime.architecture_id orelse return null;
    return registry.visionProgramByArchitectureId(arch_id);
}

pub fn hasAnyTensor(st: *SafeTensors, candidates: []const []const u8) bool {
    for (candidates) |name| {
        _ = st.getTensor(name, null) catch continue;
        return true;
    }
    return false;
}

pub fn getTensorByCandidates(st: *SafeTensors, candidates: []const []const u8) !tensor.Tensor {
    for (candidates) |name| {
        const t = st.getTensor(name, null) catch |err| switch (err) {
            error.NotFound => continue,
            else => return err,
        };
        return t;
    }
    return error.NotFound;
}

pub fn getLayerTensorByTemplates(st: *SafeTensors, layer_idx: usize, templates: []const []const u8) !tensor.Tensor {
    var name_buf: [192]u8 = undefined;
    for (templates) |template| {
        const name = generic_weights.expandLayerTemplate(name_buf[0..], template, layer_idx) catch continue;
        const t = st.getTensor(name, null) catch |err| switch (err) {
            error.NotFound => continue,
            else => return err,
        };
        return t;
    }
    return error.NotFound;
}

pub fn detectVisionAttentionLayout(loaded: *const LoadedModel) AttentionLayout {
    if (loaded.st == null) return .unknown;

    const st: *SafeTensors = @constCast(&loaded.st.?);
    const vision_metadata = resolveVisionMetadata(loaded);
    if (hasAnyTensor(st, vision_metadata.fused_qkv_probe_candidates)) {
        return .fused_qkv;
    }

    if (hasAnyTensor(st, vision_metadata.split_qkv_probe_candidates)) {
        return .split_qkv;
    }

    return .unknown;
}

pub fn hydrateVisionConfigFromWeights(loaded: *LoadedModel) !void {
    if (loaded.st == null) return;

    var cfg = &loaded.config;
    const vision_metadata = resolveVisionMetadata(loaded);
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
    if (!hasAnyTensor(st, vision_metadata.patch_embed_candidates)) return;

    const patch_tensor = getTensorByCandidates(st, vision_metadata.patch_embed_candidates) catch |err| switch (err) {
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

    const pos_tensor = getTensorByCandidates(st, vision_metadata.position_embed_candidates) catch |err| switch (err) {
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

    const merger_fc1_weight = getTensorByCandidates(st, vision_metadata.merger_fc1_candidates) catch |err| switch (err) {
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
        cfg.vision_depth = @intCast(try inferVisionDepth(st, &vision_metadata));
    }

    if (cfg.vision_num_heads <= 0 and cfg.vision_hidden_size > 0) {
        cfg.vision_num_heads = @intCast(inferVisionHeads(@intCast(cfg.vision_hidden_size)));
    }

    if (cfg.vision_intermediate_size <= 0 and cfg.vision_hidden_size > 0) {
        if (try inferVisionIntermediateSize(st, @intCast(cfg.vision_hidden_size), &vision_metadata)) |intermediate| {
            cfg.vision_intermediate_size = intermediate;
        }
    }

    if (cfg.vision_max_num_patches <= 0 and cfg.vision_num_position_embeddings > 0) {
        cfg.vision_max_num_patches = cfg.vision_num_position_embeddings;
    }
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

fn inferVisionDepth(st: *SafeTensors, vision_metadata: *const VisionMetadata) !usize {
    const split_depth = try countLayerDepth(st, vision_metadata.depth_split_qproj_templates);
    if (split_depth > 0) return split_depth;

    const fused_depth = try countLayerDepth(st, vision_metadata.depth_fused_qkv_templates);
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

fn inferVisionIntermediateSize(
    st: *SafeTensors,
    vision_hidden_size: usize,
    vision_metadata: *const VisionMetadata,
) !?i32 {
    const fc1 = getLayerTensorByTemplates(st, 0, vision_metadata.intermediate_fc1_templates) catch |err| switch (err) {
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

fn makeLoadedModelForTests() LoadedModel {
    return .{
        .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
        .config = .{
            .vocab_size = 16,
            .d_model = 8,
            .n_layers = 1,
            .n_heads = 1,
            .n_kv_groups = 1,
            .d_ff = 16,
            .max_seq_len = 32,
            .head_dim = 8,
            .rope_theta = 10000.0,
            .norm_eps = 1e-5,
            .gaffine_group_size = 128,
        },
        .token_embeddings = .{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ 1, 1, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
            .numel = 1,
            .strides = .{ 1, 1, 0, 0, 0, 0, 0, 0 },
        },
        .blocks = &.{},
        .original_weight_dtype = .f32,
    };
}

test "resolveVisionMetadata returns empty metadata when architecture is unset" {
    var loaded = makeLoadedModelForTests();
    defer loaded.arena.deinit();
    const meta = resolveVisionMetadata(&loaded);
    try std.testing.expect(std.meta.eql(VisionMetadata{}, meta));
}

test "resolveVisionProgram returns null when architecture is unset" {
    var loaded = makeLoadedModelForTests();
    defer loaded.arena.deinit();
    try std.testing.expect(resolveVisionProgram(&loaded) == null);
}

test "resolveVisionProgram returns multimodal program for youtu_vl" {
    var loaded = makeLoadedModelForTests();
    defer loaded.arena.deinit();
    loaded.runtime.architecture_id = "youtu_vl";
    const program = resolveVisionProgram(&loaded) orelse return error.TestUnexpectedResult;
    try std.testing.expect(program.len > 0);
}

test "resolveVisionProgram returns multimodal program for qwen3 and lfm2" {
    var loaded = makeLoadedModelForTests();
    defer loaded.arena.deinit();

    loaded.runtime.architecture_id = "qwen3";
    const qwen_program = resolveVisionProgram(&loaded) orelse return error.TestUnexpectedResult;
    try std.testing.expect(qwen_program.len > 0);

    loaded.runtime.architecture_id = "lfm2";
    const lfm_program = resolveVisionProgram(&loaded) orelse return error.TestUnexpectedResult;
    try std.testing.expect(lfm_program.len > 0);
}

test "hasAnyTensor and getTensorByCandidates find existing tensor" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var f32_values = [_]f32{ 1.0, 2.0 };
    const shape = [_]usize{2};
    const entries = [_]st_writer.TensorEntry{
        .{
            .name = "foo.weight",
            .dtype = .f32,
            .shape = &shape,
            .data = std.mem.sliceAsBytes(&f32_values),
        },
    };
    var out_file = try tmp.dir.createFile("vision_test.safetensors", .{});
    defer out_file.close();
    try st_writer.writeToFile(allocator, out_file, &entries);
    const path = try tmp.dir.realpathAlloc(allocator, "vision_test.safetensors");
    defer allocator.free(path);

    var st = try SafeTensors.load(allocator, path);
    defer st.deinit();
    try std.testing.expect(hasAnyTensor(&st, &.{ "missing", "foo.weight" }));
    const tensor_view = try getTensorByCandidates(&st, &.{ "missing", "foo.weight" });
    try std.testing.expectEqual(@as(usize, 2), tensor_view.numel);
}

test "getLayerTensorByTemplates resolves layer template name" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var f32_values = [_]f32{ 1.0, 2.0 };
    const shape = [_]usize{2};
    const entries = [_]st_writer.TensorEntry{
        .{
            .name = "layers.0.weight",
            .dtype = .f32,
            .shape = &shape,
            .data = std.mem.sliceAsBytes(&f32_values),
        },
    };
    var out_file = try tmp.dir.createFile("vision_layer_test.safetensors", .{});
    defer out_file.close();
    try st_writer.writeToFile(allocator, out_file, &entries);
    const path = try tmp.dir.realpathAlloc(allocator, "vision_layer_test.safetensors");
    defer allocator.free(path);

    var st = try SafeTensors.load(allocator, path);
    defer st.deinit();
    const t = try getLayerTensorByTemplates(&st, 0, &.{"layers.{d}.weight"});
    try std.testing.expectEqual(@as(usize, 2), t.numel);
}

test "detectVisionAttentionLayout returns unknown when tensors are absent" {
    var loaded = makeLoadedModelForTests();
    defer loaded.arena.deinit();
    try std.testing.expectEqual(AttentionLayout.unknown, detectVisionAttentionLayout(&loaded));
}

test "hydrateVisionConfigFromWeights is no-op without loaded tensors" {
    var loaded = makeLoadedModelForTests();
    defer loaded.arena.deinit();
    try hydrateVisionConfigFromWeights(&loaded);
    try std.testing.expectEqual(@as(i32, 0), loaded.config.vision_hidden_size);
}
