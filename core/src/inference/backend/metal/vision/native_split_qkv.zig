//! Vision runtime for local multimodal prefill.
//!
//! This module provides a ViT-style first-pass vision path:
//! - patch embedding from preprocessed image pixels
//! - absolute position embedding interpolation
//! - visual transformer blocks (bidirectional attention)
//! - spatial merger projection to language hidden size
//!
//! The output embeddings are scattered into `<|image_pad|>` token positions
//! before text prefill.

const std = @import("std");
const tensor = @import("../../../../tensor.zig");
const models = @import("../../../../models/root.zig");
const layer_ops = @import("../../../../models/layer_ops.zig");
const opcode_map = @import("../../../../models/plan/opcode_map.zig");
const compute = @import("../../../../compute/root.zig");
const image_mod = @import("../../../../image/root.zig");
const runtime_contract = @import("../../../runtime_contract/root.zig");
const common_vision = @import("../../../vision_types.zig");
const vision_tensor_convert = @import("../../../vision_tensor_convert.zig");
const vision_program_mod = @import("../../../vision_program.zig");
const vision_adapters = @import("../../../vision_program_adapters.zig");
const mlx_fused = @import("../mlx/ffi.zig");
const graph = compute.metal.graph;

const Tensor = tensor.Tensor;
const ModelConfig = tensor.ModelConfig;
const LoadedModel = models.LoadedModel;
const vision_load = models.vision;
const cpu_linalg = compute.cpu.linalg;
const cpu_common = compute.cpu.common;
const cpu_image_ops = compute.cpu.image_ops;
const cpu_norm = compute.cpu.normalization;
const cpu_rotary = compute.cpu.rotary;
const cpu_rowwise = compute.cpu.rowwise;
const cpu_memory = compute.cpu.memory;
const MatmulScratch = compute.cpu.linalg.MatmulScratch;
const ArrayHandle = graph.ArrayHandle;

pub const PrefillVisionImage = common_vision.PrefillVisionImage;
pub const PrefillVisionInput = common_vision.PrefillVisionInput;
pub const EncodedVisionOutput = common_vision.EncodedVisionOutput;

pub const VisionRuntime = struct {
    const TokenOrder = enum {
        merge_block,
        row_major,
    };

    allocator: std.mem.Allocator,

    vision_hidden_size: usize,
    vision_depth: usize,
    vision_num_heads: usize,
    vision_intermediate_size: usize,
    merger_intermediate_size: usize,
    language_hidden_size: usize,
    patch_size: usize,
    spatial_merge_size: usize,
    temporal_patch_size: usize,
    use_vision_rope: bool,
    token_order: TokenOrder,
    num_pos_embeddings: usize,
    num_grid_side: usize,

    patch_proj_weight: Tensor, // [vision_hidden, patch_dim]
    patch_proj_bias: []f32, // [vision_hidden]

    pos_embed_f32: []f32, // [num_pos_embeddings * vision_hidden]
    vision_post_norm_weight: []f32 = &.{},
    vision_post_norm_bias: []f32 = &.{},

    merger_norm_weight: []f32, // [vision_hidden]
    merger_norm_bias: []f32, // [vision_hidden]
    merger_fc1_weight: Tensor, // [vision_intermediate, vision_hidden * merge^2]
    merger_fc1_bias: []f32, // [vision_intermediate]
    merger_fc2_weight: Tensor, // [language_hidden, vision_intermediate]
    merger_fc2_bias: []f32, // [language_hidden]
    deepstack_mergers: []DeepstackMergerWeights = &.{},
    deepstack_visual_layers: [8]usize = [_]usize{0} ** 8,
    deepstack_layer_count: usize = 0,

    layer_weights: []LayerWeights,
    matmul_scratch: MatmulScratch,
    vision_program: []const layer_ops.LayerOp,
    vision_stage_plans: vision_program_mod.VisionStagePlans,
    dispatch_counters: runtime_contract.DispatchCounters = .{},

    const LayerWeights = struct {
        ln1_weight: Tensor,
        ln1_weight_handle: ArrayHandle,
        ln1_bias: []f32,
        ln1_bias_handle: ArrayHandle,
        ln2_weight: Tensor,
        ln2_weight_handle: ArrayHandle,
        ln2_bias: []f32,
        ln2_bias_handle: ArrayHandle,
        qkv_weight: Tensor,
        qkv_weight_handle: ArrayHandle,
        has_fused_qkv: bool,
        q_proj_weight: Tensor,
        q_proj_weight_handle: ArrayHandle,
        k_proj_weight: Tensor,
        k_proj_weight_handle: ArrayHandle,
        v_proj_weight: Tensor,
        v_proj_weight_handle: ArrayHandle,
        has_split_qkv: bool,
        o_weight: Tensor,
        o_weight_handle: ArrayHandle,
        fc1_weight: Tensor,
        fc1_weight_handle: ArrayHandle,
        fc2_weight: Tensor,
        fc2_weight_handle: ArrayHandle,

        qkv_bias_all: []f32,
        qkv_bias_handle: ArrayHandle,
        q_bias_handle: ArrayHandle,
        k_bias_handle: ArrayHandle,
        v_bias_handle: ArrayHandle,
        o_bias: []f32,
        o_bias_handle: ArrayHandle,
        fc1_bias: []f32,
        fc1_bias_handle: ArrayHandle,
        fc2_bias: []f32,
        fc2_bias_handle: ArrayHandle,

        fn deinit(self: *LayerWeights, allocator: std.mem.Allocator) void {
            graph.freeArray(self.ln1_weight_handle);
            graph.freeArray(self.ln1_bias_handle);
            graph.freeArray(self.ln2_weight_handle);
            graph.freeArray(self.ln2_bias_handle);
            if (self.ln1_bias.len > 0) allocator.free(self.ln1_bias);
            if (self.ln2_bias.len > 0) allocator.free(self.ln2_bias);
            if (self.has_fused_qkv) {
                graph.freeArray(self.qkv_weight_handle);
                graph.freeArray(self.qkv_bias_handle);
            } else {
                graph.freeArray(self.q_proj_weight_handle);
                graph.freeArray(self.k_proj_weight_handle);
                graph.freeArray(self.v_proj_weight_handle);
                graph.freeArray(self.q_bias_handle);
                graph.freeArray(self.k_bias_handle);
                graph.freeArray(self.v_bias_handle);
            }
            graph.freeArray(self.o_weight_handle);
            graph.freeArray(self.o_bias_handle);
            graph.freeArray(self.fc1_weight_handle);
            graph.freeArray(self.fc1_bias_handle);
            graph.freeArray(self.fc2_weight_handle);
            graph.freeArray(self.fc2_bias_handle);
            if (self.qkv_bias_all.len > 0) allocator.free(self.qkv_bias_all);
            if (self.o_bias.len > 0) allocator.free(self.o_bias);
            if (self.fc1_bias.len > 0) allocator.free(self.fc1_bias);
            if (self.fc2_bias.len > 0) allocator.free(self.fc2_bias);
            self.* = undefined;
        }
    };

    pub const DeepstackMergerWeights = struct {
        norm_weight: []f32,
        norm_bias: []f32,
        fc1_weight: Tensor,
        fc1_bias: []f32,
        fc2_weight: Tensor,
        fc2_bias: []f32,

        fn deinit(self: *DeepstackMergerWeights, allocator: std.mem.Allocator) void {
            if (self.norm_weight.len > 0) allocator.free(self.norm_weight);
            if (self.norm_bias.len > 0) allocator.free(self.norm_bias);
            if (self.fc1_bias.len > 0) allocator.free(self.fc1_bias);
            if (self.fc2_bias.len > 0) allocator.free(self.fc2_bias);
            self.* = undefined;
        }
    };

    pub fn init(allocator: std.mem.Allocator, loaded: *LoadedModel) !?VisionRuntime {
        const cfg = loaded.config;
        if (cfg.vision_hidden_size <= 0 or cfg.vision_depth <= 0) return null;
        if (loaded.st == null) return null;

        const vision_hidden_size: usize = @intCast(cfg.vision_hidden_size);
        const vision_depth: usize = @intCast(cfg.vision_depth);
        const vision_num_heads: usize = @intCast(if (cfg.vision_num_heads > 0) cfg.vision_num_heads else 16);
        const vision_intermediate_size: usize = @intCast(if (cfg.vision_intermediate_size > 0) cfg.vision_intermediate_size else 4 * cfg.vision_hidden_size);
        const merger_intermediate_size: usize = @intCast(if (cfg.projector_hidden_size > 0) cfg.projector_hidden_size else if (cfg.vision_intermediate_size > 0) cfg.vision_intermediate_size else 4 * cfg.vision_hidden_size);
        const language_hidden_size: usize = @intCast(cfg.d_model);
        const patch_size: usize = @intCast(if (cfg.vision_patch_size > 0) cfg.vision_patch_size else 16);
        var spatial_merge_size: usize = @intCast(if (cfg.vision_spatial_merge_size > 0) cfg.vision_spatial_merge_size else 1);
        const temporal_patch_size: usize = @intCast(if (cfg.vision_temporal_patch_size > 0) cfg.vision_temporal_patch_size else 1);
        const num_pos_embeddings: usize = @intCast(if (cfg.vision_num_position_embeddings > 0) cfg.vision_num_position_embeddings else 2304);
        var deepstack_visual_layers: [8]usize = [_]usize{0} ** 8;
        var deepstack_layer_count: usize = @intCast(cfg.vision_probe_layer_count);
        deepstack_layer_count = @min(deepstack_layer_count, deepstack_visual_layers.len);
        for (0..deepstack_layer_count) |idx| {
            deepstack_visual_layers[idx] = cfg.vision_probe_layers[idx];
        }
        const vision_program = vision_load.resolveVisionProgram(loaded) orelse return null;
        const parsed_program = try vision_program_mod.parseVisionProgram(
            vision_program,
            spatial_merge_size,
            deepstack_visual_layers,
            deepstack_layer_count,
        );
        var vision_stage_plans = try vision_program_mod.compileVisionStagePlans(allocator, vision_program);
        errdefer vision_program_mod.deinitVisionStagePlans(allocator, &vision_stage_plans);
        spatial_merge_size = parsed_program.spatial_merge_size;
        deepstack_visual_layers = parsed_program.deepstack_visual_layers;
        deepstack_layer_count = parsed_program.deepstack_layer_count;

        const num_grid_side = std.math.sqrt(num_pos_embeddings);
        if (num_grid_side * num_grid_side != num_pos_embeddings) return error.InvalidShape;

        const st = &loaded.st.?;
        const vision_metadata = vision_load.resolveVisionMetadata(loaded);

        const patch_proj_w_5d = try vision_load.getTensorByCandidates(st, vision_metadata.patch_embed_candidates);
        const patch_proj_weight = flattenPatchProjWeight(patch_proj_w_5d, vision_hidden_size, temporal_patch_size, patch_size);
        const patch_proj_bias_tensor = try vision_load.getTensorByCandidates(st, vision_metadata.patch_embed_bias_candidates);
        const patch_proj_bias = try vision_tensor_convert.tensorToOwnedF32(allocator, patch_proj_bias_tensor);
        errdefer allocator.free(patch_proj_bias);

        const pos_embed_tensor = try vision_load.getTensorByCandidates(st, vision_metadata.position_embed_candidates);
        const pos_embed_f32 = try vision_tensor_convert.tensorToOwnedF32(allocator, pos_embed_tensor);
        errdefer allocator.free(pos_embed_f32);

        var vision_post_norm_weight: []f32 = &.{};
        var vision_post_norm_bias: []f32 = &.{};
        const post_norm_weight_tensor = vision_load.getTensorByCandidates(st, vision_metadata.post_norm_weight_candidates) catch |err| switch (err) {
            error.NotFound => null,
            else => return err,
        };
        if (post_norm_weight_tensor) |tensor_weight| {
            vision_post_norm_weight = try vision_tensor_convert.tensorToOwnedF32(allocator, tensor_weight);
            vision_post_norm_bias = try vision_tensor_convert.tensorToOwnedF32(allocator, try vision_load.getTensorByCandidates(st, vision_metadata.post_norm_bias_candidates));
        }
        errdefer if (vision_post_norm_weight.len > 0) allocator.free(vision_post_norm_weight);
        errdefer if (vision_post_norm_bias.len > 0) allocator.free(vision_post_norm_bias);

        var merger_norm_weight: []f32 = &.{};
        var merger_norm_bias: []f32 = &.{};
        const merger_norm_tensor = vision_load.getTensorByCandidates(st, vision_metadata.merger_norm_weight_candidates) catch |err| switch (err) {
            error.NotFound => null,
            else => return err,
        };
        if (merger_norm_tensor) |tensor_weight| {
            merger_norm_weight = try vision_tensor_convert.tensorToOwnedF32(allocator, tensor_weight);
            merger_norm_bias = try vision_tensor_convert.tensorToOwnedF32(allocator, try vision_load.getTensorByCandidates(st, vision_metadata.merger_norm_bias_candidates));
        }
        errdefer if (merger_norm_weight.len > 0) allocator.free(merger_norm_weight);
        errdefer if (merger_norm_bias.len > 0) allocator.free(merger_norm_bias);

        const merger_fc1_weight = try vision_load.getTensorByCandidates(st, vision_metadata.merger_fc1_candidates);
        const merger_fc1_bias = try vision_tensor_convert.tensorToOwnedF32(allocator, try vision_load.getTensorByCandidates(st, vision_metadata.merger_fc1_bias_candidates));
        errdefer allocator.free(merger_fc1_bias);

        const merger_fc2_weight = try vision_load.getTensorByCandidates(st, vision_metadata.merger_fc2_candidates);
        const merger_fc2_bias = try vision_tensor_convert.tensorToOwnedF32(allocator, try vision_load.getTensorByCandidates(st, vision_metadata.merger_fc2_bias_candidates));
        errdefer allocator.free(merger_fc2_bias);
        const deepstack_mergers = try loadDeepstackMergers(
            allocator,
            st,
            &vision_metadata,
            vision_hidden_size,
            language_hidden_size,
            spatial_merge_size,
            deepstack_layer_count,
        );
        errdefer {
            for (deepstack_mergers) |*merger| merger.deinit(allocator);
            if (deepstack_mergers.len > 0) allocator.free(deepstack_mergers);
        }
        deepstack_layer_count = @min(deepstack_layer_count, deepstack_mergers.len);

        var layer_weights = try allocator.alloc(LayerWeights, vision_depth);
        errdefer allocator.free(layer_weights);

        var built_layers: usize = 0;
        var saw_fused_qkv = false;
        var saw_split_qkv = false;
        errdefer {
            for (0..built_layers) |idx| {
                layer_weights[idx].deinit(allocator);
            }
        }

        for (0..vision_depth) |layer_idx| {
            const ln1_weight = try vision_load.getLayerTensorByTemplates(st, layer_idx, vision_metadata.ln1_weight_templates);
            const ln1_bias = try vision_load.getLayerTensorByTemplates(st, layer_idx, vision_metadata.ln1_bias_templates);

            const ln2_weight = try vision_load.getLayerTensorByTemplates(st, layer_idx, vision_metadata.ln2_weight_templates);
            const ln2_bias = try vision_load.getLayerTensorByTemplates(st, layer_idx, vision_metadata.ln2_bias_templates);

            const fused_qkv_weight = vision_load.getLayerTensorByTemplates(st, layer_idx, vision_metadata.fused_qkv_weight_templates) catch |err| switch (err) {
                error.NotFound => null,
                else => return err,
            };
            if (fused_qkv_weight != null) {
                saw_fused_qkv = true;
            } else {
                saw_split_qkv = true;
            }
            const fused_qkv_bias_all = if (fused_qkv_weight != null)
                try vision_tensor_convert.tensorToOwnedF32(allocator, try vision_load.getLayerTensorByTemplates(st, layer_idx, vision_metadata.fused_qkv_bias_templates))
            else
                null;
            const qkv_bias_all = if (fused_qkv_bias_all) |bias| bias else blk: {
                const q_bias = try vision_tensor_convert.tensorToOwnedF32(allocator, try vision_load.getLayerTensorByTemplates(st, layer_idx, vision_metadata.split_q_bias_templates));
                errdefer allocator.free(q_bias);
                const k_bias = try vision_tensor_convert.tensorToOwnedF32(allocator, try vision_load.getLayerTensorByTemplates(st, layer_idx, vision_metadata.split_k_bias_templates));
                errdefer allocator.free(k_bias);
                const v_bias = try vision_tensor_convert.tensorToOwnedF32(allocator, try vision_load.getLayerTensorByTemplates(st, layer_idx, vision_metadata.split_v_bias_templates));
                errdefer allocator.free(v_bias);

                const merged = try allocator.alloc(f32, q_bias.len + k_bias.len + v_bias.len);
                errdefer allocator.free(merged);
                @memcpy(merged[0..q_bias.len], q_bias);
                @memcpy(merged[q_bias.len .. q_bias.len + k_bias.len], k_bias);
                @memcpy(merged[q_bias.len + k_bias.len ..], v_bias);
                allocator.free(q_bias);
                allocator.free(k_bias);
                allocator.free(v_bias);
                break :blk merged;
            };
            errdefer allocator.free(qkv_bias_all);

            const q_proj_weight = if (fused_qkv_weight == null)
                try vision_load.getLayerTensorByTemplates(st, layer_idx, vision_metadata.split_q_weight_templates)
            else
                undefined;
            const k_proj_weight = if (fused_qkv_weight == null)
                try vision_load.getLayerTensorByTemplates(st, layer_idx, vision_metadata.split_k_weight_templates)
            else
                undefined;
            const v_proj_weight = if (fused_qkv_weight == null)
                try vision_load.getLayerTensorByTemplates(st, layer_idx, vision_metadata.split_v_weight_templates)
            else
                undefined;

            const o_weight = try vision_load.getLayerTensorByTemplates(st, layer_idx, vision_metadata.out_proj_weight_templates);
            const o_bias = try vision_tensor_convert.tensorToOwnedF32(allocator, try vision_load.getLayerTensorByTemplates(st, layer_idx, vision_metadata.out_proj_bias_templates));
            errdefer allocator.free(o_bias);

            const fc1_weight = try vision_load.getLayerTensorByTemplates(st, layer_idx, vision_metadata.fc1_weight_templates);
            const fc1_bias = try vision_tensor_convert.tensorToOwnedF32(allocator, try vision_load.getLayerTensorByTemplates(st, layer_idx, vision_metadata.fc1_bias_templates));
            errdefer allocator.free(fc1_bias);
            const fc2_weight = try vision_load.getLayerTensorByTemplates(st, layer_idx, vision_metadata.fc2_weight_templates);
            const fc2_bias = try vision_tensor_convert.tensorToOwnedF32(allocator, try vision_load.getLayerTensorByTemplates(st, layer_idx, vision_metadata.fc2_bias_templates));
            errdefer allocator.free(fc2_bias);

            const q_bias = qkv_bias_all[0..vision_hidden_size];
            const k_bias = qkv_bias_all[vision_hidden_size .. 2 * vision_hidden_size];
            const v_bias = qkv_bias_all[2 * vision_hidden_size .. 3 * vision_hidden_size];

            const ln1_weight_handle = try tensorToMetalArray(&ln1_weight);
            errdefer graph.freeArray(ln1_weight_handle);
            const ln1_bias_f32 = try vision_tensor_convert.tensorToOwnedF32(allocator, ln1_bias);
            errdefer allocator.free(ln1_bias_f32);
            const ln1_bias_handle = f32SliceToMetalArray(ln1_bias_f32);
            errdefer graph.freeArray(ln1_bias_handle);
            const ln2_weight_handle = try tensorToMetalArray(&ln2_weight);
            errdefer graph.freeArray(ln2_weight_handle);
            const ln2_bias_f32 = try vision_tensor_convert.tensorToOwnedF32(allocator, ln2_bias);
            errdefer allocator.free(ln2_bias_f32);
            const ln2_bias_handle = f32SliceToMetalArray(ln2_bias_f32);
            errdefer graph.freeArray(ln2_bias_handle);
            const qkv_weight_handle = if (fused_qkv_weight != null) try tensorToMetalArray(&(fused_qkv_weight.?)) else null;
            errdefer if (qkv_weight_handle != null) graph.freeArray(qkv_weight_handle);
            const qkv_bias_handle = f32SliceToMetalArray(qkv_bias_all);
            errdefer graph.freeArray(qkv_bias_handle);
            const q_proj_weight_handle = if (fused_qkv_weight == null) try tensorToMetalArray(&q_proj_weight) else null;
            errdefer if (q_proj_weight_handle != null) graph.freeArray(q_proj_weight_handle);
            const k_proj_weight_handle = if (fused_qkv_weight == null) try tensorToMetalArray(&k_proj_weight) else null;
            errdefer if (k_proj_weight_handle != null) graph.freeArray(k_proj_weight_handle);
            const v_proj_weight_handle = if (fused_qkv_weight == null) try tensorToMetalArray(&v_proj_weight) else null;
            errdefer if (v_proj_weight_handle != null) graph.freeArray(v_proj_weight_handle);
            const q_bias_handle = f32SliceToMetalArray(q_bias);
            errdefer graph.freeArray(q_bias_handle);
            const k_bias_handle = f32SliceToMetalArray(k_bias);
            errdefer graph.freeArray(k_bias_handle);
            const v_bias_handle = f32SliceToMetalArray(v_bias);
            errdefer graph.freeArray(v_bias_handle);
            const o_weight_handle = try tensorToMetalArray(&o_weight);
            errdefer graph.freeArray(o_weight_handle);
            const o_bias_handle = f32SliceToMetalArray(o_bias);
            errdefer graph.freeArray(o_bias_handle);
            const fc1_weight_handle = try tensorToMetalArray(&fc1_weight);
            errdefer graph.freeArray(fc1_weight_handle);
            const fc1_bias_handle = f32SliceToMetalArray(fc1_bias);
            errdefer graph.freeArray(fc1_bias_handle);
            const fc2_weight_handle = try tensorToMetalArray(&fc2_weight);
            errdefer graph.freeArray(fc2_weight_handle);
            const fc2_bias_handle = f32SliceToMetalArray(fc2_bias);
            errdefer graph.freeArray(fc2_bias_handle);

            layer_weights[layer_idx] = .{
                .ln1_weight = ln1_weight,
                .ln1_weight_handle = ln1_weight_handle,
                .ln1_bias = ln1_bias_f32,
                .ln1_bias_handle = ln1_bias_handle,
                .ln2_weight = ln2_weight,
                .ln2_weight_handle = ln2_weight_handle,
                .ln2_bias = ln2_bias_f32,
                .ln2_bias_handle = ln2_bias_handle,
                .qkv_weight = fused_qkv_weight orelse undefined,
                .qkv_weight_handle = qkv_weight_handle,
                .has_fused_qkv = fused_qkv_weight != null,
                .q_proj_weight = if (fused_qkv_weight == null) q_proj_weight else undefined,
                .q_proj_weight_handle = q_proj_weight_handle,
                .k_proj_weight = if (fused_qkv_weight == null) k_proj_weight else undefined,
                .k_proj_weight_handle = k_proj_weight_handle,
                .v_proj_weight = if (fused_qkv_weight == null) v_proj_weight else undefined,
                .v_proj_weight_handle = v_proj_weight_handle,
                .has_split_qkv = fused_qkv_weight == null,
                .o_weight = o_weight,
                .o_weight_handle = o_weight_handle,
                .fc1_weight = fc1_weight,
                .fc1_weight_handle = fc1_weight_handle,
                .fc2_weight = fc2_weight,
                .fc2_weight_handle = fc2_weight_handle,
                .qkv_bias_all = qkv_bias_all,
                .qkv_bias_handle = qkv_bias_handle,
                .q_bias_handle = q_bias_handle,
                .k_bias_handle = k_bias_handle,
                .v_bias_handle = v_bias_handle,
                .o_bias = o_bias,
                .o_bias_handle = o_bias_handle,
                .fc1_bias = fc1_bias,
                .fc1_bias_handle = fc1_bias_handle,
                .fc2_bias = fc2_bias,
                .fc2_bias_handle = fc2_bias_handle,
            };

            built_layers += 1;
        }

        if (saw_fused_qkv and saw_split_qkv) return error.InvalidShape;
        const use_vision_rope = saw_fused_qkv;
        const token_order: TokenOrder = if (use_vision_rope) .merge_block else .row_major;

        var matmul_scratch = try MatmulScratch.init(allocator);
        errdefer matmul_scratch.deinit();

        return VisionRuntime{
            .allocator = allocator,
            .vision_hidden_size = vision_hidden_size,
            .vision_depth = vision_depth,
            .vision_num_heads = vision_num_heads,
            .vision_intermediate_size = vision_intermediate_size,
            .merger_intermediate_size = merger_intermediate_size,
            .language_hidden_size = language_hidden_size,
            .patch_size = patch_size,
            .spatial_merge_size = spatial_merge_size,
            .temporal_patch_size = temporal_patch_size,
            .use_vision_rope = use_vision_rope,
            .token_order = token_order,
            .num_pos_embeddings = num_pos_embeddings,
            .num_grid_side = num_grid_side,
            .patch_proj_weight = patch_proj_weight,
            .patch_proj_bias = patch_proj_bias,
            .pos_embed_f32 = pos_embed_f32,
            .vision_post_norm_weight = vision_post_norm_weight,
            .vision_post_norm_bias = vision_post_norm_bias,
            .merger_norm_weight = merger_norm_weight,
            .merger_norm_bias = merger_norm_bias,
            .merger_fc1_weight = merger_fc1_weight,
            .merger_fc1_bias = merger_fc1_bias,
            .merger_fc2_weight = merger_fc2_weight,
            .merger_fc2_bias = merger_fc2_bias,
            .deepstack_mergers = deepstack_mergers,
            .deepstack_visual_layers = deepstack_visual_layers,
            .deepstack_layer_count = deepstack_layer_count,
            .layer_weights = layer_weights,
            .matmul_scratch = matmul_scratch,
            .vision_program = vision_program,
            .vision_stage_plans = vision_stage_plans,
            .dispatch_counters = .{},
        };
    }

    pub fn deinit(self: *VisionRuntime) void {
        for (self.layer_weights) |*w| w.deinit(self.allocator);

        if (self.patch_proj_bias.len > 0) self.allocator.free(self.patch_proj_bias);
        if (self.pos_embed_f32.len > 0) self.allocator.free(self.pos_embed_f32);
        if (self.vision_post_norm_weight.len > 0) self.allocator.free(self.vision_post_norm_weight);
        if (self.vision_post_norm_bias.len > 0) self.allocator.free(self.vision_post_norm_bias);
        if (self.merger_norm_weight.len > 0) self.allocator.free(self.merger_norm_weight);
        if (self.merger_norm_bias.len > 0) self.allocator.free(self.merger_norm_bias);
        if (self.merger_fc1_bias.len > 0) self.allocator.free(self.merger_fc1_bias);
        if (self.merger_fc2_bias.len > 0) self.allocator.free(self.merger_fc2_bias);
        if (self.deepstack_mergers.len > 0) {
            for (self.deepstack_mergers) |*merger| merger.deinit(self.allocator);
            self.allocator.free(self.deepstack_mergers);
        }

        self.matmul_scratch.deinit();
        vision_program_mod.deinitVisionStagePlans(self.allocator, &self.vision_stage_plans);
        if (self.layer_weights.len > 0) self.allocator.free(self.layer_weights);

        self.* = undefined;
    }

    /// Encode images and return merged embeddings plus optional DeepStack features.
    pub fn encodeImages(
        self: *VisionRuntime,
        images: []const PrefillVisionImage,
    ) !EncodedVisionOutput {
        var total_tokens: usize = 0;
        for (images) |img| total_tokens += img.token_count;
        if (total_tokens == 0) {
            return .{
                .merged_embeddings = &.{},
                .deepstack_layer_embeddings = &.{},
            };
        }

        const total_values = total_tokens * self.language_hidden_size;
        const merged_embeddings = try self.allocator.alloc(f32, total_values);
        errdefer self.allocator.free(merged_embeddings);

        const deepstack_count = self.deepstack_mergers.len;
        var deepstack_layer_embeddings: [][]f32 = if (deepstack_count > 0)
            try self.allocator.alloc([]f32, deepstack_count)
        else
            try self.allocator.alloc([]f32, 0);
        errdefer self.allocator.free(deepstack_layer_embeddings);
        for (0..deepstack_layer_embeddings.len) |layer_idx| deepstack_layer_embeddings[layer_idx] = &.{};

        if (deepstack_count > 0) {
            errdefer {
                for (deepstack_layer_embeddings) |layer_embed| {
                    if (layer_embed.len > 0) self.allocator.free(layer_embed);
                }
            }
            for (0..deepstack_count) |layer_idx| {
                deepstack_layer_embeddings[layer_idx] = try self.allocator.alloc(f32, total_values);
            }
        }

        var offset_tokens: usize = 0;
        for (images) |img| {
            var single = try self.encodeSingleImage(img);
            defer single.deinit(self.allocator);

            const token_count = single.merged_embeddings.len / self.language_hidden_size;
            @memcpy(
                merged_embeddings[offset_tokens * self.language_hidden_size ..][0 .. token_count * self.language_hidden_size],
                single.merged_embeddings,
            );
            if (deepstack_count > 0) {
                if (single.deepstack_layer_embeddings.len != deepstack_count) return error.InvalidShape;
                for (0..deepstack_count) |layer_idx| {
                    @memcpy(
                        deepstack_layer_embeddings[layer_idx][offset_tokens * self.language_hidden_size ..][0 .. token_count * self.language_hidden_size],
                        single.deepstack_layer_embeddings[layer_idx],
                    );
                }
            }
            offset_tokens += token_count;
        }

        return .{
            .merged_embeddings = merged_embeddings,
            .deepstack_layer_embeddings = deepstack_layer_embeddings,
        };
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
        try vision_adapters.runScatterProgram(
            &self.vision_stage_plans.scatter,
            &self.dispatch_counters,
            vision_adapters.adapter_table,
            hidden_states,
            seq_len,
            d_model,
            token_ids,
            image_token_id,
            embeddings,
        );
    }

    const EncodedSingleImage = vision_adapters.EncodedSingleImage;

    fn encodeSingleImage(self: *VisionRuntime, image: PrefillVisionImage) !EncodedSingleImage {
        const patch_count: usize = @as(usize, @intCast(image.grid.temporal)) *
            @as(usize, @intCast(image.grid.height)) *
            @as(usize, @intCast(image.grid.width));
        if (patch_count == 0) return error.InvalidImageDimensions;

        const patch_dim = 3 * self.temporal_patch_size * self.patch_size * self.patch_size;
        const patch_input = try self.allocator.alloc(f32, patch_count * patch_dim);
        defer self.allocator.free(patch_input);

        try extractPatchInput(self, image, patch_input);

        const patch_hidden = try self.allocator.alloc(f32, patch_count * self.vision_hidden_size);
        defer self.allocator.free(patch_hidden);

        // patch projection: [patch_count, patch_dim] @ [vision_hidden, patch_dim]^T -> [patch_count, vision_hidden]
        var patch_input_view = Tensor.view2DSlice(patch_input, patch_count, patch_dim);
        var patch_hidden_view = Tensor.view2DSlice(patch_hidden, patch_count, self.vision_hidden_size);
        try cpu_linalg.matmulAuto(&patch_input_view, &self.patch_proj_weight, &patch_hidden_view, &self.matmul_scratch);
        cpu_common.addBiasRows(patch_hidden, self.patch_proj_bias, patch_count, self.vision_hidden_size);

        const pos_embeds = try self.allocator.alloc(f32, patch_count * self.vision_hidden_size);
        defer self.allocator.free(pos_embeds);
        try interpolatePosEmbeddings(self, image.grid, pos_embeds);

        cpu_rowwise.addScaledInPlace(patch_hidden, pos_embeds, 1.0);

        const hidden_shape = [_]i64{ 1, @intCast(patch_count), @intCast(self.vision_hidden_size) };
        var hidden_handle = graph.createArrayF32(patch_hidden, &hidden_shape);
        defer graph.freeArray(hidden_handle);

        var runtime_rope_cos_handle: ArrayHandle = null;
        var runtime_rope_sin_handle: ArrayHandle = null;
        var runtime_rope_dim: usize = 0;
        var vision_rope_cos: []f32 = &.{};
        var vision_rope_sin: []f32 = &.{};
        defer if (vision_rope_cos.len > 0) self.allocator.free(vision_rope_cos);
        defer if (vision_rope_sin.len > 0) self.allocator.free(vision_rope_sin);
        defer if (runtime_rope_cos_handle != null) graph.freeArray(runtime_rope_cos_handle);
        defer if (runtime_rope_sin_handle != null) graph.freeArray(runtime_rope_sin_handle);
        if (self.use_vision_rope) {
            runtime_rope_dim = self.vision_hidden_size / self.vision_num_heads;
            vision_rope_cos = try self.allocator.alloc(f32, patch_count * runtime_rope_dim);
            vision_rope_sin = try self.allocator.alloc(f32, patch_count * runtime_rope_dim);
            try self.fillVisionRotaryTables(image.grid, vision_rope_cos, vision_rope_sin);
            const rope_shape = [_]i64{ @intCast(patch_count), @intCast(runtime_rope_dim) };
            runtime_rope_cos_handle = graph.createArrayF32(vision_rope_cos, &rope_shape);
            runtime_rope_sin_handle = graph.createArrayF32(vision_rope_sin, &rope_shape);
        }

        const deepstack_count = self.deepstack_mergers.len;
        var deepstack_layer_embeddings: [][]const f32 = if (deepstack_count > 0)
            try self.allocator.alloc([]const f32, deepstack_count)
        else
            try self.allocator.alloc([]const f32, 0);
        var deepstack_layer_handles: []ArrayHandle = if (deepstack_count > 0)
            try self.allocator.alloc(ArrayHandle, deepstack_count)
        else
            try self.allocator.alloc(ArrayHandle, 0);
        errdefer self.allocator.free(deepstack_layer_embeddings);
        errdefer self.allocator.free(deepstack_layer_handles);
        defer self.allocator.free(deepstack_layer_handles);
        errdefer {
            for (deepstack_layer_embeddings) |layer_embed| {
                if (layer_embed.len > 0) self.allocator.free(layer_embed);
            }
        }
        for (0..deepstack_layer_embeddings.len) |i| deepstack_layer_embeddings[i] = &.{};
        for (0..deepstack_layer_handles.len) |i| deepstack_layer_handles[i] = null;

        const attention_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(self.vision_hidden_size / self.vision_num_heads)));
        for (self.layer_weights, 0..) |*lw, layer_idx| {
            hidden_handle = if (lw.has_fused_qkv)
                mlx_fused.mlx_lazy_vision_block_fused_qkv_bf16(
                    hidden_handle,
                    lw.ln1_weight_handle,
                    lw.ln1_bias_handle,
                    lw.ln2_weight_handle,
                    lw.ln2_bias_handle,
                    lw.qkv_weight_handle.?,
                    lw.qkv_bias_handle,
                    lw.o_weight_handle,
                    lw.o_bias_handle,
                    lw.fc1_weight_handle,
                    lw.fc1_bias_handle,
                    lw.fc2_weight_handle,
                    lw.fc2_bias_handle,
                    runtime_rope_cos_handle,
                    runtime_rope_sin_handle,
                    runtime_rope_dim,
                    self.vision_num_heads,
                    self.vision_hidden_size / self.vision_num_heads,
                    attention_scale,
                    1e-6,
                )
            else
                mlx_fused.mlx_lazy_vision_block_split_qkv_bf16(
                    hidden_handle,
                    lw.ln1_weight_handle,
                    lw.ln1_bias_handle,
                    lw.ln2_weight_handle,
                    lw.ln2_bias_handle,
                    lw.q_proj_weight_handle.?,
                    lw.q_bias_handle,
                    lw.k_proj_weight_handle.?,
                    lw.k_bias_handle,
                    lw.v_proj_weight_handle.?,
                    lw.v_bias_handle,
                    lw.o_weight_handle,
                    lw.o_bias_handle,
                    lw.fc1_weight_handle,
                    lw.fc1_bias_handle,
                    lw.fc2_weight_handle,
                    lw.fc2_bias_handle,
                    runtime_rope_cos_handle,
                    runtime_rope_sin_handle,
                    runtime_rope_dim,
                    self.vision_num_heads,
                    self.vision_hidden_size / self.vision_num_heads,
                    attention_scale,
                    1e-6,
                );

            if (self.deepstackLayerToMergerIndex(layer_idx)) |merger_idx| {
                deepstack_layer_handles[merger_idx] = hidden_handle;
            }
        }

        if (deepstack_count > 0) {
            for (deepstack_layer_handles) |layer_handle| {
                if (layer_handle == null) return error.InvalidState;
            }
        }

        const eval_handle_count = 1 + deepstack_count;
        var eval_handles = try self.allocator.alloc(ArrayHandle, eval_handle_count);
        defer self.allocator.free(eval_handles);
        eval_handles[0] = hidden_handle;
        for (deepstack_layer_handles, 0..) |layer_handle, idx| {
            eval_handles[idx + 1] = layer_handle;
        }
        graph.eval(eval_handles);

        if (deepstack_count > 0) {
            for (deepstack_layer_handles, 0..) |layer_handle, merger_idx| {
                const hidden_host = try self.allocator.alloc(f32, patch_count * self.vision_hidden_size);
                defer self.allocator.free(hidden_host);
                graph.copyToHost(layer_handle, hidden_host);
                deepstack_layer_embeddings[merger_idx] = try self.runDeepstackMerger(
                    image.grid,
                    hidden_host,
                    &self.deepstack_mergers[merger_idx],
                );
            }
        }

        const final_hidden = try self.allocator.alloc(f32, patch_count * self.vision_hidden_size);
        defer self.allocator.free(final_hidden);
        graph.copyToHost(hidden_handle, final_hidden);

        const merger_input = if (self.vision_post_norm_weight.len > 0) blk: {
            if (self.vision_post_norm_weight.len != self.vision_hidden_size or self.vision_post_norm_bias.len != self.vision_hidden_size) {
                return error.InvalidShape;
            }

            const normalized = try self.allocator.alloc(f32, final_hidden.len);
            errdefer self.allocator.free(normalized);
            try cpu_norm.layerNormRows(
                final_hidden,
                normalized,
                patch_count,
                self.vision_hidden_size,
                self.vision_post_norm_weight,
                self.vision_post_norm_bias,
                1e-6,
            );
            break :blk normalized;
        } else final_hidden;
        defer if (merger_input.ptr != final_hidden.ptr) self.allocator.free(merger_input);

        const VTable = vision_adapters.VTableFor(VisionRuntime);
        return try vision_adapters.runVisionProgram(
            @ptrCast(self),
            &VTable.vtable,
            self.allocator,
            &self.vision_stage_plans.vision_encode,
            &self.dispatch_counters,
            vision_adapters.adapter_table,
            image.grid,
            merger_input,
            deepstack_layer_embeddings,
        );
    }

    pub fn deepstackLayerToMergerIndex(self: *const VisionRuntime, layer_idx: usize) ?usize {
        for (self.deepstack_visual_layers[0..self.deepstack_layer_count], 0..) |visual_layer_idx, merger_idx| {
            if (merger_idx >= self.deepstack_mergers.len) break;
            if (layer_idx == visual_layer_idx) return merger_idx;
        }
        return null;
    }

    fn runDeepstackMerger(
        self: *VisionRuntime,
        grid: image_mod.VisionGrid,
        hidden: []const f32,
        merger: *const DeepstackMergerWeights,
    ) ![]f32 {
        const patch_count: usize = @as(usize, @intCast(grid.temporal)) *
            @as(usize, @intCast(grid.height)) *
            @as(usize, @intCast(grid.width));
        if (hidden.len != patch_count * self.vision_hidden_size) return error.InvalidShape;

        const merge_units = self.spatial_merge_size * self.spatial_merge_size;
        const merged_h = @as(usize, @intCast(grid.height)) / self.spatial_merge_size;
        const merged_w = @as(usize, @intCast(grid.width)) / self.spatial_merge_size;
        const merged_t = @as(usize, @intCast(grid.temporal));
        const merged_tokens = merged_t * merged_h * merged_w;
        const merged_width = self.vision_hidden_size * merge_units;

        const merged = try self.allocator.alloc(f32, merged_tokens * merged_width);
        defer self.allocator.free(merged);
        try cpu_image_ops.packMergedGridTokens(
            hidden,
            @as(usize, @intCast(grid.height)),
            @as(usize, @intCast(grid.width)),
            @as(usize, @intCast(grid.temporal)),
            self.vision_hidden_size,
            self.spatial_merge_size,
            self.token_order == .row_major,
            merged,
        );

        if (merger.norm_weight.len != merged_width or merger.norm_bias.len != merged_width) {
            return error.InvalidShape;
        }

        const normed = try self.allocator.alloc(f32, merged.len);
        defer self.allocator.free(normed);
        try cpu_norm.layerNormRows(
            merged,
            normed,
            merged_tokens,
            merged_width,
            merger.norm_weight,
            merger.norm_bias,
            1e-6,
        );

        const fc1_out = try self.allocator.alloc(f32, merged_tokens * merged_width);
        defer self.allocator.free(fc1_out);

        var normed_view = Tensor.view2DSlice(normed, merged_tokens, merged_width);
        var fc1_view = Tensor.view2DSlice(fc1_out, merged_tokens, merged_width);
        try cpu_linalg.matmulAuto(&normed_view, &merger.fc1_weight, &fc1_view, &self.matmul_scratch);

        cpu_common.addBiasRows(fc1_out, merger.fc1_bias, merged_tokens, merged_width);

        const tv = compute.cpu.tensor_view;
        const activation = compute.cpu.activation_view;
        var fc1_tensor = Tensor.view2DSlice(fc1_out, merged_tokens, merged_width);
        const fc1_tv = tv.fromSimpleTensor(&fc1_tensor) orelse return error.InvalidShape;
        activation.gelu(fc1_tv, fc1_tv);

        const out = try self.allocator.alloc(f32, merged_tokens * self.language_hidden_size);
        errdefer self.allocator.free(out);

        var out_view = Tensor.view2DSlice(out, merged_tokens, self.language_hidden_size);
        try cpu_linalg.matmulAuto(&fc1_view, &merger.fc2_weight, &out_view, &self.matmul_scratch);

        cpu_common.addBiasRows(out, merger.fc2_bias, merged_tokens, self.language_hidden_size);

        return out;
    }

    pub fn runMerger(self: *VisionRuntime, grid: image_mod.VisionGrid, hidden: []const f32) ![]f32 {
        const patch_count: usize = @as(usize, @intCast(grid.temporal)) *
            @as(usize, @intCast(grid.height)) *
            @as(usize, @intCast(grid.width));
        if (hidden.len != patch_count * self.vision_hidden_size) return error.InvalidShape;

        const merge_units = self.spatial_merge_size * self.spatial_merge_size;
        const merged_h = @as(usize, @intCast(grid.height)) / self.spatial_merge_size;
        const merged_w = @as(usize, @intCast(grid.width)) / self.spatial_merge_size;
        const merged_t = @as(usize, @intCast(grid.temporal));
        const merged_tokens = merged_t * merged_h * merged_w;
        const merged_width = self.vision_hidden_size * merge_units;

        const norm_on_patch = self.merger_norm_weight.len == self.vision_hidden_size and self.merger_norm_bias.len == self.vision_hidden_size;
        const norm_on_merged = self.merger_norm_weight.len == merged_width and self.merger_norm_bias.len == merged_width;
        const has_norm = self.merger_norm_weight.len > 0 or self.merger_norm_bias.len > 0;
        if (has_norm and !norm_on_patch and !norm_on_merged) return error.InvalidShape;

        const pack_input = if (norm_on_patch) blk: {
            const normalized = try self.allocator.alloc(f32, hidden.len);
            errdefer self.allocator.free(normalized);
            try cpu_norm.layerNormRows(
                hidden,
                normalized,
                patch_count,
                self.vision_hidden_size,
                self.merger_norm_weight,
                self.merger_norm_bias,
                1e-6,
            );
            break :blk normalized;
        } else hidden;
        defer if (pack_input.ptr != hidden.ptr) self.allocator.free(pack_input);

        const merged = try self.allocator.alloc(f32, merged_tokens * merged_width);
        defer self.allocator.free(merged);
        try cpu_image_ops.packMergedGridTokens(
            pack_input,
            @as(usize, @intCast(grid.height)),
            @as(usize, @intCast(grid.width)),
            @as(usize, @intCast(grid.temporal)),
            self.vision_hidden_size,
            self.spatial_merge_size,
            self.token_order == .row_major,
            merged,
        );

        if (norm_on_merged) {
            try cpu_norm.layerNormRows(
                merged,
                merged,
                merged_tokens,
                merged_width,
                self.merger_norm_weight,
                self.merger_norm_bias,
                1e-6,
            );
        }

        const fc1_out = try self.allocator.alloc(f32, merged_tokens * self.merger_intermediate_size);
        defer self.allocator.free(fc1_out);

        var merged_view = Tensor.view2DSlice(merged, merged_tokens, merged_width);
        var fc1_view = Tensor.view2DSlice(fc1_out, merged_tokens, self.merger_intermediate_size);
        try cpu_linalg.matmulAuto(&merged_view, &self.merger_fc1_weight, &fc1_view, &self.matmul_scratch);

        cpu_common.addBiasRows(fc1_out, self.merger_fc1_bias, merged_tokens, self.merger_intermediate_size);

        // In-place GELU
        const tv = compute.cpu.tensor_view;
        const activation = compute.cpu.activation_view;
        var fc1_tensor = Tensor.view2DSlice(fc1_out, merged_tokens, self.merger_intermediate_size);
        const fc1_tv = tv.fromSimpleTensor(&fc1_tensor) orelse return error.InvalidShape;
        activation.gelu(fc1_tv, fc1_tv);

        const out = try self.allocator.alloc(f32, merged_tokens * self.language_hidden_size);
        errdefer self.allocator.free(out);

        var out_view = Tensor.view2DSlice(out, merged_tokens, self.language_hidden_size);
        try cpu_linalg.matmulAuto(&fc1_view, &self.merger_fc2_weight, &out_view, &self.matmul_scratch);

        cpu_common.addBiasRows(out, self.merger_fc2_bias, merged_tokens, self.language_hidden_size);

        return out;
    }

    fn fillVisionRotaryTables(
        self: *const VisionRuntime,
        grid: image_mod.VisionGrid,
        cos_out: []f32,
        sin_out: []f32,
    ) !void {
        return cpu_rotary.fillSpatialRotaryTables(
            self.allocator,
            @as(usize, @intCast(grid.height)),
            @as(usize, @intCast(grid.width)),
            @as(usize, @intCast(grid.temporal)),
            self.spatial_merge_size,
            self.vision_hidden_size,
            self.vision_num_heads,
            cos_out,
            sin_out,
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
    return cpu_memory.scatterRowsByMatchedId(
        hidden_states,
        seq_len,
        d_model,
        token_ids,
        image_token_id,
        embeddings,
    );
}

fn tensorToMetalArray(t: *const Tensor) !ArrayHandle {
    const shape = t.shape[0..@as(usize, @intCast(t.n_dims))];
    switch (t.dtype) {
        .f32 => return graph.createArrayF32(t.asSlice(f32), shape),
        .bf16 => {
            const data_ptr: [*]align(1) const u16 = @ptrCast(t.data_ptr);
            return graph.createArrayBF16Unaligned(data_ptr, t.data_size / 2, shape);
        },
        .f16 => {
            const data_ptr: [*]align(1) const u16 = @ptrCast(t.data_ptr);
            return graph.createArrayF16Unaligned(data_ptr, t.data_size / 2, shape);
        },
        else => return error.UnsupportedDType,
    }
}

fn f32SliceToMetalArray(values: []const f32) ArrayHandle {
    const shape = [_]i64{@intCast(values.len)};
    return graph.createArrayF32(values, &shape);
}

fn flattenPatchProjWeight(
    weight_5d: Tensor,
    vision_hidden_size: usize,
    temporal_patch_size: usize,
    patch_size: usize,
) Tensor {
    const patch_dim = 3 * temporal_patch_size * patch_size * patch_size;
    return Tensor.view(
        weight_5d.data_ptr.?,
        &.{ vision_hidden_size, patch_dim },
        weight_5d.dtype,
        weight_5d.data_size,
    );
}

fn loadDeepstackMergers(
    allocator: std.mem.Allocator,
    st: *vision_load.SafeTensors,
    vision_metadata: *const vision_load.VisionMetadata,
    vision_hidden_size: usize,
    language_hidden_size: usize,
    spatial_merge_size: usize,
    merger_count: usize,
) ![]VisionRuntime.DeepstackMergerWeights {
    if (merger_count == 0) return &.{};
    const merge_units = spatial_merge_size * spatial_merge_size;
    const merged_width = vision_hidden_size * merge_units;

    var mergers = try allocator.alloc(VisionRuntime.DeepstackMergerWeights, merger_count);
    errdefer allocator.free(mergers);

    var loaded_count: usize = 0;
    errdefer {
        for (mergers[0..loaded_count]) |*merger| merger.deinit(allocator);
    }

    for (0..merger_count) |merger_idx| {
        const norm_weight_tensor = vision_load.getLayerTensorByTemplates(st, merger_idx, vision_metadata.deepstack_norm_weight_templates) catch |err| switch (err) {
            error.NotFound => break,
            else => return err,
        };
        const norm_weight = try vision_tensor_convert.tensorToOwnedF32(allocator, norm_weight_tensor);
        errdefer allocator.free(norm_weight);

        const norm_bias = try vision_tensor_convert.tensorToOwnedF32(allocator, try vision_load.getLayerTensorByTemplates(st, merger_idx, vision_metadata.deepstack_norm_bias_templates));
        errdefer allocator.free(norm_bias);

        const fc1_weight = try vision_load.getLayerTensorByTemplates(st, merger_idx, vision_metadata.deepstack_fc1_weight_templates);
        const fc1_bias = try vision_tensor_convert.tensorToOwnedF32(allocator, try vision_load.getLayerTensorByTemplates(st, merger_idx, vision_metadata.deepstack_fc1_bias_templates));
        errdefer allocator.free(fc1_bias);

        const fc2_weight = try vision_load.getLayerTensorByTemplates(st, merger_idx, vision_metadata.deepstack_fc2_weight_templates);
        const fc2_bias = try vision_tensor_convert.tensorToOwnedF32(allocator, try vision_load.getLayerTensorByTemplates(st, merger_idx, vision_metadata.deepstack_fc2_bias_templates));
        errdefer allocator.free(fc2_bias);

        if (norm_weight.len != merged_width or norm_bias.len != merged_width) return error.InvalidShape;
        if (fc1_bias.len != merged_width) return error.InvalidShape;
        if (fc2_bias.len != language_hidden_size) return error.InvalidShape;

        mergers[loaded_count] = .{
            .norm_weight = norm_weight,
            .norm_bias = norm_bias,
            .fc1_weight = fc1_weight,
            .fc1_bias = fc1_bias,
            .fc2_weight = fc2_weight,
            .fc2_bias = fc2_bias,
        };
        loaded_count += 1;
    }

    if (loaded_count == 0) {
        allocator.free(mergers);
        return &.{};
    }

    return allocator.realloc(mergers, loaded_count);
}

fn extractPatchInput(
    self: *const VisionRuntime,
    image: PrefillVisionImage,
    out: []f32,
) !void {
    if (self.token_order == .row_major) {
        return cpu_image_ops.extractGridBlocksRowMajor(
            image.pixels,
            @as(usize, image.height),
            @as(usize, image.width),
            @as(usize, image.grid.height),
            @as(usize, image.grid.width),
            @as(usize, image.grid.temporal),
            self.patch_size,
            self.temporal_patch_size,
            out,
        );
    }

    return cpu_image_ops.extractGridBlocksMerged(
        image.pixels,
        @as(usize, image.height),
        @as(usize, image.width),
        @as(usize, image.grid.height),
        @as(usize, image.grid.width),
        @as(usize, image.grid.temporal),
        self.patch_size,
        self.temporal_patch_size,
        self.spatial_merge_size,
        out,
    );
}

fn interpolatePosEmbeddings(
    self: *const VisionRuntime,
    grid: image_mod.VisionGrid,
    out: []f32,
) !void {
    return cpu_image_ops.interpolateGridEmbeddings(
        self.pos_embed_f32,
        self.num_grid_side,
        self.vision_hidden_size,
        @as(usize, grid.height),
        @as(usize, grid.width),
        @as(usize, grid.temporal),
        self.spatial_merge_size,
        self.token_order == .row_major,
        self.token_order == .row_major,
        out,
    );
}

test "scatterVisionEmbeddings replaces image token rows" {
    var hidden = [_]f32{
        0, 0,
        1, 1,
        2, 2,
        3, 3,
    };
    const tokens = [_]u32{ 10, 99, 99, 11 };
    const embeds = [_]f32{ 7, 8, 9, 10 };

    try scatterVisionEmbeddings(hidden[0..], 4, 2, tokens[0..], 99, embeds[0..]);

    try std.testing.expectEqualSlices(f32, &[_]f32{ 0, 0 }, hidden[0..2]);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 7, 8 }, hidden[2..4]);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 9, 10 }, hidden[4..6]);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 3, 3 }, hidden[6..8]);
}

test "vision adapter table covers vision opcodes" {
    for (vision_adapters.required_opcodes) |opcode| {
        try std.testing.expect(vision_adapters.adapter_table[@intFromEnum(opcode)] != null);
    }

    try std.testing.expect(vision_adapters.adapter_table[@intFromEnum(opcode_map.Opcode.rmsnorm)] == null);
    try std.testing.expect(vision_adapters.adapter_table[@intFromEnum(opcode_map.Opcode.mul_scalar)] == null);
}
