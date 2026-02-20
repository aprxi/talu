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
const io = @import("../../../../io/root.zig");
const layer_ops = @import("../../../../models/layer_ops.zig");
const models = @import("../../../../models/root.zig");
const compute = @import("../../../../compute/root.zig");
const cpu_blocks = @import("../executor/weights.zig");
const exec_block = @import("../executor/block.zig");
const image_mod = @import("../../../../image/root.zig");
const common_vision = @import("types.zig");
const vision_tensor_convert = @import("tensor_convert.zig");

const Tensor = tensor.Tensor;
const ModelConfig = tensor.ModelConfig;
const LoadedModel = models.LoadedModel;
const cpu_linalg = compute.cpu.linalg;
const cpu_common = compute.cpu.common;
const cpu_image_ops = compute.cpu.image_ops;
const cpu_norm = compute.cpu.normalization;
const cpu_rotary = compute.cpu.rotary;
const cpu_rowwise = compute.cpu.rowwise;
const cpu_memory = compute.cpu.memory;
const log = @import("../../../../log.zig");

pub const PrefillVisionImage = common_vision.PrefillVisionImage;
pub const PrefillVisionInput = common_vision.PrefillVisionInput;
pub const EncodedVisionOutput = common_vision.EncodedVisionOutput;

pub const VisionRuntime = struct {
    allocator: std.mem.Allocator,

    vision_hidden_size: usize,
    vision_depth: usize,
    vision_num_heads: usize,
    vision_intermediate_size: usize,
    language_hidden_size: usize,
    patch_size: usize,
    spatial_merge_size: usize,
    temporal_patch_size: usize,
    num_pos_embeddings: usize,
    num_grid_side: usize,

    patch_proj_weight: Tensor, // [vision_hidden, patch_dim]
    patch_proj_bias: []f32, // [vision_hidden]

    pos_embed_f32: []f32, // [num_pos_embeddings * vision_hidden]

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
    blocks: []cpu_blocks.TransformerBlock,
    exec_blocks: []exec_block.Block,
    scratch: cpu_blocks.ScratchBuffer,

    const LayerWeights = struct {
        ln1_weight: Tensor,
        ln1_bias: Tensor,
        ln2_weight: Tensor,
        ln2_bias: Tensor,
        qkv_weight: Tensor,
        o_weight: Tensor,
        fc1_weight: Tensor,
        fc2_weight: Tensor,

        qkv_bias_all: []f32,
        o_bias: []f32,
        fc1_bias: []f32,
        fc2_bias: []f32,
        fc1_bias_tensor: Tensor,
        fc2_bias_tensor: Tensor,

        fn deinit(self: *LayerWeights, allocator: std.mem.Allocator) void {
            if (self.qkv_bias_all.len > 0) allocator.free(self.qkv_bias_all);
            if (self.o_bias.len > 0) allocator.free(self.o_bias);
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
        const language_hidden_size: usize = @intCast(cfg.d_model);
        const patch_size: usize = @intCast(if (cfg.vision_patch_size > 0) cfg.vision_patch_size else 16);
        const spatial_merge_size: usize = @intCast(if (cfg.vision_spatial_merge_size > 0) cfg.vision_spatial_merge_size else 2);
        const temporal_patch_size: usize = @intCast(if (cfg.vision_temporal_patch_size > 0) cfg.vision_temporal_patch_size else 2);
        const num_pos_embeddings: usize = @intCast(if (cfg.vision_num_position_embeddings > 0) cfg.vision_num_position_embeddings else 2304);
        var deepstack_visual_layers: [8]usize = [_]usize{0} ** 8;
        var deepstack_layer_count: usize = @intCast(cfg.vision_probe_layer_count);
        deepstack_layer_count = @min(deepstack_layer_count, deepstack_visual_layers.len);
        for (0..deepstack_layer_count) |idx| {
            deepstack_visual_layers[idx] = cfg.vision_probe_layers[idx];
        }

        const num_grid_side = std.math.sqrt(num_pos_embeddings);
        if (num_grid_side * num_grid_side != num_pos_embeddings) return error.InvalidShape;

        var st = &loaded.st.?;

        const patch_proj_w_5d = try st.getTensor("model.visual.patch_embed.proj.weight", null);
        const patch_proj_weight = flattenPatchProjWeight(patch_proj_w_5d, vision_hidden_size, temporal_patch_size, patch_size);
        const patch_proj_bias_tensor = try st.getTensor("model.visual.patch_embed.proj.bias", null);
        const patch_proj_bias = try vision_tensor_convert.tensorToOwnedF32(allocator, patch_proj_bias_tensor);
        errdefer allocator.free(patch_proj_bias);

        const pos_embed_tensor = try st.getTensor("model.visual.pos_embed.weight", null);
        const pos_embed_f32 = try vision_tensor_convert.tensorToOwnedF32(allocator, pos_embed_tensor);
        errdefer allocator.free(pos_embed_f32);

        const merger_norm_weight = try vision_tensor_convert.tensorToOwnedF32(allocator, try st.getTensor("model.visual.merger.norm.weight", null));
        errdefer allocator.free(merger_norm_weight);
        const merger_norm_bias = try vision_tensor_convert.tensorToOwnedF32(allocator, try st.getTensor("model.visual.merger.norm.bias", null));
        errdefer allocator.free(merger_norm_bias);

        const merger_fc1_weight = try st.getTensor("model.visual.merger.linear_fc1.weight", null);
        const merger_fc1_bias = try vision_tensor_convert.tensorToOwnedF32(allocator, try st.getTensor("model.visual.merger.linear_fc1.bias", null));
        errdefer allocator.free(merger_fc1_bias);

        const merger_fc2_weight = try st.getTensor("model.visual.merger.linear_fc2.weight", null);
        const merger_fc2_bias = try vision_tensor_convert.tensorToOwnedF32(allocator, try st.getTensor("model.visual.merger.linear_fc2.bias", null));
        errdefer allocator.free(merger_fc2_bias);
        const deepstack_mergers = try loadDeepstackMergers(
            allocator,
            st,
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

        var blocks = try allocator.alloc(cpu_blocks.TransformerBlock, vision_depth);
        errdefer allocator.free(blocks);

        var exec_blocks = try allocator.alloc(exec_block.Block, vision_depth);
        errdefer allocator.free(exec_blocks);

        var built_layers: usize = 0;
        errdefer {
            for (0..built_layers) |idx| {
                blocks[idx].deinit(allocator);
                layer_weights[idx].deinit(allocator);
            }
        }

        const head_dim: usize = vision_hidden_size / vision_num_heads;
        const attention_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        for (0..vision_depth) |layer_idx| {
            var name_buf: [128]u8 = undefined;

            const ln1_weight_name = try std.fmt.bufPrint(&name_buf, "model.visual.blocks.{d}.norm1.weight", .{layer_idx});
            const ln1_weight = try st.getTensor(ln1_weight_name, null);
            const ln1_bias_name = try std.fmt.bufPrint(&name_buf, "model.visual.blocks.{d}.norm1.bias", .{layer_idx});
            const ln1_bias = try st.getTensor(ln1_bias_name, null);

            const ln2_weight_name = try std.fmt.bufPrint(&name_buf, "model.visual.blocks.{d}.norm2.weight", .{layer_idx});
            const ln2_weight = try st.getTensor(ln2_weight_name, null);
            const ln2_bias_name = try std.fmt.bufPrint(&name_buf, "model.visual.blocks.{d}.norm2.bias", .{layer_idx});
            const ln2_bias = try st.getTensor(ln2_bias_name, null);

            const qkv_weight_name = try std.fmt.bufPrint(&name_buf, "model.visual.blocks.{d}.attn.qkv.weight", .{layer_idx});
            const qkv_weight = try st.getTensor(qkv_weight_name, null);
            const qkv_bias_name = try std.fmt.bufPrint(&name_buf, "model.visual.blocks.{d}.attn.qkv.bias", .{layer_idx});
            const qkv_bias_all = try vision_tensor_convert.tensorToOwnedF32(allocator, try st.getTensor(qkv_bias_name, null));
            errdefer allocator.free(qkv_bias_all);

            const o_weight_name = try std.fmt.bufPrint(&name_buf, "model.visual.blocks.{d}.attn.proj.weight", .{layer_idx});
            const o_weight = try st.getTensor(o_weight_name, null);
            const o_bias_name = try std.fmt.bufPrint(&name_buf, "model.visual.blocks.{d}.attn.proj.bias", .{layer_idx});
            const o_bias = try vision_tensor_convert.tensorToOwnedF32(allocator, try st.getTensor(o_bias_name, null));
            errdefer allocator.free(o_bias);

            const fc1_weight_name = try std.fmt.bufPrint(&name_buf, "model.visual.blocks.{d}.mlp.linear_fc1.weight", .{layer_idx});
            const fc1_weight = try st.getTensor(fc1_weight_name, null);
            const fc1_bias_name = try std.fmt.bufPrint(&name_buf, "model.visual.blocks.{d}.mlp.linear_fc1.bias", .{layer_idx});
            const fc1_bias = try vision_tensor_convert.tensorToOwnedF32(allocator, try st.getTensor(fc1_bias_name, null));
            errdefer allocator.free(fc1_bias);
            const fc2_weight_name = try std.fmt.bufPrint(&name_buf, "model.visual.blocks.{d}.mlp.linear_fc2.weight", .{layer_idx});
            const fc2_weight = try st.getTensor(fc2_weight_name, null);
            const fc2_bias_name = try std.fmt.bufPrint(&name_buf, "model.visual.blocks.{d}.mlp.linear_fc2.bias", .{layer_idx});
            const fc2_bias = try vision_tensor_convert.tensorToOwnedF32(allocator, try st.getTensor(fc2_bias_name, null));
            errdefer allocator.free(fc2_bias);

            layer_weights[layer_idx] = .{
                .ln1_weight = ln1_weight,
                .ln1_bias = ln1_bias,
                .ln2_weight = ln2_weight,
                .ln2_bias = ln2_bias,
                .qkv_weight = qkv_weight,
                .o_weight = o_weight,
                .fc1_weight = fc1_weight,
                .fc2_weight = fc2_weight,
                .qkv_bias_all = qkv_bias_all,
                .o_bias = o_bias,
                .fc1_bias = fc1_bias,
                .fc2_bias = fc2_bias,
                .fc1_bias_tensor = Tensor.view2DSlice(fc1_bias, 1, fc1_bias.len),
                .fc2_bias_tensor = Tensor.view2DSlice(fc2_bias, 1, fc2_bias.len),
            };

            const q_bias = layer_weights[layer_idx].qkv_bias_all[0..vision_hidden_size];
            const k_bias = layer_weights[layer_idx].qkv_bias_all[vision_hidden_size .. 2 * vision_hidden_size];
            const v_bias = layer_weights[layer_idx].qkv_bias_all[2 * vision_hidden_size .. 3 * vision_hidden_size];

            const weights = cpu_blocks.BlockWeights{ .attention_mlp = .{
                .ln1_weight = &layer_weights[layer_idx].ln1_weight,
                .ln2_weight = &layer_weights[layer_idx].ln2_weight,
                .ln1_bias = &layer_weights[layer_idx].ln1_bias,
                .ln2_bias = &layer_weights[layer_idx].ln2_bias,
                .q_proj = null,
                .k_proj = null,
                .v_proj = null,
                .o_proj = &layer_weights[layer_idx].o_weight,
                .w1 = &layer_weights[layer_idx].fc1_weight,
                .w2 = &layer_weights[layer_idx].fc2_weight,
                .w3 = null,
                .w1_bias = &layer_weights[layer_idx].fc1_bias_tensor,
                .w2_bias = &layer_weights[layer_idx].fc2_bias_tensor,
                .rope = null,
                .sliding_window = 0,
                .fused = .{ .qkv_proj = layer_weights[layer_idx].qkv_weight },
                .q_norm = null,
                .k_norm = null,
                .pre_ffn_norm = null,
                .post_ffn_norm = null,
                .q_bias = q_bias,
                .k_bias = k_bias,
                .v_bias = v_bias,
                .o_bias = layer_weights[layer_idx].o_bias,
                .moe_weights = null,
                .sinks = null,
                .is_causal = false,
                .block_ops = &.{},
                .mla_config = null,
                .q_a_proj = null,
                .q_a_norm = null,
                .q_b_proj = null,
                .kv_a_proj = null,
                .kv_a_norm = null,
                .kv_b_proj = null,
            } };

            blocks[layer_idx] = try cpu_blocks.TransformerBlock.init(
                allocator,
                vision_hidden_size,
                vision_intermediate_size,
                vision_num_heads,
                vision_num_heads,
                head_dim,
                num_pos_embeddings,
                weights,
                1e-6,
                .{},
                1.0,
                attention_scale,
                true,
                layer_idx,
            );

            exec_blocks[layer_idx] = .{
                .program = &vision_block_program,
                .block = &blocks[layer_idx],
                .block_idx = layer_idx,
                .hidden_size = vision_hidden_size,
                .mamba_layer_idx = null,
                .shortconv_layer_idx = null,
            };
            try exec_blocks[layer_idx].validate();

            built_layers += 1;
        }

        var scratch = try cpu_blocks.ScratchBuffer.init(
            allocator,
            vision_hidden_size,
            vision_intermediate_size,
            vision_depth,
        );
        errdefer scratch.deinit();

        return VisionRuntime{
            .allocator = allocator,
            .vision_hidden_size = vision_hidden_size,
            .vision_depth = vision_depth,
            .vision_num_heads = vision_num_heads,
            .vision_intermediate_size = vision_intermediate_size,
            .language_hidden_size = language_hidden_size,
            .patch_size = patch_size,
            .spatial_merge_size = spatial_merge_size,
            .temporal_patch_size = temporal_patch_size,
            .num_pos_embeddings = num_pos_embeddings,
            .num_grid_side = num_grid_side,
            .patch_proj_weight = patch_proj_weight,
            .patch_proj_bias = patch_proj_bias,
            .pos_embed_f32 = pos_embed_f32,
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
            .blocks = blocks,
            .exec_blocks = exec_blocks,
            .scratch = scratch,
        };
    }

    pub fn deinit(self: *VisionRuntime) void {
        for (self.blocks) |*block| block.deinit(self.allocator);
        for (self.layer_weights) |*w| w.deinit(self.allocator);
        self.scratch.deinit();

        if (self.patch_proj_bias.len > 0) self.allocator.free(self.patch_proj_bias);
        if (self.pos_embed_f32.len > 0) self.allocator.free(self.pos_embed_f32);
        if (self.merger_norm_weight.len > 0) self.allocator.free(self.merger_norm_weight);
        if (self.merger_norm_bias.len > 0) self.allocator.free(self.merger_norm_bias);
        if (self.merger_fc1_bias.len > 0) self.allocator.free(self.merger_fc1_bias);
        if (self.merger_fc2_bias.len > 0) self.allocator.free(self.merger_fc2_bias);
        if (self.deepstack_mergers.len > 0) {
            for (self.deepstack_mergers) |*merger| merger.deinit(self.allocator);
            self.allocator.free(self.deepstack_mergers);
        }

        if (self.exec_blocks.len > 0) self.allocator.free(self.exec_blocks);
        if (self.blocks.len > 0) self.allocator.free(self.blocks);
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

    const EncodedSingleImage = struct {
        merged_embeddings: []f32,
        deepstack_layer_embeddings: []const []const f32,

        fn deinit(self: *EncodedSingleImage, allocator: std.mem.Allocator) void {
            if (self.merged_embeddings.len > 0) allocator.free(self.merged_embeddings);
            if (self.deepstack_layer_embeddings.len > 0) {
                for (self.deepstack_layer_embeddings) |layer_embed| {
                    if (layer_embed.len > 0) allocator.free(layer_embed);
                }
                allocator.free(self.deepstack_layer_embeddings);
            }
            self.* = .{
                .merged_embeddings = &.{},
                .deepstack_layer_embeddings = &.{},
            };
        }
    };

    fn encodeSingleImage(self: *VisionRuntime, image: PrefillVisionImage) !EncodedSingleImage {
        const patch_count: usize = @as(usize, @intCast(image.grid.temporal)) *
            @as(usize, @intCast(image.grid.height)) *
            @as(usize, @intCast(image.grid.width));
        if (patch_count == 0) return error.InvalidImageDimensions;

        log.debug("scheduler", "encodeSingleImage start", .{
            .patch_count = patch_count,
            .grid_t = @as(usize, image.grid.temporal),
            .grid_h = @as(usize, image.grid.height),
            .grid_w = @as(usize, image.grid.width),
            .vision_hidden = self.vision_hidden_size,
            .max_pos = self.num_pos_embeddings,
        }, @src());

        const patch_dim = 3 * self.temporal_patch_size * self.patch_size * self.patch_size;
        const patch_input = try self.allocator.alloc(f32, patch_count * patch_dim);
        defer self.allocator.free(patch_input);

        try extractPatchInputReordered(self, image, patch_input);

        const patch_hidden = try self.allocator.alloc(f32, patch_count * self.vision_hidden_size);
        defer self.allocator.free(patch_hidden);

        // patch projection: [patch_count, patch_dim] @ [vision_hidden, patch_dim]^T -> [patch_count, vision_hidden]
        var patch_input_view = Tensor.view2DSlice(patch_input, patch_count, patch_dim);
        var patch_hidden_view = Tensor.view2DSlice(patch_hidden, patch_count, self.vision_hidden_size);
        try cpu_linalg.matmulAuto(&patch_input_view, &self.patch_proj_weight, &patch_hidden_view, &self.scratch.matmul_scratch);
        cpu_common.addBiasRows(patch_hidden, self.patch_proj_bias, patch_count, self.vision_hidden_size);

        const pos_embeds = try self.allocator.alloc(f32, patch_count * self.vision_hidden_size);
        defer self.allocator.free(pos_embeds);
        try interpolatePosEmbeddings(self, image.grid, pos_embeds);

        cpu_rowwise.addScaledInPlace(patch_hidden, pos_embeds, 1.0);

        const hidden_a = try self.allocator.alloc(f32, patch_hidden.len);
        defer self.allocator.free(hidden_a);
        @memcpy(hidden_a, patch_hidden);

        const hidden_b = try self.allocator.alloc(f32, patch_hidden.len);
        defer self.allocator.free(hidden_b);

        try self.scratch.ensure(patch_count);
        self.scratch.resetCaches();

        var current = hidden_a;
        var next = hidden_b;
        const vision_head_dim = self.vision_hidden_size / self.vision_num_heads;
        const vision_rope_cos = try self.allocator.alloc(f32, patch_count * vision_head_dim);
        defer self.allocator.free(vision_rope_cos);
        const vision_rope_sin = try self.allocator.alloc(f32, patch_count * vision_head_dim);
        defer self.allocator.free(vision_rope_sin);
        try self.fillVisionRotaryTables(image.grid, vision_rope_cos, vision_rope_sin);

        for (self.blocks) |*blk| {
            if (blk.getAttentionMut()) |attn| {
                attn.runtime_rope = .{
                    .cos = vision_rope_cos,
                    .sin = vision_rope_sin,
                    .dim = vision_head_dim,
                };
            }
        }
        defer {
            for (self.blocks) |*blk| {
                if (blk.getAttentionMut()) |attn| {
                    attn.runtime_rope = null;
                }
            }
        }

        const deepstack_count = self.deepstack_mergers.len;
        var deepstack_layer_embeddings: [][]const f32 = if (deepstack_count > 0)
            try self.allocator.alloc([]const f32, deepstack_count)
        else
            try self.allocator.alloc([]const f32, 0);
        errdefer self.allocator.free(deepstack_layer_embeddings);
        errdefer {
            for (deepstack_layer_embeddings) |layer_embed| {
                if (layer_embed.len > 0) self.allocator.free(layer_embed);
            }
        }
        for (0..deepstack_layer_embeddings.len) |i| deepstack_layer_embeddings[i] = &.{};

        for (self.exec_blocks, 0..) |*block, layer_idx| {
            var in_view = Tensor.view3D(std.mem.sliceAsBytes(current), patch_count, self.vision_hidden_size);
            var out_view = Tensor.view3D(std.mem.sliceAsBytes(next), patch_count, self.vision_hidden_size);
            try block.forward(
                &in_view,
                &out_view,
                &self.scratch,
                &self.scratch.attn_caches[layer_idx],
                false,
            );

            const tmp = current;
            current = next;
            next = tmp;

            if (self.deepstackLayerToMergerIndex(layer_idx)) |merger_idx| {
                deepstack_layer_embeddings[merger_idx] = try self.runDeepstackMerger(
                    image.grid,
                    current,
                    &self.deepstack_mergers[merger_idx],
                );
            }
        }

        if (deepstack_count > 0) {
            for (deepstack_layer_embeddings) |layer_embed| {
                if (layer_embed.len == 0) return error.InvalidState;
            }
        }

        log.debug("scheduler", "Vision transformer done", .{
            .layers = self.vision_depth,
            .patch_count = patch_count,
            .hidden_len = current.len,
        }, @src());

        return .{
            .merged_embeddings = try self.runMerger(image.grid, current),
            .deepstack_layer_embeddings = deepstack_layer_embeddings,
        };
    }

    fn deepstackLayerToMergerIndex(self: *const VisionRuntime, layer_idx: usize) ?usize {
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
        try cpu_image_ops.packMergedTokens(
            hidden,
            @as(usize, @intCast(grid.height)),
            @as(usize, @intCast(grid.width)),
            @as(usize, @intCast(grid.temporal)),
            self.vision_hidden_size,
            self.spatial_merge_size,
            false,
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
        try cpu_linalg.matmulAuto(&normed_view, &merger.fc1_weight, &fc1_view, &self.scratch.matmul_scratch);

        cpu_common.addBiasRows(fc1_out, merger.fc1_bias, merged_tokens, merged_width);

        const tv = compute.cpu.tensor_view;
        const activation = compute.cpu.activation_view;
        var fc1_tensor = Tensor.view2DSlice(fc1_out, merged_tokens, merged_width);
        const fc1_tv = tv.fromSimpleTensor(&fc1_tensor) orelse return error.InvalidShape;
        activation.gelu(fc1_tv, fc1_tv);

        const out = try self.allocator.alloc(f32, merged_tokens * self.language_hidden_size);
        errdefer self.allocator.free(out);

        var out_view = Tensor.view2DSlice(out, merged_tokens, self.language_hidden_size);
        try cpu_linalg.matmulAuto(&fc1_view, &merger.fc2_weight, &out_view, &self.scratch.matmul_scratch);

        cpu_common.addBiasRows(out, merger.fc2_bias, merged_tokens, self.language_hidden_size);

        return out;
    }

    fn runMerger(self: *VisionRuntime, grid: image_mod.VisionGrid, hidden: []const f32) ![]f32 {
        const patch_count: usize = @as(usize, @intCast(grid.temporal)) *
            @as(usize, @intCast(grid.height)) *
            @as(usize, @intCast(grid.width));
        if (hidden.len != patch_count * self.vision_hidden_size) return error.InvalidShape;

        const merge_units_log = self.spatial_merge_size * self.spatial_merge_size;
        const merged_tokens_log = (@as(usize, @intCast(grid.temporal))) *
            (@as(usize, @intCast(grid.height)) / self.spatial_merge_size) *
            (@as(usize, @intCast(grid.width)) / self.spatial_merge_size);
        log.debug("scheduler", "runMerger", .{
            .patch_count = patch_count,
            .merged_tokens = merged_tokens_log,
            .merged_width = self.vision_hidden_size * merge_units_log,
            .fc1_out_dim = self.vision_intermediate_size,
            .fc2_out_dim = self.language_hidden_size,
        }, @src());

        const normed = try self.allocator.alloc(f32, hidden.len);
        defer self.allocator.free(normed);
        try cpu_norm.layerNormRows(
            hidden,
            normed,
            patch_count,
            self.vision_hidden_size,
            self.merger_norm_weight,
            self.merger_norm_bias,
            1e-6,
        );

        const merge_units = self.spatial_merge_size * self.spatial_merge_size;
        const merged_h = @as(usize, @intCast(grid.height)) / self.spatial_merge_size;
        const merged_w = @as(usize, @intCast(grid.width)) / self.spatial_merge_size;
        const merged_t = @as(usize, @intCast(grid.temporal));
        const merged_tokens = merged_t * merged_h * merged_w;
        const merged_width = self.vision_hidden_size * merge_units;

        const merged = try self.allocator.alloc(f32, merged_tokens * merged_width);
        defer self.allocator.free(merged);

        try cpu_image_ops.packMergedTokens(
            normed,
            @as(usize, @intCast(grid.height)),
            @as(usize, @intCast(grid.width)),
            @as(usize, @intCast(grid.temporal)),
            self.vision_hidden_size,
            self.spatial_merge_size,
            false,
            merged,
        );

        const fc1_out = try self.allocator.alloc(f32, merged_tokens * self.vision_intermediate_size);
        defer self.allocator.free(fc1_out);

        var merged_view = Tensor.view2DSlice(merged, merged_tokens, merged_width);
        var fc1_view = Tensor.view2DSlice(fc1_out, merged_tokens, self.vision_intermediate_size);
        try cpu_linalg.matmulAuto(&merged_view, &self.merger_fc1_weight, &fc1_view, &self.scratch.matmul_scratch);

        cpu_common.addBiasRows(fc1_out, self.merger_fc1_bias, merged_tokens, self.vision_intermediate_size);

        // In-place GELU
        const tv = compute.cpu.tensor_view;
        const activation = compute.cpu.activation_view;
        var fc1_tensor = Tensor.view2DSlice(fc1_out, merged_tokens, self.vision_intermediate_size);
        const fc1_tv = tv.fromSimpleTensor(&fc1_tensor) orelse return error.InvalidShape;
        activation.gelu(fc1_tv, fc1_tv);

        const out = try self.allocator.alloc(f32, merged_tokens * self.language_hidden_size);
        errdefer self.allocator.free(out);

        var out_view = Tensor.view2DSlice(out, merged_tokens, self.language_hidden_size);
        try cpu_linalg.matmulAuto(&fc1_view, &self.merger_fc2_weight, &out_view, &self.scratch.matmul_scratch);

        cpu_common.addBiasRows(out, self.merger_fc2_bias, merged_tokens, self.language_hidden_size);

        log.debug("scheduler", "runMerger done", .{
            .out_len = out.len,
        }, @src());

        return out;
    }

    fn fillVisionRotaryTables(
        self: *const VisionRuntime,
        grid: image_mod.VisionGrid,
        cos_out: []f32,
        sin_out: []f32,
    ) !void {
        return cpu_rotary.fillVisionRotaryTables(
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
    return cpu_memory.scatterEmbeddingsByTokenId(
        hidden_states,
        seq_len,
        d_model,
        token_ids,
        image_token_id,
        embeddings,
    );
}

const vision_block_program = [_]layer_ops.LayerOp{
    .{ .kernel = .{ .id = 0, .in = .residual, .out = .norm_out, .debug_type = .norm } },
    .{ .kernel = .{ .id = 1, .in = .norm_out, .out = .branch_out, .debug_type = .multihead_attention } },
    .{ .add = .{ .branch = .branch_out, .scale = .one } },
    .{ .kernel = .{ .id = 2, .in = .residual, .out = .norm_out, .debug_type = .norm } },
    .{ .kernel = .{ .id = 3, .in = .norm_out, .out = .branch_out, .debug_type = .mlp } },
    .{ .add = .{ .branch = .branch_out, .scale = .one } },
};

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
    st: *io.safetensors.root.UnifiedSafeTensors,
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
        var name_buf: [160]u8 = undefined;

        const norm_weight_name = try std.fmt.bufPrint(&name_buf, "model.visual.deepstack_merger_list.{d}.norm.weight", .{merger_idx});
        const norm_weight_tensor = st.getTensor(norm_weight_name, null) catch |err| switch (err) {
            error.TensorNotFound => break,
            else => return err,
        };
        const norm_weight = try vision_tensor_convert.tensorToOwnedF32(allocator, norm_weight_tensor);
        errdefer allocator.free(norm_weight);

        const norm_bias_name = try std.fmt.bufPrint(&name_buf, "model.visual.deepstack_merger_list.{d}.norm.bias", .{merger_idx});
        const norm_bias = try vision_tensor_convert.tensorToOwnedF32(allocator, try st.getTensor(norm_bias_name, null));
        errdefer allocator.free(norm_bias);

        const fc1_weight_name = try std.fmt.bufPrint(&name_buf, "model.visual.deepstack_merger_list.{d}.linear_fc1.weight", .{merger_idx});
        const fc1_weight = try st.getTensor(fc1_weight_name, null);
        const fc1_bias_name = try std.fmt.bufPrint(&name_buf, "model.visual.deepstack_merger_list.{d}.linear_fc1.bias", .{merger_idx});
        const fc1_bias = try vision_tensor_convert.tensorToOwnedF32(allocator, try st.getTensor(fc1_bias_name, null));
        errdefer allocator.free(fc1_bias);

        const fc2_weight_name = try std.fmt.bufPrint(&name_buf, "model.visual.deepstack_merger_list.{d}.linear_fc2.weight", .{merger_idx});
        const fc2_weight = try st.getTensor(fc2_weight_name, null);
        const fc2_bias_name = try std.fmt.bufPrint(&name_buf, "model.visual.deepstack_merger_list.{d}.linear_fc2.bias", .{merger_idx});
        const fc2_bias = try vision_tensor_convert.tensorToOwnedF32(allocator, try st.getTensor(fc2_bias_name, null));
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

fn extractPatchInputReordered(
    self: *const VisionRuntime,
    image: PrefillVisionImage,
    out: []f32,
) !void {
    return cpu_image_ops.extractPatchInputMerged(
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
    return cpu_image_ops.interpolatePosEmbeddings(
        self.pos_embed_f32,
        self.num_grid_side,
        self.vision_hidden_size,
        @as(usize, grid.height),
        @as(usize, grid.width),
        @as(usize, grid.temporal),
        self.spatial_merge_size,
        false,
        false,
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
