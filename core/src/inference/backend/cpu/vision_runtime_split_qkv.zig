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
const tensor = @import("../../../tensor.zig");
const dtype_mod = @import("../../../dtype.zig");
const io = @import("../../../io/root.zig");
const compute = @import("../../../compute/root.zig");
const cpu_blocks = @import("block_kernels.zig");
const exec_block = @import("../../executor/block.zig");
const layer_ops = @import("../../../graph/root.zig").layer_ops;
const image_mod = @import("../../../image/root.zig");

const Tensor = tensor.Tensor;
const ModelConfig = tensor.ModelConfig;
const LoadedModel = io.weights.LoadedModel;
const matmul = compute.ops.matmul;

pub const PrefillVisionImage = struct {
    pixels: []f32,
    width: u32,
    height: u32,
    grid: image_mod.VisionGrid,
    token_count: usize,

    pub fn deinit(self: *PrefillVisionImage, allocator: std.mem.Allocator) void {
        if (self.pixels.len > 0) allocator.free(self.pixels);
        self.* = .{
            .pixels = &.{},
            .width = 0,
            .height = 0,
            .grid = .{ .temporal = 0, .height = 0, .width = 0 },
            .token_count = 0,
        };
    }
};

pub const PrefillVisionInput = struct {
    images: []PrefillVisionImage,
    image_token_id: u32,

    pub fn deinit(self: *PrefillVisionInput, allocator: std.mem.Allocator) void {
        for (self.images) |*img| img.deinit(allocator);
        if (self.images.len > 0) allocator.free(self.images);
        self.* = .{ .images = &.{}, .image_token_id = 0 };
    }
};

pub const EncodedVisionOutput = struct {
    merged_embeddings: []f32,
    deepstack_layer_embeddings: []const []const f32,

    pub fn deinit(self: *EncodedVisionOutput, allocator: std.mem.Allocator) void {
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
    blocks: []cpu_blocks.TransformerBlock,
    exec_blocks: []exec_block.Block,
    scratch: cpu_blocks.ScratchBuffer,

    const LayerWeights = struct {
        ln1_weight: Tensor,
        ln1_bias: Tensor,
        ln2_weight: Tensor,
        ln2_bias: Tensor,
        qkv_weight: Tensor,
        has_fused_qkv: bool,
        q_proj_weight: Tensor,
        k_proj_weight: Tensor,
        v_proj_weight: Tensor,
        has_split_qkv: bool,
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
        const merger_intermediate_size: usize = @intCast(if (cfg.projector_hidden_size > 0) cfg.projector_hidden_size else if (cfg.vision_intermediate_size > 0) cfg.vision_intermediate_size else 4 * cfg.vision_hidden_size);
        const language_hidden_size: usize = @intCast(cfg.d_model);
        const patch_size: usize = @intCast(if (cfg.vision_patch_size > 0) cfg.vision_patch_size else 16);
        const spatial_merge_size: usize = @intCast(if (cfg.vision_spatial_merge_size > 0) cfg.vision_spatial_merge_size else 1);
        const temporal_patch_size: usize = @intCast(if (cfg.vision_temporal_patch_size > 0) cfg.vision_temporal_patch_size else 1);
        const num_pos_embeddings: usize = @intCast(if (cfg.vision_num_position_embeddings > 0) cfg.vision_num_position_embeddings else 2304);
        const max_patch_tokens: usize = @intCast(if (cfg.vision_max_num_patches > 0) cfg.vision_max_num_patches else if (cfg.vision_num_position_embeddings > 0) cfg.vision_num_position_embeddings else 2304);
        var deepstack_visual_layers: [8]usize = [_]usize{0} ** 8;
        var deepstack_layer_count: usize = @intCast(cfg.vision_probe_layer_count);
        deepstack_layer_count = @min(deepstack_layer_count, deepstack_visual_layers.len);
        for (0..deepstack_layer_count) |idx| {
            deepstack_visual_layers[idx] = cfg.vision_probe_layers[idx];
        }

        const num_grid_side = std.math.sqrt(num_pos_embeddings);
        if (num_grid_side * num_grid_side != num_pos_embeddings) return error.InvalidShape;

        const st = &loaded.st.?;

        const patch_proj_w_5d = try getTensorByCandidates(st, &.{
            "model.visual.patch_embed.proj.weight",
            "model.vision_tower.vision_model.embeddings.patch_embedding.weight",
        });
        const patch_proj_weight = flattenPatchProjWeight(patch_proj_w_5d, vision_hidden_size, temporal_patch_size, patch_size);
        const patch_proj_bias_tensor = try getTensorByCandidates(st, &.{
            "model.visual.patch_embed.proj.bias",
            "model.vision_tower.vision_model.embeddings.patch_embedding.bias",
        });
        const patch_proj_bias = try tensorToOwnedF32(allocator, patch_proj_bias_tensor);
        errdefer allocator.free(patch_proj_bias);

        const pos_embed_tensor = try getTensorByCandidates(st, &.{
            "model.visual.pos_embed.weight",
            "model.vision_tower.vision_model.embeddings.position_embedding.weight",
        });
        const pos_embed_f32 = try tensorToOwnedF32(allocator, pos_embed_tensor);
        errdefer allocator.free(pos_embed_f32);

        var vision_post_norm_weight: []f32 = &.{};
        var vision_post_norm_bias: []f32 = &.{};
        const post_norm_weight_tensor = getTensorByCandidates(st, &.{
            "model.visual.norm.weight",
            "model.vision_tower.vision_model.post_layernorm.weight",
        }) catch |err| switch (err) {
            error.NotFound => null,
            else => return err,
        };
        if (post_norm_weight_tensor) |tensor_weight| {
            vision_post_norm_weight = try tensorToOwnedF32(allocator, tensor_weight);
            vision_post_norm_bias = try tensorToOwnedF32(allocator, try getTensorByCandidates(st, &.{
                "model.visual.norm.bias",
                "model.vision_tower.vision_model.post_layernorm.bias",
            }));
        }
        errdefer if (vision_post_norm_weight.len > 0) allocator.free(vision_post_norm_weight);
        errdefer if (vision_post_norm_bias.len > 0) allocator.free(vision_post_norm_bias);

        var merger_norm_weight: []f32 = &.{};
        var merger_norm_bias: []f32 = &.{};
        const merger_norm_tensor = getTensorByCandidates(st, &.{
            "model.visual.merger.norm.weight",
            "model.multi_modal_projector.layer_norm.weight",
        }) catch |err| switch (err) {
            error.NotFound => null,
            else => return err,
        };
        if (merger_norm_tensor) |tensor_weight| {
            merger_norm_weight = try tensorToOwnedF32(allocator, tensor_weight);
            merger_norm_bias = try tensorToOwnedF32(allocator, try getTensorByCandidates(st, &.{
                "model.visual.merger.norm.bias",
                "model.multi_modal_projector.layer_norm.bias",
            }));
        }
        errdefer if (merger_norm_weight.len > 0) allocator.free(merger_norm_weight);
        errdefer if (merger_norm_bias.len > 0) allocator.free(merger_norm_bias);

        const merger_fc1_weight = try getTensorByCandidates(st, &.{
            "model.visual.merger.linear_fc1.weight",
            "model.multi_modal_projector.linear_1.weight",
        });
        const merger_fc1_bias = try tensorToOwnedF32(allocator, try getTensorByCandidates(st, &.{
            "model.visual.merger.linear_fc1.bias",
            "model.multi_modal_projector.linear_1.bias",
        }));
        errdefer allocator.free(merger_fc1_bias);

        const merger_fc2_weight = try getTensorByCandidates(st, &.{
            "model.visual.merger.linear_fc2.weight",
            "model.multi_modal_projector.linear_2.weight",
        });
        const merger_fc2_bias = try tensorToOwnedF32(allocator, try getTensorByCandidates(st, &.{
            "model.visual.merger.linear_fc2.bias",
            "model.multi_modal_projector.linear_2.bias",
        }));
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
        var saw_fused_qkv = false;
        var saw_split_qkv = false;
        errdefer {
            for (0..built_layers) |idx| {
                blocks[idx].deinit(allocator);
                layer_weights[idx].deinit(allocator);
            }
        }

        const head_dim: usize = vision_hidden_size / vision_num_heads;
        const attention_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        for (0..vision_depth) |layer_idx| {
            const ln1_weight = try getLayerTensorByTemplates(st, layer_idx, &.{
                "model.visual.blocks.{d}.norm1.weight",
                "model.vision_tower.vision_model.encoder.layers.{d}.layer_norm1.weight",
            });
            const ln1_bias = try getLayerTensorByTemplates(st, layer_idx, &.{
                "model.visual.blocks.{d}.norm1.bias",
                "model.vision_tower.vision_model.encoder.layers.{d}.layer_norm1.bias",
            });

            const ln2_weight = try getLayerTensorByTemplates(st, layer_idx, &.{
                "model.visual.blocks.{d}.norm2.weight",
                "model.vision_tower.vision_model.encoder.layers.{d}.layer_norm2.weight",
            });
            const ln2_bias = try getLayerTensorByTemplates(st, layer_idx, &.{
                "model.visual.blocks.{d}.norm2.bias",
                "model.vision_tower.vision_model.encoder.layers.{d}.layer_norm2.bias",
            });

            const fused_qkv_weight = getLayerTensorByTemplates(st, layer_idx, &.{
                "model.visual.blocks.{d}.attn.qkv.weight",
            }) catch |err| switch (err) {
                error.NotFound => null,
                else => return err,
            };
            if (fused_qkv_weight != null) {
                saw_fused_qkv = true;
            } else {
                saw_split_qkv = true;
            }
            const fused_qkv_bias_all = if (fused_qkv_weight != null)
                try tensorToOwnedF32(allocator, try getLayerTensorByTemplates(st, layer_idx, &.{
                    "model.visual.blocks.{d}.attn.qkv.bias",
                }))
            else
                null;
            const qkv_bias_all = if (fused_qkv_bias_all) |bias| bias else blk: {
                const q_bias = try tensorToOwnedF32(allocator, try getLayerTensorByTemplates(st, layer_idx, &.{
                    "model.vision_tower.vision_model.encoder.layers.{d}.self_attn.q_proj.bias",
                }));
                errdefer allocator.free(q_bias);
                const k_bias = try tensorToOwnedF32(allocator, try getLayerTensorByTemplates(st, layer_idx, &.{
                    "model.vision_tower.vision_model.encoder.layers.{d}.self_attn.k_proj.bias",
                }));
                errdefer allocator.free(k_bias);
                const v_bias = try tensorToOwnedF32(allocator, try getLayerTensorByTemplates(st, layer_idx, &.{
                    "model.vision_tower.vision_model.encoder.layers.{d}.self_attn.v_proj.bias",
                }));
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
                try getLayerTensorByTemplates(st, layer_idx, &.{
                    "model.vision_tower.vision_model.encoder.layers.{d}.self_attn.q_proj.weight",
                })
            else
                undefined;
            const k_proj_weight = if (fused_qkv_weight == null)
                try getLayerTensorByTemplates(st, layer_idx, &.{
                    "model.vision_tower.vision_model.encoder.layers.{d}.self_attn.k_proj.weight",
                })
            else
                undefined;
            const v_proj_weight = if (fused_qkv_weight == null)
                try getLayerTensorByTemplates(st, layer_idx, &.{
                    "model.vision_tower.vision_model.encoder.layers.{d}.self_attn.v_proj.weight",
                })
            else
                undefined;

            const o_weight = try getLayerTensorByTemplates(st, layer_idx, &.{
                "model.visual.blocks.{d}.attn.proj.weight",
                "model.vision_tower.vision_model.encoder.layers.{d}.self_attn.out_proj.weight",
            });
            const o_bias = try tensorToOwnedF32(allocator, try getLayerTensorByTemplates(st, layer_idx, &.{
                "model.visual.blocks.{d}.attn.proj.bias",
                "model.vision_tower.vision_model.encoder.layers.{d}.self_attn.out_proj.bias",
            }));
            errdefer allocator.free(o_bias);

            const fc1_weight = try getLayerTensorByTemplates(st, layer_idx, &.{
                "model.visual.blocks.{d}.mlp.linear_fc1.weight",
                "model.vision_tower.vision_model.encoder.layers.{d}.mlp.fc1.weight",
            });
            const fc1_bias = try tensorToOwnedF32(allocator, try getLayerTensorByTemplates(st, layer_idx, &.{
                "model.visual.blocks.{d}.mlp.linear_fc1.bias",
                "model.vision_tower.vision_model.encoder.layers.{d}.mlp.fc1.bias",
            }));
            errdefer allocator.free(fc1_bias);
            const fc2_weight = try getLayerTensorByTemplates(st, layer_idx, &.{
                "model.visual.blocks.{d}.mlp.linear_fc2.weight",
                "model.vision_tower.vision_model.encoder.layers.{d}.mlp.fc2.weight",
            });
            const fc2_bias = try tensorToOwnedF32(allocator, try getLayerTensorByTemplates(st, layer_idx, &.{
                "model.visual.blocks.{d}.mlp.linear_fc2.bias",
                "model.vision_tower.vision_model.encoder.layers.{d}.mlp.fc2.bias",
            }));
            errdefer allocator.free(fc2_bias);

            layer_weights[layer_idx] = .{
                .ln1_weight = ln1_weight,
                .ln1_bias = ln1_bias,
                .ln2_weight = ln2_weight,
                .ln2_bias = ln2_bias,
                .qkv_weight = fused_qkv_weight orelse undefined,
                .has_fused_qkv = fused_qkv_weight != null,
                .q_proj_weight = if (fused_qkv_weight == null) q_proj_weight else undefined,
                .k_proj_weight = if (fused_qkv_weight == null) k_proj_weight else undefined,
                .v_proj_weight = if (fused_qkv_weight == null) v_proj_weight else undefined,
                .has_split_qkv = fused_qkv_weight == null,
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
                .q_proj = if (layer_weights[layer_idx].has_split_qkv) &layer_weights[layer_idx].q_proj_weight else null,
                .k_proj = if (layer_weights[layer_idx].has_split_qkv) &layer_weights[layer_idx].k_proj_weight else null,
                .v_proj = if (layer_weights[layer_idx].has_split_qkv) &layer_weights[layer_idx].v_proj_weight else null,
                .o_proj = &layer_weights[layer_idx].o_weight,
                .w1 = &layer_weights[layer_idx].fc1_weight,
                .w2 = &layer_weights[layer_idx].fc2_weight,
                .w3 = null,
                .w1_bias = &layer_weights[layer_idx].fc1_bias_tensor,
                .w2_bias = &layer_weights[layer_idx].fc2_bias_tensor,
                .rope = null,
                .sliding_window = 0,
                .fused = .{ .qkv_proj = if (layer_weights[layer_idx].has_fused_qkv) layer_weights[layer_idx].qkv_weight else null },
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
                max_patch_tokens,
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

        if (saw_fused_qkv and saw_split_qkv) return error.InvalidShape;
        const use_vision_rope = saw_fused_qkv;
        const token_order: TokenOrder = if (use_vision_rope) .merge_block else .row_major;

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

        const patch_dim = 3 * self.temporal_patch_size * self.patch_size * self.patch_size;
        const patch_input = try self.allocator.alloc(f32, patch_count * patch_dim);
        defer self.allocator.free(patch_input);

        try extractPatchInput(self, image, patch_input);

        const patch_hidden = try self.allocator.alloc(f32, patch_count * self.vision_hidden_size);
        defer self.allocator.free(patch_hidden);

        // patch projection: [patch_count, patch_dim] @ [vision_hidden, patch_dim]^T -> [patch_count, vision_hidden]
        var patch_input_view = Tensor.view2DSlice(patch_input, patch_count, patch_dim);
        var patch_hidden_view = Tensor.view2DSlice(patch_hidden, patch_count, self.vision_hidden_size);
        try matmul.matmulAuto(&patch_input_view, &self.patch_proj_weight, &patch_hidden_view, &self.scratch.matmul_scratch);

        for (0..patch_count) |row| {
            const row_slice = patch_hidden[row * self.vision_hidden_size ..][0..self.vision_hidden_size];
            for (row_slice, self.patch_proj_bias) |*v, b| v.* += b;
        }

        const pos_embeds = try self.allocator.alloc(f32, patch_count * self.vision_hidden_size);
        defer self.allocator.free(pos_embeds);
        try interpolatePosEmbeddings(self, image.grid, pos_embeds);

        for (patch_hidden, pos_embeds) |*h, p| h.* += p;

        const hidden_a = try self.allocator.alloc(f32, patch_hidden.len);
        defer self.allocator.free(hidden_a);
        @memcpy(hidden_a, patch_hidden);

        const hidden_b = try self.allocator.alloc(f32, patch_hidden.len);
        defer self.allocator.free(hidden_b);

        try self.scratch.ensure(patch_count);
        self.scratch.resetCaches();

        var current = hidden_a;
        var next = hidden_b;
        if (self.use_vision_rope) {
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

        const merger_input = if (self.vision_post_norm_weight.len > 0) blk: {
            if (self.vision_post_norm_weight.len != self.vision_hidden_size or self.vision_post_norm_bias.len != self.vision_hidden_size) {
                return error.InvalidShape;
            }

            const normalized = try self.allocator.alloc(f32, current.len);
            errdefer self.allocator.free(normalized);
            for (0..patch_count) |row| {
                const in_row = current[row * self.vision_hidden_size ..][0..self.vision_hidden_size];
                const out_row = normalized[row * self.vision_hidden_size ..][0..self.vision_hidden_size];
                layerNormRow(in_row, out_row, self.vision_post_norm_weight, self.vision_post_norm_bias, 1e-6);
            }
            break :blk normalized;
        } else current;
        defer if (merger_input.ptr != current.ptr) self.allocator.free(merger_input);

        return .{
            .merged_embeddings = try self.runMerger(image.grid, merger_input),
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
        try self.packMergedTokens(grid, hidden, merged_tokens, merged_width, merged);

        if (merger.norm_weight.len != merged_width or merger.norm_bias.len != merged_width) {
            return error.InvalidShape;
        }

        const normed = try self.allocator.alloc(f32, merged.len);
        defer self.allocator.free(normed);
        for (0..merged_tokens) |row| {
            const in_row = merged[row * merged_width ..][0..merged_width];
            const out_row = normed[row * merged_width ..][0..merged_width];
            layerNormRow(in_row, out_row, merger.norm_weight, merger.norm_bias, 1e-6);
        }

        const fc1_out = try self.allocator.alloc(f32, merged_tokens * merged_width);
        defer self.allocator.free(fc1_out);

        var normed_view = Tensor.view2DSlice(normed, merged_tokens, merged_width);
        var fc1_view = Tensor.view2DSlice(fc1_out, merged_tokens, merged_width);
        try matmul.matmulAuto(&normed_view, &merger.fc1_weight, &fc1_view, &self.scratch.matmul_scratch);

        for (0..merged_tokens) |row| {
            const row_slice = fc1_out[row * merged_width ..][0..merged_width];
            for (row_slice, merger.fc1_bias) |*v, b| v.* += b;
        }

        const tv = compute.ops.tensor_view;
        const activation = compute.ops.activation;
        var fc1_tensor = Tensor.view2DSlice(fc1_out, merged_tokens, merged_width);
        const fc1_tv = tv.fromSimpleTensor(&fc1_tensor) orelse return error.InvalidShape;
        activation.gelu(fc1_tv, fc1_tv);

        const out = try self.allocator.alloc(f32, merged_tokens * self.language_hidden_size);
        errdefer self.allocator.free(out);

        var out_view = Tensor.view2DSlice(out, merged_tokens, self.language_hidden_size);
        try matmul.matmulAuto(&fc1_view, &merger.fc2_weight, &out_view, &self.scratch.matmul_scratch);

        for (0..merged_tokens) |row| {
            const row_slice = out[row * self.language_hidden_size ..][0..self.language_hidden_size];
            for (row_slice, merger.fc2_bias) |*v, b| v.* += b;
        }

        return out;
    }

    fn runMerger(self: *VisionRuntime, grid: image_mod.VisionGrid, hidden: []const f32) ![]f32 {
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
            for (0..patch_count) |row| {
                const in_row = hidden[row * self.vision_hidden_size ..][0..self.vision_hidden_size];
                const out_row = normalized[row * self.vision_hidden_size ..][0..self.vision_hidden_size];
                layerNormRow(in_row, out_row, self.merger_norm_weight, self.merger_norm_bias, 1e-6);
            }
            break :blk normalized;
        } else hidden;
        defer if (pack_input.ptr != hidden.ptr) self.allocator.free(pack_input);

        const merged = try self.allocator.alloc(f32, merged_tokens * merged_width);
        defer self.allocator.free(merged);
        try self.packMergedTokens(grid, pack_input, merged_tokens, merged_width, merged);

        if (norm_on_merged) {
            for (0..merged_tokens) |row| {
                const row_slice = merged[row * merged_width ..][0..merged_width];
                layerNormRow(row_slice, row_slice, self.merger_norm_weight, self.merger_norm_bias, 1e-6);
            }
        }

        const fc1_out = try self.allocator.alloc(f32, merged_tokens * self.merger_intermediate_size);
        defer self.allocator.free(fc1_out);

        var merged_view = Tensor.view2DSlice(merged, merged_tokens, merged_width);
        var fc1_view = Tensor.view2DSlice(fc1_out, merged_tokens, self.merger_intermediate_size);
        try matmul.matmulAuto(&merged_view, &self.merger_fc1_weight, &fc1_view, &self.scratch.matmul_scratch);

        for (0..merged_tokens) |row| {
            const row_slice = fc1_out[row * self.merger_intermediate_size ..][0..self.merger_intermediate_size];
            for (row_slice, self.merger_fc1_bias) |*v, b| v.* += b;
        }

        // In-place GELU
        const tv = compute.ops.tensor_view;
        const activation = compute.ops.activation;
        var fc1_tensor = Tensor.view2DSlice(fc1_out, merged_tokens, self.merger_intermediate_size);
        const fc1_tv = tv.fromSimpleTensor(&fc1_tensor) orelse return error.InvalidShape;
        activation.gelu(fc1_tv, fc1_tv);

        const out = try self.allocator.alloc(f32, merged_tokens * self.language_hidden_size);
        errdefer self.allocator.free(out);

        var out_view = Tensor.view2DSlice(out, merged_tokens, self.language_hidden_size);
        try matmul.matmulAuto(&fc1_view, &self.merger_fc2_weight, &out_view, &self.scratch.matmul_scratch);

        for (0..merged_tokens) |row| {
            const row_slice = out[row * self.language_hidden_size ..][0..self.language_hidden_size];
            for (row_slice, self.merger_fc2_bias) |*v, b| v.* += b;
        }

        return out;
    }

    fn packMergedTokens(
        self: *const VisionRuntime,
        grid: image_mod.VisionGrid,
        hidden: []const f32,
        merged_tokens: usize,
        merged_width: usize,
        out: []f32,
    ) !void {
        const merge_units = self.spatial_merge_size * self.spatial_merge_size;
        const patch_count: usize = @as(usize, @intCast(grid.temporal)) *
            @as(usize, @intCast(grid.height)) *
            @as(usize, @intCast(grid.width));

        if (hidden.len != patch_count * self.vision_hidden_size) return error.InvalidShape;
        if (out.len != merged_tokens * merged_width) return error.InvalidShape;
        if (patch_count != merged_tokens * merge_units) return error.InvalidShape;

        if (self.token_order == .merge_block) {
            for (0..merged_tokens) |dst_token| {
                const dst_base = dst_token * merged_width;
                for (0..merge_units) |u| {
                    const src_token = dst_token * merge_units + u;
                    const src_base = src_token * self.vision_hidden_size;
                    const dst_offset = dst_base + u * self.vision_hidden_size;
                    @memcpy(
                        out[dst_offset..][0..self.vision_hidden_size],
                        hidden[src_base..][0..self.vision_hidden_size],
                    );
                }
            }
            return;
        }

        const grid_h = @as(usize, @intCast(grid.height));
        const grid_w = @as(usize, @intCast(grid.width));
        const grid_t = @as(usize, @intCast(grid.temporal));
        const merge = self.spatial_merge_size;
        const merged_h = grid_h / merge;
        const merged_w = grid_w / merge;
        const patches_per_frame = grid_h * grid_w;
        const merged_per_frame = merged_h * merged_w;

        for (0..grid_t) |t_idx| {
            const src_frame_base = t_idx * patches_per_frame;
            const dst_frame_base = t_idx * merged_per_frame;
            for (0..merged_h) |bh| {
                for (0..merged_w) |bw| {
                    const dst_token = dst_frame_base + bh * merged_w + bw;
                    const dst_base = dst_token * merged_width;
                    var unit_idx: usize = 0;
                    for (0..merge) |ih| {
                        for (0..merge) |iw| {
                            const row = bh * merge + ih;
                            const col = bw * merge + iw;
                            const src_token = src_frame_base + row * grid_w + col;
                            const src_base = src_token * self.vision_hidden_size;
                            const dst_offset = dst_base + unit_idx * self.vision_hidden_size;
                            @memcpy(
                                out[dst_offset..][0..self.vision_hidden_size],
                                hidden[src_base..][0..self.vision_hidden_size],
                            );
                            unit_idx += 1;
                        }
                    }
                }
            }
        }
    }

    fn fillVisionRotaryTables(
        self: *const VisionRuntime,
        grid: image_mod.VisionGrid,
        cos_out: []f32,
        sin_out: []f32,
    ) !void {
        const grid_h = @as(usize, @intCast(grid.height));
        const grid_w = @as(usize, @intCast(grid.width));
        const grid_t = @as(usize, @intCast(grid.temporal));
        const merge = self.spatial_merge_size;
        if ((grid_h % merge) != 0 or (grid_w % merge) != 0) return error.InvalidShape;

        const merged_h = grid_h / merge;
        const merged_w = grid_w / merge;
        const token_count = grid_t * grid_h * grid_w;
        const head_dim = self.vision_hidden_size / self.vision_num_heads;
        if ((head_dim % 4) != 0) return error.InvalidShape;
        if (cos_out.len != token_count * head_dim or sin_out.len != token_count * head_dim) return error.InvalidShape;

        const half_dim = head_dim / 2;
        const freq_dim = half_dim / 2;

        const inv_freq = try self.allocator.alloc(f32, freq_dim);
        defer self.allocator.free(inv_freq);
        for (0..freq_dim) |idx| {
            const exponent = @as(f32, @floatFromInt(2 * idx)) / @as(f32, @floatFromInt(half_dim));
            inv_freq[idx] = 1.0 / std.math.pow(f32, 10000.0, exponent);
        }

        var token_idx: usize = 0;
        for (0..grid_t) |_| {
            for (0..merged_h) |bh| {
                for (0..merged_w) |bw| {
                    for (0..merge) |ih| {
                        for (0..merge) |iw| {
                            const row = bh * merge + ih;
                            const col = bw * merge + iw;
                            const row_pos = @as(f32, @floatFromInt(row));
                            const col_pos = @as(f32, @floatFromInt(col));
                            const base = token_idx * head_dim;

                            for (0..freq_dim) |f| {
                                const row_angle = row_pos * inv_freq[f];
                                const col_angle = col_pos * inv_freq[f];
                                const row_cos = @cos(row_angle);
                                const row_sin = @sin(row_angle);
                                const col_cos = @cos(col_angle);
                                const col_sin = @sin(col_angle);

                                cos_out[base + f] = row_cos;
                                sin_out[base + f] = row_sin;
                                cos_out[base + freq_dim + f] = col_cos;
                                sin_out[base + freq_dim + f] = col_sin;

                                cos_out[base + half_dim + f] = row_cos;
                                sin_out[base + half_dim + f] = row_sin;
                                cos_out[base + half_dim + freq_dim + f] = col_cos;
                                sin_out[base + half_dim + freq_dim + f] = col_sin;
                            }
                            token_idx += 1;
                        }
                    }
                }
            }
        }
        if (token_idx != token_count) return error.InvalidState;
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
    if (hidden_states.len != seq_len * d_model) return error.InvalidShape;
    if (token_ids.len != seq_len) return error.InvalidShape;
    if (embeddings.len % d_model != 0) return error.InvalidShape;

    const embed_tokens = embeddings.len / d_model;
    var embed_idx: usize = 0;

    for (token_ids, 0..) |tok, pos| {
        if (tok != image_token_id) continue;
        if (embed_idx >= embed_tokens) return error.InvalidShape;
        const src = embeddings[embed_idx * d_model ..][0..d_model];
        const dst = hidden_states[pos * d_model ..][0..d_model];
        @memcpy(dst, src);
        embed_idx += 1;
    }

    if (embed_idx != embed_tokens) return error.InvalidShape;
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

fn getTensorByCandidates(st: *io.safetensors.root.UnifiedSafeTensors, candidates: []const []const u8) !Tensor {
    for (candidates) |name| {
        const t = st.getTensor(name, null) catch |err| switch (err) {
            error.NotFound => continue,
            else => return err,
        };
        return t;
    }
    return error.NotFound;
}

fn getLayerTensorByTemplates(
    st: *io.safetensors.root.UnifiedSafeTensors,
    layer_idx: usize,
    templates: []const []const u8,
) !Tensor {
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
            error.NotFound => break,
            else => return err,
        };
        const norm_weight = try tensorToOwnedF32(allocator, norm_weight_tensor);
        errdefer allocator.free(norm_weight);

        const norm_bias_name = try std.fmt.bufPrint(&name_buf, "model.visual.deepstack_merger_list.{d}.norm.bias", .{merger_idx});
        const norm_bias = try tensorToOwnedF32(allocator, try st.getTensor(norm_bias_name, null));
        errdefer allocator.free(norm_bias);

        const fc1_weight_name = try std.fmt.bufPrint(&name_buf, "model.visual.deepstack_merger_list.{d}.linear_fc1.weight", .{merger_idx});
        const fc1_weight = try st.getTensor(fc1_weight_name, null);
        const fc1_bias_name = try std.fmt.bufPrint(&name_buf, "model.visual.deepstack_merger_list.{d}.linear_fc1.bias", .{merger_idx});
        const fc1_bias = try tensorToOwnedF32(allocator, try st.getTensor(fc1_bias_name, null));
        errdefer allocator.free(fc1_bias);

        const fc2_weight_name = try std.fmt.bufPrint(&name_buf, "model.visual.deepstack_merger_list.{d}.linear_fc2.weight", .{merger_idx});
        const fc2_weight = try st.getTensor(fc2_weight_name, null);
        const fc2_bias_name = try std.fmt.bufPrint(&name_buf, "model.visual.deepstack_merger_list.{d}.linear_fc2.bias", .{merger_idx});
        const fc2_bias = try tensorToOwnedF32(allocator, try st.getTensor(fc2_bias_name, null));
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

fn tensorToOwnedF32(allocator: std.mem.Allocator, t: Tensor) ![]f32 {
    const n = t.numel;
    var out = try allocator.alloc(f32, n);
    errdefer allocator.free(out);

    switch (t.dtype) {
        .f32 => {
            const src = t.asSlice(f32);
            @memcpy(out, src[0..n]);
        },
        .bf16 => {
            const src = t.asSliceUnaligned(u16);
            for (0..n) |i| out[i] = dtype_mod.bf16ToF32(src[i]);
        },
        .f16 => {
            const src = t.asSliceUnaligned(u16);
            for (0..n) |i| out[i] = dtype_mod.fp16ToF32(src[i]);
        },
        else => return error.UnsupportedDType,
    }

    return out;
}

fn extractPatchInput(
    self: *const VisionRuntime,
    image: PrefillVisionImage,
    out: []f32,
) !void {
    const height = @as(usize, image.height);
    const width = @as(usize, image.width);
    const grid_h = @as(usize, image.grid.height);
    const grid_w = @as(usize, image.grid.width);
    const grid_t = @as(usize, image.grid.temporal);
    const merge = self.spatial_merge_size;

    const total_frames = image.pixels.len / (3 * height * width);
    const patch_dim = 3 * self.temporal_patch_size * self.patch_size * self.patch_size;

    var patch_idx: usize = 0;
    if (self.token_order == .row_major) {
        for (0..grid_t) |t_block| {
            const frame_base = t_block * self.temporal_patch_size;
            for (0..grid_h) |patch_h| {
                for (0..grid_w) |patch_w| {
                    const y0 = patch_h * self.patch_size;
                    const x0 = patch_w * self.patch_size;

                    var dst = patch_idx * patch_dim;
                    for (0..self.temporal_patch_size) |tp| {
                        const frame_idx = frame_base + tp;
                        if (frame_idx >= total_frames) return error.InvalidImageDimensions;
                        for (0..self.patch_size) |py| {
                            for (0..self.patch_size) |px| {
                                const src_y = y0 + py;
                                const src_x = x0 + px;
                                for (0..3) |c| {
                                    const src_idx = (((c * total_frames + frame_idx) * height + src_y) * width + src_x);
                                    out[dst] = image.pixels[src_idx];
                                    dst += 1;
                                }
                            }
                        }
                    }
                    patch_idx += 1;
                }
            }
        }
        return;
    }

    const merged_h = grid_h / merge;
    const merged_w = grid_w / merge;
    for (0..grid_t) |t_block| {
        const frame_base = t_block * self.temporal_patch_size;
        for (0..merged_h) |bh| {
            for (0..merged_w) |bw| {
                for (0..merge) |ih| {
                    for (0..merge) |iw| {
                        const patch_h = bh * merge + ih;
                        const patch_w = bw * merge + iw;
                        const y0 = patch_h * self.patch_size;
                        const x0 = patch_w * self.patch_size;

                        var dst = patch_idx * patch_dim;
                        for (0..3) |c| {
                            for (0..self.temporal_patch_size) |tp| {
                                const frame_idx = frame_base + tp;
                                if (frame_idx >= total_frames) return error.InvalidImageDimensions;
                                for (0..self.patch_size) |py| {
                                    for (0..self.patch_size) |px| {
                                        const src_y = y0 + py;
                                        const src_x = x0 + px;
                                        const src_idx = (((c * total_frames + frame_idx) * height + src_y) * width + src_x);
                                        out[dst] = image.pixels[src_idx];
                                        dst += 1;
                                    }
                                }
                            }
                        }
                        patch_idx += 1;
                    }
                }
            }
        }
    }
}

fn interpolatePosEmbeddings(
    self: *const VisionRuntime,
    grid: image_mod.VisionGrid,
    out: []f32,
) !void {
    const grid_h = @as(usize, grid.height);
    const grid_w = @as(usize, grid.width);
    const grid_t = @as(usize, grid.temporal);
    const merge = self.spatial_merge_size;
    const merged_h = grid_h / merge;
    const merged_w = grid_w / merge;

    const expected_tokens = grid_t * grid_h * grid_w;
    if (out.len != expected_tokens * self.vision_hidden_size) return error.InvalidShape;

    var dst_token: usize = 0;
    if (self.token_order == .row_major) {
        for (0..grid_t) |_| {
            for (0..grid_h) |h_idx| {
                for (0..grid_w) |w_idx| {
                    try bilinearPosRow(self, h_idx, w_idx, grid_h, grid_w, out[dst_token * self.vision_hidden_size ..][0..self.vision_hidden_size]);
                    dst_token += 1;
                }
            }
        }
        return;
    }

    for (0..grid_t) |_| {
        for (0..merged_h) |bh| {
            for (0..merged_w) |bw| {
                for (0..merge) |ih| {
                    for (0..merge) |iw| {
                        const h_idx = bh * merge + ih;
                        const w_idx = bw * merge + iw;
                        try bilinearPosRow(self, h_idx, w_idx, grid_h, grid_w, out[dst_token * self.vision_hidden_size ..][0..self.vision_hidden_size]);
                        dst_token += 1;
                    }
                }
            }
        }
    }
}

fn bilinearPosRow(
    self: *const VisionRuntime,
    h_idx: usize,
    w_idx: usize,
    grid_h: usize,
    grid_w: usize,
    out_row: []f32,
) !void {
    const side_minus_1 = @as(f32, @floatFromInt(self.num_grid_side - 1));
    const hf = if (self.token_order == .row_major) blk: {
        if (grid_h <= 1) break :blk 0.0;
        const scale = @as(f32, @floatFromInt(self.num_grid_side)) / @as(f32, @floatFromInt(grid_h));
        const mapped = (@as(f32, @floatFromInt(h_idx)) + 0.5) * scale - 0.5;
        break :blk std.math.clamp(mapped, 0.0, side_minus_1);
    } else if (grid_h <= 1)
        0.0
    else
        @as(f32, @floatFromInt(h_idx)) * side_minus_1 / @as(f32, @floatFromInt(grid_h - 1));

    const wf = if (self.token_order == .row_major) blk: {
        if (grid_w <= 1) break :blk 0.0;
        const scale = @as(f32, @floatFromInt(self.num_grid_side)) / @as(f32, @floatFromInt(grid_w));
        const mapped = (@as(f32, @floatFromInt(w_idx)) + 0.5) * scale - 0.5;
        break :blk std.math.clamp(mapped, 0.0, side_minus_1);
    } else if (grid_w <= 1)
        0.0
    else
        @as(f32, @floatFromInt(w_idx)) * side_minus_1 / @as(f32, @floatFromInt(grid_w - 1));

    const h0 = @as(usize, @intFromFloat(@floor(hf)));
    const w0 = @as(usize, @intFromFloat(@floor(wf)));
    const h1 = @min(self.num_grid_side - 1, h0 + 1);
    const w1 = @min(self.num_grid_side - 1, w0 + 1);

    const dh = hf - @as(f32, @floatFromInt(h0));
    const dw = wf - @as(f32, @floatFromInt(w0));

    const w00 = (1.0 - dh) * (1.0 - dw);
    const w01 = (1.0 - dh) * dw;
    const w10 = dh * (1.0 - dw);
    const w11 = dh * dw;

    const idx00 = (h0 * self.num_grid_side + w0) * self.vision_hidden_size;
    const idx01 = (h0 * self.num_grid_side + w1) * self.vision_hidden_size;
    const idx10 = (h1 * self.num_grid_side + w0) * self.vision_hidden_size;
    const idx11 = (h1 * self.num_grid_side + w1) * self.vision_hidden_size;

    for (0..self.vision_hidden_size) |d| {
        out_row[d] = self.pos_embed_f32[idx00 + d] * w00 +
            self.pos_embed_f32[idx01 + d] * w01 +
            self.pos_embed_f32[idx10 + d] * w10 +
            self.pos_embed_f32[idx11 + d] * w11;
    }
}

fn layerNormRow(
    input: []const f32,
    output: []f32,
    weight: []const f32,
    bias: []const f32,
    eps: f32,
) void {
    var mean: f32 = 0.0;
    for (input) |v| mean += v;
    mean /= @as(f32, @floatFromInt(input.len));

    var var_sum: f32 = 0.0;
    for (input) |v| {
        const d = v - mean;
        var_sum += d * d;
    }
    const variance = var_sum / @as(f32, @floatFromInt(input.len));
    const inv_std = 1.0 / @sqrt(variance + eps);

    for (0..input.len) |i| {
        output[i] = (input[i] - mean) * inv_std * weight[i] + bias[i];
    }
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
