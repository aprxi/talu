//! Shared multimodal vision metadata and pipeline descriptors.
//!
//! This module owns cross-family vision contracts used by YouTu-VL, Qwen3-VL,
//! and LFM2-VL so families depend on a neutral shared definition.

const std = @import("std");
const layer_ops = @import("layer_ops.zig");
const types = @import("op_types.zig");

/// Static vision pipeline topology used by multimodal prefill.
/// Buffer ids are stage placeholders for backend-specific vision runtimes.
pub const vision_program: []const layer_ops.LayerOp = &.{
    .{ .patch_embed = .{
        .in = .residual,
        .out = .norm_out,
    } },
    .{ .spatial_merge = .{
        .in = .norm_out,
        .out = .branch_out,
        .merge_size = 2,
    } },
    .{ .scatter = .{
        .text_in = .residual,
        .vision_in = .branch_out,
        .out = .residual,
        .image_token_id = 0,
    } },
};

/// Vision tensor probing metadata shared by multimodal families.
pub const metadata: types.VisionMetadata = .{
    // Probe tensors used to classify fused/split attention layouts.
    .fused_qkv_probe_candidates = &.{
        "model.visual.blocks.0.attn.qkv.weight",
    },
    .split_qkv_probe_candidates = &.{
        "model.visual.blocks.0.attn.q_proj.weight",
        "model.visual.blocks.0.self_attn.q_proj.weight",
        "model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight",
        "model.vision_model.encoder.layers.0.self_attn.q_proj.weight",
    },

    // Candidate tensors used for config hydration.
    .patch_embed_candidates = &.{
        "model.visual.patch_embed.proj.weight",
        "model.vision_tower.vision_model.embeddings.patch_embedding.weight",
        "model.vision_model.embeddings.patch_embedding.weight",
    },
    .patch_embed_bias_candidates = &.{
        "model.visual.patch_embed.proj.bias",
        "model.vision_tower.vision_model.embeddings.patch_embedding.bias",
        "model.vision_model.embeddings.patch_embedding.bias",
    },
    .position_embed_candidates = &.{
        "model.visual.pos_embed.weight",
        "model.vision_tower.vision_model.embeddings.position_embedding.weight",
        "model.vision_model.embeddings.position_embedding.weight",
    },
    .post_norm_weight_candidates = &.{
        "model.visual.norm.weight",
        "model.vision_tower.vision_model.post_layernorm.weight",
        "model.vision_model.post_layernorm.weight",
    },
    .post_norm_bias_candidates = &.{
        "model.visual.norm.bias",
        "model.vision_tower.vision_model.post_layernorm.bias",
        "model.vision_model.post_layernorm.bias",
    },
    .merger_norm_weight_candidates = &.{
        "model.visual.merger.norm.weight",
        "model.multi_modal_projector.layer_norm.weight",
    },
    .merger_norm_bias_candidates = &.{
        "model.visual.merger.norm.bias",
        "model.multi_modal_projector.layer_norm.bias",
    },
    .merger_fc1_candidates = &.{
        "model.visual.merger.linear_fc1.weight",
        "model.multi_modal_projector.linear_1.weight",
    },
    .merger_fc1_bias_candidates = &.{
        "model.visual.merger.linear_fc1.bias",
        "model.multi_modal_projector.linear_1.bias",
    },
    .merger_fc2_candidates = &.{
        "model.visual.merger.linear_fc2.weight",
        "model.multi_modal_projector.linear_2.weight",
    },
    .merger_fc2_bias_candidates = &.{
        "model.visual.merger.linear_fc2.bias",
        "model.multi_modal_projector.linear_2.bias",
    },

    // Layer templates for per-block vision weights.
    .ln1_weight_templates = &.{
        "model.visual.blocks.{d}.norm1.weight",
        "model.vision_tower.vision_model.encoder.layers.{d}.layer_norm1.weight",
        "model.vision_model.encoder.layers.{d}.layer_norm1.weight",
    },
    .ln1_bias_templates = &.{
        "model.visual.blocks.{d}.norm1.bias",
        "model.vision_tower.vision_model.encoder.layers.{d}.layer_norm1.bias",
        "model.vision_model.encoder.layers.{d}.layer_norm1.bias",
    },
    .ln2_weight_templates = &.{
        "model.visual.blocks.{d}.norm2.weight",
        "model.vision_tower.vision_model.encoder.layers.{d}.layer_norm2.weight",
        "model.vision_model.encoder.layers.{d}.layer_norm2.weight",
    },
    .ln2_bias_templates = &.{
        "model.visual.blocks.{d}.norm2.bias",
        "model.vision_tower.vision_model.encoder.layers.{d}.layer_norm2.bias",
        "model.vision_model.encoder.layers.{d}.layer_norm2.bias",
    },
    .fused_qkv_weight_templates = &.{
        "model.visual.blocks.{d}.attn.qkv.weight",
    },
    .fused_qkv_bias_templates = &.{
        "model.visual.blocks.{d}.attn.qkv.bias",
    },
    .split_q_weight_templates = &.{
        "model.visual.blocks.{d}.attn.q_proj.weight",
        "model.visual.blocks.{d}.self_attn.q_proj.weight",
        "model.vision_tower.vision_model.encoder.layers.{d}.self_attn.q_proj.weight",
        "model.vision_model.encoder.layers.{d}.self_attn.q_proj.weight",
    },
    .split_q_bias_templates = &.{
        "model.visual.blocks.{d}.attn.q_proj.bias",
        "model.visual.blocks.{d}.self_attn.q_proj.bias",
        "model.vision_tower.vision_model.encoder.layers.{d}.self_attn.q_proj.bias",
        "model.vision_model.encoder.layers.{d}.self_attn.q_proj.bias",
    },
    .split_k_weight_templates = &.{
        "model.visual.blocks.{d}.attn.k_proj.weight",
        "model.visual.blocks.{d}.self_attn.k_proj.weight",
        "model.vision_tower.vision_model.encoder.layers.{d}.self_attn.k_proj.weight",
        "model.vision_model.encoder.layers.{d}.self_attn.k_proj.weight",
    },
    .split_k_bias_templates = &.{
        "model.visual.blocks.{d}.attn.k_proj.bias",
        "model.visual.blocks.{d}.self_attn.k_proj.bias",
        "model.vision_tower.vision_model.encoder.layers.{d}.self_attn.k_proj.bias",
        "model.vision_model.encoder.layers.{d}.self_attn.k_proj.bias",
    },
    .split_v_weight_templates = &.{
        "model.visual.blocks.{d}.attn.v_proj.weight",
        "model.visual.blocks.{d}.self_attn.v_proj.weight",
        "model.vision_tower.vision_model.encoder.layers.{d}.self_attn.v_proj.weight",
        "model.vision_model.encoder.layers.{d}.self_attn.v_proj.weight",
    },
    .split_v_bias_templates = &.{
        "model.visual.blocks.{d}.attn.v_proj.bias",
        "model.visual.blocks.{d}.self_attn.v_proj.bias",
        "model.vision_tower.vision_model.encoder.layers.{d}.self_attn.v_proj.bias",
        "model.vision_model.encoder.layers.{d}.self_attn.v_proj.bias",
    },
    .out_proj_weight_templates = &.{
        "model.visual.blocks.{d}.attn.proj.weight",
        "model.vision_tower.vision_model.encoder.layers.{d}.self_attn.out_proj.weight",
        "model.vision_model.encoder.layers.{d}.self_attn.out_proj.weight",
    },
    .out_proj_bias_templates = &.{
        "model.visual.blocks.{d}.attn.proj.bias",
        "model.vision_tower.vision_model.encoder.layers.{d}.self_attn.out_proj.bias",
        "model.vision_model.encoder.layers.{d}.self_attn.out_proj.bias",
    },
    .fc1_weight_templates = &.{
        "model.visual.blocks.{d}.mlp.linear_fc1.weight",
        "model.vision_tower.vision_model.encoder.layers.{d}.mlp.fc1.weight",
        "model.vision_model.encoder.layers.{d}.mlp.fc1.weight",
    },
    .fc1_bias_templates = &.{
        "model.visual.blocks.{d}.mlp.linear_fc1.bias",
        "model.vision_tower.vision_model.encoder.layers.{d}.mlp.fc1.bias",
        "model.vision_model.encoder.layers.{d}.mlp.fc1.bias",
    },
    .fc2_weight_templates = &.{
        "model.visual.blocks.{d}.mlp.linear_fc2.weight",
        "model.vision_tower.vision_model.encoder.layers.{d}.mlp.fc2.weight",
        "model.vision_model.encoder.layers.{d}.mlp.fc2.weight",
    },
    .fc2_bias_templates = &.{
        "model.visual.blocks.{d}.mlp.linear_fc2.bias",
        "model.vision_tower.vision_model.encoder.layers.{d}.mlp.fc2.bias",
        "model.vision_model.encoder.layers.{d}.mlp.fc2.bias",
    },

    // Deepstack merger templates (if present).
    .deepstack_norm_weight_templates = &.{
        "model.visual.deepstack_merger_list.{d}.norm.weight",
    },
    .deepstack_norm_bias_templates = &.{
        "model.visual.deepstack_merger_list.{d}.norm.bias",
    },
    .deepstack_fc1_weight_templates = &.{
        "model.visual.deepstack_merger_list.{d}.linear_fc1.weight",
    },
    .deepstack_fc1_bias_templates = &.{
        "model.visual.deepstack_merger_list.{d}.linear_fc1.bias",
    },
    .deepstack_fc2_weight_templates = &.{
        "model.visual.deepstack_merger_list.{d}.linear_fc2.weight",
    },
    .deepstack_fc2_bias_templates = &.{
        "model.visual.deepstack_merger_list.{d}.linear_fc2.bias",
    },

    // Layer templates used to infer depth/intermediate size.
    .depth_split_qproj_templates = &.{
        "model.visual.blocks.{d}.attn.q_proj.weight",
        "model.visual.blocks.{d}.self_attn.q_proj.weight",
        "model.vision_tower.vision_model.encoder.layers.{d}.self_attn.q_proj.weight",
        "model.vision_model.encoder.layers.{d}.self_attn.q_proj.weight",
    },
    .depth_fused_qkv_templates = &.{
        "model.visual.blocks.{d}.attn.qkv.weight",
    },
    .intermediate_fc1_templates = &.{
        "model.visual.blocks.0.mlp.linear_fc1.weight",
        "model.vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight",
        "model.vision_model.encoder.layers.0.mlp.fc1.weight",
    },
};

test "vision_program preserves patch-merge-scatter topology" {
    try std.testing.expectEqual(@as(usize, 3), vision_program.len);
}

test "metadata includes fused and split probe candidates" {
    try std.testing.expect(metadata.fused_qkv_probe_candidates.len > 0);
    try std.testing.expect(metadata.split_qkv_probe_candidates.len > 0);
}
