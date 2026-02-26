//! LFM2 model-version metadata.

const std = @import("std");
const tensor = @import("../../tensor.zig");
const layer_ops = @import("../layer_ops.zig");
const types = @import("../op_types.zig");
const config_hooks = @import("../config/hook_utils.zig");
const vision_shared = @import("../vision_shared.zig");

pub const id: []const u8 = "lfm2";
pub const family: []const u8 = "lfm2";
pub const version: []const u8 = "lfm2";
pub const model_types: []const []const u8 = &.{
    "lfm2",
    "lfm2_vl",
};

pub const attention_mlp_program: []const layer_ops.LayerOp = &.{
    .{ .kernel = .{
        .id = 0,
        .in = .residual,
        .out = .norm_out,
        .debug_type = .norm,
    } },
    .{ .kernel = .{
        .id = 1,
        .in = .norm_out,
        .out = .branch_out,
        .debug_type = .multihead_attention,
    } },
    .{ .add = .{
        .branch = .branch_out,
        .scale = .residual_multiplier,
    } },
    .{ .kernel = .{
        .id = 2,
        .in = .residual,
        .out = .norm_out,
        .debug_type = .norm,
    } },
    .{ .kernel = .{
        .id = 3,
        .in = .norm_out,
        .out = .branch_out,
        .debug_type = .mlp,
    } },
    .{ .add = .{
        .branch = .branch_out,
        .scale = .residual_multiplier,
    } },
};

pub const shortconv_program: []const layer_ops.LayerOp = &.{
    .{ .kernel = .{
        .id = 0,
        .in = .residual,
        .out = .norm_out,
        .debug_type = .norm,
    } },
    .{ .kernel = .{
        .id = 1,
        .in = .norm_out,
        .out = .branch_out,
        .debug_type = .shortconv,
    } },
    .{ .add = .{
        .branch = .branch_out,
        .scale = .residual_multiplier,
    } },
    .{ .kernel = .{
        .id = 2,
        .in = .residual,
        .out = .norm_out,
        .debug_type = .norm,
    } },
    .{ .kernel = .{
        .id = 3,
        .in = .norm_out,
        .out = .branch_out,
        .debug_type = .mlp,
    } },
    .{ .add = .{
        .branch = .branch_out,
        .scale = .residual_multiplier,
    } },
};

fn parseConfigHook(
    config_obj: std.json.ObjectMap,
    root_obj: std.json.ObjectMap,
    config: *tensor.ModelConfig,
) void {
    config_hooks.applyCommonTextConfig(config_obj, root_obj, config);
    config_hooks.applyShortConvConfig(config_obj, root_obj, config);
    config_hooks.applyVisionConfig(root_obj, config);
}

// Runtime architecture payload (migrated from runtime_architectures.zig)
const lfm2_model_types = [_][]const u8{ "lfm2", "lfm2_vl" };
const lfm2_weight_prefixes = [_][]const u8{ "model.layers.{d}.", "model.language_model.layers.{d}.", "layers.{d}.", "transformer.h.{d}.", "backbone.layers.{d}.", "language_model.model.layers.{d}." };
const lfm2_block_weights = [_]types.WeightSpec{};
const lfm2_layer_map = [_]u8{ 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0 };
const lfm2_variant_aliases = [_]types.VariantAlias{
    .{ .alias = "conv", .variant_index = 0 },
    .{ .alias = "full_attention", .variant_index = 1 },
};
const lfm2_shortconv_weights = [_]types.WeightSpec{
    .{ .id = "operator_norm.weight", .suffix = "operator_norm.weight", .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "conv.in_proj.weight", .suffix = "conv.in_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "conv.conv.weight", .suffix = "conv.conv.weight", .module_type = "Conv1d", .layout = .conv1d_depthwise, .dtype = "float32", .required = true },
    .{ .id = "conv.out_proj.weight", .suffix = "conv.out_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "ffn_norm.weight", .suffix = "ffn_norm.weight", .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "feed_forward.w1.weight", .suffix = "feed_forward.w1.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "feed_forward.w2.weight", .suffix = "feed_forward.w2.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "feed_forward.w3.weight", .suffix = "feed_forward.w3.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
};
const lfm2_attention_weights = [_]types.WeightSpec{
    .{ .id = "operator_norm.weight", .suffix = "operator_norm.weight", .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "self_attn.q_proj.weight", .suffix = "self_attn.q_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.k_proj.weight", .suffix = "self_attn.k_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.v_proj.weight", .suffix = "self_attn.v_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.out_proj.weight", .suffix = "self_attn.out_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.q_layernorm.weight", .suffix = "self_attn.q_layernorm.weight", .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "self_attn.k_layernorm.weight", .suffix = "self_attn.k_layernorm.weight", .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "ffn_norm.weight", .suffix = "ffn_norm.weight", .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "feed_forward.w1.weight", .suffix = "feed_forward.w1.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "feed_forward.w2.weight", .suffix = "feed_forward.w2.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "feed_forward.w3.weight", .suffix = "feed_forward.w3.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
};
var lfm2_block_variants = [_]types.BlockVariant{
    .{
        .name = "shortconv",
        .meta = .{
            .shortconv_config = .{
                .d_conv = 3,
                .conv_dim = 1024,
                .conv_dim_out = 1024,
                .has_bias = false,
            },
        },
        .weights = &lfm2_shortconv_weights,
    },
    .{ .name = "attention", .weights = &lfm2_attention_weights },
};
const lfm2_global_weights = [_]types.WeightSpec{
    .{ .id = "token_embeddings", .suffix = "model.embed_tokens.weight", .aliases = &.{ "model.language_model.embed_tokens.weight", "embed_tokens.weight", "transformer.wte.weight", "backbone.embedding.weight", "language_model.model.embed_tokens.weight" }, .module_type = "Embedding", .layout = .embedding, .dtype = "float32", .required = true },
    .{ .id = "ln_final", .suffix = "model.norm.weight", .aliases = &.{ "model.language_model.embedding_norm.weight", "model.language_model.norm.weight", "norm.weight", "transformer.ln_f.weight", "backbone.norm.weight", "language_model.model.norm.weight", "model.embedding_norm.weight" }, .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "lm_head", .suffix = "lm_head.weight", .aliases = &.{ "model.language_model.lm_head.weight", "output.weight", "transformer.lm_head.weight", "language_model.lm_head.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = false },
};
pub var arch: types.Architecture = .{
    .name = "lfm2",
    .model_types = &lfm2_model_types,
    .parse_config_hook = parseConfigHook,
    .block_variants = &lfm2_block_variants,
    .layer_map = &lfm2_layer_map,
    .variant_aliases = &lfm2_variant_aliases,
    .block_weights = &lfm2_block_weights,
    .global_weights = &lfm2_global_weights,
    .weight_prefixes = &lfm2_weight_prefixes,
    .d_ff_source_weight_ids = &.{"feed_forward.w1.weight"},
    .resolve_d_ff_from_weights = true,
    .shortconv_dims_source_weight_id = "conv.conv.weight",
    .resolve_shortconv_dims_from_weights = true,
    .has_qk_norm = false,
    .has_moe = false,
    .has_mamba = false,
    .has_shortconv = true,
    .has_mla = false,
    .has_fused_qkv = false,
    .has_fused_gate_up = false,
    .num_norms_per_block = 2,
    .use_gelu = false,
    .use_swiglu_oss = false,
    .norm_weight_offset = 0.0,
    .explicit_qk_norm_ops = false,
    .embedding_multiplier = 1.0,
    .vision = vision_shared.metadata,
};
