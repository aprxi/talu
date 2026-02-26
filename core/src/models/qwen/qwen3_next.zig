//! Qwen3-Next model-version metadata.

const std = @import("std");
const tensor = @import("../../tensor.zig");
const layer_ops = @import("../layer_ops.zig");
const types = @import("../op_types.zig");
const config_hooks = @import("../config/hook_utils.zig");

pub const id: []const u8 = "qwen3_next";
pub const family: []const u8 = "qwen";
pub const version: []const u8 = "qwen3_next";
pub const model_types: []const []const u8 = &.{
    "qwen3_next",
};

/// Qwen3-Next full-attention branch:
/// norm -> attention -> add -> norm -> moe -> add.
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
        .debug_type = .moe,
    } },
    .{ .add = .{
        .branch = .branch_out,
        .scale = .residual_multiplier,
    } },
};

/// Qwen3-Next linear-attention branch:
/// norm -> mamba_mixer -> add -> norm -> moe -> add.
pub const mamba_program: []const layer_ops.LayerOp = &.{
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
        .debug_type = .mamba_mixer,
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
        .debug_type = .moe,
    } },
    .{ .add = .{
        .branch = .branch_out,
        .scale = .residual_multiplier,
    } },
};

// Runtime architecture payload (migrated from runtime_architectures.zig)
const qwen3_next_model_types = [_][]const u8{"qwen3_next"};
const qwen3_next_weight_prefixes = [_][]const u8{
    "model.layers.{d}.",
    "layers.{d}.",
    "transformer.h.{d}.",
    "backbone.layers.{d}.",
    "language_model.model.layers.{d}.",
};

fn requiredLayerWeight(comptime id_suffix: []const u8, comptime module_type: []const u8, comptime layout: types.WeightLayout) types.WeightSpec {
    return .{
        .id = id_suffix,
        .suffix = id_suffix,
        .module_type = module_type,
        .layout = layout,
        .dtype = "float32",
        .required = true,
    };
}

fn buildExpertWeights(comptime expert_count: usize) [expert_count * 3]types.WeightSpec {
    @setEvalBranchQuota(expert_count * 300);
    var specs: [expert_count * 3]types.WeightSpec = undefined;
    comptime var out_i: usize = 0;
    inline for (0..expert_count) |expert_idx| {
        const prefix = std.fmt.comptimePrint("mlp.experts.{d}.", .{expert_idx});
        specs[out_i] = requiredLayerWeight(prefix ++ "gate_proj.weight", "Linear", .linear);
        out_i += 1;
        specs[out_i] = requiredLayerWeight(prefix ++ "up_proj.weight", "Linear", .linear);
        out_i += 1;
        specs[out_i] = requiredLayerWeight(prefix ++ "down_proj.weight", "Linear", .linear);
        out_i += 1;
    }
    return specs;
}

fn parseConfigHook(
    config_obj: std.json.ObjectMap,
    root_obj: std.json.ObjectMap,
    config: *tensor.ModelConfig,
) void {
    config_hooks.applyCommonTextConfig(config_obj, root_obj, config);
    config_hooks.applyMambaConfig(config_obj, root_obj, config);
}

const qwen3_next_block_weights = [_]types.WeightSpec{};
const qwen3_next_layer_map = [_]u8{ 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1 };

const qwen3_next_expert_weights = buildExpertWeights(512);
const qwen3_next_shared_expert_weights = [_]types.WeightSpec{
    requiredLayerWeight("mlp.shared_expert.gate_proj.weight", "Linear", .linear),
    requiredLayerWeight("mlp.shared_expert.up_proj.weight", "Linear", .linear),
    requiredLayerWeight("mlp.shared_expert.down_proj.weight", "Linear", .linear),
    requiredLayerWeight("mlp.shared_expert_gate.weight", "Linear", .linear),
};

const qwen3_next_linear_attention_common_weights = [_]types.WeightSpec{
    requiredLayerWeight("input_layernorm.weight", "RMSNorm", .none),
    requiredLayerWeight("linear_attn.A_log", "LinearAttention", .none),
    requiredLayerWeight("linear_attn.dt_bias", "LinearAttention", .none),
    requiredLayerWeight("linear_attn.in_proj_qkvz.weight", "Linear", .linear),
    requiredLayerWeight("linear_attn.in_proj_ba.weight", "Linear", .linear),
    requiredLayerWeight("linear_attn.conv1d.weight", "Conv1d", .conv1d_depthwise),
    .{
        .id = "linear_attn.norm.weight",
        .suffix = "linear_attn.norm.weight",
        .module_type = "RMSNorm",
        .layout = .none,
        .dtype = "float32",
        .required = true,
        .force_f32 = true,
    },
    requiredLayerWeight("linear_attn.out_proj.weight", "Linear", .linear),
    requiredLayerWeight("post_attention_layernorm.weight", "RMSNorm", .none),
    requiredLayerWeight("mlp.gate.weight", "Linear", .linear),
};
const qwen3_next_linear_attention_weights = qwen3_next_linear_attention_common_weights ++ qwen3_next_expert_weights ++ qwen3_next_shared_expert_weights;

const qwen3_next_full_attention_common_weights = [_]types.WeightSpec{
    requiredLayerWeight("input_layernorm.weight", "RMSNorm", .none),
    requiredLayerWeight("self_attn.q_proj.weight", "Linear", .linear),
    requiredLayerWeight("self_attn.k_proj.weight", "Linear", .linear),
    requiredLayerWeight("self_attn.v_proj.weight", "Linear", .linear),
    requiredLayerWeight("self_attn.o_proj.weight", "Linear", .linear),
    requiredLayerWeight("self_attn.q_norm.weight", "RMSNorm", .none),
    requiredLayerWeight("self_attn.k_norm.weight", "RMSNorm", .none),
    requiredLayerWeight("post_attention_layernorm.weight", "RMSNorm", .none),
    requiredLayerWeight("mlp.gate.weight", "Linear", .linear),
};
const qwen3_next_full_attention_weights = qwen3_next_full_attention_common_weights ++ qwen3_next_expert_weights ++ qwen3_next_shared_expert_weights;

var qwen3_next_block_variants = [_]types.BlockVariant{
    .{
        .name = "linear_attention",
        .meta = .{
            .mamba_config = .{
                .d_state = 128,
                .d_conv = 4,
                .n_heads = 32,
                .d_head = 128,
                .n_groups = 16,
                .d_inner = 4096,
            },
        },
        .weights = &qwen3_next_linear_attention_weights,
    },
    .{ .name = "full_attention", .weights = &qwen3_next_full_attention_weights },
};

const qwen3_next_global_weights = [_]types.WeightSpec{
    .{ .id = "token_embeddings", .suffix = "model.embed_tokens.weight", .aliases = &.{ "embed_tokens.weight", "transformer.wte.weight", "backbone.embedding.weight", "language_model.model.embed_tokens.weight" }, .module_type = "Embedding", .layout = .embedding, .dtype = "float32", .required = true },
    .{ .id = "ln_final", .suffix = "model.norm.weight", .aliases = &.{ "norm.weight", "transformer.ln_f.weight", "backbone.norm.weight", "language_model.model.norm.weight", "model.embedding_norm.weight" }, .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "lm_head", .suffix = "lm_head.weight", .aliases = &.{ "output.weight", "transformer.lm_head.weight", "language_model.lm_head.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = false },
};

pub var arch: types.Architecture = .{
    .name = "qwen3_next",
    .model_types = &qwen3_next_model_types,
    .parse_config_hook = parseConfigHook,
    .block_variants = &qwen3_next_block_variants,
    .layer_map = &qwen3_next_layer_map,
    .variant_aliases = null,
    .block_weights = &qwen3_next_block_weights,
    .global_weights = &qwen3_next_global_weights,
    .weight_prefixes = &qwen3_next_weight_prefixes,
    .d_ff_source_weight_ids = &.{"mlp.experts.0.gate_proj.weight"},
    .resolve_d_ff_from_weights = true,
    .has_qk_norm = false,
    .has_moe = true,
    .has_mamba = true,
    .has_shortconv = false,
    .has_mla = false,
    .has_fused_qkv = false,
    .has_fused_gate_up = false,
    .num_norms_per_block = 2,
    .use_gelu = false,
    .use_swiglu_oss = false,
    .norm_weight_offset = 1.0,
    .explicit_qk_norm_ops = false,
    .embedding_multiplier = 1.0,
};
