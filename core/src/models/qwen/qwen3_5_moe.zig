//! Qwen3.5-MoE model-version metadata.
//!
//! Combines Qwen3.5 heterogeneous blocks (gated_delta + full_attention, 3:1 ratio)
//! with Mixture-of-Experts in the FFN path (256 routed experts + shared expert).

const std = @import("std");
const tensor = @import("../../tensor.zig");
const layer_ops = @import("../layer_ops.zig");
const types = @import("../op_types.zig");
const config_hooks = @import("../config/hook_utils.zig");
const perf = @import("../perf_hints.zig");
const sp = @import("../sampling_presets.zig");

pub const id: []const u8 = "qwen3_5_moe";
pub const family: []const u8 = "qwen";
pub const version: []const u8 = "qwen3_5_moe";
pub const model_types: []const []const u8 = &.{
    "qwen3_5_moe",
    "qwen3_5_moe_text",
};

/// Full-attention + MoE branch:
/// norm -> attention -> add -> norm -> moe -> add.
pub const attention_moe_program: []const layer_ops.LayerOp = &.{
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
        .state_block_id = types.kv_cache_state_id,
        .attention_config = .{
            .query_gate = true,
        },
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

/// Linear-attention (gated delta) + MoE branch:
/// norm -> gated_delta_net -> add -> norm -> moe -> add.
pub const gated_delta_moe_program: []const layer_ops.LayerOp = &.{
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
        .debug_type = .gated_delta_net,
        .state_block_id = types.gated_delta_state_id,
        .gated_delta_config = .{
            .d_conv = 4,
            .n_heads = 16,
            .d_head = 128,
            .d_inner = 2048,
        },
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

fn optionalLayerWeight(comptime id_suffix: []const u8, comptime module_type: []const u8, comptime layout: types.WeightLayout) types.WeightSpec {
    return .{
        .id = id_suffix,
        .suffix = id_suffix,
        .module_type = module_type,
        .layout = layout,
        .dtype = "float32",
        .required = false,
    };
}

fn parseConfigHook(
    config_obj: std.json.ObjectMap,
    root_obj: std.json.ObjectMap,
    config: *tensor.ModelConfig,
) void {
    config_hooks.applyCommonTextConfig(config_obj, root_obj, config);
    config_hooks.applyLinearAttentionConfig(config_obj, root_obj, config);
    config_hooks.applyMambaConfig(config_obj, root_obj, config);
}

const qwen3_5_moe_model_types = [_][]const u8{ "qwen3_5_moe", "qwen3_5_moe_text" };
const qwen3_5_moe_weight_prefixes = [_][]const u8{
    "model.language_model.layers.{d}.",
    "model.layers.{d}.",
    "layers.{d}.",
    "transformer.h.{d}.",
    "backbone.layers.{d}.",
    "language_model.model.layers.{d}.",
};

const qwen3_5_moe_block_weights = [_]types.WeightSpec{};

// Layer map: 3:1 ratio (linear_attention=0, full_attention=1), 64 entries covers up to 64 layers.
// config.json layer_types overrides this at runtime via parseLayerTypes.
const qwen3_5_moe_layer_map = [_]u8{
    0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
    0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
    0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
    0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
};

const qwen3_5_moe_expert_weights = [_]types.WeightSpec{
    requiredLayerWeight("mlp.experts.gate_up_proj", "Parameter", .fused_linear),
    requiredLayerWeight("mlp.experts.down_proj", "Parameter", .fused_linear),
};
const qwen3_5_moe_shared_expert_weights = [_]types.WeightSpec{
    requiredLayerWeight("mlp.shared_expert.gate_proj.weight", "Linear", .linear),
    requiredLayerWeight("mlp.shared_expert.up_proj.weight", "Linear", .linear),
    requiredLayerWeight("mlp.shared_expert.down_proj.weight", "Linear", .linear),
    requiredLayerWeight("mlp.shared_expert_gate.weight", "Parameter", .none),
};

const qwen3_5_moe_moe_weights = [_]types.WeightSpec{
    requiredLayerWeight("mlp.gate.weight", "Linear", .linear),
} ++ qwen3_5_moe_expert_weights ++ qwen3_5_moe_shared_expert_weights;

// Linear attention variant: gated_delta + MoE
const qwen3_5_moe_linear_attention_common_weights = [_]types.WeightSpec{
    requiredLayerWeight("input_layernorm.weight", "RMSNorm", .none),
    optionalLayerWeight("mixer.in_proj.weight", "Linear", .linear),
    optionalLayerWeight("linear_attn.in_proj_qkv.weight", "Linear", .linear),
    optionalLayerWeight("linear_attn.in_proj_z.weight", "Linear", .linear),
    optionalLayerWeight("linear_attn.in_proj_a.weight", "Linear", .linear),
    optionalLayerWeight("linear_attn.in_proj_b.weight", "Linear", .linear),
    requiredLayerWeight("linear_attn.conv1d.weight", "Conv1d", .conv1d_depthwise),
    .{
        .id = "linear_attn.A_log",
        .suffix = "linear_attn.A_log",
        .module_type = "LinearAttention",
        .layout = .none,
        .dtype = "float32",
        .required = true,
        .force_f32 = true,
    },
    requiredLayerWeight("linear_attn.dt_bias", "LinearAttention", .none),
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
};
const qwen3_5_moe_linear_attention_weights = qwen3_5_moe_linear_attention_common_weights ++ qwen3_5_moe_moe_weights;

// Full attention variant: attention + MoE
const qwen3_5_moe_full_attention_common_weights = [_]types.WeightSpec{
    requiredLayerWeight("input_layernorm.weight", "RMSNorm", .none),
    requiredLayerWeight("self_attn.q_proj.weight", "Linear", .linear),
    requiredLayerWeight("self_attn.k_proj.weight", "Linear", .linear),
    requiredLayerWeight("self_attn.v_proj.weight", "Linear", .linear),
    requiredLayerWeight("self_attn.o_proj.weight", "Linear", .linear),
    requiredLayerWeight("self_attn.q_norm.weight", "RMSNorm", .none),
    requiredLayerWeight("self_attn.k_norm.weight", "RMSNorm", .none),
    requiredLayerWeight("post_attention_layernorm.weight", "RMSNorm", .none),
};
const qwen3_5_moe_full_attention_weights = qwen3_5_moe_full_attention_common_weights ++ qwen3_5_moe_moe_weights;

var qwen3_5_moe_block_variants = [_]types.BlockVariant{
    .{
        .name = "linear_attention",
        .meta = .{
            .gated_delta_config = .{
                .d_conv = 4,
                .n_heads = 16,
                .d_head = 128,
                .d_inner = 2048,
            },
        },
        .weights = &qwen3_5_moe_linear_attention_weights,
    },
    .{
        .name = "full_attention",
        .meta = .{
            .attention_config = .{
                .query_gate = true,
            },
        },
        .weights = &qwen3_5_moe_full_attention_weights,
    },
};

const qwen3_5_moe_conversion_fusions = [_]types.ConversionFusion{
    .{
        .kind = .gated_delta_split_in_proj,
        .trigger_suffix = "linear_attn.in_proj_qkv.weight",
        .required_input_suffixes = &.{
            "linear_attn.in_proj_qkv.weight",
            "linear_attn.in_proj_z.weight",
            "linear_attn.in_proj_b.weight",
            "linear_attn.in_proj_a.weight",
        },
        .output_suffix = "mixer.in_proj.weight",
    },
};

const qwen3_5_moe_global_weights = [_]types.WeightSpec{
    .{
        .id = "token_embeddings",
        .suffix = "model.language_model.embed_tokens.weight",
        .aliases = &.{
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "transformer.wte.weight",
            "backbone.embedding.weight",
            "language_model.model.embed_tokens.weight",
        },
        .module_type = "Embedding",
        .layout = .embedding,
        .dtype = "float32",
        .required = true,
    },
    .{
        .id = "ln_final",
        .suffix = "model.language_model.norm.weight",
        .aliases = &.{
            "model.norm.weight",
            "norm.weight",
            "transformer.ln_f.weight",
            "backbone.norm.weight",
            "language_model.model.norm.weight",
            "model.embedding_norm.weight",
        },
        .module_type = "RMSNorm",
        .layout = .none,
        .dtype = "float32",
        .required = true,
    },
    .{
        .id = "lm_head",
        .suffix = "lm_head.weight",
        .aliases = &.{
            "model.language_model.lm_head.weight",
            "output.weight",
            "transformer.lm_head.weight",
            "language_model.lm_head.weight",
        },
        .module_type = "Linear",
        .layout = .linear,
        .dtype = "float32",
        .required = false,
    },
};

const qwen3_5_moe_perf_hints = perf.attentionOnlyHints("qwen3_5_moe");
const qwen3_5_moe_sampling_presets: sp.SamplingPresets = .{
    .general = .{ .temperature = 1.0, .top_p = 0.95, .top_k = 20, .presence_penalty = 1.5 },
    .coding = .{ .temperature = 0.6, .top_p = 0.95, .top_k = 20, .presence_penalty = 0.0 },
    .instruct = .{ .temperature = 0.7, .top_p = 0.8, .top_k = 20, .presence_penalty = 1.5 },
    .deterministic = .{ .temperature = 0.0, .top_p = 1.0, .top_k = 1, .presence_penalty = 0.0 },
};

pub var arch: types.Architecture = .{
    .name = "qwen3_5_moe",
    .model_types = &qwen3_5_moe_model_types,
    .performance_hints = &qwen3_5_moe_perf_hints,
    .sampling_presets = &qwen3_5_moe_sampling_presets,
    .parse_config_hook = parseConfigHook,
    .block_variants = &qwen3_5_moe_block_variants,
    .layer_map = &qwen3_5_moe_layer_map,
    .variant_aliases = null,
    .block_weights = &qwen3_5_moe_block_weights,
    .global_weights = &qwen3_5_moe_global_weights,
    .weight_prefixes = &qwen3_5_moe_weight_prefixes,
    .conversion_fusions = &qwen3_5_moe_conversion_fusions,
    .d_ff_source_weight_ids = &.{"mlp.shared_expert.gate_proj.weight"},
    .resolve_d_ff_from_weights = false,
    .has_qk_norm = true,
    .has_moe = true,
    .has_mamba = false,
    .has_gated_delta = true,
    .has_shortconv = false,
    .has_mla = false,
    .has_fused_qkv = false,
    .has_fused_gate_up = false,
    .enable_loader_fusions = false,
    .num_norms_per_block = 2,
    .use_gelu = false,
    .use_swiglu_oss = false,
    .norm_weight_offset = 1.0,
    .explicit_qk_norm_ops = false,
    .embedding_multiplier = 1.0,
};
