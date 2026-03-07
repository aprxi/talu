//! Qwen3.5 model-version metadata.

const std = @import("std");
const tensor = @import("../../tensor.zig");
const layer_ops = @import("../layer_ops.zig");
const perf = @import("../perf_hints.zig");
const types = @import("../op_types.zig");
const config_hooks = @import("../config/hook_utils.zig");
const vision_shared = @import("../vision_shared.zig");

pub const id: []const u8 = "qwen3_5";
pub const family: []const u8 = "qwen";
pub const version: []const u8 = "qwen3_5";
pub const model_types: []const []const u8 = &.{
    "qwen3_5",
    "qwen3.5",
    "qwen3_5_text",
};

/// Qwen3.5 full-attention branch:
/// norm -> attention -> add -> norm -> ffn -> add.
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
        .state_block_id = types.kv_cache_state_id,
        .attention_config = .{
            .rope_interleaved = false,
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
        .debug_type = .mlp,
    } },
    .{ .add = .{
        .branch = .branch_out,
        .scale = .residual_multiplier,
    } },
};

/// Qwen3.5 linear-attention branch:
/// norm -> gated_delta_net -> add -> norm -> ffn -> add.
pub const gated_delta_program: []const layer_ops.LayerOp = &.{
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
    config_hooks.applyMambaConfig(config_obj, root_obj, config);
    config_hooks.applyVisionConfig(root_obj, config);
}

const qwen3_5_model_types = [_][]const u8{ "qwen3_5", "qwen3.5", "qwen3_5_text" };
const qwen3_5_weight_prefixes = [_][]const u8{
    "model.language_model.layers.{d}.",
    "model.layers.{d}.",
    "layers.{d}.",
    "transformer.h.{d}.",
    "backbone.layers.{d}.",
    "language_model.model.layers.{d}.",
};

fn requiredLayerWeight(comptime weight_id: []const u8, comptime suffix: []const u8, comptime module_type: []const u8, comptime layout: types.WeightLayout) types.WeightSpec {
    return .{
        .id = weight_id,
        .suffix = suffix,
        .module_type = module_type,
        .layout = layout,
        .dtype = "float32",
        .required = true,
    };
}

fn optionalLayerWeight(comptime weight_id: []const u8, comptime suffix: []const u8, comptime module_type: []const u8, comptime layout: types.WeightLayout) types.WeightSpec {
    return .{
        .id = weight_id,
        .suffix = suffix,
        .module_type = module_type,
        .layout = layout,
        .dtype = "float32",
        .required = false,
    };
}

const qwen3_5_block_weights = [_]types.WeightSpec{};
const qwen3_5_layer_map = [_]u8{
    0, 0, 0, 1,
    0, 0, 0, 1,
    0, 0, 0, 1,
    0, 0, 0, 1,
    0, 0, 0, 1,
    0, 0, 0, 1,
};

const qwen3_5_linear_attention_weights = [_]types.WeightSpec{
    requiredLayerWeight("input_layernorm.weight", "input_layernorm.weight", "RMSNorm", .none),
    optionalLayerWeight("mixer.in_proj.weight", "mixer.in_proj.weight", "Linear", .linear),
    optionalLayerWeight("linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_qkv.weight", "Linear", .linear),
    optionalLayerWeight("linear_attn.in_proj_z.weight", "linear_attn.in_proj_z.weight", "Linear", .linear),
    optionalLayerWeight("linear_attn.in_proj_a.weight", "linear_attn.in_proj_a.weight", "Linear", .linear),
    optionalLayerWeight("linear_attn.in_proj_b.weight", "linear_attn.in_proj_b.weight", "Linear", .linear),
    requiredLayerWeight("linear_attn.conv1d.weight", "linear_attn.conv1d.weight", "Conv1d", .conv1d_depthwise),
    .{
        .id = "linear_attn.A_log",
        .suffix = "linear_attn.A_log",
        .module_type = "LinearAttention",
        .layout = .none,
        .dtype = "float32",
        .required = true,
        .force_f32 = true,
    },
    requiredLayerWeight("linear_attn.dt_bias", "linear_attn.dt_bias", "LinearAttention", .none),
    .{
        .id = "linear_attn.norm.weight",
        .suffix = "linear_attn.norm.weight",
        .module_type = "RMSNorm",
        .layout = .none,
        .dtype = "float32",
        .required = true,
        .force_f32 = true,
    },
    requiredLayerWeight("linear_attn.out_proj.weight", "linear_attn.out_proj.weight", "Linear", .linear),
    requiredLayerWeight("post_attention_layernorm.weight", "post_attention_layernorm.weight", "RMSNorm", .none),
    requiredLayerWeight("mlp.gate_proj.weight", "mlp.gate_proj.weight", "Linear", .linear),
    requiredLayerWeight("mlp.up_proj.weight", "mlp.up_proj.weight", "Linear", .linear),
    requiredLayerWeight("mlp.down_proj.weight", "mlp.down_proj.weight", "Linear", .linear),
};

const qwen3_5_full_attention_weights = [_]types.WeightSpec{
    requiredLayerWeight("input_layernorm.weight", "input_layernorm.weight", "RMSNorm", .none),
    requiredLayerWeight("self_attn.q_proj.weight", "self_attn.q_proj.weight", "Linear", .linear),
    requiredLayerWeight("self_attn.k_proj.weight", "self_attn.k_proj.weight", "Linear", .linear),
    requiredLayerWeight("self_attn.v_proj.weight", "self_attn.v_proj.weight", "Linear", .linear),
    requiredLayerWeight("self_attn.o_proj.weight", "self_attn.o_proj.weight", "Linear", .linear),
    requiredLayerWeight("self_attn.q_norm.weight", "self_attn.q_norm.weight", "RMSNorm", .none),
    requiredLayerWeight("self_attn.k_norm.weight", "self_attn.k_norm.weight", "RMSNorm", .none),
    requiredLayerWeight("post_attention_layernorm.weight", "post_attention_layernorm.weight", "RMSNorm", .none),
    requiredLayerWeight("mlp.gate_proj.weight", "mlp.gate_proj.weight", "Linear", .linear),
    requiredLayerWeight("mlp.up_proj.weight", "mlp.up_proj.weight", "Linear", .linear),
    requiredLayerWeight("mlp.down_proj.weight", "mlp.down_proj.weight", "Linear", .linear),
};

var qwen3_5_block_variants = [_]types.BlockVariant{
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
        .weights = &qwen3_5_linear_attention_weights,
    },
    .{
        .name = "full_attention",
        .meta = .{
            .attention_config = .{
                .rope_interleaved = false,
                .query_gate = true,
            },
        },
        .weights = &qwen3_5_full_attention_weights,
    },
};

const qwen3_5_global_weights = [_]types.WeightSpec{
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

const qwen3_5_conversion_fusions = [_]types.ConversionFusion{
    .{
        .kind = .gated_delta_split_in_proj,
        .trigger_suffix = "linear_attn.in_proj_qkv.weight",
        .required_input_suffixes = &.{
            "linear_attn.in_proj_qkv.weight",
            "linear_attn.in_proj_z.weight",
            "linear_attn.in_proj_b.weight",
        },
        .optional_input_suffixes = &.{
            "linear_attn.in_proj_a.weight",
        },
        .output_suffix = "mixer.in_proj.weight",
    },
};

const qwen3_5_perf_hints = perf.PerfHints{
    .bench_model = "qwen3_5",
    .point_mappings = perf.standard_attention_mlp_point_mappings[0..],
    .hidden_rows = perf.qwen3_5_hidden_rows[0..],
    .role_dims = perf.qwen3_5_role_dims[0..],
};

pub var arch: types.Architecture = .{
    .name = "qwen3_5",
    .model_types = &qwen3_5_model_types,
    .performance_hints = &qwen3_5_perf_hints,
    .parse_config_hook = parseConfigHook,
    .block_variants = &qwen3_5_block_variants,
    .layer_map = &qwen3_5_layer_map,
    .variant_aliases = null,
    .block_weights = &qwen3_5_block_weights,
    .global_weights = &qwen3_5_global_weights,
    .weight_prefixes = &qwen3_5_weight_prefixes,
    .conversion_fusions = &qwen3_5_conversion_fusions,
    .d_ff_source_weight_ids = &.{"mlp.gate_proj.weight"},
    .resolve_d_ff_from_weights = true,
    .has_qk_norm = true,
    .has_moe = false,
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
    .vision = vision_shared.metadata,
};
