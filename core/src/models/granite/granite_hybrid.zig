//! Granite Hybrid model-version metadata.

const std = @import("std");
const tensor = @import("../../tensor.zig");
const layer_ops = @import("../layer_ops.zig");
const types = @import("../op_types.zig");
const config_hooks = @import("../config/hook_utils.zig");

pub const id: []const u8 = "granite_hybrid";
pub const family: []const u8 = "granite";
pub const version: []const u8 = "granite_hybrid";
pub const model_types: []const []const u8 = &.{
    "granite_hybrid",
    "granitehybrid",
    "granitemoehybrid",
};

/// Attention branch topology (hybrid attention blocks).
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

/// Mamba branch topology (hybrid mamba blocks).
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
}

// Runtime architecture payload (migrated from runtime_architectures.zig)
const granite_hybrid_model_types = [_][]const u8{ "granite_hybrid", "granitehybrid", "granitemoehybrid" };
const granite_hybrid_weight_prefixes = [_][]const u8{ "model.layers.{d}.", "layers.{d}.", "transformer.h.{d}.", "backbone.layers.{d}.", "language_model.model.layers.{d}." };
const granite_hybrid_block_weights = [_]types.WeightSpec{};
const granite_hybrid_layer_map = [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 };
const granite_hybrid_mamba_weights = [_]types.WeightSpec{
    .{ .id = "input_layernorm.weight", .suffix = "input_layernorm.weight", .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "mixer.A_log", .suffix = "mixer.A_log", .module_type = "Mamba2", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "mixer.D", .suffix = "mixer.D", .module_type = "Mamba2", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "mixer.dt_bias", .suffix = "mixer.dt_bias", .module_type = "Mamba2", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "mixer.in_proj.weight", .suffix = "mixer.in_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "mixer.conv1d.weight", .suffix = "mixer.conv1d.weight", .module_type = "Conv1d", .layout = .conv1d_depthwise, .dtype = "float32", .required = true },
    .{ .id = "mixer.conv1d.bias", .suffix = "mixer.conv1d.bias", .module_type = "Conv1d", .layout = .conv1d_depthwise, .dtype = "float32", .required = true },
    .{ .id = "mixer.norm.weight", .suffix = "mixer.norm.weight", .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true, .force_f32 = true },
    .{ .id = "mixer.out_proj.weight", .suffix = "mixer.out_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "post_attention_layernorm.weight", .suffix = "post_attention_layernorm.weight", .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "mlp.input_linear.weight", .suffix = "mlp.input_linear.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "mlp.output_linear.weight", .suffix = "mlp.output_linear.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
};
const granite_hybrid_attention_weights = [_]types.WeightSpec{
    .{ .id = "input_layernorm.weight", .suffix = "input_layernorm.weight", .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "mixer.q_proj.weight", .suffix = "mixer.q_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "mixer.k_proj.weight", .suffix = "mixer.k_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "mixer.v_proj.weight", .suffix = "mixer.v_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "mixer.o_proj.weight", .suffix = "mixer.o_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "post_attention_layernorm.weight", .suffix = "post_attention_layernorm.weight", .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "mlp.input_linear.weight", .suffix = "mlp.input_linear.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "mlp.output_linear.weight", .suffix = "mlp.output_linear.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
};
var granite_hybrid_block_variants = [_]types.BlockVariant{
    .{
        .name = "mamba",
        .meta = .{
            .mamba_config = .{
                .d_state = 128,
                .d_conv = 4,
                .n_heads = 48,
                .d_head = 32,
                .n_groups = 1,
                .d_inner = 1536,
            },
        },
        .weights = &granite_hybrid_mamba_weights,
    },
    .{ .name = "attention", .weights = &granite_hybrid_attention_weights },
};
const granite_hybrid_global_weights = [_]types.WeightSpec{
    .{ .id = "token_embeddings", .suffix = "model.embed_tokens.weight", .aliases = &.{ "embed_tokens.weight", "transformer.wte.weight", "backbone.embedding.weight", "language_model.model.embed_tokens.weight" }, .module_type = "Embedding", .layout = .embedding, .dtype = "float32", .required = true },
    .{ .id = "ln_final", .suffix = "model.norm.weight", .aliases = &.{ "norm.weight", "transformer.ln_f.weight", "backbone.norm.weight", "language_model.model.norm.weight", "model.embedding_norm.weight" }, .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "lm_head", .suffix = "lm_head.weight", .aliases = &.{ "output.weight", "transformer.lm_head.weight", "language_model.lm_head.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = false },
};
pub var arch: types.Architecture = .{
    .name = "granite_hybrid",
    .model_types = &granite_hybrid_model_types,
    .parse_config_hook = parseConfigHook,
    .block_variants = &granite_hybrid_block_variants,
    .layer_map = &granite_hybrid_layer_map,
    .variant_aliases = null,
    .block_weights = &granite_hybrid_block_weights,
    .global_weights = &granite_hybrid_global_weights,
    .weight_prefixes = &granite_hybrid_weight_prefixes,
    .d_ff_source_weight_ids = &.{"mlp.input_linear.weight"},
    .resolve_d_ff_from_weights = true,
    .has_qk_norm = false,
    .has_moe = false,
    .has_mamba = true,
    .has_shortconv = false,
    .has_mla = false,
    .has_fused_qkv = false,
    .has_fused_gate_up = true,
    .num_norms_per_block = 2,
    .use_gelu = false,
    .use_swiglu_oss = false,
    .norm_weight_offset = 0.0,
    .explicit_qk_norm_ops = false,
    .embedding_multiplier = 12.0,
};
