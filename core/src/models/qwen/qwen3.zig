//! Qwen3 family model-version metadata.

const std = @import("std");
const tensor = @import("../../tensor.zig");
const layer_ops = @import("../layer_ops.zig");
const types = @import("../op_types.zig");
const config_hooks = @import("../config/hook_utils.zig");
const vision_shared = @import("../vision_shared.zig");

pub const id: []const u8 = "qwen3";
pub const family: []const u8 = "qwen";
pub const version: []const u8 = "qwen3";
pub const model_types: []const []const u8 = &.{
    "qwen3",
    "qwen3_vl",
    "qwen2.5",
    "qwen2",
    "qwen",
};

/// Static block topology for Qwen3 dense decoder blocks.
/// Sequence: norm -> attention -> add -> norm -> ffn -> add.
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

fn parseConfigHook(
    config_obj: std.json.ObjectMap,
    root_obj: std.json.ObjectMap,
    config: *tensor.ModelConfig,
) void {
    config_hooks.applyCommonTextConfig(config_obj, root_obj, config);
    config_hooks.applyVisionConfig(root_obj, config);
}

// Runtime architecture payload (migrated from runtime_architectures.zig)
const qwen3_model_types = [_][]const u8{ "qwen3", "qwen3_vl", "qwen2.5", "qwen2", "qwen" };
const qwen3_weight_prefixes = [_][]const u8{ "model.layers.{d}.", "model.language_model.layers.{d}.", "layers.{d}.", "transformer.h.{d}.", "backbone.layers.{d}.", "language_model.model.layers.{d}." };
const qwen3_block_weights = [_]types.WeightSpec{
    .{ .id = "input_layernorm.weight", .suffix = "input_layernorm.weight", .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "self_attn.q_proj.weight", .suffix = "self_attn.q_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.k_proj.weight", .suffix = "self_attn.k_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.v_proj.weight", .suffix = "self_attn.v_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.o_proj.weight", .suffix = "self_attn.o_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.q_norm.weight", .suffix = "self_attn.q_norm.weight", .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "self_attn.k_norm.weight", .suffix = "self_attn.k_norm.weight", .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "post_attention_layernorm.weight", .suffix = "post_attention_layernorm.weight", .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "mlp.gate_proj.weight", .suffix = "mlp.gate_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "mlp.up_proj.weight", .suffix = "mlp.up_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "mlp.down_proj.weight", .suffix = "mlp.down_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
};
const qwen3_global_weights = [_]types.WeightSpec{
    .{ .id = "token_embeddings", .suffix = "model.embed_tokens.weight", .aliases = &.{ "model.language_model.embed_tokens.weight", "embed_tokens.weight", "transformer.wte.weight", "backbone.embedding.weight", "language_model.model.embed_tokens.weight" }, .module_type = "Embedding", .layout = .embedding, .dtype = "float32", .required = true },
    .{ .id = "ln_final", .suffix = "model.norm.weight", .aliases = &.{ "model.language_model.norm.weight", "norm.weight", "transformer.ln_f.weight", "backbone.norm.weight", "language_model.model.norm.weight", "model.embedding_norm.weight" }, .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "lm_head", .suffix = "lm_head.weight", .aliases = &.{ "output.weight", "transformer.lm_head.weight", "language_model.lm_head.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = false },
};
pub var arch: types.Architecture = .{
    .name = "qwen3",
    .model_types = &qwen3_model_types,
    .parse_config_hook = parseConfigHook,
    .block_variants = null,
    .layer_map = null,
    .variant_aliases = null,
    .block_weights = &qwen3_block_weights,
    .global_weights = &qwen3_global_weights,
    .weight_prefixes = &qwen3_weight_prefixes,
    .d_ff_source_weight_ids = &.{"mlp.gate_proj.weight"},
    .resolve_d_ff_from_weights = true,
    .has_qk_norm = true,
    .has_moe = false,
    .has_mamba = false,
    .has_shortconv = false,
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
