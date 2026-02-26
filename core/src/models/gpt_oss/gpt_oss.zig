//! GPT-OSS model-version metadata.

const layer_ops = @import("../layer_ops.zig");
const types = @import("../op_types.zig");
const config_hooks = @import("../config/hook_utils.zig");

pub const id: []const u8 = "gpt_oss";
pub const family: []const u8 = "gpt_oss";
pub const version: []const u8 = "gpt_oss";
pub const model_types: []const []const u8 = &.{
    "gpt_oss",
};

/// Static block topology for GPT-OSS decoder blocks.
/// Both sliding/full attention variants share this structure:
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

// Runtime architecture payload (migrated from runtime_architectures.zig)
const gpt_oss_model_types = [_][]const u8{"gpt_oss"};
const gpt_oss_weight_prefixes = [_][]const u8{ "model.layers.{d}.", "layers.{d}.", "transformer.h.{d}.", "backbone.layers.{d}.", "language_model.model.layers.{d}." };
const gpt_oss_block_weights = [_]types.WeightSpec{};
const gpt_oss_layer_map = [_]u8{ 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 };
const gpt_oss_sliding_attention_weights = [_]types.WeightSpec{
    .{ .id = "input_layernorm.weight", .suffix = "input_layernorm.weight", .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "self_attn.sinks", .suffix = "self_attn.sinks", .module_type = "Attention", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "self_attn.q_proj.weight", .suffix = "self_attn.q_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.q_proj.bias", .suffix = "self_attn.q_proj.bias", .module_type = "Linear", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "self_attn.k_proj.weight", .suffix = "self_attn.k_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.k_proj.bias", .suffix = "self_attn.k_proj.bias", .module_type = "Linear", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "self_attn.v_proj.weight", .suffix = "self_attn.v_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.v_proj.bias", .suffix = "self_attn.v_proj.bias", .module_type = "Linear", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "self_attn.o_proj.weight", .suffix = "self_attn.o_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.o_proj.bias", .suffix = "self_attn.o_proj.bias", .module_type = "Linear", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "post_attention_layernorm.weight", .suffix = "post_attention_layernorm.weight", .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "mlp.router.weight", .suffix = "mlp.router.weight", .module_type = "_TopKRouter", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "mlp.router.bias", .suffix = "mlp.router.bias", .module_type = "_TopKRouter", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "mlp.experts.gate_up_proj_blocks", .suffix = "mlp.experts.gate_up_proj_blocks", .module_type = "_Experts", .layout = .none, .dtype = "mxfp4", .required = true },
    .{ .id = "mlp.experts.gate_up_proj_scales", .suffix = "mlp.experts.gate_up_proj_scales", .module_type = "_Experts", .layout = .none, .dtype = "uint8", .required = true },
    .{ .id = "mlp.experts.gate_up_proj_bias", .suffix = "mlp.experts.gate_up_proj_bias", .module_type = "_Experts", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "mlp.experts.down_proj_blocks", .suffix = "mlp.experts.down_proj_blocks", .module_type = "_Experts", .layout = .none, .dtype = "mxfp4", .required = true },
    .{ .id = "mlp.experts.down_proj_scales", .suffix = "mlp.experts.down_proj_scales", .module_type = "_Experts", .layout = .none, .dtype = "uint8", .required = true },
    .{ .id = "mlp.experts.down_proj_bias", .suffix = "mlp.experts.down_proj_bias", .module_type = "_Experts", .layout = .none, .dtype = "float32", .required = true },
};
const gpt_oss_full_attention_weights = [_]types.WeightSpec{
    .{ .id = "input_layernorm.weight", .suffix = "input_layernorm.weight", .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "self_attn.sinks", .suffix = "self_attn.sinks", .module_type = "Attention", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "self_attn.q_proj.weight", .suffix = "self_attn.q_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.q_proj.bias", .suffix = "self_attn.q_proj.bias", .module_type = "Linear", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "self_attn.k_proj.weight", .suffix = "self_attn.k_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.k_proj.bias", .suffix = "self_attn.k_proj.bias", .module_type = "Linear", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "self_attn.v_proj.weight", .suffix = "self_attn.v_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.v_proj.bias", .suffix = "self_attn.v_proj.bias", .module_type = "Linear", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "self_attn.o_proj.weight", .suffix = "self_attn.o_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.o_proj.bias", .suffix = "self_attn.o_proj.bias", .module_type = "Linear", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "post_attention_layernorm.weight", .suffix = "post_attention_layernorm.weight", .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "mlp.router.weight", .suffix = "mlp.router.weight", .module_type = "_TopKRouter", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "mlp.router.bias", .suffix = "mlp.router.bias", .module_type = "_TopKRouter", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "mlp.experts.gate_up_proj_blocks", .suffix = "mlp.experts.gate_up_proj_blocks", .module_type = "_Experts", .layout = .none, .dtype = "mxfp4", .required = true },
    .{ .id = "mlp.experts.gate_up_proj_scales", .suffix = "mlp.experts.gate_up_proj_scales", .module_type = "_Experts", .layout = .none, .dtype = "uint8", .required = true },
    .{ .id = "mlp.experts.gate_up_proj_bias", .suffix = "mlp.experts.gate_up_proj_bias", .module_type = "_Experts", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "mlp.experts.down_proj_blocks", .suffix = "mlp.experts.down_proj_blocks", .module_type = "_Experts", .layout = .none, .dtype = "mxfp4", .required = true },
    .{ .id = "mlp.experts.down_proj_scales", .suffix = "mlp.experts.down_proj_scales", .module_type = "_Experts", .layout = .none, .dtype = "uint8", .required = true },
    .{ .id = "mlp.experts.down_proj_bias", .suffix = "mlp.experts.down_proj_bias", .module_type = "_Experts", .layout = .none, .dtype = "float32", .required = true },
};
var gpt_oss_block_variants = [_]types.BlockVariant{
    .{ .name = "sliding_attention", .weights = &gpt_oss_sliding_attention_weights },
    .{ .name = "full_attention", .weights = &gpt_oss_full_attention_weights },
};
const gpt_oss_global_weights = [_]types.WeightSpec{
    .{ .id = "token_embeddings", .suffix = "model.embed_tokens.weight", .aliases = &.{ "embed_tokens.weight", "transformer.wte.weight", "backbone.embedding.weight", "language_model.model.embed_tokens.weight" }, .module_type = "Embedding", .layout = .embedding, .dtype = "float32", .required = true },
    .{ .id = "ln_final", .suffix = "model.norm.weight", .aliases = &.{ "norm.weight", "transformer.ln_f.weight", "backbone.norm.weight", "language_model.model.norm.weight", "model.embedding_norm.weight" }, .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "lm_head", .suffix = "lm_head.weight", .aliases = &.{ "output.weight", "transformer.lm_head.weight", "language_model.lm_head.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = false },
};
pub var arch: types.Architecture = .{
    .name = "gpt_oss",
    .model_types = &gpt_oss_model_types,
    .parse_config_hook = config_hooks.applyCommonTextConfigHook,
    .block_variants = &gpt_oss_block_variants,
    .layer_map = &gpt_oss_layer_map,
    .variant_aliases = null,
    .block_weights = &gpt_oss_block_weights,
    .global_weights = &gpt_oss_global_weights,
    .weight_prefixes = &gpt_oss_weight_prefixes,
    .d_ff_source_weight_ids = &.{"mlp.experts.gate_up_proj_blocks"},
    .resolve_d_ff_from_weights = true,
    .has_qk_norm = false,
    .has_moe = true,
    .has_mamba = false,
    .has_shortconv = false,
    .has_mla = false,
    .has_fused_qkv = false,
    .has_fused_gate_up = false,
    .num_norms_per_block = 2,
    .use_gelu = false,
    .use_swiglu_oss = true,
    .norm_weight_offset = 0.0,
    .explicit_qk_norm_ops = false,
    .embedding_multiplier = 1.0,
};
