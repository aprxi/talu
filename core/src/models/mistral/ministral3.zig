//! Ministral3 model-version metadata.

const layer_ops = @import("../layer_ops.zig");
const types = @import("../op_types.zig");
const config_hooks = @import("../config/hook_utils.zig");

pub const id: []const u8 = "ministral3";
pub const family: []const u8 = "mistral";
pub const version: []const u8 = "ministral3";
pub const model_types: []const []const u8 = &.{
    "ministral3",
    "mistral3",
};

/// Static block topology for Ministral3 decoder blocks.
/// Sequence: norm -> attention -> add -> norm -> moe -> add.
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
const ministral3_model_types = [_][]const u8{ "ministral3", "mistral3" };
const ministral3_weight_prefixes = [_][]const u8{ "model.layers.{d}.", "layers.{d}.", "transformer.h.{d}.", "backbone.layers.{d}.", "language_model.model.layers.{d}." };
const ministral3_block_weights = [_]types.WeightSpec{
    .{ .id = "input_layernorm.weight", .suffix = "input_layernorm.weight", .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "self_attn.q_proj.weight", .suffix = "self_attn.q_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.k_proj.weight", .suffix = "self_attn.k_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.v_proj.weight", .suffix = "self_attn.v_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.o_proj.weight", .suffix = "self_attn.o_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "post_attention_layernorm.weight", .suffix = "post_attention_layernorm.weight", .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "block_sparse_moe.gate.weight", .suffix = "block_sparse_moe.gate.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "block_sparse_moe.experts.0.gate_proj.weight", .suffix = "block_sparse_moe.experts.0.gate_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "block_sparse_moe.experts.0.up_proj.weight", .suffix = "block_sparse_moe.experts.0.up_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "block_sparse_moe.experts.0.down_proj.weight", .suffix = "block_sparse_moe.experts.0.down_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "block_sparse_moe.experts.1.gate_proj.weight", .suffix = "block_sparse_moe.experts.1.gate_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "block_sparse_moe.experts.1.up_proj.weight", .suffix = "block_sparse_moe.experts.1.up_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "block_sparse_moe.experts.1.down_proj.weight", .suffix = "block_sparse_moe.experts.1.down_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "block_sparse_moe.experts.2.gate_proj.weight", .suffix = "block_sparse_moe.experts.2.gate_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "block_sparse_moe.experts.2.up_proj.weight", .suffix = "block_sparse_moe.experts.2.up_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "block_sparse_moe.experts.2.down_proj.weight", .suffix = "block_sparse_moe.experts.2.down_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "block_sparse_moe.experts.3.gate_proj.weight", .suffix = "block_sparse_moe.experts.3.gate_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "block_sparse_moe.experts.3.up_proj.weight", .suffix = "block_sparse_moe.experts.3.up_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "block_sparse_moe.experts.3.down_proj.weight", .suffix = "block_sparse_moe.experts.3.down_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "block_sparse_moe.experts.4.gate_proj.weight", .suffix = "block_sparse_moe.experts.4.gate_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "block_sparse_moe.experts.4.up_proj.weight", .suffix = "block_sparse_moe.experts.4.up_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "block_sparse_moe.experts.4.down_proj.weight", .suffix = "block_sparse_moe.experts.4.down_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "block_sparse_moe.experts.5.gate_proj.weight", .suffix = "block_sparse_moe.experts.5.gate_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "block_sparse_moe.experts.5.up_proj.weight", .suffix = "block_sparse_moe.experts.5.up_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "block_sparse_moe.experts.5.down_proj.weight", .suffix = "block_sparse_moe.experts.5.down_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "block_sparse_moe.experts.6.gate_proj.weight", .suffix = "block_sparse_moe.experts.6.gate_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "block_sparse_moe.experts.6.up_proj.weight", .suffix = "block_sparse_moe.experts.6.up_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "block_sparse_moe.experts.6.down_proj.weight", .suffix = "block_sparse_moe.experts.6.down_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "block_sparse_moe.experts.7.gate_proj.weight", .suffix = "block_sparse_moe.experts.7.gate_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "block_sparse_moe.experts.7.up_proj.weight", .suffix = "block_sparse_moe.experts.7.up_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "block_sparse_moe.experts.7.down_proj.weight", .suffix = "block_sparse_moe.experts.7.down_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
};
const ministral3_global_weights = [_]types.WeightSpec{
    .{ .id = "token_embeddings", .suffix = "model.embed_tokens.weight", .aliases = &.{ "embed_tokens.weight", "transformer.wte.weight", "backbone.embedding.weight", "language_model.model.embed_tokens.weight" }, .module_type = "Embedding", .layout = .embedding, .dtype = "float32", .required = true },
    .{ .id = "ln_final", .suffix = "model.norm.weight", .aliases = &.{ "norm.weight", "transformer.ln_f.weight", "backbone.norm.weight", "language_model.model.norm.weight", "model.embedding_norm.weight" }, .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "lm_head", .suffix = "lm_head.weight", .aliases = &.{ "output.weight", "transformer.lm_head.weight", "language_model.lm_head.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = false },
};
pub var arch: types.Architecture = .{
    .name = "ministral3",
    .model_types = &ministral3_model_types,
    .parse_config_hook = config_hooks.applyCommonTextConfigHook,
    .block_variants = null,
    .layer_map = null,
    .variant_aliases = null,
    .block_weights = &ministral3_block_weights,
    .global_weights = &ministral3_global_weights,
    .weight_prefixes = &ministral3_weight_prefixes,
    .d_ff_source_weight_ids = &.{"block_sparse_moe.experts.0.gate_proj.weight"},
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
    .use_swiglu_oss = false,
    .norm_weight_offset = 0.0,
    .explicit_qk_norm_ops = false,
    .embedding_multiplier = 1.0,
};
