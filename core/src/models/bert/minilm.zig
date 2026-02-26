//! MiniLM (BERT-family) model-version metadata.

const layer_ops = @import("../layer_ops.zig");
const types = @import("../op_types.zig");
const config_hooks = @import("../config/hook_utils.zig");

pub const id: []const u8 = "minilm";
pub const family: []const u8 = "bert";
pub const version: []const u8 = "minilm";
pub const model_types: []const []const u8 = &.{
    "bert",
    "minilm",
};

/// Static block topology for MiniLM encoder blocks.
/// Sequence: attention -> add -> norm -> mlp -> add -> norm.
pub const attention_mlp_program: []const layer_ops.LayerOp = &.{
    .{ .kernel = .{
        .id = 0,
        .in = .residual,
        .out = .branch_out,
        .debug_type = .multihead_attention,
    } },
    .{ .add = .{
        .branch = .branch_out,
        .scale = .residual_multiplier,
    } },
    .{ .kernel = .{
        .id = 1,
        .in = .residual,
        .out = .norm_out,
        .debug_type = .norm,
    } },
    .{ .kernel = .{
        .id = 2,
        .in = .norm_out,
        .out = .branch_out,
        .debug_type = .mlp,
    } },
    .{ .add = .{
        .branch = .branch_out,
        .scale = .residual_multiplier,
    } },
    .{ .kernel = .{
        .id = 3,
        .in = .residual,
        .out = .norm_out,
        .debug_type = .norm,
    } },
};

// Runtime architecture payload (migrated from runtime_architectures.zig)
const minilm_model_types = [_][]const u8{ "bert", "minilm" };
const minilm_weight_prefixes = [_][]const u8{ "bert.encoder.layer.{d}.", "encoder.layer.{d}." };
const minilm_block_weights = [_]types.WeightSpec{
    .{ .id = "self_attn.q_proj.weight", .suffix = "self_attn.q_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.q_proj.bias", .suffix = "self_attn.q_proj.bias", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.k_proj.weight", .suffix = "self_attn.k_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.k_proj.bias", .suffix = "self_attn.k_proj.bias", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.v_proj.weight", .suffix = "self_attn.v_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.v_proj.bias", .suffix = "self_attn.v_proj.bias", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.o_proj.weight", .suffix = "self_attn.o_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.o_proj.bias", .suffix = "self_attn.o_proj.bias", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "input_layernorm.weight", .suffix = "input_layernorm.weight", .module_type = "LayerNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "input_layernorm.bias", .suffix = "input_layernorm.bias", .module_type = "LayerNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "mlp.dense_in.weight", .suffix = "mlp.dense_in.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "mlp.dense_in.bias", .suffix = "mlp.dense_in.bias", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "mlp.dense_out.weight", .suffix = "mlp.dense_out.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "mlp.dense_out.bias", .suffix = "mlp.dense_out.bias", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "post_attention_layernorm.weight", .suffix = "post_attention_layernorm.weight", .module_type = "LayerNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "post_attention_layernorm.bias", .suffix = "post_attention_layernorm.bias", .module_type = "LayerNorm", .layout = .none, .dtype = "float32", .required = true },
};
const minilm_global_weights = [_]types.WeightSpec{
    .{ .id = "token_embeddings", .suffix = "bert.embeddings.word_embeddings.weight", .aliases = &.{"embeddings.word_embeddings.weight"}, .module_type = "Embedding", .layout = .embedding, .dtype = "float32", .required = true },
    .{ .id = "position_embeddings", .suffix = "bert.embeddings.position_embeddings.weight", .aliases = &.{"embeddings.position_embeddings.weight"}, .module_type = "Embedding", .layout = .embedding, .dtype = "float32", .required = true },
    .{ .id = "token_type_embeddings", .suffix = "bert.embeddings.token_type_embeddings.weight", .aliases = &.{"embeddings.token_type_embeddings.weight"}, .module_type = "Embedding", .layout = .embedding, .dtype = "float32", .required = true },
    .{ .id = "embedding_ln", .suffix = "bert.embeddings.LayerNorm.weight", .aliases = &.{"embeddings.LayerNorm.weight"}, .module_type = "LayerNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "embedding_ln_bias", .suffix = "bert.embeddings.LayerNorm.bias", .aliases = &.{"embeddings.LayerNorm.bias"}, .module_type = "LayerNorm", .layout = .none, .dtype = "float32", .required = true },
};
pub var arch: types.Architecture = .{
    .name = "minilm",
    .model_types = &minilm_model_types,
    .parse_config_hook = config_hooks.applyCommonTextConfigHook,
    .kernel_meta = .{ .is_causal = false },
    .block_variants = null,
    .layer_map = null,
    .variant_aliases = null,
    .block_weights = &minilm_block_weights,
    .global_weights = &minilm_global_weights,
    .weight_prefixes = &minilm_weight_prefixes,
    .d_ff_source_weight_ids = &.{"mlp.dense_in.weight"},
    .resolve_d_ff_from_weights = true,
    .has_qk_norm = false,
    .has_moe = false,
    .has_mamba = false,
    .has_shortconv = false,
    .has_mla = false,
    .has_fused_qkv = false,
    .has_fused_gate_up = false,
    .num_norms_per_block = 2,
    .use_gelu = true,
    .use_swiglu_oss = false,
    .norm_weight_offset = 0.0,
    .explicit_qk_norm_ops = false,
    .embedding_multiplier = 1.0,
};
