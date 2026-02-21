//! LFM2.5 model-version metadata.

const layer_ops = @import("../layer_ops.zig");
const types = @import("../op_types.zig");

pub const id: []const u8 = "lfm2_5";
pub const family: []const u8 = "lfm2";
pub const version: []const u8 = "lfm2_5";
pub const model_types: []const []const u8 = &.{
    "lfm2_5",
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

// Runtime architecture payload (migrated from runtime_architectures.zig)
const lfm2_5_model_types = [_][]const u8{"lfm2_5"};
const lfm2_5_weight_prefixes = [_][]const u8{ "model.layers.{d}.", "layers.{d}.", "transformer.h.{d}.", "backbone.layers.{d}.", "language_model.model.layers.{d}." };
const lfm2_5_pre_block_ops = [_]types.Op{
    .{ .op_type = .embedding, .inputs = &.{.{ .tensor = "input_ids" }}, .outputs = &.{"_t0"} },
};
const lfm2_5_post_block_ops = [_]types.Op{
    .{ .op_type = .norm, .inputs = &.{.{ .tensor = "_t_last" }}, .outputs = &.{"_t_out"} },
};
const lfm2_5_block_ops = [_]types.Op{};
const lfm2_5_block_weights = [_]types.WeightSpec{};
const lfm2_5_layer_map = [_]u8{ 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0 };
const lfm2_5_conv_ops = [_]types.Op{
    .{ .op_type = .norm, .name = "operator_norm", .inputs = &.{ .{ .tensor = "x" }, .{ .tensor = "operator_norm.weight" } }, .outputs = &.{"_t0"} },
    .{ .op_type = .shortconv, .name = "conv", .inputs = &.{.{ .tensor = "_t0" }}, .outputs = &.{"_t1"}, .d_conv = 3, .conv_dim = 2048, .conv_dim_out = 2048 },
    .{ .op_type = .add, .inputs = &.{ .{ .tensor = "x" }, .{ .tensor = "_t1" } }, .outputs = &.{"_t2"} },
    .{ .op_type = .norm, .name = "ffn_norm", .inputs = &.{ .{ .tensor = "_t2" }, .{ .tensor = "ffn_norm.weight" } }, .outputs = &.{"_t3"} },
    .{ .op_type = .mlp, .inputs = &.{.{ .tensor = "_t3" }}, .outputs = &.{"_t4"}, .activation = "silu" },
    .{ .op_type = .add, .inputs = &.{ .{ .tensor = "_t2" }, .{ .tensor = "_t4" } }, .outputs = &.{"_t5"} },
};
const lfm2_5_conv_weights = [_]types.WeightSpec{
    .{ .id = "operator_norm.weight", .candidates = &.{ "model.layers.{d}.operator_norm.weight", "layers.{d}.operator_norm.weight", "transformer.h.{d}.operator_norm.weight", "backbone.layers.{d}.operator_norm.weight", "language_model.model.layers.{d}.operator_norm.weight" }, .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "conv.in_proj.weight", .candidates = &.{ "model.layers.{d}.conv.in_proj.weight", "layers.{d}.conv.in_proj.weight", "transformer.h.{d}.conv.in_proj.weight", "backbone.layers.{d}.conv.in_proj.weight", "language_model.model.layers.{d}.conv.in_proj.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "conv.conv.weight", .candidates = &.{ "model.layers.{d}.conv.conv.weight", "layers.{d}.conv.conv.weight", "transformer.h.{d}.conv.conv.weight", "backbone.layers.{d}.conv.conv.weight", "language_model.model.layers.{d}.conv.conv.weight" }, .module_type = "Conv1d", .layout = .conv1d_depthwise, .dtype = "float32", .required = true },
    .{ .id = "conv.out_proj.weight", .candidates = &.{ "model.layers.{d}.conv.out_proj.weight", "layers.{d}.conv.out_proj.weight", "transformer.h.{d}.conv.out_proj.weight", "backbone.layers.{d}.conv.out_proj.weight", "language_model.model.layers.{d}.conv.out_proj.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "ffn_norm.weight", .candidates = &.{ "model.layers.{d}.ffn_norm.weight", "layers.{d}.ffn_norm.weight", "transformer.h.{d}.ffn_norm.weight", "backbone.layers.{d}.ffn_norm.weight", "language_model.model.layers.{d}.ffn_norm.weight" }, .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "feed_forward.w1.weight", .candidates = &.{ "model.layers.{d}.feed_forward.w1.weight", "layers.{d}.feed_forward.w1.weight", "transformer.h.{d}.feed_forward.w1.weight", "backbone.layers.{d}.feed_forward.w1.weight", "language_model.model.layers.{d}.feed_forward.w1.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "feed_forward.w2.weight", .candidates = &.{ "model.layers.{d}.feed_forward.w2.weight", "layers.{d}.feed_forward.w2.weight", "transformer.h.{d}.feed_forward.w2.weight", "backbone.layers.{d}.feed_forward.w2.weight", "language_model.model.layers.{d}.feed_forward.w2.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "feed_forward.w3.weight", .candidates = &.{ "model.layers.{d}.feed_forward.w3.weight", "layers.{d}.feed_forward.w3.weight", "transformer.h.{d}.feed_forward.w3.weight", "backbone.layers.{d}.feed_forward.w3.weight", "language_model.model.layers.{d}.feed_forward.w3.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
};
const lfm2_5_full_attention_ops = [_]types.Op{
    .{ .op_type = .norm, .name = "operator_norm", .inputs = &.{ .{ .tensor = "x" }, .{ .tensor = "operator_norm.weight" } }, .outputs = &.{"_t0"} },
    .{ .op_type = .multihead_attention, .inputs = &.{.{ .tensor = "_t0" }}, .outputs = &.{"_t1"}, .qk_norm = true },
    .{ .op_type = .add, .inputs = &.{ .{ .tensor = "x" }, .{ .tensor = "_t1" } }, .outputs = &.{"_t2"} },
    .{ .op_type = .norm, .name = "ffn_norm", .inputs = &.{ .{ .tensor = "_t2" }, .{ .tensor = "ffn_norm.weight" } }, .outputs = &.{"_t3"} },
    .{ .op_type = .mlp, .inputs = &.{.{ .tensor = "_t3" }}, .outputs = &.{"_t4"}, .activation = "silu" },
    .{ .op_type = .add, .inputs = &.{ .{ .tensor = "_t2" }, .{ .tensor = "_t4" } }, .outputs = &.{"_t5"} },
};
const lfm2_5_full_attention_weights = [_]types.WeightSpec{
    .{ .id = "operator_norm.weight", .candidates = &.{ "model.layers.{d}.operator_norm.weight", "layers.{d}.operator_norm.weight", "transformer.h.{d}.operator_norm.weight", "backbone.layers.{d}.operator_norm.weight", "language_model.model.layers.{d}.operator_norm.weight" }, .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "self_attn.q_proj.weight", .candidates = &.{ "model.layers.{d}.self_attn.q_proj.weight", "layers.{d}.self_attn.q_proj.weight", "transformer.h.{d}.self_attn.q_proj.weight", "backbone.layers.{d}.self_attn.q_proj.weight", "language_model.model.layers.{d}.self_attn.q_proj.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.k_proj.weight", .candidates = &.{ "model.layers.{d}.self_attn.k_proj.weight", "layers.{d}.self_attn.k_proj.weight", "transformer.h.{d}.self_attn.k_proj.weight", "backbone.layers.{d}.self_attn.k_proj.weight", "language_model.model.layers.{d}.self_attn.k_proj.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.v_proj.weight", .candidates = &.{ "model.layers.{d}.self_attn.v_proj.weight", "layers.{d}.self_attn.v_proj.weight", "transformer.h.{d}.self_attn.v_proj.weight", "backbone.layers.{d}.self_attn.v_proj.weight", "language_model.model.layers.{d}.self_attn.v_proj.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.out_proj.weight", .candidates = &.{ "model.layers.{d}.self_attn.out_proj.weight", "layers.{d}.self_attn.out_proj.weight", "transformer.h.{d}.self_attn.out_proj.weight", "backbone.layers.{d}.self_attn.out_proj.weight", "language_model.model.layers.{d}.self_attn.out_proj.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.q_layernorm.weight", .candidates = &.{ "model.layers.{d}.self_attn.q_layernorm.weight", "layers.{d}.self_attn.q_layernorm.weight", "transformer.h.{d}.self_attn.q_layernorm.weight", "backbone.layers.{d}.self_attn.q_layernorm.weight", "language_model.model.layers.{d}.self_attn.q_layernorm.weight" }, .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "self_attn.k_layernorm.weight", .candidates = &.{ "model.layers.{d}.self_attn.k_layernorm.weight", "layers.{d}.self_attn.k_layernorm.weight", "transformer.h.{d}.self_attn.k_layernorm.weight", "backbone.layers.{d}.self_attn.k_layernorm.weight", "language_model.model.layers.{d}.self_attn.k_layernorm.weight" }, .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "ffn_norm.weight", .candidates = &.{ "model.layers.{d}.ffn_norm.weight", "layers.{d}.ffn_norm.weight", "transformer.h.{d}.ffn_norm.weight", "backbone.layers.{d}.ffn_norm.weight", "language_model.model.layers.{d}.ffn_norm.weight" }, .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "feed_forward.w1.weight", .candidates = &.{ "model.layers.{d}.feed_forward.w1.weight", "layers.{d}.feed_forward.w1.weight", "transformer.h.{d}.feed_forward.w1.weight", "backbone.layers.{d}.feed_forward.w1.weight", "language_model.model.layers.{d}.feed_forward.w1.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "feed_forward.w2.weight", .candidates = &.{ "model.layers.{d}.feed_forward.w2.weight", "layers.{d}.feed_forward.w2.weight", "transformer.h.{d}.feed_forward.w2.weight", "backbone.layers.{d}.feed_forward.w2.weight", "language_model.model.layers.{d}.feed_forward.w2.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "feed_forward.w3.weight", .candidates = &.{ "model.layers.{d}.feed_forward.w3.weight", "layers.{d}.feed_forward.w3.weight", "transformer.h.{d}.feed_forward.w3.weight", "backbone.layers.{d}.feed_forward.w3.weight", "language_model.model.layers.{d}.feed_forward.w3.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
};
var lfm2_5_block_variants = [_]types.BlockVariant{
    .{ .name = "conv", .ops = &lfm2_5_conv_ops, .weights = &lfm2_5_conv_weights },
    .{ .name = "full_attention", .ops = &lfm2_5_full_attention_ops, .weights = &lfm2_5_full_attention_weights },
};
const lfm2_5_global_weights = [_]types.WeightSpec{
    .{ .id = "token_embeddings", .candidates = &.{ "model.embed_tokens.weight", "embed_tokens.weight", "transformer.wte.weight", "backbone.embedding.weight", "language_model.model.embed_tokens.weight" }, .module_type = "Embedding", .layout = .embedding, .dtype = "float32", .required = true },
    .{ .id = "ln_final", .candidates = &.{ "model.norm.weight", "norm.weight", "transformer.ln_f.weight", "backbone.norm.weight", "language_model.model.norm.weight", "model.embedding_norm.weight" }, .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "lm_head", .candidates = &.{ "lm_head.weight", "output.weight", "transformer.lm_head.weight", "language_model.lm_head.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = false },
};
pub var arch: types.Architecture = .{
    .name = "lfm2_5",
    .model_types = &lfm2_5_model_types,
    .block_ops = &lfm2_5_block_ops,
    .pre_block_ops = &lfm2_5_pre_block_ops,
    .post_block_ops = &lfm2_5_post_block_ops,
    .block_variants = &lfm2_5_block_variants,
    .layer_map = &lfm2_5_layer_map,
    .variant_aliases = null,
    .block_weights = &lfm2_5_block_weights,
    .global_weights = &lfm2_5_global_weights,
    .weight_prefixes = &lfm2_5_weight_prefixes,
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
};
