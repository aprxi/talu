//! GPT-OSS model-version metadata.

const layer_ops = @import("../layer_ops.zig");
const types = @import("../op_types.zig");

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
const gpt_oss_pre_block_ops = [_]types.Op{
    .{ .op_type = .embedding, .inputs = &.{.{ .tensor = "input_ids" }}, .outputs = &.{"_t0"} },
};
const gpt_oss_post_block_ops = [_]types.Op{
    .{ .op_type = .norm, .inputs = &.{.{ .tensor = "_t_last" }}, .outputs = &.{"_t_out"} },
};
const gpt_oss_block_ops = [_]types.Op{};
const gpt_oss_block_weights = [_]types.WeightSpec{};
const gpt_oss_layer_map = [_]u8{ 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 };
const gpt_oss_sliding_attention_ops = [_]types.Op{
    .{ .op_type = .norm, .name = "input_layernorm", .inputs = &.{ .{ .tensor = "x" }, .{ .tensor = "input_layernorm.weight" } }, .outputs = &.{"_t0"} },
    .{ .op_type = .multihead_attention, .inputs = &.{.{ .tensor = "_t0" }}, .outputs = &.{"_t1"}, .sliding_window = 128 },
    .{ .op_type = .add, .inputs = &.{ .{ .tensor = "x" }, .{ .tensor = "_t1" } }, .outputs = &.{"_t2"} },
    .{ .op_type = .norm, .name = "post_attention_layernorm", .inputs = &.{ .{ .tensor = "_t2" }, .{ .tensor = "post_attention_layernorm.weight" } }, .outputs = &.{"_t3"} },
    .{ .op_type = .moe, .inputs = &.{.{ .tensor = "_t3" }}, .outputs = &.{"_t4"}, .activation = "swiglu_oss", .num_experts = 32, .experts_per_token = 4 },
    .{ .op_type = .add, .inputs = &.{ .{ .tensor = "_t2" }, .{ .tensor = "_t4" } }, .outputs = &.{"_t5"} },
};
const gpt_oss_sliding_attention_weights = [_]types.WeightSpec{
    .{ .id = "input_layernorm.weight", .candidates = &.{ "model.layers.{d}.input_layernorm.weight", "layers.{d}.input_layernorm.weight", "transformer.h.{d}.input_layernorm.weight", "backbone.layers.{d}.input_layernorm.weight", "language_model.model.layers.{d}.input_layernorm.weight" }, .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "self_attn.sinks", .candidates = &.{ "model.layers.{d}.self_attn.sinks", "layers.{d}.self_attn.sinks", "transformer.h.{d}.self_attn.sinks", "backbone.layers.{d}.self_attn.sinks", "language_model.model.layers.{d}.self_attn.sinks" }, .module_type = "Attention", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "self_attn.q_proj.weight", .candidates = &.{ "model.layers.{d}.self_attn.q_proj.weight", "layers.{d}.self_attn.q_proj.weight", "transformer.h.{d}.self_attn.q_proj.weight", "backbone.layers.{d}.self_attn.q_proj.weight", "language_model.model.layers.{d}.self_attn.q_proj.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.q_proj.bias", .candidates = &.{ "model.layers.{d}.self_attn.q_proj.bias", "layers.{d}.self_attn.q_proj.bias", "transformer.h.{d}.self_attn.q_proj.bias", "backbone.layers.{d}.self_attn.q_proj.bias", "language_model.model.layers.{d}.self_attn.q_proj.bias" }, .module_type = "Linear", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "self_attn.k_proj.weight", .candidates = &.{ "model.layers.{d}.self_attn.k_proj.weight", "layers.{d}.self_attn.k_proj.weight", "transformer.h.{d}.self_attn.k_proj.weight", "backbone.layers.{d}.self_attn.k_proj.weight", "language_model.model.layers.{d}.self_attn.k_proj.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.k_proj.bias", .candidates = &.{ "model.layers.{d}.self_attn.k_proj.bias", "layers.{d}.self_attn.k_proj.bias", "transformer.h.{d}.self_attn.k_proj.bias", "backbone.layers.{d}.self_attn.k_proj.bias", "language_model.model.layers.{d}.self_attn.k_proj.bias" }, .module_type = "Linear", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "self_attn.v_proj.weight", .candidates = &.{ "model.layers.{d}.self_attn.v_proj.weight", "layers.{d}.self_attn.v_proj.weight", "transformer.h.{d}.self_attn.v_proj.weight", "backbone.layers.{d}.self_attn.v_proj.weight", "language_model.model.layers.{d}.self_attn.v_proj.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.v_proj.bias", .candidates = &.{ "model.layers.{d}.self_attn.v_proj.bias", "layers.{d}.self_attn.v_proj.bias", "transformer.h.{d}.self_attn.v_proj.bias", "backbone.layers.{d}.self_attn.v_proj.bias", "language_model.model.layers.{d}.self_attn.v_proj.bias" }, .module_type = "Linear", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "self_attn.o_proj.weight", .candidates = &.{ "model.layers.{d}.self_attn.o_proj.weight", "layers.{d}.self_attn.o_proj.weight", "transformer.h.{d}.self_attn.o_proj.weight", "backbone.layers.{d}.self_attn.o_proj.weight", "language_model.model.layers.{d}.self_attn.o_proj.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.o_proj.bias", .candidates = &.{ "model.layers.{d}.self_attn.o_proj.bias", "layers.{d}.self_attn.o_proj.bias", "transformer.h.{d}.self_attn.o_proj.bias", "backbone.layers.{d}.self_attn.o_proj.bias", "language_model.model.layers.{d}.self_attn.o_proj.bias" }, .module_type = "Linear", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "post_attention_layernorm.weight", .candidates = &.{ "model.layers.{d}.post_attention_layernorm.weight", "layers.{d}.post_attention_layernorm.weight", "transformer.h.{d}.post_attention_layernorm.weight", "backbone.layers.{d}.post_attention_layernorm.weight", "language_model.model.layers.{d}.post_attention_layernorm.weight" }, .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "mlp.router.weight", .candidates = &.{ "model.layers.{d}.mlp.router.weight", "layers.{d}.mlp.router.weight", "transformer.h.{d}.mlp.router.weight", "backbone.layers.{d}.mlp.router.weight", "language_model.model.layers.{d}.mlp.router.weight" }, .module_type = "_TopKRouter", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "mlp.router.bias", .candidates = &.{ "model.layers.{d}.mlp.router.bias", "layers.{d}.mlp.router.bias", "transformer.h.{d}.mlp.router.bias", "backbone.layers.{d}.mlp.router.bias", "language_model.model.layers.{d}.mlp.router.bias" }, .module_type = "_TopKRouter", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "mlp.experts.gate_up_proj_blocks", .candidates = &.{ "model.layers.{d}.mlp.experts.gate_up_proj_blocks", "layers.{d}.mlp.experts.gate_up_proj_blocks", "transformer.h.{d}.mlp.experts.gate_up_proj_blocks", "backbone.layers.{d}.mlp.experts.gate_up_proj_blocks", "language_model.model.layers.{d}.mlp.experts.gate_up_proj_blocks" }, .module_type = "_Experts", .layout = .none, .dtype = "mxfp4", .required = true },
    .{ .id = "mlp.experts.gate_up_proj_scales", .candidates = &.{ "model.layers.{d}.mlp.experts.gate_up_proj_scales", "layers.{d}.mlp.experts.gate_up_proj_scales", "transformer.h.{d}.mlp.experts.gate_up_proj_scales", "backbone.layers.{d}.mlp.experts.gate_up_proj_scales", "language_model.model.layers.{d}.mlp.experts.gate_up_proj_scales" }, .module_type = "_Experts", .layout = .none, .dtype = "uint8", .required = true },
    .{ .id = "mlp.experts.gate_up_proj_bias", .candidates = &.{ "model.layers.{d}.mlp.experts.gate_up_proj_bias", "layers.{d}.mlp.experts.gate_up_proj_bias", "transformer.h.{d}.mlp.experts.gate_up_proj_bias", "backbone.layers.{d}.mlp.experts.gate_up_proj_bias", "language_model.model.layers.{d}.mlp.experts.gate_up_proj_bias" }, .module_type = "_Experts", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "mlp.experts.down_proj_blocks", .candidates = &.{ "model.layers.{d}.mlp.experts.down_proj_blocks", "layers.{d}.mlp.experts.down_proj_blocks", "transformer.h.{d}.mlp.experts.down_proj_blocks", "backbone.layers.{d}.mlp.experts.down_proj_blocks", "language_model.model.layers.{d}.mlp.experts.down_proj_blocks" }, .module_type = "_Experts", .layout = .none, .dtype = "mxfp4", .required = true },
    .{ .id = "mlp.experts.down_proj_scales", .candidates = &.{ "model.layers.{d}.mlp.experts.down_proj_scales", "layers.{d}.mlp.experts.down_proj_scales", "transformer.h.{d}.mlp.experts.down_proj_scales", "backbone.layers.{d}.mlp.experts.down_proj_scales", "language_model.model.layers.{d}.mlp.experts.down_proj_scales" }, .module_type = "_Experts", .layout = .none, .dtype = "uint8", .required = true },
    .{ .id = "mlp.experts.down_proj_bias", .candidates = &.{ "model.layers.{d}.mlp.experts.down_proj_bias", "layers.{d}.mlp.experts.down_proj_bias", "transformer.h.{d}.mlp.experts.down_proj_bias", "backbone.layers.{d}.mlp.experts.down_proj_bias", "language_model.model.layers.{d}.mlp.experts.down_proj_bias" }, .module_type = "_Experts", .layout = .none, .dtype = "float32", .required = true },
};
const gpt_oss_full_attention_ops = [_]types.Op{
    .{ .op_type = .norm, .name = "input_layernorm", .inputs = &.{ .{ .tensor = "x" }, .{ .tensor = "input_layernorm.weight" } }, .outputs = &.{"_t0"} },
    .{ .op_type = .multihead_attention, .inputs = &.{.{ .tensor = "_t0" }}, .outputs = &.{"_t1"} },
    .{ .op_type = .add, .inputs = &.{ .{ .tensor = "x" }, .{ .tensor = "_t1" } }, .outputs = &.{"_t2"} },
    .{ .op_type = .norm, .name = "post_attention_layernorm", .inputs = &.{ .{ .tensor = "_t2" }, .{ .tensor = "post_attention_layernorm.weight" } }, .outputs = &.{"_t3"} },
    .{ .op_type = .moe, .inputs = &.{.{ .tensor = "_t3" }}, .outputs = &.{"_t4"}, .activation = "swiglu_oss", .num_experts = 32, .experts_per_token = 4 },
    .{ .op_type = .add, .inputs = &.{ .{ .tensor = "_t2" }, .{ .tensor = "_t4" } }, .outputs = &.{"_t5"} },
};
const gpt_oss_full_attention_weights = [_]types.WeightSpec{
    .{ .id = "input_layernorm.weight", .candidates = &.{ "model.layers.{d}.input_layernorm.weight", "layers.{d}.input_layernorm.weight", "transformer.h.{d}.input_layernorm.weight", "backbone.layers.{d}.input_layernorm.weight", "language_model.model.layers.{d}.input_layernorm.weight" }, .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "self_attn.sinks", .candidates = &.{ "model.layers.{d}.self_attn.sinks", "layers.{d}.self_attn.sinks", "transformer.h.{d}.self_attn.sinks", "backbone.layers.{d}.self_attn.sinks", "language_model.model.layers.{d}.self_attn.sinks" }, .module_type = "Attention", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "self_attn.q_proj.weight", .candidates = &.{ "model.layers.{d}.self_attn.q_proj.weight", "layers.{d}.self_attn.q_proj.weight", "transformer.h.{d}.self_attn.q_proj.weight", "backbone.layers.{d}.self_attn.q_proj.weight", "language_model.model.layers.{d}.self_attn.q_proj.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.q_proj.bias", .candidates = &.{ "model.layers.{d}.self_attn.q_proj.bias", "layers.{d}.self_attn.q_proj.bias", "transformer.h.{d}.self_attn.q_proj.bias", "backbone.layers.{d}.self_attn.q_proj.bias", "language_model.model.layers.{d}.self_attn.q_proj.bias" }, .module_type = "Linear", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "self_attn.k_proj.weight", .candidates = &.{ "model.layers.{d}.self_attn.k_proj.weight", "layers.{d}.self_attn.k_proj.weight", "transformer.h.{d}.self_attn.k_proj.weight", "backbone.layers.{d}.self_attn.k_proj.weight", "language_model.model.layers.{d}.self_attn.k_proj.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.k_proj.bias", .candidates = &.{ "model.layers.{d}.self_attn.k_proj.bias", "layers.{d}.self_attn.k_proj.bias", "transformer.h.{d}.self_attn.k_proj.bias", "backbone.layers.{d}.self_attn.k_proj.bias", "language_model.model.layers.{d}.self_attn.k_proj.bias" }, .module_type = "Linear", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "self_attn.v_proj.weight", .candidates = &.{ "model.layers.{d}.self_attn.v_proj.weight", "layers.{d}.self_attn.v_proj.weight", "transformer.h.{d}.self_attn.v_proj.weight", "backbone.layers.{d}.self_attn.v_proj.weight", "language_model.model.layers.{d}.self_attn.v_proj.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.v_proj.bias", .candidates = &.{ "model.layers.{d}.self_attn.v_proj.bias", "layers.{d}.self_attn.v_proj.bias", "transformer.h.{d}.self_attn.v_proj.bias", "backbone.layers.{d}.self_attn.v_proj.bias", "language_model.model.layers.{d}.self_attn.v_proj.bias" }, .module_type = "Linear", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "self_attn.o_proj.weight", .candidates = &.{ "model.layers.{d}.self_attn.o_proj.weight", "layers.{d}.self_attn.o_proj.weight", "transformer.h.{d}.self_attn.o_proj.weight", "backbone.layers.{d}.self_attn.o_proj.weight", "language_model.model.layers.{d}.self_attn.o_proj.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.o_proj.bias", .candidates = &.{ "model.layers.{d}.self_attn.o_proj.bias", "layers.{d}.self_attn.o_proj.bias", "transformer.h.{d}.self_attn.o_proj.bias", "backbone.layers.{d}.self_attn.o_proj.bias", "language_model.model.layers.{d}.self_attn.o_proj.bias" }, .module_type = "Linear", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "post_attention_layernorm.weight", .candidates = &.{ "model.layers.{d}.post_attention_layernorm.weight", "layers.{d}.post_attention_layernorm.weight", "transformer.h.{d}.post_attention_layernorm.weight", "backbone.layers.{d}.post_attention_layernorm.weight", "language_model.model.layers.{d}.post_attention_layernorm.weight" }, .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "mlp.router.weight", .candidates = &.{ "model.layers.{d}.mlp.router.weight", "layers.{d}.mlp.router.weight", "transformer.h.{d}.mlp.router.weight", "backbone.layers.{d}.mlp.router.weight", "language_model.model.layers.{d}.mlp.router.weight" }, .module_type = "_TopKRouter", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "mlp.router.bias", .candidates = &.{ "model.layers.{d}.mlp.router.bias", "layers.{d}.mlp.router.bias", "transformer.h.{d}.mlp.router.bias", "backbone.layers.{d}.mlp.router.bias", "language_model.model.layers.{d}.mlp.router.bias" }, .module_type = "_TopKRouter", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "mlp.experts.gate_up_proj_blocks", .candidates = &.{ "model.layers.{d}.mlp.experts.gate_up_proj_blocks", "layers.{d}.mlp.experts.gate_up_proj_blocks", "transformer.h.{d}.mlp.experts.gate_up_proj_blocks", "backbone.layers.{d}.mlp.experts.gate_up_proj_blocks", "language_model.model.layers.{d}.mlp.experts.gate_up_proj_blocks" }, .module_type = "_Experts", .layout = .none, .dtype = "mxfp4", .required = true },
    .{ .id = "mlp.experts.gate_up_proj_scales", .candidates = &.{ "model.layers.{d}.mlp.experts.gate_up_proj_scales", "layers.{d}.mlp.experts.gate_up_proj_scales", "transformer.h.{d}.mlp.experts.gate_up_proj_scales", "backbone.layers.{d}.mlp.experts.gate_up_proj_scales", "language_model.model.layers.{d}.mlp.experts.gate_up_proj_scales" }, .module_type = "_Experts", .layout = .none, .dtype = "uint8", .required = true },
    .{ .id = "mlp.experts.gate_up_proj_bias", .candidates = &.{ "model.layers.{d}.mlp.experts.gate_up_proj_bias", "layers.{d}.mlp.experts.gate_up_proj_bias", "transformer.h.{d}.mlp.experts.gate_up_proj_bias", "backbone.layers.{d}.mlp.experts.gate_up_proj_bias", "language_model.model.layers.{d}.mlp.experts.gate_up_proj_bias" }, .module_type = "_Experts", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "mlp.experts.down_proj_blocks", .candidates = &.{ "model.layers.{d}.mlp.experts.down_proj_blocks", "layers.{d}.mlp.experts.down_proj_blocks", "transformer.h.{d}.mlp.experts.down_proj_blocks", "backbone.layers.{d}.mlp.experts.down_proj_blocks", "language_model.model.layers.{d}.mlp.experts.down_proj_blocks" }, .module_type = "_Experts", .layout = .none, .dtype = "mxfp4", .required = true },
    .{ .id = "mlp.experts.down_proj_scales", .candidates = &.{ "model.layers.{d}.mlp.experts.down_proj_scales", "layers.{d}.mlp.experts.down_proj_scales", "transformer.h.{d}.mlp.experts.down_proj_scales", "backbone.layers.{d}.mlp.experts.down_proj_scales", "language_model.model.layers.{d}.mlp.experts.down_proj_scales" }, .module_type = "_Experts", .layout = .none, .dtype = "uint8", .required = true },
    .{ .id = "mlp.experts.down_proj_bias", .candidates = &.{ "model.layers.{d}.mlp.experts.down_proj_bias", "layers.{d}.mlp.experts.down_proj_bias", "transformer.h.{d}.mlp.experts.down_proj_bias", "backbone.layers.{d}.mlp.experts.down_proj_bias", "language_model.model.layers.{d}.mlp.experts.down_proj_bias" }, .module_type = "_Experts", .layout = .none, .dtype = "float32", .required = true },
};
var gpt_oss_block_variants = [_]types.BlockVariant{
    .{ .name = "sliding_attention", .ops = &gpt_oss_sliding_attention_ops, .weights = &gpt_oss_sliding_attention_weights },
    .{ .name = "full_attention", .ops = &gpt_oss_full_attention_ops, .weights = &gpt_oss_full_attention_weights },
};
const gpt_oss_global_weights = [_]types.WeightSpec{
    .{ .id = "token_embeddings", .candidates = &.{ "model.embed_tokens.weight", "embed_tokens.weight", "transformer.wte.weight", "backbone.embedding.weight", "language_model.model.embed_tokens.weight" }, .module_type = "Embedding", .layout = .embedding, .dtype = "float32", .required = true },
    .{ .id = "ln_final", .candidates = &.{ "model.norm.weight", "norm.weight", "transformer.ln_f.weight", "backbone.norm.weight", "language_model.model.norm.weight", "model.embedding_norm.weight" }, .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "lm_head", .candidates = &.{ "lm_head.weight", "output.weight", "transformer.lm_head.weight", "language_model.lm_head.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = false },
};
pub var arch: types.Architecture = .{
    .name = "gpt_oss",
    .model_types = &gpt_oss_model_types,
    .block_ops = &gpt_oss_block_ops,
    .pre_block_ops = &gpt_oss_pre_block_ops,
    .post_block_ops = &gpt_oss_post_block_ops,
    .block_variants = &gpt_oss_block_variants,
    .layer_map = &gpt_oss_layer_map,
    .variant_aliases = null,
    .block_weights = &gpt_oss_block_weights,
    .global_weights = &gpt_oss_global_weights,
    .weight_prefixes = &gpt_oss_weight_prefixes,
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
