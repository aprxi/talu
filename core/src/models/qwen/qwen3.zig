//! Qwen3 family model-version metadata.

const layer_ops = @import("../layer_ops.zig");
const types = @import("../op_types.zig");

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

// Runtime architecture payload (migrated from runtime_architectures.zig)
const qwen3_model_types = [_][]const u8{ "qwen3", "qwen3_vl", "qwen2.5", "qwen2", "qwen" };
const qwen3_weight_prefixes = [_][]const u8{ "model.layers.{d}.", "model.language_model.layers.{d}.", "layers.{d}.", "transformer.h.{d}.", "backbone.layers.{d}.", "language_model.model.layers.{d}." };
const qwen3_pre_block_ops = [_]types.Op{
    .{ .op_type = .embedding, .inputs = &.{.{ .tensor = "input_ids" }}, .outputs = &.{"_t0"} },
};
const qwen3_post_block_ops = [_]types.Op{
    .{ .op_type = .norm, .inputs = &.{.{ .tensor = "_t_last" }}, .outputs = &.{"_t_out"} },
};
const qwen3_block_ops = [_]types.Op{
    .{ .op_type = .norm, .name = "input_layernorm", .inputs = &.{ .{ .tensor = "x" }, .{ .tensor = "input_layernorm.weight" } }, .outputs = &.{"_t0"} },
    .{ .op_type = .multihead_attention, .inputs = &.{.{ .tensor = "_t0" }}, .outputs = &.{"_t1"}, .qk_norm = true },
    .{ .op_type = .add, .inputs = &.{ .{ .tensor = "x" }, .{ .tensor = "_t1" } }, .outputs = &.{"_t2"} },
    .{ .op_type = .norm, .name = "post_attention_layernorm", .inputs = &.{ .{ .tensor = "_t2" }, .{ .tensor = "post_attention_layernorm.weight" } }, .outputs = &.{"_t3"} },
    .{ .op_type = .mlp, .inputs = &.{.{ .tensor = "_t3" }}, .outputs = &.{"_t4"}, .activation = "silu" },
    .{ .op_type = .add, .inputs = &.{ .{ .tensor = "_t2" }, .{ .tensor = "_t4" } }, .outputs = &.{"_t5"} },
};
const qwen3_block_weights = [_]types.WeightSpec{
    .{ .id = "input_layernorm.weight", .candidates = &.{ "model.layers.{d}.input_layernorm.weight", "model.language_model.layers.{d}.input_layernorm.weight", "layers.{d}.input_layernorm.weight", "transformer.h.{d}.input_layernorm.weight", "backbone.layers.{d}.input_layernorm.weight", "language_model.model.layers.{d}.input_layernorm.weight" }, .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "self_attn.q_proj.weight", .candidates = &.{ "model.layers.{d}.self_attn.q_proj.weight", "model.language_model.layers.{d}.self_attn.q_proj.weight", "layers.{d}.self_attn.q_proj.weight", "transformer.h.{d}.self_attn.q_proj.weight", "backbone.layers.{d}.self_attn.q_proj.weight", "language_model.model.layers.{d}.self_attn.q_proj.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.k_proj.weight", .candidates = &.{ "model.layers.{d}.self_attn.k_proj.weight", "model.language_model.layers.{d}.self_attn.k_proj.weight", "layers.{d}.self_attn.k_proj.weight", "transformer.h.{d}.self_attn.k_proj.weight", "backbone.layers.{d}.self_attn.k_proj.weight", "language_model.model.layers.{d}.self_attn.k_proj.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.v_proj.weight", .candidates = &.{ "model.layers.{d}.self_attn.v_proj.weight", "model.language_model.layers.{d}.self_attn.v_proj.weight", "layers.{d}.self_attn.v_proj.weight", "transformer.h.{d}.self_attn.v_proj.weight", "backbone.layers.{d}.self_attn.v_proj.weight", "language_model.model.layers.{d}.self_attn.v_proj.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.o_proj.weight", .candidates = &.{ "model.layers.{d}.self_attn.o_proj.weight", "model.language_model.layers.{d}.self_attn.o_proj.weight", "layers.{d}.self_attn.o_proj.weight", "transformer.h.{d}.self_attn.o_proj.weight", "backbone.layers.{d}.self_attn.o_proj.weight", "language_model.model.layers.{d}.self_attn.o_proj.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.q_norm.weight", .candidates = &.{ "model.layers.{d}.self_attn.q_norm.weight", "model.language_model.layers.{d}.self_attn.q_norm.weight", "layers.{d}.self_attn.q_norm.weight", "transformer.h.{d}.self_attn.q_norm.weight", "backbone.layers.{d}.self_attn.q_norm.weight", "language_model.model.layers.{d}.self_attn.q_norm.weight" }, .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "self_attn.k_norm.weight", .candidates = &.{ "model.layers.{d}.self_attn.k_norm.weight", "model.language_model.layers.{d}.self_attn.k_norm.weight", "layers.{d}.self_attn.k_norm.weight", "transformer.h.{d}.self_attn.k_norm.weight", "backbone.layers.{d}.self_attn.k_norm.weight", "language_model.model.layers.{d}.self_attn.k_norm.weight" }, .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "post_attention_layernorm.weight", .candidates = &.{ "model.layers.{d}.post_attention_layernorm.weight", "model.language_model.layers.{d}.post_attention_layernorm.weight", "layers.{d}.post_attention_layernorm.weight", "transformer.h.{d}.post_attention_layernorm.weight", "backbone.layers.{d}.post_attention_layernorm.weight", "language_model.model.layers.{d}.post_attention_layernorm.weight" }, .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "mlp.gate_proj.weight", .candidates = &.{ "model.layers.{d}.mlp.gate_proj.weight", "model.language_model.layers.{d}.mlp.gate_proj.weight", "layers.{d}.mlp.gate_proj.weight", "transformer.h.{d}.mlp.gate_proj.weight", "backbone.layers.{d}.mlp.gate_proj.weight", "language_model.model.layers.{d}.mlp.gate_proj.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "mlp.up_proj.weight", .candidates = &.{ "model.layers.{d}.mlp.up_proj.weight", "model.language_model.layers.{d}.mlp.up_proj.weight", "layers.{d}.mlp.up_proj.weight", "transformer.h.{d}.mlp.up_proj.weight", "backbone.layers.{d}.mlp.up_proj.weight", "language_model.model.layers.{d}.mlp.up_proj.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "mlp.down_proj.weight", .candidates = &.{ "model.layers.{d}.mlp.down_proj.weight", "model.language_model.layers.{d}.mlp.down_proj.weight", "layers.{d}.mlp.down_proj.weight", "transformer.h.{d}.mlp.down_proj.weight", "backbone.layers.{d}.mlp.down_proj.weight", "language_model.model.layers.{d}.mlp.down_proj.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
};
const qwen3_global_weights = [_]types.WeightSpec{
    .{ .id = "token_embeddings", .candidates = &.{ "model.embed_tokens.weight", "model.language_model.embed_tokens.weight", "embed_tokens.weight", "transformer.wte.weight", "backbone.embedding.weight", "language_model.model.embed_tokens.weight" }, .module_type = "Embedding", .layout = .embedding, .dtype = "float32", .required = true },
    .{ .id = "ln_final", .candidates = &.{ "model.norm.weight", "model.language_model.norm.weight", "norm.weight", "transformer.ln_f.weight", "backbone.norm.weight", "language_model.model.norm.weight", "model.embedding_norm.weight" }, .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "lm_head", .candidates = &.{ "lm_head.weight", "output.weight", "transformer.lm_head.weight", "language_model.lm_head.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = false },
};
pub var arch: types.Architecture = .{
    .name = "qwen3",
    .model_types = &qwen3_model_types,
    .block_ops = &qwen3_block_ops,
    .pre_block_ops = &qwen3_pre_block_ops,
    .post_block_ops = &qwen3_post_block_ops,
    .block_variants = null,
    .layer_map = null,
    .variant_aliases = null,
    .block_weights = &qwen3_block_weights,
    .global_weights = &qwen3_global_weights,
    .weight_prefixes = &qwen3_weight_prefixes,
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
};
