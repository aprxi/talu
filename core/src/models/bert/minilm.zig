//! MiniLM (BERT-family) model-version metadata.

const layer_ops = @import("../layer_ops.zig");
const types = @import("../op_types.zig");

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
const minilm_pre_block_ops = [_]types.Op{
    .{ .op_type = .embedding, .name = "word_embeddings", .inputs = &.{.{ .tensor = "input_ids" }}, .outputs = &.{"_t0"} },
    .{ .op_type = .embedding, .name = "position_embeddings", .inputs = &.{.{ .tensor = "position_ids" }}, .outputs = &.{"_t1"} },
    .{ .op_type = .embedding, .name = "token_type_embeddings", .inputs = &.{.{ .tensor = "token_type_ids" }}, .outputs = &.{"_t2"} },
    .{ .op_type = .add, .inputs = &.{ .{ .tensor = "_t0" }, .{ .tensor = "_t1" } }, .outputs = &.{"_t3"} },
    .{ .op_type = .add, .inputs = &.{ .{ .tensor = "_t3" }, .{ .tensor = "_t2" } }, .outputs = &.{"_t4"} },
    .{ .op_type = .norm, .inputs = &.{.{ .tensor = "_t4" }}, .outputs = &.{"_t5"} },
};
const minilm_post_block_ops = [_]types.Op{};
const minilm_block_ops = [_]types.Op{
    .{ .op_type = .multihead_attention, .inputs = &.{.{ .tensor = "x" }}, .outputs = &.{"_t0"}, .is_causal = false },
    .{ .op_type = .add, .inputs = &.{ .{ .tensor = "x" }, .{ .tensor = "_t0" } }, .outputs = &.{"_t1"} },
    .{ .op_type = .norm, .inputs = &.{.{ .tensor = "_t1" }}, .outputs = &.{"_t2"} },
    .{ .op_type = .mlp, .inputs = &.{.{ .tensor = "_t2" }}, .outputs = &.{"_t3"}, .activation = "gelu" },
    .{ .op_type = .add, .inputs = &.{ .{ .tensor = "_t2" }, .{ .tensor = "_t3" } }, .outputs = &.{"_t4"} },
    .{ .op_type = .norm, .inputs = &.{.{ .tensor = "_t4" }}, .outputs = &.{"_t5"} },
};
const minilm_block_weights = [_]types.WeightSpec{
    .{ .id = "self_attn.q_proj.weight", .candidates = &.{ "bert.encoder.layer.{d}.attention.self.query.weight", "encoder.layer.{d}.attention.self.query.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.q_proj.bias", .candidates = &.{ "bert.encoder.layer.{d}.attention.self.query.bias", "encoder.layer.{d}.attention.self.query.bias" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.k_proj.weight", .candidates = &.{ "bert.encoder.layer.{d}.attention.self.key.weight", "encoder.layer.{d}.attention.self.key.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.k_proj.bias", .candidates = &.{ "bert.encoder.layer.{d}.attention.self.key.bias", "encoder.layer.{d}.attention.self.key.bias" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.v_proj.weight", .candidates = &.{ "bert.encoder.layer.{d}.attention.self.value.weight", "encoder.layer.{d}.attention.self.value.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.v_proj.bias", .candidates = &.{ "bert.encoder.layer.{d}.attention.self.value.bias", "encoder.layer.{d}.attention.self.value.bias" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.o_proj.weight", .candidates = &.{ "bert.encoder.layer.{d}.attention.output.dense.weight", "encoder.layer.{d}.attention.output.dense.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "self_attn.o_proj.bias", .candidates = &.{ "bert.encoder.layer.{d}.attention.output.dense.bias", "encoder.layer.{d}.attention.output.dense.bias" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "input_layernorm.weight", .candidates = &.{ "bert.encoder.layer.{d}.attention.output.LayerNorm.weight", "encoder.layer.{d}.attention.output.LayerNorm.weight" }, .module_type = "LayerNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "input_layernorm.bias", .candidates = &.{ "bert.encoder.layer.{d}.attention.output.LayerNorm.bias", "encoder.layer.{d}.attention.output.LayerNorm.bias" }, .module_type = "LayerNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "mlp.dense_in.weight", .candidates = &.{ "bert.encoder.layer.{d}.intermediate.dense.weight", "encoder.layer.{d}.intermediate.dense.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "mlp.dense_in.bias", .candidates = &.{ "bert.encoder.layer.{d}.intermediate.dense.bias", "encoder.layer.{d}.intermediate.dense.bias" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "mlp.dense_out.weight", .candidates = &.{ "bert.encoder.layer.{d}.output.dense.weight", "encoder.layer.{d}.output.dense.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "mlp.dense_out.bias", .candidates = &.{ "bert.encoder.layer.{d}.output.dense.bias", "encoder.layer.{d}.output.dense.bias" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "post_attention_layernorm.weight", .candidates = &.{ "bert.encoder.layer.{d}.output.LayerNorm.weight", "encoder.layer.{d}.output.LayerNorm.weight" }, .module_type = "LayerNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "post_attention_layernorm.bias", .candidates = &.{ "bert.encoder.layer.{d}.output.LayerNorm.bias", "encoder.layer.{d}.output.LayerNorm.bias" }, .module_type = "LayerNorm", .layout = .none, .dtype = "float32", .required = true },
};
const minilm_global_weights = [_]types.WeightSpec{
    .{ .id = "token_embeddings", .candidates = &.{ "bert.embeddings.word_embeddings.weight", "embeddings.word_embeddings.weight" }, .module_type = "Embedding", .layout = .embedding, .dtype = "float32", .required = true },
    .{ .id = "position_embeddings", .candidates = &.{ "bert.embeddings.position_embeddings.weight", "embeddings.position_embeddings.weight" }, .module_type = "Embedding", .layout = .embedding, .dtype = "float32", .required = true },
    .{ .id = "token_type_embeddings", .candidates = &.{ "bert.embeddings.token_type_embeddings.weight", "embeddings.token_type_embeddings.weight" }, .module_type = "Embedding", .layout = .embedding, .dtype = "float32", .required = true },
    .{ .id = "embedding_ln", .candidates = &.{ "bert.embeddings.LayerNorm.weight", "embeddings.LayerNorm.weight" }, .module_type = "LayerNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "embedding_ln_bias", .candidates = &.{ "bert.embeddings.LayerNorm.bias", "embeddings.LayerNorm.bias" }, .module_type = "LayerNorm", .layout = .none, .dtype = "float32", .required = true },
};
pub var arch: types.Architecture = .{
    .name = "minilm",
    .model_types = &minilm_model_types,
    .block_ops = &minilm_block_ops,
    .pre_block_ops = &minilm_pre_block_ops,
    .post_block_ops = &minilm_post_block_ops,
    .block_variants = null,
    .layer_map = null,
    .variant_aliases = null,
    .block_weights = &minilm_block_weights,
    .global_weights = &minilm_global_weights,
    .weight_prefixes = &minilm_weight_prefixes,
    .d_ff_source_weight_ids = &.{"mlp.dense_in.weight"},
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
