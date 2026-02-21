//! Granite Hybrid model-version metadata.

const layer_ops = @import("../layer_ops.zig");
const types = @import("../op_types.zig");

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

// Runtime architecture payload (migrated from runtime_architectures.zig)
const granite_hybrid_model_types = [_][]const u8{ "granite_hybrid", "granitehybrid", "granitemoehybrid" };
const granite_hybrid_weight_prefixes = [_][]const u8{ "model.layers.{d}.", "layers.{d}.", "transformer.h.{d}.", "backbone.layers.{d}.", "language_model.model.layers.{d}." };
const granite_hybrid_pre_block_ops = [_]types.Op{
    .{ .op_type = .embedding, .inputs = &.{.{ .tensor = "input_ids" }}, .outputs = &.{"_t0"} },
    .{ .op_type = .mul, .inputs = &.{ .{ .tensor = "_t0" }, .{ .scalar = 12.0 } }, .outputs = &.{"_t1"} },
};
const granite_hybrid_post_block_ops = [_]types.Op{
    .{ .op_type = .norm, .inputs = &.{.{ .tensor = "_t_last" }}, .outputs = &.{"_t_out"} },
};
const granite_hybrid_block_ops = [_]types.Op{};
const granite_hybrid_block_weights = [_]types.WeightSpec{};
const granite_hybrid_layer_map = [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 };
const granite_hybrid_mamba_ops = [_]types.Op{
    .{ .op_type = .norm, .name = "input_layernorm", .inputs = &.{ .{ .tensor = "x" }, .{ .tensor = "input_layernorm.weight" } }, .outputs = &.{"_t0"} },
    .{ .op_type = .mamba_mixer, .name = "mixer", .inputs = &.{.{ .tensor = "_t0" }}, .outputs = &.{"_t1"}, .d_state = 128, .d_conv = 4, .n_heads = 48, .d_head = 32, .n_groups = 1, .d_inner = 1536 },
    .{ .op_type = .mul, .inputs = &.{ .{ .tensor = "_t1" }, .{ .scalar = 0.246 } }, .outputs = &.{"_t2"} },
    .{ .op_type = .add, .inputs = &.{ .{ .tensor = "x" }, .{ .tensor = "_t2" } }, .outputs = &.{"_t3"} },
    .{ .op_type = .norm, .name = "post_attention_layernorm", .inputs = &.{ .{ .tensor = "_t3" }, .{ .tensor = "post_attention_layernorm.weight" } }, .outputs = &.{"_t4"} },
    .{ .op_type = .mlp, .inputs = &.{.{ .tensor = "_t4" }}, .outputs = &.{"_t5"}, .activation = "silu" },
    .{ .op_type = .mul, .inputs = &.{ .{ .tensor = "_t5" }, .{ .scalar = 0.246 } }, .outputs = &.{"_t6"} },
    .{ .op_type = .add, .inputs = &.{ .{ .tensor = "_t3" }, .{ .tensor = "_t6" } }, .outputs = &.{"_t7"} },
};
const granite_hybrid_mamba_weights = [_]types.WeightSpec{
    .{ .id = "input_layernorm.weight", .candidates = &.{ "model.layers.{d}.input_layernorm.weight", "layers.{d}.input_layernorm.weight", "transformer.h.{d}.input_layernorm.weight", "backbone.layers.{d}.input_layernorm.weight", "language_model.model.layers.{d}.input_layernorm.weight" }, .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "mixer.A_log", .candidates = &.{ "model.layers.{d}.mamba.A_log", "model.layers.{d}.mixer.A_log", "layers.{d}.mixer.A_log", "transformer.h.{d}.mixer.A_log", "backbone.layers.{d}.mixer.A_log", "language_model.model.layers.{d}.mixer.A_log" }, .module_type = "Mamba2", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "mixer.D", .candidates = &.{ "model.layers.{d}.mamba.D", "model.layers.{d}.mixer.D", "layers.{d}.mixer.D", "transformer.h.{d}.mixer.D", "backbone.layers.{d}.mixer.D", "language_model.model.layers.{d}.mixer.D" }, .module_type = "Mamba2", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "mixer.dt_bias", .candidates = &.{ "model.layers.{d}.mamba.dt_bias", "model.layers.{d}.mixer.dt_bias", "layers.{d}.mixer.dt_bias", "transformer.h.{d}.mixer.dt_bias", "backbone.layers.{d}.mixer.dt_bias", "language_model.model.layers.{d}.mixer.dt_bias" }, .module_type = "Mamba2", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "mixer.in_proj.weight", .candidates = &.{ "model.layers.{d}.mamba.in_proj.weight", "model.layers.{d}.mixer.in_proj.weight", "layers.{d}.mixer.in_proj.weight", "transformer.h.{d}.mixer.in_proj.weight", "backbone.layers.{d}.mixer.in_proj.weight", "language_model.model.layers.{d}.mixer.in_proj.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "mixer.conv1d.weight", .candidates = &.{ "model.layers.{d}.mamba.conv1d.weight", "model.layers.{d}.mixer.conv1d.weight", "layers.{d}.mixer.conv1d.weight", "transformer.h.{d}.mixer.conv1d.weight", "backbone.layers.{d}.mixer.conv1d.weight", "language_model.model.layers.{d}.mixer.conv1d.weight" }, .module_type = "Conv1d", .layout = .conv1d_depthwise, .dtype = "float32", .required = true },
    .{ .id = "mixer.conv1d.bias", .candidates = &.{ "model.layers.{d}.mamba.conv1d.bias", "model.layers.{d}.mixer.conv1d.bias", "layers.{d}.mixer.conv1d.bias", "transformer.h.{d}.mixer.conv1d.bias", "backbone.layers.{d}.mixer.conv1d.bias", "language_model.model.layers.{d}.mixer.conv1d.bias" }, .module_type = "Conv1d", .layout = .conv1d_depthwise, .dtype = "float32", .required = true },
    .{ .id = "mixer.norm.weight", .candidates = &.{ "model.layers.{d}.mamba.norm.weight", "model.layers.{d}.mixer.norm.weight", "layers.{d}.mixer.norm.weight", "transformer.h.{d}.mixer.norm.weight", "backbone.layers.{d}.mixer.norm.weight", "language_model.model.layers.{d}.mixer.norm.weight" }, .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true, .force_f32 = true },
    .{ .id = "mixer.out_proj.weight", .candidates = &.{ "model.layers.{d}.mamba.out_proj.weight", "model.layers.{d}.mixer.out_proj.weight", "layers.{d}.mixer.out_proj.weight", "transformer.h.{d}.mixer.out_proj.weight", "backbone.layers.{d}.mixer.out_proj.weight", "language_model.model.layers.{d}.mixer.out_proj.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "post_attention_layernorm.weight", .candidates = &.{ "model.layers.{d}.post_attention_layernorm.weight", "layers.{d}.post_attention_layernorm.weight", "transformer.h.{d}.post_attention_layernorm.weight", "backbone.layers.{d}.post_attention_layernorm.weight", "language_model.model.layers.{d}.post_attention_layernorm.weight" }, .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "mlp.input_linear.weight", .candidates = &.{ "model.layers.{d}.shared_mlp.input_linear.weight", "model.layers.{d}.mlp.input_linear.weight", "layers.{d}.mlp.input_linear.weight", "transformer.h.{d}.mlp.input_linear.weight", "backbone.layers.{d}.mlp.input_linear.weight", "language_model.model.layers.{d}.mlp.input_linear.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "mlp.output_linear.weight", .candidates = &.{ "model.layers.{d}.shared_mlp.output_linear.weight", "model.layers.{d}.mlp.output_linear.weight", "layers.{d}.mlp.output_linear.weight", "transformer.h.{d}.mlp.output_linear.weight", "backbone.layers.{d}.mlp.output_linear.weight", "language_model.model.layers.{d}.mlp.output_linear.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
};
const granite_hybrid_attention_ops = [_]types.Op{
    .{ .op_type = .norm, .name = "input_layernorm", .inputs = &.{ .{ .tensor = "x" }, .{ .tensor = "input_layernorm.weight" } }, .outputs = &.{"_t0"} },
    .{ .op_type = .multihead_attention, .inputs = &.{.{ .tensor = "_t0" }}, .outputs = &.{"_t1"} },
    .{ .op_type = .mul, .inputs = &.{ .{ .tensor = "_t1" }, .{ .scalar = 0.246 } }, .outputs = &.{"_t2"} },
    .{ .op_type = .add, .inputs = &.{ .{ .tensor = "x" }, .{ .tensor = "_t2" } }, .outputs = &.{"_t3"} },
    .{ .op_type = .norm, .name = "post_attention_layernorm", .inputs = &.{ .{ .tensor = "_t3" }, .{ .tensor = "post_attention_layernorm.weight" } }, .outputs = &.{"_t4"} },
    .{ .op_type = .mlp, .inputs = &.{.{ .tensor = "_t4" }}, .outputs = &.{"_t5"}, .activation = "silu" },
    .{ .op_type = .mul, .inputs = &.{ .{ .tensor = "_t5" }, .{ .scalar = 0.246 } }, .outputs = &.{"_t6"} },
    .{ .op_type = .add, .inputs = &.{ .{ .tensor = "_t3" }, .{ .tensor = "_t6" } }, .outputs = &.{"_t7"} },
};
const granite_hybrid_attention_weights = [_]types.WeightSpec{
    .{ .id = "input_layernorm.weight", .candidates = &.{ "model.layers.{d}.input_layernorm.weight", "layers.{d}.input_layernorm.weight", "transformer.h.{d}.input_layernorm.weight", "backbone.layers.{d}.input_layernorm.weight", "language_model.model.layers.{d}.input_layernorm.weight" }, .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "mixer.q_proj.weight", .candidates = &.{ "model.layers.{d}.self_attn.q_proj.weight", "model.layers.{d}.mixer.q_proj.weight", "layers.{d}.mixer.q_proj.weight", "transformer.h.{d}.mixer.q_proj.weight", "backbone.layers.{d}.mixer.q_proj.weight", "language_model.model.layers.{d}.mixer.q_proj.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "mixer.k_proj.weight", .candidates = &.{ "model.layers.{d}.self_attn.k_proj.weight", "model.layers.{d}.mixer.k_proj.weight", "layers.{d}.mixer.k_proj.weight", "transformer.h.{d}.mixer.k_proj.weight", "backbone.layers.{d}.mixer.k_proj.weight", "language_model.model.layers.{d}.mixer.k_proj.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "mixer.v_proj.weight", .candidates = &.{ "model.layers.{d}.self_attn.v_proj.weight", "model.layers.{d}.mixer.v_proj.weight", "layers.{d}.mixer.v_proj.weight", "transformer.h.{d}.mixer.v_proj.weight", "backbone.layers.{d}.mixer.v_proj.weight", "language_model.model.layers.{d}.mixer.v_proj.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "mixer.o_proj.weight", .candidates = &.{ "model.layers.{d}.self_attn.o_proj.weight", "model.layers.{d}.mixer.o_proj.weight", "layers.{d}.mixer.o_proj.weight", "transformer.h.{d}.mixer.o_proj.weight", "backbone.layers.{d}.mixer.o_proj.weight", "language_model.model.layers.{d}.mixer.o_proj.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "post_attention_layernorm.weight", .candidates = &.{ "model.layers.{d}.post_attention_layernorm.weight", "layers.{d}.post_attention_layernorm.weight", "transformer.h.{d}.post_attention_layernorm.weight", "backbone.layers.{d}.post_attention_layernorm.weight", "language_model.model.layers.{d}.post_attention_layernorm.weight" }, .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "mlp.input_linear.weight", .candidates = &.{ "model.layers.{d}.shared_mlp.input_linear.weight", "model.layers.{d}.mlp.input_linear.weight", "layers.{d}.mlp.input_linear.weight", "transformer.h.{d}.mlp.input_linear.weight", "backbone.layers.{d}.mlp.input_linear.weight", "language_model.model.layers.{d}.mlp.input_linear.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "mlp.output_linear.weight", .candidates = &.{ "model.layers.{d}.shared_mlp.output_linear.weight", "model.layers.{d}.mlp.output_linear.weight", "layers.{d}.mlp.output_linear.weight", "transformer.h.{d}.mlp.output_linear.weight", "backbone.layers.{d}.mlp.output_linear.weight", "language_model.model.layers.{d}.mlp.output_linear.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
};
var granite_hybrid_block_variants = [_]types.BlockVariant{
    .{ .name = "mamba", .ops = &granite_hybrid_mamba_ops, .weights = &granite_hybrid_mamba_weights },
    .{ .name = "attention", .ops = &granite_hybrid_attention_ops, .weights = &granite_hybrid_attention_weights },
};
const granite_hybrid_global_weights = [_]types.WeightSpec{
    .{ .id = "token_embeddings", .candidates = &.{ "model.embed_tokens.weight", "embed_tokens.weight", "transformer.wte.weight", "backbone.embedding.weight", "language_model.model.embed_tokens.weight" }, .module_type = "Embedding", .layout = .embedding, .dtype = "float32", .required = true },
    .{ .id = "ln_final", .candidates = &.{ "model.norm.weight", "norm.weight", "transformer.ln_f.weight", "backbone.norm.weight", "language_model.model.norm.weight", "model.embedding_norm.weight" }, .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "lm_head", .candidates = &.{ "lm_head.weight", "output.weight", "transformer.lm_head.weight", "language_model.lm_head.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = false },
};
pub var arch: types.Architecture = .{
    .name = "granite_hybrid",
    .model_types = &granite_hybrid_model_types,
    .block_ops = &granite_hybrid_block_ops,
    .pre_block_ops = &granite_hybrid_pre_block_ops,
    .post_block_ops = &granite_hybrid_post_block_ops,
    .block_variants = &granite_hybrid_block_variants,
    .layer_map = &granite_hybrid_layer_map,
    .variant_aliases = null,
    .block_weights = &granite_hybrid_block_weights,
    .global_weights = &granite_hybrid_global_weights,
    .weight_prefixes = &granite_hybrid_weight_prefixes,
    .d_ff_source_weight_ids = &.{"mlp.input_linear.weight"},
    .has_qk_norm = false,
    .has_moe = false,
    .has_mamba = true,
    .has_shortconv = false,
    .has_mla = false,
    .has_fused_qkv = false,
    .has_fused_gate_up = false,
    .num_norms_per_block = 2,
    .use_gelu = false,
    .use_swiglu_oss = false,
    .norm_weight_offset = 0.0,
    .explicit_qk_norm_ops = false,
    .embedding_multiplier = 12.0,
};
