//! YouTu-VL model-version metadata.

const std = @import("std");
const tensor = @import("../../tensor.zig");
const layer_ops = @import("../layer_ops.zig");
const types = @import("../op_types.zig");
const vision_shared = @import("../vision_shared.zig");
const config_hooks = @import("../config/hook_utils.zig");

pub const id: []const u8 = "youtu_vl";
pub const family: []const u8 = "youtu_vl";
pub const version: []const u8 = "youtu_vl";
pub const model_types: []const []const u8 = &.{
    "youtu_vl",
};

/// Static block topology for YouTu-VL text decoder blocks.
/// Attention kernel may resolve to MLA internally, but op type remains attention.
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

pub const vision_program = vision_shared.vision_program;

// Runtime architecture payload (migrated from runtime_architectures.zig)
const youtu_vl_model_types = [_][]const u8{"youtu_vl"};
const youtu_vl_weight_prefixes = [_][]const u8{ "model.layers.{d}.", "language_model.model.layers.{d}." };
const youtu_vl_block_weights = [_]types.WeightSpec{
    .{ .id = "input_layernorm.weight", .suffix = "input_layernorm.weight", .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "self_attn.q_a_proj.weight", .suffix = "self_attn.q_a_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true, .expected_shape = &.{ 1536, 2560 } },
    .{ .id = "self_attn.q_a_layernorm.weight", .suffix = "self_attn.q_a_layernorm.weight", .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "self_attn.q_b_proj.weight", .suffix = "self_attn.q_b_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true, .expected_shape = &.{ 6144, 1536 } },
    .{ .id = "self_attn.kv_a_proj_with_mqa.weight", .suffix = "self_attn.kv_a_proj_with_mqa.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true, .expected_shape = &.{ 576, 2560 } },
    .{ .id = "self_attn.kv_a_layernorm.weight", .suffix = "self_attn.kv_a_layernorm.weight", .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "self_attn.kv_b_proj.weight", .suffix = "self_attn.kv_b_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true, .expected_shape = &.{ 8192, 512 } },
    .{ .id = "self_attn.o_proj.weight", .suffix = "self_attn.o_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true, .expected_shape = &.{ 2560, 4096 } },
    .{ .id = "post_attention_layernorm.weight", .suffix = "post_attention_layernorm.weight", .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "mlp.gate_proj.weight", .suffix = "mlp.gate_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "mlp.up_proj.weight", .suffix = "mlp.up_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
    .{ .id = "mlp.down_proj.weight", .suffix = "mlp.down_proj.weight", .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = true },
};
const youtu_vl_global_weights = [_]types.WeightSpec{
    .{ .id = "token_embeddings", .suffix = "model.embed_tokens.weight", .aliases = &.{ "embed_tokens.weight", "transformer.wte.weight", "backbone.embedding.weight", "language_model.model.embed_tokens.weight" }, .module_type = "Embedding", .layout = .embedding, .dtype = "float32", .required = true },
    .{ .id = "ln_final", .suffix = "model.norm.weight", .aliases = &.{ "norm.weight", "transformer.ln_f.weight", "backbone.norm.weight", "language_model.model.norm.weight", "model.embedding_norm.weight" }, .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "lm_head", .suffix = "lm_head.weight", .aliases = &.{ "output.weight", "transformer.lm_head.weight", "language_model.lm_head.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = false },
};

fn parseConfigHook(
    config_obj: std.json.ObjectMap,
    root_obj: std.json.ObjectMap,
    config: *tensor.ModelConfig,
) void {
    config_hooks.applyCommonTextConfig(config_obj, root_obj, config);
    config_hooks.applyVisionConfig(root_obj, config);
}

pub var arch: types.Architecture = .{
    .name = "youtu_vl",
    .model_types = &youtu_vl_model_types,
    .parse_config_hook = parseConfigHook,
    .kernel_meta = .{
        .mla_config = .{
            .q_lora_rank = 1536,
            .kv_lora_rank = 512,
            .qk_head_dim = 192,
            .qk_rope_head_dim = 64,
            .qk_nope_head_dim = 128,
            .v_head_dim = 128,
            .rope_interleave = true,
        },
    },
    .block_variants = null,
    .layer_map = null,
    .variant_aliases = null,
    .block_weights = &youtu_vl_block_weights,
    .global_weights = &youtu_vl_global_weights,
    .weight_prefixes = &youtu_vl_weight_prefixes,
    .d_ff_source_weight_ids = &.{"mlp.gate_proj.weight"},
    .resolve_d_ff_from_weights = true,
    .has_qk_norm = false,
    .has_moe = false,
    .has_mamba = false,
    .has_shortconv = false,
    .has_mla = true,
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
