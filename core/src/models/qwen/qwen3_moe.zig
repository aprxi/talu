//! Qwen3-MoE model-version metadata.

const std = @import("std");
const layer_ops = @import("../layer_ops.zig");
const types = @import("../op_types.zig");

pub const id: []const u8 = "qwen3_moe";
pub const family: []const u8 = "qwen";
pub const version: []const u8 = "qwen3_moe";
pub const model_types: []const []const u8 = &.{
    "qwen3_moe",
};

/// Static block topology for Qwen3-MoE decoder blocks.
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
const qwen3_moe_model_types = [_][]const u8{"qwen3_moe"};
const qwen3_moe_weight_prefixes = [_][]const u8{
    "model.layers.{d}.",
    "layers.{d}.",
    "transformer.h.{d}.",
    "backbone.layers.{d}.",
    "language_model.model.layers.{d}.",
};

fn layerCandidates(comptime suffix: []const u8) []const []const u8 {
    return &.{
        "model.layers.{d}." ++ suffix,
        "layers.{d}." ++ suffix,
        "transformer.h.{d}." ++ suffix,
        "backbone.layers.{d}." ++ suffix,
        "language_model.model.layers.{d}." ++ suffix,
    };
}

fn requiredLayerWeight(comptime id_suffix: []const u8, comptime module_type: []const u8, comptime layout: types.WeightLayout) types.WeightSpec {
    return .{
        .id = id_suffix,
        .candidates = layerCandidates(id_suffix),
        .module_type = module_type,
        .layout = layout,
        .dtype = "float32",
        .required = true,
    };
}

fn buildExpertWeights(comptime expert_count: usize) [expert_count * 3]types.WeightSpec {
    @setEvalBranchQuota(expert_count * 300);
    var specs: [expert_count * 3]types.WeightSpec = undefined;
    comptime var out_i: usize = 0;
    inline for (0..expert_count) |expert_idx| {
        const prefix = std.fmt.comptimePrint("mlp.experts.{d}.", .{expert_idx});
        specs[out_i] = requiredLayerWeight(prefix ++ "gate_proj.weight", "Linear", .linear);
        out_i += 1;
        specs[out_i] = requiredLayerWeight(prefix ++ "up_proj.weight", "Linear", .linear);
        out_i += 1;
        specs[out_i] = requiredLayerWeight(prefix ++ "down_proj.weight", "Linear", .linear);
        out_i += 1;
    }
    return specs;
}

const qwen3_moe_pre_block_ops = [_]types.Op{
    .{ .op_type = .embedding, .inputs = &.{.{ .tensor = "input_ids" }}, .outputs = &.{"_t0"} },
};
const qwen3_moe_post_block_ops = [_]types.Op{
    .{ .op_type = .norm, .inputs = &.{.{ .tensor = "_t_last" }}, .outputs = &.{"_t_out"} },
};
const qwen3_moe_block_ops = [_]types.Op{
    .{ .op_type = .norm, .name = "input_layernorm", .inputs = &.{ .{ .tensor = "x" }, .{ .tensor = "input_layernorm.weight" } }, .outputs = &.{"_t0"} },
    .{ .op_type = .multihead_attention, .inputs = &.{.{ .tensor = "_t0" }}, .outputs = &.{"_t1"}, .qk_norm = true },
    .{ .op_type = .add, .inputs = &.{ .{ .tensor = "x" }, .{ .tensor = "_t1" } }, .outputs = &.{"_t2"} },
    .{ .op_type = .norm, .name = "post_attention_layernorm", .inputs = &.{ .{ .tensor = "_t2" }, .{ .tensor = "post_attention_layernorm.weight" } }, .outputs = &.{"_t3"} },
    .{ .op_type = .moe, .inputs = &.{.{ .tensor = "_t3" }}, .outputs = &.{"_t4"}, .num_experts = 128, .experts_per_token = 8 },
    .{ .op_type = .add, .inputs = &.{ .{ .tensor = "_t2" }, .{ .tensor = "_t4" } }, .outputs = &.{"_t5"} },
};

const qwen3_moe_common_weights = [_]types.WeightSpec{
    requiredLayerWeight("input_layernorm.weight", "RMSNorm", .none),
    requiredLayerWeight("self_attn.q_proj.weight", "Linear", .linear),
    requiredLayerWeight("self_attn.k_proj.weight", "Linear", .linear),
    requiredLayerWeight("self_attn.v_proj.weight", "Linear", .linear),
    requiredLayerWeight("self_attn.o_proj.weight", "Linear", .linear),
    requiredLayerWeight("self_attn.q_norm.weight", "RMSNorm", .none),
    requiredLayerWeight("self_attn.k_norm.weight", "RMSNorm", .none),
    requiredLayerWeight("post_attention_layernorm.weight", "RMSNorm", .none),
    requiredLayerWeight("mlp.gate.weight", "Linear", .linear),
};
const qwen3_moe_expert_weights = buildExpertWeights(128);
const qwen3_moe_block_weights = qwen3_moe_common_weights ++ qwen3_moe_expert_weights;

const qwen3_moe_global_weights = [_]types.WeightSpec{
    .{ .id = "token_embeddings", .candidates = &.{ "model.embed_tokens.weight", "embed_tokens.weight", "transformer.wte.weight", "backbone.embedding.weight", "language_model.model.embed_tokens.weight" }, .module_type = "Embedding", .layout = .embedding, .dtype = "float32", .required = true },
    .{ .id = "ln_final", .candidates = &.{ "model.norm.weight", "norm.weight", "transformer.ln_f.weight", "backbone.norm.weight", "language_model.model.norm.weight", "model.embedding_norm.weight" }, .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "lm_head", .candidates = &.{ "lm_head.weight", "output.weight", "transformer.lm_head.weight", "language_model.lm_head.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = false },
};

pub var arch: types.Architecture = .{
    .name = "qwen3_moe",
    .model_types = &qwen3_moe_model_types,
    .block_ops = &qwen3_moe_block_ops,
    .pre_block_ops = &qwen3_moe_pre_block_ops,
    .post_block_ops = &qwen3_moe_post_block_ops,
    .block_variants = null,
    .layer_map = null,
    .variant_aliases = null,
    .block_weights = &qwen3_moe_block_weights,
    .global_weights = &qwen3_moe_global_weights,
    .weight_prefixes = &qwen3_moe_weight_prefixes,
    .has_qk_norm = true,
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
