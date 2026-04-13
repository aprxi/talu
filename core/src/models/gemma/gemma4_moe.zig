//! Gemma4-MoE model-version metadata.
//!
//! Gemma4-26B-A4B uses shared MLP + 128 routed experts per layer.
//! The entire FFN block (shared MLP + routing + experts + internal norms)
//! is fused into a single .moe kernel because the layer program's 3-buffer
//! system cannot express the fork-join pattern otherwise.

const std = @import("std");
const tensor = @import("tensor_pkg");
const layer_ops = @import("models_pkg").layer_ops;
const types = @import("models_pkg").op_types;
const config_hooks = @import("../config/hook_utils.zig");
const perf = @import("../perf_hints.zig");
const sp = @import("../sampling_presets.zig");

pub const id: []const u8 = "gemma4_moe";
pub const family: []const u8 = "gemma";
pub const version: []const u8 = "gemma4_moe";
pub const model_types: []const []const u8 = &.{
    "gemma4_moe",
};

/// Layer program: norm → attn → norm → add → fused_moe → add.
/// The fused_moe kernel handles:
///   pre_feedforward_layernorm → shared MLP → post_feedforward_layernorm_1
///   + router(residual) → pre_feedforward_layernorm_2 → experts → post_feedforward_layernorm_2
///   → combine → post_feedforward_layernorm
pub const attention_moe_program: []const layer_ops.LayerOp = &.{
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
        .state_block_id = types.kv_cache_state_id,
    } },
    .{ .kernel = .{
        .id = 2,
        .in = .branch_out,
        .out = .norm_out,
        .debug_type = .norm,
    } },
    .{ .add = .{
        .branch = .norm_out,
        .scale = .residual_multiplier,
    } },
    .{ .kernel = .{
        .id = 3,
        .in = .residual,
        .out = .branch_out,
        .debug_type = .moe,
    } },
    .{ .add = .{
        .branch = .branch_out,
        .scale = .residual_multiplier,
    } },
};

fn requiredLayerWeight(
    comptime id_suffix: []const u8,
    comptime module_type: []const u8,
    comptime layout: types.WeightLayout,
) types.WeightSpec {
    return .{
        .id = id_suffix,
        .suffix = id_suffix,
        .module_type = module_type,
        .layout = layout,
        .dtype = "float32",
        .required = true,
    };
}

fn optionalLayerWeight(
    comptime id_suffix: []const u8,
    comptime module_type: []const u8,
    comptime layout: types.WeightLayout,
) types.WeightSpec {
    return .{
        .id = id_suffix,
        .suffix = id_suffix,
        .module_type = module_type,
        .layout = layout,
        .dtype = "float32",
        .required = false,
    };
}

// Weight specs for sliding attention layers (v_proj required).
const gemma4_moe_sliding_weights = [_]types.WeightSpec{
    // Norms: indices 0-6 (order matters for norm indexing)
    requiredLayerWeight("input_layernorm.weight", "RMSNorm", .none), // 0
    requiredLayerWeight("post_attention_layernorm.weight", "RMSNorm", .none), // 1
    requiredLayerWeight("pre_feedforward_layernorm.weight", "RMSNorm", .none), // 2
    requiredLayerWeight("post_feedforward_layernorm_1.weight", "RMSNorm", .none), // 3
    requiredLayerWeight("pre_feedforward_layernorm_2.weight", "RMSNorm", .none), // 4
    requiredLayerWeight("post_feedforward_layernorm_2.weight", "RMSNorm", .none), // 5
    requiredLayerWeight("post_feedforward_layernorm.weight", "RMSNorm", .none), // 6
    // Attention
    requiredLayerWeight("self_attn.q_proj.weight", "Linear", .linear),
    requiredLayerWeight("self_attn.k_proj.weight", "Linear", .linear),
    requiredLayerWeight("self_attn.v_proj.weight", "Linear", .linear),
    requiredLayerWeight("self_attn.o_proj.weight", "Linear", .linear),
    requiredLayerWeight("self_attn.q_norm.weight", "RMSNorm", .none),
    requiredLayerWeight("self_attn.k_norm.weight", "RMSNorm", .none),
    // Shared MLP
    requiredLayerWeight("mlp.gate_proj.weight", "Linear", .linear),
    requiredLayerWeight("mlp.up_proj.weight", "Linear", .linear),
    requiredLayerWeight("mlp.down_proj.weight", "Linear", .linear),
    // Router
    requiredLayerWeight("router.proj.weight", "Linear", .linear),
    requiredLayerWeight("router.scale", "Parameter", .none),
    requiredLayerWeight("router.per_expert_scale", "Parameter", .none),
    // Fused 3D expert weights (split into per-expert views at load time)
    requiredLayerWeight("experts.gate_up_proj", "Parameter", .fused_linear),
    requiredLayerWeight("experts.down_proj", "Parameter", .fused_linear),
};

// Full-attention variant: v_proj is optional (attention_k_eq_v shares K/V).
const gemma4_moe_full_weights = blk: {
    var weights = gemma4_moe_sliding_weights;
    for (&weights) |*w| {
        if (std.mem.eql(u8, w.id, "self_attn.v_proj.weight")) {
            w.required = false;
            break;
        }
    }
    break :blk weights;
};

var gemma4_moe_block_variants = [_]types.BlockVariant{
    .{ .name = "sliding_attention", .weights = &gemma4_moe_sliding_weights },
    .{ .name = "full_attention", .weights = &gemma4_moe_full_weights },
};

const gemma4_moe_block_weights = [_]types.WeightSpec{};

const gemma4_moe_global_weights = [_]types.WeightSpec{
    .{ .id = "token_embeddings", .suffix = "model.embed_tokens.weight", .aliases = &.{ "embed_tokens.weight", "language_model.model.embed_tokens.weight", "model.language_model.embed_tokens.weight" }, .module_type = "Embedding", .layout = .embedding, .dtype = "float32", .required = true },
    .{ .id = "ln_final", .suffix = "model.norm.weight", .aliases = &.{ "norm.weight", "language_model.model.norm.weight", "model.language_model.norm.weight" }, .module_type = "RMSNorm", .layout = .none, .dtype = "float32", .required = true },
    .{ .id = "lm_head", .suffix = "lm_head.weight", .aliases = &.{ "output.weight", "language_model.lm_head.weight", "model.language_model.lm_head.weight" }, .module_type = "Linear", .layout = .linear, .dtype = "float32", .required = false },
};

/// Config hook: apply Gemma4 defaults (pure RMS norms, sqrt(d_model) embedding
/// scaling, unit attention scaling).
fn gemma4MoeConfigHook(
    config_obj: std.json.ObjectMap,
    root_obj: std.json.ObjectMap,
    config: *tensor.ModelConfig,
) void {
    config_hooks.applyCommonTextConfig(config_obj, root_obj, config);

    config.use_raw_rms_norm = true;
    config.use_v_norm = true;
    if (config.embedding_multiplier == 1.0) {
        config.embedding_multiplier = @sqrt(@as(f32, @floatFromInt(config.d_model)));
    }
    if (config.attention_multiplier == 0.0 and config.query_pre_attn_scalar == 0.0) {
        config.attention_multiplier = 1.0;
    }
}

const gemma4_moe_weight_prefixes = [_][]const u8{
    "model.layers.{d}.",
    "layers.{d}.",
    "language_model.model.layers.{d}.",
    "model.language_model.layers.{d}.",
};

const gemma4_moe_perf_hints = perf.attentionOnlyHints("gemma4_moe");
const gemma4_moe_sampling_presets: sp.SamplingPresets = .{
    .general = .{ .temperature = 1.0, .top_p = 0.95, .top_k = 64, .presence_penalty = 0.0 },
    .coding = .{ .temperature = 0.6, .top_p = 0.95, .top_k = 64, .presence_penalty = 0.0 },
    .instruct = .{ .temperature = 0.7, .top_p = 0.8, .top_k = 64, .presence_penalty = 0.0 },
    .deterministic = .{ .temperature = 0.0, .top_p = 1.0, .top_k = 1, .presence_penalty = 0.0 },
};

pub var arch: types.Architecture = .{
    .name = "gemma4_moe",
    .model_types = &.{"gemma4_moe"},
    .parse_config_hook = gemma4MoeConfigHook,
    .block_variants = &gemma4_moe_block_variants,
    .layer_map = null,
    .variant_aliases = null,
    .block_weights = &gemma4_moe_block_weights,
    .global_weights = &gemma4_moe_global_weights,
    .weight_prefixes = &gemma4_moe_weight_prefixes,
    .d_ff_source_weight_ids = &.{"mlp.gate_proj.weight"},
    .resolve_d_ff_from_weights = false,
    .has_qk_norm = true,
    .has_moe = true,
    .has_mamba = false,
    .has_shortconv = false,
    .has_mla = false,
    .has_fused_qkv = false,
    .has_fused_gate_up = false,
    .num_norms_per_block = 7,
    .use_gelu = true,
    .use_swiglu_oss = false,
    .norm_weight_offset = 1.0,
    .explicit_qk_norm_ops = false,
    .norm_weights_pre_shifted = true,
    .embedding_multiplier = 1.0, // Overridden by config hook to sqrt(d_model)
    .performance_hints = &gemma4_moe_perf_hints,
    .sampling_presets = &gemma4_moe_sampling_presets,
};
