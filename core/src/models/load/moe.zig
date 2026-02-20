//! MoE Config Detection
//!
//! Provides hooks for detecting Mixture of Experts configuration from weights
//! when config.json doesn't specify num_experts/experts_per_token.
//!
const std = @import("std");
const tensor = @import("../../tensor.zig");
const st_loader = @import("../../io/safetensors/root.zig");
const log = @import("../../log.zig");

// MoE weight naming patterns for detection
const experts_gate_up_blocks_name_fmt = "model.layers.{d}.mlp.experts.gate_up_proj_blocks";
const experts_gate_proj_weight_name_fmt = "model.layers.{d}.mlp.experts.gate_proj.weight";
const indexed_expert_gate_proj_weight_fmt = "model.layers.{d}.mlp.experts.{d}.gate_proj.weight";

/// MoE configuration detection hooks.
pub const MoEHooks = struct {
    /// Detect MoE settings from weights when config.json omits them.
    pub fn inferMoEFromWeights(safetensors: *st_loader.UnifiedSafeTensors, model_config: *tensor.ModelConfig) void {
        inferMoEFromWeightsImpl(safetensors, model_config);
    }
};

fn inferMoEFromWeightsImpl(safetensors: *st_loader.UnifiedSafeTensors, model_config: *tensor.ModelConfig) void {
    if (model_config.num_experts > 0) return;

    var name_buffer: [256]u8 = undefined;

    // Try fused format first (MXFP4 style)
    const gate_up_blocks_name = std.fmt.bufPrint(&name_buffer, experts_gate_up_blocks_name_fmt, .{0}) catch return;
    if (safetensors.getTensor(gate_up_blocks_name, null)) |gate_up_blocks| {
        if (gate_up_blocks.n_dims > 0) {
            model_config.num_experts = @intCast(gate_up_blocks.shape[0]);
            if (model_config.experts_per_token <= 0) model_config.experts_per_token = 4;
            log.debug("load", "Inferred MoE from fused weights", .{
                .num_experts = model_config.num_experts,
                .experts_per_token = model_config.experts_per_token,
            }, @src());
            return;
        }
    } else |_| {}

    // Try stacked format (separate gate/up)
    const gate_name = std.fmt.bufPrint(&name_buffer, experts_gate_proj_weight_name_fmt, .{0}) catch return;
    if (safetensors.getTensor(gate_name, null)) |gate_weights| {
        if (gate_weights.n_dims > 0) {
            model_config.num_experts = @intCast(gate_weights.shape[0]);
            if (model_config.experts_per_token <= 0) model_config.experts_per_token = 4;
            log.debug("load", "Inferred MoE from stacked weights", .{
                .num_experts = model_config.num_experts,
                .experts_per_token = model_config.experts_per_token,
            }, @src());
            return;
        }
    } else |_| {}

    // Try indexed format (Qwen3 MoE style: experts.0.gate_proj.weight, etc.)
    // Count how many experts exist by probing
    var expert_count: i32 = 0;
    for (0..256) |e| { // Max 256 experts
        const probe_name = std.fmt.bufPrint(&name_buffer, indexed_expert_gate_proj_weight_fmt, .{ 0, e }) catch break;
        _ = safetensors.getTensor(probe_name, null) catch break;
        expert_count += 1;
    }

    if (expert_count > 0) {
        model_config.num_experts = expert_count;
        if (model_config.experts_per_token <= 0) model_config.experts_per_token = 4;
        log.debug("load", "Inferred MoE from indexed weights", .{
            .num_experts = model_config.num_experts,
            .experts_per_token = model_config.experts_per_token,
        }, @src());
    }
}
