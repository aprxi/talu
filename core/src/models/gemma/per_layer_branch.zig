//! Gemma-family metadata for the per-layer branch feature.

const std = @import("std");

pub const architecture_ids: []const []const u8 = &.{
    "gemma3",
    "gemma4_moe",
};

pub const layer_prefixes: []const []const u8 = &.{
    "model.language_model.layers",
    "model.layers",
    "language_model.model.layers",
    "layers",
};

pub const per_layer_embedding_names: []const []const u8 = &.{
    "model.language_model.embed_tokens_per_layer.weight",
    "model.embed_tokens_per_layer.weight",
    "embed_tokens_per_layer.weight",
};

pub const per_layer_model_projection_names: []const []const u8 = &.{
    "model.language_model.per_layer_model_projection.weight",
    "model.per_layer_model_projection.weight",
    "per_layer_model_projection.weight",
};

pub const per_layer_projection_norm_names: []const []const u8 = &.{
    "model.language_model.per_layer_projection_norm.weight",
    "model.per_layer_projection_norm.weight",
    "per_layer_projection_norm.weight",
};

pub const per_layer_input_gate_suffix = "per_layer_input_gate.weight";
pub const per_layer_projection_suffix = "per_layer_projection.weight";
pub const post_per_layer_input_norm_suffix = "post_per_layer_input_norm.weight";
pub const layer_scalar_suffix = "layer_scalar";

pub const per_layer_input_scale: f32 = 0.70710677;

pub fn supportsArchitectureId(architecture_id: ?[]const u8) bool {
    const id = architecture_id orelse return false;
    for (architecture_ids) |candidate| {
        if (std.mem.eql(u8, candidate, id)) return true;
    }
    return false;
}

test "supports Gemma per-layer branch architectures" {
    try std.testing.expect(supportsArchitectureId("gemma3"));
    try std.testing.expect(supportsArchitectureId("gemma4_moe"));
    try std.testing.expect(!supportsArchitectureId("qwen3"));
    try std.testing.expect(!supportsArchitectureId(null));
}
