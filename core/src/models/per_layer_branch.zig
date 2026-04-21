//! Model-owned contract for the per-layer branch feature.

const std = @import("std");
const LoadedModel = @import("load/weights.zig").LoadedModel;
const gemma = @import("gemma/per_layer_branch.zig");

pub const Spec = struct {
    layer_prefixes: []const []const u8,
    per_layer_embedding_names: []const []const u8,
    per_layer_model_projection_names: []const []const u8,
    per_layer_projection_norm_names: []const []const u8,
    per_layer_input_gate_suffix: []const u8,
    per_layer_projection_suffix: []const u8,
    post_per_layer_input_norm_suffix: []const u8,
    layer_scalar_suffix: []const u8,
    per_layer_input_scale: f32,
};

const gemma_spec: Spec = .{
    .layer_prefixes = gemma.layer_prefixes,
    .per_layer_embedding_names = gemma.per_layer_embedding_names,
    .per_layer_model_projection_names = gemma.per_layer_model_projection_names,
    .per_layer_projection_norm_names = gemma.per_layer_projection_norm_names,
    .per_layer_input_gate_suffix = gemma.per_layer_input_gate_suffix,
    .per_layer_projection_suffix = gemma.per_layer_projection_suffix,
    .post_per_layer_input_norm_suffix = gemma.post_per_layer_input_norm_suffix,
    .layer_scalar_suffix = gemma.layer_scalar_suffix,
    .per_layer_input_scale = gemma.per_layer_input_scale,
};

pub fn specForLoadedModel(loaded: *const LoadedModel) ?Spec {
    return specForArchitectureId(loaded.runtime.architecture_id);
}

pub fn specForArchitectureId(architecture_id: ?[]const u8) ?Spec {
    if (gemma.supportsArchitectureId(architecture_id)) return gemma_spec;
    return null;
}

test "routes Gemma architectures to per-layer branch spec" {
    try std.testing.expect(specForArchitectureId("gemma3") != null);
    try std.testing.expect(specForArchitectureId("gemma4_moe") != null);
    try std.testing.expect(specForArchitectureId("llama3") == null);
}
