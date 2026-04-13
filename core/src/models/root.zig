//! Static models subsystem root.
//!
//! Module boundary:
//! - `models/` is the only owner of architecture metadata, config parsing,
//!   and model weight-loading contracts.
//! - Inference and converter code consume this module instead of legacy paths.

const std = @import("std");
const loader = @import("loader.zig");
const progress_mod = @import("progress_pkg");

pub const registry = @import("registry.zig");
pub const layer_ops = @import("layer_ops.zig");
pub const op_types = @import("op_types.zig");
pub const perf_hints = @import("perf_hints.zig");
pub const runtime_blocks = @import("runtime_blocks.zig");
pub const rope_scaling = @import("rope_scaling.zig");
pub const vision = @import("vision.zig");
pub const plan = @import("plan/root.zig");
pub const config = @import("config/root.zig");
pub const load = struct {
    pub const transforms = @import("load/transforms.zig");
};
pub const common = struct {
    pub const types = @import("common/types.zig");
};
pub const ModelDescriptor = registry.Entry;

pub const LoadedModel = loader.LoadedModel;
pub const LoadOptions = loader.LoadOptions;
pub const weights = loader.weights;
pub const Reporter = loader.Reporter;
pub const ResolvedModelKind = loader.ResolvedModelKind;

pub fn loadModel(
    allocator: std.mem.Allocator,
    config_path: []const u8,
    safetensors_path: []const u8,
    load_options: LoadOptions,
    progress: progress_mod.Context,
) !LoadedModel {
    return loader.loadModel(allocator, config_path, safetensors_path, load_options, progress);
}

/// Load model config and per-layer block types WITHOUT loading weight tensor data.
/// Reads safetensors headers for dtype/count metadata only. Suitable for backends
/// that load weights independently (e.g., Metal/MLX).
pub fn loadModelMetadataOnly(
    allocator: std.mem.Allocator,
    config_path: []const u8,
    safetensors_path: []const u8,
) !LoadedModel {
    _ = loadArchitectureDefinitions(allocator);
    const model_kind = try resolveModelKindForConfig(allocator, config_path);
    return loader.loadModelMetadataOnly(allocator, config_path, safetensors_path, model_kind.runtime_arch, model_kind.parse_config_hook);
}

pub fn loadArchitectureDefinitions(allocator: std.mem.Allocator) bool {
    return registry.loadArchitectureDefinitions(allocator);
}

pub fn resolveModelKindForConfig(
    allocator: std.mem.Allocator,
    config_path: []const u8,
) !ResolvedModelKind {
    return loader.resolveModelKindForConfig(allocator, config_path);
}

pub fn runtimeArchitectureForConfig(
    allocator: std.mem.Allocator,
    config_path: []const u8,
) !*const op_types.Architecture {
    return loader.runtimeArchitectureForConfig(allocator, config_path);
}

pub fn applyRuntimeArchitectureMetadata(
    loaded_model: *LoadedModel,
    runtime_architecture: *const op_types.Architecture,
) void {
    loader.applyRuntimeArchitectureMetadata(loaded_model, runtime_architecture);
}

pub fn detectByModelType(model_type: []const u8) ?ModelDescriptor {
    return registry.detectByModelType(model_type);
}

pub fn isSupportedModelType(model_type: []const u8) bool {
    return registry.isSupportedModelType(model_type);
}

pub fn performanceHintsByName(name: []const u8) ?*const perf_hints.PerfHints {
    return registry.performanceHintsByName(name);
}

test "models root delegates model type lookup to static registry" {
    try std.testing.expect(isSupportedModelType("llama3"));
    try std.testing.expect(!isSupportedModelType("not_a_real_model_type"));
}
