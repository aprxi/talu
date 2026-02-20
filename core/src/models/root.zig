//! Static models subsystem root.
//!
//! Module boundary:
//! - `models/` is the only owner of architecture metadata, config parsing,
//!   and model weight-loading contracts.
//! - Inference and converter code consume this module instead of legacy paths.

const std = @import("std");
const loader = @import("loader.zig");
const progress_mod = @import("../capi/progress.zig");

pub const contract = @import("contract.zig");
pub const registry = @import("registry.zig");
pub const layer_ops = @import("layer_ops.zig");
pub const op_types = @import("op_types.zig");
pub const config = @import("config/root.zig");

pub const common = struct {
    pub const types = @import("common/types.zig");
    pub const weight_manifest = @import("common/weight_manifest.zig");
    pub const memory_requirements = @import("common/memory_requirements.zig");
};

// Compatibility exports during phased migration (old loader API surface).
pub const LoadedModel = loader.LoadedModel;
pub const LoadOptions = loader.LoadOptions;
pub const weights = loader.weights;
pub const Reporter = loader.Reporter;

pub fn loadModel(
    allocator: std.mem.Allocator,
    config_path: []const u8,
    safetensors_path: []const u8,
    load_options: LoadOptions,
    progress: progress_mod.ProgressContext,
) !LoadedModel {
    return loader.loadModel(allocator, config_path, safetensors_path, load_options, progress);
}

pub fn loadArchitectureDefinitions(allocator: std.mem.Allocator) bool {
    return registry.loadArchitectureDefinitions(allocator);
}

pub fn detectByModelType(model_type: []const u8) ?contract.ModelDescriptor {
    return registry.detectByModelType(model_type);
}

pub fn isSupportedModelType(model_type: []const u8) bool {
    return registry.isSupportedModelType(model_type);
}

test "models root delegates model type lookup to static registry" {
    try std.testing.expect(isSupportedModelType("llama3"));
    try std.testing.expect(!isSupportedModelType("not_a_real_model_type"));
}
