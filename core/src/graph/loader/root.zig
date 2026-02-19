//! Graph loader compatibility wrapper.
//!
//! Source of truth lives in `core/src/models/loader.zig`.

const std = @import("std");
const progress_mod = @import("../../capi/progress.zig");
const models_loader = @import("../../models/loader.zig");

pub const weights = models_loader.weights;
pub const LoadedModel = models_loader.LoadedModel;
pub const LoadOptions = models_loader.LoadOptions;
pub const validateLoadedModel = models_loader.validateLoadedModel;
pub const Reporter = models_loader.Reporter;

pub fn loadModel(
    backing_allocator: std.mem.Allocator,
    config_path: []const u8,
    weights_path: []const u8,
    load_options: LoadOptions,
    progress: progress_mod.ProgressContext,
) !LoadedModel {
    return models_loader.loadModel(backing_allocator, config_path, weights_path, load_options, progress);
}

pub fn loadArchitectureDefinitions(allocator: std.mem.Allocator) bool {
    return models_loader.loadArchitectureDefinitions(allocator);
}
