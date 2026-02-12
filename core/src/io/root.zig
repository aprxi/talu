//! I/O Subsystem - model loading, storage, config parsing.
//!
//! This is the single entry point for the io module. All external code should
//! import from here.
//!
//! ## Public API
//!
//! - `loadModel()` - Load a model from disk (SafeTensors format)
//! - `repository` - Model repository format (HF-style structure)
//! - `transport` - Model downloads (HTTP, HuggingFace Hub API)
//! - `config` - Model configuration parsing
//!
//! ## Internal API (for core/src/ only)
//!
//! - `weights` - Low-level weight loading
//! - `loader` - Low-level loader module
//! - `safetensors` - SafeTensors format parsing

const std = @import("std");
const loader_mod = @import("loader/root.zig");
const weights_mod = @import("loader/weights.zig");
const validation = @import("loader/validation.zig");
const graph = @import("../graph/root.zig");
const inference_mod = @import("../inference/root.zig");
const transformer = inference_mod.executor;
const capi = @import("../capi/error.zig");

// =============================================================================
// Public API
// =============================================================================

pub const LoadedModel = weights_mod.LoadedModel;

/// Load a model from disk (SafeTensors format).
pub const loadModel = loader_mod.loadModel;

/// Load architecture definitions from _graphs/ directory.
pub const loadArchitectureDefinitions = loader_mod.loadArchitectureDefinitions;

/// Validate a loaded model's weight shapes against config.
pub const validateLoadedModel = validation.validate;

const log = @import("../log.zig");

/// Get the LayerOp program for a loaded model.
pub fn blockProgramForModel(loaded: *LoadedModel) ![]const transformer.LayerOp {
    graph.init(std.heap.page_allocator);

    log.debug("load", "blockProgramForModel called", .{
        .has_runtime_arch = loaded.runtime_arch != null,
    }, @src());

    if (loaded.runtime_arch) |runtime_arch_ptr| {
        const arch: *graph.Architecture = @ptrCast(@alignCast(runtime_arch_ptr));
        return try graph.ensureCompiled(arch);
    }

    capi.setContext("Run `make graphs` to generate architecture.json files from Python definitions", .{});
    return error.MissingArchitecture;
}

/// Get the LayerOp program for a specific layer of a loaded model.
/// For heterogeneous models, returns the appropriate variant's program.
/// For homogeneous models, returns the same program for all layers.
pub fn blockProgramForLayer(loaded: *LoadedModel, layer_idx: usize) ![]const transformer.LayerOp {
    graph.init(std.heap.page_allocator);

    if (loaded.runtime_arch) |runtime_arch_ptr| {
        const arch: *graph.Architecture = @ptrCast(@alignCast(runtime_arch_ptr));
        // Use model's layer_types if available (for different-sized models of same architecture)
        return try graph.ensureCompiledForLayerWithOverride(arch, layer_idx, loaded.config.layer_types);
    }

    capi.setContext("Run `make graphs` to generate architecture.json files from Python definitions", .{});
    return error.MissingArchitecture;
}

/// Check if a loaded model is heterogeneous (has multiple block variants).
pub fn isHeterogeneousModel(loaded: *LoadedModel) bool {
    if (loaded.runtime_arch) |runtime_arch_ptr| {
        const arch: *graph.Architecture = @ptrCast(@alignCast(runtime_arch_ptr));
        return arch.isHeterogeneous();
    }
    return false;
}

// NOTE: Architecture detection is now handled by graph.detectFromModelType().
// The hardcoded detectFromModelType/getMLXModelType functions have been removed.
// Use graph.detectFromModelType() to look up architectures by model_type string.

/// Model repository format (HF-style structure).
pub const repository = @import("repository/root.zig");

/// Transport layer (HTTP, HuggingFace Hub API).
pub const transport = @import("transport/root.zig");

/// Model configuration parsing (config.json).
pub const config = @import("config/root.zig");

/// Plugin discovery (UI plugin scanner).
pub const plugins = @import("plugins/root.zig");

// =============================================================================
// Internal API (for core/src/ only)
// =============================================================================

/// Low-level weight loading.
pub const weights = weights_mod;

/// Low-level loader module.
pub const loader = loader_mod;

/// MoE (Mixture of Experts) loading.
pub const moe = @import("loader/moe.zig");

/// SafeTensors format parsing.
pub const safetensors = struct {
    pub const root = @import("safetensors/root.zig");
    pub const names = @import("safetensors/names.zig");
    pub const norm_loader = @import("safetensors/norm_loader.zig");
};

/// JSON value extraction helpers.
pub const json_helpers = @import("json_helpers.zig");

/// JSON parsing with centralized size limits and error mapping.
pub const json = @import("json/root.zig");

/// KvBuf (Key-Value Buffer) binary format for zero-copy field access.
pub const kvbuf = @import("kvbuf/root.zig");
