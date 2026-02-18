//! CPU graph bridge for inference execution.
//!
//! Exposes LayerOp/graph helpers used by CPU executor and CPU-backed vision
//! runtime without leaking these details into top-level inference exports.

const std = @import("std");
const graph = @import("../../../graph/root.zig");
const capi = @import("../../../capi/error.zig");

pub const layer_ops = @import("../../../graph/layer_ops.zig");
pub const compiler = @import("../../../graph/compiler.zig");
pub const graph_types = @import("../../../graph/types.zig");

pub const LayerOp = layer_ops.LayerOp;
pub const BufferId = layer_ops.BufferId;
pub const ResidualScale = layer_ops.ResidualScale;
pub const NormSlot = layer_ops.NormSlot;
pub const Op = graph_types.Op;
pub const OpType = graph_types.OpType;
pub const LoadedModel = graph.LoadedModel;

/// Get the LayerOp program for a loaded model.
pub fn blockProgramForModel(loaded: *LoadedModel) ![]const LayerOp {
    graph.init(std.heap.page_allocator);
    if (loaded.runtime_arch) |runtime_arch_ptr| {
        const arch: *graph.Architecture = @ptrCast(@alignCast(runtime_arch_ptr));
        return try graph.ensureCompiled(arch);
    }
    capi.setContext("Run `make graphs` to generate architecture.json files from Python definitions", .{});
    return error.MissingArchitecture;
}

/// Get the LayerOp program for a specific layer of a loaded model.
pub fn blockProgramForLayer(loaded: *LoadedModel, layer_idx: usize) ![]const LayerOp {
    graph.init(std.heap.page_allocator);
    if (loaded.runtime_arch) |runtime_arch_ptr| {
        const arch: *graph.Architecture = @ptrCast(@alignCast(runtime_arch_ptr));
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
