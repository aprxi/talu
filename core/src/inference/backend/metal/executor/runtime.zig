//! Metal executor glue module.
//!
//! Historically this file owned all Metal backend execution logic.
//! The ownership now lives in `executor/weights.zig` (weight lifecycle)
//! and `executor/model.zig` (forward orchestration). This module keeps
//! compatibility symbols for callers while remaining thin.

const std = @import("std");
const compute = @import("../../../../compute/root.zig");
const model_executor = @import("model.zig");
const weights_executor = @import("weights.zig");

const ArrayHandle = compute.metal.graph.ArrayHandle;

pub const Cache = compute.metal.graph.Cache;
pub const ShortConvCache = compute.metal.graph.ShortConvCache;

pub const DeepstackAdditions = model_executor.DeepstackAdditions;
pub const RuntimeRoPEOverride = model_executor.RuntimeRoPEOverride;

pub const MLXError = weights_executor.MLXError;
pub const WeightHandles = weights_executor.WeightHandles;

pub const loadWeightsToGPU = weights_executor.loadWeightsToGPU;
pub const createFusedModel = weights_executor.createFusedModel;
pub const freeWeights = weights_executor.freeWeights;

pub const gatherTokenEmbeddingsLazy = model_executor.gatherTokenEmbeddingsLazy;

pub fn transformerForwardLazy(
    allocator: std.mem.Allocator,
    weight_handles: *const WeightHandles,
    input_ids: []const u32,
    config: anytype,
    cache: ?Cache,
    shortconv_cache: ?ShortConvCache,
    pos_offset: usize,
    use_compiled: bool,
) !ArrayHandle {
    return model_executor.Model.forward(
        allocator,
        weight_handles,
        input_ids,
        config,
        cache,
        shortconv_cache,
        pos_offset,
        use_compiled,
    );
}

pub fn transformerForwardLazyWithEmbeddingOverride(
    allocator: std.mem.Allocator,
    weight_handles: *const WeightHandles,
    input_ids: []const u32,
    config: anytype,
    cache: ?Cache,
    shortconv_cache: ?ShortConvCache,
    pos_offset: usize,
    use_compiled: bool,
    embedding_override: ?[]const f32,
    deepstack: ?DeepstackAdditions,
    runtime_rope: ?RuntimeRoPEOverride,
) !ArrayHandle {
    return model_executor.Model.forwardWithEmbeddingOverride(
        allocator,
        weight_handles,
        input_ids,
        config,
        cache,
        shortconv_cache,
        pos_offset,
        use_compiled,
        embedding_override,
        deepstack,
        runtime_rope,
    );
}

pub fn transformerForwardFromGPUToken(
    allocator: std.mem.Allocator,
    weight_handles: *const WeightHandles,
    token_handle: ArrayHandle,
    config: anytype,
    cache: ?Cache,
    shortconv_cache: ?ShortConvCache,
    pos_offset: usize,
) !ArrayHandle {
    return model_executor.Model.forwardFromGPUToken(
        allocator,
        weight_handles,
        token_handle,
        config,
        cache,
        shortconv_cache,
        pos_offset,
    );
}
