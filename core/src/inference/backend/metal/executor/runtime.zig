//! Metal executor glue module.
//!
//! This module exposes the Metal runtime types used by executor kernels.

const std = @import("std");
const compute = @import("../../../../compute/root.zig");
const runtime_contract = @import("../../../runtime_contract/root.zig");
const runtime_graph = @import("../runtime_graph.zig");
const model_executor = @import("model.zig");
const weights_executor = @import("weights.zig");

const ArrayHandle = compute.metal.graph.ArrayHandle;

pub const Cache = runtime_graph.Cache;
pub const ShortConvCache = runtime_graph.ShortConvCache;
pub const MambaCache = runtime_graph.MambaCache;

pub const DeepstackAdditions = model_executor.DeepstackAdditions;
pub const RuntimeRoPEOverride = model_executor.RuntimeRoPEOverride;

pub const MLXError = weights_executor.MLXError;
pub const WeightHandles = weights_executor.WeightHandles;

pub const loadWeightsToGPU = weights_executor.loadWeightsToGPU;
pub const freeWeights = weights_executor.freeWeights;

pub const gatherTokenEmbeddingsLazy = model_executor.gatherTokenEmbeddingsLazy;

pub fn transformerForwardLazy(
    allocator: std.mem.Allocator,
    weight_handles: *const WeightHandles,
    input_ids: []const u32,
    state_blocks: []const runtime_contract.StateBlockHandle,
    config: anytype,
    pos_offset: usize,
    use_compiled: bool,
) !ArrayHandle {
    return model_executor.Model.forward(
        allocator,
        weight_handles,
        input_ids,
        state_blocks,
        config,
        pos_offset,
        use_compiled,
    );
}

pub fn transformerForwardHiddenLazy(
    allocator: std.mem.Allocator,
    weight_handles: *const WeightHandles,
    input_ids: []const u32,
    state_blocks: []const runtime_contract.StateBlockHandle,
    config: anytype,
    pos_offset: usize,
    use_compiled: bool,
) !ArrayHandle {
    return model_executor.Model.forwardHidden(
        allocator,
        weight_handles,
        input_ids,
        state_blocks,
        config,
        pos_offset,
        use_compiled,
    );
}

pub fn transformerForwardLazyWithEmbeddingOverride(
    allocator: std.mem.Allocator,
    weight_handles: *const WeightHandles,
    input_ids: []const u32,
    state_blocks: []const runtime_contract.StateBlockHandle,
    config: anytype,
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
        state_blocks,
        config,
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
    state_blocks: []const runtime_contract.StateBlockHandle,
    config: anytype,
    pos_offset: usize,
) !ArrayHandle {
    return model_executor.Model.forwardFromGPUToken(
        allocator,
        weight_handles,
        token_handle,
        state_blocks,
        config,
        pos_offset,
    );
}

test "transformerForwardLazy exposes stable callable signature" {
    const fn_info = @typeInfo(@TypeOf(transformerForwardLazy)).@"fn";
    try std.testing.expectEqual(@as(usize, 8), fn_info.params.len);
    const f = transformerForwardLazy;
    _ = f;
}

test "transformerForwardHiddenLazy exposes stable callable signature" {
    const fn_info = @typeInfo(@TypeOf(transformerForwardHiddenLazy)).@"fn";
    try std.testing.expectEqual(@as(usize, 8), fn_info.params.len);
    const f = transformerForwardHiddenLazy;
    _ = f;
}

test "transformerForwardLazyWithEmbeddingOverride exposes stable callable signature" {
    const fn_info = @typeInfo(@TypeOf(transformerForwardLazyWithEmbeddingOverride)).@"fn";
    try std.testing.expectEqual(@as(usize, 11), fn_info.params.len);
    const f = transformerForwardLazyWithEmbeddingOverride;
    _ = f;
}

test "transformerForwardFromGPUToken exposes stable callable signature" {
    const fn_info = @typeInfo(@TypeOf(transformerForwardFromGPUToken)).@"fn";
    try std.testing.expectEqual(@as(usize, 7), fn_info.params.len);
    const f = transformerForwardFromGPUToken;
    _ = f;
}
