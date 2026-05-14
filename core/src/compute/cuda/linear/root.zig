//! Generic CUDA linear-weight descriptors and execution.

pub const weights = @import("weights.zig");
pub const executor = @import("executor.zig");
pub const fused = @import("fused.zig");

pub const DenseU16Dtype = weights.DenseU16Dtype;
pub const EmbeddingLookupKind = weights.EmbeddingLookupKind;
pub const DeviceTensor = weights.DeviceTensor;
pub const missing_device_tensor = weights.missing_device_tensor;
pub const missing_host_tensor = weights.missing_host_tensor;
pub const EmbeddingLookup = weights.EmbeddingLookup;
pub const GaffineU4LinearWeight = weights.GaffineU4LinearWeight;
pub const GaffineU8LinearWeight = weights.GaffineU8LinearWeight;
pub const U16LinearWeight = weights.U16LinearWeight;
pub const Fp8LinearWeight = weights.Fp8LinearWeight;
pub const Mxfp8LinearWeight = weights.Mxfp8LinearWeight;
pub const Nvfp4LinearWeight = weights.Nvfp4LinearWeight;
pub const LinearWeight = weights.LinearWeight;
pub const bufferSlice = weights.bufferSlice;
pub const bufferF32RowCount = weights.bufferF32RowCount;
pub const logicalF32RowSlice = weights.logicalF32RowSlice;
pub const linearWeightHasI8Cache = weights.linearWeightHasI8Cache;

pub const Nvfp4LinearRouteKind = executor.Nvfp4LinearRouteKind;
pub const Diagnostics = executor.Diagnostics;
pub const Workspace = executor.Workspace;
pub const Context = executor.Context;
pub const execute = executor.execute;
pub const executeRows = executor.executeRows;

pub const FusedContext = fused.FusedContext;
pub const FusedDiagnostics = fused.Diagnostics;
pub const FusedCapabilityFlags = fused.CapabilityFlags;
pub const FusedPairOutputs = fused.PairOutputs;
pub const FusedTripleOutputs = fused.TripleOutputs;
pub const FusedPairActivation = fused.PairActivation;
pub const FusedConcatI8TripleWeight = fused.ConcatI8TripleWeight;
pub const FusedNvfp4RouteKind = fused.Nvfp4RouteKind;

test {
    _ = weights;
    _ = executor;
    _ = fused;
}
