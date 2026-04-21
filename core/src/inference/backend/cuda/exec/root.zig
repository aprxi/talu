//! Execution-route subsystem root for CUDA inference backend.

pub const decode_route = @import("decode_route.zig");
pub const prefill_route = @import("prefill_route.zig");
pub const kv_capacity = @import("kv_capacity.zig");
pub const transfers = @import("transfers.zig");
pub const resets = @import("resets.zig");
pub const pipeline2 = @import("pipeline2.zig");
pub const cpu_gpu = @import("cpu_gpu.zig");
pub const cpu_gpu_gpu = @import("cpu_gpu_gpu.zig");

pub const resolveStagedPrefillChunkRows = prefill_route.resolveStagedPrefillChunkRows;
pub const computeGpuPrototypePrefillLogitsWithLayerLimit = prefill_route.computeGpuPrototypePrefillLogitsWithLayerLimit;
pub const computeGpuPrototypeLogitsWithLayerLimit = decode_route.computeGpuPrototypeLogitsWithLayerLimit;
pub const computeBatchedDecodeLogits = decode_route.computeBatchedDecodeLogits;
pub const computeBatchedDecodeLogitsDeviceOnly = decode_route.computeBatchedDecodeLogitsDeviceOnly;
pub const ensureKvCapacity = kv_capacity.ensureKvCapacity;
pub const transferPipelineActivationMultiRow = transfers.transferPipelineActivationMultiRow;
pub const transferPipelineActivationStage12MultiRow = transfers.transferPipelineActivationStage12MultiRow;
pub const uploadCpuKvToMirrors = transfers.uploadCpuKvToMirrors;
pub const resetShortConvStates = resets.resetShortConvStates;
pub const resetGatedDeltaStates = resets.resetGatedDeltaStates;
pub const resetAttentionCpuStates = resets.resetAttentionCpuStates;
pub const ensureGatedDeltaHostStageCapacity = resets.ensureGatedDeltaHostStageCapacity;
pub const computeBatchedPrefillPipeline2 = pipeline2.computeBatchedPrefillPipeline2;
pub const runPipeline2WithPipelineRuntime = pipeline2.runPipeline2WithPipelineRuntime;
pub const computeBatchedDecodePipeline2WithMode = decode_route.computeBatchedDecodePipeline2WithMode;
pub const computeBatchedPrefillCpuGpu = cpu_gpu.computeBatchedPrefillCpuGpu;
pub const runCpuGpuWithPipelineRuntime = cpu_gpu.runCpuGpuWithPipelineRuntime;
pub const computeBatchedDecodeCpuGpuWithMode = decode_route.computeBatchedDecodeCpuGpuWithMode;
pub const computeBatchedPrefillCpuGpuGpu = cpu_gpu_gpu.computeBatchedPrefillCpuGpuGpu;
pub const runCpuGpuGpuWithPipelineRuntime = cpu_gpu_gpu.runCpuGpuGpuWithPipelineRuntime;
pub const computeBatchedDecodeCpuGpuGpuWithMode = decode_route.computeBatchedDecodeCpuGpuGpuWithMode;
