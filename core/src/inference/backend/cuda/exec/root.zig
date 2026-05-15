//! Execution-route subsystem root for CUDA inference backend.

pub const decode_route = @import("decode_route.zig");
pub const prefill_route = @import("prefill_route.zig");
pub const kv_capacity = @import("kv_capacity.zig");
pub const transfers = @import("transfers.zig");
pub const resets = @import("resets.zig");
pub const stage_adapters = @import("stage_adapters.zig");

pub const resolveStagedPrefillChunkRows = prefill_route.resolveStagedPrefillChunkRows;
pub const executePrefillWithLayerLimit = prefill_route.executePrefillWithLayerLimit;
pub const executeDecodeWithLayerLimit = decode_route.executeDecodeWithLayerLimit;
pub const computeBatchedDecodeLogits = decode_route.computeBatchedDecodeLogits;
pub const computeBatchedDecodeLogitsDeviceOnly = decode_route.computeBatchedDecodeLogitsDeviceOnly;
pub const ensureKvCapacity = kv_capacity.ensureKvCapacity;
pub const uploadCpuKvToMirrors = transfers.uploadCpuKvToMirrors;
pub const resetShortConvStates = resets.resetShortConvStates;
pub const resetGatedDeltaStates = resets.resetGatedDeltaStates;
pub const resetAttentionCpuStates = resets.resetAttentionCpuStates;
pub const ensureGatedDeltaHostStageCapacity = resets.ensureGatedDeltaHostStageCapacity;
