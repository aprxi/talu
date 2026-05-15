//! Inference transport executors.
//!
//! Transport owns process-local byte movement for staged activation handoff.
//! Bridge modules provide the contracts and ordering; backend adapters provide
//! concrete CPU/CUDA/Metal payload operations.

pub const local_stage = @import("local_stage.zig");
pub const cuda_activation = @import("cuda_activation.zig");
pub const cuda_kv_mirror = @import("cuda_kv_mirror.zig");

pub const LocalStageTransportValidationError = local_stage.LocalStageTransportValidationError;
pub const LocalStageTransportRequest = local_stage.LocalStageTransportRequest;
pub const LocalStageTransportEntryFailure = local_stage.LocalStageTransportEntryFailure;
pub const LocalStageTransportFailureReport = local_stage.LocalStageTransportFailureReport;
pub const LocalStageTransportFailureCapture = local_stage.LocalStageTransportFailureCapture;
pub const CudaPeerCopySynchronization = cuda_activation.CudaPeerCopySynchronization;
pub const CudaActivationStage = cuda_activation.CudaActivationStage;
pub const CudaPeerActivationStage = cuda_activation.CudaPeerActivationStage;
pub const NoopActivationStage = cuda_activation.NoopActivationStage;

pub const executeLocalStageTransport = local_stage.executeLocalStageTransport;
pub const executeLocalStageTransportWithFailureCapture = local_stage.executeLocalStageTransportWithFailureCapture;
pub const cudaBufferSlice = cuda_activation.cudaBufferSlice;
pub const downloadCudaActivation = cuda_activation.downloadCudaActivation;
pub const downloadHostSlotActivation = cuda_activation.downloadHostSlotActivation;
pub const copyCudaPeerActivation = cuda_activation.copyCudaPeerActivation;
pub const copyCudaPeerActivationAfterEvent = cuda_activation.copyCudaPeerActivationAfterEvent;
pub const peerCopyCudaActivation = cuda_activation.peerCopyCudaActivation;
pub const peerCopyCudaActivationHandlesStageSync = cuda_activation.peerCopyCudaActivationHandlesStageSync;
pub const synchronizeCudaActivationBackend = cuda_activation.synchronizeCudaActivationBackend;
pub const uploadCudaActivation = cuda_activation.uploadCudaActivation;
pub const uploadCudaActivationSegments = cuda_activation.uploadCudaActivationSegments;
pub const uploadCpuKvToCudaMirrors = cuda_kv_mirror.uploadCpuKvToCudaMirrors;
pub const uploadHostSlotActivation = cuda_activation.uploadHostSlotActivation;

test "inference transport root exports local_stage contract" {
    _ = LocalStageTransportRequest;
    _ = LocalStageTransportValidationError;
    _ = executeLocalStageTransport;
}

test "inference transport root exports cuda_activation contract" {
    _ = CudaActivationStage;
    _ = CudaPeerActivationStage;
    _ = CudaPeerCopySynchronization;
    _ = NoopActivationStage;
    _ = cudaBufferSlice;
    _ = copyCudaPeerActivation;
    _ = copyCudaPeerActivationAfterEvent;
    _ = downloadCudaActivation;
    _ = downloadHostSlotActivation;
    _ = peerCopyCudaActivation;
    _ = peerCopyCudaActivationHandlesStageSync;
    _ = synchronizeCudaActivationBackend;
    _ = uploadCudaActivation;
    _ = uploadCudaActivationSegments;
    _ = uploadCpuKvToCudaMirrors;
    _ = uploadHostSlotActivation;
}
