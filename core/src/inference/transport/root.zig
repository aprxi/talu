//! Inference transport executors.
//!
//! Transport owns process-local byte movement for staged activation handoff.
//! Pipeline modules provide contracts and ordering; backend stages provide only
//! execution handles that transport adapters turn into concrete host/CUDA
//! payload operations.

pub const local_stage = @import("local_stage.zig");
pub const cuda_activation = @import("cuda_activation.zig");
pub const cuda_kv_mirror = @import("cuda_kv_mirror.zig");

pub const LocalStageTransportValidationError = local_stage.LocalStageTransportValidationError;
pub const LocalStageTransportRequest = local_stage.LocalStageTransportRequest;
pub const LocalStageTransportEntryFailure = local_stage.LocalStageTransportEntryFailure;
pub const LocalStageTransportFailureReport = local_stage.LocalStageTransportFailureReport;
pub const LocalStageTransportFailureCapture = local_stage.LocalStageTransportFailureCapture;
pub const LocalStageTransportEndpoint = local_stage.LocalStageTransportEndpoint;
pub const LocalStageTransportEndpointRegistry = local_stage.LocalStageTransportEndpointRegistry;
pub const LocalStageTransportEndpointVTable = local_stage.LocalStageTransportEndpointVTable;
pub const StageExecutionFence = local_stage.StageExecutionFence;
pub const StageExecutionReceipt = local_stage.StageExecutionReceipt;
pub const CudaPeerCopySynchronization = cuda_activation.CudaPeerCopySynchronization;
pub const CudaActivationStage = cuda_activation.CudaActivationStage;
pub const CudaPeerActivationStage = cuda_activation.CudaPeerActivationStage;
pub const CudaActivationBufferVTable = cuda_activation.CudaActivationBufferVTable;
pub const CudaBufferDescriptor = cuda_activation.CudaBufferDescriptor;
pub const NoopActivationStage = cuda_activation.NoopActivationStage;

pub const ExternalActivationSource = union(enum) {
    host: []const u8,
    cuda: CudaBufferDescriptor,
};

pub const ExternalActivationTarget = union(enum) {
    host: []u8,
    cuda: CudaBufferDescriptor,
};

pub const executeLocalStageTransport = local_stage.executeLocalStageTransport;
pub const executeLocalStageTransportWithFailureCapture = local_stage.executeLocalStageTransportWithFailureCapture;
pub const localStageTransportAdapter = local_stage.localStageTransportAdapter;
pub const copyHostActivationToBuffer = local_stage.copyHostActivationToBuffer;
pub const copyHostBufferToActivationTarget = local_stage.copyHostBufferToActivationTarget;
pub const cudaBufferSlice = cuda_activation.cudaBufferSlice;
pub const cudaActivationBufferDescriptor = cuda_activation.cudaActivationBufferDescriptor;
pub const synchronizeCudaActivationDescriptor = cuda_activation.synchronizeCudaActivationDescriptor;
pub const downloadCudaActivationDescriptor = cuda_activation.downloadCudaActivationDescriptor;
pub const uploadCudaActivationDescriptor = cuda_activation.uploadCudaActivationDescriptor;
pub const uploadCudaActivationSegmentsDescriptor = cuda_activation.uploadCudaActivationSegmentsDescriptor;
pub const peerCopyCudaActivationDescriptorTo = cuda_activation.peerCopyCudaActivationDescriptorTo;
pub const probeCudaActivationDescriptorPeerCopy = cuda_activation.probeCudaActivationDescriptorPeerCopy;
pub const peerCopyCudaActivationDescriptorHandlesStageSync = cuda_activation.peerCopyCudaActivationDescriptorHandlesStageSync;
pub const uploadCudaBufferFromHostBytes = cuda_activation.uploadCudaBufferFromHostBytes;
pub const downloadCudaBufferToHostBytes = cuda_activation.downloadCudaBufferToHostBytes;
pub const downloadCudaActivation = cuda_activation.downloadCudaActivation;
pub const downloadHostSlotActivation = cuda_activation.downloadHostSlotActivation;
pub const copyCudaPeerActivation = cuda_activation.copyCudaPeerActivation;
pub const probeCudaPeerActivation = cuda_activation.probeCudaPeerActivation;
pub const copyCudaPeerActivationAfterEvent = cuda_activation.copyCudaPeerActivationAfterEvent;
pub const peerCopyCudaActivation = cuda_activation.peerCopyCudaActivation;
pub const peerCopyCudaActivationHandlesStageSync = cuda_activation.peerCopyCudaActivationHandlesStageSync;
pub const peerCopyCudaActivationRuntime = cuda_activation.peerCopyCudaActivationRuntime;
pub const peerCopyCudaActivationHandlesStageSyncRuntime = cuda_activation.peerCopyCudaActivationHandlesStageSyncRuntime;
pub const synchronizeLocalEndpoint = cuda_activation.synchronizeLocalEndpoint;
pub const prepareCpuBoundaryTransferToCudaEndpoint = cuda_activation.prepareCpuBoundaryTransferToCudaEndpoint;
pub const downloadLocalDecodeEndpointActivation = cuda_activation.downloadLocalDecodeEndpointActivation;
pub const downloadLocalDeviceEndpointActivation = cuda_activation.downloadLocalDeviceEndpointActivation;
pub const uploadLocalEndpointActivation = cuda_activation.uploadLocalEndpointActivation;
pub const uploadLocalEndpointActivationSegments = cuda_activation.uploadLocalEndpointActivationSegments;
pub const peerCopyLocalEndpointActivationTo = cuda_activation.peerCopyLocalEndpointActivationTo;
pub const localEndpointPeerCopyHandlesStageSync = cuda_activation.localEndpointPeerCopyHandlesStageSync;
pub const localEndpointTransportAdapter = cuda_activation.localEndpointTransportAdapter;
pub const LocalEndpointTransportOptions = cuda_activation.LocalEndpointTransportOptions;
pub const synchronizeCudaActivationBackend = cuda_activation.synchronizeCudaActivationBackend;
pub const uploadCudaActivation = cuda_activation.uploadCudaActivation;
pub const uploadCudaActivationSegments = cuda_activation.uploadCudaActivationSegments;
pub const uploadCpuKvToCudaMirrors = cuda_kv_mirror.uploadCpuKvToCudaMirrors;
pub const uploadHostSlotActivation = cuda_activation.uploadHostSlotActivation;

pub fn downloadExternalActivation(
    source: ExternalActivationSource,
    host_buf: []u8,
    byte_count: usize,
) !void {
    switch (source) {
        .host => |bytes| try copyHostActivationToBuffer(bytes, host_buf, byte_count),
        .cuda => |descriptor| try downloadCudaActivationDescriptor(descriptor, host_buf, byte_count),
    }
}

pub fn uploadExternalActivation(
    target: ExternalActivationTarget,
    host_buf: []const u8,
    byte_count: usize,
) !void {
    switch (target) {
        .host => |bytes| try copyHostBufferToActivationTarget(bytes, host_buf, byte_count),
        .cuda => |descriptor| try uploadCudaActivationDescriptor(descriptor, host_buf, byte_count),
    }
}

pub fn uploadExternalActivationSegments(
    target: ExternalActivationTarget,
    host_segments: []const []const u8,
    byte_count: usize,
) !void {
    switch (target) {
        .host => return error.LocalStageTransportSegmentedUploadUnsupported,
        .cuda => |descriptor| try uploadCudaActivationSegmentsDescriptor(descriptor, host_segments, byte_count),
    }
}

pub fn peerCopyExternalActivation(
    source: ExternalActivationSource,
    target: ExternalActivationTarget,
    byte_count: usize,
) !void {
    switch (source) {
        .cuda => |source_descriptor| switch (target) {
            .cuda => |target_descriptor| try peerCopyCudaActivationDescriptorTo(
                source_descriptor,
                target_descriptor,
                byte_count,
                .source_stream,
            ),
            else => return error.LocalStageTransportPeerCopyUnsupported,
        },
        else => return error.LocalStageTransportPeerCopyUnsupported,
    }
}

pub fn canPeerCopyExternalActivation(
    source: ExternalActivationSource,
    target: ExternalActivationTarget,
) bool {
    return switch (source) {
        .cuda => |source_descriptor| switch (target) {
            .cuda => |target_descriptor| probeCudaActivationDescriptorPeerCopy(source_descriptor, target_descriptor),
            else => false,
        },
        else => false,
    };
}

pub fn externalActivationPeerCopyHandlesStageSync(source: ExternalActivationSource) bool {
    return switch (source) {
        .cuda => |descriptor| peerCopyCudaActivationDescriptorHandlesStageSync(descriptor, .source_stream),
        else => false,
    };
}

test "inference transport root exports local_stage contract" {
    _ = LocalStageTransportRequest;
    _ = LocalStageTransportValidationError;
    _ = LocalStageTransportEndpoint;
    _ = LocalStageTransportEndpointRegistry;
    _ = LocalStageTransportEndpointVTable;
    _ = StageExecutionFence;
    _ = StageExecutionReceipt;
    _ = executeLocalStageTransport;
    _ = localStageTransportAdapter;
    _ = copyHostActivationToBuffer;
    _ = copyHostBufferToActivationTarget;
    _ = ExternalActivationSource;
    _ = ExternalActivationTarget;
    _ = downloadExternalActivation;
    _ = uploadExternalActivation;
    _ = uploadExternalActivationSegments;
    _ = peerCopyExternalActivation;
    _ = canPeerCopyExternalActivation;
    _ = externalActivationPeerCopyHandlesStageSync;
}

test "inference transport root exports cuda_activation contract" {
    _ = CudaActivationStage;
    _ = CudaPeerActivationStage;
    _ = CudaPeerCopySynchronization;
    _ = CudaActivationBufferVTable;
    _ = CudaBufferDescriptor;
    _ = NoopActivationStage;
    _ = cudaBufferSlice;
    _ = cudaActivationBufferDescriptor;
    _ = synchronizeCudaActivationDescriptor;
    _ = downloadCudaActivationDescriptor;
    _ = uploadCudaActivationDescriptor;
    _ = uploadCudaActivationSegmentsDescriptor;
    _ = peerCopyCudaActivationDescriptorTo;
    _ = probeCudaActivationDescriptorPeerCopy;
    _ = peerCopyCudaActivationDescriptorHandlesStageSync;
    _ = uploadCudaBufferFromHostBytes;
    _ = downloadCudaBufferToHostBytes;
    _ = copyCudaPeerActivation;
    _ = probeCudaPeerActivation;
    _ = copyCudaPeerActivationAfterEvent;
    _ = downloadCudaActivation;
    _ = downloadHostSlotActivation;
    _ = peerCopyCudaActivation;
    _ = peerCopyCudaActivationHandlesStageSync;
    _ = peerCopyCudaActivationRuntime;
    _ = peerCopyCudaActivationHandlesStageSyncRuntime;
    _ = synchronizeLocalEndpoint;
    _ = prepareCpuBoundaryTransferToCudaEndpoint;
    _ = downloadLocalDecodeEndpointActivation;
    _ = downloadLocalDeviceEndpointActivation;
    _ = uploadLocalEndpointActivation;
    _ = uploadLocalEndpointActivationSegments;
    _ = peerCopyLocalEndpointActivationTo;
    _ = localEndpointPeerCopyHandlesStageSync;
    _ = localEndpointTransportAdapter;
    _ = LocalEndpointTransportOptions;
    _ = synchronizeCudaActivationBackend;
    _ = uploadCudaActivation;
    _ = uploadCudaActivationSegments;
    _ = uploadCpuKvToCudaMirrors;
    _ = uploadHostSlotActivation;
}
