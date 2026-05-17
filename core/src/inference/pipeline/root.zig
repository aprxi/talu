//! Backend-neutral staged inference pipeline.
//!
//! This module owns orchestration contracts for local multi-stage inference.
//! CPU, CUDA, and Metal backends remain execution providers; pipeline code owns
//! stage ordering, activation handoff, and future tensor-frame/distributed
//! boundaries. The current implementation stays fully in-process and generic so
//! local CPU/CUDA placements keep their current performance shape.

pub const local_stage_contract = @import("local_stage_contract.zig");
pub const local_pipeline_runtime = @import("local_pipeline_runtime.zig");
pub const local_pipeline_stage_adapter = @import("local_pipeline_stage_adapter.zig");

const pipeline = @import("pipeline.zig");
const tensor_frame = @import("tensor_frame.zig");
const state_ownership = @import("state_ownership.zig");
const host_capability = @import("host_capability.zig");
const local_stage_runner = @import("local_stage_runner.zig");
const local_pipeline = @import("local_pipeline.zig");
const boundary_byte_image = @import("boundary_byte_image.zig");
const stage_transfer_mode = @import("stage_transfer_mode.zig");
const stage_transport = @import("stage_transport.zig");
const local_pipeline_topology = @import("local_pipeline_topology.zig");
const local_pipeline_builder = @import("local_pipeline_builder.zig");

pub const BoundaryDType = pipeline.BoundaryDType;
pub const BoundaryByteImageError = boundary_byte_image.BoundaryByteImageError;
pub const BoundaryByteImageReadiness = boundary_byte_image.BoundaryByteImageReadiness;
pub const BoundaryByteImageRef = boundary_byte_image.BoundaryByteImageRef;
pub const BoundaryLayout = pipeline.BoundaryLayout;
pub const StageTransferMode = stage_transfer_mode.StageTransferMode;
pub const StageTransferModeDecision = stage_transfer_mode.StageTransferModeDecision;
pub const negotiateBoundaryContract = pipeline.negotiateBoundaryContract;

pub const LocalStageChainBoundaryStep = local_stage_runner.LocalStageChainBoundaryStep;
pub const LocalStageChainStage = local_stage_runner.LocalStageChainStage;
pub const LocalStageEndpoint = local_stage_runner.LocalStageEndpoint;
pub const LocalPipelineBoundaryPayload = local_pipeline.LocalPipelineBoundaryPayload;
pub const LocalPipelineContext = local_pipeline.LocalPipelineContext;
pub const LocalPipelineBoundaryFrameSpec = local_pipeline.LocalPipelineBoundaryFrameSpec;
pub const LocalDecodeBoundaryImageSpec = local_pipeline.LocalDecodeBoundaryImageSpec;
pub const LocalDecodeBoundaryPayloadSpec = local_pipeline.LocalDecodeBoundaryPayloadSpec;
pub const LocalPrefillBoundaryImageSpec = local_pipeline.LocalPrefillBoundaryImageSpec;
pub const LocalPrefillBoundaryPayloadSpec = local_pipeline.LocalPrefillBoundaryPayloadSpec;
pub const LocalPipelineRuntime = local_pipeline_runtime.LocalPipelineRuntime;
pub const LocalPipelineStageHandle = local_pipeline_runtime.StageHandle;
pub const LocalPipelineStageVTable = local_pipeline_runtime.StageVTable;
pub const LocalStagePlan = local_pipeline_topology.LocalStagePlan;
pub const LocalStageSpec = local_pipeline_topology.LocalStageSpec;
pub const LocalPipelineStageFactoryRequest = local_pipeline_builder.StageFactoryRequest;
pub const LocalStageRunnerPlanRef = local_stage_runner.LocalStageRunnerPlanRef;
pub const StageStateDescriptorSet = state_ownership.StageStateDescriptorSet;
pub const StageStateOwnershipPlan = state_ownership.StageStateOwnershipPlan;
pub const StageStateOwnershipPlanId = state_ownership.StageStateOwnershipPlanId;
pub const StageStatePartitionFact = state_ownership.StageStatePartitionFact;
pub const BoundaryFrameEndpointRole = host_capability.BoundaryFrameEndpointRole;
pub const BoundaryFrameProfile = host_capability.BoundaryFrameProfile;
pub const BoundaryHandoffMode = host_capability.BoundaryHandoffMode;
pub const HostBackendKind = host_capability.HostBackendKind;
pub const HostCapability = host_capability.HostCapability;
pub const HostFrameCapability = host_capability.HostFrameCapability;
pub const HostId = host_capability.HostId;
pub const HostResidencySnapshot = host_capability.HostResidencySnapshot;
pub const PlacementPlan = host_capability.PlacementPlan;
pub const ResidentStageEntry = host_capability.ResidentStageEntry;
pub const StageHostBinding = host_capability.StageHostBinding;
pub const StageStatePlacementRef = host_capability.StageStatePlacementRef;
pub const StatePlacementMode = host_capability.StatePlacementMode;
pub const TensorFrameBatchEntry = tensor_frame.TensorFrameBatchEntry;
pub const TensorFrameBoundaryRef = tensor_frame.TensorFrameBoundaryRef;
pub const TensorFrameDType = tensor_frame.TensorFrameDType;
pub const TensorFrameInstanceId = tensor_frame.TensorFrameInstanceId;
pub const TensorFrameLayout = tensor_frame.TensorFrameLayout;
pub const TensorFrameMetadata = tensor_frame.TensorFrameMetadata;
pub const TensorFrameObserver = tensor_frame.TensorFrameObserver;
pub const TensorFramePayloadLocationHint = tensor_frame.TensorFramePayloadLocationHint;
pub const TensorFramePlanRef = tensor_frame.TensorFramePlanRef;
pub const TensorFrameStepKind = tensor_frame.TensorFrameStepKind;
pub const TensorFrameTensorDesc = tensor_frame.TensorFrameTensorDesc;
pub const TensorFrameValidationError = tensor_frame.TensorFrameValidationError;
pub const activationDecodeFrame = tensor_frame.activationDecodeFrame;
pub const buildStageStateOwnershipPlan = state_ownership.buildStageStateOwnershipPlan;
pub const buildHostCapability = host_capability.buildHostCapability;
pub const buildHostResidencySnapshot = host_capability.buildHostResidencySnapshot;
pub const buildPlacementPlan = host_capability.buildPlacementPlan;
pub const buildStageStatePlacementRef = host_capability.buildStageStatePlacementRef;
pub const buildLocalStageRunnerPlanRef = local_stage_runner.buildLocalStageRunnerPlanRef;
pub const dtypeByteSize = tensor_frame.dtypeByteSize;
pub const emitTensorFrame = tensor_frame.emitTensorFrame;
pub const executeLocalStageChain = local_stage_runner.executeLocalStageChain;
pub const executeLocalDecodePipelineStep = local_pipeline.executeLocalDecodePipelineStep;
pub const executeLocalDecodePipelineStepWithEndpointRegistry = local_pipeline.executeLocalDecodePipelineStepWithEndpointRegistry;
pub const executeLocalPrefillPipelineStep = local_pipeline.executeLocalPrefillPipelineStep;
pub const executeLocalPrefillPipelineStepWithEndpointRegistry = local_pipeline.executeLocalPrefillPipelineStepWithEndpointRegistry;
pub const executeLocalPipelineStep = local_pipeline.executeLocalPipelineStep;
pub const executeLocalPipelineStepWithEndpointRegistry = local_pipeline.executeLocalPipelineStepWithEndpointRegistry;
pub const autoDetectLocalStagePlanForModel = local_pipeline_topology.autoDetectLocalStagePlanForModel;
pub const cpuPrefixCudaLocalStagePlan = local_pipeline_topology.cpuPrefixCudaLocalStagePlan;
pub const defaultCudaLocalStagePlan = local_pipeline_topology.defaultCudaLocalStagePlan;
pub const initLocalPipelineRuntime = local_pipeline_builder.initLocalPipelineRuntime;
pub const localStagePlanFromSpecs = local_pipeline_topology.localStagePlanFromSpecs;
pub const loadedModelHasPackedNvfp4Weights = local_pipeline_topology.loadedModelHasPackedNvfp4Weights;
pub const parseLocalStageSpecs = local_pipeline_topology.parseLocalStageSpecs;
pub const resolveLocalStagePlan = local_pipeline_topology.resolveLocalStagePlan;
pub const validateLocalStagePlan = local_pipeline_topology.validateLocalStagePlan;
pub const selectedBoundaryTensorContract = tensor_frame.selectedBoundaryTensorContract;
pub const deviceActivationByteImage = local_stage_runner.deviceActivationByteImage;
pub const hostActivationByteImage = local_stage_runner.hostActivationByteImage;
pub const localStageAdapter = local_stage_runner.localStageAdapter;
pub const state_ownership_contract_version = state_ownership.state_ownership_contract_version;
pub const validatePlacementPlan = host_capability.validatePlacementPlan;
pub const validateStageStateOwnershipPlan = state_ownership.validateStageStateOwnershipPlan;
pub const validateBoundaryByteImage = boundary_byte_image.validateBoundaryByteImage;
pub const boundaryByteImageIsRemoteReadable = boundary_byte_image.boundaryByteImageIsRemoteReadable;
pub const chooseStageTransferMode = stage_transfer_mode.chooseStageTransferMode;
pub const buildStageTransportActivationEnvelope = stage_transport.buildStageTransportActivationEnvelope;

test "inference pipeline root exposes high-level local pipeline surface" {
    _ = local_stage_contract.StageSpec;
    _ = local_pipeline_runtime.LocalPipelineRuntime;
    _ = local_pipeline_stage_adapter.stageVTable;
    _ = LocalStageSpec;
    _ = LocalStagePlan;
    _ = LocalPipelineRuntime;
    _ = LocalStageChainBoundaryStep;
    _ = LocalStageChainStage;
    _ = LocalStageEndpoint;
    _ = executeLocalStageChain;
    _ = executeLocalPipelineStep;
    _ = chooseStageTransferMode;
    _ = buildStageTransportActivationEnvelope;
}
