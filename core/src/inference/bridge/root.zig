//! Backend-neutral staged inference bridge.
//!
//! This module owns orchestration contracts for local multi-stage inference.
//! CPU, CUDA, and Metal backends remain execution providers; bridge code owns
//! stage ordering, activation handoff, and future tensor-frame/distributed
//! boundaries. The current implementation stays fully in-process and generic so
//! existing CPU/GPU split paths keep their current performance shape.

pub const pipeline = @import("pipeline.zig");
pub const orchestrator = @import("orchestrator.zig");
pub const tensor_frame = @import("tensor_frame.zig");
pub const state_ownership = @import("state_ownership.zig");

pub const BoundaryDType = pipeline.BoundaryDType;
pub const BoundaryLayout = pipeline.BoundaryLayout;
pub const BoundaryNegotiationRequest = pipeline.BoundaryNegotiationRequest;
pub const BoundaryNegotiationResult = pipeline.BoundaryNegotiationResult;
pub const negotiateBoundaryContract = pipeline.negotiateBoundaryContract;

pub const ActivationFrameArgs = tensor_frame.ActivationFrameArgs;
pub const LocalDecodeHandoffConfig = orchestrator.LocalDecodeHandoffConfig;
pub const PipelineRuntime = pipeline.PipelineRuntime;
pub const PipelineRuntime3 = pipeline.PipelineRuntime3;
pub const StageStateBindingReport = state_ownership.StageStateBindingReport;
pub const StageStateBoundaryRef = state_ownership.StageStateBoundaryRef;
pub const StageStateCleanupTarget = state_ownership.StageStateCleanupTarget;
pub const StageStateDependencyRef = state_ownership.StageStateDependencyRef;
pub const StageStateDescriptorSet = state_ownership.StageStateDescriptorSet;
pub const StageStateDescriptorSource = state_ownership.StageStateDescriptorSource;
pub const StageStateLease = state_ownership.StageStateLease;
pub const StageStateLeaseState = state_ownership.StageStateLeaseState;
pub const StageStateLeaseValidationOptions = state_ownership.StageStateLeaseValidationOptions;
pub const StageStateOwnershipPlan = state_ownership.StageStateOwnershipPlan;
pub const StageStateOwnershipPlanId = state_ownership.StageStateOwnershipPlanId;
pub const StageStateOwnershipRequest = state_ownership.StageStateOwnershipRequest;
pub const StageStatePartitionFact = state_ownership.StageStatePartitionFact;
pub const StageStateTransition = state_ownership.StageStateTransition;
pub const StageStateTransitionKind = state_ownership.StageStateTransitionKind;
pub const StageStateTransitionResult = state_ownership.StageStateTransitionResult;
pub const StateCleanupKind = state_ownership.StateCleanupKind;
pub const StateCleanupObligation = state_ownership.StateCleanupObligation;
pub const StateOwnershipContractVersion = state_ownership.StateOwnershipContractVersion;
pub const StateOwnershipError = state_ownership.StateOwnershipError;
pub const StatePartitionOwnershipMode = state_ownership.StatePartitionOwnershipMode;
pub const BoundaryTensorContractSource = tensor_frame.BoundaryTensorContractSource;
pub const TensorFrameBatch = tensor_frame.TensorFrameBatch;
pub const TensorFrameBatchEntry = tensor_frame.TensorFrameBatchEntry;
pub const TensorFrameBoundaryRef = tensor_frame.TensorFrameBoundaryRef;
pub const TensorFrameBoundaryTensorContract = tensor_frame.TensorFrameBoundaryTensorContract;
pub const TensorFrameContractVersion = tensor_frame.TensorFrameContractVersion;
pub const TensorFrameDType = tensor_frame.TensorFrameDType;
pub const TensorFrameInstanceId = tensor_frame.TensorFrameInstanceId;
pub const TensorFrameLayout = tensor_frame.TensorFrameLayout;
pub const TensorFrameLifetime = tensor_frame.TensorFrameLifetime;
pub const TensorFrameMetadata = tensor_frame.TensorFrameMetadata;
pub const TensorFrameObserver = tensor_frame.TensorFrameObserver;
pub const TensorFrameObserverMode = tensor_frame.TensorFrameObserverMode;
pub const TensorFrameOwnership = tensor_frame.TensorFrameOwnership;
pub const TensorFramePayload = tensor_frame.TensorFramePayload;
pub const TensorFramePayloadLocationHint = tensor_frame.TensorFramePayloadLocationHint;
pub const TensorFramePlanIdentity = tensor_frame.TensorFramePlanIdentity;
pub const TensorFramePlanRef = tensor_frame.TensorFramePlanRef;
pub const TensorFrameRole = tensor_frame.TensorFrameRole;
pub const TensorFrameShapeContext = tensor_frame.TensorFrameShapeContext;
pub const TensorFrameStepKind = tensor_frame.TensorFrameStepKind;
pub const TensorFrameTensorDesc = tensor_frame.TensorFrameTensorDesc;
pub const TensorFrameValidationError = tensor_frame.TensorFrameValidationError;
pub const activationDecodeFrame = tensor_frame.activationDecodeFrame;
pub const activationPrefillFrame = tensor_frame.activationPrefillFrame;
pub const buildStageStateOwnershipPlan = state_ownership.buildStageStateOwnershipPlan;
pub const buildStateCleanupObligations = state_ownership.buildStateCleanupObligations;
pub const boundaryRefFromPlanRef = tensor_frame.boundaryRefFromPlanRef;
pub const descriptorSetForStage = state_ownership.descriptorSetForStage;
pub const dtypeByteSize = tensor_frame.dtypeByteSize;
pub const emitTensorFrame = tensor_frame.emitTensorFrame;
pub const executeTwoStageForward = orchestrator.executeTwoStageForward;
pub const executeThreeStageForward = orchestrator.executeThreeStageForward;
pub const executeLocalDecodeHandoff = orchestrator.executeLocalDecodeHandoff;
pub const fromComputeDType = tensor_frame.fromComputeDType;
pub const selectedBoundaryTensorContract = tensor_frame.selectedBoundaryTensorContract;
pub const shouldZeroStateDescriptorForLifecycleAction = state_ownership.shouldZeroStateDescriptorForLifecycleAction;
pub const state_ownership_contract_version = state_ownership.state_ownership_contract_version;
pub const stateOwnershipPlanIdEql = state_ownership.stateOwnershipPlanIdEql;
pub const tensor_frame_contract_version = tensor_frame.tensor_frame_contract_version;
pub const tensorFrameLogicalEql = tensor_frame.tensorFrameLogicalEql;
pub const tensorFrameLogicalHash = tensor_frame.tensorFrameLogicalHash;
pub const toComputeDType = tensor_frame.toComputeDType;
pub const validatePayloadBufferLength = tensor_frame.validatePayloadBufferLength;
pub const validateDecodeTransition = state_ownership.validateDecodeTransition;
pub const validateLeaseTransition = state_ownership.validateLeaseTransition;
pub const validatePrefillTransition = state_ownership.validatePrefillTransition;
pub const validateStageStateBindingReports = state_ownership.validateStageStateBindingReports;
pub const validateStageStateDescriptorSet = state_ownership.validateStageStateDescriptorSet;
pub const validateStageStateLease = state_ownership.validateStageStateLease;
pub const validateStageStateOwnershipPlan = state_ownership.validateStageStateOwnershipPlan;
pub const validateStageStatePartitionFact = state_ownership.validateStageStatePartitionFact;
pub const validateStateCleanupObligation = state_ownership.validateStateCleanupObligation;
pub const validateStateDescriptorLifecycleAction = state_ownership.validateStateDescriptorLifecycleAction;
pub const validateStateEpoch = state_ownership.validateStateEpoch;
pub const validateTensorFrameBatchEntryForStateLease = state_ownership.validateTensorFrameBatchEntryForStateLease;
pub const validateTensorFrameForPlanBoundary = tensor_frame.validateTensorFrameForPlanBoundary;
pub const validateTensorFrameForStateLeases = state_ownership.validateTensorFrameForStateLeases;
pub const validateTensorFramePlanIdentity = tensor_frame.validateTensorFramePlanIdentity;

test "inference bridge root exports state_ownership contract" {
    _ = state_ownership.StageStateOwnershipPlan;
    _ = StageStateOwnershipPlanId;
    _ = buildStageStateOwnershipPlan;
    _ = validateStageStateOwnershipPlan;
    _ = validateTensorFrameForStateLeases;
}
