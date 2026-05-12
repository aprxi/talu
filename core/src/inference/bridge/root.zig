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

pub const BoundaryDType = pipeline.BoundaryDType;
pub const BoundaryLayout = pipeline.BoundaryLayout;
pub const BoundaryNegotiationRequest = pipeline.BoundaryNegotiationRequest;
pub const BoundaryNegotiationResult = pipeline.BoundaryNegotiationResult;
pub const negotiateBoundaryContract = pipeline.negotiateBoundaryContract;

pub const ActivationFrameArgs = tensor_frame.ActivationFrameArgs;
pub const LocalDecodeHandoffConfig = orchestrator.LocalDecodeHandoffConfig;
pub const PipelineRuntime = pipeline.PipelineRuntime;
pub const PipelineRuntime3 = pipeline.PipelineRuntime3;
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
pub const boundaryRefFromPlanRef = tensor_frame.boundaryRefFromPlanRef;
pub const dtypeByteSize = tensor_frame.dtypeByteSize;
pub const emitTensorFrame = tensor_frame.emitTensorFrame;
pub const executeTwoStageForward = orchestrator.executeTwoStageForward;
pub const executeThreeStageForward = orchestrator.executeThreeStageForward;
pub const executeLocalDecodeHandoff = orchestrator.executeLocalDecodeHandoff;
pub const fromComputeDType = tensor_frame.fromComputeDType;
pub const selectedBoundaryTensorContract = tensor_frame.selectedBoundaryTensorContract;
pub const tensor_frame_contract_version = tensor_frame.tensor_frame_contract_version;
pub const tensorFrameLogicalEql = tensor_frame.tensorFrameLogicalEql;
pub const tensorFrameLogicalHash = tensor_frame.tensorFrameLogicalHash;
pub const toComputeDType = tensor_frame.toComputeDType;
pub const validatePayloadBufferLength = tensor_frame.validatePayloadBufferLength;
pub const validateTensorFrameForPlanBoundary = tensor_frame.validateTensorFrameForPlanBoundary;
pub const validateTensorFramePlanIdentity = tensor_frame.validateTensorFramePlanIdentity;
