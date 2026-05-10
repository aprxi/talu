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

pub const ActivationHandoffFrameArgs = tensor_frame.ActivationHandoffFrameArgs;
pub const BoundaryDType = pipeline.BoundaryDType;
pub const BoundaryLayout = pipeline.BoundaryLayout;
pub const BoundaryNegotiationRequest = pipeline.BoundaryNegotiationRequest;
pub const BoundaryNegotiationResult = pipeline.BoundaryNegotiationResult;
pub const negotiateBoundaryContract = pipeline.negotiateBoundaryContract;

pub const ActivationFrameArgs = tensor_frame.ActivationFrameArgs;
pub const PipelineRuntime = pipeline.PipelineRuntime;
pub const PipelineRuntime3 = pipeline.PipelineRuntime3;
pub const StageBackend = tensor_frame.StageBackend;
pub const StageBoundary = tensor_frame.StageBoundary;
pub const StageEndpoint = tensor_frame.StageEndpoint;
pub const TensorFrameDevice = tensor_frame.TensorFrameDevice;
pub const TensorFrameDType = tensor_frame.TensorFrameDType;
pub const TensorFrameLayout = tensor_frame.TensorFrameLayout;
pub const TensorFrameLifetime = tensor_frame.TensorFrameLifetime;
pub const TensorFrameMetadata = tensor_frame.TensorFrameMetadata;
pub const TensorFrameOwnership = tensor_frame.TensorFrameOwnership;
pub const TensorFrameRole = tensor_frame.TensorFrameRole;
pub const TensorFrameShape = tensor_frame.TensorFrameShape;
pub const TensorFrameValidationError = tensor_frame.TensorFrameValidationError;
pub const activationHandoffFrame = tensor_frame.activationHandoffFrame;
pub const activationFrameFromBoundary = tensor_frame.activationFrameFromBoundary;
pub const dtypeByteSize = tensor_frame.dtypeByteSize;
pub const executeTwoStageForward = orchestrator.executeTwoStageForward;
pub const executeThreeStageForward = orchestrator.executeThreeStageForward;
pub const tensor_frame_contract_version = tensor_frame.tensor_frame_contract_version;
pub const validateActivationFrameByteCount = tensor_frame.validateActivationFrameByteCount;
pub const validateStageBoundary = tensor_frame.validateStageBoundary;
