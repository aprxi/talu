//! Internal staged failure taxonomy and cleanup contracts.
//!
//! This module is metadata and validation only. It classifies typed failures
//! from the staged inference pipeline contracts, builds deterministic cleanup
//! plans from state ownership obligations, and records cleanup attempts without
//! changing scheduler, backend, transport, or public error behavior.

const std = @import("std");
const models = @import("models_pkg");
const runtime_contract = @import("runtime_contract_pkg");
const host_capability = @import("host_capability.zig");
const state_ownership = @import("state_ownership.zig");
const tensor_frame = @import("tensor_frame.zig");

const Allocator = std.mem.Allocator;
const Sha256 = std.crypto.hash.sha2.Sha256;
const StagePlan = stage_plan.StagePlan;
const StateLifecycle = runtime_contract.StateLifecycle;
const StateCleanupKind = state_ownership.StateCleanupKind;
const StageStateOwnershipPlan = state_ownership.StageStateOwnershipPlan;
const StateCleanupObligation = state_ownership.StateCleanupObligation;
const PlacementPlan = host_capability.PlacementPlan;
const HostId = host_capability.HostId;
const stage_plan = models.stage_plan;

pub const staged_error_contract_version: u32 = 1;
pub const StagedErrorContractVersion = u32;

pub const StagedErrorError = error{
    OutOfMemory,
    InvalidStagedErrorContractVersion,
    InvalidStagedFailureKind,
    InvalidStagedFailurePhase,
    InvalidStagedFailureScope,
    InvalidStagedSourceDomain,
    InvalidStagedErrorContext,
    MissingRequestId,
    MissingHostId,
    MissingStageId,
    MissingBoundaryContext,
    MissingCleanupStepIndex,
    IncompatibleSourceErrorClassification,
    CleanupPlanFingerprintMismatch,
    CleanupRequestIdMismatch,
    InvalidCleanupStepOrder,
    DuplicateCleanupStep,
    NonIdempotentCleanupStepUnsupported,
    CleanupReportFingerprintMismatch,
    CleanupReportPlanMismatch,
    InvalidCleanupAttemptOrder,
    InvalidCleanupReportCompletionFlags,
    CleanupFailureReplacedPrimaryFailure,
    ResourceExhausted,
    MissingStateOwnershipPlan,
    InvalidCleanupObligation,
    MissingCleanupPlan,
};

pub const StagedFailureKind = enum(u8) {
    unsupported_partition,
    incompatible_host_capability,
    graph_identity_mismatch,
    stage_plan_identity_mismatch,
    placement_identity_mismatch,
    invalid_tensor_frame,
    invalid_activation_payload,
    missing_resident_weights,
    stale_request_state,
    state_ownership_mismatch,
    remote_host_unavailable,
    transfer_failed,
    request_cancelled,
    stage_execution_failed,
    cleanup_failed,
    resource_exhausted,
    internal_contract_violation,
};

pub const StagedFailurePhase = enum(u8) {
    validation_before_mutation,
    stage_execution_before_state_mutation,
    stage_execution_after_state_mutation,
    frame_handoff,
    cleanup,
};

pub const StagedFailureScope = enum(u8) {
    plan,
    placement,
    request,
    stage,
    host,
    boundary,
    transport,
    cleanup,
};

pub const StagedSourceDomain = enum(u8) {
    stage_plan,
    tensor_frame,
    state_ownership,
    host_placement,
    runner,
    transport,
    cleanup,
    internal,
};

pub const StagedSourceError = struct {
    domain: StagedSourceDomain,
    source_error_name: ?[]const u8 = null,
};

pub const StagedErrorContext = struct {
    graph_digest: ?[32]u8 = null,
    graph_contract_version: ?u32 = null,
    stage_plan_contract_version: ?u32 = null,
    stage_plan_id: ?stage_plan.StagePlanId = null,
    placement_plan_id: ?host_capability.PlacementPlanId = null,
    state_ownership_plan_id: ?state_ownership.StageStateOwnershipPlanId = null,
    tensor_frame_id: ?tensor_frame.TensorFrameInstanceId = null,
    boundary_index: ?usize = null,
    source_stage_id: ?usize = null,
    target_stage_id: ?usize = null,
    stage_id: ?usize = null,
    host_id: ?HostId = null,
    source_host_id: ?HostId = null,
    target_host_id: ?HostId = null,
    request_id: ?u64 = null,
    slot_id: ?u64 = null,
    state_epoch: ?u64 = null,
    state_descriptor_id: ?u8 = null,
    state_lifecycle: ?StateLifecycle = null,
    state_cleanup_kind: ?StateCleanupKind = null,
    cleanup_step_index: ?usize = null,
    cleanup_attempt_index: ?usize = null,
};

pub const StagedFailure = struct {
    version: StagedErrorContractVersion = staged_error_contract_version,
    kind: StagedFailureKind,
    phase: StagedFailurePhase,
    scope: StagedFailureScope,
    context: StagedErrorContext = .{},
    source: StagedSourceError,
    diagnostic_detail: ?[]const u8 = null,
};

pub const StagedFailureClass = struct {
    kind: StagedFailureKind,
    phase: ?StagedFailurePhase = null,
    scope: StagedFailureScope,
    source: StagedSourceError,
};

pub const SourceClassificationContext = struct {
    stage_id: ?usize = null,
};

pub const RunnerClassificationContext = struct {
    phase: StagedFailurePhase,
};

pub const StagedFailureRequest = struct {
    kind: StagedFailureKind,
    phase: StagedFailurePhase,
    scope: StagedFailureScope,
    context: StagedErrorContext = .{},
    source: StagedSourceError,
    diagnostic_detail: ?[]const u8 = null,
};

pub const StagedFailureValidationOptions = struct {
    stage_plan: ?*const StagePlan = null,
    placement_plan: ?*const PlacementPlan = null,
    state_ownership_plan: ?*const StageStateOwnershipPlan = null,
    cleanup_plan: ?*const StagedCleanupPlan = null,
};

pub const StagedErrorReport = struct {
    arena: std.heap.ArenaAllocator,
    version: StagedErrorContractVersion,
    primary_failure: StagedFailure,
    cleanup_plan_id: ?StagedCleanupPlanId = null,
    cleanup_failures: []const StagedFailure,

    pub fn deinit(self: *StagedErrorReport) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

pub const TouchedStageCleanupRef = struct {
    stage_id: usize,
    request_id: u64,
    slot_id: u64,
    state_epoch: ?u64 = null,
};

pub const StagedCleanupPlanRequest = struct {
    primary_failure: StagedFailure,
    request_id: u64,
    placement_plan: ?*const PlacementPlan = null,
    state_ownership_plan: ?*const StageStateOwnershipPlan = null,
    touched_stages: []const TouchedStageCleanupRef = &.{},
    cleanup_obligations: []const StateCleanupObligation = &.{},
};

pub const StagedCleanupPlanId = struct {
    digest: [32]u8,
};

pub const StagedCleanupSource = enum(u8) {
    state_obligation,
};

pub const StagedCleanupStep = struct {
    index: usize,
    source: StagedCleanupSource = .state_obligation,
    host_id: ?HostId = null,
    stage_id: usize,
    request_id: u64,
    slot_id: u64,
    state_epoch: ?u64 = null,
    descriptor_id: u8,
    lifecycle: StateLifecycle,
    cleanup_kind: StateCleanupKind,
    order_key: u64,
    idempotent: bool = true,
};

pub const StagedCleanupPlan = struct {
    arena: std.heap.ArenaAllocator,
    version: StagedErrorContractVersion,
    plan_id: StagedCleanupPlanId,
    graph_digest: ?[32]u8 = null,
    graph_contract_version: ?u32 = null,
    stage_plan_contract_version: ?u32 = null,
    stage_plan_id: ?stage_plan.StagePlanId = null,
    placement_plan_id: ?host_capability.PlacementPlanId = null,
    state_ownership_plan_id: ?state_ownership.StageStateOwnershipPlanId = null,
    request_id: u64,
    steps: []const StagedCleanupStep,

    pub fn deinit(self: *StagedCleanupPlan) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

pub const StagedCleanupPlanValidationOptions = struct {
    placement_plan: ?*const PlacementPlan = null,
    state_ownership_plan: ?*const StageStateOwnershipPlan = null,
    touched_stages: []const TouchedStageCleanupRef = &.{},
    cleanup_obligations: []const StateCleanupObligation = &.{},
};

pub const StagedCleanupAttemptResult = enum(u8) {
    success,
    already_clean,
    failed,
};

pub const StagedCleanupAttempt = struct {
    cleanup_step_index: usize,
    attempt_index: usize,
    result: StagedCleanupAttemptResult,
    cleanup_failure: ?StagedFailure = null,
};

pub const StagedCleanupReportId = struct {
    digest: [32]u8,
};

pub const StagedCleanupReport = struct {
    arena: std.heap.ArenaAllocator,
    version: StagedErrorContractVersion,
    report_id: StagedCleanupReportId,
    cleanup_plan_id: StagedCleanupPlanId,
    attempts: []const StagedCleanupAttempt,
    cleanup_complete: bool,
    cleanup_failed: bool,

    pub fn deinit(self: *StagedCleanupReport) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

pub const StagedCleanupReportValidationOptions = struct {
    cleanup_plan: *const StagedCleanupPlan,
    primary_failure: ?*const StagedFailure = null,
};

pub const RunnerSourceError = error{
    OutOfMemory,
    RequestCancelled,
    StaleRequestState,
    StageExecutionFailed,
    UnknownRunnerFailure,
};

pub const TransportSourceError = error{
    OutOfMemory,
    RequestCancelled,
    RemoteHostUnavailable,
    TransferFailed,
    UnknownTransportFailure,
};

pub const CleanupCallbackSourceError = error{
    OutOfMemory,
    CleanupFailed,
    UnknownCleanupFailure,
};

pub fn stagedCleanupPlanIdEql(lhs: StagedCleanupPlanId, rhs: StagedCleanupPlanId) bool {
    return std.mem.eql(u8, &lhs.digest, &rhs.digest);
}

pub fn stagedCleanupReportIdEql(lhs: StagedCleanupReportId, rhs: StagedCleanupReportId) bool {
    return std.mem.eql(u8, &lhs.digest, &rhs.digest);
}

pub fn classifyStagePlanError(err: stage_plan.StagePlanError, context: SourceClassificationContext) StagedFailureClass {
    const source = sourceError(.stage_plan, err);
    return switch (err) {
        error.OutOfMemory => classified(.resource_exhausted, .plan, source),
        error.MissingGraphIdentity, error.GraphIdentityMismatch => classified(.graph_identity_mismatch, .plan, source),
        error.InvalidContractVersion, error.PlanFingerprintMismatch => classified(.stage_plan_identity_mismatch, .plan, source),
        error.InvalidLayerCount,
        error.InvalidSplitPoint,
        error.DuplicateSplitPoint,
        error.ForbiddenSplitPoint,
        error.MissingPartitionConstraints,
        error.MissingStageDependency,
        error.UnsupportedRoleOwnerOverride,
        error.DuplicateRoleOwnerOverride,
        error.MissingRoleOwner,
        error.UnclassifiedGlobalNotAllowed,
        error.InvalidDependency,
        error.MissingRoleSemantics,
        error.DuplicateDependency,
        error.InvalidStageRange,
        error.DuplicateStageId,
        error.NonContiguousStageRange,
        => classified(.unsupported_partition, .plan, source),
        error.UnknownStageId => classified(.unsupported_partition, if (context.stage_id != null) .stage else .plan, source),
        error.ResidencyMismatch => classified(.internal_contract_violation, .plan, source),
        else => classified(.internal_contract_violation, .plan, source),
    };
}

pub fn classifyTensorFrameError(err: tensor_frame.TensorFrameValidationError) StagedFailureClass {
    const source = sourceError(.tensor_frame, err);
    return switch (err) {
        error.OutOfMemory => classified(.resource_exhausted, .boundary, source),
        error.GraphIdentityMismatch => classified(.graph_identity_mismatch, .boundary, source),
        error.StagePlanIdentityMismatch => classified(.stage_plan_identity_mismatch, .boundary, source),
        error.BoundaryIndexOutOfRange,
        error.AbsentStageBoundary,
        error.InvalidSourceStage,
        error.InvalidTargetStage,
        error.InvalidProducerLayerRange,
        error.InvalidConsumerLayerRange,
        error.MissingSelectedBoundaryTensorContract,
        error.BoundaryTensorContractMismatch,
        => classified(.invalid_tensor_frame, .boundary, source),
        error.InvalidTensorFrameContractVersion,
        error.InvalidFrameId,
        error.InvalidActivationRole,
        error.InvalidStepKind,
        error.InvalidDType,
        error.UnsupportedActivationDType,
        error.InvalidTensorRank,
        error.InvalidTensorShape,
        error.InvalidHiddenSize,
        error.InvalidTensorStride,
        error.UnsupportedTensorLayout,
        error.NonContiguousPayload,
        error.InvalidBatch,
        error.DuplicateBatchIndex,
        error.DuplicateRequestId,
        error.DuplicateSlotId,
        error.InvalidRequestId,
        error.InvalidSlotId,
        error.InvalidSequenceRange,
        error.RaggedBatchUnsupported,
        error.InvalidOwnership,
        error.InvalidLifetime,
        => classified(.invalid_tensor_frame, .boundary, source),
        error.ByteCountOverflow,
        error.InvalidLogicalByteCount,
        error.InvalidPayloadByteCount,
        error.PayloadBufferLengthMismatch,
        => classified(.invalid_activation_payload, .boundary, source),
        error.ObserverFailure => classified(.internal_contract_violation, .boundary, source),
        else => classified(.internal_contract_violation, .boundary, source),
    };
}

pub fn classifyStateOwnershipError(err: state_ownership.StateOwnershipError) StagedFailureClass {
    const source = sourceError(.state_ownership, err);
    return switch (err) {
        error.OutOfMemory => classified(.resource_exhausted, .request, source),
        error.GraphIdentityMismatch => classified(.graph_identity_mismatch, .request, source),
        error.StagePlanIdentityMismatch => classified(.stage_plan_identity_mismatch, .request, source),
        error.StateRequestMismatch,
        error.StateSlotMismatch,
        error.StateStageMismatch,
        error.StatePositionMismatch,
        error.StaleStateEpoch,
        error.MissingStateEpoch,
        error.InvalidLeaseState,
        error.InvalidLeaseTransition,
        error.SlotAlreadyBound,
        error.StaleLeaseEpochOnSlotReuse,
        => classified(.stale_request_state, .request, source),
        error.InvalidStateOwnershipContractVersion,
        error.StateOwnershipPlanIdentityMismatch,
        error.StateOwnershipPlanFingerprintMismatch,
        error.MissingStageDescriptorSet,
        error.DuplicateStageDescriptorSet,
        error.DuplicateStateDescriptorId,
        error.InvalidStateDescriptorAlignment,
        error.InvalidStateDescriptorSize,
        error.InvalidStateLifecycle,
        error.InvalidStateLifecycleAction,
        error.MissingStateDescriptor,
        error.UnknownStateDescriptorId,
        error.UnsupportedStatePartition,
        error.MissingRequiredStateDependency,
        error.InvalidStatePartitionFact,
        error.DuplicateStatePartitionFact,
        error.CrossStageStateMigrationUnsupported,
        error.CrossStageStateAliasUnsupported,
        error.InvalidStateBindingReport,
        => classified(.state_ownership_mismatch, .request, source),
        error.InvalidCleanupObligation => classified(.cleanup_failed, .cleanup, source),
        else => classified(.internal_contract_violation, .request, source),
    };
}

pub fn classifyPlacementError(err: host_capability.PlacementError) StagedFailureClass {
    const source = sourceError(.host_placement, err);
    return switch (err) {
        error.OutOfMemory => classified(.resource_exhausted, .placement, source),
        error.GraphIdentityMismatch => classified(.graph_identity_mismatch, .placement, source),
        error.StagePlanIdentityMismatch => classified(.stage_plan_identity_mismatch, .placement, source),
        error.InvalidPlacementContractVersion, error.PlacementPlanFingerprintMismatch => classified(.placement_identity_mismatch, .placement, source),
        error.HostCapabilityFingerprintMismatch,
        error.InvalidHostId,
        error.DuplicateHostCapability,
        error.MissingHostCapability,
        error.UnsupportedBackendKind,
        error.UnsupportedReachabilityKind,
        error.RemoteReachabilityNotAllowed,
        error.UnsupportedTensorFrameContractVersion,
        error.UnsupportedBoundaryDType,
        error.UnsupportedBoundaryLayout,
        error.UnsupportedStepKind,
        error.UnsupportedHandoffMode,
        error.InvalidBatchEnvelope,
        error.InvalidTokenEnvelope,
        error.InvalidActivationPayloadEnvelope,
        error.ResidentCheckpointBudgetExceeded,
        error.InvalidResidentCheckpointBudget,
        error.InvalidDiagnosticWorkspaceBudget,
        error.DuplicateHostFrameCapability,
        error.InvalidCapabilitySet,
        => classified(.incompatible_host_capability, .placement, source),
        error.HostResidencyFingerprintMismatch,
        error.DuplicateHostResidencySnapshot,
        error.MissingHostResidency,
        error.InvalidResidentStageRange,
        error.MissingResidentStage,
        error.DuplicateResidentStage,
        error.MissingResidentGlobalRole,
        error.WrongResidentGlobalRoleOwner,
        error.ResidencyMismatch,
        => classified(.missing_resident_weights, .placement, source),
        error.UnknownHostId,
        error.MissingStageBinding,
        error.DuplicateStageBinding,
        error.ExtraStageBinding,
        error.InvalidRequiredStepKindSet,
        error.DuplicateBoundaryFrameProfile,
        error.MissingBoundaryFrameProfile,
        error.BoundaryFrameProfileMismatch,
        => classified(.incompatible_host_capability, .placement, source),
        error.InvalidStatePlacementMode,
        error.MissingStatePlacementSummary,
        error.MissingResidentStateOwnership,
        error.MismatchedResidentStateDescriptorSummary,
        error.UnsupportedStateOwnershipContractVersion,
        => classified(.state_ownership_mismatch, .placement, source),
        else => classified(.internal_contract_violation, .placement, source),
    };
}

pub fn classifyRunnerError(err: RunnerSourceError, context: RunnerClassificationContext) StagedErrorError!StagedFailureClass {
    const source = sourceError(.runner, err);
    return switch (err) {
        error.OutOfMemory => classifiedWithPhase(.resource_exhausted, .stage, source, context.phase),
        error.RequestCancelled => classifiedWithPhase(.request_cancelled, .request, source, context.phase),
        error.StaleRequestState => classifiedWithPhase(.stale_request_state, .request, source, context.phase),
        error.StageExecutionFailed => blk: {
            switch (context.phase) {
                .stage_execution_before_state_mutation,
                .stage_execution_after_state_mutation,
                => {},
                .validation_before_mutation,
                .frame_handoff,
                .cleanup,
                => return error.InvalidStagedFailurePhase,
            }
            break :blk classifiedWithPhase(.stage_execution_failed, .stage, source, context.phase);
        },
        error.UnknownRunnerFailure => classifiedWithPhase(.internal_contract_violation, .stage, source, context.phase),
    };
}

pub fn classifyTransportError(err: TransportSourceError) StagedFailureClass {
    const source = sourceError(.transport, err);
    return switch (err) {
        error.OutOfMemory => classified(.resource_exhausted, .transport, source),
        error.RequestCancelled => classified(.request_cancelled, .request, source),
        error.RemoteHostUnavailable => classified(.remote_host_unavailable, .host, source),
        error.TransferFailed => classified(.transfer_failed, .transport, source),
        error.UnknownTransportFailure => classified(.internal_contract_violation, .transport, source),
    };
}

pub fn classifyCleanupCallbackError(err: CleanupCallbackSourceError) StagedFailureClass {
    const source = sourceError(.cleanup, err);
    return switch (err) {
        error.OutOfMemory => classified(.resource_exhausted, .cleanup, source),
        error.CleanupFailed => classified(.cleanup_failed, .cleanup, source),
        error.UnknownCleanupFailure => classified(.internal_contract_violation, .cleanup, source),
    };
}

pub fn buildStagedFailure(request: StagedFailureRequest, options: StagedFailureValidationOptions) StagedErrorError!StagedFailure {
    const failure = StagedFailure{
        .version = staged_error_contract_version,
        .kind = request.kind,
        .phase = request.phase,
        .scope = request.scope,
        .context = request.context,
        .source = request.source,
        .diagnostic_detail = request.diagnostic_detail,
    };
    try validateStagedFailure(&failure, options);
    return failure;
}

pub fn validateStagedFailure(failure: *const StagedFailure, options: StagedFailureValidationOptions) StagedErrorError!void {
    if (failure.version != staged_error_contract_version) return error.InvalidStagedErrorContractVersion;
    try validateFailureContext(failure);
    try validateSourceCompatibility(failure.source.domain, failure.kind);
    try validateFailureAgainstStagePlan(failure, options.stage_plan);
    try validateFailureAgainstPlacementPlan(failure, options.placement_plan);
    try validateFailureAgainstStateOwnershipPlan(failure, options.state_ownership_plan);
    try validateFailureAgainstCleanupPlan(failure, options.cleanup_plan);
}

pub fn buildStagedErrorReport(
    allocator: Allocator,
    primary_failure: StagedFailure,
    cleanup_plan_id: ?StagedCleanupPlanId,
    cleanup_failures: []const StagedFailure,
    validation_options: StagedFailureValidationOptions,
) StagedErrorError!StagedErrorReport {
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_allocator = arena.allocator();
    const copied_failures = try arena_allocator.dupe(StagedFailure, cleanup_failures);
    var report = StagedErrorReport{
        .arena = arena,
        .version = staged_error_contract_version,
        .primary_failure = primary_failure,
        .cleanup_plan_id = cleanup_plan_id,
        .cleanup_failures = copied_failures,
    };
    try validateStagedErrorReport(&report, validation_options);
    return report;
}

pub fn attachCleanupFailures(
    allocator: Allocator,
    report: *const StagedErrorReport,
    cleanup_failures: []const StagedFailure,
    validation_options: StagedFailureValidationOptions,
) StagedErrorError!StagedErrorReport {
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_allocator = arena.allocator();
    const combined = try arena_allocator.alloc(StagedFailure, report.cleanup_failures.len + cleanup_failures.len);
    @memcpy(combined[0..report.cleanup_failures.len], report.cleanup_failures);
    @memcpy(combined[report.cleanup_failures.len..], cleanup_failures);
    std.mem.sort(StagedFailure, combined, {}, cleanupFailureLessThan);
    var next = StagedErrorReport{
        .arena = arena,
        .version = staged_error_contract_version,
        .primary_failure = report.primary_failure,
        .cleanup_plan_id = report.cleanup_plan_id,
        .cleanup_failures = combined,
    };
    try validateStagedErrorReport(&next, validation_options);
    return next;
}

pub fn validateStagedErrorReport(report: *const StagedErrorReport, validation_options: StagedFailureValidationOptions) StagedErrorError!void {
    if (report.version != staged_error_contract_version) return error.InvalidStagedErrorContractVersion;
    try validateStagedFailure(&report.primary_failure, validation_options);
    if (report.cleanup_failures.len > 0 and report.primary_failure.kind == .cleanup_failed and report.primary_failure.phase == .cleanup) {
        return error.CleanupFailureReplacedPrimaryFailure;
    }
    var previous_step: ?usize = null;
    var previous_attempt: ?usize = null;
    for (report.cleanup_failures) |failure| {
        if (failure.kind != .cleanup_failed or failure.phase != .cleanup or failure.scope != .cleanup) {
            return error.InvalidStagedErrorContext;
        }
        if (failure.context.cleanup_step_index == null or failure.context.cleanup_attempt_index == null) {
            return error.MissingCleanupStepIndex;
        }
        try validateStagedFailure(&failure, validation_options);
        const step_index = failure.context.cleanup_step_index.?;
        const attempt_index = failure.context.cleanup_attempt_index.?;
        if (previous_step) |prev_step| {
            if (step_index < prev_step or (step_index == prev_step and attempt_index < previous_attempt.?)) {
                return error.InvalidCleanupAttemptOrder;
            }
        }
        previous_step = step_index;
        previous_attempt = attempt_index;
    }
    if (report.cleanup_plan_id) |report_plan_id| {
        const cleanup_plan = validation_options.cleanup_plan orelse {
            if (report.cleanup_failures.len != 0) return error.MissingCleanupPlan;
            return;
        };
        if (!stagedCleanupPlanIdEql(report_plan_id, cleanup_plan.plan_id)) return error.CleanupReportPlanMismatch;
    }
}

pub fn buildStagedCleanupPlan(allocator: Allocator, request: StagedCleanupPlanRequest) StagedErrorError!StagedCleanupPlan {
    try validatePrimaryFailureForCleanupInputs(&request.primary_failure, request.request_id, request.placement_plan, request.state_ownership_plan);
    if ((request.touched_stages.len != 0 or request.cleanup_obligations.len != 0) and request.request_id == 0) {
        return error.MissingRequestId;
    }
    if (request.cleanup_obligations.len != 0 and request.state_ownership_plan == null) return error.MissingStateOwnershipPlan;
    try validateTouchedStageRefs(request.request_id, request.touched_stages);
    try validateCleanupObligationsInput(request.request_id, request.state_ownership_plan, request.cleanup_obligations);
    try validateCleanupInputsMatch(request.request_id, request.touched_stages, request.cleanup_obligations);
    if (request.placement_plan != null and request.state_ownership_plan != null) {
        try validatePlacementStateIdentity(request.placement_plan.?, request.state_ownership_plan.?);
    }

    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_allocator = arena.allocator();
    const steps = try arena_allocator.alloc(StagedCleanupStep, request.cleanup_obligations.len);
    var used = try arena_allocator.alloc(bool, request.cleanup_obligations.len);
    @memset(used, false);
    var step_count: usize = 0;
    var remaining_touched = request.touched_stages.len;
    while (remaining_touched > 0) {
        remaining_touched -= 1;
        const touched = request.touched_stages[remaining_touched];
        while (nextObligationIndexForTouched(request.cleanup_obligations, used, touched)) |obligation_index| {
            used[obligation_index] = true;
            const obligation = request.cleanup_obligations[obligation_index];
            steps[step_count] = try cleanupStepFromObligation(step_count, obligation, touched, request.placement_plan);
            step_count += 1;
        }
    }
    if (step_count != request.cleanup_obligations.len) return error.InvalidCleanupObligation;
    try validateDuplicateCleanupSteps(steps[0..step_count]);

    var plan = StagedCleanupPlan{
        .arena = arena,
        .version = staged_error_contract_version,
        .plan_id = zeroCleanupPlanId(),
        .request_id = request.request_id,
        .steps = steps[0..step_count],
    };
    copyCleanupPlanIdentity(&plan, request.placement_plan, request.state_ownership_plan);
    plan.plan_id = computeStagedCleanupPlanId(&plan);
    try validateStagedCleanupPlan(&plan, .{
        .placement_plan = request.placement_plan,
        .state_ownership_plan = request.state_ownership_plan,
        .touched_stages = request.touched_stages,
        .cleanup_obligations = request.cleanup_obligations,
    });
    return plan;
}

pub fn validateStagedCleanupPlan(plan: *const StagedCleanupPlan, options: StagedCleanupPlanValidationOptions) StagedErrorError!void {
    if (plan.version != staged_error_contract_version) return error.InvalidStagedErrorContractVersion;
    if (!stagedCleanupPlanIdEql(plan.plan_id, computeStagedCleanupPlanId(plan))) return error.CleanupPlanFingerprintMismatch;
    if (plan.steps.len != 0 and plan.request_id == 0) return error.MissingRequestId;
    try validateTouchedStageRefs(plan.request_id, options.touched_stages);
    if (plan.steps.len != 0 and options.state_ownership_plan == null) return error.MissingStateOwnershipPlan;
    try validateCleanupObligationsInput(plan.request_id, options.state_ownership_plan, options.cleanup_obligations);
    try validateCleanupInputsMatch(plan.request_id, options.touched_stages, options.cleanup_obligations);
    if (plan.steps.len != options.cleanup_obligations.len) return error.InvalidCleanupObligation;
    if (options.placement_plan != null and options.state_ownership_plan != null) {
        try validatePlacementStateIdentity(options.placement_plan.?, options.state_ownership_plan.?);
    }
    try validateCleanupPlanIdentity(plan, options.placement_plan, options.state_ownership_plan);
    try validateDuplicateCleanupSteps(plan.steps);
    try validateCleanupStepOrder(plan.steps, options.touched_stages);
    for (plan.steps, 0..) |step, index| {
        if (step.index != index) return error.InvalidCleanupStepOrder;
        if (step.request_id != plan.request_id) return error.CleanupRequestIdMismatch;
        if (!step.idempotent) return error.NonIdempotentCleanupStepUnsupported;
        try validateStepAgainstInputs(step, options);
    }
}

pub fn buildStagedCleanupReport(
    allocator: Allocator,
    attempts: []const StagedCleanupAttempt,
    options: StagedCleanupReportValidationOptions,
) StagedErrorError!StagedCleanupReport {
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_allocator = arena.allocator();
    const cleanup_plan = options.cleanup_plan;
    const copied_attempts = try arena_allocator.dupe(StagedCleanupAttempt, attempts);
    const flags = cleanupReportFlags(copied_attempts);
    var report = StagedCleanupReport{
        .arena = arena,
        .version = staged_error_contract_version,
        .report_id = zeroCleanupReportId(),
        .cleanup_plan_id = cleanup_plan.plan_id,
        .attempts = copied_attempts,
        .cleanup_complete = flags.cleanup_complete,
        .cleanup_failed = flags.cleanup_failed,
    };
    report.report_id = computeStagedCleanupReportId(&report);
    try validateStagedCleanupReport(&report, options);
    return report;
}

pub fn validateStagedCleanupReport(report: *const StagedCleanupReport, options: StagedCleanupReportValidationOptions) StagedErrorError!void {
    if (report.version != staged_error_contract_version) return error.InvalidStagedErrorContractVersion;
    if (!stagedCleanupReportIdEql(report.report_id, computeStagedCleanupReportId(report))) return error.CleanupReportFingerprintMismatch;
    if (!stagedCleanupPlanIdEql(report.cleanup_plan_id, options.cleanup_plan.plan_id)) return error.CleanupReportPlanMismatch;
    if (report.attempts.len != options.cleanup_plan.steps.len) return error.InvalidCleanupAttemptOrder;
    for (report.attempts, 0..) |attempt, index| {
        if (attempt.cleanup_step_index != index or attempt.attempt_index != 0) return error.InvalidCleanupAttemptOrder;
        const step = options.cleanup_plan.steps[index];
        switch (attempt.result) {
            .success, .already_clean => if (attempt.cleanup_failure != null) return error.InvalidCleanupReportCompletionFlags,
            .failed => {
                const failure = attempt.cleanup_failure orelse return error.InvalidCleanupReportCompletionFlags;
                try validateStagedFailure(&failure, .{ .cleanup_plan = options.cleanup_plan });
                try validateCleanupFailureMatchesAttempt(failure, step, attempt.attempt_index);
            },
        }
    }
    const flags = cleanupReportFlags(report.attempts);
    if (report.cleanup_complete != flags.cleanup_complete or report.cleanup_failed != flags.cleanup_failed) {
        return error.InvalidCleanupReportCompletionFlags;
    }
    if (options.primary_failure) |primary| {
        try validateStagedFailure(primary, .{ .cleanup_plan = options.cleanup_plan });
        try validatePrimaryFailureForCleanupPlan(primary, options.cleanup_plan);
        if (primary.kind == .cleanup_failed and primary.phase == .cleanup and report.cleanup_failed) {
            return error.CleanupFailureReplacedPrimaryFailure;
        }
    }
}

pub fn stagedCleanupRequired(
    primary_failure: StagedFailure,
    touched_stages: []const TouchedStageCleanupRef,
    cleanup_obligations: []const StateCleanupObligation,
) bool {
    if (cleanup_obligations.len != 0) return true;
    if (touched_stages.len != 0) return true;
    return primary_failure.phase == .stage_execution_after_state_mutation;
}

fn classified(kind: StagedFailureKind, scope: StagedFailureScope, source: StagedSourceError) StagedFailureClass {
    return .{ .kind = kind, .scope = scope, .source = source };
}

fn classifiedWithPhase(kind: StagedFailureKind, scope: StagedFailureScope, source: StagedSourceError, phase: StagedFailurePhase) StagedFailureClass {
    return .{ .kind = kind, .phase = phase, .scope = scope, .source = source };
}

fn sourceError(domain: StagedSourceDomain, err: anyerror) StagedSourceError {
    return .{ .domain = domain, .source_error_name = @errorName(err) };
}

fn validateFailureContext(failure: *const StagedFailure) StagedErrorError!void {
    const context = failure.context;
    switch (failure.scope) {
        .request => if (context.request_id == null or context.request_id.? == 0) return error.MissingRequestId,
        .host => if (context.host_id == null or context.host_id.?.value == 0) return error.MissingHostId,
        .stage => if (context.stage_id == null) return error.MissingStageId,
        .boundary, .transport => if (context.boundary_index == null) return error.MissingBoundaryContext,
        else => {},
    }
    if (failure.kind == .cleanup_failed and failure.phase != .cleanup) return error.InvalidStagedErrorContext;
    if (failure.source.domain == .cleanup and failure.kind != .cleanup_failed and failure.kind != .resource_exhausted and failure.kind != .internal_contract_violation) {
        return error.IncompatibleSourceErrorClassification;
    }
    if (failure.kind == .cleanup_failed and failure.source.domain != .cleanup and failure.source.domain != .state_ownership) {
        return error.IncompatibleSourceErrorClassification;
    }
    if (failure.phase != .cleanup and failure.source.domain == .cleanup) return error.InvalidStagedErrorContext;
    if (context.cleanup_attempt_index != null and context.cleanup_step_index == null) return error.MissingCleanupStepIndex;
    const has_state_cleanup_field = context.state_descriptor_id != null or context.state_lifecycle != null or context.state_cleanup_kind != null;
    if (has_state_cleanup_field) {
        if (context.cleanup_step_index == null) return error.MissingCleanupStepIndex;
        if (context.state_descriptor_id == null or context.state_lifecycle == null or context.state_cleanup_kind == null) {
            return error.InvalidStagedErrorContext;
        }
    }
}

fn validateSourceCompatibility(domain: StagedSourceDomain, kind: StagedFailureKind) StagedErrorError!void {
    const compatible = switch (domain) {
        .stage_plan => kind == .unsupported_partition or kind == .graph_identity_mismatch or kind == .stage_plan_identity_mismatch or kind == .resource_exhausted or kind == .internal_contract_violation,
        .tensor_frame => kind == .invalid_tensor_frame or kind == .invalid_activation_payload or kind == .graph_identity_mismatch or kind == .stage_plan_identity_mismatch or kind == .resource_exhausted or kind == .internal_contract_violation,
        .state_ownership => kind == .stale_request_state or kind == .state_ownership_mismatch or kind == .cleanup_failed or kind == .graph_identity_mismatch or kind == .stage_plan_identity_mismatch or kind == .resource_exhausted or kind == .internal_contract_violation,
        .host_placement => kind == .incompatible_host_capability or kind == .missing_resident_weights or kind == .graph_identity_mismatch or kind == .stage_plan_identity_mismatch or kind == .placement_identity_mismatch or kind == .state_ownership_mismatch or kind == .resource_exhausted or kind == .internal_contract_violation,
        .runner => kind == .stage_execution_failed or kind == .request_cancelled or kind == .stale_request_state or kind == .resource_exhausted or kind == .internal_contract_violation,
        .transport => kind == .transfer_failed or kind == .remote_host_unavailable or kind == .request_cancelled or kind == .resource_exhausted or kind == .internal_contract_violation,
        .cleanup => kind == .cleanup_failed or kind == .resource_exhausted or kind == .internal_contract_violation,
        .internal => kind == .resource_exhausted or kind == .internal_contract_violation,
    };
    if (!compatible) return error.IncompatibleSourceErrorClassification;
}

fn validatePrimaryFailureForCleanupInputs(
    primary_failure: *const StagedFailure,
    request_id: u64,
    placement_plan: ?*const PlacementPlan,
    state_ownership_plan: ?*const StageStateOwnershipPlan,
) StagedErrorError!void {
    try validateStagedFailure(primary_failure, .{
        .placement_plan = placement_plan,
        .state_ownership_plan = state_ownership_plan,
    });
    try validatePrimaryFailureRequestId(primary_failure.*, request_id);
}

fn validatePrimaryFailureForCleanupPlan(primary_failure: *const StagedFailure, cleanup_plan: *const StagedCleanupPlan) StagedErrorError!void {
    try validatePrimaryFailureRequestId(primary_failure.*, cleanup_plan.request_id);
    try validateFailureIdentityAgainstCleanupPlan(primary_failure.context, cleanup_plan);
}

fn validatePrimaryFailureRequestId(primary_failure: StagedFailure, request_id: u64) StagedErrorError!void {
    const primary_request_id = primary_failure.context.request_id orelse {
        if (request_id != 0) return error.MissingRequestId;
        return;
    };
    if (primary_request_id == 0) return error.MissingRequestId;
    if (primary_request_id != request_id) return error.CleanupRequestIdMismatch;
}

fn validateFailureAgainstStagePlan(failure: *const StagedFailure, plan_opt: ?*const StagePlan) StagedErrorError!void {
    const plan = plan_opt orelse return;
    const context = failure.context;
    if (context.graph_digest) |digest| {
        if (!std.mem.eql(u8, &digest, &plan.graph_identity.digest)) return error.InvalidStagedErrorContext;
    }
    if (context.graph_contract_version) |version| {
        if (version != plan.graph_identity.graph_contract_version) return error.InvalidStagedErrorContext;
    }
    if (context.stage_plan_contract_version) |version| {
        if (version != plan.stage_contract_version) return error.InvalidStagedErrorContext;
    }
    if (context.stage_plan_id) |id| {
        if (!std.mem.eql(u8, &id.digest, &plan.plan_id.digest)) return error.InvalidStagedErrorContext;
    }
    if (context.stage_id) |stage_id| {
        if (stage_id >= plan.stages.len or plan.stages[stage_id].id != stage_id) return error.InvalidStagedErrorContext;
    }
    if (context.boundary_index) |boundary_index| {
        if (boundary_index >= plan.boundaries.len) return error.InvalidStagedErrorContext;
        const boundary = plan.boundaries[boundary_index];
        if (context.source_stage_id) |source| {
            if (source != boundary.source_stage_id) return error.InvalidStagedErrorContext;
        }
        if (context.target_stage_id) |target| {
            if (target != boundary.target_stage_id) return error.InvalidStagedErrorContext;
        }
    }
}

fn validateFailureAgainstPlacementPlan(failure: *const StagedFailure, plan_opt: ?*const PlacementPlan) StagedErrorError!void {
    const plan = plan_opt orelse return;
    const context = failure.context;
    if (context.graph_digest) |digest| {
        if (!std.mem.eql(u8, &digest, &plan.graph_digest)) return error.InvalidStagedErrorContext;
    }
    if (context.graph_contract_version) |version| {
        if (version != plan.graph_contract_version) return error.InvalidStagedErrorContext;
    }
    if (context.stage_plan_contract_version) |version| {
        if (version != plan.stage_plan_contract_version) return error.InvalidStagedErrorContext;
    }
    if (context.stage_plan_id) |id| {
        if (!std.mem.eql(u8, &id.digest, &plan.stage_plan_id.digest)) return error.InvalidStagedErrorContext;
    }
    if (context.placement_plan_id) |id| {
        if (!host_capability.placementPlanIdEql(id, plan.plan_id)) return error.InvalidStagedErrorContext;
    }
    if (context.state_ownership_plan_id) |id| {
        if (plan.state_ownership_plan_id == null or !state_ownership.stateOwnershipPlanIdEql(id, plan.state_ownership_plan_id.?)) {
            return error.InvalidStagedErrorContext;
        }
    }
    if (context.boundary_index) |boundary_index| {
        const boundary = placementBoundaryForIndex(plan, boundary_index) orelse return error.InvalidStagedErrorContext;
        if (context.source_stage_id) |source| {
            if (source != boundary.source_stage_id) return error.InvalidStagedErrorContext;
        }
        if (context.target_stage_id) |target| {
            if (target != boundary.target_stage_id) return error.InvalidStagedErrorContext;
        }
        const source_binding = host_capability.bindingForStage(plan, boundary.source_stage_id) catch return error.InvalidStagedErrorContext;
        const target_binding = host_capability.bindingForStage(plan, boundary.target_stage_id) catch return error.InvalidStagedErrorContext;
        if (context.source_host_id) |host_id| {
            if (!hostIdEql(host_id, source_binding.host_id)) return error.InvalidStagedErrorContext;
        }
        if (context.target_host_id) |host_id| {
            if (!hostIdEql(host_id, target_binding.host_id)) return error.InvalidStagedErrorContext;
        }
    }
    if (context.stage_id) |stage_id| {
        const binding = host_capability.bindingForStage(plan, stage_id) catch return error.InvalidStagedErrorContext;
        if (context.host_id) |host_id| {
            if (!hostIdEql(host_id, binding.host_id)) return error.InvalidStagedErrorContext;
        }
    }
    if (context.source_stage_id) |stage_id| {
        const binding = host_capability.bindingForStage(plan, stage_id) catch return error.InvalidStagedErrorContext;
        if (context.source_host_id) |host_id| {
            if (!hostIdEql(host_id, binding.host_id)) return error.InvalidStagedErrorContext;
        }
    }
    if (context.target_stage_id) |stage_id| {
        const binding = host_capability.bindingForStage(plan, stage_id) catch return error.InvalidStagedErrorContext;
        if (context.target_host_id) |host_id| {
            if (!hostIdEql(host_id, binding.host_id)) return error.InvalidStagedErrorContext;
        }
    }
}

fn validateFailureAgainstStateOwnershipPlan(failure: *const StagedFailure, plan_opt: ?*const StageStateOwnershipPlan) StagedErrorError!void {
    const plan = plan_opt orelse return;
    const context = failure.context;
    if (context.graph_digest) |digest| {
        if (!std.mem.eql(u8, &digest, &plan.graph_digest)) return error.InvalidStagedErrorContext;
    }
    if (context.graph_contract_version) |version| {
        if (version != plan.graph_contract_version) return error.InvalidStagedErrorContext;
    }
    if (context.stage_plan_contract_version) |version| {
        if (version != plan.stage_plan_contract_version) return error.InvalidStagedErrorContext;
    }
    if (context.stage_plan_id) |id| {
        if (!std.mem.eql(u8, &id.digest, &plan.stage_plan_id.digest)) return error.InvalidStagedErrorContext;
    }
    if (context.state_ownership_plan_id) |id| {
        if (!state_ownership.stateOwnershipPlanIdEql(id, plan.plan_id)) return error.InvalidStagedErrorContext;
    }
    if (context.stage_id) |stage_id| {
        if (!statePlanHasStage(plan, stage_id)) return error.InvalidStagedErrorContext;
    }
    if (context.state_descriptor_id) |descriptor_id| {
        const stage_id = context.stage_id orelse return error.MissingStageId;
        const descriptor = descriptorForStage(plan, stage_id, descriptor_id) orelse return error.InvalidStagedErrorContext;
        if (context.state_lifecycle) |lifecycle| {
            if (lifecycle != descriptor.lifecycle) return error.InvalidStagedErrorContext;
        }
    }
}

fn validateFailureAgainstCleanupPlan(failure: *const StagedFailure, plan_opt: ?*const StagedCleanupPlan) StagedErrorError!void {
    const plan = plan_opt orelse return;
    const context = failure.context;
    try validateFailureIdentityAgainstCleanupPlan(context, plan);
    if (context.cleanup_step_index) |step_index| {
        if (step_index >= plan.steps.len) return error.InvalidStagedErrorContext;
        try validateCleanupFailureMatchesStepFields(failure.*, plan.steps[step_index]);
    }
}

fn validateFailureIdentityAgainstCleanupPlan(context: StagedErrorContext, plan: *const StagedCleanupPlan) StagedErrorError!void {
    if (context.request_id) |request_id| {
        if (request_id != plan.request_id) return error.CleanupRequestIdMismatch;
    }
    if (context.graph_digest) |digest| {
        if (plan.graph_digest == null or !std.mem.eql(u8, &digest, &plan.graph_digest.?)) return error.InvalidStagedErrorContext;
    }
    if (context.graph_contract_version) |version| {
        if (plan.graph_contract_version == null or version != plan.graph_contract_version.?) return error.InvalidStagedErrorContext;
    }
    if (context.stage_plan_contract_version) |version| {
        if (plan.stage_plan_contract_version == null or version != plan.stage_plan_contract_version.?) return error.InvalidStagedErrorContext;
    }
    if (context.stage_plan_id) |id| {
        if (plan.stage_plan_id == null or !std.mem.eql(u8, &id.digest, &plan.stage_plan_id.?.digest)) return error.InvalidStagedErrorContext;
    }
    if (context.placement_plan_id) |id| {
        if (plan.placement_plan_id == null or !host_capability.placementPlanIdEql(id, plan.placement_plan_id.?)) return error.InvalidStagedErrorContext;
    }
    if (context.state_ownership_plan_id) |id| {
        if (plan.state_ownership_plan_id == null or !state_ownership.stateOwnershipPlanIdEql(id, plan.state_ownership_plan_id.?)) return error.InvalidStagedErrorContext;
    }
}

fn validateTouchedStageRefs(request_id: u64, touched_stages: []const TouchedStageCleanupRef) StagedErrorError!void {
    for (touched_stages, 0..) |touched, index| {
        if (touched.request_id == 0) return error.MissingRequestId;
        if (request_id != 0 and touched.request_id != request_id) return error.CleanupRequestIdMismatch;
        if (touched.slot_id == 0) return error.InvalidStagedErrorContext;
        var previous_index: usize = 0;
        while (previous_index < index) : (previous_index += 1) {
            const previous = touched_stages[previous_index];
            if (previous.stage_id == touched.stage_id and previous.slot_id == touched.slot_id) return error.InvalidCleanupStepOrder;
        }
    }
}

fn validateCleanupObligationsInput(
    request_id: u64,
    ownership_plan: ?*const StageStateOwnershipPlan,
    obligations: []const StateCleanupObligation,
) StagedErrorError!void {
    if (obligations.len == 0) return;
    const plan = ownership_plan orelse return error.MissingStateOwnershipPlan;
    for (obligations) |obligation| {
        if (obligation.request_id == 0) return error.MissingRequestId;
        if (request_id != 0 and obligation.request_id != request_id) return error.CleanupRequestIdMismatch;
        if (!obligation.idempotent) return error.NonIdempotentCleanupStepUnsupported;
        state_ownership.validateStateCleanupObligation(plan, obligation) catch return error.InvalidCleanupObligation;
    }
}

fn validateCleanupInputsMatch(
    request_id: u64,
    touched_stages: []const TouchedStageCleanupRef,
    obligations: []const StateCleanupObligation,
) StagedErrorError!void {
    for (obligations) |obligation| {
        if (obligation.request_id != request_id) return error.CleanupRequestIdMismatch;
        _ = findTouchedForObligation(touched_stages, obligation) orelse return error.InvalidCleanupObligation;
    }
}

fn cleanupStepFromObligation(
    index: usize,
    obligation: StateCleanupObligation,
    touched: TouchedStageCleanupRef,
    placement_plan: ?*const PlacementPlan,
) StagedErrorError!StagedCleanupStep {
    if (!obligation.idempotent) return error.NonIdempotentCleanupStepUnsupported;
    var host_id: ?HostId = null;
    if (placement_plan) |plan| {
        host_id = (host_capability.bindingForStage(plan, obligation.stage_id) catch return error.InvalidStagedErrorContext).host_id;
    }
    return .{
        .index = index,
        .source = .state_obligation,
        .host_id = host_id,
        .stage_id = obligation.stage_id,
        .request_id = obligation.request_id,
        .slot_id = obligation.slot_id,
        .state_epoch = touched.state_epoch,
        .descriptor_id = obligation.descriptor_id,
        .lifecycle = obligation.lifecycle,
        .cleanup_kind = obligation.kind,
        .order_key = obligation.order_key,
        .idempotent = obligation.idempotent,
    };
}

fn validatePlacementStateIdentity(placement_plan: *const PlacementPlan, ownership_plan: *const StageStateOwnershipPlan) StagedErrorError!void {
    if (!std.mem.eql(u8, &placement_plan.graph_digest, &ownership_plan.graph_digest)) return error.InvalidStagedErrorContext;
    if (placement_plan.graph_contract_version != ownership_plan.graph_contract_version) return error.InvalidStagedErrorContext;
    if (placement_plan.stage_plan_contract_version != ownership_plan.stage_plan_contract_version) return error.InvalidStagedErrorContext;
    if (!std.mem.eql(u8, &placement_plan.stage_plan_id.digest, &ownership_plan.stage_plan_id.digest)) return error.InvalidStagedErrorContext;
}

fn validateCleanupPlanIdentity(
    plan: *const StagedCleanupPlan,
    placement_plan: ?*const PlacementPlan,
    ownership_plan: ?*const StageStateOwnershipPlan,
) StagedErrorError!void {
    if (placement_plan) |placement| {
        if (plan.graph_digest == null or !std.mem.eql(u8, &plan.graph_digest.?, &placement.graph_digest)) return error.InvalidStagedErrorContext;
        if (plan.graph_contract_version == null or plan.graph_contract_version.? != placement.graph_contract_version) return error.InvalidStagedErrorContext;
        if (plan.stage_plan_contract_version == null or plan.stage_plan_contract_version.? != placement.stage_plan_contract_version) return error.InvalidStagedErrorContext;
        if (plan.stage_plan_id == null or !std.mem.eql(u8, &plan.stage_plan_id.?.digest, &placement.stage_plan_id.digest)) return error.InvalidStagedErrorContext;
        if (plan.placement_plan_id == null or !host_capability.placementPlanIdEql(plan.placement_plan_id.?, placement.plan_id)) return error.InvalidStagedErrorContext;
    }
    if (ownership_plan) |ownership| {
        if (plan.graph_digest == null or !std.mem.eql(u8, &plan.graph_digest.?, &ownership.graph_digest)) return error.InvalidStagedErrorContext;
        if (plan.graph_contract_version == null or plan.graph_contract_version.? != ownership.graph_contract_version) return error.InvalidStagedErrorContext;
        if (plan.stage_plan_contract_version == null or plan.stage_plan_contract_version.? != ownership.stage_plan_contract_version) return error.InvalidStagedErrorContext;
        if (plan.stage_plan_id == null or !std.mem.eql(u8, &plan.stage_plan_id.?.digest, &ownership.stage_plan_id.digest)) return error.InvalidStagedErrorContext;
        if (plan.state_ownership_plan_id == null or !state_ownership.stateOwnershipPlanIdEql(plan.state_ownership_plan_id.?, ownership.plan_id)) return error.InvalidStagedErrorContext;
    }
}

fn copyCleanupPlanIdentity(
    plan: *StagedCleanupPlan,
    placement_plan: ?*const PlacementPlan,
    ownership_plan: ?*const StageStateOwnershipPlan,
) void {
    if (placement_plan) |placement| {
        plan.graph_digest = placement.graph_digest;
        plan.graph_contract_version = placement.graph_contract_version;
        plan.stage_plan_contract_version = placement.stage_plan_contract_version;
        plan.stage_plan_id = placement.stage_plan_id;
        plan.placement_plan_id = placement.plan_id;
    }
    if (ownership_plan) |ownership| {
        if (plan.graph_digest == null) plan.graph_digest = ownership.graph_digest;
        if (plan.graph_contract_version == null) plan.graph_contract_version = ownership.graph_contract_version;
        if (plan.stage_plan_contract_version == null) plan.stage_plan_contract_version = ownership.stage_plan_contract_version;
        if (plan.stage_plan_id == null) plan.stage_plan_id = ownership.stage_plan_id;
        plan.state_ownership_plan_id = ownership.plan_id;
    }
}

fn validateDuplicateCleanupSteps(steps: []const StagedCleanupStep) StagedErrorError!void {
    for (steps, 0..) |step, index| {
        var previous_index: usize = 0;
        while (previous_index < index) : (previous_index += 1) {
            const previous = steps[previous_index];
            if (previous.stage_id == step.stage_id and
                previous.descriptor_id == step.descriptor_id and
                previous.slot_id == step.slot_id and
                previous.cleanup_kind == step.cleanup_kind)
            {
                return error.DuplicateCleanupStep;
            }
        }
    }
}

fn validateCleanupStepOrder(steps: []const StagedCleanupStep, touched_stages: []const TouchedStageCleanupRef) StagedErrorError!void {
    var previous_touched_index: ?usize = null;
    var previous_order_key: u64 = 0;
    for (steps) |step| {
        const touched_index = findTouchedIndex(touched_stages, step.stage_id, step.request_id, step.slot_id) orelse return error.InvalidCleanupStepOrder;
        if (previous_touched_index) |previous| {
            if (touched_index > previous) return error.InvalidCleanupStepOrder;
            if (touched_index == previous and step.order_key < previous_order_key) return error.InvalidCleanupStepOrder;
        }
        previous_touched_index = touched_index;
        previous_order_key = step.order_key;
    }
}

fn validateStepAgainstInputs(step: StagedCleanupStep, options: StagedCleanupPlanValidationOptions) StagedErrorError!void {
    const obligation = findMatchingObligation(options.cleanup_obligations, step) orelse return error.InvalidCleanupObligation;
    if (!obligation.idempotent) return error.NonIdempotentCleanupStepUnsupported;
    _ = findTouchedForObligation(options.touched_stages, obligation) orelse return error.InvalidCleanupObligation;
    if (options.placement_plan) |placement| {
        const binding = host_capability.bindingForStage(placement, step.stage_id) catch return error.InvalidStagedErrorContext;
        if (step.host_id == null or !hostIdEql(step.host_id.?, binding.host_id)) return error.InvalidStagedErrorContext;
    }
}

fn validateCleanupFailureMatchesStepFields(failure: StagedFailure, step: StagedCleanupStep) StagedErrorError!void {
    if (failure.kind != .cleanup_failed or failure.phase != .cleanup or failure.scope != .cleanup) return error.InvalidStagedErrorContext;
    const context = failure.context;
    if (context.cleanup_step_index == null or context.cleanup_step_index.? != step.index) return error.MissingCleanupStepIndex;
    if (!optionalHostIdEql(context.host_id, step.host_id)) return error.InvalidStagedErrorContext;
    if (context.stage_id == null or context.stage_id.? != step.stage_id) return error.InvalidStagedErrorContext;
    if (context.request_id == null or context.request_id.? != step.request_id) return error.InvalidStagedErrorContext;
    if (context.slot_id == null or context.slot_id.? != step.slot_id) return error.InvalidStagedErrorContext;
    if (context.state_descriptor_id == null or context.state_descriptor_id.? != step.descriptor_id) return error.InvalidStagedErrorContext;
    if (context.state_lifecycle == null or context.state_lifecycle.? != step.lifecycle) return error.InvalidStagedErrorContext;
    if (context.state_cleanup_kind == null or context.state_cleanup_kind.? != step.cleanup_kind) return error.InvalidStagedErrorContext;
}

fn validateCleanupFailureMatchesAttempt(failure: StagedFailure, step: StagedCleanupStep, attempt_index: usize) StagedErrorError!void {
    try validateCleanupFailureMatchesStepFields(failure, step);
    if (failure.context.cleanup_attempt_index == null or failure.context.cleanup_attempt_index.? != attempt_index) {
        return error.InvalidCleanupAttemptOrder;
    }
}

fn nextObligationIndexForTouched(
    obligations: []const StateCleanupObligation,
    used: []const bool,
    touched: TouchedStageCleanupRef,
) ?usize {
    var selected_index: ?usize = null;
    for (obligations, 0..) |obligation, index| {
        if (used[index]) continue;
        if (!obligationMatchesTouched(obligation, touched)) continue;
        if (selected_index) |selected| {
            if (obligation.order_key < obligations[selected].order_key) selected_index = index;
        } else {
            selected_index = index;
        }
    }
    return selected_index;
}

fn findTouchedForObligation(touched_stages: []const TouchedStageCleanupRef, obligation: StateCleanupObligation) ?TouchedStageCleanupRef {
    for (touched_stages) |touched| {
        if (obligationMatchesTouched(obligation, touched)) return touched;
    }
    return null;
}

fn findTouchedIndex(touched_stages: []const TouchedStageCleanupRef, stage_id: usize, request_id: u64, slot_id: u64) ?usize {
    for (touched_stages, 0..) |touched, index| {
        if (touched.stage_id == stage_id and touched.request_id == request_id and touched.slot_id == slot_id) return index;
    }
    return null;
}

fn obligationMatchesTouched(obligation: StateCleanupObligation, touched: TouchedStageCleanupRef) bool {
    return obligation.stage_id == touched.stage_id and
        obligation.request_id == touched.request_id and
        obligation.slot_id == touched.slot_id;
}

fn findMatchingObligation(obligations: []const StateCleanupObligation, step: StagedCleanupStep) ?StateCleanupObligation {
    for (obligations) |obligation| {
        if (obligation.stage_id == step.stage_id and
            obligation.request_id == step.request_id and
            obligation.slot_id == step.slot_id and
            obligation.descriptor_id == step.descriptor_id and
            obligation.lifecycle == step.lifecycle and
            obligation.kind == step.cleanup_kind and
            obligation.order_key == step.order_key and
            obligation.idempotent == step.idempotent)
        {
            return obligation;
        }
    }
    return null;
}

fn statePlanHasStage(plan: *const StageStateOwnershipPlan, stage_id: usize) bool {
    for (plan.stage_entries) |entry| {
        if (entry.stage_id == stage_id) return true;
    }
    return false;
}

fn descriptorForStage(plan: *const StageStateOwnershipPlan, stage_id: usize, descriptor_id: u8) ?runtime_contract.StateDescriptor {
    for (plan.stage_entries) |entry| {
        if (entry.stage_id != stage_id) continue;
        for (entry.descriptors) |descriptor| {
            if (descriptor.id == descriptor_id) return descriptor;
        }
    }
    return null;
}

fn placementBoundaryForIndex(plan: *const PlacementPlan, boundary_index: usize) ?host_capability.PlacementBoundarySummary {
    for (plan.boundary_summaries) |boundary| {
        if (boundary.boundary_index == boundary_index) return boundary;
    }
    return null;
}

fn cleanupReportFlags(attempts: []const StagedCleanupAttempt) struct { cleanup_complete: bool, cleanup_failed: bool } {
    var failed = false;
    for (attempts) |attempt| {
        if (attempt.result == .failed) failed = true;
    }
    return .{ .cleanup_complete = !failed, .cleanup_failed = failed };
}

fn cleanupFailureLessThan(_: void, lhs: StagedFailure, rhs: StagedFailure) bool {
    const lhs_step = lhs.context.cleanup_step_index orelse std.math.maxInt(usize);
    const rhs_step = rhs.context.cleanup_step_index orelse std.math.maxInt(usize);
    if (lhs_step != rhs_step) return lhs_step < rhs_step;
    const lhs_attempt = lhs.context.cleanup_attempt_index orelse std.math.maxInt(usize);
    const rhs_attempt = rhs.context.cleanup_attempt_index orelse std.math.maxInt(usize);
    return lhs_attempt < rhs_attempt;
}

const CleanupPlanStepMutation = enum {
    stage_id,
    slot_id,
    descriptor_id,
    lifecycle,
    cleanup_kind,
    order_key,
    idempotent_flag,
    step_order,
};

fn expectCleanupPlanStepIdChange(base: *const StagedCleanupPlan, mutation: CleanupPlanStepMutation) !void {
    var steps = try std.testing.allocator.dupe(StagedCleanupStep, base.steps);
    defer std.testing.allocator.free(steps);

    switch (mutation) {
        .stage_id => steps[0].stage_id += 10,
        .slot_id => steps[0].slot_id += 10,
        .descriptor_id => steps[0].descriptor_id += 10,
        .lifecycle => steps[0].lifecycle = if (steps[0].lifecycle == .request_scoped) .slot_persistent else .request_scoped,
        .cleanup_kind => steps[0].cleanup_kind = if (steps[0].cleanup_kind == .evict_request_scoped) .unbind_slot_persistent else .evict_request_scoped,
        .order_key => steps[0].order_key += 10,
        .idempotent_flag => steps[0].idempotent = !steps[0].idempotent,
        .step_order => {
            try std.testing.expect(steps.len > 1);
            std.mem.swap(StagedCleanupStep, &steps[0], &steps[1]);
        },
    }

    var changed = base.*;
    changed.steps = steps;
    changed.plan_id = computeStagedCleanupPlanId(&changed);
    try std.testing.expect(!stagedCleanupPlanIdEql(base.plan_id, changed.plan_id));
}

fn hostIdEql(lhs: HostId, rhs: HostId) bool {
    return lhs.value == rhs.value;
}

fn optionalHostIdEql(lhs: ?HostId, rhs: ?HostId) bool {
    if (lhs == null and rhs == null) return true;
    if (lhs == null or rhs == null) return false;
    return hostIdEql(lhs.?, rhs.?);
}

fn computeStagedCleanupPlanId(plan: *const StagedCleanupPlan) StagedCleanupPlanId {
    var encoder = HashEncoder.init();
    encoder.writeString("talu.staged_cleanup_plan.v1");
    encoder.writeU32(plan.version);
    encoder.writeOptionalBytes(plan.graph_digest);
    encoder.writeOptionalU32(plan.graph_contract_version);
    encoder.writeOptionalU32(plan.stage_plan_contract_version);
    writeOptionalStagePlanId(&encoder, plan.stage_plan_id);
    writeOptionalPlacementPlanId(&encoder, plan.placement_plan_id);
    writeOptionalStateOwnershipPlanId(&encoder, plan.state_ownership_plan_id);
    encoder.writeU64(plan.request_id);
    encoder.writeUsize(plan.steps.len);
    for (plan.steps) |step| writeCleanupStep(&encoder, step);
    return .{ .digest = encoder.finish() };
}

fn computeStagedCleanupReportId(report: *const StagedCleanupReport) StagedCleanupReportId {
    var encoder = HashEncoder.init();
    encoder.writeString("talu.staged_cleanup_report.v1");
    encoder.writeU32(report.version);
    encoder.writeBytes(&report.cleanup_plan_id.digest);
    encoder.writeUsize(report.attempts.len);
    for (report.attempts) |attempt| writeCleanupAttempt(&encoder, attempt);
    encoder.writeBool(report.cleanup_complete);
    encoder.writeBool(report.cleanup_failed);
    return .{ .digest = encoder.finish() };
}

fn writeCleanupStep(encoder: *HashEncoder, step: StagedCleanupStep) void {
    encoder.writeUsize(step.index);
    encoder.writeU8(@intFromEnum(step.source));
    writeOptionalHostId(encoder, step.host_id);
    encoder.writeUsize(step.stage_id);
    encoder.writeU64(step.request_id);
    encoder.writeU64(step.slot_id);
    encoder.writeOptionalU64(step.state_epoch);
    encoder.writeU8(step.descriptor_id);
    encoder.writeU8(@intFromEnum(step.lifecycle));
    encoder.writeU8(@intFromEnum(step.cleanup_kind));
    encoder.writeU64(step.order_key);
    encoder.writeBool(step.idempotent);
}

fn writeCleanupAttempt(encoder: *HashEncoder, attempt: StagedCleanupAttempt) void {
    encoder.writeUsize(attempt.cleanup_step_index);
    encoder.writeUsize(attempt.attempt_index);
    encoder.writeU8(@intFromEnum(attempt.result));
    encoder.writeBool(attempt.cleanup_failure != null);
    if (attempt.cleanup_failure) |failure| writeStagedFailureForReportId(encoder, failure);
}

fn writeStagedFailureForReportId(encoder: *HashEncoder, failure: StagedFailure) void {
    encoder.writeU32(failure.version);
    encoder.writeU8(@intFromEnum(failure.kind));
    encoder.writeU8(@intFromEnum(failure.phase));
    encoder.writeU8(@intFromEnum(failure.scope));
    encoder.writeU8(@intFromEnum(failure.source.domain));
    writeStagedErrorContext(encoder, failure.context);
}

fn writeStagedErrorContext(encoder: *HashEncoder, context: StagedErrorContext) void {
    encoder.writeOptionalBytes(context.graph_digest);
    encoder.writeOptionalU32(context.graph_contract_version);
    encoder.writeOptionalU32(context.stage_plan_contract_version);
    writeOptionalStagePlanId(encoder, context.stage_plan_id);
    writeOptionalPlacementPlanId(encoder, context.placement_plan_id);
    writeOptionalStateOwnershipPlanId(encoder, context.state_ownership_plan_id);
    encoder.writeOptionalU64(if (context.tensor_frame_id) |id| id.value else null);
    encoder.writeOptionalUsize(context.boundary_index);
    encoder.writeOptionalUsize(context.source_stage_id);
    encoder.writeOptionalUsize(context.target_stage_id);
    encoder.writeOptionalUsize(context.stage_id);
    writeOptionalHostId(encoder, context.host_id);
    writeOptionalHostId(encoder, context.source_host_id);
    writeOptionalHostId(encoder, context.target_host_id);
    encoder.writeOptionalU64(context.request_id);
    encoder.writeOptionalU64(context.slot_id);
    encoder.writeOptionalU64(context.state_epoch);
    encoder.writeOptionalU8(context.state_descriptor_id);
    encoder.writeOptionalEnum(context.state_lifecycle);
    encoder.writeOptionalEnum(context.state_cleanup_kind);
    encoder.writeOptionalUsize(context.cleanup_step_index);
    encoder.writeOptionalUsize(context.cleanup_attempt_index);
}

fn writeOptionalStagePlanId(encoder: *HashEncoder, value: ?stage_plan.StagePlanId) void {
    encoder.writeBool(value != null);
    if (value) |id| encoder.writeBytes(&id.digest);
}

fn writeOptionalPlacementPlanId(encoder: *HashEncoder, value: ?host_capability.PlacementPlanId) void {
    encoder.writeBool(value != null);
    if (value) |id| encoder.writeBytes(&id.digest);
}

fn writeOptionalStateOwnershipPlanId(encoder: *HashEncoder, value: ?state_ownership.StageStateOwnershipPlanId) void {
    encoder.writeBool(value != null);
    if (value) |id| encoder.writeBytes(&id.digest);
}

fn writeOptionalHostId(encoder: *HashEncoder, value: ?HostId) void {
    encoder.writeBool(value != null);
    if (value) |id| encoder.writeU64(id.value);
}

fn zeroCleanupPlanId() StagedCleanupPlanId {
    return .{ .digest = [_]u8{0} ** 32 };
}

fn zeroCleanupReportId() StagedCleanupReportId {
    return .{ .digest = [_]u8{0} ** 32 };
}

const HashEncoder = struct {
    hasher: Sha256,

    fn init() HashEncoder {
        return .{ .hasher = Sha256.init(.{}) };
    }

    fn finish(self: *HashEncoder) [32]u8 {
        var digest: [32]u8 = undefined;
        self.hasher.final(&digest);
        return digest;
    }

    fn writeBytes(self: *HashEncoder, bytes: []const u8) void {
        self.hasher.update(bytes);
    }

    fn writeString(self: *HashEncoder, value: []const u8) void {
        self.writeU64(value.len);
        self.writeBytes(value);
    }

    fn writeBool(self: *HashEncoder, value: bool) void {
        self.writeU8(@intFromBool(value));
    }

    fn writeU8(self: *HashEncoder, value: u8) void {
        self.hasher.update(&.{value});
    }

    fn writeU32(self: *HashEncoder, value: u32) void {
        var bytes: [4]u8 = undefined;
        std.mem.writeInt(u32, &bytes, value, .little);
        self.hasher.update(&bytes);
    }

    fn writeU64(self: *HashEncoder, value: u64) void {
        var bytes: [8]u8 = undefined;
        std.mem.writeInt(u64, &bytes, value, .little);
        self.hasher.update(&bytes);
    }

    fn writeUsize(self: *HashEncoder, value: usize) void {
        self.writeU64(@intCast(value));
    }

    fn writeOptionalBytes(self: *HashEncoder, value: ?[32]u8) void {
        self.writeBool(value != null);
        if (value) |payload| self.writeBytes(&payload);
    }

    fn writeOptionalU8(self: *HashEncoder, value: ?u8) void {
        self.writeBool(value != null);
        if (value) |payload| self.writeU8(payload);
    }

    fn writeOptionalU32(self: *HashEncoder, value: ?u32) void {
        self.writeBool(value != null);
        if (value) |payload| self.writeU32(payload);
    }

    fn writeOptionalU64(self: *HashEncoder, value: ?u64) void {
        self.writeBool(value != null);
        if (value) |payload| self.writeU64(payload);
    }

    fn writeOptionalUsize(self: *HashEncoder, value: ?usize) void {
        self.writeBool(value != null);
        if (value) |payload| self.writeUsize(payload);
    }

    fn writeOptionalEnum(self: *HashEncoder, value: anytype) void {
        self.writeBool(value != null);
        if (value) |payload| self.writeU8(@intFromEnum(payload));
    }
};

test "inference pipeline staged_error classifyStagePlanError classifyTensorFrameError classifyStateOwnershipError classifyPlacementError classifyRunnerError classifyTransportError classifyCleanupCallbackError maps typed source errors" {
    try expectClass(classifyStagePlanError(error.MissingGraphIdentity, .{}), .graph_identity_mismatch, .plan, .stage_plan);
    try expectClass(classifyStagePlanError(error.InvalidLayerCount, .{}), .unsupported_partition, .plan, .stage_plan);
    try expectClass(classifyStagePlanError(error.UnknownStageId, .{ .stage_id = 4 }), .unsupported_partition, .stage, .stage_plan);
    try expectClass(classifyStagePlanError(error.ResidencyMismatch, .{}), .internal_contract_violation, .plan, .stage_plan);
    try expectClass(classifyStagePlanError(error.OutOfMemory, .{}), .resource_exhausted, .plan, .stage_plan);

    try expectClass(classifyTensorFrameError(error.InvalidTensorShape), .invalid_tensor_frame, .boundary, .tensor_frame);
    try expectClass(classifyTensorFrameError(error.BoundaryTensorContractMismatch), .invalid_tensor_frame, .boundary, .tensor_frame);
    try expectClass(classifyTensorFrameError(error.PayloadBufferLengthMismatch), .invalid_activation_payload, .boundary, .tensor_frame);
    try expectClass(classifyTensorFrameError(error.ObserverFailure), .internal_contract_violation, .boundary, .tensor_frame);
    try expectClass(classifyTensorFrameError(error.OutOfMemory), .resource_exhausted, .boundary, .tensor_frame);

    try expectClass(classifyStateOwnershipError(error.StaleStateEpoch), .stale_request_state, .request, .state_ownership);
    try expectClass(classifyStateOwnershipError(error.StateRequestMismatch), .stale_request_state, .request, .state_ownership);
    try expectClass(classifyStateOwnershipError(error.StateSlotMismatch), .stale_request_state, .request, .state_ownership);
    try expectClass(classifyStateOwnershipError(error.StatePositionMismatch), .stale_request_state, .request, .state_ownership);
    try expectClass(classifyStateOwnershipError(error.StateOwnershipPlanFingerprintMismatch), .state_ownership_mismatch, .request, .state_ownership);
    try expectClass(classifyStateOwnershipError(error.InvalidCleanupObligation), .cleanup_failed, .cleanup, .state_ownership);
    try expectClass(classifyStateOwnershipError(error.OutOfMemory), .resource_exhausted, .request, .state_ownership);

    try expectClass(classifyPlacementError(error.InvalidPlacementContractVersion), .placement_identity_mismatch, .placement, .host_placement);
    try expectClass(classifyPlacementError(error.UnsupportedBackendKind), .incompatible_host_capability, .placement, .host_placement);
    try expectClass(classifyPlacementError(error.MissingHostResidency), .missing_resident_weights, .placement, .host_placement);
    try expectClass(classifyPlacementError(error.MismatchedResidentStateDescriptorSummary), .state_ownership_mismatch, .placement, .host_placement);
    try expectClass(classifyPlacementError(error.OutOfMemory), .resource_exhausted, .placement, .host_placement);

    try expectClassPhase(try classifyRunnerError(error.StageExecutionFailed, .{ .phase = .stage_execution_before_state_mutation }), .stage_execution_failed, .stage_execution_before_state_mutation, .stage, .runner);
    try expectClassPhase(try classifyRunnerError(error.StageExecutionFailed, .{ .phase = .stage_execution_after_state_mutation }), .stage_execution_failed, .stage_execution_after_state_mutation, .stage, .runner);
    try expectClassPhase(try classifyRunnerError(error.RequestCancelled, .{ .phase = .validation_before_mutation }), .request_cancelled, .validation_before_mutation, .request, .runner);
    try expectClassPhase(try classifyRunnerError(error.OutOfMemory, .{ .phase = .stage_execution_before_state_mutation }), .resource_exhausted, .stage_execution_before_state_mutation, .stage, .runner);
    try expectClassPhase(try classifyRunnerError(error.UnknownRunnerFailure, .{ .phase = .stage_execution_before_state_mutation }), .internal_contract_violation, .stage_execution_before_state_mutation, .stage, .runner);
    try std.testing.expectError(error.InvalidStagedFailurePhase, classifyRunnerError(error.StageExecutionFailed, .{ .phase = .validation_before_mutation }));
    try std.testing.expectError(error.InvalidStagedFailurePhase, classifyRunnerError(error.StageExecutionFailed, .{ .phase = .frame_handoff }));
    try std.testing.expectError(error.InvalidStagedFailurePhase, classifyRunnerError(error.StageExecutionFailed, .{ .phase = .cleanup }));
    try expectClass(classifyTransportError(error.TransferFailed), .transfer_failed, .transport, .transport);
    try expectClass(classifyTransportError(error.RemoteHostUnavailable), .remote_host_unavailable, .host, .transport);
    try expectClass(classifyTransportError(error.OutOfMemory), .resource_exhausted, .transport, .transport);
    try expectClass(classifyCleanupCallbackError(error.CleanupFailed), .cleanup_failed, .cleanup, .cleanup);
    try expectClass(classifyCleanupCallbackError(error.OutOfMemory), .resource_exhausted, .cleanup, .cleanup);
}

test "inference pipeline staged_error buildStagedFailure validateStagedFailure rejects missing context and supplied contract mismatches" {
    try std.testing.expectError(error.MissingRequestId, buildStagedFailure(.{
        .kind = .request_cancelled,
        .phase = .validation_before_mutation,
        .scope = .request,
        .source = .{ .domain = .runner },
    }, .{}));
    try std.testing.expectError(error.MissingHostId, buildStagedFailure(.{
        .kind = .remote_host_unavailable,
        .phase = .frame_handoff,
        .scope = .host,
        .source = .{ .domain = .transport },
    }, .{}));
    try std.testing.expectError(error.MissingBoundaryContext, buildStagedFailure(.{
        .kind = .transfer_failed,
        .phase = .frame_handoff,
        .scope = .transport,
        .source = .{ .domain = .transport },
    }, .{}));
    try std.testing.expectError(error.IncompatibleSourceErrorClassification, buildStagedFailure(.{
        .kind = .invalid_tensor_frame,
        .phase = .validation_before_mutation,
        .scope = .boundary,
        .context = .{ .boundary_index = 0 },
        .source = .{ .domain = .stage_plan },
    }, .{}));

    var plan_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer plan_arena.deinit();
    const stages = [_]stage_plan.StagePlanStage{
        .{ .id = 0, .layer_start = 0, .layer_end = 2, .residency = .{ .layer_start = 0, .layer_end = 2 } },
        .{ .id = 1, .layer_start = 2, .layer_end = 4, .residency = .{ .layer_start = 2, .layer_end = 4 } },
    };
    const boundaries = [_]stage_plan.StageBoundary{.{ .source_stage_id = 0, .target_stage_id = 1, .producer_layer_start = 0, .producer_layer_end = 2, .consumer_layer_start = 2, .consumer_layer_end = 4 }};
    const stage_plan_value = StagePlan{
        .arena = plan_arena,
        .stage_contract_version = stage_plan.stage_plan_contract_version,
        .graph_identity = .{
            .graph_contract_version = stage_plan.graph_identity_contract_version,
            .stage_contract_version = stage_plan.stage_plan_contract_version,
            .architecture_id = "test",
            .digest = testDigest(1),
        },
        .split_points = &.{2},
        .plan_id = .{ .digest = testDigest(2) },
        .partition_constraint_source = .explicit,
        .n_layers = 4,
        .stages = &stages,
        .boundaries = &boundaries,
        .dependencies = &.{},
        .diagnostics = &.{},
    };

    const valid = try buildStagedFailure(.{
        .kind = .transfer_failed,
        .phase = .frame_handoff,
        .scope = .boundary,
        .context = .{
            .graph_digest = stage_plan_value.graph_identity.digest,
            .graph_contract_version = stage_plan_value.graph_identity.graph_contract_version,
            .stage_plan_contract_version = stage_plan_value.stage_contract_version,
            .stage_plan_id = stage_plan_value.plan_id,
            .boundary_index = 0,
            .source_stage_id = 0,
            .target_stage_id = 1,
        },
        .source = .{ .domain = .transport },
    }, .{ .stage_plan = &stage_plan_value });
    try validateStagedFailure(&valid, .{ .stage_plan = &stage_plan_value });

    var placement_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer placement_arena.deinit();
    const placement_boundaries = [_]host_capability.PlacementBoundarySummary{.{
        .boundary_index = 0,
        .source_stage_id = 0,
        .target_stage_id = 1,
        .producer_layer_start = 0,
        .producer_layer_end = 2,
        .consumer_layer_start = 2,
        .consumer_layer_end = 4,
    }};
    const placement_bindings = [_]host_capability.StageHostBinding{
        .{ .stage_id = 0, .host_id = .{ .value = 1 } },
        .{ .stage_id = 1, .host_id = .{ .value = 2 } },
    };
    const placement_plan = PlacementPlan{
        .arena = placement_arena,
        .version = host_capability.placement_contract_version,
        .graph_digest = stage_plan_value.graph_identity.digest,
        .graph_contract_version = stage_plan_value.graph_identity.graph_contract_version,
        .stage_plan_contract_version = stage_plan_value.stage_contract_version,
        .stage_plan_id = stage_plan_value.plan_id,
        .plan_id = .{ .digest = testDigest(8) },
        .stage_summaries = &.{},
        .boundary_summaries = &placement_boundaries,
        .required_step_kinds = &.{},
        .state_placement_mode = .stateless_only,
        .stage_host_bindings = &placement_bindings,
        .host_summaries = &.{},
        .boundary_frame_profiles = &.{},
    };
    var valid_placement_boundary = valid;
    valid_placement_boundary.context.source_stage_id = null;
    valid_placement_boundary.context.target_stage_id = null;
    valid_placement_boundary.context.source_host_id = .{ .value = 1 };
    valid_placement_boundary.context.target_host_id = .{ .value = 2 };
    try validateStagedFailure(&valid_placement_boundary, .{ .placement_plan = &placement_plan });

    var placement_plan_with_state = placement_plan;
    placement_plan_with_state.state_ownership_plan_id = .{ .digest = testDigest(3) };
    var valid_placement_state_identity = valid_placement_boundary;
    valid_placement_state_identity.context.state_ownership_plan_id = placement_plan_with_state.state_ownership_plan_id;
    try validateStagedFailure(&valid_placement_state_identity, .{ .placement_plan = &placement_plan_with_state });

    var wrong_placement_state_identity = valid_placement_state_identity;
    wrong_placement_state_identity.context.state_ownership_plan_id = .{ .digest = testDigest(99) };
    try std.testing.expectError(error.InvalidStagedErrorContext, validateStagedFailure(&wrong_placement_state_identity, .{ .placement_plan = &placement_plan_with_state }));
    try std.testing.expectError(error.InvalidStagedErrorContext, validateStagedFailure(&valid_placement_state_identity, .{ .placement_plan = &placement_plan }));

    var wrong_boundary_index = valid_placement_boundary;
    wrong_boundary_index.context.boundary_index = 999;
    try std.testing.expectError(error.InvalidStagedErrorContext, validateStagedFailure(&wrong_boundary_index, .{ .placement_plan = &placement_plan }));

    var wrong_boundary_host = valid_placement_boundary;
    wrong_boundary_host.context.source_host_id = .{ .value = 9 };
    try std.testing.expectError(error.InvalidStagedErrorContext, validateStagedFailure(&wrong_boundary_host, .{ .placement_plan = &placement_plan }));

    var wrong_graph = valid;
    wrong_graph.context.graph_digest = testDigest(9);
    try std.testing.expectError(error.InvalidStagedErrorContext, validateStagedFailure(&wrong_graph, .{ .stage_plan = &stage_plan_value }));

    var missing_cleanup_step = StagedFailure{
        .kind = .cleanup_failed,
        .phase = .cleanup,
        .scope = .cleanup,
        .context = .{
            .stage_id = 0,
            .request_id = 7,
            .slot_id = 1,
            .state_descriptor_id = 1,
            .state_lifecycle = .request_scoped,
            .state_cleanup_kind = .evict_request_scoped,
        },
        .source = .{ .domain = .cleanup },
    };
    try std.testing.expectError(error.MissingCleanupStepIndex, validateStagedFailure(&missing_cleanup_step, .{}));
    missing_cleanup_step.context.cleanup_step_index = 0;
    try std.testing.expectError(error.InvalidStagedErrorContext, validateStagedFailure(&missing_cleanup_step, .{}));
}

test "inference pipeline staged_error validateStagedFailure rejects state ownership lifecycle mismatch" {
    var ownership_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer ownership_arena.deinit();
    const descriptors = [_]runtime_contract.StateDescriptor{
        .{ .id = 1, .size_bytes = 16, .align_bytes = 8, .zero_init = true, .lifecycle = .request_scoped, .runtime_kind = 1 },
    };
    const descriptor_sets = [_]state_ownership.StageStateDescriptorSet{.{ .stage_id = 0, .descriptors = &descriptors }};
    const ownership_plan = StageStateOwnershipPlan{
        .arena = ownership_arena,
        .version = state_ownership.state_ownership_contract_version,
        .graph_digest = testDigest(1),
        .graph_contract_version = stage_plan.graph_identity_contract_version,
        .stage_plan_contract_version = stage_plan.stage_plan_contract_version,
        .stage_plan_id = .{ .digest = testDigest(2) },
        .plan_id = .{ .digest = testDigest(3) },
        .strict_tensor_frame_epoch = false,
        .stage_entries = &descriptor_sets,
        .boundaries = &.{},
        .stateful_dependencies = &.{},
        .partition_facts = &.{},
    };
    const lifecycle_mismatch = StagedFailure{
        .kind = .cleanup_failed,
        .phase = .cleanup,
        .scope = .cleanup,
        .context = .{
            .stage_id = 0,
            .request_id = 7,
            .slot_id = 1,
            .state_descriptor_id = 1,
            .state_lifecycle = .slot_persistent,
            .state_cleanup_kind = .evict_request_scoped,
            .cleanup_step_index = 0,
        },
        .source = .{ .domain = .cleanup, .source_error_name = "CleanupFailed" },
    };
    try std.testing.expectError(error.InvalidStagedErrorContext, validateStagedFailure(&lifecycle_mismatch, .{ .state_ownership_plan = &ownership_plan }));
}

test "inference pipeline staged_error buildStagedCleanupPlan validateStagedCleanupPlan stagedCleanupPlanIdEql stagedCleanupRequired orders idempotent single request cleanup" {
    var ownership_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer ownership_arena.deinit();
    const descriptors0 = [_]runtime_contract.StateDescriptor{
        .{ .id = 1, .size_bytes = 16, .align_bytes = 8, .zero_init = true, .lifecycle = .request_scoped, .runtime_kind = 1 },
        .{ .id = 2, .size_bytes = 16, .align_bytes = 8, .zero_init = true, .lifecycle = .step_scoped, .runtime_kind = 1 },
    };
    const descriptors1 = [_]runtime_contract.StateDescriptor{
        .{ .id = 3, .size_bytes = 32, .align_bytes = 8, .zero_init = true, .lifecycle = .slot_persistent, .runtime_kind = 1 },
    };
    const descriptor_sets = [_]state_ownership.StageStateDescriptorSet{
        .{ .stage_id = 0, .descriptors = &descriptors0 },
        .{ .stage_id = 1, .descriptors = &descriptors1 },
    };
    const ownership_plan = StageStateOwnershipPlan{
        .arena = ownership_arena,
        .version = state_ownership.state_ownership_contract_version,
        .graph_digest = testDigest(1),
        .graph_contract_version = stage_plan.graph_identity_contract_version,
        .stage_plan_contract_version = stage_plan.stage_plan_contract_version,
        .stage_plan_id = .{ .digest = testDigest(2) },
        .plan_id = .{ .digest = testDigest(3) },
        .strict_tensor_frame_epoch = false,
        .stage_entries = &descriptor_sets,
        .boundaries = &.{},
        .stateful_dependencies = &.{},
        .partition_facts = &.{},
    };
    const touched = [_]TouchedStageCleanupRef{
        .{ .stage_id = 0, .request_id = 7, .slot_id = 1, .state_epoch = 101 },
        .{ .stage_id = 1, .request_id = 7, .slot_id = 2, .state_epoch = 102 },
    };
    const obligations = [_]StateCleanupObligation{
        .{ .stage_id = 0, .request_id = 7, .slot_id = 1, .descriptor_id = 1, .lifecycle = .request_scoped, .kind = .evict_request_scoped, .order_key = 20 },
        .{ .stage_id = 1, .request_id = 7, .slot_id = 2, .descriptor_id = 3, .lifecycle = .slot_persistent, .kind = .unbind_slot_persistent, .order_key = 5 },
        .{ .stage_id = 0, .request_id = 7, .slot_id = 1, .descriptor_id = 2, .lifecycle = .step_scoped, .kind = .evict_step_scoped, .order_key = 10 },
    };
    const primary = requestFailure(.stage_execution_after_state_mutation, .stage_execution_failed, 7);
    try std.testing.expect(stagedCleanupRequired(primary, &touched, &.{}));
    try std.testing.expect(stagedCleanupRequired(requestFailure(.validation_before_mutation, .request_cancelled, 7), &touched, &.{}));
    try std.testing.expect(stagedCleanupRequired(requestFailure(.validation_before_mutation, .stage_execution_failed, 7), &.{}, &obligations));
    try std.testing.expect(!stagedCleanupRequired(requestFailure(.validation_before_mutation, .stage_execution_failed, 7), &.{}, &.{}));

    var wrong_primary_request = primary;
    wrong_primary_request.context.request_id = 8;
    try std.testing.expectError(error.CleanupRequestIdMismatch, buildStagedCleanupPlan(std.testing.allocator, .{
        .primary_failure = wrong_primary_request,
        .request_id = 7,
        .state_ownership_plan = &ownership_plan,
        .touched_stages = &touched,
        .cleanup_obligations = &obligations,
    }));

    var wrong_primary_identity = primary;
    wrong_primary_identity.context.state_ownership_plan_id = .{ .digest = testDigest(99) };
    try std.testing.expectError(error.InvalidStagedErrorContext, buildStagedCleanupPlan(std.testing.allocator, .{
        .primary_failure = wrong_primary_identity,
        .request_id = 7,
        .state_ownership_plan = &ownership_plan,
        .touched_stages = &touched,
        .cleanup_obligations = &obligations,
    }));

    var plan = try buildStagedCleanupPlan(std.testing.allocator, .{
        .primary_failure = primary,
        .request_id = 7,
        .state_ownership_plan = &ownership_plan,
        .touched_stages = &touched,
        .cleanup_obligations = &obligations,
    });
    defer plan.deinit();
    try validateStagedCleanupPlan(&plan, .{ .state_ownership_plan = &ownership_plan, .touched_stages = &touched, .cleanup_obligations = &obligations });
    try std.testing.expectEqual(@as(usize, 3), plan.steps.len);
    try std.testing.expectEqual(@as(usize, 1), plan.steps[0].stage_id);
    try std.testing.expectEqual(@as(u8, 3), plan.steps[0].descriptor_id);
    try std.testing.expectEqual(@as(u8, 2), plan.steps[1].descriptor_id);
    try std.testing.expectEqual(@as(u8, 1), plan.steps[2].descriptor_id);
    try std.testing.expect(stagedCleanupPlanIdEql(plan.plan_id, plan.plan_id));

    var changed_request_id = plan;
    changed_request_id.request_id = 8;
    changed_request_id.plan_id = computeStagedCleanupPlanId(&changed_request_id);
    try std.testing.expect(!stagedCleanupPlanIdEql(plan.plan_id, changed_request_id.plan_id));
    try expectCleanupPlanStepIdChange(&plan, .stage_id);
    try expectCleanupPlanStepIdChange(&plan, .slot_id);
    try expectCleanupPlanStepIdChange(&plan, .descriptor_id);
    try expectCleanupPlanStepIdChange(&plan, .lifecycle);
    try expectCleanupPlanStepIdChange(&plan, .cleanup_kind);
    try expectCleanupPlanStepIdChange(&plan, .order_key);
    try expectCleanupPlanStepIdChange(&plan, .idempotent_flag);
    try expectCleanupPlanStepIdChange(&plan, .step_order);

    var placement_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer placement_arena.deinit();
    const bindings = [_]host_capability.StageHostBinding{
        .{ .stage_id = 0, .host_id = .{ .value = 1 } },
        .{ .stage_id = 1, .host_id = .{ .value = 2 } },
    };
    const placement_plan = PlacementPlan{
        .arena = placement_arena,
        .version = host_capability.placement_contract_version,
        .graph_digest = ownership_plan.graph_digest,
        .graph_contract_version = ownership_plan.graph_contract_version,
        .stage_plan_contract_version = ownership_plan.stage_plan_contract_version,
        .stage_plan_id = ownership_plan.stage_plan_id,
        .plan_id = .{ .digest = testDigest(4) },
        .stage_summaries = &.{},
        .boundary_summaries = &.{},
        .required_step_kinds = &.{},
        .state_placement_mode = .validate_ref,
        .state_ownership_contract_version = state_ownership.state_ownership_contract_version,
        .state_ownership_plan_id = ownership_plan.plan_id,
        .state_stage_summaries = &.{},
        .stage_host_bindings = &bindings,
        .host_summaries = &.{},
        .boundary_frame_profiles = &.{},
    };
    var plan_with_host = try buildStagedCleanupPlan(std.testing.allocator, .{
        .primary_failure = primary,
        .request_id = 7,
        .placement_plan = &placement_plan,
        .state_ownership_plan = &ownership_plan,
        .touched_stages = &touched,
        .cleanup_obligations = &obligations,
    });
    defer plan_with_host.deinit();
    try std.testing.expectEqual(@as(u64, 2), plan_with_host.steps[0].host_id.?.value);
    try std.testing.expect(!stagedCleanupPlanIdEql(plan.plan_id, plan_with_host.plan_id));

    var changed_placement_identity = plan_with_host;
    changed_placement_identity.placement_plan_id.?.digest[0] ^= 1;
    changed_placement_identity.plan_id = computeStagedCleanupPlanId(&changed_placement_identity);
    try std.testing.expect(!stagedCleanupPlanIdEql(plan_with_host.plan_id, changed_placement_identity.plan_id));

    var changed_state_identity = plan_with_host;
    changed_state_identity.state_ownership_plan_id.?.digest[0] ^= 1;
    changed_state_identity.plan_id = computeStagedCleanupPlanId(&changed_state_identity);
    try std.testing.expect(!stagedCleanupPlanIdEql(plan_with_host.plan_id, changed_state_identity.plan_id));

    var tampered_steps = try std.testing.allocator.dupe(StagedCleanupStep, plan_with_host.steps);
    defer std.testing.allocator.free(tampered_steps);
    var tampered_host = plan_with_host;
    tampered_steps[0].host_id = .{ .value = 9 };
    tampered_host.steps = tampered_steps;
    tampered_host.plan_id = computeStagedCleanupPlanId(&tampered_host);
    try std.testing.expectError(error.InvalidStagedErrorContext, validateStagedCleanupPlan(&tampered_host, .{ .placement_plan = &placement_plan, .state_ownership_plan = &ownership_plan, .touched_stages = &touched, .cleanup_obligations = &obligations }));

    var mismatched_host_bindings = bindings;
    mismatched_host_bindings[1].host_id = .{ .value = 9 };
    var mismatched_host_placement = placement_plan;
    mismatched_host_placement.stage_host_bindings = &mismatched_host_bindings;
    try std.testing.expectError(error.InvalidStagedErrorContext, validateStagedCleanupPlan(&plan_with_host, .{ .placement_plan = &mismatched_host_placement, .state_ownership_plan = &ownership_plan, .touched_stages = &touched, .cleanup_obligations = &obligations }));

    var mismatched_stage_bindings = bindings;
    mismatched_stage_bindings[1].stage_id = 9;
    var mismatched_stage_placement = placement_plan;
    mismatched_stage_placement.stage_host_bindings = &mismatched_stage_bindings;
    try std.testing.expectError(error.InvalidStagedErrorContext, validateStagedCleanupPlan(&plan_with_host, .{ .placement_plan = &mismatched_stage_placement, .state_ownership_plan = &ownership_plan, .touched_stages = &touched, .cleanup_obligations = &obligations }));

    var mismatched_ownership_id = ownership_plan;
    mismatched_ownership_id.plan_id.digest[0] ^= 1;
    try std.testing.expectError(error.InvalidStagedErrorContext, validateStagedCleanupPlan(&plan, .{ .state_ownership_plan = &mismatched_ownership_id, .touched_stages = &touched, .cleanup_obligations = &obligations }));

    var mismatched_descriptor_sets = descriptor_sets;
    mismatched_descriptor_sets[1].stage_id = 9;
    var mismatched_ownership_stage = ownership_plan;
    mismatched_ownership_stage.stage_entries = &mismatched_descriptor_sets;
    try std.testing.expectError(error.InvalidCleanupObligation, validateStagedCleanupPlan(&plan, .{ .state_ownership_plan = &mismatched_ownership_stage, .touched_stages = &touched, .cleanup_obligations = &obligations }));

    var mismatched_descriptors0 = descriptors0;
    mismatched_descriptors0[0].id = 9;
    const mismatched_descriptor_sets_by_id = [_]state_ownership.StageStateDescriptorSet{
        .{ .stage_id = 0, .descriptors = &mismatched_descriptors0 },
        .{ .stage_id = 1, .descriptors = &descriptors1 },
    };
    var mismatched_ownership_descriptor = ownership_plan;
    mismatched_ownership_descriptor.stage_entries = &mismatched_descriptor_sets_by_id;
    try std.testing.expectError(error.InvalidCleanupObligation, validateStagedCleanupPlan(&plan, .{ .state_ownership_plan = &mismatched_ownership_descriptor, .touched_stages = &touched, .cleanup_obligations = &obligations }));

    var mismatched_touched_request = touched;
    mismatched_touched_request[0].request_id = 8;
    try std.testing.expectError(error.CleanupRequestIdMismatch, validateStagedCleanupPlan(&plan, .{ .state_ownership_plan = &ownership_plan, .touched_stages = &mismatched_touched_request, .cleanup_obligations = &obligations }));

    var mismatched_touched_slot = touched;
    mismatched_touched_slot[0].slot_id = 9;
    try std.testing.expectError(error.InvalidCleanupObligation, validateStagedCleanupPlan(&plan, .{ .state_ownership_plan = &ownership_plan, .touched_stages = &mismatched_touched_slot, .cleanup_obligations = &obligations }));

    var mismatched_step_descriptor_steps = try std.testing.allocator.dupe(StagedCleanupStep, plan.steps);
    defer std.testing.allocator.free(mismatched_step_descriptor_steps);
    mismatched_step_descriptor_steps[0].descriptor_id = 9;
    var mismatched_step_descriptor = plan;
    mismatched_step_descriptor.steps = mismatched_step_descriptor_steps;
    mismatched_step_descriptor.plan_id = computeStagedCleanupPlanId(&mismatched_step_descriptor);
    try std.testing.expectError(error.InvalidCleanupObligation, validateStagedCleanupPlan(&mismatched_step_descriptor, .{ .state_ownership_plan = &ownership_plan, .touched_stages = &touched, .cleanup_obligations = &obligations }));

    var mismatched_step_order_steps = try std.testing.allocator.dupe(StagedCleanupStep, plan.steps);
    defer std.testing.allocator.free(mismatched_step_order_steps);
    mismatched_step_order_steps[2].order_key = 0;
    var mismatched_step_order = plan;
    mismatched_step_order.steps = mismatched_step_order_steps;
    mismatched_step_order.plan_id = computeStagedCleanupPlanId(&mismatched_step_order);
    try std.testing.expectError(error.InvalidCleanupStepOrder, validateStagedCleanupPlan(&mismatched_step_order, .{ .state_ownership_plan = &ownership_plan, .touched_stages = &touched, .cleanup_obligations = &obligations }));

    var non_idempotent_step_steps = try std.testing.allocator.dupe(StagedCleanupStep, plan.steps);
    defer std.testing.allocator.free(non_idempotent_step_steps);
    non_idempotent_step_steps[0].idempotent = false;
    var non_idempotent_step_plan = plan;
    non_idempotent_step_plan.steps = non_idempotent_step_steps;
    non_idempotent_step_plan.plan_id = computeStagedCleanupPlanId(&non_idempotent_step_plan);
    try std.testing.expectError(error.NonIdempotentCleanupStepUnsupported, validateStagedCleanupPlan(&non_idempotent_step_plan, .{ .state_ownership_plan = &ownership_plan, .touched_stages = &touched, .cleanup_obligations = &obligations }));

    var changed_epoch_touched = touched;
    changed_epoch_touched[1].state_epoch = 103;
    var changed_epoch_plan = try buildStagedCleanupPlan(std.testing.allocator, .{
        .primary_failure = primary,
        .request_id = 7,
        .state_ownership_plan = &ownership_plan,
        .touched_stages = &changed_epoch_touched,
        .cleanup_obligations = &obligations,
    });
    defer changed_epoch_plan.deinit();
    try std.testing.expect(!stagedCleanupPlanIdEql(plan.plan_id, changed_epoch_plan.plan_id));

    var tampered_id = plan;
    tampered_id.plan_id.digest[0] ^= 1;
    try std.testing.expectError(error.CleanupPlanFingerprintMismatch, validateStagedCleanupPlan(&tampered_id, .{ .state_ownership_plan = &ownership_plan, .touched_stages = &touched, .cleanup_obligations = &obligations }));

    var wrong_request_obligation = obligations;
    wrong_request_obligation[0].request_id = 8;
    try std.testing.expectError(error.CleanupRequestIdMismatch, buildStagedCleanupPlan(std.testing.allocator, .{
        .primary_failure = primary,
        .request_id = 7,
        .state_ownership_plan = &ownership_plan,
        .touched_stages = &touched,
        .cleanup_obligations = &wrong_request_obligation,
    }));

    var duplicate_obligations = obligations;
    duplicate_obligations[2] = duplicate_obligations[0];
    duplicate_obligations[2].order_key = 21;
    try std.testing.expectError(error.DuplicateCleanupStep, buildStagedCleanupPlan(std.testing.allocator, .{
        .primary_failure = primary,
        .request_id = 7,
        .state_ownership_plan = &ownership_plan,
        .touched_stages = &touched,
        .cleanup_obligations = &duplicate_obligations,
    }));

    var non_idempotent_obligations = obligations;
    non_idempotent_obligations[0].idempotent = false;
    try std.testing.expectError(error.NonIdempotentCleanupStepUnsupported, buildStagedCleanupPlan(std.testing.allocator, .{
        .primary_failure = primary,
        .request_id = 7,
        .state_ownership_plan = &ownership_plan,
        .touched_stages = &touched,
        .cleanup_obligations = &non_idempotent_obligations,
    }));
}

test "inference pipeline staged_error buildStagedCleanupReport validateStagedCleanupReport stagedCleanupReportIdEql rejects lifecycle mismatch" {
    var ownership_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer ownership_arena.deinit();
    const descriptors = [_]runtime_contract.StateDescriptor{
        .{ .id = 1, .size_bytes = 16, .align_bytes = 8, .zero_init = true, .lifecycle = .request_scoped, .runtime_kind = 1 },
        .{ .id = 2, .size_bytes = 16, .align_bytes = 8, .zero_init = true, .lifecycle = .step_scoped, .runtime_kind = 1 },
        .{ .id = 3, .size_bytes = 16, .align_bytes = 8, .zero_init = true, .lifecycle = .slot_persistent, .runtime_kind = 1 },
    };
    const descriptor_sets = [_]state_ownership.StageStateDescriptorSet{.{ .stage_id = 0, .descriptors = &descriptors }};
    const ownership_plan = StageStateOwnershipPlan{
        .arena = ownership_arena,
        .version = state_ownership.state_ownership_contract_version,
        .graph_digest = testDigest(1),
        .graph_contract_version = stage_plan.graph_identity_contract_version,
        .stage_plan_contract_version = stage_plan.stage_plan_contract_version,
        .stage_plan_id = .{ .digest = testDigest(2) },
        .plan_id = .{ .digest = testDigest(3) },
        .strict_tensor_frame_epoch = false,
        .stage_entries = &descriptor_sets,
        .boundaries = &.{},
        .stateful_dependencies = &.{},
        .partition_facts = &.{},
    };
    const touched = [_]TouchedStageCleanupRef{.{ .stage_id = 0, .request_id = 9, .slot_id = 1, .state_epoch = 201 }};
    const obligations = [_]StateCleanupObligation{
        .{ .stage_id = 0, .request_id = 9, .slot_id = 1, .descriptor_id = 1, .lifecycle = .request_scoped, .kind = .evict_request_scoped, .order_key = 0 },
        .{ .stage_id = 0, .request_id = 9, .slot_id = 1, .descriptor_id = 2, .lifecycle = .step_scoped, .kind = .evict_step_scoped, .order_key = 1 },
        .{ .stage_id = 0, .request_id = 9, .slot_id = 1, .descriptor_id = 3, .lifecycle = .slot_persistent, .kind = .unbind_slot_persistent, .order_key = 2 },
    };
    const primary_failure = requestFailure(.stage_execution_after_state_mutation, .stage_execution_failed, 9);
    var plan = try buildStagedCleanupPlan(std.testing.allocator, .{
        .primary_failure = primary_failure,
        .request_id = 9,
        .state_ownership_plan = &ownership_plan,
        .touched_stages = &touched,
        .cleanup_obligations = &obligations,
    });
    defer plan.deinit();

    var identity_failure = cleanupFailureForStep(plan.steps[0], 0);
    identity_failure.context.graph_digest = plan.graph_digest;
    identity_failure.context.graph_contract_version = plan.graph_contract_version;
    identity_failure.context.stage_plan_contract_version = plan.stage_plan_contract_version;
    identity_failure.context.stage_plan_id = plan.stage_plan_id;
    identity_failure.context.state_ownership_plan_id = plan.state_ownership_plan_id;
    const attempts = [_]StagedCleanupAttempt{
        .{ .cleanup_step_index = 0, .attempt_index = 0, .result = .failed, .cleanup_failure = identity_failure },
        .{ .cleanup_step_index = 1, .attempt_index = 0, .result = .success },
        .{ .cleanup_step_index = 2, .attempt_index = 0, .result = .already_clean },
    };
    var report = try buildStagedCleanupReport(std.testing.allocator, &attempts, .{ .cleanup_plan = &plan, .primary_failure = &primary_failure });
    defer report.deinit();
    try validateStagedCleanupReport(&report, .{ .cleanup_plan = &plan, .primary_failure = &primary_failure });
    try std.testing.expect(!report.cleanup_complete);
    try std.testing.expect(report.cleanup_failed);
    try std.testing.expect(stagedCleanupReportIdEql(report.report_id, report.report_id));

    var mismatched_primary_request = primary_failure;
    mismatched_primary_request.context.request_id = 10;
    try std.testing.expectError(error.CleanupRequestIdMismatch, validateStagedCleanupReport(&report, .{ .cleanup_plan = &plan, .primary_failure = &mismatched_primary_request }));

    var mismatched_primary_identity = primary_failure;
    mismatched_primary_identity.context.state_ownership_plan_id = .{ .digest = testDigest(99) };
    try std.testing.expectError(error.InvalidStagedErrorContext, validateStagedCleanupReport(&report, .{ .cleanup_plan = &plan, .primary_failure = &mismatched_primary_identity }));

    var tampered_report = report;
    tampered_report.report_id.digest[0] ^= 1;
    try std.testing.expectError(error.CleanupReportFingerprintMismatch, validateStagedCleanupReport(&tampered_report, .{ .cleanup_plan = &plan }));

    var mismatched_plan_report = report;
    mismatched_plan_report.cleanup_plan_id.digest[0] ^= 1;
    mismatched_plan_report.report_id = computeStagedCleanupReportId(&mismatched_plan_report);
    try std.testing.expectError(error.CleanupReportPlanMismatch, validateStagedCleanupReport(&mismatched_plan_report, .{ .cleanup_plan = &plan }));
    try std.testing.expect(!stagedCleanupReportIdEql(report.report_id, mismatched_plan_report.report_id));

    var changed_result_attempts = try std.testing.allocator.dupe(StagedCleanupAttempt, report.attempts);
    defer std.testing.allocator.free(changed_result_attempts);
    changed_result_attempts[1].result = .already_clean;
    var changed_attempt_result = report;
    changed_attempt_result.attempts = changed_result_attempts;
    changed_attempt_result.report_id = computeStagedCleanupReportId(&changed_attempt_result);
    try std.testing.expect(!stagedCleanupReportIdEql(report.report_id, changed_attempt_result.report_id));

    var changed_failure_context_attempts = try std.testing.allocator.dupe(StagedCleanupAttempt, report.attempts);
    defer std.testing.allocator.free(changed_failure_context_attempts);
    if (changed_failure_context_attempts[0].cleanup_failure) |*failure| {
        failure.context.slot_id = failure.context.slot_id.? + 1;
    }
    var changed_failure_context = report;
    changed_failure_context.attempts = changed_failure_context_attempts;
    changed_failure_context.report_id = computeStagedCleanupReportId(&changed_failure_context);
    try std.testing.expect(!stagedCleanupReportIdEql(report.report_id, changed_failure_context.report_id));

    var changed_cleanup_complete = report;
    changed_cleanup_complete.cleanup_complete = !changed_cleanup_complete.cleanup_complete;
    changed_cleanup_complete.report_id = computeStagedCleanupReportId(&changed_cleanup_complete);
    try std.testing.expect(!stagedCleanupReportIdEql(report.report_id, changed_cleanup_complete.report_id));

    var changed_cleanup_failed = report;
    changed_cleanup_failed.cleanup_failed = !changed_cleanup_failed.cleanup_failed;
    changed_cleanup_failed.report_id = computeStagedCleanupReportId(&changed_cleanup_failed);
    try std.testing.expect(!stagedCleanupReportIdEql(report.report_id, changed_cleanup_failed.report_id));

    const complete_attempts = [_]StagedCleanupAttempt{
        .{ .cleanup_step_index = 0, .attempt_index = 0, .result = .success },
        .{ .cleanup_step_index = 1, .attempt_index = 0, .result = .already_clean },
        .{ .cleanup_step_index = 2, .attempt_index = 0, .result = .success },
    };
    var complete_report = try buildStagedCleanupReport(std.testing.allocator, &complete_attempts, .{ .cleanup_plan = &plan, .primary_failure = &primary_failure });
    defer complete_report.deinit();
    try std.testing.expect(complete_report.cleanup_complete);
    try std.testing.expect(!complete_report.cleanup_failed);

    const cleanup_primary = cleanupFailureForStep(plan.steps[0], 0);
    try std.testing.expectError(error.CleanupFailureReplacedPrimaryFailure, buildStagedCleanupReport(std.testing.allocator, &attempts, .{ .cleanup_plan = &plan, .primary_failure = &cleanup_primary }));

    var mismatched_failure = cleanupFailureForStep(plan.steps[0], 0);
    mismatched_failure.context.state_lifecycle = .slot_persistent;
    const bad_attempts = [_]StagedCleanupAttempt{
        .{ .cleanup_step_index = 0, .attempt_index = 0, .result = .failed, .cleanup_failure = mismatched_failure },
        .{ .cleanup_step_index = 1, .attempt_index = 0, .result = .success },
        .{ .cleanup_step_index = 2, .attempt_index = 0, .result = .already_clean },
    };
    var bad_report = StagedCleanupReport{
        .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
        .version = staged_error_contract_version,
        .report_id = zeroCleanupReportId(),
        .cleanup_plan_id = plan.plan_id,
        .attempts = &bad_attempts,
        .cleanup_complete = false,
        .cleanup_failed = true,
    };
    bad_report.report_id = computeStagedCleanupReportId(&bad_report);
    defer bad_report.arena.deinit();
    try std.testing.expectError(error.InvalidStagedErrorContext, validateStagedCleanupReport(&bad_report, .{ .cleanup_plan = &plan }));

    var mismatched_identity_failure = identity_failure;
    mismatched_identity_failure.context.graph_digest = testDigest(99);
    const bad_identity_attempts = [_]StagedCleanupAttempt{
        .{ .cleanup_step_index = 0, .attempt_index = 0, .result = .failed, .cleanup_failure = mismatched_identity_failure },
        .{ .cleanup_step_index = 1, .attempt_index = 0, .result = .success },
        .{ .cleanup_step_index = 2, .attempt_index = 0, .result = .already_clean },
    };
    var bad_identity_report = StagedCleanupReport{
        .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
        .version = staged_error_contract_version,
        .report_id = zeroCleanupReportId(),
        .cleanup_plan_id = plan.plan_id,
        .attempts = &bad_identity_attempts,
        .cleanup_complete = false,
        .cleanup_failed = true,
    };
    bad_identity_report.report_id = computeStagedCleanupReportId(&bad_identity_report);
    defer bad_identity_report.arena.deinit();
    try std.testing.expectError(error.InvalidStagedErrorContext, validateStagedCleanupReport(&bad_identity_report, .{ .cleanup_plan = &plan }));
}

test "inference pipeline staged_error buildStagedErrorReport validateStagedErrorReport attachCleanupFailures preserves primary failure" {
    const primary = requestFailure(.stage_execution_after_state_mutation, .stage_execution_failed, 11);
    var report = try buildStagedErrorReport(std.testing.allocator, primary, null, &.{}, .{});
    defer report.deinit();
    try validateStagedErrorReport(&report, .{});

    const step = StagedCleanupStep{
        .index = 0,
        .stage_id = 0,
        .request_id = 11,
        .slot_id = 1,
        .descriptor_id = 1,
        .lifecycle = .request_scoped,
        .cleanup_kind = .evict_request_scoped,
        .order_key = 0,
    };
    const failure = cleanupFailureForStep(step, 0);
    var next = try attachCleanupFailures(std.testing.allocator, &report, &.{failure}, .{});
    defer next.deinit();
    try std.testing.expectEqual(primary.kind, next.primary_failure.kind);
    try std.testing.expectEqual(@as(usize, 1), next.cleanup_failures.len);
    try validateStagedErrorReport(&next, .{});

    const cleanup_primary = cleanupFailureForStep(step, 0);
    try std.testing.expectError(error.CleanupFailureReplacedPrimaryFailure, buildStagedErrorReport(std.testing.allocator, cleanup_primary, null, &.{failure}, .{}));
}

fn expectClass(
    actual: StagedFailureClass,
    kind: StagedFailureKind,
    scope: StagedFailureScope,
    domain: StagedSourceDomain,
) !void {
    try std.testing.expectEqual(kind, actual.kind);
    try std.testing.expectEqual(@as(?StagedFailurePhase, null), actual.phase);
    try std.testing.expectEqual(scope, actual.scope);
    try std.testing.expectEqual(domain, actual.source.domain);
    try std.testing.expect(actual.source.source_error_name != null);
}

fn expectClassPhase(
    actual: StagedFailureClass,
    kind: StagedFailureKind,
    phase: StagedFailurePhase,
    scope: StagedFailureScope,
    domain: StagedSourceDomain,
) !void {
    try std.testing.expectEqual(kind, actual.kind);
    try std.testing.expectEqual(@as(?StagedFailurePhase, phase), actual.phase);
    try std.testing.expectEqual(scope, actual.scope);
    try std.testing.expectEqual(domain, actual.source.domain);
    try std.testing.expect(actual.source.source_error_name != null);
}

fn requestFailure(phase: StagedFailurePhase, kind: StagedFailureKind, request_id: u64) StagedFailure {
    return .{
        .kind = kind,
        .phase = phase,
        .scope = .request,
        .context = .{ .request_id = request_id },
        .source = .{ .domain = .runner, .source_error_name = "StageExecutionFailed" },
    };
}

fn cleanupFailureForStep(step: StagedCleanupStep, attempt_index: usize) StagedFailure {
    return .{
        .kind = .cleanup_failed,
        .phase = .cleanup,
        .scope = .cleanup,
        .context = .{
            .host_id = step.host_id,
            .stage_id = step.stage_id,
            .request_id = step.request_id,
            .slot_id = step.slot_id,
            .state_descriptor_id = step.descriptor_id,
            .state_lifecycle = step.lifecycle,
            .state_cleanup_kind = step.cleanup_kind,
            .cleanup_step_index = step.index,
            .cleanup_attempt_index = attempt_index,
        },
        .source = .{ .domain = .cleanup, .source_error_name = "CleanupFailed" },
    };
}

fn testDigest(value: u8) [32]u8 {
    return [_]u8{value} ** 32;
}
