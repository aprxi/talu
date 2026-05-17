//! Pure runtime-state ownership contracts for logical inference stages.
//!
//! This module is metadata and validation only. It does not allocate backend
//! state, move KV/runtime payloads, or wire scheduler/backend mutation paths.

const std = @import("std");
const models = @import("models_pkg");
const runtime_contract = @import("runtime_contract_pkg");
const tensor_frame = @import("tensor_frame.zig");

const Allocator = std.mem.Allocator;
const Sha256 = std.crypto.hash.sha2.Sha256;
const StateDescriptor = runtime_contract.StateDescriptor;
const StateLifecycle = runtime_contract.StateLifecycle;
const StateLifecycleAction = runtime_contract.StateLifecycleAction;
const stage_plan = models.stage_plan;

pub const state_ownership_contract_version: u32 = 1;
pub const state_block_abi_max_alignment: u16 = 64;

pub const StateOwnershipError = stage_plan.StagePlanError || tensor_frame.TensorFrameValidationError || error{
    InvalidStateOwnershipContractVersion,
    StateOwnershipPlanIdentityMismatch,
    StateOwnershipPlanFingerprintMismatch,
    MissingStageDescriptorSet,
    DuplicateStageDescriptorSet,
    DuplicateStateDescriptorId,
    InvalidStateDescriptorAlignment,
    InvalidStateDescriptorSize,
    InvalidStateLifecycle,
    InvalidStateLifecycleAction,
    MissingStateDescriptor,
    UnknownStateDescriptorId,
    UnsupportedStatePartition,
    MissingRequiredStateDependency,
    InvalidStatePartitionFact,
    DuplicateStatePartitionFact,
    CrossStageStateMigrationUnsupported,
    CrossStageStateAliasUnsupported,
    StateRequestMismatch,
    StateSlotMismatch,
    StateStageMismatch,
    StatePositionMismatch,
    StaleStateEpoch,
    MissingStateEpoch,
    InvalidLeaseState,
    InvalidLeaseTransition,
    SlotAlreadyBound,
    StaleLeaseEpochOnSlotReuse,
    InvalidStateBindingReport,
    InvalidCleanupObligation,
};

pub const StateOwnershipContractVersion = u32;

pub const StageStateOwnershipPlanId = struct {
    digest: [32]u8,
};

pub const StageStateDescriptorSource = enum(u8) {
    compiled_plan,
    loaded_stage,
    backend_report,
    test_fixture,
};

pub const StageStateDescriptorSet = struct {
    stage_id: usize,
    descriptors: []const StateDescriptor,
    source: StageStateDescriptorSource = .test_fixture,
};

pub const StatePartitionOwnershipMode = enum(u8) {
    stage_local_independent,
    stage_level_dependency_only,
    requires_cross_stage_migration,
    requires_shared_mutable_alias,
};

pub const StageStatePartitionFact = struct {
    boundary_index: usize,
    source_stage_id: usize,
    target_stage_id: usize,
    reason: stage_plan.StageDependencyReason,
    descriptor_id: ?u8 = null,
    runtime_kind: ?u8 = null,
    ownership_mode: StatePartitionOwnershipMode = .stage_level_dependency_only,
};

pub const StageStateBoundaryRef = struct {
    boundary_index: usize,
    source_stage_id: usize,
    target_stage_id: usize,
    producer_layer_start: usize,
    producer_layer_end: usize,
    consumer_layer_start: usize,
    consumer_layer_end: usize,
};

pub const StageStateDependencyRef = struct {
    boundary_index: usize,
    source_stage_id: usize,
    target_stage_id: usize,
    reason: stage_plan.StageDependencyReason,
};

pub const StageStateOwnershipRequest = struct {
    plan: *const stage_plan.StagePlan,
    descriptor_sets: []const StageStateDescriptorSet,
    partition_facts: []const StageStatePartitionFact = &.{},
    strict_tensor_frame_epoch: bool = false,
};

pub const StageStateOwnershipPlan = struct {
    arena: std.heap.ArenaAllocator,
    version: StateOwnershipContractVersion,
    graph_digest: [32]u8,
    graph_contract_version: u32,
    stage_plan_contract_version: u32,
    stage_plan_id: stage_plan.StagePlanId,
    plan_id: StageStateOwnershipPlanId,
    strict_tensor_frame_epoch: bool,
    stage_entries: []const StageStateDescriptorSet,
    boundaries: []const StageStateBoundaryRef,
    stateful_dependencies: []const StageStateDependencyRef,
    partition_facts: []const StageStatePartitionFact,

    pub fn deinit(self: *StageStateOwnershipPlan) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

pub const StageStateLeaseState = enum(u8) {
    active,
    completed,
    cancelled,
    failed,
    evicted,
    invalid,
};

pub const StageStateLease = struct {
    version: StateOwnershipContractVersion = state_ownership_contract_version,
    graph_digest: [32]u8,
    graph_contract_version: u32,
    stage_plan_contract_version: u32,
    stage_plan_id: stage_plan.StagePlanId,
    ownership_plan_id: StageStateOwnershipPlanId,
    stage_id: usize,
    request_id: u64,
    slot_id: u64,
    expected_sequence_position: u64,
    expected_token_count: u64 = 0,
    epoch: u64,
    state: StageStateLeaseState = .active,
};

pub const StageStateLeaseValidationOptions = struct {
    expected_stage_id: ?usize = null,
    expected_request_id: ?u64 = null,
    expected_slot_id: ?u64 = null,
    expected_position: ?u64 = null,
    expected_epoch: ?u64 = null,
    expected_state: ?StageStateLeaseState = null,
};

pub const StageStateTransitionKind = enum(u8) {
    bind,
    prefill,
    decode,
    complete,
    cancel,
    fail,
    evict,
    deinit,
};

pub const StageStateTransition = struct {
    kind: StageStateTransitionKind,
    lease: ?*const StageStateLease = null,
    previous_lease: ?*const StageStateLease = null,
    plan: ?*const StageStateOwnershipPlan = null,
    request_id: u64,
    stage_id: usize,
    slot_id: u64,
    sequence_start: u64 = 0,
    token_count: u64 = 0,
    expected_epoch: u64,
    expected_token_count: u64 = 0,
};

pub const StageStateTransitionResult = struct {
    state: StageStateLeaseState,
    expected_sequence_position: u64,
    cleanup_required: bool = false,
};

pub const StageStateBindingReport = struct {
    stage_id: usize,
    descriptor_id: u8,
    payload_identity: u64,
    payload_size: u64,
    payload_alignment: u16,
};

pub const TensorFrameStateLeaseRole = enum(u8) {
    producer,
    consumer,
};

pub const TensorFrameStateValidationOptions = struct {
    strict_epoch: ?bool = null,
};

pub const StateCleanupKind = enum(u8) {
    unbind_slot_persistent,
    evict_request_scoped,
    evict_step_scoped,
};

pub const StateCleanupObligation = struct {
    stage_id: usize,
    request_id: u64,
    slot_id: u64,
    descriptor_id: u8,
    lifecycle: StateLifecycle,
    kind: StateCleanupKind,
    order_key: u64,
    idempotent: bool = true,
};

pub const StageStateCleanupTarget = struct {
    stage_id: usize,
    request_id: u64,
    slot_id: u64,
};

pub const StageStateDescriptorSetValidationOptions = struct {
    known_descriptor_ids: ?[]const u8 = null,
};

pub fn stateOwnershipPlanIdEql(lhs: StageStateOwnershipPlanId, rhs: StageStateOwnershipPlanId) bool {
    return std.mem.eql(u8, &lhs.digest, &rhs.digest);
}

pub fn buildStageStateOwnershipPlan(
    allocator: Allocator,
    request: StageStateOwnershipRequest,
) StateOwnershipError!StageStateOwnershipPlan {
    try stage_plan.validateStagePlan(request.plan, .{});

    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_allocator = arena.allocator();

    const stage_entries = try copyStageDescriptorSets(arena_allocator, request.plan, request.descriptor_sets);
    try validateStageDescriptorSetCoverage(request.plan, stage_entries);

    const boundaries = try copyStageBoundaries(arena_allocator, request.plan);
    const stateful_dependencies = try copyStatefulDependencies(arena_allocator, request.plan, boundaries);

    const partition_facts = try arena_allocator.dupe(StageStatePartitionFact, request.partition_facts);
    std.mem.sort(StageStatePartitionFact, partition_facts, {}, partitionFactLess);
    try validateStageStatePartitionFactsForSummaries(stage_entries, boundaries, stateful_dependencies, partition_facts);

    const plan_id = computeStageStateOwnershipPlanId(.{
        .version = state_ownership_contract_version,
        .graph_digest = request.plan.graph_identity.digest,
        .graph_contract_version = request.plan.graph_identity.graph_contract_version,
        .stage_plan_contract_version = request.plan.stage_contract_version,
        .stage_plan_id = request.plan.plan_id,
        .stage_entries = stage_entries,
        .boundaries = boundaries,
        .stateful_dependencies = stateful_dependencies,
        .partition_facts = partition_facts,
    });

    return .{
        .arena = arena,
        .version = state_ownership_contract_version,
        .graph_digest = request.plan.graph_identity.digest,
        .graph_contract_version = request.plan.graph_identity.graph_contract_version,
        .stage_plan_contract_version = request.plan.stage_contract_version,
        .stage_plan_id = request.plan.plan_id,
        .plan_id = plan_id,
        .strict_tensor_frame_epoch = request.strict_tensor_frame_epoch,
        .stage_entries = stage_entries,
        .boundaries = boundaries,
        .stateful_dependencies = stateful_dependencies,
        .partition_facts = partition_facts,
    };
}

pub fn validateStageStateOwnershipPlan(plan: *const StageStateOwnershipPlan) StateOwnershipError!void {
    if (plan.version != state_ownership_contract_version) return error.InvalidStateOwnershipContractVersion;
    if (plan.stage_entries.len == 0) return error.MissingStageDescriptorSet;

    var previous_stage_id: ?usize = null;
    for (plan.stage_entries) |entry| {
        if (previous_stage_id) |previous| {
            if (entry.stage_id <= previous) return error.DuplicateStageDescriptorSet;
        }
        try validateStageStateDescriptorSet(entry, .{});
        previous_stage_id = entry.stage_id;
    }

    try validateBoundarySummaries(plan.stage_entries, plan.boundaries);
    try validateStatefulDependencySummaries(plan.stage_entries, plan.boundaries, plan.stateful_dependencies);
    try validateStageStatePartitionFactsForSummaries(
        plan.stage_entries,
        plan.boundaries,
        plan.stateful_dependencies,
        plan.partition_facts,
    );

    var previous_fact: ?StageStatePartitionFact = null;
    for (plan.partition_facts) |fact| {
        if (previous_fact) |previous| {
            if (!partitionFactLess({}, previous, fact)) return error.DuplicateStatePartitionFact;
        }
        previous_fact = fact;
    }

    const expected = computeStageStateOwnershipPlanId(.{
        .version = plan.version,
        .graph_digest = plan.graph_digest,
        .graph_contract_version = plan.graph_contract_version,
        .stage_plan_contract_version = plan.stage_plan_contract_version,
        .stage_plan_id = plan.stage_plan_id,
        .stage_entries = plan.stage_entries,
        .boundaries = plan.boundaries,
        .stateful_dependencies = plan.stateful_dependencies,
        .partition_facts = plan.partition_facts,
    });
    if (!stateOwnershipPlanIdEql(plan.plan_id, expected)) return error.StateOwnershipPlanFingerprintMismatch;
}

pub fn descriptorSetForStage(
    plan: *const StageStateOwnershipPlan,
    stage_id: usize,
) StateOwnershipError!*const StageStateDescriptorSet {
    for (plan.stage_entries) |*entry| {
        if (entry.stage_id == stage_id) return entry;
    }
    return error.UnknownStageId;
}

pub fn validateStageStateDescriptorSet(
    set: StageStateDescriptorSet,
    options: StageStateDescriptorSetValidationOptions,
) StateOwnershipError!void {
    var seen = [_]bool{false} ** 256;
    for (set.descriptors) |descriptor| {
        if (seen[descriptor.id]) return error.DuplicateStateDescriptorId;
        seen[descriptor.id] = true;
        try validateStateDescriptor(descriptor);
        if (options.known_descriptor_ids) |known_ids| {
            if (!containsDescriptorId(known_ids, descriptor.id)) return error.UnknownStateDescriptorId;
        }
    }
}

pub fn validateStageStatePartitionFact(
    plan: *const stage_plan.StagePlan,
    stage_entries: []const StageStateDescriptorSet,
    fact: StageStatePartitionFact,
) StateOwnershipError!void {
    try validatePartitionFactMode(fact);
    const boundary = boundaryForFact(plan, fact) orelse return error.InvalidStatePartitionFact;
    if (boundary.source_stage_id != fact.source_stage_id or boundary.target_stage_id != fact.target_stage_id) {
        return error.InvalidStatePartitionFact;
    }
    if (!hasMatchingDependency(plan, fact.source_stage_id, fact.target_stage_id, fact.reason)) {
        return error.InvalidStatePartitionFact;
    }
    try validatePartitionFactDescriptors(stage_entries, fact);
}

fn validateStageStatePartitionFactAgainstSummaries(
    stage_entries: []const StageStateDescriptorSet,
    boundaries: []const StageStateBoundaryRef,
    dependencies: []const StageStateDependencyRef,
    fact: StageStatePartitionFact,
) StateOwnershipError!void {
    try validatePartitionFactMode(fact);
    const boundary = boundaryRefForIndex(boundaries, fact.boundary_index) orelse return error.BoundaryIndexOutOfRange;
    if (boundary.source_stage_id != fact.source_stage_id or boundary.target_stage_id != fact.target_stage_id) {
        return error.InvalidStatePartitionFact;
    }
    if (!hasMatchingDependencyRef(dependencies, fact.boundary_index, fact.source_stage_id, fact.target_stage_id, fact.reason)) {
        return error.InvalidStatePartitionFact;
    }
    try validatePartitionFactDescriptors(stage_entries, fact);
}

pub fn validateStageStateLease(
    plan: *const StageStateOwnershipPlan,
    lease: *const StageStateLease,
    options: StageStateLeaseValidationOptions,
) StateOwnershipError!void {
    if (lease.version != state_ownership_contract_version or plan.version != state_ownership_contract_version) {
        return error.InvalidStateOwnershipContractVersion;
    }
    if (!std.mem.eql(u8, &lease.graph_digest, &plan.graph_digest) or lease.graph_contract_version != plan.graph_contract_version) {
        return error.GraphIdentityMismatch;
    }
    if (lease.stage_plan_contract_version != plan.stage_plan_contract_version or
        !std.mem.eql(u8, &lease.stage_plan_id.digest, &plan.stage_plan_id.digest))
    {
        return error.StagePlanIdentityMismatch;
    }
    if (!stateOwnershipPlanIdEql(lease.ownership_plan_id, plan.plan_id)) {
        return error.StateOwnershipPlanIdentityMismatch;
    }
    if (lease.request_id == 0) return error.InvalidRequestId;
    if (lease.slot_id == 0) return error.InvalidSlotId;
    if (lease.epoch == 0) return error.StaleStateEpoch;
    _ = try descriptorSetForStage(plan, lease.stage_id);

    if (options.expected_stage_id) |expected| {
        if (lease.stage_id != expected) return error.StateStageMismatch;
    }
    if (options.expected_request_id) |expected| {
        if (lease.request_id != expected) return error.StateRequestMismatch;
    }
    if (options.expected_slot_id) |expected| {
        if (lease.slot_id != expected) return error.StateSlotMismatch;
    }
    if (options.expected_position) |expected| {
        if (lease.expected_sequence_position != expected) return error.StatePositionMismatch;
    }
    if (options.expected_epoch) |expected| {
        if (lease.epoch != expected) return error.StaleStateEpoch;
    }
    if (options.expected_state) |expected| {
        if (lease.state != expected) return error.InvalidLeaseState;
    }
}

pub fn validateTensorFrameBatchEntryForStateLease(
    entry: tensor_frame.TensorFrameBatchEntry,
    lease: *const StageStateLease,
    strict_epoch: bool,
) StateOwnershipError!void {
    if (lease.version != state_ownership_contract_version) return error.InvalidStateOwnershipContractVersion;
    if (entry.request_id == 0) return error.InvalidRequestId;
    if (entry.slot_id == 0) return error.InvalidSlotId;
    if (entry.request_id != lease.request_id) return error.StateRequestMismatch;
    if (entry.slot_id != lease.slot_id) return error.StateSlotMismatch;
    if (entry.sequence_start != lease.expected_sequence_position) return error.StatePositionMismatch;
    if (lease.expected_token_count == 0 or entry.token_count != lease.expected_token_count) return error.InvalidSequenceRange;

    if (entry.state_epoch) |epoch| {
        if (epoch == 0 or epoch != lease.epoch) return error.StaleStateEpoch;
    } else if (strict_epoch) {
        return error.MissingStateEpoch;
    }
}

pub fn validateTensorFrameForStateLeases(
    metadata: *const tensor_frame.TensorFrameMetadata,
    plan: *const StageStateOwnershipPlan,
    leases: []const StageStateLease,
    role: TensorFrameStateLeaseRole,
    options: TensorFrameStateValidationOptions,
) StateOwnershipError!void {
    try metadata.validate();
    if (plan.version != state_ownership_contract_version) return error.InvalidStateOwnershipContractVersion;
    if (!std.mem.eql(u8, &metadata.plan.graph_digest, &plan.graph_digest) or
        metadata.plan.graph_contract_version != plan.graph_contract_version)
    {
        return error.GraphIdentityMismatch;
    }
    if (metadata.plan.stage_plan_contract_version != plan.stage_plan_contract_version or
        !std.mem.eql(u8, &metadata.plan.stage_plan_id.digest, &plan.stage_plan_id.digest))
    {
        return error.StagePlanIdentityMismatch;
    }
    if (metadata.batch.entries.len != leases.len) return error.InvalidBatch;

    const boundary = boundaryRefForIndex(plan.boundaries, metadata.boundary.boundary_index) orelse {
        return error.BoundaryIndexOutOfRange;
    };
    try validateTensorFrameBoundaryMatches(boundary, metadata.boundary);
    try validateTensorFrameBoundaryMatches(boundary, metadata.selected_contract.boundary);

    const expected_stage_id = switch (role) {
        .producer => boundary.source_stage_id,
        .consumer => boundary.target_stage_id,
    };
    const strict_epoch = options.strict_epoch orelse plan.strict_tensor_frame_epoch;
    for (metadata.batch.entries, leases) |entry, lease| {
        try validateStageStateLease(plan, &lease, .{
            .expected_stage_id = expected_stage_id,
            .expected_request_id = entry.request_id,
            .expected_slot_id = entry.slot_id,
            .expected_position = entry.sequence_start,
            .expected_state = .active,
        });
        try validateTensorFrameBatchEntryForStateLease(entry, &lease, strict_epoch);
    }
}

pub fn validatePrefillTransition(
    plan: *const StageStateOwnershipPlan,
    lease: *const StageStateLease,
    sequence_start: u64,
    token_count: u64,
    expected_epoch: u64,
) StateOwnershipError!u64 {
    try validateStageStateLease(plan, lease, .{ .expected_epoch = expected_epoch });
    return validatePrefillProgress(lease, sequence_start, token_count, expected_epoch);
}

pub fn validateDecodeTransition(
    plan: *const StageStateOwnershipPlan,
    lease: *const StageStateLease,
    sequence_start: u64,
    token_count: u64,
    expected_epoch: u64,
) StateOwnershipError!u64 {
    if (token_count != 1) return error.InvalidSequenceRange;
    return validatePrefillTransition(plan, lease, sequence_start, token_count, expected_epoch);
}

pub fn validateLeaseTransition(transition: StageStateTransition) StateOwnershipError!StageStateTransitionResult {
    switch (transition.kind) {
        .bind => {
            const plan = transition.plan orelse return error.InvalidLeaseTransition;
            if (plan.version != state_ownership_contract_version) return error.InvalidStateOwnershipContractVersion;
            if (transition.request_id == 0) return error.InvalidRequestId;
            if (transition.slot_id == 0) return error.InvalidSlotId;
            if (transition.expected_epoch == 0) return error.StaleStateEpoch;
            _ = try descriptorSetForStage(plan, transition.stage_id);
            if (transition.previous_lease) |previous| {
                if (previous.stage_id == transition.stage_id and
                    previous.slot_id == transition.slot_id and
                    stateOwnershipPlanIdEql(previous.ownership_plan_id, plan.plan_id))
                {
                    if (previous.state == .active) return error.SlotAlreadyBound;
                    if (previous.epoch == transition.expected_epoch) return error.StaleLeaseEpochOnSlotReuse;
                }
            }
            return .{
                .state = .active,
                .expected_sequence_position = transition.sequence_start,
            };
        },
        .prefill => {
            const plan = transition.plan orelse return error.InvalidLeaseTransition;
            const lease = transition.lease orelse return error.InvalidLeaseTransition;
            try validateTransitionLeaseIdentity(plan, lease, transition);
            return .{
                .state = .active,
                .expected_sequence_position = try validatePrefillProgress(
                    lease,
                    transition.sequence_start,
                    transition.token_count,
                    transition.expected_epoch,
                ),
            };
        },
        .decode => {
            const plan = transition.plan orelse return error.InvalidLeaseTransition;
            const lease = transition.lease orelse return error.InvalidLeaseTransition;
            try validateTransitionLeaseIdentity(plan, lease, transition);
            return .{
                .state = .active,
                .expected_sequence_position = try validateDecodeProgress(
                    lease,
                    transition.sequence_start,
                    transition.token_count,
                    transition.expected_epoch,
                ),
            };
        },
        .complete => return terminalTransition(transition, .completed, false),
        .cancel => return terminalTransition(transition, .cancelled, true),
        .fail => return terminalTransition(transition, .failed, true),
        .evict => return evictTransition(transition),
        .deinit => return deinitTransition(transition),
    }
}

pub fn validateStateDescriptorLifecycleAction(
    descriptor: *const StateDescriptor,
    action: StateLifecycleAction,
) StateOwnershipError!void {
    try runtime_contract.validateStateLifecycleAction(descriptor.lifecycle, action);
}

pub fn shouldZeroStateDescriptorForLifecycleAction(
    descriptor: *const StateDescriptor,
    action: StateLifecycleAction,
) StateOwnershipError!bool {
    return runtime_contract.shouldZeroStateForLifecycleAction(descriptor, action);
}

pub fn buildStateCleanupObligations(
    plan: *const StageStateOwnershipPlan,
    touched_stages: []const StageStateCleanupTarget,
    out: []StateCleanupObligation,
) StateOwnershipError![]StateCleanupObligation {
    var count: usize = 0;
    var order_key: u64 = 0;
    var remaining = touched_stages.len;
    while (remaining > 0) {
        remaining -= 1;
        const target = touched_stages[remaining];
        if (target.request_id == 0) return error.InvalidRequestId;
        if (target.slot_id == 0) return error.InvalidSlotId;
        const set = try descriptorSetForStage(plan, target.stage_id);
        var descriptor_index = set.descriptors.len;
        while (descriptor_index > 0) {
            descriptor_index -= 1;
            if (count >= out.len) return error.InvalidCleanupObligation;
            const descriptor = set.descriptors[descriptor_index];
            out[count] = .{
                .stage_id = target.stage_id,
                .request_id = target.request_id,
                .slot_id = target.slot_id,
                .descriptor_id = descriptor.id,
                .lifecycle = descriptor.lifecycle,
                .kind = cleanupKindForLifecycle(descriptor.lifecycle),
                .order_key = order_key,
                .idempotent = true,
            };
            count += 1;
            order_key += 1;
        }
    }
    return out[0..count];
}

pub fn validateStateCleanupObligation(
    plan: *const StageStateOwnershipPlan,
    obligation: StateCleanupObligation,
) StateOwnershipError!void {
    if (obligation.request_id == 0) return error.InvalidRequestId;
    if (obligation.slot_id == 0) return error.InvalidSlotId;
    const descriptor = try descriptorForStage(plan, obligation.stage_id, obligation.descriptor_id);
    if (descriptor.lifecycle != obligation.lifecycle) return error.InvalidCleanupObligation;
    if (cleanupKindForLifecycle(descriptor.lifecycle) != obligation.kind) return error.InvalidCleanupObligation;
    if (!obligation.idempotent) return error.InvalidCleanupObligation;
}

pub fn validateStateEpoch(lease: *const StageStateLease, epoch: u64) StateOwnershipError!void {
    if (lease.version != state_ownership_contract_version) return error.InvalidStateOwnershipContractVersion;
    if (epoch == 0 or epoch != lease.epoch) return error.StaleStateEpoch;
}

pub fn validateStageStateBindingReports(
    plan: *const StageStateOwnershipPlan,
    reports: []const StageStateBindingReport,
) StateOwnershipError!void {
    for (reports, 0..) |report, index| {
        if (report.payload_identity == 0) return error.InvalidStateBindingReport;
        const descriptor = try descriptorForStage(plan, report.stage_id, report.descriptor_id);
        if (report.payload_alignment == 0 or !std.math.isPowerOfTwo(report.payload_alignment)) {
            return error.InvalidStateBindingReport;
        }
        if (report.payload_alignment > state_block_abi_max_alignment or report.payload_alignment < descriptor.align_bytes) {
            return error.InvalidStateBindingReport;
        }
        _ = std.math.cast(usize, report.payload_size) orelse return error.InvalidStateBindingReport;
        if (report.payload_size < descriptor.size_bytes) return error.InvalidStateBindingReport;

        for (reports[0..index]) |previous| {
            if (previous.payload_identity == report.payload_identity and previous.stage_id != report.stage_id) {
                return error.CrossStageStateAliasUnsupported;
            }
        }
    }
}

const PlanIdInputs = struct {
    version: StateOwnershipContractVersion,
    graph_digest: [32]u8,
    graph_contract_version: u32,
    stage_plan_contract_version: u32,
    stage_plan_id: stage_plan.StagePlanId,
    stage_entries: []const StageStateDescriptorSet,
    boundaries: []const StageStateBoundaryRef,
    stateful_dependencies: []const StageStateDependencyRef,
    partition_facts: []const StageStatePartitionFact,
};

fn computeStageStateOwnershipPlanId(inputs: PlanIdInputs) StageStateOwnershipPlanId {
    var encoder = HashEncoder.init();
    encoder.writeString("talu.stage_state_ownership");
    encoder.writeU32(inputs.version);
    encoder.writeBytes(&inputs.graph_digest);
    encoder.writeU32(inputs.graph_contract_version);
    encoder.writeU32(inputs.stage_plan_contract_version);
    encoder.writeBytes(&inputs.stage_plan_id.digest);
    encoder.writeUsize(inputs.stage_entries.len);
    for (inputs.stage_entries) |entry| {
        encoder.writeUsize(entry.stage_id);
        encoder.writeUsize(entry.descriptors.len);
        for (entry.descriptors) |descriptor| writeStateDescriptor(&encoder, descriptor);
    }
    encoder.writeUsize(inputs.boundaries.len);
    for (inputs.boundaries) |boundary| writeBoundaryRef(&encoder, boundary);
    encoder.writeUsize(inputs.stateful_dependencies.len);
    for (inputs.stateful_dependencies) |dependency| writeDependencyRef(&encoder, dependency);
    encoder.writeUsize(inputs.partition_facts.len);
    for (inputs.partition_facts) |fact| writePartitionFact(&encoder, fact);
    return .{ .digest = encoder.finish() };
}

fn recomputeStageStateOwnershipPlanId(plan: *const StageStateOwnershipPlan) StageStateOwnershipPlanId {
    return computeStageStateOwnershipPlanId(.{
        .version = plan.version,
        .graph_digest = plan.graph_digest,
        .graph_contract_version = plan.graph_contract_version,
        .stage_plan_contract_version = plan.stage_plan_contract_version,
        .stage_plan_id = plan.stage_plan_id,
        .stage_entries = plan.stage_entries,
        .boundaries = plan.boundaries,
        .stateful_dependencies = plan.stateful_dependencies,
        .partition_facts = plan.partition_facts,
    });
}

fn copyStageDescriptorSets(
    allocator: Allocator,
    plan: *const stage_plan.StagePlan,
    input_sets: []const StageStateDescriptorSet,
) StateOwnershipError![]StageStateDescriptorSet {
    const output_len = if (plan.stages.len == 1 and input_sets.len == 0) 1 else input_sets.len;
    const output_sets = try allocator.alloc(StageStateDescriptorSet, output_len);

    if (input_sets.len == 0 and plan.stages.len == 1) {
        output_sets[0] = .{
            .stage_id = plan.stages[0].id,
            .descriptors = &.{},
            .source = .test_fixture,
        };
        return output_sets;
    }

    for (input_sets, 0..) |input, index| {
        _ = try stageById(plan, input.stage_id);
        const descriptors = try allocator.dupe(StateDescriptor, input.descriptors);
        std.mem.sort(StateDescriptor, descriptors, {}, descriptorLess);
        output_sets[index] = .{
            .stage_id = input.stage_id,
            .descriptors = descriptors,
            .source = input.source,
        };
    }
    std.mem.sort(StageStateDescriptorSet, output_sets, {}, descriptorSetLess);
    for (output_sets, 0..) |set, index| {
        if (index > 0 and output_sets[index - 1].stage_id == set.stage_id) return error.DuplicateStageDescriptorSet;
        try validateStageStateDescriptorSet(set, .{});
    }
    return output_sets;
}

fn copyStageBoundaries(
    allocator: Allocator,
    plan: *const stage_plan.StagePlan,
) StateOwnershipError![]StageStateBoundaryRef {
    const boundaries = try allocator.alloc(StageStateBoundaryRef, plan.boundaries.len);
    for (plan.boundaries, 0..) |boundary, index| {
        boundaries[index] = .{
            .boundary_index = index,
            .source_stage_id = boundary.source_stage_id,
            .target_stage_id = boundary.target_stage_id,
            .producer_layer_start = boundary.producer_layer_start,
            .producer_layer_end = boundary.producer_layer_end,
            .consumer_layer_start = boundary.consumer_layer_start,
            .consumer_layer_end = boundary.consumer_layer_end,
        };
    }
    return boundaries;
}

fn copyStatefulDependencies(
    allocator: Allocator,
    plan: *const stage_plan.StagePlan,
    boundaries: []const StageStateBoundaryRef,
) StateOwnershipError![]StageStateDependencyRef {
    var count: usize = 0;
    for (plan.dependencies) |dependency| {
        if (dependency.reason == .stateful_decoder) count += 1;
    }

    const dependencies = try allocator.alloc(StageStateDependencyRef, count);
    var index: usize = 0;
    for (plan.dependencies) |dependency| {
        if (dependency.reason != .stateful_decoder) continue;
        const boundary_index = boundaryIndexForStagesInRefs(
            boundaries,
            dependency.source_stage_id,
            dependency.target_stage_id,
        ) orelse return error.InvalidStatePartitionFact;
        dependencies[index] = .{
            .boundary_index = boundary_index,
            .source_stage_id = dependency.source_stage_id,
            .target_stage_id = dependency.target_stage_id,
            .reason = dependency.reason,
        };
        index += 1;
    }
    std.mem.sort(StageStateDependencyRef, dependencies, {}, dependencyRefLess);
    return dependencies;
}

fn validateStageDescriptorSetCoverage(
    plan: *const stage_plan.StagePlan,
    stage_entries: []const StageStateDescriptorSet,
) StateOwnershipError!void {
    if (plan.stages.len > 1 and stage_entries.len != plan.stages.len) return error.MissingStageDescriptorSet;
    for (plan.stages) |stage_entry| {
        var found = false;
        for (stage_entries) |set| {
            if (set.stage_id == stage_entry.id) {
                found = true;
                break;
            }
        }
        if (!found) return error.MissingStageDescriptorSet;
    }
}

fn validateBoundarySummaries(
    stage_entries: []const StageStateDescriptorSet,
    boundaries: []const StageStateBoundaryRef,
) StateOwnershipError!void {
    for (boundaries, 0..) |boundary, index| {
        if (boundary.boundary_index != index) return error.BoundaryIndexOutOfRange;
        if (boundary.source_stage_id == boundary.target_stage_id) return error.InvalidStatePartitionFact;
        _ = descriptorSetFromEntries(stage_entries, boundary.source_stage_id) orelse return error.UnknownStageId;
        _ = descriptorSetFromEntries(stage_entries, boundary.target_stage_id) orelse return error.UnknownStageId;
        if (boundary.producer_layer_start >= boundary.producer_layer_end) return error.InvalidStatePartitionFact;
        if (boundary.consumer_layer_start >= boundary.consumer_layer_end) return error.InvalidStatePartitionFact;
    }
}

fn validateStatefulDependencySummaries(
    stage_entries: []const StageStateDescriptorSet,
    boundaries: []const StageStateBoundaryRef,
    dependencies: []const StageStateDependencyRef,
) StateOwnershipError!void {
    var previous_dependency: ?StageStateDependencyRef = null;
    for (dependencies) |dependency| {
        if (dependency.reason != .stateful_decoder) return error.InvalidStatePartitionFact;
        if (previous_dependency) |previous| {
            if (!dependencyRefLess({}, previous, dependency)) return error.DuplicateStatePartitionFact;
        }
        const boundary = boundaryRefForIndex(boundaries, dependency.boundary_index) orelse return error.BoundaryIndexOutOfRange;
        if (boundary.source_stage_id != dependency.source_stage_id or boundary.target_stage_id != dependency.target_stage_id) {
            return error.InvalidStatePartitionFact;
        }
        _ = descriptorSetFromEntries(stage_entries, dependency.source_stage_id) orelse return error.UnknownStageId;
        _ = descriptorSetFromEntries(stage_entries, dependency.target_stage_id) orelse return error.UnknownStageId;
        previous_dependency = dependency;
    }
}

fn validateStageStatePartitionFactsForSummaries(
    stage_entries: []const StageStateDescriptorSet,
    boundaries: []const StageStateBoundaryRef,
    dependencies: []const StageStateDependencyRef,
    facts: []const StageStatePartitionFact,
) StateOwnershipError!void {
    var previous_fact: ?StageStatePartitionFact = null;
    for (facts) |fact| {
        if (previous_fact) |previous| {
            if (!partitionFactLess({}, previous, fact)) return error.DuplicateStatePartitionFact;
        }
        try validateStageStatePartitionFactAgainstSummaries(stage_entries, boundaries, dependencies, fact);
        previous_fact = fact;
    }

    for (dependencies) |dependency| {
        var matches: usize = 0;
        for (facts) |fact| {
            if (fact.boundary_index == dependency.boundary_index and
                fact.source_stage_id == dependency.source_stage_id and
                fact.target_stage_id == dependency.target_stage_id and
                fact.reason == dependency.reason)
            {
                matches += 1;
            }
        }
        if (matches == 0) return error.MissingRequiredStateDependency;
        if (matches > 1) return error.DuplicateStatePartitionFact;
    }
}

fn validateStageStatePartitionFacts(
    plan: *const stage_plan.StagePlan,
    stage_entries: []const StageStateDescriptorSet,
    facts: []const StageStatePartitionFact,
) StateOwnershipError!void {
    var previous_fact: ?StageStatePartitionFact = null;
    for (facts) |fact| {
        if (previous_fact) |previous| {
            if (!partitionFactLess({}, previous, fact)) return error.DuplicateStatePartitionFact;
        }
        try validateStageStatePartitionFact(plan, stage_entries, fact);
        previous_fact = fact;
    }

    for (plan.dependencies) |dependency| {
        if (dependency.reason != .stateful_decoder) continue;
        const boundary_index = boundaryIndexForStages(plan, dependency.source_stage_id, dependency.target_stage_id) orelse {
            return error.InvalidStatePartitionFact;
        };
        var matches: usize = 0;
        for (facts) |fact| {
            if (fact.boundary_index == boundary_index and
                fact.source_stage_id == dependency.source_stage_id and
                fact.target_stage_id == dependency.target_stage_id and
                fact.reason == .stateful_decoder)
            {
                matches += 1;
            }
        }
        if (matches == 0) return error.MissingRequiredStateDependency;
        if (matches > 1) return error.DuplicateStatePartitionFact;
    }
}

fn validateStateDescriptor(descriptor: StateDescriptor) StateOwnershipError!void {
    if (descriptor.align_bytes == 0 or !std.math.isPowerOfTwo(descriptor.align_bytes)) {
        return error.InvalidStateDescriptorAlignment;
    }
    if (descriptor.size_bytes == 0) return error.InvalidStateDescriptorSize;
    _ = runtime_contract.stateLifecyclePolicy(descriptor.lifecycle);
}

fn validatePartitionFactMode(fact: StageStatePartitionFact) StateOwnershipError!void {
    if (fact.reason != .stateful_decoder) return error.InvalidStatePartitionFact;
    switch (fact.ownership_mode) {
        .requires_cross_stage_migration => return error.CrossStageStateMigrationUnsupported,
        .requires_shared_mutable_alias => return error.CrossStageStateAliasUnsupported,
        .stage_level_dependency_only => {
            if (fact.descriptor_id != null or fact.runtime_kind != null) return error.InvalidStatePartitionFact;
        },
        .stage_local_independent => {
            if (fact.descriptor_id == null and fact.runtime_kind == null) return error.InvalidStatePartitionFact;
        },
    }
}

fn validatePartitionFactDescriptors(
    stage_entries: []const StageStateDescriptorSet,
    fact: StageStatePartitionFact,
) StateOwnershipError!void {
    if (fact.descriptor_id == null and fact.runtime_kind == null) return;
    try validateFactDescriptorForStage(stage_entries, fact.source_stage_id, fact);
    try validateFactDescriptorForStage(stage_entries, fact.target_stage_id, fact);
}

fn validateFactDescriptorForStage(
    stage_entries: []const StageStateDescriptorSet,
    stage_id: usize,
    fact: StageStatePartitionFact,
) StateOwnershipError!void {
    const set = descriptorSetFromEntries(stage_entries, stage_id) orelse return error.UnknownStageId;
    for (set.descriptors) |descriptor| {
        if (fact.descriptor_id) |expected_id| {
            if (descriptor.id != expected_id) continue;
        }
        if (fact.runtime_kind) |expected_kind| {
            if (descriptor.runtime_kind != expected_kind) continue;
        }
        return;
    }
    return error.MissingStateDescriptor;
}

fn validatePrefillProgress(
    lease: *const StageStateLease,
    sequence_start: u64,
    token_count: u64,
    expected_epoch: u64,
) StateOwnershipError!u64 {
    if (lease.version != state_ownership_contract_version) return error.InvalidStateOwnershipContractVersion;
    if (lease.state != .active) return error.InvalidLeaseTransition;
    if (expected_epoch == 0 or expected_epoch != lease.epoch) return error.StaleStateEpoch;
    if (token_count == 0) return error.InvalidSequenceRange;
    if (sequence_start != lease.expected_sequence_position) return error.StatePositionMismatch;
    return std.math.add(u64, sequence_start, token_count) catch return error.InvalidSequenceRange;
}

fn validateDecodeProgress(
    lease: *const StageStateLease,
    sequence_start: u64,
    token_count: u64,
    expected_epoch: u64,
) StateOwnershipError!u64 {
    if (token_count != 1) return error.InvalidSequenceRange;
    return validatePrefillProgress(lease, sequence_start, token_count, expected_epoch);
}

fn validateTransitionLeaseIdentity(
    plan: *const StageStateOwnershipPlan,
    lease: *const StageStateLease,
    transition: StageStateTransition,
) StateOwnershipError!void {
    try validateStageStateLease(plan, lease, .{
        .expected_stage_id = transition.stage_id,
        .expected_request_id = transition.request_id,
        .expected_slot_id = transition.slot_id,
        .expected_epoch = transition.expected_epoch,
    });
}

fn terminalTransition(
    transition: StageStateTransition,
    terminal_state: StageStateLeaseState,
    cleanup_required: bool,
) StateOwnershipError!StageStateTransitionResult {
    const plan = transition.plan orelse return error.InvalidLeaseTransition;
    const lease = transition.lease orelse return error.InvalidLeaseTransition;
    try validateTransitionLeaseIdentity(plan, lease, transition);
    if (lease.state == terminal_state and (terminal_state == .cancelled or terminal_state == .failed)) {
        return .{
            .state = terminal_state,
            .expected_sequence_position = lease.expected_sequence_position,
            .cleanup_required = cleanup_required,
        };
    }
    if (lease.state != .active) return error.InvalidLeaseTransition;
    return .{
        .state = terminal_state,
        .expected_sequence_position = lease.expected_sequence_position,
        .cleanup_required = cleanup_required,
    };
}

fn evictTransition(transition: StageStateTransition) StateOwnershipError!StageStateTransitionResult {
    const plan = transition.plan orelse return error.InvalidLeaseTransition;
    const lease = transition.lease orelse return error.InvalidLeaseTransition;
    try validateTransitionLeaseIdentity(plan, lease, transition);
    switch (lease.state) {
        .completed, .cancelled, .failed, .evicted => return .{
            .state = .evicted,
            .expected_sequence_position = lease.expected_sequence_position,
            .cleanup_required = true,
        },
        .active, .invalid => return error.InvalidLeaseTransition,
    }
}

fn deinitTransition(transition: StageStateTransition) StateOwnershipError!StageStateTransitionResult {
    const plan = transition.plan orelse return error.InvalidLeaseTransition;
    const lease = transition.lease orelse return error.InvalidLeaseTransition;
    try validateTransitionLeaseIdentity(plan, lease, transition);
    switch (lease.state) {
        .completed, .cancelled, .failed, .evicted, .invalid => return .{
            .state = .invalid,
            .expected_sequence_position = lease.expected_sequence_position,
            .cleanup_required = true,
        },
        .active => return error.InvalidLeaseTransition,
    }
}

fn stageById(plan: *const stage_plan.StagePlan, stage_id: usize) StateOwnershipError!*const stage_plan.StagePlanStage {
    for (plan.stages) |*stage_entry| {
        if (stage_entry.id == stage_id) return stage_entry;
    }
    return error.UnknownStageId;
}

fn boundaryForFact(plan: *const stage_plan.StagePlan, fact: StageStatePartitionFact) ?stage_plan.StageBoundary {
    if (fact.boundary_index >= plan.boundaries.len) return null;
    return plan.boundaries[fact.boundary_index];
}

fn boundaryRefForIndex(boundaries: []const StageStateBoundaryRef, boundary_index: usize) ?StageStateBoundaryRef {
    for (boundaries) |boundary| {
        if (boundary.boundary_index == boundary_index) return boundary;
    }
    return null;
}

fn boundaryIndexForStages(plan: *const stage_plan.StagePlan, source_stage_id: usize, target_stage_id: usize) ?usize {
    for (plan.boundaries, 0..) |boundary, index| {
        if (boundary.source_stage_id == source_stage_id and boundary.target_stage_id == target_stage_id) return index;
    }
    return null;
}

fn boundaryIndexForStagesInRefs(
    boundaries: []const StageStateBoundaryRef,
    source_stage_id: usize,
    target_stage_id: usize,
) ?usize {
    for (boundaries) |boundary| {
        if (boundary.source_stage_id == source_stage_id and boundary.target_stage_id == target_stage_id) {
            return boundary.boundary_index;
        }
    }
    return null;
}

fn hasMatchingDependency(
    plan: *const stage_plan.StagePlan,
    source_stage_id: usize,
    target_stage_id: usize,
    reason: stage_plan.StageDependencyReason,
) bool {
    for (plan.dependencies) |dependency| {
        if (dependency.source_stage_id == source_stage_id and
            dependency.target_stage_id == target_stage_id and
            dependency.reason == reason)
        {
            return true;
        }
    }
    return false;
}

fn hasMatchingDependencyRef(
    dependencies: []const StageStateDependencyRef,
    boundary_index: usize,
    source_stage_id: usize,
    target_stage_id: usize,
    reason: stage_plan.StageDependencyReason,
) bool {
    for (dependencies) |dependency| {
        if (dependency.boundary_index == boundary_index and
            dependency.source_stage_id == source_stage_id and
            dependency.target_stage_id == target_stage_id and
            dependency.reason == reason)
        {
            return true;
        }
    }
    return false;
}

fn validateTensorFrameBoundaryMatches(
    expected: StageStateBoundaryRef,
    actual: tensor_frame.TensorFrameBoundaryRef,
) StateOwnershipError!void {
    if (expected.boundary_index != actual.boundary_index or
        expected.source_stage_id != actual.source_stage_id or
        expected.target_stage_id != actual.target_stage_id or
        expected.producer_layer_start != actual.producer_layer_start or
        expected.producer_layer_end != actual.producer_layer_end or
        expected.consumer_layer_start != actual.consumer_layer_start or
        expected.consumer_layer_end != actual.consumer_layer_end)
    {
        return error.BoundaryTensorContractMismatch;
    }
}

fn descriptorSetFromEntries(
    stage_entries: []const StageStateDescriptorSet,
    stage_id: usize,
) ?*const StageStateDescriptorSet {
    for (stage_entries) |*entry| {
        if (entry.stage_id == stage_id) return entry;
    }
    return null;
}

fn descriptorForStage(
    plan: *const StageStateOwnershipPlan,
    stage_id: usize,
    descriptor_id: u8,
) StateOwnershipError!*const StateDescriptor {
    const set = try descriptorSetForStage(plan, stage_id);
    for (set.descriptors) |*descriptor| {
        if (descriptor.id == descriptor_id) return descriptor;
    }
    return error.MissingStateDescriptor;
}

fn cleanupKindForLifecycle(lifecycle: StateLifecycle) StateCleanupKind {
    return switch (lifecycle) {
        .slot_persistent => .unbind_slot_persistent,
        .request_scoped => .evict_request_scoped,
        .step_scoped => .evict_step_scoped,
    };
}

fn containsDescriptorId(ids: []const u8, descriptor_id: u8) bool {
    for (ids) |id| {
        if (id == descriptor_id) return true;
    }
    return false;
}

fn descriptorLess(_: void, lhs: StateDescriptor, rhs: StateDescriptor) bool {
    return lhs.id < rhs.id;
}

fn descriptorSetLess(_: void, lhs: StageStateDescriptorSet, rhs: StageStateDescriptorSet) bool {
    return lhs.stage_id < rhs.stage_id;
}

fn dependencyRefLess(_: void, lhs: StageStateDependencyRef, rhs: StageStateDependencyRef) bool {
    if (lhs.boundary_index != rhs.boundary_index) return lhs.boundary_index < rhs.boundary_index;
    if (lhs.source_stage_id != rhs.source_stage_id) return lhs.source_stage_id < rhs.source_stage_id;
    if (lhs.target_stage_id != rhs.target_stage_id) return lhs.target_stage_id < rhs.target_stage_id;
    if (@intFromEnum(lhs.reason) != @intFromEnum(rhs.reason)) return @intFromEnum(lhs.reason) < @intFromEnum(rhs.reason);
    return false;
}

fn partitionFactLess(_: void, lhs: StageStatePartitionFact, rhs: StageStatePartitionFact) bool {
    if (lhs.boundary_index != rhs.boundary_index) return lhs.boundary_index < rhs.boundary_index;
    if (lhs.source_stage_id != rhs.source_stage_id) return lhs.source_stage_id < rhs.source_stage_id;
    if (lhs.target_stage_id != rhs.target_stage_id) return lhs.target_stage_id < rhs.target_stage_id;
    if (@intFromEnum(lhs.reason) != @intFromEnum(rhs.reason)) return @intFromEnum(lhs.reason) < @intFromEnum(rhs.reason);
    if (lhs.descriptor_id != rhs.descriptor_id) return optionalU8Less(lhs.descriptor_id, rhs.descriptor_id);
    if (lhs.runtime_kind != rhs.runtime_kind) return optionalU8Less(lhs.runtime_kind, rhs.runtime_kind);
    if (@intFromEnum(lhs.ownership_mode) != @intFromEnum(rhs.ownership_mode)) {
        return @intFromEnum(lhs.ownership_mode) < @intFromEnum(rhs.ownership_mode);
    }
    return false;
}

fn optionalU8Less(lhs: ?u8, rhs: ?u8) bool {
    if (lhs == null) return rhs != null;
    if (rhs == null) return false;
    return lhs.? < rhs.?;
}

fn writeStateDescriptor(encoder: *HashEncoder, descriptor: StateDescriptor) void {
    encoder.writeU8(descriptor.id);
    encoder.writeU64(descriptor.size_bytes);
    encoder.writeU16(descriptor.align_bytes);
    encoder.writeBool(descriptor.zero_init);
    encoder.writeU8(@intFromEnum(descriptor.lifecycle));
    encoder.writeU8(descriptor.runtime_kind);
}

fn writeBoundaryRef(encoder: *HashEncoder, boundary: StageStateBoundaryRef) void {
    encoder.writeUsize(boundary.boundary_index);
    encoder.writeUsize(boundary.source_stage_id);
    encoder.writeUsize(boundary.target_stage_id);
    encoder.writeUsize(boundary.producer_layer_start);
    encoder.writeUsize(boundary.producer_layer_end);
    encoder.writeUsize(boundary.consumer_layer_start);
    encoder.writeUsize(boundary.consumer_layer_end);
}

fn writeDependencyRef(encoder: *HashEncoder, dependency: StageStateDependencyRef) void {
    encoder.writeUsize(dependency.boundary_index);
    encoder.writeUsize(dependency.source_stage_id);
    encoder.writeUsize(dependency.target_stage_id);
    encoder.writeU8(@intFromEnum(dependency.reason));
}

fn writePartitionFact(encoder: *HashEncoder, fact: StageStatePartitionFact) void {
    encoder.writeUsize(fact.boundary_index);
    encoder.writeUsize(fact.source_stage_id);
    encoder.writeUsize(fact.target_stage_id);
    encoder.writeU8(@intFromEnum(fact.reason));
    encoder.writeOptionalU8(fact.descriptor_id);
    encoder.writeOptionalU8(fact.runtime_kind);
    encoder.writeU8(@intFromEnum(fact.ownership_mode));
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
        self.writeBytes(&.{value});
    }

    fn writeU16(self: *HashEncoder, value: u16) void {
        var buf: [2]u8 = undefined;
        std.mem.writeInt(u16, &buf, value, .little);
        self.writeBytes(&buf);
    }

    fn writeU32(self: *HashEncoder, value: u32) void {
        var buf: [4]u8 = undefined;
        std.mem.writeInt(u32, &buf, value, .little);
        self.writeBytes(&buf);
    }

    fn writeU64(self: *HashEncoder, value: u64) void {
        var buf: [8]u8 = undefined;
        std.mem.writeInt(u64, &buf, value, .little);
        self.writeBytes(&buf);
    }

    fn writeUsize(self: *HashEncoder, value: usize) void {
        self.writeU64(@intCast(value));
    }

    fn writeOptionalU8(self: *HashEncoder, value: ?u8) void {
        self.writeBool(value != null);
        if (value) |payload| self.writeU8(payload);
    }
};

fn testConfig(layer_count: usize) models.config.ModelConfig {
    return .{
        .vocab_size = 64,
        .d_model = 8,
        .n_layers = @intCast(layer_count),
        .n_heads = 2,
        .n_kv_groups = 2,
        .d_ff = 16,
        .max_seq_len = 32,
        .head_dim = 4,
        .rope_theta = 10000,
        .norm_eps = 0.00001,
        .gaffine_group_size = 0,
        .tie_word_embeddings = false,
    };
}

fn testArch() models.op_types.Architecture {
    return .{
        .name = "state_ownership_test",
        .model_types = &.{"state_ownership_test"},
    };
}

fn testManifest(allocator: Allocator, layer_count: usize) !models.manifest.ModelManifest {
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_allocator = arena.allocator();

    const entry_count = layer_count + 3;
    const entries = try arena_allocator.alloc(models.manifest.TensorManifestEntry, entry_count);
    entries[0] = .{
        .name = "model.embed_tokens.weight",
        .dtype = .f32,
        .shape = &.{ 64, 8 },
        .checkpoint_bytes = 128,
        .role = .token_embeddings,
        .weight_id = "token_embeddings",
        .status = .architecture_weight,
    };
    for (0..layer_count) |layer_index| {
        entries[layer_index + 1] = .{
            .name = "model.layers.self_attn.q_proj.weight",
            .dtype = .f32,
            .shape = &.{ 8, 8 },
            .checkpoint_bytes = 64,
            .role = .decoder_layer,
            .layer_index = layer_index,
            .weight_id = "self_attn.q_proj.weight",
            .status = .architecture_weight,
        };
    }
    entries[layer_count + 1] = .{
        .name = "model.norm.weight",
        .dtype = .f32,
        .shape = &.{8},
        .checkpoint_bytes = 32,
        .role = .final_norm,
        .weight_id = "ln_final",
        .status = .architecture_weight,
    };
    entries[layer_count + 2] = .{
        .name = "lm_head.weight",
        .dtype = .f32,
        .shape = &.{ 64, 8 },
        .checkpoint_bytes = 128,
        .role = .lm_head,
        .weight_id = "lm_head",
        .status = .architecture_weight,
    };

    var role_bytes = [_]usize{0} ** models.manifest.role_count;
    var total_bytes: usize = 0;
    for (entries) |entry| {
        total_bytes += entry.checkpoint_bytes;
        role_bytes[@intFromEnum(entry.role)] += entry.checkpoint_bytes;
    }

    return .{
        .arena = arena,
        .architecture_id = "state_ownership_test",
        .layer_count = layer_count,
        .entries = entries,
        .total_checkpoint_bytes = total_bytes,
        .role_bytes = role_bytes,
    };
}

fn buildTestStagePlan(
    allocator: Allocator,
    splits: []const usize,
    dependencies: []const stage_plan.DependencyOverride,
) !stage_plan.StagePlan {
    var arch = testArch();
    var config = testConfig(4);
    var manifest = try testManifest(allocator, 4);
    defer manifest.deinit();

    return stage_plan.buildStagePlan(allocator, .{
        .n_layers = 4,
        .split_points = splits,
        .architecture = &arch,
        .model_config = &config,
        .manifest = &manifest,
        .partition_constraints = .{
            .decoder_cuts_allowed = true,
            .dependency_overrides = dependencies,
        },
    });
}

fn testDescriptor(id: u8, lifecycle: StateLifecycle, runtime_kind: u8) StateDescriptor {
    return .{
        .id = id,
        .size_bytes = 64,
        .align_bytes = 64,
        .zero_init = lifecycle != .slot_persistent,
        .lifecycle = lifecycle,
        .runtime_kind = runtime_kind,
    };
}

const test_stage0_descriptors = [_]StateDescriptor{
    testDescriptor(runtime_contract.kv_cache_state_id, .slot_persistent, runtime_contract.state_runtime_kind_kv_cache),
    testDescriptor(9, .request_scoped, runtime_contract.state_runtime_kind_none),
    testDescriptor(10, .step_scoped, runtime_contract.state_runtime_kind_none),
};

const test_stage1_descriptors = [_]StateDescriptor{
    testDescriptor(runtime_contract.kv_cache_state_id, .slot_persistent, runtime_contract.state_runtime_kind_kv_cache),
};

fn testDescriptorSets() [2]StageStateDescriptorSet {
    return .{
        .{ .stage_id = 0, .descriptors = &test_stage0_descriptors, .source = .test_fixture },
        .{ .stage_id = 1, .descriptors = &test_stage1_descriptors, .source = .test_fixture },
    };
}

fn testLease(plan: *const StageStateOwnershipPlan, stage_id: usize) StageStateLease {
    return .{
        .graph_digest = plan.graph_digest,
        .graph_contract_version = plan.graph_contract_version,
        .stage_plan_contract_version = plan.stage_plan_contract_version,
        .stage_plan_id = plan.stage_plan_id,
        .ownership_plan_id = plan.plan_id,
        .stage_id = stage_id,
        .request_id = 42,
        .slot_id = 7,
        .expected_sequence_position = 9,
        .expected_token_count = 1,
        .epoch = 99,
        .state = .active,
    };
}

test "inference pipeline state_ownership buildStageStateOwnershipPlan validateStageStateOwnershipPlan descriptorSetForStage stateOwnershipPlanIdEql" {
    const allocator = std.testing.allocator;
    var one_stage = try buildTestStagePlan(allocator, &.{}, &.{});
    defer one_stage.deinit();

    var empty_plan = try buildStageStateOwnershipPlan(allocator, .{ .plan = &one_stage, .descriptor_sets = &.{} });
    defer empty_plan.deinit();
    try validateStageStateOwnershipPlan(&empty_plan);
    try std.testing.expectEqual(@as(usize, 1), empty_plan.stage_entries.len);
    try std.testing.expectEqual(@as(usize, 0), (try descriptorSetForStage(&empty_plan, 0)).descriptors.len);

    var two_stage = try buildTestStagePlan(allocator, &.{2}, &.{});
    defer two_stage.deinit();
    const sets = testDescriptorSets();
    const missing_stage_sets = [_]StageStateDescriptorSet{sets[0]};
    try std.testing.expectError(error.MissingStageDescriptorSet, buildStageStateOwnershipPlan(allocator, .{
        .plan = &two_stage,
        .descriptor_sets = &missing_stage_sets,
    }));
    var unknown_stage_sets = sets;
    unknown_stage_sets[1].stage_id = 99;
    try std.testing.expectError(error.UnknownStageId, buildStageStateOwnershipPlan(allocator, .{
        .plan = &two_stage,
        .descriptor_sets = &unknown_stage_sets,
    }));

    var plan = try buildStageStateOwnershipPlan(allocator, .{
        .plan = &two_stage,
        .descriptor_sets = &sets,
    });
    defer plan.deinit();
    try validateStageStateOwnershipPlan(&plan);
    try std.testing.expect(stateOwnershipPlanIdEql(plan.plan_id, plan.plan_id));
    try std.testing.expectEqual(@as(usize, 1), plan.boundaries.len);
    try std.testing.expectEqual(@as(usize, 0), plan.boundaries[0].source_stage_id);
    try std.testing.expectEqual(@as(usize, 1), plan.boundaries[0].target_stage_id);
    try std.testing.expectEqual(@as(usize, 0), plan.stateful_dependencies.len);

    const stage0 = try descriptorSetForStage(&plan, 0);
    const stage1 = try descriptorSetForStage(&plan, 1);
    try std.testing.expectEqual(@as(u8, runtime_contract.kv_cache_state_id), stage0.descriptors[0].id);
    try std.testing.expectEqual(@as(u8, runtime_contract.kv_cache_state_id), stage1.descriptors[0].id);

    var changed_sets = sets;
    var changed_stage0 = [_]StateDescriptor{
        testDescriptor(runtime_contract.kv_cache_state_id, .request_scoped, runtime_contract.state_runtime_kind_kv_cache),
    };
    changed_sets[0] = .{ .stage_id = 0, .descriptors = &changed_stage0 };
    var changed_plan = try buildStageStateOwnershipPlan(allocator, .{
        .plan = &two_stage,
        .descriptor_sets = &changed_sets,
    });
    defer changed_plan.deinit();
    try std.testing.expect(!stateOwnershipPlanIdEql(plan.plan_id, changed_plan.plan_id));

    var strict_plan = try buildStageStateOwnershipPlan(allocator, .{
        .plan = &two_stage,
        .descriptor_sets = &sets,
        .strict_tensor_frame_epoch = true,
    });
    defer strict_plan.deinit();
    try std.testing.expect(stateOwnershipPlanIdEql(plan.plan_id, strict_plan.plan_id));

    var toggled_plan = plan;
    toggled_plan.strict_tensor_frame_epoch = true;
    try validateStageStateOwnershipPlan(&toggled_plan);
}

test "inference pipeline state_ownership validateStageStateDescriptorSet rejects invalid descriptors" {
    const duplicate = [_]StateDescriptor{
        testDescriptor(1, .slot_persistent, runtime_contract.state_runtime_kind_none),
        testDescriptor(1, .request_scoped, runtime_contract.state_runtime_kind_none),
    };
    try std.testing.expectError(
        error.DuplicateStateDescriptorId,
        validateStageStateDescriptorSet(.{ .stage_id = 0, .descriptors = &duplicate }, .{}),
    );

    const bad_align = [_]StateDescriptor{.{
        .id = 1,
        .size_bytes = 64,
        .align_bytes = 24,
        .zero_init = false,
        .lifecycle = .slot_persistent,
    }};
    try std.testing.expectError(
        error.InvalidStateDescriptorAlignment,
        validateStageStateDescriptorSet(.{ .stage_id = 0, .descriptors = &bad_align }, .{}),
    );

    const bad_size = [_]StateDescriptor{.{
        .id = 1,
        .size_bytes = 0,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
    }};
    try std.testing.expectError(
        error.InvalidStateDescriptorSize,
        validateStageStateDescriptorSet(.{ .stage_id = 0, .descriptors = &bad_size }, .{}),
    );

    const unknown = [_]StateDescriptor{testDescriptor(7, .slot_persistent, runtime_contract.state_runtime_kind_none)};
    const known = [_]u8{1};
    try std.testing.expectError(
        error.UnknownStateDescriptorId,
        validateStageStateDescriptorSet(.{ .stage_id = 0, .descriptors = &unknown }, .{ .known_descriptor_ids = &known }),
    );
}

test "inference pipeline state_ownership validateStageStatePartitionFact preserves stateful dependencies without descriptor inference" {
    const allocator = std.testing.allocator;
    const dependencies = [_]stage_plan.DependencyOverride{.{
        .source_stage_id = 0,
        .target_stage_id = 1,
        .reason = .stateful_decoder,
    }};
    var stage_plan_value = try buildTestStagePlan(allocator, &.{2}, &dependencies);
    defer stage_plan_value.deinit();
    const sets = testDescriptorSets();

    try std.testing.expectError(
        error.MissingRequiredStateDependency,
        buildStageStateOwnershipPlan(allocator, .{
            .plan = &stage_plan_value,
            .descriptor_sets = &sets,
        }),
    );

    const facts = [_]StageStatePartitionFact{.{
        .boundary_index = 0,
        .source_stage_id = 0,
        .target_stage_id = 1,
        .reason = .stateful_decoder,
        .ownership_mode = .stage_level_dependency_only,
    }};
    var plan = try buildStageStateOwnershipPlan(allocator, .{
        .plan = &stage_plan_value,
        .descriptor_sets = &sets,
        .partition_facts = &facts,
    });
    defer plan.deinit();
    try validateStageStateOwnershipPlan(&plan);
    try std.testing.expectEqual(@as(usize, 1), plan.partition_facts.len);

    const duplicate_facts = [_]StageStatePartitionFact{ facts[0], facts[0] };
    try std.testing.expectError(
        error.DuplicateStatePartitionFact,
        buildStageStateOwnershipPlan(allocator, .{
            .plan = &stage_plan_value,
            .descriptor_sets = &sets,
            .partition_facts = &duplicate_facts,
        }),
    );

    const wrong_boundary = [_]StageStatePartitionFact{.{
        .boundary_index = 0,
        .source_stage_id = 1,
        .target_stage_id = 0,
        .reason = .stateful_decoder,
        .ownership_mode = .stage_level_dependency_only,
    }};
    try std.testing.expectError(
        error.InvalidStatePartitionFact,
        buildStageStateOwnershipPlan(allocator, .{
            .plan = &stage_plan_value,
            .descriptor_sets = &sets,
            .partition_facts = &wrong_boundary,
        }),
    );
}

test "inference pipeline state_ownership validateStageStateOwnershipPlan rejects tampered copied partition summaries" {
    const allocator = std.testing.allocator;
    const dependencies = [_]stage_plan.DependencyOverride{.{
        .source_stage_id = 0,
        .target_stage_id = 1,
        .reason = .stateful_decoder,
    }};
    var stage_plan_value = try buildTestStagePlan(allocator, &.{2}, &dependencies);
    defer stage_plan_value.deinit();
    const sets = testDescriptorSets();
    const facts = [_]StageStatePartitionFact{.{
        .boundary_index = 0,
        .source_stage_id = 0,
        .target_stage_id = 1,
        .reason = .stateful_decoder,
        .ownership_mode = .stage_level_dependency_only,
    }};

    var plan = try buildStageStateOwnershipPlan(allocator, .{
        .plan = &stage_plan_value,
        .descriptor_sets = &sets,
        .partition_facts = &facts,
    });
    defer plan.deinit();
    try validateStageStateOwnershipPlan(&plan);
    try std.testing.expectEqual(@as(usize, 1), plan.boundaries.len);
    try std.testing.expectEqual(@as(usize, 1), plan.stateful_dependencies.len);

    var tampered_facts = [_]StageStatePartitionFact{plan.partition_facts[0]};
    tampered_facts[0].target_stage_id = 0;
    var tampered_fact_plan = plan;
    tampered_fact_plan.partition_facts = &tampered_facts;
    tampered_fact_plan.plan_id = recomputeStageStateOwnershipPlanId(&tampered_fact_plan);
    try std.testing.expectError(error.InvalidStatePartitionFact, validateStageStateOwnershipPlan(&tampered_fact_plan));

    var tampered_boundaries = [_]StageStateBoundaryRef{plan.boundaries[0]};
    tampered_boundaries[0].target_stage_id = 99;
    var tampered_boundary_plan = plan;
    tampered_boundary_plan.boundaries = &tampered_boundaries;
    tampered_boundary_plan.plan_id = recomputeStageStateOwnershipPlanId(&tampered_boundary_plan);
    try std.testing.expectError(error.UnknownStageId, validateStageStateOwnershipPlan(&tampered_boundary_plan));

    var tampered_dependencies = [_]StageStateDependencyRef{plan.stateful_dependencies[0]};
    tampered_dependencies[0].target_stage_id = 99;
    var tampered_dependency_plan = plan;
    tampered_dependency_plan.stateful_dependencies = &tampered_dependencies;
    tampered_dependency_plan.plan_id = recomputeStageStateOwnershipPlanId(&tampered_dependency_plan);
    try std.testing.expectError(error.InvalidStatePartitionFact, validateStageStateOwnershipPlan(&tampered_dependency_plan));
}

test "inference pipeline state_ownership validateStageStatePartitionFact validates descriptor facts and unsupported partitions" {
    const allocator = std.testing.allocator;
    const dependencies = [_]stage_plan.DependencyOverride{.{
        .source_stage_id = 0,
        .target_stage_id = 1,
        .reason = .stateful_decoder,
    }};
    var stage_plan_value = try buildTestStagePlan(allocator, &.{2}, &dependencies);
    defer stage_plan_value.deinit();
    const sets = testDescriptorSets();

    const descriptor_fact = [_]StageStatePartitionFact{.{
        .boundary_index = 0,
        .source_stage_id = 0,
        .target_stage_id = 1,
        .reason = .stateful_decoder,
        .descriptor_id = runtime_contract.kv_cache_state_id,
        .runtime_kind = runtime_contract.state_runtime_kind_kv_cache,
        .ownership_mode = .stage_local_independent,
    }};
    const stage_level_fact = [_]StageStatePartitionFact{.{
        .boundary_index = 0,
        .source_stage_id = 0,
        .target_stage_id = 1,
        .reason = .stateful_decoder,
        .ownership_mode = .stage_level_dependency_only,
    }};
    var stage_level_plan = try buildStageStateOwnershipPlan(allocator, .{
        .plan = &stage_plan_value,
        .descriptor_sets = &sets,
        .partition_facts = &stage_level_fact,
    });
    defer stage_level_plan.deinit();
    var descriptor_plan = try buildStageStateOwnershipPlan(allocator, .{
        .plan = &stage_plan_value,
        .descriptor_sets = &sets,
        .partition_facts = &descriptor_fact,
    });
    defer descriptor_plan.deinit();
    try validateStageStateOwnershipPlan(&descriptor_plan);
    try std.testing.expect(!stateOwnershipPlanIdEql(stage_level_plan.plan_id, descriptor_plan.plan_id));

    const missing_descriptor = [_]StageStatePartitionFact{.{
        .boundary_index = 0,
        .source_stage_id = 0,
        .target_stage_id = 1,
        .reason = .stateful_decoder,
        .descriptor_id = 123,
        .ownership_mode = .stage_local_independent,
    }};
    try std.testing.expectError(
        error.MissingStateDescriptor,
        buildStageStateOwnershipPlan(allocator, .{
            .plan = &stage_plan_value,
            .descriptor_sets = &sets,
            .partition_facts = &missing_descriptor,
        }),
    );

    const migration = [_]StageStatePartitionFact{.{
        .boundary_index = 0,
        .source_stage_id = 0,
        .target_stage_id = 1,
        .reason = .stateful_decoder,
        .ownership_mode = .requires_cross_stage_migration,
    }};
    try std.testing.expectError(
        error.CrossStageStateMigrationUnsupported,
        buildStageStateOwnershipPlan(allocator, .{
            .plan = &stage_plan_value,
            .descriptor_sets = &sets,
            .partition_facts = &migration,
        }),
    );

    const shared_alias = [_]StageStatePartitionFact{.{
        .boundary_index = 0,
        .source_stage_id = 0,
        .target_stage_id = 1,
        .reason = .stateful_decoder,
        .ownership_mode = .requires_shared_mutable_alias,
    }};
    try std.testing.expectError(
        error.CrossStageStateAliasUnsupported,
        buildStageStateOwnershipPlan(allocator, .{
            .plan = &stage_plan_value,
            .descriptor_sets = &sets,
            .partition_facts = &shared_alias,
        }),
    );
}

test "inference pipeline state_ownership validateStageStateLease validateStateEpoch rejects identity and epoch mismatches" {
    const allocator = std.testing.allocator;
    var stage_plan_value = try buildTestStagePlan(allocator, &.{2}, &.{});
    defer stage_plan_value.deinit();
    const sets = testDescriptorSets();
    var plan = try buildStageStateOwnershipPlan(allocator, .{ .plan = &stage_plan_value, .descriptor_sets = &sets });
    defer plan.deinit();

    var lease = testLease(&plan, 1);
    try validateStageStateLease(&plan, &lease, .{
        .expected_stage_id = 1,
        .expected_request_id = 42,
        .expected_slot_id = 7,
        .expected_position = 9,
        .expected_epoch = 99,
        .expected_state = .active,
    });
    try validateStateEpoch(&lease, 99);

    var wrong_plan = plan;
    wrong_plan.plan_id.digest[0] ^= 1;
    try std.testing.expectError(error.StateOwnershipPlanIdentityMismatch, validateStageStateLease(&wrong_plan, &lease, .{}));

    lease.epoch = 0;
    try std.testing.expectError(error.StaleStateEpoch, validateStageStateLease(&plan, &lease, .{}));
}

test "inference pipeline state_ownership validatePrefillTransition validateDecodeTransition validateLeaseTransition follows state machine" {
    const allocator = std.testing.allocator;
    var stage_plan_value = try buildTestStagePlan(allocator, &.{2}, &.{});
    defer stage_plan_value.deinit();
    const sets = testDescriptorSets();
    var plan = try buildStageStateOwnershipPlan(allocator, .{ .plan = &stage_plan_value, .descriptor_sets = &sets });
    defer plan.deinit();

    var lease = testLease(&plan, 0);
    try std.testing.expectEqual(@as(u64, 13), try validatePrefillTransition(&plan, &lease, 9, 4, 99));
    try std.testing.expectError(error.StatePositionMismatch, validatePrefillTransition(&plan, &lease, 8, 4, 99));
    try std.testing.expectError(error.InvalidSequenceRange, validateDecodeTransition(&plan, &lease, 9, 2, 99));
    try std.testing.expectEqual(@as(u64, 10), try validateDecodeTransition(&plan, &lease, 9, 1, 99));
    lease.expected_sequence_position = std.math.maxInt(u64);
    try std.testing.expectError(error.InvalidSequenceRange, validatePrefillTransition(&plan, &lease, std.math.maxInt(u64), 1, 99));
    lease.expected_sequence_position = 9;

    var other_sets = sets;
    var other_stage0 = [_]StateDescriptor{
        testDescriptor(runtime_contract.kv_cache_state_id, .request_scoped, runtime_contract.state_runtime_kind_kv_cache),
    };
    other_sets[0] = .{ .stage_id = 0, .descriptors = &other_stage0 };
    var other_plan = try buildStageStateOwnershipPlan(allocator, .{ .plan = &stage_plan_value, .descriptor_sets = &other_sets });
    defer other_plan.deinit();
    try std.testing.expectError(error.StateOwnershipPlanIdentityMismatch, validatePrefillTransition(&other_plan, &lease, 9, 1, 99));
    try std.testing.expectError(error.InvalidLeaseTransition, validateLeaseTransition(.{
        .kind = .decode,
        .lease = &lease,
        .request_id = lease.request_id,
        .stage_id = lease.stage_id,
        .slot_id = lease.slot_id,
        .sequence_start = lease.expected_sequence_position,
        .token_count = 1,
        .expected_epoch = lease.epoch,
    }));
    try std.testing.expectError(error.StateOwnershipPlanIdentityMismatch, validateLeaseTransition(.{
        .kind = .decode,
        .plan = &other_plan,
        .lease = &lease,
        .request_id = lease.request_id,
        .stage_id = lease.stage_id,
        .slot_id = lease.slot_id,
        .sequence_start = lease.expected_sequence_position,
        .token_count = 1,
        .expected_epoch = lease.epoch,
    }));
    const decoded = try validateLeaseTransition(.{
        .kind = .decode,
        .plan = &plan,
        .lease = &lease,
        .request_id = lease.request_id,
        .stage_id = lease.stage_id,
        .slot_id = lease.slot_id,
        .sequence_start = lease.expected_sequence_position,
        .token_count = 1,
        .expected_epoch = lease.epoch,
    });
    try std.testing.expectEqual(@as(u64, 10), decoded.expected_sequence_position);

    const bind_result = try validateLeaseTransition(.{
        .kind = .bind,
        .plan = &plan,
        .request_id = 88,
        .stage_id = 0,
        .slot_id = 7,
        .sequence_start = 0,
        .expected_epoch = 100,
    });
    try std.testing.expectEqual(StageStateLeaseState.active, bind_result.state);
    try std.testing.expectError(error.SlotAlreadyBound, validateLeaseTransition(.{
        .kind = .bind,
        .plan = &plan,
        .previous_lease = &lease,
        .request_id = 88,
        .stage_id = 0,
        .slot_id = 7,
        .expected_epoch = 100,
    }));

    lease.state = .completed;
    try std.testing.expectError(error.StaleLeaseEpochOnSlotReuse, validateLeaseTransition(.{
        .kind = .bind,
        .plan = &plan,
        .previous_lease = &lease,
        .request_id = 88,
        .stage_id = 0,
        .slot_id = 7,
        .expected_epoch = 99,
    }));
    const rebind = try validateLeaseTransition(.{
        .kind = .bind,
        .plan = &plan,
        .previous_lease = &lease,
        .request_id = 88,
        .stage_id = 0,
        .slot_id = 7,
        .expected_epoch = 101,
    });
    try std.testing.expectEqual(StageStateLeaseState.active, rebind.state);

    lease.state = .active;
    const completed = try validateLeaseTransition(.{
        .kind = .complete,
        .plan = &plan,
        .lease = &lease,
        .request_id = lease.request_id,
        .stage_id = lease.stage_id,
        .slot_id = lease.slot_id,
        .expected_epoch = lease.epoch,
    });
    try std.testing.expectEqual(StageStateLeaseState.completed, completed.state);
    const failed = try validateLeaseTransition(.{
        .kind = .fail,
        .plan = &plan,
        .lease = &lease,
        .request_id = lease.request_id,
        .stage_id = lease.stage_id,
        .slot_id = lease.slot_id,
        .expected_epoch = lease.epoch,
    });
    try std.testing.expectEqual(StageStateLeaseState.failed, failed.state);

    const cancelled = try validateLeaseTransition(.{
        .kind = .cancel,
        .plan = &plan,
        .lease = &lease,
        .request_id = lease.request_id,
        .stage_id = lease.stage_id,
        .slot_id = lease.slot_id,
        .expected_epoch = lease.epoch,
    });
    try std.testing.expectEqual(StageStateLeaseState.cancelled, cancelled.state);
    lease.state = .cancelled;
    const repeated_cancel = try validateLeaseTransition(.{
        .kind = .cancel,
        .plan = &plan,
        .lease = &lease,
        .request_id = lease.request_id,
        .stage_id = lease.stage_id,
        .slot_id = lease.slot_id,
        .expected_epoch = lease.epoch,
    });
    try std.testing.expect(repeated_cancel.cleanup_required);
    const repeated_fail_lease = StageStateLease{
        .graph_digest = lease.graph_digest,
        .graph_contract_version = lease.graph_contract_version,
        .stage_plan_contract_version = lease.stage_plan_contract_version,
        .stage_plan_id = lease.stage_plan_id,
        .ownership_plan_id = lease.ownership_plan_id,
        .stage_id = lease.stage_id,
        .request_id = lease.request_id,
        .slot_id = lease.slot_id,
        .expected_sequence_position = lease.expected_sequence_position,
        .epoch = lease.epoch,
        .state = .failed,
    };
    const repeated_fail = try validateLeaseTransition(.{
        .kind = .fail,
        .plan = &plan,
        .lease = &repeated_fail_lease,
        .request_id = repeated_fail_lease.request_id,
        .stage_id = repeated_fail_lease.stage_id,
        .slot_id = repeated_fail_lease.slot_id,
        .expected_epoch = repeated_fail_lease.epoch,
    });
    try std.testing.expect(repeated_fail.cleanup_required);
    const evicted = try validateLeaseTransition(.{
        .kind = .evict,
        .plan = &plan,
        .lease = &lease,
        .request_id = lease.request_id,
        .stage_id = lease.stage_id,
        .slot_id = lease.slot_id,
        .expected_epoch = lease.epoch,
    });
    try std.testing.expectEqual(StageStateLeaseState.evicted, evicted.state);
    lease.state = .evicted;
    const deinited = try validateLeaseTransition(.{
        .kind = .deinit,
        .plan = &plan,
        .lease = &lease,
        .request_id = lease.request_id,
        .stage_id = lease.stage_id,
        .slot_id = lease.slot_id,
        .expected_epoch = lease.epoch,
    });
    try std.testing.expectEqual(StageStateLeaseState.invalid, deinited.state);
}

test "inference pipeline state_ownership validateTensorFrameBatchEntryForStateLease validateTensorFrameForStateLeases enforces epoch and role" {
    const allocator = std.testing.allocator;
    var stage_plan_value = try buildTestStagePlan(allocator, &.{2}, &.{});
    defer stage_plan_value.deinit();
    const sets = testDescriptorSets();
    var ownership_plan = try buildStageStateOwnershipPlan(allocator, .{
        .plan = &stage_plan_value,
        .descriptor_sets = &sets,
        .strict_tensor_frame_epoch = true,
    });
    defer ownership_plan.deinit();

    var plan_ref = try tensor_frame.TensorFramePlanRef.fromStagePlan(allocator, &stage_plan_value);
    defer plan_ref.deinit();
    const boundary = try plan_ref.boundary(0);
    const contract = try tensor_frame.selectedBoundaryTensorContract(&plan_ref, 0, .f32, .row_major, .explicit);
    const tensor = try tensor_frame.TensorFrameTensorDesc.contiguousActivation(.f32, .{ 1, 1, 8, 0 });
    const batch = [_]tensor_frame.TensorFrameBatchEntry{.{
        .batch_index = 0,
        .request_id = 42,
        .slot_id = 7,
        .sequence_start = 9,
        .token_count = 1,
        .state_epoch = 99,
    }};
    const metadata = try tensor_frame.activationDecodeFrame(.{
        .frame_id = try tensor_frame.TensorFrameInstanceId.init(1),
        .plan_ref = &plan_ref,
        .boundary_index = 0,
        .selected_contract = &contract,
        .shape_context = .{ .expected_hidden_size = 8 },
        .tensor = tensor,
        .batch = .{ .entries = &batch },
        .payload = .{ .byte_count = tensor.payload_byte_count },
    });
    try std.testing.expectEqual(boundary.target_stage_id, metadata.boundary.target_stage_id);

    const consumer_lease = testLease(&ownership_plan, 1);
    try validateTensorFrameBatchEntryForStateLease(batch[0], &consumer_lease, true);
    try validateTensorFrameForStateLeases(&metadata, &ownership_plan, &.{consumer_lease}, .consumer, .{});
    try std.testing.expectError(
        error.StateStageMismatch,
        validateTensorFrameForStateLeases(&metadata, &ownership_plan, &.{consumer_lease}, .producer, .{}),
    );

    var fake_metadata = metadata;
    fake_metadata.boundary.target_stage_id = 0;
    fake_metadata.selected_contract.boundary = fake_metadata.boundary;
    try std.testing.expectError(
        error.BoundaryTensorContractMismatch,
        validateTensorFrameForStateLeases(&fake_metadata, &ownership_plan, &.{consumer_lease}, .consumer, .{}),
    );

    var wrong_version_plan = ownership_plan;
    wrong_version_plan.version = state_ownership_contract_version + 1;
    try std.testing.expectError(
        error.InvalidStateOwnershipContractVersion,
        validateTensorFrameForStateLeases(&metadata, &wrong_version_plan, &.{consumer_lease}, .consumer, .{}),
    );

    var missing_epoch = batch;
    missing_epoch[0].state_epoch = null;
    try std.testing.expectError(error.MissingStateEpoch, validateTensorFrameBatchEntryForStateLease(missing_epoch[0], &consumer_lease, true));

    var wrong_token = batch;
    wrong_token[0].token_count = 2;
    try std.testing.expectError(error.InvalidSequenceRange, validateTensorFrameBatchEntryForStateLease(wrong_token[0], &consumer_lease, true));
}

test "inference pipeline state_ownership validateStateDescriptorLifecycleAction shouldZeroStateDescriptorForLifecycleAction delegates lifecycle policy" {
    const request_descriptor = testDescriptor(12, .request_scoped, runtime_contract.state_runtime_kind_none);
    try validateStateDescriptorLifecycleAction(&request_descriptor, .alloc);
    try std.testing.expect(try shouldZeroStateDescriptorForLifecycleAction(&request_descriptor, .alloc));
    try std.testing.expectError(error.InvalidStateLifecycleAction, validateStateDescriptorLifecycleAction(&request_descriptor, .reset));

    const step_descriptor = testDescriptor(13, .step_scoped, runtime_contract.state_runtime_kind_none);
    try validateStateDescriptorLifecycleAction(&step_descriptor, .reset);
    try std.testing.expect(try shouldZeroStateDescriptorForLifecycleAction(&step_descriptor, .reset));
}

test "inference pipeline state_ownership buildStateCleanupObligations validateStateCleanupObligation is deterministic and idempotent" {
    const allocator = std.testing.allocator;
    var stage_plan_value = try buildTestStagePlan(allocator, &.{2}, &.{});
    defer stage_plan_value.deinit();
    const sets = testDescriptorSets();
    var plan = try buildStageStateOwnershipPlan(allocator, .{ .plan = &stage_plan_value, .descriptor_sets = &sets });
    defer plan.deinit();

    const touched = [_]StageStateCleanupTarget{
        .{ .stage_id = 0, .request_id = 42, .slot_id = 7 },
        .{ .stage_id = 1, .request_id = 42, .slot_id = 8 },
    };
    var storage_a: [8]StateCleanupObligation = undefined;
    var storage_b: [8]StateCleanupObligation = undefined;
    const obligations_a = try buildStateCleanupObligations(&plan, &touched, &storage_a);
    const obligations_b = try buildStateCleanupObligations(&plan, &touched, &storage_b);
    try std.testing.expectEqual(obligations_a.len, obligations_b.len);
    for (obligations_a, obligations_b) |a, b| {
        try validateStateCleanupObligation(&plan, a);
        try std.testing.expectEqual(a.stage_id, b.stage_id);
        try std.testing.expectEqual(a.descriptor_id, b.descriptor_id);
        try std.testing.expectEqual(a.order_key, b.order_key);
        try std.testing.expect(a.idempotent);
    }
    try std.testing.expectEqual(@as(usize, 1), obligations_a[0].stage_id);
}

test "inference pipeline state_ownership validateStageStateBindingReports catches supplied alias only when reports exist" {
    const allocator = std.testing.allocator;
    var stage_plan_value = try buildTestStagePlan(allocator, &.{2}, &.{});
    defer stage_plan_value.deinit();
    const sets = testDescriptorSets();
    var plan = try buildStageStateOwnershipPlan(allocator, .{ .plan = &stage_plan_value, .descriptor_sets = &sets });
    defer plan.deinit();

    try validateStageStateBindingReports(&plan, &.{});

    const ok = [_]StageStateBindingReport{
        .{ .stage_id = 0, .descriptor_id = runtime_contract.kv_cache_state_id, .payload_identity = 1, .payload_size = 64, .payload_alignment = 64 },
        .{ .stage_id = 1, .descriptor_id = runtime_contract.kv_cache_state_id, .payload_identity = 2, .payload_size = 64, .payload_alignment = 64 },
    };
    try validateStageStateBindingReports(&plan, &ok);

    const alias = [_]StageStateBindingReport{
        .{ .stage_id = 0, .descriptor_id = runtime_contract.kv_cache_state_id, .payload_identity = 9, .payload_size = 64, .payload_alignment = 64 },
        .{ .stage_id = 1, .descriptor_id = runtime_contract.kv_cache_state_id, .payload_identity = 9, .payload_size = 64, .payload_alignment = 64 },
    };
    try std.testing.expectError(error.CrossStageStateAliasUnsupported, validateStageStateBindingReports(&plan, &alias));

    const bad_alignment = [_]StageStateBindingReport{
        .{ .stage_id = 0, .descriptor_id = runtime_contract.kv_cache_state_id, .payload_identity = 3, .payload_size = 64, .payload_alignment = 128 },
    };
    try std.testing.expectError(error.InvalidStateBindingReport, validateStageStateBindingReports(&plan, &bad_alignment));
}
