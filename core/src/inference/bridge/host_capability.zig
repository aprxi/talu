//! Pure host capability, loaded residency, and placement contracts.
//!
//! This module validates cold placement metadata only. It does not discover
//! hosts, probe backends, load stages, move payloads, or change scheduler state.

const std = @import("std");
const models = @import("models_pkg");
const runtime_contract = @import("runtime_contract_pkg");
const state_ownership = @import("state_ownership.zig");
const tensor_frame = @import("tensor_frame.zig");

const Allocator = std.mem.Allocator;
const Sha256 = std.crypto.hash.sha2.Sha256;
const StagePlan = stage_plan.StagePlan;
const StageResidencyReport = manifest.StageResidencyReport;
const StateDescriptor = runtime_contract.StateDescriptor;
const StateLifecycle = runtime_contract.StateLifecycle;
const manifest = models.manifest;
const stage_plan = models.stage_plan;

pub const placement_contract_version: u32 = 1;

pub const PlacementError = stage_plan.StagePlanError || state_ownership.StateOwnershipError || tensor_frame.TensorFrameValidationError || error{
    InvalidPlacementContractVersion,
    HostCapabilityFingerprintMismatch,
    HostResidencyFingerprintMismatch,
    PlacementPlanFingerprintMismatch,
    InvalidHostId,
    DuplicateHostCapability,
    DuplicateHostResidencySnapshot,
    UnknownHostId,
    MissingStageBinding,
    DuplicateStageBinding,
    ExtraStageBinding,
    InvalidRequiredStepKindSet,
    MissingHostCapability,
    MissingHostResidency,
    InvalidStatePlacementMode,
    MissingStatePlacementSummary,
    MissingResidentStateOwnership,
    MismatchedResidentStateDescriptorSummary,
    InvalidResidentStageRange,
    MissingResidentStage,
    DuplicateResidentStage,
    MissingResidentGlobalRole,
    WrongResidentGlobalRoleOwner,
    DuplicateBoundaryFrameProfile,
    MissingBoundaryFrameProfile,
    BoundaryFrameProfileMismatch,
    UnsupportedBackendKind,
    UnsupportedReachabilityKind,
    RemoteReachabilityNotAllowed,
    UnsupportedTensorFrameContractVersion,
    UnsupportedBoundaryDType,
    UnsupportedBoundaryLayout,
    UnsupportedStepKind,
    UnsupportedHandoffMode,
    InvalidBatchEnvelope,
    InvalidTokenEnvelope,
    InvalidActivationPayloadEnvelope,
    ResidentCheckpointBudgetExceeded,
    InvalidResidentCheckpointBudget,
    InvalidDiagnosticWorkspaceBudget,
    DuplicateHostFrameCapability,
    InvalidCapabilitySet,
    UnsupportedStateOwnershipContractVersion,
};

pub const PlacementContractVersion = u32;

pub const HostId = struct {
    value: u64,
};

pub const HostBackendKind = enum(u8) {
    cpu,
    cuda,
    metal,
    mock,
    @"opaque",
};

pub const HostReachabilityKind = enum(u8) {
    local_in_process,
    remote_declared,
    mock,
};

pub const BoundaryHandoffMode = enum(u8) {
    same_host_direct,
    local_in_process,
    remote_declared,
    mock,
};

pub const BoundaryFrameEndpointRole = enum(u8) {
    producer,
    consumer,
};

pub const StatePlacementMode = enum(u8) {
    stateless_only,
    validate_ref,
};

pub const HostCapabilityId = struct {
    digest: [32]u8,
};

pub const HostResidencySnapshotId = struct {
    digest: [32]u8,
};

pub const PlacementPlanId = struct {
    digest: [32]u8,
};

pub const HostFrameCapability = struct {
    endpoint_role: BoundaryFrameEndpointRole,
    tensor_frame_contract_version: u32 = tensor_frame.tensor_frame_contract_version,
    step_kind: tensor_frame.TensorFrameStepKind,
    dtype: tensor_frame.TensorFrameDType,
    layout: tensor_frame.TensorFrameLayout = .row_major,
    handoff_mode: BoundaryHandoffMode,
    max_batch_entries: u64,
    max_token_count_per_frame: u64,
    max_activation_payload_bytes: u64,
};

pub const HostCapabilityRequest = struct {
    host_id: HostId,
    backend_kind: HostBackendKind,
    reachability_kind: HostReachabilityKind,
    supported_graph_contract_versions: []const u32 = &.{},
    supported_stage_plan_contract_versions: []const u32 = &.{},
    supported_state_ownership_contract_versions: []const u32 = &.{},
    frame_capabilities: []const HostFrameCapability,
    max_sequence_position: ?u64 = null,
    resident_checkpoint_budget_bytes: ?usize = null,
    diagnostic_workspace_budget_bytes: ?usize = null,
};

pub const HostCapability = struct {
    arena: std.heap.ArenaAllocator,
    version: PlacementContractVersion,
    host_id: HostId,
    capability_id: HostCapabilityId,
    backend_kind: HostBackendKind,
    reachability_kind: HostReachabilityKind,
    supported_graph_contract_versions: []const u32,
    supported_stage_plan_contract_versions: []const u32,
    supported_state_ownership_contract_versions: []const u32,
    frame_capabilities: []const HostFrameCapability,
    max_sequence_position: ?u64 = null,
    resident_checkpoint_budget_bytes: ?usize = null,
    diagnostic_workspace_budget_bytes: ?usize = null,

    pub fn deinit(self: *HostCapability) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

pub const StageStatePlacementDescriptorSummary = struct {
    descriptor_id: u8,
    size_bytes: u64,
    align_bytes: u16,
    zero_init: bool,
    lifecycle: StateLifecycle,
    runtime_kind: u8,
};

pub const StageStatePlacementStageSummary = struct {
    stage_id: usize,
    descriptor_count: usize,
    descriptors: []const StageStatePlacementDescriptorSummary,
    owns_runtime_state: bool,
};

pub const StageStatePlacementRef = struct {
    arena: std.heap.ArenaAllocator,
    state_ownership_contract_version: u32,
    graph_digest: [32]u8,
    graph_contract_version: u32,
    stage_plan_contract_version: u32,
    stage_plan_id: stage_plan.StagePlanId,
    state_ownership_plan_id: state_ownership.StageStateOwnershipPlanId,
    stage_ids: []const usize,
    stage_summaries: []const StageStatePlacementStageSummary,

    pub fn deinit(self: *StageStatePlacementRef) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

pub const ResidentStageStateSummary = struct {
    state_ownership_contract_version: u32,
    state_ownership_plan_id: state_ownership.StageStateOwnershipPlanId,
    stage_id: usize,
    descriptor_count: usize,
    descriptors: []const StageStatePlacementDescriptorSummary,
};

pub const ResidentStageEntry = struct {
    stage_id: usize,
    layer_start: usize,
    layer_end: usize,
    owned_roles: [manifest.role_count]bool,
    residency: StageResidencyReport,
    state_summary: ?ResidentStageStateSummary = null,
};

pub const HostResidencySnapshotRequest = struct {
    host_id: HostId,
    plan: *const StagePlan,
    state_ownership_contract_version: ?u32 = null,
    state_ownership_plan_id: ?state_ownership.StageStateOwnershipPlanId = null,
    resident_stages: []const ResidentStageEntry,
};

pub const HostResidencySnapshot = struct {
    arena: std.heap.ArenaAllocator,
    version: PlacementContractVersion,
    host_id: HostId,
    snapshot_id: HostResidencySnapshotId,
    graph_digest: [32]u8,
    graph_contract_version: u32,
    stage_plan_contract_version: u32,
    stage_plan_id: stage_plan.StagePlanId,
    state_ownership_contract_version: ?u32 = null,
    state_ownership_plan_id: ?state_ownership.StageStateOwnershipPlanId = null,
    resident_stages: []const ResidentStageEntry,

    pub fn deinit(self: *HostResidencySnapshot) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

pub const BoundaryFrameProfile = struct {
    boundary_index: usize,
    source_stage_id: usize,
    target_stage_id: usize,
    tensor_frame_contract_version: u32 = tensor_frame.tensor_frame_contract_version,
    step_kind: tensor_frame.TensorFrameStepKind,
    dtype: tensor_frame.TensorFrameDType,
    layout: tensor_frame.TensorFrameLayout = .row_major,
    max_batch_entries: u64,
    max_token_count_per_frame: u64,
    max_activation_payload_bytes: u64,
    handoff_mode: BoundaryHandoffMode,
};

pub const StageHostBinding = struct {
    stage_id: usize,
    host_id: HostId,
    expected_capability_id: ?HostCapabilityId = null,
    expected_residency_snapshot_id: ?HostResidencySnapshotId = null,
};

const default_allowed_reachability = [_]HostReachabilityKind{ .local_in_process, .mock };

pub const PlacementRequest = struct {
    plan: *const StagePlan,
    required_step_kinds: []const tensor_frame.TensorFrameStepKind,
    host_capabilities: []const HostCapability,
    host_residency_snapshots: []const HostResidencySnapshot,
    stage_host_bindings: []const StageHostBinding,
    boundary_frame_profiles: []const BoundaryFrameProfile,
    state_placement_mode: StatePlacementMode = .stateless_only,
    state_placement_ref: ?*const StageStatePlacementRef = null,
    allowed_reachability: []const HostReachabilityKind = &default_allowed_reachability,
    stateful_execution_required: bool = false,
};

pub const PlacementStageSummary = struct {
    stage_id: usize,
    layer_start: usize,
    layer_end: usize,
    owned_roles: [manifest.role_count]bool,
    residency: StageResidencyReport,
};

pub const PlacementBoundarySummary = struct {
    boundary_index: usize,
    source_stage_id: usize,
    target_stage_id: usize,
    producer_layer_start: usize,
    producer_layer_end: usize,
    consumer_layer_start: usize,
    consumer_layer_end: usize,
};

pub const PlacementHostSummary = struct {
    host_id: HostId,
    capability_id: HostCapabilityId,
    residency_snapshot_id: HostResidencySnapshotId,
};

pub const PlacementPlan = struct {
    arena: std.heap.ArenaAllocator,
    version: PlacementContractVersion,
    graph_digest: [32]u8,
    graph_contract_version: u32,
    stage_plan_contract_version: u32,
    stage_plan_id: stage_plan.StagePlanId,
    plan_id: PlacementPlanId,
    stage_summaries: []const PlacementStageSummary,
    boundary_summaries: []const PlacementBoundarySummary,
    required_step_kinds: []const tensor_frame.TensorFrameStepKind,
    state_placement_mode: StatePlacementMode,
    state_ownership_contract_version: ?u32 = null,
    state_ownership_plan_id: ?state_ownership.StageStateOwnershipPlanId = null,
    state_stage_summaries: []const StageStatePlacementStageSummary = &.{},
    stage_host_bindings: []const StageHostBinding,
    host_summaries: []const PlacementHostSummary,
    boundary_frame_profiles: []const BoundaryFrameProfile,

    pub fn deinit(self: *PlacementPlan) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

pub fn hostCapabilityIdEql(lhs: HostCapabilityId, rhs: HostCapabilityId) bool {
    return std.mem.eql(u8, &lhs.digest, &rhs.digest);
}

pub fn hostResidencySnapshotIdEql(lhs: HostResidencySnapshotId, rhs: HostResidencySnapshotId) bool {
    return std.mem.eql(u8, &lhs.digest, &rhs.digest);
}

pub fn placementPlanIdEql(lhs: PlacementPlanId, rhs: PlacementPlanId) bool {
    return std.mem.eql(u8, &lhs.digest, &rhs.digest);
}

pub fn buildHostCapability(allocator: Allocator, request: HostCapabilityRequest) PlacementError!HostCapability {
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_allocator = arena.allocator();

    const graph_versions = try copySortedU32(arena_allocator, request.supported_graph_contract_versions);
    const stage_versions = try copySortedU32(arena_allocator, request.supported_stage_plan_contract_versions);
    const state_versions = try copySortedU32(arena_allocator, request.supported_state_ownership_contract_versions);
    const frame_capabilities = try arena_allocator.dupe(HostFrameCapability, request.frame_capabilities);
    std.mem.sort(HostFrameCapability, frame_capabilities, {}, hostFrameCapabilityLess);

    var capability = HostCapability{
        .arena = arena,
        .version = placement_contract_version,
        .host_id = request.host_id,
        .capability_id = undefined,
        .backend_kind = request.backend_kind,
        .reachability_kind = request.reachability_kind,
        .supported_graph_contract_versions = graph_versions,
        .supported_stage_plan_contract_versions = stage_versions,
        .supported_state_ownership_contract_versions = state_versions,
        .frame_capabilities = frame_capabilities,
        .max_sequence_position = request.max_sequence_position,
        .resident_checkpoint_budget_bytes = request.resident_checkpoint_budget_bytes,
        .diagnostic_workspace_budget_bytes = request.diagnostic_workspace_budget_bytes,
    };
    try validateHostCapabilityShape(&capability);
    capability.capability_id = computeHostCapabilityId(&capability);
    try validateHostCapability(&capability);
    return capability;
}

pub fn validateHostCapability(capability: *const HostCapability) PlacementError!void {
    try validateHostCapabilityShape(capability);
    const expected = computeHostCapabilityId(capability);
    if (!hostCapabilityIdEql(capability.capability_id, expected)) return error.HostCapabilityFingerprintMismatch;
}

pub fn buildStageStatePlacementRef(
    allocator: Allocator,
    ownership_plan: *const state_ownership.StageStateOwnershipPlan,
) PlacementError!StageStatePlacementRef {
    try state_ownership.validateStageStateOwnershipPlan(ownership_plan);

    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_allocator = arena.allocator();

    const stage_ids = try arena_allocator.alloc(usize, ownership_plan.stage_entries.len);
    const summaries = try arena_allocator.alloc(StageStatePlacementStageSummary, ownership_plan.stage_entries.len);
    for (ownership_plan.stage_entries, 0..) |entry, index| {
        stage_ids[index] = entry.stage_id;
        const descriptors = try copyDescriptorSummaries(arena_allocator, entry.descriptors);
        summaries[index] = .{
            .stage_id = entry.stage_id,
            .descriptor_count = descriptors.len,
            .descriptors = descriptors,
            .owns_runtime_state = descriptors.len > 0,
        };
    }

    return .{
        .arena = arena,
        .state_ownership_contract_version = ownership_plan.version,
        .graph_digest = ownership_plan.graph_digest,
        .graph_contract_version = ownership_plan.graph_contract_version,
        .stage_plan_contract_version = ownership_plan.stage_plan_contract_version,
        .stage_plan_id = ownership_plan.stage_plan_id,
        .state_ownership_plan_id = ownership_plan.plan_id,
        .stage_ids = stage_ids,
        .stage_summaries = summaries,
    };
}

pub fn validateStageStatePlacementRef(ref: *const StageStatePlacementRef, plan: *const StagePlan) PlacementError!void {
    try stage_plan.validateStagePlan(plan, .{});
    if (ref.state_ownership_contract_version != state_ownership.state_ownership_contract_version) {
        return error.InvalidStateOwnershipContractVersion;
    }
    if (!std.mem.eql(u8, &ref.graph_digest, &plan.graph_identity.digest) or
        ref.graph_contract_version != plan.graph_identity.graph_contract_version)
    {
        return error.GraphIdentityMismatch;
    }
    if (ref.stage_plan_contract_version != plan.stage_contract_version or
        !std.mem.eql(u8, &ref.stage_plan_id.digest, &plan.plan_id.digest))
    {
        return error.StagePlanIdentityMismatch;
    }
    if (ref.stage_ids.len != plan.stages.len or ref.stage_summaries.len != plan.stages.len) {
        return error.MissingStatePlacementSummary;
    }
    for (plan.stages, 0..) |stage_entry, index| {
        if (ref.stage_ids[index] != stage_entry.id) return error.MissingStatePlacementSummary;
        const summary = ref.stage_summaries[index];
        if (summary.stage_id != stage_entry.id) return error.MissingStatePlacementSummary;
        if (summary.descriptor_count != summary.descriptors.len) return error.MismatchedResidentStateDescriptorSummary;
        if (summary.owns_runtime_state != (summary.descriptors.len > 0)) return error.MismatchedResidentStateDescriptorSummary;
        try validateDescriptorSummaryOrder(summary.descriptors);
    }
}

pub fn buildHostResidencySnapshot(
    allocator: Allocator,
    request: HostResidencySnapshotRequest,
) PlacementError!HostResidencySnapshot {
    try stage_plan.validateStagePlan(request.plan, .{});
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_allocator = arena.allocator();

    const resident_stages = try copyResidentStages(arena_allocator, request.resident_stages);
    var snapshot = HostResidencySnapshot{
        .arena = arena,
        .version = placement_contract_version,
        .host_id = request.host_id,
        .snapshot_id = undefined,
        .graph_digest = request.plan.graph_identity.digest,
        .graph_contract_version = request.plan.graph_identity.graph_contract_version,
        .stage_plan_contract_version = request.plan.stage_contract_version,
        .stage_plan_id = request.plan.plan_id,
        .state_ownership_contract_version = request.state_ownership_contract_version,
        .state_ownership_plan_id = request.state_ownership_plan_id,
        .resident_stages = resident_stages,
    };
    try validateHostResidencySnapshotShape(&snapshot);
    for (resident_stages) |entry| try validateResidentStageAgainstPlan(request.plan, entry);
    snapshot.snapshot_id = computeHostResidencySnapshotId(&snapshot);
    try validateHostResidencySnapshot(&snapshot);
    return snapshot;
}

pub fn validateHostResidencySnapshot(snapshot: *const HostResidencySnapshot) PlacementError!void {
    try validateHostResidencySnapshotShape(snapshot);
    const expected = computeHostResidencySnapshotId(snapshot);
    if (!hostResidencySnapshotIdEql(snapshot.snapshot_id, expected)) return error.HostResidencyFingerprintMismatch;
}

pub fn validateBoundaryFrameProfileForProducer(
    capability: *const HostCapability,
    profile: BoundaryFrameProfile,
) PlacementError!void {
    try validateBoundaryFrameProfileForRole(capability, profile, .producer);
}

pub fn validateBoundaryFrameProfileForConsumer(
    capability: *const HostCapability,
    profile: BoundaryFrameProfile,
) PlacementError!void {
    try validateBoundaryFrameProfileForRole(capability, profile, .consumer);
}

pub fn validateBoundaryFrameProfileCardinality(
    boundaries: []const PlacementBoundarySummary,
    required_step_kinds: []const tensor_frame.TensorFrameStepKind,
    profiles: []const BoundaryFrameProfile,
) PlacementError!void {
    try validateRequiredStepKinds(required_step_kinds);
    var cursor: usize = 0;
    for (boundaries) |boundary| {
        for (required_step_kinds) |step_kind| {
            if (cursor >= profiles.len) return error.MissingBoundaryFrameProfile;
            const profile = profiles[cursor];
            if (profile.boundary_index != boundary.boundary_index or profile.step_kind != step_kind) {
                return error.MissingBoundaryFrameProfile;
            }
            try validateProfileMatchesBoundary(profile, boundary);
            if (cursor > 0 and !boundaryFrameProfileLess({}, profiles[cursor - 1], profile)) {
                return error.DuplicateBoundaryFrameProfile;
            }
            cursor += 1;
        }
    }
    if (cursor != profiles.len) return error.DuplicateBoundaryFrameProfile;
}

pub fn buildPlacementPlan(allocator: Allocator, request: PlacementRequest) PlacementError!PlacementPlan {
    try stage_plan.validateStagePlan(request.plan, .{});
    try validateRequiredStepKinds(request.required_step_kinds);
    try validateCapabilitySet(request.host_capabilities, request.allowed_reachability, request.plan);
    try validateResidencySet(request.host_residency_snapshots, request.host_capabilities, request.plan);

    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_allocator = arena.allocator();

    const stage_summaries = try copyStageSummaries(arena_allocator, request.plan);
    const boundary_summaries = try copyBoundarySummaries(arena_allocator, request.plan);
    const required_step_kinds = try arena_allocator.dupe(tensor_frame.TensorFrameStepKind, request.required_step_kinds);
    const bindings = try arena_allocator.dupe(StageHostBinding, request.stage_host_bindings);
    std.mem.sort(StageHostBinding, bindings, {}, stageHostBindingLess);
    try validateBindings(request.plan, bindings, request.host_capabilities, request.host_residency_snapshots);

    const state_copy = try copyAndValidateStatePlacement(arena_allocator, request);
    try validateBoundResidency(request.plan, bindings, request.host_capabilities, request.host_residency_snapshots, state_copy);

    const profiles = try arena_allocator.dupe(BoundaryFrameProfile, request.boundary_frame_profiles);
    std.mem.sort(BoundaryFrameProfile, profiles, {}, boundaryFrameProfileLess);
    try validateBoundaryFrameProfileCardinality(boundary_summaries, required_step_kinds, profiles);
    try validateBoundaryProfilesForPlacement(bindings, request.host_capabilities, boundary_summaries, profiles, request.allowed_reachability);

    const host_summaries = try copyBoundHostSummaries(arena_allocator, bindings, request.host_capabilities, request.host_residency_snapshots);

    var placement = PlacementPlan{
        .arena = arena,
        .version = placement_contract_version,
        .graph_digest = request.plan.graph_identity.digest,
        .graph_contract_version = request.plan.graph_identity.graph_contract_version,
        .stage_plan_contract_version = request.plan.stage_contract_version,
        .stage_plan_id = request.plan.plan_id,
        .plan_id = undefined,
        .stage_summaries = stage_summaries,
        .boundary_summaries = boundary_summaries,
        .required_step_kinds = required_step_kinds,
        .state_placement_mode = request.state_placement_mode,
        .state_ownership_contract_version = state_copy.state_ownership_contract_version,
        .state_ownership_plan_id = state_copy.state_ownership_plan_id,
        .state_stage_summaries = state_copy.stage_summaries,
        .stage_host_bindings = bindings,
        .host_summaries = host_summaries,
        .boundary_frame_profiles = profiles,
    };
    placement.plan_id = computePlacementPlanId(&placement);
    try validatePlacementPlan(&placement);
    return placement;
}

pub fn validatePlacementPlan(plan: *const PlacementPlan) PlacementError!void {
    if (plan.version != placement_contract_version) return error.InvalidPlacementContractVersion;
    try validateRequiredStepKinds(plan.required_step_kinds);
    try validatePlacementStageSummaries(plan.stage_summaries);
    try validatePlacementBoundarySummaries(plan.stage_summaries, plan.boundary_summaries);
    try validatePlacementBindings(plan.stage_summaries, plan.stage_host_bindings, plan.host_summaries);
    try validatePlacementHostSummaries(plan.host_summaries);
    try validateBoundaryFrameProfileCardinality(plan.boundary_summaries, plan.required_step_kinds, plan.boundary_frame_profiles);
    try validatePlanStateSummary(plan);
    const expected = computePlacementPlanId(plan);
    if (!placementPlanIdEql(plan.plan_id, expected)) return error.PlacementPlanFingerprintMismatch;
}

pub fn bindingForStage(plan: *const PlacementPlan, stage_id: usize) PlacementError!StageHostBinding {
    for (plan.stage_host_bindings) |binding| {
        if (binding.stage_id == stage_id) return binding;
    }
    return error.UnknownStageId;
}

pub fn stageIdsForHost(plan: *const PlacementPlan, host_id: HostId, out: []usize) PlacementError![]usize {
    if (host_id.value == 0) return error.InvalidHostId;
    var count: usize = 0;
    for (plan.stage_host_bindings) |binding| {
        if (!hostIdEql(binding.host_id, host_id)) continue;
        if (count >= out.len) return error.InvalidCapabilitySet;
        out[count] = binding.stage_id;
        count += 1;
    }
    if (count == 0) return error.UnknownHostId;
    return out[0..count];
}

pub fn hostSummaryForStage(plan: *const PlacementPlan, stage_id: usize) PlacementError!PlacementHostSummary {
    const binding = try bindingForStage(plan, stage_id);
    for (plan.host_summaries) |summary| {
        if (hostIdEql(summary.host_id, binding.host_id)) return summary;
    }
    return error.UnknownHostId;
}

const StatePlacementCopy = struct {
    state_ownership_contract_version: ?u32 = null,
    state_ownership_plan_id: ?state_ownership.StageStateOwnershipPlanId = null,
    stage_summaries: []const StageStatePlacementStageSummary = &.{},
};

fn validateHostCapabilityShape(capability: *const HostCapability) PlacementError!void {
    if (capability.version != placement_contract_version) return error.InvalidPlacementContractVersion;
    try validateHostId(capability.host_id);
    if (capability.frame_capabilities.len == 0) return error.InvalidCapabilitySet;
    try validateSortedUniqueU32(capability.supported_graph_contract_versions);
    try validateSortedUniqueU32(capability.supported_stage_plan_contract_versions);
    try validateSortedUniqueU32(capability.supported_state_ownership_contract_versions);
    if (capability.resident_checkpoint_budget_bytes) |budget| {
        if (budget == 0) return error.InvalidResidentCheckpointBudget;
    }
    if (capability.diagnostic_workspace_budget_bytes) |budget| {
        if (budget == 0) return error.InvalidDiagnosticWorkspaceBudget;
    }
    var previous: ?HostFrameCapability = null;
    for (capability.frame_capabilities) |frame| {
        if (previous) |prev| {
            if (!hostFrameCapabilityLess({}, prev, frame)) return error.DuplicateHostFrameCapability;
        }
        try validateHostFrameCapability(frame);
        previous = frame;
    }
}

fn validateHostFrameCapability(frame: HostFrameCapability) PlacementError!void {
    if (frame.tensor_frame_contract_version != tensor_frame.tensor_frame_contract_version) {
        return error.UnsupportedTensorFrameContractVersion;
    }
    if (frame.layout != .row_major) return error.UnsupportedBoundaryLayout;
    if (frame.max_batch_entries == 0) return error.InvalidBatchEnvelope;
    if (frame.max_token_count_per_frame == 0) return error.InvalidTokenEnvelope;
    if (frame.max_activation_payload_bytes == 0) return error.InvalidActivationPayloadEnvelope;
}

fn validateHostResidencySnapshotShape(snapshot: *const HostResidencySnapshot) PlacementError!void {
    if (snapshot.version != placement_contract_version) return error.InvalidPlacementContractVersion;
    try validateHostId(snapshot.host_id);
    if (snapshot.state_ownership_contract_version == null and snapshot.state_ownership_plan_id != null) {
        return error.MissingResidentStateOwnership;
    }
    if (snapshot.state_ownership_contract_version != null and snapshot.state_ownership_plan_id == null) {
        return error.MissingResidentStateOwnership;
    }
    var previous_stage_id: ?usize = null;
    for (snapshot.resident_stages) |entry| {
        if (previous_stage_id) |previous| {
            if (entry.stage_id <= previous) return error.DuplicateResidentStage;
        }
        if (entry.layer_start >= entry.layer_end) return error.InvalidResidentStageRange;
        if (entry.residency.layer_start != entry.layer_start or entry.residency.layer_end != entry.layer_end) {
            return error.InvalidResidentStageRange;
        }
        if (entry.state_summary) |summary| {
            if (snapshot.state_ownership_contract_version == null or snapshot.state_ownership_plan_id == null) {
                return error.MissingResidentStateOwnership;
            }
            if (summary.state_ownership_contract_version != snapshot.state_ownership_contract_version.? or
                !stateOwnershipPlanIdEql(summary.state_ownership_plan_id, snapshot.state_ownership_plan_id.?))
            {
                return error.StateOwnershipPlanIdentityMismatch;
            }
            if (summary.stage_id != entry.stage_id or summary.descriptor_count != summary.descriptors.len) {
                return error.MismatchedResidentStateDescriptorSummary;
            }
            try validateDescriptorSummaryOrder(summary.descriptors);
        }
        previous_stage_id = entry.stage_id;
    }
}

fn validateBoundaryFrameProfileForRole(
    capability: *const HostCapability,
    profile: BoundaryFrameProfile,
    role: BoundaryFrameEndpointRole,
) PlacementError!void {
    try validateHostCapability(capability);
    try validateBoundaryFrameProfileShape(profile);
    for (capability.frame_capabilities) |frame| {
        if (frame.endpoint_role != role) continue;
        if (frame.tensor_frame_contract_version != profile.tensor_frame_contract_version) continue;
        if (frame.step_kind != profile.step_kind) continue;
        if (frame.dtype != profile.dtype) continue;
        if (frame.layout != profile.layout) continue;
        if (frame.handoff_mode != profile.handoff_mode) continue;
        if (profile.max_batch_entries > frame.max_batch_entries) return error.InvalidBatchEnvelope;
        if (profile.max_token_count_per_frame > frame.max_token_count_per_frame) return error.InvalidTokenEnvelope;
        if (profile.max_activation_payload_bytes > frame.max_activation_payload_bytes) return error.InvalidActivationPayloadEnvelope;
        return;
    }
    return error.BoundaryFrameProfileMismatch;
}

fn copyAndValidateStatePlacement(allocator: Allocator, request: PlacementRequest) PlacementError!StatePlacementCopy {
    switch (request.state_placement_mode) {
        .stateless_only => {
            if (request.state_placement_ref != null) return error.InvalidStatePlacementMode;
            if (request.stateful_execution_required) return error.InvalidStatePlacementMode;
            if (stagePlanHasStatefulDependency(request.plan)) return error.InvalidStatePlacementMode;
            for (request.host_residency_snapshots) |snapshot| {
                if (snapshot.state_ownership_contract_version != null or snapshot.state_ownership_plan_id != null) {
                    return error.InvalidStatePlacementMode;
                }
                for (snapshot.resident_stages) |entry| {
                    if (entry.state_summary != null) return error.InvalidStatePlacementMode;
                }
            }
            return .{};
        },
        .validate_ref => {
            const ref = request.state_placement_ref orelse return error.MissingStatePlacementSummary;
            try validateStageStatePlacementRef(ref, request.plan);
            const summaries = try copyStageStatePlacementStageSummaries(allocator, ref.stage_summaries);
            return .{
                .state_ownership_contract_version = ref.state_ownership_contract_version,
                .state_ownership_plan_id = ref.state_ownership_plan_id,
                .stage_summaries = summaries,
            };
        },
    }
}

fn validatePlanStateSummary(plan: *const PlacementPlan) PlacementError!void {
    switch (plan.state_placement_mode) {
        .stateless_only => {
            if (plan.state_ownership_contract_version != null or plan.state_ownership_plan_id != null or plan.state_stage_summaries.len != 0) {
                return error.InvalidStatePlacementMode;
            }
        },
        .validate_ref => {
            if (plan.state_ownership_contract_version == null or plan.state_ownership_plan_id == null) {
                return error.MissingStatePlacementSummary;
            }
            if (plan.state_stage_summaries.len != plan.stage_summaries.len) return error.MissingStatePlacementSummary;
            for (plan.state_stage_summaries, 0..) |summary, index| {
                if (summary.stage_id != plan.stage_summaries[index].stage_id) return error.MissingStatePlacementSummary;
                if (summary.descriptor_count != summary.descriptors.len) return error.MismatchedResidentStateDescriptorSummary;
                if (summary.owns_runtime_state != (summary.descriptors.len > 0)) return error.MismatchedResidentStateDescriptorSummary;
                try validateDescriptorSummaryOrder(summary.descriptors);
            }
        },
    }
}

fn validateCapabilitySet(
    capabilities: []const HostCapability,
    allowed_reachability: []const HostReachabilityKind,
    plan: *const StagePlan,
) PlacementError!void {
    for (capabilities, 0..) |*capability, index| {
        try validateHostCapability(capability);
        if (!reachabilityAllowed(allowed_reachability, capability.reachability_kind)) {
            return error.RemoteReachabilityNotAllowed;
        }
        if (!versionAllowed(capability.supported_graph_contract_versions, plan.graph_identity.graph_contract_version)) {
            return error.GraphIdentityMismatch;
        }
        if (!versionAllowed(capability.supported_stage_plan_contract_versions, plan.stage_contract_version)) {
            return error.StagePlanIdentityMismatch;
        }
        for (capabilities[0..index]) |*previous| {
            if (hostIdEql(previous.host_id, capability.host_id)) return error.DuplicateHostCapability;
        }
    }
}

fn validateResidencySet(
    snapshots: []const HostResidencySnapshot,
    capabilities: []const HostCapability,
    plan: *const StagePlan,
) PlacementError!void {
    for (snapshots, 0..) |*snapshot, index| {
        try validateHostResidencySnapshot(snapshot);
        _ = capabilityForHost(capabilities, snapshot.host_id) orelse return error.MissingHostCapability;
        if (!std.mem.eql(u8, &snapshot.graph_digest, &plan.graph_identity.digest) or
            snapshot.graph_contract_version != plan.graph_identity.graph_contract_version)
        {
            return error.GraphIdentityMismatch;
        }
        if (snapshot.stage_plan_contract_version != plan.stage_contract_version or
            !std.mem.eql(u8, &snapshot.stage_plan_id.digest, &plan.plan_id.digest))
        {
            return error.StagePlanIdentityMismatch;
        }
        for (snapshot.resident_stages) |entry| try validateResidentStageAgainstPlan(plan, entry);
        for (snapshots[0..index]) |*previous| {
            if (hostIdEql(previous.host_id, snapshot.host_id) and
                std.mem.eql(u8, &previous.graph_digest, &snapshot.graph_digest) and
                previous.graph_contract_version == snapshot.graph_contract_version and
                previous.stage_plan_contract_version == snapshot.stage_plan_contract_version and
                std.mem.eql(u8, &previous.stage_plan_id.digest, &snapshot.stage_plan_id.digest))
            {
                return error.DuplicateHostResidencySnapshot;
            }
        }
    }
}

fn validateBindings(
    plan: *const StagePlan,
    bindings: []const StageHostBinding,
    capabilities: []const HostCapability,
    snapshots: []const HostResidencySnapshot,
) PlacementError!void {
    if (bindings.len < plan.stages.len) return error.MissingStageBinding;
    if (bindings.len > plan.stages.len) return error.ExtraStageBinding;
    for (bindings, 0..) |binding, index| {
        if (index > 0 and bindings[index - 1].stage_id == binding.stage_id) return error.DuplicateStageBinding;
        if (index >= plan.stages.len or plan.stages[index].id != binding.stage_id) return error.MissingStageBinding;
        const capability = capabilityForHost(capabilities, binding.host_id) orelse return error.UnknownHostId;
        const snapshot = residencyForHost(snapshots, binding.host_id) orelse return error.MissingHostResidency;
        if (binding.expected_capability_id) |expected| {
            if (!hostCapabilityIdEql(expected, capability.capability_id)) return error.MissingHostCapability;
        }
        if (binding.expected_residency_snapshot_id) |expected| {
            if (!hostResidencySnapshotIdEql(expected, snapshot.snapshot_id)) return error.MissingHostResidency;
        }
    }
}

fn validateBoundResidency(
    plan: *const StagePlan,
    bindings: []const StageHostBinding,
    capabilities: []const HostCapability,
    snapshots: []const HostResidencySnapshot,
    state_copy: StatePlacementCopy,
) PlacementError!void {
    for (bindings) |binding| {
        const snapshot = residencyForHost(snapshots, binding.host_id) orelse return error.MissingHostResidency;
        const resident = residentStageForId(snapshot.resident_stages, binding.stage_id) orelse return error.MissingResidentStage;
        try validateResidentStageAgainstPlan(plan, resident);
        if (state_copy.state_ownership_plan_id) |ownership_id| {
            if (snapshot.state_ownership_contract_version) |snapshot_version| {
                if (snapshot_version != state_copy.state_ownership_contract_version.? or
                    !stateOwnershipPlanIdEql(snapshot.state_ownership_plan_id.?, ownership_id))
                {
                    return error.StateOwnershipPlanIdentityMismatch;
                }
            }
            const state_summary = stateSummaryForStage(state_copy.stage_summaries, binding.stage_id) orelse return error.MissingStatePlacementSummary;
            if (resident.state_summary) |resident_state| {
                if (resident_state.state_ownership_contract_version != state_copy.state_ownership_contract_version.? or
                    !stateOwnershipPlanIdEql(resident_state.state_ownership_plan_id, ownership_id))
                {
                    return error.StateOwnershipPlanIdentityMismatch;
                }
                if (!descriptorSummariesEql(resident_state.descriptors, state_summary.descriptors)) {
                    return error.MismatchedResidentStateDescriptorSummary;
                }
            } else if (state_summary.owns_runtime_state) {
                return error.MissingResidentStateOwnership;
            }
        }
    }
    if (state_copy.state_ownership_contract_version) |version| {
        for (capabilities) |capability| {
            if (!versionAllowed(capability.supported_state_ownership_contract_versions, version)) {
                return error.UnsupportedStateOwnershipContractVersion;
            }
        }
    }
    for (capabilities) |capability| {
        if (capability.resident_checkpoint_budget_bytes) |budget| {
            var total: usize = 0;
            var has_binding = false;
            for (bindings) |binding| {
                if (!hostIdEql(binding.host_id, capability.host_id)) continue;
                const snapshot = residencyForHost(snapshots, binding.host_id) orelse return error.MissingHostResidency;
                const resident = residentStageForId(snapshot.resident_stages, binding.stage_id) orelse return error.MissingResidentStage;
                total = std.math.add(usize, total, resident.residency.total_checkpoint_bytes) catch return error.ResidentCheckpointBudgetExceeded;
                has_binding = true;
            }
            if (has_binding and total > budget) return error.ResidentCheckpointBudgetExceeded;
        }
    }
}

fn validateBoundaryProfilesForPlacement(
    bindings: []const StageHostBinding,
    capabilities: []const HostCapability,
    boundaries: []const PlacementBoundarySummary,
    profiles: []const BoundaryFrameProfile,
    allowed_reachability: []const HostReachabilityKind,
) PlacementError!void {
    for (profiles) |profile| {
        const boundary = boundarySummaryForIndex(boundaries, profile.boundary_index) orelse return error.BoundaryFrameProfileMismatch;
        const source_binding = bindingForStageInSlice(bindings, boundary.source_stage_id) orelse return error.MissingStageBinding;
        const target_binding = bindingForStageInSlice(bindings, boundary.target_stage_id) orelse return error.MissingStageBinding;
        const source_capability = capabilityForHost(capabilities, source_binding.host_id) orelse return error.MissingHostCapability;
        const target_capability = capabilityForHost(capabilities, target_binding.host_id) orelse return error.MissingHostCapability;
        try validateHandoffMode(profile.handoff_mode, source_capability, target_capability, allowed_reachability);
        try validateBoundaryFrameProfileForProducer(source_capability, profile);
        try validateBoundaryFrameProfileForConsumer(target_capability, profile);
    }
}

fn validateHandoffMode(
    mode: BoundaryHandoffMode,
    source: *const HostCapability,
    target: *const HostCapability,
    allowed_reachability: []const HostReachabilityKind,
) PlacementError!void {
    const same_host = hostIdEql(source.host_id, target.host_id);
    switch (mode) {
        .same_host_direct => {
            if (!same_host) return error.UnsupportedHandoffMode;
        },
        .local_in_process => {
            if (same_host) return error.UnsupportedHandoffMode;
            if (source.reachability_kind == .remote_declared or target.reachability_kind == .remote_declared) {
                return error.UnsupportedHandoffMode;
            }
        },
        .remote_declared => {
            if (!reachabilityAllowed(allowed_reachability, .remote_declared)) return error.RemoteReachabilityNotAllowed;
            if (source.reachability_kind != .remote_declared and target.reachability_kind != .remote_declared) {
                return error.UnsupportedHandoffMode;
            }
        },
        .mock => {},
    }
}

fn copyBoundHostSummaries(
    allocator: Allocator,
    bindings: []const StageHostBinding,
    capabilities: []const HostCapability,
    snapshots: []const HostResidencySnapshot,
) PlacementError![]PlacementHostSummary {
    var count: usize = 0;
    for (bindings, 0..) |binding, index| {
        var seen = false;
        for (bindings[0..index]) |previous| {
            if (hostIdEql(previous.host_id, binding.host_id)) {
                seen = true;
                break;
            }
        }
        if (!seen) count += 1;
    }

    const summaries = try allocator.alloc(PlacementHostSummary, count);
    var out_index: usize = 0;
    for (bindings, 0..) |binding, index| {
        var seen = false;
        for (bindings[0..index]) |previous| {
            if (hostIdEql(previous.host_id, binding.host_id)) {
                seen = true;
                break;
            }
        }
        if (seen) continue;
        const capability = capabilityForHost(capabilities, binding.host_id) orelse return error.MissingHostCapability;
        const snapshot = residencyForHost(snapshots, binding.host_id) orelse return error.MissingHostResidency;
        summaries[out_index] = .{
            .host_id = binding.host_id,
            .capability_id = capability.capability_id,
            .residency_snapshot_id = snapshot.snapshot_id,
        };
        out_index += 1;
    }
    std.mem.sort(PlacementHostSummary, summaries, {}, placementHostSummaryLess);
    return summaries;
}

fn validatePlacementStageSummaries(stages: []const PlacementStageSummary) PlacementError!void {
    if (stages.len == 0) return error.MissingStageBinding;
    for (stages, 0..) |stage, index| {
        if (stage.stage_id != index) return error.UnknownStageId;
        if (stage.layer_start >= stage.layer_end) return error.InvalidResidentStageRange;
        if (stage.residency.layer_start != stage.layer_start or stage.residency.layer_end != stage.layer_end) {
            return error.InvalidResidentStageRange;
        }
    }
}

fn validatePlacementBoundarySummaries(
    stages: []const PlacementStageSummary,
    boundaries: []const PlacementBoundarySummary,
) PlacementError!void {
    if (stages.len == 1 and boundaries.len != 0) return error.BoundaryFrameProfileMismatch;
    if (stages.len > 1 and boundaries.len != stages.len - 1) return error.BoundaryFrameProfileMismatch;
    for (boundaries, 0..) |boundary, index| {
        if (boundary.boundary_index != index) return error.BoundaryFrameProfileMismatch;
        if (boundary.source_stage_id >= stages.len or boundary.target_stage_id >= stages.len) return error.UnknownStageId;
        if (boundary.source_stage_id + 1 != boundary.target_stage_id) return error.BoundaryFrameProfileMismatch;
        const source = stages[boundary.source_stage_id];
        const target = stages[boundary.target_stage_id];
        if (boundary.producer_layer_start != source.layer_start or boundary.producer_layer_end != source.layer_end) {
            return error.BoundaryFrameProfileMismatch;
        }
        if (boundary.consumer_layer_start != target.layer_start or boundary.consumer_layer_end != target.layer_end) {
            return error.BoundaryFrameProfileMismatch;
        }
    }
}

fn validatePlacementBindings(
    stages: []const PlacementStageSummary,
    bindings: []const StageHostBinding,
    hosts: []const PlacementHostSummary,
) PlacementError!void {
    if (bindings.len < stages.len) return error.MissingStageBinding;
    if (bindings.len > stages.len) return error.ExtraStageBinding;
    for (bindings, 0..) |binding, index| {
        if (index > 0 and bindings[index - 1].stage_id == binding.stage_id) return error.DuplicateStageBinding;
        if (binding.stage_id != stages[index].stage_id) return error.MissingStageBinding;
        _ = placementHostSummaryForHost(hosts, binding.host_id) orelse return error.UnknownHostId;
    }
}

fn validatePlacementHostSummaries(hosts: []const PlacementHostSummary) PlacementError!void {
    var previous: ?HostId = null;
    for (hosts) |host| {
        try validateHostId(host.host_id);
        if (previous) |prev| {
            if (hostIdLess(host.host_id, prev) or hostIdEql(host.host_id, prev)) return error.DuplicateHostCapability;
        }
        previous = host.host_id;
    }
}

fn validateResidentStageAgainstPlan(plan: *const StagePlan, entry: ResidentStageEntry) PlacementError!void {
    const stage = plan.stage(entry.stage_id) catch return error.UnknownStageId;
    if (entry.layer_start != stage.layer_start or entry.layer_end != stage.layer_end) {
        return error.InvalidResidentStageRange;
    }
    if (!ownedRolesEql(entry.owned_roles, stage.owned_roles)) return error.WrongResidentGlobalRoleOwner;
    if (!stageResidencyEql(entry.residency, stage.residency)) return error.ResidencyMismatch;
}

fn copyStageSummaries(allocator: Allocator, plan: *const StagePlan) PlacementError![]PlacementStageSummary {
    const summaries = try allocator.alloc(PlacementStageSummary, plan.stages.len);
    for (plan.stages, 0..) |stage, index| {
        summaries[index] = .{
            .stage_id = stage.id,
            .layer_start = stage.layer_start,
            .layer_end = stage.layer_end,
            .owned_roles = stage.owned_roles,
            .residency = stage.residency,
        };
    }
    return summaries;
}

fn copyBoundarySummaries(allocator: Allocator, plan: *const StagePlan) PlacementError![]PlacementBoundarySummary {
    const summaries = try allocator.alloc(PlacementBoundarySummary, plan.boundaries.len);
    for (plan.boundaries, 0..) |boundary, index| {
        summaries[index] = .{
            .boundary_index = index,
            .source_stage_id = boundary.source_stage_id,
            .target_stage_id = boundary.target_stage_id,
            .producer_layer_start = boundary.producer_layer_start,
            .producer_layer_end = boundary.producer_layer_end,
            .consumer_layer_start = boundary.consumer_layer_start,
            .consumer_layer_end = boundary.consumer_layer_end,
        };
    }
    return summaries;
}

fn copyResidentStages(allocator: Allocator, input: []const ResidentStageEntry) PlacementError![]ResidentStageEntry {
    const output = try allocator.alloc(ResidentStageEntry, input.len);
    for (input, 0..) |entry, index| {
        output[index] = entry;
        if (entry.state_summary) |summary| {
            const descriptors = try allocator.dupe(StageStatePlacementDescriptorSummary, summary.descriptors);
            std.mem.sort(StageStatePlacementDescriptorSummary, descriptors, {}, descriptorSummaryLess);
            output[index].state_summary = .{
                .state_ownership_contract_version = summary.state_ownership_contract_version,
                .state_ownership_plan_id = summary.state_ownership_plan_id,
                .stage_id = summary.stage_id,
                .descriptor_count = descriptors.len,
                .descriptors = descriptors,
            };
        }
    }
    std.mem.sort(ResidentStageEntry, output, {}, residentStageEntryLess);
    return output;
}

fn copyDescriptorSummaries(
    allocator: Allocator,
    descriptors: []const StateDescriptor,
) PlacementError![]StageStatePlacementDescriptorSummary {
    const output = try allocator.alloc(StageStatePlacementDescriptorSummary, descriptors.len);
    for (descriptors, 0..) |descriptor, index| {
        output[index] = .{
            .descriptor_id = descriptor.id,
            .size_bytes = descriptor.size_bytes,
            .align_bytes = descriptor.align_bytes,
            .zero_init = descriptor.zero_init,
            .lifecycle = descriptor.lifecycle,
            .runtime_kind = descriptor.runtime_kind,
        };
    }
    std.mem.sort(StageStatePlacementDescriptorSummary, output, {}, descriptorSummaryLess);
    return output;
}

fn copyStageStatePlacementStageSummaries(
    allocator: Allocator,
    input: []const StageStatePlacementStageSummary,
) PlacementError![]StageStatePlacementStageSummary {
    const output = try allocator.alloc(StageStatePlacementStageSummary, input.len);
    for (input, 0..) |summary, index| {
        const descriptors = try allocator.dupe(StageStatePlacementDescriptorSummary, summary.descriptors);
        std.mem.sort(StageStatePlacementDescriptorSummary, descriptors, {}, descriptorSummaryLess);
        output[index] = .{
            .stage_id = summary.stage_id,
            .descriptor_count = descriptors.len,
            .descriptors = descriptors,
            .owns_runtime_state = descriptors.len > 0,
        };
    }
    std.mem.sort(StageStatePlacementStageSummary, output, {}, stageStatePlacementStageSummaryLess);
    return output;
}

fn validateDescriptorSummaryOrder(descriptors: []const StageStatePlacementDescriptorSummary) PlacementError!void {
    var previous: ?StageStatePlacementDescriptorSummary = null;
    for (descriptors) |descriptor| {
        if (descriptor.size_bytes == 0) return error.MismatchedResidentStateDescriptorSummary;
        if (descriptor.align_bytes == 0 or !std.math.isPowerOfTwo(descriptor.align_bytes)) {
            return error.MismatchedResidentStateDescriptorSummary;
        }
        if (previous) |prev| {
            if (!descriptorSummaryLess({}, prev, descriptor)) return error.MismatchedResidentStateDescriptorSummary;
        }
        previous = descriptor;
    }
}

fn validateRequiredStepKinds(required_step_kinds: []const tensor_frame.TensorFrameStepKind) PlacementError!void {
    if (required_step_kinds.len == 0 or required_step_kinds.len > 2) return error.InvalidRequiredStepKindSet;
    for (required_step_kinds, 0..) |step, index| {
        switch (step) {
            .prefill, .decode => {},
        }
        if (index > 0 and @intFromEnum(required_step_kinds[index - 1]) >= @intFromEnum(step)) {
            return error.InvalidRequiredStepKindSet;
        }
    }
}

fn validateHostId(host_id: HostId) PlacementError!void {
    if (host_id.value == 0) return error.InvalidHostId;
}

fn versionAllowed(supported: []const u32, requested: u32) bool {
    if (supported.len == 0) return true;
    for (supported) |version| {
        if (version == requested) return true;
    }
    return false;
}

fn reachabilityAllowed(allowed: []const HostReachabilityKind, reachability: HostReachabilityKind) bool {
    for (allowed) |candidate| {
        if (candidate == reachability) return true;
    }
    return false;
}

fn capabilityForHost(capabilities: []const HostCapability, host_id: HostId) ?*const HostCapability {
    for (capabilities) |*capability| {
        if (hostIdEql(capability.host_id, host_id)) return capability;
    }
    return null;
}

fn residencyForHost(snapshots: []const HostResidencySnapshot, host_id: HostId) ?*const HostResidencySnapshot {
    for (snapshots) |*snapshot| {
        if (hostIdEql(snapshot.host_id, host_id)) return snapshot;
    }
    return null;
}

fn residentStageForId(entries: []const ResidentStageEntry, stage_id: usize) ?ResidentStageEntry {
    for (entries) |entry| {
        if (entry.stage_id == stage_id) return entry;
    }
    return null;
}

fn stateSummaryForStage(entries: []const StageStatePlacementStageSummary, stage_id: usize) ?StageStatePlacementStageSummary {
    for (entries) |entry| {
        if (entry.stage_id == stage_id) return entry;
    }
    return null;
}

fn bindingForStageInSlice(bindings: []const StageHostBinding, stage_id: usize) ?StageHostBinding {
    for (bindings) |binding| {
        if (binding.stage_id == stage_id) return binding;
    }
    return null;
}

fn boundarySummaryForIndex(boundaries: []const PlacementBoundarySummary, boundary_index: usize) ?PlacementBoundarySummary {
    for (boundaries) |boundary| {
        if (boundary.boundary_index == boundary_index) return boundary;
    }
    return null;
}

fn placementHostSummaryForHost(hosts: []const PlacementHostSummary, host_id: HostId) ?PlacementHostSummary {
    for (hosts) |host| {
        if (hostIdEql(host.host_id, host_id)) return host;
    }
    return null;
}

fn validateProfileMatchesBoundary(profile: BoundaryFrameProfile, boundary: PlacementBoundarySummary) PlacementError!void {
    if (profile.source_stage_id != boundary.source_stage_id or profile.target_stage_id != boundary.target_stage_id) {
        return error.BoundaryFrameProfileMismatch;
    }
    try validateBoundaryFrameProfileShape(profile);
}

fn validateBoundaryFrameProfileShape(profile: BoundaryFrameProfile) PlacementError!void {
    if (profile.tensor_frame_contract_version != tensor_frame.tensor_frame_contract_version) {
        return error.UnsupportedTensorFrameContractVersion;
    }
    if (profile.layout != .row_major) return error.UnsupportedBoundaryLayout;
    if (profile.max_batch_entries == 0) return error.InvalidBatchEnvelope;
    if (profile.max_token_count_per_frame == 0) return error.InvalidTokenEnvelope;
    if (profile.max_activation_payload_bytes == 0) return error.InvalidActivationPayloadEnvelope;
}

fn stagePlanHasStatefulDependency(plan: *const StagePlan) bool {
    for (plan.dependencies) |dependency| {
        if (dependency.reason == .stateful_decoder) return true;
    }
    return false;
}

fn descriptorSummariesEql(
    lhs: []const StageStatePlacementDescriptorSummary,
    rhs: []const StageStatePlacementDescriptorSummary,
) bool {
    if (lhs.len != rhs.len) return false;
    for (lhs, rhs) |a, b| {
        if (a.descriptor_id != b.descriptor_id or
            a.size_bytes != b.size_bytes or
            a.align_bytes != b.align_bytes or
            a.zero_init != b.zero_init or
            a.lifecycle != b.lifecycle or
            a.runtime_kind != b.runtime_kind)
        {
            return false;
        }
    }
    return true;
}

fn stageResidencyEql(lhs: StageResidencyReport, rhs: StageResidencyReport) bool {
    return lhs.layer_start == rhs.layer_start and
        lhs.layer_end == rhs.layer_end and
        lhs.total_checkpoint_bytes == rhs.total_checkpoint_bytes and
        std.mem.eql(usize, &lhs.role_bytes, &rhs.role_bytes);
}

fn ownedRolesEql(lhs: [manifest.role_count]bool, rhs: [manifest.role_count]bool) bool {
    return std.mem.eql(bool, &lhs, &rhs);
}

fn stateOwnershipPlanIdEql(
    lhs: state_ownership.StageStateOwnershipPlanId,
    rhs: state_ownership.StageStateOwnershipPlanId,
) bool {
    return std.mem.eql(u8, &lhs.digest, &rhs.digest);
}

fn hostIdEql(lhs: HostId, rhs: HostId) bool {
    return lhs.value == rhs.value;
}

fn hostIdLess(lhs: HostId, rhs: HostId) bool {
    return lhs.value < rhs.value;
}

fn copySortedU32(allocator: Allocator, values: []const u32) PlacementError![]u32 {
    const output = try allocator.dupe(u32, values);
    std.mem.sort(u32, output, {}, u32Less);
    try validateSortedUniqueU32(output);
    return output;
}

fn validateSortedUniqueU32(values: []const u32) PlacementError!void {
    for (values, 0..) |value, index| {
        if (value == 0) return error.InvalidCapabilitySet;
        if (index > 0 and values[index - 1] >= value) return error.InvalidCapabilitySet;
    }
}

fn computeHostCapabilityId(capability: *const HostCapability) HostCapabilityId {
    var encoder = HashEncoder.init();
    encoder.writeString("talu.host_capability");
    encoder.writeU32(capability.version);
    encoder.writeU64(capability.host_id.value);
    encoder.writeU8(@intFromEnum(capability.backend_kind));
    encoder.writeU8(@intFromEnum(capability.reachability_kind));
    writeU32Slice(&encoder, capability.supported_graph_contract_versions);
    writeU32Slice(&encoder, capability.supported_stage_plan_contract_versions);
    writeU32Slice(&encoder, capability.supported_state_ownership_contract_versions);
    encoder.writeUsize(capability.frame_capabilities.len);
    for (capability.frame_capabilities) |frame| writeHostFrameCapability(&encoder, frame);
    encoder.writeOptionalU64(capability.max_sequence_position);
    encoder.writeOptionalUsize(capability.resident_checkpoint_budget_bytes);
    encoder.writeOptionalUsize(capability.diagnostic_workspace_budget_bytes);
    return .{ .digest = encoder.finish() };
}

fn computeHostResidencySnapshotId(snapshot: *const HostResidencySnapshot) HostResidencySnapshotId {
    var encoder = HashEncoder.init();
    encoder.writeString("talu.host_residency");
    encoder.writeU32(snapshot.version);
    encoder.writeU64(snapshot.host_id.value);
    encoder.writeBytes(&snapshot.graph_digest);
    encoder.writeU32(snapshot.graph_contract_version);
    encoder.writeU32(snapshot.stage_plan_contract_version);
    encoder.writeBytes(&snapshot.stage_plan_id.digest);
    encoder.writeOptionalU32(snapshot.state_ownership_contract_version);
    writeOptionalStateOwnershipPlanId(&encoder, snapshot.state_ownership_plan_id);
    encoder.writeUsize(snapshot.resident_stages.len);
    for (snapshot.resident_stages) |entry| writeResidentStageEntry(&encoder, entry);
    return .{ .digest = encoder.finish() };
}

fn computePlacementPlanId(plan: *const PlacementPlan) PlacementPlanId {
    var encoder = HashEncoder.init();
    encoder.writeString("talu.placement_plan");
    encoder.writeU32(plan.version);
    encoder.writeBytes(&plan.graph_digest);
    encoder.writeU32(plan.graph_contract_version);
    encoder.writeU32(plan.stage_plan_contract_version);
    encoder.writeBytes(&plan.stage_plan_id.digest);
    encoder.writeUsize(plan.stage_summaries.len);
    for (plan.stage_summaries) |stage| writePlacementStageSummary(&encoder, stage);
    encoder.writeUsize(plan.boundary_summaries.len);
    for (plan.boundary_summaries) |boundary| writePlacementBoundarySummary(&encoder, boundary);
    encoder.writeUsize(plan.required_step_kinds.len);
    for (plan.required_step_kinds) |step| encoder.writeU8(@intFromEnum(step));
    encoder.writeU8(@intFromEnum(plan.state_placement_mode));
    encoder.writeOptionalU32(plan.state_ownership_contract_version);
    writeOptionalStateOwnershipPlanId(&encoder, plan.state_ownership_plan_id);
    encoder.writeUsize(plan.state_stage_summaries.len);
    for (plan.state_stage_summaries) |summary| writeStageStatePlacementStageSummary(&encoder, summary);
    encoder.writeUsize(plan.stage_host_bindings.len);
    for (plan.stage_host_bindings) |binding| writeStageHostBinding(&encoder, binding);
    encoder.writeUsize(plan.host_summaries.len);
    for (plan.host_summaries) |host| writePlacementHostSummary(&encoder, host);
    encoder.writeUsize(plan.boundary_frame_profiles.len);
    for (plan.boundary_frame_profiles) |profile| writeBoundaryFrameProfile(&encoder, profile);
    return .{ .digest = encoder.finish() };
}

fn writeU32Slice(encoder: *HashEncoder, values: []const u32) void {
    encoder.writeUsize(values.len);
    for (values) |value| encoder.writeU32(value);
}

fn writeHostFrameCapability(encoder: *HashEncoder, frame: HostFrameCapability) void {
    encoder.writeU8(@intFromEnum(frame.endpoint_role));
    encoder.writeU32(frame.tensor_frame_contract_version);
    encoder.writeU8(@intFromEnum(frame.step_kind));
    encoder.writeU8(@intFromEnum(frame.dtype));
    encoder.writeU8(@intFromEnum(frame.layout));
    encoder.writeU8(@intFromEnum(frame.handoff_mode));
    encoder.writeU64(frame.max_batch_entries);
    encoder.writeU64(frame.max_token_count_per_frame);
    encoder.writeU64(frame.max_activation_payload_bytes);
}

fn writeResidentStageEntry(encoder: *HashEncoder, entry: ResidentStageEntry) void {
    encoder.writeUsize(entry.stage_id);
    encoder.writeUsize(entry.layer_start);
    encoder.writeUsize(entry.layer_end);
    for (entry.owned_roles) |owned| encoder.writeBool(owned);
    writeStageResidencyReport(encoder, entry.residency);
    encoder.writeBool(entry.state_summary != null);
    if (entry.state_summary) |summary| writeResidentStageStateSummary(encoder, summary);
}

fn writeResidentStageStateSummary(encoder: *HashEncoder, summary: ResidentStageStateSummary) void {
    encoder.writeU32(summary.state_ownership_contract_version);
    encoder.writeBytes(&summary.state_ownership_plan_id.digest);
    encoder.writeUsize(summary.stage_id);
    encoder.writeUsize(summary.descriptor_count);
    encoder.writeUsize(summary.descriptors.len);
    for (summary.descriptors) |descriptor| writeDescriptorSummary(encoder, descriptor);
}

fn writePlacementStageSummary(encoder: *HashEncoder, stage: PlacementStageSummary) void {
    encoder.writeUsize(stage.stage_id);
    encoder.writeUsize(stage.layer_start);
    encoder.writeUsize(stage.layer_end);
    for (stage.owned_roles) |owned| encoder.writeBool(owned);
    writeStageResidencyReport(encoder, stage.residency);
}

fn writePlacementBoundarySummary(encoder: *HashEncoder, boundary: PlacementBoundarySummary) void {
    encoder.writeUsize(boundary.boundary_index);
    encoder.writeUsize(boundary.source_stage_id);
    encoder.writeUsize(boundary.target_stage_id);
    encoder.writeUsize(boundary.producer_layer_start);
    encoder.writeUsize(boundary.producer_layer_end);
    encoder.writeUsize(boundary.consumer_layer_start);
    encoder.writeUsize(boundary.consumer_layer_end);
}

fn writeStageStatePlacementStageSummary(encoder: *HashEncoder, summary: StageStatePlacementStageSummary) void {
    encoder.writeUsize(summary.stage_id);
    encoder.writeUsize(summary.descriptor_count);
    encoder.writeBool(summary.owns_runtime_state);
    encoder.writeUsize(summary.descriptors.len);
    for (summary.descriptors) |descriptor| writeDescriptorSummary(encoder, descriptor);
}

fn writeDescriptorSummary(encoder: *HashEncoder, descriptor: StageStatePlacementDescriptorSummary) void {
    encoder.writeU8(descriptor.descriptor_id);
    encoder.writeU64(descriptor.size_bytes);
    encoder.writeU16(descriptor.align_bytes);
    encoder.writeBool(descriptor.zero_init);
    encoder.writeU8(@intFromEnum(descriptor.lifecycle));
    encoder.writeU8(descriptor.runtime_kind);
}

fn writeStageHostBinding(encoder: *HashEncoder, binding: StageHostBinding) void {
    encoder.writeUsize(binding.stage_id);
    encoder.writeU64(binding.host_id.value);
    encoder.writeBool(binding.expected_capability_id != null);
    if (binding.expected_capability_id) |id| encoder.writeBytes(&id.digest);
    encoder.writeBool(binding.expected_residency_snapshot_id != null);
    if (binding.expected_residency_snapshot_id) |id| encoder.writeBytes(&id.digest);
}

fn writePlacementHostSummary(encoder: *HashEncoder, summary: PlacementHostSummary) void {
    encoder.writeU64(summary.host_id.value);
    encoder.writeBytes(&summary.capability_id.digest);
    encoder.writeBytes(&summary.residency_snapshot_id.digest);
}

fn writeBoundaryFrameProfile(encoder: *HashEncoder, profile: BoundaryFrameProfile) void {
    encoder.writeUsize(profile.boundary_index);
    encoder.writeUsize(profile.source_stage_id);
    encoder.writeUsize(profile.target_stage_id);
    encoder.writeU32(profile.tensor_frame_contract_version);
    encoder.writeU8(@intFromEnum(profile.step_kind));
    encoder.writeU8(@intFromEnum(profile.dtype));
    encoder.writeU8(@intFromEnum(profile.layout));
    encoder.writeU64(profile.max_batch_entries);
    encoder.writeU64(profile.max_token_count_per_frame);
    encoder.writeU64(profile.max_activation_payload_bytes);
    encoder.writeU8(@intFromEnum(profile.handoff_mode));
}

fn writeStageResidencyReport(encoder: *HashEncoder, residency: StageResidencyReport) void {
    encoder.writeUsize(residency.layer_start);
    encoder.writeUsize(residency.layer_end);
    encoder.writeUsize(residency.total_checkpoint_bytes);
    for (residency.role_bytes) |bytes| encoder.writeUsize(bytes);
}

fn writeOptionalStateOwnershipPlanId(encoder: *HashEncoder, value: ?state_ownership.StageStateOwnershipPlanId) void {
    encoder.writeBool(value != null);
    if (value) |id| encoder.writeBytes(&id.digest);
}

fn hostFrameCapabilityLess(_: void, lhs: HostFrameCapability, rhs: HostFrameCapability) bool {
    if (@intFromEnum(lhs.endpoint_role) != @intFromEnum(rhs.endpoint_role)) return @intFromEnum(lhs.endpoint_role) < @intFromEnum(rhs.endpoint_role);
    if (lhs.tensor_frame_contract_version != rhs.tensor_frame_contract_version) return lhs.tensor_frame_contract_version < rhs.tensor_frame_contract_version;
    if (@intFromEnum(lhs.step_kind) != @intFromEnum(rhs.step_kind)) return @intFromEnum(lhs.step_kind) < @intFromEnum(rhs.step_kind);
    if (@intFromEnum(lhs.dtype) != @intFromEnum(rhs.dtype)) return @intFromEnum(lhs.dtype) < @intFromEnum(rhs.dtype);
    if (@intFromEnum(lhs.layout) != @intFromEnum(rhs.layout)) return @intFromEnum(lhs.layout) < @intFromEnum(rhs.layout);
    if (@intFromEnum(lhs.handoff_mode) != @intFromEnum(rhs.handoff_mode)) return @intFromEnum(lhs.handoff_mode) < @intFromEnum(rhs.handoff_mode);
    return false;
}

fn boundaryFrameProfileLess(_: void, lhs: BoundaryFrameProfile, rhs: BoundaryFrameProfile) bool {
    if (lhs.boundary_index != rhs.boundary_index) return lhs.boundary_index < rhs.boundary_index;
    if (@intFromEnum(lhs.step_kind) != @intFromEnum(rhs.step_kind)) return @intFromEnum(lhs.step_kind) < @intFromEnum(rhs.step_kind);
    return false;
}

fn stageHostBindingLess(_: void, lhs: StageHostBinding, rhs: StageHostBinding) bool {
    return lhs.stage_id < rhs.stage_id;
}

fn placementHostSummaryLess(_: void, lhs: PlacementHostSummary, rhs: PlacementHostSummary) bool {
    return lhs.host_id.value < rhs.host_id.value;
}

fn residentStageEntryLess(_: void, lhs: ResidentStageEntry, rhs: ResidentStageEntry) bool {
    return lhs.stage_id < rhs.stage_id;
}

fn stageStatePlacementStageSummaryLess(_: void, lhs: StageStatePlacementStageSummary, rhs: StageStatePlacementStageSummary) bool {
    return lhs.stage_id < rhs.stage_id;
}

fn descriptorSummaryLess(_: void, lhs: StageStatePlacementDescriptorSummary, rhs: StageStatePlacementDescriptorSummary) bool {
    return lhs.descriptor_id < rhs.descriptor_id;
}

fn u32Less(_: void, lhs: u32, rhs: u32) bool {
    return lhs < rhs;
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
        .name = "host_capability_test",
        .model_types = &.{"host_capability_test"},
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
        .architecture_id = "host_capability_test",
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
    var model_manifest = try testManifest(allocator, 4);
    defer model_manifest.deinit();
    return stage_plan.buildStagePlan(allocator, .{
        .n_layers = 4,
        .split_points = splits,
        .architecture = &arch,
        .model_config = &config,
        .manifest = &model_manifest,
        .partition_constraints = .{
            .decoder_cuts_allowed = true,
            .dependency_overrides = dependencies,
        },
    });
}

fn testHostId(value: u64) HostId {
    return .{ .value = value };
}

fn testFrameCapabilities(mode: BoundaryHandoffMode) [4]HostFrameCapability {
    return .{
        .{ .endpoint_role = .producer, .step_kind = .prefill, .dtype = .f32, .handoff_mode = mode, .max_batch_entries = 8, .max_token_count_per_frame = 16, .max_activation_payload_bytes = 4096 },
        .{ .endpoint_role = .consumer, .step_kind = .prefill, .dtype = .f32, .handoff_mode = mode, .max_batch_entries = 8, .max_token_count_per_frame = 16, .max_activation_payload_bytes = 4096 },
        .{ .endpoint_role = .producer, .step_kind = .decode, .dtype = .f32, .handoff_mode = mode, .max_batch_entries = 8, .max_token_count_per_frame = 1, .max_activation_payload_bytes = 4096 },
        .{ .endpoint_role = .consumer, .step_kind = .decode, .dtype = .f32, .handoff_mode = mode, .max_batch_entries = 8, .max_token_count_per_frame = 1, .max_activation_payload_bytes = 4096 },
    };
}

fn buildTestCapability(
    allocator: Allocator,
    host_id: HostId,
    reachability: HostReachabilityKind,
    mode: BoundaryHandoffMode,
) !HostCapability {
    const frames = testFrameCapabilities(mode);
    return buildHostCapability(allocator, .{
        .host_id = host_id,
        .backend_kind = if (reachability == .mock) .mock else .cpu,
        .reachability_kind = reachability,
        .supported_graph_contract_versions = &.{stage_plan.graph_identity_contract_version},
        .supported_stage_plan_contract_versions = &.{stage_plan.stage_plan_contract_version},
        .supported_state_ownership_contract_versions = &.{state_ownership.state_ownership_contract_version},
        .frame_capabilities = &frames,
        .resident_checkpoint_budget_bytes = 1024,
        .diagnostic_workspace_budget_bytes = 1,
    });
}

fn residentEntryFromStage(stage: stage_plan.StagePlanStage) ResidentStageEntry {
    return .{
        .stage_id = stage.id,
        .layer_start = stage.layer_start,
        .layer_end = stage.layer_end,
        .owned_roles = stage.owned_roles,
        .residency = stage.residency,
    };
}

fn buildTestResidency(
    allocator: Allocator,
    host_id: HostId,
    plan: *const StagePlan,
    stages: []const usize,
) !HostResidencySnapshot {
    var entries: [4]ResidentStageEntry = undefined;
    for (stages, 0..) |stage_id, index| {
        entries[index] = residentEntryFromStage(plan.stages[stage_id]);
    }
    return buildHostResidencySnapshot(allocator, .{
        .host_id = host_id,
        .plan = plan,
        .resident_stages = entries[0..stages.len],
    });
}

fn testProfiles(plan: *const StagePlan, mode: BoundaryHandoffMode) [2]BoundaryFrameProfile {
    const boundary = plan.boundaries[0];
    return .{
        .{
            .boundary_index = 0,
            .source_stage_id = boundary.source_stage_id,
            .target_stage_id = boundary.target_stage_id,
            .step_kind = .prefill,
            .dtype = .f32,
            .max_batch_entries = 4,
            .max_token_count_per_frame = 8,
            .max_activation_payload_bytes = 512,
            .handoff_mode = mode,
        },
        .{
            .boundary_index = 0,
            .source_stage_id = boundary.source_stage_id,
            .target_stage_id = boundary.target_stage_id,
            .step_kind = .decode,
            .dtype = .f32,
            .max_batch_entries = 4,
            .max_token_count_per_frame = 1,
            .max_activation_payload_bytes = 512,
            .handoff_mode = mode,
        },
    };
}

test "inference bridge host_capability buildHostCapability validateHostCapability hostCapabilityIdEql boundary frame producer consumer validation" {
    const allocator = std.testing.allocator;
    var capability = try buildTestCapability(allocator, testHostId(1), .local_in_process, .local_in_process);
    defer capability.deinit();
    try validateHostCapability(&capability);
    try std.testing.expect(hostCapabilityIdEql(capability.capability_id, capability.capability_id));

    const profile = BoundaryFrameProfile{
        .boundary_index = 0,
        .source_stage_id = 0,
        .target_stage_id = 1,
        .step_kind = .prefill,
        .dtype = .f32,
        .max_batch_entries = 4,
        .max_token_count_per_frame = 8,
        .max_activation_payload_bytes = 512,
        .handoff_mode = .local_in_process,
    };
    try validateBoundaryFrameProfileForProducer(&capability, profile);
    try validateBoundaryFrameProfileForConsumer(&capability, profile);

    var too_large = profile;
    too_large.max_activation_payload_bytes = 8192;
    try std.testing.expectError(error.InvalidActivationPayloadEnvelope, validateBoundaryFrameProfileForProducer(&capability, too_large));
    var zero_batch = profile;
    zero_batch.max_batch_entries = 0;
    try std.testing.expectError(error.InvalidBatchEnvelope, validateBoundaryFrameProfileForProducer(&capability, zero_batch));
    var wrong_role = profile;
    wrong_role.handoff_mode = .same_host_direct;
    try std.testing.expectError(error.BoundaryFrameProfileMismatch, validateBoundaryFrameProfileForProducer(&capability, wrong_role));

    const frames = testFrameCapabilities(.local_in_process);
    try std.testing.expectError(error.InvalidDiagnosticWorkspaceBudget, buildHostCapability(allocator, .{
        .host_id = testHostId(2),
        .backend_kind = .cpu,
        .reachability_kind = .local_in_process,
        .frame_capabilities = &frames,
        .diagnostic_workspace_budget_bytes = 0,
    }));
    var different_workspace = try buildHostCapability(allocator, .{
        .host_id = testHostId(1),
        .backend_kind = .cpu,
        .reachability_kind = .local_in_process,
        .supported_graph_contract_versions = &.{stage_plan.graph_identity_contract_version},
        .supported_stage_plan_contract_versions = &.{stage_plan.stage_plan_contract_version},
        .supported_state_ownership_contract_versions = &.{state_ownership.state_ownership_contract_version},
        .frame_capabilities = &frames,
        .resident_checkpoint_budget_bytes = 1024,
        .diagnostic_workspace_budget_bytes = 2,
    });
    defer different_workspace.deinit();
    try std.testing.expect(!hostCapabilityIdEql(capability.capability_id, different_workspace.capability_id));

    var tampered = capability;
    tampered.resident_checkpoint_budget_bytes = 2048;
    try std.testing.expectError(error.HostCapabilityFingerprintMismatch, validateHostCapability(&tampered));
}

test "inference bridge host_capability buildHostResidencySnapshot validateHostResidencySnapshot hostResidencySnapshotIdEql exact stage residency" {
    const allocator = std.testing.allocator;
    var plan = try buildTestStagePlan(allocator, &.{2}, &.{});
    defer plan.deinit();
    var snapshot = try buildTestResidency(allocator, testHostId(1), &plan, &.{ 0, 1 });
    defer snapshot.deinit();
    try validateHostResidencySnapshot(&snapshot);
    try std.testing.expect(hostResidencySnapshotIdEql(snapshot.snapshot_id, snapshot.snapshot_id));

    var wrong_entry = residentEntryFromStage(plan.stages[0]);
    wrong_entry.residency.total_checkpoint_bytes += 1;
    try std.testing.expectError(error.ResidencyMismatch, buildHostResidencySnapshot(allocator, .{
        .host_id = testHostId(1),
        .plan = &plan,
        .resident_stages = &.{wrong_entry},
    }));

    var wrong_role_bytes = residentEntryFromStage(plan.stages[0]);
    wrong_role_bytes.residency.role_bytes[@intFromEnum(manifest.TensorRole.decoder_layer)] += 1;
    try std.testing.expectError(error.ResidencyMismatch, buildHostResidencySnapshot(allocator, .{
        .host_id = testHostId(1),
        .plan = &plan,
        .resident_stages = &.{wrong_role_bytes},
    }));

    var wrong_owner = residentEntryFromStage(plan.stages[0]);
    wrong_owner.owned_roles[@intFromEnum(manifest.TensorRole.lm_head)] = !wrong_owner.owned_roles[@intFromEnum(manifest.TensorRole.lm_head)];
    try std.testing.expectError(error.WrongResidentGlobalRoleOwner, buildHostResidencySnapshot(allocator, .{
        .host_id = testHostId(1),
        .plan = &plan,
        .resident_stages = &.{wrong_owner},
    }));

    var tampered_snapshot = snapshot;
    tampered_snapshot.snapshot_id.digest[0] ^= 1;
    try std.testing.expectError(error.HostResidencyFingerprintMismatch, validateHostResidencySnapshot(&tampered_snapshot));
}

test "inference bridge host_capability buildStageStatePlacementRef validateStageStatePlacementRef copies descriptor summaries" {
    const allocator = std.testing.allocator;
    const dependencies = [_]stage_plan.DependencyOverride{.{
        .source_stage_id = 0,
        .target_stage_id = 1,
        .reason = .stateful_decoder,
    }};
    var plan = try buildTestStagePlan(allocator, &.{2}, &dependencies);
    defer plan.deinit();
    const descriptors0 = [_]StateDescriptor{.{
        .id = runtime_contract.kv_cache_state_id,
        .size_bytes = 64,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_kv_cache,
    }};
    const descriptors1 = descriptors0;
    const sets = [_]state_ownership.StageStateDescriptorSet{
        .{ .stage_id = 0, .descriptors = &descriptors0 },
        .{ .stage_id = 1, .descriptors = &descriptors1 },
    };
    const facts = [_]state_ownership.StageStatePartitionFact{.{
        .boundary_index = 0,
        .source_stage_id = 0,
        .target_stage_id = 1,
        .reason = .stateful_decoder,
        .ownership_mode = .stage_level_dependency_only,
    }};
    var ownership = try state_ownership.buildStageStateOwnershipPlan(allocator, .{
        .plan = &plan,
        .descriptor_sets = &sets,
        .partition_facts = &facts,
    });
    defer ownership.deinit();
    var ref = try buildStageStatePlacementRef(allocator, &ownership);
    defer ref.deinit();
    try validateStageStatePlacementRef(&ref, &plan);
    try std.testing.expect(ref.stage_summaries[0].owns_runtime_state);
    try std.testing.expectEqual(@as(u8, runtime_contract.kv_cache_state_id), ref.stage_summaries[0].descriptors[0].descriptor_id);
}

test "inference bridge host_capability buildPlacementPlan validatePlacementPlan placementPlanIdEql bindingForStage stageIdsForHost hostSummaryForStage two host placement" {
    const allocator = std.testing.allocator;
    var stage_plan_value = try buildTestStagePlan(allocator, &.{2}, &.{});
    defer stage_plan_value.deinit();
    var cap0 = try buildTestCapability(allocator, testHostId(1), .local_in_process, .local_in_process);
    defer cap0.deinit();
    var cap1 = try buildTestCapability(allocator, testHostId(2), .local_in_process, .local_in_process);
    defer cap1.deinit();
    var res0 = try buildTestResidency(allocator, testHostId(1), &stage_plan_value, &.{0});
    defer res0.deinit();
    var res1 = try buildTestResidency(allocator, testHostId(2), &stage_plan_value, &.{1});
    defer res1.deinit();
    const profiles = testProfiles(&stage_plan_value, .local_in_process);
    const bindings = [_]StageHostBinding{
        .{ .stage_id = 0, .host_id = testHostId(1) },
        .{ .stage_id = 1, .host_id = testHostId(2) },
    };
    var placement = try buildPlacementPlan(allocator, .{
        .plan = &stage_plan_value,
        .required_step_kinds = &.{ .prefill, .decode },
        .host_capabilities = &.{ cap0, cap1 },
        .host_residency_snapshots = &.{ res0, res1 },
        .stage_host_bindings = &bindings,
        .boundary_frame_profiles = &profiles,
    });
    defer placement.deinit();
    try validatePlacementPlan(&placement);
    try std.testing.expect(placementPlanIdEql(placement.plan_id, placement.plan_id));
    try std.testing.expectEqual(@as(u64, 2), (try bindingForStage(&placement, 1)).host_id.value);
    var stage_storage: [2]usize = undefined;
    const host1_stages = try stageIdsForHost(&placement, testHostId(1), &stage_storage);
    try std.testing.expectEqual(@as(usize, 1), host1_stages.len);
    try std.testing.expectEqual(@as(usize, 0), host1_stages[0]);
    try std.testing.expect(hostCapabilityIdEql(cap1.capability_id, (try hostSummaryForStage(&placement, 1)).capability_id));

    var tampered = placement;
    tampered.host_summaries[0].capability_id.digest[0] ^= 1;
    try std.testing.expectError(error.PlacementPlanFingerprintMismatch, validatePlacementPlan(&tampered));
}

test "inference bridge host_capability buildPlacementPlan accepts single stage placement and rejects stateless resident state facts" {
    const allocator = std.testing.allocator;
    var plan = try buildTestStagePlan(allocator, &.{}, &.{});
    defer plan.deinit();
    const frames = testFrameCapabilities(.same_host_direct);
    var capability = try buildHostCapability(allocator, .{
        .host_id = testHostId(1),
        .backend_kind = .cpu,
        .reachability_kind = .local_in_process,
        .frame_capabilities = &frames,
        .resident_checkpoint_budget_bytes = 1024,
        .diagnostic_workspace_budget_bytes = 1,
    });
    defer capability.deinit();
    var residency = try buildTestResidency(allocator, testHostId(1), &plan, &.{0});
    defer residency.deinit();
    const bindings = [_]StageHostBinding{.{ .stage_id = 0, .host_id = testHostId(1) }};

    var placement = try buildPlacementPlan(allocator, .{
        .plan = &plan,
        .required_step_kinds = &.{.prefill},
        .host_capabilities = &.{capability},
        .host_residency_snapshots = &.{residency},
        .stage_host_bindings = &bindings,
        .boundary_frame_profiles = &.{},
    });
    defer placement.deinit();
    try validatePlacementPlan(&placement);

    const state_ownership_id = state_ownership.StageStateOwnershipPlanId{ .digest = [_]u8{7} ** 32 };
    var state_fact_residency = try buildHostResidencySnapshot(allocator, .{
        .host_id = testHostId(1),
        .plan = &plan,
        .state_ownership_contract_version = state_ownership.state_ownership_contract_version,
        .state_ownership_plan_id = state_ownership_id,
        .resident_stages = &.{residentEntryFromStage(plan.stages[0])},
    });
    defer state_fact_residency.deinit();
    try std.testing.expectError(error.InvalidStatePlacementMode, buildPlacementPlan(allocator, .{
        .plan = &plan,
        .required_step_kinds = &.{.prefill},
        .host_capabilities = &.{capability},
        .host_residency_snapshots = &.{state_fact_residency},
        .stage_host_bindings = &bindings,
        .boundary_frame_profiles = &.{},
        .state_placement_mode = .stateless_only,
    }));

    try std.testing.expectError(error.InvalidStatePlacementMode, buildPlacementPlan(allocator, .{
        .plan = &plan,
        .required_step_kinds = &.{.prefill},
        .host_capabilities = &.{capability},
        .host_residency_snapshots = &.{residency},
        .stage_host_bindings = &bindings,
        .boundary_frame_profiles = &.{},
        .state_placement_mode = .stateless_only,
        .stateful_execution_required = true,
    }));
}

test "inference bridge host_capability validateBoundaryFrameProfileCardinality and placement validation reject missing residency duplicate profiles budget and remote reachability" {
    const allocator = std.testing.allocator;
    var plan = try buildTestStagePlan(allocator, &.{2}, &.{});
    defer plan.deinit();
    var cap0 = try buildTestCapability(allocator, testHostId(1), .local_in_process, .local_in_process);
    defer cap0.deinit();
    var cap1 = try buildTestCapability(allocator, testHostId(2), .local_in_process, .local_in_process);
    defer cap1.deinit();
    var res0 = try buildTestResidency(allocator, testHostId(1), &plan, &.{0});
    defer res0.deinit();
    var res1 = try buildTestResidency(allocator, testHostId(2), &plan, &.{1});
    defer res1.deinit();
    const profiles = testProfiles(&plan, .local_in_process);
    const bindings = [_]StageHostBinding{
        .{ .stage_id = 0, .host_id = testHostId(1) },
        .{ .stage_id = 1, .host_id = testHostId(2) },
    };

    try std.testing.expectError(error.MissingHostResidency, buildPlacementPlan(allocator, .{
        .plan = &plan,
        .required_step_kinds = &.{.prefill},
        .host_capabilities = &.{ cap0, cap1 },
        .host_residency_snapshots = &.{res0},
        .stage_host_bindings = &bindings,
        .boundary_frame_profiles = profiles[0..1],
    }));

    const duplicate_profiles = [_]BoundaryFrameProfile{ profiles[0], profiles[0] };
    try std.testing.expectError(error.DuplicateBoundaryFrameProfile, buildPlacementPlan(allocator, .{
        .plan = &plan,
        .required_step_kinds = &.{ .prefill, .decode },
        .host_capabilities = &.{ cap0, cap1 },
        .host_residency_snapshots = &.{ res0, res1 },
        .stage_host_bindings = &bindings,
        .boundary_frame_profiles = &duplicate_profiles,
    }));

    const low_budget_frames = testFrameCapabilities(.local_in_process);
    var low_budget = try buildHostCapability(allocator, .{
        .host_id = testHostId(1),
        .backend_kind = .cpu,
        .reachability_kind = .local_in_process,
        .frame_capabilities = &low_budget_frames,
        .resident_checkpoint_budget_bytes = 1,
    });
    defer low_budget.deinit();
    try std.testing.expectError(error.ResidentCheckpointBudgetExceeded, buildPlacementPlan(allocator, .{
        .plan = &plan,
        .required_step_kinds = &.{ .prefill, .decode },
        .host_capabilities = &.{ low_budget, cap1 },
        .host_residency_snapshots = &.{ res0, res1 },
        .stage_host_bindings = &bindings,
        .boundary_frame_profiles = &profiles,
    }));

    const remote_frames = testFrameCapabilities(.remote_declared);
    var remote_cap = try buildHostCapability(allocator, .{
        .host_id = testHostId(3),
        .backend_kind = .@"opaque",
        .reachability_kind = .remote_declared,
        .frame_capabilities = &remote_frames,
    });
    defer remote_cap.deinit();
    var remote_res = try buildTestResidency(allocator, testHostId(3), &plan, &.{1});
    defer remote_res.deinit();
    var remote_profiles = testProfiles(&plan, .remote_declared);
    const remote_bindings = [_]StageHostBinding{
        .{ .stage_id = 0, .host_id = testHostId(1) },
        .{ .stage_id = 1, .host_id = testHostId(3) },
    };
    try std.testing.expectError(error.RemoteReachabilityNotAllowed, buildPlacementPlan(allocator, .{
        .plan = &plan,
        .required_step_kinds = &.{ .prefill, .decode },
        .host_capabilities = &.{ cap0, remote_cap },
        .host_residency_snapshots = &.{ res0, remote_res },
        .stage_host_bindings = &remote_bindings,
        .boundary_frame_profiles = &remote_profiles,
    }));
    var remote_ok = try buildPlacementPlan(allocator, .{
        .plan = &plan,
        .required_step_kinds = &.{ .prefill, .decode },
        .host_capabilities = &.{ cap0, remote_cap },
        .host_residency_snapshots = &.{ res0, remote_res },
        .stage_host_bindings = &remote_bindings,
        .boundary_frame_profiles = &remote_profiles,
        .allowed_reachability = &.{ .local_in_process, .mock, .remote_declared },
    });
    defer remote_ok.deinit();
    try validatePlacementPlan(&remote_ok);
}

test "inference bridge host_capability buildPlacementPlan rejects duplicate inputs step sets binding errors and identity mismatches" {
    const allocator = std.testing.allocator;
    var plan = try buildTestStagePlan(allocator, &.{2}, &.{});
    defer plan.deinit();
    var cap0 = try buildTestCapability(allocator, testHostId(1), .local_in_process, .local_in_process);
    defer cap0.deinit();
    var cap1 = try buildTestCapability(allocator, testHostId(2), .local_in_process, .local_in_process);
    defer cap1.deinit();
    var res0 = try buildTestResidency(allocator, testHostId(1), &plan, &.{0});
    defer res0.deinit();
    var res1 = try buildTestResidency(allocator, testHostId(2), &plan, &.{1});
    defer res1.deinit();
    const profiles = testProfiles(&plan, .local_in_process);
    const bindings = [_]StageHostBinding{
        .{ .stage_id = 0, .host_id = testHostId(1) },
        .{ .stage_id = 1, .host_id = testHostId(2) },
    };

    try std.testing.expectError(error.DuplicateHostCapability, buildPlacementPlan(allocator, .{
        .plan = &plan,
        .required_step_kinds = &.{.prefill},
        .host_capabilities = &.{ cap0, cap0 },
        .host_residency_snapshots = &.{ res0, res1 },
        .stage_host_bindings = &bindings,
        .boundary_frame_profiles = profiles[0..1],
    }));
    try std.testing.expectError(error.DuplicateHostResidencySnapshot, buildPlacementPlan(allocator, .{
        .plan = &plan,
        .required_step_kinds = &.{.prefill},
        .host_capabilities = &.{ cap0, cap1 },
        .host_residency_snapshots = &.{ res0, res1, res1 },
        .stage_host_bindings = &bindings,
        .boundary_frame_profiles = profiles[0..1],
    }));
    try std.testing.expectError(error.InvalidRequiredStepKindSet, buildPlacementPlan(allocator, .{
        .plan = &plan,
        .required_step_kinds = &.{},
        .host_capabilities = &.{ cap0, cap1 },
        .host_residency_snapshots = &.{ res0, res1 },
        .stage_host_bindings = &bindings,
        .boundary_frame_profiles = &.{},
    }));
    try std.testing.expectError(error.InvalidRequiredStepKindSet, buildPlacementPlan(allocator, .{
        .plan = &plan,
        .required_step_kinds = &.{ .prefill, .prefill },
        .host_capabilities = &.{ cap0, cap1 },
        .host_residency_snapshots = &.{ res0, res1 },
        .stage_host_bindings = &bindings,
        .boundary_frame_profiles = &profiles,
    }));
    try std.testing.expectError(error.InvalidRequiredStepKindSet, buildPlacementPlan(allocator, .{
        .plan = &plan,
        .required_step_kinds = &.{ .decode, .prefill },
        .host_capabilities = &.{ cap0, cap1 },
        .host_residency_snapshots = &.{ res0, res1 },
        .stage_host_bindings = &bindings,
        .boundary_frame_profiles = &profiles,
    }));

    try std.testing.expectError(error.MissingStageBinding, buildPlacementPlan(allocator, .{
        .plan = &plan,
        .required_step_kinds = &.{.prefill},
        .host_capabilities = &.{ cap0, cap1 },
        .host_residency_snapshots = &.{ res0, res1 },
        .stage_host_bindings = bindings[0..1],
        .boundary_frame_profiles = profiles[0..1],
    }));
    const duplicate_bindings = [_]StageHostBinding{ bindings[0], bindings[0] };
    try std.testing.expectError(error.DuplicateStageBinding, buildPlacementPlan(allocator, .{
        .plan = &plan,
        .required_step_kinds = &.{.prefill},
        .host_capabilities = &.{ cap0, cap1 },
        .host_residency_snapshots = &.{ res0, res1 },
        .stage_host_bindings = &duplicate_bindings,
        .boundary_frame_profiles = profiles[0..1],
    }));
    const extra_bindings = [_]StageHostBinding{
        bindings[0],
        bindings[1],
        .{ .stage_id = 42, .host_id = testHostId(2) },
    };
    try std.testing.expectError(error.ExtraStageBinding, buildPlacementPlan(allocator, .{
        .plan = &plan,
        .required_step_kinds = &.{.prefill},
        .host_capabilities = &.{ cap0, cap1 },
        .host_residency_snapshots = &.{ res0, res1 },
        .stage_host_bindings = &extra_bindings,
        .boundary_frame_profiles = profiles[0..1],
    }));

    var graph_mismatch = res1;
    graph_mismatch.graph_digest[0] ^= 1;
    graph_mismatch.snapshot_id = computeHostResidencySnapshotId(&graph_mismatch);
    try std.testing.expectError(error.GraphIdentityMismatch, buildPlacementPlan(allocator, .{
        .plan = &plan,
        .required_step_kinds = &.{.prefill},
        .host_capabilities = &.{ cap0, cap1 },
        .host_residency_snapshots = &.{ res0, graph_mismatch },
        .stage_host_bindings = &bindings,
        .boundary_frame_profiles = profiles[0..1],
    }));
    var stage_plan_mismatch = res1;
    stage_plan_mismatch.stage_plan_id.digest[0] ^= 1;
    stage_plan_mismatch.snapshot_id = computeHostResidencySnapshotId(&stage_plan_mismatch);
    try std.testing.expectError(error.StagePlanIdentityMismatch, buildPlacementPlan(allocator, .{
        .plan = &plan,
        .required_step_kinds = &.{.prefill},
        .host_capabilities = &.{ cap0, cap1 },
        .host_residency_snapshots = &.{ res0, stage_plan_mismatch },
        .stage_host_bindings = &bindings,
        .boundary_frame_profiles = profiles[0..1],
    }));
}

test "inference bridge host_capability buildPlacementPlan rejects unsupported boundary capability combinations" {
    const allocator = std.testing.allocator;
    var plan = try buildTestStagePlan(allocator, &.{2}, &.{});
    defer plan.deinit();
    var cap0 = try buildTestCapability(allocator, testHostId(1), .local_in_process, .local_in_process);
    defer cap0.deinit();
    var cap1 = try buildTestCapability(allocator, testHostId(2), .local_in_process, .local_in_process);
    defer cap1.deinit();
    var res0 = try buildTestResidency(allocator, testHostId(1), &plan, &.{0});
    defer res0.deinit();
    var res1 = try buildTestResidency(allocator, testHostId(2), &plan, &.{1});
    defer res1.deinit();
    var profiles = testProfiles(&plan, .local_in_process);
    const bindings = [_]StageHostBinding{
        .{ .stage_id = 0, .host_id = testHostId(1) },
        .{ .stage_id = 1, .host_id = testHostId(2) },
    };

    profiles[0].dtype = .f16;
    try std.testing.expectError(error.BoundaryFrameProfileMismatch, buildPlacementPlan(allocator, .{
        .plan = &plan,
        .required_step_kinds = &.{.prefill},
        .host_capabilities = &.{ cap0, cap1 },
        .host_residency_snapshots = &.{ res0, res1 },
        .stage_host_bindings = &bindings,
        .boundary_frame_profiles = profiles[0..1],
    }));
    profiles = testProfiles(&plan, .local_in_process);
    profiles[0].tensor_frame_contract_version = tensor_frame.tensor_frame_contract_version + 1;
    try std.testing.expectError(error.UnsupportedTensorFrameContractVersion, buildPlacementPlan(allocator, .{
        .plan = &plan,
        .required_step_kinds = &.{.prefill},
        .host_capabilities = &.{ cap0, cap1 },
        .host_residency_snapshots = &.{ res0, res1 },
        .stage_host_bindings = &bindings,
        .boundary_frame_profiles = profiles[0..1],
    }));
    profiles = testProfiles(&plan, .same_host_direct);
    try std.testing.expectError(error.UnsupportedHandoffMode, buildPlacementPlan(allocator, .{
        .plan = &plan,
        .required_step_kinds = &.{.prefill},
        .host_capabilities = &.{ cap0, cap1 },
        .host_residency_snapshots = &.{ res0, res1 },
        .stage_host_bindings = &bindings,
        .boundary_frame_profiles = profiles[0..1],
    }));

    const producer_only_frames = [_]HostFrameCapability{
        testFrameCapabilities(.local_in_process)[0],
        testFrameCapabilities(.local_in_process)[2],
    };
    var producer_only = try buildHostCapability(allocator, .{
        .host_id = testHostId(2),
        .backend_kind = .cpu,
        .reachability_kind = .local_in_process,
        .frame_capabilities = &producer_only_frames,
    });
    defer producer_only.deinit();
    profiles = testProfiles(&plan, .local_in_process);
    try std.testing.expectError(error.BoundaryFrameProfileMismatch, buildPlacementPlan(allocator, .{
        .plan = &plan,
        .required_step_kinds = &.{.prefill},
        .host_capabilities = &.{ cap0, producer_only },
        .host_residency_snapshots = &.{ res0, res1 },
        .stage_host_bindings = &bindings,
        .boundary_frame_profiles = profiles[0..1],
    }));

    const prefill_only_frames = [_]HostFrameCapability{
        testFrameCapabilities(.local_in_process)[0],
        testFrameCapabilities(.local_in_process)[1],
    };
    var prefill_only = try buildHostCapability(allocator, .{
        .host_id = testHostId(2),
        .backend_kind = .cpu,
        .reachability_kind = .local_in_process,
        .frame_capabilities = &prefill_only_frames,
    });
    defer prefill_only.deinit();
    try std.testing.expectError(error.BoundaryFrameProfileMismatch, buildPlacementPlan(allocator, .{
        .plan = &plan,
        .required_step_kinds = &.{ .prefill, .decode },
        .host_capabilities = &.{ cap0, prefill_only },
        .host_residency_snapshots = &.{ res0, res1 },
        .stage_host_bindings = &bindings,
        .boundary_frame_profiles = &testProfiles(&plan, .local_in_process),
    }));
}

test "inference bridge host_capability placementPlanIdEql changes with binding capability residency profile and state summary" {
    const allocator = std.testing.allocator;
    var plan = try buildTestStagePlan(allocator, &.{2}, &.{});
    defer plan.deinit();
    var cap0 = try buildTestCapability(allocator, testHostId(1), .local_in_process, .local_in_process);
    defer cap0.deinit();
    var cap1 = try buildTestCapability(allocator, testHostId(2), .local_in_process, .local_in_process);
    defer cap1.deinit();
    var res0 = try buildTestResidency(allocator, testHostId(1), &plan, &.{0});
    defer res0.deinit();
    var res1 = try buildTestResidency(allocator, testHostId(2), &plan, &.{1});
    defer res1.deinit();
    const profiles = testProfiles(&plan, .local_in_process);
    const bindings = [_]StageHostBinding{
        .{ .stage_id = 0, .host_id = testHostId(1) },
        .{ .stage_id = 1, .host_id = testHostId(2) },
    };
    var base = try buildPlacementPlan(allocator, .{
        .plan = &plan,
        .required_step_kinds = &.{ .prefill, .decode },
        .host_capabilities = &.{ cap0, cap1 },
        .host_residency_snapshots = &.{ res0, res1 },
        .stage_host_bindings = &bindings,
        .boundary_frame_profiles = &profiles,
    });
    defer base.deinit();

    var same_cap = try buildTestCapability(allocator, testHostId(1), .local_in_process, .same_host_direct);
    defer same_cap.deinit();
    var same_res = try buildTestResidency(allocator, testHostId(1), &plan, &.{ 0, 1 });
    defer same_res.deinit();
    const same_bindings = [_]StageHostBinding{
        .{ .stage_id = 0, .host_id = testHostId(1) },
        .{ .stage_id = 1, .host_id = testHostId(1) },
    };
    var same_host = try buildPlacementPlan(allocator, .{
        .plan = &plan,
        .required_step_kinds = &.{ .prefill, .decode },
        .host_capabilities = &.{same_cap},
        .host_residency_snapshots = &.{same_res},
        .stage_host_bindings = &same_bindings,
        .boundary_frame_profiles = &testProfiles(&plan, .same_host_direct),
    });
    defer same_host.deinit();
    try std.testing.expect(!placementPlanIdEql(base.plan_id, same_host.plan_id));

    const frames = testFrameCapabilities(.local_in_process);
    var cap1_changed = try buildHostCapability(allocator, .{
        .host_id = testHostId(2),
        .backend_kind = .cpu,
        .reachability_kind = .local_in_process,
        .supported_graph_contract_versions = &.{stage_plan.graph_identity_contract_version},
        .supported_stage_plan_contract_versions = &.{stage_plan.stage_plan_contract_version},
        .supported_state_ownership_contract_versions = &.{state_ownership.state_ownership_contract_version},
        .frame_capabilities = &frames,
        .resident_checkpoint_budget_bytes = 1024,
        .diagnostic_workspace_budget_bytes = 2,
    });
    defer cap1_changed.deinit();
    var capability_changed = try buildPlacementPlan(allocator, .{
        .plan = &plan,
        .required_step_kinds = &.{ .prefill, .decode },
        .host_capabilities = &.{ cap0, cap1_changed },
        .host_residency_snapshots = &.{ res0, res1 },
        .stage_host_bindings = &bindings,
        .boundary_frame_profiles = &profiles,
    });
    defer capability_changed.deinit();
    try std.testing.expect(!placementPlanIdEql(base.plan_id, capability_changed.plan_id));

    var res1_changed = try buildTestResidency(allocator, testHostId(2), &plan, &.{ 0, 1 });
    defer res1_changed.deinit();
    var residency_changed = try buildPlacementPlan(allocator, .{
        .plan = &plan,
        .required_step_kinds = &.{ .prefill, .decode },
        .host_capabilities = &.{ cap0, cap1 },
        .host_residency_snapshots = &.{ res0, res1_changed },
        .stage_host_bindings = &bindings,
        .boundary_frame_profiles = &profiles,
    });
    defer residency_changed.deinit();
    try std.testing.expect(!placementPlanIdEql(base.plan_id, residency_changed.plan_id));

    var profile_changed = profiles;
    profile_changed[1].max_activation_payload_bytes += 1;
    var boundary_changed = try buildPlacementPlan(allocator, .{
        .plan = &plan,
        .required_step_kinds = &.{ .prefill, .decode },
        .host_capabilities = &.{ cap0, cap1 },
        .host_residency_snapshots = &.{ res0, res1 },
        .stage_host_bindings = &bindings,
        .boundary_frame_profiles = &profile_changed,
    });
    defer boundary_changed.deinit();
    try std.testing.expect(!placementPlanIdEql(base.plan_id, boundary_changed.plan_id));

    try std.testing.expectError(error.MissingStatePlacementSummary, buildPlacementPlan(allocator, .{
        .plan = &plan,
        .required_step_kinds = &.{ .prefill, .decode },
        .host_capabilities = &.{ cap0, cap1 },
        .host_residency_snapshots = &.{ res0, res1 },
        .stage_host_bindings = &bindings,
        .boundary_frame_profiles = &profiles,
        .state_placement_mode = .validate_ref,
    }));

    const empty_descriptors = [_]StateDescriptor{};
    const sets = [_]state_ownership.StageStateDescriptorSet{
        .{ .stage_id = 0, .descriptors = &empty_descriptors },
        .{ .stage_id = 1, .descriptors = &empty_descriptors },
    };
    var ownership = try state_ownership.buildStageStateOwnershipPlan(allocator, .{
        .plan = &plan,
        .descriptor_sets = &sets,
    });
    defer ownership.deinit();
    var state_ref = try buildStageStatePlacementRef(allocator, &ownership);
    defer state_ref.deinit();
    var state_summary_changed = try buildPlacementPlan(allocator, .{
        .plan = &plan,
        .required_step_kinds = &.{ .prefill, .decode },
        .host_capabilities = &.{ cap0, cap1 },
        .host_residency_snapshots = &.{ res0, res1 },
        .stage_host_bindings = &bindings,
        .boundary_frame_profiles = &profiles,
        .state_placement_mode = .validate_ref,
        .state_placement_ref = &state_ref,
    });
    defer state_summary_changed.deinit();
    try std.testing.expect(!placementPlanIdEql(base.plan_id, state_summary_changed.plan_id));

    const wrong_ownership_id = state_ownership.StageStateOwnershipPlanId{ .digest = [_]u8{9} ** 32 };
    var wrong_ownership_entry = residentEntryFromStage(plan.stages[1]);
    wrong_ownership_entry.state_summary = .{
        .state_ownership_contract_version = state_ref.state_ownership_contract_version,
        .state_ownership_plan_id = wrong_ownership_id,
        .stage_id = 1,
        .descriptor_count = 0,
        .descriptors = &.{},
    };
    var wrong_ownership_res1 = try buildHostResidencySnapshot(allocator, .{
        .host_id = testHostId(2),
        .plan = &plan,
        .state_ownership_contract_version = state_ref.state_ownership_contract_version,
        .state_ownership_plan_id = wrong_ownership_id,
        .resident_stages = &.{wrong_ownership_entry},
    });
    defer wrong_ownership_res1.deinit();
    try std.testing.expectError(error.StateOwnershipPlanIdentityMismatch, buildPlacementPlan(allocator, .{
        .plan = &plan,
        .required_step_kinds = &.{ .prefill, .decode },
        .host_capabilities = &.{ cap0, cap1 },
        .host_residency_snapshots = &.{ res0, wrong_ownership_res1 },
        .stage_host_bindings = &bindings,
        .boundary_frame_profiles = &profiles,
        .state_placement_mode = .validate_ref,
        .state_placement_ref = &state_ref,
    }));

    const wrong_snapshot_identity_entry = residentEntryFromStage(plan.stages[1]);
    var wrong_snapshot_identity_res1 = try buildHostResidencySnapshot(allocator, .{
        .host_id = testHostId(2),
        .plan = &plan,
        .state_ownership_contract_version = state_ref.state_ownership_contract_version,
        .state_ownership_plan_id = wrong_ownership_id,
        .resident_stages = &.{wrong_snapshot_identity_entry},
    });
    defer wrong_snapshot_identity_res1.deinit();
    try std.testing.expectError(error.StateOwnershipPlanIdentityMismatch, buildPlacementPlan(allocator, .{
        .plan = &plan,
        .required_step_kinds = &.{ .prefill, .decode },
        .host_capabilities = &.{ cap0, cap1 },
        .host_residency_snapshots = &.{ res0, wrong_snapshot_identity_res1 },
        .stage_host_bindings = &bindings,
        .boundary_frame_profiles = &profiles,
        .state_placement_mode = .validate_ref,
        .state_placement_ref = &state_ref,
    }));

    const unexpected_descriptors = [_]StageStatePlacementDescriptorSummary{.{
        .descriptor_id = runtime_contract.kv_cache_state_id,
        .size_bytes = 64,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_kv_cache,
    }};
    var unexpected_descriptor_entry = residentEntryFromStage(plan.stages[1]);
    unexpected_descriptor_entry.state_summary = .{
        .state_ownership_contract_version = state_ref.state_ownership_contract_version,
        .state_ownership_plan_id = state_ref.state_ownership_plan_id,
        .stage_id = 1,
        .descriptor_count = unexpected_descriptors.len,
        .descriptors = &unexpected_descriptors,
    };
    var unexpected_descriptor_res1 = try buildHostResidencySnapshot(allocator, .{
        .host_id = testHostId(2),
        .plan = &plan,
        .state_ownership_contract_version = state_ref.state_ownership_contract_version,
        .state_ownership_plan_id = state_ref.state_ownership_plan_id,
        .resident_stages = &.{unexpected_descriptor_entry},
    });
    defer unexpected_descriptor_res1.deinit();
    try std.testing.expectError(error.MismatchedResidentStateDescriptorSummary, buildPlacementPlan(allocator, .{
        .plan = &plan,
        .required_step_kinds = &.{ .prefill, .decode },
        .host_capabilities = &.{ cap0, cap1 },
        .host_residency_snapshots = &.{ res0, unexpected_descriptor_res1 },
        .stage_host_bindings = &bindings,
        .boundary_frame_profiles = &profiles,
        .state_placement_mode = .validate_ref,
        .state_placement_ref = &state_ref,
    }));
}

test "inference bridge host_capability buildPlacementPlan validates same host and state placement modes" {
    const allocator = std.testing.allocator;
    const dependencies = [_]stage_plan.DependencyOverride{.{
        .source_stage_id = 0,
        .target_stage_id = 1,
        .reason = .stateful_decoder,
    }};
    var plan = try buildTestStagePlan(allocator, &.{2}, &dependencies);
    defer plan.deinit();
    var same_cap = try buildTestCapability(allocator, testHostId(1), .local_in_process, .same_host_direct);
    defer same_cap.deinit();
    var same_res = try buildTestResidency(allocator, testHostId(1), &plan, &.{ 0, 1 });
    defer same_res.deinit();
    var profiles = testProfiles(&plan, .same_host_direct);
    const bindings = [_]StageHostBinding{
        .{ .stage_id = 0, .host_id = testHostId(1) },
        .{ .stage_id = 1, .host_id = testHostId(1) },
    };
    try std.testing.expectError(error.InvalidStatePlacementMode, buildPlacementPlan(allocator, .{
        .plan = &plan,
        .required_step_kinds = &.{ .prefill, .decode },
        .host_capabilities = &.{same_cap},
        .host_residency_snapshots = &.{same_res},
        .stage_host_bindings = &bindings,
        .boundary_frame_profiles = &profiles,
        .state_placement_mode = .stateless_only,
    }));

    const descriptors = [_]StateDescriptor{.{
        .id = runtime_contract.kv_cache_state_id,
        .size_bytes = 64,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_kv_cache,
    }};
    const sets = [_]state_ownership.StageStateDescriptorSet{
        .{ .stage_id = 0, .descriptors = &descriptors },
        .{ .stage_id = 1, .descriptors = &descriptors },
    };
    const facts = [_]state_ownership.StageStatePartitionFact{.{
        .boundary_index = 0,
        .source_stage_id = 0,
        .target_stage_id = 1,
        .reason = .stateful_decoder,
        .ownership_mode = .stage_level_dependency_only,
    }};
    var ownership = try state_ownership.buildStageStateOwnershipPlan(allocator, .{
        .plan = &plan,
        .descriptor_sets = &sets,
        .partition_facts = &facts,
    });
    defer ownership.deinit();
    var state_ref = try buildStageStatePlacementRef(allocator, &ownership);
    defer state_ref.deinit();
    var resident_entries = [_]ResidentStageEntry{
        residentEntryFromStage(plan.stages[0]),
        residentEntryFromStage(plan.stages[1]),
    };
    for (&resident_entries) |*entry| {
        const summary = stateSummaryForStage(state_ref.stage_summaries, entry.stage_id).?;
        entry.state_summary = .{
            .state_ownership_contract_version = state_ref.state_ownership_contract_version,
            .state_ownership_plan_id = state_ref.state_ownership_plan_id,
            .stage_id = entry.stage_id,
            .descriptor_count = summary.descriptor_count,
            .descriptors = summary.descriptors,
        };
    }
    var state_res = try buildHostResidencySnapshot(allocator, .{
        .host_id = testHostId(1),
        .plan = &plan,
        .state_ownership_contract_version = state_ref.state_ownership_contract_version,
        .state_ownership_plan_id = state_ref.state_ownership_plan_id,
        .resident_stages = &resident_entries,
    });
    defer state_res.deinit();
    const unsupported_state_frames = testFrameCapabilities(.same_host_direct);
    var unsupported_state_cap = try buildHostCapability(allocator, .{
        .host_id = testHostId(1),
        .backend_kind = .cpu,
        .reachability_kind = .local_in_process,
        .supported_state_ownership_contract_versions = &.{state_ownership.state_ownership_contract_version + 1},
        .frame_capabilities = &unsupported_state_frames,
        .resident_checkpoint_budget_bytes = 1024,
    });
    defer unsupported_state_cap.deinit();
    try std.testing.expectError(error.UnsupportedStateOwnershipContractVersion, buildPlacementPlan(allocator, .{
        .plan = &plan,
        .required_step_kinds = &.{ .prefill, .decode },
        .host_capabilities = &.{unsupported_state_cap},
        .host_residency_snapshots = &.{state_res},
        .stage_host_bindings = &bindings,
        .boundary_frame_profiles = &profiles,
        .state_placement_mode = .validate_ref,
        .state_placement_ref = &state_ref,
    }));

    var placement = try buildPlacementPlan(allocator, .{
        .plan = &plan,
        .required_step_kinds = &.{ .prefill, .decode },
        .host_capabilities = &.{same_cap},
        .host_residency_snapshots = &.{state_res},
        .stage_host_bindings = &bindings,
        .boundary_frame_profiles = &profiles,
        .state_placement_mode = .validate_ref,
        .state_placement_ref = &state_ref,
    });
    defer placement.deinit();
    try validatePlacementPlan(&placement);

    resident_entries[1].state_summary.?.descriptors = &.{};
    try std.testing.expectError(error.MismatchedResidentStateDescriptorSummary, buildHostResidencySnapshot(allocator, .{
        .host_id = testHostId(1),
        .plan = &plan,
        .state_ownership_contract_version = state_ref.state_ownership_contract_version,
        .state_ownership_plan_id = state_ref.state_ownership_plan_id,
        .resident_stages = &resident_entries,
    }));
}
