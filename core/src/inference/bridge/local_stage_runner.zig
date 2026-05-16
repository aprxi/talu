//! Local staged boundary runner contracts.
//!
//! This module validates copied cold stage/placement/frame metadata and executes
//! one local adjacent stage boundary. It does not own backend adapters, allocate
//! on the success path, or change scheduler, backend, transport, or public error
//! behavior.

const std = @import("std");
const models = @import("models_pkg");

const host_capability = @import("host_capability.zig");
const boundary_byte_image = @import("boundary_byte_image.zig");
const local_stage_transport = @import("../transport/local_stage.zig");
const stage_transfer_mode = @import("stage_transfer_mode.zig");
const stage_transport = @import("stage_transport.zig");
const staged_error = @import("staged_error.zig");
const state_ownership = @import("state_ownership.zig");
const tensor_frame = @import("tensor_frame.zig");

const Allocator = std.mem.Allocator;
const Sha256 = std.crypto.hash.sha2.Sha256;
const stage_plan = models.stage_plan;

pub const ActivationTransportContract = struct {
    decision: stage_transfer_mode.StageTransferModeDecision,
    envelope: stage_transport.StageTransportEnvelope,
};

pub const DecodeActivationMetadataRequest = struct {
    plan_ref: *const tensor_frame.TensorFramePlanRef,
    hidden_size: usize,
    boundary_index: usize,
    dtype: tensor_frame.TensorFrameDType,
    layout: tensor_frame.TensorFrameLayout,
    location_hint: ?tensor_frame.TensorFramePayloadLocationHint,
    slot_request_ids: []const ?u64,
    slot_indices: []const usize,
    positions: []const usize,
    batch_entries: []tensor_frame.TensorFrameBatchEntry,
};

pub const PrefillActivationMetadataRequest = struct {
    plan_ref: *const tensor_frame.TensorFramePlanRef,
    hidden_size: usize,
    boundary_index: usize,
    dtype: tensor_frame.TensorFrameDType,
    layout: tensor_frame.TensorFrameLayout,
    location_hint: ?tensor_frame.TensorFramePayloadLocationHint,
    slot_request_ids: []const ?u64,
    slot_index: usize,
    sequence_start: usize,
    token_count: usize,
    batch_entries: []tensor_frame.TensorFrameBatchEntry,
};

pub const local_stage_runner_contract_version: u32 = 1;

pub const LocalStageRunnerError =
    stage_plan.StagePlanError ||
    tensor_frame.TensorFrameValidationError ||
    host_capability.PlacementError ||
    state_ownership.StateOwnershipError ||
    staged_error.StagedErrorError ||
    error{
        InvalidLocalStageRunnerContractVersion,
        LocalStageRunnerPlanFingerprintMismatch,
        StageRunnerPlanIdentityMismatch,
        MissingStageRef,
        DuplicateStageRef,
        MissingBoundaryRef,
        DuplicateBoundaryRef,
        BoundaryIndexOutOfRange,
        MissingBoundaryFrameProfile,
        BoundaryFrameProfileMismatch,
        UnsupportedRemoteBoundary,
        InvalidLocalHandoffMode,
        InvalidStageRange,
        InvalidStepRequest,
        InvalidRequestId,
        InvalidSlotId,
        MissingFailureScratch,
        UnknownRunnerFailure,
    };

pub const LocalStageRunnerPlanId = struct {
    digest: [32]u8,
};

pub const LocalStageRunnerStageRef = struct {
    stage_id: usize,
    host_id: host_capability.HostId,
    layer_start: usize,
    layer_end: usize,
    owned_roles: [models.manifest.role_count]bool,
};

pub const LocalStageRunnerBoundaryProfile = struct {
    step_kind: tensor_frame.TensorFrameStepKind,
    dtype: tensor_frame.TensorFrameDType,
    layout: tensor_frame.TensorFrameLayout,
    handoff_mode: host_capability.BoundaryHandoffMode,
    max_batch_entries: u64,
    max_token_count_per_frame: u64,
    max_activation_payload_bytes: u64,
};

pub const LocalStageRunnerBoundaryRef = struct {
    boundary_index: usize,
    source_stage_id: usize,
    target_stage_id: usize,
    source_host_id: host_capability.HostId,
    target_host_id: host_capability.HostId,
    producer_layer_start: usize,
    producer_layer_end: usize,
    consumer_layer_start: usize,
    consumer_layer_end: usize,
    profiles: []const LocalStageRunnerBoundaryProfile,
};

pub const LocalStageRunnerPlanRef = struct {
    arena: std.heap.ArenaAllocator,
    version: u32,
    plan_id: LocalStageRunnerPlanId,
    tensor_frame_plan_identity: tensor_frame.TensorFramePlanIdentity,
    graph_digest: [32]u8,
    graph_contract_version: u32,
    stage_plan_contract_version: u32,
    stage_plan_id: stage_plan.StagePlanId,
    tensor_frame_contract_version: u32,
    placement_contract_version: u32,
    placement_plan_id: host_capability.PlacementPlanId,
    state_ownership_contract_version: ?u32,
    state_ownership_plan_id: ?state_ownership.StageStateOwnershipPlanId,
    stages: []const LocalStageRunnerStageRef,
    boundaries: []const LocalStageRunnerBoundaryRef,
    tensor_frame_boundaries: []const tensor_frame.TensorFrameBoundaryRef,
    required_step_kinds: []const tensor_frame.TensorFrameStepKind,

    pub fn deinit(self: *LocalStageRunnerPlanRef) void {
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn boundary(
        self: *const LocalStageRunnerPlanRef,
        boundary_index: usize,
    ) LocalStageRunnerError!*const LocalStageRunnerBoundaryRef {
        for (self.boundaries) |*entry| {
            if (entry.boundary_index == boundary_index) return entry;
        }
        return error.BoundaryIndexOutOfRange;
    }
};

pub const LocalStageRunnerPlanRequest = struct {
    stage_plan: *const stage_plan.StagePlan,
    tensor_frame_plan_ref: *const tensor_frame.TensorFramePlanRef,
    placement_plan: *const host_capability.PlacementPlan,
    state_ownership_plan: ?*const state_ownership.StageStateOwnershipPlan = null,
};

pub fn buildLocalStageRunnerPlanRef(
    allocator: Allocator,
    request: LocalStageRunnerPlanRequest,
) LocalStageRunnerError!LocalStageRunnerPlanRef {
    try validatePlanInputs(request);

    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_allocator = arena.allocator();

    const stages = try copyStageRefs(arena_allocator, request.stage_plan, request.placement_plan);
    const tensor_boundaries = try arena_allocator.dupe(tensor_frame.TensorFrameBoundaryRef, request.tensor_frame_plan_ref.boundaries);
    const required_step_kinds = try arena_allocator.dupe(tensor_frame.TensorFrameStepKind, request.placement_plan.required_step_kinds);
    const boundaries = try copyBoundaryRefs(arena_allocator, request.stage_plan, request.placement_plan, required_step_kinds);

    var plan_ref = LocalStageRunnerPlanRef{
        .arena = arena,
        .version = local_stage_runner_contract_version,
        .plan_id = undefined,
        .tensor_frame_plan_identity = request.tensor_frame_plan_ref.identity,
        .graph_digest = request.stage_plan.graph_identity.digest,
        .graph_contract_version = request.stage_plan.graph_identity.graph_contract_version,
        .stage_plan_contract_version = request.stage_plan.stage_contract_version,
        .stage_plan_id = request.stage_plan.plan_id,
        .tensor_frame_contract_version = tensor_frame.tensor_frame_contract_version,
        .placement_contract_version = request.placement_plan.version,
        .placement_plan_id = request.placement_plan.plan_id,
        .state_ownership_contract_version = request.placement_plan.state_ownership_contract_version,
        .state_ownership_plan_id = request.placement_plan.state_ownership_plan_id,
        .stages = stages,
        .boundaries = boundaries,
        .tensor_frame_boundaries = tensor_boundaries,
        .required_step_kinds = required_step_kinds,
    };
    plan_ref.plan_id = computeLocalStageRunnerPlanId(&plan_ref);
    try validateLocalStageRunnerPlanRef(&plan_ref);
    return plan_ref;
}

pub fn validateLocalStageRunnerPlanRef(
    plan_ref: *const LocalStageRunnerPlanRef,
) LocalStageRunnerError!void {
    if (plan_ref.version != local_stage_runner_contract_version) {
        return error.InvalidLocalStageRunnerContractVersion;
    }
    if (plan_ref.tensor_frame_contract_version != tensor_frame.tensor_frame_contract_version) {
        return error.InvalidTensorFrameContractVersion;
    }
    if (plan_ref.placement_contract_version != host_capability.placement_contract_version) {
        return error.InvalidPlacementContractVersion;
    }
    if (!std.mem.eql(u8, &plan_ref.tensor_frame_plan_identity.graph_digest, &plan_ref.graph_digest) or
        plan_ref.tensor_frame_plan_identity.graph_contract_version != plan_ref.graph_contract_version)
    {
        return error.StageRunnerPlanIdentityMismatch;
    }
    if (plan_ref.tensor_frame_plan_identity.stage_plan_contract_version != plan_ref.stage_plan_contract_version or
        !std.mem.eql(u8, &plan_ref.tensor_frame_plan_identity.stage_plan_id.digest, &plan_ref.stage_plan_id.digest))
    {
        return error.StageRunnerPlanIdentityMismatch;
    }
    try validateRequiredStepKinds(plan_ref.required_step_kinds);
    try validateCopiedStageRefs(plan_ref.stages);
    try validateCopiedBoundaryRefs(plan_ref);
    if ((plan_ref.state_ownership_contract_version == null) != (plan_ref.state_ownership_plan_id == null)) {
        return error.StateOwnershipPlanIdentityMismatch;
    }
    const expected = computeLocalStageRunnerPlanId(plan_ref);
    if (!localStageRunnerPlanIdEql(plan_ref.plan_id, expected)) {
        return error.LocalStageRunnerPlanFingerprintMismatch;
    }
}

pub fn localStageRunnerPlanIdEql(
    lhs: LocalStageRunnerPlanId,
    rhs: LocalStageRunnerPlanId,
) bool {
    return std.mem.eql(u8, &lhs.digest, &rhs.digest);
}

pub fn buildDecodeActivationMetadata(
    request: DecodeActivationMetadataRequest,
) anyerror!tensor_frame.TensorFrameMetadata {
    if (request.slot_indices.len == 0 or request.slot_indices.len != request.positions.len) return error.InvalidArgument;
    if (request.batch_entries.len < request.slot_indices.len) return error.InvalidArgument;

    const contract = try tensor_frame.selectedBoundaryTensorContract(
        request.plan_ref,
        request.boundary_index,
        request.dtype,
        request.layout,
        .negotiated,
    );
    const hidden_size = try usizeToU64(request.hidden_size, error.InvalidHiddenSize);
    const batch_count = try usizeToU64(request.slot_indices.len, error.InvalidBatch);
    const tensor_desc = try tensor_frame.TensorFrameTensorDesc.contiguousActivation(
        request.dtype,
        .{ batch_count, 1, hidden_size, 0 },
    );

    for (request.slot_indices, request.positions, 0..) |slot_index, position, row_index| {
        const slot_id_value = try slotId(slot_index);
        request.batch_entries[row_index] = .{
            .batch_index = std.math.cast(u32, row_index) orelse return error.InvalidBatch,
            .request_id = try requestIdForSlot(request.slot_request_ids, slot_index),
            .slot_id = slot_id_value,
            .sequence_start = try usizeToU64(position, error.InvalidSequenceRange),
            .token_count = 1,
        };
    }

    const first_entry = request.batch_entries[0];
    const args = tensor_frame.ActivationFrameArgs{
        .frame_id = try activationFrameId(
            request.boundary_index,
            first_entry.slot_id,
            first_entry.sequence_start,
            first_entry.token_count,
            request.slot_indices.len,
        ),
        .plan_ref = request.plan_ref,
        .boundary_index = request.boundary_index,
        .selected_contract = &contract,
        .shape_context = .{
            .expected_hidden_size = hidden_size,
            .expected_step_kind = .decode,
        },
        .tensor = tensor_desc,
        .batch = .{ .entries = request.batch_entries[0..request.slot_indices.len] },
        .payload = .{
            .byte_count = tensor_desc.payload_byte_count,
            .location_hint = request.location_hint,
            .ownership = .borrowed_until_next_stage_call,
            .lifetime = .step_scoped,
        },
    };
    return tensor_frame.activationDecodeFrame(args);
}

pub fn buildPrefillActivationMetadata(
    request: PrefillActivationMetadataRequest,
) anyerror!tensor_frame.TensorFrameMetadata {
    if (request.token_count == 0 or request.batch_entries.len == 0) return error.InvalidArgument;

    const contract = try tensor_frame.selectedBoundaryTensorContract(
        request.plan_ref,
        request.boundary_index,
        request.dtype,
        request.layout,
        .negotiated,
    );
    const hidden_size = try usizeToU64(request.hidden_size, error.InvalidHiddenSize);
    const token_count_u64 = try usizeToU64(request.token_count, error.InvalidSequenceRange);
    const tensor_desc = try tensor_frame.TensorFrameTensorDesc.contiguousActivation(
        request.dtype,
        .{ 1, token_count_u64, hidden_size, 0 },
    );
    const slot_id_value = try slotId(request.slot_index);
    const sequence_start_u64 = try usizeToU64(request.sequence_start, error.InvalidSequenceRange);
    request.batch_entries[0] = .{
        .batch_index = 0,
        .request_id = try requestIdForSlot(request.slot_request_ids, request.slot_index),
        .slot_id = slot_id_value,
        .sequence_start = sequence_start_u64,
        .token_count = token_count_u64,
    };

    const args = tensor_frame.ActivationFrameArgs{
        .frame_id = try activationFrameId(
            request.boundary_index,
            slot_id_value,
            sequence_start_u64,
            token_count_u64,
            1,
        ),
        .plan_ref = request.plan_ref,
        .boundary_index = request.boundary_index,
        .selected_contract = &contract,
        .shape_context = .{
            .expected_hidden_size = hidden_size,
            .expected_step_kind = .prefill,
        },
        .tensor = tensor_desc,
        .batch = .{ .entries = request.batch_entries[0..1] },
        .payload = .{
            .byte_count = tensor_desc.payload_byte_count,
            .location_hint = request.location_hint,
            .ownership = .borrowed_until_next_stage_call,
            .lifetime = .step_scoped,
        },
    };
    return tensor_frame.activationPrefillFrame(args);
}

pub fn hostActivationByteImage(
    metadata: *const tensor_frame.TensorFrameMetadata,
    host_bytes: []const u8,
) boundary_byte_image.BoundaryByteImageRef {
    return .{
        .metadata = metadata,
        .byte_count = metadata.payload.byte_count,
        .host_bytes = host_bytes,
        .location_hint = metadata.payload.location_hint,
        .readiness = .host_readable_now,
        .ownership = metadata.payload.ownership,
        .lifetime = metadata.payload.lifetime,
    };
}

pub fn segmentedHostActivationByteImage(
    metadata: *const tensor_frame.TensorFrameMetadata,
    host_segments: []const []const u8,
) boundary_byte_image.BoundaryByteImageRef {
    return .{
        .metadata = metadata,
        .byte_count = metadata.payload.byte_count,
        .host_segments = host_segments,
        .location_hint = metadata.payload.location_hint,
        .readiness = .host_readable_now,
        .ownership = metadata.payload.ownership,
        .lifetime = metadata.payload.lifetime,
    };
}

pub fn deviceActivationByteImage(
    metadata: *const tensor_frame.TensorFrameMetadata,
) boundary_byte_image.BoundaryByteImageRef {
    return .{
        .metadata = metadata,
        .byte_count = metadata.payload.byte_count,
        .location_hint = metadata.payload.location_hint,
        .readiness = .device_download_required,
        .ownership = metadata.payload.ownership,
        .lifetime = metadata.payload.lifetime,
    };
}

pub fn buildActivationTransportContract(
    placement_plan: *const host_capability.PlacementPlan,
    metadata: *const tensor_frame.TensorFrameMetadata,
    image: *const boundary_byte_image.BoundaryByteImageRef,
    allow_borrow: bool,
    local_device_peer_copy_available: bool,
) anyerror!ActivationTransportContract {
    const decision = try stage_transfer_mode.chooseStageTransferMode(.{
        .placement_plan = placement_plan,
        .metadata = metadata,
        .image = image,
        .allow_borrow = allow_borrow,
        .local_device_peer_copy_available = local_device_peer_copy_available,
    });
    const envelope = try stage_transport.buildStageTransportActivationEnvelope(.{
        .metadata = metadata,
        .image = image,
        .decision = decision,
    });
    return .{
        .decision = decision,
        .envelope = envelope,
    };
}

pub const LocalStageBoundaryStep = struct {
    boundary_index: usize,
    step_kind: tensor_frame.TensorFrameStepKind,
    metadata: tensor_frame.TensorFrameMetadata,
    activation_byte_count: usize,
    observer: tensor_frame.TensorFrameObserver = .{},
    observer_mode: tensor_frame.TensorFrameObserverMode = .best_effort,
    expected_request_id: ?u64 = null,
    expected_slot_id: ?u64 = null,
};

pub const LocalStageTouchedRef = struct {
    stage_id: usize,
    host_id: host_capability.HostId,
    request_id: u64,
    slot_id: u64,
    state_epoch: ?u64 = null,
    execution_started: bool = false,
    receiver_payload_mutation_started: bool = false,
};

pub const LocalStageRunSuccess = struct {
    boundary_index: usize,
    request_id: u64,
    slot_id: u64,
};

pub const LocalStageFailureReport = struct {
    primary_failure: staged_error.StagedFailure,
    source_error: ?anyerror,
    touched_stages: []const LocalStageTouchedRef,
};

pub const LocalStageRunResult = union(enum) {
    success: LocalStageRunSuccess,
    failure: LocalStageFailureReport,
};

pub const LocalStageBoundaryRunRequest = struct {
    plan_ref: *const LocalStageRunnerPlanRef,
    placement_plan: *const host_capability.PlacementPlan,
    step: LocalStageBoundaryStep,
    host_staging: ?[]align(64) u8,
    stage0_input: []const u8 = &.{},
    stage1_input: []const u8 = &.{},
    touched_stage_scratch: []LocalStageTouchedRef,
};

pub const LocalStageEndpointVTable = struct {
    execute_decode_layer_range: *const fn (*anyopaque, []const u8, usize, usize) anyerror!void,
    execute_prefill_layer_range: *const fn (*anyopaque, []const u8, usize, usize) anyerror!void,
    project_final_logits: ?*const fn (*anyopaque) anyerror!void = null,
};

pub const LocalStageEndpoint = struct {
    stage_id: usize,
    ptr: *anyopaque,
    vtable: *const LocalStageEndpointVTable,
    transport_endpoint: ?local_stage_transport.LocalStageTransportEndpoint = null,

    pub fn executeLayers(self: *@This(), input: []const u8, layer_start: usize, layer_end: usize) anyerror!local_stage_transport.StageExecutionReceipt {
        return self.executeDecodeLayerRange(input, layer_start, layer_end);
    }

    pub fn executeDecodeLayerRange(self: *@This(), input: []const u8, layer_start: usize, layer_end: usize) anyerror!local_stage_transport.StageExecutionReceipt {
        try self.vtable.execute_decode_layer_range(self.ptr, input, layer_start, layer_end);
        return local_stage_transport.StageExecutionReceipt.completed(self.stage_id);
    }

    pub fn executePrefillLayerRange(self: *@This(), input: []const u8, layer_start: usize, layer_end: usize) anyerror!local_stage_transport.StageExecutionReceipt {
        try self.vtable.execute_prefill_layer_range(self.ptr, input, layer_start, layer_end);
        return local_stage_transport.StageExecutionReceipt.completed(self.stage_id);
    }

    pub fn projectFinalLogits(self: *@This()) anyerror!void {
        const project = self.vtable.project_final_logits orelse return;
        return project(self.ptr);
    }
};

pub const LocalStageChainStageVTable = LocalStageEndpointVTable;
pub const LocalStageChainStage = LocalStageEndpoint;

pub const LocalStageEndpointRegistry = struct {
    endpoints: []LocalStageEndpoint,
    transport_endpoints: []local_stage_transport.LocalStageTransportEndpoint = &.{},

    pub fn endpointForStageId(self: *@This(), stage_id: usize) LocalStageRunnerError!*LocalStageEndpoint {
        var found: ?*LocalStageEndpoint = null;
        for (self.endpoints) |*endpoint| {
            if (endpoint.stage_id != stage_id) continue;
            if (found != null) return error.DuplicateStageRef;
            found = endpoint;
        }
        return found orelse error.MissingStageRef;
    }

    pub fn endpointForStageIdConst(self: *const @This(), stage_id: usize) LocalStageRunnerError!*const LocalStageEndpoint {
        var found: ?*const LocalStageEndpoint = null;
        for (self.endpoints) |*endpoint| {
            if (endpoint.stage_id != stage_id) continue;
            if (found != null) return error.DuplicateStageRef;
            found = endpoint;
        }
        return found orelse error.MissingStageRef;
    }

    pub fn transportEndpointForStageId(self: *@This(), stage_id: usize) LocalStageRunnerError!*local_stage_transport.LocalStageTransportEndpoint {
        if (self.transport_endpoints.len != 0) {
            var transport_registry = local_stage_transport.LocalStageTransportEndpointRegistry{
                .endpoints = self.transport_endpoints,
            };
            return transport_registry.endpointForStageId(stage_id);
        }
        const endpoint = try self.endpointForStageId(stage_id);
        if (endpoint.transport_endpoint) |*transport_endpoint| return transport_endpoint;
        return error.MissingStageRef;
    }
};

pub fn localStageAdapter(comptime Stage: type, stage_id: usize, stage: *Stage) LocalStageEndpoint {
    const Adapter = struct {
        fn stagePtr(ptr: *anyopaque) *Stage {
            return @ptrCast(@alignCast(ptr));
        }

        fn executeDecodeLayerRange(ptr: *anyopaque, input: []const u8, layer_start: usize, layer_end: usize) anyerror!void {
            if (comptime @hasDecl(Stage, "executeDecodeLayerRange")) {
                return stagePtr(ptr).executeDecodeLayerRange(input, layer_start, layer_end);
            }
            return stagePtr(ptr).executeLayers(input, layer_start, layer_end);
        }

        fn executePrefillLayerRange(ptr: *anyopaque, input: []const u8, layer_start: usize, layer_end: usize) anyerror!void {
            if (comptime @hasDecl(Stage, "executePrefillLayerRange")) {
                return stagePtr(ptr).executePrefillLayerRange(input, layer_start, layer_end);
            }
            return stagePtr(ptr).executeLayers(input, layer_start, layer_end);
        }

        fn projectFinalLogits(ptr: *anyopaque) anyerror!void {
            if (comptime @hasDecl(Stage, "projectFinalLogits")) {
                return stagePtr(ptr).projectFinalLogits();
            }
        }

        const vtable = LocalStageEndpointVTable{
            .execute_decode_layer_range = executeDecodeLayerRange,
            .execute_prefill_layer_range = executePrefillLayerRange,
            .project_final_logits = if (@hasDecl(Stage, "projectFinalLogits")) projectFinalLogits else null,
        };
    };
    return .{
        .stage_id = stage_id,
        .ptr = stage,
        .vtable = &Adapter.vtable,
        .transport_endpoint = local_stage_transport.localStageTransportAdapter(Stage, stage_id, stage),
    };
}

pub const LocalStageChainBoundaryStep = struct {
    boundary_index: usize,
    step_kind: tensor_frame.TensorFrameStepKind,
    metadata: *const tensor_frame.TensorFrameMetadata,
    image: *const boundary_byte_image.BoundaryByteImageRef,
    staging: ?[]align(64) u8 = null,
    allow_borrow: bool = false,
    local_device_peer_copy_available: bool = false,
};

pub const LocalStageChainRequest = struct {
    allocator: ?Allocator = null,
    plan_ref: *const LocalStageRunnerPlanRef,
    placement_plan: *const host_capability.PlacementPlan,
    state_ownership_plan: ?*const state_ownership.StageStateOwnershipPlan = null,
    cleanup_obligations: []const state_ownership.StateCleanupObligation = &.{},
    stages: []LocalStageChainStage,
    boundaries: []const LocalStageChainBoundaryStep,
    stage_inputs: []const []const u8 = &.{},
    project_final_logits: bool = false,
};

pub const LocalStageEndpointRegistryChainRequest = struct {
    allocator: ?Allocator = null,
    plan_ref: *const LocalStageRunnerPlanRef,
    placement_plan: *const host_capability.PlacementPlan,
    state_ownership_plan: ?*const state_ownership.StageStateOwnershipPlan = null,
    cleanup_obligations: []const state_ownership.StateCleanupObligation = &.{},
    registry: LocalStageEndpointRegistry,
    boundaries: []const LocalStageChainBoundaryStep,
    stage_inputs: []const []const u8 = &.{},
    project_final_logits: bool = false,
};

pub fn executeLocalStageChain(request: LocalStageChainRequest) anyerror!void {
    return executeLocalStageEndpointRegistryChain(.{
        .allocator = request.allocator,
        .plan_ref = request.plan_ref,
        .placement_plan = request.placement_plan,
        .state_ownership_plan = request.state_ownership_plan,
        .cleanup_obligations = request.cleanup_obligations,
        .registry = .{ .endpoints = request.stages },
        .boundaries = request.boundaries,
        .stage_inputs = request.stage_inputs,
        .project_final_logits = request.project_final_logits,
    });
}

pub fn executeLocalStageEndpointRegistryChain(request: LocalStageEndpointRegistryChainRequest) anyerror!void {
    try validateLocalStageEndpointRegistryChainRequestShape(request);
    var registry = request.registry;

    for (request.boundaries, 0..) |step, boundary_position| {
        const boundary = try request.plan_ref.boundary(step.boundary_index);
        _ = try registry.endpointForStageId(boundary.source_stage_id);
        _ = try registry.endpointForStageId(boundary.target_stage_id);
        _ = try registry.transportEndpointForStageId(boundary.source_stage_id);
        _ = try registry.transportEndpointForStageId(boundary.target_stage_id);
        try validateLocalStageChainBoundaryStatic(request, boundary, step, boundary_position);
        try validateLocalStageChainBoundaryImageBeforeMutation(step);
        if (step.image.host_segments == null) {
            _ = try buildLocalStageChainTransportEnvelope(request, step);
        }
    }

    var current_source_receipt: ?local_stage_transport.StageExecutionReceipt = null;
    for (request.boundaries, 0..) |step, boundary_position| {
        const boundary = try request.plan_ref.boundary(step.boundary_index);
        const source_stage = try registry.endpointForStageId(boundary.source_stage_id);
        const target_stage = try registry.endpointForStageId(boundary.target_stage_id);
        const source_transport = try registry.transportEndpointForStageId(boundary.source_stage_id);
        const target_transport = try registry.transportEndpointForStageId(boundary.target_stage_id);
        if (boundary_position == 0) {
            current_source_receipt = try executeEndpointForStep(
                source_stage,
                step.step_kind,
                stageInput(request, 0),
                boundary.producer_layer_start,
                boundary.producer_layer_end,
            );
        }
        const source_receipt = current_source_receipt orelse return error.StageRunnerPlanIdentityMismatch;
        if (source_receipt.stage_id != boundary.source_stage_id) return error.StageRunnerPlanIdentityMismatch;

        try validateLocalStageChainBoundaryImage(step);
        const contract = try buildLocalStageChainTransportEnvelope(request, step);
        const transport_request = local_stage_transport.LocalStageTransportRequest{
            .placement_plan = request.placement_plan,
            .metadata = step.metadata,
            .image = step.image,
            .decision = contract.decision,
            .envelope = &contract.envelope,
            .source_receipt = source_receipt,
            .staging = step.staging,
            .allow_borrow = step.allow_borrow,
            .local_device_peer_copy_available = step.local_device_peer_copy_available,
            .state_ownership_plan = request.state_ownership_plan,
            .cleanup_obligations = request.cleanup_obligations,
        };

        if (request.allocator) |allocator| {
            var failure_capture = local_stage_transport.LocalStageTransportFailureCapture.init(allocator);
            defer failure_capture.deinit();
            try local_stage_transport.executeLocalStageTransportWithFailureCapture(
                local_stage_transport.LocalStageTransportEndpoint,
                local_stage_transport.LocalStageTransportEndpoint,
                source_transport,
                target_transport,
                transport_request,
                &failure_capture,
            );
        } else {
            try local_stage_transport.executeLocalStageTransport(
                local_stage_transport.LocalStageTransportEndpoint,
                local_stage_transport.LocalStageTransportEndpoint,
                source_transport,
                target_transport,
                transport_request,
            );
        }

        current_source_receipt = try executeEndpointForStep(
            target_stage,
            step.step_kind,
            stageInput(request, boundary_position + 1),
            boundary.consumer_layer_start,
            boundary.consumer_layer_end,
        );
    }

    if (request.project_final_logits) {
        const final_stage_ref = request.plan_ref.stages[request.plan_ref.stages.len - 1];
        const final_endpoint = try registry.endpointForStageId(final_stage_ref.stage_id);
        try final_endpoint.projectFinalLogits();
    }
}

fn buildLocalStageChainTransportEnvelope(
    request: LocalStageEndpointRegistryChainRequest,
    step: LocalStageChainBoundaryStep,
) anyerror!ActivationTransportContract {
    const decision = try stage_transfer_mode.chooseStageTransferMode(.{
        .placement_plan = request.placement_plan,
        .metadata = step.metadata,
        .image = step.image,
        .allow_borrow = step.allow_borrow,
        .local_device_peer_copy_available = step.local_device_peer_copy_available,
    });
    const envelope = try stage_transport.buildStageTransportActivationEnvelope(.{
        .metadata = step.metadata,
        .image = step.image,
        .decision = decision,
    });
    return .{
        .decision = decision,
        .envelope = envelope,
    };
}

// Private regression harness for the pre-chain one-boundary runner contract.
// Production bridge users must enter through executeLocalStageChain so ordered
// local topology execution has one bridge path.
fn executeLocalStageBoundary(
    comptime SourceStageType: type,
    comptime TargetStageType: type,
    source_stage: SourceStageType,
    target_stage: TargetStageType,
    request: LocalStageBoundaryRunRequest,
) LocalStageRunnerError!LocalStageRunResult {
    comptime {
        assertStageContract(SourceStageType);
        assertStageContract(TargetStageType);
    }

    const tensor_frame_id = request.step.metadata.frame_id;
    if (validateLocalStageRunnerPlanRef(request.plan_ref)) |_| {} else |err| {
        return runnerFailure(request.plan_ref, null, request.step.boundary_index, null, tensor_frame_id, err, &.{});
    }
    const boundary = request.plan_ref.boundary(request.step.boundary_index) catch |err| {
        return runnerFailure(request.plan_ref, null, request.step.boundary_index, null, tensor_frame_id, err, &.{});
    };
    const profile = profileForStep(boundary, request.step.step_kind) orelse {
        return runnerFailure(request.plan_ref, boundary, request.step.boundary_index, null, tensor_frame_id, error.MissingBoundaryFrameProfile, &.{});
    };
    switch (profile.handoff_mode) {
        .local_in_process, .mock => {},
        .same_host_direct, .remote_declared => {
            return runnerFailure(request.plan_ref, boundary, request.step.boundary_index, null, tensor_frame_id, error.InvalidLocalHandoffMode, &.{});
        },
    }

    if (validateTensorFrameAgainstRunnerPlan(request.plan_ref, &request.step.metadata, boundary.boundary_index)) |_| {} else |err| {
        return tensorFailure(request.plan_ref, boundary, null, tensor_frame_id, err, &.{});
    }
    if (tensor_frame.validatePayloadBufferLength(&request.step.metadata, request.step.activation_byte_count)) |_| {} else |err| {
        return tensorFailure(request.plan_ref, boundary, null, tensor_frame_id, err, &.{});
    }
    if (validateTensorFrameAgainstBoundaryProfile(&request.step.metadata, profile)) |_| {} else |err| {
        return tensorFailure(request.plan_ref, boundary, null, tensor_frame_id, err, &.{});
    }
    const batch_entry = singleBatchEntry(&request.step.metadata) catch |err| {
        return tensorFailure(request.plan_ref, boundary, null, tensor_frame_id, err, &.{});
    };
    if (request.step.expected_request_id) |expected| {
        if (batch_entry.request_id != expected) {
            return runnerFailure(request.plan_ref, boundary, boundary.boundary_index, batch_entry, tensor_frame_id, error.InvalidRequestId, &.{});
        }
    }
    if (request.step.expected_slot_id) |expected| {
        if (batch_entry.slot_id != expected) {
            return runnerFailure(request.plan_ref, boundary, boundary.boundary_index, batch_entry, tensor_frame_id, error.InvalidSlotId, &.{});
        }
    }

    const staging = request.host_staging orelse {
        return transportFailure(request.plan_ref, boundary, batch_entry, tensor_frame_id, error.LocalStageTransferNotInitialized, &.{}, .validation_before_mutation);
    };
    if (request.step.activation_byte_count > staging.len) {
        return transportFailure(request.plan_ref, boundary, batch_entry, tensor_frame_id, error.LocalStageTransferBufferTooSmall, &.{}, .validation_before_mutation);
    }
    if (request.touched_stage_scratch.len < 2) return error.MissingFailureScratch;

    var source = source_stage;
    var target = target_stage;
    var touched_len: usize = 0;
    request.touched_stage_scratch[0] = touchedRef(boundary.source_stage_id, boundary.source_host_id, batch_entry, .{
        .execution_started = true,
    });
    touched_len = 1;
    if (source.executeLayers(request.stage0_input, boundary.producer_layer_start, boundary.producer_layer_end)) |_| {} else |err| {
        return stageFailure(request.plan_ref, boundary, batch_entry, tensor_frame_id, boundary.source_stage_id, boundary.source_host_id, err, request.touched_stage_scratch[0..touched_len]);
    }

    var transport_metadata = request.step.metadata;
    transport_metadata.payload.location_hint = .{ .cuda = 0 };
    var image = deviceActivationByteImage(&transport_metadata);
    const decision = stage_transfer_mode.chooseStageTransferMode(.{
        .placement_plan = request.placement_plan,
        .metadata = &transport_metadata,
        .image = &image,
        .allow_borrow = false,
    }) catch |err| {
        return transportFailure(request.plan_ref, boundary, batch_entry, tensor_frame_id, err, request.touched_stage_scratch[0..touched_len], .validation_before_mutation);
    };
    const envelope = stage_transport.buildStageTransportActivationEnvelope(.{
        .metadata = &transport_metadata,
        .image = &image,
        .decision = decision,
    }) catch |err| {
        return transportFailure(request.plan_ref, boundary, batch_entry, tensor_frame_id, err, request.touched_stage_scratch[0..touched_len], .validation_before_mutation);
    };
    var source_transport = local_stage_transport.localStageTransportAdapter(SourceStageType, boundary.source_stage_id, &source);
    var target_transport = local_stage_transport.localStageTransportAdapter(TargetStageType, boundary.target_stage_id, &target);
    var failure_capture = local_stage_transport.LocalStageTransportFailureCapture.init(std.heap.page_allocator);
    defer failure_capture.deinit();
    local_stage_transport.executeLocalStageTransportWithFailureCapture(
        local_stage_transport.LocalStageTransportEndpoint,
        local_stage_transport.LocalStageTransportEndpoint,
        &source_transport,
        &target_transport,
        .{
            .placement_plan = request.placement_plan,
            .metadata = &transport_metadata,
            .image = &image,
            .decision = decision,
            .envelope = &envelope,
            .source_receipt = local_stage_transport.StageExecutionReceipt.completed(boundary.source_stage_id),
            .staging = staging,
            .allow_borrow = false,
        },
        &failure_capture,
    ) catch |err| {
        touched_len = localBoundaryTouchedFromTransportFailure(request, boundary, batch_entry, &failure_capture);
        return transportFailure(request.plan_ref, boundary, batch_entry, tensor_frame_id, err, request.touched_stage_scratch[0..touched_len], .frame_handoff);
    };

    request.touched_stage_scratch[1] = touchedRef(boundary.target_stage_id, boundary.target_host_id, batch_entry, .{
        .receiver_payload_mutation_started = true,
    });
    touched_len = 2;
    if (tensor_frame.emitTensorFrame(request.step.observer, request.step.observer_mode, &request.step.metadata)) |_| {} else |err| {
        return observerFailure(request.plan_ref, boundary, batch_entry, tensor_frame_id, err, request.touched_stage_scratch[0..touched_len]);
    }
    request.touched_stage_scratch[1].execution_started = true;
    if (target.executeLayers(request.stage1_input, boundary.consumer_layer_start, boundary.consumer_layer_end)) |_| {} else |err| {
        return stageFailure(request.plan_ref, boundary, batch_entry, tensor_frame_id, boundary.target_stage_id, boundary.target_host_id, err, request.touched_stage_scratch[0..touched_len]);
    }

    return .{ .success = .{
        .boundary_index = boundary.boundary_index,
        .request_id = batch_entry.request_id,
        .slot_id = batch_entry.slot_id,
    } };
}

fn localBoundaryTouchedFromTransportFailure(
    request: LocalStageBoundaryRunRequest,
    boundary: *const LocalStageRunnerBoundaryRef,
    batch_entry: tensor_frame.TensorFrameBatchEntry,
    failure_capture: *const local_stage_transport.LocalStageTransportFailureCapture,
) usize {
    var touched_len: usize = 1;
    if (failure_capture.report) |report| {
        if (report.entries.len != 0) {
            touched_len = @min(report.entries[0].touched_stages.len, @as(usize, 2));
        }
    }
    if (touched_len == 0) return 0;
    request.touched_stage_scratch[0] = touchedRef(boundary.source_stage_id, boundary.source_host_id, batch_entry, .{
        .execution_started = true,
    });
    if (touched_len >= 2) {
        request.touched_stage_scratch[1] = touchedRef(boundary.target_stage_id, boundary.target_host_id, batch_entry, .{
            .receiver_payload_mutation_started = true,
        });
    }
    return touched_len;
}

pub const LocalStageExecutionFailureEntry = struct {
    primary_failure: staged_error.StagedFailure,
    touched_stages: []const staged_error.TouchedStageCleanupRef,
    cleanup_plan: ?staged_error.StagedCleanupPlan = null,
    cleanup_report: ?staged_error.StagedCleanupReport = null,
    error_report: staged_error.StagedErrorReport,
};

pub const LocalStageExecutionFailureReport = struct {
    allocator: Allocator,
    source_error: anyerror,
    entries: []LocalStageExecutionFailureEntry,

    pub fn deinit(self: *@This()) void {
        for (self.entries) |*entry| {
            entry.error_report.deinit();
            if (entry.cleanup_report) |*report| report.deinit();
            if (entry.cleanup_plan) |*plan| plan.deinit();
            self.allocator.free(entry.touched_stages);
        }
        self.allocator.free(self.entries);
        self.* = undefined;
    }
};

pub const LocalStageExecutionFailureRequest = struct {
    placement_plan: *const host_capability.PlacementPlan,
    state_ownership_plan: ?*const state_ownership.StageStateOwnershipPlan = null,
    cleanup_obligations: []const state_ownership.StateCleanupObligation = &.{},
    metadata: *const tensor_frame.TensorFrameMetadata,
    active_stage_id: usize,
    source_error: anyerror,
};

pub fn captureLocalStageExecutionFailure(
    allocator: Allocator,
    request: LocalStageExecutionFailureRequest,
) !LocalStageExecutionFailureReport {
    const entries = try allocator.alloc(LocalStageExecutionFailureEntry, request.metadata.batch.entries.len);
    errdefer allocator.free(entries);

    for (request.metadata.batch.entries, 0..) |batch_entry, index| {
        entries[index] = try buildLocalStageExecutionFailureEntry(allocator, request, batch_entry);
        errdefer deinitLocalStageExecutionFailureEntry(allocator, &entries[index]);
    }

    return .{
        .allocator = allocator,
        .source_error = request.source_error,
        .entries = entries,
    };
}

pub fn preserveLocalStageExecutionError(
    allocator: ?Allocator,
    request: LocalStageExecutionFailureRequest,
) anyerror {
    if (allocator) |alloc| {
        var report = captureLocalStageExecutionFailure(alloc, request) catch return request.source_error;
        report.deinit();
    }
    return request.source_error;
}

pub fn executeLocalStageLayers(
    allocator: ?Allocator,
    state_ownership_plan: ?*const state_ownership.StageStateOwnershipPlan,
    placement_plan: *const host_capability.PlacementPlan,
    metadata: *const tensor_frame.TensorFrameMetadata,
    active_stage_id: usize,
    comptime Stage: type,
    stage: *Stage,
    input: []const u8,
    layer_start: usize,
    layer_end: usize,
) !void {
    stage.executeLayers(input, layer_start, layer_end) catch |err| {
        return preserveLocalStageExecutionError(allocator, .{
            .placement_plan = placement_plan,
            .state_ownership_plan = state_ownership_plan,
            .metadata = metadata,
            .active_stage_id = active_stage_id,
            .source_error = err,
        });
    };
}

fn buildLocalStageExecutionFailureEntry(
    allocator: Allocator,
    request: LocalStageExecutionFailureRequest,
    batch_entry: tensor_frame.TensorFrameBatchEntry,
) !LocalStageExecutionFailureEntry {
    const active_host = try host_capability.bindingForStage(request.placement_plan, request.active_stage_id);
    const boundary = request.metadata.boundary;
    const source_host = try host_capability.bindingForStage(request.placement_plan, boundary.source_stage_id);
    const target_host = try host_capability.bindingForStage(request.placement_plan, boundary.target_stage_id);
    const is_out_of_memory = request.source_error == error.OutOfMemory;
    const is_cancelled = request.source_error == error.RequestCancelled;
    const primary_failure = try staged_error.buildStagedFailure(.{
        .kind = if (is_out_of_memory) .resource_exhausted else if (is_cancelled) .request_cancelled else .stage_execution_failed,
        .phase = .stage_execution_after_state_mutation,
        .scope = if (is_cancelled) .request else .stage,
        .context = .{
            .graph_digest = request.placement_plan.graph_digest,
            .graph_contract_version = request.placement_plan.graph_contract_version,
            .stage_plan_contract_version = request.placement_plan.stage_plan_contract_version,
            .stage_plan_id = request.placement_plan.stage_plan_id,
            .placement_plan_id = request.placement_plan.plan_id,
            .state_ownership_plan_id = if (request.state_ownership_plan) |plan| plan.plan_id else request.placement_plan.state_ownership_plan_id,
            .tensor_frame_id = request.metadata.frame_id,
            .boundary_index = boundary.boundary_index,
            .source_stage_id = boundary.source_stage_id,
            .target_stage_id = boundary.target_stage_id,
            .source_host_id = source_host.host_id,
            .target_host_id = target_host.host_id,
            .stage_id = request.active_stage_id,
            .host_id = active_host.host_id,
            .request_id = batch_entry.request_id,
            .slot_id = batch_entry.slot_id,
            .state_epoch = batch_entry.state_epoch,
        },
        .source = .{ .domain = .runner, .source_error_name = @errorName(request.source_error) },
    }, .{
        .placement_plan = request.placement_plan,
        .state_ownership_plan = request.state_ownership_plan,
    });

    const touched = try allocator.alloc(staged_error.TouchedStageCleanupRef, 1);
    errdefer allocator.free(touched);
    touched[0] = .{
        .stage_id = request.active_stage_id,
        .request_id = batch_entry.request_id,
        .slot_id = batch_entry.slot_id,
        .state_epoch = batch_entry.state_epoch,
    };

    var cleanup_plan_opt: ?staged_error.StagedCleanupPlan = null;
    var cleanup_report_opt: ?staged_error.StagedCleanupReport = null;
    errdefer if (cleanup_report_opt) |*report| report.deinit();
    errdefer if (cleanup_plan_opt) |*plan| plan.deinit();

    var cleanup_obligation_buffer = try buildLocalStageExecutionCleanupObligationsForEntry(
        allocator,
        request,
        batch_entry,
        touched,
    );
    defer cleanup_obligation_buffer.deinit();
    const cleanup_obligations = cleanup_obligation_buffer.obligations;

    if (staged_error.stagedCleanupRequired(primary_failure, touched, cleanup_obligations)) {
        cleanup_plan_opt = try staged_error.buildStagedCleanupPlan(allocator, .{
            .primary_failure = primary_failure,
            .request_id = batch_entry.request_id,
            .placement_plan = request.placement_plan,
            .state_ownership_plan = request.state_ownership_plan,
            .touched_stages = touched,
            .cleanup_obligations = cleanup_obligations,
        });
        if (cleanup_plan_opt.?.steps.len == 0) {
            cleanup_report_opt = try staged_error.buildStagedCleanupReport(allocator, &.{}, .{
                .cleanup_plan = &cleanup_plan_opt.?,
                .primary_failure = &primary_failure,
            });
        }
    }

    const cleanup_plan_id = if (cleanup_plan_opt) |plan| plan.plan_id else null;
    var error_report = try staged_error.buildStagedErrorReport(allocator, primary_failure, cleanup_plan_id, &.{}, .{
        .placement_plan = request.placement_plan,
        .state_ownership_plan = request.state_ownership_plan,
        .cleanup_plan = if (cleanup_plan_opt) |*plan| plan else null,
    });
    errdefer error_report.deinit();

    return .{
        .primary_failure = primary_failure,
        .touched_stages = touched,
        .cleanup_plan = cleanup_plan_opt,
        .cleanup_report = cleanup_report_opt,
        .error_report = error_report,
    };
}

fn deinitLocalStageExecutionFailureEntry(
    allocator: Allocator,
    entry: *LocalStageExecutionFailureEntry,
) void {
    entry.error_report.deinit();
    if (entry.cleanup_report) |*report| report.deinit();
    if (entry.cleanup_plan) |*plan| plan.deinit();
    allocator.free(entry.touched_stages);
}

const LocalStageExecutionCleanupObligationBuffer = struct {
    allocator: Allocator,
    obligations: []const state_ownership.StateCleanupObligation = &.{},

    fn deinit(self: *@This()) void {
        if (self.obligations.len != 0) self.allocator.free(self.obligations);
        self.* = undefined;
    }
};

fn buildLocalStageExecutionCleanupObligationsForEntry(
    allocator: Allocator,
    request: LocalStageExecutionFailureRequest,
    batch_entry: tensor_frame.TensorFrameBatchEntry,
    touched_stages: []const staged_error.TouchedStageCleanupRef,
) !LocalStageExecutionCleanupObligationBuffer {
    const obligations = if (request.cleanup_obligations.len != 0)
        try copyLocalStageExecutionCleanupObligationsForEntry(
            allocator,
            request.cleanup_obligations,
            request.active_stage_id,
            batch_entry,
        )
    else if (request.state_ownership_plan) |plan|
        try deriveLocalStageExecutionCleanupObligationsForTouched(allocator, plan, touched_stages)
    else
        &.{};
    return .{
        .allocator = allocator,
        .obligations = obligations,
    };
}

fn copyLocalStageExecutionCleanupObligationsForEntry(
    allocator: Allocator,
    obligations: []const state_ownership.StateCleanupObligation,
    active_stage_id: usize,
    batch_entry: tensor_frame.TensorFrameBatchEntry,
) ![]state_ownership.StateCleanupObligation {
    var count: usize = 0;
    for (obligations) |obligation| {
        if (obligation.stage_id == active_stage_id and
            obligation.request_id == batch_entry.request_id and
            obligation.slot_id == batch_entry.slot_id)
        {
            count += 1;
        }
    }
    if (count == 0) return &.{};

    const filtered = try allocator.alloc(state_ownership.StateCleanupObligation, count);
    var index: usize = 0;
    for (obligations) |obligation| {
        if (obligation.stage_id == active_stage_id and
            obligation.request_id == batch_entry.request_id and
            obligation.slot_id == batch_entry.slot_id)
        {
            filtered[index] = obligation;
            index += 1;
        }
    }
    return filtered;
}

fn deriveLocalStageExecutionCleanupObligationsForTouched(
    allocator: Allocator,
    plan: *const state_ownership.StageStateOwnershipPlan,
    touched_stages: []const staged_error.TouchedStageCleanupRef,
) ![]state_ownership.StateCleanupObligation {
    if (touched_stages.len == 0) return &.{};

    const targets = try allocator.alloc(state_ownership.StageStateCleanupTarget, touched_stages.len);
    defer allocator.free(targets);
    var obligation_capacity: usize = 0;
    for (touched_stages, 0..) |touched, index| {
        const descriptors = try state_ownership.descriptorSetForStage(plan, touched.stage_id);
        obligation_capacity += descriptors.descriptors.len;
        targets[index] = .{
            .stage_id = touched.stage_id,
            .request_id = touched.request_id,
            .slot_id = touched.slot_id,
        };
    }
    if (obligation_capacity == 0) return &.{};

    const scratch = try allocator.alloc(state_ownership.StateCleanupObligation, obligation_capacity);
    defer allocator.free(scratch);
    const obligations = try state_ownership.buildStateCleanupObligations(plan, targets, scratch);
    if (obligations.len == 0) return &.{};
    return allocator.dupe(state_ownership.StateCleanupObligation, obligations);
}

fn validateLocalStageEndpointRegistryChainRequestShape(request: LocalStageEndpointRegistryChainRequest) anyerror!void {
    try validateLocalStageRunnerPlanRef(request.plan_ref);
    try host_capability.validatePlacementPlan(request.placement_plan);
    if (request.registry.endpoints.len < 2) return error.InvalidStageRange;
    if (request.registry.endpoints.len != request.plan_ref.stages.len) return error.InvalidStepRequest;
    if (request.boundaries.len + 1 != request.plan_ref.stages.len) return error.InvalidStepRequest;
    if (request.stage_inputs.len != 0 and request.stage_inputs.len != request.plan_ref.stages.len) {
        return error.InvalidStepRequest;
    }
    var registry = request.registry;
    for (request.plan_ref.stages) |stage_ref| {
        const endpoint = try registry.endpointForStageId(stage_ref.stage_id);
        if (endpoint.stage_id != stage_ref.stage_id) {
            return error.StageRunnerPlanIdentityMismatch;
        }
    }
    for (request.registry.endpoints, 0..) |endpoint, endpoint_index| {
        _ = stageRef(request.plan_ref.stages, endpoint.stage_id) orelse return error.MissingStageRef;
        for (request.registry.endpoints[endpoint_index + 1 ..]) |other| {
            if (other.stage_id == endpoint.stage_id) return error.DuplicateStageRef;
        }
    }
    if (request.registry.transport_endpoints.len != 0) {
        if (request.registry.transport_endpoints.len != request.plan_ref.stages.len) return error.InvalidStepRequest;
        var transport_registry = local_stage_transport.LocalStageTransportEndpointRegistry{
            .endpoints = request.registry.transport_endpoints,
        };
        for (request.plan_ref.stages) |stage_ref| {
            const endpoint = try transport_registry.endpointForStageId(stage_ref.stage_id);
            if (endpoint.stage_id != stage_ref.stage_id) return error.StageRunnerPlanIdentityMismatch;
        }
        for (request.registry.transport_endpoints, 0..) |endpoint, endpoint_index| {
            _ = stageRef(request.plan_ref.stages, endpoint.stage_id) orelse return error.MissingStageRef;
            for (request.registry.transport_endpoints[endpoint_index + 1 ..]) |other| {
                if (other.stage_id == endpoint.stage_id) return error.DuplicateStageRef;
            }
        }
    } else {
        for (request.registry.endpoints) |endpoint| {
            if (endpoint.transport_endpoint == null) return error.MissingStageRef;
        }
    }
}

fn validateLocalStageChainBoundaryStatic(
    request: LocalStageEndpointRegistryChainRequest,
    boundary: *const LocalStageRunnerBoundaryRef,
    step: LocalStageChainBoundaryStep,
    boundary_position: usize,
) anyerror!void {
    if (boundary_position > 0 and step.boundary_index != request.boundaries[boundary_position - 1].boundary_index + 1) {
        return error.BoundaryIndexOutOfRange;
    }
    if (request.plan_ref.stages[boundary_position].stage_id != boundary.source_stage_id or
        request.plan_ref.stages[boundary_position + 1].stage_id != boundary.target_stage_id)
    {
        return error.StageRunnerPlanIdentityMismatch;
    }
    if (boundary_position > 0) {
        const previous = try request.plan_ref.boundary(boundary_position - 1);
        if (previous.target_stage_id != boundary.source_stage_id or
            previous.consumer_layer_start != boundary.producer_layer_start or
            previous.consumer_layer_end != boundary.producer_layer_end)
        {
            return error.BoundaryFrameProfileMismatch;
        }
    }
    const profile = profileForStep(boundary, step.step_kind) orelse return error.MissingBoundaryFrameProfile;
    try validateTensorFrameAgainstRunnerPlan(request.plan_ref, step.metadata, step.boundary_index);
    try tensor_frame.validatePayloadBufferLength(step.metadata, step.image.byte_count);
    try validateTensorFrameAgainstBoundaryProfile(step.metadata, profile);
}

fn validateLocalStageChainBoundaryImageBeforeMutation(
    step: LocalStageChainBoundaryStep,
) anyerror!void {
    try boundary_byte_image.validateBoundaryByteImage(step.image, .{
        .allow_pending_host_segments = step.image.host_segments != null,
    });
    if (step.image.metadata != step.metadata) return error.BoundaryTensorContractMismatch;
}

fn validateLocalStageChainBoundaryImage(
    step: LocalStageChainBoundaryStep,
) anyerror!void {
    try boundary_byte_image.validateBoundaryByteImage(step.image, .{});
    if (step.image.metadata != step.metadata) return error.BoundaryTensorContractMismatch;
}

fn executeEndpointForStep(
    endpoint: *LocalStageEndpoint,
    step_kind: tensor_frame.TensorFrameStepKind,
    input: []const u8,
    layer_start: usize,
    layer_end: usize,
) anyerror!local_stage_transport.StageExecutionReceipt {
    return switch (step_kind) {
        .decode => endpoint.executeDecodeLayerRange(input, layer_start, layer_end),
        .prefill => endpoint.executePrefillLayerRange(input, layer_start, layer_end),
    };
}

fn stageInput(request: anytype, stage_index: usize) []const u8 {
    if (request.stage_inputs.len == 0) return &.{};
    return request.stage_inputs[stage_index];
}

fn usizeToU64(value: usize, comptime err: anyerror) !u64 {
    return std.math.cast(u64, value) orelse return err;
}

fn activationFrameId(
    boundary_index: usize,
    first_slot_id: u64,
    first_sequence_start: u64,
    first_token_count: u64,
    batch_entry_count: usize,
) !tensor_frame.TensorFrameInstanceId {
    var raw: u64 = 0xcbf2_9ce4_8422_2325;
    inline for (.{ boundary_index, batch_entry_count }) |value| {
        raw = (raw ^ (try usizeToU64(value, error.InvalidFrameId))) *% 0x0000_0100_0000_01b3;
    }
    raw = (raw ^ first_slot_id) *% 0x0000_0100_0000_01b3;
    raw = (raw ^ first_sequence_start) *% 0x0000_0100_0000_01b3;
    raw = (raw ^ first_token_count) *% 0x0000_0100_0000_01b3;
    return tensor_frame.TensorFrameInstanceId.init(if (raw == 0) 1 else raw);
}

fn requestIdForSlot(slot_request_ids: []const ?u64, slot_index: usize) !u64 {
    if (slot_index >= slot_request_ids.len) return error.InvalidSlotId;
    return slot_request_ids[slot_index] orelse error.InvalidRequestId;
}

fn slotId(slot_index: usize) !u64 {
    return std.math.add(u64, try usizeToU64(slot_index, error.InvalidSlotId), 1) catch return error.InvalidSlotId;
}

fn validatePlanInputs(request: LocalStageRunnerPlanRequest) LocalStageRunnerError!void {
    try stage_plan.validateStagePlan(request.stage_plan, .{});
    try validateTensorFramePlanRefForStagePlan(request.tensor_frame_plan_ref, request.stage_plan);
    try host_capability.validatePlacementPlan(request.placement_plan);
    try validatePlacementIdentity(request.placement_plan, request.stage_plan);
    if (request.state_ownership_plan) |ownership| {
        try state_ownership.validateStageStateOwnershipPlan(ownership);
        try validateStateOwnershipIdentity(ownership, request.stage_plan);
        if (request.placement_plan.state_ownership_contract_version == null or
            request.placement_plan.state_ownership_plan_id == null)
        {
            return error.StateOwnershipPlanIdentityMismatch;
        }
        if (ownership.version != request.placement_plan.state_ownership_contract_version.? or
            !state_ownership.stateOwnershipPlanIdEql(ownership.plan_id, request.placement_plan.state_ownership_plan_id.?))
        {
            return error.StateOwnershipPlanIdentityMismatch;
        }
    } else if (request.placement_plan.state_ownership_plan_id != null) {
        return error.MissingStatePlacementSummary;
    }
}

fn validateTensorFramePlanRefForStagePlan(
    plan_ref: *const tensor_frame.TensorFramePlanRef,
    plan: *const stage_plan.StagePlan,
) LocalStageRunnerError!void {
    if (!std.mem.eql(u8, &plan_ref.identity.graph_digest, &plan.graph_identity.digest) or
        plan_ref.identity.graph_contract_version != plan.graph_identity.graph_contract_version)
    {
        return error.GraphIdentityMismatch;
    }
    if (plan_ref.identity.stage_plan_contract_version != plan.stage_contract_version or
        !std.mem.eql(u8, &plan_ref.identity.stage_plan_id.digest, &plan.plan_id.digest))
    {
        return error.StagePlanIdentityMismatch;
    }
    if (plan_ref.boundaries.len != plan.boundaries.len) return error.BoundaryIndexOutOfRange;
    for (plan.boundaries, 0..) |boundary, index| {
        const expected = tensor_frame.TensorFrameBoundaryRef{
            .boundary_index = index,
            .source_stage_id = boundary.source_stage_id,
            .target_stage_id = boundary.target_stage_id,
            .producer_layer_start = boundary.producer_layer_start,
            .producer_layer_end = boundary.producer_layer_end,
            .consumer_layer_start = boundary.consumer_layer_start,
            .consumer_layer_end = boundary.consumer_layer_end,
        };
        if (!boundaryRefEql(expected, plan_ref.boundaries[index])) return error.BoundaryTensorContractMismatch;
    }
}

fn validatePlacementIdentity(
    placement: *const host_capability.PlacementPlan,
    plan: *const stage_plan.StagePlan,
) LocalStageRunnerError!void {
    if (!std.mem.eql(u8, &placement.graph_digest, &plan.graph_identity.digest) or
        placement.graph_contract_version != plan.graph_identity.graph_contract_version)
    {
        return error.GraphIdentityMismatch;
    }
    if (placement.stage_plan_contract_version != plan.stage_contract_version or
        !std.mem.eql(u8, &placement.stage_plan_id.digest, &plan.plan_id.digest))
    {
        return error.StagePlanIdentityMismatch;
    }
}

fn validateStateOwnershipIdentity(
    ownership: *const state_ownership.StageStateOwnershipPlan,
    plan: *const stage_plan.StagePlan,
) LocalStageRunnerError!void {
    if (!std.mem.eql(u8, &ownership.graph_digest, &plan.graph_identity.digest) or
        ownership.graph_contract_version != plan.graph_identity.graph_contract_version)
    {
        return error.GraphIdentityMismatch;
    }
    if (ownership.stage_plan_contract_version != plan.stage_contract_version or
        !std.mem.eql(u8, &ownership.stage_plan_id.digest, &plan.plan_id.digest))
    {
        return error.StagePlanIdentityMismatch;
    }
}

fn copyStageRefs(
    allocator: Allocator,
    plan: *const stage_plan.StagePlan,
    placement: *const host_capability.PlacementPlan,
) LocalStageRunnerError![]LocalStageRunnerStageRef {
    const stages = try allocator.alloc(LocalStageRunnerStageRef, plan.stages.len);
    for (plan.stages, 0..) |stage, index| {
        const summary = placementStageSummary(placement, stage.id) orelse return error.MissingStageRef;
        if (summary.layer_start != stage.layer_start or summary.layer_end != stage.layer_end or
            !ownedRolesEql(summary.owned_roles, stage.owned_roles))
        {
            return error.StageRunnerPlanIdentityMismatch;
        }
        const binding = placementBinding(placement, stage.id) orelse return error.MissingStageRef;
        _ = placementHostSummary(placement, binding.host_id) orelse return error.MissingStageRef;
        stages[index] = .{
            .stage_id = stage.id,
            .host_id = binding.host_id,
            .layer_start = stage.layer_start,
            .layer_end = stage.layer_end,
            .owned_roles = stage.owned_roles,
        };
    }
    return stages;
}

fn copyBoundaryRefs(
    allocator: Allocator,
    plan: *const stage_plan.StagePlan,
    placement: *const host_capability.PlacementPlan,
    required_step_kinds: []const tensor_frame.TensorFrameStepKind,
) LocalStageRunnerError![]LocalStageRunnerBoundaryRef {
    const boundaries = try allocator.alloc(LocalStageRunnerBoundaryRef, plan.boundaries.len);
    for (plan.boundaries, 0..) |boundary, index| {
        const summary = placementBoundarySummary(placement, index) orelse return error.MissingBoundaryRef;
        if (summary.source_stage_id != boundary.source_stage_id or
            summary.target_stage_id != boundary.target_stage_id or
            summary.producer_layer_start != boundary.producer_layer_start or
            summary.producer_layer_end != boundary.producer_layer_end or
            summary.consumer_layer_start != boundary.consumer_layer_start or
            summary.consumer_layer_end != boundary.consumer_layer_end)
        {
            return error.BoundaryFrameProfileMismatch;
        }
        const source_binding = placementBinding(placement, boundary.source_stage_id) orelse return error.MissingStageRef;
        const target_binding = placementBinding(placement, boundary.target_stage_id) orelse return error.MissingStageRef;
        const profiles = try allocator.alloc(LocalStageRunnerBoundaryProfile, required_step_kinds.len);
        for (required_step_kinds, 0..) |step_kind, profile_index| {
            const profile = placementProfile(placement, index, step_kind) orelse return error.MissingBoundaryFrameProfile;
            profiles[profile_index] = try copyBoundaryProfile(profile);
        }
        boundaries[index] = .{
            .boundary_index = index,
            .source_stage_id = boundary.source_stage_id,
            .target_stage_id = boundary.target_stage_id,
            .source_host_id = source_binding.host_id,
            .target_host_id = target_binding.host_id,
            .producer_layer_start = boundary.producer_layer_start,
            .producer_layer_end = boundary.producer_layer_end,
            .consumer_layer_start = boundary.consumer_layer_start,
            .consumer_layer_end = boundary.consumer_layer_end,
            .profiles = profiles,
        };
    }
    return boundaries;
}

fn copyBoundaryProfile(
    profile: host_capability.BoundaryFrameProfile,
) LocalStageRunnerError!LocalStageRunnerBoundaryProfile {
    const mode = profile.handoff_mode;
    switch (mode) {
        .local_in_process, .mock => {},
        .same_host_direct => return error.InvalidLocalHandoffMode,
        .remote_declared => return error.UnsupportedRemoteBoundary,
    }
    return .{
        .step_kind = profile.step_kind,
        .dtype = profile.dtype,
        .layout = profile.layout,
        .handoff_mode = mode,
        .max_batch_entries = profile.max_batch_entries,
        .max_token_count_per_frame = profile.max_token_count_per_frame,
        .max_activation_payload_bytes = profile.max_activation_payload_bytes,
    };
}

fn validateRequiredStepKinds(kinds: []const tensor_frame.TensorFrameStepKind) LocalStageRunnerError!void {
    if (kinds.len == 0 or kinds.len > 2) return error.InvalidStepRequest;
    for (kinds, 0..) |kind, index| {
        if (index > 0 and @intFromEnum(kinds[index - 1]) >= @intFromEnum(kind)) return error.InvalidStepRequest;
    }
}

fn validateCopiedStageRefs(stages: []const LocalStageRunnerStageRef) LocalStageRunnerError!void {
    var previous: ?usize = null;
    for (stages) |stage| {
        if (stage.layer_start >= stage.layer_end) return error.InvalidStageRange;
        if (previous) |prev| {
            if (stage.stage_id <= prev) return error.DuplicateStageRef;
        }
        previous = stage.stage_id;
    }
}

fn validateCopiedBoundaryRefs(plan_ref: *const LocalStageRunnerPlanRef) LocalStageRunnerError!void {
    if (plan_ref.boundaries.len != plan_ref.tensor_frame_boundaries.len) return error.MissingBoundaryRef;
    for (plan_ref.boundaries, 0..) |boundary, index| {
        if (boundary.boundary_index != index) return error.DuplicateBoundaryRef;
        if (boundary.source_stage_id == boundary.target_stage_id) return error.InvalidStageRange;
        if (boundary.producer_layer_start >= boundary.producer_layer_end or
            boundary.consumer_layer_start >= boundary.consumer_layer_end)
        {
            return error.InvalidStageRange;
        }
        const source_stage = stageRef(plan_ref.stages, boundary.source_stage_id) orelse return error.MissingStageRef;
        const target_stage = stageRef(plan_ref.stages, boundary.target_stage_id) orelse return error.MissingStageRef;
        if (!hostIdEql(source_stage.host_id, boundary.source_host_id) or
            !hostIdEql(target_stage.host_id, boundary.target_host_id))
        {
            return error.StageRunnerPlanIdentityMismatch;
        }
        const tensor_boundary = plan_ref.tensor_frame_boundaries[index];
        if (!boundaryRefMatchesBoundary(tensor_boundary, boundary)) return error.BoundaryFrameProfileMismatch;
        if (boundary.profiles.len != plan_ref.required_step_kinds.len) return error.MissingBoundaryFrameProfile;
        for (plan_ref.required_step_kinds, 0..) |step_kind, profile_index| {
            const profile = boundary.profiles[profile_index];
            if (profile.step_kind != step_kind) return error.MissingBoundaryFrameProfile;
            switch (profile.handoff_mode) {
                .local_in_process, .mock => {},
                .same_host_direct, .remote_declared => return error.InvalidLocalHandoffMode,
            }
        }
    }
}

fn validateTensorFrameAgainstRunnerPlan(
    plan_ref: *const LocalStageRunnerPlanRef,
    metadata: *const tensor_frame.TensorFrameMetadata,
    boundary_index: usize,
) tensor_frame.TensorFrameValidationError!void {
    try metadata.validate();
    if (!std.mem.eql(u8, &metadata.plan.graph_digest, &plan_ref.tensor_frame_plan_identity.graph_digest) or
        metadata.plan.graph_contract_version != plan_ref.tensor_frame_plan_identity.graph_contract_version)
    {
        return error.GraphIdentityMismatch;
    }
    if (metadata.plan.stage_plan_contract_version != plan_ref.tensor_frame_plan_identity.stage_plan_contract_version or
        !std.mem.eql(u8, &metadata.plan.stage_plan_id.digest, &plan_ref.tensor_frame_plan_identity.stage_plan_id.digest))
    {
        return error.StagePlanIdentityMismatch;
    }
    if (boundary_index >= plan_ref.tensor_frame_boundaries.len) return error.BoundaryIndexOutOfRange;
    const expected = plan_ref.tensor_frame_boundaries[boundary_index];
    if (!boundaryRefEql(expected, metadata.boundary) or !boundaryRefEql(expected, metadata.selected_contract.boundary)) {
        return error.BoundaryTensorContractMismatch;
    }
}

fn singleBatchEntry(metadata: *const tensor_frame.TensorFrameMetadata) tensor_frame.TensorFrameValidationError!tensor_frame.TensorFrameBatchEntry {
    if (metadata.batch.entries.len != 1) return error.InvalidBatch;
    const entry = metadata.batch.entries[0];
    if (entry.batch_index != 0) return error.InvalidBatch;
    return entry;
}

fn validateTensorFrameAgainstBoundaryProfile(
    metadata: *const tensor_frame.TensorFrameMetadata,
    profile: *const LocalStageRunnerBoundaryProfile,
) tensor_frame.TensorFrameValidationError!void {
    if (metadata.step_kind != profile.step_kind) return error.InvalidStepKind;
    if (metadata.selected_contract.dtype != profile.dtype or metadata.tensor.dtype != profile.dtype) {
        return error.InvalidDType;
    }
    if (metadata.selected_contract.layout != profile.layout or
        metadata.tensor.layout != profile.layout or
        metadata.payload.layout != profile.layout)
    {
        return error.UnsupportedTensorLayout;
    }
    if (metadata.batch.entries.len > profile.max_batch_entries) return error.InvalidBatch;

    var token_count: u64 = 0;
    for (metadata.batch.entries) |entry| {
        token_count = std.math.add(u64, token_count, entry.token_count) catch return error.InvalidSequenceRange;
    }
    if (token_count > profile.max_token_count_per_frame) return error.InvalidSequenceRange;
    if (metadata.payload.byte_count > profile.max_activation_payload_bytes) return error.InvalidPayloadByteCount;
}

fn runnerFailure(
    plan_ref: *const LocalStageRunnerPlanRef,
    boundary: ?*const LocalStageRunnerBoundaryRef,
    boundary_index: usize,
    batch_entry: ?tensor_frame.TensorFrameBatchEntry,
    tensor_frame_id: ?tensor_frame.TensorFrameInstanceId,
    source_error: anyerror,
    touched: []const LocalStageTouchedRef,
) LocalStageRunnerError!LocalStageRunResult {
    return failureWith(
        plan_ref,
        boundary,
        boundary_index,
        batch_entry,
        tensor_frame_id,
        null,
        null,
        .internal_contract_violation,
        .validation_before_mutation,
        .boundary,
        .runner,
        source_error,
        touched,
    );
}

fn tensorFailure(
    plan_ref: *const LocalStageRunnerPlanRef,
    boundary: *const LocalStageRunnerBoundaryRef,
    batch_entry: ?tensor_frame.TensorFrameBatchEntry,
    tensor_frame_id: ?tensor_frame.TensorFrameInstanceId,
    source_error: tensor_frame.TensorFrameValidationError,
    touched: []const LocalStageTouchedRef,
) LocalStageRunnerError!LocalStageRunResult {
    const class = staged_error.classifyTensorFrameError(source_error);
    return failureWith(
        plan_ref,
        boundary,
        boundary.boundary_index,
        batch_entry,
        tensor_frame_id,
        null,
        null,
        class.kind,
        .validation_before_mutation,
        class.scope,
        class.source.domain,
        source_error,
        touched,
    );
}

fn transportFailure(
    plan_ref: *const LocalStageRunnerPlanRef,
    boundary: *const LocalStageRunnerBoundaryRef,
    batch_entry: tensor_frame.TensorFrameBatchEntry,
    tensor_frame_id: ?tensor_frame.TensorFrameInstanceId,
    source_error: anyerror,
    touched: []const LocalStageTouchedRef,
    phase: staged_error.StagedFailurePhase,
) LocalStageRunnerError!LocalStageRunResult {
    const is_out_of_memory = source_error == error.OutOfMemory;
    const is_cancelled = source_error == error.RequestCancelled;
    return failureWith(
        plan_ref,
        boundary,
        boundary.boundary_index,
        batch_entry,
        tensor_frame_id,
        null,
        null,
        if (is_out_of_memory) .resource_exhausted else if (is_cancelled) .request_cancelled else .transfer_failed,
        phase,
        if (is_cancelled) .request else .transport,
        .transport,
        source_error,
        touched,
    );
}

fn observerFailure(
    plan_ref: *const LocalStageRunnerPlanRef,
    boundary: *const LocalStageRunnerBoundaryRef,
    batch_entry: tensor_frame.TensorFrameBatchEntry,
    tensor_frame_id: ?tensor_frame.TensorFrameInstanceId,
    source_error: anyerror,
    touched: []const LocalStageTouchedRef,
) LocalStageRunnerError!LocalStageRunResult {
    return failureWith(
        plan_ref,
        boundary,
        boundary.boundary_index,
        batch_entry,
        tensor_frame_id,
        null,
        null,
        .internal_contract_violation,
        .frame_handoff,
        .boundary,
        .tensor_frame,
        source_error,
        touched,
    );
}

fn stageFailure(
    plan_ref: *const LocalStageRunnerPlanRef,
    boundary: *const LocalStageRunnerBoundaryRef,
    batch_entry: tensor_frame.TensorFrameBatchEntry,
    tensor_frame_id: ?tensor_frame.TensorFrameInstanceId,
    active_stage_id: usize,
    active_host_id: host_capability.HostId,
    source_error: anyerror,
    touched: []const LocalStageTouchedRef,
) LocalStageRunnerError!LocalStageRunResult {
    const is_out_of_memory = source_error == error.OutOfMemory;
    const is_cancelled = source_error == error.RequestCancelled;
    return failureWith(
        plan_ref,
        boundary,
        boundary.boundary_index,
        batch_entry,
        tensor_frame_id,
        active_stage_id,
        active_host_id,
        if (is_out_of_memory) .resource_exhausted else if (is_cancelled) .request_cancelled else .stage_execution_failed,
        .stage_execution_after_state_mutation,
        if (is_cancelled) .request else .stage,
        .runner,
        source_error,
        touched,
    );
}

fn failureWith(
    plan_ref: *const LocalStageRunnerPlanRef,
    boundary: ?*const LocalStageRunnerBoundaryRef,
    boundary_index: usize,
    batch_entry: ?tensor_frame.TensorFrameBatchEntry,
    tensor_frame_id: ?tensor_frame.TensorFrameInstanceId,
    active_stage_id: ?usize,
    active_host_id: ?host_capability.HostId,
    kind: staged_error.StagedFailureKind,
    phase: staged_error.StagedFailurePhase,
    scope: staged_error.StagedFailureScope,
    source_domain: staged_error.StagedSourceDomain,
    source_error: anyerror,
    touched: []const LocalStageTouchedRef,
) LocalStageRunnerError!LocalStageRunResult {
    const context = failureContext(plan_ref, boundary, boundary_index, batch_entry, tensor_frame_id, active_stage_id, active_host_id);
    const failure = try staged_error.buildStagedFailure(.{
        .kind = kind,
        .phase = phase,
        .scope = scope,
        .context = context,
        .source = .{ .domain = source_domain, .source_error_name = @errorName(source_error) },
    }, .{});
    return .{ .failure = .{
        .primary_failure = failure,
        .source_error = source_error,
        .touched_stages = touched,
    } };
}

fn failureContext(
    plan_ref: *const LocalStageRunnerPlanRef,
    boundary: ?*const LocalStageRunnerBoundaryRef,
    boundary_index: usize,
    batch_entry: ?tensor_frame.TensorFrameBatchEntry,
    tensor_frame_id: ?tensor_frame.TensorFrameInstanceId,
    active_stage_id: ?usize,
    active_host_id: ?host_capability.HostId,
) staged_error.StagedErrorContext {
    var context = staged_error.StagedErrorContext{
        .graph_digest = plan_ref.graph_digest,
        .graph_contract_version = plan_ref.graph_contract_version,
        .stage_plan_contract_version = plan_ref.stage_plan_contract_version,
        .stage_plan_id = plan_ref.stage_plan_id,
        .placement_plan_id = plan_ref.placement_plan_id,
        .state_ownership_plan_id = plan_ref.state_ownership_plan_id,
        .tensor_frame_id = tensor_frame_id,
        .boundary_index = boundary_index,
        .stage_id = active_stage_id,
        .host_id = active_host_id,
    };
    if (boundary) |entry| {
        context.source_stage_id = entry.source_stage_id;
        context.target_stage_id = entry.target_stage_id;
        context.source_host_id = entry.source_host_id;
        context.target_host_id = entry.target_host_id;
    }
    if (batch_entry) |entry| {
        context.request_id = entry.request_id;
        context.slot_id = entry.slot_id;
        context.state_epoch = entry.state_epoch;
    }
    return context;
}

fn touchedRef(
    stage_id: usize,
    host_id: host_capability.HostId,
    batch_entry: tensor_frame.TensorFrameBatchEntry,
    flags: struct {
        execution_started: bool = false,
        receiver_payload_mutation_started: bool = false,
    },
) LocalStageTouchedRef {
    return .{
        .stage_id = stage_id,
        .host_id = host_id,
        .request_id = batch_entry.request_id,
        .slot_id = batch_entry.slot_id,
        .state_epoch = batch_entry.state_epoch,
        .execution_started = flags.execution_started,
        .receiver_payload_mutation_started = flags.receiver_payload_mutation_started,
    };
}

fn profileForStep(
    boundary: *const LocalStageRunnerBoundaryRef,
    step_kind: tensor_frame.TensorFrameStepKind,
) ?*const LocalStageRunnerBoundaryProfile {
    for (boundary.profiles) |*profile| {
        if (profile.step_kind == step_kind) return profile;
    }
    return null;
}

fn placementStageSummary(
    placement: *const host_capability.PlacementPlan,
    stage_id: usize,
) ?host_capability.PlacementStageSummary {
    for (placement.stage_summaries) |summary| {
        if (summary.stage_id == stage_id) return summary;
    }
    return null;
}

fn placementBoundarySummary(
    placement: *const host_capability.PlacementPlan,
    boundary_index: usize,
) ?host_capability.PlacementBoundarySummary {
    for (placement.boundary_summaries) |summary| {
        if (summary.boundary_index == boundary_index) return summary;
    }
    return null;
}

fn placementBinding(
    placement: *const host_capability.PlacementPlan,
    stage_id: usize,
) ?host_capability.StageHostBinding {
    for (placement.stage_host_bindings) |binding| {
        if (binding.stage_id == stage_id) return binding;
    }
    return null;
}

fn placementHostSummary(
    placement: *const host_capability.PlacementPlan,
    host_id: host_capability.HostId,
) ?host_capability.PlacementHostSummary {
    for (placement.host_summaries) |summary| {
        if (hostIdEql(summary.host_id, host_id)) return summary;
    }
    return null;
}

fn placementProfile(
    placement: *const host_capability.PlacementPlan,
    boundary_index: usize,
    step_kind: tensor_frame.TensorFrameStepKind,
) ?host_capability.BoundaryFrameProfile {
    for (placement.boundary_frame_profiles) |profile| {
        if (profile.boundary_index == boundary_index and profile.step_kind == step_kind) return profile;
    }
    return null;
}

fn stageRef(stages: []const LocalStageRunnerStageRef, stage_id: usize) ?LocalStageRunnerStageRef {
    for (stages) |stage| {
        if (stage.stage_id == stage_id) return stage;
    }
    return null;
}

fn boundaryRefMatchesBoundary(
    lhs: tensor_frame.TensorFrameBoundaryRef,
    rhs: LocalStageRunnerBoundaryRef,
) bool {
    return lhs.boundary_index == rhs.boundary_index and
        lhs.source_stage_id == rhs.source_stage_id and
        lhs.target_stage_id == rhs.target_stage_id and
        lhs.producer_layer_start == rhs.producer_layer_start and
        lhs.producer_layer_end == rhs.producer_layer_end and
        lhs.consumer_layer_start == rhs.consumer_layer_start and
        lhs.consumer_layer_end == rhs.consumer_layer_end;
}

fn boundaryRefEql(
    lhs: tensor_frame.TensorFrameBoundaryRef,
    rhs: tensor_frame.TensorFrameBoundaryRef,
) bool {
    return lhs.boundary_index == rhs.boundary_index and
        lhs.source_stage_id == rhs.source_stage_id and
        lhs.target_stage_id == rhs.target_stage_id and
        lhs.producer_layer_start == rhs.producer_layer_start and
        lhs.producer_layer_end == rhs.producer_layer_end and
        lhs.consumer_layer_start == rhs.consumer_layer_start and
        lhs.consumer_layer_end == rhs.consumer_layer_end;
}

fn ownedRolesEql(
    lhs: [models.manifest.role_count]bool,
    rhs: [models.manifest.role_count]bool,
) bool {
    return std.mem.eql(bool, &lhs, &rhs);
}

fn hostIdEql(lhs: host_capability.HostId, rhs: host_capability.HostId) bool {
    return lhs.value == rhs.value;
}

fn assertStageContract(comptime StageType: type) void {
    const required_methods = .{
        .{ "executeLayers", 4 },
        .{ "downloadActivation", 3 },
        .{ "uploadActivation", 3 },
        .{ "synchronize", 1 },
    };
    inline for (required_methods) |entry| {
        const name = entry[0];
        const arity = entry[1];
        if (!@hasDecl(StageType, name)) {
            @compileError("Local stage runner type '" ++ @typeName(StageType) ++ "' missing required method '" ++ name ++ "'");
        }
        const decl = @field(StageType, name);
        const info = @typeInfo(@TypeOf(decl));
        if (info != .@"fn") {
            @compileError("Local stage runner method '" ++ @typeName(StageType) ++ "." ++ name ++ "' must be a function");
        }
        if (info.@"fn".params.len != arity) {
            @compileError("Local stage runner method '" ++ @typeName(StageType) ++ "." ++ name ++ "' has invalid arity");
        }
    }
}

fn computeLocalStageRunnerPlanId(plan_ref: *const LocalStageRunnerPlanRef) LocalStageRunnerPlanId {
    var encoder = HashEncoder.init();
    encoder.writeString("talu.local_stage_runner");
    encoder.writeU32(plan_ref.version);
    encoder.writeBytes(&plan_ref.graph_digest);
    encoder.writeU32(plan_ref.graph_contract_version);
    encoder.writeU32(plan_ref.stage_plan_contract_version);
    encoder.writeBytes(&plan_ref.stage_plan_id.digest);
    encoder.writeU32(plan_ref.tensor_frame_contract_version);
    encoder.writeU32(plan_ref.placement_contract_version);
    encoder.writeBytes(&plan_ref.placement_plan_id.digest);
    encoder.writeOptionalU32(plan_ref.state_ownership_contract_version);
    if (plan_ref.state_ownership_plan_id) |id| {
        encoder.writeBool(true);
        encoder.writeBytes(&id.digest);
    } else {
        encoder.writeBool(false);
    }
    encoder.writeUsize(plan_ref.stages.len);
    for (plan_ref.stages) |stage| writeStageRef(&encoder, stage);
    encoder.writeUsize(plan_ref.boundaries.len);
    for (plan_ref.boundaries) |boundary| writeBoundaryRef(&encoder, boundary);
    encoder.writeUsize(plan_ref.tensor_frame_boundaries.len);
    for (plan_ref.tensor_frame_boundaries) |boundary| writeTensorFrameBoundaryRef(&encoder, boundary);
    encoder.writeUsize(plan_ref.required_step_kinds.len);
    for (plan_ref.required_step_kinds) |kind| encoder.writeU8(@intFromEnum(kind));
    return .{ .digest = encoder.finish() };
}

fn writeStageRef(encoder: *HashEncoder, stage: LocalStageRunnerStageRef) void {
    encoder.writeUsize(stage.stage_id);
    encoder.writeU64(stage.host_id.value);
    encoder.writeUsize(stage.layer_start);
    encoder.writeUsize(stage.layer_end);
    for (stage.owned_roles) |owned| encoder.writeBool(owned);
}

fn writeBoundaryRef(encoder: *HashEncoder, boundary: LocalStageRunnerBoundaryRef) void {
    encoder.writeUsize(boundary.boundary_index);
    encoder.writeUsize(boundary.source_stage_id);
    encoder.writeUsize(boundary.target_stage_id);
    encoder.writeU64(boundary.source_host_id.value);
    encoder.writeU64(boundary.target_host_id.value);
    encoder.writeUsize(boundary.producer_layer_start);
    encoder.writeUsize(boundary.producer_layer_end);
    encoder.writeUsize(boundary.consumer_layer_start);
    encoder.writeUsize(boundary.consumer_layer_end);
    encoder.writeUsize(boundary.profiles.len);
    for (boundary.profiles) |profile| {
        encoder.writeU8(@intFromEnum(profile.step_kind));
        encoder.writeU8(@intFromEnum(profile.dtype));
        encoder.writeU8(@intFromEnum(profile.layout));
        encoder.writeU8(@intFromEnum(profile.handoff_mode));
        encoder.writeU64(profile.max_batch_entries);
        encoder.writeU64(profile.max_token_count_per_frame);
        encoder.writeU64(profile.max_activation_payload_bytes);
    }
}

fn writeTensorFrameBoundaryRef(encoder: *HashEncoder, boundary: tensor_frame.TensorFrameBoundaryRef) void {
    encoder.writeUsize(boundary.boundary_index);
    encoder.writeUsize(boundary.source_stage_id);
    encoder.writeUsize(boundary.target_stage_id);
    encoder.writeUsize(boundary.producer_layer_start);
    encoder.writeUsize(boundary.producer_layer_end);
    encoder.writeUsize(boundary.consumer_layer_start);
    encoder.writeUsize(boundary.consumer_layer_end);
}

const HashEncoder = struct {
    hasher: Sha256,

    fn init() HashEncoder {
        return .{ .hasher = Sha256.init(.{}) };
    }

    fn writeBytes(self: *HashEncoder, bytes: []const u8) void {
        self.hasher.update(bytes);
    }

    fn writeString(self: *HashEncoder, value: []const u8) void {
        self.writeUsize(value.len);
        self.writeBytes(value);
    }

    fn writeBool(self: *HashEncoder, value: bool) void {
        self.writeU8(if (value) 1 else 0);
    }

    fn writeU8(self: *HashEncoder, value: u8) void {
        self.hasher.update(&[_]u8{value});
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

    fn writeOptionalU32(self: *HashEncoder, value: ?u32) void {
        self.writeBool(value != null);
        if (value) |payload| self.writeU32(payload);
    }

    fn finish(self: *HashEncoder) [32]u8 {
        var digest: [32]u8 = undefined;
        self.hasher.final(&digest);
        return digest;
    }
};

const TestFixture = struct {
    manifest: models.manifest.ModelManifest,
    plan: stage_plan.StagePlan,
    plan_ref: tensor_frame.TensorFramePlanRef,
    state_plan: ?state_ownership.StageStateOwnershipPlan = null,
    state_ref: ?host_capability.StageStatePlacementRef = null,
    placement: host_capability.PlacementPlan,

    fn init(
        allocator: Allocator,
        split_points: []const usize,
        mode: host_capability.BoundaryHandoffMode,
        stateful: bool,
    ) !TestFixture {
        var manifest = try testManifest(allocator, 4);
        errdefer manifest.deinit();
        var arch = testArch();
        var config = testConfig(4);
        const dependencies = [_]stage_plan.DependencyOverride{.{
            .source_stage_id = 0,
            .target_stage_id = 1,
            .reason = .stateful_decoder,
            .affects_loader_residency = false,
        }};
        var plan = try stage_plan.buildStagePlan(allocator, .{
            .n_layers = 4,
            .split_points = split_points,
            .architecture = &arch,
            .model_config = &config,
            .manifest = &manifest,
            .partition_constraints = if (split_points.len == 0) null else .{
                .decoder_cuts_allowed = true,
                .dependency_overrides = if (stateful) &dependencies else &.{},
            },
        });
        errdefer plan.deinit();
        var plan_ref = try tensor_frame.TensorFramePlanRef.fromStagePlan(allocator, &plan);
        errdefer plan_ref.deinit();

        var state_plan_opt: ?state_ownership.StageStateOwnershipPlan = null;
        errdefer if (state_plan_opt) |*value| value.deinit();
        var state_ref_opt: ?host_capability.StageStatePlacementRef = null;
        errdefer if (state_ref_opt) |*value| value.deinit();
        if (stateful) {
            state_plan_opt = try buildDescriptorFreeStatePlan(allocator, &plan);
            if (state_plan_opt) |*value| {
                state_ref_opt = try host_capability.buildStageStatePlacementRef(allocator, value);
            }
        }
        const state_ref_ptr: ?*const host_capability.StageStatePlacementRef = if (state_ref_opt) |*value| value else null;
        var placement = try buildTestPlacement(allocator, &plan, mode, state_ref_ptr);
        errdefer placement.deinit();

        const fixture = TestFixture{
            .manifest = manifest,
            .plan = plan,
            .plan_ref = plan_ref,
            .state_plan = state_plan_opt,
            .state_ref = state_ref_opt,
            .placement = placement,
        };
        state_plan_opt = null;
        state_ref_opt = null;
        return fixture;
    }

    fn deinit(self: *TestFixture) void {
        self.placement.deinit();
        if (self.state_ref) |*value| value.deinit();
        if (self.state_plan) |*value| value.deinit();
        self.plan_ref.deinit();
        self.plan.deinit();
        self.manifest.deinit();
    }
};

const MockTrace = struct {
    execute_calls: usize = 0,
    synchronize_calls: usize = 0,
    download_calls: usize = 0,
    upload_calls: usize = 0,
    deinit_calls: usize = 0,
    last_layer_start: usize = 0,
    last_layer_end: usize = 0,
};

const MockStage = struct {
    trace: *MockTrace,
    fail_execute: ?anyerror = null,
    fail_synchronize: ?anyerror = null,
    fail_download: ?anyerror = null,
    fail_upload: ?anyerror = null,

    pub fn executeLayers(self: *@This(), _: []const u8, layer_start: usize, layer_end: usize) anyerror!void {
        self.trace.execute_calls += 1;
        self.trace.last_layer_start = layer_start;
        self.trace.last_layer_end = layer_end;
        if (self.fail_execute) |err| return err;
    }

    pub fn downloadActivation(self: *@This(), host_buf: []u8, byte_count: usize) anyerror!void {
        self.trace.download_calls += 1;
        if (self.fail_download) |err| return err;
        @memset(host_buf[0..byte_count], 0xaa);
    }

    pub fn uploadActivation(self: *@This(), _: []const u8, _: usize) anyerror!void {
        self.trace.upload_calls += 1;
        if (self.fail_upload) |err| return err;
    }

    pub fn synchronize(self: *@This()) anyerror!void {
        self.trace.synchronize_calls += 1;
        if (self.fail_synchronize) |err| return err;
    }

    pub fn deinit(self: *@This(), _: Allocator) void {
        self.trace.deinit_calls += 1;
    }
};

test "buildLocalStageRunnerPlanRef validateLocalStageRunnerPlanRef localStageRunnerPlanIdEql builds plan from stage frame and placement plans" {
    const allocator = std.testing.allocator;
    var fixture = try TestFixture.init(allocator, &.{2}, .local_in_process, false);
    defer fixture.deinit();
    var runner = try buildLocalStageRunnerPlanRef(allocator, .{
        .stage_plan = &fixture.plan,
        .tensor_frame_plan_ref = &fixture.plan_ref,
        .placement_plan = &fixture.placement,
    });
    defer runner.deinit();

    try validateLocalStageRunnerPlanRef(&runner);
    try std.testing.expect(localStageRunnerPlanIdEql(runner.plan_id, runner.plan_id));
    try std.testing.expectEqual(@as(usize, 2), runner.stages.len);
    try std.testing.expectEqual(@as(usize, 1), runner.boundaries.len);
    try std.testing.expectEqual(@as(u64, 1), runner.boundaries[0].source_host_id.value);
    try std.testing.expectEqual(@as(u64, 2), runner.boundaries[0].target_host_id.value);
}

test "buildLocalStageRunnerPlanRef rejects identity mismatches and unsupported handoff profiles" {
    const allocator = std.testing.allocator;
    var fixture = try TestFixture.init(allocator, &.{2}, .local_in_process, false);
    defer fixture.deinit();

    var wrong_ref = fixture.plan_ref;
    wrong_ref.identity.stage_plan_id.digest[0] ^= 1;
    try std.testing.expectError(error.StagePlanIdentityMismatch, buildLocalStageRunnerPlanRef(allocator, .{
        .stage_plan = &fixture.plan,
        .tensor_frame_plan_ref = &wrong_ref,
        .placement_plan = &fixture.placement,
    }));

    var wrong_placement = try TestFixture.init(allocator, &.{1}, .local_in_process, false);
    defer wrong_placement.deinit();
    try std.testing.expectError(error.StagePlanIdentityMismatch, buildLocalStageRunnerPlanRef(allocator, .{
        .stage_plan = &fixture.plan,
        .tensor_frame_plan_ref = &fixture.plan_ref,
        .placement_plan = &wrong_placement.placement,
    }));

    var missing_binding = fixture.placement;
    missing_binding.stage_host_bindings = missing_binding.stage_host_bindings[0..1];
    try std.testing.expectError(error.MissingStageBinding, buildLocalStageRunnerPlanRef(allocator, .{
        .stage_plan = &fixture.plan,
        .tensor_frame_plan_ref = &fixture.plan_ref,
        .placement_plan = &missing_binding,
    }));

    var missing_profile = fixture.placement;
    missing_profile.boundary_frame_profiles = &.{};
    try std.testing.expectError(error.MissingBoundaryFrameProfile, buildLocalStageRunnerPlanRef(allocator, .{
        .stage_plan = &fixture.plan,
        .tensor_frame_plan_ref = &fixture.plan_ref,
        .placement_plan = &missing_profile,
    }));

    var same_host = try TestFixture.init(allocator, &.{2}, .same_host_direct, false);
    defer same_host.deinit();
    try std.testing.expectError(error.InvalidLocalHandoffMode, buildLocalStageRunnerPlanRef(allocator, .{
        .stage_plan = &same_host.plan,
        .tensor_frame_plan_ref = &same_host.plan_ref,
        .placement_plan = &same_host.placement,
    }));

    var remote = try TestFixture.init(allocator, &.{2}, .remote_declared, false);
    defer remote.deinit();
    try std.testing.expectError(error.UnsupportedRemoteBoundary, buildLocalStageRunnerPlanRef(allocator, .{
        .stage_plan = &remote.plan,
        .tensor_frame_plan_ref = &remote.plan_ref,
        .placement_plan = &remote.placement,
    }));
}

test "buildLocalStageRunnerPlanRef fingerprints host profile and state identity facts" {
    const allocator = std.testing.allocator;
    var local_fixture = try TestFixture.init(allocator, &.{2}, .local_in_process, false);
    defer local_fixture.deinit();
    var local_runner = try buildLocalStageRunnerPlanRef(allocator, .{
        .stage_plan = &local_fixture.plan,
        .tensor_frame_plan_ref = &local_fixture.plan_ref,
        .placement_plan = &local_fixture.placement,
    });
    defer local_runner.deinit();

    var mock_fixture = try TestFixture.init(allocator, &.{2}, .mock, false);
    defer mock_fixture.deinit();
    var mock_runner = try buildLocalStageRunnerPlanRef(allocator, .{
        .stage_plan = &mock_fixture.plan,
        .tensor_frame_plan_ref = &mock_fixture.plan_ref,
        .placement_plan = &mock_fixture.placement,
    });
    defer mock_runner.deinit();
    try std.testing.expect(!localStageRunnerPlanIdEql(local_runner.plan_id, mock_runner.plan_id));

    var stateful_fixture = try TestFixture.init(allocator, &.{2}, .local_in_process, true);
    defer stateful_fixture.deinit();
    var stateful_runner = try buildLocalStageRunnerPlanRef(allocator, .{
        .stage_plan = &stateful_fixture.plan,
        .tensor_frame_plan_ref = &stateful_fixture.plan_ref,
        .placement_plan = &stateful_fixture.placement,
        .state_ownership_plan = if (stateful_fixture.state_plan) |*value| value else null,
    });
    defer stateful_runner.deinit();
    try std.testing.expect(!localStageRunnerPlanIdEql(local_runner.plan_id, stateful_runner.plan_id));
}

test "validateLocalStageRunnerPlanRef rejects tampered adjacent host facts" {
    const allocator = std.testing.allocator;
    var fixture = try TestFixture.init(allocator, &.{2}, .local_in_process, false);
    defer fixture.deinit();
    var runner = try buildLocalStageRunnerPlanRef(allocator, .{
        .stage_plan = &fixture.plan,
        .tensor_frame_plan_ref = &fixture.plan_ref,
        .placement_plan = &fixture.placement,
    });
    defer runner.deinit();

    const mutable_boundaries: []LocalStageRunnerBoundaryRef = @constCast(runner.boundaries);
    mutable_boundaries[0].source_host_id = testHostId(99);
    runner.plan_id = computeLocalStageRunnerPlanId(&runner);
    try std.testing.expectError(error.StageRunnerPlanIdentityMismatch, validateLocalStageRunnerPlanRef(&runner));
}

test "buildDecodeActivationMetadata buildPrefillActivationMetadata hostActivationByteImage segmentedHostActivationByteImage deviceActivationByteImage buildActivationTransportContract builds bridge handoff facts" {
    const allocator = std.testing.allocator;
    var fixture = try TestFixture.init(allocator, &.{2}, .local_in_process, false);
    defer fixture.deinit();
    const slot_request_ids = [_]?u64{ 101, 202, 303 };

    const slots = [_]usize{ 0, 2 };
    const positions = [_]usize{ 7, 9 };
    var decode_entries: [2]tensor_frame.TensorFrameBatchEntry = undefined;
    const decode_metadata = try buildDecodeActivationMetadata(.{
        .plan_ref = &fixture.plan_ref,
        .hidden_size = 8,
        .boundary_index = 0,
        .dtype = .f32,
        .layout = .row_major,
        .location_hint = .{ .cpu = {} },
        .slot_request_ids = &slot_request_ids,
        .slot_indices = &slots,
        .positions = &positions,
        .batch_entries = decode_entries[0..],
    });
    try std.testing.expectEqual(tensor_frame.TensorFrameStepKind.decode, decode_metadata.step_kind);
    try std.testing.expectEqual(@as(usize, 2), decode_metadata.batch.entries.len);
    try std.testing.expectEqual(@as(u64, 101), decode_metadata.batch.entries[0].request_id);
    try std.testing.expectEqual(@as(u64, 303), decode_metadata.batch.entries[1].request_id);

    var row0 = [_]u8{1} ** (@sizeOf(f32) * 8);
    var row1 = [_]u8{2} ** (@sizeOf(f32) * 8);
    const segments = [_][]const u8{ row0[0..], row1[0..] };
    const segmented = segmentedHostActivationByteImage(&decode_metadata, &segments);
    try boundary_byte_image.validateBoundaryByteImage(&segmented, .{});

    var prefill_entries: [1]tensor_frame.TensorFrameBatchEntry = undefined;
    const prefill_metadata = try buildPrefillActivationMetadata(.{
        .plan_ref = &fixture.plan_ref,
        .hidden_size = 8,
        .boundary_index = 0,
        .dtype = .f32,
        .layout = .row_major,
        .location_hint = .{ .cpu = {} },
        .slot_request_ids = &slot_request_ids,
        .slot_index = 1,
        .sequence_start = 4,
        .token_count = 3,
        .batch_entries = prefill_entries[0..],
    });
    try std.testing.expectEqual(tensor_frame.TensorFrameStepKind.prefill, prefill_metadata.step_kind);
    try std.testing.expectEqual(@as(u64, 202), prefill_metadata.batch.entries[0].request_id);

    var contiguous = [_]u8{0x5a} ** (@sizeOf(f32) * 8 * 3);
    const host_image = hostActivationByteImage(&prefill_metadata, contiguous[0..]);
    const device_image = deviceActivationByteImage(&prefill_metadata);
    try std.testing.expectEqual(boundary_byte_image.BoundaryByteImageReadiness.host_readable_now, host_image.readiness);
    try std.testing.expectEqual(boundary_byte_image.BoundaryByteImageReadiness.device_download_required, device_image.readiness);

    const contract = try buildActivationTransportContract(&fixture.placement, &prefill_metadata, &host_image, false, false);
    try std.testing.expectEqual(stage_transfer_mode.StageTransferMode.copy_in_process, contract.decision.mode);
    try std.testing.expectEqual(stage_transport.StageTransportActivationScope.single_entry_header, contract.envelope.activation_scope.?);
}

test "executeLocalStageBoundary rejects boundary execution for one stage runner" {
    const allocator = std.testing.allocator;
    var single_fixture = try TestFixture.init(allocator, &.{}, .local_in_process, false);
    defer single_fixture.deinit();
    var runner = try buildLocalStageRunnerPlanRef(allocator, .{
        .stage_plan = &single_fixture.plan,
        .tensor_frame_plan_ref = &single_fixture.plan_ref,
        .placement_plan = &single_fixture.placement,
    });
    defer runner.deinit();
    try validateLocalStageRunnerPlanRef(&runner);
    try std.testing.expectEqual(@as(usize, 1), runner.stages.len);
    try std.testing.expectEqual(@as(usize, 0), runner.boundaries.len);

    var frame_fixture = try TestFixture.init(allocator, &.{2}, .local_in_process, false);
    defer frame_fixture.deinit();
    var source_trace = MockTrace{};
    var target_trace = MockTrace{};
    var staging: [32]u8 align(64) = undefined;
    var touched: [2]LocalStageTouchedRef = undefined;
    const frame = try testDecodeFrame(&frame_fixture.plan_ref, 0, 1);
    try expectFailureSource(error.BoundaryIndexOutOfRange, try executeLocalStageBoundary(MockStage, MockStage, .{ .trace = &source_trace }, .{ .trace = &target_trace }, .{
        .plan_ref = &runner,
        .placement_plan = &frame_fixture.placement,
        .step = .{ .boundary_index = 0, .step_kind = .decode, .metadata = frame, .activation_byte_count = @sizeOf(f32) * 8 },
        .host_staging = staging[0..],
        .touched_stage_scratch = &touched,
    }));
}

test "executeLocalStageBoundary returns runner plan validation failures as failure results" {
    const allocator = std.testing.allocator;
    var fixture = try TestFixture.init(allocator, &.{2}, .local_in_process, false);
    defer fixture.deinit();
    var runner = try buildLocalStageRunnerPlanRef(allocator, .{
        .stage_plan = &fixture.plan,
        .tensor_frame_plan_ref = &fixture.plan_ref,
        .placement_plan = &fixture.placement,
    });
    defer runner.deinit();

    var source_trace = MockTrace{};
    var target_trace = MockTrace{};
    var staging: [32]u8 align(64) = undefined;
    var touched: [2]LocalStageTouchedRef = undefined;
    const frame = try testDecodeFrame(&fixture.plan_ref, 0, 1);

    const original_version = runner.version;
    runner.version = local_stage_runner_contract_version + 1;
    try expectFailureSourceFrameId(error.InvalidLocalStageRunnerContractVersion, frame.frame_id, try executeLocalStageBoundary(MockStage, MockStage, .{ .trace = &source_trace }, .{ .trace = &target_trace }, .{
        .plan_ref = &runner,
        .placement_plan = &fixture.placement,
        .step = .{ .boundary_index = 0, .step_kind = .decode, .metadata = frame, .activation_byte_count = @sizeOf(f32) * 8 },
        .host_staging = staging[0..],
        .touched_stage_scratch = &touched,
    }));

    runner.version = original_version;
    const mutable_boundaries: []LocalStageRunnerBoundaryRef = @constCast(runner.boundaries);
    mutable_boundaries[0].boundary_index = 1;
    runner.plan_id = computeLocalStageRunnerPlanId(&runner);
    try expectFailureSourceFrameId(error.DuplicateBoundaryRef, frame.frame_id, try executeLocalStageBoundary(MockStage, MockStage, .{ .trace = &source_trace }, .{ .trace = &target_trace }, .{
        .plan_ref = &runner,
        .placement_plan = &fixture.placement,
        .step = .{ .boundary_index = 0, .step_kind = .decode, .metadata = frame, .activation_byte_count = @sizeOf(f32) * 8 },
        .host_staging = staging[0..],
        .touched_stage_scratch = &touched,
    }));

    try std.testing.expectEqual(@as(usize, 0), source_trace.execute_calls);
    try std.testing.expectEqual(@as(usize, 0), target_trace.execute_calls);
}

test "executeLocalStageBoundary executes mock handoff through host staging and preserves adapter ownership" {
    const allocator = std.testing.allocator;
    var fixture = try TestFixture.init(allocator, &.{2}, .mock, false);
    defer fixture.deinit();
    var runner = try buildLocalStageRunnerPlanRef(allocator, .{
        .stage_plan = &fixture.plan,
        .tensor_frame_plan_ref = &fixture.plan_ref,
        .placement_plan = &fixture.placement,
    });
    defer runner.deinit();
    var source_trace = MockTrace{};
    var target_trace = MockTrace{};
    var staging: [32]u8 align(64) = undefined;
    var touched: [2]LocalStageTouchedRef = undefined;
    const frame = try testDecodeFrame(&fixture.plan_ref, 0, 1);

    const result = try executeLocalStageBoundary(MockStage, MockStage, .{ .trace = &source_trace }, .{ .trace = &target_trace }, .{
        .plan_ref = &runner,
        .placement_plan = &fixture.placement,
        .step = .{
            .boundary_index = 0,
            .step_kind = .decode,
            .metadata = frame,
            .activation_byte_count = @sizeOf(f32) * 8,
            .expected_request_id = 123,
            .expected_slot_id = 7,
        },
        .host_staging = staging[0..],
        .touched_stage_scratch = &touched,
    });

    try std.testing.expectEqual(LocalStageRunResult{ .success = .{ .boundary_index = 0, .request_id = 123, .slot_id = 7 } }, result);
    try std.testing.expectEqual(@as(usize, 1), source_trace.execute_calls);
    try std.testing.expectEqual(@as(usize, 1), source_trace.synchronize_calls);
    try std.testing.expectEqual(@as(usize, 1), source_trace.download_calls);
    try std.testing.expectEqual(@as(usize, 1), target_trace.upload_calls);
    try std.testing.expectEqual(@as(usize, 1), target_trace.execute_calls);
    try std.testing.expectEqual(@as(usize, 0), source_trace.deinit_calls);
    try std.testing.expectEqual(@as(usize, 0), target_trace.deinit_calls);
}

test "localStageAdapter executeLocalStageChain executes bridge boundary" {
    const allocator = std.testing.allocator;
    var fixture = try TestFixture.init(allocator, &.{2}, .local_in_process, false);
    defer fixture.deinit();
    var runner = try buildLocalStageRunnerPlanRef(allocator, .{
        .stage_plan = &fixture.plan,
        .tensor_frame_plan_ref = &fixture.plan_ref,
        .placement_plan = &fixture.placement,
    });
    defer runner.deinit();

    var source_trace = MockTrace{};
    var target_trace = MockTrace{};
    var source = MockStage{ .trace = &source_trace };
    var target = MockStage{ .trace = &target_trace };
    var payload = [_]u8{0x5a} ** (@sizeOf(f32) * 8);
    const frame = try testDecodeFrame(&fixture.plan_ref, 0, 1);
    const image = hostActivationByteImage(&frame, payload[0..]);
    var stages = [_]LocalStageChainStage{
        localStageAdapter(MockStage, frame.boundary.source_stage_id, &source),
        localStageAdapter(MockStage, frame.boundary.target_stage_id, &target),
    };
    const boundaries = [_]LocalStageChainBoundaryStep{.{
        .boundary_index = frame.boundary.boundary_index,
        .step_kind = .decode,
        .metadata = &frame,
        .image = &image,
        .allow_borrow = false,
    }};

    try executeLocalStageChain(.{
        .allocator = allocator,
        .plan_ref = &runner,
        .placement_plan = &fixture.placement,
        .stages = stages[0..],
        .boundaries = boundaries[0..],
    });

    try std.testing.expectEqual(@as(usize, 1), source_trace.execute_calls);
    try std.testing.expectEqual(@as(usize, 1), source_trace.synchronize_calls);
    try std.testing.expectEqual(@as(usize, 0), source_trace.download_calls);
    try std.testing.expectEqual(@as(usize, 1), target_trace.upload_calls);
    try std.testing.expectEqual(@as(usize, 1), target_trace.execute_calls);
}

test "captureLocalStageExecutionFailure records staged cleanup metadata and preserves source error" {
    const allocator = std.testing.allocator;
    var fixture = try TestFixture.init(allocator, &.{2}, .local_in_process, false);
    defer fixture.deinit();
    var frame = try testDecodeFrame(&fixture.plan_ref, 0, 1);

    var report = try captureLocalStageExecutionFailure(allocator, .{
        .placement_plan = &fixture.placement,
        .metadata = &frame,
        .active_stage_id = 1,
        .source_error = error.StageExecutionFailed,
    });
    defer report.deinit();

    try std.testing.expectEqual(error.StageExecutionFailed, report.source_error);
    try std.testing.expectEqual(@as(usize, 1), report.entries.len);
    const entry = report.entries[0];
    try std.testing.expectEqual(staged_error.StagedFailureKind.stage_execution_failed, entry.primary_failure.kind);
    try std.testing.expectEqual(staged_error.StagedFailurePhase.stage_execution_after_state_mutation, entry.primary_failure.phase);
    try std.testing.expectEqual(staged_error.StagedFailureScope.stage, entry.primary_failure.scope);
    try std.testing.expectEqual(@as(usize, 1), entry.primary_failure.context.stage_id.?);
    try std.testing.expectEqual(@as(usize, 1), entry.touched_stages.len);
    try std.testing.expectEqual(@as(usize, 1), entry.touched_stages[0].stage_id);
    try std.testing.expectEqual(staged_error.StagedFailureKind.stage_execution_failed, entry.error_report.primary_failure.kind);
}

test "preserveLocalStageExecutionError returns original stage error" {
    const allocator = std.testing.allocator;
    var fixture = try TestFixture.init(allocator, &.{2}, .local_in_process, false);
    defer fixture.deinit();
    var frame = try testDecodeFrame(&fixture.plan_ref, 0, 1);

    const err = preserveLocalStageExecutionError(null, .{
        .placement_plan = &fixture.placement,
        .metadata = &frame,
        .active_stage_id = 0,
        .source_error = error.RequestCancelled,
    });
    try std.testing.expectEqual(error.RequestCancelled, err);
}

test "executeLocalStageLayers reports failure metadata without replacing adapter error" {
    const FailingStage = struct {
        pub fn executeLayers(_: *@This(), _: []const u8, _: usize, _: usize) anyerror!void {
            return error.RequestCancelled;
        }
    };
    const allocator = std.testing.allocator;
    var fixture = try TestFixture.init(allocator, &.{2}, .local_in_process, false);
    defer fixture.deinit();
    var frame = try testDecodeFrame(&fixture.plan_ref, 0, 1);
    var stage = FailingStage{};

    try std.testing.expectError(
        error.RequestCancelled,
        executeLocalStageLayers(
            allocator,
            null,
            &fixture.placement,
            &frame,
            0,
            FailingStage,
            &stage,
            &.{},
            0,
            2,
        ),
    );
}

test "executeLocalStageBoundary accepts three stage plans and executes nonzero producer boundary" {
    const allocator = std.testing.allocator;
    var fixture = try TestFixture.init(allocator, &.{ 1, 3 }, .local_in_process, false);
    defer fixture.deinit();
    var runner = try buildLocalStageRunnerPlanRef(allocator, .{
        .stage_plan = &fixture.plan,
        .tensor_frame_plan_ref = &fixture.plan_ref,
        .placement_plan = &fixture.placement,
    });
    defer runner.deinit();
    try std.testing.expectEqual(@as(usize, 3), runner.stages.len);
    try std.testing.expectEqual(@as(usize, 2), runner.boundaries.len);

    var source_trace = MockTrace{};
    var target_trace = MockTrace{};
    var staging: [32]u8 align(64) = undefined;
    var touched: [2]LocalStageTouchedRef = undefined;
    const frame = try testDecodeFrame(&fixture.plan_ref, 1, 1);
    const result = try executeLocalStageBoundary(MockStage, MockStage, .{ .trace = &source_trace }, .{ .trace = &target_trace }, .{
        .plan_ref = &runner,
        .placement_plan = &fixture.placement,
        .step = .{
            .boundary_index = 1,
            .step_kind = .decode,
            .metadata = frame,
            .activation_byte_count = @sizeOf(f32) * 8,
        },
        .host_staging = staging[0..],
        .touched_stage_scratch = &touched,
    });
    try std.testing.expect(result == .success);
    try std.testing.expectEqual(@as(usize, 1), source_trace.last_layer_start);
    try std.testing.expectEqual(@as(usize, 3), source_trace.last_layer_end);
    try std.testing.expectEqual(@as(usize, 3), target_trace.last_layer_start);
    try std.testing.expectEqual(@as(usize, 4), target_trace.last_layer_end);
}

test "executeLocalStageBoundary validates frame from copied runner facts without original plan ref" {
    const allocator = std.testing.allocator;
    var fixture = try TestFixture.init(allocator, &.{2}, .local_in_process, false);
    defer fixture.deinit();
    var runner = try buildLocalStageRunnerPlanRef(allocator, .{
        .stage_plan = &fixture.plan,
        .tensor_frame_plan_ref = &fixture.plan_ref,
        .placement_plan = &fixture.placement,
    });
    const frame = try testDecodeFrame(&fixture.plan_ref, 0, 1);
    defer runner.deinit();

    var source_trace = MockTrace{};
    var target_trace = MockTrace{};
    var staging: [32]u8 align(64) = undefined;
    var touched: [2]LocalStageTouchedRef = undefined;
    const result = try executeLocalStageBoundary(MockStage, MockStage, .{ .trace = &source_trace }, .{ .trace = &target_trace }, .{
        .plan_ref = &runner,
        .placement_plan = &fixture.placement,
        .step = .{ .boundary_index = 0, .step_kind = .decode, .metadata = frame, .activation_byte_count = @sizeOf(f32) * 8 },
        .host_staging = staging[0..],
        .touched_stage_scratch = &touched,
    });
    try std.testing.expect(result == .success);
}

test "executeLocalStageBoundary enforces selected boundary profile facts before source execute" {
    const allocator = std.testing.allocator;
    var fixture = try TestFixture.init(allocator, &.{2}, .local_in_process, false);
    defer fixture.deinit();
    var runner = try buildLocalStageRunnerPlanRef(allocator, .{
        .stage_plan = &fixture.plan,
        .tensor_frame_plan_ref = &fixture.plan_ref,
        .placement_plan = &fixture.placement,
    });
    defer runner.deinit();
    var source_trace = MockTrace{};
    var target_trace = MockTrace{};
    var staging: [32]u8 align(64) = undefined;
    var touched: [2]LocalStageTouchedRef = undefined;
    const frame = try testDecodeFrame(&fixture.plan_ref, 0, 1);
    const mutable_profiles: []LocalStageRunnerBoundaryProfile = @constCast(runner.boundaries[0].profiles);
    const original_profile = mutable_profiles[0];

    var wrong_step = frame;
    wrong_step.step_kind = .prefill;
    wrong_step.shape_context.expected_step_kind = .prefill;
    try expectFailureSourceFrameId(error.InvalidStepKind, wrong_step.frame_id, try executeLocalStageBoundary(MockStage, MockStage, .{ .trace = &source_trace }, .{ .trace = &target_trace }, .{
        .plan_ref = &runner,
        .placement_plan = &fixture.placement,
        .step = .{ .boundary_index = 0, .step_kind = .decode, .metadata = wrong_step, .activation_byte_count = @sizeOf(f32) * 8 },
        .host_staging = staging[0..],
        .touched_stage_scratch = &touched,
    }));

    mutable_profiles[0] = original_profile;
    mutable_profiles[0].dtype = .f16;
    runner.plan_id = computeLocalStageRunnerPlanId(&runner);
    try expectFailureSourceFrameId(error.InvalidDType, frame.frame_id, try executeLocalStageBoundary(MockStage, MockStage, .{ .trace = &source_trace }, .{ .trace = &target_trace }, .{
        .plan_ref = &runner,
        .placement_plan = &fixture.placement,
        .step = .{ .boundary_index = 0, .step_kind = .decode, .metadata = frame, .activation_byte_count = @sizeOf(f32) * 8 },
        .host_staging = staging[0..],
        .touched_stage_scratch = &touched,
    }));

    mutable_profiles[0] = original_profile;
    mutable_profiles[0].layout = @enumFromInt(1);
    runner.plan_id = computeLocalStageRunnerPlanId(&runner);
    try expectFailureSourceFrameId(error.UnsupportedTensorLayout, frame.frame_id, try executeLocalStageBoundary(MockStage, MockStage, .{ .trace = &source_trace }, .{ .trace = &target_trace }, .{
        .plan_ref = &runner,
        .placement_plan = &fixture.placement,
        .step = .{ .boundary_index = 0, .step_kind = .decode, .metadata = frame, .activation_byte_count = @sizeOf(f32) * 8 },
        .host_staging = staging[0..],
        .touched_stage_scratch = &touched,
    }));

    mutable_profiles[0] = original_profile;
    mutable_profiles[0].max_batch_entries = 0;
    runner.plan_id = computeLocalStageRunnerPlanId(&runner);
    try expectFailureSourceFrameId(error.InvalidBatch, frame.frame_id, try executeLocalStageBoundary(MockStage, MockStage, .{ .trace = &source_trace }, .{ .trace = &target_trace }, .{
        .plan_ref = &runner,
        .placement_plan = &fixture.placement,
        .step = .{ .boundary_index = 0, .step_kind = .decode, .metadata = frame, .activation_byte_count = @sizeOf(f32) * 8 },
        .host_staging = staging[0..],
        .touched_stage_scratch = &touched,
    }));

    mutable_profiles[0] = original_profile;
    mutable_profiles[0].max_token_count_per_frame = 0;
    runner.plan_id = computeLocalStageRunnerPlanId(&runner);
    try expectFailureSourceFrameId(error.InvalidSequenceRange, frame.frame_id, try executeLocalStageBoundary(MockStage, MockStage, .{ .trace = &source_trace }, .{ .trace = &target_trace }, .{
        .plan_ref = &runner,
        .placement_plan = &fixture.placement,
        .step = .{ .boundary_index = 0, .step_kind = .decode, .metadata = frame, .activation_byte_count = @sizeOf(f32) * 8 },
        .host_staging = staging[0..],
        .touched_stage_scratch = &touched,
    }));

    mutable_profiles[0] = original_profile;
    mutable_profiles[0].max_activation_payload_bytes = frame.payload.byte_count - 1;
    runner.plan_id = computeLocalStageRunnerPlanId(&runner);
    try expectFailureSourceFrameId(error.InvalidPayloadByteCount, frame.frame_id, try executeLocalStageBoundary(MockStage, MockStage, .{ .trace = &source_trace }, .{ .trace = &target_trace }, .{
        .plan_ref = &runner,
        .placement_plan = &fixture.placement,
        .step = .{ .boundary_index = 0, .step_kind = .decode, .metadata = frame, .activation_byte_count = @sizeOf(f32) * 8 },
        .host_staging = staging[0..],
        .touched_stage_scratch = &touched,
    }));

    try std.testing.expectEqual(@as(usize, 0), source_trace.execute_calls);
    try std.testing.expectEqual(@as(usize, 0), target_trace.execute_calls);
}

test "executeLocalStageBoundary returns exact source_error for validation failures before source execute" {
    const allocator = std.testing.allocator;
    var fixture = try TestFixture.init(allocator, &.{2}, .local_in_process, false);
    defer fixture.deinit();
    var runner = try buildLocalStageRunnerPlanRef(allocator, .{
        .stage_plan = &fixture.plan,
        .tensor_frame_plan_ref = &fixture.plan_ref,
        .placement_plan = &fixture.placement,
    });
    defer runner.deinit();
    var source_trace = MockTrace{};
    var target_trace = MockTrace{};
    var staging: [32]u8 align(64) = undefined;
    var touched: [2]LocalStageTouchedRef = undefined;
    var frame = try testDecodeFrame(&fixture.plan_ref, 0, 1);

    try expectFailureSourceFrameId(error.LocalStageTransferNotInitialized, frame.frame_id, try executeLocalStageBoundary(MockStage, MockStage, .{ .trace = &source_trace }, .{ .trace = &target_trace }, .{
        .plan_ref = &runner,
        .placement_plan = &fixture.placement,
        .step = .{ .boundary_index = 0, .step_kind = .decode, .metadata = frame, .activation_byte_count = @sizeOf(f32) * 8 },
        .host_staging = null,
        .touched_stage_scratch = &touched,
    }));
    try expectFailureSource(error.LocalStageTransferBufferTooSmall, try executeLocalStageBoundary(MockStage, MockStage, .{ .trace = &source_trace }, .{ .trace = &target_trace }, .{
        .plan_ref = &runner,
        .placement_plan = &fixture.placement,
        .step = .{ .boundary_index = 0, .step_kind = .decode, .metadata = frame, .activation_byte_count = @sizeOf(f32) * 8 },
        .host_staging = staging[0..4],
        .touched_stage_scratch = &touched,
    }));
    try expectFailureSourceFrameId(error.InvalidRequestId, frame.frame_id, try executeLocalStageBoundary(MockStage, MockStage, .{ .trace = &source_trace }, .{ .trace = &target_trace }, .{
        .plan_ref = &runner,
        .placement_plan = &fixture.placement,
        .step = .{ .boundary_index = 0, .step_kind = .decode, .metadata = frame, .activation_byte_count = @sizeOf(f32) * 8, .expected_request_id = 999 },
        .host_staging = staging[0..],
        .touched_stage_scratch = &touched,
    }));
    try expectFailureSource(error.BoundaryIndexOutOfRange, try executeLocalStageBoundary(MockStage, MockStage, .{ .trace = &source_trace }, .{ .trace = &target_trace }, .{
        .plan_ref = &runner,
        .placement_plan = &fixture.placement,
        .step = .{ .boundary_index = 99, .step_kind = .decode, .metadata = frame, .activation_byte_count = @sizeOf(f32) * 8 },
        .host_staging = staging[0..],
        .touched_stage_scratch = &touched,
    }));
    try expectFailureSource(error.MissingBoundaryFrameProfile, try executeLocalStageBoundary(MockStage, MockStage, .{ .trace = &source_trace }, .{ .trace = &target_trace }, .{
        .plan_ref = &runner,
        .placement_plan = &fixture.placement,
        .step = .{ .boundary_index = 0, .step_kind = .prefill, .metadata = frame, .activation_byte_count = @sizeOf(f32) * 8 },
        .host_staging = staging[0..],
        .touched_stage_scratch = &touched,
    }));
    try expectFailureSource(error.PayloadBufferLengthMismatch, try executeLocalStageBoundary(MockStage, MockStage, .{ .trace = &source_trace }, .{ .trace = &target_trace }, .{
        .plan_ref = &runner,
        .placement_plan = &fixture.placement,
        .step = .{ .boundary_index = 0, .step_kind = .decode, .metadata = frame, .activation_byte_count = @sizeOf(f32) * 8 + 4 },
        .host_staging = staging[0..],
        .touched_stage_scratch = &touched,
    }));
    frame.batch = .{ .entries = &.{} };
    try expectFailureSource(error.InvalidBatch, try executeLocalStageBoundary(MockStage, MockStage, .{ .trace = &source_trace }, .{ .trace = &target_trace }, .{
        .plan_ref = &runner,
        .placement_plan = &fixture.placement,
        .step = .{ .boundary_index = 0, .step_kind = .decode, .metadata = frame, .activation_byte_count = @sizeOf(f32) * 8 },
        .host_staging = staging[0..],
        .touched_stage_scratch = &touched,
    }));
    frame = try testDecodeFrame(&fixture.plan_ref, 0, 2);
    try expectFailureSource(error.InvalidBatch, try executeLocalStageBoundary(MockStage, MockStage, .{ .trace = &source_trace }, .{ .trace = &target_trace }, .{
        .plan_ref = &runner,
        .placement_plan = &fixture.placement,
        .step = .{ .boundary_index = 0, .step_kind = .decode, .metadata = frame, .activation_byte_count = @as(usize, @intCast(frame.payload.byte_count)) },
        .host_staging = staging[0..],
        .touched_stage_scratch = &touched,
    }));
    try std.testing.expectEqual(@as(usize, 0), source_trace.execute_calls);
    try std.testing.expectEqual(@as(usize, 0), target_trace.execute_calls);
}

test "executeLocalStageBoundary reports transfer and target mutation touched stages" {
    const allocator = std.testing.allocator;
    var fixture = try TestFixture.init(allocator, &.{2}, .local_in_process, false);
    defer fixture.deinit();
    var runner = try buildLocalStageRunnerPlanRef(allocator, .{
        .stage_plan = &fixture.plan,
        .tensor_frame_plan_ref = &fixture.plan_ref,
        .placement_plan = &fixture.placement,
    });
    defer runner.deinit();
    var source_trace = MockTrace{};
    var target_trace = MockTrace{};
    var staging: [32]u8 align(64) = undefined;
    var touched: [2]LocalStageTouchedRef = undefined;
    const frame = try testDecodeFrame(&fixture.plan_ref, 0, 1);

    const transfer_result = try executeLocalStageBoundary(MockStage, MockStage, .{ .trace = &source_trace, .fail_download = error.DiskQuota }, .{ .trace = &target_trace }, .{
        .plan_ref = &runner,
        .placement_plan = &fixture.placement,
        .step = .{ .boundary_index = 0, .step_kind = .decode, .metadata = frame, .activation_byte_count = @sizeOf(f32) * 8 },
        .host_staging = staging[0..],
        .touched_stage_scratch = &touched,
    });
    try expectFailureSourceFrameId(error.DiskQuota, frame.frame_id, transfer_result);
    try std.testing.expectEqual(@as(usize, 1), transfer_result.failure.touched_stages.len);
    try std.testing.expect(transfer_result.failure.touched_stages[0].execution_started);

    source_trace = .{};
    target_trace = .{};
    const upload_result = try executeLocalStageBoundary(MockStage, MockStage, .{ .trace = &source_trace }, .{ .trace = &target_trace, .fail_upload = error.AccessDenied }, .{
        .plan_ref = &runner,
        .placement_plan = &fixture.placement,
        .step = .{ .boundary_index = 0, .step_kind = .decode, .metadata = frame, .activation_byte_count = @sizeOf(f32) * 8 },
        .host_staging = staging[0..],
        .touched_stage_scratch = &touched,
    });
    try expectFailureSourceFrameId(error.AccessDenied, frame.frame_id, upload_result);
    try std.testing.expectEqual(@as(usize, 2), upload_result.failure.touched_stages.len);
    try std.testing.expect(upload_result.failure.touched_stages[1].receiver_payload_mutation_started);
}

test "executeLocalStageBoundary classifies out of memory and transfer cancellation before generic failures" {
    const allocator = std.testing.allocator;
    var fixture = try TestFixture.init(allocator, &.{2}, .local_in_process, false);
    defer fixture.deinit();
    var runner = try buildLocalStageRunnerPlanRef(allocator, .{
        .stage_plan = &fixture.plan,
        .tensor_frame_plan_ref = &fixture.plan_ref,
        .placement_plan = &fixture.placement,
    });
    defer runner.deinit();
    var source_trace = MockTrace{};
    var target_trace = MockTrace{};
    var staging: [32]u8 align(64) = undefined;
    var touched: [2]LocalStageTouchedRef = undefined;
    const frame = try testDecodeFrame(&fixture.plan_ref, 0, 1);

    const stage_oom = try executeLocalStageBoundary(MockStage, MockStage, .{ .trace = &source_trace, .fail_execute = error.OutOfMemory }, .{ .trace = &target_trace }, .{
        .plan_ref = &runner,
        .placement_plan = &fixture.placement,
        .step = .{ .boundary_index = 0, .step_kind = .decode, .metadata = frame, .activation_byte_count = @sizeOf(f32) * 8 },
        .host_staging = staging[0..],
        .touched_stage_scratch = &touched,
    });
    try expectFailureClassification(error.OutOfMemory, .resource_exhausted, .stage_execution_after_state_mutation, .stage, .runner, stage_oom);
    try std.testing.expectEqual(@as(?usize, 0), stage_oom.failure.primary_failure.context.stage_id);

    source_trace = .{};
    target_trace = .{};
    const transport_cancelled = try executeLocalStageBoundary(MockStage, MockStage, .{ .trace = &source_trace, .fail_synchronize = error.RequestCancelled }, .{ .trace = &target_trace }, .{
        .plan_ref = &runner,
        .placement_plan = &fixture.placement,
        .step = .{ .boundary_index = 0, .step_kind = .decode, .metadata = frame, .activation_byte_count = @sizeOf(f32) * 8 },
        .host_staging = staging[0..],
        .touched_stage_scratch = &touched,
    });
    try expectFailureClassification(error.RequestCancelled, .request_cancelled, .frame_handoff, .request, .transport, transport_cancelled);

    source_trace = .{};
    target_trace = .{};
    const transport_oom = try executeLocalStageBoundary(MockStage, MockStage, .{ .trace = &source_trace, .fail_download = error.OutOfMemory }, .{ .trace = &target_trace }, .{
        .plan_ref = &runner,
        .placement_plan = &fixture.placement,
        .step = .{ .boundary_index = 0, .step_kind = .decode, .metadata = frame, .activation_byte_count = @sizeOf(f32) * 8 },
        .host_staging = staging[0..],
        .touched_stage_scratch = &touched,
    });
    try expectFailureClassification(error.OutOfMemory, .resource_exhausted, .frame_handoff, .transport, .transport, transport_oom);
}

test "executeLocalStageBoundary preserves observer best effort and strict modes" {
    const allocator = std.testing.allocator;
    var fixture = try TestFixture.init(allocator, &.{2}, .local_in_process, false);
    defer fixture.deinit();
    var runner = try buildLocalStageRunnerPlanRef(allocator, .{
        .stage_plan = &fixture.plan,
        .tensor_frame_plan_ref = &fixture.plan_ref,
        .placement_plan = &fixture.placement,
    });
    defer runner.deinit();
    var source_trace = MockTrace{};
    var target_trace = MockTrace{};
    var staging: [32]u8 align(64) = undefined;
    var touched: [2]LocalStageTouchedRef = undefined;
    const frame = try testDecodeFrame(&fixture.plan_ref, 0, 1);
    const observer = tensor_frame.TensorFrameObserver{ .emit_fn = failingObserver };

    const best_effort = try executeLocalStageBoundary(MockStage, MockStage, .{ .trace = &source_trace }, .{ .trace = &target_trace }, .{
        .plan_ref = &runner,
        .placement_plan = &fixture.placement,
        .step = .{ .boundary_index = 0, .step_kind = .decode, .metadata = frame, .activation_byte_count = @sizeOf(f32) * 8, .observer = observer },
        .host_staging = staging[0..],
        .touched_stage_scratch = &touched,
    });
    try std.testing.expect(best_effort == .success);

    source_trace = .{};
    target_trace = .{};
    const strict = try executeLocalStageBoundary(MockStage, MockStage, .{ .trace = &source_trace }, .{ .trace = &target_trace }, .{
        .plan_ref = &runner,
        .placement_plan = &fixture.placement,
        .step = .{ .boundary_index = 0, .step_kind = .decode, .metadata = frame, .activation_byte_count = @sizeOf(f32) * 8, .observer = observer, .observer_mode = .strict },
        .host_staging = staging[0..],
        .touched_stage_scratch = &touched,
    });
    try expectFailureSource(error.ObserverFailure, strict);
    try std.testing.expectEqual(staged_error.StagedFailurePhase.frame_handoff, strict.failure.primary_failure.phase);
    try std.testing.expectEqual(@as(usize, 2), strict.failure.touched_stages.len);
}

test "buildLocalStageRunnerPlanRef validates supplied state ownership identity and descriptor free placement" {
    const allocator = std.testing.allocator;
    var fixture = try TestFixture.init(allocator, &.{2}, .local_in_process, true);
    defer fixture.deinit();
    var runner = try buildLocalStageRunnerPlanRef(allocator, .{
        .stage_plan = &fixture.plan,
        .tensor_frame_plan_ref = &fixture.plan_ref,
        .placement_plan = &fixture.placement,
        .state_ownership_plan = if (fixture.state_plan) |*value| value else null,
    });
    defer runner.deinit();
    try std.testing.expect(runner.state_ownership_plan_id != null);
    try std.testing.expectEqual(@as(usize, 0), fixture.placement.state_stage_summaries[0].descriptors.len);
    const frame = try testDecodeFrame(&fixture.plan_ref, 0, 1);
    try std.testing.expectEqual(@as(u64, @sizeOf(f32) * 8), frame.payload.byte_count);

    try std.testing.expectError(error.MissingStatePlacementSummary, buildLocalStageRunnerPlanRef(allocator, .{
        .stage_plan = &fixture.plan,
        .tensor_frame_plan_ref = &fixture.plan_ref,
        .placement_plan = &fixture.placement,
    }));
}

fn expectFailureSource(expected: anyerror, result: LocalStageRunResult) !void {
    try std.testing.expect(result == .failure);
    try std.testing.expect(result.failure.source_error != null);
    try std.testing.expectEqual(expected, result.failure.source_error.?);
}

fn expectFailureSourceFrameId(
    expected: anyerror,
    expected_frame_id: tensor_frame.TensorFrameInstanceId,
    result: LocalStageRunResult,
) !void {
    try expectFailureSource(expected, result);
    const actual_frame_id = result.failure.primary_failure.context.tensor_frame_id;
    try std.testing.expect(actual_frame_id != null);
    try std.testing.expectEqual(expected_frame_id.value, actual_frame_id.?.value);
}

fn expectFailureClassification(
    expected: anyerror,
    expected_kind: staged_error.StagedFailureKind,
    expected_phase: staged_error.StagedFailurePhase,
    expected_scope: staged_error.StagedFailureScope,
    expected_domain: staged_error.StagedSourceDomain,
    result: LocalStageRunResult,
) !void {
    try expectFailureSource(expected, result);
    try std.testing.expectEqual(expected_kind, result.failure.primary_failure.kind);
    try std.testing.expectEqual(expected_phase, result.failure.primary_failure.phase);
    try std.testing.expectEqual(expected_scope, result.failure.primary_failure.scope);
    try std.testing.expectEqual(expected_domain, result.failure.primary_failure.source.domain);
}

fn failingObserver(_: ?*anyopaque, _: *const tensor_frame.TensorFrameMetadata) anyerror!void {
    return error.ObserverSinkFailed;
}

const test_single_batch_entries = [_]tensor_frame.TensorFrameBatchEntry{
    .{ .batch_index = 0, .request_id = 123, .slot_id = 7, .sequence_start = 9, .token_count = 1 },
};

const test_two_batch_entries = [_]tensor_frame.TensorFrameBatchEntry{
    .{ .batch_index = 0, .request_id = 123, .slot_id = 7, .sequence_start = 9, .token_count = 1 },
    .{ .batch_index = 1, .request_id = 124, .slot_id = 8, .sequence_start = 9, .token_count = 1 },
};

fn testDecodeFrame(
    plan_ref: *const tensor_frame.TensorFramePlanRef,
    boundary_index: usize,
    batch_count: u64,
) !tensor_frame.TensorFrameMetadata {
    const contract = try tensor_frame.selectedBoundaryTensorContract(plan_ref, boundary_index, .f32, .row_major, .explicit);
    const tensor = try tensor_frame.TensorFrameTensorDesc.contiguousActivation(.f32, .{ batch_count, 1, 8, 0 });
    const entries: []const tensor_frame.TensorFrameBatchEntry = if (batch_count == 1)
        &test_single_batch_entries
    else
        &test_two_batch_entries;
    return tensor_frame.activationDecodeFrame(.{
        .frame_id = try tensor_frame.TensorFrameInstanceId.init(1 + boundary_index),
        .plan_ref = plan_ref,
        .boundary_index = boundary_index,
        .selected_contract = &contract,
        .shape_context = .{ .expected_hidden_size = 8, .expected_step_kind = .decode },
        .tensor = tensor,
        .batch = .{ .entries = entries[0..@as(usize, @intCast(batch_count))] },
        .payload = .{
            .byte_count = tensor.payload_byte_count,
            .ownership = .borrowed_until_next_stage_call,
            .lifetime = .step_scoped,
        },
    });
}

fn buildDescriptorFreeStatePlan(
    allocator: Allocator,
    plan: *const stage_plan.StagePlan,
) !state_ownership.StageStateOwnershipPlan {
    const descriptor_sets = try allocator.alloc(state_ownership.StageStateDescriptorSet, plan.stages.len);
    defer allocator.free(descriptor_sets);
    for (plan.stages, 0..) |stage, index| {
        descriptor_sets[index] = .{ .stage_id = stage.id, .descriptors = &.{} };
    }
    const facts = [_]state_ownership.StageStatePartitionFact{.{
        .boundary_index = 0,
        .source_stage_id = 0,
        .target_stage_id = 1,
        .reason = .stateful_decoder,
        .ownership_mode = .stage_level_dependency_only,
    }};
    return state_ownership.buildStageStateOwnershipPlan(allocator, .{
        .plan = plan,
        .descriptor_sets = descriptor_sets,
        .partition_facts = &facts,
    });
}

fn buildTestPlacement(
    allocator: Allocator,
    plan: *const stage_plan.StagePlan,
    mode: host_capability.BoundaryHandoffMode,
    state_ref: ?*const host_capability.StageStatePlacementRef,
) !host_capability.PlacementPlan {
    const same_host_direct = mode == .same_host_direct;
    var capabilities: [4]host_capability.HostCapability = undefined;
    var residencies: [4]host_capability.HostResidencySnapshot = undefined;
    var initialized_capabilities: usize = 0;
    var initialized_residencies: usize = 0;
    defer {
        for (capabilities[0..initialized_capabilities]) |*capability| capability.deinit();
        for (residencies[0..initialized_residencies]) |*residency| residency.deinit();
    }

    if (same_host_direct) {
        capabilities[initialized_capabilities] = try buildTestCapability(allocator, testHostId(1), mode);
        initialized_capabilities += 1;
        var residents: [4]host_capability.ResidentStageEntry = undefined;
        for (plan.stages, 0..) |stage, index| {
            residents[index] = residentEntryFromStage(stage, state_ref);
        }
        residencies[initialized_residencies] = try host_capability.buildHostResidencySnapshot(allocator, .{
            .host_id = testHostId(1),
            .plan = plan,
            .state_ownership_contract_version = if (state_ref) |ref| ref.state_ownership_contract_version else null,
            .state_ownership_plan_id = if (state_ref) |ref| ref.state_ownership_plan_id else null,
            .resident_stages = residents[0..plan.stages.len],
        });
        initialized_residencies += 1;
    } else {
        for (plan.stages) |stage| {
            capabilities[initialized_capabilities] = try buildTestCapability(allocator, testHostId(stage.id + 1), mode);
            initialized_capabilities += 1;
            const resident = [_]host_capability.ResidentStageEntry{residentEntryFromStage(stage, state_ref)};
            residencies[initialized_residencies] = try host_capability.buildHostResidencySnapshot(allocator, .{
                .host_id = testHostId(stage.id + 1),
                .plan = plan,
                .state_ownership_contract_version = if (state_ref) |ref| ref.state_ownership_contract_version else null,
                .state_ownership_plan_id = if (state_ref) |ref| ref.state_ownership_plan_id else null,
                .resident_stages = &resident,
            });
            initialized_residencies += 1;
        }
    }

    var bindings: [4]host_capability.StageHostBinding = undefined;
    for (plan.stages, 0..) |stage, index| {
        bindings[index] = .{
            .stage_id = stage.id,
            .host_id = if (same_host_direct) testHostId(1) else testHostId(stage.id + 1),
        };
    }
    var profiles: [8]host_capability.BoundaryFrameProfile = undefined;
    var profile_count: usize = 0;
    for (plan.boundaries, 0..) |boundary, boundary_index| {
        profiles[profile_count] = .{
            .boundary_index = boundary_index,
            .source_stage_id = boundary.source_stage_id,
            .target_stage_id = boundary.target_stage_id,
            .step_kind = .decode,
            .dtype = .f32,
            .max_batch_entries = 4,
            .max_token_count_per_frame = 1,
            .max_activation_payload_bytes = 512,
            .handoff_mode = mode,
        };
        profile_count += 1;
    }
    const stateful = state_ref != null;
    return host_capability.buildPlacementPlan(allocator, .{
        .plan = plan,
        .required_step_kinds = &.{.decode},
        .host_capabilities = capabilities[0..initialized_capabilities],
        .host_residency_snapshots = residencies[0..initialized_residencies],
        .stage_host_bindings = bindings[0..plan.stages.len],
        .boundary_frame_profiles = profiles[0..profile_count],
        .state_placement_mode = if (stateful) .validate_ref else .stateless_only,
        .state_placement_ref = state_ref,
        .allowed_reachability = if (mode == .remote_declared) &.{.remote_declared} else &.{ .local_in_process, .mock },
        .stateful_execution_required = stateful,
    });
}

fn buildTestCapability(
    allocator: Allocator,
    host_id: host_capability.HostId,
    mode: host_capability.BoundaryHandoffMode,
) !host_capability.HostCapability {
    const frames = [_]host_capability.HostFrameCapability{
        .{ .endpoint_role = .producer, .step_kind = .decode, .dtype = .f32, .handoff_mode = mode, .max_batch_entries = 4, .max_token_count_per_frame = 1, .max_activation_payload_bytes = 512 },
        .{ .endpoint_role = .consumer, .step_kind = .decode, .dtype = .f32, .handoff_mode = mode, .max_batch_entries = 4, .max_token_count_per_frame = 1, .max_activation_payload_bytes = 512 },
    };
    return host_capability.buildHostCapability(allocator, .{
        .host_id = host_id,
        .backend_kind = if (mode == .mock) .mock else .cpu,
        .reachability_kind = if (mode == .remote_declared) .remote_declared else .local_in_process,
        .supported_graph_contract_versions = &.{stage_plan.graph_identity_contract_version},
        .supported_stage_plan_contract_versions = &.{stage_plan.stage_plan_contract_version},
        .supported_state_ownership_contract_versions = &.{state_ownership.state_ownership_contract_version},
        .frame_capabilities = &frames,
    });
}

fn residentEntryFromStage(
    stage: stage_plan.StagePlanStage,
    state_ref: ?*const host_capability.StageStatePlacementRef,
) host_capability.ResidentStageEntry {
    var entry = host_capability.ResidentStageEntry{
        .stage_id = stage.id,
        .layer_start = stage.layer_start,
        .layer_end = stage.layer_end,
        .owned_roles = stage.owned_roles,
        .residency = stage.residency,
    };
    if (state_ref) |ref| {
        for (ref.stage_summaries) |summary| {
            if (summary.stage_id != stage.id or !summary.owns_runtime_state) continue;
            entry.state_summary = .{
                .state_ownership_contract_version = ref.state_ownership_contract_version,
                .state_ownership_plan_id = ref.state_ownership_plan_id,
                .stage_id = summary.stage_id,
                .descriptor_count = summary.descriptors.len,
                .descriptors = summary.descriptors,
            };
        }
    }
    return entry;
}

fn testHostId(value: usize) host_capability.HostId {
    return .{ .value = @intCast(value) };
}

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
        .name = "local_stage_runner_test",
        .model_types = &.{"local_stage_runner_test"},
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
        .architecture_id = "local_stage_runner_test",
        .layer_count = layer_count,
        .entries = entries,
        .total_checkpoint_bytes = total_bytes,
        .role_bytes = role_bytes,
    };
}
