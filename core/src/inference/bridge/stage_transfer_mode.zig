//! Transfer-mode selection for a validated staged boundary.
//!
//! This module chooses a transfer mode from placement, tensor-frame, and
//! byte-image facts. It does not execute the selected transfer.

const std = @import("std");
const tensor_frame = @import("tensor_frame.zig");
const host_capability = @import("host_capability.zig");
const boundary_byte_image = @import("boundary_byte_image.zig");
const models = @import("models_pkg");

const Allocator = std.mem.Allocator;
const stage_plan = models.stage_plan;

pub const StageTransferModeError =
    host_capability.PlacementError ||
    tensor_frame.TensorFrameValidationError ||
    boundary_byte_image.BoundaryByteImageError ||
    error{
        MissingStageTransferBoundaryProfile,
        StageTransferMetadataMismatch,
        StageTransferBoundaryMismatch,
        StageTransferTensorProfileMismatch,
        StageTransferPayloadNotHostReadable,
        StageTransferOpaquePayloadNotRemoteReadable,
    };

pub const StageTransferMode = enum(u8) {
    borrow_in_process,
    copy_in_process,
    device_download_then_copy,
    remote_stream,
    device_download_then_remote_stream,
};

pub const StageTransferModeRequest = struct {
    placement_plan: *const host_capability.PlacementPlan,
    metadata: *const tensor_frame.TensorFrameMetadata,
    image: *const boundary_byte_image.BoundaryByteImageRef,
    allow_borrow: bool = true,
};

pub const StageTransferModeDecision = struct {
    mode: StageTransferMode,
    boundary_profile: host_capability.BoundaryFrameProfile,
    source_host_id: host_capability.HostId,
    target_host_id: host_capability.HostId,
};

pub fn chooseStageTransferMode(
    request: StageTransferModeRequest,
) StageTransferModeError!StageTransferModeDecision {
    try host_capability.validatePlacementPlan(request.placement_plan);
    try request.metadata.validate();
    try boundary_byte_image.validateBoundaryByteImage(request.image, .{});
    if (request.image.metadata != request.metadata) return error.StageTransferMetadataMismatch;

    if (!placementIdentityMatchesMetadata(request.placement_plan, request.metadata)) {
        return error.StageTransferBoundaryMismatch;
    }

    const boundary_summary = try boundarySummaryForMetadata(request.placement_plan, request.metadata);
    const source_binding = try host_capability.bindingForStage(request.placement_plan, request.metadata.boundary.source_stage_id);
    const target_binding = try host_capability.bindingForStage(request.placement_plan, request.metadata.boundary.target_stage_id);
    const profile = findBoundaryProfile(request.placement_plan, request.metadata, boundary_summary) orelse {
        return error.MissingStageTransferBoundaryProfile;
    };

    if (profile.dtype != request.metadata.tensor.dtype) {
        return error.StageTransferTensorProfileMismatch;
    }
    try validateEnvelopeLimits(profile, request.metadata, request.image);

    return .{
        .mode = try selectMode(profile.handoff_mode, request.image, request.allow_borrow),
        .boundary_profile = profile,
        .source_host_id = source_binding.host_id,
        .target_host_id = target_binding.host_id,
    };
}

fn placementIdentityMatchesMetadata(
    placement_plan: *const host_capability.PlacementPlan,
    metadata: *const tensor_frame.TensorFrameMetadata,
) bool {
    return std.mem.eql(u8, &placement_plan.graph_digest, &metadata.plan.graph_digest) and
        placement_plan.graph_contract_version == metadata.plan.graph_contract_version and
        placement_plan.stage_plan_contract_version == metadata.plan.stage_plan_contract_version and
        std.mem.eql(u8, &placement_plan.stage_plan_id.digest, &metadata.plan.stage_plan_id.digest);
}

fn boundarySummaryForMetadata(
    placement_plan: *const host_capability.PlacementPlan,
    metadata: *const tensor_frame.TensorFrameMetadata,
) StageTransferModeError!host_capability.PlacementBoundarySummary {
    var matching_index_count: usize = 0;
    var matched_summary: ?host_capability.PlacementBoundarySummary = null;
    for (placement_plan.boundary_summaries) |summary| {
        if (summary.boundary_index != metadata.boundary.boundary_index) continue;
        matching_index_count += 1;
        if (!boundarySummaryMatchesMetadata(summary, metadata.boundary)) {
            return error.StageTransferBoundaryMismatch;
        }
        matched_summary = summary;
    }
    if (matching_index_count != 1) return error.StageTransferBoundaryMismatch;
    return matched_summary.?;
}

fn boundarySummaryMatchesMetadata(
    summary: host_capability.PlacementBoundarySummary,
    boundary: tensor_frame.TensorFrameBoundaryRef,
) bool {
    return summary.boundary_index == boundary.boundary_index and
        summary.source_stage_id == boundary.source_stage_id and
        summary.target_stage_id == boundary.target_stage_id and
        summary.producer_layer_start == boundary.producer_layer_start and
        summary.producer_layer_end == boundary.producer_layer_end and
        summary.consumer_layer_start == boundary.consumer_layer_start and
        summary.consumer_layer_end == boundary.consumer_layer_end;
}

fn findBoundaryProfile(
    placement_plan: *const host_capability.PlacementPlan,
    metadata: *const tensor_frame.TensorFrameMetadata,
    boundary_summary: host_capability.PlacementBoundarySummary,
) ?host_capability.BoundaryFrameProfile {
    for (placement_plan.boundary_frame_profiles) |profile| {
        if (profile.boundary_index != boundary_summary.boundary_index) continue;
        if (profile.source_stage_id != boundary_summary.source_stage_id) continue;
        if (profile.target_stage_id != boundary_summary.target_stage_id) continue;
        if (profile.step_kind != metadata.step_kind) continue;
        return profile;
    }
    return null;
}

fn validateEnvelopeLimits(
    profile: host_capability.BoundaryFrameProfile,
    metadata: *const tensor_frame.TensorFrameMetadata,
    image: *const boundary_byte_image.BoundaryByteImageRef,
) StageTransferModeError!void {
    const batch_count = std.math.cast(u64, metadata.batch.entries.len) orelse return error.StageTransferTensorProfileMismatch;
    if (batch_count > profile.max_batch_entries) return error.StageTransferTensorProfileMismatch;
    for (metadata.batch.entries) |entry| {
        if (entry.token_count > profile.max_token_count_per_frame) return error.StageTransferTensorProfileMismatch;
    }
    if (image.byte_count > profile.max_activation_payload_bytes) return error.StageTransferTensorProfileMismatch;
}

fn selectMode(
    handoff_mode: host_capability.BoundaryHandoffMode,
    image: *const boundary_byte_image.BoundaryByteImageRef,
    allow_borrow: bool,
) StageTransferModeError!StageTransferMode {
    return switch (image.readiness) {
        .host_readable_now => switch (handoff_mode) {
            .same_host_direct => if (allow_borrow and image.ownership == .borrowed_until_next_stage_call)
                .borrow_in_process
            else
                .copy_in_process,
            .local_in_process, .mock => .copy_in_process,
            .remote_declared => .remote_stream,
        },
        .device_download_required => switch (handoff_mode) {
            .same_host_direct, .local_in_process, .mock => .device_download_then_copy,
            .remote_declared => .device_download_then_remote_stream,
        },
        .producer_sync_required => error.StageTransferPayloadNotHostReadable,
        .local_only_opaque => error.StageTransferOpaquePayloadNotRemoteReadable,
    };
}

const all_step_kinds = [_]tensor_frame.TensorFrameStepKind{ .prefill, .decode };
const prefill_step_kind = [_]tensor_frame.TensorFrameStepKind{.prefill};
const all_reachability = [_]host_capability.HostReachabilityKind{ .local_in_process, .mock, .remote_declared };

const PlacementTopology = enum {
    same_host,
    two_host_local,
    two_host_mock,
    two_host_remote,
};

const ProfileConfig = struct {
    required_step_kinds: []const tensor_frame.TensorFrameStepKind = &all_step_kinds,
    max_batch_entries: ?u64 = null,
    prefill_max_token_count: ?u64 = null,
    decode_max_token_count: ?u64 = null,
    max_activation_payload_bytes: ?u64 = null,
};

const PlacementBundle = struct {
    stage_plan_value: stage_plan.StagePlan,
    source_capability: host_capability.HostCapability,
    target_capability: ?host_capability.HostCapability = null,
    source_residency: host_capability.HostResidencySnapshot,
    target_residency: ?host_capability.HostResidencySnapshot = null,
    placement: host_capability.PlacementPlan,

    fn deinit(self: *PlacementBundle) void {
        self.placement.deinit();
        if (self.target_residency) |*residency| residency.deinit();
        self.source_residency.deinit();
        if (self.target_capability) |*capability| capability.deinit();
        self.source_capability.deinit();
        self.stage_plan_value.deinit();
        self.* = undefined;
    }
};

fn buildPlacementBundle(
    allocator: Allocator,
    topology: PlacementTopology,
    mode: host_capability.BoundaryHandoffMode,
    profile_config: ProfileConfig,
) !PlacementBundle {
    var plan = try buildTestStagePlan(allocator, &.{2});
    errdefer plan.deinit();
    var profiles = testProfiles(&plan, mode);
    applyProfileConfig(&profiles, profile_config);
    const profile_slice = profilesForRequiredStepKinds(&profiles, profile_config.required_step_kinds);

    switch (topology) {
        .same_host => {
            var capability = try buildTestCapability(allocator, testHostId(1), .local_in_process, mode);
            errdefer capability.deinit();
            var residency = try buildTestResidency(allocator, testHostId(1), &plan, &.{ 0, 1 });
            errdefer residency.deinit();
            const bindings = [_]host_capability.StageHostBinding{
                .{ .stage_id = 0, .host_id = testHostId(1) },
                .{ .stage_id = 1, .host_id = testHostId(1) },
            };
            var placement = try host_capability.buildPlacementPlan(allocator, .{
                .plan = &plan,
                .required_step_kinds = profile_config.required_step_kinds,
                .host_capabilities = &.{capability},
                .host_residency_snapshots = &.{residency},
                .stage_host_bindings = &bindings,
                .boundary_frame_profiles = profile_slice,
            });
            errdefer placement.deinit();
            return .{
                .stage_plan_value = plan,
                .source_capability = capability,
                .source_residency = residency,
                .placement = placement,
            };
        },
        .two_host_local, .two_host_mock, .two_host_remote => {
            const source_reachability: host_capability.HostReachabilityKind = switch (topology) {
                .two_host_mock => .mock,
                else => .local_in_process,
            };
            const target_reachability: host_capability.HostReachabilityKind = switch (topology) {
                .two_host_mock => .mock,
                .two_host_remote => .remote_declared,
                else => .local_in_process,
            };
            var source_capability = try buildTestCapability(allocator, testHostId(1), source_reachability, mode);
            errdefer source_capability.deinit();
            var target_capability = try buildTestCapability(allocator, testHostId(2), target_reachability, mode);
            errdefer target_capability.deinit();
            var source_residency = try buildTestResidency(allocator, testHostId(1), &plan, &.{0});
            errdefer source_residency.deinit();
            var target_residency = try buildTestResidency(allocator, testHostId(2), &plan, &.{1});
            errdefer target_residency.deinit();
            const bindings = [_]host_capability.StageHostBinding{
                .{ .stage_id = 0, .host_id = testHostId(1) },
                .{ .stage_id = 1, .host_id = testHostId(2) },
            };
            var placement = try host_capability.buildPlacementPlan(allocator, .{
                .plan = &plan,
                .required_step_kinds = profile_config.required_step_kinds,
                .host_capabilities = &.{ source_capability, target_capability },
                .host_residency_snapshots = &.{ source_residency, target_residency },
                .stage_host_bindings = &bindings,
                .boundary_frame_profiles = profile_slice,
                .allowed_reachability = if (topology == .two_host_remote) &all_reachability else &.{ .local_in_process, .mock },
            });
            errdefer placement.deinit();
            return .{
                .stage_plan_value = plan,
                .source_capability = source_capability,
                .target_capability = target_capability,
                .source_residency = source_residency,
                .target_residency = target_residency,
                .placement = placement,
            };
        },
    }
}

fn applyProfileConfig(
    profiles: *[2]host_capability.BoundaryFrameProfile,
    config: ProfileConfig,
) void {
    if (config.max_batch_entries) |limit| {
        for (profiles) |*profile| profile.max_batch_entries = limit;
    }
    if (config.prefill_max_token_count) |limit| {
        profiles[0].max_token_count_per_frame = limit;
    }
    if (config.decode_max_token_count) |limit| {
        profiles[1].max_token_count_per_frame = limit;
    }
    if (config.max_activation_payload_bytes) |limit| {
        for (profiles) |*profile| profile.max_activation_payload_bytes = limit;
    }
}

fn profilesForRequiredStepKinds(
    profiles: *[2]host_capability.BoundaryFrameProfile,
    required_step_kinds: []const tensor_frame.TensorFrameStepKind,
) []const host_capability.BoundaryFrameProfile {
    if (required_step_kinds.len == 1) {
        switch (required_step_kinds[0]) {
            .prefill => return profiles[0..1],
            .decode => {
                profiles[0] = profiles[1];
                return profiles[0..1];
            },
        }
    }
    return profiles[0..];
}

fn testHostId(value: u64) host_capability.HostId {
    return .{ .value = value };
}

fn testFrameCapabilities(mode: host_capability.BoundaryHandoffMode) [4]host_capability.HostFrameCapability {
    return .{
        .{ .endpoint_role = .producer, .step_kind = .prefill, .dtype = .f32, .handoff_mode = mode, .max_batch_entries = 8, .max_token_count_per_frame = 16, .max_activation_payload_bytes = 4096 },
        .{ .endpoint_role = .consumer, .step_kind = .prefill, .dtype = .f32, .handoff_mode = mode, .max_batch_entries = 8, .max_token_count_per_frame = 16, .max_activation_payload_bytes = 4096 },
        .{ .endpoint_role = .producer, .step_kind = .decode, .dtype = .f32, .handoff_mode = mode, .max_batch_entries = 8, .max_token_count_per_frame = 1, .max_activation_payload_bytes = 4096 },
        .{ .endpoint_role = .consumer, .step_kind = .decode, .dtype = .f32, .handoff_mode = mode, .max_batch_entries = 8, .max_token_count_per_frame = 1, .max_activation_payload_bytes = 4096 },
    };
}

fn buildTestCapability(
    allocator: Allocator,
    host_id: host_capability.HostId,
    reachability: host_capability.HostReachabilityKind,
    mode: host_capability.BoundaryHandoffMode,
) !host_capability.HostCapability {
    const frames = testFrameCapabilities(mode);
    return host_capability.buildHostCapability(allocator, .{
        .host_id = host_id,
        .backend_kind = if (reachability == .mock) .mock else .cpu,
        .reachability_kind = reachability,
        .supported_graph_contract_versions = &.{stage_plan.graph_identity_contract_version},
        .supported_stage_plan_contract_versions = &.{stage_plan.stage_plan_contract_version},
        .frame_capabilities = &frames,
        .resident_checkpoint_budget_bytes = 1024,
        .diagnostic_workspace_budget_bytes = 1,
    });
}

fn testProfiles(
    plan: *const stage_plan.StagePlan,
    mode: host_capability.BoundaryHandoffMode,
) [2]host_capability.BoundaryFrameProfile {
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

fn buildTestStagePlan(
    allocator: Allocator,
    splits: []const usize,
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
        },
    });
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
        .name = "stage_transfer_mode_test",
        .model_types = &.{"stage_transfer_mode_test"},
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
        .architecture_id = "stage_transfer_mode_test",
        .layer_count = layer_count,
        .entries = entries,
        .total_checkpoint_bytes = total_bytes,
        .role_bytes = role_bytes,
    };
}

fn residentEntryFromStage(stage: stage_plan.StagePlanStage) host_capability.ResidentStageEntry {
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
    host_id: host_capability.HostId,
    plan: *const stage_plan.StagePlan,
    stages: []const usize,
) !host_capability.HostResidencySnapshot {
    var entries: [4]host_capability.ResidentStageEntry = undefined;
    for (stages, 0..) |stage_id, index| {
        entries[index] = residentEntryFromStage(plan.stages[stage_id]);
    }
    return host_capability.buildHostResidencySnapshot(allocator, .{
        .host_id = host_id,
        .plan = plan,
        .resident_stages = entries[0..stages.len],
    });
}

const decode_entries = [_]tensor_frame.TensorFrameBatchEntry{.{
    .batch_index = 0,
    .request_id = 101,
    .slot_id = 88,
    .sequence_start = 12,
    .token_count = 1,
}};

const prefill_entries = [_]tensor_frame.TensorFrameBatchEntry{.{
    .batch_index = 0,
    .request_id = 102,
    .slot_id = 89,
    .sequence_start = 12,
    .token_count = 4,
}};

const two_decode_entries = [_]tensor_frame.TensorFrameBatchEntry{
    .{
        .batch_index = 0,
        .request_id = 103,
        .slot_id = 90,
        .sequence_start = 12,
        .token_count = 1,
    },
    .{
        .batch_index = 1,
        .request_id = 104,
        .slot_id = 91,
        .sequence_start = 12,
        .token_count = 1,
    },
};

fn metadataForPlacement(
    placement: *const host_capability.PlacementPlan,
    step_kind: tensor_frame.TensorFrameStepKind,
    dtype: tensor_frame.TensorFrameDType,
    shape: [4]u64,
    entries: []const tensor_frame.TensorFrameBatchEntry,
) !tensor_frame.TensorFrameMetadata {
    const boundary = boundaryRefFromSummary(placement.boundary_summaries[0]);
    const tensor = try tensor_frame.TensorFrameTensorDesc.contiguousActivation(dtype, shape);
    return .{
        .frame_id = try tensor_frame.TensorFrameInstanceId.init(55),
        .plan = .{
            .graph_digest = placement.graph_digest,
            .graph_contract_version = placement.graph_contract_version,
            .stage_plan_contract_version = placement.stage_plan_contract_version,
            .stage_plan_id = placement.stage_plan_id,
        },
        .boundary = boundary,
        .selected_contract = .{
            .boundary = boundary,
            .dtype = dtype,
            .layout = .row_major,
            .source = .explicit,
        },
        .role = .activation,
        .step_kind = step_kind,
        .shape_context = .{
            .expected_hidden_size = shape[2],
            .expected_step_kind = step_kind,
        },
        .tensor = tensor,
        .batch = .{ .entries = entries },
        .payload = .{
            .byte_count = tensor.payload_byte_count,
            .location_hint = .cpu,
            .ownership = .borrowed_until_next_stage_call,
            .lifetime = .step_scoped,
        },
    };
}

fn boundaryRefFromSummary(summary: host_capability.PlacementBoundarySummary) tensor_frame.TensorFrameBoundaryRef {
    return .{
        .boundary_index = summary.boundary_index,
        .source_stage_id = summary.source_stage_id,
        .target_stage_id = summary.target_stage_id,
        .producer_layer_start = summary.producer_layer_start,
        .producer_layer_end = summary.producer_layer_end,
        .consumer_layer_start = summary.consumer_layer_start,
        .consumer_layer_end = summary.consumer_layer_end,
    };
}

fn testImage(
    metadata: *const tensor_frame.TensorFrameMetadata,
    readiness: boundary_byte_image.BoundaryByteImageReadiness,
    host_bytes: ?[]const u8,
) boundary_byte_image.BoundaryByteImageRef {
    return .{
        .metadata = metadata,
        .byte_count = metadata.payload.byte_count,
        .host_bytes = host_bytes,
        .location_hint = metadata.payload.location_hint,
        .readiness = readiness,
        .ownership = metadata.payload.ownership,
        .lifetime = metadata.payload.lifetime,
    };
}

fn expectDecision(
    decision: StageTransferModeDecision,
    expected_mode: StageTransferMode,
    placement: *const host_capability.PlacementPlan,
    metadata: *const tensor_frame.TensorFrameMetadata,
) !void {
    try std.testing.expectEqual(expected_mode, decision.mode);
    const expected_profile = expectedProfileFor(placement, metadata);
    try expectProfileEqual(expected_profile, decision.boundary_profile);
    const source_binding = try host_capability.bindingForStage(placement, metadata.boundary.source_stage_id);
    const target_binding = try host_capability.bindingForStage(placement, metadata.boundary.target_stage_id);
    try std.testing.expectEqual(source_binding.host_id.value, decision.source_host_id.value);
    try std.testing.expectEqual(target_binding.host_id.value, decision.target_host_id.value);
}

fn expectedProfileFor(
    placement: *const host_capability.PlacementPlan,
    metadata: *const tensor_frame.TensorFrameMetadata,
) host_capability.BoundaryFrameProfile {
    for (placement.boundary_frame_profiles) |profile| {
        if (profile.boundary_index == metadata.boundary.boundary_index and
            profile.source_stage_id == metadata.boundary.source_stage_id and
            profile.target_stage_id == metadata.boundary.target_stage_id and
            profile.step_kind == metadata.step_kind)
        {
            return profile;
        }
    }
    unreachable;
}

fn expectProfileEqual(
    expected: host_capability.BoundaryFrameProfile,
    actual: host_capability.BoundaryFrameProfile,
) !void {
    try std.testing.expectEqual(expected.boundary_index, actual.boundary_index);
    try std.testing.expectEqual(expected.source_stage_id, actual.source_stage_id);
    try std.testing.expectEqual(expected.target_stage_id, actual.target_stage_id);
    try std.testing.expectEqual(expected.tensor_frame_contract_version, actual.tensor_frame_contract_version);
    try std.testing.expectEqual(expected.step_kind, actual.step_kind);
    try std.testing.expectEqual(expected.dtype, actual.dtype);
    try std.testing.expectEqual(expected.layout, actual.layout);
    try std.testing.expectEqual(expected.max_batch_entries, actual.max_batch_entries);
    try std.testing.expectEqual(expected.max_token_count_per_frame, actual.max_token_count_per_frame);
    try std.testing.expectEqual(expected.max_activation_payload_bytes, actual.max_activation_payload_bytes);
    try std.testing.expectEqual(expected.handoff_mode, actual.handoff_mode);
}

test "inference bridge stage_transfer_mode chooseStageTransferMode selects borrow_in_process for valid same host borrowed bytes" {
    const allocator = std.testing.allocator;
    var bundle = try buildPlacementBundle(allocator, .same_host, .same_host_direct, .{});
    defer bundle.deinit();
    var metadata = try metadataForPlacement(&bundle.placement, .decode, .f32, .{ 1, 1, 4, 0 }, &decode_entries);
    const payload = [_]u8{0xaa} ** 16;
    const image = testImage(&metadata, .host_readable_now, &payload);

    const decision = try chooseStageTransferMode(.{
        .placement_plan = &bundle.placement,
        .metadata = &metadata,
        .image = &image,
    });
    try expectDecision(decision, .borrow_in_process, &bundle.placement, &metadata);
}

test "inference bridge stage_transfer_mode chooseStageTransferMode selects copy_in_process when borrowing is not allowed" {
    const allocator = std.testing.allocator;
    var bundle = try buildPlacementBundle(allocator, .same_host, .same_host_direct, .{});
    defer bundle.deinit();
    var metadata = try metadataForPlacement(&bundle.placement, .decode, .f32, .{ 1, 1, 4, 0 }, &decode_entries);
    const payload = [_]u8{0xbb} ** 16;
    const image = testImage(&metadata, .host_readable_now, &payload);

    const decision = try chooseStageTransferMode(.{
        .placement_plan = &bundle.placement,
        .metadata = &metadata,
        .image = &image,
        .allow_borrow = false,
    });
    try expectDecision(decision, .copy_in_process, &bundle.placement, &metadata);
}

test "inference bridge stage_transfer_mode chooseStageTransferMode selects copy_in_process for local and mock host readable bytes" {
    const allocator = std.testing.allocator;
    const payload = [_]u8{0xcc} ** 16;

    var local_bundle = try buildPlacementBundle(allocator, .two_host_local, .local_in_process, .{});
    defer local_bundle.deinit();
    var local_metadata = try metadataForPlacement(&local_bundle.placement, .decode, .f32, .{ 1, 1, 4, 0 }, &decode_entries);
    const local_image = testImage(&local_metadata, .host_readable_now, &payload);
    const local_decision = try chooseStageTransferMode(.{
        .placement_plan = &local_bundle.placement,
        .metadata = &local_metadata,
        .image = &local_image,
    });
    try expectDecision(local_decision, .copy_in_process, &local_bundle.placement, &local_metadata);

    var mock_bundle = try buildPlacementBundle(allocator, .two_host_mock, .mock, .{});
    defer mock_bundle.deinit();
    var mock_metadata = try metadataForPlacement(&mock_bundle.placement, .decode, .f32, .{ 1, 1, 4, 0 }, &decode_entries);
    const mock_image = testImage(&mock_metadata, .host_readable_now, &payload);
    const mock_decision = try chooseStageTransferMode(.{
        .placement_plan = &mock_bundle.placement,
        .metadata = &mock_metadata,
        .image = &mock_image,
    });
    try expectDecision(mock_decision, .copy_in_process, &mock_bundle.placement, &mock_metadata);
}

test "inference bridge stage_transfer_mode chooseStageTransferMode selects device_download_then_copy for local device resident bytes" {
    const allocator = std.testing.allocator;
    var bundle = try buildPlacementBundle(allocator, .two_host_local, .local_in_process, .{});
    defer bundle.deinit();
    var metadata = try metadataForPlacement(&bundle.placement, .decode, .f32, .{ 1, 1, 4, 0 }, &decode_entries);
    metadata.payload.location_hint = .{ .cuda = 0 };
    const image = testImage(&metadata, .device_download_required, null);

    const decision = try chooseStageTransferMode(.{
        .placement_plan = &bundle.placement,
        .metadata = &metadata,
        .image = &image,
    });
    try expectDecision(decision, .device_download_then_copy, &bundle.placement, &metadata);
}

test "inference bridge stage_transfer_mode chooseStageTransferMode selects remote_stream for remote host readable bytes" {
    const allocator = std.testing.allocator;
    var bundle = try buildPlacementBundle(allocator, .two_host_remote, .remote_declared, .{});
    defer bundle.deinit();
    var metadata = try metadataForPlacement(&bundle.placement, .decode, .f32, .{ 1, 1, 4, 0 }, &decode_entries);
    const payload = [_]u8{0xdd} ** 16;
    const image = testImage(&metadata, .host_readable_now, &payload);

    const decision = try chooseStageTransferMode(.{
        .placement_plan = &bundle.placement,
        .metadata = &metadata,
        .image = &image,
    });
    try expectDecision(decision, .remote_stream, &bundle.placement, &metadata);
}

test "inference bridge stage_transfer_mode chooseStageTransferMode selects device_download_then_remote_stream for remote device resident bytes" {
    const allocator = std.testing.allocator;
    var bundle = try buildPlacementBundle(allocator, .two_host_remote, .remote_declared, .{});
    defer bundle.deinit();
    var metadata = try metadataForPlacement(&bundle.placement, .decode, .f32, .{ 1, 1, 4, 0 }, &decode_entries);
    metadata.payload.location_hint = .{ .metal = 0 };
    const image = testImage(&metadata, .device_download_required, null);

    const decision = try chooseStageTransferMode(.{
        .placement_plan = &bundle.placement,
        .metadata = &metadata,
        .image = &image,
    });
    try expectDecision(decision, .device_download_then_remote_stream, &bundle.placement, &metadata);
}

test "inference bridge stage_transfer_mode chooseStageTransferMode rejects producer sync and opaque local payloads for every handoff mode" {
    const allocator = std.testing.allocator;
    try expectReadinessErrors(allocator, .same_host, .same_host_direct);
    try expectReadinessErrors(allocator, .two_host_local, .local_in_process);
    try expectReadinessErrors(allocator, .two_host_mock, .mock);
    try expectReadinessErrors(allocator, .two_host_remote, .remote_declared);
}

fn expectReadinessErrors(
    allocator: Allocator,
    topology: PlacementTopology,
    mode: host_capability.BoundaryHandoffMode,
) !void {
    var bundle = try buildPlacementBundle(allocator, topology, mode, .{});
    defer bundle.deinit();

    var producer_metadata = try metadataForPlacement(&bundle.placement, .decode, .f32, .{ 1, 1, 4, 0 }, &decode_entries);
    producer_metadata.payload.location_hint = .cpu;
    const producer_image = testImage(&producer_metadata, .producer_sync_required, null);
    try std.testing.expectError(error.StageTransferPayloadNotHostReadable, chooseStageTransferMode(.{
        .placement_plan = &bundle.placement,
        .metadata = &producer_metadata,
        .image = &producer_image,
    }));

    var opaque_metadata = try metadataForPlacement(&bundle.placement, .decode, .f32, .{ 1, 1, 4, 0 }, &decode_entries);
    opaque_metadata.payload.location_hint = .{ .opaque_local = 7 };
    const opaque_image = testImage(&opaque_metadata, .local_only_opaque, null);
    try std.testing.expectError(error.StageTransferOpaquePayloadNotRemoteReadable, chooseStageTransferMode(.{
        .placement_plan = &bundle.placement,
        .metadata = &opaque_metadata,
        .image = &opaque_image,
    }));
}

test "inference bridge stage_transfer_mode chooseStageTransferMode rejects mismatched metadata image placement graph full boundary and dtype facts" {
    const allocator = std.testing.allocator;
    var bundle = try buildPlacementBundle(allocator, .two_host_local, .local_in_process, .{});
    defer bundle.deinit();
    var metadata = try metadataForPlacement(&bundle.placement, .decode, .f32, .{ 1, 1, 4, 0 }, &decode_entries);
    var other_metadata = try metadataForPlacement(&bundle.placement, .decode, .f32, .{ 1, 1, 4, 0 }, &decode_entries);
    other_metadata.frame_id = try tensor_frame.TensorFrameInstanceId.init(56);
    const payload = [_]u8{0xee} ** 16;
    const image = testImage(&metadata, .host_readable_now, &payload);
    try std.testing.expectError(error.StageTransferMetadataMismatch, chooseStageTransferMode(.{
        .placement_plan = &bundle.placement,
        .metadata = &other_metadata,
        .image = &image,
    }));

    var graph_mismatch = metadata;
    graph_mismatch.plan.graph_digest[0] ^= 1;
    const graph_image = testImage(&graph_mismatch, .host_readable_now, &payload);
    try std.testing.expectError(error.StageTransferBoundaryMismatch, chooseStageTransferMode(.{
        .placement_plan = &bundle.placement,
        .metadata = &graph_mismatch,
        .image = &graph_image,
    }));

    var boundary_mismatch = metadata;
    boundary_mismatch.boundary.producer_layer_end += 1;
    boundary_mismatch.selected_contract.boundary = boundary_mismatch.boundary;
    const boundary_image = testImage(&boundary_mismatch, .host_readable_now, &payload);
    try std.testing.expectError(error.StageTransferBoundaryMismatch, chooseStageTransferMode(.{
        .placement_plan = &bundle.placement,
        .metadata = &boundary_mismatch,
        .image = &boundary_image,
    }));

    var dtype_metadata = try metadataForPlacement(&bundle.placement, .decode, .f16, .{ 1, 1, 4, 0 }, &decode_entries);
    const dtype_payload = [_]u8{0xef} ** 8;
    const dtype_image = testImage(&dtype_metadata, .host_readable_now, &dtype_payload);
    try std.testing.expectError(error.StageTransferTensorProfileMismatch, chooseStageTransferMode(.{
        .placement_plan = &bundle.placement,
        .metadata = &dtype_metadata,
        .image = &dtype_image,
    }));
}

test "inference bridge stage_transfer_mode chooseStageTransferMode rejects missing dtype mismatched and envelope mismatched boundary profiles" {
    const allocator = std.testing.allocator;
    const payload = [_]u8{0xab} ** 16;

    var missing_bundle = try buildPlacementBundle(allocator, .two_host_local, .local_in_process, .{
        .required_step_kinds = &prefill_step_kind,
    });
    defer missing_bundle.deinit();
    var missing_metadata = try metadataForPlacement(&missing_bundle.placement, .decode, .f32, .{ 1, 1, 4, 0 }, &decode_entries);
    const missing_image = testImage(&missing_metadata, .host_readable_now, &payload);
    try std.testing.expectError(error.MissingStageTransferBoundaryProfile, chooseStageTransferMode(.{
        .placement_plan = &missing_bundle.placement,
        .metadata = &missing_metadata,
        .image = &missing_image,
    }));

    var batch_bundle = try buildPlacementBundle(allocator, .two_host_local, .local_in_process, .{
        .max_batch_entries = 1,
    });
    defer batch_bundle.deinit();
    var batch_metadata = try metadataForPlacement(&batch_bundle.placement, .decode, .f32, .{ 2, 1, 4, 0 }, &two_decode_entries);
    const batch_payload = [_]u8{0xac} ** 32;
    const batch_image = testImage(&batch_metadata, .host_readable_now, &batch_payload);
    try std.testing.expectError(error.StageTransferTensorProfileMismatch, chooseStageTransferMode(.{
        .placement_plan = &batch_bundle.placement,
        .metadata = &batch_metadata,
        .image = &batch_image,
    }));

    var token_bundle = try buildPlacementBundle(allocator, .two_host_local, .local_in_process, .{
        .prefill_max_token_count = 3,
    });
    defer token_bundle.deinit();
    var token_metadata = try metadataForPlacement(&token_bundle.placement, .prefill, .f32, .{ 1, 4, 4, 0 }, &prefill_entries);
    const token_payload = [_]u8{0xad} ** 64;
    const token_image = testImage(&token_metadata, .host_readable_now, &token_payload);
    try std.testing.expectError(error.StageTransferTensorProfileMismatch, chooseStageTransferMode(.{
        .placement_plan = &token_bundle.placement,
        .metadata = &token_metadata,
        .image = &token_image,
    }));

    var payload_bundle = try buildPlacementBundle(allocator, .two_host_local, .local_in_process, .{
        .max_activation_payload_bytes = 8,
    });
    defer payload_bundle.deinit();
    var payload_metadata = try metadataForPlacement(&payload_bundle.placement, .decode, .f32, .{ 1, 1, 4, 0 }, &decode_entries);
    const payload_image = testImage(&payload_metadata, .host_readable_now, &payload);
    try std.testing.expectError(error.StageTransferTensorProfileMismatch, chooseStageTransferMode(.{
        .placement_plan = &payload_bundle.placement,
        .metadata = &payload_metadata,
        .image = &payload_image,
    }));
}

test "inference bridge stage_transfer_mode chooseStageTransferMode returns exact placement errors for invalid profile cardinality" {
    const allocator = std.testing.allocator;
    var bundle = try buildPlacementBundle(allocator, .two_host_local, .local_in_process, .{});
    defer bundle.deinit();
    var metadata = try metadataForPlacement(&bundle.placement, .decode, .f32, .{ 1, 1, 4, 0 }, &decode_entries);
    const payload = [_]u8{0xba} ** 16;
    const image = testImage(&metadata, .host_readable_now, &payload);

    var missing = bundle.placement;
    missing.boundary_frame_profiles = bundle.placement.boundary_frame_profiles[0..1];
    try std.testing.expectError(error.MissingBoundaryFrameProfile, chooseStageTransferMode(.{
        .placement_plan = &missing,
        .metadata = &metadata,
        .image = &image,
    }));

    const duplicate_profiles = [_]host_capability.BoundaryFrameProfile{
        bundle.placement.boundary_frame_profiles[0],
        bundle.placement.boundary_frame_profiles[1],
        bundle.placement.boundary_frame_profiles[1],
    };
    var duplicate = bundle.placement;
    duplicate.boundary_frame_profiles = &duplicate_profiles;
    try std.testing.expectError(error.DuplicateBoundaryFrameProfile, chooseStageTransferMode(.{
        .placement_plan = &duplicate,
        .metadata = &metadata,
        .image = &image,
    }));
}
