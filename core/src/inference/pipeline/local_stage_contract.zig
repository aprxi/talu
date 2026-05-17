//! Backend-neutral local stage contract construction.
//!
//! This module owns the cold contracts that describe an ordered process-local
//! stage chain. Backends provide concrete stage executors; the pipeline contracts
//! built here define the stage plan, placement, tensor-frame, state ownership,
//! and local runner facts used by route execution.

const std = @import("std");

const host_capability = @import("host_capability.zig");
const local_stage_runner = @import("local_stage_runner.zig");
const pipeline_contract = @import("pipeline.zig");
const state_ownership = @import("state_ownership.zig");
const tensor_frame = @import("tensor_frame.zig");
const models = @import("models_pkg");

const pipeline = struct {
    const BoundaryDType = pipeline_contract.BoundaryDType;
    const BoundaryFrameEndpointRole = host_capability.BoundaryFrameEndpointRole;
    const BoundaryFrameProfile = host_capability.BoundaryFrameProfile;
    const BoundaryLayout = pipeline_contract.BoundaryLayout;
    const HostBackendKind = host_capability.HostBackendKind;
    const HostCapability = host_capability.HostCapability;
    const HostFrameCapability = host_capability.HostFrameCapability;
    const HostId = host_capability.HostId;
    const HostResidencySnapshot = host_capability.HostResidencySnapshot;
    const LocalStageRunnerPlanRef = local_stage_runner.LocalStageRunnerPlanRef;
    const PlacementPlan = host_capability.PlacementPlan;
    const ResidentStageEntry = host_capability.ResidentStageEntry;
    const StageHostBinding = host_capability.StageHostBinding;
    const StageStateDescriptorSet = state_ownership.StageStateDescriptorSet;
    const StageStateOwnershipPlan = state_ownership.StageStateOwnershipPlan;
    const StageStateOwnershipPlanId = state_ownership.StageStateOwnershipPlanId;
    const StageStatePartitionFact = state_ownership.StageStatePartitionFact;
    const StageStatePlacementRef = host_capability.StageStatePlacementRef;
    const TensorFramePlanRef = tensor_frame.TensorFramePlanRef;
    const TensorFrameStepKind = tensor_frame.TensorFrameStepKind;
    const buildHostCapability = host_capability.buildHostCapability;
    const buildHostResidencySnapshot = host_capability.buildHostResidencySnapshot;
    const buildLocalStageRunnerPlanRef = local_stage_runner.buildLocalStageRunnerPlanRef;
    const buildPlacementPlan = host_capability.buildPlacementPlan;
    const buildStageStateOwnershipPlan = state_ownership.buildStageStateOwnershipPlan;
    const buildStageStatePlacementRef = host_capability.buildStageStatePlacementRef;
    const dtypeByteSize = tensor_frame.dtypeByteSize;
    const state_ownership_contract_version = state_ownership.state_ownership_contract_version;
};

pub const required_step_kinds = [_]pipeline.TensorFrameStepKind{ .prefill, .decode };

pub const StageSpec = struct {
    stage_id: usize,
    backend_kind: pipeline.HostBackendKind,
    layer_start: usize,
    layer_end: usize,
    owns_embedding: bool,
    owns_projection: bool,
};

pub const BoundaryRuntime = struct {
    boundary_index: usize,
    dtype: pipeline.BoundaryDType,
    layout: pipeline.BoundaryLayout,
    staging: ?[]align(4096) u8 = null,
    local_device_peer_copy_available: bool = false,
};

pub const BoundaryConfig = struct {
    dtype: pipeline.BoundaryDType,
    layout: pipeline.BoundaryLayout,
    decode_max_batch_entries: usize,
    prefill_max_token_count_per_frame: usize,
};

pub const ContractField = enum {
    local_stage_runner_plan_ref,
    placement_plan,
    state_placement_ref,
    state_ownership_plan,
    tensor_frame_plan_ref,
    stage_plan,
};

pub const contract_deinit_order = [_]ContractField{
    .local_stage_runner_plan_ref,
    .placement_plan,
    .state_placement_ref,
    .state_ownership_plan,
    .tensor_frame_plan_ref,
    .stage_plan,
};

pub const ContractBundle = struct {
    stage_plan: ?models.stage_plan.StagePlan = null,
    tensor_frame_plan_ref: ?pipeline.TensorFramePlanRef = null,
    state_ownership_plan: ?pipeline.StageStateOwnershipPlan = null,
    state_placement_ref: ?pipeline.StageStatePlacementRef = null,
    placement_plan: ?pipeline.PlacementPlan = null,
    local_stage_runner_plan_ref: ?pipeline.LocalStageRunnerPlanRef = null,

    pub fn deinit(self: *@This()) void {
        inline for (contract_deinit_order) |field| {
            switch (field) {
                .local_stage_runner_plan_ref => if (self.local_stage_runner_plan_ref) |*plan_ref| {
                    plan_ref.deinit();
                    self.local_stage_runner_plan_ref = null;
                },
                .placement_plan => if (self.placement_plan) |*plan| {
                    plan.deinit();
                    self.placement_plan = null;
                },
                .state_placement_ref => if (self.state_placement_ref) |*state_ref| {
                    state_ref.deinit();
                    self.state_placement_ref = null;
                },
                .state_ownership_plan => if (self.state_ownership_plan) |*state_plan| {
                    state_plan.deinit();
                    self.state_ownership_plan = null;
                },
                .tensor_frame_plan_ref => if (self.tensor_frame_plan_ref) |*plan_ref| {
                    plan_ref.deinit();
                    self.tensor_frame_plan_ref = null;
                },
                .stage_plan => if (self.stage_plan) |*plan| {
                    plan.deinit();
                    self.stage_plan = null;
                },
            }
        }
    }
};

pub const ContractBuildRequest = struct {
    allocator: std.mem.Allocator,
    loaded: *models.LoadedModel,
    d_model: usize,
    total_layers: usize,
    split_points: []const usize,
    stage_backend_kinds: []const pipeline.HostBackendKind,
    boundary_configs: []const BoundaryConfig,
    load_semantics: models.stage_plan.LoadSemantics,
};

pub fn validateStageSpecs(total_layers: usize, specs: []const StageSpec) !void {
    if (total_layers == 0 or specs.len == 0) return error.InvalidTopologyConfig;
    var expected_start: usize = 0;
    for (specs, 0..) |spec, index| {
        if (spec.stage_id != index) return error.InvalidTopologyConfig;
        if (spec.layer_start != expected_start) return error.InvalidTopologyConfig;
        if (spec.layer_end <= spec.layer_start or spec.layer_end > total_layers) return error.InvalidTopologyConfig;
        if (spec.owns_embedding != (index == 0)) return error.InvalidTopologyConfig;
        if (spec.owns_projection != (index + 1 == specs.len)) return error.InvalidTopologyConfig;
        expected_start = spec.layer_end;
    }
    if (expected_start != total_layers) return error.InvalidTopologyConfig;
}

pub fn validateBoundaryRuntimes(stage_count: usize, boundaries: anytype) !void {
    if (stage_count == 0) return error.InvalidTopologyConfig;
    if (boundaries.len + 1 != stage_count) return error.InvalidTopologyConfig;
    for (boundaries, 0..) |boundary, index| {
        if (boundary.boundary_index != index) return error.InvalidTopologyConfig;
    }
}

pub fn finalStageHasBackendKind(specs: []const StageSpec, kind: pipeline.HostBackendKind) bool {
    if (specs.len == 0) return false;
    return specs[specs.len - 1].backend_kind == kind;
}

pub fn deterministicHostId(stage_id: usize) !pipeline.HostId {
    const base = std.math.cast(u64, stage_id) orelse return error.InvalidTopologyConfig;
    return .{ .value = std.math.add(u64, base, 1) catch return error.InvalidTopologyConfig };
}

pub fn boundaryConfig(
    dtype_value: pipeline.BoundaryDType,
    layout: pipeline.BoundaryLayout,
    decode_max_batch_entries: usize,
    prefill_max_token_count_per_frame: usize,
) BoundaryConfig {
    return .{
        .dtype = dtype_value,
        .layout = layout,
        .decode_max_batch_entries = decode_max_batch_entries,
        .prefill_max_token_count_per_frame = prefill_max_token_count_per_frame,
    };
}

pub fn minBoundaryConfig(
    dtype_value: pipeline.BoundaryDType,
    layout: pipeline.BoundaryLayout,
    stage0_max_batch_size: usize,
    stage1_max_batch_size: usize,
    stage0_prefill_chunk_rows_cap: usize,
    stage1_prefill_chunk_rows_cap: usize,
) BoundaryConfig {
    return boundaryConfig(
        dtype_value,
        layout,
        @min(stage0_max_batch_size, stage1_max_batch_size),
        @min(stage0_prefill_chunk_rows_cap, stage1_prefill_chunk_rows_cap),
    );
}

pub fn twoBoundaryConfigs(
    boundary01_dtype: pipeline.BoundaryDType,
    boundary01_layout: pipeline.BoundaryLayout,
    boundary12_dtype: pipeline.BoundaryDType,
    boundary12_layout: pipeline.BoundaryLayout,
    gpu_stage1_max_batch_size: usize,
    gpu_stage2_max_batch_size: usize,
    gpu_stage1_prefill_chunk_rows_cap: usize,
    gpu_stage2_prefill_chunk_rows_cap: usize,
) [2]BoundaryConfig {
    const decode_batch_entries = @min(gpu_stage1_max_batch_size, gpu_stage2_max_batch_size);
    const prefill_token_count = @min(gpu_stage1_prefill_chunk_rows_cap, gpu_stage2_prefill_chunk_rows_cap);
    return .{
        boundaryConfig(boundary01_dtype, boundary01_layout, decode_batch_entries, prefill_token_count),
        boundaryConfig(boundary12_dtype, boundary12_layout, decode_batch_entries, prefill_token_count),
    };
}

pub fn boundaryRowByteCount(d_model: usize, boundary_dtype: pipeline.BoundaryDType) !u64 {
    if (d_model == 0) return error.InvalidArgument;
    const d_model_u64 = std.math.cast(u64, d_model) orelse return error.InvalidArgument;
    return std.math.mul(u64, d_model_u64, pipeline.dtypeByteSize(boundary_dtype)) catch error.InvalidArgument;
}

pub fn boundaryProfilePair(
    d_model: usize,
    boundary_index: usize,
    boundary: models.stage_plan.StageBoundary,
    config: BoundaryConfig,
) ![2]pipeline.BoundaryFrameProfile {
    if (boundary.source_stage_id + 1 != boundary.target_stage_id) return error.InvalidTopologyConfig;
    const row_bytes = try boundaryRowByteCount(d_model, config.dtype);
    const prefill_token_count = try usizeToNonZeroU64(config.prefill_max_token_count_per_frame);
    const decode_batch_entries = try usizeToNonZeroU64(config.decode_max_batch_entries);
    return .{
        .{
            .boundary_index = boundary_index,
            .source_stage_id = boundary.source_stage_id,
            .target_stage_id = boundary.target_stage_id,
            .step_kind = .prefill,
            .dtype = config.dtype,
            .layout = config.layout,
            .max_batch_entries = 1,
            .max_token_count_per_frame = prefill_token_count,
            .max_activation_payload_bytes = std.math.mul(u64, prefill_token_count, row_bytes) catch return error.InvalidArgument,
            .handoff_mode = .local_in_process,
        },
        .{
            .boundary_index = boundary_index,
            .source_stage_id = boundary.source_stage_id,
            .target_stage_id = boundary.target_stage_id,
            .step_kind = .decode,
            .dtype = config.dtype,
            .layout = config.layout,
            .max_batch_entries = decode_batch_entries,
            .max_token_count_per_frame = 1,
            .max_activation_payload_bytes = std.math.mul(u64, decode_batch_entries, row_bytes) catch return error.InvalidArgument,
            .handoff_mode = .local_in_process,
        },
    };
}

pub fn buildContractBundle(request: ContractBuildRequest) !ContractBundle {
    if (request.stage_backend_kinds.len != request.split_points.len + 1 or
        request.boundary_configs.len != request.split_points.len or
        request.total_layers == 0)
    {
        return error.InvalidTopologyConfig;
    }
    const model_manifest = request.loaded.manifestPtr() orelse return error.MissingManifest;
    const architecture = models.registry.runtimeArchitectureById(model_manifest.architecture_id) orelse return error.UnsupportedModel;
    const max_dependency_overrides = request.split_points.len + 1;
    const dependency_overrides_buffer = try request.allocator.alloc(models.stage_plan.DependencyOverride, max_dependency_overrides);
    defer request.allocator.free(dependency_overrides_buffer);
    var dependency_override_count: usize = 0;
    if (models.stage_plan.requiresBoundaryDependenciesFor(architecture, &request.loaded.config)) {
        for (0..request.split_points.len) |stage_id| {
            dependency_overrides_buffer[dependency_override_count] = .{
                .source_stage_id = stage_id,
                .target_stage_id = stage_id + 1,
                .reason = .stateful_decoder,
                .affects_loader_residency = false,
            };
            dependency_override_count += 1;
        }
    }
    if (model_manifest.hasRole(.vision_side)) {
        if (request.split_points.len == 0) return error.InvalidTopologyConfig;
        dependency_overrides_buffer[dependency_override_count] = .{
            .source_stage_id = 0,
            .target_stage_id = 1,
            .role = .vision_side,
            .reason = .vision_side,
            .affects_loader_residency = true,
        };
        dependency_override_count += 1;
    }
    const dependency_overrides = dependency_overrides_buffer[0..dependency_override_count];
    var bundle = ContractBundle{};
    errdefer bundle.deinit();

    bundle.stage_plan = try models.stage_plan.buildStagePlan(request.allocator, .{
        .n_layers = request.total_layers,
        .split_points = request.split_points,
        .architecture = architecture,
        .model_config = &request.loaded.config,
        .manifest = model_manifest,
        .load_semantics = request.load_semantics,
        .partition_constraints = .{
            .decoder_cuts_allowed = true,
            .dependency_overrides = dependency_overrides,
        },
    });
    const plan_ptr = &bundle.stage_plan.?;

    bundle.tensor_frame_plan_ref = try pipeline.TensorFramePlanRef.fromStagePlan(request.allocator, plan_ptr);

    if (countStatefulDependencies(plan_ptr) > 0) {
        bundle.state_ownership_plan = try buildStateOwnershipPlan(request.allocator, plan_ptr);
        if (bundle.state_ownership_plan) |*state_plan| {
            bundle.state_placement_ref = try pipeline.buildStageStatePlacementRef(request.allocator, state_plan);
        }
    }

    const state_ref_ptr: ?*const pipeline.StageStatePlacementRef = if (bundle.state_placement_ref) |*state_ref| state_ref else null;
    bundle.placement_plan = try buildPlacementPlan(
        request.allocator,
        request.d_model,
        plan_ptr,
        request.stage_backend_kinds,
        request.boundary_configs,
        state_ref_ptr,
    );
    const state_plan_ptr: ?*const pipeline.StageStateOwnershipPlan = if (bundle.state_ownership_plan) |*state_plan| state_plan else null;
    bundle.local_stage_runner_plan_ref = try pipeline.buildLocalStageRunnerPlanRef(request.allocator, .{
        .stage_plan = plan_ptr,
        .tensor_frame_plan_ref = &bundle.tensor_frame_plan_ref.?,
        .placement_plan = &bundle.placement_plan.?,
        .state_ownership_plan = state_plan_ptr,
    });
    return bundle;
}

pub fn countStatefulDependencies(plan: *const models.stage_plan.StagePlan) usize {
    var count: usize = 0;
    for (plan.dependencies) |dependency| {
        if (dependency.reason == .stateful_decoder) count += 1;
    }
    return count;
}

fn usizeToNonZeroU64(value: usize) !u64 {
    if (value == 0) return error.InvalidArgument;
    return std.math.cast(u64, value) orelse return error.InvalidArgument;
}

fn boundaryIndexForDependency(
    plan: *const models.stage_plan.StagePlan,
    dependency: models.stage_plan.StageDependency,
) ?usize {
    for (plan.boundaries, 0..) |boundary, index| {
        if (boundary.source_stage_id == dependency.source_stage_id and
            boundary.target_stage_id == dependency.target_stage_id)
        {
            return index;
        }
    }
    return null;
}

fn residentEntryFromStage(
    stage: models.stage_plan.StagePlanStage,
    state_ref: ?*const pipeline.StageStatePlacementRef,
) pipeline.ResidentStageEntry {
    var entry = pipeline.ResidentStageEntry{
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
            break;
        }
    }
    return entry;
}

fn buildBoundaryProfiles(
    allocator: std.mem.Allocator,
    d_model: usize,
    plan: *const models.stage_plan.StagePlan,
    boundary_configs: []const BoundaryConfig,
) ![]pipeline.BoundaryFrameProfile {
    if (plan.boundaries.len != boundary_configs.len) return error.InvalidTopologyConfig;
    const profile_count = std.math.mul(usize, plan.boundaries.len, required_step_kinds.len) catch return error.InvalidArgument;
    const profiles = try allocator.alloc(pipeline.BoundaryFrameProfile, profile_count);
    errdefer allocator.free(profiles);

    var profile_index: usize = 0;
    for (plan.boundaries, boundary_configs, 0..) |boundary, config, boundary_index| {
        const pair = try boundaryProfilePair(d_model, boundary_index, boundary, config);
        profiles[profile_index] = pair[0];
        profile_index += 1;
        profiles[profile_index] = pair[1];
        profile_index += 1;
    }
    return profiles;
}

fn hostFrameCapabilityFromProfile(
    role: pipeline.BoundaryFrameEndpointRole,
    profile: pipeline.BoundaryFrameProfile,
) pipeline.HostFrameCapability {
    return .{
        .endpoint_role = role,
        .step_kind = profile.step_kind,
        .dtype = profile.dtype,
        .layout = profile.layout,
        .handoff_mode = profile.handoff_mode,
        .max_batch_entries = profile.max_batch_entries,
        .max_token_count_per_frame = profile.max_token_count_per_frame,
        .max_activation_payload_bytes = profile.max_activation_payload_bytes,
    };
}

fn countHostFramesForStage(
    profiles: []const pipeline.BoundaryFrameProfile,
    stage_id: usize,
) usize {
    var count: usize = 0;
    for (profiles) |profile| {
        if (profile.source_stage_id == stage_id) count += 1;
        if (profile.target_stage_id == stage_id) count += 1;
    }
    return count;
}

fn buildHostCapabilities(
    allocator: std.mem.Allocator,
    plan: *const models.stage_plan.StagePlan,
    stage_backend_kinds: []const pipeline.HostBackendKind,
    profiles: []const pipeline.BoundaryFrameProfile,
    state_ref: ?*const pipeline.StageStatePlacementRef,
) ![]pipeline.HostCapability {
    if (stage_backend_kinds.len != plan.stages.len) return error.InvalidTopologyConfig;
    const host_capabilities = try allocator.alloc(pipeline.HostCapability, plan.stages.len);
    var initialized: usize = 0;
    errdefer {
        for (host_capabilities[0..initialized]) |*capability| capability.deinit();
        allocator.free(host_capabilities);
    }

    const state_versions: []const u32 = if (state_ref != null)
        &.{pipeline.state_ownership_contract_version}
    else
        &.{};

    for (plan.stages, stage_backend_kinds, 0..) |stage, backend_kind, index| {
        if (stage.id != index) return error.InvalidTopologyConfig;
        const frame_count = countHostFramesForStage(profiles, stage.id);
        if (frame_count == 0) return error.InvalidTopologyConfig;
        const frames = try allocator.alloc(pipeline.HostFrameCapability, frame_count);
        defer allocator.free(frames);
        var frame_index: usize = 0;
        for (profiles) |profile| {
            if (profile.source_stage_id == stage.id) {
                frames[frame_index] = hostFrameCapabilityFromProfile(.producer, profile);
                frame_index += 1;
            }
            if (profile.target_stage_id == stage.id) {
                frames[frame_index] = hostFrameCapabilityFromProfile(.consumer, profile);
                frame_index += 1;
            }
        }
        host_capabilities[index] = try pipeline.buildHostCapability(allocator, .{
            .host_id = try deterministicHostId(stage.id),
            .backend_kind = backend_kind,
            .reachability_kind = .local_in_process,
            .supported_graph_contract_versions = &.{plan.graph_identity.graph_contract_version},
            .supported_stage_plan_contract_versions = &.{plan.stage_contract_version},
            .supported_state_ownership_contract_versions = state_versions,
            .frame_capabilities = frames,
        });
        initialized += 1;
    }
    return host_capabilities;
}

fn buildHostResidencies(
    allocator: std.mem.Allocator,
    plan: *const models.stage_plan.StagePlan,
    state_ref: ?*const pipeline.StageStatePlacementRef,
) ![]pipeline.HostResidencySnapshot {
    const residencies = try allocator.alloc(pipeline.HostResidencySnapshot, plan.stages.len);
    var initialized: usize = 0;
    errdefer {
        for (residencies[0..initialized]) |*residency| residency.deinit();
        allocator.free(residencies);
    }

    const state_contract_version: ?u32 = if (state_ref) |ref| ref.state_ownership_contract_version else null;
    const state_plan_id: ?pipeline.StageStateOwnershipPlanId = if (state_ref) |ref| ref.state_ownership_plan_id else null;
    for (plan.stages, 0..) |stage, index| {
        if (stage.id != index) return error.InvalidTopologyConfig;
        const resident = [_]pipeline.ResidentStageEntry{residentEntryFromStage(stage, state_ref)};
        residencies[index] = try pipeline.buildHostResidencySnapshot(allocator, .{
            .host_id = try deterministicHostId(stage.id),
            .plan = plan,
            .state_ownership_contract_version = state_contract_version,
            .state_ownership_plan_id = state_plan_id,
            .resident_stages = &resident,
        });
        initialized += 1;
    }
    return residencies;
}

fn buildStageHostBindings(
    allocator: std.mem.Allocator,
    plan: *const models.stage_plan.StagePlan,
) ![]pipeline.StageHostBinding {
    const bindings = try allocator.alloc(pipeline.StageHostBinding, plan.stages.len);
    errdefer allocator.free(bindings);
    for (plan.stages, 0..) |stage, index| {
        if (stage.id != index) return error.InvalidTopologyConfig;
        bindings[index] = .{
            .stage_id = stage.id,
            .host_id = try deterministicHostId(stage.id),
        };
    }
    return bindings;
}

pub fn buildStateOwnershipPlan(
    allocator: std.mem.Allocator,
    plan: *const models.stage_plan.StagePlan,
) !pipeline.StageStateOwnershipPlan {
    const descriptor_sets = try allocator.alloc(pipeline.StageStateDescriptorSet, plan.stages.len);
    defer allocator.free(descriptor_sets);
    for (plan.stages, 0..) |stage, index| {
        descriptor_sets[index] = .{ .stage_id = stage.id, .descriptors = &.{} };
    }

    const fact_count = countStatefulDependencies(plan);
    const facts = try allocator.alloc(pipeline.StageStatePartitionFact, fact_count);
    defer allocator.free(facts);
    var fact_index: usize = 0;
    for (plan.dependencies) |dependency| {
        if (dependency.reason != .stateful_decoder) continue;
        facts[fact_index] = .{
            .boundary_index = boundaryIndexForDependency(plan, dependency) orelse return error.InvalidTopologyConfig,
            .source_stage_id = dependency.source_stage_id,
            .target_stage_id = dependency.target_stage_id,
            .reason = .stateful_decoder,
            .ownership_mode = .stage_level_dependency_only,
        };
        fact_index += 1;
    }

    return pipeline.buildStageStateOwnershipPlan(allocator, .{
        .plan = plan,
        .descriptor_sets = descriptor_sets,
        .partition_facts = facts,
    });
}

pub fn buildPlacementPlan(
    allocator: std.mem.Allocator,
    d_model: usize,
    plan: *const models.stage_plan.StagePlan,
    stage_backend_kinds: []const pipeline.HostBackendKind,
    boundary_configs: []const BoundaryConfig,
    state_ref: ?*const pipeline.StageStatePlacementRef,
) !pipeline.PlacementPlan {
    if (plan.stages.len != stage_backend_kinds.len or plan.boundaries.len != boundary_configs.len) return error.InvalidTopologyConfig;
    const profiles = try buildBoundaryProfiles(allocator, d_model, plan, boundary_configs);
    defer allocator.free(profiles);
    const host_capabilities = try buildHostCapabilities(allocator, plan, stage_backend_kinds, profiles, state_ref);
    defer {
        for (host_capabilities) |*capability| capability.deinit();
        allocator.free(host_capabilities);
    }
    const residencies = try buildHostResidencies(allocator, plan, state_ref);
    defer {
        for (residencies) |*residency| residency.deinit();
        allocator.free(residencies);
    }
    const bindings = try buildStageHostBindings(allocator, plan);
    defer allocator.free(bindings);

    return pipeline.buildPlacementPlan(allocator, .{
        .plan = plan,
        .required_step_kinds = &required_step_kinds,
        .host_capabilities = host_capabilities,
        .host_residency_snapshots = residencies,
        .stage_host_bindings = bindings,
        .boundary_frame_profiles = profiles,
        .state_placement_mode = if (state_ref != null) .validate_ref else .stateless_only,
        .state_placement_ref = state_ref,
        .allowed_reachability = &.{.local_in_process},
        .stateful_execution_required = state_ref != null,
    });
}

test "validateStageSpecs accepts contiguous ordered local stages" {
    try validateStageSpecs(4, &.{
        .{ .stage_id = 0, .backend_kind = .cpu, .layer_start = 0, .layer_end = 2, .owns_embedding = true, .owns_projection = false },
        .{ .stage_id = 1, .backend_kind = .cuda, .layer_start = 2, .layer_end = 4, .owns_embedding = false, .owns_projection = true },
    });
}

test "validateStageSpecs rejects gaps and wrong ownership" {
    try std.testing.expectError(error.InvalidTopologyConfig, validateStageSpecs(4, &.{
        .{ .stage_id = 0, .backend_kind = .cpu, .layer_start = 0, .layer_end = 1, .owns_embedding = true, .owns_projection = false },
        .{ .stage_id = 1, .backend_kind = .cuda, .layer_start = 2, .layer_end = 4, .owns_embedding = false, .owns_projection = true },
    }));
    try std.testing.expectError(error.InvalidTopologyConfig, validateStageSpecs(4, &.{
        .{ .stage_id = 0, .backend_kind = .cpu, .layer_start = 0, .layer_end = 2, .owns_embedding = false, .owns_projection = false },
        .{ .stage_id = 1, .backend_kind = .cuda, .layer_start = 2, .layer_end = 4, .owns_embedding = false, .owns_projection = true },
    }));
}

test "boundaryProfilePair emits prefill and decode profiles" {
    const pair = try boundaryProfilePair(8, 0, .{
        .source_stage_id = 0,
        .target_stage_id = 1,
        .producer_layer_start = 0,
        .producer_layer_end = 2,
        .consumer_layer_start = 2,
        .consumer_layer_end = 4,
    }, boundaryConfig(.f32, .row_major, 3, 5));
    try std.testing.expectEqual(pipeline.TensorFrameStepKind.prefill, pair[0].step_kind);
    try std.testing.expectEqual(@as(u64, 5 * 8 * @sizeOf(f32)), pair[0].max_activation_payload_bytes);
    try std.testing.expectEqual(pipeline.TensorFrameStepKind.decode, pair[1].step_kind);
    try std.testing.expectEqual(@as(u64, 3), pair[1].max_batch_entries);
}
