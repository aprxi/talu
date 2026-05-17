//! Pipeline integration tests for local stage contract bundles.

const std = @import("std");
const main = @import("main");

const pipeline = main.inference.pipeline;
const models = main.models.dispatcher;
const local_stage_testing = @import("local_stage_test_helpers.zig");

const LocalStageStateFixture = struct {
    plan: pipeline.StageStateOwnershipPlan,
    ref: pipeline.StageStatePlacementRef,

    fn deinit(self: *@This()) void {
        self.ref.deinit();
        self.plan.deinit();
    }
};

fn buildLocalStageStateFixture(
    allocator: std.mem.Allocator,
    plan: *const models.stage_plan.StagePlan,
) !LocalStageStateFixture {
    var state_plan = try local_stage_testing.buildLocalStageStateOwnershipPlan(allocator, plan);
    errdefer state_plan.deinit();
    var state_ref = try pipeline.buildStageStatePlacementRef(allocator, &state_plan);
    errdefer state_ref.deinit();
    return .{ .plan = state_plan, .ref = state_ref };
}

fn expectLocalStagePlacement(
    placement: *const pipeline.PlacementPlan,
    d_model: usize,
    stage_count: usize,
    configs: []const local_stage_testing.BoundaryConfig,
) !void {
    try pipeline.validatePlacementPlan(placement);
    try std.testing.expectEqual(stage_count, placement.stage_summaries.len);
    try std.testing.expectEqual(stage_count - 1, placement.boundary_summaries.len);
    try std.testing.expectEqual(stage_count, placement.stage_host_bindings.len);
    try std.testing.expectEqualSlices(
        pipeline.TensorFrameStepKind,
        &local_stage_testing.runtime_stage_required_step_kinds,
        placement.required_step_kinds,
    );

    for (placement.stage_host_bindings, 0..) |binding, stage_id| {
        try std.testing.expectEqual(stage_id, binding.stage_id);
        try std.testing.expectEqual(@as(u64, @intCast(stage_id + 1)), binding.host_id.value);
    }

    try std.testing.expectEqual(configs.len * local_stage_testing.runtime_stage_required_step_kinds.len, placement.boundary_frame_profiles.len);
    for (configs, 0..) |config, boundary_index| {
        const row_bytes = try local_stage_testing.boundaryRowByteCount(d_model, config.dtype);
        const prefill = placement.boundary_frame_profiles[boundary_index * 2];
        try std.testing.expectEqual(boundary_index, prefill.boundary_index);
        try std.testing.expectEqual(pipeline.TensorFrameStepKind.prefill, prefill.step_kind);
        try std.testing.expectEqual(config.dtype, prefill.dtype);
        try std.testing.expectEqual(config.layout, prefill.layout);
        try std.testing.expectEqual(@as(u64, 1), prefill.max_batch_entries);
        try std.testing.expectEqual(@as(u64, @intCast(config.prefill_max_token_count_per_frame)), prefill.max_token_count_per_frame);
        try std.testing.expectEqual(
            row_bytes * @as(u64, @intCast(config.prefill_max_token_count_per_frame)),
            prefill.max_activation_payload_bytes,
        );

        const decode = placement.boundary_frame_profiles[boundary_index * 2 + 1];
        try std.testing.expectEqual(boundary_index, decode.boundary_index);
        try std.testing.expectEqual(pipeline.TensorFrameStepKind.decode, decode.step_kind);
        try std.testing.expectEqual(config.dtype, decode.dtype);
        try std.testing.expectEqual(config.layout, decode.layout);
        try std.testing.expectEqual(@as(u64, @intCast(config.decode_max_batch_entries)), decode.max_batch_entries);
        try std.testing.expect(decode.max_batch_entries > 1);
        try std.testing.expectEqual(@as(u64, 1), decode.max_token_count_per_frame);
        try std.testing.expectEqual(
            row_bytes * @as(u64, @intCast(config.decode_max_batch_entries)),
            decode.max_activation_payload_bytes,
        );
    }
}

fn expectPlacementBuildFailureCleanup(
    d_model: usize,
    plan: *const models.stage_plan.StagePlan,
    stage_backend_kinds: []const pipeline.HostBackendKind,
    configs: []const local_stage_testing.BoundaryConfig,
    state_ref: ?*const pipeline.StageStatePlacementRef,
) !void {
    var saw_success = false;
    for (0..64) |fail_index| {
        var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        var placement_or_error = local_stage_testing.buildLocalStagePlacementPlan(
            failing.allocator(),
            d_model,
            plan,
            stage_backend_kinds,
            configs,
            state_ref,
        );
        if (placement_or_error) |*placement| {
            placement.deinit();
            try std.testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
            saw_success = true;
            break;
        } else |err| {
            try std.testing.expectEqual(error.OutOfMemory, err);
            try std.testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
        }
    }
    try std.testing.expect(saw_success);
}

test "pipeline local stage contract validates stage specs and boundary runtimes" {
    const valid = [_]local_stage_testing.StageSpec{
        .{ .stage_id = 0, .backend_kind = .cpu, .layer_start = 0, .layer_end = 1, .owns_embedding = true, .owns_projection = false },
        .{ .stage_id = 1, .backend_kind = .cuda, .layer_start = 1, .layer_end = 3, .owns_embedding = false, .owns_projection = false },
        .{ .stage_id = 2, .backend_kind = .cuda, .layer_start = 3, .layer_end = 4, .owns_embedding = false, .owns_projection = true },
    };
    try local_stage_testing.validateStageSpecs(4, &valid);

    var gap = valid;
    gap[1].layer_start = 2;
    try std.testing.expectError(error.InvalidTopologyConfig, local_stage_testing.validateStageSpecs(4, &gap));

    const boundaries = [_]local_stage_testing.BoundaryRuntime{
        .{ .boundary_index = 0, .dtype = .f32, .layout = .row_major },
        .{ .boundary_index = 1, .dtype = .f16, .layout = .row_major, .local_device_peer_copy_available = true },
    };
    try local_stage_testing.validateBoundaryRuntimes(3, &boundaries);

    var wrong_order = boundaries;
    wrong_order[1].boundary_index = 7;
    try std.testing.expectError(error.InvalidTopologyConfig, local_stage_testing.validateBoundaryRuntimes(3, &wrong_order));
}

test "pipeline local stage contract builds placement and state facts for ordered host chains" {
    const d_model: usize = 8;
    const two_stage_deps = [_]models.stage_plan.DependencyOverride{.{
        .source_stage_id = 0,
        .target_stage_id = 1,
        .reason = .stateful_decoder,
        .affects_loader_residency = false,
    }};

    var two_stage_plan = try local_stage_testing.buildLocalStageTestStagePlan(std.testing.allocator, 4, &.{2}, &two_stage_deps);
    defer two_stage_plan.deinit();
    var two_stage_state = try buildLocalStageStateFixture(std.testing.allocator, &two_stage_plan);
    defer two_stage_state.deinit();
    const two_stage_kinds = [_]pipeline.HostBackendKind{ .cpu, .cuda };
    const two_stage_configs = [_]local_stage_testing.BoundaryConfig{
        local_stage_testing.localBoundaryConfig(.f32, .row_major, 4, 9),
    };
    var two_stage_placement = try local_stage_testing.buildLocalStagePlacementPlan(
        std.testing.allocator,
        d_model,
        &two_stage_plan,
        &two_stage_kinds,
        &two_stage_configs,
        &two_stage_state.ref,
    );
    defer two_stage_placement.deinit();
    try expectLocalStagePlacement(&two_stage_placement, d_model, 2, &two_stage_configs);
    try std.testing.expectEqual(pipeline.StatePlacementMode.validate_ref, two_stage_placement.state_placement_mode);
    try std.testing.expectEqual(@as(usize, 2), two_stage_placement.state_stage_summaries.len);

    const three_stage_deps = [_]models.stage_plan.DependencyOverride{
        .{ .source_stage_id = 0, .target_stage_id = 1, .reason = .stateful_decoder, .affects_loader_residency = false },
        .{ .source_stage_id = 1, .target_stage_id = 2, .reason = .stateful_decoder, .affects_loader_residency = false },
    };
    var three_stage_plan = try local_stage_testing.buildLocalStageTestStagePlan(std.testing.allocator, 5, &.{ 1, 3 }, &three_stage_deps);
    defer three_stage_plan.deinit();
    var three_stage_state = try buildLocalStageStateFixture(std.testing.allocator, &three_stage_plan);
    defer three_stage_state.deinit();
    const three_stage_kinds = [_]pipeline.HostBackendKind{ .cpu, .cuda, .cuda };
    const three_stage_configs = local_stage_testing.localTwoBoundaryConfigs(.f32, .row_major, .f16, .row_major, 5, 9, 16, 64);
    var three_stage_placement = try local_stage_testing.buildLocalStagePlacementPlan(
        std.testing.allocator,
        d_model,
        &three_stage_plan,
        &three_stage_kinds,
        &three_stage_configs,
        &three_stage_state.ref,
    );
    defer three_stage_placement.deinit();
    try expectLocalStagePlacement(&three_stage_placement, d_model, 3, &three_stage_configs);
    try pipeline.validateStageStateOwnershipPlan(&three_stage_state.plan);
    try std.testing.expectEqual(@as(usize, 2), three_stage_state.plan.boundaries.len);
}

test "pipeline local stage contract bundle deinit is ordered and idempotent" {
    const expected_order = [_]local_stage_testing.ContractField{
        .local_stage_runner_plan_ref,
        .placement_plan,
        .state_placement_ref,
        .state_ownership_plan,
        .tensor_frame_plan_ref,
        .stage_plan,
    };
    try std.testing.expectEqualSlices(local_stage_testing.ContractField, &expected_order, &local_stage_testing.contract_deinit_order);

    const deps = [_]models.stage_plan.DependencyOverride{.{
        .source_stage_id = 0,
        .target_stage_id = 1,
        .reason = .stateful_decoder,
        .affects_loader_residency = false,
    }};
    const kinds = [_]pipeline.HostBackendKind{ .cpu, .cuda };
    const configs = [_]local_stage_testing.BoundaryConfig{
        local_stage_testing.localBoundaryConfig(.f32, .row_major, 4, 9),
    };
    const plan = try local_stage_testing.buildLocalStageTestStagePlan(std.testing.allocator, 4, &.{2}, &deps);
    var bundle = try local_stage_testing.buildLocalStageContractBundleFromOwnedPlan(
        std.testing.allocator,
        8,
        plan,
        &kinds,
        &configs,
    );
    defer bundle.deinit();
    local_stage_testing.deinitLocalStageContractBundleTwice(&bundle);
}

test "pipeline local stage contract placement builder cleans allocation failures" {
    const deps = [_]models.stage_plan.DependencyOverride{.{
        .source_stage_id = 0,
        .target_stage_id = 1,
        .reason = .stateful_decoder,
        .affects_loader_residency = false,
    }};
    var plan = try local_stage_testing.buildLocalStageTestStagePlan(std.testing.allocator, 4, &.{2}, &deps);
    defer plan.deinit();
    var state = try buildLocalStageStateFixture(std.testing.allocator, &plan);
    defer state.deinit();
    const kinds = [_]pipeline.HostBackendKind{ .cpu, .cuda };
    const configs = [_]local_stage_testing.BoundaryConfig{
        local_stage_testing.localBoundaryConfig(.f32, .row_major, 4, 9),
    };

    try expectPlacementBuildFailureCleanup(8, &plan, &kinds, &configs, &state.ref);
}
