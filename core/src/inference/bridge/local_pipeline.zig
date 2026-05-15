//! Bridge-owned local pipeline execution helpers.
//!
//! This module is the production entry point for local adjacent stage-chain
//! execution. Backends provide concrete stage adapters; bridge owns the ordered
//! stage-chain request shape and delegates activation movement through the
//! local stage transport contract.

const std = @import("std");

const boundary_byte_image = @import("boundary_byte_image.zig");
const host_capability = @import("host_capability.zig");
const local_stage_runner = @import("local_stage_runner.zig");
const state_ownership = @import("state_ownership.zig");
const tensor_frame = @import("tensor_frame.zig");

const Allocator = std.mem.Allocator;

pub const LocalPipelineContext = struct {
    allocator: ?Allocator = null,
    plan_ref: *const local_stage_runner.LocalStageRunnerPlanRef,
    placement_plan: *const host_capability.PlacementPlan,
    state_ownership_plan: ?*const state_ownership.StageStateOwnershipPlan = null,
    cleanup_obligations: []const state_ownership.StateCleanupObligation = &.{},
};

pub const LocalPipelineBoundaryRuntime = struct {
    staging: ?[]align(64) u8 = null,
    allow_borrow: bool = false,
    local_device_peer_copy_available: bool = false,
};

pub const LocalPipelineBoundaryPayload = struct {
    metadata: *const tensor_frame.TensorFrameMetadata,
    image: *const boundary_byte_image.BoundaryByteImageRef,
    runtime: LocalPipelineBoundaryRuntime = .{},
};

pub const LocalPipelineStageBinding = struct {
    stage_id: usize,
    backend_kind: host_capability.HostBackendKind,
};

pub const LocalPipelinePlacementKind = enum {
    cuda_cuda,
    cpu_cuda,
    cpu_cuda_cuda,
    generic_local_chain,
};

pub fn resolveLocalPipelinePlacementKind(
    context: LocalPipelineContext,
    stage_bindings: []const LocalPipelineStageBinding,
) anyerror!LocalPipelinePlacementKind {
    try validateLocalPipelineStageBindings(context, stage_bindings);
    return classifyLocalPipelinePlacementKind(stage_bindings);
}

pub fn classifyLocalPipelinePlacementKind(
    stage_bindings: []const LocalPipelineStageBinding,
) anyerror!LocalPipelinePlacementKind {
    if (stage_bindings.len < 2) return error.InvalidStageRange;
    if (stage_bindings.len == 2) {
        if (stage_bindings[0].backend_kind == .cuda and stage_bindings[1].backend_kind == .cuda) {
            return .cuda_cuda;
        }
        if (stage_bindings[0].backend_kind == .cpu and stage_bindings[1].backend_kind == .cuda) {
            return .cpu_cuda;
        }
    }
    if (stage_bindings.len == 3 and
        stage_bindings[0].backend_kind == .cpu and
        stage_bindings[1].backend_kind == .cuda and
        stage_bindings[2].backend_kind == .cuda)
    {
        return .cpu_cuda_cuda;
    }
    return .generic_local_chain;
}

pub fn validateLocalPipelineStageBindings(
    context: LocalPipelineContext,
    stage_bindings: []const LocalPipelineStageBinding,
) anyerror!void {
    try local_stage_runner.validateLocalStageRunnerPlanRef(context.plan_ref);
    try host_capability.validatePlacementPlan(context.placement_plan);
    if (stage_bindings.len < 2) return error.InvalidStageRange;
    if (stage_bindings.len != context.plan_ref.stages.len) return error.InvalidStepRequest;
    if (context.plan_ref.boundaries.len + 1 != stage_bindings.len) return error.InvalidStepRequest;

    for (stage_bindings, context.plan_ref.stages, 0..) |binding, stage_ref, index| {
        if (binding.stage_id != stage_ref.stage_id) return error.StageRunnerPlanIdentityMismatch;
        if (index > 0 and stage_bindings[index - 1].stage_id >= binding.stage_id) {
            return error.DuplicateStageRef;
        }
    }
    for (context.plan_ref.boundaries, 0..) |boundary, index| {
        if (boundary.boundary_index != index) return error.BoundaryIndexOutOfRange;
        if (boundary.source_stage_id != stage_bindings[index].stage_id or
            boundary.target_stage_id != stage_bindings[index + 1].stage_id)
        {
            return error.StageRunnerPlanIdentityMismatch;
        }
    }
}

test "classifyLocalPipelinePlacementKind classifies current local chain placements" {
    try std.testing.expectEqual(
        LocalPipelinePlacementKind.cuda_cuda,
        try classifyLocalPipelinePlacementKind(&.{
            .{ .stage_id = 0, .backend_kind = .cuda },
            .{ .stage_id = 1, .backend_kind = .cuda },
        }),
    );
    try std.testing.expectEqual(
        LocalPipelinePlacementKind.cpu_cuda,
        try classifyLocalPipelinePlacementKind(&.{
            .{ .stage_id = 0, .backend_kind = .cpu },
            .{ .stage_id = 1, .backend_kind = .cuda },
        }),
    );
    try std.testing.expectEqual(
        LocalPipelinePlacementKind.cpu_cuda_cuda,
        try classifyLocalPipelinePlacementKind(&.{
            .{ .stage_id = 0, .backend_kind = .cpu },
            .{ .stage_id = 1, .backend_kind = .cuda },
            .{ .stage_id = 2, .backend_kind = .cuda },
        }),
    );
    try std.testing.expectEqual(
        LocalPipelinePlacementKind.generic_local_chain,
        try classifyLocalPipelinePlacementKind(&.{
            .{ .stage_id = 0, .backend_kind = .cuda },
            .{ .stage_id = 1, .backend_kind = .cpu },
        }),
    );
    try std.testing.expectError(error.InvalidStageRange, classifyLocalPipelinePlacementKind(&.{
        .{ .stage_id = 0, .backend_kind = .cuda },
    }));
}

test "validateLocalPipelineStageBindings rejects invalid bridge runner contract" {
    var plan_ref: local_stage_runner.LocalStageRunnerPlanRef = undefined;
    plan_ref.version = local_stage_runner.local_stage_runner_contract_version + 1;
    var placement_plan: host_capability.PlacementPlan = undefined;

    try std.testing.expectError(
        error.InvalidLocalStageRunnerContractVersion,
        validateLocalPipelineStageBindings(.{
            .plan_ref = &plan_ref,
            .placement_plan = &placement_plan,
        }, &.{
            .{ .stage_id = 0, .backend_kind = .cpu },
            .{ .stage_id = 1, .backend_kind = .cuda },
        }),
    );
}

test "resolveLocalPipelinePlacementKind rejects invalid bridge runner contract" {
    var plan_ref: local_stage_runner.LocalStageRunnerPlanRef = undefined;
    plan_ref.version = local_stage_runner.local_stage_runner_contract_version + 1;
    var placement_plan: host_capability.PlacementPlan = undefined;

    try std.testing.expectError(
        error.InvalidLocalStageRunnerContractVersion,
        resolveLocalPipelinePlacementKind(.{
            .plan_ref = &plan_ref,
            .placement_plan = &placement_plan,
        }, &.{
            .{ .stage_id = 0, .backend_kind = .cpu },
            .{ .stage_id = 1, .backend_kind = .cuda },
        }),
    );
}

pub fn executeLocalPipelineChain(
    context: LocalPipelineContext,
    stages: []local_stage_runner.LocalStageChainStage,
    boundaries: []const local_stage_runner.LocalStageChainBoundaryStep,
    stage_inputs: []const []const u8,
) anyerror!void {
    try local_stage_runner.executeLocalStageChain(.{
        .allocator = context.allocator,
        .plan_ref = context.plan_ref,
        .placement_plan = context.placement_plan,
        .state_ownership_plan = context.state_ownership_plan,
        .cleanup_obligations = context.cleanup_obligations,
        .stages = stages,
        .boundaries = boundaries,
        .stage_inputs = stage_inputs,
    });
}

pub fn executeLocalPipelineStep(
    context: LocalPipelineContext,
    stages: []local_stage_runner.LocalStageChainStage,
    boundary_payloads: []const LocalPipelineBoundaryPayload,
    comptime step_kind: tensor_frame.TensorFrameStepKind,
    stage_inputs: []const []const u8,
) anyerror!void {
    if (stages.len < 2) return error.InvalidStageRange;
    if (boundary_payloads.len + 1 != stages.len) return error.InvalidStepRequest;

    var inline_steps: [8]local_stage_runner.LocalStageChainBoundaryStep = undefined;
    var heap_steps: []local_stage_runner.LocalStageChainBoundaryStep = &.{};
    defer if (heap_steps.len != 0) context.allocator.?.free(heap_steps);

    const steps = if (boundary_payloads.len <= inline_steps.len)
        inline_steps[0..boundary_payloads.len]
    else blk: {
        const allocator = context.allocator orelse return error.InvalidStepRequest;
        heap_steps = try allocator.alloc(local_stage_runner.LocalStageChainBoundaryStep, boundary_payloads.len);
        break :blk heap_steps;
    };

    for (boundary_payloads, steps) |payload, *step| {
        step.* = .{
            .boundary_index = payload.metadata.boundary.boundary_index,
            .step_kind = step_kind,
            .metadata = payload.metadata,
            .image = payload.image,
            .staging = payload.runtime.staging,
            .allow_borrow = payload.runtime.allow_borrow,
            .local_device_peer_copy_available = payload.runtime.local_device_peer_copy_available,
        };
    }

    try executeLocalPipelineChain(context, stages, steps, stage_inputs);
}

pub fn executeLocalPipelineBoundary(
    comptime Source: type,
    comptime Target: type,
    context: LocalPipelineContext,
    comptime step_kind: tensor_frame.TensorFrameStepKind,
    source: *Source,
    target: *Target,
    metadata: *const tensor_frame.TensorFrameMetadata,
    image: *const boundary_byte_image.BoundaryByteImageRef,
    runtime: LocalPipelineBoundaryRuntime,
) anyerror!void {
    var stages = [_]local_stage_runner.LocalStageChainStage{
        local_stage_runner.localStageAdapter(Source, metadata.boundary.source_stage_id, source),
        local_stage_runner.localStageAdapter(Target, metadata.boundary.target_stage_id, target),
    };
    const boundary_payloads = [_]LocalPipelineBoundaryPayload{.{
        .metadata = metadata,
        .image = image,
        .runtime = runtime,
    }};
    try executeLocalPipelineStep(context, stages[0..], boundary_payloads[0..], step_kind, &.{});
}

test "executeLocalPipelineChain rejects invalid bridge runner contract" {
    var plan_ref: local_stage_runner.LocalStageRunnerPlanRef = undefined;
    plan_ref.version = local_stage_runner.local_stage_runner_contract_version + 1;
    var placement_plan: host_capability.PlacementPlan = undefined;
    var chain_stages: [1]local_stage_runner.LocalStageChainStage = undefined;

    try std.testing.expectError(
        error.InvalidLocalStageRunnerContractVersion,
        executeLocalPipelineChain(.{
            .plan_ref = &plan_ref,
            .placement_plan = &placement_plan,
        }, chain_stages[0..], &.{}, &.{}),
    );
}

test "executeLocalPipelineStep rejects invalid bridge runner contract" {
    var plan_ref: local_stage_runner.LocalStageRunnerPlanRef = undefined;
    plan_ref.version = local_stage_runner.local_stage_runner_contract_version + 1;
    var placement_plan: host_capability.PlacementPlan = undefined;
    var chain_stages: [2]local_stage_runner.LocalStageChainStage = undefined;
    var metadata: tensor_frame.TensorFrameMetadata = undefined;
    metadata.boundary = .{
        .boundary_index = 0,
        .source_stage_id = 0,
        .target_stage_id = 1,
        .producer_layer_start = 0,
        .producer_layer_end = 1,
        .consumer_layer_start = 1,
        .consumer_layer_end = 2,
    };
    var image: boundary_byte_image.BoundaryByteImageRef = undefined;
    const payloads = [_]LocalPipelineBoundaryPayload{.{
        .metadata = &metadata,
        .image = &image,
    }};

    try std.testing.expectError(
        error.InvalidLocalStageRunnerContractVersion,
        executeLocalPipelineStep(.{
            .plan_ref = &plan_ref,
            .placement_plan = &placement_plan,
        }, chain_stages[0..], payloads[0..], .decode, &.{}),
    );
}

test "executeLocalPipelineStep rejects invalid stage boundary contract shape" {
    var plan_ref: local_stage_runner.LocalStageRunnerPlanRef = undefined;
    plan_ref.version = local_stage_runner.local_stage_runner_contract_version + 1;
    var placement_plan: host_capability.PlacementPlan = undefined;
    var chain_stages: [2]local_stage_runner.LocalStageChainStage = undefined;

    try std.testing.expectError(
        error.InvalidStepRequest,
        executeLocalPipelineStep(.{
            .plan_ref = &plan_ref,
            .placement_plan = &placement_plan,
        }, chain_stages[0..], &.{}, .decode, &.{}),
    );
}

test "executeLocalPipelineBoundary rejects invalid bridge runner contract" {
    const Stage = struct {
        pub fn executeLayers(_: *@This(), _: []const u8, _: usize, _: usize) anyerror!void {}
        pub fn synchronize(_: *@This()) anyerror!void {}
        pub fn downloadActivation(_: *@This(), _: []u8, _: usize) anyerror!void {}
        pub fn uploadActivation(_: *@This(), _: []const u8, _: usize) anyerror!void {}
    };

    var plan_ref: local_stage_runner.LocalStageRunnerPlanRef = undefined;
    plan_ref.version = local_stage_runner.local_stage_runner_contract_version + 1;
    var placement_plan: host_capability.PlacementPlan = undefined;
    var metadata: tensor_frame.TensorFrameMetadata = undefined;
    metadata.boundary = .{
        .boundary_index = 0,
        .source_stage_id = 0,
        .target_stage_id = 1,
        .producer_layer_start = 0,
        .producer_layer_end = 1,
        .consumer_layer_start = 1,
        .consumer_layer_end = 2,
    };
    var image: boundary_byte_image.BoundaryByteImageRef = undefined;
    var source = Stage{};
    var target = Stage{};

    try std.testing.expectError(
        error.InvalidLocalStageRunnerContractVersion,
        executeLocalPipelineBoundary(
            Stage,
            Stage,
            .{
                .plan_ref = &plan_ref,
                .placement_plan = &placement_plan,
            },
            .decode,
            &source,
            &target,
            &metadata,
            &image,
            .{},
        ),
    );
}
