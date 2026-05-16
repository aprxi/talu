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

pub const LocalPipelineBoundaryFrameSpec = struct {
    boundary_index: usize,
    dtype: tensor_frame.TensorFrameDType,
    layout: tensor_frame.TensorFrameLayout,
    staging: ?[]align(64) u8 = null,
    local_device_peer_copy_available: bool = false,
};

pub const LocalDecodeBoundaryImageSpec = union(enum) {
    device,
    host_bytes: []const u8,
    host_segments: []const []const u8,
};

pub const LocalDecodeBoundaryPayloadSpec = struct {
    frame: LocalPipelineBoundaryFrameSpec,
    activation_byte_count: usize,
    location_hint: ?tensor_frame.TensorFramePayloadLocationHint,
    image: LocalDecodeBoundaryImageSpec,
    local_device_peer_copy_available: ?bool = null,
};

pub const LocalDecodePipelineStepRequest = struct {
    tensor_frame_plan_ref: *const tensor_frame.TensorFramePlanRef,
    hidden_size: usize,
    slot_request_ids: []const ?u64,
    slot_indices: []const usize,
    positions: []const usize,
    boundary_payloads: []const LocalDecodeBoundaryPayloadSpec,
    stage_inputs: []const []const u8 = &.{},
};

pub const LocalPrefillBoundaryImageSpec = union(enum) {
    device,
    host_bytes: []const u8,
};

pub const LocalPrefillBoundaryPayloadSpec = struct {
    frame: LocalPipelineBoundaryFrameSpec,
    slot_index: usize,
    sequence_start: usize,
    token_count: usize,
    activation_byte_count: usize,
    location_hint: ?tensor_frame.TensorFramePayloadLocationHint,
    image: LocalPrefillBoundaryImageSpec,
    local_device_peer_copy_available: ?bool = null,
};

pub const LocalPrefillPipelineStepRequest = struct {
    tensor_frame_plan_ref: *const tensor_frame.TensorFramePlanRef,
    hidden_size: usize,
    slot_request_ids: []const ?u64,
    boundary_payloads: []const LocalPrefillBoundaryPayloadSpec,
    stage_inputs: []const []const u8 = &.{},
};

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

pub fn executeLocalDecodePipelineStep(
    context: LocalPipelineContext,
    stages: []local_stage_runner.LocalStageChainStage,
    request: LocalDecodePipelineStepRequest,
) anyerror!void {
    if (request.slot_indices.len != request.positions.len) return error.InvalidArgument;
    if (request.boundary_payloads.len == 0 or request.boundary_payloads.len + 1 != stages.len) {
        return error.InvalidStepRequest;
    }
    if (request.slot_indices.len == 0) return error.InvalidArgument;

    const boundary_count = request.boundary_payloads.len;
    var inline_metadata: [8]tensor_frame.TensorFrameMetadata = undefined;
    var inline_images: [8]boundary_byte_image.BoundaryByteImageRef = undefined;
    var inline_payloads: [8]LocalPipelineBoundaryPayload = undefined;
    var heap_metadata: []tensor_frame.TensorFrameMetadata = &.{};
    var heap_images: []boundary_byte_image.BoundaryByteImageRef = &.{};
    var heap_payloads: []LocalPipelineBoundaryPayload = &.{};
    defer {
        if (heap_payloads.len != 0) context.allocator.?.free(heap_payloads);
        if (heap_images.len != 0) context.allocator.?.free(heap_images);
        if (heap_metadata.len != 0) context.allocator.?.free(heap_metadata);
    }

    const metadata = if (boundary_count <= inline_metadata.len)
        inline_metadata[0..boundary_count]
    else blk: {
        const allocator = context.allocator orelse return error.InvalidStepRequest;
        heap_metadata = try allocator.alloc(tensor_frame.TensorFrameMetadata, boundary_count);
        break :blk heap_metadata;
    };
    const images = if (boundary_count <= inline_images.len)
        inline_images[0..boundary_count]
    else blk: {
        const allocator = context.allocator orelse return error.InvalidStepRequest;
        heap_images = try allocator.alloc(boundary_byte_image.BoundaryByteImageRef, boundary_count);
        break :blk heap_images;
    };
    const payloads = if (boundary_count <= inline_payloads.len)
        inline_payloads[0..boundary_count]
    else blk: {
        const allocator = context.allocator orelse return error.InvalidStepRequest;
        heap_payloads = try allocator.alloc(LocalPipelineBoundaryPayload, boundary_count);
        break :blk heap_payloads;
    };

    const entry_count = std.math.mul(usize, boundary_count, request.slot_indices.len) catch return error.InvalidArgument;
    var inline_entries: [256]tensor_frame.TensorFrameBatchEntry = undefined;
    var heap_entries: []tensor_frame.TensorFrameBatchEntry = &.{};
    defer if (heap_entries.len != 0) context.allocator.?.free(heap_entries);
    const entries = if (entry_count <= inline_entries.len)
        inline_entries[0..entry_count]
    else blk: {
        const allocator = context.allocator orelse return error.InvalidStepRequest;
        heap_entries = try allocator.alloc(tensor_frame.TensorFrameBatchEntry, entry_count);
        break :blk heap_entries;
    };

    for (request.boundary_payloads, 0..) |spec, index| {
        const batch_start = index * request.slot_indices.len;
        const batch_entries = entries[batch_start..][0..request.slot_indices.len];
        metadata[index] = try local_stage_runner.buildDecodeActivationMetadata(.{
            .plan_ref = request.tensor_frame_plan_ref,
            .hidden_size = request.hidden_size,
            .boundary_index = spec.frame.boundary_index,
            .dtype = spec.frame.dtype,
            .layout = spec.frame.layout,
            .location_hint = spec.location_hint,
            .slot_request_ids = request.slot_request_ids,
            .slot_indices = request.slot_indices,
            .positions = request.positions,
            .batch_entries = batch_entries,
        });
        try tensor_frame.validatePayloadBufferLength(&metadata[index], spec.activation_byte_count);
        images[index] = switch (spec.image) {
            .device => local_stage_runner.deviceActivationByteImage(&metadata[index]),
            .host_bytes => |host_bytes| blk: {
                if (spec.activation_byte_count > host_bytes.len) return error.InvalidArgument;
                break :blk local_stage_runner.hostActivationByteImage(
                    &metadata[index],
                    host_bytes[0..spec.activation_byte_count],
                );
            },
            .host_segments => |host_segments| blk: {
                if (host_segments.len != request.slot_indices.len) return error.InvalidArgument;
                break :blk local_stage_runner.segmentedHostActivationByteImage(&metadata[index], host_segments);
            },
        };
        payloads[index] = .{
            .metadata = &metadata[index],
            .image = &images[index],
            .runtime = .{
                .staging = spec.frame.staging,
                .allow_borrow = false,
                .local_device_peer_copy_available = spec.local_device_peer_copy_available orelse
                    spec.frame.local_device_peer_copy_available,
            },
        };
    }

    try executeLocalPipelineStep(context, stages, payloads, .decode, request.stage_inputs);
}

pub fn executeLocalPrefillPipelineStep(
    context: LocalPipelineContext,
    stages: []local_stage_runner.LocalStageChainStage,
    request: LocalPrefillPipelineStepRequest,
) anyerror!void {
    if (request.boundary_payloads.len == 0 or request.boundary_payloads.len + 1 != stages.len) {
        return error.InvalidStepRequest;
    }

    const boundary_count = request.boundary_payloads.len;
    var inline_metadata: [8]tensor_frame.TensorFrameMetadata = undefined;
    var inline_images: [8]boundary_byte_image.BoundaryByteImageRef = undefined;
    var inline_payloads: [8]LocalPipelineBoundaryPayload = undefined;
    var inline_entries: [8]tensor_frame.TensorFrameBatchEntry = undefined;
    var heap_metadata: []tensor_frame.TensorFrameMetadata = &.{};
    var heap_images: []boundary_byte_image.BoundaryByteImageRef = &.{};
    var heap_payloads: []LocalPipelineBoundaryPayload = &.{};
    var heap_entries: []tensor_frame.TensorFrameBatchEntry = &.{};
    defer {
        if (heap_entries.len != 0) context.allocator.?.free(heap_entries);
        if (heap_payloads.len != 0) context.allocator.?.free(heap_payloads);
        if (heap_images.len != 0) context.allocator.?.free(heap_images);
        if (heap_metadata.len != 0) context.allocator.?.free(heap_metadata);
    }

    const metadata = if (boundary_count <= inline_metadata.len)
        inline_metadata[0..boundary_count]
    else blk: {
        const allocator = context.allocator orelse return error.InvalidStepRequest;
        heap_metadata = try allocator.alloc(tensor_frame.TensorFrameMetadata, boundary_count);
        break :blk heap_metadata;
    };
    const images = if (boundary_count <= inline_images.len)
        inline_images[0..boundary_count]
    else blk: {
        const allocator = context.allocator orelse return error.InvalidStepRequest;
        heap_images = try allocator.alloc(boundary_byte_image.BoundaryByteImageRef, boundary_count);
        break :blk heap_images;
    };
    const payloads = if (boundary_count <= inline_payloads.len)
        inline_payloads[0..boundary_count]
    else blk: {
        const allocator = context.allocator orelse return error.InvalidStepRequest;
        heap_payloads = try allocator.alloc(LocalPipelineBoundaryPayload, boundary_count);
        break :blk heap_payloads;
    };
    const entries = if (boundary_count <= inline_entries.len)
        inline_entries[0..boundary_count]
    else blk: {
        const allocator = context.allocator orelse return error.InvalidStepRequest;
        heap_entries = try allocator.alloc(tensor_frame.TensorFrameBatchEntry, boundary_count);
        break :blk heap_entries;
    };

    for (request.boundary_payloads, 0..) |spec, index| {
        metadata[index] = try local_stage_runner.buildPrefillActivationMetadata(.{
            .plan_ref = request.tensor_frame_plan_ref,
            .hidden_size = request.hidden_size,
            .boundary_index = spec.frame.boundary_index,
            .dtype = spec.frame.dtype,
            .layout = spec.frame.layout,
            .location_hint = spec.location_hint,
            .slot_request_ids = request.slot_request_ids,
            .slot_index = spec.slot_index,
            .sequence_start = spec.sequence_start,
            .token_count = spec.token_count,
            .batch_entries = entries[index .. index + 1],
        });
        try tensor_frame.validateTensorFrameForPlanBoundary(
            &metadata[index],
            request.tensor_frame_plan_ref,
            spec.frame.boundary_index,
        );
        try tensor_frame.validatePayloadBufferLength(&metadata[index], spec.activation_byte_count);
        images[index] = switch (spec.image) {
            .device => local_stage_runner.deviceActivationByteImage(&metadata[index]),
            .host_bytes => |host_bytes| blk: {
                if (spec.activation_byte_count > host_bytes.len) return error.InvalidArgument;
                break :blk local_stage_runner.hostActivationByteImage(
                    &metadata[index],
                    host_bytes[0..spec.activation_byte_count],
                );
            },
        };
        payloads[index] = .{
            .metadata = &metadata[index],
            .image = &images[index],
            .runtime = .{
                .staging = spec.frame.staging,
                .allow_borrow = false,
                .local_device_peer_copy_available = spec.local_device_peer_copy_available orelse
                    spec.frame.local_device_peer_copy_available,
            },
        };
    }

    try executeLocalPipelineStep(context, stages, payloads, .prefill, request.stage_inputs);
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
