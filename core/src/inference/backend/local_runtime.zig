//! Backend-neutral local stage runtime planning.
//!
//! This module derives ordered bridge/runtime facts from an explicit local
//! stage list and per-stage capabilities. Concrete backends still instantiate
//! their own executors; this planner owns the topology-independent contract
//! inputs and host staging allocation.

const std = @import("std");

const bridge = @import("../bridge/root.zig");
const cpu_stage_capabilities = @import("cpu/stage_capabilities.zig");
const cuda_stage_capabilities = @import("cuda/stage_capabilities.zig");
const local_stage = @import("local_stage.zig");
const topology = @import("topology.zig");

pub const StageSpec = local_stage.StageSpec;
pub const BoundaryConfig = local_stage.BoundaryConfig;
pub const BoundaryRuntime = local_stage.BoundaryRuntime;

pub const StageRuntimeCapability = struct {
    stage_id: usize,
    backend_kind: bridge.HostBackendKind,
    max_batch_size: usize,
    prefill_chunk_rows_cap: usize,
    supported_boundary_dtypes: []const bridge.BoundaryDType,
};

pub const Plan = struct {
    allocator: std.mem.Allocator,
    split_points: []usize = &.{},
    stage_backend_kinds: []bridge.HostBackendKind = &.{},
    stage_specs: []StageSpec = &.{},
    boundary_configs: []BoundaryConfig = &.{},
    boundary_runtimes: []BoundaryRuntime = &.{},
    boundary_staging: []?[]align(4096) u8 = &.{},

    pub fn deinit(self: *@This()) void {
        for (self.boundary_staging) |staging| {
            if (staging) |buf| self.allocator.free(buf);
        }
        if (self.boundary_staging.len != 0) self.allocator.free(self.boundary_staging);
        if (self.boundary_runtimes.len != 0) self.allocator.free(self.boundary_runtimes);
        if (self.boundary_configs.len != 0) self.allocator.free(self.boundary_configs);
        if (self.stage_specs.len != 0) self.allocator.free(self.stage_specs);
        if (self.stage_backend_kinds.len != 0) self.allocator.free(self.stage_backend_kinds);
        if (self.split_points.len != 0) self.allocator.free(self.split_points);
        self.* = undefined;
    }
};

pub const BuildRequest = struct {
    allocator: std.mem.Allocator,
    d_model: usize,
    total_layers: usize,
    stages: []const topology.LocalStageSpec,
    stage_capabilities: []const StageRuntimeCapability,
    boundary_peer_copy_available: []const bool,
};

pub fn bridgeBackendKind(kind: topology.LocalStageBackendKind) bridge.HostBackendKind {
    return switch (kind) {
        .cpu => .cpu,
        .cuda => .cuda,
        .metal => .metal,
    };
}

pub fn supportedBoundaryDTypes(kind: topology.LocalStageBackendKind) []const bridge.BoundaryDType {
    return switch (kind) {
        .cpu => cpu_stage_capabilities.supported_boundary_dtypes[0..],
        .cuda => cuda_stage_capabilities.supported_boundary_dtypes[0..],
        .metal => &.{.f32},
    };
}

pub fn buildPlan(request: BuildRequest) !Plan {
    if (request.total_layers == 0 or request.stages.len <= 1) return error.InvalidTopologyConfig;
    const boundary_count = request.stages.len - 1;
    if (request.stage_capabilities.len != request.stages.len or
        request.boundary_peer_copy_available.len != boundary_count)
    {
        return error.InvalidTopologyConfig;
    }

    var plan = Plan{ .allocator = request.allocator };
    errdefer plan.deinit();

    plan.split_points = try request.allocator.alloc(usize, boundary_count);
    plan.stage_backend_kinds = try request.allocator.alloc(bridge.HostBackendKind, request.stages.len);
    plan.stage_specs = try request.allocator.alloc(StageSpec, request.stages.len);
    plan.boundary_configs = try request.allocator.alloc(BoundaryConfig, boundary_count);
    plan.boundary_runtimes = try request.allocator.alloc(BoundaryRuntime, boundary_count);
    plan.boundary_staging = try request.allocator.alloc(?[]align(4096) u8, boundary_count);
    @memset(plan.boundary_staging, null);

    for (request.stages, request.stage_capabilities, 0..) |stage, capability, stage_id| {
        if (capability.stage_id != stage_id) return error.InvalidTopologyConfig;
        const backend_kind = bridgeBackendKind(stage.backend_kind);
        if (capability.backend_kind != backend_kind) return error.InvalidTopologyConfig;
        plan.stage_backend_kinds[stage_id] = backend_kind;
        plan.stage_specs[stage_id] = .{
            .stage_id = stage_id,
            .backend_kind = backend_kind,
            .layer_start = stage.layer_start,
            .layer_end = stage.layer_end,
            .owns_embedding = stage_id == 0,
            .owns_projection = stage_id + 1 == request.stages.len,
        };
        if (stage_id < boundary_count) plan.split_points[stage_id] = stage.layer_end;
    }

    try local_stage.validateStageSpecs(request.total_layers, plan.stage_specs);

    for (0..boundary_count) |boundary_index| {
        const source_capability = request.stage_capabilities[boundary_index];
        const target_capability = request.stage_capabilities[boundary_index + 1];
        const boundary = try bridge.negotiateBoundaryContract(.{
            .stage0_native_dtype = .f32,
            .stage1_native_dtype = .f32,
            .stage0_supported_boundary_dtypes = source_capability.supported_boundary_dtypes,
            .stage1_supported_boundary_dtypes = target_capability.supported_boundary_dtypes,
        });
        if (boundary.stage0_requires_conversion or boundary.stage1_requires_conversion) {
            return error.InvalidTopologyConfig;
        }

        const decode_max_batch_entries = @min(source_capability.max_batch_size, target_capability.max_batch_size);
        const prefill_max_token_count_per_frame = @min(
            source_capability.prefill_chunk_rows_cap,
            target_capability.prefill_chunk_rows_cap,
        );
        plan.boundary_configs[boundary_index] = local_stage.boundaryConfig(
            boundary.boundary_dtype,
            boundary.layout,
            decode_max_batch_entries,
            prefill_max_token_count_per_frame,
        );

        if (!request.boundary_peer_copy_available[boundary_index]) {
            const row_bytes = try local_stage.boundaryRowByteCount(request.d_model, boundary.boundary_dtype);
            const rows = @max(@as(usize, 1), prefill_max_token_count_per_frame);
            const transfer_bytes = std.math.mul(usize, std.math.cast(usize, row_bytes) orelse return error.InvalidArgument, rows) catch return error.InvalidArgument;
            plan.boundary_staging[boundary_index] = try request.allocator.alignedAlloc(u8, .fromByteUnits(4096), transfer_bytes);
        }

        plan.boundary_runtimes[boundary_index] = .{
            .boundary_index = boundary_index,
            .dtype = boundary.boundary_dtype,
            .layout = boundary.layout,
            .staging = plan.boundary_staging[boundary_index],
            .local_device_peer_copy_available = request.boundary_peer_copy_available[boundary_index],
        };
    }

    return plan;
}

test "inference backend local_runtime buildPlan builds generic CPU CUDA CPU runtime facts" {
    const stages = [_]topology.LocalStageSpec{
        .{ .backend_kind = .cpu, .layer_start = 0, .layer_end = 1 },
        .{ .backend_kind = .cuda, .device_ordinal = 0, .layer_start = 1, .layer_end = 3 },
        .{ .backend_kind = .cpu, .layer_start = 3, .layer_end = 4 },
    };
    const caps = [_]StageRuntimeCapability{
        .{ .stage_id = 0, .backend_kind = .cpu, .max_batch_size = 2, .prefill_chunk_rows_cap = 5, .supported_boundary_dtypes = &.{.f32} },
        .{ .stage_id = 1, .backend_kind = .cuda, .max_batch_size = 3, .prefill_chunk_rows_cap = 4, .supported_boundary_dtypes = &.{.f32} },
        .{ .stage_id = 2, .backend_kind = .cpu, .max_batch_size = 1, .prefill_chunk_rows_cap = 6, .supported_boundary_dtypes = &.{.f32} },
    };
    var plan = try buildPlan(.{
        .allocator = std.testing.allocator,
        .d_model = 8,
        .total_layers = 4,
        .stages = &stages,
        .stage_capabilities = &caps,
        .boundary_peer_copy_available = &.{ false, false },
    });
    defer plan.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 1, 3 }, plan.split_points);
    try std.testing.expectEqual(bridge.HostBackendKind.cpu, plan.stage_specs[0].backend_kind);
    try std.testing.expectEqual(bridge.HostBackendKind.cuda, plan.stage_specs[1].backend_kind);
    try std.testing.expectEqual(bridge.HostBackendKind.cpu, plan.stage_specs[2].backend_kind);
    try std.testing.expectEqual(@as(usize, 2), plan.boundary_configs[0].decode_max_batch_entries);
    try std.testing.expectEqual(@as(usize, 1), plan.boundary_configs[1].decode_max_batch_entries);
    try std.testing.expect(plan.boundary_runtimes[0].staging != null);
    try std.testing.expect(plan.boundary_runtimes[1].staging != null);
}

test "inference backend local_runtime buildPlan skips host staging for peer-copy boundary" {
    const stages = [_]topology.LocalStageSpec{
        .{ .backend_kind = .cuda, .device_ordinal = 0, .layer_start = 0, .layer_end = 1 },
        .{ .backend_kind = .cuda, .device_ordinal = 1, .layer_start = 1, .layer_end = 2 },
    };
    const caps = [_]StageRuntimeCapability{
        .{ .stage_id = 0, .backend_kind = .cuda, .max_batch_size = 2, .prefill_chunk_rows_cap = 5, .supported_boundary_dtypes = &.{.f32} },
        .{ .stage_id = 1, .backend_kind = .cuda, .max_batch_size = 2, .prefill_chunk_rows_cap = 5, .supported_boundary_dtypes = &.{.f32} },
    };
    var plan = try buildPlan(.{
        .allocator = std.testing.allocator,
        .d_model = 8,
        .total_layers = 2,
        .stages = &stages,
        .stage_capabilities = &caps,
        .boundary_peer_copy_available = &.{true},
    });
    defer plan.deinit();

    try std.testing.expectEqual(@as(usize, 1), plan.boundary_runtimes.len);
    try std.testing.expect(plan.boundary_runtimes[0].local_device_peer_copy_available);
    try std.testing.expect(plan.boundary_runtimes[0].staging == null);
}
