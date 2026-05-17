//! Backend-neutral local stage runtime planning.
//!
//! This module derives ordered bridge/runtime facts from an explicit local
//! stage list and per-stage capabilities. Concrete backends still instantiate
//! their own executors; this planner owns the topology-independent contract
//! inputs and host staging allocation.

const std = @import("std");

const host_capability = @import("host_capability.zig");
const pipeline = @import("pipeline.zig");
const local_stage = @import("local_stage_contract.zig");

const bridge = struct {
    const BoundaryDType = pipeline.BoundaryDType;
    const HostBackendKind = host_capability.HostBackendKind;
    const negotiateBoundaryContract = pipeline.negotiateBoundaryContract;
};

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
    stages: []const StageInputSpec,
    stage_capabilities: []const StageRuntimeCapability,
    boundary_peer_copy_available: []const bool,
};

pub const StageInputSpec = struct {
    backend_kind: bridge.HostBackendKind,
    layer_start: usize,
    layer_end: usize,
};

fn hostLogitCandidateBetter(
    logit: f32,
    token_id: u32,
    existing_logit: f32,
    existing_token_id: u32,
) bool {
    return logit > existing_logit or (logit == existing_logit and token_id < existing_token_id);
}

fn insertHostTopKCandidate(
    logits: []f32,
    ids: []u32,
    count: *usize,
    max_count: usize,
    logit: f32,
    token_id: u32,
) void {
    if (max_count == 0) return;
    var insert_at = count.*;
    if (insert_at == max_count and !hostLogitCandidateBetter(logit, token_id, logits[max_count - 1], ids[max_count - 1])) {
        return;
    }
    if (insert_at < max_count) count.* += 1;
    if (insert_at >= max_count) insert_at = max_count - 1;
    while (insert_at > 0 and hostLogitCandidateBetter(logit, token_id, logits[insert_at - 1], ids[insert_at - 1])) {
        logits[insert_at] = logits[insert_at - 1];
        ids[insert_at] = ids[insert_at - 1];
        insert_at -= 1;
    }
    logits[insert_at] = logit;
    ids[insert_at] = token_id;
}

pub fn extractTopKFromHostLogitsRow(
    logits: []const f32,
    top_k: usize,
    candidate_logits_out: []f32,
    candidate_ids_out: []u32,
) !usize {
    if (top_k == 0 or top_k > 256) return error.InvalidArgument;
    if (candidate_logits_out.len < top_k or candidate_ids_out.len < top_k) return error.InvalidArgument;
    if (logits.len == 0 or logits.len > std.math.maxInt(u32)) return error.InvalidArgument;
    const count_limit = @min(top_k, logits.len);
    var count: usize = 0;
    for (logits, 0..) |logit, token_index| {
        insertHostTopKCandidate(
            candidate_logits_out,
            candidate_ids_out,
            &count,
            count_limit,
            logit,
            @intCast(token_index),
        );
    }
    if (top_k == 1 and count == 1) candidate_logits_out[0] = 0.0;
    return count;
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
        if (capability.backend_kind != stage.backend_kind) return error.InvalidTopologyConfig;
        plan.stage_backend_kinds[stage_id] = stage.backend_kind;
        plan.stage_specs[stage_id] = .{
            .stage_id = stage_id,
            .backend_kind = stage.backend_kind,
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

test "local_stage_runtime buildPlan builds generic mixed runtime facts" {
    const stages = [_]StageInputSpec{
        .{ .backend_kind = .cpu, .layer_start = 0, .layer_end = 1 },
        .{ .backend_kind = .cuda, .layer_start = 1, .layer_end = 3 },
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

test "local_stage_runtime buildPlan skips host staging for peer-copy boundary" {
    const stages = [_]StageInputSpec{
        .{ .backend_kind = .cuda, .layer_start = 0, .layer_end = 1 },
        .{ .backend_kind = .cuda, .layer_start = 1, .layer_end = 2 },
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

test "local_stage_runtime extractTopKFromHostLogitsRow returns deterministic candidates" {
    const logits = [_]f32{ 1.0, 8.0, 3.0, 8.0, -2.0, 5.0 };
    var candidate_logits: [3]f32 = undefined;
    var candidate_ids: [3]u32 = undefined;

    const count = try extractTopKFromHostLogitsRow(
        logits[0..],
        3,
        candidate_logits[0..],
        candidate_ids[0..],
    );

    try std.testing.expectEqual(@as(usize, 3), count);
    try std.testing.expectEqualSlices(u32, &.{ 1, 3, 5 }, candidate_ids[0..]);
    try std.testing.expectEqualSlices(f32, &.{ 8.0, 8.0, 5.0 }, candidate_logits[0..]);
}

fn testCapability(stage_id: usize, kind: bridge.HostBackendKind) StageRuntimeCapability {
    return .{
        .stage_id = stage_id,
        .backend_kind = kind,
        .max_batch_size = 4,
        .prefill_chunk_rows_cap = 8,
        .supported_boundary_dtypes = &.{.f32},
    };
}

test "local_stage_runtime buildPlan accepts ordered mixed chain shapes" {
    const Case = struct {
        stages: []const StageInputSpec,
        caps: []const StageRuntimeCapability,
        split_points: []const usize,
    };
    const chain_a_stages = [_]StageInputSpec{
        .{ .backend_kind = .cpu, .layer_start = 0, .layer_end = 1 },
        .{ .backend_kind = .cuda, .layer_start = 1, .layer_end = 4 },
    };
    const chain_a_caps = [_]StageRuntimeCapability{
        testCapability(0, .cpu),
        testCapability(1, .cuda),
    };
    const chain_b_stages = [_]StageInputSpec{
        .{ .backend_kind = .cuda, .layer_start = 0, .layer_end = 3 },
        .{ .backend_kind = .cpu, .layer_start = 3, .layer_end = 4 },
    };
    const chain_b_caps = [_]StageRuntimeCapability{
        testCapability(0, .cuda),
        testCapability(1, .cpu),
    };
    const chain_c_stages = [_]StageInputSpec{
        .{ .backend_kind = .cuda, .layer_start = 0, .layer_end = 1 },
        .{ .backend_kind = .cpu, .layer_start = 1, .layer_end = 3 },
        .{ .backend_kind = .cuda, .layer_start = 3, .layer_end = 5 },
    };
    const chain_c_caps = [_]StageRuntimeCapability{
        testCapability(0, .cuda),
        testCapability(1, .cpu),
        testCapability(2, .cuda),
    };
    const cases = [_]Case{
        .{ .stages = chain_a_stages[0..], .caps = chain_a_caps[0..], .split_points = &.{1} },
        .{ .stages = chain_b_stages[0..], .caps = chain_b_caps[0..], .split_points = &.{3} },
        .{ .stages = chain_c_stages[0..], .caps = chain_c_caps[0..], .split_points = &.{ 1, 3 } },
    };

    for (cases) |case| {
        const no_peer_copy = [_]bool{ false, false };
        var plan = try buildPlan(.{
            .allocator = std.testing.allocator,
            .d_model = 8,
            .total_layers = case.stages[case.stages.len - 1].layer_end,
            .stages = case.stages,
            .stage_capabilities = case.caps,
            .boundary_peer_copy_available = no_peer_copy[0 .. case.stages.len - 1],
        });
        defer plan.deinit();

        try std.testing.expectEqualSlices(usize, case.split_points, plan.split_points);
        try std.testing.expectEqual(case.stages.len, plan.stage_specs.len);
        try std.testing.expectEqual(case.stages.len - 1, plan.boundary_runtimes.len);
    }
}
