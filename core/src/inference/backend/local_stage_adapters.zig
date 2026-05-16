//! Shared adapter helpers for local staged decode and prefill routes.
//!
//! This module owns common stage-method validation and stage metadata helpers
//! used by local staged-route adapters.

const std = @import("std");
const bridge = @import("../bridge/root.zig");
const transport = @import("../transport/root.zig");

pub const max_decode_transport_rows: usize = 128;

pub const DecodeContext = struct {
    token: u32,
    position: usize,
    slot_index: usize,
    logits_out_opt: ?[]f32,
    compute_logits: bool,
    download_logits: bool,
    ensure_kv_capacity: bool,
    trace_seq_len_u32: u32,
    trace_pos_offset: usize,
};

pub const DecodeBatchEntryScratch = struct {
    allocator: ?std.mem.Allocator = null,
    heap_entries: []bridge.TensorFrameBatchEntry = &.{},
    inline_entries: [max_decode_transport_rows]bridge.TensorFrameBatchEntry = undefined,

    pub fn init(allocator: ?std.mem.Allocator, len: usize) !DecodeBatchEntryScratch {
        if (len <= max_decode_transport_rows) return .{};
        const alloc = allocator orelse return error.InvalidTopologyConfig;
        return .{
            .allocator = alloc,
            .heap_entries = try alloc.alloc(bridge.TensorFrameBatchEntry, len),
        };
    }

    pub fn deinit(self: *DecodeBatchEntryScratch) void {
        if (self.heap_entries.len != 0) {
            self.allocator.?.free(self.heap_entries);
        }
        self.* = undefined;
    }

    pub fn slice(self: *DecodeBatchEntryScratch, len: usize) []bridge.TensorFrameBatchEntry {
        if (self.heap_entries.len != 0) return self.heap_entries[0..len];
        return self.inline_entries[0..len];
    }
};

pub const HostSegmentScratch = struct {
    allocator: ?std.mem.Allocator = null,
    heap_segments: [][]const u8 = &.{},
    inline_segments: [max_decode_transport_rows][]const u8 = undefined,

    pub fn init(allocator: ?std.mem.Allocator, len: usize) !HostSegmentScratch {
        if (len <= max_decode_transport_rows) return .{};
        const alloc = allocator orelse return error.InvalidTopologyConfig;
        return .{
            .allocator = alloc,
            .heap_segments = try alloc.alloc([]const u8, len),
        };
    }

    pub fn deinit(self: *HostSegmentScratch) void {
        if (self.heap_segments.len != 0) {
            self.allocator.?.free(self.heap_segments);
        }
        self.* = undefined;
    }

    pub fn slice(self: *HostSegmentScratch, len: usize) [][]const u8 {
        if (self.heap_segments.len != 0) return self.heap_segments[0..len];
        return self.inline_segments[0..len];
    }
};

pub fn backendAllocator(backend: anytype) ?std.mem.Allocator {
    const BackendType = @TypeOf(backend.*);
    if (comptime !@hasField(BackendType, "allocator")) return null;
    return backend.allocator;
}

pub fn localStageTensorFramePlanRef(backend: anytype) !*const bridge.TensorFramePlanRef {
    const BackendType = @TypeOf(backend.*);
    if (comptime !@hasField(BackendType, "local_tensor_frame_plan_ref")) return error.InvalidTopologyConfig;
    if (backend.local_tensor_frame_plan_ref) |*plan_ref| return plan_ref;
    return error.InvalidTopologyConfig;
}

pub fn localStagePlacementPlan(backend: anytype) !*const bridge.PlacementPlan {
    const BackendType = @TypeOf(backend.*);
    if (comptime !@hasField(BackendType, "local_placement_plan")) return error.InvalidTopologyConfig;
    if (backend.local_placement_plan) |*placement_plan| return placement_plan;
    return error.InvalidTopologyConfig;
}

pub fn localStageStateOwnershipPlan(backend: anytype) ?*const bridge.StageStateOwnershipPlan {
    const BackendType = @TypeOf(backend.*);
    if (comptime !@hasField(BackendType, "local_state_ownership_plan")) return null;
    if (backend.local_state_ownership_plan) |*state_plan| return state_plan;
    return null;
}

pub fn localStageRunnerPlanRef(backend: anytype) !*const bridge.LocalStageRunnerPlanRef {
    const BackendType = @TypeOf(backend.*);
    if (comptime !@hasField(BackendType, "local_stage_runner_plan_ref")) return error.InvalidTopologyConfig;
    if (backend.local_stage_runner_plan_ref) |*plan_ref| return plan_ref;
    return error.InvalidTopologyConfig;
}

pub fn localPipelineContext(backend: anytype) !bridge.LocalPipelineContext {
    return .{
        .allocator = backendAllocator(backend),
        .plan_ref = try localStageRunnerPlanRef(backend),
        .placement_plan = try localStagePlacementPlan(backend),
        .state_ownership_plan = localStageStateOwnershipPlan(backend),
    };
}

pub fn localPipelineFactsAvailable(backend: anytype) !bool {
    const BackendType = @TypeOf(backend.*);
    if (comptime !@hasField(BackendType, "local_stage_specs")) return false;
    if (backend.local_stage_specs.len == 0) return false;

    const allocator = backendAllocator(backend);
    var inline_bindings: [8]bridge.LocalPipelineStageBinding = undefined;
    var heap_bindings: []bridge.LocalPipelineStageBinding = &.{};
    defer if (heap_bindings.len != 0) allocator.?.free(heap_bindings);

    const bindings = if (backend.local_stage_specs.len <= inline_bindings.len)
        inline_bindings[0..backend.local_stage_specs.len]
    else blk: {
        const alloc = allocator orelse return error.InvalidTopologyConfig;
        heap_bindings = try alloc.alloc(bridge.LocalPipelineStageBinding, backend.local_stage_specs.len);
        break :blk heap_bindings;
    };

    for (backend.local_stage_specs, bindings) |spec, *binding| {
        binding.* = .{
            .stage_id = spec.stage_id,
            .backend_kind = spec.backend_kind,
        };
    }
    try bridge.validateLocalPipelineStageBindings(try localPipelineContext(backend), bindings);
    return true;
}

pub const LocalBoundaryRuntimeView = struct {
    boundary_index: usize,
    dtype: bridge.BoundaryDType,
    layout: bridge.BoundaryLayout,
    staging: ?[]align(64) u8 = null,
    local_device_peer_copy_available: bool = false,
    peer_copy_synchronization: transport.CudaPeerCopySynchronization = .source_stream,
};

pub fn localBoundaryRuntime(backend: anytype, boundary_index: usize) !LocalBoundaryRuntimeView {
    const BackendType = @TypeOf(backend.*);
    if (comptime @hasDecl(BackendType, "localBoundaryRuntime")) {
        const boundary = try backend.localBoundaryRuntime(boundary_index);
        return .{
            .boundary_index = boundary.boundary_index,
            .dtype = boundary.dtype,
            .layout = boundary.layout,
            .staging = boundary.staging,
            .local_device_peer_copy_available = boundary.local_device_peer_copy_available,
            .peer_copy_synchronization = if (comptime @hasField(@TypeOf(boundary.*), "peer_copy_synchronization"))
                boundary.peer_copy_synchronization
            else
                .source_stream,
        };
    }
    if (comptime @hasField(BackendType, "local_boundary_runtimes")) {
        if (boundary_index >= backend.local_boundary_runtimes.len) return error.InvalidTopologyConfig;
        const boundary = backend.local_boundary_runtimes[boundary_index];
        if (boundary.boundary_index != boundary_index) return error.InvalidTopologyConfig;
        return .{
            .boundary_index = boundary.boundary_index,
            .dtype = boundary.dtype,
            .layout = boundary.layout,
            .staging = boundary.staging,
            .local_device_peer_copy_available = boundary.local_device_peer_copy_available,
            .peer_copy_synchronization = if (comptime @hasField(@TypeOf(boundary), "peer_copy_synchronization"))
                boundary.peer_copy_synchronization
            else
                .source_stream,
        };
    }
    return error.InvalidTopologyConfig;
}

pub fn localBoundaryFrameSpec(backend: anytype, boundary_index: usize) !bridge.LocalPipelineBoundaryFrameSpec {
    const boundary = try localBoundaryRuntime(backend, boundary_index);
    return .{
        .boundary_index = boundary.boundary_index,
        .dtype = boundary.dtype,
        .layout = boundary.layout,
        .staging = boundary.staging,
        .local_device_peer_copy_available = boundary.local_device_peer_copy_available,
    };
}

pub fn localBoundaryActivationByteCount(backend: anytype, boundary_index: usize) !usize {
    const boundary = try localBoundaryRuntime(backend, boundary_index);
    const element_bytes: usize = switch (boundary.dtype) {
        .bf16, .f16 => @sizeOf(u16),
        .f32 => @sizeOf(f32),
    };
    return std.math.mul(usize, backend.d_model, element_bytes) catch return error.InvalidArgument;
}

pub fn localLayerOffset(backend: anytype) usize {
    const BackendType = @TypeOf(backend.*);
    if (comptime @hasField(BackendType, "local_layer_start")) {
        return backend.local_layer_start;
    }
    return 0;
}

pub fn cudaPayloadLocationHint(backend: anytype) !bridge.TensorFramePayloadLocationHint {
    const BackendType = @TypeOf(backend.*);
    if (comptime !@hasField(BackendType, "device")) return error.InvalidTopologyConfig;
    const ordinal = backend.device.ordinal();
    return .{ .cuda = std.math.cast(u16, ordinal) orelse return error.InvalidTopologyConfig };
}

pub fn buildDecodeActivationMetadata(
    backend: anytype,
    boundary_index: usize,
    boundary_dtype: bridge.BoundaryDType,
    boundary_layout: bridge.BoundaryLayout,
    location_hint: ?bridge.TensorFramePayloadLocationHint,
    slot_indices: []const usize,
    positions: []const usize,
    batch_entries: []bridge.TensorFrameBatchEntry,
) !bridge.TensorFrameMetadata {
    const BackendType = @TypeOf(backend.*);
    if (comptime !@hasField(BackendType, "slot_request_ids")) return error.InvalidRequestId;
    return bridge.buildDecodeActivationMetadata(.{
        .plan_ref = try localStageTensorFramePlanRef(backend),
        .hidden_size = backend.d_model,
        .boundary_index = boundary_index,
        .dtype = boundary_dtype,
        .layout = boundary_layout,
        .location_hint = location_hint,
        .slot_request_ids = backend.slot_request_ids[0..],
        .slot_indices = slot_indices,
        .positions = positions,
        .batch_entries = batch_entries,
    });
}

pub fn buildPrefillActivationMetadata(
    backend: anytype,
    boundary_index: usize,
    boundary_dtype: bridge.BoundaryDType,
    boundary_layout: bridge.BoundaryLayout,
    location_hint: ?bridge.TensorFramePayloadLocationHint,
    slot_index: usize,
    sequence_start: usize,
    token_count: usize,
    batch_entries: []bridge.TensorFrameBatchEntry,
) !bridge.TensorFrameMetadata {
    const BackendType = @TypeOf(backend.*);
    if (comptime !@hasField(BackendType, "slot_request_ids")) return error.InvalidRequestId;
    return bridge.buildPrefillActivationMetadata(.{
        .plan_ref = try localStageTensorFramePlanRef(backend),
        .hidden_size = backend.d_model,
        .boundary_index = boundary_index,
        .dtype = boundary_dtype,
        .layout = boundary_layout,
        .location_hint = location_hint,
        .slot_request_ids = backend.slot_request_ids[0..],
        .slot_index = slot_index,
        .sequence_start = sequence_start,
        .token_count = token_count,
        .batch_entries = batch_entries,
    });
}

pub fn validateEmptyInput(input: []const u8) !void {
    if (input.len != 0) return error.InvalidArgument;
}

pub fn decodeLayerLimit(layer_start: usize, layer_end: usize) !usize {
    if (layer_end < layer_start) return error.InvalidArgument;
    return layer_end - layer_start;
}

pub fn executeCpuDecodeLayerRange(
    backend: anytype,
    ctx: *const DecodeContext,
    layer_start: usize,
    layer_end: usize,
    logits_out_opt: ?[]f32,
    compute_logits: bool,
    download_logits: bool,
    use_preloaded_input: bool,
) !void {
    if (comptime !hasDecl(@TypeOf(backend.*), "executeDecodeLayerRange")) {
        return error.InvalidTopologyConfig;
    }
    try backend.executeDecodeLayerRange(
        ctx.token,
        ctx.position,
        ctx.slot_index,
        logits_out_opt,
        layer_start,
        layer_end,
        compute_logits,
        download_logits,
        ctx.ensure_kv_capacity,
        use_preloaded_input,
    );
}

pub fn executeCudaDecodeLayerRange(
    comptime execute_decode_with_layer_limit: anytype,
    backend: anytype,
    ctx: *const DecodeContext,
    layer_start: usize,
    layer_end: usize,
    logits_out_opt: ?[]f32,
    compute_logits: bool,
    download_logits: bool,
    hidden_override: ?[]const f32,
    deepstack_layer_features_opt: ?[]const []const f32,
    deepstack_feature_index_opt: ?usize,
    use_preloaded_input: bool,
) !void {
    const local_layer_limit = try decodeLayerLimit(layer_start, layer_end);
    try execute_decode_with_layer_limit(
        backend,
        ctx.token,
        ctx.position,
        ctx.slot_index,
        logits_out_opt,
        local_layer_limit,
        compute_logits,
        download_logits,
        ctx.ensure_kv_capacity,
        ctx.trace_seq_len_u32,
        ctx.trace_pos_offset,
        hidden_override,
        deepstack_layer_features_opt,
        deepstack_feature_index_opt,
        use_preloaded_input,
    );
}

pub fn executeCpuPrefillLayerRange(
    backend: anytype,
    slot_index: usize,
    tokens: []const u32,
    sequence_start: usize,
    layer_start: usize,
    layer_end: usize,
    use_preloaded_input: bool,
    compute_logits: bool,
    logits_out_opt: ?[]f32,
    source_embeddings_out: ?[]f32,
) !void {
    if (comptime !hasDecl(@TypeOf(backend.*), "executePrefillLayerRange")) {
        return error.InvalidTopologyConfig;
    }
    try backend.executePrefillLayerRange(
        slot_index,
        tokens,
        sequence_start,
        layer_start,
        layer_end,
        use_preloaded_input,
        compute_logits,
        logits_out_opt,
        source_embeddings_out,
    );
}

fn hasDecl(comptime T: type, comptime name: []const u8) bool {
    return switch (@typeInfo(T)) {
        .@"struct", .@"enum", .@"union", .@"opaque" => @hasDecl(T, name),
        else => false,
    };
}

test "decodeLayerLimit rejects inverted ranges" {
    try std.testing.expectError(error.InvalidArgument, decodeLayerLimit(4, 3));
}

test "validateEmptyInput rejects route payloads" {
    try std.testing.expectError(error.InvalidArgument, validateEmptyInput(&.{1}));
}

test "executeCpuDecodeLayerRange forwards final logits controls" {
    const MockCpuBackend = struct {
        saw_logits: bool = false,
        compute_logits: bool = false,
        download_logits: bool = false,
        use_preloaded_input: bool = false,
        layer_start: usize = 0,
        layer_end: usize = 0,

        pub fn executeDecodeLayerRange(
            self: *@This(),
            token: u32,
            position: usize,
            slot_index: usize,
            logits_out_opt: ?[]f32,
            layer_start: usize,
            layer_end: usize,
            compute_logits: bool,
            download_logits: bool,
            ensure_kv_capacity: bool,
            use_preloaded_input: bool,
        ) !void {
            try std.testing.expectEqual(@as(u32, 17), token);
            try std.testing.expectEqual(@as(usize, 23), position);
            try std.testing.expectEqual(@as(usize, 3), slot_index);
            try std.testing.expect(ensure_kv_capacity);
            self.saw_logits = logits_out_opt != null;
            self.compute_logits = compute_logits;
            self.download_logits = download_logits;
            self.use_preloaded_input = use_preloaded_input;
            self.layer_start = layer_start;
            self.layer_end = layer_end;
        }
    };

    var backend = MockCpuBackend{};
    var logits = [_]f32{0} ** 4;
    const ctx = DecodeContext{
        .token = 17,
        .position = 23,
        .slot_index = 3,
        .logits_out_opt = logits[0..],
        .compute_logits = true,
        .download_logits = true,
        .ensure_kv_capacity = true,
        .trace_seq_len_u32 = 24,
        .trace_pos_offset = 0,
    };

    try executeCpuDecodeLayerRange(&backend, &ctx, 4, 7, logits[0..], true, true, true);

    try std.testing.expect(backend.saw_logits);
    try std.testing.expect(backend.compute_logits);
    try std.testing.expect(backend.download_logits);
    try std.testing.expect(backend.use_preloaded_input);
    try std.testing.expectEqual(@as(usize, 4), backend.layer_start);
    try std.testing.expectEqual(@as(usize, 7), backend.layer_end);
}

test "inference.backend local_stage_adapters executeCpuPrefillLayerRange forwards generic prefill controls" {
    const MockCpuBackend = struct {
        slot_seen: usize = 0,
        sequence_start_seen: usize = 0,
        layer_start_seen: usize = 0,
        layer_end_seen: usize = 0,
        use_preloaded_seen: bool = false,
        compute_logits_seen: bool = false,
        logits_len_seen: usize = 0,
        source_embeddings_len_seen: usize = 0,

        pub fn executePrefillLayerRange(
            self: *@This(),
            slot_index: usize,
            tokens: []const u32,
            sequence_start: usize,
            layer_start: usize,
            layer_end: usize,
            use_preloaded_input: bool,
            compute_logits: bool,
            logits_out_opt: ?[]f32,
            source_embeddings_out: ?[]f32,
        ) !void {
            try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3 }, tokens);
            self.slot_seen = slot_index;
            self.sequence_start_seen = sequence_start;
            self.layer_start_seen = layer_start;
            self.layer_end_seen = layer_end;
            self.use_preloaded_seen = use_preloaded_input;
            self.compute_logits_seen = compute_logits;
            self.logits_len_seen = if (logits_out_opt) |logits| logits.len else 0;
            self.source_embeddings_len_seen = if (source_embeddings_out) |embeddings| embeddings.len else 0;
        }
    };

    var backend = MockCpuBackend{};
    var logits = [_]f32{0} ** 8;
    var embeddings = [_]f32{0} ** 12;
    try executeCpuPrefillLayerRange(
        &backend,
        4,
        &.{ 1, 2, 3 },
        9,
        2,
        5,
        true,
        true,
        logits[0..],
        embeddings[0..],
    );

    try std.testing.expectEqual(@as(usize, 4), backend.slot_seen);
    try std.testing.expectEqual(@as(usize, 9), backend.sequence_start_seen);
    try std.testing.expectEqual(@as(usize, 2), backend.layer_start_seen);
    try std.testing.expectEqual(@as(usize, 5), backend.layer_end_seen);
    try std.testing.expect(backend.use_preloaded_seen);
    try std.testing.expect(backend.compute_logits_seen);
    try std.testing.expectEqual(@as(usize, 8), backend.logits_len_seen);
    try std.testing.expectEqual(@as(usize, 12), backend.source_embeddings_len_seen);
}

test "localStageStateOwnershipPlan returns optional local stage ownership plan field" {
    const WithoutPlan = struct {
        allocator: std.mem.Allocator = std.testing.allocator,
    };
    const WithPlan = struct {
        local_state_ownership_plan: ?bridge.StageStateOwnershipPlan = null,
    };
    var without_plan = WithoutPlan{};
    var with_plan = WithPlan{};

    try std.testing.expect(localStageStateOwnershipPlan(&without_plan) == null);
    try std.testing.expect(localStageStateOwnershipPlan(&with_plan) == null);
}

test "localPipelineContext rejects missing bridge contracts" {
    const MockBackend = struct {
        allocator: std.mem.Allocator = std.testing.allocator,
        local_stage_runner_plan_ref: ?bridge.LocalStageRunnerPlanRef = null,
        local_placement_plan: ?bridge.PlacementPlan = null,
        local_state_ownership_plan: ?bridge.StageStateOwnershipPlan = null,
    };
    var backend = MockBackend{};

    try std.testing.expectError(error.InvalidTopologyConfig, localPipelineContext(&backend));
}

test "localPipelineFactsAvailable returns false without local stage specs and rejects incomplete contracts" {
    const WithoutSpecs = struct {
        allocator: std.mem.Allocator = std.testing.allocator,
    };
    const Spec = struct {
        stage_id: usize,
        backend_kind: bridge.HostBackendKind,
    };
    const WithSpecs = struct {
        allocator: std.mem.Allocator = std.testing.allocator,
        local_stage_specs: []const Spec,
        local_stage_runner_plan_ref: ?bridge.LocalStageRunnerPlanRef = null,
        local_placement_plan: ?bridge.PlacementPlan = null,
        local_state_ownership_plan: ?bridge.StageStateOwnershipPlan = null,
    };
    var without_specs = WithoutSpecs{};
    const specs = [_]Spec{
        .{ .stage_id = 0, .backend_kind = .cpu },
        .{ .stage_id = 1, .backend_kind = .cuda },
    };
    var with_specs = WithSpecs{ .local_stage_specs = &specs };

    try std.testing.expect(!try localPipelineFactsAvailable(&without_specs));
    try std.testing.expectError(error.InvalidTopologyConfig, localPipelineFactsAvailable(&with_specs));
}

test "localBoundaryRuntime reads generic boundary runtime by index" {
    const Runtime = struct {
        boundary_index: usize,
        dtype: bridge.BoundaryDType,
        layout: bridge.BoundaryLayout,
        staging: ?[]align(64) u8 = null,
        local_device_peer_copy_available: bool = false,
    };
    const MockBackend = struct {
        runtimes: [2]Runtime,

        pub fn localBoundaryRuntime(self: *@This(), boundary_index: usize) !*const Runtime {
            if (boundary_index >= self.runtimes.len) return error.InvalidTopologyConfig;
            return &self.runtimes[boundary_index];
        }
    };
    var staging: [16]u8 align(64) = [_]u8{0} ** 16;
    var backend = MockBackend{ .runtimes = .{
        .{ .boundary_index = 0, .dtype = .f32, .layout = .row_major },
        .{
            .boundary_index = 1,
            .dtype = .f16,
            .layout = .row_major,
            .staging = staging[0..],
            .local_device_peer_copy_available = true,
        },
    } };

    const boundary = try localBoundaryRuntime(&backend, 1);
    try std.testing.expectEqual(@as(usize, 1), boundary.boundary_index);
    try std.testing.expectEqual(bridge.BoundaryDType.f16, boundary.dtype);
    try std.testing.expectEqual(bridge.BoundaryLayout.row_major, boundary.layout);
    try std.testing.expect(boundary.staging != null);
    try std.testing.expect(boundary.local_device_peer_copy_available);
    try std.testing.expectError(error.InvalidTopologyConfig, localBoundaryRuntime(&backend, 2));
}

test "executeLocalDecodePipelineStep rejects invalid stage boundary shape" {
    const MockBackend = struct {};
    _ = MockBackend{};
    var stages: [2]bridge.LocalStageChainStage = undefined;
    const slots = [_]usize{0};
    const positions = [_]usize{0};
    var plan_ref: bridge.TensorFramePlanRef = undefined;
    var runner_ref: bridge.LocalStageRunnerPlanRef = undefined;
    var placement_plan: bridge.PlacementPlan = undefined;

    try std.testing.expectError(
        error.InvalidStepRequest,
        bridge.executeLocalDecodePipelineStep(.{
            .plan_ref = &runner_ref,
            .placement_plan = &placement_plan,
        }, stages[0..], .{
            .tensor_frame_plan_ref = &plan_ref,
            .hidden_size = 1,
            .slot_request_ids = &.{null},
            .slot_indices = &slots,
            .positions = &positions,
            .boundary_payloads = &.{},
        }),
    );
}

const PrefillBoundaryTestRuntime = struct {
    boundary_index: usize,
    dtype: bridge.BoundaryDType,
    layout: bridge.BoundaryLayout,
    staging: ?[]align(64) u8 = null,
    local_device_peer_copy_available: bool = false,
};

const PrefillBoundaryTestBackend = struct {
    d_model: usize = 4,
    local_tensor_frame_plan_ref: ?bridge.TensorFramePlanRef = null,
    slot_request_ids: [2]?u64 = .{ 101, 202 },
    runtime: PrefillBoundaryTestRuntime = .{
        .boundary_index = 0,
        .dtype = .f32,
        .layout = .row_major,
    },

    pub fn localBoundaryRuntime(self: *@This(), boundary_index: usize) !*const PrefillBoundaryTestRuntime {
        if (boundary_index != self.runtime.boundary_index) return error.InvalidTopologyConfig;
        return &self.runtime;
    }
};

fn prefillBoundaryTestPlanRef(boundaries: []const bridge.TensorFrameBoundaryRef) bridge.TensorFramePlanRef {
    return .{
        .allocator = std.testing.allocator,
        .identity = .{
            .graph_digest = [_]u8{1} ** 32,
            .graph_contract_version = 1,
            .stage_plan_contract_version = 1,
            .stage_plan_id = .{ .digest = [_]u8{2} ** 32 },
        },
        .boundaries = boundaries,
    };
}

test "executeLocalPrefillPipelineStep rejects invalid stage boundary shape" {
    const MockBackend = struct {};
    _ = MockBackend{};
    var stages: [2]bridge.LocalStageChainStage = undefined;
    var plan_ref: bridge.TensorFramePlanRef = undefined;
    var runner_ref: bridge.LocalStageRunnerPlanRef = undefined;
    var placement_plan: bridge.PlacementPlan = undefined;

    try std.testing.expectError(
        error.InvalidStepRequest,
        bridge.executeLocalPrefillPipelineStep(.{
            .plan_ref = &runner_ref,
            .placement_plan = &placement_plan,
        }, stages[0..], .{
            .tensor_frame_plan_ref = &plan_ref,
            .hidden_size = 1,
            .slot_request_ids = &.{null},
            .boundary_payloads = &.{},
        }),
    );
}

test "executeLocalPrefillPipelineStep builds host and device prefill payload specs" {
    const boundaries = [_]bridge.TensorFrameBoundaryRef{.{
        .boundary_index = 0,
        .source_stage_id = 0,
        .target_stage_id = 1,
        .producer_layer_start = 0,
        .producer_layer_end = 2,
        .consumer_layer_start = 2,
        .consumer_layer_end = 4,
    }};
    var backend = PrefillBoundaryTestBackend{
        .local_tensor_frame_plan_ref = prefillBoundaryTestPlanRef(&boundaries),
    };
    var stages: [2]bridge.LocalStageChainStage = undefined;
    const activation_byte_count = 2 * 4 * @sizeOf(f32);
    var short_host_storage = [_]u8{0x5a} ** (activation_byte_count - 1);
    var runner_ref: bridge.LocalStageRunnerPlanRef = undefined;
    runner_ref.version = bridge.local_stage_runner_contract_version + 1;
    var placement_plan: bridge.PlacementPlan = undefined;

    try std.testing.expectError(
        error.InvalidArgument,
        bridge.executeLocalPrefillPipelineStep(.{
            .plan_ref = &runner_ref,
            .placement_plan = &placement_plan,
        }, stages[0..], .{
            .tensor_frame_plan_ref = &backend.local_tensor_frame_plan_ref.?,
            .hidden_size = backend.d_model,
            .slot_request_ids = backend.slot_request_ids[0..],
            .boundary_payloads = &.{.{
                .frame = try localBoundaryFrameSpec(&backend, 0),
                .slot_index = 0,
                .sequence_start = 3,
                .token_count = 2,
                .activation_byte_count = activation_byte_count,
                .location_hint = .{ .cpu = {} },
                .image = .{ .host_bytes = short_host_storage[0..] },
            }},
        }),
    );
    try std.testing.expectError(
        error.InvalidLocalStageRunnerContractVersion,
        bridge.executeLocalPrefillPipelineStep(.{
            .plan_ref = &runner_ref,
            .placement_plan = &placement_plan,
        }, stages[0..], .{
            .tensor_frame_plan_ref = &backend.local_tensor_frame_plan_ref.?,
            .hidden_size = backend.d_model,
            .slot_request_ids = backend.slot_request_ids[0..],
            .boundary_payloads = &.{.{
                .frame = try localBoundaryFrameSpec(&backend, 0),
                .slot_index = 0,
                .sequence_start = 3,
                .token_count = 2,
                .activation_byte_count = activation_byte_count,
                .location_hint = .{ .cuda = 0 },
                .image = .device,
            }},
        }),
    );
}

test "buildDecodeActivationMetadata creates multi-entry decode frame" {
    const MockBackend = struct {
        d_model: usize = 4,
        local_tensor_frame_plan_ref: ?bridge.TensorFramePlanRef = null,
        slot_request_ids: [3]?u64 = .{ 101, 202, 303 },
    };
    const boundaries = [_]bridge.TensorFrameBoundaryRef{.{
        .boundary_index = 0,
        .source_stage_id = 0,
        .target_stage_id = 1,
        .producer_layer_start = 0,
        .producer_layer_end = 2,
        .consumer_layer_start = 2,
        .consumer_layer_end = 4,
    }};
    const plan_ref = bridge.TensorFramePlanRef{
        .allocator = std.testing.allocator,
        .identity = .{
            .graph_digest = [_]u8{1} ** 32,
            .graph_contract_version = 1,
            .stage_plan_contract_version = 1,
            .stage_plan_id = .{ .digest = [_]u8{2} ** 32 },
        },
        .boundaries = &boundaries,
    };
    var backend = MockBackend{ .local_tensor_frame_plan_ref = plan_ref };
    var entries: [max_decode_transport_rows]bridge.TensorFrameBatchEntry = undefined;
    const slots = [_]usize{ 0, 2 };
    const positions = [_]usize{ 7, 9 };
    const metadata = try buildDecodeActivationMetadata(
        &backend,
        0,
        .f32,
        .row_major,
        .{ .cpu = {} },
        &slots,
        &positions,
        entries[0..],
    );

    try std.testing.expectEqual(bridge.TensorFrameStepKind.decode, metadata.step_kind);
    try std.testing.expectEqual(@as(usize, 2), metadata.batch.entries.len);
    try std.testing.expectEqual(@as(u64, 2), metadata.tensor.shape[0]);
    try std.testing.expectEqual(@as(u64, 1), metadata.tensor.shape[1]);
    try std.testing.expectEqual(@as(u64, 4), metadata.tensor.shape[2]);
    try std.testing.expectEqual(@as(u64, 101), metadata.batch.entries[0].request_id);
    try std.testing.expectEqual(@as(u64, 303), metadata.batch.entries[1].request_id);
    try bridge.validatePayloadBufferLength(&metadata, 2 * 4 * @sizeOf(f32));
}

test "buildPrefillActivationMetadata hostActivationByteImage deviceActivationByteImage creates single-entry prefill frame images" {
    const MockBackend = struct {
        d_model: usize = 4,
        local_tensor_frame_plan_ref: ?bridge.TensorFramePlanRef = null,
        slot_request_ids: [2]?u64 = .{ 101, 202 },
    };
    const boundaries = [_]bridge.TensorFrameBoundaryRef{.{
        .boundary_index = 0,
        .source_stage_id = 0,
        .target_stage_id = 1,
        .producer_layer_start = 0,
        .producer_layer_end = 2,
        .consumer_layer_start = 2,
        .consumer_layer_end = 4,
    }};
    const plan_ref = bridge.TensorFramePlanRef{
        .allocator = std.testing.allocator,
        .identity = .{
            .graph_digest = [_]u8{1} ** 32,
            .graph_contract_version = 1,
            .stage_plan_contract_version = 1,
            .stage_plan_id = .{ .digest = [_]u8{2} ** 32 },
        },
        .boundaries = &boundaries,
    };
    var backend = MockBackend{ .local_tensor_frame_plan_ref = plan_ref };
    var entries: [1]bridge.TensorFrameBatchEntry = undefined;
    const metadata = try buildPrefillActivationMetadata(
        &backend,
        0,
        .f32,
        .row_major,
        .{ .cpu = {} },
        1,
        7,
        3,
        entries[0..],
    );

    try std.testing.expectEqual(bridge.TensorFrameStepKind.prefill, metadata.step_kind);
    try std.testing.expectEqual(@as(usize, 1), metadata.batch.entries.len);
    try std.testing.expectEqual(@as(u64, 1), metadata.tensor.shape[0]);
    try std.testing.expectEqual(@as(u64, 3), metadata.tensor.shape[1]);
    try std.testing.expectEqual(@as(u64, 4), metadata.tensor.shape[2]);
    try std.testing.expectEqual(@as(u64, 202), metadata.batch.entries[0].request_id);
    try std.testing.expectEqual(@as(u64, 2), metadata.batch.entries[0].slot_id);
    try std.testing.expectEqual(@as(u64, 7), metadata.batch.entries[0].sequence_start);
    try std.testing.expectEqual(@as(u64, 3), metadata.batch.entries[0].token_count);
    try bridge.validatePayloadBufferLength(&metadata, 3 * 4 * @sizeOf(f32));

    var host_storage = [_]u8{0x5a} ** (3 * 4 * @sizeOf(f32));
    const host_image = bridge.hostActivationByteImage(&metadata, host_storage[0..]);
    try std.testing.expectEqual(bridge.BoundaryByteImageReadiness.host_readable_now, host_image.readiness);
    try std.testing.expectEqual(metadata.payload.byte_count, host_image.byte_count);
    try std.testing.expectEqualSlices(u8, host_storage[0..], host_image.host_bytes.?);

    const device_image = bridge.deviceActivationByteImage(&metadata);
    try std.testing.expectEqual(bridge.BoundaryByteImageReadiness.device_download_required, device_image.readiness);
    try std.testing.expectEqual(metadata.payload.byte_count, device_image.byte_count);
    try std.testing.expect(device_image.host_bytes == null);
    try std.testing.expect(device_image.host_segments == null);

    try std.testing.expectError(
        error.InvalidArgument,
        buildPrefillActivationMetadata(&backend, 0, .f32, .row_major, .{ .cpu = {} }, 1, 7, 0, entries[0..]),
    );
}
