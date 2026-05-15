//! Shared adapter helpers for CUDA staged decode and prefill routes.
//!
//! This module owns common stage-method validation and stage metadata helpers
//! used by local staged-route adapters.

const std = @import("std");
const bridge = @import("../../../bridge/root.zig");
const transport = @import("../../../transport/root.zig");

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

pub fn localTopologyTensorFramePlanRef(backend: anytype) !*const bridge.TensorFramePlanRef {
    const BackendType = @TypeOf(backend.*);
    if (comptime !@hasField(BackendType, "local_tensor_frame_plan_ref")) return error.InvalidTopologyConfig;
    if (backend.local_tensor_frame_plan_ref) |*plan_ref| return plan_ref;
    return error.InvalidTopologyConfig;
}

pub fn localTopologyPlacementPlan(backend: anytype) !*const bridge.PlacementPlan {
    const BackendType = @TypeOf(backend.*);
    if (comptime !@hasField(BackendType, "local_placement_plan")) return error.InvalidTopologyConfig;
    if (backend.local_placement_plan) |*placement_plan| return placement_plan;
    return error.InvalidTopologyConfig;
}

pub fn localTopologyStateOwnershipPlan(backend: anytype) ?*const bridge.StageStateOwnershipPlan {
    const BackendType = @TypeOf(backend.*);
    if (comptime !@hasField(BackendType, "local_state_ownership_plan")) return null;
    if (backend.local_state_ownership_plan) |*state_plan| return state_plan;
    return null;
}

pub fn localTopologyRunnerPlanRef(backend: anytype) !*const bridge.LocalStageRunnerPlanRef {
    const BackendType = @TypeOf(backend.*);
    if (comptime !@hasField(BackendType, "local_stage_runner_plan_ref")) return error.InvalidTopologyConfig;
    if (backend.local_stage_runner_plan_ref) |*plan_ref| return plan_ref;
    return error.InvalidTopologyConfig;
}

pub fn localPipelineContext(backend: anytype) !bridge.LocalPipelineContext {
    return .{
        .allocator = backendAllocator(backend),
        .plan_ref = try localTopologyRunnerPlanRef(backend),
        .placement_plan = try localTopologyPlacementPlan(backend),
        .state_ownership_plan = localTopologyStateOwnershipPlan(backend),
    };
}

pub fn localPipelinePlacementKind(backend: anytype) !?bridge.LocalPipelinePlacementKind {
    const BackendType = @TypeOf(backend.*);
    if (comptime !@hasField(BackendType, "local_stage_specs")) return null;
    if (backend.local_stage_specs.len == 0) return null;

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
    return try bridge.resolveLocalPipelinePlacementKind(try localPipelineContext(backend), bindings);
}

pub const LocalBoundaryRuntimeView = struct {
    boundary_index: usize,
    dtype: bridge.BoundaryDType,
    layout: bridge.BoundaryLayout,
    staging: ?[]align(64) u8 = null,
    local_device_peer_copy_available: bool = false,
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
        };
    }
    return error.InvalidTopologyConfig;
}

pub fn localLayerOffset(backend: anytype) usize {
    const BackendType = @TypeOf(backend.*);
    if (comptime @hasDecl(BackendType, "localSplitLayer")) {
        return backend.localSplitLayer();
    }
    if (comptime @hasField(BackendType, "split_layer")) {
        return backend.split_layer;
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
        .plan_ref = try localTopologyTensorFramePlanRef(backend),
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

pub const LocalBoundaryStageIds = struct {
    source_stage_id: usize,
    target_stage_id: usize,
};

pub fn localBoundaryStageIds(backend: anytype, boundary_index: usize) !LocalBoundaryStageIds {
    const boundary = try (try localTopologyRunnerPlanRef(backend)).boundary(boundary_index);
    return .{
        .source_stage_id = boundary.source_stage_id,
        .target_stage_id = boundary.target_stage_id,
    };
}

pub const DecodeBoundaryImageSpec = union(enum) {
    device,
    host_bytes: []const u8,
    host_segments: []const []const u8,
};

pub const DecodeBoundaryPayloadSpec = struct {
    boundary_index: usize,
    activation_byte_count: usize,
    location_hint: ?bridge.TensorFramePayloadLocationHint,
    image: DecodeBoundaryImageSpec,
    local_device_peer_copy_available: ?bool = null,
};

pub fn executeDecodeBoundaryPipeline(
    root_backend: anytype,
    stages: []bridge.LocalStageChainStage,
    slot_indices: []const usize,
    positions: []const usize,
    payload_specs: []const DecodeBoundaryPayloadSpec,
) !void {
    if (slot_indices.len != positions.len) return error.InvalidArgument;
    if (payload_specs.len == 0 or payload_specs.len + 1 != stages.len) return error.InvalidStepRequest;
    if (payload_specs.len > 2) return error.InvalidStepRequest;

    var entry_scratches: [2]DecodeBatchEntryScratch = undefined;
    var initialized_scratches: usize = 0;
    defer {
        for (entry_scratches[0..initialized_scratches]) |*scratch| scratch.deinit();
    }

    var metadata: [2]bridge.TensorFrameMetadata = undefined;
    var images: [2]bridge.BoundaryByteImageRef = undefined;
    var payloads: [2]bridge.LocalPipelineBoundaryPayload = undefined;

    for (payload_specs, 0..) |spec, index| {
        const boundary = try localBoundaryRuntime(root_backend, spec.boundary_index);
        entry_scratches[index] = try DecodeBatchEntryScratch.init(
            backendAllocator(root_backend),
            slot_indices.len,
        );
        initialized_scratches += 1;
        metadata[index] = try buildDecodeActivationMetadata(
            root_backend,
            boundary.boundary_index,
            boundary.dtype,
            boundary.layout,
            spec.location_hint,
            slot_indices,
            positions,
            entry_scratches[index].slice(slot_indices.len),
        );
        try bridge.validatePayloadBufferLength(&metadata[index], spec.activation_byte_count);
        images[index] = switch (spec.image) {
            .device => bridge.deviceActivationByteImage(&metadata[index]),
            .host_bytes => |host_bytes| blk: {
                if (spec.activation_byte_count > host_bytes.len) return error.InvalidArgument;
                break :blk bridge.hostActivationByteImage(&metadata[index], host_bytes[0..spec.activation_byte_count]);
            },
            .host_segments => |host_segments| blk: {
                if (host_segments.len != slot_indices.len) return error.InvalidArgument;
                break :blk bridge.segmentedHostActivationByteImage(&metadata[index], host_segments);
            },
        };
        payloads[index] = .{
            .metadata = &metadata[index],
            .image = &images[index],
            .runtime = .{
                .staging = boundary.staging,
                .allow_borrow = false,
                .local_device_peer_copy_available = spec.local_device_peer_copy_available orelse boundary.local_device_peer_copy_available,
            },
        };
    }

    try bridge.executeLocalPipelineStep(
        try localPipelineContext(root_backend),
        stages,
        payloads[0..payload_specs.len],
        .decode,
        &.{},
    );
}

pub fn executeCudaDecodeActivationBoundary(
    root_backend: anytype,
    source_backend: anytype,
    target_backend: anytype,
    boundary_index: usize,
    slot_indices: []const usize,
    positions: []const usize,
    activation_byte_count: usize,
    comptime synchronization: transport.CudaPeerCopySynchronization,
) !void {
    const Source = transport.CudaPeerActivationStage(@TypeOf(source_backend), @TypeOf(target_backend), synchronization);
    const Target = transport.CudaActivationStage(@TypeOf(target_backend));
    const stage_ids = try localBoundaryStageIds(root_backend, boundary_index);
    var source = Source{ .backend = source_backend, .target_backend = target_backend };
    var target = Target{ .backend = target_backend };
    var stages = [_]bridge.LocalStageChainStage{
        bridge.localStageAdapter(Source, stage_ids.source_stage_id, &source),
        bridge.localStageAdapter(Target, stage_ids.target_stage_id, &target),
    };
    const payload_specs = [_]DecodeBoundaryPayloadSpec{.{
        .boundary_index = boundary_index,
        .activation_byte_count = activation_byte_count,
        .location_hint = try cudaPayloadLocationHint(source_backend),
        .image = .device,
    }};
    try executeDecodeBoundaryPipeline(root_backend, stages[0..], slot_indices, positions, payload_specs[0..]);
}

pub fn executeCpuSegmentedDecodeActivationBoundary(
    root_backend: anytype,
    cpu_stage0: anytype,
    gpu_target: anytype,
    boundary_index: usize,
    slot_indices: []const usize,
    positions: []const usize,
    host_segments: []const []const u8,
    activation_byte_count: usize,
) !void {
    if (slot_indices.len != host_segments.len) return error.InvalidArgument;
    const Source = CpuSegmentedBatchedDecodeSourceStage(@TypeOf(cpu_stage0), @TypeOf(gpu_target));
    const Target = transport.CudaActivationStage(@TypeOf(gpu_target));
    const stage_ids = try localBoundaryStageIds(root_backend, boundary_index);
    var source = Source{
        .backend = cpu_stage0,
        .gpu_backend = gpu_target,
        .slot_indices = slot_indices,
        .positions = positions,
    };
    var target = Target{ .backend = gpu_target };
    var stages = [_]bridge.LocalStageChainStage{
        bridge.localStageAdapter(Source, stage_ids.source_stage_id, &source),
        bridge.localStageAdapter(Target, stage_ids.target_stage_id, &target),
    };
    const payload_specs = [_]DecodeBoundaryPayloadSpec{.{
        .boundary_index = boundary_index,
        .activation_byte_count = activation_byte_count,
        .location_hint = .{ .cpu = {} },
        .image = .{ .host_segments = host_segments },
        .local_device_peer_copy_available = false,
    }};
    try executeDecodeBoundaryPipeline(root_backend, stages[0..], slot_indices, positions, payload_specs[0..]);
}

pub const PrefillBoundaryImageSpec = union(enum) {
    device,
    host_bytes: []const u8,
};

pub const PrefillBoundaryPayloadSpec = struct {
    boundary_index: usize,
    slot_index: usize,
    sequence_start: usize,
    token_count: usize,
    activation_byte_count: usize,
    location_hint: ?bridge.TensorFramePayloadLocationHint,
    image: PrefillBoundaryImageSpec,
    local_device_peer_copy_available: ?bool = null,
};

pub fn executePrefillBoundaryPipeline(
    root_backend: anytype,
    stages: []bridge.LocalStageChainStage,
    payload_specs: []const PrefillBoundaryPayloadSpec,
) !void {
    if (payload_specs.len == 0 or payload_specs.len + 1 != stages.len) return error.InvalidStepRequest;
    if (payload_specs.len > 2) return error.InvalidStepRequest;

    var batch_entries: [2][1]bridge.TensorFrameBatchEntry = undefined;
    var metadata: [2]bridge.TensorFrameMetadata = undefined;
    var images: [2]bridge.BoundaryByteImageRef = undefined;
    var payloads: [2]bridge.LocalPipelineBoundaryPayload = undefined;
    const plan_ref = try localTopologyTensorFramePlanRef(root_backend);

    for (payload_specs, 0..) |spec, index| {
        const boundary = try localBoundaryRuntime(root_backend, spec.boundary_index);
        metadata[index] = try buildPrefillActivationMetadata(
            root_backend,
            boundary.boundary_index,
            boundary.dtype,
            boundary.layout,
            spec.location_hint,
            spec.slot_index,
            spec.sequence_start,
            spec.token_count,
            batch_entries[index][0..],
        );
        try bridge.validateTensorFrameForPlanBoundary(&metadata[index], plan_ref, boundary.boundary_index);
        try bridge.validatePayloadBufferLength(&metadata[index], spec.activation_byte_count);
        images[index] = switch (spec.image) {
            .device => bridge.deviceActivationByteImage(&metadata[index]),
            .host_bytes => |host_bytes| blk: {
                if (spec.activation_byte_count > host_bytes.len) return error.InvalidArgument;
                break :blk bridge.hostActivationByteImage(&metadata[index], host_bytes[0..spec.activation_byte_count]);
            },
        };
        payloads[index] = .{
            .metadata = &metadata[index],
            .image = &images[index],
            .runtime = .{
                .staging = boundary.staging,
                .allow_borrow = false,
                .local_device_peer_copy_available = spec.local_device_peer_copy_available orelse boundary.local_device_peer_copy_available,
            },
        };
    }

    try bridge.executeLocalPipelineStep(
        try localPipelineContext(root_backend),
        stages,
        payloads[0..payload_specs.len],
        .prefill,
        &.{},
    );
}

pub fn executeHostToCudaPrefillBoundary(
    root_backend: anytype,
    cuda_target: anytype,
    boundary_index: usize,
    slot_index: usize,
    sequence_start: usize,
    token_count: usize,
    host_bytes: []const u8,
    activation_byte_count: usize,
) !void {
    const Source = transport.NoopActivationStage;
    const Target = transport.CudaActivationStage(@TypeOf(cuda_target));
    const stage_ids = try localBoundaryStageIds(root_backend, boundary_index);
    var source = Source{};
    var target = Target{ .backend = cuda_target, .slot_index = slot_index };
    var stages = [_]bridge.LocalStageChainStage{
        bridge.localStageAdapter(Source, stage_ids.source_stage_id, &source),
        bridge.localStageAdapter(Target, stage_ids.target_stage_id, &target),
    };
    const payload_specs = [_]PrefillBoundaryPayloadSpec{.{
        .boundary_index = boundary_index,
        .slot_index = slot_index,
        .sequence_start = sequence_start,
        .token_count = token_count,
        .activation_byte_count = activation_byte_count,
        .location_hint = .{ .cpu = {} },
        .image = .{ .host_bytes = host_bytes },
    }};
    try executePrefillBoundaryPipeline(root_backend, stages[0..], payload_specs[0..]);
}

pub fn executeCudaDevicePrefillBoundary(
    root_backend: anytype,
    source_backend: anytype,
    target_backend: anytype,
    boundary_index: usize,
    slot_index: usize,
    sequence_start: usize,
    token_count: usize,
    activation_byte_count: usize,
    comptime synchronization: transport.CudaPeerCopySynchronization,
) !void {
    const Source = transport.CudaPeerActivationStage(@TypeOf(source_backend), @TypeOf(target_backend), synchronization);
    const Target = transport.CudaActivationStage(@TypeOf(target_backend));
    const stage_ids = try localBoundaryStageIds(root_backend, boundary_index);
    var source = Source{ .backend = source_backend, .target_backend = target_backend, .slot_index = slot_index };
    var target = Target{ .backend = target_backend, .slot_index = slot_index };
    var stages = [_]bridge.LocalStageChainStage{
        bridge.localStageAdapter(Source, stage_ids.source_stage_id, &source),
        bridge.localStageAdapter(Target, stage_ids.target_stage_id, &target),
    };
    const payload_specs = [_]PrefillBoundaryPayloadSpec{.{
        .boundary_index = boundary_index,
        .slot_index = slot_index,
        .sequence_start = sequence_start,
        .token_count = token_count,
        .activation_byte_count = activation_byte_count,
        .location_hint = try cudaPayloadLocationHint(source_backend),
        .image = .device,
    }};
    try executePrefillBoundaryPipeline(root_backend, stages[0..], payload_specs[0..]);
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
        .plan_ref = try localTopologyTensorFramePlanRef(backend),
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
    use_preloaded_input: bool,
) !void {
    if (comptime !hasDecl(@TypeOf(backend.*), "executeDecodeLayerRange")) {
        return error.InvalidTopologyConfig;
    }
    try backend.executeDecodeLayerRange(
        ctx.token,
        ctx.position,
        ctx.slot_index,
        null,
        layer_start,
        layer_end,
        false,
        false,
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

pub fn CpuDecodeSourceStage(
    comptime CpuBackend: type,
    comptime GpuBackend: type,
) type {
    return struct {
        backend: CpuBackend,
        gpu_backend: GpuBackend,
        ctx: *const DecodeContext,

        pub fn executeLayers(stage: *@This(), input: []const u8, layer_start: usize, layer_end: usize) anyerror!void {
            try validateEmptyInput(input);
            try executeCpuDecodeLayerRange(stage.backend, stage.ctx, layer_start, layer_end, false);
        }

        pub fn synchronize(stage: *@This()) anyerror!void {
            try transport.uploadCpuKvToCudaMirrors(stage.gpu_backend, stage.backend, stage.ctx.slot_index, stage.ctx.position, 1);
        }

        pub fn downloadActivation(stage: *@This(), host_buf: []u8, byte_count: usize) anyerror!void {
            try transport.downloadHostSlotActivation(stage.backend, stage.ctx.slot_index, host_buf, byte_count);
        }

        pub fn uploadActivation(_: *@This(), _: []const u8, _: usize) anyerror!void {
            return error.InvalidTopologyConfig;
        }
    };
}

pub fn CudaDecodeSourceStage(
    comptime Backend: type,
    comptime TargetBackend: type,
    comptime execute_decode_with_layer_limit: anytype,
    comptime synchronization: transport.CudaPeerCopySynchronization,
) type {
    return struct {
        backend: Backend,
        target_backend: TargetBackend,
        ctx: *const DecodeContext,
        hidden_override: ?[]const f32 = null,
        deepstack_layer_features_opt: ?[]const []const f32 = null,
        deepstack_feature_index_opt: ?usize = null,
        use_preloaded_input: bool = false,

        pub fn executeLayers(stage: *@This(), input: []const u8, layer_start: usize, layer_end: usize) anyerror!void {
            try validateEmptyInput(input);
            try executeCudaDecodeLayerRange(
                execute_decode_with_layer_limit,
                stage.backend,
                stage.ctx,
                layer_start,
                layer_end,
                null,
                false,
                false,
                stage.hidden_override,
                stage.deepstack_layer_features_opt,
                stage.deepstack_feature_index_opt,
                stage.use_preloaded_input,
            );
        }

        pub fn synchronize(stage: *@This()) anyerror!void {
            try transport.synchronizeCudaActivationBackend(stage.backend);
        }

        pub fn downloadActivation(stage: *@This(), host_buf: []u8, byte_count: usize) anyerror!void {
            try transport.downloadCudaActivation(stage.backend, host_buf, byte_count);
        }

        pub fn uploadActivation(stage: *@This(), host_buf: []const u8, byte_count: usize) anyerror!void {
            try transport.uploadCudaActivation(stage.backend, stage.ctx.slot_index, host_buf, byte_count);
        }

        pub fn peerCopyActivationToErased(stage: *@This(), target_ptr: *anyopaque, byte_count: usize) anyerror!void {
            _ = target_ptr;
            try transport.peerCopyCudaActivation(stage.backend, stage.target_backend, byte_count, synchronization);
        }

        pub fn peerCopyHandlesStageSync(stage: *const @This()) bool {
            return transport.peerCopyCudaActivationHandlesStageSync(stage.backend, synchronization);
        }
    };
}

pub fn CudaDecodeTargetStage(
    comptime Backend: type,
    comptime execute_decode_with_layer_limit: anytype,
) type {
    return struct {
        backend: Backend,
        ctx: *const DecodeContext,
        deepstack_layer_features_opt: ?[]const []const f32 = null,
        deepstack_feature_index_opt: ?usize = null,

        pub fn executeLayers(stage: *@This(), input: []const u8, layer_start: usize, layer_end: usize) anyerror!void {
            try validateEmptyInput(input);
            try executeCudaDecodeLayerRange(
                execute_decode_with_layer_limit,
                stage.backend,
                stage.ctx,
                layer_start,
                layer_end,
                stage.ctx.logits_out_opt,
                stage.ctx.compute_logits,
                stage.ctx.download_logits,
                null,
                stage.deepstack_layer_features_opt,
                stage.deepstack_feature_index_opt,
                true,
            );
        }

        pub fn synchronize(stage: *@This()) anyerror!void {
            try transport.synchronizeCudaActivationBackend(stage.backend);
        }

        pub fn downloadActivation(stage: *@This(), host_buf: []u8, byte_count: usize) anyerror!void {
            try transport.downloadCudaActivation(stage.backend, host_buf, byte_count);
        }

        pub fn uploadActivation(stage: *@This(), host_buf: []const u8, byte_count: usize) anyerror!void {
            try transport.uploadCudaActivation(stage.backend, stage.ctx.slot_index, host_buf, byte_count);
        }
    };
}

pub fn CpuSegmentedBatchedDecodeSourceStage(
    comptime CpuBackend: type,
    comptime GpuBackend: type,
) type {
    return struct {
        backend: CpuBackend,
        gpu_backend: GpuBackend,
        slot_indices: []const usize,
        positions: []const usize,

        pub fn executeLayers(_: *@This(), input: []const u8, _: usize, _: usize) anyerror!void {
            try validateEmptyInput(input);
        }

        pub fn synchronize(stage: *@This()) anyerror!void {
            for (stage.slot_indices, stage.positions) |slot_index, position| {
                try transport.uploadCpuKvToCudaMirrors(stage.gpu_backend, stage.backend, slot_index, position, 1);
            }
        }

        pub fn downloadActivation(_: *@This(), _: []u8, _: usize) anyerror!void {
            return error.InvalidTopologyConfig;
        }

        pub fn uploadActivation(_: *@This(), _: []const u8, _: usize) anyerror!void {
            return error.InvalidTopologyConfig;
        }
    };
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

test "localTopologyStateOwnershipPlan returns optional topology ownership plan field" {
    const WithoutPlan = struct {
        allocator: std.mem.Allocator = std.testing.allocator,
    };
    const WithPlan = struct {
        local_state_ownership_plan: ?bridge.StageStateOwnershipPlan = null,
    };
    var without_plan = WithoutPlan{};
    var with_plan = WithPlan{};

    try std.testing.expect(localTopologyStateOwnershipPlan(&without_plan) == null);
    try std.testing.expect(localTopologyStateOwnershipPlan(&with_plan) == null);
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

test "localPipelinePlacementKind returns null without local stage specs and rejects incomplete contracts" {
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

    try std.testing.expect(try localPipelinePlacementKind(&without_specs) == null);
    try std.testing.expectError(error.InvalidTopologyConfig, localPipelinePlacementKind(&with_specs));
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

test "localBoundaryStageIds rejects missing runner plan" {
    const MockBackend = struct {};
    var backend = MockBackend{};

    try std.testing.expectError(error.InvalidTopologyConfig, localBoundaryStageIds(&backend, 0));
}

test "executeDecodeBoundaryPipeline rejects invalid stage boundary shape" {
    const MockBackend = struct {};
    var backend = MockBackend{};
    var stages: [2]bridge.LocalStageChainStage = undefined;
    const slots = [_]usize{0};
    const positions = [_]usize{0};

    try std.testing.expectError(
        error.InvalidStepRequest,
        executeDecodeBoundaryPipeline(&backend, stages[0..], &slots, &positions, &.{}),
    );
}

test "executeCudaDecodeActivationBoundary rejects missing runner plan" {
    const MockBackend = struct {};
    var backend = MockBackend{};
    const slots = [_]usize{0};
    const positions = [_]usize{0};

    try std.testing.expectError(
        error.InvalidTopologyConfig,
        executeCudaDecodeActivationBoundary(&backend, &backend, &backend, 0, &slots, &positions, 4, .source_stream),
    );
}

test "executeCpuSegmentedDecodeActivationBoundary rejects mismatched host segments" {
    const MockBackend = struct {};
    var backend = MockBackend{};
    const slots = [_]usize{0};
    const positions = [_]usize{0};

    try std.testing.expectError(
        error.InvalidArgument,
        executeCpuSegmentedDecodeActivationBoundary(&backend, &backend, &backend, 0, &slots, &positions, &.{}, 4),
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

test "executePrefillBoundaryPipeline rejects invalid stage boundary shape" {
    const MockBackend = struct {};
    var backend = MockBackend{};
    var stages: [2]bridge.LocalStageChainStage = undefined;

    try std.testing.expectError(
        error.InvalidStepRequest,
        executePrefillBoundaryPipeline(&backend, stages[0..], &.{}),
    );
}

test "executePrefillBoundaryPipeline builds host and device prefill payload specs" {
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

    try std.testing.expectError(
        error.InvalidArgument,
        executePrefillBoundaryPipeline(&backend, stages[0..], &.{.{
            .boundary_index = 0,
            .slot_index = 0,
            .sequence_start = 3,
            .token_count = 2,
            .activation_byte_count = activation_byte_count,
            .location_hint = .{ .cpu = {} },
            .image = .{ .host_bytes = short_host_storage[0..] },
        }}),
    );
    try std.testing.expectError(
        error.InvalidTopologyConfig,
        executePrefillBoundaryPipeline(&backend, stages[0..], &.{.{
            .boundary_index = 0,
            .slot_index = 0,
            .sequence_start = 3,
            .token_count = 2,
            .activation_byte_count = activation_byte_count,
            .location_hint = .{ .cuda = 0 },
            .image = .device,
        }}),
    );
}

test "executeHostToCudaPrefillBoundary rejects missing runner plan" {
    var backend = PrefillBoundaryTestBackend{};
    var host_storage = [_]u8{0x5a} ** (2 * 4 * @sizeOf(f32));

    try std.testing.expectError(
        error.InvalidTopologyConfig,
        executeHostToCudaPrefillBoundary(&backend, &backend, 0, 0, 3, 2, host_storage[0..], host_storage.len),
    );
}

test "executeCudaDevicePrefillBoundary rejects missing runner plan" {
    var backend = PrefillBoundaryTestBackend{};

    try std.testing.expectError(
        error.InvalidTopologyConfig,
        executeCudaDevicePrefillBoundary(&backend, &backend, &backend, 0, 0, 3, 2, 2 * 4 * @sizeOf(f32), .source_stream),
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
