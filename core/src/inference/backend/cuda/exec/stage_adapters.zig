//! Shared adapter helpers for CUDA staged decode and prefill routes.
//!
//! This module owns common stage-method validation and activation movement used
//! by local staged-route adapters.

const std = @import("std");
const bridge = @import("../../../bridge/root.zig");
const engine_weights = @import("../weights/root.zig");

const bufferSlice = engine_weights.bufferSlice;

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
    if (comptime !@hasField(BackendType, "cpu_gpu_tensor_frame_plan_ref")) return error.InvalidTopologyConfig;
    if (backend.cpu_gpu_tensor_frame_plan_ref) |*plan_ref| return plan_ref;
    return error.InvalidTopologyConfig;
}

pub fn localTopologyPlacementPlan(backend: anytype) !*const bridge.PlacementPlan {
    const BackendType = @TypeOf(backend.*);
    if (comptime !@hasField(BackendType, "cpu_gpu_placement_plan")) return error.InvalidTopologyConfig;
    if (backend.cpu_gpu_placement_plan) |*placement_plan| return placement_plan;
    return error.InvalidTopologyConfig;
}

pub fn localTopologyStateOwnershipPlan(backend: anytype) ?*const bridge.StageStateOwnershipPlan {
    const BackendType = @TypeOf(backend.*);
    if (comptime !@hasField(BackendType, "cpu_gpu_state_ownership_plan")) return null;
    if (backend.cpu_gpu_state_ownership_plan) |*state_plan| return state_plan;
    return null;
}

pub fn localTopologyRunnerPlanRef(backend: anytype) !*const bridge.LocalStageRunnerPlanRef {
    const BackendType = @TypeOf(backend.*);
    if (comptime !@hasField(BackendType, "cpu_gpu_local_stage_runner_plan_ref")) return error.InvalidTopologyConfig;
    if (backend.cpu_gpu_local_stage_runner_plan_ref) |*plan_ref| return plan_ref;
    return error.InvalidTopologyConfig;
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
        null,
        null,
        null,
        use_preloaded_input,
    );
}

pub fn downloadCpuActivation(backend: anytype, slot_index: usize, host_buf: []u8, byte_count: usize) !void {
    if (byte_count > host_buf.len) return error.InvalidArgument;
    const source = backend.slotActivationBytes(slot_index);
    if (byte_count > source.len) return error.InvalidArgument;
    @memcpy(host_buf[0..byte_count], source[0..byte_count]);
}

pub fn uploadCpuActivation(backend: anytype, slot_index: usize, host_buf: []const u8, byte_count: usize) !void {
    if (byte_count > host_buf.len) return error.InvalidArgument;
    const dst = backend.slotActivationBytesMut(slot_index);
    if (byte_count > dst.len) return error.InvalidArgument;
    @memcpy(dst[0..byte_count], host_buf[0..byte_count]);
}

pub fn downloadCudaActivation(backend: anytype, host_buf: []u8, byte_count: usize) !void {
    if (byte_count > host_buf.len) return error.InvalidArgument;
    const BackendType = @TypeOf(backend.*);
    if (comptime @hasDecl(BackendType, "downloadPipelineActivationToHost")) {
        return backend.downloadPipelineActivationToHost(host_buf[0..byte_count], byte_count);
    }
    if (comptime @hasField(BackendType, "runtime_buffers") and @hasField(BackendType, "device")) {
        return backend.runtime_buffers.input_dev.download(&backend.device, host_buf[0..byte_count]);
    }
    return error.InvalidTopologyConfig;
}

pub fn uploadCudaActivation(backend: anytype, slot_index: usize, host_buf: []const u8, byte_count: usize) !void {
    if (byte_count > host_buf.len) return error.InvalidArgument;
    const BackendType = @TypeOf(backend.*);
    if (comptime @hasDecl(BackendType, "uploadPipelineActivationFromHost")) {
        return backend.uploadPipelineActivationFromHost(slot_index, host_buf[0..byte_count], byte_count);
    }
    if (comptime @hasField(BackendType, "runtime_buffers") and @hasField(BackendType, "device")) {
        return backend.runtime_buffers.input_dev.upload(&backend.device, host_buf[0..byte_count]);
    }
    return error.InvalidTopologyConfig;
}

pub fn uploadCudaActivationSegments(backend: anytype, host_segments: []const []const u8, byte_count: usize) !void {
    const BackendType = @TypeOf(backend.*);
    if (comptime !@hasField(BackendType, "runtime_buffers") or !@hasField(BackendType, "device")) {
        return error.InvalidTopologyConfig;
    }
    var offset: usize = 0;
    for (host_segments) |segment| {
        offset = std.math.add(usize, offset, segment.len) catch return error.InvalidArgument;
        if (offset > byte_count) return error.InvalidArgument;
        const start = offset - segment.len;
        var dst_slice = try bufferSlice(&backend.runtime_buffers.input_dev, start, segment.len);
        try dst_slice.upload(&backend.device, segment);
    }
    if (offset != byte_count) return error.InvalidArgument;
}

pub fn synchronizeCudaBackend(backend: anytype) !void {
    const BackendType = @TypeOf(backend.*);
    if (comptime @hasDecl(BackendType, "synchronizePipelineActivation")) {
        return backend.synchronizePipelineActivation();
    }
    if (comptime !@hasField(BackendType, "device")) return error.InvalidTopologyConfig;
    if (comptime @hasField(BackendType, "compute_stream")) {
        if (backend.compute_stream) |stream| {
            const DeviceType = @TypeOf(backend.device);
            if (comptime @hasDecl(DeviceType, "synchronizeStream")) {
                try backend.device.synchronizeStream(stream);
                return;
            }
        }
    }
    const DeviceType = @TypeOf(backend.device);
    if (comptime @hasDecl(DeviceType, "synchronize")) {
        try backend.device.synchronize();
        return;
    }
    return error.InvalidTopologyConfig;
}

pub fn peerCopyPipelineActivation(source_backend: anytype, target_backend: anytype, byte_count: usize) !void {
    if (byte_count == 0) return;
    const SourceType = @TypeOf(source_backend.*);
    const TargetType = @TypeOf(target_backend.*);
    if (comptime !@hasField(SourceType, "device") or
        !@hasField(SourceType, "runtime_buffers") or
        !@hasField(SourceType, "compute_stream") or
        !@hasField(TargetType, "device") or
        !@hasField(TargetType, "runtime_buffers") or
        !@hasField(TargetType, "compute_stream"))
    {
        return error.InvalidTopologyConfig;
    }
    if (!target_backend.device.canAccessPeer(&source_backend.device)) return error.InvalidTopologyConfig;
    target_backend.device.enablePeerAccess(&source_backend.device) catch {};
    source_backend.device.enablePeerAccess(&target_backend.device) catch {};

    if (comptime @hasField(SourceType, "pipeline_stage0_event")) {
        if (source_backend.pipeline_stage0_event) |event| {
            try source_backend.device.recordEvent(event, source_backend.compute_stream);
            try target_backend.device.streamWaitEvent(target_backend.compute_stream, event);
            try target_backend.device.makeCurrent();
            return source_backend.device.memcpyPeerAsync(
                &target_backend.device,
                &target_backend.runtime_buffers.input_dev,
                &source_backend.runtime_buffers.input_dev,
                byte_count,
                target_backend.compute_stream,
            );
        }
    }

    try source_backend.device.memcpyPeerAsync(
        &target_backend.device,
        &target_backend.runtime_buffers.input_dev,
        &source_backend.runtime_buffers.input_dev,
        byte_count,
        source_backend.compute_stream,
    );
    if (source_backend.compute_stream) |stream| {
        try source_backend.device.synchronizeStream(stream);
    } else {
        try source_backend.device.synchronize();
    }
}

pub fn peerCopyStage12Activation(source_backend: anytype, target_backend: anytype, byte_count: usize) !void {
    if (byte_count == 0) return;
    const SourceType = @TypeOf(source_backend.*);
    const TargetType = @TypeOf(target_backend.*);
    if (comptime !@hasField(SourceType, "device") or
        !@hasField(SourceType, "runtime_buffers") or
        !@hasField(SourceType, "compute_stream") or
        !@hasField(TargetType, "device") or
        !@hasField(TargetType, "runtime_buffers"))
    {
        return error.InvalidTopologyConfig;
    }
    if (!target_backend.device.canAccessPeer(&source_backend.device)) return error.InvalidTopologyConfig;
    target_backend.device.enablePeerAccess(&source_backend.device) catch {};
    source_backend.device.enablePeerAccess(&target_backend.device) catch {};
    try source_backend.device.memcpyPeerAsync(
        &target_backend.device,
        &target_backend.runtime_buffers.input_dev,
        &source_backend.runtime_buffers.input_dev,
        byte_count,
        source_backend.compute_stream,
    );
    if (source_backend.compute_stream) |stream| {
        try source_backend.device.synchronizeStream(stream);
    } else {
        try source_backend.device.synchronize();
    }
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
        cpu_gpu_state_ownership_plan: ?bridge.StageStateOwnershipPlan = null,
    };
    var without_plan = WithoutPlan{};
    var with_plan = WithPlan{};

    try std.testing.expect(localTopologyStateOwnershipPlan(&without_plan) == null);
    try std.testing.expect(localTopologyStateOwnershipPlan(&with_plan) == null);
}

test "buildDecodeActivationMetadata creates multi-entry decode frame" {
    const MockBackend = struct {
        d_model: usize = 4,
        cpu_gpu_tensor_frame_plan_ref: ?bridge.TensorFramePlanRef = null,
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
    var backend = MockBackend{ .cpu_gpu_tensor_frame_plan_ref = plan_ref };
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
        cpu_gpu_tensor_frame_plan_ref: ?bridge.TensorFramePlanRef = null,
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
    var backend = MockBackend{ .cpu_gpu_tensor_frame_plan_ref = plan_ref };
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
