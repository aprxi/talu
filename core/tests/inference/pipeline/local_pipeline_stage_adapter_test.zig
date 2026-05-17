//! Tests for pipeline-owned backend stage adaptation.

const std = @import("std");
const main = @import("main");

const pipeline = main.inference.pipeline;
const transport = main.inference.transport;
const runtime_contract = main.inference.runtime_contract;

const MockTrace = struct {
    executed: [8]usize = undefined,
    executed_count: usize = 0,

    fn push(self: *@This(), stage_id: usize) void {
        self.executed[self.executed_count] = stage_id;
        self.executed_count += 1;
    }
};

const MockDevice = struct {
    ordinal_value: usize,

    pub fn ordinal(self: *const @This()) usize {
        return self.ordinal_value;
    }

    pub fn synchronize(_: *@This()) !void {}

    pub fn synchronizeStream(_: *@This(), _: *anyopaque) !void {}

    pub fn canAccessPeer(_: *@This(), _: *@This()) bool {
        return true;
    }

    pub fn enablePeerAccess(_: *@This(), _: *@This()) !void {}

    pub fn memcpyPeerAsync(
        _: *@This(),
        _: *@This(),
        target: *MockDeviceBuffer,
        source: *MockDeviceBuffer,
        byte_count: usize,
        _: ?*anyopaque,
    ) !void {
        if (byte_count > target.bytes.len or byte_count > source.bytes.len) return error.InvalidArgument;
        @memcpy(target.bytes[0..byte_count], source.bytes[0..byte_count]);
    }
};

const MockDeviceBuffer = struct {
    bytes: [16]u8 = [_]u8{0} ** 16,
    size: usize = 16,
    pointer: u64 = 0,

    pub fn download(self: *@This(), _: anytype, host_buf: []u8) !void {
        if (host_buf.len > self.bytes.len) return error.InvalidArgument;
        @memcpy(host_buf, self.bytes[0..host_buf.len]);
    }

    pub fn upload(self: *@This(), _: anytype, host_buf: []const u8) !void {
        if (host_buf.len > self.bytes.len) return error.InvalidArgument;
        @memcpy(self.bytes[0..host_buf.len], host_buf);
    }
};

const MockStage = struct {
    const RuntimeBuffers = struct {
        input_dev: MockDeviceBuffer = .{},
    };

    id: usize,
    trace: *MockTrace,
    host_bytes: [16]u8 = [_]u8{0} ** 16,
    runtime_buffers: RuntimeBuffers = .{},
    device: MockDevice,
    compute_stream: ?*anyopaque = null,
    max_batch_size: usize = 2,
    prefill_chunk_rows_cap: usize = 4,
    last_use_preloaded_input: bool = false,

    pub fn deinit(_: *@This()) void {}

    pub fn stateDescriptors(_: *const @This()) []const runtime_contract.StateDescriptor {
        return &.{};
    }

    pub fn allocSlot(_: *@This()) ?usize {
        return 0;
    }

    pub fn freeSlot(_: *@This(), _: usize) void {}

    pub fn resetSlot(_: *@This(), _: usize) void {}

    pub fn bindSlotStateBlocks(_: *@This(), _: usize, blocks: []const runtime_contract.StateBlockHandle) !void {
        if (blocks.len != 0) return error.InvalidStateDescriptorBinding;
    }

    pub fn unbindSlotStateBlocks(_: *@This(), _: usize) void {}

    pub fn synchronize(_: *@This()) !void {}

    fn outputByte(self: *const @This()) u8 {
        return @intCast(self.id + 1);
    }
};

const HostResidentModule = struct {
    pub const interface = struct {
        pub const stage_executor = struct {
            pub const supports_local_stage_execution = true;

            pub fn backendKind() pipeline.HostBackendKind {
                return .cpu;
            }

            pub fn maxBatchSize(stage: anytype) usize {
                return stage.max_batch_size;
            }

            pub fn prefillChunkRowsCap(stage: anytype) usize {
                return stage.prefill_chunk_rows_cap;
            }

            pub fn executeDecodeLayerRange(
                stage: anytype,
                _: anytype,
                layer_start: usize,
                layer_end: usize,
                _: ?[]f32,
                _: bool,
                _: bool,
                use_preloaded_input: bool,
            ) !void {
                if (layer_end <= layer_start) return error.InvalidArgument;
                stage.last_use_preloaded_input = use_preloaded_input;
                stage.host_bytes[0] = stage.outputByte();
                stage.trace.push(stage.id);
            }

            pub fn executePrefillLayerRange(
                stage: anytype,
                _: usize,
                tokens: []const u32,
                _: usize,
                layer_start: usize,
                layer_end: usize,
                use_preloaded_input: bool,
                _: bool,
                _: ?[]f32,
                _: ?[]f32,
            ) !void {
                if (tokens.len == 0 or layer_end <= layer_start) return error.InvalidArgument;
                stage.last_use_preloaded_input = use_preloaded_input;
                stage.host_bytes[0] = stage.outputByte();
                stage.trace.push(stage.id);
            }
        };

        pub const transport_endpoint = struct {
            pub const supports_transport_endpoint_descriptors = true;
            pub const HostActivationOutput = struct { bytes: []const u8 };
            pub const HostActivationInput = struct { bytes: []u8 };

            pub fn deviceLocationHint(_: anytype) !pipeline.TensorFramePayloadLocationHint {
                return .{ .cpu = {} };
            }

            pub fn decodeExternalOutput(stage: anytype, _: usize, byte_count: usize) !HostActivationOutput {
                if (byte_count > stage.host_bytes.len) return error.InvalidArgument;
                return .{ .bytes = stage.host_bytes[0..byte_count] };
            }

            pub fn prefillExternalOutput(stage: anytype, byte_count: usize) !HostActivationOutput {
                return decodeExternalOutput(stage, 0, byte_count);
            }

            pub fn decodeExternalInput(stage: anytype, _: usize, byte_count: usize) !HostActivationInput {
                if (byte_count > stage.host_bytes.len) return error.InvalidArgument;
                return .{ .bytes = stage.host_bytes[0..byte_count] };
            }

            pub fn prefillExternalInput(stage: anytype, byte_count: usize) !HostActivationInput {
                return decodeExternalInput(stage, 0, byte_count);
            }

            pub fn sideExternalInput(_: anytype, _: usize) !HostActivationInput {
                return error.UnsupportedContentType;
            }
        };
    };
};

const DeviceResidentModule = struct {
    pub const interface = struct {
        pub const stage_executor = struct {
            pub const supports_local_stage_execution = true;

            pub fn backendKind() pipeline.HostBackendKind {
                return .cuda;
            }

            pub fn maxBatchSize(stage: anytype) usize {
                return stage.max_batch_size;
            }

            pub fn prefillChunkRowsCap(stage: anytype) usize {
                return stage.prefill_chunk_rows_cap;
            }

            pub fn executeDecodeLayerRange(
                stage: anytype,
                _: anytype,
                layer_start: usize,
                layer_end: usize,
                _: ?[]f32,
                _: bool,
                _: bool,
                use_preloaded_input: bool,
            ) !void {
                if (layer_end <= layer_start) return error.InvalidArgument;
                stage.last_use_preloaded_input = use_preloaded_input;
                stage.runtime_buffers.input_dev.bytes[0] = stage.outputByte();
                stage.trace.push(stage.id);
            }

            pub fn executePrefillLayerRange(
                stage: anytype,
                _: usize,
                tokens: []const u32,
                _: usize,
                layer_start: usize,
                layer_end: usize,
                use_preloaded_input: bool,
                _: bool,
                _: ?[]f32,
                _: ?[]f32,
            ) !void {
                if (tokens.len == 0 or layer_end <= layer_start) return error.InvalidArgument;
                stage.last_use_preloaded_input = use_preloaded_input;
                stage.runtime_buffers.input_dev.bytes[0] = stage.outputByte();
                stage.trace.push(stage.id);
            }
        };

        pub const transport_endpoint = struct {
            pub const supports_transport_endpoint_descriptors = true;
            pub const CudaExternalActivation = transport.CudaBufferDescriptor;

            pub fn deviceLocationHint(stage: anytype) !pipeline.TensorFramePayloadLocationHint {
                return .{ .cuda = std.math.cast(u16, stage.device.ordinal()) orelse return error.InvalidTopologyConfig };
            }

            pub fn decodeExternalOutput(stage: anytype, slot_index: usize, byte_count: usize) !CudaExternalActivation {
                return transport.cudaActivationBufferDescriptor(@TypeOf(stage.*), stage, slot_index, byte_count);
            }

            pub fn prefillExternalOutput(stage: anytype, byte_count: usize) !CudaExternalActivation {
                return decodeExternalOutput(stage, 0, byte_count);
            }

            pub fn decodeExternalInput(stage: anytype, slot_index: usize, byte_count: usize) !CudaExternalActivation {
                return transport.cudaActivationBufferDescriptor(@TypeOf(stage.*), stage, slot_index, byte_count);
            }

            pub fn prefillExternalInput(stage: anytype, byte_count: usize) !CudaExternalActivation {
                return decodeExternalInput(stage, 0, byte_count);
            }

            pub fn sideExternalInput(_: anytype, _: usize) !CudaExternalActivation {
                return error.UnsupportedContentType;
            }
        };
    };
};

fn stageHandle(
    stage_id: usize,
    kind: pipeline.HostBackendKind,
    ptr: *MockStage,
    vtable: *const pipeline.LocalPipelineStageVTable,
) pipeline.LocalPipelineStageHandle {
    return .{
        .stage_id = stage_id,
        .backend_kind = kind,
        .layer_start = stage_id,
        .layer_end = stage_id + 1,
        .supported_boundary_dtypes = &.{.f32},
        .ptr = ptr,
        .vtable = vtable,
    };
}

fn executeDecode(vtable: *const pipeline.LocalPipelineStageVTable, stage: *MockStage, use_preloaded_input: bool) !void {
    try vtable.execute_decode(stage, .{
        .token = 1,
        .position = 0,
        .slot_index = 0,
        .logits_out_opt = null,
        .layer_start = stage.id,
        .layer_end = stage.id + 1,
        .compute_logits = false,
        .download_logits = false,
        .ensure_kv_capacity = true,
        .use_preloaded_input = use_preloaded_input,
    });
}

fn moveExternalActivation(
    source_vtable: *const pipeline.LocalPipelineStageVTable,
    source: *MockStage,
    target_vtable: *const pipeline.LocalPipelineStageVTable,
    target: *MockStage,
) !void {
    const output = try source_vtable.decode_external_output(source, 0, 1);
    const input = try target_vtable.decode_external_input(target, 0, 1);
    if (transport.canPeerCopyExternalActivation(output, input)) {
        try transport.peerCopyExternalActivation(output, input, 1);
        return;
    }
    var staging = [_]u8{0};
    try transport.downloadExternalActivation(output, staging[0..], staging.len);
    try transport.uploadExternalActivation(input, staging[0..], staging.len);
}

fn expectPipelineComposesExternalSurfaces(
    comptime SourceModule: type,
    comptime TargetModule: type,
    expected_target_byte: *const fn (*const MockStage) u8,
) !void {
    var trace = MockTrace{};
    var source = MockStage{ .id = 0, .trace = &trace, .device = .{ .ordinal_value = 0 } };
    var target = MockStage{ .id = 1, .trace = &trace, .device = .{ .ordinal_value = 1 } };
    const source_vtable = pipeline.local_pipeline_stage_adapter.stageVTable(SourceModule, MockStage);
    const target_vtable = pipeline.local_pipeline_stage_adapter.stageVTable(TargetModule, MockStage);

    try executeDecode(source_vtable, &source, false);
    try moveExternalActivation(source_vtable, &source, target_vtable, &target);
    try std.testing.expectEqual(@as(u8, 1), expected_target_byte(&target));
    try executeDecode(target_vtable, &target, true);

    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, trace.executed[0..trace.executed_count]);
    try std.testing.expect(!source.last_use_preloaded_input);
    try std.testing.expect(target.last_use_preloaded_input);
}

fn hostByte(stage: *const MockStage) u8 {
    return stage.host_bytes[0];
}

fn deviceByte(stage: *const MockStage) u8 {
    return stage.runtime_buffers.input_dev.bytes[0];
}

test "pipeline adapter composes independent external activation surfaces" {
    try expectPipelineComposesExternalSurfaces(HostResidentModule, HostResidentModule, hostByte);
    try expectPipelineComposesExternalSurfaces(HostResidentModule, DeviceResidentModule, deviceByte);
    try expectPipelineComposesExternalSurfaces(DeviceResidentModule, HostResidentModule, hostByte);
    try expectPipelineComposesExternalSurfaces(DeviceResidentModule, DeviceResidentModule, deviceByte);
}

test "pipeline adapter probes peer copy without backend pair logic" {
    var trace = MockTrace{};
    var source = MockStage{ .id = 0, .trace = &trace, .device = .{ .ordinal_value = 0 } };
    var target = MockStage{ .id = 1, .trace = &trace, .device = .{ .ordinal_value = 1 } };
    const vtable = pipeline.local_pipeline_stage_adapter.stageVTable(DeviceResidentModule, MockStage);
    var source_handle = stageHandle(0, .cuda, &source, vtable);
    var target_handle = stageHandle(1, .cuda, &target, vtable);

    source.runtime_buffers.input_dev.bytes[0] = 99;
    try std.testing.expect(pipeline.local_pipeline_stage_adapter.canPeerCopy(&source_handle, &target_handle));
    try std.testing.expectEqual(@as(u8, 99), target.runtime_buffers.input_dev.bytes[0]);
}
