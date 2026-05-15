//! CUDA activation transport tests.

const std = @import("std");
const main = @import("main");

const cuda_activation = main.inference.transport.cuda_activation;

const MockBuffer = struct {
    id: u8,
};

const MockEvent = struct {};

const MockCounters = struct {
    enable_peer_calls: usize = 0,
    copy_calls: usize = 0,
    record_event_calls: usize = 0,
    wait_event_calls: usize = 0,
    make_current_calls: usize = 0,
    synchronize_stream_calls: usize = 0,
    synchronize_calls: usize = 0,
};

const MockDevice = struct {
    can_access: bool = true,
    counters: *MockCounters,

    pub fn canAccessPeer(self: *MockDevice, _: *MockDevice) bool {
        return self.can_access;
    }

    pub fn enablePeerAccess(self: *MockDevice, _: *MockDevice) !void {
        self.counters.enable_peer_calls += 1;
    }

    pub fn memcpyPeerAsync(
        self: *MockDevice,
        _: *MockDevice,
        _: *MockBuffer,
        _: *MockBuffer,
        byte_count: usize,
        _: ?usize,
    ) !void {
        try std.testing.expect(byte_count > 0);
        self.counters.copy_calls += 1;
    }

    pub fn recordEvent(self: *MockDevice, _: *MockEvent, _: ?usize) !void {
        self.counters.record_event_calls += 1;
    }

    pub fn streamWaitEvent(self: *MockDevice, _: ?usize, _: *MockEvent) !void {
        self.counters.wait_event_calls += 1;
    }

    pub fn makeCurrent(self: *MockDevice) !void {
        self.counters.make_current_calls += 1;
    }

    pub fn synchronizeStream(self: *MockDevice, _: usize) !void {
        self.counters.synchronize_stream_calls += 1;
    }

    pub fn synchronize(self: *MockDevice) !void {
        self.counters.synchronize_calls += 1;
    }
};

const MockSlotBackend = struct {
    slots: [2][8]u8 = .{
        [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 },
        [_]u8{0} ** 8,
    },

    pub fn slotActivationBytes(self: *@This(), slot_index: usize) []const u8 {
        return self.slots[slot_index][0..];
    }

    pub fn slotActivationBytesMut(self: *@This(), slot_index: usize) []u8 {
        return self.slots[slot_index][0..];
    }
};

const MockCudaActivationTrace = struct {
    upload_calls: usize = 0,
    download_calls: usize = 0,
    synchronize_stream_calls: usize = 0,
    synchronize_calls: usize = 0,
    upload_pointers: [4]u64 = [_]u64{0} ** 4,
    upload_sizes: [4]usize = [_]usize{0} ** 4,
    upload_lengths: [4]usize = [_]usize{0} ** 4,
    download_length: usize = 0,
};

const MockCudaActivationDevice = struct {
    trace: *MockCudaActivationTrace,

    pub fn synchronizeStream(self: *@This(), _: usize) !void {
        self.trace.synchronize_stream_calls += 1;
    }

    pub fn synchronize(self: *@This()) !void {
        self.trace.synchronize_calls += 1;
    }
};

const MockCudaDeviceBuffer = struct {
    pointer: u64,
    size: usize,
    trace: ?*MockCudaActivationTrace = null,

    pub fn upload(self: *const @This(), _: *MockCudaActivationDevice, data: []const u8) !void {
        const trace = self.trace orelse return error.TestUnexpectedResult;
        const index = trace.upload_calls;
        trace.upload_calls += 1;
        trace.upload_pointers[index] = self.pointer;
        trace.upload_sizes[index] = self.size;
        trace.upload_lengths[index] = data.len;
    }

    pub fn download(self: *const @This(), _: *MockCudaActivationDevice, data: []u8) !void {
        const trace = self.trace orelse return error.TestUnexpectedResult;
        trace.download_calls += 1;
        trace.download_length = data.len;
    }
};

const MockCudaRuntimeBuffers = struct {
    input_dev: MockCudaDeviceBuffer,
};

const MockCudaActivationBackend = struct {
    device: MockCudaActivationDevice,
    compute_stream: ?usize = null,
    runtime_buffers: MockCudaRuntimeBuffers,
};

const MockPeerRuntimeBuffers = struct {
    input_dev: MockBuffer,
};

const MockPeerBackend = struct {
    device: MockDevice,
    compute_stream: ?usize = null,
    local_stage_peer_copy_event: ?*MockEvent = null,
    runtime_buffers: MockPeerRuntimeBuffers,
};

test "noop activation stage validates empty bridge input" {
    var stage = cuda_activation.NoopActivationStage{};
    const non_empty = [_]u8{1};
    var host: [1]u8 = undefined;

    try stage.executeLayers(&.{}, 0, 0);
    try std.testing.expectError(error.InvalidArgument, stage.executeLayers(non_empty[0..], 0, 0));
    try stage.synchronize();
    try std.testing.expectError(error.InvalidTopologyConfig, stage.downloadActivation(host[0..], host.len));
    try std.testing.expectError(error.InvalidTopologyConfig, stage.uploadActivation(host[0..], host.len));
}

test "host slot activation transport copies bounded bytes" {
    var backend = MockSlotBackend{};
    var host: [4]u8 = undefined;

    try cuda_activation.downloadHostSlotActivation(&backend, 0, host[0..], host.len);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4 }, host[0..]);

    const update = [_]u8{ 9, 8, 7, 6 };
    try cuda_activation.uploadHostSlotActivation(&backend, 1, update[0..], update.len);
    try std.testing.expectEqualSlices(u8, update[0..], backend.slots[1][0..4]);
    try std.testing.expectError(error.InvalidArgument, cuda_activation.downloadHostSlotActivation(&backend, 0, host[0..], 5));
}

test "cuda activation transport uploads downloads and synchronizes backend" {
    var trace = MockCudaActivationTrace{};
    var backend = MockCudaActivationBackend{
        .device = .{ .trace = &trace },
        .compute_stream = 11,
        .runtime_buffers = .{ .input_dev = .{ .pointer = 100, .size = 16, .trace = &trace } },
    };
    var host: [8]u8 = [_]u8{0} ** 8;

    try cuda_activation.uploadCudaActivation(&backend, 0, host[0..], 6);
    try cuda_activation.downloadCudaActivation(&backend, host[0..], 5);
    try cuda_activation.synchronizeCudaActivationBackend(&backend);

    try std.testing.expectEqual(@as(usize, 1), trace.upload_calls);
    try std.testing.expectEqual(@as(usize, 6), trace.upload_lengths[0]);
    try std.testing.expectEqual(@as(usize, 1), trace.download_calls);
    try std.testing.expectEqual(@as(usize, 5), trace.download_length);
    try std.testing.expectEqual(@as(usize, 1), trace.synchronize_stream_calls);
    try std.testing.expectEqual(@as(usize, 0), trace.synchronize_calls);
}

test "cuda activation stage exposes local bridge transport methods" {
    var trace = MockCudaActivationTrace{};
    var backend = MockCudaActivationBackend{
        .device = .{ .trace = &trace },
        .compute_stream = 11,
        .runtime_buffers = .{ .input_dev = .{ .pointer = 100, .size = 16, .trace = &trace } },
    };
    const Stage = cuda_activation.CudaActivationStage(*MockCudaActivationBackend);
    var stage = Stage{ .backend = &backend };
    var host: [8]u8 = [_]u8{0} ** 8;

    try stage.executeLayers(&.{}, 0, 0);
    try stage.uploadActivation(host[0..], 4);
    try stage.downloadActivation(host[0..], 3);
    try stage.synchronize();

    try std.testing.expectEqual(@as(usize, 1), trace.upload_calls);
    try std.testing.expectEqual(@as(usize, 4), trace.upload_lengths[0]);
    try std.testing.expectEqual(@as(usize, 1), trace.download_calls);
    try std.testing.expectEqual(@as(usize, 3), trace.download_length);
    try std.testing.expectEqual(@as(usize, 1), trace.synchronize_stream_calls);
}

test "cuda activation segment upload slices contiguous target buffer" {
    var trace = MockCudaActivationTrace{};
    var backend = MockCudaActivationBackend{
        .device = .{ .trace = &trace },
        .runtime_buffers = .{ .input_dev = .{ .pointer = 100, .size = 16, .trace = &trace } },
    };
    const first = [_]u8{ 1, 2, 3 };
    const second = [_]u8{ 4, 5 };
    const segments = [_][]const u8{ first[0..], second[0..] };

    try cuda_activation.uploadCudaActivationSegments(&backend, segments[0..], 5);

    try std.testing.expectEqual(@as(usize, 2), trace.upload_calls);
    try std.testing.expectEqual(@as(u64, 100), trace.upload_pointers[0]);
    try std.testing.expectEqual(@as(usize, 3), trace.upload_sizes[0]);
    try std.testing.expectEqual(@as(u64, 103), trace.upload_pointers[1]);
    try std.testing.expectEqual(@as(usize, 2), trace.upload_sizes[1]);
    try std.testing.expectError(error.InvalidArgument, cuda_activation.uploadCudaActivationSegments(&backend, segments[0..], 4));
}

test "cuda activation stage uploads segmented bridge payloads" {
    var trace = MockCudaActivationTrace{};
    var backend = MockCudaActivationBackend{
        .device = .{ .trace = &trace },
        .runtime_buffers = .{ .input_dev = .{ .pointer = 100, .size = 16, .trace = &trace } },
    };
    const Stage = cuda_activation.CudaActivationStage(*MockCudaActivationBackend);
    var stage = Stage{ .backend = &backend };
    const first = [_]u8{ 1, 2, 3 };
    const second = [_]u8{ 4, 5 };
    const segments = [_][]const u8{ first[0..], second[0..] };

    try stage.uploadActivationSegments(segments[0..], 5);

    try std.testing.expectEqual(@as(usize, 2), trace.upload_calls);
    try std.testing.expectEqual(@as(u64, 100), trace.upload_pointers[0]);
    try std.testing.expectEqual(@as(u64, 103), trace.upload_pointers[1]);
}

test "copyCudaPeerActivation copies on source stream and synchronizes" {
    var source_counters = MockCounters{};
    var target_counters = MockCounters{};
    var source = MockDevice{ .counters = &source_counters };
    var target = MockDevice{ .counters = &target_counters };
    var source_buffer = MockBuffer{ .id = 1 };
    var target_buffer = MockBuffer{ .id = 2 };

    try cuda_activation.copyCudaPeerActivation(&source, &target, &target_buffer, &source_buffer, @as(?usize, 7), 128);

    try std.testing.expectEqual(@as(usize, 1), source_counters.copy_calls);
    try std.testing.expectEqual(@as(usize, 1), source_counters.synchronize_stream_calls);
    try std.testing.expectEqual(@as(usize, 1), source_counters.enable_peer_calls);
    try std.testing.expectEqual(@as(usize, 1), target_counters.enable_peer_calls);
}

test "cuda peer activation stage issues event ordered peer copy" {
    var source_counters = MockCounters{};
    var target_counters = MockCounters{};
    var event = MockEvent{};
    var source_backend = MockPeerBackend{
        .device = .{ .counters = &source_counters },
        .compute_stream = 3,
        .local_stage_peer_copy_event = &event,
        .runtime_buffers = .{ .input_dev = .{ .id = 1 } },
    };
    var target_backend = MockPeerBackend{
        .device = .{ .counters = &target_counters },
        .compute_stream = 5,
        .runtime_buffers = .{ .input_dev = .{ .id = 2 } },
    };
    const SourceStage = cuda_activation.CudaPeerActivationStage(
        *MockPeerBackend,
        *MockPeerBackend,
        .source_event_target_stream,
    );
    const TargetStage = cuda_activation.CudaActivationStage(*MockPeerBackend);
    var source_stage = SourceStage{ .backend = &source_backend, .target_backend = &target_backend };
    var target_stage = TargetStage{ .backend = &target_backend };

    try std.testing.expect(source_stage.peerCopyHandlesStageSync());
    try source_stage.peerCopyActivationToErased(&target_stage, 64);

    try std.testing.expectEqual(@as(usize, 1), source_counters.record_event_calls);
    try std.testing.expectEqual(@as(usize, 1), target_counters.wait_event_calls);
    try std.testing.expectEqual(@as(usize, 1), source_counters.copy_calls);
    try std.testing.expectEqual(@as(usize, 0), source_counters.synchronize_stream_calls);
}

test "copyCudaPeerActivation rejects missing peer access" {
    var source_counters = MockCounters{};
    var target_counters = MockCounters{};
    var source = MockDevice{ .counters = &source_counters };
    var target = MockDevice{ .can_access = false, .counters = &target_counters };
    var source_buffer = MockBuffer{ .id = 1 };
    var target_buffer = MockBuffer{ .id = 2 };

    try std.testing.expectError(
        error.InvalidTopologyConfig,
        cuda_activation.copyCudaPeerActivation(&source, &target, &target_buffer, &source_buffer, @as(?usize, null), 128),
    );
    try std.testing.expectEqual(@as(usize, 0), source_counters.copy_calls);
}

test "copyCudaPeerActivationAfterEvent waits on target stream" {
    var source_counters = MockCounters{};
    var target_counters = MockCounters{};
    var source = MockDevice{ .counters = &source_counters };
    var target = MockDevice{ .counters = &target_counters };
    var source_buffer = MockBuffer{ .id = 1 };
    var target_buffer = MockBuffer{ .id = 2 };
    var event = MockEvent{};

    try cuda_activation.copyCudaPeerActivationAfterEvent(
        &source,
        &target,
        &target_buffer,
        &source_buffer,
        @as(?usize, 3),
        @as(?usize, 5),
        &event,
        128,
    );

    try std.testing.expectEqual(@as(usize, 1), source_counters.record_event_calls);
    try std.testing.expectEqual(@as(usize, 1), target_counters.wait_event_calls);
    try std.testing.expectEqual(@as(usize, 1), target_counters.make_current_calls);
    try std.testing.expectEqual(@as(usize, 1), source_counters.copy_calls);
    try std.testing.expectEqual(@as(usize, 0), source_counters.synchronize_stream_calls);
}
