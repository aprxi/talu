//! CUDA activation transport primitives used by local stage handoff adapters.
//!
//! This module owns the device-to-device copy mechanics. Backend adapters are
//! responsible for extracting concrete devices, buffers, streams, and events.

const std = @import("std");

pub const CudaPeerCopySynchronization = enum {
    source_stream,
    source_event_target_stream,
};

fn validateEmptyStageInput(input: []const u8) !void {
    if (input.len != 0) return error.InvalidArgument;
}

pub const NoopActivationStage = struct {
    pub fn executeLayers(_: *@This(), input: []const u8, _: usize, _: usize) anyerror!void {
        try validateEmptyStageInput(input);
    }

    pub fn synchronize(_: *@This()) anyerror!void {}

    pub fn downloadActivation(_: *@This(), _: []u8, _: usize) anyerror!void {
        return error.InvalidTopologyConfig;
    }

    pub fn uploadActivation(_: *@This(), _: []const u8, _: usize) anyerror!void {
        return error.InvalidTopologyConfig;
    }
};

pub fn CudaActivationStage(comptime Backend: type) type {
    return struct {
        backend: Backend,
        slot_index: usize = 0,

        pub fn executeLayers(_: *@This(), input: []const u8, _: usize, _: usize) anyerror!void {
            try validateEmptyStageInput(input);
        }

        pub fn synchronize(stage: *@This()) anyerror!void {
            try synchronizeCudaActivationBackend(stage.backend);
        }

        pub fn downloadActivation(stage: *@This(), host_buf: []u8, byte_count: usize) anyerror!void {
            try downloadCudaActivation(stage.backend, host_buf, byte_count);
        }

        pub fn uploadActivation(stage: *@This(), host_buf: []const u8, byte_count: usize) anyerror!void {
            try uploadCudaActivation(stage.backend, stage.slot_index, host_buf, byte_count);
        }

        pub fn uploadActivationSegments(stage: *@This(), segments: []const []const u8, byte_count: usize) anyerror!void {
            try uploadCudaActivationSegments(stage.backend, segments, byte_count);
        }
    };
}

pub fn CudaPeerActivationStage(
    comptime Backend: type,
    comptime TargetBackend: type,
    comptime synchronization: CudaPeerCopySynchronization,
) type {
    return struct {
        backend: Backend,
        target_backend: TargetBackend,
        slot_index: usize = 0,

        pub fn executeLayers(_: *@This(), input: []const u8, _: usize, _: usize) anyerror!void {
            try validateEmptyStageInput(input);
        }

        pub fn synchronize(stage: *@This()) anyerror!void {
            try synchronizeCudaActivationBackend(stage.backend);
        }

        pub fn downloadActivation(stage: *@This(), host_buf: []u8, byte_count: usize) anyerror!void {
            try downloadCudaActivation(stage.backend, host_buf, byte_count);
        }

        pub fn uploadActivation(stage: *@This(), host_buf: []const u8, byte_count: usize) anyerror!void {
            try uploadCudaActivation(stage.backend, stage.slot_index, host_buf, byte_count);
        }

        pub fn peerCopyActivationToErased(stage: *@This(), target_ptr: *anyopaque, byte_count: usize) anyerror!void {
            _ = target_ptr;
            try peerCopyCudaActivation(stage.backend, stage.target_backend, byte_count, synchronization);
        }

        pub fn peerCopyHandlesStageSync(stage: *const @This()) bool {
            return peerCopyCudaActivationHandlesStageSync(stage.backend, synchronization);
        }
    };
}

pub fn downloadHostSlotActivation(
    backend: anytype,
    slot_index: usize,
    host_buffer: []u8,
    byte_count: usize,
) !void {
    if (byte_count > host_buffer.len) return error.InvalidArgument;
    const source = backend.slotActivationBytes(slot_index);
    if (byte_count > source.len) return error.InvalidArgument;
    @memcpy(host_buffer[0..byte_count], source[0..byte_count]);
}

pub fn uploadHostSlotActivation(
    backend: anytype,
    slot_index: usize,
    host_buffer: []const u8,
    byte_count: usize,
) !void {
    if (byte_count > host_buffer.len) return error.InvalidArgument;
    const target = backend.slotActivationBytesMut(slot_index);
    if (byte_count > target.len) return error.InvalidArgument;
    @memcpy(target[0..byte_count], host_buffer[0..byte_count]);
}

pub fn downloadCudaActivation(backend: anytype, host_buffer: []u8, byte_count: usize) !void {
    if (byte_count > host_buffer.len) return error.InvalidArgument;
    const BackendType = @TypeOf(backend.*);
    if (comptime @hasDecl(BackendType, "downloadLocalActivationToHost")) {
        return backend.downloadLocalActivationToHost(host_buffer[0..byte_count], byte_count);
    }
    if (comptime @hasField(BackendType, "runtime_buffers") and @hasField(BackendType, "device")) {
        return backend.runtime_buffers.input_dev.download(&backend.device, host_buffer[0..byte_count]);
    }
    return error.InvalidTopologyConfig;
}

pub fn uploadCudaActivation(backend: anytype, slot_index: usize, host_buffer: []const u8, byte_count: usize) !void {
    if (byte_count > host_buffer.len) return error.InvalidArgument;
    const BackendType = @TypeOf(backend.*);
    if (comptime @hasDecl(BackendType, "uploadLocalActivationFromHost")) {
        return backend.uploadLocalActivationFromHost(slot_index, host_buffer[0..byte_count], byte_count);
    }
    if (comptime @hasField(BackendType, "runtime_buffers") and @hasField(BackendType, "device")) {
        return backend.runtime_buffers.input_dev.upload(&backend.device, host_buffer[0..byte_count]);
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
        var target_slice = try cudaBufferSlice(&backend.runtime_buffers.input_dev, start, segment.len);
        try target_slice.upload(&backend.device, segment);
    }
    if (offset != byte_count) return error.InvalidArgument;
}

pub fn cudaBufferSlice(buffer: anytype, byte_offset: usize, byte_len: usize) !@TypeOf(buffer.*) {
    if (byte_offset > buffer.size) return error.InvalidArgument;
    const end = std.math.add(usize, byte_offset, byte_len) catch return error.InvalidArgument;
    if (end > buffer.size) return error.InvalidArgument;
    const ptr = std.math.add(u64, buffer.pointer, @intCast(byte_offset)) catch return error.InvalidArgument;
    var sliced = buffer.*;
    sliced.pointer = ptr;
    sliced.size = byte_len;
    return sliced;
}

pub fn synchronizeCudaActivationBackend(backend: anytype) !void {
    const BackendType = @TypeOf(backend.*);
    if (comptime @hasDecl(BackendType, "synchronizeLocalActivation")) {
        return backend.synchronizeLocalActivation();
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

pub fn copyCudaPeerActivation(
    source_device: anytype,
    target_device: anytype,
    target_buffer: anytype,
    source_buffer: anytype,
    source_stream: anytype,
    byte_count: usize,
) !void {
    if (byte_count == 0) return;
    if (!target_device.canAccessPeer(source_device)) return error.InvalidTopologyConfig;
    target_device.enablePeerAccess(source_device) catch {};
    source_device.enablePeerAccess(target_device) catch {};
    try source_device.memcpyPeerAsync(
        target_device,
        target_buffer,
        source_buffer,
        byte_count,
        source_stream,
    );
    if (source_stream) |stream| {
        try source_device.synchronizeStream(stream);
    } else {
        try source_device.synchronize();
    }
}

pub fn copyCudaPeerActivationAfterEvent(
    source_device: anytype,
    target_device: anytype,
    target_buffer: anytype,
    source_buffer: anytype,
    source_stream: anytype,
    target_stream: anytype,
    event: anytype,
    byte_count: usize,
) !void {
    if (byte_count == 0) return;
    if (!target_device.canAccessPeer(source_device)) return error.InvalidTopologyConfig;
    target_device.enablePeerAccess(source_device) catch {};
    source_device.enablePeerAccess(target_device) catch {};
    try source_device.recordEvent(event, source_stream);
    try target_device.streamWaitEvent(target_stream, event);
    try target_device.makeCurrent();
    try source_device.memcpyPeerAsync(
        target_device,
        target_buffer,
        source_buffer,
        byte_count,
        target_stream,
    );
}

pub fn peerCopyCudaActivation(
    source_backend: anytype,
    target_backend: anytype,
    byte_count: usize,
    comptime synchronization: CudaPeerCopySynchronization,
) !void {
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

    if (comptime synchronization == .source_event_target_stream) {
        if (comptime !@hasField(TargetType, "compute_stream")) return error.InvalidTopologyConfig;
        if (comptime @hasField(SourceType, "local_stage_peer_copy_event")) {
            if (source_backend.local_stage_peer_copy_event) |event| {
                return copyCudaPeerActivationAfterEvent(
                    &source_backend.device,
                    &target_backend.device,
                    &target_backend.runtime_buffers.input_dev,
                    &source_backend.runtime_buffers.input_dev,
                    source_backend.compute_stream,
                    target_backend.compute_stream,
                    event,
                    byte_count,
                );
            }
        }
    }

    try copyCudaPeerActivation(
        &source_backend.device,
        &target_backend.device,
        &target_backend.runtime_buffers.input_dev,
        &source_backend.runtime_buffers.input_dev,
        source_backend.compute_stream,
        byte_count,
    );
}

pub fn peerCopyCudaActivationHandlesStageSync(
    source_backend: anytype,
    comptime synchronization: CudaPeerCopySynchronization,
) bool {
    if (comptime synchronization == .source_event_target_stream) {
        const SourceType = @TypeOf(source_backend.*);
        if (comptime @hasField(SourceType, "local_stage_peer_copy_event")) {
            return source_backend.local_stage_peer_copy_event != null;
        }
    }
    return false;
}
