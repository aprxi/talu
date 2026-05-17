//! CUDA activation transport primitives used by local stage handoff adapters.
//!
//! This module owns the device-to-device copy mechanics. Backend adapters are
//! responsible for extracting concrete devices, buffers, streams, and events.

const std = @import("std");
const tensor_frame = @import("../bridge/tensor_frame.zig");
const cuda_kv_mirror = @import("cuda_kv_mirror.zig");
const local_stage = @import("local_stage.zig");

pub const CudaPeerCopySynchronization = enum {
    source_stream,
    source_event_target_stream,
};

pub const LocalEndpointTransportOptions = struct {
    pub const CpuActivationScope = enum {
        decode_slot,
        prefill_stage,
    };

    has_cpu_stage: bool = false,
    allow_cpu_decode_download: bool = false,
    prepare_cpu_boundary: bool = false,
    cpu_activation_scope: CpuActivationScope = .decode_slot,
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

pub fn uploadHostPrefillActivation(
    backend: anytype,
    host_buffer: []const u8,
    byte_count: usize,
) !void {
    if (byte_count > host_buffer.len) return error.InvalidArgument;
    const BackendType = @TypeOf(backend.*);
    if (comptime !@hasDecl(BackendType, "localPrefillActivationBytesMut")) return error.InvalidTopologyConfig;
    const target = backend.localPrefillActivationBytesMut(byte_count);
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
        try ensureCudaActivationUploadCapacity(backend, byte_count);
        return backend.runtime_buffers.input_dev.upload(&backend.device, host_buffer[0..byte_count]);
    }
    return error.InvalidTopologyConfig;
}

pub fn uploadCudaActivationSegments(backend: anytype, host_segments: []const []const u8, byte_count: usize) !void {
    const BackendType = @TypeOf(backend.*);
    if (comptime !@hasField(BackendType, "runtime_buffers") or !@hasField(BackendType, "device")) {
        return error.InvalidTopologyConfig;
    }
    try ensureCudaActivationUploadCapacity(backend, byte_count);
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

fn ensureCudaActivationUploadCapacity(backend: anytype, byte_count: usize) !void {
    if (byte_count == 0) return;
    if (byte_count <= backend.runtime_buffers.input_dev.size) return;
    const BackendType = @TypeOf(backend.*);
    if (comptime !@hasField(BackendType, "d_model")) return error.InvalidTopologyConfig;
    const row_bytes = std.math.mul(usize, backend.d_model, @sizeOf(f32)) catch return error.InvalidArgument;
    if (row_bytes == 0 or byte_count % row_bytes != 0) return error.InvalidArgument;
    const required_rows = byte_count / row_bytes;
    const fixed_alloc_mode = if (comptime @hasField(BackendType, "fixed_alloc_mode"))
        backend.fixed_alloc_mode
    else
        false;
    try backend.runtime_buffers.ensureRowCapacity(&backend.device, required_rows, fixed_alloc_mode);
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

pub fn uploadCudaBufferFromHostBytes(
    device: anytype,
    buffer: anytype,
    byte_offset: usize,
    host_bytes: []const u8,
) !@TypeOf(buffer.*) {
    var target = try cudaBufferSlice(buffer, byte_offset, host_bytes.len);
    try target.upload(device, host_bytes);
    return target;
}

pub fn downloadCudaBufferToHostBytes(
    device: anytype,
    buffer: anytype,
    host_bytes: []u8,
) !void {
    if (host_bytes.len > buffer.size) return error.InvalidArgument;
    try buffer.download(device, host_bytes);
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

pub fn probeCudaPeerActivation(source_backend: anytype, target_backend: anytype) bool {
    const SourceType = @TypeOf(source_backend.*);
    const TargetType = @TypeOf(target_backend.*);
    if (comptime !@hasField(SourceType, "device") or
        !@hasField(SourceType, "runtime_buffers") or
        !@hasField(SourceType, "compute_stream") or
        !@hasField(TargetType, "device") or
        !@hasField(TargetType, "runtime_buffers"))
    {
        return false;
    }
    const src_size = source_backend.runtime_buffers.input_dev.size;
    const dst_size = target_backend.runtime_buffers.input_dev.size;
    const min_size = @min(src_size, dst_size);
    if (min_size == 0) return false;
    const probe_bytes = @min(min_size, @as(usize, 256));
    copyCudaPeerActivation(
        &source_backend.device,
        &target_backend.device,
        &target_backend.runtime_buffers.input_dev,
        &source_backend.runtime_buffers.input_dev,
        source_backend.compute_stream,
        probe_bytes,
    ) catch return false;
    return true;
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
    _ = source_backend;
    _ = synchronization;
    return false;
}

pub fn peerCopyCudaActivationRuntime(
    source_backend: anytype,
    target_backend: anytype,
    byte_count: usize,
    synchronization: CudaPeerCopySynchronization,
) !void {
    switch (synchronization) {
        .source_stream => try peerCopyCudaActivation(source_backend, target_backend, byte_count, .source_stream),
        .source_event_target_stream => try peerCopyCudaActivation(source_backend, target_backend, byte_count, .source_event_target_stream),
    }
}

pub fn peerCopyCudaActivationHandlesStageSyncRuntime(
    source_backend: anytype,
    synchronization: CudaPeerCopySynchronization,
) bool {
    return switch (synchronization) {
        .source_stream => peerCopyCudaActivationHandlesStageSync(source_backend, .source_stream),
        .source_event_target_stream => peerCopyCudaActivationHandlesStageSync(source_backend, .source_event_target_stream),
    };
}

pub fn synchronizeLocalEndpoint(stage: anytype) !void {
    switch (stage.kind) {
        .cpu => {},
        .cuda => try synchronizeCudaActivationBackend(stage.cuda_backend orelse return error.InvalidTopologyConfig),
    }
}

pub fn prepareCpuBoundaryTransferToCudaEndpoint(
    stage: anytype,
    target_ptr: *anyopaque,
    metadata: *const tensor_frame.TensorFrameMetadata,
    comptime Endpoint: type,
    comptime has_cpu_stage: bool,
) !void {
    if (stage.kind != .cpu) return;
    if (comptime !has_cpu_stage) return error.InvalidTopologyConfig;
    const cpu = stage.cpu_backend orelse return error.InvalidTopologyConfig;
    const target: *Endpoint = @ptrCast(@alignCast(target_ptr));
    switch (target.kind) {
        .cuda => {
            const cuda_backend = target.cuda_backend orelse return error.InvalidTopologyConfig;
            for (metadata.batch.entries) |entry| {
                const slot_index = std.math.cast(usize, entry.slot_id) orelse return error.InvalidArgument;
                const position = std.math.cast(usize, entry.sequence_start) orelse return error.InvalidArgument;
                const token_count = std.math.cast(usize, entry.token_count) orelse return error.InvalidArgument;
                try cuda_kv_mirror.uploadCpuKvToCudaMirrors(cuda_backend, cpu, slot_index, position, token_count);
            }
        },
        .cpu => return error.InvalidTopologyConfig,
    }
}

pub fn downloadLocalDecodeEndpointActivation(
    stage: anytype,
    host_buf: []u8,
    byte_count: usize,
    comptime has_cpu_stage: bool,
) !void {
    switch (stage.kind) {
        .cpu => {
            if (comptime !has_cpu_stage) return error.InvalidTopologyConfig;
            try downloadHostSlotActivation(
                stage.cpu_backend orelse return error.InvalidTopologyConfig,
                stage.activation_slot_index,
                host_buf,
                byte_count,
            );
        },
        .cuda => try downloadCudaActivation(stage.cuda_backend orelse return error.InvalidTopologyConfig, host_buf, byte_count),
    }
}

pub fn downloadLocalDeviceEndpointActivation(
    stage: anytype,
    host_buf: []u8,
    byte_count: usize,
) !void {
    switch (stage.kind) {
        .cpu => return error.InvalidTopologyConfig,
        .cuda => try downloadCudaActivation(stage.cuda_backend orelse return error.InvalidTopologyConfig, host_buf, byte_count),
    }
}

pub fn uploadLocalEndpointActivation(
    stage: anytype,
    host_buf: []const u8,
    byte_count: usize,
    comptime has_cpu_stage: bool,
    comptime cpu_activation_scope: LocalEndpointTransportOptions.CpuActivationScope,
) !void {
    switch (stage.kind) {
        .cpu => {
            if (comptime !has_cpu_stage) return error.InvalidTopologyConfig;
            switch (comptime cpu_activation_scope) {
                .decode_slot => try uploadHostSlotActivation(stage.cpu_backend orelse return error.InvalidTopologyConfig, stage.activation_slot_index, host_buf, byte_count),
                .prefill_stage => try uploadHostPrefillActivation(stage.cpu_backend orelse return error.InvalidTopologyConfig, host_buf, byte_count),
            }
        },
        .cuda => try uploadCudaActivation(stage.cuda_backend orelse return error.InvalidTopologyConfig, stage.activation_slot_index, host_buf, byte_count),
    }
}

pub fn uploadLocalEndpointActivationSegments(stage: anytype, segments: []const []const u8, byte_count: usize) !void {
    switch (stage.kind) {
        .cpu => return error.LocalStageTransportSegmentedUploadUnsupported,
        .cuda => try uploadCudaActivationSegments(stage.cuda_backend orelse return error.InvalidTopologyConfig, segments, byte_count),
    }
}

pub fn peerCopyLocalEndpointActivationTo(
    stage: anytype,
    target_ptr: *anyopaque,
    byte_count: usize,
    comptime Endpoint: type,
) !void {
    if (stage.kind == .cpu) return error.LocalStageTransportPeerCopyUnsupported;
    const target: *Endpoint = @ptrCast(@alignCast(target_ptr));
    if (target.kind == .cpu) return error.LocalStageTransportPeerCopyUnsupported;
    try peerCopyCudaActivationRuntime(
        stage.cuda_backend orelse return error.InvalidTopologyConfig,
        target.cuda_backend orelse return error.InvalidTopologyConfig,
        byte_count,
        stage.peer_copy_synchronization,
    );
}

pub fn localEndpointPeerCopyHandlesStageSync(stage: anytype) bool {
    return switch (stage.kind) {
        .cpu => false,
        .cuda => peerCopyCudaActivationHandlesStageSyncRuntime(stage.cuda_backend orelse return false, stage.peer_copy_synchronization),
    };
}

pub fn localEndpointTransportAdapter(
    comptime Endpoint: type,
    stage_id: usize,
    endpoint: *Endpoint,
    comptime options: LocalEndpointTransportOptions,
) local_stage.LocalStageTransportEndpoint {
    const Adapter = struct {
        fn endpointPtr(ptr: *anyopaque) *Endpoint {
            return @ptrCast(@alignCast(ptr));
        }

        fn synchronize(ptr: *anyopaque, receipt: local_stage.StageExecutionReceipt) anyerror!void {
            _ = receipt;
            try synchronizeLocalEndpoint(endpointPtr(ptr));
        }

        fn prepareBoundaryTransferTo(
            ptr: *anyopaque,
            target_ptr: *anyopaque,
            metadata: *const tensor_frame.TensorFrameMetadata,
        ) anyerror!void {
            if (comptime !options.prepare_cpu_boundary) return;
            try prepareCpuBoundaryTransferToCudaEndpoint(
                endpointPtr(ptr),
                target_ptr,
                metadata,
                Endpoint,
                options.has_cpu_stage,
            );
        }

        fn downloadActivation(ptr: *anyopaque, host_buf: []u8, byte_count: usize) anyerror!void {
            if (comptime options.allow_cpu_decode_download) {
                try downloadLocalDecodeEndpointActivation(
                    endpointPtr(ptr),
                    host_buf,
                    byte_count,
                    options.has_cpu_stage,
                );
            } else {
                try downloadLocalDeviceEndpointActivation(
                    endpointPtr(ptr),
                    host_buf,
                    byte_count,
                );
            }
        }

        fn uploadActivation(ptr: *anyopaque, host_buf: []const u8, byte_count: usize) anyerror!void {
            try uploadLocalEndpointActivation(
                endpointPtr(ptr),
                host_buf,
                byte_count,
                options.has_cpu_stage,
                options.cpu_activation_scope,
            );
        }

        fn uploadActivationSegments(ptr: *anyopaque, segments: []const []const u8, byte_count: usize) anyerror!void {
            try uploadLocalEndpointActivationSegments(
                endpointPtr(ptr),
                segments,
                byte_count,
            );
        }

        fn peerCopyActivationTo(ptr: *anyopaque, target_ptr: *anyopaque, byte_count: usize) anyerror!void {
            try peerCopyLocalEndpointActivationTo(
                endpointPtr(ptr),
                target_ptr,
                byte_count,
                Endpoint,
            );
        }

        fn peerCopyHandlesStageSync(ptr: *anyopaque) bool {
            return localEndpointPeerCopyHandlesStageSync(endpointPtr(ptr));
        }

        const vtable = local_stage.LocalStageTransportEndpointVTable{
            .synchronize = synchronize,
            .prepare_boundary_transfer_to = if (options.prepare_cpu_boundary) prepareBoundaryTransferTo else null,
            .download_activation = downloadActivation,
            .upload_activation = uploadActivation,
            .upload_activation_segments = uploadActivationSegments,
            .peer_copy_activation_to = peerCopyActivationTo,
            .peer_copy_handles_stage_sync = peerCopyHandlesStageSync,
        };
    };
    return .{
        .stage_id = stage_id,
        .ptr = endpoint,
        .vtable = &Adapter.vtable,
    };
}
