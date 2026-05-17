//! CUDA implementation of the backend transport endpoint descriptor interface.

const std = @import("std");
const bridge = @import("../../../bridge/root.zig");

pub const supports_transport_endpoint_descriptors = true;

pub const HostActivationSlice = struct {
    bytes: []const u8,
};

pub const CudaBufferDescriptor = struct {
    buffer: *anyopaque,
    device: *anyopaque,
    device_ordinal: u16,
    stream: ?*anyopaque,
    byte_count: usize,
};

pub fn deviceLocationHint(backend: anytype) !bridge.TensorFramePayloadLocationHint {
    const ordinal = backend.device.ordinal();
    return .{ .cuda = std.math.cast(u16, ordinal) orelse return error.InvalidTopologyConfig };
}

pub fn hostDecodeActivationSlice(_: anytype, _: usize, _: usize) !HostActivationSlice {
    return error.InvalidTopologyConfig;
}

pub fn hostPrefillActivationSlice(_: anytype, _: usize) !HostActivationSlice {
    return error.InvalidTopologyConfig;
}

pub fn decodeInputBuffer(backend: anytype, _: usize, byte_count: usize) !CudaBufferDescriptor {
    return cudaInputBuffer(backend, byte_count);
}

pub fn prefillInputBuffer(backend: anytype, byte_count: usize) !CudaBufferDescriptor {
    return cudaInputBuffer(backend, byte_count);
}

pub fn sideInputBuffer(_: anytype, _: usize) !CudaBufferDescriptor {
    return error.UnsupportedContentType;
}

fn cudaInputBuffer(backend: anytype, byte_count: usize) !CudaBufferDescriptor {
    _ = try deviceLocationHint(backend);
    return .{
        .buffer = @ptrCast(&backend.runtime_buffers.input_dev),
        .device = @ptrCast(&backend.device),
        .device_ordinal = std.math.cast(u16, backend.device.ordinal()) orelse return error.InvalidTopologyConfig,
        .stream = backend.compute_stream,
        .byte_count = byte_count,
    };
}
