//! CUDA implementation of the backend external activation surface interface.
//!
//! These functions expose only CUDA-owned activation inputs/outputs. They do not
//! select routes, inspect adjacent stages, or perform handoff copies.

const std = @import("std");
const pipeline = @import("../../../pipeline/root.zig");
const transport = @import("../../../transport/root.zig");

pub const supports_transport_endpoint_descriptors = true;

pub const CudaExternalActivation = transport.CudaBufferDescriptor;

pub fn deviceLocationHint(backend: anytype) !pipeline.TensorFramePayloadLocationHint {
    const ordinal = backend.device.ordinal();
    return .{ .cuda = std.math.cast(u16, ordinal) orelse return error.InvalidTopologyConfig };
}

pub fn decodeExternalOutput(backend: anytype, slot_index: usize, byte_count: usize) !CudaExternalActivation {
    return cudaActivationSurface(backend, slot_index, byte_count);
}

pub fn prefillExternalOutput(backend: anytype, byte_count: usize) !CudaExternalActivation {
    return cudaActivationSurface(backend, 0, byte_count);
}

pub fn decodeExternalInput(backend: anytype, slot_index: usize, byte_count: usize) !CudaExternalActivation {
    return cudaActivationSurface(backend, slot_index, byte_count);
}

pub fn prefillExternalInput(backend: anytype, byte_count: usize) !CudaExternalActivation {
    return cudaActivationSurface(backend, 0, byte_count);
}

pub fn sideExternalInput(_: anytype, _: usize) !CudaExternalActivation {
    return error.UnsupportedContentType;
}

fn cudaActivationSurface(backend: anytype, slot_index: usize, byte_count: usize) !CudaExternalActivation {
    _ = try deviceLocationHint(backend);
    return transport.cudaActivationBufferDescriptor(@TypeOf(backend.*), backend, slot_index, byte_count);
}
