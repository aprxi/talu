//! CPU implementation of the backend external activation surface interface.
//!
//! These functions expose only CPU-owned activation inputs/outputs. They do not
//! select routes, inspect adjacent stages, or perform handoff copies.

const pipeline = @import("../../../pipeline/root.zig");

pub const supports_transport_endpoint_descriptors = true;

pub const HostActivationOutput = struct {
    bytes: []const u8,
};

pub const HostActivationInput = struct {
    bytes: []u8,
};

pub fn deviceLocationHint(_: anytype) !pipeline.TensorFramePayloadLocationHint {
    return .{ .cpu = {} };
}

pub fn decodeExternalOutput(backend: anytype, slot_index: usize, byte_count: usize) !HostActivationOutput {
    const bytes = backend.slotActivationBytes(slot_index);
    if (byte_count > bytes.len) return error.InvalidArgument;
    return .{ .bytes = bytes[0..byte_count] };
}

pub fn prefillExternalOutput(backend: anytype, byte_count: usize) !HostActivationOutput {
    const BackendType = @TypeOf(backend.*);
    const bytes = if (comptime hasDecl(BackendType, "ensureLocalPrefillActivationBytes"))
        try backend.ensureLocalPrefillActivationBytes(byte_count)
    else
        backend.localPrefillActivationBytes(byte_count);
    if (byte_count > bytes.len) return error.InvalidArgument;
    return .{ .bytes = bytes[0..byte_count] };
}

pub fn decodeExternalInput(backend: anytype, slot_index: usize, byte_count: usize) !HostActivationInput {
    const input = backend.slotActivationBytesMut(slot_index);
    if (byte_count > input.len) return error.InvalidArgument;
    return .{ .bytes = input[0..byte_count] };
}

pub fn prefillExternalInput(backend: anytype, byte_count: usize) !HostActivationInput {
    const input = backend.localPrefillActivationBytesMut(byte_count);
    if (byte_count > input.len) return error.InvalidArgument;
    return .{ .bytes = input[0..byte_count] };
}

pub fn sideExternalInput(_: anytype, _: usize) !HostActivationInput {
    return error.UnsupportedContentType;
}

fn hasDecl(comptime T: type, comptime name: []const u8) bool {
    return switch (@typeInfo(T)) {
        .@"struct", .@"enum", .@"union", .@"opaque" => @hasDecl(T, name),
        else => false,
    };
}
