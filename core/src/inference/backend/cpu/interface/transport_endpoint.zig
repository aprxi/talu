//! CPU implementation of the backend transport endpoint descriptor interface.

const bridge = @import("../../../bridge/root.zig");

pub const supports_transport_endpoint_descriptors = true;

pub const HostActivationSlice = struct {
    bytes: []const u8,
};

pub const HostActivationTarget = struct {
    bytes: []u8,
};

pub fn deviceLocationHint(_: anytype) !bridge.TensorFramePayloadLocationHint {
    return .{ .cpu = {} };
}

pub fn hostDecodeActivationSlice(backend: anytype, slot_index: usize, byte_count: usize) !HostActivationSlice {
    const bytes = backend.slotActivationBytes(slot_index);
    if (byte_count > bytes.len) return error.InvalidArgument;
    return .{ .bytes = bytes[0..byte_count] };
}

pub fn hostPrefillActivationSlice(backend: anytype, byte_count: usize) !HostActivationSlice {
    const bytes = backend.localPrefillActivationBytes(byte_count);
    if (byte_count > bytes.len) return error.InvalidArgument;
    return .{ .bytes = bytes[0..byte_count] };
}

pub fn decodeInputBuffer(backend: anytype, slot_index: usize, byte_count: usize) !HostActivationTarget {
    const target = backend.slotActivationBytesMut(slot_index);
    if (byte_count > target.len) return error.InvalidArgument;
    return .{ .bytes = target[0..byte_count] };
}

pub fn prefillInputBuffer(backend: anytype, byte_count: usize) !HostActivationTarget {
    const target = backend.localPrefillActivationBytesMut(byte_count);
    if (byte_count > target.len) return error.InvalidArgument;
    return .{ .bytes = target[0..byte_count] };
}

pub fn sideInputBuffer(_: anytype, _: usize) !HostActivationTarget {
    return error.UnsupportedContentType;
}
