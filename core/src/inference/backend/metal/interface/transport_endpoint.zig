//! Metal transport endpoint interface placeholder.

const bridge = @import("../../../bridge/root.zig");

pub const supports_transport_endpoint_descriptors = false;

pub const HostActivationSlice = struct {
    bytes: []const u8,
};

pub const HostActivationTarget = struct {
    bytes: []u8,
};

pub fn deviceLocationHint(_: anytype) !bridge.TensorFramePayloadLocationHint {
    return error.UnsupportedBackend;
}

pub fn hostDecodeActivationSlice(_: anytype, _: usize, _: usize) !HostActivationSlice {
    return error.UnsupportedBackend;
}

pub fn hostPrefillActivationSlice(_: anytype, _: usize) !HostActivationSlice {
    return error.UnsupportedBackend;
}

pub fn decodeInputBuffer(_: anytype, _: usize, _: usize) !HostActivationTarget {
    return error.UnsupportedBackend;
}

pub fn prefillInputBuffer(_: anytype, _: usize) !HostActivationTarget {
    return error.UnsupportedBackend;
}

pub fn sideInputBuffer(_: anytype, _: usize) !HostActivationTarget {
    return error.UnsupportedBackend;
}
