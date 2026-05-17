//! Metal external activation surface interface placeholder.

const pipeline = @import("../../../pipeline/root.zig");

pub const supports_transport_endpoint_descriptors = false;

pub const HostActivationOutput = struct {
    bytes: []const u8,
};

pub const HostActivationInput = struct {
    bytes: []u8,
};

pub fn deviceLocationHint(_: anytype) !pipeline.TensorFramePayloadLocationHint {
    return error.UnsupportedBackend;
}

pub fn decodeExternalOutput(_: anytype, _: usize, _: usize) !HostActivationOutput {
    return error.UnsupportedBackend;
}

pub fn prefillExternalOutput(_: anytype, _: usize) !HostActivationOutput {
    return error.UnsupportedBackend;
}

pub fn decodeExternalInput(_: anytype, _: usize, _: usize) !HostActivationInput {
    return error.UnsupportedBackend;
}

pub fn prefillExternalInput(_: anytype, _: usize) !HostActivationInput {
    return error.UnsupportedBackend;
}

pub fn sideExternalInput(_: anytype, _: usize) !HostActivationInput {
    return error.UnsupportedBackend;
}
