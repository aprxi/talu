//! Tests for fail-closed Metal transport endpoint descriptors.

const std = @import("std");
const main = @import("main");

const endpoint = main.inference.backend.metal.interface.transport_endpoint;

test "Metal transport endpoint shape is present and fail-closed" {
    const Backend = struct {};
    var backend = Backend{};

    try std.testing.expectError(error.UnsupportedBackend, endpoint.deviceLocationHint(&backend));
    try std.testing.expectError(error.UnsupportedBackend, endpoint.hostDecodeActivationSlice(&backend, 0, 4));
    try std.testing.expectError(error.UnsupportedBackend, endpoint.hostPrefillActivationSlice(&backend, 4));
    try std.testing.expectError(error.UnsupportedBackend, endpoint.decodeInputBuffer(&backend, 0, 4));
    try std.testing.expectError(error.UnsupportedBackend, endpoint.prefillInputBuffer(&backend, 4));
    try std.testing.expectError(error.UnsupportedBackend, endpoint.sideInputBuffer(&backend, 4));
}
