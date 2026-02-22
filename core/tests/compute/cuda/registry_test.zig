//! Integration tests for CUDA kernel registry selection behavior.

const std = @import("std");
const main = @import("main");
const cuda = main.core.compute.cuda;

test "Registry.init starts with empty modules and manifest" {
    var fake_device: cuda.Device = undefined;
    var registry = cuda.registry.Registry.init(std.testing.allocator, &fake_device);
    defer {
        // No modules loaded in this test.
        registry.embedded_module = null;
        registry.sideload_module = null;
        registry.sideload_manifest = null;
    }

    try std.testing.expect(registry.embedded_module == null);
    try std.testing.expect(registry.sideload_module == null);
    try std.testing.expect(registry.sideload_manifest == null);
}
