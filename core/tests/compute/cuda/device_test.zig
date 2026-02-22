//! Integration tests for CUDA device/runtime lifecycle.

const std = @import("std");
const main = @import("main");
const cuda = main.core.compute.cuda;

test "probeRuntime returns disabled when isRuntimeSupported is false" {
    if (cuda.device.isRuntimeSupported()) return;
    try std.testing.expectEqual(cuda.Probe.disabled, cuda.probeRuntime());
}

test "probeRuntime returns a non-disabled value when isRuntimeSupported is true" {
    if (!cuda.device.isRuntimeSupported()) return;
    try std.testing.expect(cuda.probeRuntime() != .disabled);
}

test "Device.init and Device.deinit succeed when probeRuntime is available" {
    if (cuda.probeRuntime() != .available) return error.SkipZigTest;

    var device = try cuda.Device.init();
    defer device.deinit();

    try std.testing.expect(device.name().len > 0);
}

test "Device.computeCapability returns non-zero major when available" {
    if (cuda.probeRuntime() != .available) return error.SkipZigTest;

    var device = try cuda.Device.init();
    defer device.deinit();

    const capability = device.computeCapability() catch |err| {
        try std.testing.expect(err == error.CudaQueryUnavailable);
        return;
    };
    try std.testing.expect(capability.major > 0);
}
