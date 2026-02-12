//! Integration tests for Metal Device
//!
//! Metal Device provides GPU device abstraction for macOS Metal API.
//! Tests are skipped on non-macOS platforms at compile time.

const std = @import("std");
const builtin = @import("builtin");
const main = @import("main");
const metal = main.core.compute.metal;
const Device = metal.Device;

// =============================================================================
// Device Creation Tests
// =============================================================================

test "Device init creates valid device" {
    if (comptime builtin.os.tag != .macos) return;

    var device = Device.init() catch |err| {
        try std.testing.expect(err == error.MetalUnavailable);
        return;
    };
    defer device.deinit();

    try std.testing.expect(@intFromPtr(device.handle) != 0);
}

test "Device name returns GPU name" {
    if (comptime builtin.os.tag != .macos) return;

    var device = Device.init() catch return;
    defer device.deinit();

    const name = device.name();
    try std.testing.expect(name.len > 0);
    // Common Apple GPU names contain "Apple" or "AMD" or "Intel"
}

test "Device synchronize completes" {
    if (comptime builtin.os.tag != .macos) return;

    var device = Device.init() catch return;
    defer device.deinit();

    // Should complete without error
    device.synchronize();
}

// =============================================================================
// Multiple Device Tests
// =============================================================================

test "Device multiple init/deinit cycles" {
    if (comptime builtin.os.tag != .macos) return;

    for (0..3) |_| {
        var device = Device.init() catch return;
        const name = device.name();
        try std.testing.expect(name.len > 0);
        device.deinit();
    }
}

test "Device concurrent instances" {
    if (comptime builtin.os.tag != .macos) return;

    var device1 = Device.init() catch return;
    defer device1.deinit();

    var device2 = Device.init() catch return;
    defer device2.deinit();

    // Both devices should be valid
    try std.testing.expect(device1.name().len > 0);
    try std.testing.expect(device2.name().len > 0);
}
