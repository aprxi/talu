//! Integration tests for root.Device
//!
//! Device is the device descriptor from compute/device.zig, re-exported in root.zig.

const std = @import("std");
const main = @import("main");
const Device = main.core.Device;
const DeviceType = main.core.DeviceType;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "Device type is accessible" {
    const T = Device;
    _ = T;
}

test "Device is a struct" {
    const info = @typeInfo(Device);
    try std.testing.expect(info == .@"struct");
}

test "Device has expected fields" {
    const info = @typeInfo(Device);
    const fields = info.@"struct".fields;

    var has_device_type = false;
    var has_device_id = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "device_type")) has_device_type = true;
        if (comptime std.mem.eql(u8, field.name, "device_id")) has_device_id = true;
    }

    try std.testing.expect(has_device_type);
    try std.testing.expect(has_device_id);
}

// =============================================================================
// Method Tests
// =============================================================================

test "Device has cpu method" {
    try std.testing.expect(@hasDecl(Device, "cpu"));
}

test "Device has cuda method" {
    try std.testing.expect(@hasDecl(Device, "cuda"));
}

test "Device has metal method" {
    try std.testing.expect(@hasDecl(Device, "metal"));
}

test "Device has isCPU method" {
    try std.testing.expect(@hasDecl(Device, "isCPU"));
}

test "Device has isCUDA method" {
    try std.testing.expect(@hasDecl(Device, "isCUDA"));
}

// =============================================================================
// Factory Method Tests
// =============================================================================

test "Device cpu creates CPU device with id 0" {
    const dev = Device.cpu();
    try std.testing.expectEqual(DeviceType.CPU, dev.device_type);
    try std.testing.expectEqual(@as(i32, 0), dev.device_id);
}

test "Device cuda creates CUDA device with specified id" {
    const dev = Device.cuda(2);
    try std.testing.expectEqual(DeviceType.CUDA, dev.device_type);
    try std.testing.expectEqual(@as(i32, 2), dev.device_id);
}

test "Device metal creates Metal device with specified id" {
    const dev = Device.metal(1);
    try std.testing.expectEqual(DeviceType.Metal, dev.device_type);
    try std.testing.expectEqual(@as(i32, 1), dev.device_id);
}

// =============================================================================
// Predicate Tests
// =============================================================================

test "Device isCPU returns true only for CPU" {
    const cpu = Device.cpu();
    const cuda = Device.cuda(0);
    const metal = Device.metal(0);

    try std.testing.expect(cpu.isCPU());
    try std.testing.expect(!cuda.isCPU());
    try std.testing.expect(!metal.isCPU());
}

test "Device isCUDA returns true only for CUDA" {
    const cpu = Device.cpu();
    const cuda = Device.cuda(0);
    const metal = Device.metal(0);

    try std.testing.expect(!cpu.isCUDA());
    try std.testing.expect(cuda.isCUDA());
    try std.testing.expect(!metal.isCUDA());
}

// =============================================================================
// DeviceType Tests
// =============================================================================

test "DeviceType type is accessible" {
    const T = DeviceType;
    _ = T;
}

test "DeviceType is an enum" {
    const info = @typeInfo(DeviceType);
    try std.testing.expect(info == .@"enum");
}

test "DeviceType has expected variants" {
    // Check for DLPack-compatible device types
    _ = DeviceType.CPU;
    _ = DeviceType.CUDA;
    _ = DeviceType.Metal;
}
