//! Integration tests for Device
//!
//! Device is the descriptor for tensor location (DLPack-compatible).
//! Supports multiple device types including CPU, CUDA, and Metal.

const std = @import("std");
const main = @import("main");
const Device = main.core.Device;
const DeviceType = main.core.DeviceType;

// =============================================================================
// Factory Function Tests
// =============================================================================

test "cpu returns CPU device" {
    const device = Device.cpu();
    try std.testing.expectEqual(DeviceType.CPU, device.device_type);
    try std.testing.expectEqual(@as(i32, 0), device.device_id);
}

test "cuda returns CUDA device" {
    const device = Device.cuda(0);
    try std.testing.expectEqual(DeviceType.CUDA, device.device_type);
    try std.testing.expectEqual(@as(i32, 0), device.device_id);

    const device1 = Device.cuda(1);
    try std.testing.expectEqual(DeviceType.CUDA, device1.device_type);
    try std.testing.expectEqual(@as(i32, 1), device1.device_id);

    const device2 = Device.cuda(7);
    try std.testing.expectEqual(DeviceType.CUDA, device2.device_type);
    try std.testing.expectEqual(@as(i32, 7), device2.device_id);
}

test "metal returns Metal device" {
    const device = Device.metal(0);
    try std.testing.expectEqual(DeviceType.Metal, device.device_type);
    try std.testing.expectEqual(@as(i32, 0), device.device_id);

    const device1 = Device.metal(1);
    try std.testing.expectEqual(DeviceType.Metal, device1.device_type);
    try std.testing.expectEqual(@as(i32, 1), device1.device_id);

    const device2 = Device.metal(3);
    try std.testing.expectEqual(DeviceType.Metal, device2.device_type);
    try std.testing.expectEqual(@as(i32, 3), device2.device_id);
}

// =============================================================================
// Type Predicate Tests
// =============================================================================

test "isCPU returns true for CPU device" {
    const cpu_device = Device.cpu();
    try std.testing.expect(cpu_device.isCPU());

    const cuda_device = Device.cuda(0);
    try std.testing.expect(!cuda_device.isCPU());

    const metal_device = Device.metal(0);
    try std.testing.expect(!metal_device.isCPU());
}

test "isCUDA returns true for CUDA device" {
    const cuda_device = Device.cuda(0);
    try std.testing.expect(cuda_device.isCUDA());

    const cpu_device = Device.cpu();
    try std.testing.expect(!cpu_device.isCUDA());

    const metal_device = Device.metal(0);
    try std.testing.expect(!metal_device.isCUDA());
}
