//! Integration tests for compute.DLDevice
//!
//! DLDevice is the DLPack device representation for tensor exchange.

const std = @import("std");
const main = @import("main");
const DLDevice = main.compute.DLDevice;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "DLDevice type is accessible" {
    const T = DLDevice;
    _ = T;
}

test "DLDevice is a struct" {
    const info = @typeInfo(DLDevice);
    try std.testing.expect(info == .@"struct");
}

test "DLDevice has expected fields" {
    const info = @typeInfo(DLDevice);
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
// Factory Tests
// =============================================================================

test "DLDevice has cpu method" {
    try std.testing.expect(@hasDecl(DLDevice, "cpu"));
}

test "DLDevice cpu creates CPU device" {
    const device = DLDevice.cpu();
    try std.testing.expectEqual(@as(i32, 0), device.device_id);
}
