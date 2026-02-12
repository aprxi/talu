//! Metal device management for GPU acceleration.
//!
//! Provides Zig bindings to Metal device creation, buffer allocation,
//! and synchronization primitives for macOS GPU compute.

const std = @import("std");

/// Opaque Metal device handle.
pub const MetalDevice = opaque {};

/// Opaque Metal buffer handle.
pub const MetalBuffer = opaque {};

/// C API imports.
extern fn metal_is_available() bool;
extern fn metal_device_create() ?*MetalDevice;
extern fn metal_device_destroy(device: *MetalDevice) void;
extern fn metal_device_name(device: *MetalDevice) [*:0]const u8;
extern fn metal_buffer_create(device: *MetalDevice, size: usize) ?*MetalBuffer;
extern fn metal_buffer_upload(buffer: *MetalBuffer, data: *const anyopaque, size: usize) void;
extern fn metal_buffer_download(buffer: *MetalBuffer, data: *anyopaque, size: usize) void;
extern fn metal_buffer_contents(buffer: *MetalBuffer) ?*anyopaque;
extern fn metal_buffer_destroy(buffer: *MetalBuffer) void;
extern fn metal_device_synchronize(device: *MetalDevice) void;

/// Check if Metal is available on this system.
pub fn isAvailable() bool {
    return metal_is_available();
}

/// Managed Metal device context.
pub const Device = struct {
    handle: *MetalDevice,

    pub fn init() !Device {
        const handle = metal_device_create() orelse return error.MetalUnavailable;
        return .{ .handle = handle };
    }

    pub fn deinit(self: *Device) void {
        metal_device_destroy(self.handle);
    }

    pub fn name(self: *Device) []const u8 {
        return std.mem.span(metal_device_name(self.handle));
    }

    pub fn synchronize(self: *Device) void {
        metal_device_synchronize(self.handle);
    }

    /// Allocate a Metal buffer.
    pub fn allocBuffer(self: *Device, buffer_size: usize) !Buffer {
        const handle = metal_buffer_create(self.handle, buffer_size) orelse return error.OutOfMemory;
        return .{ .handle = handle, .size = buffer_size };
    }
};

/// Managed Metal buffer.
pub const Buffer = struct {
    handle: *MetalBuffer,
    size: usize,

    pub fn deinit(self: *Buffer) void {
        metal_buffer_destroy(self.handle);
    }

    pub fn upload(self: *Buffer, data: []const u8) void {
        metal_buffer_upload(self.handle, data.ptr, @min(data.len, self.size));
    }

    pub fn download(self: *Buffer, data: []u8) void {
        metal_buffer_download(self.handle, data.ptr, @min(data.len, self.size));
    }

    pub fn contents(self: *Buffer) ?*anyopaque {
        return metal_buffer_contents(self.handle);
    }
};

// =============================================================================
// Unit Tests - compiled only on macOS where Metal is available
// =============================================================================

const builtin = @import("builtin");

test "isAvailable returns consistent result" {
    if (comptime builtin.os.tag != .macos) return; // Skip on non-macOS

    // Call multiple times - result should be consistent
    const first = isAvailable();
    const second = isAvailable();
    const third = isAvailable();

    try std.testing.expectEqual(first, second);
    try std.testing.expectEqual(second, third);
}

test "Device init and deinit multiple times" {
    if (comptime builtin.os.tag != .macos) return;

    // Test that we can create and destroy multiple devices
    for (0..3) |_| {
        var device = Device.init() catch |err| {
            // MetalUnavailable is acceptable if no GPU
            try std.testing.expect(err == error.MetalUnavailable);
            return;
        };
        // Verify device has valid handle (non-zero address)
        try std.testing.expect(@intFromPtr(device.handle) != 0);
        device.deinit();
    }
}

test "Device name returns non-empty string" {
    if (comptime builtin.os.tag != .macos) return;
    var device = Device.init() catch return;
    defer device.deinit();
    const device_name = device.name();
    try std.testing.expect(device_name.len > 0);
}

test "Device synchronize after buffer operations" {
    if (comptime builtin.os.tag != .macos) return;
    var device = Device.init() catch return;
    defer device.deinit();

    // Allocate buffer and upload data
    var buffer = device.allocBuffer(256) catch return;
    defer buffer.deinit();

    const input = [_]u8{0xAB} ** 256;
    buffer.upload(&input);

    // Synchronize to ensure upload completes
    device.synchronize();

    // Verify data after sync
    var output: [256]u8 = undefined;
    buffer.download(&output);
    try std.testing.expectEqualSlices(u8, &input, &output);
}

test "Device allocBuffer returns buffer with correct size" {
    if (comptime builtin.os.tag != .macos) return;
    var device = Device.init() catch return;
    defer device.deinit();
    var buffer = device.allocBuffer(1024) catch return;
    defer buffer.deinit();
    try std.testing.expectEqual(@as(usize, 1024), buffer.size);
}

test "Buffer upload and download preserve data" {
    if (comptime builtin.os.tag != .macos) return;
    var device = Device.init() catch return;
    defer device.deinit();
    var buffer = device.allocBuffer(16) catch return;
    defer buffer.deinit();

    const input = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    buffer.upload(&input);

    var output: [16]u8 = undefined;
    buffer.download(&output);
    try std.testing.expectEqualSlices(u8, &input, &output);
}

test "Buffer contents returns valid pointer for read/write" {
    if (comptime builtin.os.tag != .macos) return;
    var device = Device.init() catch return;
    defer device.deinit();
    var buffer = device.allocBuffer(64) catch return;
    defer buffer.deinit();
    const ptr = buffer.contents();
    try std.testing.expect(ptr != null);
    // Verify pointer is usable by writing and reading back
    const slice = @as([*]u8, @ptrCast(ptr.?))[0..64];
    @memset(slice, 0xAB);
    try std.testing.expectEqual(@as(u8, 0xAB), slice[0]);
    try std.testing.expectEqual(@as(u8, 0xAB), slice[63]);
}
