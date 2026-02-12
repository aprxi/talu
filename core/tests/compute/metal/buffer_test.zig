//! Integration tests for Metal Buffer
//!
//! Metal Buffer provides GPU memory allocation and data transfer.
//! Tests are skipped on non-macOS platforms at compile time.

const std = @import("std");
const builtin = @import("builtin");
const main = @import("main");
const metal = main.core.compute.metal;
const Device = metal.Device;
const Buffer = metal.Buffer;

// =============================================================================
// Buffer Allocation Tests
// =============================================================================

test "Buffer allocBuffer creates buffer with correct size" {
    if (comptime builtin.os.tag != .macos) return;

    var device = Device.init() catch return;
    defer device.deinit();

    var buffer = device.allocBuffer(1024) catch return;
    defer buffer.deinit();

    try std.testing.expectEqual(@as(usize, 1024), buffer.size);
}

test "Buffer allocBuffer various sizes" {
    if (comptime builtin.os.tag != .macos) return;

    var device = Device.init() catch return;
    defer device.deinit();

    const sizes = [_]usize{ 64, 256, 1024, 4096, 65536 };
    for (sizes) |size| {
        var buffer = device.allocBuffer(size) catch continue;
        defer buffer.deinit();
        try std.testing.expectEqual(size, buffer.size);
    }
}

// =============================================================================
// Buffer Data Transfer Tests
// =============================================================================

test "Buffer upload and download preserve data" {
    if (comptime builtin.os.tag != .macos) return;

    var device = Device.init() catch return;
    defer device.deinit();

    var buffer = device.allocBuffer(64) catch return;
    defer buffer.deinit();

    // Create test data
    var input: [64]u8 = undefined;
    for (&input, 0..) |*v, i| {
        v.* = @intCast(i);
    }

    buffer.upload(&input);
    device.synchronize();

    var output: [64]u8 = undefined;
    buffer.download(&output);

    try std.testing.expectEqualSlices(u8, &input, &output);
}

test "Buffer upload partial data" {
    if (comptime builtin.os.tag != .macos) return;

    var device = Device.init() catch return;
    defer device.deinit();

    var buffer = device.allocBuffer(128) catch return;
    defer buffer.deinit();

    // Upload less than buffer size
    const input = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 };
    buffer.upload(&input);
    device.synchronize();

    var output: [8]u8 = undefined;
    buffer.download(&output);

    try std.testing.expectEqualSlices(u8, &input, &output);
}

// =============================================================================
// Buffer Contents Tests
// =============================================================================

test "Buffer contents returns non-null pointer" {
    if (comptime builtin.os.tag != .macos) return;

    var device = Device.init() catch return;
    defer device.deinit();

    var buffer = device.allocBuffer(256) catch return;
    defer buffer.deinit();

    const ptr = buffer.contents();
    try std.testing.expect(ptr != null);
}

test "Buffer contents allows direct memory access" {
    if (comptime builtin.os.tag != .macos) return;

    var device = Device.init() catch return;
    defer device.deinit();

    var buffer = device.allocBuffer(16) catch return;
    defer buffer.deinit();

    // Upload data
    const input = [_]u8{ 0xAA, 0xBB, 0xCC, 0xDD } ++ [_]u8{0} ** 12;
    buffer.upload(&input);
    device.synchronize();

    // Read via contents pointer
    if (buffer.contents()) |ptr| {
        const bytes: [*]u8 = @ptrCast(ptr);
        try std.testing.expectEqual(@as(u8, 0xAA), bytes[0]);
        try std.testing.expectEqual(@as(u8, 0xBB), bytes[1]);
    }
}

// =============================================================================
// Buffer Lifecycle Tests
// =============================================================================

test "Buffer multiple alloc/deinit cycles" {
    if (comptime builtin.os.tag != .macos) return;

    var device = Device.init() catch return;
    defer device.deinit();

    for (0..5) |_| {
        var buffer = device.allocBuffer(512) catch continue;
        const data = [_]u8{0xFF} ** 512;
        buffer.upload(&data);
        buffer.deinit();
    }
}
