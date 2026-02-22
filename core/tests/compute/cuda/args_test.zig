//! Integration tests for CUDA argument packing.

const std = @import("std");
const main = @import("main");
const cuda = main.core.compute.cuda;

test "ArgPack.appendScalar stores scalar payloads" {
    var pack = cuda.args.ArgPack.init(std.testing.allocator);
    defer pack.deinit();

    try pack.appendScalar(u32, 42);
    try std.testing.expectEqual(@as(usize, 1), pack.len());
    try std.testing.expect(pack.asKernelParams() != null);
}

test "ArgPack.asKernelParams returns null for empty pack" {
    var pack = cuda.args.ArgPack.init(std.testing.allocator);
    defer pack.deinit();

    try std.testing.expect(pack.asKernelParams() == null);
}

test "ArgPack.appendBufferPtr stores buffer pointer as u64" {
    var pack = cuda.args.ArgPack.init(std.testing.allocator);
    defer pack.deinit();

    const buffer = cuda.Buffer{
        .pointer = 0x1234,
        .size = 16,
    };
    try pack.appendBufferPtr(&buffer);
    try std.testing.expectEqual(@as(usize, 1), pack.len());
}

test "ArgPack.appendDevicePtr stores raw device pointer" {
    var pack = cuda.args.ArgPack.init(std.testing.allocator);
    defer pack.deinit();

    try pack.appendDevicePtr(0xDEAD_BEEF);
    try std.testing.expectEqual(@as(usize, 1), pack.len());
}

test "ArgPack.appendScalar rejects unsupported argument types" {
    var pack = cuda.args.ArgPack.init(std.testing.allocator);
    defer pack.deinit();

    const unsupported = struct { x: u32 };
    try std.testing.expectError(error.UnsupportedKernelArgType, pack.appendScalar(unsupported, .{ .x = 1 }));
}

test "ArgPack.appendScalar errors when scalar storage is exhausted" {
    var pack = cuda.args.ArgPack.init(std.testing.allocator);
    defer pack.deinit();

    const fit_count = cuda.args.ArgPack.scalar_storage_bytes / @sizeOf(u64);
    for (0..fit_count) |_| {
        try pack.appendScalar(u64, 1);
    }

    try std.testing.expectError(error.KernelArgStorageExceeded, pack.appendScalar(u64, 2));
}

test "ArgPack.appendScalar errors when max parameter count is exceeded" {
    var pack = cuda.args.ArgPack.init(std.testing.allocator);
    defer pack.deinit();

    for (0..cuda.args.ArgPack.max_params) |_| {
        try pack.appendScalar(bool, true);
    }

    try std.testing.expectError(error.KernelArgCapacityExceeded, pack.appendScalar(bool, false));
}

test "ArgPack.reset clears kernel arguments" {
    var pack = cuda.args.ArgPack.init(std.testing.allocator);
    defer pack.deinit();

    try pack.appendScalar(u32, 7);
    try std.testing.expectEqual(@as(usize, 1), pack.len());
    pack.reset();
    try std.testing.expectEqual(@as(usize, 0), pack.len());
    try std.testing.expect(pack.asKernelParams() == null);
}
