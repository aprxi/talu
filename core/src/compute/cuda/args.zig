//! CUDA kernel argument packing for cuLaunchKernel.
//!
//! cuLaunchKernel expects a `void**` array where each entry points to host
//! memory containing the value of one kernel argument.

const std = @import("std");
const device_mod = @import("device.zig");

pub const ArgPack = struct {
    pub const max_params: usize = 64;
    pub const scalar_storage_bytes: usize = 256;
    const scalar_storage_align = @alignOf(u128);

    param_ptrs: [max_params]?*anyopaque = [_]?*anyopaque{null} ** max_params,
    scalar_storage: [scalar_storage_bytes]u8 align(scalar_storage_align) = undefined,
    param_count: usize = 0,
    scalar_offset: usize = 0,

    pub fn init(_: std.mem.Allocator) ArgPack {
        return .{
            .param_ptrs = [_]?*anyopaque{null} ** max_params,
        };
    }

    pub fn deinit(self: *ArgPack) void {
        self.* = undefined;
    }

    pub fn len(self: *const ArgPack) usize {
        return self.param_count;
    }

    pub fn reset(self: *ArgPack) void {
        self.param_count = 0;
        self.scalar_offset = 0;
    }

    pub fn asKernelParams(self: *const ArgPack) ?[*]const ?*anyopaque {
        if (self.param_count == 0) return null;
        return self.param_ptrs[0..self.param_count].ptr;
    }

    pub fn appendScalar(self: *ArgPack, comptime T: type, value: T) !void {
        switch (@typeInfo(T)) {
            .int, .float, .bool, .@"enum" => {},
            else => return error.UnsupportedKernelArgType,
        }

        const aligned_offset = std.mem.alignForward(usize, self.scalar_offset, @alignOf(T));
        const scalar_end = std.math.add(usize, aligned_offset, @sizeOf(T)) catch return error.KernelArgStorageExceeded;
        if (scalar_end > scalar_storage_bytes) return error.KernelArgStorageExceeded;

        if (self.param_count >= max_params) return error.KernelArgCapacityExceeded;

        const slot_bytes = self.scalar_storage[aligned_offset..scalar_end];
        const slot: *T = @ptrCast(@alignCast(slot_bytes.ptr));
        slot.* = value;
        self.param_ptrs[self.param_count] = @ptrCast(slot);
        self.param_count += 1;
        self.scalar_offset = scalar_end;
    }

    pub fn appendDevicePtr(self: *ArgPack, device_pointer: u64) !void {
        return self.appendScalar(u64, device_pointer);
    }

    pub fn appendBufferPtr(self: *ArgPack, buffer: *const device_mod.Buffer) !void {
        return self.appendDevicePtr(buffer.pointer);
    }
};

test "ArgPack.appendScalar stores f32 value address" {
    var pack = ArgPack.init(std.testing.allocator);
    defer pack.deinit();

    try pack.appendScalar(f32, 1.25);
    try std.testing.expectEqual(@as(usize, 1), pack.len());

    const ptr = pack.asKernelParams().?[0].?;
    const value_ptr: *const f32 = @ptrCast(@alignCast(ptr));
    try std.testing.expectApproxEqAbs(@as(f32, 1.25), value_ptr.*, 0.0001);
}

test "ArgPack.appendDevicePtr stores u64 pointer payload" {
    var pack = ArgPack.init(std.testing.allocator);
    defer pack.deinit();

    try pack.appendDevicePtr(0x1234_5678_90AB_CDEF);
    try std.testing.expectEqual(@as(usize, 1), pack.len());

    const ptr = pack.asKernelParams().?[0].?;
    const value_ptr: *const u64 = @ptrCast(@alignCast(ptr));
    try std.testing.expectEqual(@as(u64, 0x1234_5678_90AB_CDEF), value_ptr.*);
}

test "ArgPack.asKernelParams returns null for empty pack" {
    var pack = ArgPack.init(std.testing.allocator);
    defer pack.deinit();

    try std.testing.expect(pack.asKernelParams() == null);
}

test "ArgPack.appendScalar rejects unsupported types" {
    var pack = ArgPack.init(std.testing.allocator);
    defer pack.deinit();

    const unsupported = struct { x: u32 };
    try std.testing.expectError(error.UnsupportedKernelArgType, pack.appendScalar(unsupported, .{ .x = 7 }));
}

test "ArgPack.appendScalar enforces max_params capacity" {
    var pack = ArgPack.init(std.testing.allocator);
    defer pack.deinit();

    for (0..ArgPack.max_params) |_| {
        try pack.appendScalar(bool, true);
    }
    try std.testing.expectError(error.KernelArgCapacityExceeded, pack.appendScalar(bool, false));
}

test "ArgPack.appendScalar enforces scalar storage capacity" {
    var pack = ArgPack.init(std.testing.allocator);
    defer pack.deinit();

    const fit_count = ArgPack.scalar_storage_bytes / @sizeOf(u64);
    for (0..fit_count) |_| {
        try pack.appendScalar(u64, 0);
    }
    try std.testing.expectError(error.KernelArgStorageExceeded, pack.appendScalar(u64, 1));
}

test "ArgPack.reset clears packed arguments" {
    var pack = ArgPack.init(std.testing.allocator);
    defer pack.deinit();

    try pack.appendScalar(u32, 123);
    try std.testing.expectEqual(@as(usize, 1), pack.len());
    pack.reset();
    try std.testing.expectEqual(@as(usize, 0), pack.len());
    try std.testing.expect(pack.asKernelParams() == null);
}
