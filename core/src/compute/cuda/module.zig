//! CUDA module and function lifecycle wrappers.

const std = @import("std");
const device_mod = @import("device.zig");

pub const Module = struct {
    handle: device_mod.ModuleHandle,

    pub fn load(device: *device_mod.Device, image: []const u8) !Module {
        if (image.len == 0) return error.InvalidArgument;

        // PTX input is expected to be a null-terminated string. Appending a
        // trailing zero is harmless for cubin payloads and ensures PTX safety.
        const storage = try std.heap.page_allocator.alloc(u8, image.len + 1);
        defer std.heap.page_allocator.free(storage);
        @memcpy(storage[0..image.len], image);
        storage[image.len] = 0;

        const handle = try device.moduleLoadData(storage.ptr);
        return .{ .handle = handle };
    }

    pub fn deinit(self: *Module, device: *device_mod.Device) void {
        device.moduleUnload(self.handle);
        self.* = undefined;
    }

    pub fn getFunction(self: *const Module, device: *device_mod.Device, symbol: [:0]const u8) !Function {
        return .{ .handle = try device.moduleGetFunction(self.handle, symbol) };
    }
};

pub const Function = struct {
    handle: device_mod.FunctionHandle,
};

test "Module.load rejects empty module bytes" {
    if (device_mod.probeRuntime() != .available) return error.SkipZigTest;

    var device = try device_mod.Device.init();
    defer device.deinit();

    try std.testing.expectError(error.InvalidArgument, Module.load(&device, &.{}));
}

test "Module.load returns CudaModuleApiUnavailable when module API symbols are unavailable" {
    if (device_mod.probeRuntime() != .available) return error.SkipZigTest;

    var device = try device_mod.Device.init();
    defer device.deinit();

    if (device.supportsModuleLaunch()) return;

    const dummy = [_]u8{0x7f};
    try std.testing.expectError(error.CudaModuleApiUnavailable, Module.load(&device, &dummy));
}
