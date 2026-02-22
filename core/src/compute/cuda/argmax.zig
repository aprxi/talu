//! Argmax kernel wrapper for modular CUDA runtime.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");
const registry_mod = @import("registry.zig");

pub const embedded_ptx = @embedFile("kernels/fallback_kernels.ptx");
pub const embedded_symbol: [:0]const u8 = "talu_argmax_f32_v1";

pub fn run(
    allocator: std.mem.Allocator,
    device: *device_mod.Device,
    registry: *registry_mod.Registry,
    input: *const device_mod.Buffer,
    out_index: *device_mod.Buffer,
    count: u32,
) !registry_mod.KernelSource {
    if (count == 0) return error.InvalidArgument;
    const bytes = @as(usize, count) * @sizeOf(f32);
    if (input.size < bytes or out_index.size < @sizeOf(u32)) return error.InvalidArgument;

    if (registry.embedded_module == null) try registry.loadEmbeddedModule(embedded_ptx);
    const resolved = try registry.resolveFunction("argmax_f32", embedded_symbol);
    var arg_pack = args_mod.ArgPack.init(allocator);
    defer arg_pack.deinit();
    try runWithFunction(&arg_pack, device, resolved.function, input, out_index, count);
    return resolved.source;
}

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    input: *const device_mod.Buffer,
    out_index: *device_mod.Buffer,
    count: u32,
) !void {
    if (count == 0) return error.InvalidArgument;
    const bytes = @as(usize, count) * @sizeOf(f32);
    if (input.size < bytes or out_index.size < @sizeOf(u32)) return error.InvalidArgument;

    arg_pack.reset();
    try arg_pack.appendBufferPtr(input);
    try arg_pack.appendScalar(u32, count);
    try arg_pack.appendBufferPtr(out_index);

    try launch_mod.launch(device, function, .{
        .grid_x = 1,
        .block_x = 256,
    }, arg_pack);
}

test "run rejects zero count" {
    var fake_device: device_mod.Device = undefined;
    var registry = registry_mod.Registry.init(std.testing.allocator, &fake_device);
    defer {
        registry.embedded_module = null;
        registry.sideload_module = null;
        registry.sideload_manifest = null;
    }

    const input = device_mod.Buffer{ .pointer = 0, .size = 16 };
    var out_index = device_mod.Buffer{ .pointer = 0, .size = @sizeOf(u32) };
    try std.testing.expectError(
        error.InvalidArgument,
        run(std.testing.allocator, &fake_device, &registry, &input, &out_index, 0),
    );
}

test "run validates input size" {
    var fake_device: device_mod.Device = undefined;
    var registry = registry_mod.Registry.init(std.testing.allocator, &fake_device);
    defer {
        registry.embedded_module = null;
        registry.sideload_module = null;
        registry.sideload_manifest = null;
    }

    const input = device_mod.Buffer{ .pointer = 0, .size = 4 };
    var out_index = device_mod.Buffer{ .pointer = 0, .size = @sizeOf(u32) };
    try std.testing.expectError(
        error.InvalidArgument,
        run(std.testing.allocator, &fake_device, &registry, &input, &out_index, 8),
    );
}
