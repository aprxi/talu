//! Vector add kernel wrapper for modular CUDA runtime.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");
const registry_mod = @import("registry.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_vector_add_f32";

pub fn run(
    allocator: std.mem.Allocator,
    device: *device_mod.Device,
    registry: *registry_mod.Registry,
    a: *const device_mod.Buffer,
    b: *const device_mod.Buffer,
    out: *device_mod.Buffer,
    count: u32,
) !registry_mod.KernelSource {
    if (count == 0) return error.InvalidArgument;

    const required_bytes = @as(usize, count) * @sizeOf(f32);
    if (a.size < required_bytes or b.size < required_bytes or out.size < required_bytes) {
        return error.InvalidArgument;
    }

    if (registry.embedded_module == null) try registry.loadEmbeddedModule(embedded_module);
    const resolved = try registry.resolveFunction("vector_add_f32", embedded_symbol);
    var arg_pack = args_mod.ArgPack.init(allocator);
    defer arg_pack.deinit();
    try runWithFunction(&arg_pack, device, resolved.function, a, b, out, count);
    return resolved.source;
}

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    a: *const device_mod.Buffer,
    b: *const device_mod.Buffer,
    out: *device_mod.Buffer,
    count: u32,
) !void {
    if (count == 0) return error.InvalidArgument;

    const required_bytes = @as(usize, count) * @sizeOf(f32);
    if (a.size < required_bytes or b.size < required_bytes or out.size < required_bytes) {
        return error.InvalidArgument;
    }

    arg_pack.reset();
    try arg_pack.appendBufferPtr(out);
    try arg_pack.appendBufferPtr(a);
    try arg_pack.appendBufferPtr(b);
    try arg_pack.appendScalar(u32, count);

    const block_x: u32 = 256;
    const grid_x: u32 = ceilDiv(count, block_x);
    try launch_mod.launch(device, function, .{
        .grid_x = grid_x,
        .block_x = block_x,
    }, arg_pack);
}

fn ceilDiv(numerator: u32, denominator: u32) u32 {
    return (numerator + denominator - 1) / denominator;
}

test "run rejects zero count" {
    var fake_device: device_mod.Device = undefined;
    var registry = registry_mod.Registry.init(std.testing.allocator, &fake_device);
    defer {
        // Registry contains no loaded modules in this test; avoid calling deinit
        // on undefined fake device by clearing slots explicitly.
        registry.embedded_module = null;
        registry.sideload_module = null;
        registry.sideload_manifest = null;
    }

    const fake_buffer = device_mod.Buffer{ .pointer = 0, .size = 4 };
    var out = fake_buffer;
    try std.testing.expectError(
        error.InvalidArgument,
        run(std.testing.allocator, &fake_device, &registry, &fake_buffer, &fake_buffer, &out, 0),
    );
}

test "ceilDiv computes expected block count" {
    try std.testing.expectEqual(@as(u32, 1), ceilDiv(1, 256));
    try std.testing.expectEqual(@as(u32, 2), ceilDiv(257, 256));
}
