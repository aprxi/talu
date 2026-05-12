//! Device-to-device copy kernel wrapper for modular CUDA runtime.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");
const registry_mod = @import("registry.zig");
const copy_cast = @import("../copy_cast.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_copy_f32";
pub const op_name: []const u8 = "copy_f32";

pub fn run(
    allocator: std.mem.Allocator,
    device: *device_mod.Device,
    registry: *registry_mod.Registry,
    src: *const device_mod.Buffer,
    dst: *device_mod.Buffer,
    count: u32,
) !registry_mod.KernelSource {
    try validateArgs(src, dst, count);

    if (registry.embedded_module == null) try registry.loadEmbeddedModule(embedded_module);
    const resolved = try registry.resolveFunction(op_name, embedded_symbol);
    var arg_pack = args_mod.ArgPack.init(allocator);
    defer arg_pack.deinit();
    try runWithFunction(&arg_pack, device, resolved.function, src, dst, count);
    return resolved.source;
}

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    src: *const device_mod.Buffer,
    dst: *device_mod.Buffer,
    count: u32,
) !void {
    try validateArgs(src, dst, count);

    arg_pack.reset();
    try arg_pack.appendBufferPtr(dst);
    try arg_pack.appendBufferPtr(src);
    try arg_pack.appendScalar(u32, count);

    const block_x: u32 = 256;
    const grid_x: u32 = ceilDiv(count, block_x);
    try launch_mod.launchWithFamily(device, function, .{
        .grid_x = grid_x,
        .block_x = block_x,
    }, arg_pack, .copy_cast);
}

fn validateArgs(src: *const device_mod.Buffer, dst: *device_mod.Buffer, count: u32) !void {
    _ = try copy_cast.validateCopyBuffers(.{
        .backend = .cuda,
        .direction = .device_to_device,
        .dtype = .f32,
        .layout = .row_major_contiguous,
        .element_count = @intCast(count),
        .src_size = src.size,
        .dst_size = dst.size,
        .src_address = @intCast(src.pointer),
        .dst_address = @intCast(dst.pointer),
    });
}

fn ceilDiv(numerator: u32, denominator: u32) u32 {
    return (numerator + denominator - 1) / denominator;
}

test "run rejects zero count" {
    var fake_device: device_mod.Device = undefined;
    var registry = registry_mod.Registry.init(std.testing.allocator, &fake_device);
    defer {
        registry.embedded_module = null;
        registry.sideload_module = null;
        registry.sideload_manifest = null;
    }

    const fake_buffer = device_mod.Buffer{ .pointer = 0, .size = 4 };
    var dst = fake_buffer;
    try std.testing.expectError(
        error.InvalidShape,
        run(std.testing.allocator, &fake_device, &registry, &fake_buffer, &dst, 0),
    );
}

test "validateArgs rejects undersized destination buffer" {
    const src = device_mod.Buffer{ .pointer = 0x1000, .size = 16 };
    var dst = device_mod.Buffer{ .pointer = 0x2000, .size = 12 };
    try std.testing.expectError(error.BufferTooSmall, validateArgs(&src, &dst, 4));
}

test "validateArgs rejects misaligned f32 copy buffers" {
    const src = device_mod.Buffer{ .pointer = 0x1002, .size = 16 };
    var dst = device_mod.Buffer{ .pointer = 0x2000, .size = 16 };
    try std.testing.expectError(error.AlignmentMismatch, validateArgs(&src, &dst, 4));
}

test "ceilDiv computes expected block count" {
    try std.testing.expectEqual(@as(u32, 1), ceilDiv(1, 256));
    try std.testing.expectEqual(@as(u32, 2), ceilDiv(257, 256));
}
