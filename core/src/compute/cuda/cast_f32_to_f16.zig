//! f32 -> f16 conversion kernel wrapper.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");
const registry_mod = @import("registry.zig");
const copy_cast = @import("../copy_cast.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_cast_f32_to_f16";
pub const op_name: []const u8 = "cast_f32_to_f16";

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    src_f32: *const device_mod.Buffer,
    dst_f16: *device_mod.Buffer,
    count: u32,
) !void {
    try validateArgs(src_f32, dst_f16, count);

    arg_pack.reset();
    try arg_pack.appendBufferPtr(dst_f16);
    try arg_pack.appendBufferPtr(src_f32);
    try arg_pack.appendScalar(u32, count);

    const block_x: u32 = 256;
    const grid_x: u32 = ceilDiv(count, block_x);
    try launch_mod.launchWithFamily(device, function, .{
        .grid_x = grid_x,
        .block_x = block_x,
    }, arg_pack, .copy_cast);
}

fn validateArgs(src_f32: *const device_mod.Buffer, dst_f16: *device_mod.Buffer, count: u32) !void {
    _ = try copy_cast.validateCastBuffers(.{
        .backend = .cuda,
        .src_dtype = .f32,
        .dst_dtype = .f16,
        .layout = .row_major_contiguous,
        .element_count = @intCast(count),
        .src_size = src_f32.size,
        .dst_size = dst_f16.size,
    });
}

fn ceilDiv(numerator: u32, denominator: u32) u32 {
    return (numerator + denominator - 1) / denominator;
}

test "validateArgs rejects zero count" {
    const src = device_mod.Buffer{ .pointer = 0, .size = 16 };
    var dst = src;
    try std.testing.expectError(error.InvalidShape, validateArgs(&src, &dst, 0));
}

test "validateArgs rejects undersized destination buffer" {
    const src = device_mod.Buffer{ .pointer = 0, .size = 16 };
    var dst = device_mod.Buffer{ .pointer = 0, .size = 6 };
    try std.testing.expectError(error.BufferTooSmall, validateArgs(&src, &dst, 4));
}

test "validateArgs runWithFunction rejects undersized destination before mutating arg pack" {
    var arg_pack = args_mod.ArgPack.init(std.testing.allocator);
    defer arg_pack.deinit();
    try arg_pack.appendScalar(u32, 123);
    const before_len = arg_pack.len();

    var fake_device: device_mod.Device = undefined;
    const fake_function = module_mod.Function{ .handle = @ptrFromInt(1) };
    const src = device_mod.Buffer{ .pointer = 0, .size = 16 };
    var dst = device_mod.Buffer{ .pointer = 0, .size = 6 };
    try std.testing.expectError(error.BufferTooSmall, runWithFunction(&arg_pack, &fake_device, fake_function, &src, &dst, 4));
    try std.testing.expectEqual(before_len, arg_pack.len());
}

test "ceilDiv computes expected block count" {
    try std.testing.expectEqual(@as(u32, 1), ceilDiv(1, 256));
    try std.testing.expectEqual(@as(u32, 2), ceilDiv(257, 256));
}
