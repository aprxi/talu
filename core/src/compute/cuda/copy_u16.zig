//! Device-to-device copy kernel wrapper for u16 elements.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");
const registry_mod = @import("registry.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_copy_u16";
pub const op_name: []const u8 = "copy_u16";

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
    try launch_mod.launch(device, function, .{
        .grid_x = grid_x,
        .block_x = block_x,
    }, arg_pack);
}

fn validateArgs(src: *const device_mod.Buffer, dst: *device_mod.Buffer, count: u32) !void {
    if (count == 0) return error.InvalidArgument;
    const required_bytes = std.math.mul(usize, @as(usize, count), @sizeOf(u16)) catch return error.InvalidArgument;
    if (src.size < required_bytes or dst.size < required_bytes) return error.InvalidArgument;
}

fn ceilDiv(numerator: u32, denominator: u32) u32 {
    return (numerator + denominator - 1) / denominator;
}

test "validateArgs rejects zero count" {
    const src = device_mod.Buffer{ .pointer = 0, .size = 16 };
    var dst = src;
    try std.testing.expectError(error.InvalidArgument, validateArgs(&src, &dst, 0));
}

test "validateArgs rejects undersized destination buffer" {
    const src = device_mod.Buffer{ .pointer = 0, .size = 16 };
    var dst = device_mod.Buffer{ .pointer = 0, .size = 6 };
    try std.testing.expectError(error.InvalidArgument, validateArgs(&src, &dst, 4));
}

test "ceilDiv computes expected block count" {
    try std.testing.expectEqual(@as(u32, 1), ceilDiv(1, 256));
    try std.testing.expectEqual(@as(u32, 2), ceilDiv(257, 256));
}
