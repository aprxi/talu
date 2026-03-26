//! Increments decode metadata arrays (seq_lens and positions) in-place.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_decode_u32_increment";
pub const op_name: []const u8 = "decode_u32_increment";

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    seq_lens_inout: *device_mod.Buffer,
    positions_inout: *device_mod.Buffer,
    count: u32,
) !void {
    try validateArgs(seq_lens_inout, positions_inout, count);

    arg_pack.reset();
    try arg_pack.appendBufferPtr(seq_lens_inout);
    try arg_pack.appendBufferPtr(positions_inout);
    try arg_pack.appendScalar(u32, count);

    const block_x: u32 = 256;
    try launch_mod.launchWithFamily(device, function, .{
        .grid_x = ceilDiv(count, block_x),
        .block_x = block_x,
    }, arg_pack, .pointwise);
}

fn validateArgs(
    seq_lens_inout: *const device_mod.Buffer,
    positions_inout: *const device_mod.Buffer,
    count: u32,
) !void {
    if (count == 0) return error.InvalidArgument;
    const bytes = std.math.mul(usize, @as(usize, count), @sizeOf(u32)) catch return error.InvalidArgument;
    if (seq_lens_inout.size < bytes or positions_inout.size < bytes) return error.InvalidArgument;
}

fn ceilDiv(numerator: u32, denominator: u32) u32 {
    return (numerator + denominator - 1) / denominator;
}

test "validateArgs rejects zero count" {
    const b = device_mod.Buffer{ .pointer = 0, .size = 64 };
    try std.testing.expectError(error.InvalidArgument, validateArgs(&b, &b, 0));
}

test "validateArgs rejects undersized buffers" {
    const small = device_mod.Buffer{ .pointer = 0, .size = 4 };
    try std.testing.expectError(error.InvalidArgument, validateArgs(&small, &small, 2));
}
