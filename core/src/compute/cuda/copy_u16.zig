//! Device-to-device copy kernel wrapper for u16 elements.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");
const registry_mod = @import("registry.zig");

pub const embedded_ptx = @embedFile("kernels/kernels.ptx");
pub const embedded_symbol: [:0]const u8 = "talu_copy_u16_v1";

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    src: *const device_mod.Buffer,
    dst: *device_mod.Buffer,
    count: u32,
) !void {
    if (count == 0) return error.InvalidArgument;

    const required_bytes = @as(usize, count) * @sizeOf(u16);
    if (src.size < required_bytes or dst.size < required_bytes) {
        return error.InvalidArgument;
    }

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

fn ceilDiv(numerator: u32, denominator: u32) u32 {
    return (numerator + denominator - 1) / denominator;
}
