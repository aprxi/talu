//! f32 -> f16 conversion kernel wrapper.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");
const registry_mod = @import("registry.zig");

pub const embedded_ptx = @embedFile("kernels/kernels.ptx");
pub const embedded_symbol: [:0]const u8 = "talu_cast_f32_to_f16_v1";

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    src_f32: *const device_mod.Buffer,
    dst_f16: *device_mod.Buffer,
    count: u32,
) !void {
    if (count == 0) return error.InvalidArgument;

    const src_bytes = @as(usize, count) * @sizeOf(f32);
    const dst_bytes = @as(usize, count) * @sizeOf(u16);
    if (src_f32.size < src_bytes or dst_f16.size < dst_bytes) {
        return error.InvalidArgument;
    }

    arg_pack.reset();
    try arg_pack.appendBufferPtr(dst_f16);
    try arg_pack.appendBufferPtr(src_f32);
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
