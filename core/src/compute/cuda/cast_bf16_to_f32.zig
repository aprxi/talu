//! bf16 -> f32 conversion kernel wrapper.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_cast_bf16_to_f32";
pub const op_name: []const u8 = "cast_bf16_to_f32";

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    src_bf16: *const device_mod.Buffer,
    dst_f32: *device_mod.Buffer,
    count: u32,
) !void {
    if (count == 0) return error.InvalidArgument;
    const src_bytes = std.math.mul(usize, @as(usize, count), @sizeOf(u16)) catch return error.InvalidArgument;
    const dst_bytes = std.math.mul(usize, @as(usize, count), @sizeOf(f32)) catch return error.InvalidArgument;
    if (src_bf16.size < src_bytes or dst_f32.size < dst_bytes) return error.InvalidArgument;

    arg_pack.reset();
    try arg_pack.appendBufferPtr(dst_f32);
    try arg_pack.appendBufferPtr(src_bf16);
    try arg_pack.appendScalar(u32, count);

    const block_x: u32 = 256;
    const grid_x: u32 = (count + block_x - 1) / block_x;
    try launch_mod.launchWithFamily(device, function, .{
        .grid_x = grid_x,
        .block_x = block_x,
    }, arg_pack, .copy_cast);
}
