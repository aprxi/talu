//! Row-wise softmax kernel wrapper for batched attention.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_softmax_rows_f32";
pub const op_name: []const u8 = "softmax_rows_f32";

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    input: *const device_mod.Buffer,
    out: *device_mod.Buffer,
    rows: u32,
    cols: u32,
) !void {
    try validateArgs(input, out, rows, cols);

    arg_pack.reset();
    try arg_pack.appendBufferPtr(out);
    try arg_pack.appendBufferPtr(input);
    try arg_pack.appendScalar(u32, rows);
    try arg_pack.appendScalar(u32, cols);

    try launch_mod.launch(device, function, .{
        .grid_x = rows,
        .block_x = 256,
    }, arg_pack);
}

fn validateArgs(
    input: *const device_mod.Buffer,
    out: *device_mod.Buffer,
    rows: u32,
    cols: u32,
) !void {
    if (rows == 0 or cols == 0) return error.InvalidArgument;
    const elems = std.math.mul(usize, @as(usize, rows), @as(usize, cols)) catch return error.InvalidArgument;
    const bytes = std.math.mul(usize, elems, @sizeOf(f32)) catch return error.InvalidArgument;
    if (input.size < bytes or out.size < bytes) return error.InvalidArgument;
}

test "validateArgs rejects zero shapes" {
    const input = device_mod.Buffer{ .pointer = 0, .size = 16 };
    var out = device_mod.Buffer{ .pointer = 0, .size = 16 };
    try std.testing.expectError(error.InvalidArgument, validateArgs(&input, &out, 0, 4));
    try std.testing.expectError(error.InvalidArgument, validateArgs(&input, &out, 1, 0));
}
