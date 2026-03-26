//! Batched RoPE kernel wrapper: applies rotary position embedding to multiple
//! rows, reading per-row positions from a device buffer.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_rope_rows_ptrs";
pub const op_name: []const u8 = "rope_rows_ptrs";

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    io: *device_mod.Buffer,
    positions: *const device_mod.Buffer,
    batch_rows: u32,
    n_heads: u32,
    head_dim: u32,
    rope_dim: u32,
    theta: f32,
) !void {
    if (batch_rows == 0 or n_heads == 0 or head_dim == 0) return error.InvalidArgument;
    if (rope_dim == 0 or rope_dim > head_dim or (rope_dim & 1) != 0) return error.InvalidArgument;
    if (!std.math.isFinite(theta) or theta <= 1.0) return error.InvalidArgument;

    const half = rope_dim / 2;
    const total_pairs = std.math.mul(u32, n_heads, half) catch return error.InvalidArgument;
    const io_elems = std.math.mul(usize, @as(usize, batch_rows) * @as(usize, n_heads), @as(usize, head_dim)) catch return error.InvalidArgument;
    if (io.size < io_elems * @sizeOf(f32)) return error.InvalidArgument;
    if (positions.size < @as(usize, batch_rows) * @sizeOf(u32)) return error.InvalidArgument;

    arg_pack.reset();
    try arg_pack.appendBufferPtr(io);
    try arg_pack.appendBufferPtr(positions);
    try arg_pack.appendScalar(u32, batch_rows);
    try arg_pack.appendScalar(u32, n_heads);
    try arg_pack.appendScalar(u32, head_dim);
    try arg_pack.appendScalar(u32, rope_dim);
    try arg_pack.appendScalar(f32, theta);

    const block_x: u32 = 128;
    try launch_mod.launchWithFamily(device, function, .{
        .grid_x = ceilDiv(total_pairs, block_x),
        .grid_y = batch_rows,
        .block_x = block_x,
    }, arg_pack, .rope);
}

fn ceilDiv(numerator: u32, denominator: u32) u32 {
    return (numerator + denominator - 1) / denominator;
}

test "runWithFunction rejects invalid rope_dim" {
    const b = device_mod.Buffer{ .pointer = 0, .size = 65536 };
    var io = device_mod.Buffer{ .pointer = 0, .size = 65536 };
    var ap: args_mod.ArgPack = undefined;
    // rope_dim=0
    try std.testing.expectError(
        error.InvalidArgument,
        runWithFunction(&ap, undefined, undefined, &io, &b, 1, 16, 128, 0, 10000.0),
    );
}
