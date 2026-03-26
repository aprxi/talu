//! In-place softmax with per-row dynamic column counts read from device buffer.
//! Each block handles one row of a [batch_rows * n_heads, max_cols] matrix,
//! reading the actual column count from seq_lens (graph-compatible).

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_softmax_rows_dynamic_cols_ptrs";
pub const op_name: []const u8 = "softmax_rows_dynamic_cols_ptrs";

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    data: *device_mod.Buffer,
    seq_lens: *const device_mod.Buffer,
    positions: *const device_mod.Buffer,
    batch_rows: u32,
    n_heads: u32,
    max_cols: u32,
    sliding_window: u32,
) !void {
    if (batch_rows == 0 or n_heads == 0 or max_cols == 0) return error.InvalidArgument;
    const total_rows = std.math.mul(u32, batch_rows, n_heads) catch return error.InvalidArgument;
    const total_elems = std.math.mul(usize, @as(usize, total_rows), @as(usize, max_cols)) catch return error.InvalidArgument;
    if (data.size < total_elems * @sizeOf(f32)) return error.InvalidArgument;
    if (seq_lens.size < @as(usize, batch_rows) * @sizeOf(u32)) return error.InvalidArgument;
    if (positions.size < @as(usize, batch_rows) * @sizeOf(u32)) return error.InvalidArgument;

    arg_pack.reset();
    try arg_pack.appendBufferPtr(data);
    try arg_pack.appendBufferPtr(seq_lens);
    try arg_pack.appendBufferPtr(positions);
    try arg_pack.appendScalar(u32, n_heads);
    try arg_pack.appendScalar(u32, max_cols);
    try arg_pack.appendScalar(u32, sliding_window);

    const block_x: u32 = 128;
    try launch_mod.launchWithFamily(device, function, .{
        .grid_x = total_rows,
        .block_x = block_x,
    }, arg_pack, .attention);
}

test "runWithFunction rejects zero batch_rows" {
    const b = device_mod.Buffer{ .pointer = 0, .size = 65536 };
    var data = device_mod.Buffer{ .pointer = 0, .size = 65536 };
    var ap: args_mod.ArgPack = undefined;
    try std.testing.expectError(
        error.InvalidArgument,
        runWithFunction(&ap, undefined, undefined, &data, &b, &b, 0, 16, 200, 0),
    );
}
