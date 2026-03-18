//! Causal attention softmax kernel wrapper.
//!
//! Applies causal mask (future positions set to zero) and row-wise softmax
//! in a single pass.  Used by the GEMM-based prefill attention path.
//!
//! Grid: (total_rows, 1).  Block: (128, 1).

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_causal_attn_softmax_f32";
pub const op_name: []const u8 = "causal_attn_softmax_f32";

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    scores: *device_mod.Buffer,
    total_rows: u32,
    cols: u32,
    q_rows: u32,
    position_base: u32,
    sliding_window: u32,
) !void {
    try validateArgs(scores, total_rows, cols, q_rows, position_base);

    arg_pack.reset();
    try arg_pack.appendBufferPtr(scores);
    try arg_pack.appendScalar(u32, total_rows);
    try arg_pack.appendScalar(u32, cols);
    try arg_pack.appendScalar(u32, q_rows);
    try arg_pack.appendScalar(u32, position_base);
    try arg_pack.appendScalar(u32, sliding_window);

    const block_x: u32 = 128;
    try launch_mod.launchWithFamily(device, function, .{
        .grid_x = total_rows,
        .block_x = block_x,
    }, arg_pack, .attention);
}

fn validateArgs(
    scores: *device_mod.Buffer,
    total_rows: u32,
    cols: u32,
    q_rows: u32,
    position_base: u32,
) !void {
    if (total_rows == 0 or cols == 0 or q_rows == 0) return error.InvalidArgument;
    if (total_rows % q_rows != 0) return error.InvalidArgument;
    // position_base + q_rows - 1 must be < cols for the last query row.
    if (@as(u64, position_base) + @as(u64, q_rows) > @as(u64, cols)) return error.InvalidArgument;
    const elems = std.math.mul(usize, @as(usize, total_rows), @as(usize, cols)) catch return error.InvalidArgument;
    const bytes = std.math.mul(usize, elems, @sizeOf(f32)) catch return error.InvalidArgument;
    if (scores.size < bytes) return error.InvalidArgument;
}

test "validateArgs rejects zero shapes" {
    var scores = device_mod.Buffer{ .pointer = 0, .size = 65536 };
    try std.testing.expectError(error.InvalidArgument, validateArgs(&scores, 0, 4, 1, 0));
    try std.testing.expectError(error.InvalidArgument, validateArgs(&scores, 4, 0, 1, 0));
    try std.testing.expectError(error.InvalidArgument, validateArgs(&scores, 4, 4, 0, 0));
}

test "validateArgs rejects total_rows not divisible by q_rows" {
    var scores = device_mod.Buffer{ .pointer = 0, .size = 65536 };
    try std.testing.expectError(error.InvalidArgument, validateArgs(&scores, 7, 16, 4, 0));
}

test "validateArgs rejects position_base + q_rows > cols" {
    var scores = device_mod.Buffer{ .pointer = 0, .size = 65536 };
    try std.testing.expectError(error.InvalidArgument, validateArgs(&scores, 4, 8, 4, 6));
}

test "validateArgs accepts valid params" {
    var scores = device_mod.Buffer{ .pointer = 0, .size = 65536 };
    // 20 rows (5 groups × 4 q_rows), 8 cols, position_base=4
    // Last q_row position: 4 + 3 = 7 < 8. OK.
    try validateArgs(&scores, 20, 8, 4, 4);
}
