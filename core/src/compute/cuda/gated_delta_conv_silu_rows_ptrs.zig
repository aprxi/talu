//! Gated-delta depthwise conv + SiLU fused rows wrapper with per-row state pointers.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_gated_delta_conv_silu_values_rows_ptrs_f32";
pub const op_name: []const u8 = "gated_delta_conv_silu_values_rows_ptrs_f32";

pub const embedded_symbol_advance: [:0]const u8 = "talu_gated_delta_advance_ring_heads_f32";
pub const op_name_advance: []const u8 = "gated_delta_advance_ring_heads_f32";

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    values: *const device_mod.Buffer,
    state_ptrs: *const device_mod.Buffer,
    positions: *const device_mod.Buffer,
    weight_time_major: *const device_mod.Buffer,
    bias: ?*const device_mod.Buffer,
    out: *device_mod.Buffer,
    conv_dim: u32,
    d_conv: u32,
    rows: u32,
    row_stride: u32,
) !void {
    try validateArgs(values, state_ptrs, positions, weight_time_major, bias, out, conv_dim, d_conv, rows, row_stride);

    arg_pack.reset();
    try arg_pack.appendBufferPtr(out);
    try arg_pack.appendBufferPtr(state_ptrs);
    try arg_pack.appendBufferPtr(positions);
    try arg_pack.appendBufferPtr(values);
    try arg_pack.appendBufferPtr(weight_time_major);
    try arg_pack.appendDevicePtr(if (bias) |buf| buf.pointer else 0);
    try arg_pack.appendScalar(u32, conv_dim);
    try arg_pack.appendScalar(u32, d_conv);
    try arg_pack.appendScalar(u32, if (bias != null) 1 else 0);
    try arg_pack.appendScalar(u32, rows);
    try arg_pack.appendScalar(u32, row_stride);

    const block_x: u32 = 256;
    try launch_mod.launchWithFamily(device, function, .{
        .grid_x = ceilDiv(conv_dim, block_x),
        .grid_y = rows,
        .block_x = block_x,
    }, arg_pack, .gated_delta);
}

pub fn runAdvanceWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    positions: *device_mod.Buffer,
    d_conv: u32,
    rows: u32,
) !void {
    if (d_conv == 0 or rows == 0) return error.InvalidArgument;
    const bytes = std.math.mul(usize, @as(usize, rows), @sizeOf(u32)) catch return error.InvalidArgument;
    if (positions.size < bytes) return error.InvalidArgument;

    arg_pack.reset();
    try arg_pack.appendBufferPtr(positions);
    try arg_pack.appendScalar(u32, d_conv);
    try arg_pack.appendScalar(u32, rows);
    const block_x: u32 = 256;
    try launch_mod.launchWithFamily(device, function, .{
        .grid_x = ceilDiv(rows, block_x),
        .block_x = block_x,
    }, arg_pack, .gated_delta);
}

fn validateArgs(
    values: *const device_mod.Buffer,
    state_ptrs: *const device_mod.Buffer,
    positions: *const device_mod.Buffer,
    weight_time_major: *const device_mod.Buffer,
    bias: ?*const device_mod.Buffer,
    out: *const device_mod.Buffer,
    conv_dim: u32,
    d_conv: u32,
    rows: u32,
    row_stride: u32,
) !void {
    if (conv_dim == 0 or d_conv == 0 or rows == 0) return error.InvalidArgument;
    if (row_stride < conv_dim) return error.InvalidArgument;

    const row_elems = std.math.mul(usize, @as(usize, rows), @as(usize, row_stride)) catch return error.InvalidArgument;
    const row_bytes = std.math.mul(usize, row_elems, @sizeOf(f32)) catch return error.InvalidArgument;
    if (values.size < row_bytes or out.size < row_bytes) return error.InvalidArgument;

    const ptr_bytes = std.math.mul(usize, @as(usize, rows), @sizeOf(u64)) catch return error.InvalidArgument;
    const position_bytes = std.math.mul(usize, @as(usize, rows), @sizeOf(u32)) catch return error.InvalidArgument;
    if (state_ptrs.size < ptr_bytes or positions.size < position_bytes) return error.InvalidArgument;

    const taps = std.math.mul(usize, @as(usize, conv_dim), @as(usize, d_conv)) catch return error.InvalidArgument;
    const taps_bytes = std.math.mul(usize, taps, @sizeOf(f32)) catch return error.InvalidArgument;
    if (weight_time_major.size < taps_bytes) return error.InvalidArgument;
    if (bias) |bias_buf| {
        const vec_bytes = std.math.mul(usize, @as(usize, conv_dim), @sizeOf(f32)) catch return error.InvalidArgument;
        if (bias_buf.size < vec_bytes) return error.InvalidArgument;
    }
}

fn ceilDiv(numerator: u32, denominator: u32) u32 {
    return (numerator + denominator - 1) / denominator;
}

test "validateArgs rejects invalid row shape" {
    const values = device_mod.Buffer{ .pointer = 0, .size = 128 * @sizeOf(f32) };
    const ptrs = device_mod.Buffer{ .pointer = 0, .size = 16 * @sizeOf(u64) };
    const positions = device_mod.Buffer{ .pointer = 0, .size = 16 * @sizeOf(u32) };
    const taps = device_mod.Buffer{ .pointer = 0, .size = 64 * @sizeOf(f32) };
    var out = values;
    try std.testing.expectError(
        error.InvalidArgument,
        validateArgs(&values, &ptrs, &positions, &taps, null, &out, 0, 2, 2, 8),
    );
    try std.testing.expectError(
        error.InvalidArgument,
        validateArgs(&values, &ptrs, &positions, &taps, null, &out, 8, 2, 2, 4),
    );
}
