//! Gated-delta state-space rows kernel wrapper with int8 state + f32 scales.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_gated_delta_ssm_rows_i8_f32";
pub const op_name: []const u8 = "gated_delta_ssm_rows_i8_f32";
const out_tile: u32 = 32;
const warp_size: u32 = 32;

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    qkv_rows: *const device_mod.Buffer,
    a_log: *const device_mod.Buffer,
    dt_bias: ?*const device_mod.Buffer,
    state: *device_mod.Buffer,
    out: *device_mod.Buffer,
    n_qk_heads: u32,
    n_v_heads: u32,
    d_head: u32,
    rows: u32,
    row_stride: u32,
    beta_offset: u32,
    a_offset: u32,
    out_row_stride: u32,
    state_scales_offset: u32,
) !void {
    try validateArgs(qkv_rows, a_log, dt_bias, state, out, n_qk_heads, n_v_heads, d_head, rows, row_stride, beta_offset, a_offset, out_row_stride, state_scales_offset);

    arg_pack.reset();
    try arg_pack.appendBufferPtr(out);
    try arg_pack.appendBufferPtr(state);
    try arg_pack.appendBufferPtr(qkv_rows);
    try arg_pack.appendBufferPtr(a_log);
    try arg_pack.appendDevicePtr(if (dt_bias) |buf| buf.pointer else 0);
    try arg_pack.appendScalar(u32, n_qk_heads);
    try arg_pack.appendScalar(u32, n_v_heads);
    try arg_pack.appendScalar(u32, d_head);
    try arg_pack.appendScalar(u32, if (dt_bias != null) 1 else 0);
    try arg_pack.appendScalar(u32, rows);
    try arg_pack.appendScalar(u32, row_stride);
    try arg_pack.appendScalar(u32, beta_offset);
    try arg_pack.appendScalar(u32, a_offset);
    try arg_pack.appendScalar(u32, out_row_stride);
    try arg_pack.appendScalar(u32, state_scales_offset);

    const block_x = blockSizeForDHead(d_head);
    const shared_bytes = std.math.mul(usize, 2 * @as(usize, d_head), @sizeOf(f32)) catch return error.InvalidArgument;
    const tiles_per_head = ceilDiv(d_head, out_tile);
    try launch_mod.launchWithFamily(device, function, .{
        .grid_x = std.math.mul(u32, n_v_heads, tiles_per_head) catch return error.InvalidArgument,
        .block_x = block_x,
        .shared_mem_bytes = @intCast(shared_bytes),
    }, arg_pack, .gated_delta);
}

fn validateArgs(
    qkv_rows: *const device_mod.Buffer,
    a_log: *const device_mod.Buffer,
    dt_bias: ?*const device_mod.Buffer,
    state: *const device_mod.Buffer,
    out: *const device_mod.Buffer,
    n_qk_heads: u32,
    n_v_heads: u32,
    d_head: u32,
    rows: u32,
    row_stride: u32,
    beta_offset: u32,
    a_offset: u32,
    out_row_stride: u32,
    state_scales_offset: u32,
) !void {
    if (n_qk_heads == 0 or n_v_heads == 0 or d_head == 0 or rows == 0) return error.InvalidArgument;
    if ((n_v_heads % n_qk_heads) != 0) return error.InvalidArgument;
    if (row_stride == 0 or out_row_stride == 0) return error.InvalidArgument;
    if ((state_scales_offset % @as(u32, @intCast(@sizeOf(f32)))) != 0) return error.InvalidArgument;

    const qk_inner = std.math.mul(usize, @as(usize, n_qk_heads), @as(usize, d_head)) catch return error.InvalidArgument;
    const d_inner = std.math.mul(usize, @as(usize, n_v_heads), @as(usize, d_head)) catch return error.InvalidArgument;
    const qkv_len = std.math.add(usize, std.math.mul(usize, 2, qk_inner) catch return error.InvalidArgument, d_inner) catch return error.InvalidArgument;
    if (row_stride < qkv_len) return error.InvalidArgument;
    if (beta_offset + n_v_heads > row_stride or a_offset + n_v_heads > row_stride) return error.InvalidArgument;
    if (out_row_stride < d_inner) return error.InvalidArgument;

    const qkv_rows_elems = std.math.mul(usize, @as(usize, rows), @as(usize, row_stride)) catch return error.InvalidArgument;
    const qkv_rows_bytes = std.math.mul(usize, qkv_rows_elems, @sizeOf(f32)) catch return error.InvalidArgument;
    if (qkv_rows.size < qkv_rows_bytes) return error.InvalidArgument;

    const out_rows_elems = std.math.mul(usize, @as(usize, rows), @as(usize, out_row_stride)) catch return error.InvalidArgument;
    const out_rows_bytes = std.math.mul(usize, out_rows_elems, @sizeOf(f32)) catch return error.InvalidArgument;
    if (out.size < out_rows_bytes) return error.InvalidArgument;

    const head_bytes = std.math.mul(usize, @as(usize, n_v_heads), @sizeOf(f32)) catch return error.InvalidArgument;
    if (a_log.size < head_bytes) return error.InvalidArgument;
    if (dt_bias) |bias_buf| {
        if (bias_buf.size < head_bytes) return error.InvalidArgument;
    }

    const state_i8_bytes = std.math.mul(usize, d_inner, @as(usize, d_head)) catch return error.InvalidArgument;
    const state_scales_offset_usize: usize = @intCast(state_scales_offset);
    if (state_scales_offset_usize < state_i8_bytes) return error.InvalidArgument;
    const scale_bytes = std.math.mul(usize, d_inner, @sizeOf(f32)) catch return error.InvalidArgument;
    const required_state_bytes = std.math.add(usize, state_scales_offset_usize, scale_bytes) catch return error.InvalidArgument;
    if (state.size < required_state_bytes) return error.InvalidArgument;
}

fn ceilDiv(numerator: u32, denominator: u32) u32 {
    return (numerator + denominator - 1) / denominator;
}

fn blockSizeForDHead(d_head: u32) u32 {
    _ = d_head;
    return warp_size * 4;
}

test "validateArgs rejects invalid state scale offset" {
    const buf = device_mod.Buffer{ .pointer = 0, .size = 4096 };
    try std.testing.expectError(error.InvalidArgument, validateArgs(&buf, &buf, null, &buf, &buf, 1, 2, 8, 2, 16, 8, 10, 16, 258));
}

test "blockSizeForDHead rounds up to full warps" {
    try std.testing.expectEqual(@as(u32, 128), blockSizeForDHead(1));
    try std.testing.expectEqual(@as(u32, 128), blockSizeForDHead(32));
    try std.testing.expectEqual(@as(u32, 128), blockSizeForDHead(80));
    try std.testing.expectEqual(@as(u32, 128), blockSizeForDHead(128));
}
