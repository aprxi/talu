//! Dequantize i8 K/V cache rows to f16 using per-token-per-head scales.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_dequant_kv_i8_to_f16";
pub const op_name: []const u8 = "dequant_kv_i8_to_f16";

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    key_cache_i8: *const device_mod.Buffer,
    value_cache_i8: *const device_mod.Buffer,
    k_scales: *const device_mod.Buffer,
    v_scales: *const device_mod.Buffer,
    out_key_f16: *device_mod.Buffer,
    out_value_f16: *device_mod.Buffer,
    seq_len: u32,
    n_kv_heads: u32,
    row_stride: u32,
    head_dim: u32,
) !void {
    try validateArgs(
        key_cache_i8,
        value_cache_i8,
        k_scales,
        v_scales,
        out_key_f16,
        out_value_f16,
        seq_len,
        n_kv_heads,
        row_stride,
        head_dim,
    );

    const total_elems = std.math.mul(u32, seq_len, row_stride) catch return error.InvalidArgument;

    arg_pack.reset();
    try arg_pack.appendBufferPtr(out_key_f16);
    try arg_pack.appendBufferPtr(out_value_f16);
    try arg_pack.appendBufferPtr(key_cache_i8);
    try arg_pack.appendBufferPtr(value_cache_i8);
    try arg_pack.appendBufferPtr(k_scales);
    try arg_pack.appendBufferPtr(v_scales);
    try arg_pack.appendScalar(u32, seq_len);
    try arg_pack.appendScalar(u32, n_kv_heads);
    try arg_pack.appendScalar(u32, row_stride);
    try arg_pack.appendScalar(u32, head_dim);

    const block_x: u32 = 256;
    try launch_mod.launchWithFamily(device, function, .{
        .grid_x = ceilDiv(total_elems, block_x),
        .block_x = block_x,
    }, arg_pack, .copy_cast);
}

fn validateArgs(
    key_cache_i8: *const device_mod.Buffer,
    value_cache_i8: *const device_mod.Buffer,
    k_scales: *const device_mod.Buffer,
    v_scales: *const device_mod.Buffer,
    out_key_f16: *device_mod.Buffer,
    out_value_f16: *device_mod.Buffer,
    seq_len: u32,
    n_kv_heads: u32,
    row_stride: u32,
    head_dim: u32,
) !void {
    if (seq_len == 0 or n_kv_heads == 0 or row_stride == 0 or head_dim == 0) return error.InvalidArgument;

    const min_row_stride = std.math.mul(u32, n_kv_heads, head_dim) catch return error.InvalidArgument;
    if (row_stride < min_row_stride) return error.InvalidArgument;

    const cache_elems = std.math.mul(usize, @as(usize, seq_len), @as(usize, row_stride)) catch return error.InvalidArgument;
    const cache_bytes = cache_elems; // i8
    const f16_bytes = std.math.mul(usize, cache_elems, @sizeOf(u16)) catch return error.InvalidArgument;
    const scale_elems = std.math.mul(usize, @as(usize, seq_len), @as(usize, n_kv_heads)) catch return error.InvalidArgument;
    const scale_bytes = std.math.mul(usize, scale_elems, @sizeOf(f32)) catch return error.InvalidArgument;

    if (key_cache_i8.size < cache_bytes or value_cache_i8.size < cache_bytes) return error.InvalidArgument;
    if (k_scales.size < scale_bytes or v_scales.size < scale_bytes) return error.InvalidArgument;
    if (out_key_f16.size < f16_bytes or out_value_f16.size < f16_bytes) return error.InvalidArgument;
}

fn ceilDiv(numerator: u32, denominator: u32) u32 {
    return (numerator + denominator - 1) / denominator;
}

test "validateArgs rejects invalid geometry" {
    const src = device_mod.Buffer{ .pointer = 0, .size = 1024 };
    const scales = device_mod.Buffer{ .pointer = 0, .size = 128 };
    var dst = device_mod.Buffer{ .pointer = 0, .size = 2048 };
    try std.testing.expectError(error.InvalidArgument, validateArgs(&src, &src, &scales, &scales, &dst, &dst, 0, 1, 1, 1));
    try std.testing.expectError(error.InvalidArgument, validateArgs(&src, &src, &scales, &scales, &dst, &dst, 8, 2, 7, 4));
}
