//! Batched K/V cache write kernel wrapper (K: RoPE+store fp8, V: cast+store fp8).
//! Per-head-per-token symmetric quantization with scale = max_abs / 448.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_kv_write_fp8_rows";
pub const op_name: []const u8 = "kv_write_fp8_rows";

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    input_k_f32: *const device_mod.Buffer,
    input_v_f32: *const device_mod.Buffer,
    out_k_fp8: *device_mod.Buffer,
    out_v_fp8: *device_mod.Buffer,
    k_scales: *device_mod.Buffer,
    v_scales: *device_mod.Buffer,
    n_heads: u32,
    head_dim: u32,
    rope_dim: u32,
    q_rows: u32,
    row_stride: u32,
    position_base: u32,
    theta: f32,
) !void {
    try validateArgs(
        input_k_f32,
        input_v_f32,
        out_k_fp8,
        out_v_fp8,
        k_scales,
        v_scales,
        n_heads,
        head_dim,
        rope_dim,
        q_rows,
        row_stride,
        theta,
    );

    arg_pack.reset();
    try arg_pack.appendBufferPtr(out_k_fp8);
    try arg_pack.appendBufferPtr(out_v_fp8);
    try arg_pack.appendBufferPtr(k_scales);
    try arg_pack.appendBufferPtr(v_scales);
    try arg_pack.appendBufferPtr(input_k_f32);
    try arg_pack.appendBufferPtr(input_v_f32);
    try arg_pack.appendScalar(u32, n_heads);
    try arg_pack.appendScalar(u32, head_dim);
    try arg_pack.appendScalar(u32, rope_dim);
    try arg_pack.appendScalar(u32, q_rows);
    try arg_pack.appendScalar(u32, row_stride);
    try arg_pack.appendScalar(u32, position_base);
    try arg_pack.appendScalar(f32, theta);

    // Grid: (n_heads, q_rows) — one block per head per row.
    try launch_mod.launchWithFamily(device, function, .{
        .grid_x = n_heads,
        .grid_y = q_rows,
        .block_x = 256,
    }, arg_pack, .kv_write);
}

fn validateArgs(
    input_k_f32: *const device_mod.Buffer,
    input_v_f32: *const device_mod.Buffer,
    out_k_fp8: *device_mod.Buffer,
    out_v_fp8: *device_mod.Buffer,
    k_scales: *device_mod.Buffer,
    v_scales: *device_mod.Buffer,
    n_heads: u32,
    head_dim: u32,
    rope_dim: u32,
    q_rows: u32,
    row_stride: u32,
    theta: f32,
) !void {
    if (n_heads == 0 or head_dim == 0 or rope_dim == 0 or q_rows == 0 or row_stride == 0) return error.InvalidArgument;
    if (rope_dim > head_dim or (rope_dim & 1) != 0) return error.InvalidArgument;
    if (!std.math.isFinite(theta) or theta <= 1.0) return error.InvalidArgument;

    const row_width = std.math.mul(u32, n_heads, head_dim) catch return error.InvalidArgument;
    if (row_stride < row_width) return error.InvalidArgument;

    const input_count = std.math.mul(usize, @as(usize, q_rows), @as(usize, row_width)) catch return error.InvalidArgument;
    const input_bytes = std.math.mul(usize, input_count, @sizeOf(f32)) catch return error.InvalidArgument;
    // Output: fp8 (1 byte per element), with row_stride spacing.
    const output_count = std.math.mul(usize, @as(usize, q_rows), @as(usize, row_stride)) catch return error.InvalidArgument;
    // Scale: one f32 per head per row.
    const scale_count = std.math.mul(usize, @as(usize, q_rows), @as(usize, n_heads)) catch return error.InvalidArgument;
    const scale_bytes = std.math.mul(usize, scale_count, @sizeOf(f32)) catch return error.InvalidArgument;
    if (input_k_f32.size < input_bytes or input_v_f32.size < input_bytes) return error.InvalidArgument;
    if (out_k_fp8.size < output_count or out_v_fp8.size < output_count) return error.InvalidArgument;
    if (k_scales.size < scale_bytes or v_scales.size < scale_bytes) return error.InvalidArgument;
}

fn ceilDiv(numerator: u32, denominator: u32) u32 {
    return (numerator + denominator - 1) / denominator;
}

test "validateArgs rejects invalid row shape" {
    const input = device_mod.Buffer{ .pointer = 0, .size = 4096 };
    var output = device_mod.Buffer{ .pointer = 0, .size = 4096 };
    var scales = device_mod.Buffer{ .pointer = 0, .size = 256 };
    try std.testing.expectError(error.InvalidArgument, validateArgs(&input, &input, &output, &output, &scales, &scales, 2, 8, 8, 0, 16, 10000.0));
    try std.testing.expectError(error.InvalidArgument, validateArgs(&input, &input, &output, &output, &scales, &scales, 2, 8, 8, 2, 15, 10000.0));
}
