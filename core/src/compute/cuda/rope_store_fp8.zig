//! RoPE + fp8 cache-store kernel wrapper.
//! Per-head symmetric FP8 E4M3 quantization with scale = max_abs / 448.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_rope_store_fp8";
pub const op_name: []const u8 = "rope_store_fp8";

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    input_f32: *const device_mod.Buffer,
    out_fp8: *device_mod.Buffer,
    scales: *device_mod.Buffer,
    n_heads: u32,
    head_dim: u32,
    rope_dim: u32,
    position: u32,
    theta: f32,
) !void {
    try validateArgs(input_f32, out_fp8, scales, n_heads, head_dim, rope_dim, theta);

    arg_pack.reset();
    try arg_pack.appendBufferPtr(out_fp8);
    try arg_pack.appendBufferPtr(scales);
    try arg_pack.appendBufferPtr(input_f32);
    try arg_pack.appendScalar(u32, n_heads);
    try arg_pack.appendScalar(u32, head_dim);
    try arg_pack.appendScalar(u32, rope_dim);
    try arg_pack.appendScalar(u32, position);
    try arg_pack.appendScalar(f32, theta);

    // One block per head for per-head max_abs reduction + quantization.
    try launch_mod.launchWithFamily(device, function, .{
        .grid_x = n_heads,
        .block_x = 256,
    }, arg_pack, .rope);
}

fn validateArgs(
    input_f32: *const device_mod.Buffer,
    out_fp8: *device_mod.Buffer,
    scales: *device_mod.Buffer,
    n_heads: u32,
    head_dim: u32,
    rope_dim: u32,
    theta: f32,
) !void {
    if (n_heads == 0 or head_dim == 0 or rope_dim == 0) return error.InvalidArgument;
    if (rope_dim > head_dim or (rope_dim & 1) != 0) return error.InvalidArgument;
    if (!std.math.isFinite(theta) or theta <= 1.0) return error.InvalidArgument;

    const count = std.math.mul(usize, @as(usize, n_heads), @as(usize, head_dim)) catch return error.InvalidArgument;
    const input_bytes = std.math.mul(usize, count, @sizeOf(f32)) catch return error.InvalidArgument;
    const out_bytes = count; // 1 byte per fp8 element
    const scale_bytes = std.math.mul(usize, @as(usize, n_heads), @sizeOf(f32)) catch return error.InvalidArgument;
    if (input_f32.size < input_bytes or out_fp8.size < out_bytes or scales.size < scale_bytes) return error.InvalidArgument;
}

test "validateArgs rejects invalid rope dimensions" {
    const in = device_mod.Buffer{ .pointer = 0, .size = 4096 };
    var out = device_mod.Buffer{ .pointer = 0, .size = 2048 };
    var scales = device_mod.Buffer{ .pointer = 0, .size = 256 };
    try std.testing.expectError(error.InvalidArgument, validateArgs(&in, &out, &scales, 1, 8, 9, 10000.0));
    try std.testing.expectError(error.InvalidArgument, validateArgs(&in, &out, &scales, 1, 8, 10, 10000.0));
}

test "validateArgs rejects undersized output buffer" {
    const in = device_mod.Buffer{ .pointer = 0, .size = 8 * @sizeOf(f32) };
    var out = device_mod.Buffer{ .pointer = 0, .size = 4 };
    var scales = device_mod.Buffer{ .pointer = 0, .size = @sizeOf(f32) };
    try std.testing.expectError(error.InvalidArgument, validateArgs(&in, &out, &scales, 1, 8, 8, 10000.0));
}
