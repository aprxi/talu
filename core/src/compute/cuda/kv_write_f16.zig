//! K/V cache write kernel wrapper (K: RoPE+store f16, V: cast+store f16).

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_kv_write_f16";
pub const op_name: []const u8 = "kv_write_f16";

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    input_k_f32: *const device_mod.Buffer,
    input_v_f32: *const device_mod.Buffer,
    out_k_f16: *device_mod.Buffer,
    out_v_f16: *device_mod.Buffer,
    n_heads: u32,
    head_dim: u32,
    rope_dim: u32,
    position: u32,
    theta: f32,
) !void {
    try validateArgs(
        input_k_f32,
        input_v_f32,
        out_k_f16,
        out_v_f16,
        n_heads,
        head_dim,
        rope_dim,
        theta,
    );

    arg_pack.reset();
    try arg_pack.appendBufferPtr(out_k_f16);
    try arg_pack.appendBufferPtr(out_v_f16);
    try arg_pack.appendBufferPtr(input_k_f32);
    try arg_pack.appendBufferPtr(input_v_f32);
    try arg_pack.appendScalar(u32, n_heads);
    try arg_pack.appendScalar(u32, head_dim);
    try arg_pack.appendScalar(u32, rope_dim);
    try arg_pack.appendScalar(u32, position);
    try arg_pack.appendScalar(f32, theta);

    const count = std.math.mul(u32, n_heads, head_dim) catch return error.InvalidArgument;
    const block_x: u32 = 256;
    try launch_mod.launch(device, function, .{
        .grid_x = ceilDiv(count, block_x),
        .block_x = block_x,
    }, arg_pack);
}

fn validateArgs(
    input_k_f32: *const device_mod.Buffer,
    input_v_f32: *const device_mod.Buffer,
    out_k_f16: *device_mod.Buffer,
    out_v_f16: *device_mod.Buffer,
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
    const out_bytes = std.math.mul(usize, count, @sizeOf(u16)) catch return error.InvalidArgument;
    if (input_k_f32.size < input_bytes or input_v_f32.size < input_bytes) return error.InvalidArgument;
    if (out_k_f16.size < out_bytes or out_v_f16.size < out_bytes) return error.InvalidArgument;
}

fn ceilDiv(numerator: u32, denominator: u32) u32 {
    return (numerator + denominator - 1) / denominator;
}

test "validateArgs rejects invalid rope dims" {
    const input = device_mod.Buffer{ .pointer = 0, .size = 4096 };
    var output = device_mod.Buffer{ .pointer = 0, .size = 2048 };
    try std.testing.expectError(error.InvalidArgument, validateArgs(&input, &input, &output, &output, 1, 8, 9, 10000.0));
    try std.testing.expectError(error.InvalidArgument, validateArgs(&input, &input, &output, &output, 1, 8, 10, 10000.0));
}

test "validateArgs rejects undersized output" {
    const input = device_mod.Buffer{ .pointer = 0, .size = 8 * @sizeOf(f32) };
    var output = device_mod.Buffer{ .pointer = 0, .size = 8 };
    try std.testing.expectError(error.InvalidArgument, validateArgs(&input, &input, &output, &output, 1, 8, 8, 10000.0));
}
