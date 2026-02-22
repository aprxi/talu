//! Dense u16-weight (f16/bf16) matvec kernel wrapper.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_matvec_u16_f32";
pub const dtype_f16: u32 = 0;
pub const dtype_bf16: u32 = 1;

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    input: *const device_mod.Buffer,
    weight_u16: *const device_mod.Buffer,
    out: *device_mod.Buffer,
    in_dim: u32,
    out_dim: u32,
    weight_dtype_tag: u32,
) !void {
    try validateArgs(input, weight_u16, out, in_dim, out_dim, weight_dtype_tag);

    arg_pack.reset();
    try arg_pack.appendBufferPtr(input);
    try arg_pack.appendBufferPtr(weight_u16);
    try arg_pack.appendBufferPtr(out);
    try arg_pack.appendScalar(u32, in_dim);
    try arg_pack.appendScalar(u32, out_dim);
    try arg_pack.appendScalar(u32, weight_dtype_tag);

    const block_x: u32 = 256;
    const shared_mem_bytes: u32 = block_x * @sizeOf(f32);
    try launch_mod.launch(device, function, .{
        .grid_x = ceilDiv(out_dim, block_x),
        .block_x = block_x,
        .shared_mem_bytes = shared_mem_bytes,
    }, arg_pack);
}

fn validateArgs(
    input: *const device_mod.Buffer,
    weight_u16: *const device_mod.Buffer,
    out: *device_mod.Buffer,
    in_dim: u32,
    out_dim: u32,
    weight_dtype_tag: u32,
) !void {
    if (in_dim == 0 or out_dim == 0) return error.InvalidArgument;
    if (weight_dtype_tag != dtype_f16 and weight_dtype_tag != dtype_bf16) return error.InvalidArgument;

    const input_bytes = std.math.mul(usize, @as(usize, in_dim), @sizeOf(f32)) catch return error.InvalidArgument;
    const weight_elems = std.math.mul(usize, @as(usize, in_dim), @as(usize, out_dim)) catch return error.InvalidArgument;
    const weight_bytes = std.math.mul(usize, weight_elems, @sizeOf(u16)) catch return error.InvalidArgument;
    const out_bytes = std.math.mul(usize, @as(usize, out_dim), @sizeOf(f32)) catch return error.InvalidArgument;
    if (input.size < input_bytes or weight_u16.size < weight_bytes or out.size < out_bytes) return error.InvalidArgument;
}

fn ceilDiv(numerator: u32, denominator: u32) u32 {
    return (numerator + denominator - 1) / denominator;
}

test "validateArgs rejects invalid dtype tag" {
    const b = device_mod.Buffer{ .pointer = 0, .size = 1024 };
    var out = b;
    try std.testing.expectError(error.InvalidArgument, validateArgs(&b, &b, &out, 4, 4, 3));
}
