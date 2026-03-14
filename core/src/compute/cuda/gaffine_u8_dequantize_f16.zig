//! Grouped-affine U8 weight dequantization to F16 kernel wrapper.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");
const gaffine = @import("gaffine_u8_matvec.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_gaffine_u8_dequantize_to_f16";
pub const op_name: []const u8 = "gaffine_u8_dequantize_to_f16";

const block_x: u32 = 256;

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    packed_weight: *const device_mod.Buffer,
    scales: *const device_mod.Buffer,
    biases: *const device_mod.Buffer,
    out_f16: *device_mod.Buffer,
    out_dim: u32,
    in_dim: u32,
    group_size: u32,
    scales_dtype_tag: u32,
) !void {
    try validateArgs(packed_weight, scales, biases, out_f16, out_dim, in_dim, group_size, scales_dtype_tag);

    arg_pack.reset();
    try arg_pack.appendBufferPtr(packed_weight);
    try arg_pack.appendBufferPtr(scales);
    try arg_pack.appendBufferPtr(biases);
    try arg_pack.appendBufferPtr(out_f16);
    try arg_pack.appendScalar(u32, out_dim);
    try arg_pack.appendScalar(u32, in_dim);
    try arg_pack.appendScalar(u32, group_size);
    try arg_pack.appendScalar(u32, scales_dtype_tag);

    try launch_mod.launchWithFamily(device, function, .{
        .grid_x = out_dim,
        .block_x = block_x,
    }, arg_pack, .other);
}

fn validateArgs(
    packed_weight: *const device_mod.Buffer,
    scales: *const device_mod.Buffer,
    biases: *const device_mod.Buffer,
    out_f16: *const device_mod.Buffer,
    out_dim: u32,
    in_dim: u32,
    group_size: u32,
    scales_dtype_tag: u32,
) !void {
    if (in_dim == 0 or out_dim == 0 or group_size == 0) return error.InvalidArgument;
    if ((in_dim % 16) != 0 or (group_size % 4) != 0) return error.InvalidArgument;
    if ((in_dim % group_size) != 0) return error.InvalidArgument;
    if (scales_dtype_tag != gaffine.scales_dtype_f16 and scales_dtype_tag != gaffine.scales_dtype_bf16) return error.InvalidArgument;

    const in_dim_usize: usize = @intCast(in_dim);
    const out_dim_usize: usize = @intCast(out_dim);
    const group_size_usize: usize = @intCast(group_size);
    const groups_per_row = in_dim_usize / group_size_usize;
    const packed_row_words = in_dim_usize / 4;

    const packed_bytes = std.math.mul(usize, std.math.mul(usize, out_dim_usize, packed_row_words) catch return error.InvalidArgument, @sizeOf(u32)) catch return error.InvalidArgument;
    const sb_bytes = std.math.mul(usize, std.math.mul(usize, out_dim_usize, groups_per_row) catch return error.InvalidArgument, @sizeOf(u16)) catch return error.InvalidArgument;
    const out_bytes = std.math.mul(usize, std.math.mul(usize, out_dim_usize, in_dim_usize) catch return error.InvalidArgument, @sizeOf(u16)) catch return error.InvalidArgument;

    if (packed_weight.size < packed_bytes or scales.size < sb_bytes or biases.size < sb_bytes or out_f16.size < out_bytes) {
        return error.InvalidArgument;
    }
}

test "validateArgs rejects zero dimensions" {
    const b = device_mod.Buffer{ .pointer = 0, .size = 8192 };
    var out = b;
    try std.testing.expectError(error.InvalidArgument, validateArgs(&b, &b, &b, &out, 0, 32, 16, gaffine.scales_dtype_f16));
    try std.testing.expectError(error.InvalidArgument, validateArgs(&b, &b, &b, &out, 8, 0, 16, gaffine.scales_dtype_f16));
    try std.testing.expectError(error.InvalidArgument, validateArgs(&b, &b, &b, &out, 8, 32, 0, gaffine.scales_dtype_f16));
}

test "validateArgs rejects invalid alignment" {
    const b = device_mod.Buffer{ .pointer = 0, .size = 8192 };
    var out = b;
    // in_dim must be multiple of 16 for 128-bit loads.
    try std.testing.expectError(error.InvalidArgument, validateArgs(&b, &b, &b, &out, 8, 12, 4, gaffine.scales_dtype_bf16));
    // group_size must be multiple of 4.
    try std.testing.expectError(error.InvalidArgument, validateArgs(&b, &b, &b, &out, 8, 32, 6, gaffine.scales_dtype_bf16));
}

test "validateArgs rejects invalid scales dtype tag" {
    const b = device_mod.Buffer{ .pointer = 0, .size = 8192 };
    var out = b;
    try std.testing.expectError(error.InvalidArgument, validateArgs(&b, &b, &b, &out, 8, 32, 16, 7));
}

test "validateArgs rejects undersized output buffer" {
    const b = device_mod.Buffer{ .pointer = 0, .size = 8192 };
    var small_out = device_mod.Buffer{ .pointer = 0, .size = 4 };
    try std.testing.expectError(error.InvalidArgument, validateArgs(&b, &b, &b, &small_out, 8, 32, 16, gaffine.scales_dtype_f16));
}
