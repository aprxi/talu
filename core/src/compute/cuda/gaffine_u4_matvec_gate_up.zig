//! Fused grouped-affine U4 gate/up matvec kernel wrapper.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");
const gaffine = @import("gaffine_u4_matvec.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_gaffine_u4_matvec_gate_up_f32";
pub const op_name: []const u8 = "gaffine_u4_matvec_gate_up_f32";
const warp_size: u32 = 32;
const block_x: u32 = 256;

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    input: *const device_mod.Buffer,
    gate_packed_weight: *const device_mod.Buffer,
    gate_scales: *const device_mod.Buffer,
    gate_biases: *const device_mod.Buffer,
    gate_out: *device_mod.Buffer,
    gate_out_dim: u32,
    gate_group_size: u32,
    gate_scales_dtype_tag: u32,
    up_packed_weight: *const device_mod.Buffer,
    up_scales: *const device_mod.Buffer,
    up_biases: *const device_mod.Buffer,
    up_out: *device_mod.Buffer,
    up_out_dim: u32,
    up_group_size: u32,
    up_scales_dtype_tag: u32,
    in_dim: u32,
) !void {
    try validateArgs(
        input,
        gate_packed_weight,
        gate_scales,
        gate_biases,
        gate_out,
        gate_out_dim,
        gate_group_size,
        gate_scales_dtype_tag,
        up_packed_weight,
        up_scales,
        up_biases,
        up_out,
        up_out_dim,
        up_group_size,
        up_scales_dtype_tag,
        in_dim,
    );

    arg_pack.reset();
    try arg_pack.appendBufferPtr(input);
    try arg_pack.appendBufferPtr(gate_packed_weight);
    try arg_pack.appendBufferPtr(gate_scales);
    try arg_pack.appendBufferPtr(gate_biases);
    try arg_pack.appendBufferPtr(gate_out);
    try arg_pack.appendScalar(u32, gate_out_dim);
    try arg_pack.appendScalar(u32, gate_group_size);
    try arg_pack.appendScalar(u32, gate_scales_dtype_tag);
    try arg_pack.appendBufferPtr(up_packed_weight);
    try arg_pack.appendBufferPtr(up_scales);
    try arg_pack.appendBufferPtr(up_biases);
    try arg_pack.appendBufferPtr(up_out);
    try arg_pack.appendScalar(u32, up_out_dim);
    try arg_pack.appendScalar(u32, up_group_size);
    try arg_pack.appendScalar(u32, up_scales_dtype_tag);
    try arg_pack.appendScalar(u32, in_dim);

    const total_out = std.math.add(u32, gate_out_dim, up_out_dim) catch return error.InvalidArgument;
    const rows_per_block = block_x / warp_size;
    try launch_mod.launch(device, function, .{
        .grid_x = ceilDiv(total_out, rows_per_block),
        .block_x = block_x,
    }, arg_pack);
}

fn validateArgs(
    input: *const device_mod.Buffer,
    gate_packed_weight: *const device_mod.Buffer,
    gate_scales: *const device_mod.Buffer,
    gate_biases: *const device_mod.Buffer,
    gate_out: *device_mod.Buffer,
    gate_out_dim: u32,
    gate_group_size: u32,
    gate_scales_dtype_tag: u32,
    up_packed_weight: *const device_mod.Buffer,
    up_scales: *const device_mod.Buffer,
    up_biases: *const device_mod.Buffer,
    up_out: *device_mod.Buffer,
    up_out_dim: u32,
    up_group_size: u32,
    up_scales_dtype_tag: u32,
    in_dim: u32,
) !void {
    try validateOne(
        input,
        gate_packed_weight,
        gate_scales,
        gate_biases,
        gate_out,
        in_dim,
        gate_out_dim,
        gate_group_size,
        gate_scales_dtype_tag,
    );
    try validateOne(
        input,
        up_packed_weight,
        up_scales,
        up_biases,
        up_out,
        in_dim,
        up_out_dim,
        up_group_size,
        up_scales_dtype_tag,
    );
}

fn validateOne(
    input: *const device_mod.Buffer,
    packed_weight: *const device_mod.Buffer,
    scales: *const device_mod.Buffer,
    biases: *const device_mod.Buffer,
    out: *device_mod.Buffer,
    in_dim: u32,
    out_dim: u32,
    group_size: u32,
    scales_dtype_tag: u32,
) !void {
    if (in_dim == 0 or out_dim == 0 or group_size == 0) return error.InvalidArgument;
    if ((in_dim % 8) != 0 or (group_size % 8) != 0) return error.InvalidArgument;
    if ((in_dim % group_size) != 0) return error.InvalidArgument;
    if (scales_dtype_tag != gaffine.scales_dtype_f16 and scales_dtype_tag != gaffine.scales_dtype_bf16) {
        return error.InvalidArgument;
    }

    const in_dim_usize: usize = @intCast(in_dim);
    const out_dim_usize: usize = @intCast(out_dim);
    const group_size_usize: usize = @intCast(group_size);
    const groups_per_row = in_dim_usize / group_size_usize;
    const packed_row_words = in_dim_usize / 8;

    const input_bytes = std.math.mul(usize, in_dim_usize, @sizeOf(f32)) catch return error.InvalidArgument;
    const out_bytes = std.math.mul(usize, out_dim_usize, @sizeOf(f32)) catch return error.InvalidArgument;
    const packed_words = std.math.mul(usize, out_dim_usize, packed_row_words) catch return error.InvalidArgument;
    const packed_bytes = std.math.mul(usize, packed_words, @sizeOf(u32)) catch return error.InvalidArgument;
    const sb_count = std.math.mul(usize, out_dim_usize, groups_per_row) catch return error.InvalidArgument;
    const sb_bytes = std.math.mul(usize, sb_count, @sizeOf(u16)) catch return error.InvalidArgument;

    if (input.size < input_bytes or
        out.size < out_bytes or
        packed_weight.size < packed_bytes or
        scales.size < sb_bytes or
        biases.size < sb_bytes)
    {
        return error.InvalidArgument;
    }
}

fn ceilDiv(numerator: u32, denominator: u32) u32 {
    return (numerator + denominator - 1) / denominator;
}

test "validateArgs rejects invalid group alignment" {
    const b = device_mod.Buffer{ .pointer = 0, .size = 8192 };
    var out = b;
    try std.testing.expectError(
        error.InvalidArgument,
        validateArgs(
            &b,
            &b,
            &b,
            &b,
            &out,
            8,
            12,
            gaffine.scales_dtype_bf16,
            &b,
            &b,
            &b,
            &out,
            8,
            8,
            gaffine.scales_dtype_bf16,
            16,
        ),
    );
}

test "ceilDiv computes expected block count" {
    try std.testing.expectEqual(@as(u32, 1), ceilDiv(1, 256));
    try std.testing.expectEqual(@as(u32, 2), ceilDiv(257, 256));
}
