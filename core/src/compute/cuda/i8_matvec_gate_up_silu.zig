//! Fused i8 gate/up + SiLU matvec kernel wrapper.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_i8_matvec_gate_up_silu_f32";
pub const op_name: []const u8 = "i8_matvec_gate_up_silu_f32";
pub const embedded_symbol_tile8: [:0]const u8 = "talu_i8_matvec_gate_up_silu_f32_tile8";
pub const op_name_tile8: []const u8 = "i8_matvec_gate_up_silu_f32_tile8";
const warp_size: u32 = 32;
const block_x: u32 = 128;
const inner_batch_rows: u32 = 4;
const inner_batch_rows_tile8: u32 = 8;

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    input: *const device_mod.Buffer,
    gate_weight: *const device_mod.Buffer,
    gate_scales: *const device_mod.Buffer,
    up_weight: *const device_mod.Buffer,
    up_scales: *const device_mod.Buffer,
    out: *device_mod.Buffer,
    out_dim: u32,
    in_dim: u32,
    batch_rows: u32,
) !void {
    return runWithFunctionBatchTile(
        arg_pack,
        device,
        function,
        input,
        gate_weight,
        gate_scales,
        up_weight,
        up_scales,
        out,
        out_dim,
        in_dim,
        batch_rows,
        inner_batch_rows,
    );
}

pub fn runWithFunctionTile8(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    input: *const device_mod.Buffer,
    gate_weight: *const device_mod.Buffer,
    gate_scales: *const device_mod.Buffer,
    up_weight: *const device_mod.Buffer,
    up_scales: *const device_mod.Buffer,
    out: *device_mod.Buffer,
    out_dim: u32,
    in_dim: u32,
    batch_rows: u32,
) !void {
    return runWithFunctionBatchTile(
        arg_pack,
        device,
        function,
        input,
        gate_weight,
        gate_scales,
        up_weight,
        up_scales,
        out,
        out_dim,
        in_dim,
        batch_rows,
        inner_batch_rows_tile8,
    );
}

fn runWithFunctionBatchTile(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    input: *const device_mod.Buffer,
    gate_weight: *const device_mod.Buffer,
    gate_scales: *const device_mod.Buffer,
    up_weight: *const device_mod.Buffer,
    up_scales: *const device_mod.Buffer,
    out: *device_mod.Buffer,
    out_dim: u32,
    in_dim: u32,
    batch_rows: u32,
    batch_tile_rows: u32,
) !void {
    try validateArgs(input, gate_weight, gate_scales, up_weight, up_scales, out, out_dim, in_dim, batch_rows);
    if (batch_tile_rows == 0) return error.InvalidArgument;

    arg_pack.reset();
    try arg_pack.appendBufferPtr(input);
    try arg_pack.appendBufferPtr(gate_weight);
    try arg_pack.appendBufferPtr(gate_scales);
    try arg_pack.appendBufferPtr(up_weight);
    try arg_pack.appendBufferPtr(up_scales);
    try arg_pack.appendBufferPtr(out);
    try arg_pack.appendScalar(u32, out_dim);
    try arg_pack.appendScalar(u32, in_dim);
    try arg_pack.appendScalar(u32, batch_rows);

    const rows_per_block = block_x / warp_size;
    try launch_mod.launchWithFamily(device, function, .{
        .grid_x = ceilDiv(out_dim, rows_per_block),
        .grid_y = ceilDiv(batch_rows, batch_tile_rows),
        .block_x = block_x,
    }, arg_pack, .matvec_gate_up_silu);
}

fn validateArgs(
    input: *const device_mod.Buffer,
    gate_weight: *const device_mod.Buffer,
    gate_scales: *const device_mod.Buffer,
    up_weight: *const device_mod.Buffer,
    up_scales: *const device_mod.Buffer,
    out: *const device_mod.Buffer,
    out_dim: u32,
    in_dim: u32,
    batch_rows: u32,
) !void {
    if (in_dim == 0 or out_dim == 0 or batch_rows == 0) return error.InvalidArgument;

    const batch: usize = @intCast(batch_rows);
    const in_dim_usize: usize = @intCast(in_dim);
    const out_dim_usize: usize = @intCast(out_dim);
    const input_bytes = std.math.mul(usize, batch * in_dim_usize, @sizeOf(f32)) catch return error.InvalidArgument;
    const out_bytes = std.math.mul(usize, batch * out_dim_usize, @sizeOf(f32)) catch return error.InvalidArgument;
    const weight_bytes = std.math.mul(usize, out_dim_usize * in_dim_usize, @sizeOf(i8)) catch return error.InvalidArgument;
    const scale_bytes = std.math.mul(usize, out_dim_usize, @sizeOf(f32)) catch return error.InvalidArgument;

    if (input.size < input_bytes or
        out.size < out_bytes or
        gate_weight.size < weight_bytes or
        gate_scales.size < scale_bytes or
        up_weight.size < weight_bytes or
        up_scales.size < scale_bytes)
    {
        return error.InvalidArgument;
    }
}

fn ceilDiv(numerator: u32, denominator: u32) u32 {
    return (numerator + denominator - 1) / denominator;
}

test "validateArgs rejects zero dimensions" {
    const b = device_mod.Buffer{ .pointer = 0, .size = 4096 };
    var out = b;
    try std.testing.expectError(error.InvalidArgument, validateArgs(&b, &b, &b, &b, &b, &out, 0, 4, 1));
    try std.testing.expectError(error.InvalidArgument, validateArgs(&b, &b, &b, &b, &b, &out, 4, 0, 1));
    try std.testing.expectError(error.InvalidArgument, validateArgs(&b, &b, &b, &b, &b, &out, 4, 4, 0));
}
