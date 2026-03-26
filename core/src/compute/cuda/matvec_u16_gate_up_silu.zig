//! Dense u16-weight (f16/bf16) gate/up matvec + SiLU fusion wrapper.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol_f16: [:0]const u8 = "talu_matvec_gate_up_silu_f16_f32_batch";
pub const embedded_symbol_bf16: [:0]const u8 = "talu_matvec_gate_up_silu_bf16_f32_batch";
pub const op_name_f16: []const u8 = "matvec_gate_up_silu_f16_f32";
pub const op_name_bf16: []const u8 = "matvec_gate_up_silu_bf16_f32";
const warp_size: u32 = 32;
const block_x: u32 = 256;
const inner_batch_rows: u32 = 8;

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    input: *const device_mod.Buffer,
    gate_weight_u16: *const device_mod.Buffer,
    up_weight_u16: *const device_mod.Buffer,
    out: *device_mod.Buffer,
    out_dim: u32,
    in_dim: u32,
    batch_rows: u32,
) !void {
    try validateArgs(input, gate_weight_u16, up_weight_u16, out, out_dim, in_dim, batch_rows);

    arg_pack.reset();
    try arg_pack.appendBufferPtr(input);
    try arg_pack.appendBufferPtr(gate_weight_u16);
    try arg_pack.appendBufferPtr(up_weight_u16);
    try arg_pack.appendBufferPtr(out);
    try arg_pack.appendScalar(u32, out_dim);
    try arg_pack.appendScalar(u32, in_dim);
    try arg_pack.appendScalar(u32, batch_rows);

    const rows_per_block = block_x / warp_size;
    try launch_mod.launchWithFamily(device, function, .{
        .grid_x = ceilDiv(out_dim, rows_per_block),
        .grid_y = ceilDiv(batch_rows, inner_batch_rows),
        .block_x = block_x,
    }, arg_pack, .matvec_gate_up_silu);
}

/// Canonical batched launch: rows are mapped to grid_y.
pub fn runWithFunctionGridBatch(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    input: *const device_mod.Buffer,
    gate_weight_u16: *const device_mod.Buffer,
    up_weight_u16: *const device_mod.Buffer,
    out: *device_mod.Buffer,
    out_dim: u32,
    in_dim: u32,
    batch_rows: u32,
) !void {
    try validateArgs(input, gate_weight_u16, up_weight_u16, out, out_dim, in_dim, batch_rows);

    arg_pack.reset();
    try arg_pack.appendBufferPtr(input);
    try arg_pack.appendBufferPtr(gate_weight_u16);
    try arg_pack.appendBufferPtr(up_weight_u16);
    try arg_pack.appendBufferPtr(out);
    try arg_pack.appendScalar(u32, out_dim);
    try arg_pack.appendScalar(u32, in_dim);
    try arg_pack.appendScalar(u32, batch_rows);

    const rows_per_block = block_x / warp_size;
    try launch_mod.launchWithFamily(device, function, .{
        .grid_x = ceilDiv(out_dim, rows_per_block),
        .grid_y = ceilDiv(batch_rows, inner_batch_rows),
        .block_x = block_x,
    }, arg_pack, .matvec_gate_up_silu);
}

fn validateArgs(
    input: *const device_mod.Buffer,
    gate_weight_u16: *const device_mod.Buffer,
    up_weight_u16: *const device_mod.Buffer,
    out: *device_mod.Buffer,
    out_dim: u32,
    in_dim: u32,
    batch_rows: u32,
) !void {
    if (in_dim == 0 or out_dim == 0 or batch_rows == 0) return error.InvalidArgument;

    const batch: usize = @intCast(batch_rows);
    const input_row_bytes = std.math.mul(usize, @as(usize, in_dim), @sizeOf(f32)) catch return error.InvalidArgument;
    const input_bytes = std.math.mul(usize, input_row_bytes, batch) catch return error.InvalidArgument;
    const weight_elems = std.math.mul(usize, @as(usize, in_dim), @as(usize, out_dim)) catch return error.InvalidArgument;
    const weight_bytes = std.math.mul(usize, weight_elems, @sizeOf(u16)) catch return error.InvalidArgument;
    const out_row_bytes = std.math.mul(usize, @as(usize, out_dim), @sizeOf(f32)) catch return error.InvalidArgument;
    const out_bytes = std.math.mul(usize, out_row_bytes, batch) catch return error.InvalidArgument;
    if (input.size < input_bytes or gate_weight_u16.size < weight_bytes or up_weight_u16.size < weight_bytes or out.size < out_bytes) {
        return error.InvalidArgument;
    }
}

fn ceilDiv(numerator: u32, denominator: u32) u32 {
    return (numerator + denominator - 1) / denominator;
}

test "validateArgs rejects zero dimensions" {
    const b = device_mod.Buffer{ .pointer = 0, .size = 4096 };
    var out = b;
    try std.testing.expectError(error.InvalidArgument, validateArgs(&b, &b, &b, &out, 0, 4, 1));
    try std.testing.expectError(error.InvalidArgument, validateArgs(&b, &b, &b, &out, 4, 0, 1));
}

test "validateArgs rejects zero batch rows" {
    const b = device_mod.Buffer{ .pointer = 0, .size = 4096 };
    var out = b;
    try std.testing.expectError(error.InvalidArgument, validateArgs(&b, &b, &b, &out, 4, 4, 0));
}

test "validateArgs rejects undersized output" {
    const input = device_mod.Buffer{ .pointer = 0, .size = 8 * @sizeOf(f32) };
    const weight = device_mod.Buffer{ .pointer = 0, .size = 8 * 8 * @sizeOf(u16) };
    var out = device_mod.Buffer{ .pointer = 0, .size = 4 };
    try std.testing.expectError(error.InvalidArgument, validateArgs(&input, &weight, &weight, &out, 8, 8, 1));
}
