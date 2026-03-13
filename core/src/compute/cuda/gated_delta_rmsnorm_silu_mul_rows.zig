//! Fused gated-delta RMSNorm + SiLU(gate) multiply rows wrapper.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_gated_delta_rmsnorm_silu_mul_rows_f32";
pub const op_name: []const u8 = "gated_delta_rmsnorm_silu_mul_rows_f32";

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    input: *const device_mod.Buffer,
    gate_rows: *const device_mod.Buffer,
    weight: *const device_mod.Buffer,
    out: *device_mod.Buffer,
    rows_total: u32,
    cols: u32,
    n_v_heads: u32,
    gate_row_stride: u32,
    gate_offset: u32,
    eps: f32,
    weight_row_stride: u32,
) !void {
    try validateArgs(input, gate_rows, weight, out, rows_total, cols, n_v_heads, gate_row_stride, gate_offset, eps, weight_row_stride);

    arg_pack.reset();
    try arg_pack.appendBufferPtr(out);
    try arg_pack.appendBufferPtr(input);
    try arg_pack.appendBufferPtr(gate_rows);
    try arg_pack.appendBufferPtr(weight);
    try arg_pack.appendScalar(u32, rows_total);
    try arg_pack.appendScalar(u32, cols);
    try arg_pack.appendScalar(u32, n_v_heads);
    try arg_pack.appendScalar(u32, gate_row_stride);
    try arg_pack.appendScalar(u32, gate_offset);
    try arg_pack.appendScalar(f32, eps);
    try arg_pack.appendScalar(u32, weight_row_stride);

    try launch_mod.launch(device, function, .{
        .grid_x = rows_total,
        .block_x = 256,
    }, arg_pack);
}

fn validateArgs(
    input: *const device_mod.Buffer,
    gate_rows: *const device_mod.Buffer,
    weight: *const device_mod.Buffer,
    out: *device_mod.Buffer,
    rows_total: u32,
    cols: u32,
    n_v_heads: u32,
    gate_row_stride: u32,
    gate_offset: u32,
    eps: f32,
    weight_row_stride: u32,
) !void {
    if (rows_total == 0 or cols == 0 or n_v_heads == 0) return error.InvalidArgument;
    if (gate_row_stride == 0 or gate_offset + cols > gate_row_stride) return error.InvalidArgument;
    if (!std.math.isFinite(eps) or eps <= 0.0) return error.InvalidArgument;
    if (!(weight_row_stride == 0 or weight_row_stride >= cols)) return error.InvalidArgument;
    if ((rows_total % n_v_heads) != 0) return error.InvalidArgument;

    const io_elems = std.math.mul(usize, @as(usize, rows_total), @as(usize, cols)) catch return error.InvalidArgument;
    const io_bytes = std.math.mul(usize, io_elems, @sizeOf(f32)) catch return error.InvalidArgument;
    if (input.size < io_bytes or out.size < io_bytes) return error.InvalidArgument;

    const token_rows = rows_total / n_v_heads;
    const gate_elems = std.math.mul(usize, @as(usize, token_rows), @as(usize, gate_row_stride)) catch return error.InvalidArgument;
    const gate_bytes = std.math.mul(usize, gate_elems, @sizeOf(f32)) catch return error.InvalidArgument;
    if (gate_rows.size < gate_bytes) return error.InvalidArgument;

    const weight_rows = if (weight_row_stride == 0) 1 else @as(usize, n_v_heads);
    const weight_elems = std.math.mul(usize, weight_rows, @as(usize, if (weight_row_stride == 0) cols else weight_row_stride)) catch return error.InvalidArgument;
    const weight_bytes = std.math.mul(usize, weight_elems, @sizeOf(f32)) catch return error.InvalidArgument;
    if (weight.size < weight_bytes) return error.InvalidArgument;
}

test "validateArgs rejects invalid head mapping" {
    const buf = device_mod.Buffer{ .pointer = 0, .size = 4096 };
    var out = buf;
    try std.testing.expectError(error.InvalidArgument, validateArgs(&buf, &buf, &buf, &out, 7, 16, 2, 64, 0, 1.0e-6, 0));
}
