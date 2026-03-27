//! Row-strided RMSNorm kernel wrapper for modular CUDA runtime.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_rmsnorm_rows_strided_f32";
pub const op_name: []const u8 = "rmsnorm_rows_strided_f32";

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    input: *const device_mod.Buffer,
    weight: *const device_mod.Buffer,
    output: *device_mod.Buffer,
    rows: u32,
    cols: u32,
    input_stride_elems: u32,
    output_stride_elems: u32,
    eps: f32,
    weight_offset: f32,
) !void {
    try validateArgs(input, weight, output, rows, cols, input_stride_elems, output_stride_elems, eps);

    arg_pack.reset();
    try arg_pack.appendBufferPtr(output);
    try arg_pack.appendBufferPtr(input);
    try arg_pack.appendBufferPtr(weight);
    try arg_pack.appendScalar(u32, rows);
    try arg_pack.appendScalar(u32, cols);
    try arg_pack.appendScalar(u32, input_stride_elems);
    try arg_pack.appendScalar(u32, output_stride_elems);
    try arg_pack.appendScalar(f32, eps);
    try arg_pack.appendScalar(f32, weight_offset);

    try launch_mod.launchWithFamily(device, function, .{
        .grid_x = rows,
        .block_x = 256,
        .shared_mem_bytes = 256 * @sizeOf(f32),
    }, arg_pack, .norm);
}

fn validateArgs(
    input: *const device_mod.Buffer,
    weight: *const device_mod.Buffer,
    output: *device_mod.Buffer,
    rows: u32,
    cols: u32,
    input_stride_elems: u32,
    output_stride_elems: u32,
    eps: f32,
) !void {
    if (rows == 0 or cols == 0) return error.InvalidArgument;
    if (!std.math.isFinite(eps) or eps <= 0.0) return error.InvalidArgument;
    if (input_stride_elems < cols or output_stride_elems < cols) return error.InvalidArgument;

    const weight_bytes = std.math.mul(usize, @as(usize, cols), @sizeOf(f32)) catch return error.InvalidArgument;
    if (weight.size < weight_bytes) return error.InvalidArgument;

    const input_required_bytes = try requiredBytes(rows, cols, input_stride_elems);
    const output_required_bytes = try requiredBytes(rows, cols, output_stride_elems);
    if (input.size < input_required_bytes or output.size < output_required_bytes) return error.InvalidArgument;
}

fn requiredBytes(rows: u32, cols: u32, stride_elems: u32) !usize {
    const row_count: usize = @intCast(rows);
    const cols_usize: usize = @intCast(cols);
    const stride_usize: usize = @intCast(stride_elems);
    const leading = if (row_count <= 1)
        @as(usize, 0)
    else
        try std.math.mul(usize, row_count - 1, stride_usize);
    const elems = try std.math.add(usize, leading, cols_usize);
    return std.math.mul(usize, elems, @sizeOf(f32));
}

test "validateArgs rejects invalid eps" {
    const fake = device_mod.Buffer{ .pointer = 0, .size = 4096 };
    var out = fake;
    try std.testing.expectError(
        error.InvalidArgument,
        validateArgs(&fake, &fake, &out, 1, 32, 32, 32, 0.0),
    );
}

test "validateArgs rejects stride smaller than cols" {
    const fake = device_mod.Buffer{ .pointer = 0, .size = 4096 };
    var out = fake;
    try std.testing.expectError(
        error.InvalidArgument,
        validateArgs(&fake, &fake, &out, 2, 64, 32, 64, 1e-5),
    );
}
