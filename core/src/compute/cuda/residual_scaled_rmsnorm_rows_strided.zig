//! Row-strided fused residual-add + RMSNorm kernel wrapper for CUDA runtime.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_residual_scaled_rmsnorm_rows_strided_f32";
pub const op_name: []const u8 = "residual_scaled_rmsnorm_rows_strided_f32";

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    residual_out: *device_mod.Buffer,
    norm_out: *device_mod.Buffer,
    residual_in: *const device_mod.Buffer,
    branch: *const device_mod.Buffer,
    weight: *const device_mod.Buffer,
    residual_scale: f32,
    rows: u32,
    cols: u32,
    residual_out_stride_elems: u32,
    norm_out_stride_elems: u32,
    residual_in_stride_elems: u32,
    branch_stride_elems: u32,
    eps: f32,
    weight_offset: f32,
) !void {
    try validateArgs(
        residual_out,
        norm_out,
        residual_in,
        branch,
        weight,
        rows,
        cols,
        residual_out_stride_elems,
        norm_out_stride_elems,
        residual_in_stride_elems,
        branch_stride_elems,
        eps,
    );

    arg_pack.reset();
    try arg_pack.appendBufferPtr(residual_out);
    try arg_pack.appendBufferPtr(norm_out);
    try arg_pack.appendBufferPtr(residual_in);
    try arg_pack.appendBufferPtr(branch);
    try arg_pack.appendBufferPtr(weight);
    try arg_pack.appendScalar(f32, residual_scale);
    try arg_pack.appendScalar(u32, rows);
    try arg_pack.appendScalar(u32, cols);
    try arg_pack.appendScalar(u32, residual_out_stride_elems);
    try arg_pack.appendScalar(u32, norm_out_stride_elems);
    try arg_pack.appendScalar(u32, residual_in_stride_elems);
    try arg_pack.appendScalar(u32, branch_stride_elems);
    try arg_pack.appendScalar(f32, eps);
    try arg_pack.appendScalar(f32, weight_offset);

    try launch_mod.launchWithFamily(device, function, .{
        .grid_x = rows,
        .block_x = 256,
        .shared_mem_bytes = 256 * @sizeOf(f32),
    }, arg_pack, .norm);
}

fn validateArgs(
    residual_out: *const device_mod.Buffer,
    norm_out: *const device_mod.Buffer,
    residual_in: *const device_mod.Buffer,
    branch: *const device_mod.Buffer,
    weight: *const device_mod.Buffer,
    rows: u32,
    cols: u32,
    residual_out_stride_elems: u32,
    norm_out_stride_elems: u32,
    residual_in_stride_elems: u32,
    branch_stride_elems: u32,
    eps: f32,
) !void {
    if (rows == 0 or cols == 0) return error.InvalidArgument;
    if (!std.math.isFinite(eps) or eps <= 0.0) return error.InvalidArgument;
    if (residual_out_stride_elems < cols or
        norm_out_stride_elems < cols or
        residual_in_stride_elems < cols or
        branch_stride_elems < cols)
    {
        return error.InvalidArgument;
    }

    const weight_bytes = std.math.mul(usize, @as(usize, cols), @sizeOf(f32)) catch return error.InvalidArgument;
    if (weight.size < weight_bytes) return error.InvalidArgument;

    const residual_out_required = try requiredBytes(rows, cols, residual_out_stride_elems);
    const norm_out_required = try requiredBytes(rows, cols, norm_out_stride_elems);
    const residual_in_required = try requiredBytes(rows, cols, residual_in_stride_elems);
    const branch_required = try requiredBytes(rows, cols, branch_stride_elems);
    if (residual_out.size < residual_out_required or
        norm_out.size < norm_out_required or
        residual_in.size < residual_in_required or
        branch.size < branch_required)
    {
        return error.InvalidArgument;
    }
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
    try std.testing.expectError(
        error.InvalidArgument,
        validateArgs(&fake, &fake, &fake, &fake, &fake, 1, 32, 32, 32, 32, 32, 0.0),
    );
}

test "validateArgs rejects stride smaller than cols" {
    const fake = device_mod.Buffer{ .pointer = 0, .size = 4096 };
    try std.testing.expectError(
        error.InvalidArgument,
        validateArgs(&fake, &fake, &fake, &fake, &fake, 2, 64, 32, 64, 64, 64, 1e-5),
    );
}
