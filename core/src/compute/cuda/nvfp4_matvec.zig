//! NVFP4 matvec kernel wrapper for modular CUDA runtime.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");
const registry_mod = @import("registry.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_nvfp4_matvec_f32";
pub const op_name: []const u8 = "nvfp4_matvec_f32";
pub const embedded_symbol_tile8: [:0]const u8 = "talu_nvfp4_matvec_f32_tile8";
pub const op_name_tile8: []const u8 = "nvfp4_matvec_f32_tile8";
const warp_size: u32 = 32;
const block_x: u32 = 128;
const inner_batch_rows: u32 = 4;
const inner_batch_rows_tile8: u32 = 8;

pub fn run(
    allocator: std.mem.Allocator,
    device: *device_mod.Device,
    registry: *registry_mod.Registry,
    input: *const device_mod.Buffer,
    weight_packed: *const device_mod.Buffer,
    scales: *const device_mod.Buffer,
    out: *device_mod.Buffer,
    in_dim: u32,
    out_dim: u32,
    scale_cols: u32,
    group_size: u32,
    weight_global_scale: f32,
    batch_rows: u32,
) !registry_mod.KernelSource {
    try validateArgs(input, weight_packed, scales, out, in_dim, out_dim, scale_cols, group_size, weight_global_scale, batch_rows);

    if (registry.embedded_module == null) try registry.loadEmbeddedModule(embedded_module);
    const resolved = try registry.resolveFunction(op_name, embedded_symbol);
    var arg_pack = args_mod.ArgPack.init(allocator);
    defer arg_pack.deinit();
    try runWithFunction(
        &arg_pack,
        device,
        resolved.function,
        input,
        weight_packed,
        scales,
        out,
        in_dim,
        out_dim,
        scale_cols,
        group_size,
        weight_global_scale,
        batch_rows,
    );
    return resolved.source;
}

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    input: *const device_mod.Buffer,
    weight_packed: *const device_mod.Buffer,
    scales: *const device_mod.Buffer,
    out: *device_mod.Buffer,
    in_dim: u32,
    out_dim: u32,
    scale_cols: u32,
    group_size: u32,
    weight_global_scale: f32,
    batch_rows: u32,
) !void {
    return runWithFunctionBatchTile(
        arg_pack,
        device,
        function,
        input,
        weight_packed,
        scales,
        out,
        in_dim,
        out_dim,
        scale_cols,
        group_size,
        weight_global_scale,
        batch_rows,
        inner_batch_rows,
    );
}

pub fn runWithFunctionTile8(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    input: *const device_mod.Buffer,
    weight_packed: *const device_mod.Buffer,
    scales: *const device_mod.Buffer,
    out: *device_mod.Buffer,
    in_dim: u32,
    out_dim: u32,
    scale_cols: u32,
    group_size: u32,
    weight_global_scale: f32,
    batch_rows: u32,
) !void {
    return runWithFunctionBatchTile(
        arg_pack,
        device,
        function,
        input,
        weight_packed,
        scales,
        out,
        in_dim,
        out_dim,
        scale_cols,
        group_size,
        weight_global_scale,
        batch_rows,
        inner_batch_rows_tile8,
    );
}

fn runWithFunctionBatchTile(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    input: *const device_mod.Buffer,
    weight_packed: *const device_mod.Buffer,
    scales: *const device_mod.Buffer,
    out: *device_mod.Buffer,
    in_dim: u32,
    out_dim: u32,
    scale_cols: u32,
    group_size: u32,
    weight_global_scale: f32,
    batch_rows: u32,
    batch_tile_rows: u32,
) !void {
    try validateArgs(input, weight_packed, scales, out, in_dim, out_dim, scale_cols, group_size, weight_global_scale, batch_rows);
    if (batch_tile_rows == 0) return error.InvalidArgument;

    arg_pack.reset();
    try arg_pack.appendBufferPtr(input);
    try arg_pack.appendBufferPtr(weight_packed);
    try arg_pack.appendBufferPtr(scales);
    try arg_pack.appendBufferPtr(out);
    try arg_pack.appendScalar(u32, in_dim);
    try arg_pack.appendScalar(u32, out_dim);
    try arg_pack.appendScalar(u32, scale_cols);
    try arg_pack.appendScalar(u32, group_size);
    try arg_pack.appendScalar(f32, weight_global_scale);
    try arg_pack.appendScalar(u32, batch_rows);

    const rows_per_block = block_x / warp_size;
    const grid_x: u32 = ceilDiv(out_dim, rows_per_block);
    try launch_mod.launchWithFamily(device, function, .{
        .grid_x = grid_x,
        .grid_y = ceilDiv(batch_rows, batch_tile_rows),
        .block_x = block_x,
    }, arg_pack, .matvec);
}

fn validateArgs(
    input: *const device_mod.Buffer,
    weight_packed: *const device_mod.Buffer,
    scales: *const device_mod.Buffer,
    out: *device_mod.Buffer,
    in_dim: u32,
    out_dim: u32,
    scale_cols: u32,
    group_size: u32,
    weight_global_scale: f32,
    batch_rows: u32,
) !void {
    if (in_dim == 0 or out_dim == 0 or scale_cols == 0 or group_size == 0 or batch_rows == 0) return error.InvalidArgument;
    if (group_size != 16 or (in_dim & 15) != 0) return error.InvalidArgument;
    if (weight_global_scale == 0.0 or !std.math.isFinite(weight_global_scale)) return error.InvalidArgument;

    const in_dim_usize: usize = @intCast(in_dim);
    const out_dim_usize: usize = @intCast(out_dim);
    const scale_cols_usize: usize = @intCast(scale_cols);
    const batch_rows_usize: usize = @intCast(batch_rows);
    const packed_cols = (in_dim_usize + 1) >> 1;
    const input_elems = std.math.mul(usize, in_dim_usize, batch_rows_usize) catch return error.InvalidArgument;
    const out_elems = std.math.mul(usize, out_dim_usize, batch_rows_usize) catch return error.InvalidArgument;
    const input_bytes = std.math.mul(usize, input_elems, @sizeOf(f32)) catch return error.InvalidArgument;
    const out_bytes = std.math.mul(usize, out_elems, @sizeOf(f32)) catch return error.InvalidArgument;
    const packed_bytes = std.math.mul(usize, out_dim_usize, packed_cols) catch return error.InvalidArgument;
    const scale_bytes = std.math.mul(usize, out_dim_usize, scale_cols_usize) catch return error.InvalidArgument;

    if (input.size < input_bytes or
        out.size < out_bytes or
        weight_packed.size < packed_bytes or
        scales.size < scale_bytes)
    {
        return error.InvalidArgument;
    }
}

fn ceilDiv(numerator: u32, denominator: u32) u32 {
    return (numerator + denominator - 1) / denominator;
}

test "ceilDiv rounds up" {
    try std.testing.expectEqual(@as(u32, 1), ceilDiv(1, 4));
    try std.testing.expectEqual(@as(u32, 2), ceilDiv(5, 4));
}
