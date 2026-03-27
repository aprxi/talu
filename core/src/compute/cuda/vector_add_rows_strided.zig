//! Row-strided vector add kernel wrapper for modular CUDA runtime.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_vector_add_rows_strided_f32";
pub const op_name: []const u8 = "vector_add_rows_strided_f32";

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    a: *const device_mod.Buffer,
    b: *const device_mod.Buffer,
    out: *device_mod.Buffer,
    rows: u32,
    cols: u32,
    a_stride_elems: u32,
    b_stride_elems: u32,
    out_stride_elems: u32,
) !void {
    try validateArgs(a, b, out, rows, cols, a_stride_elems, b_stride_elems, out_stride_elems);

    arg_pack.reset();
    try arg_pack.appendBufferPtr(out);
    try arg_pack.appendBufferPtr(a);
    try arg_pack.appendBufferPtr(b);
    try arg_pack.appendScalar(u32, rows);
    try arg_pack.appendScalar(u32, cols);
    try arg_pack.appendScalar(u32, out_stride_elems);
    try arg_pack.appendScalar(u32, a_stride_elems);
    try arg_pack.appendScalar(u32, b_stride_elems);

    const total = std.math.mul(u32, rows, cols) catch return error.InvalidArgument;
    const block_x: u32 = 256;
    const grid_x: u32 = ceilDiv(total, block_x);
    try launch_mod.launchWithFamily(device, function, .{
        .grid_x = grid_x,
        .block_x = block_x,
    }, arg_pack, .pointwise);
}

fn validateArgs(
    a: *const device_mod.Buffer,
    b: *const device_mod.Buffer,
    out: *device_mod.Buffer,
    rows: u32,
    cols: u32,
    a_stride_elems: u32,
    b_stride_elems: u32,
    out_stride_elems: u32,
) !void {
    if (rows == 0 or cols == 0) return error.InvalidArgument;
    if (a_stride_elems < cols or b_stride_elems < cols or out_stride_elems < cols) return error.InvalidArgument;

    const a_required_bytes = try requiredBytes(rows, cols, a_stride_elems);
    const b_required_bytes = try requiredBytes(rows, cols, b_stride_elems);
    const out_required_bytes = try requiredBytes(rows, cols, out_stride_elems);
    if (a.size < a_required_bytes or b.size < b_required_bytes or out.size < out_required_bytes) {
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

fn ceilDiv(numerator: u32, denominator: u32) u32 {
    return (numerator + denominator - 1) / denominator;
}

test "validateArgs rejects stride smaller than cols" {
    const fake = device_mod.Buffer{ .pointer = 0, .size = 1024 };
    var out = fake;
    try std.testing.expectError(
        error.InvalidArgument,
        validateArgs(&fake, &fake, &out, 2, 64, 32, 64, 64),
    );
}

test "requiredBytes handles single-row input" {
    try std.testing.expectEqual(@as(usize, 128), try requiredBytes(1, 32, 128));
}
