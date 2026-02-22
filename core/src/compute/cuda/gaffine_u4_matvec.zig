//! Grouped-affine U4 matvec kernel wrapper for modular CUDA runtime.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");
const registry_mod = @import("registry.zig");

pub const embedded_ptx = @embedFile("kernels/kernels.ptx");
pub const embedded_symbol: [:0]const u8 = "talu_gaffine_u4_matvec_f32_v1";
pub const scales_dtype_f16: u32 = 0;
pub const scales_dtype_bf16: u32 = 1;

pub fn run(
    allocator: std.mem.Allocator,
    device: *device_mod.Device,
    registry: *registry_mod.Registry,
    input: *const device_mod.Buffer,
    packed_weight: *const device_mod.Buffer,
    scales: *const device_mod.Buffer,
    biases: *const device_mod.Buffer,
    out: *device_mod.Buffer,
    in_dim: u32,
    out_dim: u32,
    group_size: u32,
    scales_dtype_tag: u32,
) !registry_mod.KernelSource {
    try validateArgs(input, packed_weight, scales, biases, out, in_dim, out_dim, group_size, scales_dtype_tag);

    if (registry.embedded_module == null) try registry.loadEmbeddedModule(embedded_ptx);
    const resolved = try registry.resolveFunction("gaffine_u4_matvec_f32", embedded_symbol);
    var arg_pack = args_mod.ArgPack.init(allocator);
    defer arg_pack.deinit();
    try runWithFunction(
        &arg_pack,
        device,
        resolved.function,
        input,
        packed_weight,
        scales,
        biases,
        out,
        in_dim,
        out_dim,
        group_size,
        scales_dtype_tag,
    );
    return resolved.source;
}

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
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
    try validateArgs(input, packed_weight, scales, biases, out, in_dim, out_dim, group_size, scales_dtype_tag);

    arg_pack.reset();
    try arg_pack.appendBufferPtr(input);
    try arg_pack.appendBufferPtr(packed_weight);
    try arg_pack.appendBufferPtr(scales);
    try arg_pack.appendBufferPtr(biases);
    try arg_pack.appendBufferPtr(out);
    try arg_pack.appendScalar(u32, in_dim);
    try arg_pack.appendScalar(u32, out_dim);
    try arg_pack.appendScalar(u32, group_size);
    try arg_pack.appendScalar(u32, scales_dtype_tag);

    const block_x: u32 = 256;
    const grid_x: u32 = ceilDiv(out_dim, block_x);
    try launch_mod.launch(device, function, .{
        .grid_x = grid_x,
        .block_x = block_x,
    }, arg_pack);
}

fn validateArgs(
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
    if (scales_dtype_tag != scales_dtype_f16 and scales_dtype_tag != scales_dtype_bf16) return error.InvalidArgument;

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
    const b = device_mod.Buffer{ .pointer = 0, .size = 4096 };
    var out = b;
    try std.testing.expectError(
        error.InvalidArgument,
        validateArgs(&b, &b, &b, &b, &out, 16, 8, 12, scales_dtype_bf16),
    );
}

test "validateArgs rejects invalid scales dtype tag" {
    const b = device_mod.Buffer{ .pointer = 0, .size = 4096 };
    var out = b;
    try std.testing.expectError(
        error.InvalidArgument,
        validateArgs(&b, &b, &b, &b, &out, 16, 8, 8, 7),
    );
}
