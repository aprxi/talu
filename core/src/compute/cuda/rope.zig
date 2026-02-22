//! RoPE kernel wrapper for modular CUDA runtime.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");
const registry_mod = @import("registry.zig");

pub const embedded_ptx = @embedFile("kernels/kernels.ptx");
pub const embedded_symbol: [:0]const u8 = "talu_rope_f32_v1";

pub fn run(
    allocator: std.mem.Allocator,
    device: *device_mod.Device,
    registry: *registry_mod.Registry,
    io: *device_mod.Buffer,
    n_heads: u32,
    head_dim: u32,
    rope_dim: u32,
    position: u32,
    theta: f32,
) !registry_mod.KernelSource {
    validateArgs(io, n_heads, head_dim, rope_dim, theta) catch |err| return err;

    if (registry.embedded_module == null) try registry.loadEmbeddedModule(embedded_ptx);
    const resolved = try registry.resolveFunction("rope_f32", embedded_symbol);
    var arg_pack = args_mod.ArgPack.init(allocator);
    defer arg_pack.deinit();
    try runWithFunction(&arg_pack, device, resolved.function, io, n_heads, head_dim, rope_dim, position, theta);
    return resolved.source;
}

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    io: *device_mod.Buffer,
    n_heads: u32,
    head_dim: u32,
    rope_dim: u32,
    position: u32,
    theta: f32,
) !void {
    try validateArgs(io, n_heads, head_dim, rope_dim, theta);

    const pair_dim = rope_dim / 2;
    const total_pairs = n_heads * pair_dim;

    arg_pack.reset();
    try arg_pack.appendBufferPtr(io);
    try arg_pack.appendScalar(u32, n_heads);
    try arg_pack.appendScalar(u32, head_dim);
    try arg_pack.appendScalar(u32, rope_dim);
    try arg_pack.appendScalar(u32, position);
    try arg_pack.appendScalar(f32, theta);

    const block_x: u32 = 256;
    const grid_x: u32 = ceilDiv(total_pairs, block_x);
    try launch_mod.launch(device, function, .{
        .grid_x = grid_x,
        .block_x = block_x,
    }, arg_pack);
}

fn validateArgs(io: *const device_mod.Buffer, n_heads: u32, head_dim: u32, rope_dim: u32, theta: f32) !void {
    if (n_heads == 0 or head_dim == 0 or rope_dim == 0) return error.InvalidArgument;
    if (rope_dim > head_dim) return error.InvalidArgument;
    if ((rope_dim & 1) != 0) return error.InvalidArgument;
    if (!std.math.isFinite(theta) or theta <= 1.0) return error.InvalidArgument;

    const required_count = @as(usize, n_heads) * @as(usize, head_dim);
    const required_bytes = required_count * @sizeOf(f32);
    if (io.size < required_bytes) return error.InvalidArgument;
}

fn ceilDiv(numerator: u32, denominator: u32) u32 {
    return (numerator + denominator - 1) / denominator;
}

test "validateArgs rejects odd rope_dim" {
    const buffer = device_mod.Buffer{ .pointer = 0, .size = 1024 };
    try std.testing.expectError(error.InvalidArgument, validateArgs(&buffer, 4, 64, 63, 10000.0));
}

test "validateArgs rejects rope_dim greater than head_dim" {
    const buffer = device_mod.Buffer{ .pointer = 0, .size = 1024 };
    try std.testing.expectError(error.InvalidArgument, validateArgs(&buffer, 4, 64, 128, 10000.0));
}
