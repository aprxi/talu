//! Fused SiLU+mul kernel wrapper for modular CUDA runtime.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_silu_mul_f32";
pub const op_name: []const u8 = "silu_mul_f32";

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    gate: *const device_mod.Buffer,
    up: *const device_mod.Buffer,
    out: *device_mod.Buffer,
    count: u32,
) !void {
    try validateArgs(gate, up, out, count);

    arg_pack.reset();
    try arg_pack.appendBufferPtr(out);
    try arg_pack.appendBufferPtr(gate);
    try arg_pack.appendBufferPtr(up);
    try arg_pack.appendScalar(u32, count);

    const block_x: u32 = 256;
    try launch_mod.launch(device, function, .{
        .grid_x = ceilDiv(count, block_x),
        .block_x = block_x,
    }, arg_pack);
}

fn validateArgs(
    gate: *const device_mod.Buffer,
    up: *const device_mod.Buffer,
    out: *device_mod.Buffer,
    count: u32,
) !void {
    if (count == 0) return error.InvalidArgument;
    const bytes = std.math.mul(usize, @as(usize, count), @sizeOf(f32)) catch return error.InvalidArgument;
    if (gate.size < bytes or up.size < bytes or out.size < bytes) return error.InvalidArgument;
}

fn ceilDiv(numerator: u32, denominator: u32) u32 {
    return (numerator + denominator - 1) / denominator;
}

test "validateArgs rejects zero count" {
    const b = device_mod.Buffer{ .pointer = 0, .size = 64 };
    var out = b;
    try std.testing.expectError(error.InvalidArgument, validateArgs(&b, &b, &out, 0));
}

test "validateArgs rejects undersized output" {
    const b = device_mod.Buffer{ .pointer = 0, .size = 8 * @sizeOf(f32) };
    var small = device_mod.Buffer{ .pointer = 0, .size = 4 };
    try std.testing.expectError(error.InvalidArgument, validateArgs(&b, &b, &small, 8));
}
