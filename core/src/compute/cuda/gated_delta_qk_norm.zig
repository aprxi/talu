//! Gated-delta query/key normalization kernel wrapper.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_gated_delta_qk_norm_f32";
pub const op_name: []const u8 = "gated_delta_qk_norm_f32";

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    query: *device_mod.Buffer,
    key: *device_mod.Buffer,
    n_heads: u32,
    d_head: u32,
) !void {
    try validateArgs(query, key, n_heads, d_head);

    arg_pack.reset();
    try arg_pack.appendBufferPtr(query);
    try arg_pack.appendBufferPtr(key);
    try arg_pack.appendScalar(u32, n_heads);
    try arg_pack.appendScalar(u32, d_head);

    try launch_mod.launch(device, function, .{
        .grid_x = n_heads,
        .block_x = 256,
    }, arg_pack);
}

fn validateArgs(
    query: *const device_mod.Buffer,
    key: *const device_mod.Buffer,
    n_heads: u32,
    d_head: u32,
) !void {
    if (n_heads == 0 or d_head == 0) return error.InvalidArgument;
    const bytes = std.math.mul(usize, std.math.mul(usize, n_heads, d_head) catch return error.InvalidArgument, @sizeOf(f32)) catch return error.InvalidArgument;
    if (query.size < bytes or key.size < bytes) return error.InvalidArgument;
}

test "validateArgs rejects zero dimensions" {
    var buf = device_mod.Buffer{ .pointer = 0, .size = 64 };
    try std.testing.expectError(error.InvalidArgument, validateArgs(&buf, &buf, 0, 8));
    try std.testing.expectError(error.InvalidArgument, validateArgs(&buf, &buf, 2, 0));
}

test "validateArgs rejects undersized buffers" {
    var small = device_mod.Buffer{ .pointer = 0, .size = 4 };
    try std.testing.expectError(error.InvalidArgument, validateArgs(&small, &small, 2, 8));
}
