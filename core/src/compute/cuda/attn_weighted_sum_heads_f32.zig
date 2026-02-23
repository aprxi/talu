//! Batched attention weighted-sum kernel wrapper for f32 KV cache.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_attn_weighted_sum_heads_f32";
pub const op_name: []const u8 = "attn_weighted_sum_heads_f32";

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    probs: *const device_mod.Buffer,
    value_cache: *const device_mod.Buffer,
    out: *device_mod.Buffer,
    n_heads: u32,
    seq_len: u32,
    row_stride: u32,
    kv_groups: u32,
    head_dim: u32,
) !void {
    try validateArgs(probs, value_cache, out, n_heads, seq_len, row_stride, kv_groups, head_dim);

    arg_pack.reset();
    try arg_pack.appendBufferPtr(out);
    try arg_pack.appendBufferPtr(probs);
    try arg_pack.appendBufferPtr(value_cache);
    try arg_pack.appendScalar(u32, n_heads);
    try arg_pack.appendScalar(u32, seq_len);
    try arg_pack.appendScalar(u32, row_stride);
    try arg_pack.appendScalar(u32, kv_groups);
    try arg_pack.appendScalar(u32, head_dim);

    const block_x: u32 = 128;
    try launch_mod.launch(device, function, .{
        .grid_x = ceilDiv(head_dim, warpsPerBlock(block_x)),
        .grid_y = n_heads,
        .block_x = block_x,
    }, arg_pack);
}

fn validateArgs(
    probs: *const device_mod.Buffer,
    value_cache: *const device_mod.Buffer,
    out: *device_mod.Buffer,
    n_heads: u32,
    seq_len: u32,
    row_stride: u32,
    kv_groups: u32,
    head_dim: u32,
) !void {
    if (n_heads == 0 or seq_len == 0 or row_stride == 0 or kv_groups == 0 or head_dim == 0) return error.InvalidArgument;
    if (n_heads % kv_groups != 0) return error.InvalidArgument;
    const n_kv_heads = n_heads / kv_groups;
    const required_row = std.math.mul(u32, n_kv_heads, head_dim) catch return error.InvalidArgument;
    if (row_stride < required_row) return error.InvalidArgument;

    const prob_elems = std.math.mul(usize, @as(usize, n_heads), @as(usize, seq_len)) catch return error.InvalidArgument;
    const prob_bytes = std.math.mul(usize, prob_elems, @sizeOf(f32)) catch return error.InvalidArgument;
    const cache_elems = std.math.mul(usize, @as(usize, seq_len), @as(usize, row_stride)) catch return error.InvalidArgument;
    const cache_bytes = std.math.mul(usize, cache_elems, @sizeOf(f32)) catch return error.InvalidArgument;
    const out_elems = std.math.mul(usize, @as(usize, n_heads), @as(usize, head_dim)) catch return error.InvalidArgument;
    const out_bytes = std.math.mul(usize, out_elems, @sizeOf(f32)) catch return error.InvalidArgument;
    if (probs.size < prob_bytes or value_cache.size < cache_bytes or out.size < out_bytes) {
        return error.InvalidArgument;
    }
}

fn ceilDiv(numerator: u32, denominator: u32) u32 {
    return (numerator + denominator - 1) / denominator;
}

fn warpsPerBlock(block_x: u32) u32 {
    return block_x / 32;
}

test "validateArgs rejects invalid shapes" {
    const probs = device_mod.Buffer{ .pointer = 0, .size = 1024 };
    const cache = device_mod.Buffer{ .pointer = 0, .size = 1024 };
    var out = device_mod.Buffer{ .pointer = 0, .size = 1024 };

    try std.testing.expectError(
        error.InvalidArgument,
        validateArgs(&probs, &cache, &out, 8, 2, 4, 3, 2),
    );
    try std.testing.expectError(
        error.InvalidArgument,
        validateArgs(&probs, &cache, &out, 0, 2, 4, 1, 2),
    );
}

test "warpsPerBlock computes expected warp count" {
    try std.testing.expectEqual(@as(u32, 4), warpsPerBlock(128));
    try std.testing.expectEqual(@as(u32, 8), warpsPerBlock(256));
}
