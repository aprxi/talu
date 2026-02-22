//! Batched attention-score kernel wrapper for f16 KV cache.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_attn_scores_heads_f16_kv";

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    query: *const device_mod.Buffer,
    key_cache_f16: *const device_mod.Buffer,
    scores_out: *device_mod.Buffer,
    n_heads: u32,
    seq_len: u32,
    row_stride: u32,
    kv_groups: u32,
    head_dim: u32,
    scale: f32,
) !void {
    try validateArgs(query, key_cache_f16, scores_out, n_heads, seq_len, row_stride, kv_groups, head_dim, scale);

    arg_pack.reset();
    try arg_pack.appendBufferPtr(scores_out);
    try arg_pack.appendBufferPtr(query);
    try arg_pack.appendBufferPtr(key_cache_f16);
    try arg_pack.appendScalar(u32, n_heads);
    try arg_pack.appendScalar(u32, seq_len);
    try arg_pack.appendScalar(u32, row_stride);
    try arg_pack.appendScalar(u32, kv_groups);
    try arg_pack.appendScalar(u32, head_dim);
    try arg_pack.appendScalar(f32, scale);

    const block_x: u32 = 256;
    try launch_mod.launch(device, function, .{
        .grid_x = ceilDiv(seq_len, block_x),
        .grid_y = n_heads,
        .block_x = block_x,
    }, arg_pack);
}

fn validateArgs(
    query: *const device_mod.Buffer,
    key_cache_f16: *const device_mod.Buffer,
    scores_out: *device_mod.Buffer,
    n_heads: u32,
    seq_len: u32,
    row_stride: u32,
    kv_groups: u32,
    head_dim: u32,
    scale: f32,
) !void {
    if (n_heads == 0 or seq_len == 0 or row_stride == 0 or kv_groups == 0 or head_dim == 0) return error.InvalidArgument;
    if (n_heads % kv_groups != 0) return error.InvalidArgument;
    if (!std.math.isFinite(scale)) return error.InvalidArgument;

    const n_kv_heads = n_heads / kv_groups;
    const required_row = std.math.mul(u32, n_kv_heads, head_dim) catch return error.InvalidArgument;
    if (row_stride < required_row) return error.InvalidArgument;

    const query_bytes = std.math.mul(usize, @as(usize, n_heads), @as(usize, head_dim) * @sizeOf(f32)) catch return error.InvalidArgument;
    const cache_elems = std.math.mul(usize, @as(usize, seq_len), @as(usize, row_stride)) catch return error.InvalidArgument;
    const cache_bytes = std.math.mul(usize, cache_elems, @sizeOf(u16)) catch return error.InvalidArgument;
    const score_elems = std.math.mul(usize, @as(usize, n_heads), @as(usize, seq_len)) catch return error.InvalidArgument;
    const score_bytes = std.math.mul(usize, score_elems, @sizeOf(f32)) catch return error.InvalidArgument;
    if (query.size < query_bytes or key_cache_f16.size < cache_bytes or scores_out.size < score_bytes) {
        return error.InvalidArgument;
    }
}

fn ceilDiv(numerator: u32, denominator: u32) u32 {
    return (numerator + denominator - 1) / denominator;
}

test "validateArgs rejects invalid shapes" {
    const query = device_mod.Buffer{ .pointer = 0, .size = 1024 };
    const cache = device_mod.Buffer{ .pointer = 0, .size = 1024 };
    var scores = device_mod.Buffer{ .pointer = 0, .size = 1024 };

    try std.testing.expectError(
        error.InvalidArgument,
        validateArgs(&query, &cache, &scores, 0, 1, 1, 1, 1, 1.0),
    );
    try std.testing.expectError(
        error.InvalidArgument,
        validateArgs(&query, &cache, &scores, 8, 2, 4, 3, 2, 1.0),
    );
}

test "ceilDiv computes expected block count" {
    try std.testing.expectEqual(@as(u32, 1), ceilDiv(1, 256));
    try std.testing.expectEqual(@as(u32, 2), ceilDiv(257, 256));
}
