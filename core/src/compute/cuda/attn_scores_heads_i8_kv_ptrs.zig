//! Batched attention-score kernel wrapper for i8 KV cache with pointer tables.
//! Reads KV cache pointers and seq_lens from device buffers (graph-compatible).

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_attn_scores_heads_i8_kv_ptrs";
pub const op_name: []const u8 = "attn_scores_heads_i8_kv_ptrs";

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    scores_out: *device_mod.Buffer,
    query: *const device_mod.Buffer,
    key_cache_ptrs: *const device_mod.Buffer,
    k_scale_ptrs: *const device_mod.Buffer,
    seq_lens: *const device_mod.Buffer,
    positions: *const device_mod.Buffer,
    batch_rows: u32,
    n_heads: u32,
    n_kv_heads: u32,
    max_seq_len: u32,
    row_stride: u32,
    kv_groups: u32,
    head_dim: u32,
    scale: f32,
    sliding_window: u32,
) !void {
    if (batch_rows == 0 or n_heads == 0 or n_kv_heads == 0 or max_seq_len == 0 or row_stride == 0 or kv_groups == 0 or head_dim == 0) return error.InvalidArgument;
    if (n_heads % kv_groups != 0) return error.InvalidArgument;
    if (!std.math.isFinite(scale)) return error.InvalidArgument;

    const required_row = std.math.mul(u32, n_kv_heads, head_dim) catch return error.InvalidArgument;
    if (row_stride < required_row) return error.InvalidArgument;

    const query_elems = std.math.mul(usize, @as(usize, batch_rows) * @as(usize, n_heads), @as(usize, head_dim)) catch return error.InvalidArgument;
    if (query.size < query_elems * @sizeOf(f32)) return error.InvalidArgument;

    const score_elems = std.math.mul(usize, @as(usize, batch_rows) * @as(usize, n_heads), @as(usize, max_seq_len)) catch return error.InvalidArgument;
    if (scores_out.size < score_elems * @sizeOf(f32)) return error.InvalidArgument;

    if (key_cache_ptrs.size < @as(usize, batch_rows) * @sizeOf(u64)) return error.InvalidArgument;
    if (k_scale_ptrs.size < @as(usize, batch_rows) * @sizeOf(u64)) return error.InvalidArgument;
    if (seq_lens.size < @as(usize, batch_rows) * @sizeOf(u32)) return error.InvalidArgument;
    if (positions.size < @as(usize, batch_rows) * @sizeOf(u32)) return error.InvalidArgument;

    arg_pack.reset();
    try arg_pack.appendBufferPtr(scores_out);
    try arg_pack.appendBufferPtr(query);
    try arg_pack.appendBufferPtr(key_cache_ptrs);
    try arg_pack.appendBufferPtr(k_scale_ptrs);
    try arg_pack.appendBufferPtr(seq_lens);
    try arg_pack.appendBufferPtr(positions);
    try arg_pack.appendScalar(u32, n_heads);
    try arg_pack.appendScalar(u32, n_kv_heads);
    try arg_pack.appendScalar(u32, max_seq_len);
    try arg_pack.appendScalar(u32, row_stride);
    try arg_pack.appendScalar(u32, kv_groups);
    try arg_pack.appendScalar(u32, head_dim);
    try arg_pack.appendScalar(f32, scale);
    try arg_pack.appendScalar(u32, sliding_window);

    const block_x: u32 = 128;
    try launch_mod.launchWithFamily(device, function, .{
        .grid_x = ceilDiv(max_seq_len, warpsPerBlock(block_x)),
        .grid_y = n_heads,
        .grid_z = batch_rows,
        .block_x = block_x,
    }, arg_pack, .attention);
}

fn ceilDiv(numerator: u32, denominator: u32) u32 {
    return (numerator + denominator - 1) / denominator;
}

fn warpsPerBlock(block_x: u32) u32 {
    return block_x / 32;
}

test "ceilDiv computes expected block count" {
    try std.testing.expectEqual(@as(u32, 1), ceilDiv(1, 4));
    try std.testing.expectEqual(@as(u32, 50), ceilDiv(200, 4));
    try std.testing.expectEqual(@as(u32, 2048), ceilDiv(8192, 4));
}
