//! Fused attention kernel wrapper for f16 KV cache.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_attn_fused_heads_f16_kv";
pub const op_name: []const u8 = "attn_fused_heads_f16_kv";

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    query: *const device_mod.Buffer,
    key_cache_f16: *const device_mod.Buffer,
    value_cache_f16: *const device_mod.Buffer,
    out: *device_mod.Buffer,
    n_heads: u32,
    seq_len: u32,
    row_stride: u32,
    kv_groups: u32,
    head_dim: u32,
    scale: f32,
    rope_dim: u32,
    position: u32,
    theta: f32,
) !void {
    try validateArgs(
        query,
        key_cache_f16,
        value_cache_f16,
        out,
        n_heads,
        seq_len,
        row_stride,
        kv_groups,
        head_dim,
        scale,
        rope_dim,
        theta,
    );

    arg_pack.reset();
    try arg_pack.appendBufferPtr(out);
    try arg_pack.appendBufferPtr(query);
    try arg_pack.appendBufferPtr(key_cache_f16);
    try arg_pack.appendBufferPtr(value_cache_f16);
    try arg_pack.appendScalar(u32, n_heads);
    try arg_pack.appendScalar(u32, seq_len);
    try arg_pack.appendScalar(u32, row_stride);
    try arg_pack.appendScalar(u32, kv_groups);
    try arg_pack.appendScalar(u32, head_dim);
    try arg_pack.appendScalar(f32, scale);
    try arg_pack.appendScalar(u32, rope_dim);
    try arg_pack.appendScalar(u32, position);
    try arg_pack.appendScalar(f32, theta);

    try launch_mod.launch(device, function, .{
        .grid_x = n_heads,
        .block_x = 32,
    }, arg_pack);
}

fn validateArgs(
    query: *const device_mod.Buffer,
    key_cache_f16: *const device_mod.Buffer,
    value_cache_f16: *const device_mod.Buffer,
    out: *device_mod.Buffer,
    n_heads: u32,
    seq_len: u32,
    row_stride: u32,
    kv_groups: u32,
    head_dim: u32,
    scale: f32,
    rope_dim: u32,
    theta: f32,
) !void {
    if (n_heads == 0 or seq_len == 0 or row_stride == 0 or kv_groups == 0 or head_dim == 0) return error.InvalidArgument;
    if (n_heads % kv_groups != 0) return error.InvalidArgument;
    if (head_dim > 512) return error.InvalidArgument;
    if (!std.math.isFinite(scale)) return error.InvalidArgument;
    if (rope_dim == 0 or rope_dim > head_dim or (rope_dim & 1) != 0) return error.InvalidArgument;
    if (!std.math.isFinite(theta) or theta <= 1.0) return error.InvalidArgument;

    const n_kv_heads = n_heads / kv_groups;
    const required_row = std.math.mul(u32, n_kv_heads, head_dim) catch return error.InvalidArgument;
    if (row_stride < required_row) return error.InvalidArgument;

    const query_elems = std.math.mul(usize, @as(usize, n_heads), @as(usize, head_dim)) catch return error.InvalidArgument;
    const query_bytes = std.math.mul(usize, query_elems, @sizeOf(f32)) catch return error.InvalidArgument;
    const cache_elems = std.math.mul(usize, @as(usize, seq_len), @as(usize, row_stride)) catch return error.InvalidArgument;
    const cache_bytes = std.math.mul(usize, cache_elems, @sizeOf(u16)) catch return error.InvalidArgument;
    const out_bytes = query_bytes;
    if (query.size < query_bytes or key_cache_f16.size < cache_bytes or value_cache_f16.size < cache_bytes or out.size < out_bytes) {
        return error.InvalidArgument;
    }
}

test "validateArgs rejects invalid shape params" {
    const query = device_mod.Buffer{ .pointer = 0, .size = 1024 };
    const cache = device_mod.Buffer{ .pointer = 0, .size = 1024 };
    var out = device_mod.Buffer{ .pointer = 0, .size = 1024 };

    try std.testing.expectError(error.InvalidArgument, validateArgs(&query, &cache, &cache, &out, 0, 1, 1, 1, 1, 1.0, 1, 10000.0));
    try std.testing.expectError(error.InvalidArgument, validateArgs(&query, &cache, &cache, &out, 8, 2, 4, 3, 2, 1.0, 2, 10000.0));
}
