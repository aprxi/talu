//! Flash prefill attention kernel with online softmax and KV tiling.
//!
//! Grid: (n_heads, ceil(q_rows / 4)) — each block processes 4 query rows for
//! one attention head. K/V tiles loaded into shared memory and reused across
//! all 4 warps, reducing KV cache DRAM reads by 4×.
//!
//! 4 warps per block (128 threads). Online softmax per warp with causal masking.
//! Supports f16, i8, and fp8 KV cache dtypes.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;

pub const symbol_f16: [:0]const u8 = "talu_flash_prefill_f16";
pub const symbol_i8: [:0]const u8 = "talu_flash_prefill_i8";
pub const symbol_fp8: [:0]const u8 = "talu_flash_prefill_fp8";
pub const op_name_f16: []const u8 = "flash_prefill_f16";
pub const op_name_i8: []const u8 = "flash_prefill_i8";
pub const op_name_fp8: []const u8 = "flash_prefill_fp8";

const fp_br: u32 = 4; // query rows per block
const fp_tile: u32 = 16; // KV tokens per tile
const block_size: u32 = fp_br * 32;

/// Compute shared memory bytes for the flash prefill kernel.
/// K tile + V tile, each [FP_TILE × head_dim] in the native element size.
fn smemBytes(head_dim: u32, kv_elem_bytes: u32) u32 {
    return 2 * fp_tile * head_dim * kv_elem_bytes;
}

/// Launch flash prefill for f16 KV cache.
pub fn runF16(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    query: *const device_mod.Buffer,
    key_cache: *const device_mod.Buffer,
    value_cache: *const device_mod.Buffer,
    out: *device_mod.Buffer,
    n_heads: u32,
    q_rows: u32,
    seq_len: u32,
    row_stride: u32,
    kv_groups: u32,
    head_dim: u32,
    scale: f32,
    rope_dim: u32,
    position_base: u32,
    sliding_window: u32,
    theta: f32,
) !void {
    try validateArgs(n_heads, q_rows, seq_len, row_stride, kv_groups, head_dim, scale, rope_dim, theta);

    arg_pack.reset();
    try arg_pack.appendBufferPtr(out);
    try arg_pack.appendBufferPtr(query);
    try arg_pack.appendBufferPtr(key_cache);
    try arg_pack.appendBufferPtr(value_cache);
    try arg_pack.appendScalar(u32, n_heads);
    try arg_pack.appendScalar(u32, q_rows);
    try arg_pack.appendScalar(u32, seq_len);
    try arg_pack.appendScalar(u32, row_stride);
    try arg_pack.appendScalar(u32, kv_groups);
    try arg_pack.appendScalar(u32, head_dim);
    try arg_pack.appendScalar(f32, scale);
    try arg_pack.appendScalar(u32, rope_dim);
    try arg_pack.appendScalar(u32, position_base);
    try arg_pack.appendScalar(u32, sliding_window);
    try arg_pack.appendScalar(f32, theta);

    const grid_x_blocks = (q_rows + fp_br - 1) / fp_br;
    try launch_mod.launchWithFamily(device, function, .{
        .grid_x = grid_x_blocks,
        .grid_y = n_heads,
        .block_x = block_size,
        .shared_mem_bytes = smemBytes(head_dim, 2), // f16 = 2 bytes
    }, arg_pack, .attention);
}

/// Launch flash prefill for i8/fp8 KV cache (with per-head scales).
pub fn runWithScales(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    query: *const device_mod.Buffer,
    key_cache: *const device_mod.Buffer,
    value_cache: *const device_mod.Buffer,
    k_scales: *const device_mod.Buffer,
    v_scales: *const device_mod.Buffer,
    out: *device_mod.Buffer,
    n_heads: u32,
    n_kv_heads: u32,
    q_rows: u32,
    seq_len: u32,
    row_stride: u32,
    kv_groups: u32,
    head_dim: u32,
    scale: f32,
    rope_dim: u32,
    position_base: u32,
    sliding_window: u32,
    theta: f32,
) !void {
    try validateArgs(n_heads, q_rows, seq_len, row_stride, kv_groups, head_dim, scale, rope_dim, theta);

    arg_pack.reset();
    try arg_pack.appendBufferPtr(out);
    try arg_pack.appendBufferPtr(query);
    try arg_pack.appendBufferPtr(key_cache);
    try arg_pack.appendBufferPtr(value_cache);
    try arg_pack.appendBufferPtr(k_scales);
    try arg_pack.appendBufferPtr(v_scales);
    try arg_pack.appendScalar(u32, n_heads);
    try arg_pack.appendScalar(u32, n_kv_heads);
    try arg_pack.appendScalar(u32, q_rows);
    try arg_pack.appendScalar(u32, seq_len);
    try arg_pack.appendScalar(u32, row_stride);
    try arg_pack.appendScalar(u32, kv_groups);
    try arg_pack.appendScalar(u32, head_dim);
    try arg_pack.appendScalar(f32, scale);
    try arg_pack.appendScalar(u32, rope_dim);
    try arg_pack.appendScalar(u32, position_base);
    try arg_pack.appendScalar(u32, sliding_window);
    try arg_pack.appendScalar(f32, theta);

    const grid_x_blocks = (q_rows + fp_br - 1) / fp_br;
    try launch_mod.launchWithFamily(device, function, .{
        .grid_x = grid_x_blocks,
        .grid_y = n_heads,
        .block_x = block_size,
        .shared_mem_bytes = smemBytes(head_dim, 1), // i8/fp8 = 1 byte
    }, arg_pack, .attention);
}

fn validateArgs(
    n_heads: u32,
    q_rows: u32,
    seq_len: u32,
    row_stride: u32,
    kv_groups: u32,
    head_dim: u32,
    scale: f32,
    rope_dim: u32,
    theta: f32,
) !void {
    if (n_heads == 0 or q_rows == 0 or seq_len == 0 or row_stride == 0 or
        kv_groups == 0 or head_dim == 0) return error.InvalidArgument;
    if (q_rows > seq_len) return error.InvalidArgument;
    if (n_heads % kv_groups != 0) return error.InvalidArgument;
    if (head_dim > 256) return error.InvalidArgument;
    if (!std.math.isFinite(scale)) return error.InvalidArgument;
    if (rope_dim == 0 or rope_dim > head_dim or (rope_dim & 1) != 0) return error.InvalidArgument;
    if (!std.math.isFinite(theta) or theta <= 1.0) return error.InvalidArgument;
    const n_kv_heads = n_heads / kv_groups;
    const required_row = std.math.mul(u32, n_kv_heads, head_dim) catch return error.InvalidArgument;
    if (row_stride < required_row) return error.InvalidArgument;
}

test "validateArgs rejects invalid dimensions" {
    try std.testing.expectError(error.InvalidArgument, validateArgs(0, 1, 1, 1, 1, 1, 1.0, 2, 10000.0));
    try std.testing.expectError(error.InvalidArgument, validateArgs(8, 3, 2, 4, 2, 2, 1.0, 2, 10000.0));
    try std.testing.expectError(error.InvalidArgument, validateArgs(8, 1, 1, 64, 3, 64, 1.0, 64, 10000.0));
}

test "validateArgs accepts valid dimensions" {
    try validateArgs(8, 100, 100, 256, 4, 128, 0.088, 128, 10000.0);
    try validateArgs(12, 500, 500, 768, 6, 128, 0.088, 128, 1000000.0);
}

test "smemBytes computes correct sizes" {
    // f16: 2 * 16 * 128 * 2 = 8192
    try std.testing.expectEqual(@as(u32, 8192), smemBytes(128, 2));
    // i8: 2 * 16 * 128 * 1 = 4096
    try std.testing.expectEqual(@as(u32, 4096), smemBytes(128, 1));
    // f16 head_dim=256: 2 * 16 * 256 * 2 = 16384
    try std.testing.expectEqual(@as(u32, 16384), smemBytes(256, 2));
}
