//! Flash decode attention kernel with GQA optimization and split-K.
//!
//! Grid: (n_kv_heads × n_seq_chunks, batch_rows) — each block reads K/V
//! for one KV head's token chunk and computes attention for all kv_groups
//! query heads. Split-K divides the sequence across blocks for occupancy.
//!
//! When n_seq_chunks > 1, a reduce kernel merges partial results.
//!
//! 8 warps per block (256 threads). Online softmax with cross-warp merge.
//! Supports f16, i8, and fp8 KV cache dtypes.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;

// Entry point symbols for each KV dtype.
pub const symbol_f16: [:0]const u8 = "talu_flash_decode_f16";
pub const symbol_i8: [:0]const u8 = "talu_flash_decode_i8";
pub const symbol_fp8: [:0]const u8 = "talu_flash_decode_fp8";
pub const symbol_reduce: [:0]const u8 = "talu_flash_decode_reduce";
pub const op_name_f16: []const u8 = "flash_decode_f16";
pub const op_name_i8: []const u8 = "flash_decode_i8";
pub const op_name_fp8: []const u8 = "flash_decode_fp8";
pub const op_name_reduce: []const u8 = "flash_decode_reduce";

const num_warps: u32 = 8;
const max_groups: u32 = 4;

/// Compute shared memory for the flash decode kernel.
/// Layout: per_group × actual_groups, where per_group = (2 * num_warps + num_warps * head_dim) floats.
fn smemBytes(kv_groups: u32, head_dim: u32) u32 {
    const groups = @min(kv_groups, max_groups);
    const per_group: u32 = (2 * num_warps + num_warps * head_dim);
    return per_group * groups * @sizeOf(f32);
}

/// Compute optimal n_seq_chunks for split-K.
/// For decode, attention is rarely the bottleneck (GEMV dominates), so
/// split-K reduce overhead typically outweighs occupancy gains. The GQA
/// optimization (reading KV once per KV head) already provides the main
/// bandwidth win. Only split for extreme cases where a single block per
/// KV head would be underutilized.
pub fn computeSeqChunks(n_kv_heads: u32, batch_rows: u32) u32 {
    _ = n_kv_heads;
    _ = batch_rows;
    return 1;
}

/// Compute partial buffer size in bytes for split-K.
/// Layout: [batch_rows × n_heads × n_seq_chunks] for m and s (floats),
///         [batch_rows × n_heads × n_seq_chunks × head_dim] for out (floats).
pub fn partialBufBytes(batch_rows: u32, n_heads: u32, n_seq_chunks: u32, head_dim: u32) usize {
    const entries = @as(usize, batch_rows) * @as(usize, n_heads) * @as(usize, n_seq_chunks);
    // m + s + out[head_dim], all f32
    return entries * (2 + @as(usize, head_dim)) * @sizeOf(f32);
}

/// Launch flash decode with scale pointers (i8/fp8 path).
pub fn runWithScales(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    out: *device_mod.Buffer,
    query: *const device_mod.Buffer,
    key_cache_ptrs: *const device_mod.Buffer,
    value_cache_ptrs: *const device_mod.Buffer,
    k_scale_ptrs: *const device_mod.Buffer,
    v_scale_ptrs: *const device_mod.Buffer,
    seq_lens: *const device_mod.Buffer,
    positions: *const device_mod.Buffer,
    batch_rows: u32,
    n_heads: u32,
    n_kv_heads: u32,
    row_stride: u32,
    kv_groups: u32,
    head_dim: u32,
    scale: f32,
    rope_dim: u32,
    sliding_window: u32,
    theta: f32,
    gate_proj: ?*const device_mod.Buffer,
    gate_proj_stride: u32,
    n_seq_chunks: u32,
    partial_m: ?*device_mod.Buffer,
    partial_s: ?*device_mod.Buffer,
    partial_out: ?*device_mod.Buffer,
) !void {
    try validateArgs(batch_rows, n_heads, n_kv_heads, row_stride, kv_groups, head_dim, scale, rope_dim, theta);
    if (n_seq_chunks == 0) return error.InvalidArgument;
    if (n_seq_chunks > 1 and (partial_m == null or partial_s == null or partial_out == null))
        return error.InvalidArgument;

    arg_pack.reset();
    try arg_pack.appendBufferPtr(out);
    try arg_pack.appendBufferPtr(query);
    try arg_pack.appendBufferPtr(key_cache_ptrs);
    try arg_pack.appendBufferPtr(value_cache_ptrs);
    try arg_pack.appendBufferPtr(k_scale_ptrs);
    try arg_pack.appendBufferPtr(v_scale_ptrs);
    try arg_pack.appendBufferPtr(seq_lens);
    try arg_pack.appendBufferPtr(positions);
    try arg_pack.appendScalar(u32, batch_rows);
    try arg_pack.appendScalar(u32, n_heads);
    try arg_pack.appendScalar(u32, n_kv_heads);
    try arg_pack.appendScalar(u32, row_stride);
    try arg_pack.appendScalar(u32, kv_groups);
    try arg_pack.appendScalar(u32, head_dim);
    try arg_pack.appendScalar(f32, scale);
    try arg_pack.appendScalar(u32, rope_dim);
    try arg_pack.appendScalar(u32, sliding_window);
    try arg_pack.appendScalar(f32, theta);
    if (gate_proj) |gp| {
        try arg_pack.appendBufferPtr(gp);
    } else {
        try arg_pack.appendScalar(u64, 0);
    }
    try arg_pack.appendScalar(u32, gate_proj_stride);
    try arg_pack.appendScalar(u32, n_seq_chunks);
    if (partial_m) |pm| {
        try arg_pack.appendBufferPtr(pm);
    } else {
        try arg_pack.appendScalar(u64, 0);
    }
    if (partial_s) |ps| {
        try arg_pack.appendBufferPtr(ps);
    } else {
        try arg_pack.appendScalar(u64, 0);
    }
    if (partial_out) |po| {
        try arg_pack.appendBufferPtr(po);
    } else {
        try arg_pack.appendScalar(u64, 0);
    }

    try launch_mod.launchWithFamily(device, function, .{
        .grid_x = n_kv_heads * n_seq_chunks,
        .grid_y = batch_rows,
        .block_x = num_warps * 32,
        .shared_mem_bytes = smemBytes(kv_groups, head_dim),
    }, arg_pack, .attention);
}

/// Launch flash decode without scale pointers (f16 path).
pub fn runNoScales(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    out: *device_mod.Buffer,
    query: *const device_mod.Buffer,
    key_cache_ptrs: *const device_mod.Buffer,
    value_cache_ptrs: *const device_mod.Buffer,
    seq_lens: *const device_mod.Buffer,
    positions: *const device_mod.Buffer,
    batch_rows: u32,
    n_heads: u32,
    n_kv_heads: u32,
    row_stride: u32,
    kv_groups: u32,
    head_dim: u32,
    scale: f32,
    rope_dim: u32,
    sliding_window: u32,
    theta: f32,
    gate_proj: ?*const device_mod.Buffer,
    gate_proj_stride: u32,
    n_seq_chunks: u32,
    partial_m: ?*device_mod.Buffer,
    partial_s: ?*device_mod.Buffer,
    partial_out: ?*device_mod.Buffer,
) !void {
    try validateArgs(batch_rows, n_heads, n_kv_heads, row_stride, kv_groups, head_dim, scale, rope_dim, theta);
    if (n_seq_chunks == 0) return error.InvalidArgument;
    if (n_seq_chunks > 1 and (partial_m == null or partial_s == null or partial_out == null))
        return error.InvalidArgument;

    arg_pack.reset();
    try arg_pack.appendBufferPtr(out);
    try arg_pack.appendBufferPtr(query);
    try arg_pack.appendBufferPtr(key_cache_ptrs);
    try arg_pack.appendBufferPtr(value_cache_ptrs);
    try arg_pack.appendBufferPtr(seq_lens);
    try arg_pack.appendBufferPtr(positions);
    try arg_pack.appendScalar(u32, batch_rows);
    try arg_pack.appendScalar(u32, n_heads);
    try arg_pack.appendScalar(u32, n_kv_heads);
    try arg_pack.appendScalar(u32, row_stride);
    try arg_pack.appendScalar(u32, kv_groups);
    try arg_pack.appendScalar(u32, head_dim);
    try arg_pack.appendScalar(f32, scale);
    try arg_pack.appendScalar(u32, rope_dim);
    try arg_pack.appendScalar(u32, sliding_window);
    try arg_pack.appendScalar(f32, theta);
    if (gate_proj) |gp| {
        try arg_pack.appendBufferPtr(gp);
    } else {
        try arg_pack.appendScalar(u64, 0);
    }
    try arg_pack.appendScalar(u32, gate_proj_stride);
    try arg_pack.appendScalar(u32, n_seq_chunks);
    if (partial_m) |pm| {
        try arg_pack.appendBufferPtr(pm);
    } else {
        try arg_pack.appendScalar(u64, 0);
    }
    if (partial_s) |ps| {
        try arg_pack.appendBufferPtr(ps);
    } else {
        try arg_pack.appendScalar(u64, 0);
    }
    if (partial_out) |po| {
        try arg_pack.appendBufferPtr(po);
    } else {
        try arg_pack.appendScalar(u64, 0);
    }

    try launch_mod.launchWithFamily(device, function, .{
        .grid_x = n_kv_heads * n_seq_chunks,
        .grid_y = batch_rows,
        .block_x = num_warps * 32,
        .shared_mem_bytes = smemBytes(kv_groups, head_dim),
    }, arg_pack, .attention);
}

/// Launch the reduce kernel to merge split-K partial results.
pub fn runReduce(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    out: *device_mod.Buffer,
    partial_m: *device_mod.Buffer,
    partial_s: *device_mod.Buffer,
    partial_out_buf: *device_mod.Buffer,
    batch_rows: u32,
    n_heads: u32,
    head_dim: u32,
    n_seq_chunks: u32,
    gate_proj: ?*const device_mod.Buffer,
    gate_proj_stride: u32,
) !void {
    if (batch_rows == 0 or n_heads == 0 or head_dim == 0 or n_seq_chunks < 2)
        return error.InvalidArgument;

    arg_pack.reset();
    try arg_pack.appendBufferPtr(out);
    try arg_pack.appendBufferPtr(partial_m);
    try arg_pack.appendBufferPtr(partial_s);
    try arg_pack.appendBufferPtr(partial_out_buf);
    try arg_pack.appendScalar(u32, batch_rows);
    try arg_pack.appendScalar(u32, n_heads);
    try arg_pack.appendScalar(u32, head_dim);
    try arg_pack.appendScalar(u32, n_seq_chunks);
    if (gate_proj) |gp| {
        try arg_pack.appendBufferPtr(gp);
    } else {
        try arg_pack.appendScalar(u64, 0);
    }
    try arg_pack.appendScalar(u32, gate_proj_stride);

    try launch_mod.launchWithFamily(device, function, .{
        .grid_x = n_heads,
        .grid_y = batch_rows,
        .block_x = 32, // 1 warp
        .shared_mem_bytes = 0,
    }, arg_pack, .attention);
}

fn validateArgs(
    batch_rows: u32,
    n_heads: u32,
    n_kv_heads: u32,
    row_stride: u32,
    kv_groups: u32,
    head_dim: u32,
    scale: f32,
    rope_dim: u32,
    theta: f32,
) !void {
    if (batch_rows == 0 or n_heads == 0 or n_kv_heads == 0 or row_stride == 0 or
        kv_groups == 0 or head_dim == 0) return error.InvalidArgument;
    if (n_heads % kv_groups != 0) return error.InvalidArgument;
    if (head_dim > 256) return error.InvalidArgument;
    if (kv_groups > max_groups) return error.InvalidArgument;
    if (!std.math.isFinite(scale)) return error.InvalidArgument;
    if (rope_dim == 0 or rope_dim > head_dim or (rope_dim & 1) != 0) return error.InvalidArgument;
    if (!std.math.isFinite(theta) or theta <= 1.0) return error.InvalidArgument;
    const required_row = std.math.mul(u32, n_kv_heads, head_dim) catch return error.InvalidArgument;
    if (row_stride < required_row) return error.InvalidArgument;
}

test "validateArgs rejects invalid dimensions" {
    try std.testing.expectError(error.InvalidArgument, validateArgs(0, 8, 2, 64, 4, 64, 1.0, 64, 10000.0));
    try std.testing.expectError(error.InvalidArgument, validateArgs(1, 8, 2, 64, 3, 64, 1.0, 64, 10000.0));
    try std.testing.expectError(error.InvalidArgument, validateArgs(1, 8, 2, 64, 4, 64, 1.0, 64, 0.5));
}

test "validateArgs accepts valid dimensions" {
    try validateArgs(1, 8, 2, 512, 4, 256, 0.088, 64, 10000.0);
    try validateArgs(8, 12, 2, 768, 6, 128, 0.088, 128, 1000000.0);
}

test "smemBytes computes correct sizes" {
    // groups=4, head_dim=256, num_warps=8:
    // per_group = (2*8 + 8*256) = 2064 floats = 8256 bytes
    // total = 4 * 8256 = 33024 bytes
    try std.testing.expectEqual(@as(u32, 33024), smemBytes(4, 256));
    // groups=1, head_dim=128: per_group = (16 + 1024) = 1040 * 4 = 4160
    try std.testing.expectEqual(@as(u32, 4160), smemBytes(1, 128));
}

test "computeSeqChunks returns 1 (no split-K)" {
    // Split-K disabled: reduce overhead outweighs occupancy gains for decode.
    try std.testing.expectEqual(@as(u32, 1), computeSeqChunks(2, 1));
    try std.testing.expectEqual(@as(u32, 1), computeSeqChunks(2, 4));
    try std.testing.expectEqual(@as(u32, 1), computeSeqChunks(32, 2));
    try std.testing.expectEqual(@as(u32, 1), computeSeqChunks(2, 8));
}

test "partialBufBytes" {
    // 8 rows, 12 heads, 4 chunks, 128 dim
    // entries = 8*12*4 = 384
    // per entry = (2+128)*4 = 520 bytes
    // total = 384*520 = 199680
    try std.testing.expectEqual(@as(usize, 199680), partialBufBytes(8, 12, 4, 128));
}
