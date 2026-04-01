//! Batched fused decode-attention wrapper using per-row KV cache pointers (fp8).
//! Supports optional query gate fusion.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_attn_fused_decode_heads_fp8_kv_ptrs";
pub const op_name: []const u8 = "attn_fused_decode_heads_fp8_kv_ptrs";

pub fn runWithFunction(
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
) !void {
    try validateArgs(
        out,
        query,
        key_cache_ptrs,
        value_cache_ptrs,
        k_scale_ptrs,
        v_scale_ptrs,
        seq_lens,
        positions,
        batch_rows,
        n_heads,
        n_kv_heads,
        row_stride,
        kv_groups,
        head_dim,
        scale,
        rope_dim,
        theta,
        gate_proj,
        gate_proj_stride,
    );

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

    // 16 warps per block; shared memory: [num_warps] m + [num_warps] s + [num_warps * head_dim] out.
    const num_warps: u32 = 16;
    const smem_bytes: u32 = (2 * num_warps + num_warps * head_dim) * @sizeOf(f32);
    try launch_mod.launchWithFamily(device, function, .{
        .grid_x = n_heads,
        .grid_y = batch_rows,
        .block_x = num_warps * 32,
        .shared_mem_bytes = smem_bytes,
    }, arg_pack, .attention);
}

fn validateArgs(
    out: *const device_mod.Buffer,
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
    theta: f32,
    gate_proj: ?*const device_mod.Buffer,
    gate_proj_stride: u32,
) !void {
    if (batch_rows == 0 or n_heads == 0 or n_kv_heads == 0 or row_stride == 0 or kv_groups == 0 or head_dim == 0) return error.InvalidArgument;
    if (n_heads % kv_groups != 0) return error.InvalidArgument;
    if (head_dim > 512) return error.InvalidArgument;
    if (!std.math.isFinite(scale)) return error.InvalidArgument;
    if (rope_dim == 0 or rope_dim > head_dim or (rope_dim & 1) != 0) return error.InvalidArgument;
    if (!std.math.isFinite(theta) or theta <= 1.0) return error.InvalidArgument;

    const required_row = std.math.mul(u32, n_kv_heads, head_dim) catch return error.InvalidArgument;
    if (row_stride < required_row) return error.InvalidArgument;

    const query_elems = std.math.mul(usize, @as(usize, batch_rows), @as(usize, n_heads)) catch return error.InvalidArgument;
    const query_elems_full = std.math.mul(usize, query_elems, @as(usize, head_dim)) catch return error.InvalidArgument;
    const query_bytes = std.math.mul(usize, query_elems_full, @sizeOf(f32)) catch return error.InvalidArgument;
    const ptr_bytes = std.math.mul(usize, @as(usize, batch_rows), @sizeOf(u64)) catch return error.InvalidArgument;
    const idx_bytes = std.math.mul(usize, @as(usize, batch_rows), @sizeOf(u32)) catch return error.InvalidArgument;

    if (query.size < query_bytes or
        out.size < query_bytes or
        key_cache_ptrs.size < ptr_bytes or
        value_cache_ptrs.size < ptr_bytes or
        k_scale_ptrs.size < ptr_bytes or
        v_scale_ptrs.size < ptr_bytes or
        seq_lens.size < idx_bytes or
        positions.size < idx_bytes)
    {
        return error.InvalidArgument;
    }

    if (gate_proj) |gp| {
        if (gate_proj_stride == 0) return error.InvalidArgument;
        const gate_bytes = std.math.mul(
            usize,
            std.math.mul(usize, @as(usize, batch_rows), @as(usize, gate_proj_stride)) catch return error.InvalidArgument,
            @sizeOf(f32),
        ) catch return error.InvalidArgument;
        if (gp.size < gate_bytes) return error.InvalidArgument;
    }
}

test "validateArgs rejects invalid dimensions" {
    const b = device_mod.Buffer{ .pointer = 0, .size = 4096 };
    try std.testing.expectError(
        error.InvalidArgument,
        validateArgs(&b, &b, &b, &b, &b, &b, &b, &b, 0, 8, 2, 64, 1, 64, 1.0, 64, 10000.0, null, 0),
    );
    try std.testing.expectError(
        error.InvalidArgument,
        validateArgs(&b, &b, &b, &b, &b, &b, &b, &b, 1, 8, 2, 64, 3, 64, 1.0, 64, 10000.0, null, 0),
    );
}
