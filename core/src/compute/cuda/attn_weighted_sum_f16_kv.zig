//! Attention weighted-sum kernel wrapper using f16 KV cache.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");
const registry_mod = @import("registry.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_attn_weighted_sum_f16_kv";

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    probs: *const device_mod.Buffer,
    value_cache_f16: *const device_mod.Buffer,
    out_head: *device_mod.Buffer,
    seq_len: u32,
    row_stride: u32,
    head_offset: u32,
    head_dim: u32,
) !void {
    try validateArgs(probs, value_cache_f16, out_head, seq_len, row_stride, head_offset, head_dim);

    arg_pack.reset();
    try arg_pack.appendBufferPtr(out_head);
    try arg_pack.appendBufferPtr(probs);
    try arg_pack.appendBufferPtr(value_cache_f16);
    try arg_pack.appendScalar(u32, seq_len);
    try arg_pack.appendScalar(u32, row_stride);
    try arg_pack.appendScalar(u32, head_offset);
    try arg_pack.appendScalar(u32, head_dim);

    const block_x: u32 = 256;
    const grid_x: u32 = ceilDiv(head_dim, block_x);
    try launch_mod.launch(device, function, .{
        .grid_x = grid_x,
        .block_x = block_x,
    }, arg_pack);
}

fn validateArgs(
    probs: *const device_mod.Buffer,
    value_cache_f16: *const device_mod.Buffer,
    out_head: *device_mod.Buffer,
    seq_len: u32,
    row_stride: u32,
    head_offset: u32,
    head_dim: u32,
) !void {
    if (seq_len == 0 or row_stride == 0 or head_dim == 0) return error.InvalidArgument;
    if (head_offset + head_dim > row_stride) return error.InvalidArgument;

    const probs_bytes = @as(usize, seq_len) * @sizeOf(f32);
    const cache_bytes = @as(usize, seq_len) * @as(usize, row_stride) * @sizeOf(u16);
    const out_bytes = @as(usize, head_dim) * @sizeOf(f32);
    if (probs.size < probs_bytes or value_cache_f16.size < cache_bytes or out_head.size < out_bytes) {
        return error.InvalidArgument;
    }
}

fn ceilDiv(numerator: u32, denominator: u32) u32 {
    return (numerator + denominator - 1) / denominator;
}
