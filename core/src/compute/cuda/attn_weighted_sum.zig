//! Attention weighted-sum kernel wrapper for modular CUDA runtime.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");
const registry_mod = @import("registry.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_attn_weighted_sum_f32";
pub const op_name: []const u8 = "attn_weighted_sum_f32";

pub fn run(
    allocator: std.mem.Allocator,
    device: *device_mod.Device,
    registry: *registry_mod.Registry,
    probs: *const device_mod.Buffer,
    value_cache: *const device_mod.Buffer,
    out_head: *device_mod.Buffer,
    seq_len: u32,
    row_stride: u32,
    head_offset: u32,
    head_dim: u32,
) !registry_mod.KernelSource {
    try validateArgs(probs, value_cache, out_head, seq_len, row_stride, head_offset, head_dim);

    if (registry.embedded_module == null) try registry.loadEmbeddedModule(embedded_module);
    const resolved = try registry.resolveFunction(op_name, embedded_symbol);
    var arg_pack = args_mod.ArgPack.init(allocator);
    defer arg_pack.deinit();
    try runWithFunction(&arg_pack, device, resolved.function, probs, value_cache, out_head, seq_len, row_stride, head_offset, head_dim);
    return resolved.source;
}

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    probs: *const device_mod.Buffer,
    value_cache: *const device_mod.Buffer,
    out_head: *device_mod.Buffer,
    seq_len: u32,
    row_stride: u32,
    head_offset: u32,
    head_dim: u32,
) !void {
    try validateArgs(probs, value_cache, out_head, seq_len, row_stride, head_offset, head_dim);

    arg_pack.reset();
    try arg_pack.appendBufferPtr(out_head);
    try arg_pack.appendBufferPtr(probs);
    try arg_pack.appendBufferPtr(value_cache);
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
    value_cache: *const device_mod.Buffer,
    out_head: *device_mod.Buffer,
    seq_len: u32,
    row_stride: u32,
    head_offset: u32,
    head_dim: u32,
) !void {
    if (seq_len == 0 or row_stride == 0 or head_dim == 0) return error.InvalidArgument;
    if (head_offset + head_dim > row_stride) return error.InvalidArgument;

    const probs_bytes = @as(usize, seq_len) * @sizeOf(f32);
    const cache_bytes = @as(usize, seq_len) * @as(usize, row_stride) * @sizeOf(f32);
    const out_bytes = @as(usize, head_dim) * @sizeOf(f32);
    if (probs.size < probs_bytes or value_cache.size < cache_bytes or out_head.size < out_bytes) {
        return error.InvalidArgument;
    }
}

fn ceilDiv(numerator: u32, denominator: u32) u32 {
    return (numerator + denominator - 1) / denominator;
}

test "validateArgs rejects invalid head window" {
    const b = device_mod.Buffer{ .pointer = 0, .size = 4096 };
    var out = b;
    try std.testing.expectError(error.InvalidArgument, validateArgs(&b, &b, &out, 8, 64, 60, 8));
}
