//! Attention score kernel wrapper for modular CUDA runtime.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");
const registry_mod = @import("registry.zig");

pub const embedded_ptx = @embedFile("kernels/fallback_kernels.ptx");
pub const embedded_symbol: [:0]const u8 = "talu_attn_scores_f32_v1";

pub fn run(
    allocator: std.mem.Allocator,
    device: *device_mod.Device,
    registry: *registry_mod.Registry,
    query_head: *const device_mod.Buffer,
    key_cache: *const device_mod.Buffer,
    scores_out: *device_mod.Buffer,
    seq_len: u32,
    row_stride: u32,
    head_offset: u32,
    head_dim: u32,
    scale: f32,
) !registry_mod.KernelSource {
    try validateArgs(query_head, key_cache, scores_out, seq_len, row_stride, head_offset, head_dim, scale);

    if (registry.embedded_module == null) try registry.loadEmbeddedModule(embedded_ptx);
    const resolved = try registry.resolveFunction("attn_scores_f32", embedded_symbol);
    var arg_pack = args_mod.ArgPack.init(allocator);
    defer arg_pack.deinit();
    try runWithFunction(&arg_pack, device, resolved.function, query_head, key_cache, scores_out, seq_len, row_stride, head_offset, head_dim, scale);
    return resolved.source;
}

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    query_head: *const device_mod.Buffer,
    key_cache: *const device_mod.Buffer,
    scores_out: *device_mod.Buffer,
    seq_len: u32,
    row_stride: u32,
    head_offset: u32,
    head_dim: u32,
    scale: f32,
) !void {
    try validateArgs(query_head, key_cache, scores_out, seq_len, row_stride, head_offset, head_dim, scale);

    arg_pack.reset();
    try arg_pack.appendBufferPtr(scores_out);
    try arg_pack.appendBufferPtr(query_head);
    try arg_pack.appendBufferPtr(key_cache);
    try arg_pack.appendScalar(u32, seq_len);
    try arg_pack.appendScalar(u32, row_stride);
    try arg_pack.appendScalar(u32, head_offset);
    try arg_pack.appendScalar(u32, head_dim);
    try arg_pack.appendScalar(f32, scale);

    const block_x: u32 = 256;
    const grid_x: u32 = ceilDiv(seq_len, block_x);
    try launch_mod.launch(device, function, .{
        .grid_x = grid_x,
        .block_x = block_x,
    }, arg_pack);
}

fn validateArgs(
    query_head: *const device_mod.Buffer,
    key_cache: *const device_mod.Buffer,
    scores_out: *device_mod.Buffer,
    seq_len: u32,
    row_stride: u32,
    head_offset: u32,
    head_dim: u32,
    scale: f32,
) !void {
    if (seq_len == 0 or row_stride == 0 or head_dim == 0) return error.InvalidArgument;
    if (head_offset + head_dim > row_stride) return error.InvalidArgument;
    if (!std.math.isFinite(scale)) return error.InvalidArgument;

    const query_bytes = @as(usize, head_dim) * @sizeOf(f32);
    const cache_bytes = @as(usize, seq_len) * @as(usize, row_stride) * @sizeOf(f32);
    const scores_bytes = @as(usize, seq_len) * @sizeOf(f32);
    if (query_head.size < query_bytes or key_cache.size < cache_bytes or scores_out.size < scores_bytes) {
        return error.InvalidArgument;
    }
}

fn ceilDiv(numerator: u32, denominator: u32) u32 {
    return (numerator + denominator - 1) / denominator;
}

test "validateArgs rejects zero seq_len" {
    const b = device_mod.Buffer{ .pointer = 0, .size = 1024 };
    var out = b;
    try std.testing.expectError(error.InvalidArgument, validateArgs(&b, &b, &out, 0, 64, 0, 64, 1.0));
}

test "validateArgs rejects invalid head window" {
    const b = device_mod.Buffer{ .pointer = 0, .size = 4096 };
    var out = b;
    try std.testing.expectError(error.InvalidArgument, validateArgs(&b, &b, &out, 8, 64, 48, 32, 1.0));
}
