//! Grouped-affine U4 embedding lookup kernel wrapper.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");
const registry_mod = @import("registry.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_embedding_lookup_gaffine_u4_f32";
pub const op_name: []const u8 = "embedding_lookup_gaffine_u4_f32";

pub fn run(
    allocator: std.mem.Allocator,
    device: *device_mod.Device,
    registry: *registry_mod.Registry,
    out: *device_mod.Buffer,
    packed_vals: *const device_mod.Buffer,
    scales: *const device_mod.Buffer,
    biases: *const device_mod.Buffer,
    vocab_size: u32,
    hidden_dim: u32,
    token: u32,
    group_size: u32,
    scales_dtype_tag: u32,
    multiplier: f32,
) !registry_mod.KernelSource {
    try validateArgs(out, packed_vals, scales, biases, vocab_size, hidden_dim, token, group_size, scales_dtype_tag);

    if (registry.embedded_module == null) try registry.loadEmbeddedModule(embedded_module);
    const resolved = try registry.resolveFunction(op_name, embedded_symbol);
    var arg_pack = args_mod.ArgPack.init(allocator);
    defer arg_pack.deinit();
    try runWithFunction(
        &arg_pack,
        device,
        resolved.function,
        out,
        packed_vals,
        scales,
        biases,
        vocab_size,
        hidden_dim,
        token,
        group_size,
        scales_dtype_tag,
        multiplier,
    );
    return resolved.source;
}

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    out: *device_mod.Buffer,
    packed_vals: *const device_mod.Buffer,
    scales: *const device_mod.Buffer,
    biases: *const device_mod.Buffer,
    vocab_size: u32,
    hidden_dim: u32,
    token: u32,
    group_size: u32,
    scales_dtype_tag: u32,
    multiplier: f32,
) !void {
    try validateArgs(out, packed_vals, scales, biases, vocab_size, hidden_dim, token, group_size, scales_dtype_tag);

    arg_pack.reset();
    try arg_pack.appendBufferPtr(out);
    try arg_pack.appendBufferPtr(packed_vals);
    try arg_pack.appendBufferPtr(scales);
    try arg_pack.appendBufferPtr(biases);
    try arg_pack.appendScalar(u32, vocab_size);
    try arg_pack.appendScalar(u32, hidden_dim);
    try arg_pack.appendScalar(u32, token);
    try arg_pack.appendScalar(u32, group_size);
    try arg_pack.appendScalar(u32, scales_dtype_tag);
    try arg_pack.appendScalar(f32, multiplier);

    const block_x: u32 = 256;
    try launch_mod.launch(device, function, .{
        .grid_x = ceilDiv(hidden_dim, block_x),
        .block_x = block_x,
    }, arg_pack);
}

fn validateArgs(
    out: *device_mod.Buffer,
    packed_vals: *const device_mod.Buffer,
    scales: *const device_mod.Buffer,
    biases: *const device_mod.Buffer,
    vocab_size: u32,
    hidden_dim: u32,
    token: u32,
    group_size: u32,
    scales_dtype_tag: u32,
) !void {
    if (vocab_size == 0 or hidden_dim == 0) return error.InvalidArgument;
    if (group_size == 0 or (hidden_dim % group_size) != 0 or (group_size % 8) != 0) return error.InvalidArgument;
    if (token >= vocab_size) return error.InvalidArgument;
    if (scales_dtype_tag != 0 and scales_dtype_tag != 1) return error.InvalidArgument;

    const out_bytes = std.math.mul(usize, hidden_dim, @sizeOf(f32)) catch return error.InvalidArgument;
    if (out.size < out_bytes) return error.InvalidArgument;

    const packed_words_per_row = std.math.divExact(usize, hidden_dim, 8) catch return error.InvalidArgument;
    const packed_words_total = std.math.mul(usize, vocab_size, packed_words_per_row) catch return error.InvalidArgument;
    const packed_bytes = std.math.mul(usize, packed_words_total, @sizeOf(u32)) catch return error.InvalidArgument;
    if (packed_vals.size < packed_bytes) return error.InvalidArgument;

    const groups_per_row = std.math.divExact(usize, hidden_dim, group_size) catch return error.InvalidArgument;
    const sb_elems = std.math.mul(usize, vocab_size, groups_per_row) catch return error.InvalidArgument;
    const sb_bytes = std.math.mul(usize, sb_elems, @sizeOf(u16)) catch return error.InvalidArgument;
    if (scales.size < sb_bytes or biases.size < sb_bytes) return error.InvalidArgument;
}

fn ceilDiv(numerator: u32, denominator: u32) u32 {
    return (numerator + denominator - 1) / denominator;
}

test "validateArgs rejects invalid group size" {
    var out = device_mod.Buffer{ .pointer = 0, .size = 64 };
    const packed_vals = device_mod.Buffer{ .pointer = 0, .size = 64 };
    const scales = device_mod.Buffer{ .pointer = 0, .size = 64 };
    const biases = device_mod.Buffer{ .pointer = 0, .size = 64 };
    try std.testing.expectError(
        error.InvalidArgument,
        validateArgs(&out, &packed_vals, &scales, &biases, 2, 32, 0, 7, 1),
    );
}
