//! Batched U16 embedding lookup kernel wrapper (f16/bf16 source, f32 output).

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_embedding_lookup_u16_rows_f32";
pub const op_name: []const u8 = "embedding_lookup_u16_rows_f32";

pub const layout_vocab_hidden: u32 = 0;
pub const layout_hidden_vocab: u32 = 1;
pub const dtype_f16: u32 = 0;
pub const dtype_bf16: u32 = 1;

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    out: *device_mod.Buffer,
    embeddings: *const device_mod.Buffer,
    tokens: *const device_mod.Buffer,
    rows: u32,
    dim0: u32,
    dim1: u32,
    hidden_dim: u32,
    layout_tag: u32,
    dtype_tag: u32,
    multiplier: f32,
) !void {
    try validateArgs(out, embeddings, tokens, rows, dim0, dim1, hidden_dim, layout_tag, dtype_tag);

    arg_pack.reset();
    try arg_pack.appendBufferPtr(out);
    try arg_pack.appendBufferPtr(embeddings);
    try arg_pack.appendBufferPtr(tokens);
    try arg_pack.appendScalar(u32, rows);
    try arg_pack.appendScalar(u32, dim0);
    try arg_pack.appendScalar(u32, dim1);
    try arg_pack.appendScalar(u32, hidden_dim);
    try arg_pack.appendScalar(u32, layout_tag);
    try arg_pack.appendScalar(u32, dtype_tag);
    try arg_pack.appendScalar(f32, multiplier);

    const total = std.math.mul(u32, rows, hidden_dim) catch return error.InvalidArgument;
    const block_x: u32 = 256;
    const grid_x: u32 = ceilDiv(total, block_x);
    try launch_mod.launchWithFamily(device, function, .{
        .grid_x = grid_x,
        .block_x = block_x,
    }, arg_pack, .embedding);
}

fn validateArgs(
    out: *device_mod.Buffer,
    embeddings: *const device_mod.Buffer,
    tokens: *const device_mod.Buffer,
    rows: u32,
    dim0: u32,
    dim1: u32,
    hidden_dim: u32,
    layout_tag: u32,
    dtype_tag: u32,
) !void {
    if (rows == 0 or dim0 == 0 or dim1 == 0 or hidden_dim == 0) return error.InvalidArgument;
    if (layout_tag != layout_vocab_hidden and layout_tag != layout_hidden_vocab) return error.InvalidArgument;
    if (dtype_tag != dtype_f16 and dtype_tag != dtype_bf16) return error.InvalidArgument;

    const embed_count = std.math.mul(usize, dim0, dim1) catch return error.InvalidArgument;
    const embed_bytes = std.math.mul(usize, embed_count, @sizeOf(u16)) catch return error.InvalidArgument;
    if (embeddings.size < embed_bytes) return error.InvalidArgument;

    const token_bytes = std.math.mul(usize, rows, @sizeOf(u32)) catch return error.InvalidArgument;
    if (tokens.size < token_bytes) return error.InvalidArgument;

    const out_elems = std.math.mul(usize, rows, hidden_dim) catch return error.InvalidArgument;
    const out_bytes = std.math.mul(usize, out_elems, @sizeOf(f32)) catch return error.InvalidArgument;
    if (out.size < out_bytes) return error.InvalidArgument;

    if (layout_tag == layout_vocab_hidden) {
        if (hidden_dim > dim1) return error.InvalidArgument;
    } else {
        if (hidden_dim > dim0) return error.InvalidArgument;
    }
}

fn ceilDiv(numerator: u32, denominator: u32) u32 {
    return (numerator + denominator - 1) / denominator;
}

test "validateArgs rejects invalid dtype tag" {
    var out = device_mod.Buffer{ .pointer = 0, .size = 32 * @sizeOf(f32) };
    const embeddings = device_mod.Buffer{ .pointer = 0, .size = 32 * @sizeOf(u16) };
    const tokens = device_mod.Buffer{ .pointer = 0, .size = 4 * @sizeOf(u32) };
    try std.testing.expectError(
        error.InvalidArgument,
        validateArgs(
            &out,
            &embeddings,
            &tokens,
            4,
            4,
            8,
            8,
            layout_vocab_hidden,
            99,
        ),
    );
}
