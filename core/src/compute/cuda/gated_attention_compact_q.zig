//! Query-gated attention projection compaction kernel wrapper.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_gated_attention_compact_q_f32";
pub const op_name: []const u8 = "gated_attention_compact_q_f32";

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    packed_query: *const device_mod.Buffer,
    compact_query: *device_mod.Buffer,
    seq_len: u32,
    query_dim: u32,
    query_projection_dim: u32,
    head_count: u32,
    head_dim: u32,
) !void {
    try validateArgs(packed_query, compact_query, seq_len, query_dim, query_projection_dim, head_count, head_dim);

    arg_pack.reset();
    try arg_pack.appendBufferPtr(compact_query);
    try arg_pack.appendBufferPtr(packed_query);
    try arg_pack.appendScalar(u32, seq_len);
    try arg_pack.appendScalar(u32, query_dim);
    try arg_pack.appendScalar(u32, query_projection_dim);
    try arg_pack.appendScalar(u32, head_count);
    try arg_pack.appendScalar(u32, head_dim);

    const compact_elements = std.math.mul(u32, seq_len, query_dim) catch return error.InvalidArgument;
    const block_x: u32 = 256;
    try launch_mod.launchWithFamily(device, function, .{
        .grid_x = ceilDiv(compact_elements, block_x),
        .block_x = block_x,
    }, arg_pack, .attention);
}

fn validateArgs(
    packed_query: *const device_mod.Buffer,
    compact_query: *const device_mod.Buffer,
    seq_len: u32,
    query_dim: u32,
    query_projection_dim: u32,
    head_count: u32,
    head_dim: u32,
) !void {
    if (seq_len == 0 or query_dim == 0 or query_projection_dim == 0 or head_count == 0 or head_dim == 0) {
        return error.InvalidArgument;
    }
    if ((std.math.mul(u32, head_count, head_dim) catch return error.InvalidArgument) != query_dim) {
        return error.InvalidArgument;
    }
    if ((std.math.mul(u32, query_dim, 2) catch return error.InvalidArgument) != query_projection_dim) {
        return error.InvalidArgument;
    }

    const packed_bytes = std.math.mul(
        usize,
        std.math.mul(usize, seq_len, query_projection_dim) catch return error.InvalidArgument,
        @sizeOf(f32),
    ) catch return error.InvalidArgument;
    const compact_bytes = std.math.mul(
        usize,
        std.math.mul(usize, seq_len, query_dim) catch return error.InvalidArgument,
        @sizeOf(f32),
    ) catch return error.InvalidArgument;
    if (packed_query.size < packed_bytes or compact_query.size < compact_bytes) {
        return error.InvalidArgument;
    }
}

fn ceilDiv(numerator: u32, denominator: u32) u32 {
    return (numerator + denominator - 1) / denominator;
}

test "validateArgs rejects invalid dimensions" {
    const buf = device_mod.Buffer{ .pointer = 0, .size = 64 };
    try std.testing.expectError(
        error.InvalidArgument,
        validateArgs(&buf, &buf, 1, 4, 7, 2, 2),
    );
    try std.testing.expectError(
        error.InvalidArgument,
        validateArgs(&buf, &buf, 0, 4, 8, 2, 2),
    );
}

test "validateArgs rejects undersized buffers" {
    const packed_buf = device_mod.Buffer{ .pointer = 0, .size = 4 * @sizeOf(f32) };
    const compact = device_mod.Buffer{ .pointer = 0, .size = 2 * @sizeOf(f32) };
    try std.testing.expectError(
        error.InvalidArgument,
        validateArgs(&packed_buf, &compact, 2, 4, 8, 2, 2),
    );
}
