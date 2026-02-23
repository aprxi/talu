//! U16 embedding lookup kernel wrapper (f16/bf16 source, f32 output).

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");
const registry_mod = @import("registry.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_embedding_lookup_u16_f32";
pub const op_name: []const u8 = "embedding_lookup_u16_f32";

pub const layout_vocab_hidden: u32 = 0;
pub const layout_hidden_vocab: u32 = 1;
pub const dtype_f16: u32 = 0;
pub const dtype_bf16: u32 = 1;

pub fn run(
    allocator: std.mem.Allocator,
    device: *device_mod.Device,
    registry: *registry_mod.Registry,
    out: *device_mod.Buffer,
    embeddings: *const device_mod.Buffer,
    dim0: u32,
    dim1: u32,
    hidden_dim: u32,
    token: u32,
    layout_tag: u32,
    dtype_tag: u32,
    multiplier: f32,
) !registry_mod.KernelSource {
    try validateArgs(out, embeddings, dim0, dim1, hidden_dim, token, layout_tag, dtype_tag);

    if (registry.embedded_module == null) try registry.loadEmbeddedModule(embedded_module);
    const resolved = try registry.resolveFunction(op_name, embedded_symbol);
    var arg_pack = args_mod.ArgPack.init(allocator);
    defer arg_pack.deinit();
    try runWithFunction(
        &arg_pack,
        device,
        resolved.function,
        out,
        embeddings,
        dim0,
        dim1,
        hidden_dim,
        token,
        layout_tag,
        dtype_tag,
        multiplier,
    );
    return resolved.source;
}

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    out: *device_mod.Buffer,
    embeddings: *const device_mod.Buffer,
    dim0: u32,
    dim1: u32,
    hidden_dim: u32,
    token: u32,
    layout_tag: u32,
    dtype_tag: u32,
    multiplier: f32,
) !void {
    try validateArgs(out, embeddings, dim0, dim1, hidden_dim, token, layout_tag, dtype_tag);

    arg_pack.reset();
    try arg_pack.appendBufferPtr(out);
    try arg_pack.appendBufferPtr(embeddings);
    try arg_pack.appendScalar(u32, dim0);
    try arg_pack.appendScalar(u32, dim1);
    try arg_pack.appendScalar(u32, hidden_dim);
    try arg_pack.appendScalar(u32, token);
    try arg_pack.appendScalar(u32, layout_tag);
    try arg_pack.appendScalar(u32, dtype_tag);
    try arg_pack.appendScalar(f32, multiplier);

    const block_x: u32 = 256;
    const grid_x: u32 = ceilDiv(hidden_dim, block_x);
    try launch_mod.launch(device, function, .{
        .grid_x = grid_x,
        .block_x = block_x,
    }, arg_pack);
}

fn validateArgs(
    out: *device_mod.Buffer,
    embeddings: *const device_mod.Buffer,
    dim0: u32,
    dim1: u32,
    hidden_dim: u32,
    token: u32,
    layout_tag: u32,
    dtype_tag: u32,
) !void {
    if (dim0 == 0 or dim1 == 0 or hidden_dim == 0) return error.InvalidArgument;
    if (layout_tag != layout_vocab_hidden and layout_tag != layout_hidden_vocab) return error.InvalidArgument;
    if (dtype_tag != dtype_f16 and dtype_tag != dtype_bf16) return error.InvalidArgument;

    const embed_count = std.math.mul(usize, dim0, dim1) catch return error.InvalidArgument;
    const embed_bytes = std.math.mul(usize, embed_count, @sizeOf(u16)) catch return error.InvalidArgument;
    if (embeddings.size < embed_bytes) return error.InvalidArgument;

    const out_bytes = std.math.mul(usize, hidden_dim, @sizeOf(f32)) catch return error.InvalidArgument;
    if (out.size < out_bytes) return error.InvalidArgument;

    if (layout_tag == layout_vocab_hidden) {
        if (token >= dim0 or hidden_dim > dim1) return error.InvalidArgument;
    } else {
        if (token >= dim1 or hidden_dim > dim0) return error.InvalidArgument;
    }
}

fn ceilDiv(numerator: u32, denominator: u32) u32 {
    return (numerator + denominator - 1) / denominator;
}

test "run rejects invalid dtype tag" {
    var fake_device: device_mod.Device = undefined;
    var registry = registry_mod.Registry.init(std.testing.allocator, &fake_device);
    defer {
        registry.embedded_module = null;
        registry.sideload_module = null;
        registry.sideload_manifest = null;
    }

    var out = device_mod.Buffer{ .pointer = 0, .size = 32 };
    const embeddings = device_mod.Buffer{ .pointer = 0, .size = 32 };
    try std.testing.expectError(
        error.InvalidArgument,
        run(
            std.testing.allocator,
            &fake_device,
            &registry,
            &out,
            &embeddings,
            2,
            2,
            2,
            0,
            layout_vocab_hidden,
            99,
            1.0,
        ),
    );
}
