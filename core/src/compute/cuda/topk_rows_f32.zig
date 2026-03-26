//! Batched top-k kernel wrapper over row-major logits [rows, vocab].

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");
const registry_mod = @import("registry.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_topk_rows_f32";
pub const op_name: []const u8 = "topk_rows_f32";

pub fn run(
    allocator: std.mem.Allocator,
    device: *device_mod.Device,
    registry: *registry_mod.Registry,
    values_out: *device_mod.Buffer,
    ids_out: *device_mod.Buffer,
    logits_inout: *device_mod.Buffer,
    rows: u32,
    vocab: u32,
    row_stride: u32,
    k: u32,
) !registry_mod.KernelSource {
    try validateArgs(values_out, ids_out, logits_inout, rows, vocab, row_stride, k);

    if (registry.embedded_module == null) try registry.loadEmbeddedModule(embedded_module);
    const resolved = try registry.resolveFunction(op_name, embedded_symbol);

    var arg_pack = args_mod.ArgPack.init(allocator);
    defer arg_pack.deinit();
    try runWithFunction(&arg_pack, device, resolved.function, values_out, ids_out, logits_inout, rows, vocab, row_stride, k);
    return resolved.source;
}

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    values_out: *device_mod.Buffer,
    ids_out: *device_mod.Buffer,
    logits_inout: *device_mod.Buffer,
    rows: u32,
    vocab: u32,
    row_stride: u32,
    k: u32,
) !void {
    try validateArgs(values_out, ids_out, logits_inout, rows, vocab, row_stride, k);

    arg_pack.reset();
    try arg_pack.appendBufferPtr(values_out);
    try arg_pack.appendBufferPtr(ids_out);
    try arg_pack.appendBufferPtr(logits_inout);
    try arg_pack.appendScalar(u32, rows);
    try arg_pack.appendScalar(u32, vocab);
    try arg_pack.appendScalar(u32, row_stride);
    try arg_pack.appendScalar(u32, k);

    const block_x: u32 = 256;
    try launch_mod.launchWithFamily(device, function, .{
        .grid_x = rows,
        .block_x = block_x,
    }, arg_pack, .pointwise);
}

fn validateArgs(
    values_out: *device_mod.Buffer,
    ids_out: *device_mod.Buffer,
    logits_inout: *device_mod.Buffer,
    rows: u32,
    vocab: u32,
    row_stride: u32,
    k: u32,
) !void {
    if (rows == 0 or vocab == 0 or row_stride == 0 or k == 0) return error.InvalidArgument;
    if (k > row_stride) return error.InvalidArgument;

    const logits_count = std.math.mul(usize, @as(usize, rows), @as(usize, vocab)) catch return error.InvalidArgument;
    const logits_bytes = std.math.mul(usize, logits_count, @sizeOf(f32)) catch return error.InvalidArgument;
    const out_count = std.math.mul(usize, @as(usize, rows), @as(usize, row_stride)) catch return error.InvalidArgument;
    const out_values_bytes = std.math.mul(usize, out_count, @sizeOf(f32)) catch return error.InvalidArgument;
    const out_ids_bytes = std.math.mul(usize, out_count, @sizeOf(u32)) catch return error.InvalidArgument;

    if (logits_inout.size < logits_bytes) return error.InvalidArgument;
    if (values_out.size < out_values_bytes) return error.InvalidArgument;
    if (ids_out.size < out_ids_bytes) return error.InvalidArgument;
}

test "validateArgs rejects invalid top-k row shape" {
    var values = device_mod.Buffer{ .pointer = 0, .size = 4096 };
    var ids = device_mod.Buffer{ .pointer = 0, .size = 4096 };
    var logits = device_mod.Buffer{ .pointer = 0, .size = 4096 };

    try std.testing.expectError(error.InvalidArgument, validateArgs(&values, &ids, &logits, 0, 8, 4, 4));
    try std.testing.expectError(error.InvalidArgument, validateArgs(&values, &ids, &logits, 1, 0, 4, 4));
    try std.testing.expectError(error.InvalidArgument, validateArgs(&values, &ids, &logits, 1, 8, 0, 1));
    try std.testing.expectError(error.InvalidArgument, validateArgs(&values, &ids, &logits, 1, 8, 4, 5));
}
