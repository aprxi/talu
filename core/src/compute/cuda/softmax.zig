//! Softmax kernel wrapper for modular CUDA runtime.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");
const registry_mod = @import("registry.zig");

pub const embedded_ptx = @embedFile("kernels/fallback_kernels.ptx");
pub const embedded_symbol: [:0]const u8 = "talu_softmax_f32_v1";

pub fn run(
    allocator: std.mem.Allocator,
    device: *device_mod.Device,
    registry: *registry_mod.Registry,
    input: *const device_mod.Buffer,
    output: *device_mod.Buffer,
    count: u32,
) !registry_mod.KernelSource {
    try validateArgs(input, output, count);

    if (registry.embedded_module == null) try registry.loadEmbeddedModule(embedded_ptx);
    const resolved = try registry.resolveFunction("softmax_f32", embedded_symbol);
    var arg_pack = args_mod.ArgPack.init(allocator);
    defer arg_pack.deinit();
    try runWithFunction(&arg_pack, device, resolved.function, input, output, count);
    return resolved.source;
}

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    input: *const device_mod.Buffer,
    output: *device_mod.Buffer,
    count: u32,
) !void {
    try validateArgs(input, output, count);

    arg_pack.reset();
    try arg_pack.appendBufferPtr(output);
    try arg_pack.appendBufferPtr(input);
    try arg_pack.appendScalar(u32, count);

    try launch_mod.launch(device, function, .{
        .grid_x = 1,
        .block_x = 1,
    }, arg_pack);
}

fn validateArgs(input: *const device_mod.Buffer, output: *device_mod.Buffer, count: u32) !void {
    if (count == 0) return error.InvalidArgument;
    const required_bytes = @as(usize, count) * @sizeOf(f32);
    if (input.size < required_bytes or output.size < required_bytes) return error.InvalidArgument;
}

test "validateArgs rejects zero count" {
    const b = device_mod.Buffer{ .pointer = 0, .size = 1024 };
    var out = b;
    try std.testing.expectError(error.InvalidArgument, validateArgs(&b, &out, 0));
}
