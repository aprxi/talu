//! RMSNorm kernel wrapper for modular CUDA runtime.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");
const registry_mod = @import("registry.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_rmsnorm_f32";

pub fn run(
    allocator: std.mem.Allocator,
    device: *device_mod.Device,
    registry: *registry_mod.Registry,
    input: *const device_mod.Buffer,
    weight: *const device_mod.Buffer,
    output: *device_mod.Buffer,
    rows: u32,
    cols: u32,
    eps: f32,
    weight_offset: f32,
) !registry_mod.KernelSource {
    if (rows == 0 or cols == 0) return error.InvalidArgument;
    if (!std.math.isFinite(eps) or eps <= 0.0) return error.InvalidArgument;

    const input_output_bytes = @as(usize, rows) * @as(usize, cols) * @sizeOf(f32);
    const weight_bytes = @as(usize, cols) * @sizeOf(f32);
    if (input.size < input_output_bytes or output.size < input_output_bytes or weight.size < weight_bytes) {
        return error.InvalidArgument;
    }

    if (registry.embedded_module == null) try registry.loadEmbeddedModule(embedded_module);
    const resolved = try registry.resolveFunction("rmsnorm_f32", embedded_symbol);
    var arg_pack = args_mod.ArgPack.init(allocator);
    defer arg_pack.deinit();
    try runWithFunction(&arg_pack, device, resolved.function, input, weight, output, rows, cols, eps, weight_offset);
    return resolved.source;
}

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    input: *const device_mod.Buffer,
    weight: *const device_mod.Buffer,
    output: *device_mod.Buffer,
    rows: u32,
    cols: u32,
    eps: f32,
    weight_offset: f32,
) !void {
    if (rows == 0 or cols == 0) return error.InvalidArgument;
    if (!std.math.isFinite(eps) or eps <= 0.0) return error.InvalidArgument;

    const input_output_bytes = @as(usize, rows) * @as(usize, cols) * @sizeOf(f32);
    const weight_bytes = @as(usize, cols) * @sizeOf(f32);
    if (input.size < input_output_bytes or output.size < input_output_bytes or weight.size < weight_bytes) {
        return error.InvalidArgument;
    }

    arg_pack.reset();
    try arg_pack.appendBufferPtr(output);
    try arg_pack.appendBufferPtr(input);
    try arg_pack.appendBufferPtr(weight);
    try arg_pack.appendScalar(u32, rows);
    try arg_pack.appendScalar(u32, cols);
    try arg_pack.appendScalar(f32, eps);
    try arg_pack.appendScalar(f32, weight_offset);

    try launch_mod.launch(device, function, .{
        .grid_x = rows,
        .block_x = 256,
        .shared_mem_bytes = 256 * @sizeOf(f32),
    }, arg_pack);
}

test "run validates rows and cols" {
    var fake_device: device_mod.Device = undefined;
    var registry = registry_mod.Registry.init(std.testing.allocator, &fake_device);
    defer {
        registry.embedded_module = null;
        registry.sideload_module = null;
        registry.sideload_manifest = null;
    }

    const fake_buffer = device_mod.Buffer{ .pointer = 0, .size = 1024 };
    var output = fake_buffer;
    try std.testing.expectError(
        error.InvalidArgument,
        run(
            std.testing.allocator,
            &fake_device,
            &registry,
            &fake_buffer,
            &fake_buffer,
            &output,
            0,
            128,
            1e-5,
            0.0,
        ),
    );
}

test "run validates eps" {
    var fake_device: device_mod.Device = undefined;
    var registry = registry_mod.Registry.init(std.testing.allocator, &fake_device);
    defer {
        registry.embedded_module = null;
        registry.sideload_module = null;
        registry.sideload_manifest = null;
    }

    const fake_buffer = device_mod.Buffer{ .pointer = 0, .size = 1024 };
    var output = fake_buffer;
    try std.testing.expectError(
        error.InvalidArgument,
        run(
            std.testing.allocator,
            &fake_device,
            &registry,
            &fake_buffer,
            &fake_buffer,
            &output,
            1,
            128,
            0.0,
            0.0,
        ),
    );
}
