//! Integration tests for modular CUDA RMSNorm wrapper.

const std = @import("std");
const main = @import("main");
const cuda = main.core.compute.cuda;

test "rmsnorm.run returns InvalidArgument for non-positive epsilon" {
    var fake_device: cuda.Device = undefined;
    var registry = cuda.Registry.init(std.testing.allocator, &fake_device);
    defer {
        registry.embedded_module = null;
        registry.sideload_module = null;
        registry.sideload_manifest = null;
    }

    const input = cuda.Buffer{ .pointer = 0, .size = 16 };
    const weight = cuda.Buffer{ .pointer = 0, .size = 16 };
    var output = cuda.Buffer{ .pointer = 0, .size = 16 };

    try std.testing.expectError(
        error.InvalidArgument,
        cuda.rmsnorm.run(
            std.testing.allocator,
            &fake_device,
            &registry,
            &input,
            &weight,
            &output,
            1,
            4,
            0.0,
            0.0,
        ),
    );
}

test "rmsnorm.run computes expected output on CUDA" {
    if (cuda.probeRuntime() != .available) return error.SkipZigTest;

    var device = cuda.Device.init() catch |err| {
        if (err == error.CudaInitFailed or err == error.CudaNoDevices) return error.SkipZigTest;
        return err;
    };
    defer device.deinit();

    if (!device.supportsModuleLaunch()) return error.SkipZigTest;

    var registry = cuda.Registry.init(std.testing.allocator, &device);
    defer registry.deinit();

    const rows: u32 = 2;
    const cols: u32 = 4;
    const eps: f32 = 1e-5;
    const input = [_]f32{
        1.0,  2.0, 3.0, 4.0,
        -1.0, 0.0, 1.0, 2.0,
    };
    const weight = [_]f32{ 1.0, 1.5, 0.5, 2.0 };
    var expected = [_]f32{0.0} ** input.len;
    var actual = [_]f32{0.0} ** input.len;

    computeReference(&expected, &input, &weight, rows, cols, eps);

    var input_dev = try device.allocBuffer(input.len * @sizeOf(f32));
    defer input_dev.deinit(&device);
    var weight_dev = try device.allocBuffer(weight.len * @sizeOf(f32));
    defer weight_dev.deinit(&device);
    var output_dev = try device.allocBuffer(actual.len * @sizeOf(f32));
    defer output_dev.deinit(&device);

    try input_dev.upload(&device, std.mem.sliceAsBytes(input[0..]));
    try weight_dev.upload(&device, std.mem.sliceAsBytes(weight[0..]));
    const source = try cuda.rmsnorm.run(
        std.testing.allocator,
        &device,
        &registry,
        &input_dev,
        &weight_dev,
        &output_dev,
        rows,
        cols,
        eps,
        0.0,
    );
    try std.testing.expectEqual(cuda.registry.KernelSource.embedded_module, source);
    try device.synchronize();
    try output_dev.download(&device, std.mem.sliceAsBytes(actual[0..]));

    for (expected, actual) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.001);
    }
}

fn computeReference(
    out: []f32,
    input: []const f32,
    weight: []const f32,
    rows: u32,
    cols: u32,
    eps: f32,
) void {
    const rows_usize: usize = @intCast(rows);
    const cols_usize: usize = @intCast(cols);
    var row: usize = 0;
    while (row < rows_usize) : (row += 1) {
        const base = row * cols_usize;
        var sum_sq: f32 = 0.0;
        var col: usize = 0;
        while (col < cols_usize) : (col += 1) {
            const v = input[base + col];
            sum_sq += v * v;
        }
        const mean_sq = sum_sq / @as(f32, @floatFromInt(cols_usize));
        const inv_rms = 1.0 / std.math.sqrt(mean_sq + eps);
        col = 0;
        while (col < cols_usize) : (col += 1) {
            out[base + col] = input[base + col] * inv_rms * weight[col];
        }
    }
}
