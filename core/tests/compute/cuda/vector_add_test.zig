//! Integration tests for modular CUDA vector add wrapper.

const std = @import("std");
const main = @import("main");
const cuda = main.core.compute.cuda;

test "vector_add.run returns InvalidArgument for zero element count" {
    var fake_device: cuda.Device = undefined;
    var registry = cuda.Registry.init(std.testing.allocator, &fake_device);
    defer {
        registry.embedded_module = null;
        registry.sideload_module = null;
        registry.sideload_manifest = null;
    }

    const a = cuda.Buffer{ .pointer = 0, .size = 16 };
    const b = cuda.Buffer{ .pointer = 0, .size = 16 };
    var out = cuda.Buffer{ .pointer = 0, .size = 16 };

    try std.testing.expectError(
        error.InvalidArgument,
        cuda.vector_add.run(std.testing.allocator, &fake_device, &registry, &a, &b, &out, 0),
    );
}

test "vector_add.run computes expected output on CUDA" {
    if (cuda.probeRuntime() != .available) return error.SkipZigTest;

    var device = cuda.Device.init() catch |err| {
        if (err == error.CudaInitFailed or err == error.CudaNoDevices) return error.SkipZigTest;
        return err;
    };
    defer device.deinit();

    if (!device.supportsModuleLaunch()) return error.SkipZigTest;

    var registry = cuda.Registry.init(std.testing.allocator, &device);
    defer registry.deinit();

    const lhs = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const rhs = [_]f32{ 10.0, 20.0, 30.0, 40.0 };
    const expected = [_]f32{ 11.0, 22.0, 33.0, 44.0 };
    var actual = [_]f32{0.0} ** lhs.len;

    var lhs_dev = try device.allocBuffer(lhs.len * @sizeOf(f32));
    defer lhs_dev.deinit(&device);
    var rhs_dev = try device.allocBuffer(rhs.len * @sizeOf(f32));
    defer rhs_dev.deinit(&device);
    var out_dev = try device.allocBuffer(actual.len * @sizeOf(f32));
    defer out_dev.deinit(&device);

    try lhs_dev.upload(&device, std.mem.sliceAsBytes(lhs[0..]));
    try rhs_dev.upload(&device, std.mem.sliceAsBytes(rhs[0..]));
    const source = try cuda.vector_add.run(
        std.testing.allocator,
        &device,
        &registry,
        &lhs_dev,
        &rhs_dev,
        &out_dev,
        @intCast(lhs.len),
    );
    try std.testing.expectEqual(cuda.registry.KernelSource.embedded_ptx, source);
    try device.synchronize();
    try out_dev.download(&device, std.mem.sliceAsBytes(actual[0..]));

    for (expected, actual) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.0001);
    }
}
