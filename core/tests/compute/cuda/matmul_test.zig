//! Integration tests for CUDA f32 matmul primitive.

const std = @import("std");
const main = @import("main");
const cuda = main.core.compute.cuda;

fn toBytesConst(comptime T: type, values: []const T) []const u8 {
    return std.mem.sliceAsBytes(values);
}

fn toBytesMut(comptime T: type, values: []T) []u8 {
    return std.mem.sliceAsBytes(values);
}

fn matmulCpuRef(a: []const f32, m: usize, k: usize, b: []const f32, n: usize, out: []f32) void {
    @memset(out, 0.0);
    for (0..m) |row| {
        for (0..n) |col| {
            var acc: f32 = 0.0;
            for (0..k) |inner| {
                acc += a[row * k + inner] * b[inner * n + col];
            }
            out[row * n + col] = acc;
        }
    }
}

test "cuda matmulF32 matches CPU reference within tolerance" {
    if (cuda.probeRuntime() != .available) return error.SkipZigTest;

    var device = try cuda.Device.init();
    defer device.deinit();

    var blas = cuda.Blas.init(&device) catch |err| {
        // Some CUDA driver installs omit cuBLAS runtime.
        if (err == error.CublasUnavailable or err == error.CublasSymbolMissing) return error.SkipZigTest;
        return err;
    };
    defer blas.deinit(&device);

    const m: usize = 3;
    const k: usize = 4;
    const n: usize = 2;

    const a = [_]f32{
        1.0, 2.0,  3.0,  4.0,
        5.0, 6.0,  7.0,  8.0,
        9.0, 10.0, 11.0, 12.0,
    };
    const b = [_]f32{
        0.5,  -1.0,
        1.5,  2.0,
        -0.5, 0.25,
        4.0,  3.0,
    };

    var out_gpu = [_]f32{0.0} ** (m * n);
    var out_cpu = [_]f32{0.0} ** (m * n);

    var a_dev = try device.allocBuffer(@sizeOf(f32) * a.len);
    defer a_dev.deinit(&device);
    var b_dev = try device.allocBuffer(@sizeOf(f32) * b.len);
    defer b_dev.deinit(&device);
    var c_dev = try device.allocBuffer(@sizeOf(f32) * out_gpu.len);
    defer c_dev.deinit(&device);

    try a_dev.upload(&device, toBytesConst(f32, &a));
    try b_dev.upload(&device, toBytesConst(f32, &b));
    try blas.matmulF32(&device, &a_dev, m, k, &b_dev, n, &c_dev);
    try device.synchronize();
    try c_dev.download(&device, toBytesMut(f32, &out_gpu));

    matmulCpuRef(&a, m, k, &b, n, &out_cpu);

    for (out_cpu, out_gpu) |expected, actual| {
        try std.testing.expectApproxEqAbs(expected, actual, 0.001);
    }
}

test "cuda matmulF32 rejects undersized output buffer" {
    if (cuda.probeRuntime() != .available) return error.SkipZigTest;

    var device = try cuda.Device.init();
    defer device.deinit();

    var blas = cuda.Blas.init(&device) catch |err| {
        if (err == error.CublasUnavailable or err == error.CublasSymbolMissing) return error.SkipZigTest;
        return err;
    };
    defer blas.deinit(&device);

    const m: usize = 2;
    const k: usize = 2;
    const n: usize = 2;

    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 1, 0, 0, 1 };

    var a_dev = try device.allocBuffer(@sizeOf(f32) * a.len);
    defer a_dev.deinit(&device);
    var b_dev = try device.allocBuffer(@sizeOf(f32) * b.len);
    defer b_dev.deinit(&device);
    var c_too_small = try device.allocBuffer(@sizeOf(f32) * (m * n - 1));
    defer c_too_small.deinit(&device);

    try a_dev.upload(&device, toBytesConst(f32, &a));
    try b_dev.upload(&device, toBytesConst(f32, &b));

    try std.testing.expectError(
        error.InvalidArgument,
        blas.matmulF32(&device, &a_dev, m, k, &b_dev, n, &c_too_small),
    );
}
