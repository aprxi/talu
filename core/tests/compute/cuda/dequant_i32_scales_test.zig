//! Integration tests for CUDA I32 dequantization kernels.

const std = @import("std");
const main = @import("main");
const cuda = main.core.compute.cuda;

fn approxEqSlices(expected: []const f32, actual: []const f32, tolerance: f32) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, tolerance);
    }
}

test "dequant_i32_scales kernel matches CPU reference with 2D launch mapping" {
    if (cuda.probeRuntime() != .available) return error.SkipZigTest;

    var device = cuda.Device.init() catch |err| {
        if (err == error.CudaInitFailed or err == error.CudaNoDevices) return error.SkipZigTest;
        return err;
    };
    defer device.deinit();
    if (!device.supportsModuleLaunch()) return error.SkipZigTest;

    var registry = cuda.Registry.init(std.testing.allocator, &device);
    defer registry.deinit();
    try registry.loadEmbeddedModule(cuda.topk_rows_f32.embedded_module);

    const resolved = try registry.resolveFunction("dequant_i32_scales", "talu_dequant_i32_scales");
    try std.testing.expectEqual(cuda.registry.KernelSource.embedded_module, resolved.source);

    const rows: usize = 4;
    const out_dim: usize = 513;
    const count = rows * out_dim;

    var gemm_host: [count]i32 = undefined;
    var input_scales_host: [rows]f32 = undefined;
    var weight_scales_host: [out_dim]f32 = undefined;
    var expected_host: [count]f32 = undefined;
    var output_host: [count]f32 = [_]f32{0.0} ** count;

    for (0..rows) |row| {
        input_scales_host[row] = 0.125 + @as(f32, @floatFromInt(row)) * 0.25;
    }
    for (0..out_dim) |col| {
        weight_scales_host[col] = 0.01 + @as(f32, @floatFromInt(col % 17)) * 0.001;
    }
    for (0..rows) |row| {
        for (0..out_dim) |col| {
            const idx = row * out_dim + col;
            const raw_u = (row * 37 + col * 13 + 17) % 2048;
            const raw: i32 = @as(i32, @intCast(raw_u)) - 1024;
            gemm_host[idx] = raw;
            expected_host[idx] = @as(f32, @floatFromInt(raw)) * input_scales_host[row] * weight_scales_host[col];
        }
    }

    var gemm_dev = try device.allocBuffer(count * @sizeOf(i32));
    defer gemm_dev.deinit(&device);
    var input_scales_dev = try device.allocBuffer(rows * @sizeOf(f32));
    defer input_scales_dev.deinit(&device);
    var weight_scales_dev = try device.allocBuffer(out_dim * @sizeOf(f32));
    defer weight_scales_dev.deinit(&device);
    var output_dev = try device.allocBuffer(count * @sizeOf(f32));
    defer output_dev.deinit(&device);

    try gemm_dev.upload(&device, std.mem.sliceAsBytes(gemm_host[0..]));
    try input_scales_dev.upload(&device, std.mem.sliceAsBytes(input_scales_host[0..]));
    try weight_scales_dev.upload(&device, std.mem.sliceAsBytes(weight_scales_host[0..]));

    var arg_pack = cuda.ArgPack.init(std.testing.allocator);
    defer arg_pack.deinit();
    arg_pack.reset();
    try arg_pack.appendBufferPtr(&gemm_dev);
    try arg_pack.appendBufferPtr(&input_scales_dev);
    try arg_pack.appendBufferPtr(&weight_scales_dev);
    try arg_pack.appendBufferPtr(&output_dev);
    try arg_pack.appendScalar(u32, @intCast(rows));
    try arg_pack.appendScalar(u32, @intCast(out_dim));

    const blocks_x: u32 = @intCast((out_dim + 255) / 256);
    try cuda.launch.launchWithFamily(&device, resolved.function, .{
        .grid_x = blocks_x,
        .grid_y = @intCast(rows),
        .block_x = 256,
    }, &arg_pack, .other);
    try device.synchronize();

    try output_dev.download(&device, std.mem.sliceAsBytes(output_host[0..]));
    try approxEqSlices(expected_host[0..], output_host[0..], 0.0001);
}

test "dequant_i32_scales_split3 kernel matches CPU reference and target buffers" {
    if (cuda.probeRuntime() != .available) return error.SkipZigTest;

    var device = cuda.Device.init() catch |err| {
        if (err == error.CudaInitFailed or err == error.CudaNoDevices) return error.SkipZigTest;
        return err;
    };
    defer device.deinit();
    if (!device.supportsModuleLaunch()) return error.SkipZigTest;

    var registry = cuda.Registry.init(std.testing.allocator, &device);
    defer registry.deinit();
    try registry.loadEmbeddedModule(cuda.topk_rows_f32.embedded_module);

    const resolved = try registry.resolveFunction("dequant_i32_scales_split3", "talu_dequant_i32_scales_split3");
    try std.testing.expectEqual(cuda.registry.KernelSource.embedded_module, resolved.source);

    const rows: usize = 3;
    const dim_a: usize = 129;
    const dim_b: usize = 257;
    const dim_c: usize = 33;
    const total_dim = dim_a + dim_b + dim_c;
    const total_count = rows * total_dim;

    var gemm_host: [total_count]i32 = undefined;
    var input_scales_host: [rows]f32 = undefined;
    var weight_scales_host: [total_dim]f32 = undefined;
    var out_a_host: [rows * dim_a]f32 = [_]f32{0.0} ** (rows * dim_a);
    var out_b_host: [rows * dim_b]f32 = [_]f32{0.0} ** (rows * dim_b);
    var out_c_host: [rows * dim_c]f32 = [_]f32{0.0} ** (rows * dim_c);
    var exp_a_host: [rows * dim_a]f32 = undefined;
    var exp_b_host: [rows * dim_b]f32 = undefined;
    var exp_c_host: [rows * dim_c]f32 = undefined;

    for (0..rows) |row| {
        input_scales_host[row] = 0.2 + @as(f32, @floatFromInt(row)) * 0.15;
    }
    for (0..total_dim) |col| {
        weight_scales_host[col] = 0.015 + @as(f32, @floatFromInt(col % 23)) * 0.0005;
    }
    for (0..rows) |row| {
        for (0..total_dim) |col| {
            const idx = row * total_dim + col;
            const raw_u = (row * 97 + col * 11 + 9) % 4096;
            const raw: i32 = @as(i32, @intCast(raw_u)) - 2048;
            const value = @as(f32, @floatFromInt(raw)) * input_scales_host[row] * weight_scales_host[col];
            gemm_host[idx] = raw;
            if (col < dim_a) {
                exp_a_host[row * dim_a + col] = value;
            } else if (col < dim_a + dim_b) {
                exp_b_host[row * dim_b + (col - dim_a)] = value;
            } else {
                exp_c_host[row * dim_c + (col - dim_a - dim_b)] = value;
            }
        }
    }

    var gemm_dev = try device.allocBuffer(total_count * @sizeOf(i32));
    defer gemm_dev.deinit(&device);
    var input_scales_dev = try device.allocBuffer(rows * @sizeOf(f32));
    defer input_scales_dev.deinit(&device);
    var weight_scales_dev = try device.allocBuffer(total_dim * @sizeOf(f32));
    defer weight_scales_dev.deinit(&device);
    var out_a_dev = try device.allocBuffer(rows * dim_a * @sizeOf(f32));
    defer out_a_dev.deinit(&device);
    var out_b_dev = try device.allocBuffer(rows * dim_b * @sizeOf(f32));
    defer out_b_dev.deinit(&device);
    var out_c_dev = try device.allocBuffer(rows * dim_c * @sizeOf(f32));
    defer out_c_dev.deinit(&device);

    try gemm_dev.upload(&device, std.mem.sliceAsBytes(gemm_host[0..]));
    try input_scales_dev.upload(&device, std.mem.sliceAsBytes(input_scales_host[0..]));
    try weight_scales_dev.upload(&device, std.mem.sliceAsBytes(weight_scales_host[0..]));

    var arg_pack = cuda.ArgPack.init(std.testing.allocator);
    defer arg_pack.deinit();
    arg_pack.reset();
    try arg_pack.appendBufferPtr(&gemm_dev);
    try arg_pack.appendBufferPtr(&input_scales_dev);
    try arg_pack.appendBufferPtr(&weight_scales_dev);
    try arg_pack.appendBufferPtr(&out_a_dev);
    try arg_pack.appendBufferPtr(&out_b_dev);
    try arg_pack.appendBufferPtr(&out_c_dev);
    try arg_pack.appendScalar(u32, @intCast(rows));
    try arg_pack.appendScalar(u32, @intCast(dim_a));
    try arg_pack.appendScalar(u32, @intCast(dim_b));
    try arg_pack.appendScalar(u32, @intCast(dim_c));

    const blocks_x: u32 = @intCast((total_dim + 255) / 256);
    try cuda.launch.launchWithFamily(&device, resolved.function, .{
        .grid_x = blocks_x,
        .grid_y = @intCast(rows),
        .block_x = 256,
    }, &arg_pack, .other);
    try device.synchronize();

    try out_a_dev.download(&device, std.mem.sliceAsBytes(out_a_host[0..]));
    try out_b_dev.download(&device, std.mem.sliceAsBytes(out_b_host[0..]));
    try out_c_dev.download(&device, std.mem.sliceAsBytes(out_c_host[0..]));

    try approxEqSlices(exp_a_host[0..], out_a_host[0..], 0.0001);
    try approxEqSlices(exp_b_host[0..], out_b_host[0..], 0.0001);
    try approxEqSlices(exp_c_host[0..], out_c_host[0..], 0.0001);
}
