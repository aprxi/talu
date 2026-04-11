//! Integration tests for CUDA RoPE wrappers.

const std = @import("std");
const main = @import("main");
const cuda = main.core.compute.cuda;
const dtype = main.core.dtype;

test "rope.run matches interleaved partial-rope CPU reference on CUDA" {
    if (cuda.probeRuntime() != .available) return error.SkipZigTest;

    var device = cuda.Device.init() catch |err| {
        if (err == error.CudaInitFailed or err == error.CudaNoDevices) return error.SkipZigTest;
        return err;
    };
    defer device.deinit();

    if (!device.supportsModuleLaunch()) return error.SkipZigTest;

    var registry = cuda.Registry.init(std.testing.allocator, &device);
    defer registry.deinit();

    const n_heads: u32 = 2;
    const head_dim: u32 = 8;
    const rope_dim: u32 = 4;
    const position: u32 = 3;
    const theta: f32 = 10000.0;
    const rope_interleaved = true;

    const input = [_]f32{
        1.0, 2.0, 3.0, 4.0, 50.0, 60.0,  70.0,  80.0,
        5.0, 6.0, 7.0, 8.0, 90.0, 100.0, 110.0, 120.0,
    };
    var expected = input;
    var actual = [_]f32{0.0} ** input.len;

    ropeReference(expected[0..], n_heads, head_dim, rope_dim, rope_interleaved, position, theta);

    var io_dev = try device.allocBuffer(input.len * @sizeOf(f32));
    defer io_dev.deinit(&device);
    try io_dev.upload(&device, std.mem.sliceAsBytes(input[0..]));

    const source = try cuda.rope.run(
        std.testing.allocator,
        &device,
        &registry,
        &io_dev,
        n_heads,
        head_dim,
        rope_dim,
        rope_interleaved,
        position,
        theta,
    );
    try std.testing.expectEqual(cuda.registry.KernelSource.embedded_module, source);

    try device.synchronize();
    try io_dev.download(&device, std.mem.sliceAsBytes(actual[0..]));

    for (expected, actual) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.001);
    }
}

test "rope.run matches non-interleaved partial-rope CPU reference on CUDA" {
    if (cuda.probeRuntime() != .available) return error.SkipZigTest;

    var device = cuda.Device.init() catch |err| {
        if (err == error.CudaInitFailed or err == error.CudaNoDevices) return error.SkipZigTest;
        return err;
    };
    defer device.deinit();

    if (!device.supportsModuleLaunch()) return error.SkipZigTest;

    var registry = cuda.Registry.init(std.testing.allocator, &device);
    defer registry.deinit();

    const n_heads: u32 = 2;
    const head_dim: u32 = 8;
    const rope_dim: u32 = 4;
    const position: u32 = 3;
    const theta: f32 = 10000.0;
    const rope_interleaved = false;

    const input = [_]f32{
        1.0, 2.0, 3.0, 4.0, 50.0, 60.0,  70.0,  80.0,
        5.0, 6.0, 7.0, 8.0, 90.0, 100.0, 110.0, 120.0,
    };
    var expected = input;
    var actual = [_]f32{0.0} ** input.len;

    ropeReference(expected[0..], n_heads, head_dim, rope_dim, rope_interleaved, position, theta);

    var io_dev = try device.allocBuffer(input.len * @sizeOf(f32));
    defer io_dev.deinit(&device);
    try io_dev.upload(&device, std.mem.sliceAsBytes(input[0..]));

    const source = try cuda.rope.run(
        std.testing.allocator,
        &device,
        &registry,
        &io_dev,
        n_heads,
        head_dim,
        rope_dim,
        rope_interleaved,
        position,
        theta,
    );
    try std.testing.expectEqual(cuda.registry.KernelSource.embedded_module, source);

    try device.synchronize();
    try io_dev.download(&device, std.mem.sliceAsBytes(actual[0..]));

    for (expected, actual) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.001);
    }
}

test "rope_store_f16 matches non-interleaved partial-rope CPU reference on CUDA" {
    if (cuda.probeRuntime() != .available) return error.SkipZigTest;

    var device = cuda.Device.init() catch |err| {
        if (err == error.CudaInitFailed or err == error.CudaNoDevices) return error.SkipZigTest;
        return err;
    };
    defer device.deinit();

    if (!device.supportsModuleLaunch()) return error.SkipZigTest;

    var registry = cuda.Registry.init(std.testing.allocator, &device);
    defer registry.deinit();
    try registry.loadEmbeddedModule(cuda.rope_store_f16.embedded_module);
    const resolved = try registry.resolveFunction(
        cuda.rope_store_f16.op_name,
        cuda.rope_store_f16.embedded_symbol,
    );

    const n_heads: u32 = 2;
    const head_dim: u32 = 8;
    const rope_dim: u32 = 4;
    const position: u32 = 3;
    const theta: f32 = 10000.0;
    const rope_interleaved = false;

    const input = [_]f32{
        1.0, 2.0, 3.0, 4.0, 50.0, 60.0,  70.0,  80.0,
        5.0, 6.0, 7.0, 8.0, 90.0, 100.0, 110.0, 120.0,
    };
    var expected = input;
    ropeReference(expected[0..], n_heads, head_dim, rope_dim, rope_interleaved, position, theta);

    var output_bits = [_]u16{0} ** input.len;
    var actual = [_]f32{0.0} ** input.len;

    var input_dev = try device.allocBuffer(input.len * @sizeOf(f32));
    defer input_dev.deinit(&device);
    var output_dev = try device.allocBuffer(output_bits.len * @sizeOf(u16));
    defer output_dev.deinit(&device);

    try input_dev.upload(&device, std.mem.sliceAsBytes(input[0..]));
    var arg_pack = cuda.ArgPack.init(std.testing.allocator);
    defer arg_pack.deinit();

    try cuda.rope_store_f16.runWithFunction(
        &arg_pack,
        &device,
        resolved.function,
        &input_dev,
        &output_dev,
        n_heads,
        head_dim,
        rope_dim,
        position,
        theta,
    );
    try device.synchronize();
    try output_dev.download(&device, std.mem.sliceAsBytes(output_bits[0..]));

    for (output_bits, 0..) |bits, i| actual[i] = dtype.fp16ToF32(bits);
    for (expected, actual) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.01);
    }
}

test "kv_write_f16 matches non-interleaved partial-rope CPU reference on CUDA" {
    if (cuda.probeRuntime() != .available) return error.SkipZigTest;

    var device = cuda.Device.init() catch |err| {
        if (err == error.CudaInitFailed or err == error.CudaNoDevices) return error.SkipZigTest;
        return err;
    };
    defer device.deinit();

    if (!device.supportsModuleLaunch()) return error.SkipZigTest;

    var registry = cuda.Registry.init(std.testing.allocator, &device);
    defer registry.deinit();
    try registry.loadEmbeddedModule(cuda.kv_write_f16.embedded_module);
    const resolved = try registry.resolveFunction(
        cuda.kv_write_f16.op_name,
        cuda.kv_write_f16.embedded_symbol,
    );

    const n_heads: u32 = 2;
    const head_dim: u32 = 8;
    const rope_dim: u32 = 4;
    const position: u32 = 3;
    const theta: f32 = 10000.0;
    const rope_interleaved = false;

    const k_input = [_]f32{
        1.0, 2.0, 3.0, 4.0, 50.0, 60.0,  70.0,  80.0,
        5.0, 6.0, 7.0, 8.0, 90.0, 100.0, 110.0, 120.0,
    };
    const v_input = [_]f32{
        -1.0, -2.0, -3.0, -4.0, 10.0, 20.0, 30.0, 40.0,
        0.5,  1.5,  2.5,  3.5,  11.0, 21.0, 31.0, 41.0,
    };
    var expected_k = k_input;
    ropeReference(expected_k[0..], n_heads, head_dim, rope_dim, rope_interleaved, position, theta);

    var out_k_bits = [_]u16{0} ** k_input.len;
    var out_v_bits = [_]u16{0} ** v_input.len;
    var actual_k = [_]f32{0.0} ** k_input.len;
    var actual_v = [_]f32{0.0} ** v_input.len;

    var input_k_dev = try device.allocBuffer(k_input.len * @sizeOf(f32));
    defer input_k_dev.deinit(&device);
    var input_v_dev = try device.allocBuffer(v_input.len * @sizeOf(f32));
    defer input_v_dev.deinit(&device);
    var out_k_dev = try device.allocBuffer(out_k_bits.len * @sizeOf(u16));
    defer out_k_dev.deinit(&device);
    var out_v_dev = try device.allocBuffer(out_v_bits.len * @sizeOf(u16));
    defer out_v_dev.deinit(&device);

    try input_k_dev.upload(&device, std.mem.sliceAsBytes(k_input[0..]));
    try input_v_dev.upload(&device, std.mem.sliceAsBytes(v_input[0..]));
    var arg_pack = cuda.ArgPack.init(std.testing.allocator);
    defer arg_pack.deinit();

    try cuda.kv_write_f16.runWithFunction(
        &arg_pack,
        &device,
        resolved.function,
        &input_k_dev,
        &input_v_dev,
        &out_k_dev,
        &out_v_dev,
        n_heads,
        head_dim,
        rope_dim,
        position,
        theta,
    );
    try device.synchronize();
    try out_k_dev.download(&device, std.mem.sliceAsBytes(out_k_bits[0..]));
    try out_v_dev.download(&device, std.mem.sliceAsBytes(out_v_bits[0..]));

    for (out_k_bits, 0..) |bits, i| actual_k[i] = dtype.fp16ToF32(bits);
    for (out_v_bits, 0..) |bits, i| actual_v[i] = dtype.fp16ToF32(bits);

    for (expected_k, actual_k) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.01);
    }
    for (v_input, actual_v) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.01);
    }
}

fn ropeReference(
    io: []f32,
    n_heads: u32,
    head_dim: u32,
    rope_dim: u32,
    rope_interleaved: bool,
    position: u32,
    theta: f32,
) void {
    const half = rope_dim / 2;
    const n_heads_usize: usize = @intCast(n_heads);
    const head_dim_usize: usize = @intCast(head_dim);
    const half_usize: usize = @intCast(half);
    const rope_dim_f32: f32 = @floatFromInt(rope_dim);
    const position_f32: f32 = @floatFromInt(position);

    for (0..n_heads_usize) |head| {
        const base = head * head_dim_usize;
        for (0..half_usize) |pair| {
            const lo_dim = if (rope_interleaved) pair * 2 else pair;
            const hi_dim = if (rope_interleaved) (pair * 2) + 1 else half_usize + pair;
            const lo_idx = base + lo_dim;
            const hi_idx = base + hi_dim;
            const x0 = io[lo_idx];
            const x1 = io[hi_idx];
            const pair_f32: f32 = @floatFromInt(pair);
            const inv_freq = std.math.pow(f32, theta, (-2.0 * pair_f32) / rope_dim_f32);
            const angle = position_f32 * inv_freq;
            io[lo_idx] = x0 * @cos(angle) - x1 * @sin(angle);
            io[hi_idx] = x0 * @sin(angle) + x1 * @cos(angle);
        }
    }
}
