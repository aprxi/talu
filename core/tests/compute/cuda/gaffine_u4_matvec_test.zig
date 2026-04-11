//! Integration tests for CUDA grouped-affine U4 matvec wrappers.

const std = @import("std");
const main = @import("main");
const cuda = main.core.compute.cuda;

const TestShape = struct {
    in_dim: usize,
    out_dim: usize,
    group_size: usize,
    batch_rows: usize,
};

test "gaffine_u4_matvec batched kernel matches row-by-row reference" {
    try expectBatchedMatchesRowByRow(.{
        .in_dim = 5120,
        .out_dim = 192,
        .group_size = 128,
        .batch_rows = 4,
    }, false);
}

test "gaffine_u4_matvec batched kernel matches qwen3.5 27b q projection shape" {
    try expectBatchedMatchesRowByRow(.{
        .in_dim = 5120,
        .out_dim = 6144,
        .group_size = 32,
        .batch_rows = 8,
    }, false);
}

test "gaffine_u4_matvec batched kernel matches qwen3.5 27b kv projection shape" {
    try expectBatchedMatchesRowByRow(.{
        .in_dim = 5120,
        .out_dim = 1024,
        .group_size = 32,
        .batch_rows = 8,
    }, false);
}

test "gaffine_u4_matvec batched kernel matches qwen3.5 27b gated-delta in projection shape" {
    try expectBatchedMatchesRowByRow(.{
        .in_dim = 5120,
        .out_dim = 16480,
        .group_size = 32,
        .batch_rows = 8,
    }, false);
}

test "gaffine_u4_matvec batched kernel matches qwen3.5 27b gated-delta out projection shape" {
    try expectBatchedMatchesRowByRow(.{
        .in_dim = 6144,
        .out_dim = 5120,
        .group_size = 32,
        .batch_rows = 8,
    }, false);
}

test "gaffine_u4_matvec batched kernel matches large-vocab output geometry" {
    try expectBatchedMatchesRowByRow(.{
        .in_dim = 5120,
        .out_dim = 65536,
        .group_size = 32,
        .batch_rows = 8,
    }, false);
}

test "gaffine_u4_matvec tile8 batched kernel matches row-by-row reference" {
    try expectBatchedMatchesRowByRow(.{
        .in_dim = 5120,
        .out_dim = 192,
        .group_size = 128,
        .batch_rows = 8,
    }, true);
}

fn expectBatchedMatchesRowByRow(shape: TestShape, use_tile8: bool) !void {
    if (cuda.probeRuntime() != .available) return error.SkipZigTest;

    var device = cuda.Device.init() catch |err| {
        if (err == error.CudaInitFailed or err == error.CudaNoDevices) return error.SkipZigTest;
        return err;
    };
    defer device.deinit();
    if (!device.supportsModuleLaunch()) return error.SkipZigTest;

    var registry = cuda.Registry.init(std.testing.allocator, &device);
    defer registry.deinit();
    try registry.loadEmbeddedModule(cuda.gaffine_u4_matvec.embedded_module);

    const base_resolved = try registry.resolveFunction(
        cuda.gaffine_u4_matvec.op_name,
        cuda.gaffine_u4_matvec.embedded_symbol,
    );
    try std.testing.expectEqual(cuda.registry.KernelSource.embedded_module, base_resolved.source);

    const tile8_resolved = try registry.resolveFunction(
        cuda.gaffine_u4_matvec.op_name_tile8,
        cuda.gaffine_u4_matvec.embedded_symbol_tile8,
    );
    try std.testing.expectEqual(cuda.registry.KernelSource.embedded_module, tile8_resolved.source);

    const groups_per_row = shape.in_dim / shape.group_size;
    const input_count = shape.batch_rows * shape.in_dim;
    const output_count = shape.batch_rows * shape.out_dim;
    const packed_word_count = shape.out_dim * (shape.in_dim / 8);
    const scale_bias_count = shape.out_dim * groups_per_row;

    var prng = std.Random.DefaultPrng.init(0x5A17_E9C2_731D_0042);
    const random = prng.random();

    const input_host = try std.testing.allocator.alloc(f32, input_count);
    defer std.testing.allocator.free(input_host);
    const packed_weight_host = try std.testing.allocator.alloc(u32, packed_word_count);
    defer std.testing.allocator.free(packed_weight_host);
    const scales_host = try std.testing.allocator.alloc(u16, scale_bias_count);
    defer std.testing.allocator.free(scales_host);
    const biases_host = try std.testing.allocator.alloc(u16, scale_bias_count);
    defer std.testing.allocator.free(biases_host);
    const batched_host = try std.testing.allocator.alloc(f32, output_count);
    defer std.testing.allocator.free(batched_host);
    const row_host = try std.testing.allocator.alloc(f32, output_count);
    defer std.testing.allocator.free(row_host);

    fillInput(input_host, random, shape);
    fillQuantizedWeights(packed_weight_host, scales_host, biases_host, random, shape);
    @memset(batched_host, 0.0);
    @memset(row_host, 0.0);

    var input_dev = try device.allocBuffer(input_count * @sizeOf(f32));
    defer input_dev.deinit(&device);
    var packed_weight_dev = try device.allocBuffer(packed_word_count * @sizeOf(u32));
    defer packed_weight_dev.deinit(&device);
    var scales_dev = try device.allocBuffer(scale_bias_count * @sizeOf(u16));
    defer scales_dev.deinit(&device);
    var biases_dev = try device.allocBuffer(scale_bias_count * @sizeOf(u16));
    defer biases_dev.deinit(&device);
    var batched_out_dev = try device.allocBuffer(output_count * @sizeOf(f32));
    defer batched_out_dev.deinit(&device);
    var row_out_dev = try device.allocBuffer(output_count * @sizeOf(f32));
    defer row_out_dev.deinit(&device);

    try input_dev.upload(&device, std.mem.sliceAsBytes(input_host));
    try packed_weight_dev.upload(&device, std.mem.sliceAsBytes(packed_weight_host));
    try scales_dev.upload(&device, std.mem.sliceAsBytes(scales_host));
    try biases_dev.upload(&device, std.mem.sliceAsBytes(biases_host));
    try batched_out_dev.upload(&device, std.mem.sliceAsBytes(batched_host));
    try row_out_dev.upload(&device, std.mem.sliceAsBytes(row_host));

    {
        var arg_pack = cuda.ArgPack.init(std.testing.allocator);
        defer arg_pack.deinit();
        if (use_tile8) {
            try cuda.gaffine_u4_matvec.runWithFunctionTile8(
                &arg_pack,
                &device,
                tile8_resolved.function,
                &input_dev,
                &packed_weight_dev,
                &scales_dev,
                &biases_dev,
                &batched_out_dev,
                @intCast(shape.in_dim),
                @intCast(shape.out_dim),
                @intCast(shape.group_size),
                cuda.gaffine_u4_matvec.scales_dtype_f16,
                @intCast(shape.batch_rows),
                0,
            );
        } else {
            try cuda.gaffine_u4_matvec.runWithFunction(
                &arg_pack,
                &device,
                base_resolved.function,
                &input_dev,
                &packed_weight_dev,
                &scales_dev,
                &biases_dev,
                &batched_out_dev,
                @intCast(shape.in_dim),
                @intCast(shape.out_dim),
                @intCast(shape.group_size),
                cuda.gaffine_u4_matvec.scales_dtype_f16,
                @intCast(shape.batch_rows),
                0,
            );
        }
    }
    try device.synchronize();

    {
        var arg_pack = cuda.ArgPack.init(std.testing.allocator);
        defer arg_pack.deinit();

        const input_row_bytes = shape.in_dim * @sizeOf(f32);
        const output_row_bytes = shape.out_dim * @sizeOf(f32);
        for (0..shape.batch_rows) |row_index| {
            var input_row = sliceBuffer(&input_dev, row_index * input_row_bytes, input_row_bytes) orelse return error.InvalidArgument;
            var output_row = sliceBuffer(&row_out_dev, row_index * output_row_bytes, output_row_bytes) orelse return error.InvalidArgument;
            try cuda.gaffine_u4_matvec.runWithFunction(
                &arg_pack,
                &device,
                base_resolved.function,
                &input_row,
                &packed_weight_dev,
                &scales_dev,
                &biases_dev,
                &output_row,
                @intCast(shape.in_dim),
                @intCast(shape.out_dim),
                @intCast(shape.group_size),
                cuda.gaffine_u4_matvec.scales_dtype_f16,
                1,
                0,
            );
        }
    }
    try device.synchronize();

    try batched_out_dev.download(&device, std.mem.sliceAsBytes(batched_host));
    try row_out_dev.download(&device, std.mem.sliceAsBytes(row_host));

    try expectApproxEqSlices(row_host, batched_host, 0.001);
}

fn fillInput(input: []f32, random: std.Random, shape: TestShape) void {
    for (0..shape.batch_rows) |row| {
        for (0..shape.in_dim) |col| {
            const idx = row * shape.in_dim + col;
            const centered = random.float(f32) - 0.5;
            input[idx] = centered * 0.5 + @as(f32, @floatFromInt((row + col) % 5)) * 0.01;
        }
    }
}

fn fillQuantizedWeights(
    packed_weight: []u32,
    scales: []u16,
    biases: []u16,
    random: std.Random,
    shape: TestShape,
) void {
    const words_per_row = shape.in_dim / 8;
    const groups_per_row = shape.in_dim / shape.group_size;
    const words_per_group = shape.group_size / 8;

    @memset(packed_weight, 0);

    for (0..shape.out_dim) |out_idx| {
        for (0..groups_per_row) |group_idx| {
            const sb_index = out_idx * groups_per_row + group_idx;
            const scale = 0.015 + random.float(f32) * 0.08;
            const bias = (random.float(f32) - 0.5) * 0.06;
            scales[sb_index] = encodeF16(scale);
            biases[sb_index] = encodeF16(bias);
        }

        for (0..words_per_row) |word_idx| {
            var packed_word: u32 = 0;
            for (0..8) |nibble| {
                const quant = random.intRangeAtMost(u32, 0, 15);
                packed_word |= quant << @intCast(nibble * 4);
            }
            packed_weight[out_idx * words_per_row + word_idx] = packed_word;
        }

        // Keep group transitions hot by perturbing the final word in each group.
        for (1..groups_per_row) |group_idx| {
            const word_index = out_idx * words_per_row + group_idx * words_per_group - 1;
            packed_weight[word_index] ^= 0x0102_0408;
        }
    }
}

fn sliceBuffer(buffer: *const cuda.Buffer, byte_offset: usize, byte_count: usize) ?cuda.Buffer {
    if (byte_offset > buffer.size or byte_count > buffer.size - byte_offset) return null;
    return .{
        .pointer = buffer.pointer + byte_offset,
        .size = byte_count,
    };
}

fn encodeF16(value: f32) u16 {
    const narrowed: f16 = @floatCast(value);
    return @bitCast(narrowed);
}

fn expectApproxEqSlices(expected: []const f32, actual: []const f32, tolerance: f32) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual, 0..) |want, got, idx| {
        std.testing.expectApproxEqAbs(want, got, tolerance) catch |err| {
            std.debug.print(
                "gaffine_u4 mismatch idx={} want={d:.6} got={d:.6} abs_diff={d:.6}\n",
                .{ idx, want, got, @abs(want - got) },
            );
            return err;
        };
    }
}

// --- CPU reference tests: verify CUDA output against ground-truth CPU computation ---

const ScalesDtype = enum { f16, bf16 };

fn encodeBF16(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    // Round-to-nearest-even: add rounding bias based on LSB of result.
    const lsb: u32 = (bits >> 16) & 1;
    const rounded = bits +% (0x7FFF +% lsb);
    return @intCast(rounded >> 16);
}

fn decodeScaleBias(raw: u16, dtype: ScalesDtype) f32 {
    return switch (dtype) {
        .f16 => blk: {
            const narrowed: f16 = @bitCast(raw);
            break :blk @as(f32, @floatCast(narrowed));
        },
        .bf16 => blk: {
            const bits: u32 = @as(u32, raw) << 16;
            break :blk @bitCast(bits);
        },
    };
}

/// CPU reference GEMV for grouped-affine U4.
/// Computes output[out_idx] = sum_k(input[k] * dequant(weight[out_idx][k]))
/// where dequant(nibble) = nibble * scale + bias (per-group).
fn cpuReferenceGemv(
    input: []const f32,
    packed_weight: []const u32,
    scales: []const u16,
    biases: []const u16,
    output: []f32,
    in_dim: usize,
    out_dim: usize,
    group_size: usize,
    dtype: ScalesDtype,
) void {
    const words_per_row = in_dim / 8;
    const groups_per_row = in_dim / group_size;

    for (0..out_dim) |out_idx| {
        var acc: f64 = 0.0; // F64 for reference accuracy
        const weight_row = packed_weight[out_idx * words_per_row ..][0..words_per_row];
        const scale_row = scales[out_idx * groups_per_row ..][0..groups_per_row];
        const bias_row = biases[out_idx * groups_per_row ..][0..groups_per_row];

        for (0..words_per_row) |word_idx| {
            const word = weight_row[word_idx];
            const elem_base = word_idx * 8;
            const group_idx = elem_base / group_size;
            const scale = decodeScaleBias(scale_row[group_idx], dtype);
            const bias = decodeScaleBias(bias_row[group_idx], dtype);

            for (0..8) |nibble_idx| {
                const nibble: u32 = (word >> @intCast(nibble_idx * 4)) & 0xF;
                const dequant: f64 = @as(f64, @floatFromInt(nibble)) * @as(f64, scale) + @as(f64, bias);
                acc += @as(f64, input[elem_base + nibble_idx]) * dequant;
            }
        }
        output[out_idx] = @floatCast(acc);
    }
}

fn fillQuantizedWeightsWithDtype(
    packed_weight: []u32,
    scales: []u16,
    biases: []u16,
    random: std.Random,
    shape: TestShape,
    dtype: ScalesDtype,
) void {
    const words_per_row = shape.in_dim / 8;
    const groups_per_row = shape.in_dim / shape.group_size;
    const words_per_group = shape.group_size / 8;

    @memset(packed_weight, 0);

    for (0..shape.out_dim) |out_idx| {
        for (0..groups_per_row) |group_idx| {
            const sb_index = out_idx * groups_per_row + group_idx;
            const scale = 0.015 + random.float(f32) * 0.08;
            const bias = (random.float(f32) - 0.5) * 0.06;
            scales[sb_index] = switch (dtype) {
                .f16 => encodeF16(scale),
                .bf16 => encodeBF16(scale),
            };
            biases[sb_index] = switch (dtype) {
                .f16 => encodeF16(bias),
                .bf16 => encodeBF16(bias),
            };
        }

        for (0..words_per_row) |word_idx| {
            var packed_word: u32 = 0;
            for (0..8) |nibble| {
                const quant = random.intRangeAtMost(u32, 0, 15);
                packed_word |= quant << @intCast(nibble * 4);
            }
            packed_weight[out_idx * words_per_row + word_idx] = packed_word;
        }

        for (1..groups_per_row) |group_idx| {
            const word_index = out_idx * words_per_row + group_idx * words_per_group - 1;
            packed_weight[word_index] ^= 0x0102_0408;
        }
    }
}

fn expectCudaMatchesCpuReference(shape: TestShape, dtype: ScalesDtype) !void {
    if (cuda.probeRuntime() != .available) return error.SkipZigTest;

    var device = cuda.Device.init() catch |err| {
        if (err == error.CudaInitFailed or err == error.CudaNoDevices) return error.SkipZigTest;
        return err;
    };
    defer device.deinit();
    if (!device.supportsModuleLaunch()) return error.SkipZigTest;

    var registry = cuda.Registry.init(std.testing.allocator, &device);
    defer registry.deinit();
    try registry.loadEmbeddedModule(cuda.gaffine_u4_matvec.embedded_module);
    const resolved = try registry.resolveFunction(
        cuda.gaffine_u4_matvec.op_name,
        cuda.gaffine_u4_matvec.embedded_symbol,
    );

    const groups_per_row = shape.in_dim / shape.group_size;
    const input_count = shape.batch_rows * shape.in_dim;
    const output_count = shape.batch_rows * shape.out_dim;
    const packed_word_count = shape.out_dim * (shape.in_dim / 8);
    const scale_bias_count = shape.out_dim * groups_per_row;

    var prng = std.Random.DefaultPrng.init(0xDEAD_BEEF_1234_5678);
    const random = prng.random();

    const input_host = try std.testing.allocator.alloc(f32, input_count);
    defer std.testing.allocator.free(input_host);
    const packed_weight_host = try std.testing.allocator.alloc(u32, packed_word_count);
    defer std.testing.allocator.free(packed_weight_host);
    const scales_host = try std.testing.allocator.alloc(u16, scale_bias_count);
    defer std.testing.allocator.free(scales_host);
    const biases_host = try std.testing.allocator.alloc(u16, scale_bias_count);
    defer std.testing.allocator.free(biases_host);
    const cuda_output = try std.testing.allocator.alloc(f32, output_count);
    defer std.testing.allocator.free(cuda_output);
    const cpu_output = try std.testing.allocator.alloc(f32, output_count);
    defer std.testing.allocator.free(cpu_output);

    fillInput(input_host, random, shape);
    fillQuantizedWeightsWithDtype(packed_weight_host, scales_host, biases_host, random, shape, dtype);
    @memset(cuda_output, 0.0);

    // CPU reference
    for (0..shape.batch_rows) |row| {
        const in_row = input_host[row * shape.in_dim ..][0..shape.in_dim];
        const out_row = cpu_output[row * shape.out_dim ..][0..shape.out_dim];
        cpuReferenceGemv(in_row, packed_weight_host, scales_host, biases_host, out_row, shape.in_dim, shape.out_dim, shape.group_size, dtype);
    }

    // CUDA
    var input_dev = try device.allocBuffer(input_count * @sizeOf(f32));
    defer input_dev.deinit(&device);
    var packed_weight_dev = try device.allocBuffer(packed_word_count * @sizeOf(u32));
    defer packed_weight_dev.deinit(&device);
    var scales_dev = try device.allocBuffer(scale_bias_count * @sizeOf(u16));
    defer scales_dev.deinit(&device);
    var biases_dev = try device.allocBuffer(scale_bias_count * @sizeOf(u16));
    defer biases_dev.deinit(&device);
    var output_dev = try device.allocBuffer(output_count * @sizeOf(f32));
    defer output_dev.deinit(&device);

    try input_dev.upload(&device, std.mem.sliceAsBytes(input_host));
    try packed_weight_dev.upload(&device, std.mem.sliceAsBytes(packed_weight_host));
    try scales_dev.upload(&device, std.mem.sliceAsBytes(scales_host));
    try biases_dev.upload(&device, std.mem.sliceAsBytes(biases_host));

    const scales_tag: u32 = switch (dtype) {
        .f16 => cuda.gaffine_u4_matvec.scales_dtype_f16,
        .bf16 => cuda.gaffine_u4_matvec.scales_dtype_bf16,
    };

    {
        var arg_pack = cuda.ArgPack.init(std.testing.allocator);
        defer arg_pack.deinit();
        try cuda.gaffine_u4_matvec.runWithFunction(
            &arg_pack,
            &device,
            resolved.function,
            &input_dev,
            &packed_weight_dev,
            &scales_dev,
            &biases_dev,
            &output_dev,
            @intCast(shape.in_dim),
            @intCast(shape.out_dim),
            @intCast(shape.group_size),
            scales_tag,
            @intCast(shape.batch_rows),
            0,
        );
    }
    try device.synchronize();
    try output_dev.download(&device, std.mem.sliceAsBytes(cuda_output));

    // Compare with tolerance. Use relative tolerance for larger values.
    var max_abs_err: f32 = 0.0;
    var max_rel_err: f32 = 0.0;
    var worst_idx: usize = 0;
    for (cpu_output, cuda_output, 0..) |want, got, idx| {
        const abs_err = @abs(want - got);
        const rel_err = if (@abs(want) > 1e-6) abs_err / @abs(want) else abs_err;
        if (abs_err > max_abs_err) {
            max_abs_err = abs_err;
            worst_idx = idx;
        }
        if (rel_err > max_rel_err) max_rel_err = rel_err;
    }

    // Fail if max absolute error exceeds tolerance.
    // For dot products of ~5120 terms with U4 weights, expect < 0.1 absolute error.
    const tolerance: f32 = 0.1;
    if (max_abs_err > tolerance) {
        std.debug.print(
            "CPU vs CUDA mismatch: max_abs_err={d:.6} max_rel_err={d:.6} worst_idx={} cpu={d:.6} cuda={d:.6} dtype={s}\n",
            .{
                max_abs_err,
                max_rel_err,
                worst_idx,
                cpu_output[worst_idx],
                cuda_output[worst_idx],
                @tagName(dtype),
            },
        );
        return error.TestExpectedApproxEqAbs;
    }
}

test "gaffine_u4_matvec matches CPU reference with bf16 scales at 27b gate_proj shape" {
    try expectCudaMatchesCpuReference(.{
        .in_dim = 5120,
        .out_dim = 17408,
        .group_size = 32,
        .batch_rows = 1,
    }, .bf16);
}

test "gaffine_u4_matvec matches CPU reference with bf16 scales at 27b down_proj shape" {
    try expectCudaMatchesCpuReference(.{
        .in_dim = 17408,
        .out_dim = 5120,
        .group_size = 32,
        .batch_rows = 1,
    }, .bf16);
}

test "gaffine_u4_matvec matches CPU reference with f16 scales at 27b gate_proj shape" {
    try expectCudaMatchesCpuReference(.{
        .in_dim = 5120,
        .out_dim = 17408,
        .group_size = 32,
        .batch_rows = 1,
    }, .f16);
}

test "gaffine_u4_matvec matches CPU reference with actual 27b model weights" {
    // This test loads actual model weight data (gate_proj layer 0) and compares
    // the CUDA kernel output against a Python-computed CPU reference.
    // If this fails, the kernel has a data-dependent bug with real model weights.
    if (cuda.probeRuntime() != .available) return error.SkipZigTest;

    var device = cuda.Device.init() catch |err| {
        if (err == error.CudaInitFailed or err == error.CudaNoDevices) return error.SkipZigTest;
        return err;
    };
    defer device.deinit();
    if (!device.supportsModuleLaunch()) return error.SkipZigTest;

    var registry = cuda.Registry.init(std.testing.allocator, &device);
    defer registry.deinit();
    try registry.loadEmbeddedModule(cuda.gaffine_u4_matvec.embedded_module);
    const resolved = try registry.resolveFunction(
        cuda.gaffine_u4_matvec.op_name,
        cuda.gaffine_u4_matvec.embedded_symbol,
    );

    const in_dim: u32 = 5120;
    const out_dim: u32 = 16;
    const group_size: u32 = 32;

    // Load test data from disk (generated by Python script with actual model weights)
    const input_host = loadBinaryFile(f32, "/tmp/test_27b_input.bin") orelse {
        std.debug.print("Skipping: /tmp/test_27b_input.bin not found\n", .{});
        return error.SkipZigTest;
    };
    defer std.testing.allocator.free(input_host);
    if (input_host.len != in_dim) return error.SkipZigTest;

    const packed_weight_host = loadBinaryFile(u32, "/tmp/test_27b_weight.bin") orelse return error.SkipZigTest;
    defer std.testing.allocator.free(packed_weight_host);

    const scales_host = loadBinaryFile(u16, "/tmp/test_27b_scales.bin") orelse return error.SkipZigTest;
    defer std.testing.allocator.free(scales_host);

    const biases_host = loadBinaryFile(u16, "/tmp/test_27b_biases.bin") orelse return error.SkipZigTest;
    defer std.testing.allocator.free(biases_host);

    const cpu_ref = loadBinaryFile(f32, "/tmp/test_27b_cpu_ref.bin") orelse return error.SkipZigTest;
    defer std.testing.allocator.free(cpu_ref);
    if (cpu_ref.len != out_dim) return error.SkipZigTest;

    // Upload to CUDA
    var input_dev = try device.allocBuffer(in_dim * @sizeOf(f32));
    defer input_dev.deinit(&device);
    var weight_dev = try device.allocBuffer(packed_weight_host.len * @sizeOf(u32));
    defer weight_dev.deinit(&device);
    var scales_dev = try device.allocBuffer(scales_host.len * @sizeOf(u16));
    defer scales_dev.deinit(&device);
    var biases_dev = try device.allocBuffer(biases_host.len * @sizeOf(u16));
    defer biases_dev.deinit(&device);
    var output_dev = try device.allocBuffer(out_dim * @sizeOf(f32));
    defer output_dev.deinit(&device);

    try input_dev.upload(&device, std.mem.sliceAsBytes(input_host));
    try weight_dev.upload(&device, std.mem.sliceAsBytes(packed_weight_host));
    try scales_dev.upload(&device, std.mem.sliceAsBytes(scales_host));
    try biases_dev.upload(&device, std.mem.sliceAsBytes(biases_host));

    // Run CUDA kernel with BF16 scales (matching actual model)
    {
        var arg_pack = cuda.ArgPack.init(std.testing.allocator);
        defer arg_pack.deinit();
        try cuda.gaffine_u4_matvec.runWithFunction(
            &arg_pack,
            &device,
            resolved.function,
            &input_dev,
            &weight_dev,
            &scales_dev,
            &biases_dev,
            &output_dev,
            in_dim,
            out_dim,
            group_size,
            cuda.gaffine_u4_matvec.scales_dtype_bf16,
            1,
            0,
        );
    }
    try device.synchronize();

    var cuda_output: [16]f32 = undefined;
    try output_dev.download(&device, std.mem.sliceAsBytes(cuda_output[0..]));

    // Compare
    var max_abs_err: f32 = 0.0;
    for (cpu_ref, cuda_output[0..], 0..) |want, got, idx| {
        const abs_err = @abs(want - got);
        if (abs_err > max_abs_err) max_abs_err = abs_err;
        if (abs_err > 0.05) {
            std.debug.print(
                "ACTUAL MODEL WEIGHT mismatch idx={} cpu_ref={d:.6} cuda={d:.6} abs_diff={d:.6}\n",
                .{ idx, want, got, abs_err },
            );
        }
    }
    std.debug.print("Max abs error (actual weights): {d:.6}\n", .{max_abs_err});
    try std.testing.expect(max_abs_err < 0.05);
}

fn loadBinaryFile(comptime T: type, path: []const u8) ?[]T {
    const file = std.fs.openFileAbsolute(path, .{}) catch return null;
    defer file.close();
    const stat = file.stat() catch return null;
    const byte_count = stat.size;
    if (byte_count == 0 or byte_count % @sizeOf(T) != 0) return null;
    const elem_count = byte_count / @sizeOf(T);
    const buffer = std.testing.allocator.alloc(T, elem_count) catch return null;
    const bytes_read = file.readAll(std.mem.sliceAsBytes(buffer)) catch {
        std.testing.allocator.free(buffer);
        return null;
    };
    if (bytes_read != byte_count) {
        std.testing.allocator.free(buffer);
        return null;
    }
    return buffer;
}

test "gaffine_u4_matvec matches CPU reference with bf16 scales at 9b gate_proj shape" {
    try expectCudaMatchesCpuReference(.{
        .in_dim = 4096,
        .out_dim = 12288,
        .group_size = 32,
        .batch_rows = 1,
    }, .bf16);
}
