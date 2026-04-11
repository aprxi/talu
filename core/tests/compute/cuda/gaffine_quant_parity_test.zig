//! Regression tests: CUDA grouped-affine U4 quantization must produce
//! BF16 scales/biases and packed nibbles matching the CPU quantization path.
//!
//! Background: The CUDA kernel's F32→BF16 conversion previously used
//! round-to-nearest-even while the CPU path uses truncation (bits >> 16).
//! This caused scale/bias divergence that accumulated over 64 layers in
//! large models (Qwen3.5-27B), producing corrupt model files.

const std = @import("std");
const main = @import("main");
const cuda = main.core.compute.cuda;

const quant_kernel_symbol: [:0]const u8 = "talu_gaffine_quantize_u4_f32";

/// CPU truncation BF16 — the authoritative reference for the converter.
/// Must match core/src/converter/root.zig:f32ToBf16.
fn f32ToBf16Truncate(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    return @intCast(bits >> 16);
}

/// CPU-side grouped-affine U4 quantization matching quantizeRowSlice in
/// core/src/converter/grouped_affine.zig. Produces packed nibbles + BF16
/// scale/bias using truncation.
fn cpuQuantizeRow(
    row_values: []const f32,
    group_len: usize,
    group_scale_factors: []const f32,
    group_bias_shifts: []const f32,
    group_round_shifts: []const f32,
    packed_out: []u32,
    scales_out: []u16,
    biases_out: []u16,
) void {
    const col_count = row_values.len;
    const group_count = col_count / group_len;

    for (0..group_count) |group_idx| {
        const group_start = group_idx * group_len;
        const group_values = row_values[group_start .. group_start + group_len];

        var min_val: f32 = group_values[0];
        var max_val: f32 = group_values[0];
        for (group_values) |value| {
            if (value < min_val) min_val = value;
            if (value > max_val) max_val = value;
        }
        const base_scale: f32 = if (max_val > min_val) (max_val - min_val) / 15.0 else 0;
        const group_scale_factor = if (group_idx < group_scale_factors.len)
            group_scale_factors[group_idx]
        else
            1.0;
        const group_bias_shift = if (group_idx < group_bias_shifts.len)
            group_bias_shifts[group_idx]
        else
            0.0;
        const group_round_shift = if (group_idx < group_round_shifts.len)
            group_round_shifts[group_idx]
        else
            0.0;
        const group_scale = base_scale * group_scale_factor;
        const group_bias = if (group_scale > 0)
            min_val + group_bias_shift * group_scale
        else
            min_val;

        scales_out[group_idx] = f32ToBf16Truncate(group_scale);
        biases_out[group_idx] = f32ToBf16Truncate(group_bias);

        const words_per_group = group_len / 8;
        for (0..words_per_group) |pack_word_idx| {
            const value_base = group_start + pack_word_idx * 8;
            var packed_word: u32 = 0;

            for (0..8) |value_idx| {
                const value = row_values[value_base + value_idx];
                var quantized: u32 = 0;
                if (group_scale > 0) {
                    const normalized = (value - group_bias) / group_scale + group_round_shift;
                    quantized = @intFromFloat(@max(0, @min(15, @round(normalized))));
                }
                packed_word |= quantized << @intCast(value_idx * 4);
            }
            packed_out[(group_start / 8) + pack_word_idx] = packed_word;
        }
    }
}

test "CUDA gaffine U4 quantization produces BF16 scales matching CPU truncation" {
    // Regression: CUDA previously used round-to-nearest-even for BF16, while
    // CPU uses truncation. Both must produce bit-identical BF16 scale/bias.
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
    const quant_fn = registry.resolveFunction(
        quant_kernel_symbol,
        quant_kernel_symbol,
    ) catch return error.SkipZigTest;

    // Test dimensions matching Qwen3.5-27B gate_proj: 17408 rows × 5120 cols, group_size=32
    // We use a smaller slice (4 rows) to keep the test fast.
    const row_count: u32 = 4;
    const col_count: u32 = 5120;
    const group_size: u32 = 32;
    const group_count: u32 = col_count / group_size;
    const packed_col_count: u32 = col_count / 8;

    // Generate deterministic test data with varied weight distributions.
    var rng = std.Random.DefaultPrng.init(0xDEAD_BEEF_1234);
    const random = rng.random();

    const source = try std.testing.allocator.alloc(f32, row_count * col_count);
    defer std.testing.allocator.free(source);
    for (source) |*v| {
        v.* = (random.float(f32) - 0.5) * 0.1;
    }

    // Use non-trivial calibration parameters to exercise the full code path.
    const scale_factors = try std.testing.allocator.alloc(f32, group_count);
    defer std.testing.allocator.free(scale_factors);
    const bias_shifts = try std.testing.allocator.alloc(f32, group_count);
    defer std.testing.allocator.free(bias_shifts);
    const round_shifts = try std.testing.allocator.alloc(f32, group_count);
    defer std.testing.allocator.free(round_shifts);
    for (0..group_count) |g| {
        scale_factors[g] = 0.85 + random.float(f32) * 0.3;
        bias_shifts[g] = (random.float(f32) - 0.5) * 0.2;
        round_shifts[g] = (random.float(f32) - 0.5) * 0.1;
    }

    // --- CPU quantization ---
    const cpu_packed = try std.testing.allocator.alloc(u32, row_count * packed_col_count);
    defer std.testing.allocator.free(cpu_packed);
    const cpu_scales = try std.testing.allocator.alloc(u16, row_count * group_count);
    defer std.testing.allocator.free(cpu_scales);
    const cpu_biases = try std.testing.allocator.alloc(u16, row_count * group_count);
    defer std.testing.allocator.free(cpu_biases);

    for (0..row_count) |row| {
        const row_values = source[row * col_count .. (row + 1) * col_count];
        const row_packed = cpu_packed[row * packed_col_count .. (row + 1) * packed_col_count];
        const row_scales = cpu_scales[row * group_count .. (row + 1) * group_count];
        const row_biases = cpu_biases[row * group_count .. (row + 1) * group_count];
        cpuQuantizeRow(
            row_values,
            group_size,
            scale_factors,
            bias_shifts,
            round_shifts,
            row_packed,
            row_scales,
            row_biases,
        );
    }

    // --- CUDA quantization ---
    const input_bytes = row_count * col_count * @sizeOf(f32);
    const packed_bytes = row_count * packed_col_count * @sizeOf(u32);
    const sb_bytes = row_count * group_count * @sizeOf(u16);
    const factor_bytes = group_count * @sizeOf(f32);

    var input_dev = try device.allocBuffer(input_bytes);
    defer input_dev.deinit(&device);
    var packed_dev = try device.allocBuffer(packed_bytes);
    defer packed_dev.deinit(&device);
    var scales_dev = try device.allocBuffer(sb_bytes);
    defer scales_dev.deinit(&device);
    var biases_dev = try device.allocBuffer(sb_bytes);
    defer biases_dev.deinit(&device);
    var sf_dev = try device.allocBuffer(factor_bytes);
    defer sf_dev.deinit(&device);
    var bs_dev = try device.allocBuffer(factor_bytes);
    defer bs_dev.deinit(&device);
    var rs_dev = try device.allocBuffer(factor_bytes);
    defer rs_dev.deinit(&device);

    try input_dev.upload(&device, std.mem.sliceAsBytes(source));
    try sf_dev.upload(&device, std.mem.sliceAsBytes(scale_factors));
    try bs_dev.upload(&device, std.mem.sliceAsBytes(bias_shifts));
    try rs_dev.upload(&device, std.mem.sliceAsBytes(round_shifts));

    // Launch: grid=(group_count, row_count), block=(128)
    var arg_pack = cuda.args.ArgPack.init(std.testing.allocator);
    defer arg_pack.deinit();
    try arg_pack.appendBufferPtr(&input_dev);
    try arg_pack.appendBufferPtr(&sf_dev);
    try arg_pack.appendBufferPtr(&bs_dev);
    try arg_pack.appendBufferPtr(&rs_dev);
    try arg_pack.appendBufferPtr(&packed_dev);
    try arg_pack.appendBufferPtr(&scales_dev);
    try arg_pack.appendBufferPtr(&biases_dev);
    try arg_pack.appendScalar(u32, row_count);
    try arg_pack.appendScalar(u32, col_count);
    try arg_pack.appendScalar(u32, group_size);
    try arg_pack.appendScalar(u32, packed_col_count);

    try cuda.launch.launchWithFamily(
        &device,
        quant_fn,
        .{
            .grid_x = group_count,
            .grid_y = row_count,
            .block_x = 128,
        },
        &arg_pack,
        .pointwise,
    );
    try device.synchronize();

    // --- Download CUDA results ---
    const cuda_packed = try std.testing.allocator.alloc(u32, row_count * packed_col_count);
    defer std.testing.allocator.free(cuda_packed);
    const cuda_scales = try std.testing.allocator.alloc(u16, row_count * group_count);
    defer std.testing.allocator.free(cuda_scales);
    const cuda_biases = try std.testing.allocator.alloc(u16, row_count * group_count);
    defer std.testing.allocator.free(cuda_biases);

    try packed_dev.download(&device, std.mem.sliceAsBytes(cuda_packed));
    try scales_dev.download(&device, std.mem.sliceAsBytes(cuda_scales));
    try biases_dev.download(&device, std.mem.sliceAsBytes(cuda_biases));

    // --- Verify BF16 scales and biases are bit-identical ---
    var scale_mismatches: usize = 0;
    var bias_mismatches: usize = 0;
    for (0..row_count * group_count) |i| {
        if (cuda_scales[i] != cpu_scales[i]) {
            if (scale_mismatches < 5) {
                std.debug.print(
                    "Scale mismatch at [{d}]: CUDA=0x{X:0>4} CPU=0x{X:0>4}\n",
                    .{ i, cuda_scales[i], cpu_scales[i] },
                );
            }
            scale_mismatches += 1;
        }
        if (cuda_biases[i] != cpu_biases[i]) {
            if (bias_mismatches < 5) {
                std.debug.print(
                    "Bias mismatch at [{d}]: CUDA=0x{X:0>4} CPU=0x{X:0>4}\n",
                    .{ i, cuda_biases[i], cpu_biases[i] },
                );
            }
            bias_mismatches += 1;
        }
    }
    if (scale_mismatches > 0 or bias_mismatches > 0) {
        std.debug.print(
            "BF16 parity FAILED: {d}/{d} scale mismatches, {d}/{d} bias mismatches\n",
            .{ scale_mismatches, row_count * group_count, bias_mismatches, row_count * group_count },
        );
        return error.TestExpectedEqual;
    }

    // --- Verify packed nibbles are bit-identical ---
    var nibble_mismatches: usize = 0;
    for (0..row_count * packed_col_count) |i| {
        if (cuda_packed[i] != cpu_packed[i]) {
            if (nibble_mismatches < 5) {
                std.debug.print(
                    "Packed word mismatch at [{d}]: CUDA=0x{X:0>8} CPU=0x{X:0>8}\n",
                    .{ i, cuda_packed[i], cpu_packed[i] },
                );
            }
            nibble_mismatches += 1;
        }
    }
    if (nibble_mismatches > 0) {
        std.debug.print(
            "Nibble parity FAILED: {d}/{d} word mismatches\n",
            .{ nibble_mismatches, row_count * packed_col_count },
        );
        return error.TestExpectedEqual;
    }
}
