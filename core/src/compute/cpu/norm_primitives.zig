//! Normalization operations with stride-aware implementations.

const std = @import("std");
const tensor = @import("../../tensor.zig");
const tv = @import("tensor_view.zig");
const math = @import("math.zig");

const TensorView = tv.TensorView;
const DType = tv.DType;
const MAX_NDIM = tv.MAX_NDIM;
const TensorDType = tensor.DType;

// SIMD infrastructure
const simd = math.simd;
const VEC_LEN = simd.f32_vec_len;
const F32Vec = simd.F32Vec;

/// Dtype conversion helpers - wrapped to remove inline calling convention
const dtype_mod = @import("../../dtype.zig");

fn fp16ToF32(x: u16) f32 {
    return dtype_mod.fp16ToF32(x);
}

fn f32ToFp16(x: f32) u16 {
    return dtype_mod.f32ToFp16(x);
}

fn bf16ToF32(x: u16) f32 {
    return dtype_mod.bf16ToF32(x);
}

fn f32ToBf16(x: f32) u16 {
    return dtype_mod.f32ToBf16(x);
}

/// Read a weight/bias value converting from the tensor's dtype to f32
/// For contiguous tensors, offset == index. For strided, use coordsToOffset.
fn readWeightF32(t: TensorView, offset: usize) f32 {
    return switch (t.dtype) {
        .f32 => @as([*]const f32, @ptrCast(@alignCast(t.data)))[offset],
        .f16 => fp16ToF32(@as([*]const u16, @ptrCast(@alignCast(t.data)))[offset]),
        .bf16 => bf16ToF32(@as([*]const u16, @ptrCast(@alignCast(t.data)))[offset]),
        else => 0.0,
    };
}

fn viewDTypeToTensor(dt: DType) TensorDType {
    return switch (dt) {
        .f32 => .f32,
        .f16 => .f16,
        .bf16 => .bf16,
        else => .f32,
    };
}

/// RMS Normalization: out = x * rsqrt(mean(x^2) + eps) * weight
pub fn rmsNorm(out: TensorView, input: TensorView, weight: TensorView, eps: f32) void {
    const weight_dtype_ok = weight.dtype == .f32 or weight.dtype == .f16 or weight.dtype == .bf16;
    if (input.dtype == .f32 and out.dtype == .f32 and input.isContiguous() and out.isContiguous() and weight.isContiguous() and weight_dtype_ok) {
        const hidden_size = input.shape[input.ndim - 1];
        const num_tokens = input.numel / hidden_size;
        const weight_f32_slice = if (weight.dtype == .f32) weight.asSlice(f32) else null;
        const weight_u16_slice = if (weight.dtype == .f16 or weight.dtype == .bf16) weight.asSlice(u16) else null;
        math.rmsnormContiguous(
            out.asSlice(f32),
            input.asSlice(f32),
            weight_f32_slice,
            weight_u16_slice,
            viewDTypeToTensor(weight.dtype),
            num_tokens,
            hidden_size,
            eps,
            0.0,
        );
        return;
    }

    switch (input.dtype) {
        .f32 => rmsNormTyped(f32, f32Identity, f32Identity, out, input, weight, eps),
        .f16 => rmsNormTyped(u16, fp16ToF32, f32ToFp16, out, input, weight, eps),
        .bf16 => rmsNormTyped(u16, bf16ToF32, f32ToBf16, out, input, weight, eps),
        else => unreachable,
    }
}

fn f32Identity(x: f32) f32 {
    return x;
}

fn rmsNormTyped(
    comptime T: type,
    comptime toF32: fn (T) f32,
    comptime fromF32: fn (f32) T,
    out: TensorView,
    input: TensorView,
    weight: TensorView,
    eps: f32,
) void {
    const out_data = @as([*]T, @ptrCast(@alignCast(out.data)));
    const in_data = @as([*]const T, @ptrCast(@alignCast(input.data)));
    const w_data = @as([*]const T, @ptrCast(@alignCast(weight.data)));

    std.debug.assert(input.ndim >= 1);
    const hidden_size = input.shape[input.ndim - 1];
    const num_tokens = input.numel / hidden_size;

    if (input.isContiguous() and out.isContiguous() and weight.isContiguous()) {
        // Fast path: all contiguous
        for (0..num_tokens) |token_idx| {
            const token_offset = token_idx * hidden_size;

            // Compute sum of squares
            var sum_sq: f32 = 0;
            for (0..hidden_size) |elem_idx| {
                const input_value = toF32(in_data[token_offset + elem_idx]);
                sum_sq += input_value * input_value;
            }

            // Compute inverse RMS
            const inv_rms = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(hidden_size)) + eps);

            // Apply normalization with weight
            for (0..hidden_size) |elem_idx| {
                const input_value = toF32(in_data[token_offset + elem_idx]);
                const weight_value = toF32(w_data[elem_idx]);
                out_data[token_offset + elem_idx] = fromF32(input_value * inv_rms * weight_value);
            }
        }
    } else {
        // Optimized strided path: use incremental offset calculation.
        // For each row, compute base offset once, then stride along last dim.
        const in_last_stride = input.strides[input.ndim - 1];
        const out_last_stride = out.strides[out.ndim - 1];
        const w_stride = weight.strides[0]; // weight is 1D

        // Track outer coordinates with incremental offset updates
        var outer_coords: [MAX_NDIM]usize = [_]usize{0} ** MAX_NDIM;
        var in_base: usize = 0;
        var out_base: usize = 0;

        for (0..num_tokens) |_| {
            // Compute sum of squares along last dim using incremental offsets
            var sum_sq: f32 = 0;
            var in_off = in_base;
            for (0..hidden_size) |_| {
                const input_value = toF32(in_data[in_off]);
                sum_sq += input_value * input_value;
                in_off += in_last_stride;
            }

            const inv_rms = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(hidden_size)) + eps);

            // Apply normalization using incremental offsets
            in_off = in_base;
            var out_off = out_base;
            var w_off: usize = 0;
            for (0..hidden_size) |_| {
                const input_value = toF32(in_data[in_off]);
                const weight_value = toF32(w_data[w_off]);
                out_data[out_off] = fromF32(input_value * inv_rms * weight_value);
                in_off += in_last_stride;
                out_off += out_last_stride;
                w_off += w_stride;
            }

            // Advance outer coordinates (dims 0 to ndim-2)
            if (input.ndim > 1) {
                var dim_idx: usize = input.ndim - 1;
                while (dim_idx > 0) {
                    dim_idx -= 1;
                    outer_coords[dim_idx] += 1;
                    in_base += input.strides[dim_idx];
                    out_base += out.strides[dim_idx];

                    if (outer_coords[dim_idx] < input.shape[dim_idx]) break;

                    // Carry: reset this dimension
                    const carried = input.shape[dim_idx];
                    outer_coords[dim_idx] = 0;
                    in_base -= carried * input.strides[dim_idx];
                    out_base -= carried * out.strides[dim_idx];
                }
            }
        }
    }
}

/// Layer Normalization: out = (x - mean) / sqrt(var + eps) * weight + bias
pub fn layerNorm(out: TensorView, input: TensorView, weight: TensorView, bias: ?TensorView, eps: f32) void {
    const bias_dtype_ok = if (bias) |b| b.dtype == .f32 else true;
    if (input.dtype == .f32 and out.dtype == .f32 and weight.dtype == .f32 and bias_dtype_ok and
        input.isContiguous() and out.isContiguous() and weight.isContiguous() and (bias == null or bias.?.isContiguous()))
    {
        const hidden_size = input.shape[input.ndim - 1];
        const num_tokens = input.numel / hidden_size;
        const bias_slice = if (bias) |b| b.asSlice(f32) else null;
        math.layerNormContiguous(out.asSlice(f32), input.asSlice(f32), weight.asSlice(f32), bias_slice, num_tokens, hidden_size, eps);
        return;
    }

    switch (input.dtype) {
        .f32 => layerNormTyped(f32, f32Identity, f32Identity, out, input, weight, bias, eps),
        .f16 => layerNormTyped(u16, fp16ToF32, f32ToFp16, out, input, weight, bias, eps),
        .bf16 => layerNormTyped(u16, bf16ToF32, f32ToBf16, out, input, weight, bias, eps),
        else => unreachable,
    }
}

fn layerNormTyped(
    comptime T: type,
    comptime toF32: fn (T) f32,
    comptime fromF32: fn (f32) T,
    out: TensorView,
    input: TensorView,
    weight: TensorView,
    bias: ?TensorView,
    eps: f32,
) void {
    const out_data = @as([*]T, @ptrCast(@alignCast(out.data)));
    const in_data = @as([*]const T, @ptrCast(@alignCast(input.data)));
    // Note: weight/bias may have different dtypes than input, use readWeightF32 helper

    std.debug.assert(input.ndim >= 1);
    const hidden_size = input.shape[input.ndim - 1];
    const num_tokens = input.numel / hidden_size;
    const hidden_f32 = @as(f32, @floatFromInt(hidden_size));

    if (input.isContiguous() and out.isContiguous() and weight.isContiguous()) {
        // Fast path: all contiguous
        for (0..num_tokens) |token_idx| {
            const token_offset = token_idx * hidden_size;

            // Compute mean
            var sum: f32 = 0;
            for (0..hidden_size) |elem_idx| {
                sum += toF32(in_data[token_offset + elem_idx]);
            }
            const mean = sum / hidden_f32;

            // Compute variance
            var var_sum: f32 = 0;
            for (0..hidden_size) |elem_idx| {
                const diff = toF32(in_data[token_offset + elem_idx]) - mean;
                var_sum += diff * diff;
            }
            const inv_std = 1.0 / @sqrt(var_sum / hidden_f32 + eps);

            // Apply normalization with weight and bias (handle mixed dtypes)
            for (0..hidden_size) |elem_idx| {
                const normalized = (toF32(in_data[token_offset + elem_idx]) - mean) * inv_std;
                const weight_value = readWeightF32(weight, elem_idx);
                const bias_value = if (bias) |b| readWeightF32(b, elem_idx) else 0.0;
                out_data[token_offset + elem_idx] = fromF32(normalized * weight_value + bias_value);
            }
        }
    } else {
        // Optimized strided path: use incremental offset calculation.
        const in_last_stride = input.strides[input.ndim - 1];
        const out_last_stride = out.strides[out.ndim - 1];
        const w_stride = weight.strides[0];
        const b_stride = if (bias) |b| b.strides[0] else 0;

        // Track outer coordinates with incremental offset updates
        var outer_coords: [MAX_NDIM]usize = [_]usize{0} ** MAX_NDIM;
        var in_base: usize = 0;
        var out_base: usize = 0;

        for (0..num_tokens) |_| {
            // Compute mean using incremental offsets
            var sum: f32 = 0;
            var in_off = in_base;
            for (0..hidden_size) |_| {
                sum += toF32(in_data[in_off]);
                in_off += in_last_stride;
            }
            const mean = sum / hidden_f32;

            // Compute variance using incremental offsets
            var var_sum: f32 = 0;
            in_off = in_base;
            for (0..hidden_size) |_| {
                const diff = toF32(in_data[in_off]) - mean;
                var_sum += diff * diff;
                in_off += in_last_stride;
            }
            const inv_std = 1.0 / @sqrt(var_sum / hidden_f32 + eps);

            // Apply normalization using incremental offsets
            in_off = in_base;
            var out_off = out_base;
            var w_off: usize = 0;
            var b_off: usize = 0;
            for (0..hidden_size) |_| {
                const normalized = (toF32(in_data[in_off]) - mean) * inv_std;
                const weight_value = readWeightF32(weight, w_off);
                const bias_value = if (bias) |b| readWeightF32(b, b_off) else 0.0;
                out_data[out_off] = fromF32(normalized * weight_value + bias_value);
                in_off += in_last_stride;
                out_off += out_last_stride;
                w_off += w_stride;
                b_off += b_stride;
            }

            // Advance outer coordinates (dims 0 to ndim-2)
            if (input.ndim > 1) {
                var dim_idx: usize = input.ndim - 1;
                while (dim_idx > 0) {
                    dim_idx -= 1;
                    outer_coords[dim_idx] += 1;
                    in_base += input.strides[dim_idx];
                    out_base += out.strides[dim_idx];

                    if (outer_coords[dim_idx] < input.shape[dim_idx]) break;

                    // Carry: reset this dimension
                    const carried = input.shape[dim_idx];
                    outer_coords[dim_idx] = 0;
                    in_base -= carried * input.strides[dim_idx];
                    out_base -= carried * out.strides[dim_idx];
                }
            }
        }
    }
}

test "rmsNorm simple" {
    var in_data = [_]f32{ 1, 2, 3, 4 };
    var out_data = [_]f32{ 0, 0, 0, 0 };
    var w_data = [_]f32{ 1, 1, 1, 1 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{ 1, 4 }, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{ 1, 4 }, .f32);
    const weight = TensorView.initContiguous(@ptrCast(&w_data), &.{4}, .f32);

    rmsNorm(out, input, weight, 1e-6);

    // Verify output is normalized
    var sum_sq: f32 = 0;
    for (out_data) |v| sum_sq += v * v;
    const rms = @sqrt(sum_sq / 4.0);

    // RMS of normalized output should be close to 1 (within tolerance)
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), rms, 0.01);
}

test "layerNorm simple" {
    var in_data = [_]f32{ 1, 2, 3, 4 };
    var out_data = [_]f32{ 0, 0, 0, 0 };
    var w_data = [_]f32{ 1, 1, 1, 1 };
    var b_data = [_]f32{ 0, 0, 0, 0 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{ 1, 4 }, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{ 1, 4 }, .f32);
    const weight = TensorView.initContiguous(@ptrCast(&w_data), &.{4}, .f32);
    const bias = TensorView.initContiguous(@ptrCast(&b_data), &.{4}, .f32);

    layerNorm(out, input, weight, bias, 1e-6);

    // Verify mean is ~0 and std is ~1
    var sum: f32 = 0;
    for (out_data) |v| sum += v;
    const mean = sum / 4.0;

    var var_sum: f32 = 0;
    for (out_data) |v| {
        const diff = v - mean;
        var_sum += diff * diff;
    }
    const std_val = @sqrt(var_sum / 4.0);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mean, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), std_val, 0.01);
}

// ============================================================================
// Extended Unit Tests for RMSNorm and LayerNorm
// ============================================================================

test "rmsNorm correctness - verify formula x / sqrt(mean(x^2) + eps) * weight" {
    // Test the exact RMSNorm formula with known inputs
    var in_data = [_]f32{ 2.0, 4.0, 6.0, 8.0 };
    var out_data = [_]f32{ 0, 0, 0, 0 };
    var w_data = [_]f32{ 0.5, 1.0, 1.5, 2.0 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{4}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{4}, .f32);
    const weight = TensorView.initContiguous(@ptrCast(&w_data), &.{4}, .f32);

    const eps: f32 = 1e-6;
    rmsNorm(out, input, weight, eps);

    // Manually calculate expected values
    // mean(x^2) = (4 + 16 + 36 + 64) / 4 = 30
    // sqrt(30 + 1e-6) ≈ 5.477226
    const sum_sq: f32 = 4.0 + 16.0 + 36.0 + 64.0;
    const mean_sq = sum_sq / 4.0;
    const inv_rms = 1.0 / @sqrt(mean_sq + eps);

    // Expected: input[i] * inv_rms * weight[i]
    try std.testing.expectApproxEqRel(2.0 * inv_rms * 0.5, out_data[0], 1e-5);
    try std.testing.expectApproxEqRel(4.0 * inv_rms * 1.0, out_data[1], 1e-5);
    try std.testing.expectApproxEqRel(6.0 * inv_rms * 1.5, out_data[2], 1e-5);
    try std.testing.expectApproxEqRel(8.0 * inv_rms * 2.0, out_data[3], 1e-5);
}

test "rmsNorm epsilon handling - numerical stability with very small values" {
    // Test that epsilon prevents division by zero with very small inputs
    var in_data = [_]f32{ 1e-8, 1e-8, 1e-8, 1e-8 };
    var out_data = [_]f32{ 0, 0, 0, 0 };
    var w_data = [_]f32{ 1, 1, 1, 1 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{4}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{4}, .f32);
    const weight = TensorView.initContiguous(@ptrCast(&w_data), &.{4}, .f32);

    const eps: f32 = 1e-5;
    rmsNorm(out, input, weight, eps);

    // With very small inputs, epsilon dominates: inv_rms ≈ 1/sqrt(eps)
    // Output should be finite and not NaN
    for (out_data) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }

    // Test with different epsilon values
    const eps_small: f32 = 1e-10;
    rmsNorm(out, input, weight, eps_small);
    for (out_data) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }
}

test "rmsNorm weight scaling - verify weights applied correctly" {
    // Test that different weights produce proportional outputs
    var in_data = [_]f32{ 1, 1, 1, 1 };
    var out_data = [_]f32{ 0, 0, 0, 0 };
    var w_data = [_]f32{ 0.5, 1.0, 2.0, 4.0 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{4}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{4}, .f32);
    const weight = TensorView.initContiguous(@ptrCast(&w_data), &.{4}, .f32);

    rmsNorm(out, input, weight, 1e-6);

    // With uniform input, output should be proportional to weights
    // Check that weight ratios are preserved
    try std.testing.expectApproxEqRel(out_data[1], out_data[0] * 2.0, 1e-5);
    try std.testing.expectApproxEqRel(out_data[2], out_data[0] * 4.0, 1e-5);
    try std.testing.expectApproxEqRel(out_data[3], out_data[0] * 8.0, 1e-5);

    // Test with zero weight
    w_data[2] = 0.0;
    rmsNorm(out, input, weight, 1e-6);
    try std.testing.expectEqual(@as(f32, 0.0), out_data[2]);
}

test "rmsNorm different tensor shapes - 1D vector" {
    // Test with simple 1D vector
    var in_data = [_]f32{ 3.0, 4.0 };
    var out_data = [_]f32{ 0, 0 };
    var w_data = [_]f32{ 1, 1 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{2}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{2}, .f32);
    const weight = TensorView.initContiguous(@ptrCast(&w_data), &.{2}, .f32);

    rmsNorm(out, input, weight, 1e-6);

    // mean(x^2) = (9 + 16) / 2 = 12.5
    // inv_rms = 1 / sqrt(12.5) ≈ 0.2828
    const inv_rms = 1.0 / @sqrt(12.5 + 1e-6);
    try std.testing.expectApproxEqRel(3.0 * inv_rms, out_data[0], 1e-5);
    try std.testing.expectApproxEqRel(4.0 * inv_rms, out_data[1], 1e-5);
}

test "rmsNorm different tensor shapes - 2D batched" {
    // Test with 2D tensor (batch_size=2, hidden_size=3)
    var in_data = [_]f32{
        1.0, 2.0, 3.0, // batch 0
        4.0, 5.0, 6.0, // batch 1
    };
    var out_data = [_]f32{ 0, 0, 0, 0, 0, 0 };
    var w_data = [_]f32{ 1.0, 1.0, 1.0 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{ 2, 3 }, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{ 2, 3 }, .f32);
    const weight = TensorView.initContiguous(@ptrCast(&w_data), &.{3}, .f32);

    rmsNorm(out, input, weight, 1e-6);

    // Each row should be normalized independently
    // Batch 0: mean(x^2) = (1 + 4 + 9) / 3 = 14/3
    const inv_rms_0 = 1.0 / @sqrt(14.0 / 3.0 + 1e-6);
    try std.testing.expectApproxEqRel(1.0 * inv_rms_0, out_data[0], 1e-5);
    try std.testing.expectApproxEqRel(2.0 * inv_rms_0, out_data[1], 1e-5);
    try std.testing.expectApproxEqRel(3.0 * inv_rms_0, out_data[2], 1e-5);

    // Batch 1: mean(x^2) = (16 + 25 + 36) / 3 = 77/3
    const inv_rms_1 = 1.0 / @sqrt(77.0 / 3.0 + 1e-6);
    try std.testing.expectApproxEqRel(4.0 * inv_rms_1, out_data[3], 1e-5);
    try std.testing.expectApproxEqRel(5.0 * inv_rms_1, out_data[4], 1e-5);
    try std.testing.expectApproxEqRel(6.0 * inv_rms_1, out_data[5], 1e-5);
}

test "rmsNorm different tensor shapes - 3D batched" {
    // Test with 3D tensor (batch=2, seq=2, hidden=2)
    var in_data = [_]f32{
        1.0, 2.0, // batch 0, seq 0
        3.0, 4.0, // batch 0, seq 1
        5.0, 6.0, // batch 1, seq 0
        7.0, 8.0, // batch 1, seq 1
    };
    var out_data = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    var w_data = [_]f32{ 1.0, 1.0 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{ 2, 2, 2 }, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{ 2, 2, 2 }, .f32);
    const weight = TensorView.initContiguous(@ptrCast(&w_data), &.{2}, .f32);

    rmsNorm(out, input, weight, 1e-6);

    // Each row (last dimension) should be normalized independently
    // [1, 2]: mean(x^2) = (1 + 4) / 2 = 2.5
    const inv_rms_0 = 1.0 / @sqrt(2.5 + 1e-6);
    try std.testing.expectApproxEqRel(1.0 * inv_rms_0, out_data[0], 1e-5);
    try std.testing.expectApproxEqRel(2.0 * inv_rms_0, out_data[1], 1e-5);

    // All outputs should be valid
    for (out_data) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }
}

test "rmsNorm edge case - all zeros" {
    // Test behavior with all zero inputs
    var in_data = [_]f32{ 0, 0, 0, 0 };
    var out_data = [_]f32{ 0, 0, 0, 0 };
    var w_data = [_]f32{ 1, 1, 1, 1 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{4}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{4}, .f32);
    const weight = TensorView.initContiguous(@ptrCast(&w_data), &.{4}, .f32);

    rmsNorm(out, input, weight, 1e-6);

    // With all zeros, output should be zero (0 * anything = 0)
    for (out_data) |v| {
        try std.testing.expectEqual(@as(f32, 0.0), v);
    }
}

test "rmsNorm edge case - all same value" {
    // Test with all elements having the same value
    var in_data = [_]f32{ 5.0, 5.0, 5.0, 5.0 };
    var out_data = [_]f32{ 0, 0, 0, 0 };
    var w_data = [_]f32{ 1, 1, 1, 1 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{4}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{4}, .f32);
    const weight = TensorView.initContiguous(@ptrCast(&w_data), &.{4}, .f32);

    rmsNorm(out, input, weight, 1e-6);

    // With uniform input: mean(x^2) = 25, inv_rms = 1/5
    // Output should be: 5 * (1/5) * 1 = 1
    for (out_data) |v| {
        try std.testing.expectApproxEqRel(@as(f32, 1.0), v, 1e-5);
    }
}

test "rmsNorm edge case - very large values" {
    // Test numerical stability with large values
    var in_data = [_]f32{ 1e6, 2e6, 3e6, 4e6 };
    var out_data = [_]f32{ 0, 0, 0, 0 };
    var w_data = [_]f32{ 1, 1, 1, 1 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{4}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{4}, .f32);
    const weight = TensorView.initContiguous(@ptrCast(&w_data), &.{4}, .f32);

    rmsNorm(out, input, weight, 1e-6);

    // Output should be finite and maintain proportions
    for (out_data) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }

    // Check proportions are maintained
    try std.testing.expectApproxEqRel(out_data[1], out_data[0] * 2.0, 1e-5);
    try std.testing.expectApproxEqRel(out_data[2], out_data[0] * 3.0, 1e-5);
    try std.testing.expectApproxEqRel(out_data[3], out_data[0] * 4.0, 1e-5);
}

test "rmsNorm edge case - negative values" {
    // Test that negative values are handled correctly (squared in computation)
    var in_data = [_]f32{ -1.0, 2.0, -3.0, 4.0 };
    var out_data = [_]f32{ 0, 0, 0, 0 };
    var w_data = [_]f32{ 1, 1, 1, 1 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{4}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{4}, .f32);
    const weight = TensorView.initContiguous(@ptrCast(&w_data), &.{4}, .f32);

    rmsNorm(out, input, weight, 1e-6);

    // mean(x^2) = (1 + 4 + 9 + 16) / 4 = 7.5
    const inv_rms = 1.0 / @sqrt(7.5 + 1e-6);

    // Negative inputs produce negative outputs
    try std.testing.expectApproxEqRel(-1.0 * inv_rms, out_data[0], 1e-5);
    try std.testing.expectApproxEqRel(2.0 * inv_rms, out_data[1], 1e-5);
    try std.testing.expectApproxEqRel(-3.0 * inv_rms, out_data[2], 1e-5);
    try std.testing.expectApproxEqRel(4.0 * inv_rms, out_data[3], 1e-5);
}

test "layerNorm correctness - verify formula (x - mean) / sqrt(var + eps) * weight + bias" {
    // Test exact LayerNorm formula with known inputs
    var in_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var out_data = [_]f32{ 0, 0, 0, 0 };
    var w_data = [_]f32{ 0.5, 1.0, 1.5, 2.0 };
    var b_data = [_]f32{ 0.1, 0.2, 0.3, 0.4 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{4}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{4}, .f32);
    const weight = TensorView.initContiguous(@ptrCast(&w_data), &.{4}, .f32);
    const bias = TensorView.initContiguous(@ptrCast(&b_data), &.{4}, .f32);

    const eps: f32 = 1e-6;
    layerNorm(out, input, weight, bias, eps);

    // mean = (1 + 2 + 3 + 4) / 4 = 2.5
    // var = ((1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2) / 4 = 1.25
    // inv_std = 1 / sqrt(1.25 + 1e-6)
    const mean: f32 = 2.5;
    const variance: f32 = 1.25;
    const inv_std = 1.0 / @sqrt(variance + eps);

    // Expected: (input[i] - mean) * inv_std * weight[i] + bias[i]
    try std.testing.expectApproxEqRel((1.0 - mean) * inv_std * 0.5 + 0.1, out_data[0], 1e-5);
    try std.testing.expectApproxEqRel((2.0 - mean) * inv_std * 1.0 + 0.2, out_data[1], 1e-5);
    try std.testing.expectApproxEqRel((3.0 - mean) * inv_std * 1.5 + 0.3, out_data[2], 1e-5);
    try std.testing.expectApproxEqRel((4.0 - mean) * inv_std * 2.0 + 0.4, out_data[3], 1e-5);
}

test "layerNorm without bias - null bias parameter" {
    // Test LayerNorm with no bias term
    var in_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var out_data = [_]f32{ 0, 0, 0, 0 };
    var w_data = [_]f32{ 1, 1, 1, 1 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{4}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{4}, .f32);
    const weight = TensorView.initContiguous(@ptrCast(&w_data), &.{4}, .f32);

    layerNorm(out, input, weight, null, 1e-6);

    // Output should have zero mean and unit variance
    var sum: f32 = 0;
    for (out_data) |v| sum += v;
    const out_mean = sum / 4.0;

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out_mean, 1e-5);
}

test "layerNorm epsilon handling - numerical stability" {
    // Test with uniform input (zero variance)
    var in_data = [_]f32{ 3.0, 3.0, 3.0, 3.0 };
    var out_data = [_]f32{ 0, 0, 0, 0 };
    var w_data = [_]f32{ 1, 1, 1, 1 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{4}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{4}, .f32);
    const weight = TensorView.initContiguous(@ptrCast(&w_data), &.{4}, .f32);

    const eps: f32 = 1e-5;
    layerNorm(out, input, weight, null, eps);

    // With zero variance, epsilon prevents division by zero
    for (out_data) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }

    // With uniform input and no bias, output should be close to zero
    for (out_data) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), v, 1e-3);
    }
}

test "layerNorm weight and bias scaling" {
    // Test that weight scales and bias shifts the output correctly
    var in_data = [_]f32{ 0.0, 1.0, 2.0, 3.0 };
    var out_data = [_]f32{ 0, 0, 0, 0 };
    var w_data = [_]f32{ 2.0, 2.0, 2.0, 2.0 };
    var b_data = [_]f32{ 1.0, 1.0, 1.0, 1.0 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{4}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{4}, .f32);
    const weight = TensorView.initContiguous(@ptrCast(&w_data), &.{4}, .f32);
    const bias = TensorView.initContiguous(@ptrCast(&b_data), &.{4}, .f32);

    layerNorm(out, input, weight, bias, 1e-6);

    // With uniform weight and bias, check that bias shifts all values
    for (out_data) |v| {
        // Each normalized value * 2 + 1 should be close to 1 (on average)
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }

    // Mean should be shifted by bias (approximately 1.0)
    var sum: f32 = 0;
    for (out_data) |v| sum += v;
    const mean = sum / 4.0;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), mean, 0.1);
}

test "layerNorm different tensor shapes - batched 2D" {
    // Test with multiple sequences
    var in_data = [_]f32{
        1.0, 2.0, 3.0, // seq 0
        4.0, 5.0, 6.0, // seq 1
    };
    var out_data = [_]f32{ 0, 0, 0, 0, 0, 0 };
    var w_data = [_]f32{ 1.0, 1.0, 1.0 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{ 2, 3 }, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{ 2, 3 }, .f32);
    const weight = TensorView.initContiguous(@ptrCast(&w_data), &.{3}, .f32);

    layerNorm(out, input, weight, null, 1e-6);

    // Each sequence should be normalized independently
    // Check that each row has approximately zero mean
    const mean_0 = (out_data[0] + out_data[1] + out_data[2]) / 3.0;
    const mean_1 = (out_data[3] + out_data[4] + out_data[5]) / 3.0;

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mean_0, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mean_1, 1e-5);
}

test "layerNorm edge case - all zeros" {
    // Test with all zero inputs
    var in_data = [_]f32{ 0, 0, 0, 0 };
    var out_data = [_]f32{ 0, 0, 0, 0 };
    var w_data = [_]f32{ 1, 1, 1, 1 };
    var b_data = [_]f32{ 0.5, 0.5, 0.5, 0.5 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{4}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{4}, .f32);
    const weight = TensorView.initContiguous(@ptrCast(&w_data), &.{4}, .f32);
    const bias = TensorView.initContiguous(@ptrCast(&b_data), &.{4}, .f32);

    layerNorm(out, input, weight, bias, 1e-6);

    // With zero input, mean=0, var=0, output should be 0 * weight + bias = bias
    for (out_data) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.5), v, 1e-3);
    }
}

test "layerNorm edge case - very large values" {
    // Test numerical stability with large inputs
    var in_data = [_]f32{ 1e5, 2e5, 3e5, 4e5 };
    var out_data = [_]f32{ 0, 0, 0, 0 };
    var w_data = [_]f32{ 1, 1, 1, 1 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{4}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{4}, .f32);
    const weight = TensorView.initContiguous(@ptrCast(&w_data), &.{4}, .f32);

    layerNorm(out, input, weight, null, 1e-6);

    // Output should be finite and normalized
    for (out_data) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }

    // Mean should be close to zero
    var sum: f32 = 0;
    for (out_data) |v| sum += v;
    const mean = sum / 4.0;
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mean, 1e-4);
}

// ============================================================================
// Tests for Different Weight Data Types (f16, bf16)
// ============================================================================

test "rmsNorm with f16 weights" {
    // Test RMSNorm with fp16 weights
    var in_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var out_data = [_]f32{ 0, 0, 0, 0 };
    var w_data = [_]u16{
        f32ToFp16(0.5),
        f32ToFp16(1.0),
        f32ToFp16(1.5),
        f32ToFp16(2.0),
    };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{4}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{4}, .f32);
    const weight = TensorView.initContiguous(@ptrCast(&w_data), &.{4}, .f16);

    rmsNorm(out, input, weight, 1e-6);

    // Verify output is valid and proportional to weights
    for (out_data) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }

    // Check weight scaling is applied correctly (with some fp16 precision loss)
    // out[i] = normalized[i] * weight[i], where normalized[i] = input[i] * inv_rms
    // Ratio: out[i]/out[0] = (input[i] * weight[i]) / (input[0] * weight[0])
    // With input=[1,2,3,4] and weights=[0.5,1.0,1.5,2.0]:
    // out[1]/out[0] = (2*1.0)/(1*0.5) = 4.0
    // out[2]/out[0] = (3*1.5)/(1*0.5) = 9.0
    // out[3]/out[0] = (4*2.0)/(1*0.5) = 16.0
    try std.testing.expectApproxEqRel(out_data[1], out_data[0] * 4.0, 1e-2);
    try std.testing.expectApproxEqRel(out_data[2], out_data[0] * 9.0, 1e-2);
    try std.testing.expectApproxEqRel(out_data[3], out_data[0] * 16.0, 1e-2);
}

test "rmsNorm with bf16 weights" {
    // Test RMSNorm with bfloat16 weights
    var in_data = [_]f32{ 2.0, 4.0, 6.0, 8.0 };
    var out_data = [_]f32{ 0, 0, 0, 0 };
    var w_data = [_]u16{
        f32ToBf16(1.0),
        f32ToBf16(2.0),
        f32ToBf16(3.0),
        f32ToBf16(4.0),
    };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{4}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{4}, .f32);
    const weight = TensorView.initContiguous(@ptrCast(&w_data), &.{4}, .bf16);

    rmsNorm(out, input, weight, 1e-6);

    // Verify output is valid
    for (out_data) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }

    // Check that weight scaling maintains proportions (with bf16 precision)
    // With input=[2,4,6,8] and weights=[1,2,3,4]:
    // out[1]/out[0] = (4*2)/(2*1) = 4.0
    // out[2]/out[0] = (6*3)/(2*1) = 9.0
    try std.testing.expectApproxEqRel(out_data[1], out_data[0] * 4.0, 1e-2);
    try std.testing.expectApproxEqRel(out_data[2], out_data[0] * 9.0, 1e-2);
}

test "rmsNorm with f16 weights - batched" {
    // Test batched RMSNorm with fp16 weights
    var in_data = [_]f32{
        1.0, 2.0, 3.0, // batch 0
        4.0, 5.0, 6.0, // batch 1
    };
    var out_data = [_]f32{ 0, 0, 0, 0, 0, 0 };
    var w_data = [_]u16{
        f32ToFp16(1.0),
        f32ToFp16(1.0),
        f32ToFp16(1.0),
    };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{ 2, 3 }, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{ 2, 3 }, .f32);
    const weight = TensorView.initContiguous(@ptrCast(&w_data), &.{3}, .f16);

    rmsNorm(out, input, weight, 1e-6);

    // Both batches should be normalized correctly
    for (out_data) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }
}

test "layerNorm with f16 weights" {
    // Test LayerNorm with fp16 weights
    var in_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var out_data = [_]f32{ 0, 0, 0, 0 };
    var w_data = [_]u16{
        f32ToFp16(0.5),
        f32ToFp16(1.0),
        f32ToFp16(1.5),
        f32ToFp16(2.0),
    };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{4}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{4}, .f32);
    const weight = TensorView.initContiguous(@ptrCast(&w_data), &.{4}, .f16);

    // Note: f16 weights currently not supported for layerNorm fast path,
    // so this tests the fallback typed path
    layerNorm(out, input, weight, null, 1e-6);

    // Verify output is valid (non-NaN, non-Inf)
    for (out_data) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }

    // With non-uniform weights, output mean won't be zero.
    // Just verify the computation produces finite values (f16 path may have precision issues).
    // Note: The f16 typed path may produce different results than fast path.
}

test "layerNorm with bf16 weights" {
    // Test LayerNorm with bfloat16 weights
    var in_data = [_]f32{ 0.0, 1.0, 2.0, 3.0 };
    var out_data = [_]f32{ 0, 0, 0, 0 };
    var w_data = [_]u16{
        f32ToBf16(1.0),
        f32ToBf16(1.0),
        f32ToBf16(1.0),
        f32ToBf16(1.0),
    };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{4}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{4}, .f32);
    const weight = TensorView.initContiguous(@ptrCast(&w_data), &.{4}, .bf16);

    layerNorm(out, input, weight, null, 1e-6);

    // Verify outputs are valid
    for (out_data) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }

    // With uniform weights, normalized output should have mean close to 0
    // Note: bf16 precision limits may cause small deviations
    var sum: f32 = 0;
    for (out_data) |v| sum += v;
    const mean = sum / 4.0;
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mean, 0.5);
}

// ============================================================================
// Tests for Non-Contiguous / Strided Tensors
// ============================================================================

test "rmsNorm strided tensor - simple transpose" {
    // Create a 3x4 tensor and test on transposed view
    var in_data = [_]f32{
        1.0, 2.0,  3.0,  4.0,
        5.0, 6.0,  7.0,  8.0,
        9.0, 10.0, 11.0, 12.0,
    };
    var out_data = [_]f32{0} ** 12;
    var w_data = [_]f32{ 1.0, 1.0, 1.0 };

    // Create a strided view (transpose: 4x3 with non-contiguous layout)
    // Each "row" should be normalized along dimension of size 3
    var input = TensorView.initStrided(@ptrCast(&in_data), &.{ 4, 3 }, &.{ 1, 4 }, .f32);
    var out = TensorView.initStrided(@ptrCast(&out_data), &.{ 4, 3 }, &.{ 1, 4 }, .f32);
    const weight = TensorView.initContiguous(@ptrCast(&w_data), &.{3}, .f32);

    try std.testing.expect(!input.isContiguous());
    try std.testing.expect(!out.isContiguous());

    rmsNorm(out, input, weight, 1e-6);

    // Verify outputs are valid (tests strided path)
    for (out_data) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }
}

test "layerNorm strided tensor - non-contiguous weight" {
    var in_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var out_data = [_]f32{ 0, 0, 0, 0 };
    // Weight stored with stride (every other element)
    var w_storage = [_]f32{ 1.0, 999.0, 1.0, 999.0, 1.0, 999.0, 1.0, 999.0 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{4}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{4}, .f32);
    var weight = TensorView.initStrided(@ptrCast(&w_storage), &.{4}, &.{2}, .f32);

    try std.testing.expect(!weight.isContiguous());

    layerNorm(out, input, weight, null, 1e-6);

    // Verify normalization worked despite strided weight
    for (out_data) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }

    var sum: f32 = 0;
    for (out_data) |v| sum += v;
    const mean = sum / 4.0;
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mean, 1e-5);
}

test "rmsNorm strided output tensor" {
    var in_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var out_storage = [_]f32{ 0, 999, 0, 999, 0, 999, 0, 999 };
    var w_data = [_]f32{ 1.0, 1.0, 1.0, 1.0 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{4}, .f32);
    var out = TensorView.initStrided(@ptrCast(&out_storage), &.{4}, &.{2}, .f32);
    const weight = TensorView.initContiguous(@ptrCast(&w_data), &.{4}, .f32);

    try std.testing.expect(!out.isContiguous());

    rmsNorm(out, input, weight, 1e-6);

    // Check that output was written to correct strided positions
    try std.testing.expect(!std.math.isNan(out_storage[0]));
    try std.testing.expect(!std.math.isNan(out_storage[2]));
    try std.testing.expect(!std.math.isNan(out_storage[4]));
    try std.testing.expect(!std.math.isNan(out_storage[6]));

    // Sentinel values should be unchanged
    try std.testing.expectEqual(@as(f32, 999.0), out_storage[1]);
    try std.testing.expectEqual(@as(f32, 999.0), out_storage[3]);
}

// ============================================================================
// Tests for Edge Cases: Single Token and Large Batches
// ============================================================================

test "rmsNorm single token" {
    // Test with just one token (batch_size=1, seq_len=1)
    var in_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var out_data = [_]f32{ 0, 0, 0, 0 };
    var w_data = [_]f32{ 1.0, 1.0, 1.0, 1.0 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{ 1, 4 }, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{ 1, 4 }, .f32);
    const weight = TensorView.initContiguous(@ptrCast(&w_data), &.{4}, .f32);

    rmsNorm(out, input, weight, 1e-6);

    // mean(x^2) = (1 + 4 + 9 + 16) / 4 = 7.5
    const inv_rms = 1.0 / @sqrt(7.5 + 1e-6);

    try std.testing.expectApproxEqRel(1.0 * inv_rms, out_data[0], 1e-5);
    try std.testing.expectApproxEqRel(2.0 * inv_rms, out_data[1], 1e-5);
    try std.testing.expectApproxEqRel(3.0 * inv_rms, out_data[2], 1e-5);
    try std.testing.expectApproxEqRel(4.0 * inv_rms, out_data[3], 1e-5);
}

test "layerNorm single token" {
    // Test LayerNorm with single token
    var in_data = [_]f32{ 10.0, 20.0, 30.0, 40.0 };
    var out_data = [_]f32{ 0, 0, 0, 0 };
    var w_data = [_]f32{ 1.0, 1.0, 1.0, 1.0 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{4}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{4}, .f32);
    const weight = TensorView.initContiguous(@ptrCast(&w_data), &.{4}, .f32);

    layerNorm(out, input, weight, null, 1e-6);

    // Verify zero mean
    var sum: f32 = 0;
    for (out_data) |v| sum += v;
    const mean = sum / 4.0;
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mean, 1e-5);

    // Verify unit variance
    var var_sum: f32 = 0;
    for (out_data) |v| {
        const diff = v - mean;
        var_sum += diff * diff;
    }
    const variance = var_sum / 4.0;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), variance, 1e-5);
}

test "rmsNorm large batch" {
    const alloc = std.testing.allocator;

    // Test with a larger batch to verify batch processing
    const batch_size = 32;
    const hidden_size = 64;

    const in_data = try alloc.alloc(f32, batch_size * hidden_size);
    defer alloc.free(in_data);
    const out_data = try alloc.alloc(f32, batch_size * hidden_size);
    defer alloc.free(out_data);
    const w_data = try alloc.alloc(f32, hidden_size);
    defer alloc.free(w_data);

    // Initialize with varying values
    for (0..batch_size) |b| {
        for (0..hidden_size) |h| {
            const idx = b * hidden_size + h;
            in_data[idx] = @floatFromInt(h + 1);
        }
    }
    for (0..hidden_size) |h| {
        w_data[h] = 1.0;
    }

    const input = TensorView.initContiguous(@ptrCast(in_data.ptr), &.{ batch_size, hidden_size }, .f32);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{ batch_size, hidden_size }, .f32);
    const weight = TensorView.initContiguous(@ptrCast(w_data.ptr), &.{hidden_size}, .f32);

    rmsNorm(out, input, weight, 1e-6);

    // Verify all outputs are valid
    for (out_data) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }

    // All batches should have same normalization (same input pattern)
    // Check first and last batch are similar
    try std.testing.expectApproxEqRel(out_data[0], out_data[(batch_size - 1) * hidden_size], 1e-5);
}

test "layerNorm large batch" {
    const alloc = std.testing.allocator;

    const batch_size = 16;
    const seq_len = 8;
    const hidden_size = 32;
    const total_tokens = batch_size * seq_len;

    const in_data = try alloc.alloc(f32, total_tokens * hidden_size);
    defer alloc.free(in_data);
    const out_data = try alloc.alloc(f32, total_tokens * hidden_size);
    defer alloc.free(out_data);
    const w_data = try alloc.alloc(f32, hidden_size);
    defer alloc.free(w_data);

    // Initialize with different patterns per token
    for (0..total_tokens) |t| {
        for (0..hidden_size) |h| {
            const idx = t * hidden_size + h;
            in_data[idx] = @as(f32, @floatFromInt(h)) * 0.1 + @as(f32, @floatFromInt(t)) * 0.01;
        }
    }
    for (0..hidden_size) |h| {
        w_data[h] = 1.0;
    }

    const input = TensorView.initContiguous(@ptrCast(in_data.ptr), &.{ batch_size, seq_len, hidden_size }, .f32);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{ batch_size, seq_len, hidden_size }, .f32);
    const weight = TensorView.initContiguous(@ptrCast(w_data.ptr), &.{hidden_size}, .f32);

    layerNorm(out, input, weight, null, 1e-6);

    // Verify all outputs are valid and normalized
    for (out_data) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }

    // Check that each token is normalized independently
    // Sample a few tokens and verify zero mean
    for (0..3) |t| {
        const token_offset = t * hidden_size;
        var sum: f32 = 0;
        for (0..hidden_size) |h| {
            sum += out_data[token_offset + h];
        }
        const mean = sum / @as(f32, @floatFromInt(hidden_size));
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), mean, 1e-4);
    }
}

// ============================================================================
// Tests for Mixed Precision Input (f16, bf16)
// ============================================================================

test "rmsNorm with f16 input" {
    // Test RMSNorm with fp16 input tensors
    var in_data = [_]u16{
        f32ToFp16(1.0),
        f32ToFp16(2.0),
        f32ToFp16(3.0),
        f32ToFp16(4.0),
    };
    var out_data = [_]u16{ 0, 0, 0, 0 };
    var w_data = [_]u16{
        f32ToFp16(1.0),
        f32ToFp16(1.0),
        f32ToFp16(1.0),
        f32ToFp16(1.0),
    };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{4}, .f16);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{4}, .f16);
    const weight = TensorView.initContiguous(@ptrCast(&w_data), &.{4}, .f16);

    rmsNorm(out, input, weight, 1e-6);

    // Convert back to f32 and verify
    for (out_data) |v| {
        const val = fp16ToF32(v);
        try std.testing.expect(!std.math.isNan(val));
        try std.testing.expect(!std.math.isInf(val));
    }

    // Verify RMS normalization property (approximately)
    var sum_sq: f32 = 0;
    for (out_data) |v| {
        const val = fp16ToF32(v);
        sum_sq += val * val;
    }
    const rms = @sqrt(sum_sq / 4.0);
    // Should be close to 1.0 with fp16 precision
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), rms, 0.1);
}

test "rmsNorm with bf16 input" {
    // Test RMSNorm with bfloat16 input tensors
    var in_data = [_]u16{
        f32ToBf16(2.0),
        f32ToBf16(4.0),
        f32ToBf16(6.0),
        f32ToBf16(8.0),
    };
    var out_data = [_]u16{ 0, 0, 0, 0 };
    var w_data = [_]u16{
        f32ToBf16(1.0),
        f32ToBf16(1.0),
        f32ToBf16(1.0),
        f32ToBf16(1.0),
    };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{4}, .bf16);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{4}, .bf16);
    const weight = TensorView.initContiguous(@ptrCast(&w_data), &.{4}, .bf16);

    rmsNorm(out, input, weight, 1e-6);

    // Verify outputs are valid
    for (out_data) |v| {
        const val = bf16ToF32(v);
        try std.testing.expect(!std.math.isNan(val));
        try std.testing.expect(!std.math.isInf(val));
    }

    // Check proportions are maintained
    const v0 = bf16ToF32(out_data[0]);
    const v1 = bf16ToF32(out_data[1]);
    try std.testing.expectApproxEqRel(v1, v0 * 2.0, 0.1);
}

test "layerNorm with f16 input" {
    // Test LayerNorm with fp16 input
    var in_data = [_]u16{
        f32ToFp16(1.0),
        f32ToFp16(2.0),
        f32ToFp16(3.0),
        f32ToFp16(4.0),
    };
    var out_data = [_]u16{ 0, 0, 0, 0 };
    var w_data = [_]u16{
        f32ToFp16(1.0),
        f32ToFp16(1.0),
        f32ToFp16(1.0),
        f32ToFp16(1.0),
    };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{4}, .f16);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{4}, .f16);
    const weight = TensorView.initContiguous(@ptrCast(&w_data), &.{4}, .f16);

    layerNorm(out, input, weight, null, 1e-6);

    // Verify normalized output
    var sum: f32 = 0;
    for (out_data) |v| {
        const val = fp16ToF32(v);
        try std.testing.expect(!std.math.isNan(val));
        try std.testing.expect(!std.math.isInf(val));
        sum += val;
    }
    const mean = sum / 4.0;
    // Mean should be close to zero (with fp16 precision)
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mean, 0.1);
}

test "layerNorm with bf16 input" {
    // Test LayerNorm with bfloat16 input
    var in_data = [_]u16{
        f32ToBf16(10.0),
        f32ToBf16(20.0),
        f32ToBf16(30.0),
        f32ToBf16(40.0),
    };
    var out_data = [_]u16{ 0, 0, 0, 0 };
    var w_data = [_]u16{
        f32ToBf16(1.0),
        f32ToBf16(1.0),
        f32ToBf16(1.0),
        f32ToBf16(1.0),
    };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{4}, .bf16);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{4}, .bf16);
    const weight = TensorView.initContiguous(@ptrCast(&w_data), &.{4}, .bf16);

    layerNorm(out, input, weight, null, 1e-6);

    // Verify normalization
    var sum: f32 = 0;
    for (out_data) |v| {
        const val = bf16ToF32(v);
        try std.testing.expect(!std.math.isNan(val));
        sum += val;
    }
    const mean = sum / 4.0;
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mean, 0.1);
}

// ============================================================================
// Tests for Specific Optimized Paths (Contiguous vs Strided)
// ============================================================================

test "rmsNorm fast path vs slow path equivalence" {
    // Verify that contiguous and strided paths produce same results
    var in_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var out_contiguous = [_]f32{ 0, 0, 0, 0, 0, 0 };
    var out_strided = [_]f32{ 0, 999, 0, 999, 0, 999, 0, 999, 0, 999, 0, 999 };
    var w_data = [_]f32{ 1.0, 1.0, 1.0 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{ 2, 3 }, .f32);
    const weight = TensorView.initContiguous(@ptrCast(&w_data), &.{3}, .f32);

    // Fast path (contiguous)
    const out_fast = TensorView.initContiguous(@ptrCast(&out_contiguous), &.{ 2, 3 }, .f32);
    rmsNorm(out_fast, input, weight, 1e-6);

    // Slow path (strided output)
    // Shape {2, 3}: 2 rows of 3 elements. Stride {6, 2}: elements at indices 0,2,4 (row 0) and 6,8,10 (row 1)
    const out_slow = TensorView.initStrided(@ptrCast(&out_strided), &.{ 2, 3 }, &.{ 6, 2 }, .f32);
    rmsNorm(out_slow, input, weight, 1e-6);

    // Compare results (accounting for stride)
    try std.testing.expectApproxEqRel(out_contiguous[0], out_strided[0], 1e-5);
    try std.testing.expectApproxEqRel(out_contiguous[1], out_strided[2], 1e-5);
    try std.testing.expectApproxEqRel(out_contiguous[2], out_strided[4], 1e-5);
    try std.testing.expectApproxEqRel(out_contiguous[3], out_strided[6], 1e-5);
    try std.testing.expectApproxEqRel(out_contiguous[4], out_strided[8], 1e-5);
    try std.testing.expectApproxEqRel(out_contiguous[5], out_strided[10], 1e-5);
}

test "layerNorm fast path vs slow path equivalence" {
    // Verify that contiguous and strided paths produce same results
    var in_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var out_contiguous = [_]f32{ 0, 0, 0, 0 };
    var out_strided = [_]f32{ 0, 999, 0, 999, 0, 999, 0, 999 };
    var w_data = [_]f32{ 1.0, 1.0, 1.0, 1.0 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{4}, .f32);
    const weight = TensorView.initContiguous(@ptrCast(&w_data), &.{4}, .f32);

    // Fast path
    const out_fast = TensorView.initContiguous(@ptrCast(&out_contiguous), &.{4}, .f32);
    layerNorm(out_fast, input, weight, null, 1e-6);

    // Slow path
    const out_slow = TensorView.initStrided(@ptrCast(&out_strided), &.{4}, &.{2}, .f32);
    layerNorm(out_slow, input, weight, null, 1e-6);

    // Compare results
    for (0..4) |i| {
        try std.testing.expectApproxEqRel(out_contiguous[i], out_strided[i * 2], 1e-5);
    }
}

test "rmsNorm with very small hidden dimension" {
    // Test edge case with minimal hidden size
    var in_data = [_]f32{ 3.0, 4.0 }; // 2D vector
    var out_data = [_]f32{ 0, 0 };
    var w_data = [_]f32{ 1.0, 1.0 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{2}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{2}, .f32);
    const weight = TensorView.initContiguous(@ptrCast(&w_data), &.{2}, .f32);

    rmsNorm(out, input, weight, 1e-6);

    // 3-4-5 triangle: mean(x^2) = (9+16)/2 = 12.5
    const inv_rms = 1.0 / @sqrt(12.5 + 1e-6);
    try std.testing.expectApproxEqRel(3.0 * inv_rms, out_data[0], 1e-5);
    try std.testing.expectApproxEqRel(4.0 * inv_rms, out_data[1], 1e-5);
}

test "layerNorm with very small hidden dimension" {
    // Test with minimal dimension
    var in_data = [_]f32{ 1.0, 2.0 };
    var out_data = [_]f32{ 0, 0 };
    var w_data = [_]f32{ 1.0, 1.0 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{2}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{2}, .f32);
    const weight = TensorView.initContiguous(@ptrCast(&w_data), &.{2}, .f32);

    layerNorm(out, input, weight, null, 1e-6);

    // Mean should be 0, elements should be normalized
    const sum = out_data[0] + out_data[1];
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sum, 1e-5);

    // Check they're opposite signs with equal magnitude
    try std.testing.expectApproxEqRel(out_data[0], -out_data[1], 1e-5);
}
