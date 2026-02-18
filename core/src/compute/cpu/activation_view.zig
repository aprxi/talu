//! Activation functions with stride-aware, dtype-generic implementations.
//!
//! All ops work with TensorView and handle both contiguous and strided tensors.
//! Uses comptime generics to eliminate dtype dispatch repetition.

const std = @import("std");
const tv = @import("tensor_view.zig");
const math = @import("math_primitives/root.zig");

const TensorView = tv.TensorView;
const DType = tv.DType;

// Use existing SIMD infrastructure
const simd = math.simd;
const VEC_LEN = simd.f32_vec_len;
const F32Vec = simd.F32Vec;
const fastExp = math.fastExp;
const fastExpScalar = math.fastExpScalar;

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

/// Generic element-wise unary op that handles strides and dtype conversion.
/// Computes: out[i] = op(input[i]) for all elements
fn unaryOpGeneric(
    comptime T: type,
    comptime toF32: fn (T) f32,
    comptime fromF32: fn (f32) T,
    out: TensorView,
    input: TensorView,
    comptime op: fn (f32) f32,
) void {
    const out_data = @as([*]T, @ptrCast(@alignCast(out.data)));
    const in_data = @as([*]const T, @ptrCast(@alignCast(input.data)));

    if (input.isContiguous() and out.isContiguous()) {
        // Fast path: contiguous tensors
        for (0..input.numel) |elem_idx| {
            out_data[elem_idx] = fromF32(op(toF32(in_data[elem_idx])));
        }
    } else {
        // Optimized strided path: nested loops with incremental offsets.
        // Avoids expensive indexToCoords/coordsToOffset per element.
        unaryOpStrided(T, toF32, fromF32, out_data, in_data, input, out, op);
    }
}

/// Strided unary op using nested loops for efficient offset computation.
/// For an N-dimensional tensor, uses incremental offset calculation.
fn unaryOpStrided(
    comptime T: type,
    comptime toF32: fn (T) f32,
    comptime fromF32: fn (f32) T,
    out_data: [*]T,
    in_data: [*]const T,
    input: TensorView,
    out: TensorView,
    comptime op: fn (f32) f32,
) void {
    const ndim = input.ndim;

    // Handle common cases with unrolled loops for better performance
    switch (ndim) {
        1 => {
            const n0 = input.shape[0];
            const in_s0 = input.strides[0];
            const out_s0 = out.strides[0];
            var in_off: usize = 0;
            var out_off: usize = 0;
            for (0..n0) |_| {
                out_data[out_off] = fromF32(op(toF32(in_data[in_off])));
                in_off += in_s0;
                out_off += out_s0;
            }
        },
        2 => {
            const n0 = input.shape[0];
            const n1 = input.shape[1];
            const in_s0 = input.strides[0];
            const in_s1 = input.strides[1];
            const out_s0 = out.strides[0];
            const out_s1 = out.strides[1];
            var in_base0: usize = 0;
            var out_base0: usize = 0;
            for (0..n0) |_| {
                var in_off = in_base0;
                var out_off = out_base0;
                for (0..n1) |_| {
                    out_data[out_off] = fromF32(op(toF32(in_data[in_off])));
                    in_off += in_s1;
                    out_off += out_s1;
                }
                in_base0 += in_s0;
                out_base0 += out_s0;
            }
        },
        3 => {
            const n0 = input.shape[0];
            const n1 = input.shape[1];
            const n2 = input.shape[2];
            const in_s0 = input.strides[0];
            const in_s1 = input.strides[1];
            const in_s2 = input.strides[2];
            const out_s0 = out.strides[0];
            const out_s1 = out.strides[1];
            const out_s2 = out.strides[2];
            var in_base0: usize = 0;
            var out_base0: usize = 0;
            for (0..n0) |_| {
                var in_base1 = in_base0;
                var out_base1 = out_base0;
                for (0..n1) |_| {
                    var in_off = in_base1;
                    var out_off = out_base1;
                    for (0..n2) |_| {
                        out_data[out_off] = fromF32(op(toF32(in_data[in_off])));
                        in_off += in_s2;
                        out_off += out_s2;
                    }
                    in_base1 += in_s1;
                    out_base1 += out_s1;
                }
                in_base0 += in_s0;
                out_base0 += out_s0;
            }
        },
        else => {
            // General case: use coordinate tracking with incremental offsets
            var coords: [tv.MAX_NDIM]usize = [_]usize{0} ** tv.MAX_NDIM;
            var in_offset: usize = 0;
            var out_offset: usize = 0;

            for (0..input.numel) |_| {
                out_data[out_offset] = fromF32(op(toF32(in_data[in_offset])));

                // Increment coordinates and offsets (like a multi-digit counter)
                var dim_idx: usize = ndim;
                while (dim_idx > 0) {
                    dim_idx -= 1;
                    coords[dim_idx] += 1;
                    in_offset += input.strides[dim_idx];
                    out_offset += out.strides[dim_idx];

                    if (coords[dim_idx] < input.shape[dim_idx]) break;

                    // Carry: reset this dimension
                    const carried = input.shape[dim_idx];
                    coords[dim_idx] = 0;
                    in_offset -= carried * input.strides[dim_idx];
                    out_offset -= carried * out.strides[dim_idx];
                }
            }
        },
    }
}

// Identity conversions for f32
fn f32Identity(x: f32) f32 {
    return x;
}

/// SiLU activation: x * sigmoid(x)
pub fn silu(out: TensorView, input: TensorView) void {
    if (input.dtype == .f32 and out.dtype == .f32 and input.isContiguous() and out.isContiguous()) {
        math.siluContiguous(out.asSlice(f32), input.asSlice(f32));
        return;
    }

    const siluFn = struct {
        fn apply(x: f32) f32 {
            return x / (1.0 + fastExpScalar(-x));
        }
    }.apply;

    switch (input.dtype) {
        .f32 => unaryOpGeneric(f32, f32Identity, f32Identity, out, input, siluFn),
        .f16 => unaryOpGeneric(u16, fp16ToF32, f32ToFp16, out, input, siluFn),
        .bf16 => unaryOpGeneric(u16, bf16ToF32, f32ToBf16, out, input, siluFn),
        else => unreachable,
    }
}

/// GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu(out: TensorView, input: TensorView) void {
    if (input.dtype == .f32 and out.dtype == .f32 and input.isContiguous() and out.isContiguous()) {
        math.geluContiguous(out.asSlice(f32), input.asSlice(f32));
        return;
    }

    const geluFn = struct {
        fn apply(x: f32) f32 {
            const sqrt_2_over_pi: f32 = 0.7978845608028654;
            const coeff: f32 = 0.044715;
            const x3 = x * x * x;
            const inner = sqrt_2_over_pi * (x + coeff * x3);
            return 0.5 * x * (1.0 + std.math.tanh(inner));
        }
    }.apply;

    switch (input.dtype) {
        .f32 => unaryOpGeneric(f32, f32Identity, f32Identity, out, input, geluFn),
        .f16 => unaryOpGeneric(u16, fp16ToF32, f32ToFp16, out, input, geluFn),
        .bf16 => unaryOpGeneric(u16, bf16ToF32, f32ToBf16, out, input, geluFn),
        else => unreachable,
    }
}

/// ReLU activation: max(0, x)
pub fn relu(out: TensorView, input: TensorView) void {
    if (input.dtype == .f32 and out.dtype == .f32 and input.isContiguous() and out.isContiguous()) {
        math.reluContiguous(out.asSlice(f32), input.asSlice(f32));
        return;
    }

    const reluFn = struct {
        fn apply(x: f32) f32 {
            return @max(0, x);
        }
    }.apply;

    switch (input.dtype) {
        .f32 => unaryOpGeneric(f32, f32Identity, f32Identity, out, input, reluFn),
        .f16 => unaryOpGeneric(u16, fp16ToF32, f32ToFp16, out, input, reluFn),
        .bf16 => unaryOpGeneric(u16, bf16ToF32, f32ToBf16, out, input, reluFn),
        else => unreachable,
    }
}

/// Sigmoid activation: 1 / (1 + exp(-x))
pub fn sigmoid(out: TensorView, input: TensorView) void {
    if (input.dtype == .f32 and out.dtype == .f32 and input.isContiguous() and out.isContiguous()) {
        math.sigmoidContiguous(out.asSlice(f32), input.asSlice(f32));
        return;
    }

    const sigmoidFn = struct {
        fn apply(x: f32) f32 {
            return 1.0 / (1.0 + fastExpScalar(-x));
        }
    }.apply;

    switch (input.dtype) {
        .f32 => unaryOpGeneric(f32, f32Identity, f32Identity, out, input, sigmoidFn),
        .f16 => unaryOpGeneric(u16, fp16ToF32, f32ToFp16, out, input, sigmoidFn),
        .bf16 => unaryOpGeneric(u16, bf16ToF32, f32ToBf16, out, input, sigmoidFn),
        else => unreachable,
    }
}

/// Tanh activation
pub fn tanh(out: TensorView, input: TensorView) void {
    if (input.dtype == .f32 and out.dtype == .f32 and input.isContiguous() and out.isContiguous()) {
        math.tanhContiguous(out.asSlice(f32), input.asSlice(f32));
        return;
    }

    const tanhFn = struct {
        fn apply(x: f32) f32 {
            return std.math.tanh(x);
        }
    }.apply;

    switch (input.dtype) {
        .f32 => unaryOpGeneric(f32, f32Identity, f32Identity, out, input, tanhFn),
        .f16 => unaryOpGeneric(u16, fp16ToF32, f32ToFp16, out, input, tanhFn),
        .bf16 => unaryOpGeneric(u16, bf16ToF32, f32ToBf16, out, input, tanhFn),
        else => unreachable,
    }
}

/// Softmax over specified dimension (PyTorch-compatible)
/// dim: dimension to apply softmax over (supports negative indexing)
pub fn softmaxDim(out: TensorView, input: TensorView, dim: i32) void {
    switch (input.dtype) {
        .f32 => softmaxDimTyped(f32, f32Identity, f32Identity, out, input, dim),
        .f16 => softmaxDimTyped(u16, fp16ToF32, f32ToFp16, out, input, dim),
        .bf16 => softmaxDimTyped(u16, bf16ToF32, f32ToBf16, out, input, dim),
        else => unreachable,
    }
}

/// Softmax over last dimension (convenience wrapper)
pub fn softmax(out: TensorView, input: TensorView) void {
    softmaxDim(out, input, -1);
}

fn softmaxDimTyped(
    comptime T: type,
    comptime toF32: fn (T) f32,
    comptime fromF32: fn (f32) T,
    out: TensorView,
    input: TensorView,
    dim_arg: i32,
) void {
    std.debug.assert(input.ndim >= 1);
    const out_data = @as([*]T, @ptrCast(@alignCast(out.data)));
    const in_data = @as([*]const T, @ptrCast(@alignCast(input.data)));

    // Handle negative dimension (PyTorch convention)
    const ndim_i32: i32 = @intCast(input.ndim);
    const dim: usize = @intCast(if (dim_arg < 0) dim_arg + ndim_i32 else dim_arg);
    std.debug.assert(dim < input.ndim);

    const dim_size = input.shape[dim];

    // Fast path: if dim is last dimension and contiguous
    if (dim == input.ndim - 1 and input.isContiguous() and out.isContiguous()) {
        const outer_size = input.numel / dim_size;
        if (input.dtype == .f32 and out.dtype == .f32) {
            math.softmaxContiguous(out.asSlice(f32), input.asSlice(f32), outer_size, dim_size);
            return;
        }

        for (0..outer_size) |outer_idx| {
            const base_offset = outer_idx * dim_size;

            // Find max
            var max_val: f32 = -std.math.inf(f32);
            for (0..dim_size) |dim_idx| {
                max_val = @max(max_val, toF32(in_data[base_offset + dim_idx]));
            }

            // Exp and sum
            var sum: f32 = 0;
            for (0..dim_size) |dim_idx| {
                const exp_val = fastExpScalar(toF32(in_data[base_offset + dim_idx]) - max_val);
                out_data[base_offset + dim_idx] = fromF32(exp_val);
                sum += exp_val;
            }

            // Normalize
            const inv_sum = 1.0 / sum;
            for (0..dim_size) |dim_idx| {
                out_data[base_offset + dim_idx] = fromF32(toF32(out_data[base_offset + dim_idx]) * inv_sum);
            }
        }
    } else {
        // General path: strided access for any dimension
        // Compute outer_size = product of all dims except dim
        var outer_size: usize = 1;
        for (0..input.ndim) |d| {
            if (d != dim) outer_size *= input.shape[d];
        }

        var coords: [tv.MAX_NDIM]usize = [_]usize{0} ** tv.MAX_NDIM;
        for (0..outer_size) |outer_idx| {
            // Convert outer index to coords, skipping the softmax dimension
            var remaining = outer_idx;
            var dim_idx: usize = input.ndim - 1;
            while (true) {
                if (dim_idx != dim) {
                    coords[dim_idx] = remaining % input.shape[dim_idx];
                    remaining /= input.shape[dim_idx];
                }
                if (dim_idx == 0) break;
                dim_idx -= 1;
            }

            // Find max along dim
            var max_val: f32 = -std.math.inf(f32);
            for (0..dim_size) |softmax_idx| {
                coords[dim] = softmax_idx;
                const in_off = input.coordsToOffset(coords[0..input.ndim]);
                max_val = @max(max_val, toF32(in_data[in_off]));
            }

            // Exp, sum, and store
            var sum: f32 = 0;
            for (0..dim_size) |softmax_idx| {
                coords[dim] = softmax_idx;
                const in_off = input.coordsToOffset(coords[0..input.ndim]);
                const out_off = out.coordsToOffset(coords[0..out.ndim]);
                const exp_val = fastExpScalar(toF32(in_data[in_off]) - max_val);
                out_data[out_off] = fromF32(exp_val);
                sum += exp_val;
            }

            // Normalize
            const inv_sum = 1.0 / sum;
            for (0..dim_size) |softmax_idx| {
                coords[dim] = softmax_idx;
                const out_off = out.coordsToOffset(coords[0..out.ndim]);
                out_data[out_off] = fromF32(toF32(out_data[out_off]) * inv_sum);
            }
        }
    }
}

/// Reciprocal square root: 1 / sqrt(x)
pub fn rsqrt(out: TensorView, input: TensorView) void {
    const rsqrtFn = struct {
        fn apply(x: f32) f32 {
            return 1.0 / @sqrt(x);
        }
    }.apply;

    switch (input.dtype) {
        .f32 => unaryOpGeneric(f32, f32Identity, f32Identity, out, input, rsqrtFn),
        .f16 => unaryOpGeneric(u16, fp16ToF32, f32ToFp16, out, input, rsqrtFn),
        .bf16 => unaryOpGeneric(u16, bf16ToF32, f32ToBf16, out, input, rsqrtFn),
        else => unreachable,
    }
}

test "silu contiguous" {
    var in_data = [_]f32{ 0, 1, -1, 2 };
    var out_data = [_]f32{ 0, 0, 0, 0 };

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{4}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{4}, .f32);

    silu(out, input);

    // silu(0) = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0), out_data[0], 1e-5);
    // silu(1) ≈ 0.731
    try std.testing.expectApproxEqAbs(@as(f32, 0.731), out_data[1], 1e-2);
}

// ============================================================================
// Additional unit tests
// ============================================================================

// Test SiLU correctness: verifies silu(x) = x * sigmoid(x) identity
// for a range of known input values
test "silu correctness - verify identity" {
    var in_data = [_]f32{ -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0 };
    var out_data = [_]f32{0} ** 8;

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{8}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{8}, .f32);

    silu(out, input);

    // Verify silu(x) = x * sigmoid(x) for each value
    for (in_data, 0..) |x, i| {
        const sigmoid_x = 1.0 / (1.0 + @exp(-x));
        const expected = x * sigmoid_x;
        try std.testing.expectApproxEqRel(expected, out_data[i], 1e-5);
    }
}

// Test SiLU with edge cases: zero, very negative, very positive values
test "silu edge cases" {
    var in_data = [_]f32{ 0.0, -10.0, 10.0, -100.0, 100.0 };
    var out_data = [_]f32{0} ** 5;

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{5}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{5}, .f32);

    silu(out, input);

    // silu(0) = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out_data[0], 1e-6);

    // silu(-10) ≈ -0.000454 (very small negative)
    try std.testing.expect(out_data[1] < 0.0 and out_data[1] > -0.001);

    // silu(10) ≈ 9.99954 (approaches x for large positive)
    try std.testing.expectApproxEqRel(@as(f32, 10.0), out_data[2], 1e-3);

    // silu(-100) ≈ 0 (saturates to 0 for very negative)
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out_data[3], 1e-6);

    // silu(100) ≈ 100 (approaches x for very large positive)
    try std.testing.expectApproxEqRel(@as(f32, 100.0), out_data[4], 1e-3);
}

// Test ReLU correctness: positive values pass through, negative values clamp to 0
test "relu correctness" {
    var in_data = [_]f32{ -5.0, -2.5, -0.1, 0.0, 0.1, 2.5, 5.0 };
    var out_data = [_]f32{0} ** 7;

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{7}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{7}, .f32);

    relu(out, input);

    // All negative values should be 0
    try std.testing.expectEqual(@as(f32, 0.0), out_data[0]);
    try std.testing.expectEqual(@as(f32, 0.0), out_data[1]);
    try std.testing.expectEqual(@as(f32, 0.0), out_data[2]);

    // Zero stays zero
    try std.testing.expectEqual(@as(f32, 0.0), out_data[3]);

    // Positive values pass through unchanged
    try std.testing.expectEqual(@as(f32, 0.1), out_data[4]);
    try std.testing.expectEqual(@as(f32, 2.5), out_data[5]);
    try std.testing.expectEqual(@as(f32, 5.0), out_data[6]);
}

// Test ReLU with edge cases: very large positive/negative values
test "relu edge cases" {
    var in_data = [_]f32{ -1e10, -1e6, 1e6, 1e10 };
    var out_data = [_]f32{0} ** 4;

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{4}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{4}, .f32);

    relu(out, input);

    // Very large negatives clamp to 0
    try std.testing.expectEqual(@as(f32, 0.0), out_data[0]);
    try std.testing.expectEqual(@as(f32, 0.0), out_data[1]);

    // Very large positives pass through
    try std.testing.expectEqual(@as(f32, 1e6), out_data[2]);
    try std.testing.expectEqual(@as(f32, 1e10), out_data[3]);
}

// Test GELU correctness: verifies approximate values for known inputs
// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
test "gelu correctness" {
    var in_data = [_]f32{ -3.0, -1.0, 0.0, 1.0, 3.0 };
    var out_data = [_]f32{0} ** 5;

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{5}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{5}, .f32);

    gelu(out, input);

    // Known approximate GELU values
    // gelu(-3.0) ≈ -0.0036
    try std.testing.expectApproxEqAbs(@as(f32, -0.0036), out_data[0], 1e-3);

    // gelu(-1.0) ≈ -0.1588
    try std.testing.expectApproxEqAbs(@as(f32, -0.1588), out_data[1], 1e-3);

    // gelu(0.0) = 0.0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out_data[2], 1e-6);

    // gelu(1.0) ≈ 0.8412
    try std.testing.expectApproxEqAbs(@as(f32, 0.8412), out_data[3], 1e-3);

    // gelu(3.0) ≈ 2.9964
    try std.testing.expectApproxEqAbs(@as(f32, 2.9964), out_data[4], 1e-3);
}

// Test GELU edge cases and symmetry properties
test "gelu edge cases" {
    var in_data = [_]f32{ -10.0, -5.0, 5.0, 10.0 };
    var out_data = [_]f32{0} ** 4;

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{4}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{4}, .f32);

    gelu(out, input);

    // gelu(-10) ≈ 0 (saturates to 0 for large negative)
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out_data[0], 1e-5);

    // gelu(-5) ≈ 0 (near 0 for moderate negative)
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out_data[1], 1e-4);

    // gelu(5) ≈ 5 (approaches x for large positive)
    try std.testing.expectApproxEqRel(@as(f32, 5.0), out_data[2], 1e-4);

    // gelu(10) ≈ 10 (approaches x for very large positive)
    try std.testing.expectApproxEqRel(@as(f32, 10.0), out_data[3], 1e-4);
}

// Test strided tensor handling for SiLU: tests non-contiguous memory layout
// Creates a transposed 2x3 tensor to verify stride-aware computation
test "silu strided tensor" {
    const allocator = std.testing.allocator;

    // Create a 2x3 tensor, then transpose to get non-contiguous layout
    var in_data = [_]f32{ 0.0, 1.0, 2.0, -1.0, -2.0, -3.0 };
    const out_data = try allocator.alloc(f32, 6);
    defer allocator.free(out_data);
    @memset(out_data, 0);

    // Transposed shape: [3, 2] with swapped strides (non-contiguous)
    const input_t = TensorView.initStrided(
        @ptrCast(&in_data),
        &.{ 3, 2 },
        &.{ 1, 3 }, // column-major strides (non-contiguous)
        .f32,
    );

    const out = TensorView.initStrided(
        @ptrCast(out_data.ptr),
        &.{ 3, 2 },
        &.{ 2, 1 }, // contiguous output
        .f32,
    );

    // Test non-contiguous input
    silu(out, input_t);

    // Verify correctness: the strided access should correctly compute SiLU
    // input_t logical layout: [[0, -1], [1, -2], [2, -3]]
    // Check a few values
    const expected_0_0 = 0.0 * (1.0 / (1.0 + @exp(-0.0)));
    const expected_0_1 = -1.0 * (1.0 / (1.0 + @exp(1.0)));
    const expected_1_0 = 1.0 * (1.0 / (1.0 + @exp(-1.0)));

    try std.testing.expectApproxEqAbs(expected_0_0, out_data[0], 1e-5);
    try std.testing.expectApproxEqAbs(expected_0_1, out_data[1], 1e-5);
    try std.testing.expectApproxEqAbs(expected_1_0, out_data[2], 1e-5);
}

// Test strided tensor handling for ReLU: verifies non-contiguous memory works
test "relu strided tensor" {
    const allocator = std.testing.allocator;

    var in_data = [_]f32{ -1.0, 2.0, -3.0, 4.0, -5.0, 6.0 };
    const out_data = try allocator.alloc(f32, 6);
    defer allocator.free(out_data);
    @memset(out_data, 99.0); // Initialize with sentinel value

    // Create transposed/strided tensor [3, 2]
    const input_t = TensorView.initStrided(
        @ptrCast(&in_data),
        &.{ 3, 2 },
        &.{ 1, 3 }, // non-contiguous column-major strides
        .f32,
    );

    const out = TensorView.initStrided(
        @ptrCast(out_data.ptr),
        &.{ 3, 2 },
        &.{ 2, 1 }, // contiguous output
        .f32,
    );

    relu(out, input_t);

    // Logical input layout: [[-1, 4], [2, -5], [-3, 6]]
    // Expected output: [[0, 4], [2, 0], [0, 6]]
    try std.testing.expectEqual(@as(f32, 0.0), out_data[0]);
    try std.testing.expectEqual(@as(f32, 4.0), out_data[1]);
    try std.testing.expectEqual(@as(f32, 2.0), out_data[2]);
    try std.testing.expectEqual(@as(f32, 0.0), out_data[3]);
    try std.testing.expectEqual(@as(f32, 0.0), out_data[4]);
    try std.testing.expectEqual(@as(f32, 6.0), out_data[5]);
}

// Test GELU with strided tensor
test "gelu strided tensor" {
    const allocator = std.testing.allocator;

    var in_data = [_]f32{ -1.0, 0.0, 1.0, 2.0 };
    const out_data = try allocator.alloc(f32, 4);
    defer allocator.free(out_data);
    @memset(out_data, 0);

    // Create 2x2 strided tensor
    const input = TensorView.initStrided(
        @ptrCast(&in_data),
        &.{ 2, 2 },
        &.{ 1, 2 }, // strided access
        .f32,
    );

    const out = TensorView.initStrided(
        @ptrCast(out_data.ptr),
        &.{ 2, 2 },
        &.{ 2, 1 }, // contiguous
        .f32,
    );

    gelu(out, input);

    // Verify values are computed correctly despite stride
    // Input logical layout: [[-1, 1], [0, 2]]
    try std.testing.expectApproxEqAbs(@as(f32, -0.1588), out_data[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8412), out_data[1], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out_data[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.9546), out_data[3], 1e-3);
}

// Test sigmoid correctness: verifies 1/(1+exp(-x)) for known values
test "sigmoid correctness" {
    var in_data = [_]f32{ -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0 };
    var out_data = [_]f32{0} ** 7;

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{7}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{7}, .f32);

    sigmoid(out, input);

    // Known sigmoid values
    try std.testing.expectApproxEqAbs(@as(f32, 0.0067), out_data[0], 1e-4); // sigmoid(-5)
    try std.testing.expectApproxEqAbs(@as(f32, 0.1192), out_data[1], 1e-4); // sigmoid(-2)
    try std.testing.expectApproxEqAbs(@as(f32, 0.2689), out_data[2], 1e-4); // sigmoid(-1)
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out_data[3], 1e-6); // sigmoid(0)
    try std.testing.expectApproxEqAbs(@as(f32, 0.7311), out_data[4], 1e-4); // sigmoid(1)
    try std.testing.expectApproxEqAbs(@as(f32, 0.8808), out_data[5], 1e-4); // sigmoid(2)
    try std.testing.expectApproxEqAbs(@as(f32, 0.9933), out_data[6], 1e-4); // sigmoid(5)
}

// Test tanh correctness and range [-1, 1]
test "tanh correctness" {
    var in_data = [_]f32{ -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0 };
    var out_data = [_]f32{0} ** 7;

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{7}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{7}, .f32);

    tanh(out, input);

    // Verify output range [-1, 1]
    for (out_data) |val| {
        try std.testing.expect(val >= -1.0 and val <= 1.0);
    }

    // Known tanh values
    try std.testing.expectApproxEqAbs(@as(f32, -0.9999), out_data[0], 1e-4); // tanh(-5)
    try std.testing.expectApproxEqAbs(@as(f32, -0.9640), out_data[1], 1e-4); // tanh(-2)
    try std.testing.expectApproxEqAbs(@as(f32, -0.7616), out_data[2], 1e-4); // tanh(-1)
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out_data[3], 1e-6); // tanh(0)
    try std.testing.expectApproxEqAbs(@as(f32, 0.7616), out_data[4], 1e-4); // tanh(1)
    try std.testing.expectApproxEqAbs(@as(f32, 0.9640), out_data[5], 1e-4); // tanh(2)
    try std.testing.expectApproxEqAbs(@as(f32, 0.9999), out_data[6], 1e-4); // tanh(5)
}
// Test multi-dtype support for SiLU: f32, f16, bf16
test "silu multi-dtype" {
    const allocator = std.testing.allocator;

    var in_data_f32 = [_]f32{ 0.0, 1.0, -1.0, 2.0 };
    var out_data_f32 = [_]f32{0} ** 4;

    // f16 test
    const in_data_f16 = try allocator.alloc(u16, 4);
    defer allocator.free(in_data_f16);
    const out_data_f16 = try allocator.alloc(u16, 4);
    defer allocator.free(out_data_f16);

    for (in_data_f32, 0..) |val, i| {
        in_data_f16[i] = f32ToFp16(val);
    }

    // bf16 test
    const in_data_bf16 = try allocator.alloc(u16, 4);
    defer allocator.free(in_data_bf16);
    const out_data_bf16 = try allocator.alloc(u16, 4);
    defer allocator.free(out_data_bf16);

    for (in_data_f32, 0..) |val, i| {
        in_data_bf16[i] = f32ToBf16(val);
    }

    // Test f32
    const input_f32 = TensorView.initContiguous(@ptrCast(&in_data_f32), &.{4}, .f32);
    const out_f32 = TensorView.initContiguous(@ptrCast(&out_data_f32), &.{4}, .f32);
    silu(out_f32, input_f32);

    // Test f16
    const input_f16 = TensorView.initContiguous(@ptrCast(in_data_f16.ptr), &.{4}, .f16);
    const out_f16 = TensorView.initContiguous(@ptrCast(out_data_f16.ptr), &.{4}, .f16);
    silu(out_f16, input_f16);

    // Test bf16
    const input_bf16 = TensorView.initContiguous(@ptrCast(in_data_bf16.ptr), &.{4}, .bf16);
    const out_bf16 = TensorView.initContiguous(@ptrCast(out_data_bf16.ptr), &.{4}, .bf16);
    silu(out_bf16, input_bf16);

    // Verify f16 results match f32 within tolerance (f16 has ~3 decimal digits precision)
    for (0..4) |i| {
        const f16_result = fp16ToF32(out_data_f16[i]);
        try std.testing.expectApproxEqRel(out_data_f32[i], f16_result, 1e-2);
    }

    // Verify bf16 results match f32 within tolerance (bf16 has ~2-3 decimal digits precision)
    for (0..4) |i| {
        const bf16_result = bf16ToF32(out_data_bf16[i]);
        try std.testing.expectApproxEqRel(out_data_f32[i], bf16_result, 1e-2);
    }
}

// Test multi-dtype support for ReLU
test "relu multi-dtype" {
    const allocator = std.testing.allocator;

    const in_data_f32 = [_]f32{ -2.0, -0.5, 0.0, 0.5, 2.0 };

    // f16 test
    const in_data_f16 = try allocator.alloc(u16, 5);
    defer allocator.free(in_data_f16);
    const out_data_f16 = try allocator.alloc(u16, 5);
    defer allocator.free(out_data_f16);

    for (in_data_f32, 0..) |val, i| {
        in_data_f16[i] = f32ToFp16(val);
    }

    const input_f16 = TensorView.initContiguous(@ptrCast(in_data_f16.ptr), &.{5}, .f16);
    const out_f16 = TensorView.initContiguous(@ptrCast(out_data_f16.ptr), &.{5}, .f16);
    relu(out_f16, input_f16);

    // Verify ReLU behavior with f16
    try std.testing.expectEqual(@as(f32, 0.0), fp16ToF32(out_data_f16[0])); // -2 -> 0
    try std.testing.expectEqual(@as(f32, 0.0), fp16ToF32(out_data_f16[1])); // -0.5 -> 0
    try std.testing.expectEqual(@as(f32, 0.0), fp16ToF32(out_data_f16[2])); // 0 -> 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), fp16ToF32(out_data_f16[3]), 1e-3); // 0.5 -> 0.5
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), fp16ToF32(out_data_f16[4]), 1e-3); // 2 -> 2
}

// Test rsqrt correctness: verifies 1/sqrt(x) for known values
test "rsqrt correctness" {
    var in_data = [_]f32{ 1.0, 4.0, 9.0, 16.0, 25.0 };
    var out_data = [_]f32{0} ** 5;

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{5}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{5}, .f32);

    rsqrt(out, input);

    // Verify 1/sqrt(x) values
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out_data[0], 1e-6); // 1/sqrt(1) = 1
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out_data[1], 1e-6); // 1/sqrt(4) = 0.5
    try std.testing.expectApproxEqAbs(@as(f32, 0.3333), out_data[2], 1e-4); // 1/sqrt(9) = 0.333...
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), out_data[3], 1e-6); // 1/sqrt(16) = 0.25
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), out_data[4], 1e-6); // 1/sqrt(25) = 0.2
}

// Test numerical accuracy of activations against reference implementations
// This test verifies that our fast approximations are accurate enough
test "silu gelu relu sigmoid tanh numerical accuracy" {
    var test_values = [_]f32{ -3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0 };
    var silu_out = [_]f32{0} ** 7;
    var gelu_out = [_]f32{0} ** 7;
    var sigmoid_out = [_]f32{0} ** 7;

    const input = TensorView.initContiguous(@ptrCast(&test_values), &.{7}, .f32);
    const silu_tensor = TensorView.initContiguous(@ptrCast(&silu_out), &.{7}, .f32);
    const gelu_tensor = TensorView.initContiguous(@ptrCast(&gelu_out), &.{7}, .f32);
    const sigmoid_tensor = TensorView.initContiguous(@ptrCast(&sigmoid_out), &.{7}, .f32);

    silu(silu_tensor, input);
    gelu(gelu_tensor, input);
    sigmoid(sigmoid_tensor, input);

    // Reference implementations using std.math
    for (test_values, 0..) |x, i| {
        // Reference SiLU: x / (1 + exp(-x))
        const ref_sigmoid = 1.0 / (1.0 + @exp(-x));
        const ref_silu = x * ref_sigmoid;
        try std.testing.expectApproxEqRel(ref_silu, silu_out[i], 1e-5);

        // Reference sigmoid
        try std.testing.expectApproxEqRel(ref_sigmoid, sigmoid_out[i], 1e-5);

        // Reference GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const sqrt_2_over_pi: f32 = 0.7978845608028654;
        const coeff: f32 = 0.044715;
        const x3 = x * x * x;
        const inner = sqrt_2_over_pi * (x + coeff * x3);
        const ref_gelu = 0.5 * x * (1.0 + std.math.tanh(inner));
        try std.testing.expectApproxEqRel(ref_gelu, gelu_out[i], 1e-5);
    }
}

// Test sigmoid edge cases: extreme values and saturation
test "sigmoid edge cases" {
    var in_data = [_]f32{ -100.0, -50.0, -10.0, 10.0, 50.0, 100.0 };
    var out_data = [_]f32{0} ** 6;

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{6}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{6}, .f32);

    sigmoid(out, input);

    // Very negative values saturate to 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out_data[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out_data[1], 1e-6);
    try std.testing.expect(out_data[2] < 0.0001);

    // Very positive values saturate to 1
    try std.testing.expect(out_data[3] > 0.9999);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out_data[4], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out_data[5], 1e-6);

    // All outputs are in [0, 1]
    for (out_data) |val| {
        try std.testing.expect(val >= 0.0 and val <= 1.0);
    }
}

// Test sigmoid with strided tensor
test "sigmoid strided tensor" {
    const allocator = std.testing.allocator;

    var in_data = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0, 3.0 };
    const out_data = try allocator.alloc(f32, 6);
    defer allocator.free(out_data);
    @memset(out_data, 0);

    // Create 2x3 strided tensor
    const input = TensorView.initStrided(
        @ptrCast(&in_data),
        &.{ 2, 3 },
        &.{ 1, 2 }, // strided access
        .f32,
    );

    const out = TensorView.initStrided(
        @ptrCast(out_data.ptr),
        &.{ 2, 3 },
        &.{ 3, 1 }, // contiguous
        .f32,
    );

    sigmoid(out, input);

    // Verify correctness with strided access
    // Input logical layout: [[-2, 0, 2], [-1, 1, 3]]
    try std.testing.expectApproxEqAbs(@as(f32, 0.1192), out_data[0], 1e-4); // sigmoid(-2)
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out_data[1], 1e-6); // sigmoid(0)
    try std.testing.expectApproxEqAbs(@as(f32, 0.8808), out_data[2], 1e-4); // sigmoid(2)
    try std.testing.expectApproxEqAbs(@as(f32, 0.2689), out_data[3], 1e-4); // sigmoid(-1)
    try std.testing.expectApproxEqAbs(@as(f32, 0.7311), out_data[4], 1e-4); // sigmoid(1)
    try std.testing.expectApproxEqAbs(@as(f32, 0.9526), out_data[5], 1e-4); // sigmoid(3)
}

// Test tanh edge cases: extreme values and saturation
test "tanh edge cases" {
    var in_data = [_]f32{ -100.0, -50.0, -10.0, 10.0, 50.0, 100.0 };
    var out_data = [_]f32{0} ** 6;

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{6}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{6}, .f32);

    tanh(out, input);

    // Very negative values saturate to -1
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), out_data[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), out_data[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), out_data[2], 1e-6);

    // Very positive values saturate to 1
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out_data[3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out_data[4], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out_data[5], 1e-6);
}

// Test tanh with strided tensor
test "tanh strided tensor" {
    const allocator = std.testing.allocator;

    var in_data = [_]f32{ -1.0, 0.0, 1.0, 2.0 };
    const out_data = try allocator.alloc(f32, 4);
    defer allocator.free(out_data);
    @memset(out_data, 0);

    // Create 2x2 strided tensor
    const input = TensorView.initStrided(
        @ptrCast(&in_data),
        &.{ 2, 2 },
        &.{ 1, 2 }, // strided access
        .f32,
    );

    const out = TensorView.initStrided(
        @ptrCast(out_data.ptr),
        &.{ 2, 2 },
        &.{ 2, 1 }, // contiguous
        .f32,
    );

    tanh(out, input);

    // Verify correctness with strided access
    // Input logical layout: [[-1, 1], [0, 2]]
    try std.testing.expectApproxEqAbs(@as(f32, -0.7616), out_data[0], 1e-4); // tanh(-1)
    try std.testing.expectApproxEqAbs(@as(f32, 0.7616), out_data[1], 1e-4); // tanh(1)
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out_data[2], 1e-6); // tanh(0)
    try std.testing.expectApproxEqAbs(@as(f32, 0.9640), out_data[3], 1e-4); // tanh(2)
}

// Test rsqrt edge cases: very small and very large values
test "rsqrt edge cases" {
    var in_data = [_]f32{ 0.0001, 0.01, 100.0, 10000.0 };
    var out_data = [_]f32{0} ** 4;

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{4}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{4}, .f32);

    rsqrt(out, input);

    // Verify 1/sqrt(x) for edge cases
    try std.testing.expectApproxEqAbs(@as(f32, 100.0), out_data[0], 1e-3); // 1/sqrt(0.0001) = 100
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), out_data[1], 1e-3); // 1/sqrt(0.01) = 10
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), out_data[2], 1e-6); // 1/sqrt(100) = 0.1
    try std.testing.expectApproxEqAbs(@as(f32, 0.01), out_data[3], 1e-6); // 1/sqrt(10000) = 0.01
}

// Test rsqrt with strided tensor
test "rsqrt strided tensor" {
    const allocator = std.testing.allocator;

    var in_data = [_]f32{ 1.0, 4.0, 9.0, 16.0 };
    const out_data = try allocator.alloc(f32, 4);
    defer allocator.free(out_data);
    @memset(out_data, 0);

    // Create 2x2 strided tensor
    const input = TensorView.initStrided(
        @ptrCast(&in_data),
        &.{ 2, 2 },
        &.{ 1, 2 }, // strided access
        .f32,
    );

    const out = TensorView.initStrided(
        @ptrCast(out_data.ptr),
        &.{ 2, 2 },
        &.{ 2, 1 }, // contiguous
        .f32,
    );

    rsqrt(out, input);

    // Verify correctness with strided access
    // Input logical layout: [[1, 9], [4, 16]]
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out_data[0], 1e-6); // 1/sqrt(1) = 1
    try std.testing.expectApproxEqAbs(@as(f32, 0.3333), out_data[1], 1e-4); // 1/sqrt(9) = 0.333...
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out_data[2], 1e-6); // 1/sqrt(4) = 0.5
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), out_data[3], 1e-6); // 1/sqrt(16) = 0.25
}

// Test multi-dtype support for GELU
test "gelu multi-dtype" {
    const allocator = std.testing.allocator;

    var in_data_f32 = [_]f32{ -1.0, 0.0, 1.0, 2.0 };
    var out_data_f32 = [_]f32{0} ** 4;

    // f16 test
    const in_data_f16 = try allocator.alloc(u16, 4);
    defer allocator.free(in_data_f16);
    const out_data_f16 = try allocator.alloc(u16, 4);
    defer allocator.free(out_data_f16);

    for (in_data_f32, 0..) |val, i| {
        in_data_f16[i] = f32ToFp16(val);
    }

    // bf16 test
    const in_data_bf16 = try allocator.alloc(u16, 4);
    defer allocator.free(in_data_bf16);
    const out_data_bf16 = try allocator.alloc(u16, 4);
    defer allocator.free(out_data_bf16);

    for (in_data_f32, 0..) |val, i| {
        in_data_bf16[i] = f32ToBf16(val);
    }

    // Test f32
    const input_f32 = TensorView.initContiguous(@ptrCast(&in_data_f32), &.{4}, .f32);
    const out_f32 = TensorView.initContiguous(@ptrCast(&out_data_f32), &.{4}, .f32);
    gelu(out_f32, input_f32);

    // Test f16
    const input_f16 = TensorView.initContiguous(@ptrCast(in_data_f16.ptr), &.{4}, .f16);
    const out_f16 = TensorView.initContiguous(@ptrCast(out_data_f16.ptr), &.{4}, .f16);
    gelu(out_f16, input_f16);

    // Test bf16
    const input_bf16 = TensorView.initContiguous(@ptrCast(in_data_bf16.ptr), &.{4}, .bf16);
    const out_bf16 = TensorView.initContiguous(@ptrCast(out_data_bf16.ptr), &.{4}, .bf16);
    gelu(out_bf16, input_bf16);

    // Verify f16 results match f32 within tolerance
    for (0..4) |i| {
        const f16_result = fp16ToF32(out_data_f16[i]);
        try std.testing.expectApproxEqRel(out_data_f32[i], f16_result, 5e-2);
    }

    // Verify bf16 results match f32 within tolerance
    for (0..4) |i| {
        const bf16_result = bf16ToF32(out_data_bf16[i]);
        try std.testing.expectApproxEqRel(out_data_f32[i], bf16_result, 5e-2);
    }
}

// Test multi-dtype support for sigmoid
test "sigmoid multi-dtype" {
    const allocator = std.testing.allocator;

    const in_data_f32 = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 };

    // f16 test
    const in_data_f16 = try allocator.alloc(u16, 5);
    defer allocator.free(in_data_f16);
    const out_data_f16 = try allocator.alloc(u16, 5);
    defer allocator.free(out_data_f16);

    for (in_data_f32, 0..) |val, i| {
        in_data_f16[i] = f32ToFp16(val);
    }

    const input_f16 = TensorView.initContiguous(@ptrCast(in_data_f16.ptr), &.{5}, .f16);
    const out_f16 = TensorView.initContiguous(@ptrCast(out_data_f16.ptr), &.{5}, .f16);
    sigmoid(out_f16, input_f16);

    // Verify sigmoid behavior with f16
    // Expected: [0.1192, 0.2689, 0.5, 0.7311, 0.8808]
    try std.testing.expectApproxEqAbs(@as(f32, 0.1192), fp16ToF32(out_data_f16[0]), 5e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2689), fp16ToF32(out_data_f16[1]), 5e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), fp16ToF32(out_data_f16[2]), 5e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7311), fp16ToF32(out_data_f16[3]), 5e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8808), fp16ToF32(out_data_f16[4]), 5e-3);
}

// Test multi-dtype support for tanh
test "tanh multi-dtype" {
    const allocator = std.testing.allocator;

    const in_data_f32 = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 };

    // bf16 test
    const in_data_bf16 = try allocator.alloc(u16, 5);
    defer allocator.free(in_data_bf16);
    const out_data_bf16 = try allocator.alloc(u16, 5);
    defer allocator.free(out_data_bf16);

    for (in_data_f32, 0..) |val, i| {
        in_data_bf16[i] = f32ToBf16(val);
    }

    const input_bf16 = TensorView.initContiguous(@ptrCast(in_data_bf16.ptr), &.{5}, .bf16);
    const out_bf16 = TensorView.initContiguous(@ptrCast(out_data_bf16.ptr), &.{5}, .bf16);
    tanh(out_bf16, input_bf16);

    // Verify tanh behavior with bf16
    // Expected: [-0.9640, -0.7616, 0.0, 0.7616, 0.9640]
    try std.testing.expectApproxEqAbs(@as(f32, -0.9640), bf16ToF32(out_data_bf16[0]), 5e-3);
    try std.testing.expectApproxEqAbs(@as(f32, -0.7616), bf16ToF32(out_data_bf16[1]), 5e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), bf16ToF32(out_data_bf16[2]), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7616), bf16ToF32(out_data_bf16[3]), 5e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.9640), bf16ToF32(out_data_bf16[4]), 5e-3);
}

// Test multi-dtype support for rsqrt
test "rsqrt multi-dtype" {
    const allocator = std.testing.allocator;

    const in_data_f32 = [_]f32{ 1.0, 4.0, 9.0, 16.0, 25.0 };

    // f16 test
    const in_data_f16 = try allocator.alloc(u16, 5);
    defer allocator.free(in_data_f16);
    const out_data_f16 = try allocator.alloc(u16, 5);
    defer allocator.free(out_data_f16);

    for (in_data_f32, 0..) |val, i| {
        in_data_f16[i] = f32ToFp16(val);
    }

    const input_f16 = TensorView.initContiguous(@ptrCast(in_data_f16.ptr), &.{5}, .f16);
    const out_f16 = TensorView.initContiguous(@ptrCast(out_data_f16.ptr), &.{5}, .f16);
    rsqrt(out_f16, input_f16);

    // Verify rsqrt behavior with f16
    // Expected: [1.0, 0.5, 0.333..., 0.25, 0.2]
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), fp16ToF32(out_data_f16[0]), 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), fp16ToF32(out_data_f16[1]), 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.3333), fp16ToF32(out_data_f16[2]), 5e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), fp16ToF32(out_data_f16[3]), 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), fp16ToF32(out_data_f16[4]), 1e-3);
}

// Test softmax on 1D tensor
test "softmax 1D correctness" {
    var in_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var out_data = [_]f32{0} ** 5;

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{5}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{5}, .f32);

    softmax(out, input);

    // Verify outputs sum to 1
    var sum: f32 = 0.0;
    for (out_data) |val| {
        sum += val;
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);

    // Verify monotonicity: larger inputs -> larger outputs
    try std.testing.expect(out_data[0] < out_data[1]);
    try std.testing.expect(out_data[1] < out_data[2]);
    try std.testing.expect(out_data[2] < out_data[3]);
    try std.testing.expect(out_data[3] < out_data[4]);

    // Verify approximate known values (computed via reference implementation)
    // softmax([1,2,3,4,5]) ≈ [0.01165, 0.03168, 0.08612, 0.23412, 0.63643]
    try std.testing.expectApproxEqAbs(@as(f32, 0.01165), out_data[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.63643), out_data[4], 1e-4);
}

// Test softmax on 2D tensor (last dimension)
test "softmax 2D last dimension" {
    var in_data = [_]f32{
        1.0, 2.0, 3.0, // row 0
        4.0, 5.0, 6.0, // row 1
    };
    var out_data = [_]f32{0} ** 6;

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{ 2, 3 }, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{ 2, 3 }, .f32);

    softmax(out, input); // defaults to last dimension

    // Each row should sum to 1
    const row0_sum = out_data[0] + out_data[1] + out_data[2];
    const row1_sum = out_data[3] + out_data[4] + out_data[5];
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), row0_sum, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), row1_sum, 1e-5);

    // Within each row, verify monotonicity
    try std.testing.expect(out_data[0] < out_data[1] and out_data[1] < out_data[2]);
    try std.testing.expect(out_data[3] < out_data[4] and out_data[4] < out_data[5]);
}

// Test softmaxDim with different dimensions
test "softmaxDim different dimensions" {
    var in_data = [_]f32{
        1.0, 2.0, 3.0, // row 0
        4.0, 5.0, 6.0, // row 1
    };
    var out_data = [_]f32{0} ** 6;

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{ 2, 3 }, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{ 2, 3 }, .f32);

    // Apply softmax over dim 0 (column-wise)
    softmaxDim(out, input, 0);

    // Each column should sum to 1
    const col0_sum = out_data[0] + out_data[3];
    const col1_sum = out_data[1] + out_data[4];
    const col2_sum = out_data[2] + out_data[5];
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), col0_sum, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), col1_sum, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), col2_sum, 1e-5);

    // Second row values should be larger (4,5,6 > 1,2,3)
    try std.testing.expect(out_data[3] > out_data[0]); // col 0
    try std.testing.expect(out_data[4] > out_data[1]); // col 1
    try std.testing.expect(out_data[5] > out_data[2]); // col 2
}

// Test softmax with negative dimension indexing
test "softmax negative dimension" {
    var in_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var out_data = [_]f32{0} ** 6;

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{ 2, 3 }, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{ 2, 3 }, .f32);

    // dim=-1 should be equivalent to last dimension (dim=1)
    softmaxDim(out, input, -1);

    // Each row should sum to 1
    const row0_sum = out_data[0] + out_data[1] + out_data[2];
    const row1_sum = out_data[3] + out_data[4] + out_data[5];
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), row0_sum, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), row1_sum, 1e-5);
}

// Test softmax with uniform input (all equal)
test "softmax uniform input" {
    var in_data = [_]f32{ 2.5, 2.5, 2.5, 2.5 };
    var out_data = [_]f32{0} ** 4;

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{4}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{4}, .f32);

    softmax(out, input);

    // All outputs should be equal (0.25 each)
    for (out_data) |val| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.25), val, 1e-6);
    }
}

// Test softmax with extreme values (numerical stability)
test "softmax numerical stability" {
    var in_data = [_]f32{ 1000.0, 1000.0, 1000.0 };
    var out_data = [_]f32{0} ** 3;

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{3}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{3}, .f32);

    softmax(out, input);

    // Should handle large values without overflow
    for (out_data) |val| {
        try std.testing.expect(!std.math.isNan(val));
        try std.testing.expect(!std.math.isInf(val));
        try std.testing.expectApproxEqAbs(@as(f32, 0.3333), val, 1e-3);
    }

    // Test with very negative values
    var in_data_neg = [_]f32{ -1000.0, -1000.0, -1000.0 };
    var out_data_neg = [_]f32{0} ** 3;

    const input_neg = TensorView.initContiguous(@ptrCast(&in_data_neg), &.{3}, .f32);
    const out_neg = TensorView.initContiguous(@ptrCast(&out_data_neg), &.{3}, .f32);

    softmax(out_neg, input_neg);

    // Should handle very negative values
    for (out_data_neg) |val| {
        try std.testing.expect(!std.math.isNan(val));
        try std.testing.expect(!std.math.isInf(val));
        try std.testing.expectApproxEqAbs(@as(f32, 0.3333), val, 1e-3);
    }
}

// Test softmax with strided tensor
test "softmax strided tensor" {
    const allocator = std.testing.allocator;

    var in_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const out_data = try allocator.alloc(f32, 6);
    defer allocator.free(out_data);
    @memset(out_data, 0);

    // Create 2x3 strided tensor (transposed)
    const input = TensorView.initStrided(
        @ptrCast(&in_data),
        &.{ 3, 2 },
        &.{ 1, 3 }, // non-contiguous strides
        .f32,
    );

    const out = TensorView.initStrided(
        @ptrCast(out_data.ptr),
        &.{ 3, 2 },
        &.{ 2, 1 }, // contiguous output
        .f32,
    );

    // Apply softmax over last dimension (dim=-1)
    softmaxDim(out, input, -1);

    // Logical input: [[1, 4], [2, 5], [3, 6]]
    // Each row should sum to 1
    const row0_sum = out_data[0] + out_data[1];
    const row1_sum = out_data[2] + out_data[3];
    const row2_sum = out_data[4] + out_data[5];
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), row0_sum, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), row1_sum, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), row2_sum, 1e-5);
}

// Test softmax on 3D tensor
test "softmax 3D tensor" {
    const allocator = std.testing.allocator;

    // Create 2x2x3 tensor
    var in_data = [_]f32{
        1.0, 2.0, 3.0, // [0, 0, :]
        4.0, 5.0, 6.0, // [0, 1, :]
        7.0, 8.0, 9.0, // [1, 0, :]
        10.0, 11.0, 12.0, // [1, 1, :]
    };
    const out_data = try allocator.alloc(f32, 12);
    defer allocator.free(out_data);
    @memset(out_data, 0);

    const input = TensorView.initContiguous(@ptrCast(&in_data), &.{ 2, 2, 3 }, .f32);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{ 2, 2, 3 }, .f32);

    // Apply softmax over last dimension
    softmaxDim(out, input, -1);

    // Each slice [:, :, i] should sum to 1 along last dim
    // Check first row of first matrix
    const sum_0_0 = out_data[0] + out_data[1] + out_data[2];
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum_0_0, 1e-5);

    // Check second row of second matrix
    const sum_1_1 = out_data[9] + out_data[10] + out_data[11];
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum_1_1, 1e-5);
}

// Test softmax multi-dtype (f16)
test "softmax multi-dtype f16" {
    const allocator = std.testing.allocator;

    const in_data_f32 = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    // f16 test
    const in_data_f16 = try allocator.alloc(u16, 4);
    defer allocator.free(in_data_f16);
    const out_data_f16 = try allocator.alloc(u16, 4);
    defer allocator.free(out_data_f16);

    for (in_data_f32, 0..) |val, i| {
        in_data_f16[i] = f32ToFp16(val);
    }

    const input_f16 = TensorView.initContiguous(@ptrCast(in_data_f16.ptr), &.{4}, .f16);
    const out_f16 = TensorView.initContiguous(@ptrCast(out_data_f16.ptr), &.{4}, .f16);
    softmax(out_f16, input_f16);

    // Verify output sums to 1
    var sum: f32 = 0.0;
    for (out_data_f16) |val| {
        sum += fp16ToF32(val);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-3);

    // Verify monotonicity
    try std.testing.expect(fp16ToF32(out_data_f16[0]) < fp16ToF32(out_data_f16[1]));
    try std.testing.expect(fp16ToF32(out_data_f16[1]) < fp16ToF32(out_data_f16[2]));
    try std.testing.expect(fp16ToF32(out_data_f16[2]) < fp16ToF32(out_data_f16[3]));
}

// Test large tensor to ensure SIMD path is exercised
test "silu large tensor SIMD" {
    const allocator = std.testing.allocator;

    // Create large tensor (>= 256 elements to ensure SIMD vectorization)
    const size: usize = 1024;
    const in_data = try allocator.alloc(f32, size);
    defer allocator.free(in_data);
    const out_data = try allocator.alloc(f32, size);
    defer allocator.free(out_data);

    // Initialize with varying values
    for (0..size) |i| {
        const fi: f32 = @floatFromInt(i);
        in_data[i] = (fi / 100.0) - 5.0; // Range from -5 to ~5
    }

    const input = TensorView.initContiguous(@ptrCast(in_data.ptr), &.{size}, .f32);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{size}, .f32);

    silu(out, input);

    // Verify correctness for sample elements
    for (0..size) |i| {
        const x = in_data[i];
        const expected_sigmoid = 1.0 / (1.0 + @exp(-x));
        const expected_silu = x * expected_sigmoid;
        try std.testing.expectApproxEqRel(expected_silu, out_data[i], 1e-4);
    }
}

// Test large tensor with GELU to ensure SIMD path
test "gelu large tensor SIMD" {
    const allocator = std.testing.allocator;

    const size: usize = 512;
    const in_data = try allocator.alloc(f32, size);
    defer allocator.free(in_data);
    const out_data = try allocator.alloc(f32, size);
    defer allocator.free(out_data);

    // Initialize with varying values
    for (0..size) |i| {
        const fi: f32 = @floatFromInt(i);
        in_data[i] = (fi / 50.0) - 5.0; // Range from -5 to ~5
    }

    const input = TensorView.initContiguous(@ptrCast(in_data.ptr), &.{size}, .f32);
    const out = TensorView.initContiguous(@ptrCast(out_data.ptr), &.{size}, .f32);

    gelu(out, input);

    // Verify correctness for sample elements
    const sqrt_2_over_pi: f32 = 0.7978845608028654;
    const coeff: f32 = 0.044715;

    for (0..size) |i| {
        const x = in_data[i];
        const x3 = x * x * x;
        const inner = sqrt_2_over_pi * (x + coeff * x3);
        const expected_gelu = 0.5 * x * (1.0 + std.math.tanh(inner));
        try std.testing.expectApproxEqRel(expected_gelu, out_data[i], 1e-4);
    }
}

// Test in-place operation: output and input are the same tensor
test "silu gelu relu in-place operations" {
    var data = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    const original = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 };

    const tensor = TensorView.initContiguous(@ptrCast(&data), &.{5}, .f32);

    // Test in-place ReLU
    relu(tensor, tensor);

    // Verify ReLU was applied in-place
    try std.testing.expectEqual(@as(f32, 0.0), data[0]); // -2 -> 0
    try std.testing.expectEqual(@as(f32, 0.0), data[1]); // -1 -> 0
    try std.testing.expectEqual(@as(f32, 0.0), data[2]); // 0 -> 0
    try std.testing.expectEqual(@as(f32, 1.0), data[3]); // 1 -> 1
    try std.testing.expectEqual(@as(f32, 2.0), data[4]); // 2 -> 2

    // Reset data for next test
    @memcpy(&data, &original);

    // Test in-place sigmoid
    sigmoid(tensor, tensor);

    // Verify sigmoid was applied (values should be in (0, 1))
    for (data) |val| {
        try std.testing.expect(val > 0.0 and val < 1.0);
    }

    // Reset and test in-place SiLU
    @memcpy(&data, &original);
    silu(tensor, tensor);

    // Verify SiLU: silu(0) = 0, silu(positive) > 0, silu(negative) < 0
    try std.testing.expect(data[0] < 0.0); // silu(-2)
    try std.testing.expect(data[1] < 0.0); // silu(-1)
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data[2], 1e-6); // silu(0)
    try std.testing.expect(data[3] > 0.0); // silu(1)
    try std.testing.expect(data[4] > 0.0); // silu(2)
}
