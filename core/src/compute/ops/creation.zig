//! Tensor creation operations.

const std = @import("std");
const tv = @import("tensor_view.zig");
const tensor = @import("../../tensor.zig");
const matmul_ops = @import("matmul_primitives.zig");
const parallel = @import("../parallel.zig");
const simd = @import("math_primitives/root.zig").simd;

const TensorView = tv.TensorView;
const DType = tv.DType;

// SIMD configuration
const VEC_LEN = simd.f32_vec_len;
const F32Vec = simd.F32Vec;

/// Fill tensor with zeros
pub fn zeros(out: TensorView) void {
    switch (out.dtype) {
        .f32 => zerosTyped(f32, out),
        .f16, .bf16 => zerosTyped(u16, out),
        .i32 => zerosTyped(i32, out),
        .i64 => zerosTyped(i64, out),
    }
}

fn zerosTyped(comptime T: type, out: TensorView) void {
    const data = @as([*]T, @ptrCast(@alignCast(out.data)));
    @memset(data[0..out.numel], 0);
}

/// Fill tensor with ones
pub fn ones(out: TensorView) void {
    switch (out.dtype) {
        .f32 => onesTyped(f32, out),
        .f16, .bf16 => onesTyped(u16, out),
        .i32 => onesTyped(i32, out),
        .i64 => onesTyped(i64, out),
    }
}

fn onesTyped(comptime T: type, out: TensorView) void {
    const data = @as([*]T, @ptrCast(@alignCast(out.data)));
    const one: T = if (T == f32) 1.0 else if (T == u16) 0x3C00 else 1; // 0x3C00 is fp16(1.0)
    @memset(data[0..out.numel], one);
}

/// Fill tensor with range [0, n)
pub fn arange(out: TensorView) void {
    switch (out.dtype) {
        .f32 => arangeTyped(f32, out),
        .i32 => arangeTyped(i32, out),
        .i64 => arangeTyped(i64, out),
        else => unreachable,
    }
}

fn arangeTyped(comptime T: type, out: TensorView) void {
    const data = @as([*]T, @ptrCast(@alignCast(out.data)));
    for (0..out.numel) |elem_idx| {
        data[elem_idx] = if (@typeInfo(T) == .float) @floatFromInt(elem_idx) else @intCast(elem_idx);
    }
}

/// Create causal attention mask
/// out: [seq_len, seq_len] with 0.0 for valid positions, -inf for masked
pub fn causalMask(out: TensorView) void {
    std.debug.assert(out.ndim == 2);
    std.debug.assert(out.shape[0] == out.shape[1]);

    switch (out.dtype) {
        .f32 => causalMaskTyped(f32, out),
        .f16, .bf16 => causalMaskTyped(u16, out),
        else => unreachable,
    }
}

fn causalMaskTyped(comptime T: type, out: TensorView) void {
    const data = @as([*]T, @ptrCast(@alignCast(out.data)));
    const seq_len = out.shape[0];

    const zero: T = 0;
    const neg_inf: T = if (T == f32) -std.math.inf(f32) else 0xFC00; // 0xFC00 is fp16(-inf)

    for (0..seq_len) |row| {
        for (0..seq_len) |col| {
            const out_offset = row * @as(usize, @intCast(out.strides[0])) +
                col * @as(usize, @intCast(out.strides[1]));
            data[out_offset] = if (col <= row) zero else neg_inf;
        }
    }
}

/// Upper triangular matrix
/// Zeros out elements below the diagonal
/// diagonal: 0 = main diagonal, positive = above, negative = below
pub fn triu(out: TensorView, input: TensorView, diagonal: i32) void {
    switch (out.dtype) {
        .f32 => triuTyped(f32, out, input, diagonal),
        .f16, .bf16 => triuTyped(u16, out, input, diagonal),
        else => {},
    }
}

fn triuTyped(comptime T: type, out: TensorView, input: TensorView, diagonal: i32) void {
    const out_data = @as([*]T, @ptrCast(@alignCast(out.data)));
    const in_data = @as([*]const T, @ptrCast(@alignCast(input.data)));

    std.debug.assert(input.ndim >= 2);
    const rows = input.shape[input.ndim - 2];
    const cols = input.shape[input.ndim - 1];
    const batch_size = input.numel / (rows * cols);

    const zero: T = 0;

    for (0..batch_size) |b| {
        const batch_offset = b * rows * cols;
        for (0..rows) |row| {
            for (0..cols) |col| {
                const elem_offset = batch_offset + row * cols + col;
                const signed_col: i64 = @intCast(col);
                const signed_row: i64 = @intCast(row);
                if (signed_col >= signed_row + diagonal) {
                    out_data[elem_offset] = in_data[elem_offset];
                } else {
                    out_data[elem_offset] = zero;
                }
            }
        }
    }
}

/// Embedding lookup
/// indices: [batch, seq] or [seq]
/// weight: [vocab_size, hidden_dim]
/// out: [batch, seq, hidden_dim] or [seq, hidden_dim]
pub fn embedding(out: TensorView, weight: TensorView, indices: TensorView) void {
    switch (out.dtype) {
        .f32 => embeddingTyped(f32, out, weight, indices),
        .f16, .bf16 => embeddingTyped(u16, out, weight, indices),
        else => unreachable,
    }
}

fn embeddingTyped(comptime T: type, out: TensorView, weight: TensorView, indices: TensorView) void {
    const out_data = @as([*]T, @ptrCast(@alignCast(out.data)));
    const w_data = @as([*]const T, @ptrCast(@alignCast(weight.data)));

    const hidden_dim = weight.shape[1];
    const num_tokens = indices.numel;

    // Handle different index types
    const indices_i64 = @as([*]const i64, @ptrCast(@alignCast(indices.data)));
    const indices_i32 = @as([*]const i32, @ptrCast(@alignCast(indices.data)));

    for (0..num_tokens) |token_idx| {
        const vocab_idx: usize = switch (indices.dtype) {
            .i64 => @intCast(indices_i64[token_idx]),
            .i32 => @intCast(indices_i32[token_idx]),
            else => unreachable,
        };

        const out_offset = token_idx * hidden_dim;
        const w_offset = vocab_idx * hidden_dim;

        @memcpy(out_data[out_offset .. out_offset + hidden_dim], w_data[w_offset .. w_offset + hidden_dim]);
    }
}

/// Matrix multiplication for linear layers
/// out = input @ weight^T
/// input: [batch, seq, in_features]
/// weight: [out_features, in_features]
/// out: [batch, seq, out_features]
pub fn linear(out: TensorView, input: TensorView, weight: TensorView) void {
    switch (out.dtype) {
        .f32 => linearTyped(f32, f32Identity, f32Identity, out, input, weight),
        .f16 => linearTyped(u16, fp16ToF32, f32ToFp16, out, input, weight),
        .bf16 => linearTyped(u16, bf16ToF32, f32ToBf16, out, input, weight),
        else => unreachable,
    }
}

/// Matrix multiplication with bias for linear layers
/// out = input @ weight^T + bias
/// input: [batch, seq, in_features]
/// weight: [out_features, in_features]
/// bias: [out_features]
/// out: [batch, seq, out_features]
pub fn linearWithBias(out: TensorView, input: TensorView, weight: TensorView, bias: TensorView) void {
    linear(out, input, weight);
    addBias(out, bias);
}

/// Add bias to output tensor (broadcast along last dimension)
/// out: [..., features] += bias: [features]
pub fn addBias(out: TensorView, bias: TensorView) void {
    const dtype_m = @import("../../dtype.zig");
    const features = out.shape[out.ndim - 1];
    const num_rows = out.numel / features;

    switch (out.dtype) {
        .f32 => {
            const out_data = @as([*]f32, @ptrCast(@alignCast(out.data)));
            const bias_data = @as([*]const f32, @ptrCast(@alignCast(bias.data)));
            for (0..num_rows) |row| {
                for (0..features) |col| {
                    out_data[row * features + col] += bias_data[col];
                }
            }
        },
        .bf16 => {
            const out_data = @as([*]u16, @ptrCast(@alignCast(out.data)));
            const bias_data = @as([*]const u16, @ptrCast(@alignCast(bias.data)));
            for (0..num_rows) |row| {
                for (0..features) |col| {
                    const out_val = dtype_m.bf16ToF32(out_data[row * features + col]);
                    const bias_val = dtype_m.bf16ToF32(bias_data[col]);
                    out_data[row * features + col] = dtype_m.f32ToBf16(out_val + bias_val);
                }
            }
        },
        .f16 => {
            const out_data = @as([*]u16, @ptrCast(@alignCast(out.data)));
            const bias_data = @as([*]const u16, @ptrCast(@alignCast(bias.data)));
            for (0..num_rows) |row| {
                for (0..features) |col| {
                    const out_val = dtype_m.fp16ToF32(out_data[row * features + col]);
                    const bias_val = dtype_m.fp16ToF32(bias_data[col]);
                    out_data[row * features + col] = dtype_m.f32ToFp16(out_val + bias_val);
                }
            }
        },
        else => unreachable,
    }
}

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

fn f32Identity(x: f32) f32 {
    return x;
}

fn linearTyped(
    comptime T: type,
    comptime toF32: fn (T) f32,
    comptime fromF32: fn (f32) T,
    out: TensorView,
    input: TensorView,
    weight: TensorView,
) void {
    const out_data = @as([*]T, @ptrCast(@alignCast(out.data)));
    const in_data = @as([*]const T, @ptrCast(@alignCast(input.data)));
    const w_data = @as([*]const T, @ptrCast(@alignCast(weight.data)));

    const num_tokens = input.numel / input.shape[input.ndim - 1];
    const in_features = input.shape[input.ndim - 1];
    const out_features = weight.shape[0];

    // For f32, use SIMD + parallel. For other types, use parallel only.
    if (T == f32) {
        linearF32Parallel(out_data, in_data, w_data, num_tokens, in_features, out_features);
    } else {
        linearTypedParallel(T, toF32, fromF32, out_data, in_data, w_data, num_tokens, in_features, out_features);
    }
}

/// SIMD-optimized parallel linear for f32
fn linearF32Parallel(
    out_data: [*]f32,
    in_data: [*]const f32,
    w_data: [*]const f32,
    num_tokens: usize,
    in_features: usize,
    out_features: usize,
) void {
    const LinearF32Ctx = struct {
        out_data: [*]f32,
        in_data: [*]const f32,
        w_data: [*]const f32,
        num_tokens: usize,
        in_features: usize,
        out_features: usize,

        fn runLinearTiles(start: usize, end: usize, tile_ctx: *@This()) void {
            for (start..end) |tile_idx| {
                const token_idx = tile_idx / tile_ctx.out_features;
                const out_idx = tile_idx % tile_ctx.out_features;

                const in_row = tile_ctx.in_data[token_idx * tile_ctx.in_features ..][0..tile_ctx.in_features];
                const w_row = tile_ctx.w_data[out_idx * tile_ctx.in_features ..][0..tile_ctx.in_features];

                // SIMD dot product
                var sum_vec: F32Vec = @splat(0);
                var elem_idx: usize = 0;

                // Main SIMD loop
                while (elem_idx + VEC_LEN <= tile_ctx.in_features) : (elem_idx += VEC_LEN) {
                    const in_vec: F32Vec = in_row[elem_idx..][0..VEC_LEN].*;
                    const w_vec: F32Vec = w_row[elem_idx..][0..VEC_LEN].*;
                    sum_vec = @mulAdd(F32Vec, in_vec, w_vec, sum_vec);
                }

                // Reduce vector to scalar
                var sum: f32 = @reduce(.Add, sum_vec);

                // Scalar remainder
                while (elem_idx < tile_ctx.in_features) : (elem_idx += 1) {
                    sum += in_row[elem_idx] * w_row[elem_idx];
                }

                tile_ctx.out_data[token_idx * tile_ctx.out_features + out_idx] = sum;
            }
        }
    };

    var linear_ctx = LinearF32Ctx{
        .out_data = out_data,
        .in_data = in_data,
        .w_data = w_data,
        .num_tokens = num_tokens,
        .in_features = in_features,
        .out_features = out_features,
    };

    parallel.global().parallelFor(num_tokens * out_features, LinearF32Ctx.runLinearTiles, &linear_ctx);
}

/// Parallel linear for non-f32 types (bf16, f16)
fn linearTypedParallel(
    comptime T: type,
    comptime toF32: fn (T) f32,
    comptime fromF32: fn (f32) T,
    out_data: [*]T,
    in_data: [*]const T,
    w_data: [*]const T,
    num_tokens: usize,
    in_features: usize,
    out_features: usize,
) void {
    const LinearTypedCtx = struct {
        out_data: [*]T,
        in_data: [*]const T,
        w_data: [*]const T,
        num_tokens: usize,
        in_features: usize,
        out_features: usize,

        fn runLinearTiles(start: usize, end: usize, tile_ctx: *@This()) void {
            for (start..end) |tile_idx| {
                const token_idx = tile_idx / tile_ctx.out_features;
                const out_idx = tile_idx % tile_ctx.out_features;

                const in_offset = token_idx * tile_ctx.in_features;
                const w_offset = out_idx * tile_ctx.in_features;

                var sum: f32 = 0;
                for (0..tile_ctx.in_features) |feature_idx| {
                    sum += toF32(tile_ctx.in_data[in_offset + feature_idx]) * toF32(tile_ctx.w_data[w_offset + feature_idx]);
                }

                tile_ctx.out_data[token_idx * tile_ctx.out_features + out_idx] = fromF32(sum);
            }
        }
    };

    var linear_ctx = LinearTypedCtx{
        .out_data = out_data,
        .in_data = in_data,
        .w_data = w_data,
        .num_tokens = num_tokens,
        .in_features = in_features,
        .out_features = out_features,
    };

    parallel.global().parallelFor(num_tokens * out_features, LinearTypedCtx.runLinearTiles, &linear_ctx);
}

/// General matrix multiplication
/// out = a @ b
/// a: [..., M, K]
/// b: [..., K, N]
/// out: [..., M, N]
pub fn matmul(out: TensorView, a: TensorView, b: TensorView) void {
    if (out.dtype == .f32 and a.dtype == .f32 and b.dtype == .f32 and a.ndim == 2 and b.ndim == 2 and out.ndim == 2 and
        a.isContiguous() and b.isContiguous() and out.isContiguous())
    {
        var a_tensor = tensor.Tensor.view(a.data, &.{ a.shape[0], a.shape[1] }, .f32, null);
        var b_tensor = tensor.Tensor.view(b.data, &.{ b.shape[0], b.shape[1] }, .f32, null);
        var out_tensor = tensor.Tensor.view(out.data, &.{ out.shape[0], out.shape[1] }, .f32, null);
        var scratch = emptyMatmulScratch();
        matmul_ops.matmulF32(&a_tensor, &b_tensor, &out_tensor, &scratch);
        return;
    }

    switch (out.dtype) {
        .f32 => matmulTyped(f32, f32Identity, f32Identity, out, a, b),
        .f16 => matmulTyped(u16, fp16ToF32, f32ToFp16, out, a, b),
        .bf16 => matmulTyped(u16, bf16ToF32, f32ToBf16, out, a, b),
        else => unreachable,
    }
}

fn emptyMatmulScratch() matmul_ops.MatmulScratch {
    return .{
        .allocator = std.heap.page_allocator,
    };
}

fn matmulTyped(
    comptime T: type,
    comptime toF32: fn (T) f32,
    comptime fromF32: fn (f32) T,
    out: TensorView,
    a: TensorView,
    b: TensorView,
) void {
    std.debug.assert(a.ndim >= 2 and b.ndim >= 2);

    const out_data = @as([*]T, @ptrCast(@alignCast(out.data)));
    const a_data = @as([*]const T, @ptrCast(@alignCast(a.data)));
    const b_data = @as([*]const T, @ptrCast(@alignCast(b.data)));

    const M = a.shape[a.ndim - 2];
    const K = a.shape[a.ndim - 1];
    const N = b.shape[b.ndim - 1];
    const batch = a.numel / (M * K);

    // Simple batched matmul (can be optimized with SIMD/tiling later)
    for (0..batch) |bat| {
        const a_batch_off = bat * M * K;
        const b_batch_off = bat * K * N;
        const out_batch_off = bat * M * N;

        for (0..M) |m| {
            for (0..N) |n| {
                var sum: f32 = 0;
                for (0..K) |k| {
                    sum += toF32(a_data[a_batch_off + m * K + k]) *
                        toF32(b_data[b_batch_off + k * N + n]);
                }
                out_data[out_batch_off + m * N + n] = fromF32(sum);
            }
        }
    }
}

test "zeros fills with zeros" {
    var data = [_]f32{ 1, 2, 3, 4 };
    const out = TensorView.initContiguous(@ptrCast(&data), &.{4}, .f32);
    zeros(out);
    for (data) |v| try std.testing.expectEqual(@as(f32, 0), v);
}

test "causalMask correct shape" {
    var data = [_]f32{0} ** 9;
    const out = TensorView.initContiguous(@ptrCast(&data), &.{ 3, 3 }, .f32);
    causalMask(out);

    // Lower triangle should be 0, upper should be -inf
    try std.testing.expectEqual(@as(f32, 0), data[0]); // [0,0]
    try std.testing.expect(data[1] < -1e30); // [0,1] = -inf
    try std.testing.expectEqual(@as(f32, 0), data[4]); // [1,1]
}

test "zeros fills i32 with zeros" {
    var data = [_]i32{ 1, 2, 3, 4 };
    const out = TensorView.initContiguous(@ptrCast(&data), &.{4}, .i32);
    zeros(out);
    for (data) |v| try std.testing.expectEqual(@as(i32, 0), v);
}

test "ones fills f32 with ones" {
    var data = [_]f32{ 0, 0, 0, 0 };
    const out = TensorView.initContiguous(@ptrCast(&data), &.{4}, .f32);
    ones(out);
    for (data) |v| try std.testing.expectEqual(@as(f32, 1.0), v);
}

test "ones fills i32 with ones" {
    var data = [_]i32{ 0, 0, 0, 0 };
    const out = TensorView.initContiguous(@ptrCast(&data), &.{4}, .i32);
    ones(out);
    for (data) |v| try std.testing.expectEqual(@as(i32, 1), v);
}

test "arange creates sequential f32 values" {
    var data = [_]f32{ 0, 0, 0, 0, 0 };
    const out = TensorView.initContiguous(@ptrCast(&data), &.{5}, .f32);
    arange(out);
    try std.testing.expectEqual(@as(f32, 0), data[0]);
    try std.testing.expectEqual(@as(f32, 1), data[1]);
    try std.testing.expectEqual(@as(f32, 2), data[2]);
    try std.testing.expectEqual(@as(f32, 3), data[3]);
    try std.testing.expectEqual(@as(f32, 4), data[4]);
}

test "arange creates sequential i32 values" {
    var data = [_]i32{ 0, 0, 0, 0 };
    const out = TensorView.initContiguous(@ptrCast(&data), &.{4}, .i32);
    arange(out);
    try std.testing.expectEqual(@as(i32, 0), data[0]);
    try std.testing.expectEqual(@as(i32, 1), data[1]);
    try std.testing.expectEqual(@as(i32, 2), data[2]);
    try std.testing.expectEqual(@as(i32, 3), data[3]);
}

test "causalMask full 4x4 verification" {
    var data = [_]f32{0} ** 16;
    const out = TensorView.initContiguous(@ptrCast(&data), &.{ 4, 4 }, .f32);
    causalMask(out);

    // Row 0: [0, -inf, -inf, -inf]
    try std.testing.expectEqual(@as(f32, 0), data[0]);
    try std.testing.expect(data[1] < -1e30);
    try std.testing.expect(data[2] < -1e30);
    try std.testing.expect(data[3] < -1e30);

    // Row 1: [0, 0, -inf, -inf]
    try std.testing.expectEqual(@as(f32, 0), data[4]);
    try std.testing.expectEqual(@as(f32, 0), data[5]);
    try std.testing.expect(data[6] < -1e30);
    try std.testing.expect(data[7] < -1e30);

    // Row 2: [0, 0, 0, -inf]
    try std.testing.expectEqual(@as(f32, 0), data[8]);
    try std.testing.expectEqual(@as(f32, 0), data[9]);
    try std.testing.expectEqual(@as(f32, 0), data[10]);
    try std.testing.expect(data[11] < -1e30);

    // Row 3: [0, 0, 0, 0]
    try std.testing.expectEqual(@as(f32, 0), data[12]);
    try std.testing.expectEqual(@as(f32, 0), data[13]);
    try std.testing.expectEqual(@as(f32, 0), data[14]);
    try std.testing.expectEqual(@as(f32, 0), data[15]);
}

test "triu main diagonal" {
    // Input: [[1,2,3],[4,5,6],[7,8,9]]
    var input_data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    var out_data = [_]f32{0} ** 9;
    const input = TensorView.initContiguous(@ptrCast(&input_data), &.{ 3, 3 }, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{ 3, 3 }, .f32);

    triu(out, input, 0);

    // Expected: [[1,2,3],[0,5,6],[0,0,9]]
    try std.testing.expectEqual(@as(f32, 1), out_data[0]);
    try std.testing.expectEqual(@as(f32, 2), out_data[1]);
    try std.testing.expectEqual(@as(f32, 3), out_data[2]);
    try std.testing.expectEqual(@as(f32, 0), out_data[3]);
    try std.testing.expectEqual(@as(f32, 5), out_data[4]);
    try std.testing.expectEqual(@as(f32, 6), out_data[5]);
    try std.testing.expectEqual(@as(f32, 0), out_data[6]);
    try std.testing.expectEqual(@as(f32, 0), out_data[7]);
    try std.testing.expectEqual(@as(f32, 9), out_data[8]);
}

test "triu positive diagonal" {
    // Input: [[1,2,3],[4,5,6],[7,8,9]]
    var input_data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    var out_data = [_]f32{0} ** 9;
    const input = TensorView.initContiguous(@ptrCast(&input_data), &.{ 3, 3 }, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{ 3, 3 }, .f32);

    triu(out, input, 1); // One above main diagonal

    // Expected: [[0,2,3],[0,0,6],[0,0,0]]
    try std.testing.expectEqual(@as(f32, 0), out_data[0]);
    try std.testing.expectEqual(@as(f32, 2), out_data[1]);
    try std.testing.expectEqual(@as(f32, 3), out_data[2]);
    try std.testing.expectEqual(@as(f32, 0), out_data[3]);
    try std.testing.expectEqual(@as(f32, 0), out_data[4]);
    try std.testing.expectEqual(@as(f32, 6), out_data[5]);
    try std.testing.expectEqual(@as(f32, 0), out_data[6]);
    try std.testing.expectEqual(@as(f32, 0), out_data[7]);
    try std.testing.expectEqual(@as(f32, 0), out_data[8]);
}

test "triu negative diagonal" {
    // Input: [[1,2,3],[4,5,6],[7,8,9]]
    var input_data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    var out_data = [_]f32{0} ** 9;
    const input = TensorView.initContiguous(@ptrCast(&input_data), &.{ 3, 3 }, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{ 3, 3 }, .f32);

    triu(out, input, -1); // One below main diagonal

    // Expected: [[1,2,3],[4,5,6],[0,8,9]]
    try std.testing.expectEqual(@as(f32, 1), out_data[0]);
    try std.testing.expectEqual(@as(f32, 2), out_data[1]);
    try std.testing.expectEqual(@as(f32, 3), out_data[2]);
    try std.testing.expectEqual(@as(f32, 4), out_data[3]);
    try std.testing.expectEqual(@as(f32, 5), out_data[4]);
    try std.testing.expectEqual(@as(f32, 6), out_data[5]);
    try std.testing.expectEqual(@as(f32, 0), out_data[6]);
    try std.testing.expectEqual(@as(f32, 8), out_data[7]);
    try std.testing.expectEqual(@as(f32, 9), out_data[8]);
}

test "embedding lookup basic" {
    // Weight matrix: vocab_size=3, hidden_dim=2
    // [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    var weight_data = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 };
    var indices_data = [_]i32{ 0, 2, 1 }; // Look up tokens 0, 2, 1
    var out_data = [_]f32{0} ** 6;

    const weight = TensorView.initContiguous(@ptrCast(&weight_data), &.{ 3, 2 }, .f32);
    const indices = TensorView.initContiguous(@ptrCast(&indices_data), &.{3}, .i32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{ 3, 2 }, .f32);

    embedding(out, weight, indices);

    // Token 0: [0.1, 0.2]
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), out_data[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), out_data[1], 1e-6);
    // Token 2: [0.5, 0.6]
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out_data[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), out_data[3], 1e-6);
    // Token 1: [0.3, 0.4]
    try std.testing.expectApproxEqAbs(@as(f32, 0.3), out_data[4], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.4), out_data[5], 1e-6);
}

test "embedding lookup i64 indices" {
    var weight_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var indices_data = [_]i64{1}; // Look up token 1
    var out_data = [_]f32{0} ** 2;

    const weight = TensorView.initContiguous(@ptrCast(&weight_data), &.{ 2, 2 }, .f32);
    const indices = TensorView.initContiguous(@ptrCast(&indices_data), &.{1}, .i64);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{ 1, 2 }, .f32);

    embedding(out, weight, indices);

    // Token 1: [3.0, 4.0]
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), out_data[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), out_data[1], 1e-6);
}

test "linear f32 basic matmul" {
    // input: [1, 2] @ weight^T: [[1, 2], [3, 4]]^T = [[1, 3], [2, 4]]
    // [1, 2] @ [[1, 3], [2, 4]] = [1*1+2*2, 1*3+2*4] = [5, 11]
    var input_data = [_]f32{ 1.0, 2.0 };
    var weight_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 }; // [out_features=2, in_features=2]
    var out_data = [_]f32{0} ** 2;

    const input = TensorView.initContiguous(@ptrCast(&input_data), &.{ 1, 2 }, .f32);
    const weight = TensorView.initContiguous(@ptrCast(&weight_data), &.{ 2, 2 }, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{ 1, 2 }, .f32);

    linear(out, input, weight);

    // out = input @ weight^T
    // [1, 2] @ [[1, 2], [3, 4]]^T is actually input[i] * weight[j, i] for each output j
    // out[0] = 1*1 + 2*2 = 5
    // out[1] = 1*3 + 2*4 = 11
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), out_data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 11.0), out_data[1], 1e-5);
}

test "matmul 2x2 matrices" {
    // A: [[1, 2], [3, 4]]
    // B: [[5, 6], [7, 8]]
    // C = A @ B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
    //           = [[19, 22], [43, 50]]
    var a_data = [_]f32{ 1, 2, 3, 4 };
    var b_data = [_]f32{ 5, 6, 7, 8 };
    var out_data = [_]f32{0} ** 4;

    const a = TensorView.initContiguous(@ptrCast(&a_data), &.{ 2, 2 }, .f32);
    const b = TensorView.initContiguous(@ptrCast(&b_data), &.{ 2, 2 }, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{ 2, 2 }, .f32);

    matmul(out, a, b);

    try std.testing.expectApproxEqAbs(@as(f32, 19.0), out_data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 22.0), out_data[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 43.0), out_data[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 50.0), out_data[3], 1e-5);
}

test "matmul non-square matrices" {
    // A: [2, 3] matrix [[1, 2, 3], [4, 5, 6]]
    // B: [3, 2] matrix [[1, 2], [3, 4], [5, 6]]
    // C = A @ B = [2, 2]
    // C[0,0] = 1*1 + 2*3 + 3*5 = 22
    // C[0,1] = 1*2 + 2*4 + 3*6 = 28
    // C[1,0] = 4*1 + 5*3 + 6*5 = 49
    // C[1,1] = 4*2 + 5*4 + 6*6 = 64
    var a_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var b_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var out_data = [_]f32{0} ** 4;

    const a = TensorView.initContiguous(@ptrCast(&a_data), &.{ 2, 3 }, .f32);
    const b = TensorView.initContiguous(@ptrCast(&b_data), &.{ 3, 2 }, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{ 2, 2 }, .f32);

    matmul(out, a, b);

    try std.testing.expectApproxEqAbs(@as(f32, 22.0), out_data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 28.0), out_data[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 49.0), out_data[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 64.0), out_data[3], 1e-5);
}

test "addBias single row" {
    // out: [1, 3] with values [1, 2, 3]
    // bias: [3] with values [10, 20, 30]
    // result: [11, 22, 33]
    var out_data = [_]f32{ 1.0, 2.0, 3.0 };
    var bias_data = [_]f32{ 10.0, 20.0, 30.0 };

    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{ 1, 3 }, .f32);
    const bias = TensorView.initContiguous(@ptrCast(&bias_data), &.{3}, .f32);

    addBias(out, bias);

    try std.testing.expectApproxEqAbs(@as(f32, 11.0), out_data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 22.0), out_data[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 33.0), out_data[2], 1e-5);
}

test "addBias multiple rows" {
    // out: [2, 3] with values [[1, 2, 3], [4, 5, 6]]
    // bias: [3] with values [10, 20, 30]
    // result: [[11, 22, 33], [14, 25, 36]]
    var out_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var bias_data = [_]f32{ 10.0, 20.0, 30.0 };

    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{ 2, 3 }, .f32);
    const bias = TensorView.initContiguous(@ptrCast(&bias_data), &.{3}, .f32);

    addBias(out, bias);

    try std.testing.expectApproxEqAbs(@as(f32, 11.0), out_data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 22.0), out_data[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 33.0), out_data[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 14.0), out_data[3], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 25.0), out_data[4], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 36.0), out_data[5], 1e-5);
}

test "addBias 3D tensor" {
    // out: [2, 2, 2] - batch of 2, each 2x2
    // bias: [2]
    // Each row of each batch gets bias added
    var out_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var bias_data = [_]f32{ 100.0, 200.0 };

    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{ 2, 2, 2 }, .f32);
    const bias = TensorView.initContiguous(@ptrCast(&bias_data), &.{2}, .f32);

    addBias(out, bias);

    // All 4 rows get [100, 200] added
    try std.testing.expectApproxEqAbs(@as(f32, 101.0), out_data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 202.0), out_data[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 103.0), out_data[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 204.0), out_data[3], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 105.0), out_data[4], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 206.0), out_data[5], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 107.0), out_data[6], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 208.0), out_data[7], 1e-5);
}

test "linearWithBias combines linear and bias" {
    // input: [1, 2] = [1, 2]
    // weight: [2, 2] = [[1, 2], [3, 4]]  (out_features=2, in_features=2)
    // bias: [2] = [10, 20]
    // linear: out[0] = 1*1 + 2*2 = 5, out[1] = 1*3 + 2*4 = 11
    // with bias: [15, 31]
    var input_data = [_]f32{ 1.0, 2.0 };
    var weight_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var bias_data = [_]f32{ 10.0, 20.0 };
    var out_data = [_]f32{ 0.0, 0.0 };

    const input = TensorView.initContiguous(@ptrCast(&input_data), &.{ 1, 2 }, .f32);
    const weight = TensorView.initContiguous(@ptrCast(&weight_data), &.{ 2, 2 }, .f32);
    const bias = TensorView.initContiguous(@ptrCast(&bias_data), &.{2}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{ 1, 2 }, .f32);

    linearWithBias(out, input, weight, bias);

    try std.testing.expectApproxEqAbs(@as(f32, 15.0), out_data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 31.0), out_data[1], 1e-5);
}

test "linearWithBias batched" {
    // input: [2, 2] = [[1, 2], [3, 4]]
    // weight: [3, 2] = [[1, 0], [0, 1], [1, 1]]  (out_features=3, in_features=2)
    // bias: [3] = [10, 20, 30]
    // Row 0: linear([1,2]) @ weight^T = [1, 2, 3], with bias = [11, 22, 33]
    // Row 1: linear([3,4]) @ weight^T = [3, 4, 7], with bias = [13, 24, 37]
    var input_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var weight_data = [_]f32{ 1.0, 0.0, 0.0, 1.0, 1.0, 1.0 };
    var bias_data = [_]f32{ 10.0, 20.0, 30.0 };
    var out_data = [_]f32{0} ** 6;

    const input = TensorView.initContiguous(@ptrCast(&input_data), &.{ 2, 2 }, .f32);
    const weight = TensorView.initContiguous(@ptrCast(&weight_data), &.{ 3, 2 }, .f32);
    const bias = TensorView.initContiguous(@ptrCast(&bias_data), &.{3}, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{ 2, 3 }, .f32);

    linearWithBias(out, input, weight, bias);

    // Row 0
    try std.testing.expectApproxEqAbs(@as(f32, 11.0), out_data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 22.0), out_data[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 33.0), out_data[2], 1e-5);
    // Row 1
    try std.testing.expectApproxEqAbs(@as(f32, 13.0), out_data[3], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 24.0), out_data[4], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 37.0), out_data[5], 1e-5);
}
