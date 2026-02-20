//! Normalization operations: RMSNorm, LayerNorm.

const std = @import("std");
const tensor = @import("../../../tensor.zig");
const dtype_mod = @import("../../../dtype.zig");
const simd = @import("../simd/arch/root.zig");

const VEC_LEN = simd.f32_vec_len;
const F32Vec = simd.F32Vec;
const fp16ToF32 = dtype_mod.fp16ToF32;

pub fn rmsnormContiguous(
    out: []f32,
    input: []const f32,
    weight_f32: ?[]const f32,
    weight_u16: ?[]const u16,
    weight_dtype: tensor.DType,
    num_tokens: usize,
    dim: usize,
    eps: f32,
    weight_offset: f32,
) void {
    @setFloatMode(.optimized);
    std.debug.assert(input.len == num_tokens * dim);
    std.debug.assert(out.len == input.len);

    const w_f32 = weight_f32 orelse &[_]f32{};
    const w_u16 = weight_u16 orelse &[_]u16{};
    std.debug.assert(weight_dtype == .f32 or weight_dtype == .f16 or weight_dtype == .bf16);
    if (weight_dtype == .f32) {
        std.debug.assert(weight_f32 != null);
    } else {
        std.debug.assert(weight_u16 != null);
    }

    const UNROLL = 4;
    const UNROLL_STRIDE = UNROLL * VEC_LEN;
    const has_offset = weight_offset != 0.0;
    const offset_vec: F32Vec = @splat(weight_offset);

    var token_idx: usize = 0;
    while (token_idx < num_tokens) : (token_idx += 1) {
        const token_offset = token_idx * dim;
        const x_row = input[token_offset..][0..dim];
        const out_row = out[token_offset..][0..dim];

        var sum_vecs: [UNROLL]F32Vec = .{@as(F32Vec, @splat(0))} ** UNROLL;
        var vec_idx: usize = 0;
        while (vec_idx + UNROLL_STRIDE - 1 < dim) : (vec_idx += UNROLL_STRIDE) {
            inline for (0..UNROLL) |unroll_idx| {
                const input_vec: F32Vec = x_row[vec_idx + unroll_idx * VEC_LEN ..][0..VEC_LEN].*;
                sum_vecs[unroll_idx] = @mulAdd(F32Vec, input_vec, input_vec, sum_vecs[unroll_idx]);
            }
        }
        var sum_vec = sum_vecs[0] + sum_vecs[1] + sum_vecs[2] + sum_vecs[3];
        while (vec_idx + VEC_LEN - 1 < dim) : (vec_idx += VEC_LEN) {
            const input_vec: F32Vec = x_row[vec_idx..][0..VEC_LEN].*;
            sum_vec = @mulAdd(F32Vec, input_vec, input_vec, sum_vec);
        }
        var sum = @reduce(.Add, sum_vec);
        while (vec_idx < dim) : (vec_idx += 1) {
            sum += x_row[vec_idx] * x_row[vec_idx];
        }

        const inv_rms = 1.0 / std.math.sqrt(sum / @as(f32, @floatFromInt(dim)) + eps);
        const inv_rms_vec: F32Vec = @splat(inv_rms);

        vec_idx = 0;
        switch (weight_dtype) {
            .f32 => {
                while (vec_idx + UNROLL_STRIDE - 1 < dim) : (vec_idx += UNROLL_STRIDE) {
                    inline for (0..UNROLL) |unroll_idx| {
                        const off = vec_idx + unroll_idx * VEC_LEN;
                        const input_vec: F32Vec = x_row[off..][0..VEC_LEN].*;
                        var weight_vec: F32Vec = w_f32[off..][0..VEC_LEN].*;
                        if (has_offset) weight_vec += offset_vec;
                        out_row[off..][0..VEC_LEN].* = input_vec * inv_rms_vec * weight_vec;
                    }
                }
                while (vec_idx + VEC_LEN - 1 < dim) : (vec_idx += VEC_LEN) {
                    const input_vec: F32Vec = x_row[vec_idx..][0..VEC_LEN].*;
                    var weight_vec: F32Vec = w_f32[vec_idx..][0..VEC_LEN].*;
                    if (has_offset) weight_vec += offset_vec;
                    out_row[vec_idx..][0..VEC_LEN].* = input_vec * inv_rms_vec * weight_vec;
                }
                while (vec_idx < dim) : (vec_idx += 1) {
                    var weight_value = w_f32[vec_idx];
                    if (has_offset) weight_value += weight_offset;
                    out_row[vec_idx] = x_row[vec_idx] * inv_rms * weight_value;
                }
            },
            .bf16 => {
                while (vec_idx + UNROLL_STRIDE - 1 < dim) : (vec_idx += UNROLL_STRIDE) {
                    inline for (0..UNROLL) |unroll_idx| {
                        const off = vec_idx + unroll_idx * VEC_LEN;
                        const input_vec: F32Vec = x_row[off..][0..VEC_LEN].*;
                        const w_raw: @Vector(VEC_LEN, u16) = w_u16[off..][0..VEC_LEN].*;
                        var weight_vec: F32Vec = @bitCast(@as(@Vector(VEC_LEN, u32), w_raw) << @as(@Vector(VEC_LEN, u5), @splat(16)));
                        if (has_offset) weight_vec += offset_vec;
                        out_row[off..][0..VEC_LEN].* = input_vec * inv_rms_vec * weight_vec;
                    }
                }
                while (vec_idx + VEC_LEN - 1 < dim) : (vec_idx += VEC_LEN) {
                    const input_vec: F32Vec = x_row[vec_idx..][0..VEC_LEN].*;
                    const w_raw: @Vector(VEC_LEN, u16) = w_u16[vec_idx..][0..VEC_LEN].*;
                    var weight_vec: F32Vec = @bitCast(@as(@Vector(VEC_LEN, u32), w_raw) << @as(@Vector(VEC_LEN, u5), @splat(16)));
                    if (has_offset) weight_vec += offset_vec;
                    out_row[vec_idx..][0..VEC_LEN].* = input_vec * inv_rms_vec * weight_vec;
                }
                while (vec_idx < dim) : (vec_idx += 1) {
                    const w_bits: u32 = @as(u32, w_u16[vec_idx]) << 16;
                    var weight_value: f32 = @bitCast(w_bits);
                    if (has_offset) weight_value += weight_offset;
                    out_row[vec_idx] = x_row[vec_idx] * inv_rms * weight_value;
                }
            },
            .f16 => {
                while (vec_idx + UNROLL_STRIDE - 1 < dim) : (vec_idx += UNROLL_STRIDE) {
                    inline for (0..UNROLL) |unroll_idx| {
                        const off = vec_idx + unroll_idx * VEC_LEN;
                        const input_vec: F32Vec = x_row[off..][0..VEC_LEN].*;
                        var weight_vec: F32Vec = undefined;
                        inline for (0..VEC_LEN) |lane_idx| weight_vec[lane_idx] = fp16ToF32(w_u16[off + lane_idx]);
                        if (has_offset) weight_vec += offset_vec;
                        out_row[off..][0..VEC_LEN].* = input_vec * inv_rms_vec * weight_vec;
                    }
                }
                while (vec_idx + VEC_LEN - 1 < dim) : (vec_idx += VEC_LEN) {
                    const input_vec: F32Vec = x_row[vec_idx..][0..VEC_LEN].*;
                    var weight_vec: F32Vec = undefined;
                    inline for (0..VEC_LEN) |lane_idx| weight_vec[lane_idx] = fp16ToF32(w_u16[vec_idx + lane_idx]);
                    if (has_offset) weight_vec += offset_vec;
                    out_row[vec_idx..][0..VEC_LEN].* = input_vec * inv_rms_vec * weight_vec;
                }
                while (vec_idx < dim) : (vec_idx += 1) {
                    var weight_value = fp16ToF32(w_u16[vec_idx]);
                    if (has_offset) weight_value += weight_offset;
                    out_row[vec_idx] = x_row[vec_idx] * inv_rms * weight_value;
                }
            },
            else => unreachable,
        }
    }
}

pub fn layerNormContiguous(
    out: []f32,
    input: []const f32,
    weight: []const f32,
    bias: ?[]const f32,
    num_tokens: usize,
    dim: usize,
    eps: f32,
) void {
    @setFloatMode(.optimized);
    std.debug.assert(input.len == num_tokens * dim);
    std.debug.assert(out.len == input.len);
    std.debug.assert(weight.len >= dim);
    if (bias) |b| std.debug.assert(b.len >= dim);

    const dim_f: f32 = @floatFromInt(dim);

    var token_idx: usize = 0;
    while (token_idx < num_tokens) : (token_idx += 1) {
        const token_offset = token_idx * dim;
        const row = input[token_offset..][0..dim];
        const out_row = out[token_offset..][0..dim];

        var sum_vec: F32Vec = @splat(0);
        var sumsq_vec: F32Vec = @splat(0);
        var vec_idx: usize = 0;
        while (vec_idx + VEC_LEN - 1 < dim) : (vec_idx += VEC_LEN) {
            const row_vec: F32Vec = row[vec_idx..][0..VEC_LEN].*;
            sum_vec += row_vec;
            sumsq_vec = @mulAdd(F32Vec, row_vec, row_vec, sumsq_vec);
        }
        var sum = @reduce(.Add, sum_vec);
        var sumsq = @reduce(.Add, sumsq_vec);
        while (vec_idx < dim) : (vec_idx += 1) {
            const row_value = row[vec_idx];
            sum += row_value;
            sumsq += row_value * row_value;
        }

        const mean = sum / dim_f;
        const variance = @max(sumsq / dim_f - mean * mean, 0);
        const inv_std = 1.0 / @sqrt(variance + eps);

        const mean_vec: F32Vec = @splat(mean);
        const inv_std_vec: F32Vec = @splat(inv_std);

        vec_idx = 0;
        if (bias) |b| {
            while (vec_idx + VEC_LEN - 1 < dim) : (vec_idx += VEC_LEN) {
                const row_vec: F32Vec = row[vec_idx..][0..VEC_LEN].*;
                const weight_vec: F32Vec = weight[vec_idx..][0..VEC_LEN].*;
                const bias_vec: F32Vec = b[vec_idx..][0..VEC_LEN].*;
                out_row[vec_idx..][0..VEC_LEN].* = (row_vec - mean_vec) * inv_std_vec * weight_vec + bias_vec;
            }
            while (vec_idx < dim) : (vec_idx += 1) {
                out_row[vec_idx] = (row[vec_idx] - mean) * inv_std * weight[vec_idx] + b[vec_idx];
            }
        } else {
            while (vec_idx + VEC_LEN - 1 < dim) : (vec_idx += VEC_LEN) {
                const row_vec: F32Vec = row[vec_idx..][0..VEC_LEN].*;
                const weight_vec: F32Vec = weight[vec_idx..][0..VEC_LEN].*;
                out_row[vec_idx..][0..VEC_LEN].* = (row_vec - mean_vec) * inv_std_vec * weight_vec;
            }
            while (vec_idx < dim) : (vec_idx += 1) {
                out_row[vec_idx] = (row[vec_idx] - mean) * inv_std * weight[vec_idx];
            }
        }
    }
}

pub fn rmsnormInPlaceWeightTensor(vec: []f32, weight_tensor: *const tensor.Tensor, eps: f32, weight_offset: f32) void {
    const len = vec.len;
    const w_dtype = weight_tensor.dtype;
    const has_offset = weight_offset != 0.0;

    var sum_sq: f32 = 0;
    for (vec) |v| sum_sq += v * v;
    const inv_rms = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(len)) + eps);

    if (w_dtype == .bf16) {
        const w_u16 = weight_tensor.asSliceUnaligned(u16);
        for (0..len) |elem_idx| {
            var weight_value = dtype_mod.bf16ToF32(w_u16[elem_idx]);
            if (has_offset) weight_value += weight_offset;
            vec[elem_idx] = vec[elem_idx] * inv_rms * weight_value;
        }
    } else if (w_dtype == .f16) {
        const w_u16 = weight_tensor.asSliceUnaligned(u16);
        for (0..len) |elem_idx| {
            var weight_value = dtype_mod.fp16ToF32(w_u16[elem_idx]);
            if (has_offset) weight_value += weight_offset;
            vec[elem_idx] = vec[elem_idx] * inv_rms * weight_value;
        }
    } else {
        const w_f32 = weight_tensor.asSliceUnaligned(f32);
        for (0..len) |elem_idx| {
            const weight_value = if (has_offset) weight_offset + w_f32[elem_idx] else w_f32[elem_idx];
            vec[elem_idx] = vec[elem_idx] * inv_rms * weight_value;
        }
    }
}

test "rmsnormContiguous basic output normalized" {
    const allocator = std.testing.allocator;

    // Single token, simple dimensions
    const dim: usize = 4;
    const num_tokens: usize = 1;
    const eps: f32 = 1e-6;

    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const weight = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const output = try allocator.alloc(f32, input.len);
    defer allocator.free(output);

    rmsnormContiguous(output, &input, &weight, null, .f32, num_tokens, dim, eps, 0.0);

    // Compute expected RMS of input: sqrt(mean(x²))
    // sum of squares = 1 + 4 + 9 + 16 = 30
    // mean of squares = 30/4 = 7.5
    // rms = sqrt(7.5) = 2.7386...
    // inv_rms = 1/rms = 0.3651...

    // For weight=1, output = input * inv_rms
    const sum_sq = 1.0 + 4.0 + 9.0 + 16.0;
    const rms = @sqrt(sum_sq / 4.0);
    const inv_rms = 1.0 / rms;

    for (input, 0..) |in_val, i| {
        try std.testing.expectApproxEqRel(in_val * inv_rms, output[i], 1e-5);
    }

    // Verify the output itself has RMS close to 1.0 (since weight=1)
    var out_sum_sq: f32 = 0;
    for (output) |val| {
        out_sum_sq += val * val;
    }
    const out_rms = @sqrt(out_sum_sq / @as(f32, @floatFromInt(dim)));
    try std.testing.expectApproxEqRel(1.0, out_rms, 1e-5);
}

test "rmsnormContiguous weight scaling" {
    const allocator = std.testing.allocator;

    const dim: usize = 4;
    const num_tokens: usize = 1;
    const eps: f32 = 1e-6;

    const input = [_]f32{ 2.0, 4.0, 6.0, 8.0 };
    const weight = [_]f32{ 0.5, 1.0, 1.5, 2.0 };
    const output = try allocator.alloc(f32, input.len);
    defer allocator.free(output);

    rmsnormContiguous(output, &input, &weight, null, .f32, num_tokens, dim, eps, 0.0);

    // Compute RMS: sqrt(mean([4, 16, 36, 64])) = sqrt(30) ≈ 5.477
    const sum_sq = 4.0 + 16.0 + 36.0 + 64.0;
    const inv_rms = 1.0 / @sqrt(sum_sq / 4.0);

    // output[i] = input[i] * inv_rms * weight[i]
    for (input, 0..) |in_val, i| {
        const expected = in_val * inv_rms * weight[i];
        try std.testing.expectApproxEqRel(expected, output[i], 1e-5);
    }
}

test "rmsnormContiguous weight offset" {
    const allocator = std.testing.allocator;

    const dim: usize = 4;
    const num_tokens: usize = 1;
    const eps: f32 = 1e-6;
    const weight_offset: f32 = 0.1;

    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const weight = [_]f32{ 0.9, 0.9, 0.9, 0.9 };
    const output = try allocator.alloc(f32, input.len);
    defer allocator.free(output);

    rmsnormContiguous(output, &input, &weight, null, .f32, num_tokens, dim, eps, weight_offset);

    // With offset, effective weight = weight + offset = 0.9 + 0.1 = 1.0
    const sum_sq = 1.0 + 4.0 + 9.0 + 16.0;
    const inv_rms = 1.0 / @sqrt(sum_sq / 4.0 + eps);

    for (input, 0..) |in_val, i| {
        const effective_weight = weight[i] + weight_offset;
        const expected = in_val * inv_rms * effective_weight;
        try std.testing.expectApproxEqRel(expected, output[i], 1e-5);
    }
}

test "rmsnormContiguous multiple tokens" {
    const allocator = std.testing.allocator;

    const dim: usize = 3;
    const num_tokens: usize = 2;
    const eps: f32 = 1e-6;

    // Two tokens with different values
    const input = [_]f32{
        1.0, 2.0, 3.0, // Token 0
        4.0, 5.0, 6.0, // Token 1
    };
    const weight = [_]f32{ 1.0, 1.0, 1.0 };
    const output = try allocator.alloc(f32, input.len);
    defer allocator.free(output);

    rmsnormContiguous(output, &input, &weight, null, .f32, num_tokens, dim, eps, 0.0);

    // Token 0: RMS = sqrt((1 + 4 + 9) / 3) = sqrt(14/3) ≈ 2.160
    const sum_sq_0 = 1.0 + 4.0 + 9.0;
    const inv_rms_0 = 1.0 / @sqrt(sum_sq_0 / 3.0 + eps);

    // Token 1: RMS = sqrt((16 + 25 + 36) / 3) = sqrt(77/3) ≈ 5.066
    const sum_sq_1 = 16.0 + 25.0 + 36.0;
    const inv_rms_1 = 1.0 / @sqrt(sum_sq_1 / 3.0 + eps);

    // Verify token 0
    for (0..dim) |i| {
        try std.testing.expectApproxEqRel(input[i] * inv_rms_0, output[i], 1e-5);
    }

    // Verify token 1
    for (0..dim) |i| {
        const idx = dim + i;
        try std.testing.expectApproxEqRel(input[idx] * inv_rms_1, output[idx], 1e-5);
    }
}

test "rmsnormContiguous bf16 weights" {
    const allocator = std.testing.allocator;

    const dim: usize = 4;
    const num_tokens: usize = 1;
    const eps: f32 = 1e-6;

    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    // bf16 representation of [1.0, 1.0, 1.0, 1.0]
    // bf16 is stored as u16 with value = f32 bits >> 16
    const weight_bf16 = [_]u16{ 0x3F80, 0x3F80, 0x3F80, 0x3F80 };
    const output = try allocator.alloc(f32, input.len);
    defer allocator.free(output);

    rmsnormContiguous(output, &input, null, &weight_bf16, .bf16, num_tokens, dim, eps, 0.0);

    const sum_sq = 1.0 + 4.0 + 9.0 + 16.0;
    const inv_rms = 1.0 / @sqrt(sum_sq / 4.0 + eps);

    // With weight=1.0, output should match input * inv_rms
    for (input, 0..) |in_val, i| {
        try std.testing.expectApproxEqRel(in_val * inv_rms, output[i], 1e-5);
    }
}

test "rmsnormContiguous f16 weights" {
    const allocator = std.testing.allocator;

    const dim: usize = 4;
    const num_tokens: usize = 1;
    const eps: f32 = 1e-6;

    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    // f16 representation of [1.0, 1.0, 1.0, 1.0]
    const weight_f16 = [_]u16{ 0x3C00, 0x3C00, 0x3C00, 0x3C00 };
    const output = try allocator.alloc(f32, input.len);
    defer allocator.free(output);

    rmsnormContiguous(output, &input, null, &weight_f16, .f16, num_tokens, dim, eps, 0.0);

    const sum_sq = 1.0 + 4.0 + 9.0 + 16.0;
    const inv_rms = 1.0 / @sqrt(sum_sq / 4.0 + eps);

    // With weight=1.0, output should match input * inv_rms
    for (input, 0..) |in_val, i| {
        try std.testing.expectApproxEqRel(in_val * inv_rms, output[i], 1e-4);
    }
}

test "rmsnormContiguous epsilon prevents div zero" {
    const allocator = std.testing.allocator;

    const dim: usize = 4;
    const num_tokens: usize = 1;
    const eps: f32 = 1e-6;

    // All zeros - without epsilon would cause division by zero
    const input = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    const weight = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const output = try allocator.alloc(f32, input.len);
    defer allocator.free(output);

    rmsnormContiguous(output, &input, &weight, null, .f32, num_tokens, dim, eps, 0.0);

    // With zero input, RMS = sqrt(0 + eps) = sqrt(eps)
    // inv_rms = 1/sqrt(eps)
    // output = 0 * inv_rms * weight = 0
    for (output) |val| {
        try std.testing.expectEqual(0.0, val);
    }

    // Verify no NaN or Inf
    for (output) |val| {
        try std.testing.expect(std.math.isFinite(val));
    }
}

test "rmsnormContiguous large dimension SIMD" {
    const allocator = std.testing.allocator;

    // Use dimension that exercises both SIMD and scalar paths
    const dim: usize = VEC_LEN * 4 + 3;
    const num_tokens: usize = 1;
    const eps: f32 = 1e-6;

    const input = try allocator.alloc(f32, dim);
    defer allocator.free(input);
    const weight = try allocator.alloc(f32, dim);
    defer allocator.free(weight);
    const output = try allocator.alloc(f32, dim);
    defer allocator.free(output);

    // Fill with sequential values
    for (0..dim) |i| {
        input[i] = @floatFromInt(i + 1);
        weight[i] = 1.0;
    }

    rmsnormContiguous(output, input, weight, null, .f32, num_tokens, dim, eps, 0.0);

    // Compute expected RMS
    var sum_sq: f32 = 0;
    for (input) |val| {
        sum_sq += val * val;
    }
    const inv_rms = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(dim)) + eps);

    // Verify output
    for (input, 0..) |in_val, i| {
        try std.testing.expectApproxEqRel(in_val * inv_rms, output[i], 1e-5);
    }

    // Verify output RMS is close to 1.0
    var out_sum_sq: f32 = 0;
    for (output) |val| {
        out_sum_sq += val * val;
    }
    const out_rms = @sqrt(out_sum_sq / @as(f32, @floatFromInt(dim)));
    try std.testing.expectApproxEqRel(1.0, out_rms, 1e-4);
}

test "layerNormContiguous basic zero mean unit variance" {
    const allocator = std.testing.allocator;

    const dim: usize = 4;
    const num_tokens: usize = 1;
    const eps: f32 = 1e-6;

    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const weight = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const output = try allocator.alloc(f32, input.len);
    defer allocator.free(output);

    layerNormContiguous(output, &input, &weight, null, num_tokens, dim, eps);

    // Compute expected: mean = (1+2+3+4)/4 = 2.5
    const mean = (1.0 + 2.0 + 3.0 + 4.0) / 4.0;
    // variance = mean((x - mean)²) = mean([2.25, 0.25, 0.25, 2.25]) = 1.25
    const variance = 1.25;
    const inv_std = 1.0 / @sqrt(variance + eps);

    // output = (input - mean) * inv_std * weight
    for (input, 0..) |in_val, i| {
        const expected = (in_val - mean) * inv_std;
        try std.testing.expectApproxEqRel(expected, output[i], 1e-5);
    }

    // Verify output has zero mean
    var out_mean: f32 = 0;
    for (output) |val| {
        out_mean += val;
    }
    out_mean /= @as(f32, @floatFromInt(dim));
    try std.testing.expectApproxEqAbs(0.0, out_mean, 1e-5);

    // Verify output has unit variance (before weight/bias)
    var out_variance: f32 = 0;
    for (output) |val| {
        out_variance += val * val;
    }
    out_variance /= @as(f32, @floatFromInt(dim));
    try std.testing.expectApproxEqRel(1.0, out_variance, 1e-5);
}

test "layerNormContiguous weight scaling" {
    const allocator = std.testing.allocator;

    const dim: usize = 4;
    const num_tokens: usize = 1;
    const eps: f32 = 1e-6;

    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const weight = [_]f32{ 0.5, 1.0, 1.5, 2.0 };
    const output = try allocator.alloc(f32, input.len);
    defer allocator.free(output);

    layerNormContiguous(output, &input, &weight, null, num_tokens, dim, eps);

    const mean = 2.5;
    const variance = 1.25;
    const inv_std = 1.0 / @sqrt(variance + eps);

    // output[i] = (input[i] - mean) * inv_std * weight[i]
    for (input, 0..) |in_val, i| {
        const expected = (in_val - mean) * inv_std * weight[i];
        try std.testing.expectApproxEqRel(expected, output[i], 1e-5);
    }
}

test "layerNormContiguous with bias" {
    const allocator = std.testing.allocator;

    const dim: usize = 4;
    const num_tokens: usize = 1;
    const eps: f32 = 1e-6;

    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const weight = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const bias = [_]f32{ 0.1, 0.2, 0.3, 0.4 };
    const output = try allocator.alloc(f32, input.len);
    defer allocator.free(output);

    layerNormContiguous(output, &input, &weight, &bias, num_tokens, dim, eps);

    const mean = 2.5;
    const variance = 1.25;
    const inv_std = 1.0 / @sqrt(variance + eps);

    // output[i] = (input[i] - mean) * inv_std * weight[i] + bias[i]
    for (input, 0..) |in_val, i| {
        const expected = (in_val - mean) * inv_std * weight[i] + bias[i];
        try std.testing.expectApproxEqRel(expected, output[i], 1e-5);
    }
}

test "layerNormContiguous weight and bias" {
    const allocator = std.testing.allocator;

    const dim: usize = 4;
    const num_tokens: usize = 1;
    const eps: f32 = 1e-6;

    const input = [_]f32{ -2.0, -1.0, 1.0, 2.0 };
    const weight = [_]f32{ 2.0, 2.0, 2.0, 2.0 };
    const bias = [_]f32{ -1.0, -0.5, 0.5, 1.0 };
    const output = try allocator.alloc(f32, input.len);
    defer allocator.free(output);

    layerNormContiguous(output, &input, &weight, &bias, num_tokens, dim, eps);

    const mean = 0.0;
    // variance = mean([4, 1, 1, 4]) = 2.5
    const variance = 2.5;
    const inv_std = 1.0 / @sqrt(variance + eps);

    for (input, 0..) |in_val, i| {
        const expected = (in_val - mean) * inv_std * weight[i] + bias[i];
        try std.testing.expectApproxEqRel(expected, output[i], 1e-5);
    }
}

test "layerNormContiguous multiple tokens" {
    const allocator = std.testing.allocator;

    const dim: usize = 3;
    const num_tokens: usize = 2;
    const eps: f32 = 1e-6;

    const input = [_]f32{
        1.0, 2.0, 3.0, // Token 0: mean=2, variance=2/3
        4.0, 5.0, 6.0, // Token 1: mean=5, variance=2/3
    };
    const weight = [_]f32{ 1.0, 1.0, 1.0 };
    const output = try allocator.alloc(f32, input.len);
    defer allocator.free(output);

    layerNormContiguous(output, &input, &weight, null, num_tokens, dim, eps);

    // Token 0: mean = 2.0
    const mean_0: f32 = 2.0;
    const variance_0: f32 = ((1.0 - 2.0) * (1.0 - 2.0) + (2.0 - 2.0) * (2.0 - 2.0) + (3.0 - 2.0) * (3.0 - 2.0)) / 3.0;
    const inv_std_0 = 1.0 / @sqrt(variance_0 + eps);

    // Token 1: mean = 5.0
    const mean_1: f32 = 5.0;
    const variance_1: f32 = ((4.0 - 5.0) * (4.0 - 5.0) + (5.0 - 5.0) * (5.0 - 5.0) + (6.0 - 5.0) * (6.0 - 5.0)) / 3.0;
    const inv_std_1 = 1.0 / @sqrt(variance_1 + eps);

    // Verify token 0
    for (0..dim) |i| {
        const expected = (input[i] - mean_0) * inv_std_0;
        try std.testing.expectApproxEqRel(expected, output[i], 1e-5);
    }

    // Verify token 1
    for (0..dim) |i| {
        const idx = dim + i;
        const expected = (input[idx] - mean_1) * inv_std_1;
        try std.testing.expectApproxEqRel(expected, output[idx], 1e-5);
    }
}

test "layerNormContiguous epsilon prevents div zero" {
    const allocator = std.testing.allocator;

    const dim: usize = 4;
    const num_tokens: usize = 1;
    const eps: f32 = 1e-6;

    // All same values - zero variance
    const input = [_]f32{ 5.0, 5.0, 5.0, 5.0 };
    const weight = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const output = try allocator.alloc(f32, input.len);
    defer allocator.free(output);

    layerNormContiguous(output, &input, &weight, null, num_tokens, dim, eps);

    // With zero variance, inv_std = 1/sqrt(eps)
    // output = (5 - 5) * inv_std = 0
    for (output) |val| {
        try std.testing.expectApproxEqAbs(0.0, val, 1e-5);
    }

    // Verify no NaN or Inf
    for (output) |val| {
        try std.testing.expect(std.math.isFinite(val));
    }
}

test "layerNormContiguous large dimension SIMD" {
    const allocator = std.testing.allocator;

    // Use dimension that exercises both SIMD and scalar paths
    const dim: usize = VEC_LEN * 4 + 3;
    const num_tokens: usize = 1;
    const eps: f32 = 1e-6;

    const input = try allocator.alloc(f32, dim);
    defer allocator.free(input);
    const weight = try allocator.alloc(f32, dim);
    defer allocator.free(weight);
    const output = try allocator.alloc(f32, dim);
    defer allocator.free(output);

    // Fill with sequential values
    for (0..dim) |i| {
        input[i] = @floatFromInt(i + 1);
        weight[i] = 1.0;
    }

    layerNormContiguous(output, input, weight, null, num_tokens, dim, eps);

    // Compute expected mean and variance
    var sum: f32 = 0;
    for (input) |val| {
        sum += val;
    }
    const mean = sum / @as(f32, @floatFromInt(dim));

    var sum_sq: f32 = 0;
    for (input) |val| {
        const diff = val - mean;
        sum_sq += diff * diff;
    }
    const variance = sum_sq / @as(f32, @floatFromInt(dim));
    const inv_std = 1.0 / @sqrt(variance + eps);

    // Verify output
    for (input, 0..) |in_val, i| {
        const expected = (in_val - mean) * inv_std;
        try std.testing.expectApproxEqRel(expected, output[i], 1e-4);
    }

    // Verify output has zero mean and unit variance
    var out_mean: f32 = 0;
    for (output) |val| {
        out_mean += val;
    }
    out_mean /= @as(f32, @floatFromInt(dim));
    try std.testing.expectApproxEqAbs(0.0, out_mean, 1e-4);

    var out_variance: f32 = 0;
    for (output) |val| {
        out_variance += val * val;
    }
    out_variance /= @as(f32, @floatFromInt(dim));
    try std.testing.expectApproxEqRel(1.0, out_variance, 1e-4);
}

test "layerNormContiguous vs rmsnormContiguous comparison" {
    const allocator = std.testing.allocator;

    const dim: usize = 4;
    const num_tokens: usize = 1;
    const eps: f32 = 1e-6;

    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const weight = [_]f32{ 1.0, 1.0, 1.0, 1.0 };

    const ln_output = try allocator.alloc(f32, input.len);
    defer allocator.free(ln_output);
    const rms_output = try allocator.alloc(f32, input.len);
    defer allocator.free(rms_output);

    layerNormContiguous(ln_output, &input, &weight, null, num_tokens, dim, eps);
    rmsnormContiguous(rms_output, &input, &weight, null, .f32, num_tokens, dim, eps, 0.0);

    // LayerNorm: centers (zero mean) then scales (unit variance)
    // RMSNorm: only scales by RMS (no centering)
    // These should produce different results

    // LayerNorm output should have zero mean
    var ln_mean: f32 = 0;
    for (ln_output) |val| {
        ln_mean += val;
    }
    ln_mean /= @as(f32, @floatFromInt(dim));
    try std.testing.expectApproxEqAbs(0.0, ln_mean, 1e-5);

    // RMSNorm output should NOT have zero mean (unless input happened to)
    var rms_mean: f32 = 0;
    for (rms_output) |val| {
        rms_mean += val;
    }
    rms_mean /= @as(f32, @floatFromInt(dim));
    // This should not be zero for this input
    try std.testing.expect(@abs(rms_mean) > 0.01);

    // Both should have similar scale (close to 1.0 RMS/std)
    var ln_variance: f32 = 0;
    for (ln_output) |val| {
        ln_variance += val * val;
    }
    ln_variance /= @as(f32, @floatFromInt(dim));
    try std.testing.expectApproxEqRel(1.0, ln_variance, 1e-4);

    var rms_sq: f32 = 0;
    for (rms_output) |val| {
        rms_sq += val * val;
    }
    const rms_rms = @sqrt(rms_sq / @as(f32, @floatFromInt(dim)));
    try std.testing.expectApproxEqRel(1.0, rms_rms, 1e-4);
}

test "rmsnormInPlaceWeightTensor - f32 weights no offset" {
    const allocator = std.testing.allocator;

    const dim: usize = 4;
    var vec = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const eps: f32 = 1e-6;

    var weight_owned = try tensor.OwnedTensor.init(allocator, .f32, &[_]usize{dim});
    defer weight_owned.deinit();
    const weight_slice = weight_owned.asSlice(f32);
    for (weight_slice) |*w| w.* = 1.0;

    const weight_tensor = weight_owned.toTensor();

    // Compute expected RMS
    var sum_sq: f32 = 0;
    for (vec) |v| sum_sq += v * v;
    const expected_rms = @sqrt(sum_sq / @as(f32, @floatFromInt(dim)));
    const expected_inv_rms = 1.0 / expected_rms;

    rmsnormInPlaceWeightTensor(&vec, &weight_tensor, eps, 0.0);

    // With weight=1.0, output should be input / rms
    for (0..dim) |i| {
        const expected = (@as(f32, @floatFromInt(i + 1))) * expected_inv_rms;
        try std.testing.expectApproxEqRel(expected, vec[i], 1e-5);
    }

    // Verify output has RMS close to 1.0
    var out_sum_sq: f32 = 0;
    for (vec) |v| out_sum_sq += v * v;
    const out_rms = @sqrt(out_sum_sq / @as(f32, @floatFromInt(dim)));
    try std.testing.expectApproxEqRel(@as(f32, 1.0), out_rms, 1e-4);
}

test "rmsnormInPlaceWeightTensor - f32 weights with offset" {
    const allocator = std.testing.allocator;

    const dim: usize = 4;
    var vec = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const eps: f32 = 1e-6;
    const weight_offset: f32 = 0.5;

    var weight_owned = try tensor.OwnedTensor.init(allocator, .f32, &[_]usize{dim});
    defer weight_owned.deinit();
    const weight_slice = weight_owned.asSlice(f32);
    for (weight_slice) |*w| w.* = 0.5;

    const weight_tensor = weight_owned.toTensor();

    rmsnormInPlaceWeightTensor(&vec, &weight_tensor, eps, weight_offset);

    // Effective weight = 0.5 + 0.5 = 1.0
    // Output should be similar to no-offset case
    var out_sum_sq: f32 = 0;
    for (vec) |v| out_sum_sq += v * v;
    const out_rms = @sqrt(out_sum_sq / @as(f32, @floatFromInt(dim)));
    try std.testing.expectApproxEqRel(@as(f32, 1.0), out_rms, 1e-4);
}

test "rmsnormInPlaceWeightTensor - bf16 weights" {
    const allocator = std.testing.allocator;

    const dim: usize = 4;
    var vec = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const eps: f32 = 1e-6;

    var weight_owned = try tensor.OwnedTensor.init(allocator, .bf16, &[_]usize{dim});
    defer weight_owned.deinit();
    const weight_slice = weight_owned.asSlice(u16);
    for (weight_slice) |*w| w.* = dtype_mod.f32ToBf16(1.0);

    const weight_tensor = weight_owned.toTensor();

    rmsnormInPlaceWeightTensor(&vec, &weight_tensor, eps, 0.0);

    // Verify output has RMS close to 1.0
    var out_sum_sq: f32 = 0;
    for (vec) |v| out_sum_sq += v * v;
    const out_rms = @sqrt(out_sum_sq / @as(f32, @floatFromInt(dim)));
    try std.testing.expectApproxEqRel(@as(f32, 1.0), out_rms, 1e-3); // Slightly relaxed for bf16
}

test "rmsnormInPlaceWeightTensor - f16 weights" {
    const allocator = std.testing.allocator;

    const dim: usize = 4;
    var vec = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const eps: f32 = 1e-6;

    var weight_owned = try tensor.OwnedTensor.init(allocator, .f16, &[_]usize{dim});
    defer weight_owned.deinit();
    const weight_slice = weight_owned.asSlice(u16);
    for (weight_slice) |*w| w.* = dtype_mod.f32ToFp16(1.0);

    const weight_tensor = weight_owned.toTensor();

    rmsnormInPlaceWeightTensor(&vec, &weight_tensor, eps, 0.0);

    // Verify output has RMS close to 1.0
    var out_sum_sq: f32 = 0;
    for (vec) |v| out_sum_sq += v * v;
    const out_rms = @sqrt(out_sum_sq / @as(f32, @floatFromInt(dim)));
    try std.testing.expectApproxEqRel(@as(f32, 1.0), out_rms, 1e-4);
}

test "rmsnormInPlaceWeightTensor - epsilon prevents division by zero" {
    const allocator = std.testing.allocator;

    const dim: usize = 4;
    var vec = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    const eps: f32 = 1e-6;

    var weight_owned = try tensor.OwnedTensor.init(allocator, .f32, &[_]usize{dim});
    defer weight_owned.deinit();
    const weight_slice = weight_owned.asSlice(f32);
    for (weight_slice) |*w| w.* = 1.0;

    const weight_tensor = weight_owned.toTensor();

    rmsnormInPlaceWeightTensor(&vec, &weight_tensor, eps, 0.0);

    // All outputs should be finite (not NaN or Inf)
    for (vec) |v| {
        try std.testing.expect(std.math.isFinite(v));
    }
}

test "rmsnormInPlaceWeightTensor - non-uniform weights" {
    const allocator = std.testing.allocator;

    const dim: usize = 4;
    var vec = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const weight_data = [_]f32{ 0.5, 1.0, 1.5, 2.0 };
    const eps: f32 = 1e-6;

    var weight_owned = try tensor.OwnedTensor.init(allocator, .f32, &[_]usize{dim});
    defer weight_owned.deinit();
    const weight_slice = weight_owned.asSlice(f32);
    for (weight_data, 0..) |w, i| weight_slice[i] = w;

    const weight_tensor = weight_owned.toTensor();

    const original = vec;

    rmsnormInPlaceWeightTensor(&vec, &weight_tensor, eps, 0.0);

    // Compute expected RMS from original
    var sum_sq: f32 = 0;
    for (original) |v| sum_sq += v * v;
    const inv_rms = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(dim)) + eps);

    // Verify each element is scaled correctly
    for (0..dim) |i| {
        const expected = original[i] * inv_rms * weight_data[i];
        try std.testing.expectApproxEqRel(expected, vec[i], 1e-5);
    }
}

test "rmsnormContiguous basic" {
    const allocator = std.testing.allocator;
    const dim: usize = 4;
    const num_tokens: usize = 1;
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const weight = [_]f32{ 1.0, 1.0, 1.0, 1.0 };

    const output = try allocator.alloc(f32, dim);
    defer allocator.free(output);

    rmsnormContiguous(output, &input, &weight, null, .f32, num_tokens, dim, 1e-5, 0.0);

    // Compute expected RMS norm
    var sum_sq: f32 = 0;
    for (input) |v| sum_sq += v * v;
    const rms = @sqrt(sum_sq / @as(f32, dim) + 1e-5);

    for (input, output) |in, out| {
        const expected = in / rms;
        try std.testing.expectApproxEqRel(expected, out, 1e-4);
    }
}

test "layerNormContiguous basic" {
    const allocator = std.testing.allocator;
    const dim: usize = 4;
    const num_tokens: usize = 1;
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const weight = [_]f32{ 1.0, 1.0, 1.0, 1.0 };

    const output = try allocator.alloc(f32, dim);
    defer allocator.free(output);

    layerNormContiguous(output, &input, &weight, null, num_tokens, dim, 1e-5);

    // Compute expected mean and variance
    var mean: f32 = 0;
    for (input) |v| mean += v;
    mean /= @as(f32, dim);

    var variance: f32 = 0;
    for (input) |v| variance += (v - mean) * (v - mean);
    variance /= @as(f32, dim);

    const inv_std = 1.0 / @sqrt(variance + 1e-5);

    for (input, output) |in, out| {
        const expected = (in - mean) * inv_std;
        try std.testing.expectApproxEqRel(expected, out, 1e-4);
    }
}
