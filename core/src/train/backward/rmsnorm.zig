//! RMSNorm backward pass.
//!
//! Forward: y[i,j] = x[i,j] * inv_rms[i] * (w[j] + weight_offset)
//!   where inv_rms[i] = 1 / sqrt(mean(x[i,:]^2) + eps)
//!
//! Backward computes:
//!   grad_input:  [rows, cols]
//!   grad_weight: [cols]

const std = @import("std");
const compute = @import("../../compute/root.zig");

const simd = compute.cpu.simd.arch;
const VEC = simd.f32_vec_len;
const F32Vec = simd.F32Vec;

/// Compute gradients for RMSNorm with optional fused residual addition.
///
/// Given saved forward activations:
///   input:    [rows * cols] — input to RMSNorm
///   inv_rms:  [rows]        — 1/sqrt(mean(x^2) + eps) per row
///   weight:   [cols]        — learnable scale parameter
///
/// Outputs:
///   grad_input:  [rows * cols] — overwritten with input gradient (+ residual_grad if provided)
///   grad_weight: [cols]        — accumulated (not zeroed)
///
/// grad_output:   [rows * cols] — gradient from upstream
/// residual_grad: optional [rows * cols] — if provided, fused into grad_input (dx += residual)
pub fn rmsnormBackward(
    grad_input: []f32,
    grad_weight: []f32,
    grad_output: []const f32,
    input: []const f32,
    inv_rms: []const f32,
    weight: []const f32,
    rows: usize,
    cols: usize,
    weight_offset: f32,
    residual_grad: ?[]const f32,
) void {
    @setFloatMode(.optimized);
    std.debug.assert(grad_input.len == rows * cols);
    std.debug.assert(grad_weight.len == cols);
    std.debug.assert(grad_output.len == rows * cols);
    std.debug.assert(input.len == rows * cols);
    std.debug.assert(inv_rms.len == rows);
    std.debug.assert(weight.len == cols);
    if (residual_grad) |rg| std.debug.assert(rg.len == rows * cols);

    const cols_f: f32 = @floatFromInt(cols);
    const wo_v: F32Vec = @splat(weight_offset);

    for (0..rows) |i| {
        const x_row = input[i * cols ..][0..cols];
        const dy_row = grad_output[i * cols ..][0..cols];
        const dx_row = grad_input[i * cols ..][0..cols];
        const rms_inv = inv_rms[i];

        // SIMD dot product: sum(dy * x * (w + offset))
        var dot_vec: F32Vec = @splat(0.0);
        var j: usize = 0;
        while (j + VEC <= cols) : (j += VEC) {
            const dy: F32Vec = dy_row[j..][0..VEC].*;
            const x: F32Vec = x_row[j..][0..VEC].*;
            const w: F32Vec = weight[j..][0..VEC].*;
            dot_vec = @mulAdd(F32Vec, dy * x, w + wo_v, dot_vec);
        }
        var dot_dy_xw = @reduce(.Add, dot_vec);
        while (j < cols) : (j += 1) {
            dot_dy_xw += dy_row[j] * x_row[j] * (weight[j] + weight_offset);
        }

        // SIMD grad_input: dx = irms * (w + offset) * dy - coeff * x [+ residual]
        const coeff = rms_inv * rms_inv * rms_inv / cols_f * dot_dy_xw;
        const irms_v: F32Vec = @splat(rms_inv);
        const coeff_v: F32Vec = @splat(coeff);
        j = 0;
        if (residual_grad) |rg| {
            const res_row = rg[i * cols ..][0..cols];
            while (j + VEC <= cols) : (j += VEC) {
                const dy: F32Vec = dy_row[j..][0..VEC].*;
                const x: F32Vec = x_row[j..][0..VEC].*;
                const w: F32Vec = weight[j..][0..VEC].*;
                const r: F32Vec = res_row[j..][0..VEC].*;
                dx_row[j..][0..VEC].* = irms_v * (w + wo_v) * dy - coeff_v * x + r;
            }
            while (j < cols) : (j += 1) {
                dx_row[j] = rms_inv * (weight[j] + weight_offset) * dy_row[j] - coeff * x_row[j] + res_row[j];
            }
        } else {
            while (j + VEC <= cols) : (j += VEC) {
                const dy: F32Vec = dy_row[j..][0..VEC].*;
                const x: F32Vec = x_row[j..][0..VEC].*;
                const w: F32Vec = weight[j..][0..VEC].*;
                dx_row[j..][0..VEC].* = irms_v * (w + wo_v) * dy - coeff_v * x;
            }
            while (j < cols) : (j += 1) {
                dx_row[j] = rms_inv * (weight[j] + weight_offset) * dy_row[j] - coeff * x_row[j];
            }
        }

        // SIMD grad_weight: dw += dy * x * irms
        j = 0;
        while (j + VEC <= cols) : (j += VEC) {
            const dy: F32Vec = dy_row[j..][0..VEC].*;
            const x: F32Vec = x_row[j..][0..VEC].*;
            var dw: F32Vec = grad_weight[j..][0..VEC].*;
            dw = @mulAdd(F32Vec, dy * x, irms_v, dw);
            grad_weight[j..][0..VEC].* = dw;
        }
        while (j < cols) : (j += 1) {
            grad_weight[j] += dy_row[j] * x_row[j] * rms_inv;
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

test "rmsnormBackward gradient row sums are consistent" {
    // Simple case: input = [1,1,1,1], weight = [1,1,1,1], weight_offset = 0
    // inv_rms = 1/sqrt(mean(1^2) + 0) = 1
    // y = [1,1,1,1]
    const cols: usize = 4;
    const rows: usize = 1;

    const input = [_]f32{ 1, 1, 1, 1 };
    const inv_rms = [_]f32{1.0};
    const weight = [_]f32{ 1, 1, 1, 1 };
    const grad_output = [_]f32{ 1, 0, 0, 0 };

    var grad_input: [4]f32 = undefined;
    var grad_weight = [_]f32{ 0, 0, 0, 0 };

    rmsnormBackward(&grad_input, &grad_weight, &grad_output, &input, &inv_rms, &weight, rows, cols, 0.0, null);

    // grad_weight[0] should be non-zero (target of grad_output)
    try std.testing.expect(grad_weight[0] != 0.0);

    // grad_input should not be all zeros
    var any_nonzero = false;
    for (grad_input) |v| {
        if (v != 0.0) {
            any_nonzero = true;
            break;
        }
    }
    try std.testing.expect(any_nonzero);
}

test "rmsnormBackward weight gradient accumulates across rows" {
    const cols: usize = 2;
    const rows: usize = 2;

    // Two identical rows
    const input = [_]f32{ 1, 2, 1, 2 };
    const inv_rms = [_]f32{ 0.5, 0.5 };
    const weight = [_]f32{ 1, 1 };
    const grad_output = [_]f32{ 1, 1, 1, 1 };

    var grad_input: [4]f32 = undefined;
    var grad_weight = [_]f32{ 0, 0 };

    rmsnormBackward(&grad_input, &grad_weight, &grad_output, &input, &inv_rms, &weight, rows, cols, 0.0, null);

    // dw[j] = sum_i(dy[i,j] * x[i,j] * inv_rms[i])
    // dw[0] = 1*1*0.5 + 1*1*0.5 = 1.0
    // dw[1] = 1*2*0.5 + 1*2*0.5 = 2.0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), grad_weight[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), grad_weight[1], 1e-5);
}

test "rmsnormBackward with weight_offset" {
    const cols: usize = 2;
    const rows: usize = 1;

    const input = [_]f32{ 1, 0 };
    const inv_rms = [_]f32{1.0};
    const weight = [_]f32{ 0, 0 };
    const grad_output = [_]f32{ 1, 0 };

    var grad_input: [2]f32 = undefined;
    var grad_weight = [_]f32{ 0, 0 };

    // weight_offset = 1.0 means effective weight = [1, 1]
    rmsnormBackward(&grad_input, &grad_weight, &grad_output, &input, &inv_rms, &weight, rows, cols, 1.0, null);

    // With weight_offset=1, effective weight is [1,1]
    // dx[0] = 1.0 * 1.0 * 1.0 - (1*1*1/2)*1 = 1.0 - 0.5 = 0.5
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), grad_input[0], 1e-5);
}
