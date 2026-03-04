//! RMSNorm backward pass.
//!
//! Forward: y[i,j] = x[i,j] * inv_rms[i] * (w[j] + weight_offset)
//!   where inv_rms[i] = 1 / sqrt(mean(x[i,:]^2) + eps)
//!
//! Backward computes:
//!   grad_input:  [rows, cols]
//!   grad_weight: [cols]

const std = @import("std");

/// Compute gradients for RMSNorm.
///
/// Given saved forward activations:
///   input:    [rows * cols] — input to RMSNorm
///   inv_rms:  [rows]        — 1/sqrt(mean(x^2) + eps) per row
///   weight:   [cols]        — learnable scale parameter
///
/// Outputs:
///   grad_input:  [rows * cols] — overwritten with input gradient
///   grad_weight: [cols]        — accumulated (not zeroed)
///
/// grad_output: [rows * cols] — gradient from upstream
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
) void {
    std.debug.assert(grad_input.len == rows * cols);
    std.debug.assert(grad_weight.len == cols);
    std.debug.assert(grad_output.len == rows * cols);
    std.debug.assert(input.len == rows * cols);
    std.debug.assert(inv_rms.len == rows);
    std.debug.assert(weight.len == cols);

    const cols_f: f32 = @floatFromInt(cols);

    for (0..rows) |i| {
        const x_row = input[i * cols ..][0..cols];
        const dy_row = grad_output[i * cols ..][0..cols];
        const dx_row = grad_input[i * cols ..][0..cols];
        const rms_inv = inv_rms[i];

        // Compute: sum_j( dy[j] * x[j] * (w[j] + offset) ) * inv_rms
        var dot_dy_xw: f32 = 0.0;
        for (0..cols) |j| {
            dot_dy_xw += dy_row[j] * x_row[j] * (weight[j] + weight_offset);
        }

        // dx[j] = inv_rms * (w[j] + offset) * dy[j]
        //        - inv_rms^3 / cols * x[j] * dot(dy * x * w)
        const coeff = rms_inv * rms_inv * rms_inv / cols_f * dot_dy_xw;
        for (0..cols) |j| {
            dx_row[j] = rms_inv * (weight[j] + weight_offset) * dy_row[j] - coeff * x_row[j];
        }

        // dw[j] += dy[j] * x[j] * inv_rms
        for (0..cols) |j| {
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

    rmsnormBackward(&grad_input, &grad_weight, &grad_output, &input, &inv_rms, &weight, rows, cols, 0.0);

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

    rmsnormBackward(&grad_input, &grad_weight, &grad_output, &input, &inv_rms, &weight, rows, cols, 0.0);

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
    rmsnormBackward(&grad_input, &grad_weight, &grad_output, &input, &inv_rms, &weight, rows, cols, 1.0);

    // With weight_offset=1, effective weight is [1,1]
    // dx[0] = 1.0 * 1.0 * 1.0 - (1*1*1/2)*1 = 1.0 - 0.5 = 0.5
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), grad_input[0], 1e-5);
}
