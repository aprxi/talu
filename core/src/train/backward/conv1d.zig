//! Causal depthwise 1D convolution backward pass.
//!
//! Forward: for each channel independently,
//!   y[ch, t] = sum_{k=0}^{kernel_size-1} weight[ch, k] * x[ch, t - kernel_size + 1 + k] + bias[ch]
//! where x values before time 0 are zero (causal padding).
//!
//! Backward computes gradients for input, weight, and bias.

const std = @import("std");

/// Depthwise causal conv1d backward for a single channel across a sequence.
///
/// Computes:
///   grad_input:  [seq_len]       — overwritten
///   grad_weight: [kernel_size]   — accumulated
///   grad_bias:   [1]             — accumulated
///   grad_output: [seq_len]
///   input:       [seq_len]       — saved from forward
///   weight:      [kernel_size]
pub fn conv1dBackward(
    grad_input: []f32,
    grad_weight: []f32,
    grad_bias: []f32,
    grad_output: []const f32,
    input: []const f32,
    weight: []const f32,
    seq_len: usize,
    kernel_size: usize,
) void {
    std.debug.assert(grad_input.len == seq_len);
    std.debug.assert(grad_weight.len == kernel_size);
    std.debug.assert(grad_bias.len >= 1);
    std.debug.assert(grad_output.len == seq_len);
    std.debug.assert(input.len == seq_len);
    std.debug.assert(weight.len == kernel_size);

    @memset(grad_input, 0);

    for (0..seq_len) |t| {
        const dy = grad_output[t];

        // Bias gradient: dB += dy
        grad_bias[0] += dy;

        // For each kernel position k:
        //   y[t] used x[t - kernel_size + 1 + k] * w[k]
        for (0..kernel_size) |k| {
            const input_idx_signed: i64 = @as(i64, @intCast(t)) - @as(i64, @intCast(kernel_size)) + 1 + @as(i64, @intCast(k));
            if (input_idx_signed >= 0 and input_idx_signed < @as(i64, @intCast(seq_len))) {
                const input_idx: usize = @intCast(input_idx_signed);

                // Weight gradient: dW[k] += dy * x[input_idx]
                grad_weight[k] += dy * input[input_idx];

                // Input gradient: dx[input_idx] += dy * w[k]
                grad_input[input_idx] += dy * weight[k];
            }
        }
    }
}

/// Batched depthwise conv1d backward across all channels.
///
/// Each channel has its own weight and bias (depthwise separable).
///
/// grad_input:  [channels * seq_len]   — overwritten
/// grad_weight: [channels * kernel_size] — accumulated
/// grad_bias:   [channels]             — accumulated
/// grad_output: [channels * seq_len]
/// input:       [channels * seq_len]
/// weight:      [channels * kernel_size]
pub fn conv1dBackwardBatched(
    grad_input: []f32,
    grad_weight: []f32,
    grad_bias: []f32,
    grad_output: []const f32,
    input: []const f32,
    weight: []const f32,
    channels: usize,
    seq_len: usize,
    kernel_size: usize,
) void {
    for (0..channels) |ch| {
        conv1dBackward(
            grad_input[ch * seq_len ..][0..seq_len],
            grad_weight[ch * kernel_size ..][0..kernel_size],
            grad_bias[ch..][0..1],
            grad_output[ch * seq_len ..][0..seq_len],
            input[ch * seq_len ..][0..seq_len],
            weight[ch * kernel_size ..][0..kernel_size],
            seq_len,
            kernel_size,
        );
    }
}

// =============================================================================
// Tests
// =============================================================================

test "conv1dBackward identity kernel" {
    // kernel = [1], so y[t] = x[t], backward is identity
    var grad_input = [_]f32{ 0, 0, 0 };
    var grad_weight = [_]f32{0};
    var grad_bias = [_]f32{0};
    const grad_output = [_]f32{ 1.0, 2.0, 3.0 };
    const input = [_]f32{ 10.0, 20.0, 30.0 };
    const weight = [_]f32{1.0};

    conv1dBackward(&grad_input, &grad_weight, &grad_bias, &grad_output, &input, &weight, 3, 1);

    // dx = dy (since w=[1])
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), grad_input[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), grad_input[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), grad_input[2], 1e-5);

    // dW = sum(dy * x) = 1*10 + 2*20 + 3*30 = 140
    try std.testing.expectApproxEqAbs(@as(f32, 140.0), grad_weight[0], 1e-5);

    // dB = sum(dy) = 6
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), grad_bias[0], 1e-5);
}

test "conv1dBackward kernel_size=2 causal" {
    // kernel=[0.5, 0.5], causal: y[0] = 0.5*0 + 0.5*x[0], y[1] = 0.5*x[0] + 0.5*x[1]
    var grad_input = [_]f32{ 0, 0 };
    var grad_weight = [_]f32{ 0, 0 };
    var grad_bias = [_]f32{0};
    const grad_output = [_]f32{ 1.0, 1.0 };
    const input = [_]f32{ 2.0, 3.0 };
    const weight = [_]f32{ 0.5, 0.5 };

    conv1dBackward(&grad_input, &grad_weight, &grad_bias, &grad_output, &input, &weight, 2, 2);

    // dx[0]: contributes to y[0] via w[1] and y[1] via w[0]
    //   = dy[0]*w[1] + dy[1]*w[0] = 1*0.5 + 1*0.5 = 1.0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), grad_input[0], 1e-5);

    // dx[1]: contributes to y[1] via w[1]
    //   = dy[1]*w[1] = 1*0.5 = 0.5
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), grad_input[1], 1e-5);
}

test "conv1dBackwardBatched processes channels independently" {
    // 2 channels, seq_len=2, kernel_size=1
    var grad_input = [_]f32{ 0, 0, 0, 0 };
    var grad_weight = [_]f32{ 0, 0 };
    var grad_bias = [_]f32{ 0, 0 };
    const grad_output = [_]f32{ 1, 2, 3, 4 };
    const input = [_]f32{ 1, 1, 1, 1 };
    const weight = [_]f32{ 2, 3 };

    conv1dBackwardBatched(&grad_input, &grad_weight, &grad_bias, &grad_output, &input, &weight, 2, 2, 1);

    // Channel 0: dx = dy * w[0] = [1*2, 2*2] = [2, 4]
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), grad_input[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), grad_input[1], 1e-5);
    // Channel 1: dx = dy * w[0] = [3*3, 4*3] = [9, 12]
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), grad_input[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), grad_input[3], 1e-5);
}
