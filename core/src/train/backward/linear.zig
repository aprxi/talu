//! Backward pass for linear layers (Y = X @ W^T).
//!
//! Given grad_output [batch, out_dim]:
//!   grad_weight += grad_output^T @ input    -> [out_dim, in_dim]
//!   grad_input   = grad_output  @ weight    -> [batch, in_dim]
//!   grad_bias   += sum(grad_output, dim=0)  -> [out_dim]
//!
//! All functions accumulate into gradient buffers (do NOT zero first).
//! Reuses matmulF32 from compute/cpu for SIMD-optimized matmul.

const std = @import("std");
const tensor_mod = @import("../../tensor.zig");
const compute = @import("../../compute/root.zig");

const Tensor = tensor_mod.Tensor;
const MatmulScratch = compute.cpu.linalg.MatmulScratch;
const matmulF32 = compute.cpu.linalg.matmulF32;

/// Compute gradient w.r.t. weight: dW += grad_out^T @ input.
///
/// grad_weight: [out_dim, in_dim] — accumulated (not zeroed).
/// grad_output: [batch, out_dim]
/// input:       [batch, in_dim]
pub fn gradWeight(
    grad_weight: []f32,
    grad_output: []const f32,
    input: []const f32,
    batch_size: usize,
    out_dim: usize,
    in_dim: usize,
    scratch: *MatmulScratch,
) void {
    _ = scratch;
    std.debug.assert(grad_weight.len == out_dim * in_dim);
    std.debug.assert(grad_output.len == batch_size * out_dim);
    std.debug.assert(input.len == batch_size * in_dim);

    // dW = grad_out^T @ input
    // grad_out^T: [out_dim, batch], input: [batch, in_dim] -> [out_dim, in_dim]
    //
    // matmulF32 expects: a[M,K] @ b[K,N] -> out[M,N]
    // We need: grad_out^T[out_dim, batch] @ input[batch, in_dim] -> [out_dim, in_dim]
    //
    // For accumulation, we compute into a temporary and add.
    // Direct approach: iterate and accumulate.

    // Simple row-wise accumulation: for each output row o,
    // dW[o, :] += sum over batch b of grad_output[b, o] * input[b, :]
    for (0..out_dim) |o| {
        const dw_row = grad_weight[o * in_dim ..][0..in_dim];
        for (0..batch_size) |b| {
            const g = grad_output[b * out_dim + o];
            const in_row = input[b * in_dim ..][0..in_dim];
            for (dw_row, in_row) |*dw, inp| {
                dw.* += g * inp;
            }
        }
    }
}

/// Compute gradient w.r.t. input: dX = grad_out @ W.
///
/// grad_input:  [batch, in_dim] — overwritten (not accumulated).
/// grad_output: [batch, out_dim]
/// weight:      [out_dim, in_dim]
pub fn gradInput(
    grad_input: []f32,
    grad_output: []const f32,
    weight: []const f32,
    batch_size: usize,
    out_dim: usize,
    in_dim: usize,
    scratch: *MatmulScratch,
) void {
    std.debug.assert(grad_input.len == batch_size * in_dim);
    std.debug.assert(grad_output.len == batch_size * out_dim);
    std.debug.assert(weight.len == out_dim * in_dim);

    // dX = grad_out @ W
    // grad_out: [batch, out_dim], W: [out_dim, in_dim] -> [batch, in_dim]
    //
    // Use matmulF32: a[batch, out_dim] @ b[out_dim, in_dim] -> out[batch, in_dim]
    var a = Tensor.view2DSlice(@constCast(grad_output), batch_size, out_dim);
    var b = Tensor.view2DSlice(@constCast(weight), out_dim, in_dim);
    var out = Tensor.view2DSlice(grad_input, batch_size, in_dim);
    matmulF32(&a, &b, &out, scratch);
}

/// Compute gradient w.r.t. bias: dBias += sum(grad_out, dim=0).
///
/// grad_bias:   [out_dim] — accumulated (not zeroed).
/// grad_output: [batch, out_dim]
pub fn gradBias(
    grad_bias: []f32,
    grad_output: []const f32,
    batch_size: usize,
    out_dim: usize,
) void {
    std.debug.assert(grad_bias.len == out_dim);
    std.debug.assert(grad_output.len == batch_size * out_dim);

    for (0..batch_size) |b| {
        const row = grad_output[b * out_dim ..][0..out_dim];
        for (grad_bias, row) |*gb, g| {
            gb.* += g;
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

test "gradWeight accumulates dW = grad_out^T @ input" {
    // batch=2, out_dim=2, in_dim=3
    // grad_out = [[1, 0], [0, 1]]  (2x2)
    // input    = [[1, 2, 3], [4, 5, 6]]  (2x3)
    // dW = grad_out^T @ input = [[1,0],[0,1]]^T @ [[1,2,3],[4,5,6]]
    //    = [[1, 2, 3], [4, 5, 6]]
    var dw = [_]f32{ 0, 0, 0, 0, 0, 0 };
    const grad_out = [_]f32{ 1, 0, 0, 1 };
    const input = [_]f32{ 1, 2, 3, 4, 5, 6 };

    var scratch = try MatmulScratch.init(std.testing.allocator);
    defer scratch.deinit();

    gradWeight(&dw, &grad_out, &input, 2, 2, 3, &scratch);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), dw[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), dw[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), dw[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), dw[3], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), dw[4], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), dw[5], 1e-5);
}

test "gradWeight accumulates (calling twice doubles values)" {
    var dw = [_]f32{ 0, 0 };
    const grad_out = [_]f32{2.0};
    const input = [_]f32{ 3.0, 4.0 };

    var scratch = try MatmulScratch.init(std.testing.allocator);
    defer scratch.deinit();

    gradWeight(&dw, &grad_out, &input, 1, 1, 2, &scratch);
    gradWeight(&dw, &grad_out, &input, 1, 1, 2, &scratch);

    // dW = 2 * [2*3, 2*4] = [12, 16]
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), dw[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 16.0), dw[1], 1e-5);
}

test "gradInput computes dX = grad_out @ W" {
    // batch=1, out_dim=2, in_dim=3
    // grad_out = [[1, 2]]  (1x2)
    // W = [[1, 0, 0], [0, 1, 0]]  (2x3)
    // dX = [[1,2]] @ [[1,0,0],[0,1,0]] = [[1, 2, 0]]
    var dx = [_]f32{ 0, 0, 0 };
    const grad_out = [_]f32{ 1, 2 };
    const weight = [_]f32{ 1, 0, 0, 0, 1, 0 };

    var scratch = try MatmulScratch.init(std.testing.allocator);
    defer scratch.deinit();

    gradInput(&dx, &grad_out, &weight, 1, 2, 3, &scratch);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), dx[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), dx[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), dx[2], 1e-5);
}

test "gradBias accumulates sum over batch" {
    // batch=3, out_dim=2
    // grad_out = [[1, 2], [3, 4], [5, 6]]
    // dBias = [1+3+5, 2+4+6] = [9, 12]
    var db = [_]f32{ 0, 0 };
    const grad_out = [_]f32{ 1, 2, 3, 4, 5, 6 };

    gradBias(&db, &grad_out, 3, 2);

    try std.testing.expectApproxEqAbs(@as(f32, 9.0), db[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), db[1], 1e-5);
}
