//! Linear projection forward pass for training.
//!
//! Computes output = input @ weight^T using matmulF32 from compute.cpu.linalg.

const std = @import("std");
const tensor_mod = @import("../../tensor.zig");
const compute = @import("../../compute/root.zig");

const Tensor = tensor_mod.Tensor;
const MatmulScratch = compute.cpu.linalg.MatmulScratch;
const matmulF32TransB = compute.cpu.linalg.matmulF32TransB;

/// Linear layer forward: output = input @ weight^T
/// input: [rows, in_dim], weight: [out_dim, in_dim] → output: [rows, out_dim]
pub fn linearForward(output: []f32, input: []const f32, weight: []const f32, rows: usize, in_dim: usize, out_dim: usize, scratch: *MatmulScratch) void {
    std.debug.assert(output.len >= rows * out_dim);
    std.debug.assert(input.len >= rows * in_dim);
    std.debug.assert(weight.len >= out_dim * in_dim);

    // C = A @ B^T where A=[rows, in_dim], B=[out_dim, in_dim] → C=[rows, out_dim]
    // Weight is already stored [out_dim, in_dim] — no transpose needed.
    var a = Tensor.view2DSlice(@constCast(input[0 .. rows * in_dim]), rows, in_dim);
    var b = Tensor.view2DSlice(@constCast(weight[0 .. out_dim * in_dim]), out_dim, in_dim);
    var out = Tensor.view2DSlice(output[0 .. rows * out_dim], rows, out_dim);
    matmulF32TransB(&a, &b, &out, scratch);
}

/// LM head forward: logits[i, v] = sum_d(input[i, d] * weight[v, d])
/// Same as linearForward but separated for clarity.
pub fn lmHeadForward(logits: []f32, input: []const f32, weight: []const f32, rows: usize, d_model: usize, vocab_size: usize, scratch: *MatmulScratch) void {
    linearForward(logits, input, weight, rows, d_model, vocab_size, scratch);
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

test "linearForward computes input @ weight^T" {
    // input: [2, 3], weight: [2, 3] (out_dim=2, in_dim=3)
    const input = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const weight = [_]f32{
        1, 0, 0, // row 0: select dim 0
        0, 1, 0, // row 1: select dim 1
    };
    var output: [4]f32 = undefined;
    var scratch = try MatmulScratch.init(testing.allocator);
    defer scratch.deinit();

    linearForward(&output, &input, &weight, 2, 3, 2, &scratch);

    // Row 0: [1,2,3] @ [[1,0],[0,1],[0,0]] = [1, 2]
    try testing.expectApproxEqAbs(@as(f32, 1.0), output[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 2.0), output[1], 1e-5);
    // Row 1: [4,5,6] @ ... = [4, 5]
    try testing.expectApproxEqAbs(@as(f32, 4.0), output[2], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 5.0), output[3], 1e-5);
}
