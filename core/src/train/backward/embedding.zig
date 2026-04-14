//! Embedding backward: scatter-add gradients into embedding table gradient.
//!
//! The forward pass is a table lookup: output[b] = embedding_table[token_id[b]].
//! The backward pass scatters the output gradient back to the corresponding rows
//! of the embedding table gradient.

const std = @import("std");
const compute = @import("compute_pkg");

const simd = compute.cpu.simd.arch;
const VEC = simd.f32_vec_len;
const F32Vec = simd.F32Vec;

/// Scatter-add grad_output into grad_embedding at indices given by token_ids.
///
/// For each batch element b:
///   grad_embedding[token_ids[b], :] += grad_output[b, :]
///
/// grad_embedding: [vocab_size * d_model] — accumulated (not zeroed).
/// grad_output:    [batch_size * d_model]
/// token_ids:      [batch_size]
pub fn embeddingBackward(
    grad_embedding: []f32,
    grad_output: []const f32,
    token_ids: []const u32,
    batch_size: usize,
    d_model: usize,
) void {
    @setFloatMode(.optimized);
    std.debug.assert(grad_output.len == batch_size * d_model);
    std.debug.assert(token_ids.len == batch_size);

    for (0..batch_size) |b| {
        const token_id: usize = token_ids[b];
        const emb_row = grad_embedding[token_id * d_model ..][0..d_model];
        const grad_row = grad_output[b * d_model ..][0..d_model];

        var i: usize = 0;
        while (i + VEC <= d_model) : (i += VEC) {
            var e: F32Vec = emb_row[i..][0..VEC].*;
            const g: F32Vec = grad_row[i..][0..VEC].*;
            e += g;
            emb_row[i..][0..VEC].* = e;
        }
        while (i < d_model) : (i += 1) {
            emb_row[i] += grad_row[i];
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

test "embeddingBackward scatters gradients to correct rows" {
    // vocab=4, d_model=3, batch=2
    // token_ids = [1, 3]
    // grad_output = [[1,2,3], [4,5,6]]
    // Expected: row 1 += [1,2,3], row 3 += [4,5,6]
    var grad_emb = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }; // 4x3
    const grad_out = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const token_ids = [_]u32{ 1, 3 };

    embeddingBackward(&grad_emb, &grad_out, &token_ids, 2, 3);

    // Row 0: unchanged
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), grad_emb[0], 1e-6);
    // Row 1: [1, 2, 3]
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), grad_emb[3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), grad_emb[4], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), grad_emb[5], 1e-6);
    // Row 2: unchanged
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), grad_emb[6], 1e-6);
    // Row 3: [4, 5, 6]
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), grad_emb[9], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), grad_emb[10], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), grad_emb[11], 1e-6);
}

test "embeddingBackward accumulates for duplicate tokens" {
    // batch=3, vocab=2, d_model=2
    // token_ids = [0, 0, 1]
    // grad_output = [[1,1], [2,2], [3,3]]
    // Expected: row 0 = [1+2, 1+2] = [3, 3], row 1 = [3, 3]
    var grad_emb = [_]f32{ 0, 0, 0, 0 }; // 2x2
    const grad_out = [_]f32{ 1, 1, 2, 2, 3, 3 };
    const token_ids = [_]u32{ 0, 0, 1 };

    embeddingBackward(&grad_emb, &grad_out, &token_ids, 3, 2);

    try std.testing.expectApproxEqAbs(@as(f32, 3.0), grad_emb[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), grad_emb[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), grad_emb[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), grad_emb[3], 1e-6);
}

test "embeddingBackward accumulates with existing values" {
    // Pre-filled gradient, should add not overwrite
    var grad_emb = [_]f32{ 10, 20 }; // 1x2, pre-filled
    const grad_out = [_]f32{ 1, 2 };
    const token_ids = [_]u32{0};

    embeddingBackward(&grad_emb, &grad_out, &token_ids, 1, 2);

    try std.testing.expectApproxEqAbs(@as(f32, 11.0), grad_emb[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 22.0), grad_emb[1], 1e-6);
}
