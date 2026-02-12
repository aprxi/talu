//! Fused attention helpers for CPU backend.
//! Provides utilities to project QKV with a single matmul and split views.

const std = @import("std");
const tensor = @import("../../../../tensor.zig");
const compute = @import("../../../../compute/root.zig");
const matmul = compute.ops.matmul;

const Tensor = tensor.Tensor;
const MatmulFn = matmul.MatmulFn;

/// Resulting views from a fused QKV projection.
pub const QkvViews = struct {
    q: Tensor,
    k: Tensor,
    v: Tensor,
};

/// Run one matmul against concatenated QKV weights and split into Q/K/V views.
/// After matmul, output is [seq, total_dim] where total_dim = q_dim + 2*kv_dim.
/// Each row contains [Q..., K..., V...] which we split along the feature dimension.
///
/// `qkv_buffer` must be at least 2 * seq * total_dim to allow for rearrangement.
/// Layout after this function:
///   [0 .. seq*q_dim]: Q (contiguous)
///   [seq*q_dim .. seq*(q_dim+kv_dim)]: K (contiguous)
///   [seq*(q_dim+kv_dim) .. seq*total_dim]: V (contiguous)
pub inline fn projectQkv(
    input_tensor: *const Tensor,
    fused_weights: *const Tensor,
    qkv_buffer: []f32,
    seq_len: usize,
    query_dim: usize,
    kv_dim: usize,
    matmul_kernel: MatmulFn,
    matmul_scratch: *matmul.MatmulScratch,
) QkvViews {
    const total_dim = query_dim + 2 * kv_dim;
    const total_elems = seq_len * total_dim;
    // We need space for matmul output + final rearranged output
    // Use second half of buffer for matmul, then copy to first half
    std.debug.assert(qkv_buffer.len >= 2 * total_elems);

    // Matmul into second half of buffer
    const matmul_output = qkv_buffer[total_elems .. 2 * total_elems];
    var fused_output = Tensor.view2DSlice(matmul_output, seq_len, total_dim);
    matmul_kernel(input_tensor, fused_weights, &fused_output, matmul_scratch);

    // Rearrange: from [row0: Q0 K0 V0, row1: Q1 K1 V1, ...]
    //              to [Q0 Q1 ..., K0 K1 ..., V0 V1 ...]
    // Since we're copying from second half to first half, no overlap issues
    const output_buffer = qkv_buffer[0..total_elems];

    for (0..seq_len) |token_idx| {
        const src_base = token_idx * total_dim;

        // Copy Q
        const q_dst = token_idx * query_dim;
        const q_src = src_base;
        @memcpy(output_buffer[q_dst..][0..query_dim], matmul_output[q_src..][0..query_dim]);

        // Copy K
        const k_dst = seq_len * query_dim + token_idx * kv_dim;
        const k_src = src_base + query_dim;
        @memcpy(output_buffer[k_dst..][0..kv_dim], matmul_output[k_src..][0..kv_dim]);

        // Copy V
        const v_dst = seq_len * query_dim + seq_len * kv_dim + token_idx * kv_dim;
        const v_src = src_base + query_dim + kv_dim;
        @memcpy(output_buffer[v_dst..][0..kv_dim], matmul_output[v_src..][0..kv_dim]);
    }

    return .{
        .q = Tensor.view2DSlice(output_buffer[0 .. seq_len * query_dim], seq_len, query_dim),
        .k = Tensor.view2DSlice(output_buffer[seq_len * query_dim .. seq_len * query_dim + seq_len * kv_dim], seq_len, kv_dim),
        .v = Tensor.view2DSlice(output_buffer[seq_len * query_dim + seq_len * kv_dim .. seq_len * total_dim], seq_len, kv_dim),
    };
}

// =============================================================================
// Unit Tests
// =============================================================================

test "projectQkv: basic rearrangement from [Q K V] per-row to contiguous blocks" {
    const allocator = std.testing.allocator;

    // Setup dimensions: seq=2, q_dim=4, kv_dim=2
    const seq_len = 2;
    const q_dim = 4;
    const kv_dim = 2;
    const total_dim = q_dim + 2 * kv_dim; // 8
    const d_model = 4;

    // Create input tensor [seq, d_model]
    var input_owned = try tensor.OwnedTensor.init(allocator, .f32, &[_]usize{ seq_len, d_model });
    defer input_owned.deinit();

    // Fill input with recognizable values
    const input_data = input_owned.asSlice(f32);
    for (input_data, 0..) |*v, i| {
        v.* = @floatFromInt(i + 1);
    }

    // Create fused weight tensor [d_model, total_dim] (transposed layout for matmul)
    // matmul computes: input [seq, d_model] @ weight [d_model, total_dim] -> output [seq, total_dim]
    var weight_owned = try tensor.OwnedTensor.init(allocator, .f32, &[_]usize{ d_model, total_dim });
    defer weight_owned.deinit();

    // Fill weights with identity-like values for predictable output
    const weight_data = weight_owned.asSlice(f32);
    @memset(weight_data, 0.0);
    // Make each output just copy corresponding input element (identity-ish)
    for (0..d_model) |in_idx| {
        if (in_idx < total_dim) {
            // Diagonal for first d_model outputs
            weight_data[in_idx * total_dim + in_idx] = 1.0;
        }
    }

    // Create QKV buffer (needs 2 * seq * total_dim)
    const buffer_size = 2 * seq_len * total_dim;
    const qkv_buffer = try allocator.alloc(f32, buffer_size);
    defer allocator.free(qkv_buffer);

    // Create matmul scratch
    var matmul_scratch = try matmul.MatmulScratch.init(allocator);
    defer matmul_scratch.deinit();

    // Get matmul kernel
    const dk = try matmul.matmulKernel(.f32);

    const input_tensor = input_owned.toTensor();
    const weight_tensor = weight_owned.toTensor();

    // Call projectQkv
    const views = projectQkv(
        &input_tensor,
        &weight_tensor,
        qkv_buffer,
        seq_len,
        q_dim,
        kv_dim,
        dk.func,
        &matmul_scratch,
    );

    // Verify output shapes
    try std.testing.expectEqual(@as(usize, seq_len), @as(usize, @intCast(views.q.shape[0])));
    try std.testing.expectEqual(@as(usize, q_dim), @as(usize, @intCast(views.q.shape[1])));
    try std.testing.expectEqual(@as(usize, seq_len), @as(usize, @intCast(views.k.shape[0])));
    try std.testing.expectEqual(@as(usize, kv_dim), @as(usize, @intCast(views.k.shape[1])));
    try std.testing.expectEqual(@as(usize, seq_len), @as(usize, @intCast(views.v.shape[0])));
    try std.testing.expectEqual(@as(usize, kv_dim), @as(usize, @intCast(views.v.shape[1])));

    // Verify Q, K, V views are contiguous and don't overlap
    const q_data = views.q.asSlice(f32);
    const k_data = views.k.asSlice(f32);
    const v_data = views.v.asSlice(f32);

    try std.testing.expectEqual(@as(usize, seq_len * q_dim), q_data.len);
    try std.testing.expectEqual(@as(usize, seq_len * kv_dim), k_data.len);
    try std.testing.expectEqual(@as(usize, seq_len * kv_dim), v_data.len);

    // Verify memory layout: Q, K, V should be in separate contiguous regions
    // Q starts at offset 0
    // K starts at offset seq_len * q_dim
    // V starts at offset seq_len * q_dim + seq_len * kv_dim
    const q_start = @intFromPtr(q_data.ptr) - @intFromPtr(qkv_buffer.ptr);
    const k_start = @intFromPtr(k_data.ptr) - @intFromPtr(qkv_buffer.ptr);
    const v_start = @intFromPtr(v_data.ptr) - @intFromPtr(qkv_buffer.ptr);

    try std.testing.expectEqual(@as(usize, 0), q_start / @sizeOf(f32));
    try std.testing.expectEqual(@as(usize, seq_len * q_dim), k_start / @sizeOf(f32));
    try std.testing.expectEqual(@as(usize, seq_len * q_dim + seq_len * kv_dim), v_start / @sizeOf(f32));
}

test "projectQkv: single token sequence" {
    const allocator = std.testing.allocator;

    const seq_len = 1;
    const q_dim = 8;
    const kv_dim = 4;
    const total_dim = q_dim + 2 * kv_dim;
    const d_model = 8;

    var input_owned = try tensor.OwnedTensor.init(allocator, .f32, &[_]usize{ seq_len, d_model });
    defer input_owned.deinit();
    @memset(input_owned.asSlice(f32), 1.0);

    // Weight tensor: [d_model, total_dim] for matmul
    var weight_owned = try tensor.OwnedTensor.init(allocator, .f32, &[_]usize{ d_model, total_dim });
    defer weight_owned.deinit();
    @memset(weight_owned.asSlice(f32), 0.1);

    const buffer_size = 2 * seq_len * total_dim;
    const qkv_buffer = try allocator.alloc(f32, buffer_size);
    defer allocator.free(qkv_buffer);

    var matmul_scratch = try matmul.MatmulScratch.init(allocator);
    defer matmul_scratch.deinit();

    const dk = try matmul.matmulKernel(.f32);

    const input_tensor = input_owned.toTensor();
    const weight_tensor = weight_owned.toTensor();

    const views = projectQkv(
        &input_tensor,
        &weight_tensor,
        qkv_buffer,
        seq_len,
        q_dim,
        kv_dim,
        dk.func,
        &matmul_scratch,
    );

    // Single token should still produce correct shapes
    try std.testing.expectEqual(@as(usize, 1), @as(usize, @intCast(views.q.shape[0])));
    try std.testing.expectEqual(@as(usize, q_dim), @as(usize, @intCast(views.q.shape[1])));
    try std.testing.expectEqual(@as(usize, 1), @as(usize, @intCast(views.k.shape[0])));
    try std.testing.expectEqual(@as(usize, kv_dim), @as(usize, @intCast(views.k.shape[1])));
    try std.testing.expectEqual(@as(usize, 1), @as(usize, @intCast(views.v.shape[0])));
    try std.testing.expectEqual(@as(usize, kv_dim), @as(usize, @intCast(views.v.shape[1])));
}
