//! CPU Embedding Kernel
//! Token embedding lookup for various quantization formats
//!
//! This module provides embedding lookup operations for CPU inference.
//! Supports F32 and grouped-affine u4/u8 formats.

pub const supported = true;

const std = @import("std");
const tensor = @import("../../../../tensor.zig");
const dtype = @import("../../../../dtype.zig");
const compute = @import("../../../../compute/root.zig");
const cpu_quant_decode = compute.cpu.quant_decode;
const cpu_tensor_gather = compute.cpu.tensor_gather;

const Tensor = tensor.Tensor;

/// Gather embeddings for a sequence of token IDs.
/// Supports multiple quantization formats.
pub fn gatherEmbeddings(embedding_weights: *const Tensor, token_ids: []const u32, output_tensor: *Tensor) !void {
    // Internal invariants: output must be f32 with matching dimensions
    if (output_tensor.dtype != .f32) {
        return error.InvalidDType;
    }
    const vocab_size: usize = @intCast(embedding_weights.shape[0]);
    const embed_dim: usize = @intCast(embedding_weights.shape[1]);
    if (!(output_tensor.shape[1] == token_ids.len and output_tensor.shape[2] == embedding_weights.shape[1])) {
        return error.InvalidShape;
    }
    const output_values = output_tensor.asSlice(f32);

    switch (embedding_weights.dtype) {
        .f32 => {
            const embedding_weights_data = embedding_weights.asSlice(f32);
            try cpu_tensor_gather.gatherRowsF32(embedding_weights_data, vocab_size, embed_dim, token_ids, output_values);
        },
        .f16 => {
            const embedding_weights_data = embedding_weights.asSliceUnaligned(u16);
            try cpu_quant_decode.gatherDecodeF16Rows(
                embedding_weights_data,
                vocab_size,
                embed_dim,
                token_ids,
                output_values,
            );
        },
        .bf16 => {
            const embedding_weights_data = embedding_weights.asSliceUnaligned(u16);
            try cpu_quant_decode.gatherDecodeBf16Rows(
                embedding_weights_data,
                vocab_size,
                embed_dim,
                token_ids,
                output_values,
            );
        },
        .grouped_affine_u4 => {
            const gaffine_params = embedding_weights.gaffine orelse return error.InvalidShape;
            const group_size = gaffine_params.group_size;
            const scales_dtype = gaffine_params.scales_dtype;
            const scales: []align(1) const u16 = @as([*]align(1) const u16, @ptrCast(gaffine_params.scales.ptr))[0 .. gaffine_params.scales.len / 2];
            const biases: []align(1) const u16 = @as([*]align(1) const u16, @ptrCast(gaffine_params.biases.ptr))[0 .. gaffine_params.biases.len / 2];
            const packed_values: []align(1) const u32 = @as([*]align(1) const u32, @ptrCast(embedding_weights.data().ptr))[0 .. embedding_weights.data().len / 4];
            try cpu_quant_decode.gatherDecodeGroupedAffineU4Rows(
                packed_values,
                scales,
                biases,
                scales_dtype,
                group_size,
                vocab_size,
                embed_dim,
                token_ids,
                output_values,
            );
        },
        .grouped_affine_u8 => {
            const gaffine_params = embedding_weights.gaffine orelse return error.InvalidShape;
            const group_size = gaffine_params.group_size;
            const scales_dtype = gaffine_params.scales_dtype;
            const scales: []align(1) const u16 = @as([*]align(1) const u16, @ptrCast(gaffine_params.scales.ptr))[0 .. gaffine_params.scales.len / 2];
            const biases: []align(1) const u16 = @as([*]align(1) const u16, @ptrCast(gaffine_params.biases.ptr))[0 .. gaffine_params.biases.len / 2];
            const packed_values: []align(1) const u32 = @as([*]align(1) const u32, @ptrCast(embedding_weights.data().ptr))[0 .. embedding_weights.data().len / 4];
            try cpu_quant_decode.gatherDecodeGroupedAffineU8Rows(
                packed_values,
                scales,
                biases,
                scales_dtype,
                group_size,
                vocab_size,
                embed_dim,
                token_ids,
                output_values,
            );
        },
        else => return error.InvalidDType,
    }
}

pub const EmbeddingLookup = struct {
    /// Canonical kernel-call contract for backend parity checks.
    pub const ForwardParams = struct {
        token_ids: []const u32,
        output_tensor: *Tensor,
    };

    embedding_weights: *const Tensor,

    pub fn forward(
        self: *const EmbeddingLookup,
        token_ids: []const u32,
        output_tensor: *Tensor,
    ) !void {
        return gatherEmbeddings(self.embedding_weights, token_ids, output_tensor);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "gatherEmbeddings f32 basic" {
    const allocator = std.testing.allocator;

    // Create a simple embedding table: 4 tokens x 8 dims
    const vocab_size = 4;
    const embed_dim = 8;

    var embed_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ vocab_size, embed_dim });
    defer embed_tensor.deinit();

    const embed_data = embed_tensor.asSlice(f32);
    // Fill with simple pattern: row i has values [i*10, i*10+1, ..., i*10+7]
    for (0..vocab_size) |i| {
        for (0..embed_dim) |j| {
            embed_data[i * embed_dim + j] = @as(f32, @floatFromInt(i * 10 + j));
        }
    }

    // Create output tensor for 3 tokens
    const token_ids = [_]u32{ 0, 2, 1 };
    var output_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ 1, token_ids.len, embed_dim });
    defer output_tensor.deinit();

    // Gather embeddings
    var out_view = output_tensor.view();
    try gatherEmbeddings(&embed_tensor.view(), &token_ids, &out_view);

    const output_values = output_tensor.asSlice(f32);

    // Verify token 0 (row 0)
    for (0..embed_dim) |j| {
        try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(j)), output_values[j], 0.0001);
    }

    // Verify token 2 (row 2)
    for (0..embed_dim) |j| {
        try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(20 + j)), output_values[embed_dim + j], 0.0001);
    }

    // Verify token 1 (row 1)
    for (0..embed_dim) |j| {
        try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(10 + j)), output_values[2 * embed_dim + j], 0.0001);
    }
}

test "gatherEmbeddings f32 boundary tokens" {
    const allocator = std.testing.allocator;

    const vocab_size = 100;
    const embed_dim = 4;

    var embed_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ vocab_size, embed_dim });
    defer embed_tensor.deinit();

    const embed_data = embed_tensor.asSlice(f32);
    for (0..vocab_size) |i| {
        for (0..embed_dim) |j| {
            embed_data[i * embed_dim + j] = @as(f32, @floatFromInt(i * 100 + j));
        }
    }

    // Test first and last token
    const token_ids = [_]u32{ 0, vocab_size - 1 };
    var output_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ 1, token_ids.len, embed_dim });
    defer output_tensor.deinit();

    var out_view = output_tensor.view();
    try gatherEmbeddings(&embed_tensor.view(), &token_ids, &out_view);

    const output_values = output_tensor.asSlice(f32);

    // Verify first token (0)
    for (0..embed_dim) |j| {
        try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(j)), output_values[j], 0.0001);
    }

    // Verify last token (vocab_size-1)
    for (0..embed_dim) |j| {
        try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt((vocab_size - 1) * 100 + j)), output_values[embed_dim + j], 0.0001);
    }
}

test "gatherEmbeddings f16 dtype" {
    const allocator = std.testing.allocator;

    const vocab_size = 3;
    const embed_dim = 4;

    var embed_tensor = try tensor.OwnedTensor.init(allocator, .f16, &.{ vocab_size, embed_dim });
    defer embed_tensor.deinit();

    const embed_data = embed_tensor.view().asSliceUnaligned(u16);
    // Known f16 values: 1.0, 2.0, 0.5, -1.0
    const fp16_vals = [_]u16{ 0x3C00, 0x4000, 0x3800, 0xBC00 }; // 1.0, 2.0, 0.5, -1.0
    const expected_f32 = [_]f32{ 1.0, 2.0, 0.5, -1.0 };

    // Fill embedding table with known values
    for (0..vocab_size) |i| {
        for (0..embed_dim) |j| {
            embed_data[i * embed_dim + j] = fp16_vals[j];
        }
    }

    // Lookup token 1
    const token_ids = [_]u32{1};
    var output_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ 1, token_ids.len, embed_dim });
    defer output_tensor.deinit();

    var out_view = output_tensor.view();
    try gatherEmbeddings(&embed_tensor.view(), &token_ids, &out_view);

    const output_values = output_tensor.asSlice(f32);

    // Verify conversion accuracy
    for (0..embed_dim) |j| {
        try std.testing.expectApproxEqAbs(expected_f32[j], output_values[j], 0.0001);
    }
}

test "gatherEmbeddings bf16 dtype" {
    const allocator = std.testing.allocator;

    const vocab_size = 2;
    const embed_dim = 4;

    var embed_tensor = try tensor.OwnedTensor.init(allocator, .bf16, &.{ vocab_size, embed_dim });
    defer embed_tensor.deinit();

    const embed_data = embed_tensor.view().asSliceUnaligned(u16);
    // Known bf16 values: 1.0, -2.0, 3.5, 0.0
    const bf16_vals = [_]u16{ 0x3F80, 0xC000, 0x4060, 0x0000 }; // 1.0, -2.0, 3.5, 0.0
    const expected_f32 = [_]f32{ 1.0, -2.0, 3.5, 0.0 };

    for (0..vocab_size) |i| {
        for (0..embed_dim) |j| {
            embed_data[i * embed_dim + j] = bf16_vals[j];
        }
    }

    const token_ids = [_]u32{0};
    var output_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ 1, token_ids.len, embed_dim });
    defer output_tensor.deinit();

    var out_view = output_tensor.view();
    try gatherEmbeddings(&embed_tensor.view(), &token_ids, &out_view);

    const output_values = output_tensor.asSlice(f32);

    for (0..embed_dim) |j| {
        try std.testing.expectApproxEqAbs(expected_f32[j], output_values[j], 0.001);
    }
}

test "gatherEmbeddings gaffine_u4 scale bias" {
    // Grouped affine quantization requires complex setup with custom data buffers
    // Skipping for now to focus on standard formats
    // TODO: Add grouped_affine tests with proper buffer allocation lint:ignore no-todo
}

test "gatherEmbeddings gaffine_u8" {
    // Grouped affine quantization requires complex setup with custom data buffers
    // Skipping for now to focus on standard formats
    // TODO: Add grouped_affine tests with proper buffer allocation lint:ignore no-todo
}

test "EmbeddingLookup.forward delegates to gatherEmbeddings" {
    const allocator = std.testing.allocator;

    var embed_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ 3, 2 });
    defer embed_tensor.deinit();
    const data = embed_tensor.asSlice(f32);
    data[0] = 1.0;
    data[1] = 2.0;
    data[2] = 3.0;
    data[3] = 4.0;
    data[4] = 5.0;
    data[5] = 6.0;

    var output_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ 1, 1, 2 });
    defer output_tensor.deinit();

    const token_ids = [_]u32{2};
    var embed_view = embed_tensor.view();
    var out_view = output_tensor.view();
    const lookup = EmbeddingLookup{ .embedding_weights = &embed_view };
    try lookup.forward(&token_ids, &out_view);

    try std.testing.expectEqualSlices(f32, &.{ 5.0, 6.0 }, output_tensor.asSlice(f32));
}

test "gatherEmbeddings invalid token" {
    const allocator = std.testing.allocator;

    const vocab_size = 10;
    const embed_dim = 4;

    var embed_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ vocab_size, embed_dim });
    defer embed_tensor.deinit();

    // Try to lookup token beyond vocab
    const token_ids = [_]u32{vocab_size}; // Out of bounds
    var output_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ 1, token_ids.len, embed_dim });
    defer output_tensor.deinit();

    var out_view = output_tensor.view();
    const result = gatherEmbeddings(&embed_tensor.view(), &token_ids, &out_view);
    try std.testing.expectError(error.InvalidTokenId, result);
}

test "gatherEmbeddings dimension validation" {
    const allocator = std.testing.allocator;

    const vocab_size = 10;
    const embed_dim = 8;

    var embed_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ vocab_size, embed_dim });
    defer embed_tensor.deinit();

    const token_ids = [_]u32{0};

    // Output tensor with wrong dimension
    var output_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ 1, token_ids.len, 4 }); // Wrong dim
    defer output_tensor.deinit();

    var out_view = output_tensor.view();
    const result = gatherEmbeddings(&embed_tensor.view(), &token_ids, &out_view);
    try std.testing.expectError(error.InvalidShape, result);
}

test "gatherEmbeddings single element" {
    const allocator = std.testing.allocator;

    const vocab_size = 5;
    const embed_dim = 1; // Edge case: single element

    var embed_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ vocab_size, embed_dim });
    defer embed_tensor.deinit();

    const embed_data = embed_tensor.asSlice(f32);
    for (0..vocab_size) |i| {
        embed_data[i] = @as(f32, @floatFromInt(i)) * 10.0;
    }

    const token_ids = [_]u32{ 0, 3, 4 };
    var output_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ 1, token_ids.len, embed_dim });
    defer output_tensor.deinit();

    var out_view = output_tensor.view();
    try gatherEmbeddings(&embed_tensor.view(), &token_ids, &out_view);

    const output_values = output_tensor.asSlice(f32);

    try std.testing.expectApproxEqAbs(0.0, output_values[0], 0.001);
    try std.testing.expectApproxEqAbs(30.0, output_values[1], 0.001);
    try std.testing.expectApproxEqAbs(40.0, output_values[2], 0.001);
}

test "gatherEmbeddings large vocab" {
    const allocator = std.testing.allocator;

    const vocab_size = 50000; // Realistic vocab size
    const embed_dim = 8;

    var embed_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ vocab_size, embed_dim });
    defer embed_tensor.deinit();

    const embed_data = embed_tensor.asSlice(f32);
    // Just set specific rows we'll test
    const test_token: usize = 12345;
    for (0..embed_dim) |j| {
        embed_data[test_token * embed_dim + j] = @as(f32, @floatFromInt(test_token + j));
    }

    const token_ids = [_]u32{test_token};
    var output_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ 1, token_ids.len, embed_dim });
    defer output_tensor.deinit();

    var out_view = output_tensor.view();
    try gatherEmbeddings(&embed_tensor.view(), &token_ids, &out_view);

    const output_values = output_tensor.asSlice(f32);

    for (0..embed_dim) |j| {
        try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(test_token + j)), output_values[j], 0.001);
    }
}

test "gatherEmbeddings multiple sequences" {
    const allocator = std.testing.allocator;

    const vocab_size = 10;
    const embed_dim = 4;

    var embed_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ vocab_size, embed_dim });
    defer embed_tensor.deinit();

    const embed_data = embed_tensor.asSlice(f32);
    for (0..vocab_size) |i| {
        for (0..embed_dim) |j| {
            embed_data[i * embed_dim + j] = @as(f32, @floatFromInt(i * 10 + j));
        }
    }

    // Lookup sequence of tokens
    const token_ids = [_]u32{ 0, 1, 2, 3, 4 };
    var output_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ 1, token_ids.len, embed_dim });
    defer output_tensor.deinit();

    var out_view = output_tensor.view();
    try gatherEmbeddings(&embed_tensor.view(), &token_ids, &out_view);

    const output_values = output_tensor.asSlice(f32);

    // Verify each token in sequence
    for (0..token_ids.len) |tok_idx| {
        const token_id = token_ids[tok_idx];
        for (0..embed_dim) |j| {
            const expected = @as(f32, @floatFromInt(token_id * 10 + j));
            try std.testing.expectApproxEqAbs(expected, output_values[tok_idx * embed_dim + j], 0.001);
        }
    }
}

test "gatherEmbeddings f16 special values" {
    const allocator = std.testing.allocator;

    const vocab_size = 1;
    const embed_dim = 4;

    var embed_tensor = try tensor.OwnedTensor.init(allocator, .f16, &.{ vocab_size, embed_dim });
    defer embed_tensor.deinit();

    const embed_data = embed_tensor.view().asSliceUnaligned(u16);
    // Test: 0.0, positive zero, small denormal, large value
    embed_data[0] = 0x0000; // +0.0
    embed_data[1] = 0x8000; // -0.0
    embed_data[2] = 0x0001; // Smallest denormal
    embed_data[3] = 0x7BFF; // Max normal (~65504)

    const token_ids = [_]u32{0};
    var output_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ 1, token_ids.len, embed_dim });
    defer output_tensor.deinit();

    var out_view = output_tensor.view();
    try gatherEmbeddings(&embed_tensor.view(), &token_ids, &out_view);

    const output_values = output_tensor.asSlice(f32);

    try std.testing.expectApproxEqAbs(0.0, output_values[0], 0.0001);
    try std.testing.expectApproxEqAbs(0.0, @abs(output_values[1]), 0.0001);
    try std.testing.expect(output_values[2] > 0.0 and output_values[2] < 0.0001);
    try std.testing.expect(output_values[3] > 60000.0 and output_values[3] < 70000.0);
}

test "gatherEmbeddings empty sequence" {
    const allocator = std.testing.allocator;

    const vocab_size = 10;
    const embed_dim = 4;

    var embed_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ vocab_size, embed_dim });
    defer embed_tensor.deinit();

    // Empty token sequence
    const token_ids = [_]u32{};
    var output_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ 1, token_ids.len, embed_dim });
    defer output_tensor.deinit();

    var out_view = output_tensor.view();
    try gatherEmbeddings(&embed_tensor.view(), &token_ids, &out_view);

    // Should succeed with no embeddings gathered
}

test "gatherEmbeddings duplicate tokens" {
    const allocator = std.testing.allocator;

    const vocab_size = 5;
    const embed_dim = 4;

    var embed_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ vocab_size, embed_dim });
    defer embed_tensor.deinit();

    const embed_data = embed_tensor.asSlice(f32);
    for (0..vocab_size) |i| {
        for (0..embed_dim) |j| {
            embed_data[i * embed_dim + j] = @as(f32, @floatFromInt(i * 100 + j));
        }
    }

    // Duplicate token IDs: same token appears multiple times
    const token_ids = [_]u32{ 2, 2, 2, 1, 2 };
    var output_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ 1, token_ids.len, embed_dim });
    defer output_tensor.deinit();

    var out_view = output_tensor.view();
    try gatherEmbeddings(&embed_tensor.view(), &token_ids, &out_view);

    const output_values = output_tensor.asSlice(f32);

    // Verify each duplicate correctly retrieves the same embedding
    // Token 2 appears at indices 0, 1, 2, 4
    for ([_]usize{ 0, 1, 2, 4 }) |idx| {
        for (0..embed_dim) |j| {
            const expected = @as(f32, @floatFromInt(2 * 100 + j));
            try std.testing.expectApproxEqAbs(expected, output_values[idx * embed_dim + j], 0.001);
        }
    }

    // Token 1 appears at index 3
    for (0..embed_dim) |j| {
        const expected = @as(f32, @floatFromInt(1 * 100 + j));
        try std.testing.expectApproxEqAbs(expected, output_values[3 * embed_dim + j], 0.001);
    }
}

test "gatherEmbeddings large dimension" {
    const allocator = std.testing.allocator;

    const vocab_size = 2;
    const embed_dim = 4096; // Large but realistic (e.g., LLaMA-2 70B)

    var embed_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ vocab_size, embed_dim });
    defer embed_tensor.deinit();

    const embed_data = embed_tensor.asSlice(f32);
    // Only populate first and last positions for testing
    embed_data[0] = 1.0; // First element of token 0
    embed_data[embed_dim - 1] = 2.0; // Last element of token 0
    embed_data[embed_dim] = 3.0; // First element of token 1
    embed_data[2 * embed_dim - 1] = 4.0; // Last element of token 1

    const token_ids = [_]u32{ 0, 1 };
    var output_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ 1, token_ids.len, embed_dim });
    defer output_tensor.deinit();

    var out_view = output_tensor.view();
    try gatherEmbeddings(&embed_tensor.view(), &token_ids, &out_view);

    const output_values = output_tensor.asSlice(f32);

    // Verify boundary values
    try std.testing.expectApproxEqAbs(1.0, output_values[0], 0.001);
    try std.testing.expectApproxEqAbs(2.0, output_values[embed_dim - 1], 0.001);
    try std.testing.expectApproxEqAbs(3.0, output_values[embed_dim], 0.001);
    try std.testing.expectApproxEqAbs(4.0, output_values[2 * embed_dim - 1], 0.001);
}

test "gatherEmbeddings all zeros" {
    const allocator = std.testing.allocator;

    const vocab_size = 3;
    const embed_dim = 8;

    var embed_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ vocab_size, embed_dim });
    defer embed_tensor.deinit();

    const embed_data = embed_tensor.asSlice(f32);
    @memset(embed_data, 0.0);

    const token_ids = [_]u32{ 0, 1, 2 };
    var output_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ 1, token_ids.len, embed_dim });
    defer output_tensor.deinit();

    var out_view = output_tensor.view();
    try gatherEmbeddings(&embed_tensor.view(), &token_ids, &out_view);

    const output_values = output_tensor.asSlice(f32);

    // All output should be zeros
    for (0..token_ids.len * embed_dim) |i| {
        try std.testing.expectApproxEqAbs(0.0, output_values[i], 0.0001);
    }
}

test "gatherEmbeddings extreme f32" {
    const allocator = std.testing.allocator;

    const vocab_size = 1;
    const embed_dim = 8;

    var embed_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ vocab_size, embed_dim });
    defer embed_tensor.deinit();

    const embed_data = embed_tensor.asSlice(f32);
    embed_data[0] = std.math.floatMax(f32); // Max positive
    embed_data[1] = std.math.floatMin(f32); // Min positive (denormal)
    embed_data[2] = -std.math.floatMax(f32); // Max negative
    embed_data[3] = 0.0; // Positive zero
    embed_data[4] = -0.0; // Negative zero
    embed_data[5] = 1.0e-38; // Small positive
    embed_data[6] = -1.0e-38; // Small negative
    embed_data[7] = 1.234567e20; // Large value

    const token_ids = [_]u32{0};
    var output_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ 1, token_ids.len, embed_dim });
    defer output_tensor.deinit();

    var out_view = output_tensor.view();
    try gatherEmbeddings(&embed_tensor.view(), &token_ids, &out_view);

    const output_values = output_tensor.asSlice(f32);

    // Verify extreme values are preserved exactly
    try std.testing.expectEqual(std.math.floatMax(f32), output_values[0]);
    try std.testing.expectEqual(std.math.floatMin(f32), output_values[1]);
    try std.testing.expectEqual(-std.math.floatMax(f32), output_values[2]);
    try std.testing.expectEqual(@as(f32, 0.0), output_values[3]);
    try std.testing.expectApproxEqAbs(0.0, output_values[4], 0.0);
    try std.testing.expectApproxEqAbs(1.0e-38, output_values[5], 1.0e-45);
    try std.testing.expectApproxEqAbs(-1.0e-38, output_values[6], 1.0e-45);
    try std.testing.expectApproxEqAbs(1.234567e20, output_values[7], 1.0e12);
}

test "gatherEmbeddings sequential tokens" {
    const allocator = std.testing.allocator;

    const vocab_size = 100;
    const embed_dim = 16;

    var embed_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ vocab_size, embed_dim });
    defer embed_tensor.deinit();

    const embed_data = embed_tensor.asSlice(f32);
    for (0..vocab_size) |i| {
        for (0..embed_dim) |j| {
            embed_data[i * embed_dim + j] = @as(f32, @floatFromInt(i + j));
        }
    }

    // Sequential tokens from 10 to 19
    const token_ids = [_]u32{ 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };
    var output_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ 1, token_ids.len, embed_dim });
    defer output_tensor.deinit();

    var out_view = output_tensor.view();
    try gatherEmbeddings(&embed_tensor.view(), &token_ids, &out_view);

    const output_values = output_tensor.asSlice(f32);

    // Verify each token's embedding
    for (token_ids, 0..) |token_id, tok_idx| {
        for (0..embed_dim) |j| {
            const expected = @as(f32, @floatFromInt(token_id + j));
            try std.testing.expectApproxEqAbs(expected, output_values[tok_idx * embed_dim + j], 0.001);
        }
    }
}

test "gatherEmbeddings reverse order" {
    const allocator = std.testing.allocator;

    const vocab_size = 10;
    const embed_dim = 4;

    var embed_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ vocab_size, embed_dim });
    defer embed_tensor.deinit();

    const embed_data = embed_tensor.asSlice(f32);
    for (0..vocab_size) |i| {
        for (0..embed_dim) |j| {
            embed_data[i * embed_dim + j] = @as(f32, @floatFromInt(i * 10 + j));
        }
    }

    // Reverse order: 9, 8, 7, ..., 0
    const token_ids = [_]u32{ 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 };
    var output_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ 1, token_ids.len, embed_dim });
    defer output_tensor.deinit();

    var out_view = output_tensor.view();
    try gatherEmbeddings(&embed_tensor.view(), &token_ids, &out_view);

    const output_values = output_tensor.asSlice(f32);

    // Verify order is preserved in output
    for (token_ids, 0..) |token_id, tok_idx| {
        for (0..embed_dim) |j| {
            const expected = @as(f32, @floatFromInt(token_id * 10 + j));
            try std.testing.expectApproxEqAbs(expected, output_values[tok_idx * embed_dim + j], 0.001);
        }
    }
}

test "gatherEmbeddings bf16 multiple tokens" {
    const allocator = std.testing.allocator;

    const vocab_size = 5;
    const embed_dim = 8;

    var embed_tensor = try tensor.OwnedTensor.init(allocator, .bf16, &.{ vocab_size, embed_dim });
    defer embed_tensor.deinit();

    const embed_data = embed_tensor.view().asSliceUnaligned(u16);

    // Fill with distinct patterns per row
    for (0..vocab_size) |i| {
        for (0..embed_dim) |j| {
            const val = @as(f32, @floatFromInt(i)) + @as(f32, @floatFromInt(j)) * 0.1;
            embed_data[i * embed_dim + j] = dtype.f32ToBf16(val);
        }
    }

    const token_ids = [_]u32{ 0, 2, 4 };
    var output_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ 1, token_ids.len, embed_dim });
    defer output_tensor.deinit();

    var out_view = output_tensor.view();
    try gatherEmbeddings(&embed_tensor.view(), &token_ids, &out_view);

    const output_values = output_tensor.asSlice(f32);

    // Verify each token
    for (token_ids, 0..) |token_id, tok_idx| {
        for (0..embed_dim) |j| {
            const expected = @as(f32, @floatFromInt(token_id)) + @as(f32, @floatFromInt(j)) * 0.1;
            // BF16 has only ~7 bits mantissa, precision loss can be ~1.5% of value
            // For value 4.4, error is ~0.03, so use 5% tolerance
            try std.testing.expectApproxEqAbs(expected, output_values[tok_idx * embed_dim + j], 0.05);
        }
    }
}

test "gatherEmbeddings f16 negative positive" {
    const allocator = std.testing.allocator;

    const vocab_size = 2;
    const embed_dim = 8;

    var embed_tensor = try tensor.OwnedTensor.init(allocator, .f16, &.{ vocab_size, embed_dim });
    defer embed_tensor.deinit();

    const embed_data = embed_tensor.view().asSliceUnaligned(u16);

    // Token 0: positive values
    for (0..embed_dim) |j| {
        embed_data[j] = dtype.f32ToFp16(@as(f32, @floatFromInt(j)) + 1.0);
    }

    // Token 1: negative values
    for (0..embed_dim) |j| {
        embed_data[embed_dim + j] = dtype.f32ToFp16(-@as(f32, @floatFromInt(j)) - 1.0);
    }

    const token_ids = [_]u32{ 0, 1 };
    var output_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ 1, token_ids.len, embed_dim });
    defer output_tensor.deinit();

    var out_view = output_tensor.view();
    try gatherEmbeddings(&embed_tensor.view(), &token_ids, &out_view);

    const output_values = output_tensor.asSlice(f32);

    // Verify token 0 (positive values)
    for (0..embed_dim) |j| {
        const expected = @as(f32, @floatFromInt(j)) + 1.0;
        try std.testing.expectApproxEqAbs(expected, output_values[j], 0.001);
    }

    // Verify token 1 (negative values)
    for (0..embed_dim) |j| {
        const expected = -@as(f32, @floatFromInt(j)) - 1.0;
        try std.testing.expectApproxEqAbs(expected, output_values[embed_dim + j], 0.001);
    }
}

test "gatherEmbeddings wrong dtype" {
    const allocator = std.testing.allocator;

    const vocab_size = 10;
    const embed_dim = 4;

    var embed_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ vocab_size, embed_dim });
    defer embed_tensor.deinit();

    const token_ids = [_]u32{0};

    // Output tensor with wrong dtype (i32 instead of f32)
    var output_tensor = try tensor.OwnedTensor.init(allocator, .i32, &.{ 1, token_ids.len, embed_dim });
    defer output_tensor.deinit();

    var out_view = output_tensor.view();
    const result = gatherEmbeddings(&embed_tensor.view(), &token_ids, &out_view);
    try std.testing.expectError(error.InvalidDType, result);
}

test "gatherEmbeddings mismatched length" {
    const allocator = std.testing.allocator;

    const vocab_size = 10;
    const embed_dim = 4;

    var embed_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ vocab_size, embed_dim });
    defer embed_tensor.deinit();

    const token_ids = [_]u32{ 0, 1, 2 }; // 3 tokens

    // Output tensor sized for 5 tokens instead of 3
    var output_tensor = try tensor.OwnedTensor.init(allocator, .f32, &.{ 1, 5, embed_dim });
    defer output_tensor.deinit();

    var out_view = output_tensor.view();
    const result = gatherEmbeddings(&embed_tensor.view(), &token_ids, &out_view);
    try std.testing.expectError(error.InvalidShape, result);
}
