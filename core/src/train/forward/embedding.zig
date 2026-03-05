//! Token embedding lookup for training forward pass.

const std = @import("std");

/// Token embedding lookup.
/// output[i * d .. (i+1) * d] = embedding[tokens[i] * d .. (tokens[i]+1) * d]
pub fn embeddingForward(output: []f32, embedding: []const f32, tokens: []const u32, d_model: usize) void {
    for (tokens, 0..) |token, i| {
        const src = embedding[token * d_model ..][0..d_model];
        const dst = output[i * d_model ..][0..d_model];
        @memcpy(dst, src);
    }
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

test "embeddingForward looks up correct rows" {
    // Embedding: 4 tokens, d_model=3
    const embed = [_]f32{
        0.1, 0.2, 0.3, // token 0
        0.4, 0.5, 0.6, // token 1
        0.7, 0.8, 0.9, // token 2
        1.0, 1.1, 1.2, // token 3
    };
    const tokens = [_]u32{ 2, 0, 3 };
    var output: [9]f32 = undefined;

    embeddingForward(&output, &embed, &tokens, 3);

    // token 2 → [0.7, 0.8, 0.9]
    try testing.expectApproxEqAbs(@as(f32, 0.7), output[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.8), output[1], 1e-6);
    // token 0 → [0.1, 0.2, 0.3]
    try testing.expectApproxEqAbs(@as(f32, 0.1), output[3], 1e-6);
    // token 3 → [1.0, 1.1, 1.2]
    try testing.expectApproxEqAbs(@as(f32, 1.0), output[6], 1e-6);
}
