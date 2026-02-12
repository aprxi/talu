//! Integration tests for inference.MultiHeadAttention
//!
//! Tests the MultiHeadAttention struct which performs multi-head self-attention.
//! Note: Full forward pass tests require loaded weights.

const std = @import("std");
const main = @import("main");

const MultiHeadAttention = main.inference.backend.MultiHeadAttention;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "MultiHeadAttention type is accessible" {
    const T = MultiHeadAttention;
    _ = T;
}

test "MultiHeadAttention has expected fields" {
    const info = @typeInfo(MultiHeadAttention);
    try std.testing.expect(info == .@"struct");

    const fields = info.@"struct".fields;
    var has_n_heads = false;
    var has_head_dim = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "n_heads")) has_n_heads = true;
        if (comptime std.mem.eql(u8, field.name, "head_dim")) has_head_dim = true;
    }

    try std.testing.expect(has_n_heads);
    try std.testing.expect(has_head_dim);
}
