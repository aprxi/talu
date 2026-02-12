//! Integration tests for inference.TransformerBlock
//!
//! Tests the TransformerBlock struct which contains weights and config for one
//! transformer layer. Note: Full forward pass tests require a loaded model.

const std = @import("std");
const main = @import("main");

const TransformerBlock = main.inference.backend.TransformerBlock;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "TransformerBlock type is accessible" {
    const T = TransformerBlock;
    _ = T;
}

test "TransformerBlock has expected fields" {
    const info = @typeInfo(TransformerBlock);
    try std.testing.expect(info == .@"struct");

    const fields = info.@"struct".fields;
    var has_attention = false;
    var has_ln1 = false;
    var has_ln2 = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "attention")) has_attention = true;
        if (comptime std.mem.eql(u8, field.name, "ln1")) has_ln1 = true;
        if (comptime std.mem.eql(u8, field.name, "ln2")) has_ln2 = true;
    }

    try std.testing.expect(has_attention);
    try std.testing.expect(has_ln1);
    try std.testing.expect(has_ln2);
}
