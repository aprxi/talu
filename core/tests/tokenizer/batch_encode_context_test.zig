//! Integration tests for tokenizer.BatchEncodeContext
//!
//! BatchEncodeContext provides context for parallel batch encoding operations.
//! It tracks tokenizer state, text inputs, and encoded results.

const std = @import("std");
const main = @import("main");

const BatchEncodeContext = main.tokenizer.BatchEncodeContext;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "BatchEncodeContext type is accessible" {
    const T = BatchEncodeContext;
    _ = T;
}

test "BatchEncodeContext is a struct" {
    const info = @typeInfo(BatchEncodeContext);
    try std.testing.expect(info == .@"struct");
}

test "BatchEncodeContext has expected fields" {
    const info = @typeInfo(BatchEncodeContext);
    const fields = info.@"struct".fields;

    var has_tokenizer = false;
    var has_text_ptrs = false;
    var has_text_lengths = false;
    var has_add_special_tokens = false;
    var has_encoded_batches = false;
    var has_had_error_flag = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "tokenizer")) has_tokenizer = true;
        if (comptime std.mem.eql(u8, field.name, "text_ptrs")) has_text_ptrs = true;
        if (comptime std.mem.eql(u8, field.name, "text_lengths")) has_text_lengths = true;
        if (comptime std.mem.eql(u8, field.name, "add_special_tokens")) has_add_special_tokens = true;
        if (comptime std.mem.eql(u8, field.name, "encoded_batches")) has_encoded_batches = true;
        if (comptime std.mem.eql(u8, field.name, "had_error_flag")) has_had_error_flag = true;
    }

    try std.testing.expect(has_tokenizer);
    try std.testing.expect(has_text_ptrs);
    try std.testing.expect(has_text_lengths);
    try std.testing.expect(has_add_special_tokens);
    try std.testing.expect(has_encoded_batches);
    try std.testing.expect(has_had_error_flag);
}

// =============================================================================
// Method Tests
// =============================================================================

test "BatchEncodeContext has init method" {
    try std.testing.expect(@hasDecl(BatchEncodeContext, "init"));
}

test "BatchEncodeContext has deinit method" {
    try std.testing.expect(@hasDecl(BatchEncodeContext, "deinit"));
}
