//! Integration tests for tokenizer.TokenizeBytesResult
//!
//! TokenizeBytesResult contains the raw byte data and offsets for tokenized text.
//! It uses a CSR-style layout where token i spans data[offsets[i]..offsets[i+1]].

const std = @import("std");
const main = @import("main");

const TokenizeBytesResult = main.tokenizer.TokenizeBytesResult;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "TokenizeBytesResult type is accessible" {
    const T = TokenizeBytesResult;
    _ = T;
}

test "TokenizeBytesResult is a struct" {
    const info = @typeInfo(TokenizeBytesResult);
    try std.testing.expect(info == .@"struct");
}

test "TokenizeBytesResult has expected fields" {
    const info = @typeInfo(TokenizeBytesResult);
    const fields = info.@"struct".fields;

    var has_data = false;
    var has_offsets = false;
    var has_allocator = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "data")) has_data = true;
        if (comptime std.mem.eql(u8, field.name, "offsets")) has_offsets = true;
        if (comptime std.mem.eql(u8, field.name, "allocator")) has_allocator = true;
    }

    try std.testing.expect(has_data);
    try std.testing.expect(has_offsets);
    try std.testing.expect(has_allocator);
}

// =============================================================================
// Method Tests
// =============================================================================

test "TokenizeBytesResult has deinit method" {
    try std.testing.expect(@hasDecl(TokenizeBytesResult, "deinit"));
}

// =============================================================================
// deinit Tests
// =============================================================================

test "TokenizeBytesResult.deinit is safe on default-initialized struct" {
    const allocator = std.testing.allocator;
    var result = TokenizeBytesResult{
        .data = &.{},
        .offsets = &.{},
        .allocator = allocator,
    };
    result.deinit();
}
