//! Integration tests for tokenizer.OffsetsResult
//!
//! OffsetsResult contains token offsets mapping each token back to its
//! position in the original source text (UTF-8 byte indices).

const std = @import("std");
const main = @import("main");

const OffsetsResult = main.tokenizer.OffsetsResult;
const TokenOffset = main.tokenizer.TokenOffset;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "OffsetsResult type is accessible" {
    const T = OffsetsResult;
    _ = T;
}

test "OffsetsResult is a struct" {
    const info = @typeInfo(OffsetsResult);
    try std.testing.expect(info == .@"struct");
}

test "OffsetsResult has expected fields" {
    const info = @typeInfo(OffsetsResult);
    const fields = info.@"struct".fields;

    var has_offsets = false;
    var has_allocator = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "offsets")) has_offsets = true;
        if (comptime std.mem.eql(u8, field.name, "allocator")) has_allocator = true;
    }

    try std.testing.expect(has_offsets);
    try std.testing.expect(has_allocator);
}

// =============================================================================
// Method Tests
// =============================================================================

test "OffsetsResult has deinit method" {
    try std.testing.expect(@hasDecl(OffsetsResult, "deinit"));
}

// =============================================================================
// deinit Tests
// =============================================================================

test "OffsetsResult.deinit is safe on default-initialized struct" {
    const allocator = std.testing.allocator;
    var result = OffsetsResult{
        .offsets = &.{},
        .allocator = allocator,
    };
    result.deinit();
}

// =============================================================================
// TokenOffset Tests
// =============================================================================

test "TokenOffset type is accessible" {
    const T = TokenOffset;
    _ = T;
}

test "TokenOffset is an extern struct" {
    const info = @typeInfo(TokenOffset);
    try std.testing.expect(info == .@"struct");
    try std.testing.expect(info.@"struct".layout == .@"extern");
}

test "TokenOffset has start and end fields" {
    const info = @typeInfo(TokenOffset);
    const fields = info.@"struct".fields;

    var has_start = false;
    var has_end = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "start")) has_start = true;
        if (comptime std.mem.eql(u8, field.name, "end")) has_end = true;
    }

    try std.testing.expect(has_start);
    try std.testing.expect(has_end);
}

test "TokenOffset can be created with values" {
    const offset = TokenOffset{ .start = 0, .end = 5 };
    try std.testing.expectEqual(@as(u32, 0), offset.start);
    try std.testing.expectEqual(@as(u32, 5), offset.end);
}
