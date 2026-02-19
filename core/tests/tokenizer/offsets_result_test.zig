//! Integration tests for tokenizer offset-related types.
//!
//! `OffsetsResult` was replaced by `tokenizer.Encoding`, which carries ids,
//! offsets, and masks in one owner-managed struct.

const std = @import("std");
const main = @import("main");

const Encoding = main.tokenizer.Encoding;
const TokenOffset = main.tokenizer.TokenOffset;

test "Encoding type is accessible" {
    const T = Encoding;
    _ = T;
}

test "Encoding has expected fields" {
    const fields = @typeInfo(Encoding).@"struct".fields;

    var has_ids = false;
    var has_offsets = false;
    var has_attention_mask = false;
    var has_special_tokens_mask = false;
    var has_allocator = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "ids")) has_ids = true;
        if (comptime std.mem.eql(u8, field.name, "offsets")) has_offsets = true;
        if (comptime std.mem.eql(u8, field.name, "attention_mask")) has_attention_mask = true;
        if (comptime std.mem.eql(u8, field.name, "special_tokens_mask")) has_special_tokens_mask = true;
        if (comptime std.mem.eql(u8, field.name, "allocator")) has_allocator = true;
    }

    try std.testing.expect(has_ids);
    try std.testing.expect(has_offsets);
    try std.testing.expect(has_attention_mask);
    try std.testing.expect(has_special_tokens_mask);
    try std.testing.expect(has_allocator);
}

test "Encoding has deinit and truncate methods" {
    try std.testing.expect(@hasDecl(Encoding, "deinit"));
    try std.testing.expect(@hasDecl(Encoding, "truncate"));
}

test "Encoding.deinit is safe for empty slices" {
    var encoding = Encoding{
        .ids = &.{},
        .offsets = &.{},
        .attention_mask = &.{},
        .special_tokens_mask = &.{},
        .allocator = std.testing.allocator,
    };
    encoding.deinit();
}

test "TokenOffset type is accessible" {
    const T = TokenOffset;
    _ = T;
}

test "TokenOffset is an extern struct with start/end" {
    const info = @typeInfo(TokenOffset);
    try std.testing.expect(info == .@"struct");
    try std.testing.expect(info.@"struct".layout == .@"extern");

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
