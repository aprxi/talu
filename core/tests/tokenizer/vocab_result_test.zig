//! Integration tests for tokenizer.VocabResult
//!
//! VocabResult contains the vocabulary entries and backing string data
//! returned by getVocab(). It provides efficient access to token strings.

const std = @import("std");
const main = @import("main");

const VocabResult = main.tokenizer.VocabResult;
const VocabEntry = main.tokenizer.VocabEntry;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "VocabResult type is accessible" {
    const T = VocabResult;
    _ = T;
}

test "VocabResult is a struct" {
    const info = @typeInfo(VocabResult);
    try std.testing.expect(info == .@"struct");
}

test "VocabResult has expected fields" {
    const info = @typeInfo(VocabResult);
    const fields = info.@"struct".fields;

    var has_entries = false;
    var has_string_data = false;
    var has_allocator = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "entries")) has_entries = true;
        if (comptime std.mem.eql(u8, field.name, "string_data")) has_string_data = true;
        if (comptime std.mem.eql(u8, field.name, "allocator")) has_allocator = true;
    }

    try std.testing.expect(has_entries);
    try std.testing.expect(has_string_data);
    try std.testing.expect(has_allocator);
}

// =============================================================================
// Method Tests
// =============================================================================

test "VocabResult has deinit method" {
    try std.testing.expect(@hasDecl(VocabResult, "deinit"));
}

// =============================================================================
// deinit Tests
// =============================================================================

test "VocabResult.deinit is safe on default-initialized struct" {
    const allocator = std.testing.allocator;
    var result = VocabResult{
        .entries = &.{},
        .string_data = &.{},
        .allocator = allocator,
    };
    result.deinit();
}

// =============================================================================
// VocabEntry Tests
// =============================================================================

test "VocabEntry type is accessible" {
    const T = VocabEntry;
    _ = T;
}

test "VocabEntry is a struct" {
    const info = @typeInfo(VocabEntry);
    try std.testing.expect(info == .@"struct");
}

test "VocabEntry has expected fields" {
    const info = @typeInfo(VocabEntry);
    const fields = info.@"struct".fields;

    var has_token = false;
    var has_id = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "token")) has_token = true;
        if (comptime std.mem.eql(u8, field.name, "id")) has_id = true;
    }

    try std.testing.expect(has_token);
    try std.testing.expect(has_id);
}
