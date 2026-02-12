//! Integration tests for responses.FunctionOutputValue
//!
//! FunctionOutputValue is the output union for FunctionCallOutputData.
//! It can contain either simple text or an array of content parts.

const std = @import("std");
const main = @import("main");
const FunctionOutputValue = main.responses.FunctionOutputValue;

test "FunctionOutputValue is a tagged union" {
    const info = @typeInfo(FunctionOutputValue);
    try std.testing.expect(info == .@"union");
}

test "FunctionOutputValue has text and parts variants" {
    const info = @typeInfo(FunctionOutputValue);
    const fields = info.@"union".fields;

    var has_text = false;
    var has_parts = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "text")) has_text = true;
        if (comptime std.mem.eql(u8, field.name, "parts")) has_parts = true;
    }

    try std.testing.expect(has_text);
    try std.testing.expect(has_parts);
}
