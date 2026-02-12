//! Integration tests for responses.ContentVariant
//!
//! ContentVariant is a tagged union containing type-specific content data.

const std = @import("std");
const main = @import("main");
const ContentVariant = main.responses.ContentVariant;

test "ContentVariant is a tagged union" {
    const info = @typeInfo(ContentVariant);
    try std.testing.expect(info == .@"union");
}

test "ContentVariant has expected variants" {
    const info = @typeInfo(ContentVariant);
    const fields = info.@"union".fields;

    var has_input_text = false;
    var has_output_text = false;
    var has_text = false;
    var has_input_image = false;
    var has_input_audio = false;
    var has_input_file = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "input_text")) has_input_text = true;
        if (comptime std.mem.eql(u8, field.name, "output_text")) has_output_text = true;
        if (comptime std.mem.eql(u8, field.name, "text")) has_text = true;
        if (comptime std.mem.eql(u8, field.name, "input_image")) has_input_image = true;
        if (comptime std.mem.eql(u8, field.name, "input_audio")) has_input_audio = true;
        if (comptime std.mem.eql(u8, field.name, "input_file")) has_input_file = true;
    }

    try std.testing.expect(has_input_text);
    try std.testing.expect(has_output_text);
    try std.testing.expect(has_text);
    try std.testing.expect(has_input_image);
    try std.testing.expect(has_input_audio);
    try std.testing.expect(has_input_file);
}
