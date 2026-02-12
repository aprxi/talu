//! Integration tests for responses.ContentPart
//!
//! ContentPart represents a single piece of content within an Item.
//! It uses a tagged union (ContentVariant) to support different content types
//! with type-specific data.

const std = @import("std");
const main = @import("main");

const ContentPart = main.responses.ContentPart;
const ContentType = main.responses.ContentType;
const ContentVariant = main.responses.ContentVariant;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "ContentPart type is accessible" {
    const T = ContentPart;
    _ = T;
}

test "ContentPart is a struct" {
    const info = @typeInfo(ContentPart);
    try std.testing.expect(info == .@"struct");
}

test "ContentPart has variant field" {
    const info = @typeInfo(ContentPart);
    const fields = info.@"struct".fields;

    var has_variant = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "variant")) has_variant = true;
    }

    try std.testing.expect(has_variant);
}

// =============================================================================
// ContentVariant Tests
// =============================================================================

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

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "input_text")) has_input_text = true;
        if (comptime std.mem.eql(u8, field.name, "output_text")) has_output_text = true;
        if (comptime std.mem.eql(u8, field.name, "text")) has_text = true;
        if (comptime std.mem.eql(u8, field.name, "input_image")) has_input_image = true;
    }

    try std.testing.expect(has_input_text);
    try std.testing.expect(has_output_text);
    try std.testing.expect(has_text);
    try std.testing.expect(has_input_image);
}

// =============================================================================
// ContentType Tests
// =============================================================================

test "ContentType has expected values" {
    const info = @typeInfo(ContentType);
    try std.testing.expect(info == .@"enum");

    const fields = info.@"enum".fields;
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

test "ContentType values are distinct" {
    try std.testing.expect(ContentType.input_text != ContentType.output_text);
    try std.testing.expect(ContentType.output_text != ContentType.text);
    try std.testing.expect(ContentType.input_image != ContentType.input_audio);
    try std.testing.expect(ContentType.input_text != ContentType.input_file);
}

// =============================================================================
// Usage Pattern Tests
// =============================================================================

test "ContentVariant can be switched on" {
    // Test that the union works as expected
    const variant = ContentVariant{ .input_text = .{ .text = "test" } };

    switch (variant) {
        .input_text => |data| {
            try std.testing.expectEqualStrings("test", data.text);
        },
        else => unreachable,
    }
}

test "ContentType can identify text variants" {
    const text_types = [_]ContentType{
        .input_text,
        .output_text,
        .text,
    };

    for (text_types) |ct| {
        const is_text = switch (ct) {
            .input_text, .output_text, .text => true,
            else => false,
        };
        try std.testing.expect(is_text);
    }
}
