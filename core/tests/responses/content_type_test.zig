//! Integration tests for ContentType
//!
//! ContentType identifies the type of content part in an Item (input_text, output_text, etc.).

const std = @import("std");
const main = @import("main");
const ContentType = main.responses.ContentType;

// =============================================================================
// Enum Value Tests
// =============================================================================

test "ContentType has expected values" {
    // Content type variants from Open Responses schema
    try std.testing.expect(@intFromEnum(ContentType.input_text) >= 0);
    try std.testing.expect(@intFromEnum(ContentType.output_text) >= 0);
    try std.testing.expect(@intFromEnum(ContentType.text) >= 0);
    try std.testing.expect(@intFromEnum(ContentType.input_image) >= 0);
    try std.testing.expect(@intFromEnum(ContentType.input_audio) >= 0);
    try std.testing.expect(@intFromEnum(ContentType.input_file) >= 0);
}

test "ContentType values are distinct" {
    try std.testing.expect(ContentType.input_text != ContentType.output_text);
    try std.testing.expect(ContentType.output_text != ContentType.text);
    try std.testing.expect(ContentType.input_image != ContentType.input_audio);
}

// =============================================================================
// Usage Pattern Tests
// =============================================================================

test "ContentType works in switch statements" {
    const types = [_]ContentType{ .input_text, .output_text, .text, .input_image };
    var text_count: u32 = 0;
    var non_text_count: u32 = 0;

    for (types) |content_type| {
        switch (content_type) {
            .input_text, .output_text, .text => text_count += 1,
            else => non_text_count += 1,
        }
    }

    try std.testing.expectEqual(@as(u32, 3), text_count);
    try std.testing.expectEqual(@as(u32, 1), non_text_count);
}

test "ContentType can be compared" {
    try std.testing.expect(ContentType.input_text == ContentType.input_text);
    try std.testing.expect(ContentType.input_text != ContentType.output_text);
    try std.testing.expect(ContentType.text != ContentType.input_image);
}

test "ContentType can be stored in arrays" {
    const multimodal_content = [_]ContentType{
        .input_text,
        .input_image,
        .output_text,
    };

    try std.testing.expectEqual(@as(usize, 3), multimodal_content.len);
    try std.testing.expectEqual(ContentType.input_text, multimodal_content[0]);
    try std.testing.expectEqual(ContentType.input_image, multimodal_content[1]);
    try std.testing.expectEqual(ContentType.output_text, multimodal_content[2]);
}

// =============================================================================
// Text Detection Tests
// =============================================================================

test "ContentType distinguishes text from non-text" {
    // Text types
    const text_types = [_]ContentType{ .input_text, .output_text, .text };
    for (text_types) |ct| {
        const is_text = switch (ct) {
            .input_text, .output_text, .text => true,
            else => false,
        };
        try std.testing.expect(is_text);
    }

    // Non-text types
    const non_text_types = [_]ContentType{ .input_image, .input_audio, .input_file };
    for (non_text_types) |ct| {
        const is_text = switch (ct) {
            .input_text, .output_text, .text => true,
            else => false,
        };
        try std.testing.expect(!is_text);
    }
}
