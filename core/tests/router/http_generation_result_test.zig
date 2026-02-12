//! Integration tests for HttpGenerationResult
//!
//! HttpGenerationResult contains the output from HttpEngine.generate(),
//! including generated text, token counts, and finish reason.
//!
//! Note: HttpGenerationResult is used for remote OpenAI-compatible inference.

const std = @import("std");
const main = @import("main");
const HttpGenerationResult = main.router.HttpGenerationResult;

// =============================================================================
// Struct Layout Tests
// =============================================================================

test "HttpGenerationResult has expected fields" {
    const fields = @typeInfo(HttpGenerationResult).@"struct".fields;

    var has_text = false;
    var has_prompt_tokens = false;
    var has_completion_tokens = false;
    var has_finish_reason = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "text")) has_text = true;
        if (comptime std.mem.eql(u8, field.name, "prompt_tokens")) has_prompt_tokens = true;
        if (comptime std.mem.eql(u8, field.name, "completion_tokens")) has_completion_tokens = true;
        if (comptime std.mem.eql(u8, field.name, "finish_reason")) has_finish_reason = true;
    }

    try std.testing.expect(has_text);
    try std.testing.expect(has_prompt_tokens);
    try std.testing.expect(has_completion_tokens);
    try std.testing.expect(has_finish_reason);
}

test "HttpGenerationResult can be constructed" {
    const text = try std.testing.allocator.dupe(u8, "Hello, world!");

    const result = HttpGenerationResult{
        .text = text,
        .prompt_tokens = 10,
        .completion_tokens = 4,
        .finish_reason = .stop,
    };

    try std.testing.expectEqualStrings("Hello, world!", result.text);
    try std.testing.expectEqual(@as(usize, 10), result.prompt_tokens);
    try std.testing.expectEqual(@as(usize, 4), result.completion_tokens);

    result.deinit(std.testing.allocator);
}

// =============================================================================
// Method Tests
// =============================================================================

test "HttpGenerationResult has deinit method" {
    try std.testing.expect(@hasDecl(HttpGenerationResult, "deinit"));
}

test "HttpGenerationResult.deinit frees memory" {
    const text = try std.testing.allocator.dupe(u8, "Test response");

    const result = HttpGenerationResult{
        .text = text,
        .prompt_tokens = 5,
        .completion_tokens = 3,
        .finish_reason = .stop,
    };

    result.deinit(std.testing.allocator);
    // No leak = test passes
}

// =============================================================================
// Edge Case Tests
// =============================================================================

test "HttpGenerationResult with empty text" {
    const text = try std.testing.allocator.dupe(u8, "");

    const result = HttpGenerationResult{
        .text = text,
        .prompt_tokens = 0,
        .completion_tokens = 0,
        .finish_reason = .unknown,
    };

    try std.testing.expectEqual(@as(usize, 0), result.text.len);

    result.deinit(std.testing.allocator);
}

test "HttpGenerationResult finish reasons" {
    const text = try std.testing.allocator.dupe(u8, "x");

    // Test with different finish reasons
    var result = HttpGenerationResult{
        .text = text,
        .prompt_tokens = 1,
        .completion_tokens = 1,
        .finish_reason = .length,
    };

    try std.testing.expect(result.finish_reason == .length);

    result.deinit(std.testing.allocator);
}
