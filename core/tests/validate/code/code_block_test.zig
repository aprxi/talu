//! Integration tests for CodeBlock.
//!
//! Tests the CodeBlock type exported from core/src/validate/code/root.zig.

const std = @import("std");
const main = @import("main");
const code = main.validate.code;
const CodeBlock = code.CodeBlock;

// ============================================================================
// Construction and Field Access Tests
// ============================================================================

test "CodeBlock fields are accessible" {
    const block = CodeBlock{
        .index = 0,
        .fence_start = 0,
        .fence_end = 25,
        .language_start = 3,
        .language_end = 9,
        .content_start = 10,
        .content_end = 22,
        .complete = true,
    };

    try std.testing.expectEqual(@as(u32, 0), block.index);
    try std.testing.expectEqual(@as(u32, 0), block.fence_start);
    try std.testing.expectEqual(@as(u32, 25), block.fence_end);
    try std.testing.expect(block.complete);
}

test "CodeBlock incomplete state" {
    const block = CodeBlock{
        .index = 1,
        .fence_start = 100,
        .fence_end = 200,
        .language_start = 103,
        .language_end = 109,
        .content_start = 110,
        .content_end = 195,
        .complete = false,
    };

    try std.testing.expect(!block.complete);
}

// ============================================================================
// getLanguage Tests
// ============================================================================

test "CodeBlock.getLanguage extracts language from source" {
    const source = "```python\nprint('hello')\n```";
    const block = CodeBlock{
        .index = 0,
        .fence_start = 0,
        .fence_end = 28,
        .language_start = 3,
        .language_end = 9,
        .content_start = 10,
        .content_end = 25,
        .complete = true,
    };

    try std.testing.expectEqualStrings("python", block.getLanguage(source));
}

test "CodeBlock.getLanguage returns empty for no language" {
    const source = "```\ncode\n```";
    const block = CodeBlock{
        .index = 0,
        .fence_start = 0,
        .fence_end = 12,
        .language_start = 3,
        .language_end = 3,
        .content_start = 4,
        .content_end = 9,
        .complete = true,
    };

    try std.testing.expectEqualStrings("", block.getLanguage(source));
}

test "CodeBlock.getLanguage handles out of bounds safely" {
    const source = "short";
    const block = CodeBlock{
        .index = 0,
        .fence_start = 0,
        .fence_end = 100,
        .language_start = 10,
        .language_end = 20,
        .content_start = 21,
        .content_end = 50,
        .complete = false,
    };

    try std.testing.expectEqualStrings("", block.getLanguage(source));
}

// ============================================================================
// getContent Tests
// ============================================================================

test "CodeBlock.getContent extracts content from source" {
    const source = "```python\nprint('hi')\n```";
    const block = CodeBlock{
        .index = 0,
        .fence_start = 0,
        .fence_end = 25,
        .language_start = 3,
        .language_end = 9,
        .content_start = 10,
        .content_end = 22,
        .complete = true,
    };

    try std.testing.expectEqualStrings("print('hi')\n", block.getContent(source));
}

test "CodeBlock.getContent returns empty for empty content" {
    const source = "```python\n```";
    const block = CodeBlock{
        .index = 0,
        .fence_start = 0,
        .fence_end = 13,
        .language_start = 3,
        .language_end = 9,
        .content_start = 10,
        .content_end = 10,
        .complete = true,
    };

    try std.testing.expectEqualStrings("", block.getContent(source));
}

test "CodeBlock.getContent handles out of bounds safely" {
    const source = "tiny";
    const block = CodeBlock{
        .index = 0,
        .fence_start = 0,
        .fence_end = 100,
        .language_start = 3,
        .language_end = 9,
        .content_start = 50,
        .content_end = 90,
        .complete = false,
    };

    try std.testing.expectEqualStrings("", block.getContent(source));
}

// ============================================================================
// Edge Cases
// ============================================================================

test "CodeBlock handles multiline content" {
    const source = "```rust\nfn main() {\n    println!(\"hello\");\n}\n```";
    const block = CodeBlock{
        .index = 0,
        .fence_start = 0,
        .fence_end = 48,
        .language_start = 3,
        .language_end = 7,
        .content_start = 8,
        .content_end = 44,
        .complete = true,
    };

    try std.testing.expectEqualStrings("rust", block.getLanguage(source));
    const content = block.getContent(source);
    try std.testing.expect(std.mem.indexOf(u8, content, "fn main()") != null);
    try std.testing.expect(std.mem.indexOf(u8, content, "println!") != null);
}
