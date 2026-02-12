//! Code validation module.
//!
//! Provides detection and validation of code blocks in LLM output.
//! Currently supports markdown-style code fences (CommonMark compliant).
//!
//! # Future Scope
//!
//! This module is designed to expand into language-specific syntax
//! validation (e.g., via tree-sitter), enabling enforcement of
//! syntactically valid Python, Zig, Rust, etc.
//!
//! Thread safety: Types are NOT thread-safe unless documented otherwise.

// ===== Public API =====

pub const fence = @import("fence.zig");
pub const block = @import("block.zig");

// Re-export commonly used types at module level
pub const CodeBlock = block.CodeBlock;
pub const CodeBlockList = block.CodeBlockList;
pub const FenceTracker = fence.FenceTracker;
pub const FenceState = fence.FenceState;

// Primary extraction function
pub const extractCodeBlocks = block.extractCodeBlocks;

// Utility functions
pub const isFenceChar = fence.isFenceChar;
pub const isLanguageChar = fence.isLanguageChar;

// ============================================================================
// Tests
// ============================================================================

test "extractCodeBlocks detects single block" {
    const std = @import("std");
    const text = "```python\nprint('hello')\n```\n";

    var blocks = try extractCodeBlocks(std.testing.allocator, text);
    defer blocks.deinit();

    try std.testing.expectEqual(@as(usize, 1), blocks.count());
    const b = blocks.get(0).?;
    try std.testing.expect(b.complete);
    try std.testing.expectEqualStrings("python", b.getLanguage(text));
    try std.testing.expectEqualStrings("print('hello')\n", b.getContent(text));
}

test "extractCodeBlocks detects multiple blocks" {
    const std = @import("std");
    const text =
        \\Here's some Python:
        \\```python
        \\print('hello')
        \\```
        \\
        \\And some Rust:
        \\```rust
        \\fn main() {}
        \\```
        \\
    ;

    var blocks = try extractCodeBlocks(std.testing.allocator, text);
    defer blocks.deinit();

    try std.testing.expectEqual(@as(usize, 2), blocks.count());
    try std.testing.expectEqualStrings("python", blocks.get(0).?.getLanguage(text));
    try std.testing.expectEqualStrings("rust", blocks.get(1).?.getLanguage(text));
}

test "extractCodeBlocks handles incomplete block" {
    const std = @import("std");
    const text = "```python\ncode without closing fence";

    var blocks = try extractCodeBlocks(std.testing.allocator, text);
    defer blocks.deinit();

    try std.testing.expectEqual(@as(usize, 1), blocks.count());
    try std.testing.expect(!blocks.get(0).?.complete);
}

test "extractCodeBlocks returns empty for plain text" {
    const std = @import("std");
    const text = "Just some regular text without any code blocks.";

    var blocks = try extractCodeBlocks(std.testing.allocator, text);
    defer blocks.deinit();

    try std.testing.expectEqual(@as(usize, 0), blocks.count());
}

test "extractCodeBlocks handles tilde fences" {
    const std = @import("std");
    const text = "~~~markdown\nSome `code` here\n~~~\n";

    var blocks = try extractCodeBlocks(std.testing.allocator, text);
    defer blocks.deinit();

    try std.testing.expectEqual(@as(usize, 1), blocks.count());
    try std.testing.expectEqualStrings("markdown", blocks.get(0).?.getLanguage(text));
}

test "extractCodeBlocks handles nested fences" {
    const std = @import("std");
    const text =
        \\`````markdown
        \\Here's how to write a code block:
        \\```python
        \\print("hello")
        \\```
        \\`````
        \\
    ;

    var blocks = try extractCodeBlocks(std.testing.allocator, text);
    defer blocks.deinit();

    try std.testing.expectEqual(@as(usize, 1), blocks.count());
    const content = blocks.get(0).?.getContent(text);
    // The inner ``` should be preserved as content
    try std.testing.expect(std.mem.indexOf(u8, content, "```python") != null);
}
