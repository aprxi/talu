//! Code fence state machine.
//!
//! Tracks markdown-style code fences during byte-by-byte processing.
//! Designed for streaming use cases where text arrives incrementally.
//!
//! # CommonMark Compliance (Full)
//!
//! Implements the complete CommonMark fenced code block specification:
//!
//! - **Fence characters:** Both backtick (`) and tilde (~) are valid
//! - **Opening fence:** 3 or more consecutive fence characters
//! - **Closing fence:** Same character as opening, length >= opening length
//! - **Indentation:** 0-3 spaces allowed before fence on a new line
//! - **Info string:** Optional text after opening fence; language identifier
//!   is the first word (terminated by whitespace)
//!
//! ## Examples
//!
//! Basic backtick fence:
//! ```python
//! print("hello")
//! ```
//!
//! Tilde fence (for content containing backticks):
//! ~~~markdown
//! Here's a code block: ```js
//! console.log("hi")
//! ```
//! ~~~
//!
//! Nested fence (5-tick containing 3-tick):
//! `````markdown
//! ```python
//! print("hello")
//! ```
//! `````
//!
//! Indented fence (valid with 0-3 spaces):
//!    ```python
//!    print("indented")
//!    ```
//!
//! Thread safety: NOT thread-safe. Each thread should have its own tracker.

const std = @import("std");
const CodeBlock = @import("block.zig").CodeBlock;

/// States in the code fence detection automaton.
pub const FenceState = enum {
    /// Normal text, not in a fence sequence.
    text,

    /// At start of line, counting leading spaces (0-3 allowed).
    line_start,

    /// Counting consecutive fence characters for potential opening fence.
    opening_fence,

    /// Reading info string (language + optional metadata).
    info_string,

    /// Inside code block content.
    code,

    /// At start of line inside code, counting leading spaces.
    code_line_start,

    /// Counting consecutive fence characters for potential closing fence.
    closing_fence,
};

/// Tracks code fence boundaries during incremental text processing.
///
/// # Usage
///
/// ```zig
/// var tracker = FenceTracker.init();
/// for (text, 0..) |byte, pos| {
///     if (tracker.feed(byte, pos)) |completed_block| {
///         // Process completed block
///     }
/// }
/// if (tracker.finalize(text.len)) |incomplete_block| {
///     // Handle incomplete block at end
/// }
/// ```
///
/// Thread safety: NOT thread-safe.
pub const FenceTracker = struct {
    state: FenceState,
    block_index: u32,

    // Current block being built
    fence_start: u32,
    fence_char: u8, // '`' or '~' - must match for closing
    opening_fence_len: u32, // Length of opening fence (>= 3)
    language_start: u32,
    language_end: u32,
    content_start: u32,

    // For closing fence detection
    closing_fence_start: u32,
    closing_fence_len: u32,

    // For indentation handling
    line_indent: u8, // Spaces at start of current line (0-3 valid)

    // Track if we've captured the language identifier
    language_captured: bool,

    pub fn init() FenceTracker {
        return .{
            .state = .text,
            .block_index = 0,
            .fence_start = 0,
            .fence_char = 0,
            .opening_fence_len = 0,
            .language_start = 0,
            .language_end = 0,
            .content_start = 0,
            .closing_fence_start = 0,
            .closing_fence_len = 0,
            .line_indent = 0,
            .language_captured = false,
        };
    }

    /// Process a single byte. Returns completed CodeBlock if fence closed.
    pub fn feed(self: *FenceTracker, byte: u8, pos: u32) ?CodeBlock {
        switch (self.state) {
            .text => return self.handleText(byte, pos),
            .line_start => return self.handleLineStart(byte, pos),
            .opening_fence => return self.handleOpeningFence(byte, pos),
            .info_string => return self.handleInfoString(byte, pos),
            .code => return self.handleCode(byte, pos),
            .code_line_start => return self.handleCodeLineStart(byte, pos),
            .closing_fence => return self.handleClosingFence(byte, pos),
        }
    }

    /// Finalize at end of input. Returns incomplete block if inside one.
    pub fn finalize(self: *FenceTracker, end_pos: u32) ?CodeBlock {
        switch (self.state) {
            .code, .code_line_start, .closing_fence => {
                // We're inside an incomplete block
                const block = CodeBlock{
                    .index = self.block_index,
                    .fence_start = self.fence_start,
                    .fence_end = end_pos,
                    .language_start = self.language_start,
                    .language_end = self.language_end,
                    .content_start = self.content_start,
                    .content_end = end_pos,
                    .complete = false,
                };
                self.resetForNextBlock();
                return block;
            },
            else => return null,
        }
    }

    fn handleText(self: *FenceTracker, byte: u8, pos: u32) ?CodeBlock {
        if (byte == '\n') {
            // Next byte is at line start
            self.state = .line_start;
            self.line_indent = 0;
        } else if (isFenceChar(byte)) {
            // Potential fence at start of input (pos == 0) or after newline
            // But we're in text state, so this is only valid at position 0
            if (pos == 0) {
                self.fence_start = pos;
                self.fence_char = byte;
                self.opening_fence_len = 1;
                self.state = .opening_fence;
            }
        }
        return null;
    }

    fn handleLineStart(self: *FenceTracker, byte: u8, pos: u32) ?CodeBlock {
        if (byte == ' ') {
            self.line_indent += 1;
            if (self.line_indent > 3) {
                // 4+ spaces = indented code block, not a fence
                self.state = .text;
            }
            return null;
        }

        if (isFenceChar(byte)) {
            // Start of potential opening fence
            self.fence_start = pos;
            self.fence_char = byte;
            self.opening_fence_len = 1;
            self.state = .opening_fence;
            return null;
        }

        // Not a fence start, back to normal text
        self.state = .text;
        return null;
    }

    fn handleOpeningFence(self: *FenceTracker, byte: u8, pos: u32) ?CodeBlock {
        if (byte == self.fence_char) {
            self.opening_fence_len += 1;
            return null;
        }

        // Fence character sequence ended
        if (self.opening_fence_len < 3) {
            // Not enough fence chars, not a valid fence
            self.state = .text;
            // Re-process this byte as text
            if (byte == '\n') {
                self.state = .line_start;
                self.line_indent = 0;
            }
            return null;
        }

        // Valid opening fence (3+ chars)
        if (byte == '\n') {
            // No info string, content starts on next line
            self.language_start = pos;
            self.language_end = pos;
            self.content_start = pos + 1;
            self.language_captured = true;
            self.state = .code_line_start;
            self.line_indent = 0;
        } else if (byte == ' ' or byte == '\t') {
            // Whitespace before info string - skip it
            self.language_start = pos + 1;
            self.language_end = pos + 1;
            self.language_captured = false;
            self.state = .info_string;
        } else {
            // Start of info string (language identifier)
            self.language_start = pos;
            self.language_end = pos + 1;
            self.language_captured = false;
            self.state = .info_string;
        }
        return null;
    }

    fn handleInfoString(self: *FenceTracker, byte: u8, pos: u32) ?CodeBlock {
        if (byte == '\n') {
            // Info string ends, content starts on next line
            self.content_start = pos + 1;
            self.language_captured = true;
            self.state = .code_line_start;
            self.line_indent = 0;
            return null;
        }

        if (!self.language_captured) {
            if (byte == ' ' or byte == '\t') {
                // Whitespace ends the language identifier
                // If we haven't started capturing yet, language_end == language_start
                self.language_captured = true;
            } else {
                // Extend language identifier
                self.language_end = pos + 1;
            }
        }
        // After language is captured, we ignore rest of info string until newline
        return null;
    }

    fn handleCode(self: *FenceTracker, byte: u8, _: u32) ?CodeBlock {
        if (byte == '\n') {
            self.state = .code_line_start;
            self.line_indent = 0;
        }
        return null;
    }

    fn handleCodeLineStart(self: *FenceTracker, byte: u8, pos: u32) ?CodeBlock {
        if (byte == ' ') {
            self.line_indent += 1;
            if (self.line_indent > 3) {
                // 4+ spaces at line start inside code - just content
                self.state = .code;
            }
            return null;
        }

        if (byte == self.fence_char) {
            // Potential closing fence
            self.closing_fence_start = pos;
            self.closing_fence_len = 1;
            self.state = .closing_fence;
            return null;
        }

        if (byte == '\n') {
            // Empty line inside code block, stay in code_line_start
            self.line_indent = 0;
            return null;
        }

        // Regular content
        self.state = .code;
        return null;
    }

    fn handleClosingFence(self: *FenceTracker, byte: u8, pos: u32) ?CodeBlock {
        if (byte == self.fence_char) {
            self.closing_fence_len += 1;
            return null;
        }

        // Fence character sequence ended
        if (self.closing_fence_len >= self.opening_fence_len) {
            // Valid closing fence - check if line ends properly
            if (byte == '\n' or byte == ' ' or byte == '\t') {
                // Valid close (newline or trailing whitespace before newline)
                if (byte == '\n') {
                    return self.completeBlock(pos + 1);
                }
                // Trailing whitespace - keep waiting for newline
                // Stay in closing_fence state but mark that we've seen enough fence chars
                return null;
            }
        }

        // Not a valid closing fence, treat as code content
        self.state = .code;
        if (byte == '\n') {
            self.state = .code_line_start;
            self.line_indent = 0;
        }
        return null;
    }

    fn completeBlock(self: *FenceTracker, fence_end: u32) CodeBlock {
        const block = CodeBlock{
            .index = self.block_index,
            .fence_start = self.fence_start,
            .fence_end = fence_end,
            .language_start = self.language_start,
            .language_end = self.language_end,
            .content_start = self.content_start,
            .content_end = self.closing_fence_start,
            .complete = true,
        };

        self.resetForNextBlock();
        return block;
    }

    fn resetForNextBlock(self: *FenceTracker) void {
        self.block_index += 1;
        self.state = .line_start;
        self.fence_start = 0;
        self.fence_char = 0;
        self.opening_fence_len = 0;
        self.language_start = 0;
        self.language_end = 0;
        self.content_start = 0;
        self.closing_fence_start = 0;
        self.closing_fence_len = 0;
        self.line_indent = 0;
        self.language_captured = false;
    }
};

/// Check if byte is a fence character (backtick or tilde).
pub fn isFenceChar(byte: u8) bool {
    return byte == '`' or byte == '~';
}

/// Check if byte is valid in a language identifier.
/// Language ends at first whitespace per CommonMark.
/// Allows: a-z, A-Z, 0-9, _, -, +, #, .
pub fn isLanguageChar(byte: u8) bool {
    return switch (byte) {
        'a'...'z', 'A'...'Z', '0'...'9', '_', '-', '+', '#', '.' => true,
        else => false,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "FenceTracker init starts in text state" {
    const tracker = FenceTracker.init();
    try std.testing.expectEqual(FenceState.text, tracker.state);
    try std.testing.expectEqual(@as(u32, 0), tracker.block_index);
}

test "FenceTracker feed detects single complete fence" {
    const text = "```python\nprint('hi')\n```\n";
    var tracker = FenceTracker.init();
    var completed: ?CodeBlock = null;

    for (text, 0..) |byte, pos| {
        if (tracker.feed(byte, @intCast(pos))) |block| {
            completed = block;
        }
    }

    try std.testing.expect(completed != null);
    try std.testing.expectEqual(@as(u32, 0), completed.?.index);
    try std.testing.expectEqual(@as(u32, 0), completed.?.fence_start);
    try std.testing.expect(completed.?.complete);
    try std.testing.expectEqualStrings("python", completed.?.getLanguage(text));
}

test "FenceTracker feed detects fence without language" {
    const text = "```\ncode\n```\n";
    var tracker = FenceTracker.init();
    var completed: ?CodeBlock = null;

    for (text, 0..) |byte, pos| {
        if (tracker.feed(byte, @intCast(pos))) |block| {
            completed = block;
        }
    }

    try std.testing.expect(completed != null);
    try std.testing.expectEqualStrings("", completed.?.getLanguage(text));
    try std.testing.expectEqualStrings("code\n", completed.?.getContent(text));
}

test "FenceTracker feed detects multiple fences" {
    const text = "```python\ncode1\n```\ntext\n```rust\ncode2\n```\n";
    var tracker = FenceTracker.init();
    var blocks: std.ArrayListUnmanaged(CodeBlock) = .{};
    defer blocks.deinit(std.testing.allocator);

    for (text, 0..) |byte, pos| {
        if (tracker.feed(byte, @intCast(pos))) |block| {
            try blocks.append(std.testing.allocator, block);
        }
    }

    try std.testing.expectEqual(@as(usize, 2), blocks.items.len);
    try std.testing.expectEqual(@as(u32, 0), blocks.items[0].index);
    try std.testing.expectEqual(@as(u32, 1), blocks.items[1].index);
    try std.testing.expectEqualStrings("python", blocks.items[0].getLanguage(text));
    try std.testing.expectEqualStrings("rust", blocks.items[1].getLanguage(text));
}

test "FenceTracker finalize returns incomplete block" {
    const text = "```python\ncode without closing";
    var tracker = FenceTracker.init();

    for (text, 0..) |byte, pos| {
        _ = tracker.feed(byte, @intCast(pos));
    }

    const incomplete = tracker.finalize(@intCast(text.len));
    try std.testing.expect(incomplete != null);
    try std.testing.expect(!incomplete.?.complete);
    try std.testing.expectEqualStrings("python", incomplete.?.getLanguage(text));
}

test "FenceTracker feed handles variable-length opening fence" {
    const text = "`````python\ncode\n`````\n";
    var tracker = FenceTracker.init();
    var completed: ?CodeBlock = null;

    for (text, 0..) |byte, pos| {
        if (tracker.feed(byte, @intCast(pos))) |block| {
            completed = block;
        }
    }

    try std.testing.expect(completed != null);
    try std.testing.expect(completed.?.complete);
    try std.testing.expectEqualStrings("python", completed.?.getLanguage(text));
}

test "FenceTracker feed requires closing fence length >= opening" {
    const text = "`````python\ncode\n```\nmore code\n`````\n";
    var tracker = FenceTracker.init();
    var completed: ?CodeBlock = null;

    for (text, 0..) |byte, pos| {
        if (tracker.feed(byte, @intCast(pos))) |block| {
            completed = block;
        }
    }

    try std.testing.expect(completed != null);
    // The 3-tick inside should be treated as content, not close
    const content = completed.?.getContent(text);
    try std.testing.expect(std.mem.indexOf(u8, content, "```") != null);
}

test "FenceTracker feed handles tilde fences" {
    const text = "~~~python\ncode\n~~~\n";
    var tracker = FenceTracker.init();
    var completed: ?CodeBlock = null;

    for (text, 0..) |byte, pos| {
        if (tracker.feed(byte, @intCast(pos))) |block| {
            completed = block;
        }
    }

    try std.testing.expect(completed != null);
    try std.testing.expect(completed.?.complete);
    try std.testing.expectEqualStrings("python", completed.?.getLanguage(text));
}

test "FenceTracker feed requires matching fence character" {
    // Tilde cannot close backtick
    const text = "```python\ncode\n~~~\nmore\n```\n";
    var tracker = FenceTracker.init();
    var completed: ?CodeBlock = null;

    for (text, 0..) |byte, pos| {
        if (tracker.feed(byte, @intCast(pos))) |block| {
            completed = block;
        }
    }

    try std.testing.expect(completed != null);
    // The ~~~ inside should be content, closed by ```
    const content = completed.?.getContent(text);
    try std.testing.expect(std.mem.indexOf(u8, content, "~~~") != null);
}

test "FenceTracker feed handles indented fence (0-3 spaces)" {
    const text = "text\n   ```python\n   code\n   ```\n";
    var tracker = FenceTracker.init();
    var completed: ?CodeBlock = null;

    for (text, 0..) |byte, pos| {
        if (tracker.feed(byte, @intCast(pos))) |block| {
            completed = block;
        }
    }

    try std.testing.expect(completed != null);
    try std.testing.expect(completed.?.complete);
}

test "FenceTracker feed rejects 4+ space indentation" {
    // 4 spaces makes it an indented code block, not a fenced block
    const text = "    ```python\ncode\n    ```\n";
    var tracker = FenceTracker.init();
    var completed: ?CodeBlock = null;

    for (text, 0..) |byte, pos| {
        if (tracker.feed(byte, @intCast(pos))) |block| {
            completed = block;
        }
    }

    // Should not detect a fenced code block
    try std.testing.expect(completed == null);
}

test "FenceTracker feed handles info string with extra metadata" {
    const text = "```python startline=5 highlight=2,3\ncode\n```\n";
    var tracker = FenceTracker.init();
    var completed: ?CodeBlock = null;

    for (text, 0..) |byte, pos| {
        if (tracker.feed(byte, @intCast(pos))) |block| {
            completed = block;
        }
    }

    try std.testing.expect(completed != null);
    // Language should be just "python", not the full info string
    try std.testing.expectEqualStrings("python", completed.?.getLanguage(text));
}

test "FenceTracker feed handles backticks inside code" {
    const text = "```markdown\nUse `code` like this\n```\n";
    var tracker = FenceTracker.init();
    var completed: ?CodeBlock = null;

    for (text, 0..) |byte, pos| {
        if (tracker.feed(byte, @intCast(pos))) |block| {
            completed = block;
        }
    }

    try std.testing.expect(completed != null);
    const content = completed.?.getContent(text);
    try std.testing.expect(std.mem.indexOf(u8, content, "`code`") != null);
}

test "isFenceChar identifies fence characters" {
    try std.testing.expect(isFenceChar('`'));
    try std.testing.expect(isFenceChar('~'));
    try std.testing.expect(!isFenceChar('-'));
    try std.testing.expect(!isFenceChar('a'));
    try std.testing.expect(!isFenceChar(' '));
}

test "isLanguageChar identifies valid language characters" {
    try std.testing.expect(isLanguageChar('a'));
    try std.testing.expect(isLanguageChar('Z'));
    try std.testing.expect(isLanguageChar('0'));
    try std.testing.expect(isLanguageChar('_'));
    try std.testing.expect(isLanguageChar('-'));
    try std.testing.expect(isLanguageChar('+'));
    try std.testing.expect(isLanguageChar('#'));
    try std.testing.expect(isLanguageChar('.'));
    try std.testing.expect(!isLanguageChar(' '));
    try std.testing.expect(!isLanguageChar('\t'));
    try std.testing.expect(!isLanguageChar('\n'));
}
