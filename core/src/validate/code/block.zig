//! Code block metadata types and extraction.
//!
//! Defines the structural representation of detected code blocks
//! and provides extraction from raw text.
//!
//! Thread safety: NOT thread-safe. All access must be from a single thread.

const std = @import("std");
const fence_mod = @import("fence.zig");

/// Metadata for a single detected code block.
///
/// Positions are byte offsets into the source text. All ranges are
/// half-open: [start, end).
///
/// # Invariants
///
/// - `fence_start <= language_start <= language_end <= content_start`
/// - `content_start <= content_end <= fence_end`
/// - If `complete == false`, `fence_end` points to current parse position
pub const CodeBlock = struct {
    /// Sequential index of this block (0-based).
    index: u32,

    /// Byte offset where opening fence begins.
    fence_start: u32,

    /// Byte offset after closing fence (or current position if incomplete).
    fence_end: u32,

    /// Byte offset where language identifier begins.
    language_start: u32,

    /// Byte offset after language identifier.
    language_end: u32,

    /// Byte offset where code content begins (after newline following info string).
    content_start: u32,

    /// Byte offset where code content ends (before closing fence).
    content_end: u32,

    /// True if closing fence was found.
    complete: bool,

    /// Extract the language string from the source text.
    /// Returns empty slice if no language was specified.
    pub fn getLanguage(self: CodeBlock, source: []const u8) []const u8 {
        if (self.language_start >= source.len or self.language_end > source.len) {
            return "";
        }
        if (self.language_end <= self.language_start) {
            return "";
        }
        return source[self.language_start..self.language_end];
    }

    /// Extract the code content from the source text.
    /// Returns empty slice if block has no content.
    pub fn getContent(self: CodeBlock, source: []const u8) []const u8 {
        if (self.content_start >= source.len or self.content_end > source.len) {
            return "";
        }
        if (self.content_end <= self.content_start) {
            return "";
        }
        return source[self.content_start..self.content_end];
    }
};

/// List of detected code blocks with owned memory.
pub const CodeBlockList = struct {
    items: std.ArrayListUnmanaged(CodeBlock),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) CodeBlockList {
        return .{
            .items = .{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *CodeBlockList) void {
        self.items.deinit(self.allocator);
    }

    pub fn append(self: *CodeBlockList, block: CodeBlock) !void {
        try self.items.append(self.allocator, block);
    }

    pub fn count(self: *const CodeBlockList) usize {
        return self.items.items.len;
    }

    pub fn get(self: *const CodeBlockList, index: usize) ?CodeBlock {
        if (index >= self.items.items.len) {
            return null;
        }
        return self.items.items[index];
    }

    /// Serialize to JSON. Caller owns returned memory.
    pub fn toJson(self: *const CodeBlockList, allocator: std.mem.Allocator) ![]u8 {
        var buf: std.ArrayListUnmanaged(u8) = .{};
        errdefer buf.deinit(allocator);

        try buf.append(allocator, '[');

        for (self.items.items, 0..) |block, i| {
            if (i > 0) {
                try buf.append(allocator, ',');
            }
            try std.fmt.format(buf.writer(allocator), "{{\"index\":{d},\"fence_start\":{d},\"fence_end\":{d},\"language_start\":{d},\"language_end\":{d},\"content_start\":{d},\"content_end\":{d},\"complete\":{}}}", .{
                block.index,
                block.fence_start,
                block.fence_end,
                block.language_start,
                block.language_end,
                block.content_start,
                block.content_end,
                block.complete,
            });
        }

        try buf.append(allocator, ']');

        return buf.toOwnedSlice(allocator);
    }
};

/// Extract all code blocks from text.
///
/// Scans the input text for markdown-style code fences (CommonMark compliant)
/// and returns metadata for each detected block.
///
/// Caller owns returned CodeBlockList and must call deinit().
pub fn extractCodeBlocks(allocator: std.mem.Allocator, text: []const u8) !CodeBlockList {
    var list = CodeBlockList.init(allocator);
    errdefer list.deinit();

    var tracker = fence_mod.FenceTracker.init();

    for (text, 0..) |byte, pos| {
        if (tracker.feed(byte, @intCast(pos))) |completed_block| {
            try list.append(completed_block);
        }
    }

    // Handle incomplete block at end of input
    if (tracker.finalize(@intCast(text.len))) |incomplete_block| {
        try list.append(incomplete_block);
    }

    return list;
}

// ============================================================================
// Tests
// ============================================================================

test "CodeBlock getLanguage extracts language identifier" {
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

    try std.testing.expectEqualStrings("python", block.getLanguage(source));
}

test "CodeBlock getLanguage returns empty for no language" {
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

test "CodeBlock getContent extracts code content" {
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

test "CodeBlock getContent handles out of bounds safely" {
    const source = "short";
    const block = CodeBlock{
        .index = 0,
        .fence_start = 0,
        .fence_end = 100,
        .language_start = 3,
        .language_end = 9,
        .content_start = 10,
        .content_end = 50,
        .complete = false,
    };

    try std.testing.expectEqualStrings("", block.getContent(source));
}

test "CodeBlockList init creates empty list" {
    var list = CodeBlockList.init(std.testing.allocator);
    defer list.deinit();

    try std.testing.expectEqual(@as(usize, 0), list.count());
}

test "CodeBlockList append and count work correctly" {
    var list = CodeBlockList.init(std.testing.allocator);
    defer list.deinit();

    try list.append(.{
        .index = 0,
        .fence_start = 0,
        .fence_end = 10,
        .language_start = 3,
        .language_end = 6,
        .content_start = 7,
        .content_end = 9,
        .complete = true,
    });

    try std.testing.expectEqual(@as(usize, 1), list.count());
}

test "CodeBlockList get returns correct block" {
    var list = CodeBlockList.init(std.testing.allocator);
    defer list.deinit();

    try list.append(.{
        .index = 0,
        .fence_start = 0,
        .fence_end = 10,
        .language_start = 3,
        .language_end = 6,
        .content_start = 7,
        .content_end = 9,
        .complete = true,
    });

    const block = list.get(0);
    try std.testing.expect(block != null);
    try std.testing.expectEqual(@as(u32, 0), block.?.index);
}

test "CodeBlockList get returns null for out of bounds" {
    var list = CodeBlockList.init(std.testing.allocator);
    defer list.deinit();

    try std.testing.expect(list.get(0) == null);
    try std.testing.expect(list.get(100) == null);
}

test "CodeBlockList toJson serializes empty list" {
    var list = CodeBlockList.init(std.testing.allocator);
    defer list.deinit();

    const json = try list.toJson(std.testing.allocator);
    defer std.testing.allocator.free(json);

    try std.testing.expectEqualStrings("[]", json);
}

test "CodeBlockList toJson serializes single block" {
    var list = CodeBlockList.init(std.testing.allocator);
    defer list.deinit();

    try list.append(.{
        .index = 0,
        .fence_start = 0,
        .fence_end = 25,
        .language_start = 3,
        .language_end = 9,
        .content_start = 10,
        .content_end = 22,
        .complete = true,
    });

    const json = try list.toJson(std.testing.allocator);
    defer std.testing.allocator.free(json);

    const expected = "[{\"index\":0,\"fence_start\":0,\"fence_end\":25,\"language_start\":3,\"language_end\":9,\"content_start\":10,\"content_end\":22,\"complete\":true}]";
    try std.testing.expectEqualStrings(expected, json);
}

test "CodeBlockList toJson serializes multiple blocks" {
    var list = CodeBlockList.init(std.testing.allocator);
    defer list.deinit();

    try list.append(.{
        .index = 0,
        .fence_start = 0,
        .fence_end = 10,
        .language_start = 3,
        .language_end = 6,
        .content_start = 7,
        .content_end = 9,
        .complete = true,
    });

    try list.append(.{
        .index = 1,
        .fence_start = 20,
        .fence_end = 30,
        .language_start = 23,
        .language_end = 26,
        .content_start = 27,
        .content_end = 29,
        .complete = false,
    });

    const json = try list.toJson(std.testing.allocator);
    defer std.testing.allocator.free(json);

    // Verify it's valid JSON with two objects
    try std.testing.expect(std.mem.startsWith(u8, json, "[{"));
    try std.testing.expect(std.mem.endsWith(u8, json, "}]"));
    try std.testing.expect(std.mem.indexOf(u8, json, "},{") != null);
}
