//! Shared JSON serialization helpers for tree-sitter modules.
//!
//! Consolidates JSON string escaping used by ast.zig, highlight.zig,
//! query.zig, and graph/json_output.zig.

const std = @import("std");

/// Write a JSON-escaped string value (without surrounding quotes) into the buffer.
/// Handles: `"`, `\`, `\n`, `\r`, `\t`, and control chars `<0x20` as `\u00XX`.
pub fn writeJsonEscaped(allocator: std.mem.Allocator, buf: *std.ArrayList(u8), text: []const u8) !void {
    for (text) |ch| {
        switch (ch) {
            '"' => try buf.appendSlice(allocator, "\\\""),
            '\\' => try buf.appendSlice(allocator, "\\\\"),
            '\n' => try buf.appendSlice(allocator, "\\n"),
            '\r' => try buf.appendSlice(allocator, "\\r"),
            '\t' => try buf.appendSlice(allocator, "\\t"),
            else => |c| {
                if (c < 0x20) {
                    try std.fmt.format(buf.writer(allocator), "\\u{x:0>4}", .{c});
                } else {
                    try buf.append(allocator, c);
                }
            },
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

test "writeJsonEscaped escapes quotes" {
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(std.testing.allocator);
    try writeJsonEscaped(std.testing.allocator, &buf, "say \"hello\"");
    try std.testing.expectEqualStrings("say \\\"hello\\\"", buf.items);
}

test "writeJsonEscaped escapes backslash" {
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(std.testing.allocator);
    try writeJsonEscaped(std.testing.allocator, &buf, "a\\b");
    try std.testing.expectEqualStrings("a\\\\b", buf.items);
}

test "writeJsonEscaped escapes control chars" {
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(std.testing.allocator);
    try writeJsonEscaped(std.testing.allocator, &buf, "a\nb\r\t");
    try std.testing.expectEqualStrings("a\\nb\\r\\t", buf.items);
}

test "writeJsonEscaped escapes low control chars as unicode" {
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(std.testing.allocator);
    try writeJsonEscaped(std.testing.allocator, &buf, &[_]u8{ 0x01, 0x1f });
    try std.testing.expectEqualStrings("\\u0001\\u001f", buf.items);
}

test "writeJsonEscaped passes plain ASCII through" {
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(std.testing.allocator);
    try writeJsonEscaped(std.testing.allocator, &buf, "hello world 123");
    try std.testing.expectEqualStrings("hello world 123", buf.items);
}

test "writeJsonEscaped handles empty input" {
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(std.testing.allocator);
    try writeJsonEscaped(std.testing.allocator, &buf, "");
    try std.testing.expectEqual(@as(usize, 0), buf.items.len);
}
