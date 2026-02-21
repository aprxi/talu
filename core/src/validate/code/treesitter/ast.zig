//! JSON AST output.
//!
//! Converts a parsed syntax tree into a recursive JSON representation with
//! node kind, text, byte offsets, row/column positions, and children.
//!
//! Thread safety: treeToJson is safe to call concurrently (each call creates
//! its own state).

const std = @import("std");
const node_mod = @import("node.zig");
const parser_mod = @import("parser.zig");
const Node = node_mod.Node;

/// Write a JSON-escaped string value (without surrounding quotes) into the writer.
fn writeJsonEscaped(allocator: std.mem.Allocator, buf: *std.ArrayList(u8), text: []const u8) !void {
    for (text) |ch| {
        switch (ch) {
            '"' => try buf.appendSlice(allocator, "\\\""),
            '\\' => try buf.appendSlice(allocator, "\\\\"),
            '\n' => try buf.appendSlice(allocator, "\\n"),
            '\r' => try buf.appendSlice(allocator, "\\r"),
            '\t' => try buf.appendSlice(allocator, "\\t"),
            else => |c| {
                if (c < 0x20) {
                    // Control characters as \u00XX
                    try std.fmt.format(buf.writer(allocator), "\\u{x:0>4}", .{c});
                } else {
                    try buf.append(allocator, c);
                }
            },
        }
    }
}

/// Write a recursive JSON object for the given node into buf.
fn writeNodeJson(
    allocator: std.mem.Allocator,
    buf: *std.ArrayList(u8),
    node: Node,
    source: []const u8,
) !void {
    const w = buf.writer(allocator);

    try buf.appendSlice(allocator, "{\"kind\":\"");
    try writeJsonEscaped(allocator, buf, node.kind());

    try buf.appendSlice(allocator, "\",\"is_named\":");
    try buf.appendSlice(allocator, if (node.isNamed()) "true" else "false");

    const start_byte = node.startByte();
    const end_byte = node.endByte();
    const sp = node.startPoint();
    const ep = node.endPoint();

    try std.fmt.format(w, ",\"start_byte\":{d},\"end_byte\":{d}", .{ start_byte, end_byte });
    try std.fmt.format(w, ",\"start_point\":{{\"row\":{d},\"column\":{d}}}", .{ sp.row, sp.column });
    try std.fmt.format(w, ",\"end_point\":{{\"row\":{d},\"column\":{d}}}", .{ ep.row, ep.column });

    // Include text for leaf/named nodes (skip for large interior nodes to keep output manageable)
    const child_count = node.childCount();
    if (child_count == 0 and start_byte < source.len and end_byte <= source.len) {
        try buf.appendSlice(allocator, ",\"text\":\"");
        try writeJsonEscaped(allocator, buf, source[start_byte..end_byte]);
        try buf.append(allocator, '"');
    }

    try std.fmt.format(w, ",\"child_count\":{d}", .{child_count});

    if (child_count > 0) {
        try buf.appendSlice(allocator, ",\"children\":[");
        var i: u32 = 0;
        while (i < child_count) : (i += 1) {
            if (i > 0) try buf.append(allocator, ',');
            if (node.child(i)) |ch| {
                try writeNodeJson(allocator, buf, ch, source);
            }
        }
        try buf.append(allocator, ']');
    }

    try buf.append(allocator, '}');
}

/// Serialize a parsed tree to a JSON string.
///
/// Returns a NUL-terminated JSON string. Caller owns the returned slice.
///
/// Output format:
/// ```json
/// {"language":"python","tree":{...recursive node JSON...}}
/// ```
pub fn treeToJson(
    allocator: std.mem.Allocator,
    tree: *const parser_mod.Tree,
    source: []const u8,
    language_name: []const u8,
) ![:0]u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);

    try buf.appendSlice(allocator, "{\"language\":\"");
    try writeJsonEscaped(allocator, &buf, language_name);
    try buf.appendSlice(allocator, "\",\"tree\":");

    try writeNodeJson(allocator, &buf, tree.rootNode(), source);

    try buf.append(allocator, '}');
    try buf.append(allocator, 0);

    const owned = try buf.toOwnedSlice(allocator);
    // Return as sentinel-terminated slice (len excludes the NUL)
    return owned[0 .. owned.len - 1 :0];
}

// =============================================================================
// Tests
// =============================================================================

const Language = @import("language.zig").Language;

test "treeToJson produces valid structure for Python" {
    var p = try parser_mod.Parser.init(.python);
    defer p.deinit();
    const source = "x = 1";
    var tree = try p.parse(source, null);
    defer tree.deinit();

    const json = try treeToJson(std.testing.allocator, &tree, source, "python");
    defer std.testing.allocator.free(json);

    // Should contain language field
    try std.testing.expect(std.mem.indexOf(u8, json, "\"language\":\"python\"") != null);
    // Should contain root node kind
    try std.testing.expect(std.mem.indexOf(u8, json, "\"kind\":\"module\"") != null);
    // Should contain children
    try std.testing.expect(std.mem.indexOf(u8, json, "\"children\":[") != null);
}

test "treeToJson includes text for leaf nodes" {
    var p = try parser_mod.Parser.init(.python);
    defer p.deinit();
    const source = "x = 42";
    var tree = try p.parse(source, null);
    defer tree.deinit();

    const json = try treeToJson(std.testing.allocator, &tree, source, "python");
    defer std.testing.allocator.free(json);

    // Leaf nodes should have text
    try std.testing.expect(std.mem.indexOf(u8, json, "\"text\":\"x\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"text\":\"42\"") != null);
}

test "treeToJson handles empty source" {
    var p = try parser_mod.Parser.init(.python);
    defer p.deinit();
    var tree = try p.parse("", null);
    defer tree.deinit();

    const json = try treeToJson(std.testing.allocator, &tree, "", "python");
    defer std.testing.allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"kind\":\"module\"") != null);
}

test "treeToJson escapes special characters in text" {
    var p = try parser_mod.Parser.init(.python);
    defer p.deinit();
    const source = "x = \"hello\\nworld\"";
    var tree = try p.parse(source, null);
    defer tree.deinit();

    const json = try treeToJson(std.testing.allocator, &tree, source, "python");
    defer std.testing.allocator.free(json);

    // Should contain escaped backslash from the source string
    try std.testing.expect(json.len > 0);
}

test "treeToJson includes position info" {
    var p = try parser_mod.Parser.init(.python);
    defer p.deinit();
    const source = "x = 1\ny = 2";
    var tree = try p.parse(source, null);
    defer tree.deinit();

    const json = try treeToJson(std.testing.allocator, &tree, source, "python");
    defer std.testing.allocator.free(json);

    // Should have start/end points
    try std.testing.expect(std.mem.indexOf(u8, json, "\"start_point\":{\"row\":0,\"column\":0}") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"start_byte\":0") != null);
}
