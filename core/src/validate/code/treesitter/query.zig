//! Tree-sitter query pattern matching.
//!
//! Queries use S-expression patterns to find nodes in a syntax tree.
//! This is the foundation for syntax highlighting (via highlight queries)
//! and structural code analysis.
//!
//! Thread safety: Query is immutable after creation (safe to share).
//!                QueryCursor is NOT thread-safe (create one per use).

const std = @import("std");
const c = @import("c.zig").c;
const Language = @import("language.zig").Language;
const node_mod = @import("node.zig");
const Node = node_mod.Node;

pub const QueryError = error{
    Syntax,
    NodeType,
    Field,
    Capture,
    Structure,
    Language,
    OutOfMemory,
};

pub const QueryCapture = struct {
    node: Node,
    index: u32,
};

pub const QueryMatch = struct {
    id: u32,
    pattern_index: u16,
    captures: []const c.TSQueryCapture,

    /// Get the capture at the given index as a Node.
    pub fn captureNode(self: QueryMatch, idx: usize) ?Node {
        if (idx >= self.captures.len) return null;
        return .{ .raw = self.captures[idx].node };
    }

    /// Get the capture index (maps to capture name in query) at the given position.
    pub fn captureIndex(self: QueryMatch, idx: usize) ?u32 {
        if (idx >= self.captures.len) return null;
        return self.captures[idx].index;
    }
};

/// A compiled query pattern for matching against syntax trees.
///
/// Thread safety: Immutable after creation. Safe to share across threads.
pub const Query = struct {
    handle: *c.TSQuery,

    /// Compile a query pattern for the given language.
    /// The pattern uses tree-sitter's S-expression query syntax.
    pub fn init(lang: Language, pattern: []const u8) QueryError!Query {
        var error_offset: u32 = 0;
        var error_type: c.TSQueryError = c.TSQueryErrorNone;
        const handle = c.ts_query_new(
            lang.grammar(),
            pattern.ptr,
            @intCast(pattern.len),
            &error_offset,
            &error_type,
        ) orelse {
            return switch (error_type) {
                c.TSQueryErrorSyntax => error.Syntax,
                c.TSQueryErrorNodeType => error.NodeType,
                c.TSQueryErrorField => error.Field,
                c.TSQueryErrorCapture => error.Capture,
                c.TSQueryErrorStructure => error.Structure,
                c.TSQueryErrorLanguage => error.Language,
                else => error.Syntax,
            };
        };
        return .{ .handle = handle };
    }

    pub fn captureCount(self: *const Query) u32 {
        return c.ts_query_capture_count(self.handle);
    }

    pub fn captureNameForId(self: *const Query, id: u32) []const u8 {
        var length: u32 = 0;
        const ptr = c.ts_query_capture_name_for_id(self.handle, id, &length);
        if (ptr == null or length == 0) return "";
        return ptr[0..length];
    }

    pub fn patternCount(self: *const Query) u32 {
        return c.ts_query_pattern_count(self.handle);
    }

    pub fn deinit(self: *Query) void {
        c.ts_query_delete(self.handle);
    }
};

/// Cursor for executing queries against tree nodes.
///
/// Thread safety: NOT thread-safe. Create one per thread/use.
pub const QueryCursor = struct {
    handle: *c.TSQueryCursor,

    pub fn init() !QueryCursor {
        return .{
            .handle = c.ts_query_cursor_new() orelse return error.OutOfMemory,
        };
    }

    /// Execute a query starting at the given node.
    /// Call nextMatch() to iterate results.
    pub fn exec(self: *QueryCursor, query: *const Query, node: Node) void {
        c.ts_query_cursor_exec(self.handle, query.handle, node.raw);
    }

    /// Returns the next match, or null when exhausted.
    pub fn nextMatch(self: *QueryCursor) ?QueryMatch {
        var match: c.TSQueryMatch = undefined;
        if (!c.ts_query_cursor_next_match(self.handle, &match)) return null;
        return .{
            .id = match.id,
            .pattern_index = match.pattern_index,
            .captures = if (match.capture_count > 0)
                match.captures[0..match.capture_count]
            else
                &.{},
        };
    }

    pub fn deinit(self: *QueryCursor) void {
        c.ts_query_cursor_delete(self.handle);
    }
};

// =============================================================================
// JSON Serialization
// =============================================================================

const json_helpers = @import("json_helpers.zig");

/// Serialize query matches to a NUL-terminated JSON array string.
/// Iterates the cursor to exhaustion. Caller owns the returned slice.
pub fn queryMatchesToJson(
    allocator: std.mem.Allocator,
    cursor: *QueryCursor,
    source: []const u8,
    query_ref: *const Query,
) ![:0]u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);
    const w = buf.writer(allocator);

    try buf.append(allocator, '[');

    var match_count: usize = 0;
    while (cursor.nextMatch()) |match| {
        if (match_count > 0) try buf.append(allocator, ',');

        try std.fmt.format(w, "{{\"id\":{d},\"captures\":[", .{match.id});

        for (match.captures, 0..) |capture, ci| {
            if (ci > 0) try buf.append(allocator, ',');

            const cap_node = Node{ .raw = capture.node };
            const cap_name = query_ref.captureNameForId(capture.index);
            const start = cap_node.startByte();
            const end = cap_node.endByte();

            try std.fmt.format(w, "{{\"name\":\"{s}\",\"start\":{d},\"end\":{d},\"text\":\"", .{ cap_name, start, end });

            const text_slice = cap_node.text(source);
            try json_helpers.writeJsonEscaped(allocator, &buf, text_slice);

            try buf.appendSlice(allocator, "\"}");
        }

        try buf.appendSlice(allocator, "]}");
        match_count += 1;
    }

    try buf.append(allocator, ']');
    try buf.append(allocator, 0);

    const owned = try buf.toOwnedSlice(allocator);
    return owned[0 .. owned.len - 1 :0];
}

// =============================================================================
// Tests
// =============================================================================

const parser_mod = @import("parser.zig");

test "Query.init compiles valid pattern" {
    var q = try Query.init(.python, "(identifier) @id");
    defer q.deinit();

    try std.testing.expect(q.captureCount() > 0);
    try std.testing.expectEqualStrings("id", q.captureNameForId(0));
}

test "Query.init returns error for invalid pattern" {
    const result = Query.init(.python, "(nonexistent_node_xyz) @x");
    try std.testing.expectError(error.NodeType, result);
}

test "QueryCursor finds matches" {
    var p = try parser_mod.Parser.init(.python);
    defer p.deinit();

    var tree = try p.parse("x = y + z", null);
    defer tree.deinit();

    var q = try Query.init(.python, "(identifier) @id");
    defer q.deinit();

    var cursor = try QueryCursor.init();
    defer cursor.deinit();

    cursor.exec(&q, tree.rootNode());

    // Should find at least 3 identifiers: x, y, z
    var count: u32 = 0;
    while (cursor.nextMatch()) |_| {
        count += 1;
    }
    try std.testing.expect(count >= 3);
}

test "QueryMatch provides capture nodes" {
    var p = try parser_mod.Parser.init(.python);
    defer p.deinit();

    const source = "foo = 42";
    var tree = try p.parse(source, null);
    defer tree.deinit();

    var q = try Query.init(.python, "(identifier) @name");
    defer q.deinit();

    var cursor = try QueryCursor.init();
    defer cursor.deinit();

    cursor.exec(&q, tree.rootNode());

    if (cursor.nextMatch()) |match| {
        const node = match.captureNode(0).?;
        try std.testing.expectEqualStrings("foo", node.text(source));
    } else {
        return error.TestUnexpectedResult;
    }
}

test "queryMatchesToJson produces valid JSON" {
    var p = try parser_mod.Parser.init(.python);
    defer p.deinit();

    const source = "x = 1";
    var tree = try p.parse(source, null);
    defer tree.deinit();

    var q = try Query.init(.python, "(identifier) @id");
    defer q.deinit();

    var cursor = try QueryCursor.init();
    defer cursor.deinit();
    cursor.exec(&q, tree.rootNode());

    const json = try queryMatchesToJson(std.testing.allocator, &cursor, source, &q);
    defer std.testing.allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"name\":\"id\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"text\":\"x\"") != null);
}

test "queryMatchesToJson handles no matches" {
    var p = try parser_mod.Parser.init(.python);
    defer p.deinit();

    const source = "42";
    var tree = try p.parse(source, null);
    defer tree.deinit();

    // Pattern that won't match a bare integer
    var q = try Query.init(.python, "(function_definition) @fn");
    defer q.deinit();

    var cursor = try QueryCursor.init();
    defer cursor.deinit();
    cursor.exec(&q, tree.rootNode());

    const json = try queryMatchesToJson(std.testing.allocator, &cursor, source, &q);
    defer std.testing.allocator.free(json);

    try std.testing.expectEqualStrings("[]", json);
}

test "Query.captureNameForId returns empty for invalid id" {
    var q = try Query.init(.python, "(identifier) @id");
    defer q.deinit();

    const name = q.captureNameForId(999);
    try std.testing.expectEqual(@as(usize, 0), name.len);
}
