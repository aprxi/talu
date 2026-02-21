//! Tree-sitter AST node types.
//!
//! Provides safe wrappers around TSNode (value type) and TSTreeCursor.
//! Nodes are lightweight value types (no heap allocation) that reference
//! positions within a Tree. A Node is only valid while its parent Tree lives.
//!
//! Thread safety: Nodes are immutable value types. TreeCursor is NOT thread-safe.

const std = @import("std");
const c = @import("c.zig").c;

pub const Point = struct {
    row: u32,
    column: u32,
};

/// A syntax tree node. Value type — does not own memory.
/// Valid only while the parent Tree is alive.
pub const Node = struct {
    raw: c.TSNode,

    /// The grammar rule name for this node (e.g., "function_definition", "identifier").
    pub fn kind(self: Node) []const u8 {
        const ptr = c.ts_node_type(self.raw);
        if (ptr) |p| {
            return std.mem.span(p);
        }
        return "";
    }

    /// Extract this node's source text from the original source code.
    pub fn text(self: Node, source: []const u8) []const u8 {
        const start = self.startByte();
        const end = self.endByte();
        if (start >= source.len or end > source.len or start > end) return "";
        return source[start..end];
    }

    /// True if this is a named node (not anonymous punctuation/keywords).
    pub fn isNamed(self: Node) bool {
        return c.ts_node_is_named(self.raw);
    }

    pub fn startByte(self: Node) u32 {
        return c.ts_node_start_byte(self.raw);
    }

    pub fn endByte(self: Node) u32 {
        return c.ts_node_end_byte(self.raw);
    }

    pub fn startPoint(self: Node) Point {
        const p = c.ts_node_start_point(self.raw);
        return .{ .row = p.row, .column = p.column };
    }

    pub fn endPoint(self: Node) Point {
        const p = c.ts_node_end_point(self.raw);
        return .{ .row = p.row, .column = p.column };
    }

    pub fn parent(self: Node) ?Node {
        const p = c.ts_node_parent(self.raw);
        return if (c.ts_node_is_null(p)) null else .{ .raw = p };
    }

    pub fn childCount(self: Node) u32 {
        return c.ts_node_child_count(self.raw);
    }

    pub fn child(self: Node, index: u32) ?Node {
        const ch = c.ts_node_child(self.raw, index);
        return if (c.ts_node_is_null(ch)) null else .{ .raw = ch };
    }

    pub fn namedChildCount(self: Node) u32 {
        return c.ts_node_named_child_count(self.raw);
    }

    pub fn namedChild(self: Node, index: u32) ?Node {
        const ch = c.ts_node_named_child(self.raw, index);
        return if (c.ts_node_is_null(ch)) null else .{ .raw = ch };
    }

    /// Get the child node with the given field name.
    /// Field names are grammar-specific (e.g., "name", "body", "parameters").
    pub fn childByFieldName(self: Node, field_name: []const u8) ?Node {
        const ch = c.ts_node_child_by_field_name(
            self.raw,
            field_name.ptr,
            @intCast(field_name.len),
        );
        return if (c.ts_node_is_null(ch)) null else .{ .raw = ch };
    }

    pub fn nextSibling(self: Node) ?Node {
        const s = c.ts_node_next_sibling(self.raw);
        return if (c.ts_node_is_null(s)) null else .{ .raw = s };
    }

    pub fn prevSibling(self: Node) ?Node {
        const s = c.ts_node_prev_sibling(self.raw);
        return if (c.ts_node_is_null(s)) null else .{ .raw = s };
    }

    pub fn nextNamedSibling(self: Node) ?Node {
        const s = c.ts_node_next_named_sibling(self.raw);
        return if (c.ts_node_is_null(s)) null else .{ .raw = s };
    }

    /// Returns the S-expression representation of this node's subtree.
    /// Caller owns the returned string.
    pub fn toSexp(self: Node, allocator: std.mem.Allocator) ![:0]u8 {
        const ptr = c.ts_node_string(self.raw) orelse return error.OutOfMemory;
        defer std.c.free(ptr);
        const slice = std.mem.span(ptr);
        return allocator.dupeZ(u8, slice);
    }

    /// Create a TreeCursor positioned at this node.
    pub fn walk(self: Node) TreeCursor {
        return .{ .raw = c.ts_tree_cursor_new(self.raw) };
    }

    pub fn isNull(self: Node) bool {
        return c.ts_node_is_null(self.raw);
    }
};

/// Stateful cursor for walking the tree. More efficient than repeated
/// child/parent calls when traversing depth-first.
///
/// Thread safety: NOT thread-safe. Create one per thread.
pub const TreeCursor = struct {
    raw: c.TSTreeCursor,

    pub fn currentNode(self: *const TreeCursor) Node {
        return .{ .raw = c.ts_tree_cursor_current_node(&self.raw) };
    }

    pub fn gotoFirstChild(self: *TreeCursor) bool {
        return c.ts_tree_cursor_goto_first_child(&self.raw);
    }

    pub fn gotoNextSibling(self: *TreeCursor) bool {
        return c.ts_tree_cursor_goto_next_sibling(&self.raw);
    }

    pub fn gotoParent(self: *TreeCursor) bool {
        return c.ts_tree_cursor_goto_parent(&self.raw);
    }

    pub fn deinit(self: *TreeCursor) void {
        c.ts_tree_cursor_delete(&self.raw);
    }
};

// =============================================================================
// Tests
// =============================================================================

const Language = @import("language.zig").Language;
const parser_mod = @import("parser.zig");

test "Node.kind returns correct type for Python" {
    var p = try parser_mod.Parser.init(.python);
    defer p.deinit();
    var tree = try p.parse("x = 1", null);
    defer tree.deinit();

    const root = tree.rootNode();
    try std.testing.expectEqualStrings("module", root.kind());
}

test "Node.text extracts source text" {
    const source = "hello = 42";
    var p = try parser_mod.Parser.init(.python);
    defer p.deinit();
    var tree = try p.parse(source, null);
    defer tree.deinit();

    const root = tree.rootNode();
    try std.testing.expectEqualStrings(source, root.text(source));
}

test "Node.childCount and child traversal" {
    var p = try parser_mod.Parser.init(.python);
    defer p.deinit();
    var tree = try p.parse("x = 1\ny = 2\n", null);
    defer tree.deinit();

    const root = tree.rootNode();
    // Python module should have 2 expression statements
    try std.testing.expect(root.namedChildCount() >= 2);

    // First child should be an expression_statement
    const first = root.namedChild(0).?;
    try std.testing.expectEqualStrings("expression_statement", first.kind());
}

test "Node.startByte and endByte" {
    const source = "abc";
    var p = try parser_mod.Parser.init(.python);
    defer p.deinit();
    var tree = try p.parse(source, null);
    defer tree.deinit();

    const root = tree.rootNode();
    try std.testing.expectEqual(@as(u32, 0), root.startByte());
    try std.testing.expectEqual(@as(u32, 3), root.endByte());
}

test "Node.startPoint and endPoint" {
    const source = "line1\nline2";
    var p = try parser_mod.Parser.init(.python);
    defer p.deinit();
    var tree = try p.parse(source, null);
    defer tree.deinit();

    const root = tree.rootNode();
    const start = root.startPoint();
    try std.testing.expectEqual(@as(u32, 0), start.row);
    try std.testing.expectEqual(@as(u32, 0), start.column);
}

test "Node.toSexp produces S-expression" {
    var p = try parser_mod.Parser.init(.python);
    defer p.deinit();
    var tree = try p.parse("1", null);
    defer tree.deinit();

    const root = tree.rootNode();
    const sexp = try root.toSexp(std.testing.allocator);
    defer std.testing.allocator.free(sexp);

    // Should contain "module" as root node type
    try std.testing.expect(std.mem.indexOf(u8, sexp, "module") != null);
}

test "Node.isNull for invalid node" {
    var p = try parser_mod.Parser.init(.python);
    defer p.deinit();
    var tree = try p.parse("x", null);
    defer tree.deinit();

    // Accessing child beyond range returns null
    const root = tree.rootNode();
    const invalid = root.child(999);
    try std.testing.expect(invalid == null);
}

test "Node.childByFieldName finds Python function name" {
    var p = try parser_mod.Parser.init(.python);
    defer p.deinit();
    const source = "def hello(): pass";
    var tree = try p.parse(source, null);
    defer tree.deinit();

    const root = tree.rootNode();
    const func_def = root.namedChild(0).?;
    try std.testing.expectEqualStrings("function_definition", func_def.kind());

    const name_node = func_def.childByFieldName("name").?;
    try std.testing.expectEqualStrings("hello", name_node.text(source));
}

test "Node.childByFieldName returns null for missing field" {
    var p = try parser_mod.Parser.init(.python);
    defer p.deinit();
    var tree = try p.parse("x = 1", null);
    defer tree.deinit();

    const root = tree.rootNode();
    try std.testing.expect(root.childByFieldName("nonexistent_field") == null);
}

test "TreeCursor depth-first traversal" {
    var p = try parser_mod.Parser.init(.python);
    defer p.deinit();
    var tree = try p.parse("x = 1", null);
    defer tree.deinit();

    var cursor = tree.rootNode().walk();
    defer cursor.deinit();

    // Start at root
    try std.testing.expectEqualStrings("module", cursor.currentNode().kind());

    // Go to first child
    try std.testing.expect(cursor.gotoFirstChild());

    // Should be at expression_statement
    try std.testing.expectEqualStrings("expression_statement", cursor.currentNode().kind());

    // Go back to parent
    try std.testing.expect(cursor.gotoParent());
    try std.testing.expectEqualStrings("module", cursor.currentNode().kind());
}
