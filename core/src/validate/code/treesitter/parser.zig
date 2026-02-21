//! Tree-sitter parser and tree types.
//!
//! Parser owns the TSParser handle and manages its lifecycle.
//! Tree owns the TSTree handle — an immutable parse result.
//!
//! Thread safety: Parser is NOT thread-safe (create one per thread).
//!                Tree is immutable after creation (safe to share read-only).

const std = @import("std");
const c = @import("c.zig").c;
const Language = @import("language.zig").Language;
const node_mod = @import("node.zig");
const Node = node_mod.Node;

pub const Parser = struct {
    handle: *c.TSParser,
    language: Language,

    /// Create a parser configured for the given language.
    pub fn init(lang: Language) !Parser {
        const handle = c.ts_parser_new() orelse return error.OutOfMemory;
        if (!c.ts_parser_set_language(handle, lang.grammar())) {
            c.ts_parser_delete(handle);
            return error.InvalidLanguage;
        }
        return .{ .handle = handle, .language = lang };
    }

    /// Parse source code into a syntax tree.
    /// If `old_tree` is provided, performs incremental parsing.
    pub fn parse(self: *Parser, source: []const u8, old_tree: ?*const Tree) !Tree {
        const old = if (old_tree) |t| t.handle else null;
        const tree = c.ts_parser_parse_string(
            self.handle,
            old,
            source.ptr,
            @intCast(source.len),
        ) orelse return error.ParseFailed;
        return .{ .handle = tree };
    }

    pub fn deinit(self: *Parser) void {
        c.ts_parser_delete(self.handle);
    }
};

/// Describes a text edit for incremental parsing.
///
/// You must describe the edit both in terms of byte offsets and
/// (row, column) coordinates so tree-sitter can update the tree.
pub const InputEdit = struct {
    start_byte: u32,
    old_end_byte: u32,
    new_end_byte: u32,
    start_row: u32,
    start_column: u32,
    old_end_row: u32,
    old_end_column: u32,
    new_end_row: u32,
    new_end_column: u32,
};

/// A syntax tree produced by parsing.
/// The tree references the original source text by byte offsets.
///
/// After creation, the tree is immutable — except for `edit()`, which
/// must be called before re-parsing to inform tree-sitter what changed.
pub const Tree = struct {
    handle: *c.TSTree,

    pub fn rootNode(self: *const Tree) Node {
        return .{ .raw = c.ts_tree_root_node(self.handle) };
    }

    /// Inform tree-sitter that the source has been edited.
    ///
    /// You must call this before passing the tree as `old_tree` to
    /// `Parser.parse()` for correct incremental parsing. Without it,
    /// tree-sitter cannot identify which nodes to reuse.
    pub fn edit(self: *Tree, input_edit: InputEdit) void {
        var ts_edit = c.TSInputEdit{
            .start_byte = input_edit.start_byte,
            .old_end_byte = input_edit.old_end_byte,
            .new_end_byte = input_edit.new_end_byte,
            .start_point = .{ .row = input_edit.start_row, .column = input_edit.start_column },
            .old_end_point = .{ .row = input_edit.old_end_row, .column = input_edit.old_end_column },
            .new_end_point = .{ .row = input_edit.new_end_row, .column = input_edit.new_end_column },
        };
        c.ts_tree_edit(self.handle, &ts_edit);
    }

    /// Create an independent copy of this tree.
    pub fn copy(self: *const Tree) !Tree {
        return .{
            .handle = c.ts_tree_copy(self.handle) orelse return error.OutOfMemory,
        };
    }

    pub fn deinit(self: *Tree) void {
        c.ts_tree_delete(self.handle);
    }
};

// =============================================================================
// Tests
// =============================================================================

test "Parser.init and parse Python" {
    var p = try Parser.init(.python);
    defer p.deinit();

    var tree = try p.parse("x = 1", null);
    defer tree.deinit();

    const root = tree.rootNode();
    try std.testing.expectEqualStrings("module", root.kind());
    try std.testing.expect(root.childCount() > 0);
}

test "Parser.init and parse JavaScript" {
    var p = try Parser.init(.javascript);
    defer p.deinit();

    var tree = try p.parse("const x = 1;", null);
    defer tree.deinit();

    const root = tree.rootNode();
    try std.testing.expectEqualStrings("program", root.kind());
}

test "Parser.init and parse TypeScript" {
    var p = try Parser.init(.typescript);
    defer p.deinit();

    var tree = try p.parse("const x: number = 1;", null);
    defer tree.deinit();

    const root = tree.rootNode();
    try std.testing.expectEqualStrings("program", root.kind());
}

test "Parser.init and parse Rust" {
    var p = try Parser.init(.rust);
    defer p.deinit();

    var tree = try p.parse("fn main() {}", null);
    defer tree.deinit();

    const root = tree.rootNode();
    try std.testing.expectEqualStrings("source_file", root.kind());
}

test "Parser.init and parse Go" {
    var p = try Parser.init(.go);
    defer p.deinit();

    var tree = try p.parse("package main", null);
    defer tree.deinit();

    const root = tree.rootNode();
    try std.testing.expectEqualStrings("source_file", root.kind());
}

test "Parser.init and parse C" {
    var p = try Parser.init(.c_lang);
    defer p.deinit();

    var tree = try p.parse("int main() { return 0; }", null);
    defer tree.deinit();

    const root = tree.rootNode();
    try std.testing.expectEqualStrings("translation_unit", root.kind());
}

test "Parser.init and parse Zig" {
    var p = try Parser.init(.zig_lang);
    defer p.deinit();

    var tree = try p.parse("const x = 1;", null);
    defer tree.deinit();

    const root = tree.rootNode();
    try std.testing.expect(root.childCount() > 0);
}

test "Parser.init and parse JSON" {
    var p = try Parser.init(.json);
    defer p.deinit();

    var tree = try p.parse("{\"key\": 42}", null);
    defer tree.deinit();

    const root = tree.rootNode();
    try std.testing.expectEqualStrings("document", root.kind());
}

test "Parser.init and parse HTML" {
    var p = try Parser.init(.html);
    defer p.deinit();

    var tree = try p.parse("<div>hello</div>", null);
    defer tree.deinit();

    const root = tree.rootNode();
    try std.testing.expectEqualStrings("document", root.kind());
}

test "Parser.init and parse CSS" {
    var p = try Parser.init(.css);
    defer p.deinit();

    var tree = try p.parse("body { color: red; }", null);
    defer tree.deinit();

    const root = tree.rootNode();
    try std.testing.expectEqualStrings("stylesheet", root.kind());
}

test "Parser.init and parse Bash" {
    var p = try Parser.init(.bash);
    defer p.deinit();

    var tree = try p.parse("echo hello", null);
    defer tree.deinit();

    const root = tree.rootNode();
    try std.testing.expectEqualStrings("program", root.kind());
}

test "Parser.parse empty string" {
    var p = try Parser.init(.python);
    defer p.deinit();

    var tree = try p.parse("", null);
    defer tree.deinit();

    const root = tree.rootNode();
    try std.testing.expectEqualStrings("module", root.kind());
    try std.testing.expectEqual(@as(u32, 0), root.childCount());
}

test "Tree.edit enables correct incremental re-parse" {
    var p = try Parser.init(.python);
    defer p.deinit();

    // Initial parse: "x = 1"
    var tree = try p.parse("x = 1", null);

    // Edit: replace "1" (byte 4..5) with "42" (byte 4..6)
    tree.edit(.{
        .start_byte = 4,
        .old_end_byte = 5,
        .new_end_byte = 6,
        .start_row = 0,
        .start_column = 4,
        .old_end_row = 0,
        .old_end_column = 5,
        .new_end_row = 0,
        .new_end_column = 6,
    });

    // Re-parse with edited old tree
    var new_tree = try p.parse("x = 42", &tree);
    defer new_tree.deinit();
    tree.deinit();

    const root = new_tree.rootNode();
    try std.testing.expectEqualStrings("module", root.kind());
    try std.testing.expect(root.childCount() > 0);
}

test "Tree.copy produces independent copy" {
    var p = try Parser.init(.python);
    defer p.deinit();

    var tree = try p.parse("x = 1", null);
    defer tree.deinit();

    var tree_copy = try tree.copy();
    defer tree_copy.deinit();

    // Both trees should have the same structure
    try std.testing.expectEqualStrings("module", tree_copy.rootNode().kind());
}
