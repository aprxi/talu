//! C API for tree-sitter code parsing, querying, and syntax highlighting.
//!
//! Provides three capability groups:
//!
//! 1. Parser/Tree API (talu_treesitter_parser_*, talu_treesitter_tree_*):
//!    - Parse source code into syntax trees
//!    - Get S-expression representation of tree
//!
//! 2. Highlight API (talu_treesitter_highlight):
//!    - Stateless: parse + highlight in one call
//!    - Returns JSON array of {start, end, type} tokens
//!
//! 3. Query API (talu_treesitter_query_*):
//!    - Compile custom S-expression patterns
//!    - Execute against trees, get JSON results

const std = @import("std");
const capi_error = @import("error.zig");
const error_codes = @import("error_codes.zig");
const treesitter = @import("../validate/code/treesitter/root.zig");
const Language = treesitter.Language;
const Parser = treesitter.Parser;
const Tree = treesitter.Tree;
const Query = treesitter.Query;
const QueryCursor = treesitter.QueryCursor;
const highlight_mod = treesitter.highlight;
const node_mod = treesitter.node;
const ast_mod = treesitter.ast;
const graph_mod = treesitter.graph;

// =============================================================================
// Opaque Handles
// =============================================================================

/// Opaque handle for tree-sitter parser.
/// Thread safety: NOT thread-safe. Create one per thread.
pub const TreeSitterParserHandle = opaque {
    fn fromPtr(ptr: *Parser) *TreeSitterParserHandle {
        return @ptrCast(ptr);
    }
    fn toPtr(self: *TreeSitterParserHandle) *Parser {
        return @ptrCast(@alignCast(self));
    }
};

/// Opaque handle for parsed syntax tree.
/// Thread safety: Immutable after creation. Safe to share read-only.
pub const TreeSitterTreeHandle = opaque {
    fn fromPtr(ptr: *Tree) *TreeSitterTreeHandle {
        return @ptrCast(ptr);
    }
    fn toPtr(self: *TreeSitterTreeHandle) *Tree {
        return @ptrCast(@alignCast(self));
    }
};

/// Opaque handle for compiled query pattern.
/// Thread safety: Immutable after creation. Safe to share read-only.
pub const TreeSitterQueryHandle = opaque {
    fn fromPtr(ptr: *Query) *TreeSitterQueryHandle {
        return @ptrCast(ptr);
    }
    fn toPtr(self: *TreeSitterQueryHandle) *Query {
        return @ptrCast(@alignCast(self));
    }
};

// =============================================================================
// Parser Lifecycle
// =============================================================================

/// Create a parser for the specified language.
///
/// Parameters:
///   lang: Language name string (e.g., "python", "javascript", "rust")
///
/// Returns handle on success, null on unknown language or OOM.
/// Caller must call talu_treesitter_parser_free() when done.
pub export fn talu_treesitter_parser_create(
    lang: [*:0]const u8,
) callconv(.c) ?*TreeSitterParserHandle {
    capi_error.clearError();
    const allocator = std.heap.c_allocator;

    const lang_str = std.mem.span(lang);
    const language = Language.fromString(lang_str) orelse {
        capi_error.setError(error.InvalidArgument, "Unknown language: {s}", .{lang_str});
        return null;
    };

    const parser_ptr = allocator.create(Parser) catch {
        capi_error.setError(error.OutOfMemory, "Failed to allocate parser", .{});
        return null;
    };

    parser_ptr.* = Parser.init(language) catch {
        allocator.destroy(parser_ptr);
        capi_error.setError(error.Unexpected, "Failed to initialize parser for {s}", .{lang_str});
        return null;
    };

    return TreeSitterParserHandle.fromPtr(parser_ptr);
}

/// Free a parser handle.
pub export fn talu_treesitter_parser_free(
    handle: ?*TreeSitterParserHandle,
) callconv(.c) void {
    const h = handle orelse return;
    const parser_ptr = h.toPtr();
    parser_ptr.deinit();
    std.heap.c_allocator.destroy(parser_ptr);
}

// =============================================================================
// Parsing
// =============================================================================

/// Parse source code into a syntax tree.
///
/// Parameters:
///   handle: Parser handle
///   source: Source code bytes
///   source_len: Length of source code
///   out_tree: Output: receives tree handle on success
///
/// Returns 0 on success, error code on failure.
/// Caller must call talu_treesitter_tree_free() on the output tree.
pub export fn talu_treesitter_parse(
    handle: ?*TreeSitterParserHandle,
    source: [*]const u8,
    source_len: u32,
    out_tree: ?*?*TreeSitterTreeHandle,
) callconv(.c) i32 {
    capi_error.clearError();
    const allocator = std.heap.c_allocator;

    const result = out_tree orelse {
        capi_error.setError(error.InvalidArgument, "out_tree is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    result.* = null;

    const parser_ptr = (handle orelse {
        capi_error.setError(error.InvalidArgument, "parser handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }).toPtr();

    const source_slice = source[0..source_len];

    const tree_ptr = allocator.create(Tree) catch {
        capi_error.setError(error.OutOfMemory, "Failed to allocate tree", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };

    tree_ptr.* = parser_ptr.parse(source_slice, null) catch {
        allocator.destroy(tree_ptr);
        capi_error.setError(error.Unexpected, "Parse failed", .{});
        return @intFromEnum(error_codes.ErrorCode.internal_error);
    };

    result.* = TreeSitterTreeHandle.fromPtr(tree_ptr);
    return 0;
}

/// Parse source code into a syntax tree with optional incremental parsing.
///
/// If `old_tree` is provided (non-null), tree-sitter will reuse unchanged
/// subtrees from the previous parse for faster re-parsing.
///
/// Parameters:
///   handle: Parser handle
///   source: Source code bytes
///   source_len: Length of source code
///   old_tree: Previous tree handle for incremental parsing (null for full parse)
///   out_tree: Output: receives tree handle on success
///
/// Returns 0 on success, error code on failure.
/// Caller must call talu_treesitter_tree_free() on the output tree.
/// The old_tree is NOT freed — caller retains ownership and must free it.
pub export fn talu_treesitter_parse_incremental(
    handle: ?*TreeSitterParserHandle,
    source: [*]const u8,
    source_len: u32,
    old_tree: ?*TreeSitterTreeHandle,
    out_tree: ?*?*TreeSitterTreeHandle,
) callconv(.c) i32 {
    capi_error.clearError();
    const allocator = std.heap.c_allocator;

    const result = out_tree orelse {
        capi_error.setError(error.InvalidArgument, "out_tree is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    result.* = null;

    const parser_ptr = (handle orelse {
        capi_error.setError(error.InvalidArgument, "parser handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }).toPtr();

    const source_slice = source[0..source_len];

    const old_tree_ptr: ?*const Tree = if (old_tree) |ot| ot.toPtr() else null;

    const tree_ptr = allocator.create(Tree) catch {
        capi_error.setError(error.OutOfMemory, "Failed to allocate tree", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };

    tree_ptr.* = parser_ptr.parse(source_slice, old_tree_ptr) catch {
        allocator.destroy(tree_ptr);
        capi_error.setError(error.Unexpected, "Incremental parse failed", .{});
        return @intFromEnum(error_codes.ErrorCode.internal_error);
    };

    result.* = TreeSitterTreeHandle.fromPtr(tree_ptr);
    return 0;
}

/// Edit a syntax tree to reflect a source code change.
///
/// You must call this before re-parsing with the old tree to enable
/// correct incremental parsing. Describes the edit in both byte offsets
/// and (row, column) coordinates.
///
/// Parameters:
///   handle: Tree handle to edit (mutated in place)
///   start_byte: Byte offset where the edit starts
///   old_end_byte: Byte offset where the old text ended
///   new_end_byte: Byte offset where the new text ends
///   start_row: Row (0-indexed) where the edit starts
///   start_column: Column (0-indexed) where the edit starts
///   old_end_row: Row where the old text ended
///   old_end_column: Column where the old text ended
///   new_end_row: Row where the new text ends
///   new_end_column: Column where the new text ends
///
/// Returns 0 on success, error code on failure (null handle).
pub export fn talu_treesitter_tree_edit(
    handle: ?*TreeSitterTreeHandle,
    start_byte: u32,
    old_end_byte: u32,
    new_end_byte: u32,
    start_row: u32,
    start_column: u32,
    old_end_row: u32,
    old_end_column: u32,
    new_end_row: u32,
    new_end_column: u32,
) callconv(.c) i32 {
    capi_error.clearError();

    const tree_ptr = (handle orelse {
        capi_error.setError(error.InvalidArgument, "tree handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }).toPtr();

    tree_ptr.edit(.{
        .start_byte = start_byte,
        .old_end_byte = old_end_byte,
        .new_end_byte = new_end_byte,
        .start_row = start_row,
        .start_column = start_column,
        .old_end_row = old_end_row,
        .old_end_column = old_end_column,
        .new_end_row = new_end_row,
        .new_end_column = new_end_column,
    });

    return 0;
}

/// Free a syntax tree handle.
pub export fn talu_treesitter_tree_free(
    handle: ?*TreeSitterTreeHandle,
) callconv(.c) void {
    const h = handle orelse return;
    const tree_ptr = h.toPtr();
    tree_ptr.deinit();
    std.heap.c_allocator.destroy(tree_ptr);
}

// =============================================================================
// AST Output
// =============================================================================

/// Get the S-expression representation of a parsed tree.
///
/// Parameters:
///   handle: Tree handle
///   out_str: Output: receives NUL-terminated S-expression string
///
/// Returns 0 on success, error code on failure.
/// Caller must call talu_free_string() on the output string.
pub export fn talu_treesitter_tree_sexp(
    handle: ?*TreeSitterTreeHandle,
    out_str: ?*[*:0]u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const allocator = std.heap.c_allocator;

    const result = out_str orelse {
        capi_error.setError(error.InvalidArgument, "out_str is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const tree_ptr = (handle orelse {
        capi_error.setError(error.InvalidArgument, "tree handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }).toPtr();

    const sexp = tree_ptr.rootNode().toSexp(allocator) catch {
        capi_error.setError(error.OutOfMemory, "Failed to generate S-expression", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };

    result.* = sexp.ptr;
    return 0;
}

// =============================================================================
// JSON AST Output
// =============================================================================

/// Get the JSON AST representation of a parsed tree.
///
/// Returns a JSON object: {"language":"<lang>","tree":{...recursive node JSON...}}
///
/// Parameters:
///   tree_handle: Parsed tree handle
///   source: Original source code bytes
///   source_len: Length of source
///   lang: Language name string
///   out_json: Output: receives NUL-terminated JSON string
///
/// Returns 0 on success, error code on failure.
/// Caller must call talu_free_string() on the output JSON.
pub export fn talu_treesitter_tree_json(
    tree_handle: ?*TreeSitterTreeHandle,
    source: [*]const u8,
    source_len: u32,
    lang: [*:0]const u8,
    out_json: ?*[*:0]u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const allocator = std.heap.c_allocator;

    const result = out_json orelse {
        capi_error.setError(error.InvalidArgument, "out_json is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const tree_ptr = (tree_handle orelse {
        capi_error.setError(error.InvalidArgument, "tree handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }).toPtr();

    const source_slice = source[0..source_len];
    const lang_str = std.mem.span(lang);

    const json = ast_mod.treeToJson(allocator, tree_ptr, source_slice, lang_str) catch {
        capi_error.setError(error.OutOfMemory, "Failed to generate JSON AST", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };

    result.* = json.ptr;
    return 0;
}

// =============================================================================
// Highlighting (stateless)
// =============================================================================

/// Parse and highlight source code in one call.
///
/// Returns a JSON array of token objects: [{"s":0,"e":3,"t":"keyword"}, ...]
/// where s=start byte, e=end byte, t=token type CSS class name.
///
/// Parameters:
///   source: Source code bytes
///   source_len: Length of source code
///   lang: Language name string (e.g., "python")
///   out_json: Output: receives NUL-terminated JSON string
///
/// Returns 0 on success, error code on failure.
/// Caller must call talu_free_string() on the output JSON.
pub export fn talu_treesitter_highlight(
    source: [*]const u8,
    source_len: u32,
    lang: [*:0]const u8,
    out_json: ?*[*:0]u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const allocator = std.heap.c_allocator;

    const result = out_json orelse {
        capi_error.setError(error.InvalidArgument, "out_json is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const lang_str = std.mem.span(lang);
    const language = Language.fromString(lang_str) orelse {
        capi_error.setError(error.InvalidArgument, "Unknown language: {s}", .{lang_str});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const source_slice = source[0..source_len];

    const tokens = highlight_mod.highlightTokens(allocator, source_slice, language) catch {
        capi_error.setError(error.Unexpected, "Highlight failed for {s}", .{lang_str});
        return @intFromEnum(error_codes.ErrorCode.internal_error);
    };
    defer allocator.free(tokens);

    // Build JSON array
    var json = std.ArrayList(u8).empty;
    defer json.deinit(allocator);

    json.append(allocator, '[') catch {
        capi_error.setError(error.OutOfMemory, "JSON allocation failed", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };

    for (tokens, 0..) |token, i| {
        if (i > 0) json.append(allocator, ',') catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);
        std.fmt.format(json.writer(allocator),
            \\{{"s":{d},"e":{d},"t":"{s}"}}
        , .{ token.start, token.end, token.token_type.cssClass() }) catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    }

    json.append(allocator, ']') catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    json.append(allocator, 0) catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);

    const owned = json.toOwnedSlice(allocator) catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    result.* = @ptrCast(owned.ptr);
    return 0;
}

/// Parse and highlight source code with rich token output.
///
/// Like talu_treesitter_highlight but includes node_kind, text, and positions:
/// [{"s":0,"e":3,"t":"keyword","nk":"keyword","tx":"def","sr":0,"sc":0,"er":0,"ec":3}, ...]
///
/// Parameters:
///   source: Source code bytes
///   source_len: Length of source code
///   lang: Language name string
///   out_json: Output: receives NUL-terminated JSON string
///
/// Returns 0 on success, error code on failure.
/// Caller must call talu_free_string() on the output JSON.
pub export fn talu_treesitter_highlight_rich(
    source: [*]const u8,
    source_len: u32,
    lang: [*:0]const u8,
    out_json: ?*[*:0]u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const allocator = std.heap.c_allocator;

    const result = out_json orelse {
        capi_error.setError(error.InvalidArgument, "out_json is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const lang_str = std.mem.span(lang);
    const language = Language.fromString(lang_str) orelse {
        capi_error.setError(error.InvalidArgument, "Unknown language: {s}", .{lang_str});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const source_slice = source[0..source_len];

    const tokens = highlight_mod.highlightTokensRich(allocator, source_slice, language) catch {
        capi_error.setError(error.Unexpected, "Rich highlight failed for {s}", .{lang_str});
        return @intFromEnum(error_codes.ErrorCode.internal_error);
    };
    defer allocator.free(tokens);

    // Build JSON array
    var json = std.ArrayList(u8).empty;
    defer json.deinit(allocator);

    json.append(allocator, '[') catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);

    for (tokens, 0..) |token, i| {
        if (i > 0) json.append(allocator, ',') catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);

        // Write token fields
        std.fmt.format(json.writer(allocator),
            \\{{"s":{d},"e":{d},"t":"{s}","nk":"{s}","tx":"
        , .{ token.start, token.end, token.token_type.cssClass(), token.node_kind }) catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);

        // Escape the text portion
        const text_slice = if (token.start < source_slice.len and token.end <= source_slice.len)
            source_slice[token.start..token.end]
        else
            "";
        for (text_slice) |ch| {
            switch (ch) {
                '"' => json.appendSlice(allocator, "\\\"") catch return @intFromEnum(error_codes.ErrorCode.out_of_memory),
                '\\' => json.appendSlice(allocator, "\\\\") catch return @intFromEnum(error_codes.ErrorCode.out_of_memory),
                '\n' => json.appendSlice(allocator, "\\n") catch return @intFromEnum(error_codes.ErrorCode.out_of_memory),
                '\r' => json.appendSlice(allocator, "\\r") catch return @intFromEnum(error_codes.ErrorCode.out_of_memory),
                '\t' => json.appendSlice(allocator, "\\t") catch return @intFromEnum(error_codes.ErrorCode.out_of_memory),
                else => json.append(allocator, ch) catch return @intFromEnum(error_codes.ErrorCode.out_of_memory),
            }
        }

        std.fmt.format(json.writer(allocator),
            \\","sr":{d},"sc":{d},"er":{d},"ec":{d}}}
        , .{ token.start_row, token.start_column, token.end_row, token.end_column }) catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    }

    json.append(allocator, ']') catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    json.append(allocator, 0) catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);

    const owned = json.toOwnedSlice(allocator) catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    result.* = @ptrCast(owned.ptr);
    return 0;
}

// =============================================================================
// Highlighting (from existing tree)
// =============================================================================

/// Highlight source code using an already-parsed tree.
///
/// Skips parsing — use with trees from talu_treesitter_parse or
/// talu_treesitter_parse_incremental for session-based workflows.
///
/// Parameters:
///   tree_handle: Parsed tree handle
///   source: Source code bytes (must match the tree)
///   source_len: Length of source code
///   lang: Language name string
///   out_json: Output: receives NUL-terminated JSON string
///
/// Returns 0 on success, error code on failure.
/// Caller must call talu_free_string() on the output JSON.
pub export fn talu_treesitter_highlight_from_tree(
    tree_handle: ?*TreeSitterTreeHandle,
    source: [*]const u8,
    source_len: u32,
    lang: [*:0]const u8,
    out_json: ?*[*:0]u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const allocator = std.heap.c_allocator;

    const result = out_json orelse {
        capi_error.setError(error.InvalidArgument, "out_json is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const tree_ptr = (tree_handle orelse {
        capi_error.setError(error.InvalidArgument, "tree handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }).toPtr();

    const lang_str = std.mem.span(lang);
    const language = Language.fromString(lang_str) orelse {
        capi_error.setError(error.InvalidArgument, "Unknown language: {s}", .{lang_str});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const source_slice = source[0..source_len];

    const tokens = highlight_mod.highlightTokensFromTree(allocator, source_slice, tree_ptr, language) catch {
        capi_error.setError(error.Unexpected, "Highlight from tree failed for {s}", .{lang_str});
        return @intFromEnum(error_codes.ErrorCode.internal_error);
    };
    defer allocator.free(tokens);

    var json = std.ArrayList(u8).empty;
    defer json.deinit(allocator);

    json.append(allocator, '[') catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);

    for (tokens, 0..) |token, i| {
        if (i > 0) json.append(allocator, ',') catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);
        std.fmt.format(json.writer(allocator),
            \\{{"s":{d},"e":{d},"t":"{s}"}}
        , .{ token.start, token.end, token.token_type.cssClass() }) catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    }

    json.append(allocator, ']') catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    json.append(allocator, 0) catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);

    const owned = json.toOwnedSlice(allocator) catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    result.* = @ptrCast(owned.ptr);
    return 0;
}

/// Highlight source code using an already-parsed tree, with rich output.
///
/// Like talu_treesitter_highlight_from_tree but includes node_kind, text,
/// and line/column positions in the output.
pub export fn talu_treesitter_highlight_rich_from_tree(
    tree_handle: ?*TreeSitterTreeHandle,
    source: [*]const u8,
    source_len: u32,
    lang: [*:0]const u8,
    out_json: ?*[*:0]u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const allocator = std.heap.c_allocator;

    const result = out_json orelse {
        capi_error.setError(error.InvalidArgument, "out_json is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const tree_ptr = (tree_handle orelse {
        capi_error.setError(error.InvalidArgument, "tree handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }).toPtr();

    const lang_str = std.mem.span(lang);
    const language = Language.fromString(lang_str) orelse {
        capi_error.setError(error.InvalidArgument, "Unknown language: {s}", .{lang_str});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const source_slice = source[0..source_len];

    const tokens = highlight_mod.highlightTokensRichFromTree(allocator, source_slice, tree_ptr, language) catch {
        capi_error.setError(error.Unexpected, "Rich highlight from tree failed for {s}", .{lang_str});
        return @intFromEnum(error_codes.ErrorCode.internal_error);
    };
    defer allocator.free(tokens);

    var json = std.ArrayList(u8).empty;
    defer json.deinit(allocator);

    json.append(allocator, '[') catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);

    for (tokens, 0..) |token, i| {
        if (i > 0) json.append(allocator, ',') catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);

        std.fmt.format(json.writer(allocator),
            \\{{"s":{d},"e":{d},"t":"{s}","nk":"{s}","tx":"
        , .{ token.start, token.end, token.token_type.cssClass(), token.node_kind }) catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);

        const text_slice = if (token.start < source_slice.len and token.end <= source_slice.len)
            source_slice[token.start..token.end]
        else
            "";
        for (text_slice) |ch| {
            switch (ch) {
                '"' => json.appendSlice(allocator, "\\\"") catch return @intFromEnum(error_codes.ErrorCode.out_of_memory),
                '\\' => json.appendSlice(allocator, "\\\\") catch return @intFromEnum(error_codes.ErrorCode.out_of_memory),
                '\n' => json.appendSlice(allocator, "\\n") catch return @intFromEnum(error_codes.ErrorCode.out_of_memory),
                '\r' => json.appendSlice(allocator, "\\r") catch return @intFromEnum(error_codes.ErrorCode.out_of_memory),
                '\t' => json.appendSlice(allocator, "\\t") catch return @intFromEnum(error_codes.ErrorCode.out_of_memory),
                else => json.append(allocator, ch) catch return @intFromEnum(error_codes.ErrorCode.out_of_memory),
            }
        }

        std.fmt.format(json.writer(allocator),
            \\","sr":{d},"sc":{d},"er":{d},"ec":{d}}}
        , .{ token.start_row, token.start_column, token.end_row, token.end_column }) catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    }

    json.append(allocator, ']') catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    json.append(allocator, 0) catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);

    const owned = json.toOwnedSlice(allocator) catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    result.* = @ptrCast(owned.ptr);
    return 0;
}

// =============================================================================
// Query API
// =============================================================================

/// Compile a query pattern for the given language.
///
/// Parameters:
///   lang: Language name string
///   pattern: S-expression query pattern bytes
///   pattern_len: Length of pattern
///   out_handle: Output: receives query handle on success
///
/// Returns 0 on success, error code on failure.
/// Caller must call talu_treesitter_query_free() when done.
pub export fn talu_treesitter_query_create(
    lang: [*:0]const u8,
    pattern: [*]const u8,
    pattern_len: u32,
    out_handle: ?*?*TreeSitterQueryHandle,
) callconv(.c) i32 {
    capi_error.clearError();
    const allocator = std.heap.c_allocator;

    const result = out_handle orelse {
        capi_error.setError(error.InvalidArgument, "out_handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    result.* = null;

    const lang_str = std.mem.span(lang);
    const language = Language.fromString(lang_str) orelse {
        capi_error.setError(error.InvalidArgument, "Unknown language: {s}", .{lang_str});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const query_ptr = allocator.create(Query) catch {
        capi_error.setError(error.OutOfMemory, "Failed to allocate query", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };

    query_ptr.* = Query.init(language, pattern[0..pattern_len]) catch {
        allocator.destroy(query_ptr);
        capi_error.setError(error.Unexpected, "Query compilation failed", .{});
        return @intFromEnum(error_codes.ErrorCode.internal_error);
    };

    result.* = TreeSitterQueryHandle.fromPtr(query_ptr);
    return 0;
}

/// Free a query handle.
pub export fn talu_treesitter_query_free(
    handle: ?*TreeSitterQueryHandle,
) callconv(.c) void {
    const h = handle orelse return;
    const query_ptr = h.toPtr();
    query_ptr.deinit();
    std.heap.c_allocator.destroy(query_ptr);
}

/// Execute a query against a tree and return matches as JSON.
///
/// Returns a JSON array of match objects:
/// [{"id":0,"captures":[{"name":"id","start":0,"end":3,"text":"foo"}]}, ...]
///
/// Parameters:
///   query_handle: Compiled query handle
///   tree_handle: Parsed tree handle
///   source: Original source code bytes (for text extraction)
///   source_len: Length of source
///   out_json: Output: receives NUL-terminated JSON string
///
/// Returns 0 on success, error code on failure.
/// Caller must call talu_free_string() on the output JSON.
pub export fn talu_treesitter_query_exec(
    query_handle: ?*TreeSitterQueryHandle,
    tree_handle: ?*TreeSitterTreeHandle,
    source: [*]const u8,
    source_len: u32,
    out_json: ?*[*:0]u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const allocator = std.heap.c_allocator;

    const result = out_json orelse {
        capi_error.setError(error.InvalidArgument, "out_json is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const query_ptr = (query_handle orelse {
        capi_error.setError(error.InvalidArgument, "query handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }).toPtr();

    const tree_ptr = (tree_handle orelse {
        capi_error.setError(error.InvalidArgument, "tree handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }).toPtr();

    const source_slice = source[0..source_len];

    var cursor = QueryCursor.init() catch {
        capi_error.setError(error.OutOfMemory, "Failed to create query cursor", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };
    defer cursor.deinit();

    cursor.exec(query_ptr, tree_ptr.rootNode());

    // Build JSON array of matches
    var json = std.ArrayList(u8).empty;
    defer json.deinit(allocator);

    json.append(allocator, '[') catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);

    var match_count: usize = 0;
    while (cursor.nextMatch()) |match| {
        if (match_count > 0) json.append(allocator, ',') catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);

        std.fmt.format(json.writer(allocator), "{{\"id\":{d},\"captures\":[", .{match.id}) catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);

        for (match.captures, 0..) |capture, ci| {
            if (ci > 0) json.append(allocator, ',') catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);

            const cap_node = node_mod.Node{ .raw = capture.node };
            const cap_name = query_ptr.captureNameForId(capture.index);
            const start = cap_node.startByte();
            const end = cap_node.endByte();
            const text_slice = cap_node.text(source_slice);

            // Escape text for JSON
            std.fmt.format(json.writer(allocator), "{{\"name\":\"{s}\",\"start\":{d},\"end\":{d},\"text\":\"", .{ cap_name, start, end }) catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);

            for (text_slice) |ch| {
                switch (ch) {
                    '"' => json.appendSlice(allocator, "\\\"") catch return @intFromEnum(error_codes.ErrorCode.out_of_memory),
                    '\\' => json.appendSlice(allocator, "\\\\") catch return @intFromEnum(error_codes.ErrorCode.out_of_memory),
                    '\n' => json.appendSlice(allocator, "\\n") catch return @intFromEnum(error_codes.ErrorCode.out_of_memory),
                    '\r' => json.appendSlice(allocator, "\\r") catch return @intFromEnum(error_codes.ErrorCode.out_of_memory),
                    '\t' => json.appendSlice(allocator, "\\t") catch return @intFromEnum(error_codes.ErrorCode.out_of_memory),
                    else => json.append(allocator, ch) catch return @intFromEnum(error_codes.ErrorCode.out_of_memory),
                }
            }

            json.appendSlice(allocator, "\"}") catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);
        }

        json.appendSlice(allocator, "]}") catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);
        match_count += 1;
    }

    json.append(allocator, ']') catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    json.append(allocator, 0) catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);

    const owned = json.toOwnedSlice(allocator) catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    result.* = @ptrCast(owned.ptr);
    return 0;
}

// =============================================================================
// Supported Languages
// =============================================================================

/// Get a comma-separated list of supported language names.
///
/// Returns a NUL-terminated string like "python,javascript,typescript,...".
/// Caller must call talu_free_string() on the output string.
pub export fn talu_treesitter_languages(
    out_str: ?*[*:0]u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const allocator = std.heap.c_allocator;

    const result = out_str orelse {
        capi_error.setError(error.InvalidArgument, "out_str is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const langs = "python,javascript,typescript,rust,go,c,zig,json,html,css,bash";
    const duped = allocator.dupeZ(u8, langs) catch {
        capi_error.setError(error.OutOfMemory, "Allocation failed", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };

    result.* = duped.ptr;
    return 0;
}

// =============================================================================
// Language Detection
// =============================================================================

/// Detect language from a filename or file extension.
///
/// Parameters:
///   filename: NUL-terminated filename or path (e.g., "main.py", "src/app.js")
///   out_lang: Output: receives NUL-terminated language name string
///
/// Returns 0 on success, error code if language not recognized.
/// Caller must call talu_free_string() on the output string.
pub export fn talu_treesitter_language_from_filename(
    filename: [*:0]const u8,
    out_lang: ?*[*:0]u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const allocator = std.heap.c_allocator;

    const result = out_lang orelse {
        capi_error.setError(error.InvalidArgument, "out_lang is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const filename_str = std.mem.span(filename);
    const language = Language.fromFilename(filename_str) orelse {
        capi_error.setError(error.InvalidArgument, "Unrecognized file type: {s}", .{filename_str});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const duped = allocator.dupeZ(u8, language.name()) catch {
        capi_error.setError(error.OutOfMemory, "Allocation failed", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };

    result.* = duped.ptr;
    return 0;
}

// =============================================================================
// Call Graph Analysis
// =============================================================================

/// Extract callable definitions and import aliases from source code.
///
/// Returns a JSON object: {"callables":[...],"aliases_count":N}
///
/// Parameters:
///   source: Source code bytes
///   source_len: Length of source code
///   lang: Language name string (e.g., "python", "rust", "javascript")
///   file_path: NUL-terminated file path for FQN generation
///   project_root: NUL-terminated project root for relative paths
///   out_json: Output: receives NUL-terminated JSON string
///
/// Returns 0 on success, error code on failure.
/// Caller must call talu_free_string() on the output JSON.
pub export fn talu_treesitter_extract_callables(
    source: [*]const u8,
    source_len: u32,
    lang: [*:0]const u8,
    file_path: [*:0]const u8,
    project_root: [*:0]const u8,
    out_json: ?*[*:0]u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const allocator = std.heap.c_allocator;

    const out = out_json orelse {
        capi_error.setError(error.InvalidArgument, "out_json is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const lang_str = std.mem.span(lang);
    const language = Language.fromString(lang_str) orelse {
        capi_error.setError(error.InvalidArgument, "Unknown language: {s}", .{lang_str});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const source_slice = source[0..source_len];
    const file_path_str = std.mem.span(file_path);
    const project_root_str = std.mem.span(project_root);

    const extraction = graph_mod.extractCallablesAndAliases(
        allocator,
        source_slice,
        language,
        file_path_str,
        project_root_str,
    ) catch {
        capi_error.setError(error.Unexpected, "Callable extraction failed for {s}", .{lang_str});
        return @intFromEnum(error_codes.ErrorCode.internal_error);
    };
    defer {
        for (extraction.callables) |c| {
            allocator.free(c.fqn);
            allocator.free(c.parameters);
        }
        allocator.free(extraction.callables);
        for (extraction.aliases) |a| {
            allocator.free(a.alias_fqn);
        }
        allocator.free(extraction.aliases);
    }

    const callables_json = graph_mod.callablesToJson(allocator, extraction.callables) catch {
        capi_error.setError(error.OutOfMemory, "JSON serialization failed", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };
    defer allocator.free(callables_json);

    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(allocator);

    const aliases_json = graph_mod.aliasesToJson(allocator, extraction.aliases) catch {
        capi_error.setError(error.OutOfMemory, "Alias JSON serialization failed", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };
    defer allocator.free(aliases_json);

    buf.appendSlice(allocator, "{\"callables\":") catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    buf.appendSlice(allocator, callables_json) catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    buf.appendSlice(allocator, ",\"aliases\":") catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    buf.appendSlice(allocator, aliases_json) catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    buf.append(allocator, '}') catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    buf.append(allocator, 0) catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);

    const owned = buf.toOwnedSlice(allocator) catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    out.* = @ptrCast(owned.ptr);
    return 0;
}

/// Extract call sites from source code with import-aware resolution.
///
/// Returns a JSON array of call site objects with resolved paths.
///
/// Parameters:
///   source: Source code bytes
///   source_len: Length of source code
///   lang: Language name string
///   definer_fqn: NUL-terminated FQN of the containing callable
///   file_path: NUL-terminated file path (for module context)
///   project_root: NUL-terminated project root path
///   out_json: Output: receives NUL-terminated JSON string
///
/// Returns 0 on success, error code on failure.
/// Caller must call talu_free_string() on the output JSON.
pub export fn talu_treesitter_extract_call_sites(
    source: [*]const u8,
    source_len: u32,
    lang: [*:0]const u8,
    definer_fqn: [*:0]const u8,
    file_path: [*:0]const u8,
    project_root: [*:0]const u8,
    out_json: ?*[*:0]u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const allocator = std.heap.c_allocator;

    const out = out_json orelse {
        capi_error.setError(error.InvalidArgument, "out_json is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const lang_str = std.mem.span(lang);
    const language = Language.fromString(lang_str) orelse {
        capi_error.setError(error.InvalidArgument, "Unknown language: {s}", .{lang_str});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const source_slice = source[0..source_len];
    const definer_fqn_str = std.mem.span(definer_fqn);
    const file_path_str = std.mem.span(file_path);
    const project_root_str = std.mem.span(project_root);

    const call_sites = graph_mod.extractCallSites(
        allocator,
        source_slice,
        language,
        definer_fqn_str,
        file_path_str,
        project_root_str,
    ) catch {
        capi_error.setError(error.Unexpected, "Call site extraction failed for {s}", .{lang_str});
        return @intFromEnum(error_codes.ErrorCode.internal_error);
    };
    defer {
        for (call_sites) |cs| {
            allocator.free(cs.potential_resolved_paths);
            allocator.free(cs.arguments);
        }
        allocator.free(call_sites);
    }

    const json = graph_mod.callSitesToJson(allocator, call_sites) catch {
        capi_error.setError(error.OutOfMemory, "JSON serialization failed", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };

    out.* = json.ptr;
    return 0;
}
