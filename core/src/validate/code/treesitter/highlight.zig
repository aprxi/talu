//! Syntax highlighting via tree-sitter queries.
//!
//! Uses each language's highlight query (.scm) to extract tokens with
//! semantic types (keyword, function, string, etc.) from source code.
//!
//! Thread safety: highlightTokens is safe to call from multiple threads
//! concurrently (each call creates its own parser/query state).

const std = @import("std");
const Language = @import("language.zig").Language;
const parser_mod = @import("parser.zig");
const query_mod = @import("query.zig");
const query_cache = @import("query_cache.zig");
const node_mod = @import("node.zig");
const token_refine = @import("token_refine.zig");

pub const TokenType = enum {
    plain,
    comment,
    string,
    number,
    keyword,
    operator,
    punctuation,
    identifier,
    function,
    property,
    variable,
    type_name,
    literal,
    builtin,
    decorator,
    function_call,
    string_interpolation,
    attribute,
    regex,

    /// Returns the CSS class name for syntax highlighting (e.g., "syntax-keyword").
    pub fn cssClass(self: TokenType) []const u8 {
        return switch (self) {
            .plain => "syntax-plain",
            .comment => "syntax-comment",
            .string => "syntax-string",
            .number => "syntax-number",
            .keyword => "syntax-keyword",
            .operator => "syntax-operator",
            .punctuation => "syntax-punctuation",
            .identifier => "syntax-identifier",
            .function => "syntax-function",
            .property => "syntax-property",
            .variable => "syntax-variable",
            .type_name => "syntax-type",
            .literal => "syntax-literal",
            .builtin => "syntax-builtin",
            .decorator => "syntax-decorator",
            .function_call => "syntax-function-call",
            .string_interpolation => "syntax-string-interpolation",
            .attribute => "syntax-attribute",
            .regex => "syntax-regex",
        };
    }
};

pub const Token = struct {
    start: u32,
    end: u32,
    token_type: TokenType,
};

/// Extended token with node kind, text source offsets, and line/column positions.
pub const RichToken = struct {
    start: u32,
    end: u32,
    token_type: TokenType,
    node_kind: []const u8,
    start_row: u32,
    start_column: u32,
    end_row: u32,
    end_column: u32,
};

/// Map tree-sitter highlight capture names to TokenType.
fn captureNameToTokenType(capture_name: []const u8) TokenType {
    // Tree-sitter highlight captures use @name convention.
    // Common captures: @keyword, @function, @string, @number, @comment, etc.
    const map = std.StaticStringMap(TokenType).initComptime(.{
        .{ "keyword", .keyword },
        .{ "keyword.function", .keyword },
        .{ "keyword.return", .keyword },
        .{ "keyword.operator", .operator },
        .{ "keyword.import", .keyword },
        .{ "keyword.conditional", .keyword },
        .{ "keyword.repeat", .keyword },
        .{ "keyword.exception", .keyword },
        .{ "keyword.type", .keyword },
        .{ "keyword.modifier", .keyword },
        .{ "keyword.coroutine", .keyword },
        .{ "keyword.directive", .keyword },
        .{ "keyword.storage", .keyword },
        .{ "function", .function },
        .{ "function.builtin", .builtin },
        .{ "function.call", .function_call },
        .{ "function.method", .function },
        .{ "function.method.call", .function_call },
        .{ "function.macro", .function },
        .{ "method", .function },
        .{ "method.call", .function_call },
        .{ "string", .string },
        .{ "string.special", .string },
        .{ "string.escape", .string },
        .{ "string.regex", .regex },
        .{ "string.special.path", .string },
        .{ "string.special.url", .string },
        .{ "string.special.symbol", .string },
        .{ "number", .number },
        .{ "number.float", .number },
        .{ "float", .number },
        .{ "comment", .comment },
        .{ "comment.documentation", .comment },
        .{ "comment.line", .comment },
        .{ "comment.block", .comment },
        .{ "operator", .operator },
        .{ "punctuation", .punctuation },
        .{ "punctuation.bracket", .punctuation },
        .{ "punctuation.delimiter", .punctuation },
        .{ "punctuation.special", .punctuation },
        .{ "type", .type_name },
        .{ "type.builtin", .type_name },
        .{ "type.definition", .type_name },
        .{ "type.qualifier", .keyword },
        .{ "constructor", .type_name },
        .{ "variable", .variable },
        .{ "variable.builtin", .builtin },
        .{ "variable.parameter", .variable },
        .{ "variable.member", .property },
        .{ "property", .property },
        .{ "constant", .literal },
        .{ "constant.builtin", .builtin },
        .{ "boolean", .literal },
        .{ "label", .identifier },
        .{ "tag", .keyword },
        .{ "tag.delimiter", .punctuation },
        .{ "tag.attribute", .property },
        .{ "attribute", .decorator },
        .{ "attribute.builtin", .decorator },
        .{ "namespace", .type_name },
        .{ "module", .type_name },
        .{ "include", .keyword },
        .{ "conditional", .keyword },
        .{ "repeat", .keyword },
        .{ "exception", .keyword },
        .{ "escape", .string },
        .{ "string.interpolation", .string_interpolation },
    });

    return map.get(capture_name) orelse .plain;
}

/// Extract syntax highlighting tokens from source code.
/// Caller owns the returned slice (allocated with `allocator`).
pub fn highlightTokens(
    allocator: std.mem.Allocator,
    source: []const u8,
    language: Language,
) ![]Token {
    if (source.len == 0) return allocator.alloc(Token, 0);

    // Parse
    var p = try parser_mod.Parser.init(language);
    defer p.deinit();

    var tree = try p.parse(source, null);
    defer tree.deinit();

    // Get cached highlight query (compiled once, reused across calls)
    const query = query_cache.getHighlightQuery(language) catch {
        return allocator.alloc(Token, 0);
    };

    // Execute query
    var cursor = try query_mod.QueryCursor.init();
    defer cursor.deinit();

    cursor.exec(query, tree.rootNode());

    // Collect tokens
    var tokens = std.ArrayList(Token).empty;
    errdefer tokens.deinit(allocator);

    while (cursor.nextMatch()) |match| {
        for (match.captures) |capture| {
            const capture_name = query.captureNameForId(capture.index);
            const token_type = captureNameToTokenType(capture_name);

            if (token_type == .plain) continue;

            const node = node_mod.Node{ .raw = capture.node };
            const start = node.startByte();
            const end = node.endByte();

            if (start >= end) continue;
            if (end > source.len) continue;

            try tokens.append(allocator, .{
                .start = start,
                .end = end,
                .token_type = token_type,
            });
        }
    }

    // Apply per-language refinement
    token_refine.refineTokens(tokens.items, source, language);

    // Sort by start position for deterministic output
    std.mem.sort(Token, tokens.items, {}, struct {
        fn lessThan(_: void, a: Token, b: Token) bool {
            if (a.start != b.start) return a.start < b.start;
            return a.end < b.end;
        }
    }.lessThan);

    return tokens.toOwnedSlice(allocator);
}

/// Like highlightTokens, but includes node kind and line/column position info.
/// The node_kind slices point into static tree-sitter grammar data (no allocation).
/// Caller owns the returned slice.
pub fn highlightTokensRich(
    allocator: std.mem.Allocator,
    source: []const u8,
    language: Language,
) ![]RichToken {
    if (source.len == 0) return allocator.alloc(RichToken, 0);

    var p = try parser_mod.Parser.init(language);
    defer p.deinit();

    var tree = try p.parse(source, null);
    defer tree.deinit();

    const query = query_cache.getHighlightQuery(language) catch {
        return allocator.alloc(RichToken, 0);
    };

    var cursor = try query_mod.QueryCursor.init();
    defer cursor.deinit();

    cursor.exec(query, tree.rootNode());

    var tokens = std.ArrayList(RichToken).empty;
    errdefer tokens.deinit(allocator);

    // Also collect basic tokens for refinement
    var basic_tokens = std.ArrayList(Token).empty;
    defer basic_tokens.deinit(allocator);

    while (cursor.nextMatch()) |match| {
        for (match.captures) |capture| {
            const capture_name = query.captureNameForId(capture.index);
            const token_type = captureNameToTokenType(capture_name);

            if (token_type == .plain) continue;

            const node = node_mod.Node{ .raw = capture.node };
            const start = node.startByte();
            const end = node.endByte();

            if (start >= end) continue;
            if (end > source.len) continue;

            const sp = node.startPoint();
            const ep = node.endPoint();

            try tokens.append(allocator, .{
                .start = start,
                .end = end,
                .token_type = token_type,
                .node_kind = node.kind(),
                .start_row = sp.row,
                .start_column = sp.column,
                .end_row = ep.row,
                .end_column = ep.column,
            });

            try basic_tokens.append(allocator, .{
                .start = start,
                .end = end,
                .token_type = token_type,
            });
        }
    }

    // Apply per-language refinement on basic tokens, then copy back
    token_refine.refineTokens(basic_tokens.items, source, language);
    for (basic_tokens.items, 0..) |bt, i| {
        tokens.items[i].token_type = bt.token_type;
    }

    std.mem.sort(RichToken, tokens.items, {}, struct {
        fn lessThan(_: void, a: RichToken, b: RichToken) bool {
            if (a.start != b.start) return a.start < b.start;
            return a.end < b.end;
        }
    }.lessThan);

    return tokens.toOwnedSlice(allocator);
}

// =============================================================================
// Tests
// =============================================================================

test "highlightTokens returns tokens for Python" {
    const source = "def hello():\n    return 42\n";
    const tokens = try highlightTokens(std.testing.allocator, source, .python);
    defer std.testing.allocator.free(tokens);

    try std.testing.expect(tokens.len > 0);

    // Should find a keyword ("def") and a number (42)
    var found_keyword = false;
    var found_number = false;
    for (tokens) |token| {
        const text = source[token.start..token.end];
        if (token.token_type == .keyword and std.mem.eql(u8, text, "def")) found_keyword = true;
        if (token.token_type == .number and std.mem.eql(u8, text, "42")) found_number = true;
    }
    try std.testing.expect(found_keyword);
    try std.testing.expect(found_number);
}

test "highlightTokens returns tokens for JavaScript" {
    const source = "function greet() { return \"hello\"; }";
    const tokens = try highlightTokens(std.testing.allocator, source, .javascript);
    defer std.testing.allocator.free(tokens);

    try std.testing.expect(tokens.len > 0);

    var found_keyword = false;
    for (tokens) |token| {
        const text = source[token.start..token.end];
        if (token.token_type == .keyword and std.mem.eql(u8, text, "function")) found_keyword = true;
    }
    try std.testing.expect(found_keyword);
}

test "highlightTokens returns tokens for Rust" {
    const source = "fn main() { let x = 42; }";
    const tokens = try highlightTokens(std.testing.allocator, source, .rust);
    defer std.testing.allocator.free(tokens);

    try std.testing.expect(tokens.len > 0);
}

test "highlightTokens handles empty source" {
    const tokens = try highlightTokens(std.testing.allocator, "", .python);
    defer std.testing.allocator.free(tokens);

    try std.testing.expectEqual(@as(usize, 0), tokens.len);
}

test "highlightTokens tokens are sorted by position" {
    const source = "x = 1\ny = 2\nz = 3\n";
    const tokens = try highlightTokens(std.testing.allocator, source, .python);
    defer std.testing.allocator.free(tokens);

    for (1..tokens.len) |i| {
        try std.testing.expect(tokens[i].start >= tokens[i - 1].start);
    }
}

test "TokenType.cssClass returns correct prefix" {
    try std.testing.expectEqualStrings("syntax-keyword", TokenType.keyword.cssClass());
    try std.testing.expectEqualStrings("syntax-function", TokenType.function.cssClass());
    try std.testing.expectEqualStrings("syntax-string", TokenType.string.cssClass());
    try std.testing.expectEqualStrings("syntax-comment", TokenType.comment.cssClass());
    try std.testing.expectEqualStrings("syntax-number", TokenType.number.cssClass());
}

test "TokenType.cssClass returns correct names for new types" {
    try std.testing.expectEqualStrings("syntax-string-interpolation", TokenType.string_interpolation.cssClass());
    try std.testing.expectEqualStrings("syntax-attribute", TokenType.attribute.cssClass());
    try std.testing.expectEqualStrings("syntax-regex", TokenType.regex.cssClass());
}

test "captureNameToTokenType maps known names" {
    try std.testing.expectEqual(TokenType.keyword, captureNameToTokenType("keyword"));
    try std.testing.expectEqual(TokenType.function, captureNameToTokenType("function"));
    try std.testing.expectEqual(TokenType.string, captureNameToTokenType("string"));
    try std.testing.expectEqual(TokenType.number, captureNameToTokenType("number"));
    try std.testing.expectEqual(TokenType.comment, captureNameToTokenType("comment"));
    try std.testing.expectEqual(TokenType.operator, captureNameToTokenType("operator"));
    try std.testing.expectEqual(TokenType.type_name, captureNameToTokenType("type"));
    try std.testing.expectEqual(TokenType.variable, captureNameToTokenType("variable"));
    try std.testing.expectEqual(TokenType.property, captureNameToTokenType("property"));
    try std.testing.expectEqual(TokenType.builtin, captureNameToTokenType("function.builtin"));
    try std.testing.expectEqual(TokenType.plain, captureNameToTokenType("unknown_capture"));
}

/// Extract syntax highlighting tokens using an already-parsed tree.
/// Skips parsing — use when the caller owns the tree (e.g., session-based incremental parsing).
/// Caller owns the returned slice.
pub fn highlightTokensFromTree(
    allocator: std.mem.Allocator,
    source: []const u8,
    tree: *const parser_mod.Tree,
    language: Language,
) ![]Token {
    if (source.len == 0) return allocator.alloc(Token, 0);

    const query = query_cache.getHighlightQuery(language) catch {
        return allocator.alloc(Token, 0);
    };

    var cursor = try query_mod.QueryCursor.init();
    defer cursor.deinit();

    cursor.exec(query, tree.rootNode());

    var tokens = std.ArrayList(Token).empty;
    errdefer tokens.deinit(allocator);

    while (cursor.nextMatch()) |match| {
        for (match.captures) |capture| {
            const capture_name = query.captureNameForId(capture.index);
            const token_type = captureNameToTokenType(capture_name);

            if (token_type == .plain) continue;

            const node = node_mod.Node{ .raw = capture.node };
            const start = node.startByte();
            const end = node.endByte();

            if (start >= end) continue;
            if (end > source.len) continue;

            try tokens.append(allocator, .{
                .start = start,
                .end = end,
                .token_type = token_type,
            });
        }
    }

    token_refine.refineTokens(tokens.items, source, language);

    std.mem.sort(Token, tokens.items, {}, struct {
        fn lessThan(_: void, a: Token, b: Token) bool {
            if (a.start != b.start) return a.start < b.start;
            return a.end < b.end;
        }
    }.lessThan);

    return tokens.toOwnedSlice(allocator);
}

/// Like highlightTokensFromTree, but with rich position info.
/// Caller owns the returned slice.
pub fn highlightTokensRichFromTree(
    allocator: std.mem.Allocator,
    source: []const u8,
    tree: *const parser_mod.Tree,
    language: Language,
) ![]RichToken {
    if (source.len == 0) return allocator.alloc(RichToken, 0);

    const query = query_cache.getHighlightQuery(language) catch {
        return allocator.alloc(RichToken, 0);
    };

    var cursor = try query_mod.QueryCursor.init();
    defer cursor.deinit();

    cursor.exec(query, tree.rootNode());

    var tokens = std.ArrayList(RichToken).empty;
    errdefer tokens.deinit(allocator);

    var basic_tokens = std.ArrayList(Token).empty;
    defer basic_tokens.deinit(allocator);

    while (cursor.nextMatch()) |match| {
        for (match.captures) |capture| {
            const capture_name = query.captureNameForId(capture.index);
            const token_type = captureNameToTokenType(capture_name);

            if (token_type == .plain) continue;

            const node = node_mod.Node{ .raw = capture.node };
            const start = node.startByte();
            const end = node.endByte();

            if (start >= end) continue;
            if (end > source.len) continue;

            const sp = node.startPoint();
            const ep = node.endPoint();

            try tokens.append(allocator, .{
                .start = start,
                .end = end,
                .token_type = token_type,
                .node_kind = node.kind(),
                .start_row = sp.row,
                .start_column = sp.column,
                .end_row = ep.row,
                .end_column = ep.column,
            });

            try basic_tokens.append(allocator, .{
                .start = start,
                .end = end,
                .token_type = token_type,
            });
        }
    }

    token_refine.refineTokens(basic_tokens.items, source, language);
    for (basic_tokens.items, 0..) |bt, i| {
        tokens.items[i].token_type = bt.token_type;
    }

    std.mem.sort(RichToken, tokens.items, {}, struct {
        fn lessThan(_: void, a: RichToken, b: RichToken) bool {
            if (a.start != b.start) return a.start < b.start;
            return a.end < b.end;
        }
    }.lessThan);

    return tokens.toOwnedSlice(allocator);
}

test "highlightTokensFromTree matches highlightTokens output" {
    const source = "def hello():\n    return 42\n";

    // Parse manually
    var p = try parser_mod.Parser.init(.python);
    defer p.deinit();
    var tree = try p.parse(source, null);
    defer tree.deinit();

    // Highlight from tree
    const from_tree = try highlightTokensFromTree(std.testing.allocator, source, &tree, .python);
    defer std.testing.allocator.free(from_tree);

    // Highlight stateless
    const stateless = try highlightTokens(std.testing.allocator, source, .python);
    defer std.testing.allocator.free(stateless);

    // Should produce identical results
    try std.testing.expectEqual(stateless.len, from_tree.len);
    for (stateless, from_tree) |s, f| {
        try std.testing.expectEqual(s.start, f.start);
        try std.testing.expectEqual(s.end, f.end);
        try std.testing.expectEqual(s.token_type, f.token_type);
    }
}

test "highlightTokensRichFromTree matches highlightTokensRich output" {
    const source = "def hello():\n    return 42\n";

    var p = try parser_mod.Parser.init(.python);
    defer p.deinit();
    var tree = try p.parse(source, null);
    defer tree.deinit();

    const from_tree = try highlightTokensRichFromTree(std.testing.allocator, source, &tree, .python);
    defer std.testing.allocator.free(from_tree);

    const stateless = try highlightTokensRich(std.testing.allocator, source, .python);
    defer std.testing.allocator.free(stateless);

    try std.testing.expectEqual(stateless.len, from_tree.len);
    for (stateless, from_tree) |s, f| {
        try std.testing.expectEqual(s.start, f.start);
        try std.testing.expectEqual(s.end, f.end);
        try std.testing.expectEqual(s.token_type, f.token_type);
    }
}

test "highlightTokensRich includes position info" {
    const source = "def hello():\n    return 42\n";
    const tokens = try highlightTokensRich(std.testing.allocator, source, .python);
    defer std.testing.allocator.free(tokens);

    try std.testing.expect(tokens.len > 0);

    // First token should have position data
    const first = tokens[0];
    try std.testing.expect(first.node_kind.len > 0);
    // start_row should be valid (0-indexed)
    try std.testing.expect(first.start_row <= 1);
}

test "highlightTokensRich handles empty source" {
    const tokens = try highlightTokensRich(std.testing.allocator, "", .python);
    defer std.testing.allocator.free(tokens);

    try std.testing.expectEqual(@as(usize, 0), tokens.len);
}
