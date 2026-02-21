//! Tree-sitter code parsing, querying, and syntax highlighting.
//!
//! Provides safe Zig wrappers around the tree-sitter C library for parsing
//! source code into concrete syntax trees, executing pattern queries, and
//! extracting syntax highlighting tokens.
//!
//! Supports 11 languages: Python, JavaScript, TypeScript, Rust, Go, C, Zig,
//! JSON, HTML, CSS, Bash.

pub const language = @import("language.zig");
pub const parser = @import("parser.zig");
pub const node = @import("node.zig");
pub const query = @import("query.zig");
pub const highlight = @import("highlight.zig");
pub const query_cache = @import("query_cache.zig");
pub const ast = @import("ast.zig");
pub const token_refine = @import("token_refine.zig");
pub const json_helpers = @import("json_helpers.zig");
pub const graph = @import("graph/root.zig");

// Re-export commonly used types at module level
pub const Language = language.Language;
pub const Parser = parser.Parser;
pub const Tree = parser.Tree;
pub const Node = node.Node;
pub const TreeCursor = node.TreeCursor;
pub const Point = node.Point;
pub const Query = query.Query;
pub const QueryCursor = query.QueryCursor;
pub const QueryMatch = query.QueryMatch;
pub const Token = highlight.Token;
pub const TokenType = highlight.TokenType;
pub const highlightTokens = highlight.highlightTokens;
pub const treeToJson = ast.treeToJson;
pub const RichToken = highlight.RichToken;
pub const highlightTokensRich = highlight.highlightTokensRich;
pub const tokensToJson = highlight.tokensToJson;
pub const richTokensToJson = highlight.richTokensToJson;
pub const queryMatchesToJson = query.queryMatchesToJson;

// Force compilation/testing of all submodules
comptime {
    _ = language;
    _ = parser;
    _ = node;
    _ = query;
    _ = highlight;
    _ = query_cache;
    _ = ast;
    _ = token_refine;
    _ = json_helpers;
    _ = graph;
}
