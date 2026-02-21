//! Per-language token refinement.
//!
//! Post-processes tokens after the generic highlight query pass to apply
//! language-specific rules that cannot be expressed in .scm queries alone.
//!
//! Thread safety: refineTokens is pure (no shared state). Safe to call concurrently.

const std = @import("std");
const highlight = @import("highlight.zig");
const TokenType = highlight.TokenType;
const Token = highlight.Token;
const Language = @import("language.zig").Language;

/// Refine token types for the given language.
/// Modifies token_type in-place where language-specific rules override.
pub fn refineTokens(tokens: []Token, source: []const u8, language: Language) void {
    switch (language) {
        .python => refinePython(tokens, source),
        .rust => refineRust(tokens, source),
        .javascript, .typescript => refineJavaScript(tokens, source),
        else => {},
    }
}

fn refinePython(tokens: []Token, source: []const u8) void {
    for (tokens) |*token| {
        if (token.start >= source.len or token.end > source.len) continue;
        const text = source[token.start..token.end];

        // Detect f-strings: string tokens starting with f" or f'
        if (token.token_type == .string and text.len >= 2) {
            const first = text[0];
            if ((first == 'f' or first == 'F') and
                (text[1] == '"' or text[1] == '\''))
            {
                token.token_type = .string_interpolation;
            }
        }
    }
}

fn refineRust(tokens: []Token, source: []const u8) void {
    for (tokens) |*token| {
        if (token.start >= source.len or token.end > source.len) continue;
        const text = source[token.start..token.end];

        // Detect attributes: decorator tokens starting with #
        if (token.token_type == .decorator and text.len >= 1 and text[0] == '#') {
            token.token_type = .attribute;
        }

        // Detect type primitives as type_name
        if (token.token_type == .identifier or token.token_type == .variable) {
            if (isRustPrimitive(text)) {
                token.token_type = .type_name;
            }
        }
    }
}

fn isRustPrimitive(text: []const u8) bool {
    const primitives = std.StaticStringMap(void).initComptime(.{
        .{ "bool", {} },
        .{ "char", {} },
        .{ "str", {} },
        .{ "i8", {} },
        .{ "i16", {} },
        .{ "i32", {} },
        .{ "i64", {} },
        .{ "i128", {} },
        .{ "isize", {} },
        .{ "u8", {} },
        .{ "u16", {} },
        .{ "u32", {} },
        .{ "u64", {} },
        .{ "u128", {} },
        .{ "usize", {} },
        .{ "f32", {} },
        .{ "f64", {} },
    });
    return primitives.has(text);
}

fn refineJavaScript(tokens: []Token, source: []const u8) void {
    for (tokens) |*token| {
        if (token.start >= source.len or token.end > source.len) continue;
        const text = source[token.start..token.end];

        // Detect template literals: string tokens starting with backtick
        if (token.token_type == .string and text.len >= 1 and text[0] == '`') {
            token.token_type = .string_interpolation;
        }

        // Detect regex literals: string tokens starting with /
        if (token.token_type == .string and text.len >= 2 and text[0] == '/') {
            token.token_type = .regex;
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

test "refinePython detects f-strings" {
    var tokens = [_]Token{
        .{ .start = 0, .end = 11, .token_type = .string },
    };
    const source = "f\"hello {x}\"";
    refinePython(&tokens, source);
    try std.testing.expectEqual(TokenType.string_interpolation, tokens[0].token_type);
}

test "refinePython leaves regular strings unchanged" {
    var tokens = [_]Token{
        .{ .start = 0, .end = 7, .token_type = .string },
    };
    const source = "\"hello\"";
    refinePython(&tokens, source);
    try std.testing.expectEqual(TokenType.string, tokens[0].token_type);
}

test "refineRust detects type primitives" {
    var tokens = [_]Token{
        .{ .start = 0, .end = 3, .token_type = .identifier },
    };
    const source = "i32";
    refineRust(&tokens, source);
    try std.testing.expectEqual(TokenType.type_name, tokens[0].token_type);
}

test "refineJavaScript detects template literals" {
    var tokens = [_]Token{
        .{ .start = 0, .end = 12, .token_type = .string },
    };
    const source = "`hello ${x}`";
    refineJavaScript(&tokens, source);
    try std.testing.expectEqual(TokenType.string_interpolation, tokens[0].token_type);
}

test "refineTokens dispatches to correct language" {
    var tokens = [_]Token{
        .{ .start = 0, .end = 11, .token_type = .string },
    };
    const source = "f\"hello {x}\"";
    refineTokens(&tokens, source, .python);
    try std.testing.expectEqual(TokenType.string_interpolation, tokens[0].token_type);
}

test "refineTokens is no-op for unsupported languages" {
    var tokens = [_]Token{
        .{ .start = 0, .end = 3, .token_type = .string },
    };
    const source = "abc";
    refineTokens(&tokens, source, .json);
    try std.testing.expectEqual(TokenType.string, tokens[0].token_type);
}
