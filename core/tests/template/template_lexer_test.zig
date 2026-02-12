//! Integration tests for TemplateLexer
//!
//! TemplateLexer tokenizes a Jinja2 template string into tokens.
//! Handles {{ }}, {% %}, {# #} delimiters and whitespace control.

const std = @import("std");
const main = @import("main");
const TemplateLexer = main.template.TemplateLexer;
const TokenType = main.template.TokenType;

// =============================================================================
// Basic Tokenization Tests
// =============================================================================

test "Lexer tokenizes plain text" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "Hello World");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(@as(usize, 2), tokens.len);
    try std.testing.expectEqual(TokenType.text, tokens[0].type);
    try std.testing.expectEqualStrings("Hello World", tokens[0].value);
    try std.testing.expectEqual(TokenType.eof, tokens[1].type);
}

test "Lexer tokenizes print statement" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{{ name }}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.print_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.name, tokens[1].type);
    try std.testing.expectEqualStrings("name", tokens[1].value);
    try std.testing.expectEqual(TokenType.print_close, tokens[2].type);
}

test "Lexer tokenizes statement block" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{% if x %}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.stmt_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.kw_if, tokens[1].type);
    try std.testing.expectEqual(TokenType.name, tokens[2].type);
    try std.testing.expectEqual(TokenType.stmt_close, tokens[3].type);
}

test "Lexer tokenizes mixed content" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "Hello {{ name }}!");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.text, tokens[0].type);
    try std.testing.expectEqualStrings("Hello ", tokens[0].value);
    try std.testing.expectEqual(TokenType.print_open, tokens[1].type);
    try std.testing.expectEqual(TokenType.name, tokens[2].type);
    try std.testing.expectEqual(TokenType.print_close, tokens[3].type);
    try std.testing.expectEqual(TokenType.text, tokens[4].type);
    try std.testing.expectEqualStrings("!", tokens[4].value);
}

// =============================================================================
// Keyword Tests
// =============================================================================

test "Lexer recognizes keywords" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{% if elif else endif for endfor in set %}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.stmt_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.kw_if, tokens[1].type);
    try std.testing.expectEqual(TokenType.kw_elif, tokens[2].type);
    try std.testing.expectEqual(TokenType.kw_else, tokens[3].type);
    try std.testing.expectEqual(TokenType.kw_endif, tokens[4].type);
    try std.testing.expectEqual(TokenType.kw_for, tokens[5].type);
    try std.testing.expectEqual(TokenType.kw_endfor, tokens[6].type);
    try std.testing.expectEqual(TokenType.kw_in, tokens[7].type);
    try std.testing.expectEqual(TokenType.kw_set, tokens[8].type);
}

test "Lexer recognizes boolean keywords" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{{ true false none }}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.print_open, tokens[0].type);
    try std.testing.expectEqual(TokenType.kw_true, tokens[1].type);
    try std.testing.expectEqual(TokenType.kw_false, tokens[2].type);
    try std.testing.expectEqual(TokenType.kw_none, tokens[3].type);
}

test "Lexer recognizes logical keywords" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{{ a and b or not c }}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.name, tokens[1].type);
    try std.testing.expectEqual(TokenType.kw_and, tokens[2].type);
    try std.testing.expectEqual(TokenType.name, tokens[3].type);
    try std.testing.expectEqual(TokenType.kw_or, tokens[4].type);
    try std.testing.expectEqual(TokenType.kw_not, tokens[5].type);
}

// =============================================================================
// Operator Tests
// =============================================================================

test "Lexer tokenizes operators" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{{ a + b - c * d / e }}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.plus, tokens[2].type);
    try std.testing.expectEqual(TokenType.minus, tokens[4].type);
    try std.testing.expectEqual(TokenType.star, tokens[6].type);
    try std.testing.expectEqual(TokenType.slash, tokens[8].type);
}

test "Lexer tokenizes comparison operators" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{{ a == b != c < d > e <= f >= g }}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.eq, tokens[2].type);
    try std.testing.expectEqual(TokenType.ne, tokens[4].type);
    try std.testing.expectEqual(TokenType.lt, tokens[6].type);
    try std.testing.expectEqual(TokenType.gt, tokens[8].type);
    try std.testing.expectEqual(TokenType.le, tokens[10].type);
    try std.testing.expectEqual(TokenType.ge, tokens[12].type);
}

test "Lexer tokenizes brackets and parens" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{{ items[0] (a, b) }}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.lbracket, tokens[2].type);
    try std.testing.expectEqual(TokenType.rbracket, tokens[4].type);
    try std.testing.expectEqual(TokenType.lparen, tokens[5].type);
    try std.testing.expectEqual(TokenType.comma, tokens[7].type);
    try std.testing.expectEqual(TokenType.rparen, tokens[9].type);
}

test "Lexer tokenizes pipe and dot" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{{ user.name | upper }}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.name, tokens[1].type);
    try std.testing.expectEqual(TokenType.dot, tokens[2].type);
    try std.testing.expectEqual(TokenType.name, tokens[3].type);
    try std.testing.expectEqual(TokenType.pipe, tokens[4].type);
    try std.testing.expectEqual(TokenType.name, tokens[5].type);
}

// =============================================================================
// Literal Tests
// =============================================================================

test "Lexer tokenizes integer literals" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{{ 42 }}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.integer, tokens[1].type);
    try std.testing.expectEqualStrings("42", tokens[1].value);
}

test "Lexer tokenizes float literals" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{{ 3.14 }}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.float, tokens[1].type);
    try std.testing.expectEqualStrings("3.14", tokens[1].value);
}

test "Lexer tokenizes single-quoted strings" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{{ 'hello' }}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.string, tokens[1].type);
    try std.testing.expectEqualStrings("hello", tokens[1].value);
}

test "Lexer tokenizes double-quoted strings" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{{ \"hello\" }}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.string, tokens[1].type);
    try std.testing.expectEqualStrings("hello", tokens[1].value);
}

// =============================================================================
// Whitespace Control Tests
// =============================================================================

test "Lexer handles left trim" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{{- name }}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.print_open, tokens[0].type);
    try std.testing.expect(tokens[0].trim_left);
}

test "Lexer handles right trim" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{{ name -}}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    const close_idx = for (tokens, 0..) |t, i| {
        if (t.type == .print_close) break i;
    } else unreachable;

    try std.testing.expect(tokens[close_idx].trim_right);
}

test "Lexer handles both trims" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{{- name -}}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expect(tokens[0].trim_left);

    const close_idx = for (tokens, 0..) |t, i| {
        if (t.type == .print_close) break i;
    } else unreachable;

    try std.testing.expect(tokens[close_idx].trim_right);
}

// =============================================================================
// Complex Template Tests
// =============================================================================

test "Lexer tokenizes ChatML template" {
    const allocator = std.testing.allocator;

    const template =
        \\{%- if messages[0].role == 'system' -%}
        \\<|im_start|>system
        \\{{ messages[0].content }}<|im_end|>
        \\{%- endif -%}
    ;

    var lexer = TemplateLexer.init(allocator, template);
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    // Just verify it tokenizes without error and has reasonable structure
    try std.testing.expect(tokens.len > 10);

    // Find the 'if' keyword
    var found_if = false;
    for (tokens) |t| {
        if (t.type == .kw_if) {
            found_if = true;
            break;
        }
    }
    try std.testing.expect(found_if);
}

test "Lexer tokenizes escaped quotes in strings" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{{ \"hello \\\"world\\\"\" }}");
    defer lexer.deinit();

    const tokens = try lexer.tokenize();

    try std.testing.expectEqual(TokenType.string, tokens[1].type);
}
