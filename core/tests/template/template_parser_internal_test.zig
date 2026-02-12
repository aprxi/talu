//! Integration tests for TemplateParserInternal
//!
//! TemplateParserInternal converts a token stream into an AST.
//! Uses a Pratt parser for expression precedence.

const std = @import("std");
const main = @import("main");
const TemplateParserInternal = main.template.TemplateParserInternal;
const TemplateLexer = main.template.TemplateLexer;
const ParseError = main.template.ParseError;

// =============================================================================
// Basic Parsing Tests
// =============================================================================

test "Parser parses static text" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "Hello World");
    defer lexer.deinit();
    const tokens = try lexer.tokenize();

    var parser = TemplateParserInternal.init(allocator, tokens);
    defer parser.deinit();

    const nodes = try parser.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
}

test "Parser parses print statement" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{{ name }}");
    defer lexer.deinit();
    const tokens = try lexer.tokenize();

    var parser = TemplateParserInternal.init(allocator, tokens);
    defer parser.deinit();

    const nodes = try parser.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
}

test "Parser parses multiple nodes" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "Hello {{ name }}!");
    defer lexer.deinit();
    const tokens = try lexer.tokenize();

    var parser = TemplateParserInternal.init(allocator, tokens);
    defer parser.deinit();

    const nodes = try parser.parse();
    defer allocator.free(nodes);

    // "Hello ", {{ name }}, "!"
    try std.testing.expectEqual(@as(usize, 3), nodes.len);
}

// =============================================================================
// Control Flow Parsing Tests
// =============================================================================

test "Parser parses if statement" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{% if x %}yes{% endif %}");
    defer lexer.deinit();
    const tokens = try lexer.tokenize();

    var parser = TemplateParserInternal.init(allocator, tokens);
    defer parser.deinit();

    const nodes = try parser.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
}

test "Parser parses if/else statement" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{% if x %}yes{% else %}no{% endif %}");
    defer lexer.deinit();
    const tokens = try lexer.tokenize();

    var parser = TemplateParserInternal.init(allocator, tokens);
    defer parser.deinit();

    const nodes = try parser.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
}

test "Parser parses if/elif/else statement" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{% if a %}A{% elif b %}B{% else %}C{% endif %}");
    defer lexer.deinit();
    const tokens = try lexer.tokenize();

    var parser = TemplateParserInternal.init(allocator, tokens);
    defer parser.deinit();

    const nodes = try parser.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
}

test "Parser parses for loop" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{% for x in items %}{{ x }}{% endfor %}");
    defer lexer.deinit();
    const tokens = try lexer.tokenize();

    var parser = TemplateParserInternal.init(allocator, tokens);
    defer parser.deinit();

    const nodes = try parser.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
}

test "Parser parses set statement" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{% set x = 5 %}{{ x }}");
    defer lexer.deinit();
    const tokens = try lexer.tokenize();

    var parser = TemplateParserInternal.init(allocator, tokens);
    defer parser.deinit();

    const nodes = try parser.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 2), nodes.len);
}

// =============================================================================
// Expression Parsing Tests
// =============================================================================

test "Parser parses arithmetic expressions" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{{ a + b * c }}");
    defer lexer.deinit();
    const tokens = try lexer.tokenize();

    var parser = TemplateParserInternal.init(allocator, tokens);
    defer parser.deinit();

    const nodes = try parser.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
}

test "Parser parses comparison expressions" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{{ a > b }}");
    defer lexer.deinit();
    const tokens = try lexer.tokenize();

    var parser = TemplateParserInternal.init(allocator, tokens);
    defer parser.deinit();

    const nodes = try parser.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
}

test "Parser parses logical expressions" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{{ a and b or c }}");
    defer lexer.deinit();
    const tokens = try lexer.tokenize();

    var parser = TemplateParserInternal.init(allocator, tokens);
    defer parser.deinit();

    const nodes = try parser.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
}

test "Parser parses filter expressions" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{{ name | upper }}");
    defer lexer.deinit();
    const tokens = try lexer.tokenize();

    var parser = TemplateParserInternal.init(allocator, tokens);
    defer parser.deinit();

    const nodes = try parser.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
}

test "Parser parses subscript expressions" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{{ items[0] }}");
    defer lexer.deinit();
    const tokens = try lexer.tokenize();

    var parser = TemplateParserInternal.init(allocator, tokens);
    defer parser.deinit();

    const nodes = try parser.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
}

test "Parser parses attribute access" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{{ user.name }}");
    defer lexer.deinit();
    const tokens = try lexer.tokenize();

    var parser = TemplateParserInternal.init(allocator, tokens);
    defer parser.deinit();

    const nodes = try parser.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
}

test "Parser parses method calls" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{{ text.upper() }}");
    defer lexer.deinit();
    const tokens = try lexer.tokenize();

    var parser = TemplateParserInternal.init(allocator, tokens);
    defer parser.deinit();

    const nodes = try parser.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
}

// =============================================================================
// Error Handling Tests
// =============================================================================

test "Parser returns error for unclosed block" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{% if x %}yes");
    defer lexer.deinit();
    const tokens = try lexer.tokenize();

    var parser = TemplateParserInternal.init(allocator, tokens);
    defer parser.deinit();

    const result = parser.parse();
    try std.testing.expectError(ParseError.UnclosedBlock, result);
}

test "Parser returns error for unexpected token" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{{ + }}");
    defer lexer.deinit();
    const tokens = try lexer.tokenize();

    var parser = TemplateParserInternal.init(allocator, tokens);
    defer parser.deinit();

    const result = parser.parse();
    try std.testing.expectError(ParseError.UnexpectedToken, result);
}

// =============================================================================
// Complex Template Tests
// =============================================================================

test "Parser parses nested control flow" {
    const allocator = std.testing.allocator;

    const template =
        \\{% for x in items %}
        \\{% if x > 0 %}{{ x }}{% endif %}
        \\{% endfor %}
    ;

    var lexer = TemplateLexer.init(allocator, template);
    defer lexer.deinit();
    const tokens = try lexer.tokenize();

    var parser = TemplateParserInternal.init(allocator, tokens);
    defer parser.deinit();

    const nodes = try parser.parse();
    defer allocator.free(nodes);

    // Just verify it parses without error
    try std.testing.expect(nodes.len > 0);
}

test "Parser parses slice expressions" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{{ items[1:3] }}");
    defer lexer.deinit();
    const tokens = try lexer.tokenize();

    var parser = TemplateParserInternal.init(allocator, tokens);
    defer parser.deinit();

    const nodes = try parser.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
}

test "Parser parses reverse slice" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{{ items[::-1] }}");
    defer lexer.deinit();
    const tokens = try lexer.tokenize();

    var parser = TemplateParserInternal.init(allocator, tokens);
    defer parser.deinit();

    const nodes = try parser.parse();
    defer allocator.free(nodes);

    try std.testing.expectEqual(@as(usize, 1), nodes.len);
}
