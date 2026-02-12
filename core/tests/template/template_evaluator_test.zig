//! Integration tests for TemplateEvaluator
//!
//! TemplateEvaluator renders parsed AST nodes with a context.
//! It handles variable substitution, control flow, filters, and methods.

const std = @import("std");
const main = @import("main");
const TemplateEvaluator = main.template.TemplateEvaluator;
const TemplateParser = main.template.TemplateParser;
const TemplateParserInternal = main.template.TemplateParserInternal;
const TemplateLexer = main.template.TemplateLexer;
const TemplateInput = main.template.TemplateInput;

// =============================================================================
// Basic Evaluation Tests
// =============================================================================

test "Evaluator renders static text" {
    const allocator = std.testing.allocator;

    // Tokenize and parse
    var lexer = TemplateLexer.init(allocator, "Hello World");
    defer lexer.deinit();
    const tokens = try lexer.tokenize();

    var parser = TemplateParserInternal.init(allocator, tokens);
    defer parser.deinit();
    const nodes = try parser.parse();
    defer allocator.free(nodes);

    // Evaluate
    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    var eval = TemplateEvaluator.init(allocator, &ctx);
    defer eval.deinit();

    const result = try eval.render(nodes);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Hello World", result);
}

test "Evaluator substitutes variables" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "Hello {{ name }}!");
    defer lexer.deinit();
    const tokens = try lexer.tokenize();

    var parser = TemplateParserInternal.init(allocator, tokens);
    defer parser.deinit();
    const nodes = try parser.parse();
    defer allocator.free(nodes);

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();
    try ctx.set("name", .{ .string = "Alice" });

    var eval = TemplateEvaluator.init(allocator, &ctx);
    defer eval.deinit();

    const result = try eval.render(nodes);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Hello Alice!", result);
}

test "Evaluator handles integer values" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "Count: {{ count }}");
    defer lexer.deinit();
    const tokens = try lexer.tokenize();

    var parser = TemplateParserInternal.init(allocator, tokens);
    defer parser.deinit();
    const nodes = try parser.parse();
    defer allocator.free(nodes);

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();
    try ctx.set("count", .{ .integer = 42 });

    var eval = TemplateEvaluator.init(allocator, &ctx);
    defer eval.deinit();

    const result = try eval.render(nodes);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Count: 42", result);
}

// =============================================================================
// Control Flow Tests
// =============================================================================

test "Evaluator handles if/else true branch" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{% if show %}yes{% else %}no{% endif %}");
    defer lexer.deinit();
    const tokens = try lexer.tokenize();

    var parser = TemplateParserInternal.init(allocator, tokens);
    defer parser.deinit();
    const nodes = try parser.parse();
    defer allocator.free(nodes);

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();
    try ctx.set("show", .{ .boolean = true });

    var eval = TemplateEvaluator.init(allocator, &ctx);
    defer eval.deinit();

    const result = try eval.render(nodes);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("yes", result);
}

test "Evaluator handles if/else false branch" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{% if show %}yes{% else %}no{% endif %}");
    defer lexer.deinit();
    const tokens = try lexer.tokenize();

    var parser = TemplateParserInternal.init(allocator, tokens);
    defer parser.deinit();
    const nodes = try parser.parse();
    defer allocator.free(nodes);

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();
    try ctx.set("show", .{ .boolean = false });

    var eval = TemplateEvaluator.init(allocator, &ctx);
    defer eval.deinit();

    const result = try eval.render(nodes);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("no", result);
}

test "Evaluator handles for loop" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{% for x in items %}{{ x }}{% endfor %}");
    defer lexer.deinit();
    const tokens = try lexer.tokenize();

    var parser = TemplateParserInternal.init(allocator, tokens);
    defer parser.deinit();
    const nodes = try parser.parse();
    defer allocator.free(nodes);

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    const items = [_]TemplateInput{
        .{ .string = "a" },
        .{ .string = "b" },
        .{ .string = "c" },
    };
    try ctx.set("items", .{ .array = &items });

    var eval = TemplateEvaluator.init(allocator, &ctx);
    defer eval.deinit();

    const result = try eval.render(nodes);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("abc", result);
}

// =============================================================================
// Expression Tests
// =============================================================================

test "Evaluator handles arithmetic" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{{ a + b }}");
    defer lexer.deinit();
    const tokens = try lexer.tokenize();

    var parser = TemplateParserInternal.init(allocator, tokens);
    defer parser.deinit();
    const nodes = try parser.parse();
    defer allocator.free(nodes);

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();
    try ctx.set("a", .{ .integer = 10 });
    try ctx.set("b", .{ .integer = 5 });

    var eval = TemplateEvaluator.init(allocator, &ctx);
    defer eval.deinit();

    const result = try eval.render(nodes);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("15", result);
}

test "Evaluator handles comparison" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{% if x > 5 %}big{% else %}small{% endif %}");
    defer lexer.deinit();
    const tokens = try lexer.tokenize();

    var parser = TemplateParserInternal.init(allocator, tokens);
    defer parser.deinit();
    const nodes = try parser.parse();
    defer allocator.free(nodes);

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();
    try ctx.set("x", .{ .integer = 10 });

    var eval = TemplateEvaluator.init(allocator, &ctx);
    defer eval.deinit();

    const result = try eval.render(nodes);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("big", result);
}

// =============================================================================
// Filter Tests
// =============================================================================

test "Evaluator applies upper filter" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{{ name | upper }}");
    defer lexer.deinit();
    const tokens = try lexer.tokenize();

    var parser = TemplateParserInternal.init(allocator, tokens);
    defer parser.deinit();
    const nodes = try parser.parse();
    defer allocator.free(nodes);

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();
    try ctx.set("name", .{ .string = "hello" });

    var eval = TemplateEvaluator.init(allocator, &ctx);
    defer eval.deinit();

    const result = try eval.render(nodes);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("HELLO", result);
}

test "Evaluator applies length filter" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{{ items | length }}");
    defer lexer.deinit();
    const tokens = try lexer.tokenize();

    var parser = TemplateParserInternal.init(allocator, tokens);
    defer parser.deinit();
    const nodes = try parser.parse();
    defer allocator.free(nodes);

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    const items = [_]TemplateInput{
        .{ .integer = 1 },
        .{ .integer = 2 },
        .{ .integer = 3 },
    };
    try ctx.set("items", .{ .array = &items });

    var eval = TemplateEvaluator.init(allocator, &ctx);
    defer eval.deinit();

    const result = try eval.render(nodes);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("3", result);
}

// =============================================================================
// Map Access Tests
// =============================================================================

test "Evaluator handles nested map access" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "{{ user.name }}");
    defer lexer.deinit();
    const tokens = try lexer.tokenize();

    var parser = TemplateParserInternal.init(allocator, tokens);
    defer parser.deinit();
    const nodes = try parser.parse();
    defer allocator.free(nodes);

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    var user = std.StringHashMapUnmanaged(TemplateInput){};
    try user.put(allocator, "name", .{ .string = "Bob" });
    defer user.deinit(allocator);
    try ctx.set("user", .{ .map = user });

    var eval = TemplateEvaluator.init(allocator, &ctx);
    defer eval.deinit();

    const result = try eval.render(nodes);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Bob", result);
}

// =============================================================================
// Debug Mode Tests
// =============================================================================

test "Evaluator debug mode tracks spans" {
    const allocator = std.testing.allocator;

    var lexer = TemplateLexer.init(allocator, "Hello {{ name }}!");
    defer lexer.deinit();
    const tokens = try lexer.tokenize();

    var parser = TemplateParserInternal.init(allocator, tokens);
    defer parser.deinit();
    const nodes = try parser.parse();
    defer allocator.free(nodes);

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();
    try ctx.set("name", .{ .string = "World" });

    var eval = TemplateEvaluator.initDebug(allocator, &ctx);
    defer eval.deinit();

    const result = try eval.renderWithSpans(nodes);
    defer allocator.free(result.output);
    defer allocator.free(result.spans);

    try std.testing.expectEqualStrings("Hello World!", result.output);
    try std.testing.expectEqual(@as(usize, 3), result.spans.len);
}
