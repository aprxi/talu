//! Integration tests for text.template.TemplateParser
//!
//! TemplateParser is the template execution environment that holds variables
//! for Jinja2 template rendering. Tests verify the full lifecycle:
//! init → set/get variables → deinit

const std = @import("std");
const main = @import("main");

const TemplateParser = main.template.TemplateParser;
const TemplateInput = main.template.TemplateInput;

// =============================================================================
// Lifecycle Tests
// =============================================================================

test "TemplateParser init and deinit" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    // Fresh context should have no variables
    try std.testing.expect(ctx.get("anything") == null);
}

test "TemplateParser survives multiple init/deinit cycles" {
    const allocator = std.testing.allocator;

    for (0..3) |_| {
        var ctx = TemplateParser.init(allocator);
        try ctx.set("x", .{ .integer = 42 });
        ctx.deinit();
    }
}

// =============================================================================
// Variable Storage Tests
// =============================================================================

test "TemplateParser set and get string variable" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    try ctx.set("name", .{ .string = "Alice" });

    const result = ctx.get("name");
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("Alice", result.?.string);
}

test "TemplateParser set and get integer variable" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    try ctx.set("count", .{ .integer = 42 });

    const result = ctx.get("count");
    try std.testing.expect(result != null);
    try std.testing.expectEqual(@as(i64, 42), result.?.integer);
}

test "TemplateParser set and get boolean variable" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    try ctx.set("enabled", .{ .boolean = true });
    try ctx.set("disabled", .{ .boolean = false });

    try std.testing.expectEqual(true, ctx.get("enabled").?.boolean);
    try std.testing.expectEqual(false, ctx.get("disabled").?.boolean);
}

test "TemplateParser set and get float variable" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    try ctx.set("pi", .{ .float = 3.14159 });

    const result = ctx.get("pi");
    try std.testing.expect(result != null);
    try std.testing.expectApproxEqAbs(@as(f64, 3.14159), result.?.float, 0.00001);
}

test "TemplateParser set and get array variable" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    const items = [_]TemplateInput{
        .{ .string = "a" },
        .{ .string = "b" },
        .{ .string = "c" },
    };
    try ctx.set("items", .{ .array = &items });

    const result = ctx.get("items");
    try std.testing.expect(result != null);
    try std.testing.expectEqual(@as(usize, 3), result.?.array.len);
    try std.testing.expectEqualStrings("a", result.?.array[0].string);
    try std.testing.expectEqualStrings("b", result.?.array[1].string);
    try std.testing.expectEqualStrings("c", result.?.array[2].string);
}

test "TemplateParser set and get map variable" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    var map = std.StringHashMapUnmanaged(TemplateInput){};
    try map.put(allocator, "key", .{ .string = "value" });
    defer map.deinit(allocator);

    try ctx.set("obj", .{ .map = map });

    const result = ctx.get("obj");
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("value", result.?.map.get("key").?.string);
}

test "TemplateParser set and get none variable" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    try ctx.set("empty", .none);

    const result = ctx.get("empty");
    try std.testing.expect(result != null);
    try std.testing.expect(result.? == .none);
}

test "TemplateParser get returns null for undefined variable" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    try std.testing.expect(ctx.get("undefined") == null);
}

test "TemplateParser overwrites existing variable" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    try ctx.set("x", .{ .integer = 1 });
    try std.testing.expectEqual(@as(i64, 1), ctx.get("x").?.integer);

    try ctx.set("x", .{ .integer = 2 });
    try std.testing.expectEqual(@as(i64, 2), ctx.get("x").?.integer);
}

test "TemplateParser stores multiple variables" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    try ctx.set("a", .{ .integer = 1 });
    try ctx.set("b", .{ .integer = 2 });
    try ctx.set("c", .{ .integer = 3 });

    try std.testing.expectEqual(@as(i64, 1), ctx.get("a").?.integer);
    try std.testing.expectEqual(@as(i64, 2), ctx.get("b").?.integer);
    try std.testing.expectEqual(@as(i64, 3), ctx.get("c").?.integer);
}

// =============================================================================
// TemplateInput Type Tests
// =============================================================================

test "TemplateInput isTruthy for different types" {
    // Truthy values
    try std.testing.expect(TemplateInput.isTruthy(.{ .string = "hello" }));
    try std.testing.expect(TemplateInput.isTruthy(.{ .integer = 1 }));
    try std.testing.expect(TemplateInput.isTruthy(.{ .integer = -1 }));
    try std.testing.expect(TemplateInput.isTruthy(.{ .float = 0.1 }));
    try std.testing.expect(TemplateInput.isTruthy(.{ .boolean = true }));

    // Falsy values
    try std.testing.expect(!TemplateInput.isTruthy(.{ .string = "" }));
    try std.testing.expect(!TemplateInput.isTruthy(.{ .integer = 0 }));
    try std.testing.expect(!TemplateInput.isTruthy(.{ .float = 0.0 }));
    try std.testing.expect(!TemplateInput.isTruthy(.{ .boolean = false }));
    try std.testing.expect(!TemplateInput.isTruthy(.none));
}

test "TemplateInput eql compares same types" {
    try std.testing.expect(TemplateInput.eql(.{ .string = "hello" }, .{ .string = "hello" }));
    try std.testing.expect(!TemplateInput.eql(.{ .string = "hello" }, .{ .string = "world" }));

    try std.testing.expect(TemplateInput.eql(.{ .integer = 42 }, .{ .integer = 42 }));
    try std.testing.expect(!TemplateInput.eql(.{ .integer = 42 }, .{ .integer = 43 }));

    try std.testing.expect(TemplateInput.eql(.{ .boolean = true }, .{ .boolean = true }));
    try std.testing.expect(!TemplateInput.eql(.{ .boolean = true }, .{ .boolean = false }));

    try std.testing.expect(TemplateInput.eql(.none, .none));
}

test "TemplateInput eql with numeric type coercion" {
    // Integer and float comparison
    try std.testing.expect(TemplateInput.eql(.{ .integer = 42 }, .{ .float = 42.0 }));
    try std.testing.expect(TemplateInput.eql(.{ .float = 3.0 }, .{ .integer = 3 }));
    try std.testing.expect(!TemplateInput.eql(.{ .integer = 42 }, .{ .float = 42.5 }));
}

test "TemplateInput asNumber extracts numeric value" {
    try std.testing.expectEqual(@as(?f64, 42.0), TemplateInput.asNumber(.{ .integer = 42 }));
    try std.testing.expectEqual(@as(?f64, 3.14), TemplateInput.asNumber(.{ .float = 3.14 }));
    try std.testing.expect(TemplateInput.asNumber(.{ .string = "not a number" }) == null);
    try std.testing.expect(TemplateInput.asNumber(.none) == null);
}

test "TemplateInput asString converts types" {
    const allocator = std.testing.allocator;

    // String stays as-is
    {
        const result = try TemplateInput.asString(.{ .string = "hello" }, allocator);
        try std.testing.expectEqualStrings("hello", result);
    }

    // Integer converts to string
    {
        const result = try TemplateInput.asString(.{ .integer = 42 }, allocator);
        defer allocator.free(result);
        try std.testing.expectEqualStrings("42", result);
    }

    // Boolean converts to Python-style True/False
    {
        const true_result = try TemplateInput.asString(.{ .boolean = true }, allocator);
        try std.testing.expectEqualStrings("True", true_result);

        const false_result = try TemplateInput.asString(.{ .boolean = false }, allocator);
        try std.testing.expectEqualStrings("False", false_result);
    }

    // None converts to empty string
    {
        const result = try TemplateInput.asString(.none, allocator);
        try std.testing.expectEqualStrings("", result);
    }
}

test "TemplateInput toJson serializes to JSON" {
    const allocator = std.testing.allocator;

    // String with quotes
    {
        const result = try TemplateInput.toJson(.{ .string = "hello" }, allocator);
        defer allocator.free(result);
        try std.testing.expectEqualStrings("\"hello\"", result);
    }

    // Integer
    {
        const result = try TemplateInput.toJson(.{ .integer = 42 }, allocator);
        defer allocator.free(result);
        try std.testing.expectEqualStrings("42", result);
    }

    // Boolean (lowercase in JSON)
    {
        const result = try TemplateInput.toJson(.{ .boolean = true }, allocator);
        defer allocator.free(result);
        try std.testing.expectEqualStrings("true", result);
    }

    // None becomes null
    {
        const result = try TemplateInput.toJson(.none, allocator);
        defer allocator.free(result);
        try std.testing.expectEqualStrings("null", result);
    }

    // Array
    {
        const items = [_]TemplateInput{ .{ .integer = 1 }, .{ .integer = 2 } };
        const result = try TemplateInput.toJson(.{ .array = &items }, allocator);
        defer allocator.free(result);
        try std.testing.expectEqualStrings("[1, 2]", result);
    }
}

// =============================================================================
// Integration with Template Rendering
// =============================================================================

test "TemplateParser works with template render" {
    const allocator = std.testing.allocator;
    const render = main.template.render;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    try ctx.set("name", .{ .string = "World" });
    try ctx.set("count", .{ .integer = 3 });

    const result = try render(allocator, "Hello {{ name }}, count={{ count }}", &ctx);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Hello World, count=3", result);
}

test "TemplateParser with complex nested data for chat template" {
    const allocator = std.testing.allocator;
    const render = main.template.render;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    // Build a chat message (typical LLM usage)
    var msg1 = std.StringHashMapUnmanaged(TemplateInput){};
    try msg1.put(allocator, "role", .{ .string = "user" });
    try msg1.put(allocator, "content", .{ .string = "Hello!" });
    defer msg1.deinit(allocator);

    const messages = [_]TemplateInput{.{ .map = msg1 }};
    try ctx.set("messages", .{ .array = &messages });

    const template =
        \\{% for msg in messages %}<{{ msg.role }}>{{ msg.content }}</{{ msg.role }}>{% endfor %}
    ;

    const result = try render(allocator, template, &ctx);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("<user>Hello!</user>", result);
}

// =============================================================================
// Strict Mode Tests
// =============================================================================

test "TemplateParser initStrict creates strict parser" {
    const allocator = std.testing.allocator;

    var ctx = TemplateParser.initStrict(allocator);
    defer ctx.deinit();

    try std.testing.expect(ctx.strict);
}

test "TemplateParser strict=false allows undefined variables" {
    const allocator = std.testing.allocator;
    const render = main.template.render;

    var ctx = TemplateParser.init(allocator);
    defer ctx.deinit();

    // Non-strict mode: undefined variables return empty
    const result = try render(allocator, "Hello {{ name }}!", &ctx);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Hello !", result);
}

test "TemplateParser strict=true errors on undefined variable" {
    const allocator = std.testing.allocator;
    const render = main.template.render;

    var ctx = TemplateParser.initStrict(allocator);
    defer ctx.deinit();

    // Strict mode: undefined variables raise error
    const result = render(allocator, "Hello {{ name }}!", &ctx);
    try std.testing.expectError(error.UndefinedVariable, result);
}

test "TemplateParser strict=true works with defined variables" {
    const allocator = std.testing.allocator;
    const render = main.template.render;

    var ctx = TemplateParser.initStrict(allocator);
    defer ctx.deinit();
    try ctx.set("name", .{ .string = "Alice" });

    // Strict mode works fine when variables are defined
    const result = try render(allocator, "Hello {{ name }}!", &ctx);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Hello Alice!", result);
}

test "TemplateParser strict=true errors on undefined attribute" {
    const allocator = std.testing.allocator;
    const render = main.template.render;

    var ctx = TemplateParser.initStrict(allocator);
    defer ctx.deinit();

    var user = std.StringHashMapUnmanaged(TemplateInput){};
    try user.put(allocator, "name", .{ .string = "Alice" });
    defer user.deinit(allocator);
    try ctx.set("user", .{ .map = user });

    // Accessing existing attribute works
    const good_result = try render(allocator, "{{ user.name }}", &ctx);
    defer allocator.free(good_result);
    try std.testing.expectEqualStrings("Alice", good_result);

    // Accessing missing attribute errors
    const bad_result = render(allocator, "{{ user.missing }}", &ctx);
    try std.testing.expectError(error.UndefinedVariable, bad_result);
}

test "TemplateParser strict=true errors on 'in' with undefined" {
    const allocator = std.testing.allocator;
    const render = main.template.render;

    var ctx = TemplateParser.initStrict(allocator);
    defer ctx.deinit();

    // 'x in undefined_var' should error in strict mode
    const result = render(allocator, "{% if 'x' in items %}yes{% endif %}", &ctx);
    try std.testing.expectError(error.UndefinedVariable, result);
}

test "TemplateParser strict=true allows 'is defined' test" {
    const allocator = std.testing.allocator;
    const render = main.template.render;

    var ctx = TemplateParser.initStrict(allocator);
    defer ctx.deinit();

    // Even in strict mode, 'is defined' should work
    const result = try render(allocator, "{% if name is defined %}yes{% else %}no{% endif %}", &ctx);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("no", result);
}

test "TemplateParser strict=true with default filter works" {
    const allocator = std.testing.allocator;
    const render = main.template.render;

    var ctx = TemplateParser.initStrict(allocator);
    defer ctx.deinit();

    // default filter should catch undefined before strict error
    const result = try render(allocator, "{{ name | default('Guest') }}", &ctx);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Guest", result);
}
