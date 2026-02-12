//! Integration tests for text.template_engine.RenderResult
//!
//! RenderResult is returned by renderFromJson and contains either the rendered
//! output or detailed error information for debugging template issues.

const std = @import("std");
const main = @import("main");

const RenderResult = main.template.RenderResult;
const renderFromJson = main.template.renderFromJson;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "RenderResult type is accessible" {
    const T = RenderResult;
    _ = T;
}

test "RenderResult is a struct" {
    const info = @typeInfo(RenderResult);
    try std.testing.expect(info == .@"struct");
}

test "RenderResult has expected fields" {
    const info = @typeInfo(RenderResult);
    const fields = info.@"struct".fields;

    var has_output = false;
    var has_err = false;
    var has_error_message = false;
    var has_undefined_path = false;
    var has_raise_message = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "output")) has_output = true;
        if (comptime std.mem.eql(u8, field.name, "err")) has_err = true;
        if (comptime std.mem.eql(u8, field.name, "error_message")) has_error_message = true;
        if (comptime std.mem.eql(u8, field.name, "undefined_path")) has_undefined_path = true;
        if (comptime std.mem.eql(u8, field.name, "raise_message")) has_raise_message = true;
    }

    try std.testing.expect(has_output);
    try std.testing.expect(has_err);
    try std.testing.expect(has_error_message);
    try std.testing.expect(has_undefined_path);
    try std.testing.expect(has_raise_message);
}

// =============================================================================
// Successful Render Tests
// =============================================================================

test "RenderResult: success returns rendered output" {
    const allocator = std.testing.allocator;
    var result = renderFromJson(allocator, "Hello {{ name }}!", "{\"name\": \"World\"}", false);
    defer result.deinit(allocator);

    try std.testing.expect(result.success());
    try std.testing.expect(result.output != null);
    try std.testing.expectEqualStrings("Hello World!", result.output.?);
    try std.testing.expect(result.err == null);
}

test "RenderResult: success with multiple variables" {
    const allocator = std.testing.allocator;
    var result = renderFromJson(allocator, "{{ a }} + {{ b }} = {{ c }}", "{\"a\": 1, \"b\": 2, \"c\": 3}", false);
    defer result.deinit(allocator);

    try std.testing.expect(result.success());
    try std.testing.expectEqualStrings("1 + 2 = 3", result.output.?);
}

test "RenderResult: success with empty template" {
    const allocator = std.testing.allocator;
    var result = renderFromJson(allocator, "", "{}", false);
    defer result.deinit(allocator);

    try std.testing.expect(result.success());
    try std.testing.expectEqualStrings("", result.output.?);
}

test "RenderResult: success with static text only" {
    const allocator = std.testing.allocator;
    var result = renderFromJson(allocator, "Hello World!", "{}", false);
    defer result.deinit(allocator);

    try std.testing.expect(result.success());
    try std.testing.expectEqualStrings("Hello World!", result.output.?);
}

// =============================================================================
// JSON Parse Error Tests
// =============================================================================

test "RenderResult: invalid JSON sets error_message" {
    const allocator = std.testing.allocator;
    var result = renderFromJson(allocator, "{{ x }}", "not valid json", false);
    defer result.deinit(allocator);

    try std.testing.expect(result.error_message != null);
    try std.testing.expect(result.output == null);
}

test "RenderResult: empty JSON string sets error_message" {
    const allocator = std.testing.allocator;
    var result = renderFromJson(allocator, "{{ x }}", "", false);
    defer result.deinit(allocator);

    try std.testing.expect(result.error_message != null);
}

// =============================================================================
// Template Error Tests
// =============================================================================

test "RenderResult: syntax error sets err" {
    const allocator = std.testing.allocator;
    var result = renderFromJson(allocator, "{% if x %}no endif", "{\"x\": true}", false);
    defer result.deinit(allocator);

    try std.testing.expect(!result.success());
    try std.testing.expect(result.err != null);
    try std.testing.expect(result.output == null);
}

test "RenderResult: raise_exception captures message" {
    const allocator = std.testing.allocator;
    var result = renderFromJson(allocator, "{{ raise_exception('custom error') }}", "{}", false);
    defer result.deinit(allocator);

    try std.testing.expect(!result.success());
    try std.testing.expect(result.raise_message != null);
    try std.testing.expectEqualStrings("custom error", result.raise_message.?);
}

// =============================================================================
// Strict Mode Tests
// =============================================================================

test "RenderResult: undefined variable in strict mode sets undefined_path" {
    const allocator = std.testing.allocator;
    var result = renderFromJson(allocator, "{{ missing_var }}", "{}", true);
    defer result.deinit(allocator);

    try std.testing.expect(!result.success());
    try std.testing.expect(result.undefined_path != null);
    try std.testing.expectEqualStrings("missing_var", result.undefined_path.?);
}

test "RenderResult: undefined variable in non-strict mode succeeds" {
    const allocator = std.testing.allocator;
    var result = renderFromJson(allocator, "{{ missing_var }}", "{}", false);
    defer result.deinit(allocator);

    try std.testing.expect(result.success());
    try std.testing.expectEqualStrings("", result.output.?);
}

test "RenderResult: nested undefined path in strict mode" {
    const allocator = std.testing.allocator;
    var result = renderFromJson(allocator, "{{ user.name }}", "{\"user\": {}}", true);
    defer result.deinit(allocator);

    try std.testing.expect(!result.success());
    try std.testing.expect(result.undefined_path != null);
}

// =============================================================================
// deinit Tests
// =============================================================================

test "RenderResult.deinit frees all allocated memory" {
    const allocator = std.testing.allocator;

    // Success case
    var result1 = renderFromJson(allocator, "Hello {{ name }}!", "{\"name\": \"World\"}", false);
    result1.deinit(allocator);

    // Error case with undefined_path
    var result2 = renderFromJson(allocator, "{{ x }}", "{}", true);
    result2.deinit(allocator);

    // Error case with raise_message
    var result3 = renderFromJson(allocator, "{{ raise_exception('test') }}", "{}", false);
    result3.deinit(allocator);

    // No leak = success
}

test "RenderResult.deinit is safe to call multiple times on default" {
    const allocator = std.testing.allocator;
    var result = RenderResult{};
    result.deinit(allocator);
    result.deinit(allocator); // Should not crash
}

// =============================================================================
// success() Method Tests
// =============================================================================

test "RenderResult.success returns true when err is null" {
    var result = RenderResult{ .output = "test" };
    try std.testing.expect(result.success());
}

test "RenderResult.success returns false when err is set" {
    var result = RenderResult{ .err = error.ParseError };
    try std.testing.expect(!result.success());
}

// =============================================================================
// Complex Template Tests
// =============================================================================

test "RenderResult: for loop renders correctly" {
    const allocator = std.testing.allocator;
    var result = renderFromJson(allocator, "{% for i in items %}{{ i }}{% endfor %}", "{\"items\": [1, 2, 3]}", false);
    defer result.deinit(allocator);

    try std.testing.expect(result.success());
    try std.testing.expectEqualStrings("123", result.output.?);
}

test "RenderResult: if statement renders correctly" {
    const allocator = std.testing.allocator;
    var result = renderFromJson(allocator, "{% if show %}visible{% endif %}", "{\"show\": true}", false);
    defer result.deinit(allocator);

    try std.testing.expect(result.success());
    try std.testing.expectEqualStrings("visible", result.output.?);
}

test "RenderResult: filter chain works" {
    const allocator = std.testing.allocator;
    var result = renderFromJson(allocator, "{{ name | upper | trim }}", "{\"name\": \"  hello  \"}", false);
    defer result.deinit(allocator);

    try std.testing.expect(result.success());
    try std.testing.expectEqualStrings("HELLO", result.output.?);
}
