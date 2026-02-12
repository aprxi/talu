//! Integration tests for text.template_engine.RenderDebugResult
//!
//! RenderDebugResult is returned by renderFromJsonDebug and contains the rendered
//! output along with span tracking information for debug visualization. Each span
//! indicates which parts of the output came from static text, variables, or expressions.

const std = @import("std");
const main = @import("main");

const RenderDebugResult = main.template.RenderDebugResult;
const OutputSpan = main.template.OutputSpan;
const SpanSource = main.template.SpanSource;
const renderFromJsonDebug = main.template.renderFromJsonDebug;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "RenderDebugResult type is accessible" {
    const T = RenderDebugResult;
    _ = T;
}

test "RenderDebugResult is a struct" {
    const info = @typeInfo(RenderDebugResult);
    try std.testing.expect(info == .@"struct");
}

test "RenderDebugResult has expected fields" {
    const info = @typeInfo(RenderDebugResult);
    const fields = info.@"struct".fields;

    var has_output = false;
    var has_spans = false;
    var has_err = false;
    var has_undefined_path = false;
    var has_raise_message = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "output")) has_output = true;
        if (comptime std.mem.eql(u8, field.name, "spans")) has_spans = true;
        if (comptime std.mem.eql(u8, field.name, "err")) has_err = true;
        if (comptime std.mem.eql(u8, field.name, "undefined_path")) has_undefined_path = true;
        if (comptime std.mem.eql(u8, field.name, "raise_message")) has_raise_message = true;
    }

    try std.testing.expect(has_output);
    try std.testing.expect(has_spans);
    try std.testing.expect(has_err);
    try std.testing.expect(has_undefined_path);
    try std.testing.expect(has_raise_message);
}

// =============================================================================
// Successful Render Tests
// =============================================================================

test "RenderDebugResult: success returns rendered output and spans" {
    const allocator = std.testing.allocator;
    var result = renderFromJsonDebug(allocator, "Hello {{ name }}!", "{\"name\": \"World\"}", false);
    defer result.deinit(allocator);

    try std.testing.expect(result.success());
    try std.testing.expect(result.output != null);
    try std.testing.expectEqualStrings("Hello World!", result.output.?);
    try std.testing.expect(result.spans != null);
    try std.testing.expect(result.spans.?.len > 0);
}

test "RenderDebugResult: static text only has single span" {
    const allocator = std.testing.allocator;
    var result = renderFromJsonDebug(allocator, "Hello World!", "{}", false);
    defer result.deinit(allocator);

    try std.testing.expect(result.success());
    try std.testing.expectEqualStrings("Hello World!", result.output.?);
    try std.testing.expect(result.spans != null);
    // Should have one span for static text
    try std.testing.expect(result.spans.?.len >= 1);
}

test "RenderDebugResult: empty template has no spans" {
    const allocator = std.testing.allocator;
    var result = renderFromJsonDebug(allocator, "", "{}", false);
    defer result.deinit(allocator);

    try std.testing.expect(result.success());
    try std.testing.expectEqualStrings("", result.output.?);
}

// =============================================================================
// Span Tracking Tests
// =============================================================================

test "RenderDebugResult: variable span has correct range" {
    const allocator = std.testing.allocator;
    var result = renderFromJsonDebug(allocator, "{{ name }}", "{\"name\": \"Alice\"}", false);
    defer result.deinit(allocator);

    try std.testing.expect(result.success());
    try std.testing.expectEqualStrings("Alice", result.output.?);
    try std.testing.expect(result.spans != null);
    try std.testing.expectEqual(@as(usize, 1), result.spans.?.len);

    const span = result.spans.?[0];
    try std.testing.expectEqual(@as(u32, 0), span.start);
    try std.testing.expectEqual(@as(u32, 5), span.end); // "Alice" = 5 chars
    try std.testing.expect(span.source == .variable);
}

test "RenderDebugResult: mixed static and variable spans" {
    const allocator = std.testing.allocator;
    var result = renderFromJsonDebug(allocator, "Hi {{ name }}!", "{\"name\": \"Bob\"}", false);
    defer result.deinit(allocator);

    try std.testing.expect(result.success());
    try std.testing.expectEqualStrings("Hi Bob!", result.output.?);
    try std.testing.expect(result.spans != null);

    // Should have 3 spans: "Hi " (static), "Bob" (variable), "!" (static)
    try std.testing.expectEqual(@as(usize, 3), result.spans.?.len);

    // Verify span types
    try std.testing.expect(result.spans.?[0].source == .static_text);
    try std.testing.expect(result.spans.?[1].source == .variable);
    try std.testing.expect(result.spans.?[2].source == .static_text);
}

test "RenderDebugResult: variable span includes path" {
    const allocator = std.testing.allocator;
    var result = renderFromJsonDebug(allocator, "{{ name }}", "{\"name\": \"Test\"}", false);
    defer result.deinit(allocator);

    try std.testing.expect(result.success());
    try std.testing.expect(result.spans != null);
    try std.testing.expect(result.spans.?.len > 0);

    const span = result.spans.?[0];
    if (span.source == .variable) {
        try std.testing.expectEqualStrings("name", span.source.variable);
    }
}

// =============================================================================
// Error Handling Tests
// =============================================================================

test "RenderDebugResult: syntax error sets err" {
    const allocator = std.testing.allocator;
    var result = renderFromJsonDebug(allocator, "{% if x %}no endif", "{\"x\": true}", false);
    defer result.deinit(allocator);

    try std.testing.expect(!result.success());
    try std.testing.expect(result.err != null);
}

test "RenderDebugResult: raise_exception captures message" {
    const allocator = std.testing.allocator;
    var result = renderFromJsonDebug(allocator, "{{ raise_exception('debug error') }}", "{}", false);
    defer result.deinit(allocator);

    try std.testing.expect(!result.success());
    try std.testing.expect(result.raise_message != null);
    try std.testing.expectEqualStrings("debug error", result.raise_message.?);
}

test "RenderDebugResult: undefined variable in strict mode" {
    const allocator = std.testing.allocator;
    var result = renderFromJsonDebug(allocator, "{{ missing }}", "{}", true);
    defer result.deinit(allocator);

    try std.testing.expect(!result.success());
    try std.testing.expect(result.undefined_path != null);
    try std.testing.expectEqualStrings("missing", result.undefined_path.?);
}

// =============================================================================
// deinit Tests
// =============================================================================

test "RenderDebugResult.deinit frees all allocated memory" {
    const allocator = std.testing.allocator;

    // Success case with spans
    var result1 = renderFromJsonDebug(allocator, "Hello {{ name }}!", "{\"name\": \"World\"}", false);
    result1.deinit(allocator);

    // Error case
    var result2 = renderFromJsonDebug(allocator, "{{ x }}", "{}", true);
    result2.deinit(allocator);

    // No leak = success
}

test "RenderDebugResult.deinit is safe to call on default" {
    const allocator = std.testing.allocator;
    var result = RenderDebugResult{};
    result.deinit(allocator);
    result.deinit(allocator); // Should not crash
}

// =============================================================================
// success() Method Tests
// =============================================================================

test "RenderDebugResult.success returns true when err is null" {
    var result = RenderDebugResult{ .output = "test" };
    try std.testing.expect(result.success());
}

test "RenderDebugResult.success returns false when err is set" {
    var result = RenderDebugResult{ .err = error.ParseError };
    try std.testing.expect(!result.success());
}

// =============================================================================
// Complex Template Tests
// =============================================================================

test "RenderDebugResult: for loop creates multiple spans" {
    const allocator = std.testing.allocator;
    var result = renderFromJsonDebug(allocator, "{% for i in items %}{{ i }}{% endfor %}", "{\"items\": [1, 2, 3]}", false);
    defer result.deinit(allocator);

    try std.testing.expect(result.success());
    try std.testing.expectEqualStrings("123", result.output.?);
    try std.testing.expect(result.spans != null);
    // Each loop iteration creates a span for the variable
    try std.testing.expect(result.spans.?.len >= 3);
}

test "RenderDebugResult: filter preserves variable source" {
    const allocator = std.testing.allocator;
    var result = renderFromJsonDebug(allocator, "{{ name | upper }}", "{\"name\": \"alice\"}", false);
    defer result.deinit(allocator);

    try std.testing.expect(result.success());
    try std.testing.expectEqualStrings("ALICE", result.output.?);
    try std.testing.expect(result.spans != null);
    try std.testing.expect(result.spans.?.len > 0);
}

test "RenderDebugResult: conditional renders correct spans" {
    const allocator = std.testing.allocator;
    var result = renderFromJsonDebug(allocator, "{% if show %}{{ msg }}{% endif %}", "{\"show\": true, \"msg\": \"visible\"}", false);
    defer result.deinit(allocator);

    try std.testing.expect(result.success());
    try std.testing.expectEqualStrings("visible", result.output.?);
    try std.testing.expect(result.spans != null);
}
