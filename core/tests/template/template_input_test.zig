//! Integration tests for TemplateInput
//!
//! TemplateInput represents dynamic values that can be passed to templates.
//! It supports strings, integers, floats, booleans, arrays, maps, and none.

const std = @import("std");
const main = @import("main");
const TemplateInput = main.template.TemplateInput;

// =============================================================================
// Type Construction Tests
// =============================================================================

test "TemplateInput: construct string" {
    const value: TemplateInput = .{ .string = "hello" };
    try std.testing.expectEqualStrings("hello", value.string);
}

test "TemplateInput: construct integer" {
    const value: TemplateInput = .{ .integer = 42 };
    try std.testing.expectEqual(@as(i64, 42), value.integer);
}

test "TemplateInput: construct float" {
    const value: TemplateInput = .{ .float = 3.14 };
    try std.testing.expectEqual(@as(f64, 3.14), value.float);
}

test "TemplateInput: construct boolean true" {
    const value: TemplateInput = .{ .boolean = true };
    try std.testing.expect(value.boolean);
}

test "TemplateInput: construct boolean false" {
    const value: TemplateInput = .{ .boolean = false };
    try std.testing.expect(!value.boolean);
}

test "TemplateInput: construct none" {
    const value: TemplateInput = .none;
    try std.testing.expectEqual(TemplateInput.none, value);
}

test "TemplateInput: construct array" {
    const items = [_]TemplateInput{
        .{ .integer = 1 },
        .{ .integer = 2 },
        .{ .integer = 3 },
    };
    const value: TemplateInput = .{ .array = &items };
    try std.testing.expectEqual(@as(usize, 3), value.array.len);
    try std.testing.expectEqual(@as(i64, 1), value.array[0].integer);
    try std.testing.expectEqual(@as(i64, 2), value.array[1].integer);
    try std.testing.expectEqual(@as(i64, 3), value.array[2].integer);
}

// =============================================================================
// isTruthy Tests
// =============================================================================

test "TemplateInput.isTruthy: non-empty string is truthy" {
    const value: TemplateInput = .{ .string = "hello" };
    try std.testing.expect(value.isTruthy());
}

test "TemplateInput.isTruthy: empty string is falsy" {
    const value: TemplateInput = .{ .string = "" };
    try std.testing.expect(!value.isTruthy());
}

test "TemplateInput.isTruthy: non-zero integer is truthy" {
    const positive: TemplateInput = .{ .integer = 42 };
    const negative: TemplateInput = .{ .integer = -1 };
    try std.testing.expect(positive.isTruthy());
    try std.testing.expect(negative.isTruthy());
}

test "TemplateInput.isTruthy: zero integer is falsy" {
    const value: TemplateInput = .{ .integer = 0 };
    try std.testing.expect(!value.isTruthy());
}

test "TemplateInput.isTruthy: non-zero float is truthy" {
    const value: TemplateInput = .{ .float = 0.001 };
    try std.testing.expect(value.isTruthy());
}

test "TemplateInput.isTruthy: zero float is falsy" {
    const value: TemplateInput = .{ .float = 0.0 };
    try std.testing.expect(!value.isTruthy());
}

test "TemplateInput.isTruthy: true boolean is truthy" {
    const value: TemplateInput = .{ .boolean = true };
    try std.testing.expect(value.isTruthy());
}

test "TemplateInput.isTruthy: false boolean is falsy" {
    const value: TemplateInput = .{ .boolean = false };
    try std.testing.expect(!value.isTruthy());
}

test "TemplateInput.isTruthy: none is falsy" {
    const value: TemplateInput = .none;
    try std.testing.expect(!value.isTruthy());
}

test "TemplateInput.isTruthy: non-empty array is truthy" {
    const items = [_]TemplateInput{.{ .integer = 1 }};
    const value: TemplateInput = .{ .array = &items };
    try std.testing.expect(value.isTruthy());
}

test "TemplateInput.isTruthy: empty array is falsy" {
    const value: TemplateInput = .{ .array = &[_]TemplateInput{} };
    try std.testing.expect(!value.isTruthy());
}

// =============================================================================
// asNumber Tests
// =============================================================================

test "TemplateInput.asNumber: integer returns float equivalent" {
    const value: TemplateInput = .{ .integer = 42 };
    const num = value.asNumber();
    try std.testing.expect(num != null);
    try std.testing.expectEqual(@as(f64, 42.0), num.?);
}

test "TemplateInput.asNumber: float returns itself" {
    const value: TemplateInput = .{ .float = 3.14 };
    const num = value.asNumber();
    try std.testing.expect(num != null);
    try std.testing.expectEqual(@as(f64, 3.14), num.?);
}

test "TemplateInput.asNumber: string returns null" {
    const value: TemplateInput = .{ .string = "hello" };
    try std.testing.expect(value.asNumber() == null);
}

test "TemplateInput.asNumber: boolean returns null" {
    const value: TemplateInput = .{ .boolean = true };
    try std.testing.expect(value.asNumber() == null);
}

test "TemplateInput.asNumber: none returns null" {
    const value: TemplateInput = .none;
    try std.testing.expect(value.asNumber() == null);
}

// =============================================================================
// asString Tests
// =============================================================================

test "TemplateInput.asString: string returns itself" {
    const allocator = std.testing.allocator;
    const value: TemplateInput = .{ .string = "hello" };
    const result = try value.asString(allocator);
    try std.testing.expectEqualStrings("hello", result);
    // String values return the original slice, no allocation
}

test "TemplateInput.asString: integer formats as decimal" {
    const allocator = std.testing.allocator;
    const value: TemplateInput = .{ .integer = 42 };
    const result = try value.asString(allocator);
    defer allocator.free(result);
    try std.testing.expectEqualStrings("42", result);
}

test "TemplateInput.asString: negative integer" {
    const allocator = std.testing.allocator;
    const value: TemplateInput = .{ .integer = -123 };
    const result = try value.asString(allocator);
    defer allocator.free(result);
    try std.testing.expectEqualStrings("-123", result);
}

test "TemplateInput.asString: float formats with decimal" {
    const allocator = std.testing.allocator;
    const value: TemplateInput = .{ .float = 3.14 };
    const result = try value.asString(allocator);
    defer allocator.free(result);
    try std.testing.expectEqualStrings("3.14", result);
}

test "TemplateInput.asString: whole float gets .0 suffix" {
    const allocator = std.testing.allocator;
    const value: TemplateInput = .{ .float = 5.0 };
    const result = try value.asString(allocator);
    defer allocator.free(result);
    try std.testing.expectEqualStrings("5.0", result);
}

test "TemplateInput.asString: true boolean returns 'True'" {
    const allocator = std.testing.allocator;
    const value: TemplateInput = .{ .boolean = true };
    const result = try value.asString(allocator);
    try std.testing.expectEqualStrings("True", result);
}

test "TemplateInput.asString: false boolean returns 'False'" {
    const allocator = std.testing.allocator;
    const value: TemplateInput = .{ .boolean = false };
    const result = try value.asString(allocator);
    try std.testing.expectEqualStrings("False", result);
}

test "TemplateInput.asString: none returns empty string" {
    const allocator = std.testing.allocator;
    const value: TemplateInput = .none;
    const result = try value.asString(allocator);
    try std.testing.expectEqualStrings("", result);
}

// =============================================================================
// eql Tests
// =============================================================================

test "TemplateInput.eql: same strings are equal" {
    const a: TemplateInput = .{ .string = "hello" };
    const b: TemplateInput = .{ .string = "hello" };
    try std.testing.expect(a.eql(b));
}

test "TemplateInput.eql: different strings are not equal" {
    const a: TemplateInput = .{ .string = "hello" };
    const b: TemplateInput = .{ .string = "world" };
    try std.testing.expect(!a.eql(b));
}

test "TemplateInput.eql: same integers are equal" {
    const a: TemplateInput = .{ .integer = 42 };
    const b: TemplateInput = .{ .integer = 42 };
    try std.testing.expect(a.eql(b));
}

test "TemplateInput.eql: different integers are not equal" {
    const a: TemplateInput = .{ .integer = 42 };
    const b: TemplateInput = .{ .integer = 43 };
    try std.testing.expect(!a.eql(b));
}

test "TemplateInput.eql: integer and equivalent float are equal" {
    const a: TemplateInput = .{ .integer = 42 };
    const b: TemplateInput = .{ .float = 42.0 };
    try std.testing.expect(a.eql(b));
}

test "TemplateInput.eql: same booleans are equal" {
    const a: TemplateInput = .{ .boolean = true };
    const b: TemplateInput = .{ .boolean = true };
    try std.testing.expect(a.eql(b));
}

test "TemplateInput.eql: different booleans are not equal" {
    const a: TemplateInput = .{ .boolean = true };
    const b: TemplateInput = .{ .boolean = false };
    try std.testing.expect(!a.eql(b));
}

test "TemplateInput.eql: none equals none" {
    const a: TemplateInput = .none;
    const b: TemplateInput = .none;
    try std.testing.expect(a.eql(b));
}

test "TemplateInput.eql: different types are not equal" {
    const str: TemplateInput = .{ .string = "42" };
    const int: TemplateInput = .{ .integer = 42 };
    try std.testing.expect(!str.eql(int));
}

// =============================================================================
// toJson Tests
// =============================================================================

test "TemplateInput.toJson: string with quotes escaped" {
    const allocator = std.testing.allocator;
    const value: TemplateInput = .{ .string = "hello" };
    const json = try value.toJson(allocator);
    defer allocator.free(json);
    try std.testing.expectEqualStrings("\"hello\"", json);
}

test "TemplateInput.toJson: string with special characters" {
    const allocator = std.testing.allocator;
    const value: TemplateInput = .{ .string = "line1\nline2\ttab" };
    const json = try value.toJson(allocator);
    defer allocator.free(json);
    try std.testing.expectEqualStrings("\"line1\\nline2\\ttab\"", json);
}

test "TemplateInput.toJson: integer" {
    const allocator = std.testing.allocator;
    const value: TemplateInput = .{ .integer = 42 };
    const json = try value.toJson(allocator);
    defer allocator.free(json);
    try std.testing.expectEqualStrings("42", json);
}

test "TemplateInput.toJson: float" {
    const allocator = std.testing.allocator;
    const value: TemplateInput = .{ .float = 3.14 };
    const json = try value.toJson(allocator);
    defer allocator.free(json);
    try std.testing.expectEqualStrings("3.14", json);
}

test "TemplateInput.toJson: boolean true" {
    const allocator = std.testing.allocator;
    const value: TemplateInput = .{ .boolean = true };
    const json = try value.toJson(allocator);
    defer allocator.free(json);
    try std.testing.expectEqualStrings("true", json);
}

test "TemplateInput.toJson: boolean false" {
    const allocator = std.testing.allocator;
    const value: TemplateInput = .{ .boolean = false };
    const json = try value.toJson(allocator);
    defer allocator.free(json);
    try std.testing.expectEqualStrings("false", json);
}

test "TemplateInput.toJson: none becomes null" {
    const allocator = std.testing.allocator;
    const value: TemplateInput = .none;
    const json = try value.toJson(allocator);
    defer allocator.free(json);
    try std.testing.expectEqualStrings("null", json);
}

test "TemplateInput.toJson: array" {
    const allocator = std.testing.allocator;
    const items = [_]TemplateInput{
        .{ .integer = 1 },
        .{ .integer = 2 },
        .{ .integer = 3 },
    };
    const value: TemplateInput = .{ .array = &items };
    const json = try value.toJson(allocator);
    defer allocator.free(json);
    try std.testing.expectEqualStrings("[1, 2, 3]", json);
}

test "TemplateInput.toJson: empty array" {
    const allocator = std.testing.allocator;
    const value: TemplateInput = .{ .array = &[_]TemplateInput{} };
    const json = try value.toJson(allocator);
    defer allocator.free(json);
    try std.testing.expectEqualStrings("[]", json);
}

test "TemplateInput.toJson: nested array" {
    const allocator = std.testing.allocator;
    const inner = [_]TemplateInput{ .{ .integer = 1 }, .{ .integer = 2 } };
    const outer = [_]TemplateInput{
        .{ .array = &inner },
        .{ .string = "test" },
    };
    const value: TemplateInput = .{ .array = &outer };
    const json = try value.toJson(allocator);
    defer allocator.free(json);
    try std.testing.expectEqualStrings("[[1, 2], \"test\"]", json);
}
