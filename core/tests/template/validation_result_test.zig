//! Integration tests for text.template_engine.ValidationResult
//!
//! ValidationResult contains the results of validating template inputs against
//! the required and optional variables extracted from a template. It indicates
//! which variables are missing, which are optional, and which are extra.

const std = @import("std");
const main = @import("main");

const ValidationResult = main.template.ValidationResult;
const validate = main.template.validate;
const validateJson = main.template.validateJson;
const extractVariables = main.template.extractVariables;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "ValidationResult type is accessible" {
    const T = ValidationResult;
    _ = T;
}

test "ValidationResult is a struct" {
    const info = @typeInfo(ValidationResult);
    try std.testing.expect(info == .@"struct");
}

test "ValidationResult has expected fields" {
    const info = @typeInfo(ValidationResult);
    const fields = info.@"struct".fields;

    var has_required = false;
    var has_optional = false;
    var has_extra = false;
    var has_valid = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "required")) has_required = true;
        if (comptime std.mem.eql(u8, field.name, "optional")) has_optional = true;
        if (comptime std.mem.eql(u8, field.name, "extra")) has_extra = true;
        if (comptime std.mem.eql(u8, field.name, "valid")) has_valid = true;
    }

    try std.testing.expect(has_required);
    try std.testing.expect(has_optional);
    try std.testing.expect(has_extra);
    try std.testing.expect(has_valid);
}

// =============================================================================
// Validation with All Inputs Provided Tests
// =============================================================================

test "ValidationResult: valid when all required inputs provided" {
    const allocator = std.testing.allocator;
    const inputs = [_][]const u8{ "name", "age" };
    var result = try validate(allocator, "{{ name }} {{ age }}", &inputs);
    defer result.deinit(allocator);

    try std.testing.expect(result.valid);
    try std.testing.expectEqual(@as(usize, 0), result.required.len);
    try std.testing.expectEqual(@as(usize, 0), result.extra.len);
}

test "ValidationResult: valid with no variables in template" {
    const allocator = std.testing.allocator;
    const inputs = [_][]const u8{};
    var result = try validate(allocator, "Hello world!", &inputs);
    defer result.deinit(allocator);

    try std.testing.expect(result.valid);
    try std.testing.expectEqual(@as(usize, 0), result.required.len);
}

// =============================================================================
// Missing Required Variables Tests
// =============================================================================

test "ValidationResult: invalid when required variable missing" {
    const allocator = std.testing.allocator;
    const inputs = [_][]const u8{"name"};
    var result = try validate(allocator, "{{ name }} {{ age }}", &inputs);
    defer result.deinit(allocator);

    try std.testing.expect(!result.valid);
    try std.testing.expectEqual(@as(usize, 1), result.required.len);
    try std.testing.expectEqualStrings("age", result.required[0]);
}

test "ValidationResult: tracks multiple missing required variables" {
    const allocator = std.testing.allocator;
    const inputs = [_][]const u8{};
    var result = try validate(allocator, "{{ a }} {{ b }} {{ c }}", &inputs);
    defer result.deinit(allocator);

    try std.testing.expect(!result.valid);
    try std.testing.expectEqual(@as(usize, 3), result.required.len);
}

// =============================================================================
// Optional Variables Tests
// =============================================================================

test "ValidationResult: optional variable missing is still valid" {
    const allocator = std.testing.allocator;
    const inputs = [_][]const u8{"name"};
    var result = try validate(allocator, "{{ name }} {{ context | default('') }}", &inputs);
    defer result.deinit(allocator);

    try std.testing.expect(result.valid);
    try std.testing.expectEqual(@as(usize, 0), result.required.len);
    try std.testing.expectEqual(@as(usize, 1), result.optional.len);
    try std.testing.expectEqualStrings("context", result.optional[0]);
}

test "ValidationResult: d() filter alias marks variable as optional" {
    const allocator = std.testing.allocator;
    const inputs = [_][]const u8{};
    var result = try validate(allocator, "{{ value | d('fallback') }}", &inputs);
    defer result.deinit(allocator);

    try std.testing.expect(result.valid);
    try std.testing.expectEqual(@as(usize, 1), result.optional.len);
}

// =============================================================================
// Extra Variables Tests
// =============================================================================

test "ValidationResult: tracks extra unused variables" {
    const allocator = std.testing.allocator;
    const inputs = [_][]const u8{ "name", "unused1", "unused2" };
    var result = try validate(allocator, "{{ name }}", &inputs);
    defer result.deinit(allocator);

    try std.testing.expect(result.valid); // Extra vars don't invalidate
    try std.testing.expectEqual(@as(usize, 2), result.extra.len);
}

test "ValidationResult: extra variables do not cause invalid" {
    const allocator = std.testing.allocator;
    const inputs = [_][]const u8{ "name", "extra" };
    var result = try validate(allocator, "{{ name }}", &inputs);
    defer result.deinit(allocator);

    try std.testing.expect(result.valid);
    try std.testing.expectEqual(@as(usize, 1), result.extra.len);
    try std.testing.expectEqualStrings("extra", result.extra[0]);
}

// =============================================================================
// deinit Tests
// =============================================================================

test "ValidationResult.deinit frees memory" {
    const allocator = std.testing.allocator;
    const inputs = [_][]const u8{"name"};
    var result = try validate(allocator, "{{ name }} {{ missing }} {{ opt | default('') }}", &inputs);
    // deinit should free required, optional, and extra slices
    result.deinit(allocator);
    // No leak = success
}

// =============================================================================
// Complex Template Tests
// =============================================================================

test "ValidationResult: handles nested attribute access" {
    const allocator = std.testing.allocator;
    const inputs = [_][]const u8{"user"};
    var result = try validate(allocator, "{{ user.name }}", &inputs);
    defer result.deinit(allocator);

    try std.testing.expect(result.valid);
    // Root variable "user" should be tracked, not "user.name"
}

test "ValidationResult: for loop variable not considered required" {
    const allocator = std.testing.allocator;
    const inputs = [_][]const u8{"items"};
    var result = try validate(allocator, "{% for item in items %}{{ item }}{% endfor %}", &inputs);
    defer result.deinit(allocator);

    try std.testing.expect(result.valid);
    // "item" is loop variable, not required input
}

test "ValidationResult: set variable not considered required" {
    const allocator = std.testing.allocator;
    const inputs = [_][]const u8{"value"};
    var result = try validate(allocator, "{% set x = value %}{{ x }}", &inputs);
    defer result.deinit(allocator);

    try std.testing.expect(result.valid);
    // "x" is set variable, "value" is required
}

test "ValidationResult: variable used both naked and with default is required" {
    const allocator = std.testing.allocator;
    const inputs = [_][]const u8{};
    var result = try validate(allocator, "{{ x }} {{ x | default('') }}", &inputs);
    defer result.deinit(allocator);

    try std.testing.expect(!result.valid);
    try std.testing.expectEqual(@as(usize, 1), result.required.len);
    try std.testing.expectEqualStrings("x", result.required[0]);
    // Should not appear in optional since it's required
    try std.testing.expectEqual(@as(usize, 0), result.optional.len);
}

// =============================================================================
// JSON Validation Tests
// =============================================================================

test "ValidationResult: validateJson with valid JSON" {
    const allocator = std.testing.allocator;
    var result = try validateJson(allocator, "{{ name }} {{ age }}", "{\"name\": \"Alice\", \"age\": 30}");
    defer result.deinit(allocator);

    try std.testing.expect(result.valid);
}

test "ValidationResult: validateJson detects missing variables" {
    const allocator = std.testing.allocator;
    var result = try validateJson(allocator, "{{ name }} {{ age }}", "{\"name\": \"Alice\"}");
    defer result.deinit(allocator);

    try std.testing.expect(!result.valid);
    try std.testing.expectEqual(@as(usize, 1), result.required.len);
}
