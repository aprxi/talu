//! Integration tests for ParsedToolCall
//!
//! ParsedToolCall represents a parsed tool/function call from LLM output,
//! containing the function name and JSON arguments.

const std = @import("std");
const main = @import("main");
const ParsedToolCall = main.router.ParsedToolCall;
const parseToolCall = main.router.parseToolCall;
const ToolSchemaError = main.router.ToolSchemaError;

// =============================================================================
// Struct Layout Tests
// =============================================================================

test "ParsedToolCall has expected fields" {
    // Verify all fields exist with correct types via comptime reflection
    const fields = @typeInfo(ParsedToolCall).@"struct".fields;

    try std.testing.expectEqual(@as(usize, 2), fields.len);

    // Check field names exist using inline for (comptime iteration)
    var has_name = false;
    var has_arguments = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "name")) has_name = true;
        if (comptime std.mem.eql(u8, field.name, "arguments")) has_arguments = true;
    }

    try std.testing.expect(has_name);
    try std.testing.expect(has_arguments);
}

// =============================================================================
// Construction Tests
// =============================================================================

test "ParsedToolCall can be constructed via parseToolCall" {
    const allocator = std.testing.allocator;

    const json =
        \\{"name":"get_weather","arguments":{"location":"Paris","unit":"celsius"}}
    ;

    var parsed = try parseToolCall(allocator, json);
    defer parsed.deinit(allocator);

    try std.testing.expectEqualStrings("get_weather", parsed.name);
    try std.testing.expect(std.mem.indexOf(u8, parsed.arguments, "Paris") != null);
    try std.testing.expect(std.mem.indexOf(u8, parsed.arguments, "celsius") != null);
}

test "ParsedToolCall handles empty arguments object" {
    const allocator = std.testing.allocator;

    const json =
        \\{"name":"ping","arguments":{}}
    ;

    var parsed = try parseToolCall(allocator, json);
    defer parsed.deinit(allocator);

    try std.testing.expectEqualStrings("ping", parsed.name);
    try std.testing.expectEqualStrings("{}", parsed.arguments);
}

test "ParsedToolCall handles complex nested arguments" {
    const allocator = std.testing.allocator;

    const json =
        \\{"name":"search","arguments":{"filters":{"min":0,"max":100},"options":["a","b"]}}
    ;

    var parsed = try parseToolCall(allocator, json);
    defer parsed.deinit(allocator);

    try std.testing.expectEqualStrings("search", parsed.name);
    try std.testing.expect(std.mem.indexOf(u8, parsed.arguments, "filters") != null);
    try std.testing.expect(std.mem.indexOf(u8, parsed.arguments, "options") != null);
}

// =============================================================================
// Method Tests
// =============================================================================

test "ParsedToolCall.deinit frees memory" {
    const allocator = std.testing.allocator;

    const json =
        \\{"name":"test_function","arguments":{"key":"value"}}
    ;

    var parsed = try parseToolCall(allocator, json);
    parsed.deinit(allocator);
    // No leak = test passes (allocator detects leaks)
}

// =============================================================================
// Error Tests
// =============================================================================

test "parseToolCall returns error for missing name" {
    const allocator = std.testing.allocator;

    const json =
        \\{"arguments":{"location":"Paris"}}
    ;

    const result = parseToolCall(allocator, json);
    try std.testing.expectError(ToolSchemaError.MissingNameField, result);
}

test "parseToolCall returns error for missing arguments" {
    const allocator = std.testing.allocator;

    const json =
        \\{"name":"get_weather"}
    ;

    const result = parseToolCall(allocator, json);
    try std.testing.expectError(ToolSchemaError.MissingParametersField, result);
}

test "parseToolCall returns error for invalid JSON" {
    const allocator = std.testing.allocator;

    const json = "not valid json";

    const result = parseToolCall(allocator, json);
    try std.testing.expectError(ToolSchemaError.InvalidToolsJson, result);
}

test "parseToolCall returns error for non-object JSON" {
    const allocator = std.testing.allocator;

    const json = "[1, 2, 3]";

    const result = parseToolCall(allocator, json);
    try std.testing.expectError(ToolSchemaError.InvalidToolsJson, result);
}

test "parseToolCall returns error for non-string name" {
    const allocator = std.testing.allocator;

    const json =
        \\{"name":123,"arguments":{}}
    ;

    const result = parseToolCall(allocator, json);
    try std.testing.expectError(ToolSchemaError.MissingNameField, result);
}
