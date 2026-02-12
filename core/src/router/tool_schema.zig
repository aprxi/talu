//! Tool Schema Conversion
//!
//! Converts OpenAI-style tool definitions into JSON Schema for grammar-based
//! constrained sampling. The generated schema enforces valid tool call JSON.
//!
//! Input format (OpenAI tools):
//! [
//!   {
//!     "type": "function",
//!     "function": {
//!       "name": "get_weather",
//!       "description": "Get weather for a location",
//!       "parameters": {
//!         "type": "object",
//!         "properties": {
//!           "location": {"type": "string"}
//!         },
//!         "required": ["location"]
//!       }
//!     }
//!   }
//! ]
//!
//! Output format (JSON Schema for tool call):
//! {
//!   "type": "object",
//!   "properties": {
//!     "name": {"enum": ["get_weather"]},
//!     "arguments": <parameters schema>
//!   },
//!   "required": ["name", "arguments"]
//! }

const std = @import("std");
const io = @import("../io/root.zig");

/// Error types for tool schema conversion.
pub const ToolSchemaError = error{
    InvalidToolsJson,
    MissingFunctionField,
    MissingNameField,
    MissingParametersField,
    OutOfMemory,
};

/// Parsed tool definition.
pub const ToolDef = struct {
    name: []const u8,
    parameters_schema: []const u8,
};

/// Convert OpenAI tools JSON to a grammar schema for tool calls.
///
/// The generated schema matches a single tool call object:
/// {"name": "<one_of_tool_names>", "arguments": {...}}
///
/// Caller owns returned memory.
pub fn toolsToGrammarSchema(allocator: std.mem.Allocator, tools_json: []const u8) ![]u8 {
    // Parse the tools array
    const parsed = io.json.parseValue(allocator, tools_json, .{ .max_size_bytes = 1 * 1024 * 1024 }) catch {
        return ToolSchemaError.InvalidToolsJson;
    };
    defer parsed.deinit();

    if (parsed.value != .array) {
        return ToolSchemaError.InvalidToolsJson;
    }

    const tools = parsed.value.array.items;
    if (tools.len == 0) {
        return ToolSchemaError.InvalidToolsJson;
    }

    const ToolInfo = struct {
        name: []const u8,
        has_params: bool,
        params: ?std.json.Value,
    };

    // Extract tool names and find parameters schema
    var tool_infos = std.ArrayList(ToolInfo).empty;
    defer tool_infos.deinit(allocator);

    for (tools) |tool| {
        if (tool != .object) continue;

        const func = tool.object.get("function") orelse continue;
        if (func != .object) continue;

        const name_val = func.object.get("name") orelse continue;
        if (name_val != .string) continue;

        // Get parameters schema (may be null)
        const params = func.object.get("parameters");

        try tool_infos.append(allocator, ToolInfo{
            .name = name_val.string,
            .has_params = params != null,
            .params = params,
        });
    }

    if (tool_infos.items.len == 0) {
        return ToolSchemaError.InvalidToolsJson;
    }

    // Build the grammar schema
    var schema_buf = std.ArrayList(u8).empty;
    errdefer schema_buf.deinit(allocator);
    const writer = schema_buf.writer(allocator);

    // Helper to write parameters schema
    const writeParamsSchema = struct {
        fn write(alloc: std.mem.Allocator, w: anytype, info: ToolInfo) !void {
            if (info.has_params and info.params != null) {
                const params_json = try std.json.Stringify.valueAlloc(alloc, info.params.?, .{});
                defer alloc.free(params_json);
                try w.writeAll(params_json);
            } else {
                // Default empty object schema
                try w.writeAll(
                    \\{"type":"object"}
                );
            }
        }
    };

    // For single tool, use its parameters directly
    // For multiple tools, use anyOf with each tool's schema
    if (tool_infos.items.len == 1) {
        const info = tool_infos.items[0];
        try writer.writeAll(
            \\{"type":"object","properties":{"name":{"const":"
        );
        try writer.writeAll(info.name);
        try writer.writeAll(
            \\"},"arguments":
        );
        try writeParamsSchema.write(allocator, writer, info);
        try writer.writeAll(
            \\},"required":["name","arguments"],"additionalProperties":false}
        );
    } else {
        // Multiple tools: use anyOf
        try writer.writeAll(
            \\{"anyOf":[
        );
        for (tool_infos.items, 0..) |info, i| {
            if (i > 0) try writer.writeByte(',');
            try writer.writeAll(
                \\{"type":"object","properties":{"name":{"const":"
            );
            try writer.writeAll(info.name);
            try writer.writeAll(
                \\"},"arguments":
            );
            try writeParamsSchema.write(allocator, writer, info);
            try writer.writeAll(
                \\},"required":["name","arguments"],"additionalProperties":false}
            );
        }
        try writer.writeAll(
            \\]}
        );
    }

    return schema_buf.toOwnedSlice(allocator);
}

/// Generate a unique call_id for a tool call.
/// Format: "call_<random_hex>"
pub fn generateCallId(allocator: std.mem.Allocator) ![]u8 {
    var buf: [16]u8 = undefined;
    std.crypto.random.bytes(&buf);

    var result = try allocator.alloc(u8, 5 + 32); // "call_" + 32 hex chars
    @memcpy(result[0..5], "call_");

    const hex_chars = "0123456789abcdef";
    for (buf, 0..) |byte, i| {
        result[5 + i * 2] = hex_chars[byte >> 4];
        result[5 + i * 2 + 1] = hex_chars[byte & 0x0f];
    }

    return result;
}

/// Parse a tool call JSON string into name and arguments.
/// Returns error if JSON is invalid or missing required fields.
pub const ParsedToolCall = struct {
    name: []const u8,
    arguments: []const u8,

    pub fn deinit(self: *ParsedToolCall, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.arguments);
    }
};

pub fn parseToolCall(allocator: std.mem.Allocator, json: []const u8) !ParsedToolCall {
    const parsed = io.json.parseValue(allocator, json, .{ .max_size_bytes = 1 * 1024 * 1024 }) catch {
        return ToolSchemaError.InvalidToolsJson;
    };
    defer parsed.deinit();

    if (parsed.value != .object) {
        return ToolSchemaError.InvalidToolsJson;
    }

    const name_val = parsed.value.object.get("name") orelse {
        return ToolSchemaError.MissingNameField;
    };
    if (name_val != .string) {
        return ToolSchemaError.MissingNameField;
    }

    const arguments_val = parsed.value.object.get("arguments") orelse {
        return ToolSchemaError.MissingParametersField;
    };

    // Serialize arguments back to JSON string
    const arguments_json = try std.json.Stringify.valueAlloc(allocator, arguments_val, .{});
    errdefer allocator.free(arguments_json);

    return ParsedToolCall{
        .name = try allocator.dupe(u8, name_val.string),
        .arguments = arguments_json,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "toolsToGrammarSchema single tool" {
    const allocator = std.testing.allocator;

    const tools_json =
        \\[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}]
    ;

    const schema = try toolsToGrammarSchema(allocator, tools_json);
    defer allocator.free(schema);

    // Should produce valid JSON
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, schema, .{});
    defer parsed.deinit();

    // Check structure
    try std.testing.expect(parsed.value == .object);
    const props = parsed.value.object.get("properties").?;
    try std.testing.expect(props == .object);

    // Check name constraint
    const name_prop = props.object.get("name").?;
    const name_const = name_prop.object.get("const").?;
    try std.testing.expectEqualStrings("get_weather", name_const.string);
}

test "toolsToGrammarSchema multiple tools" {
    const allocator = std.testing.allocator;

    const tools_json =
        \\[
        \\  {"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}},
        \\  {"type":"function","function":{"name":"search","parameters":{"type":"object"}}}
        \\]
    ;

    const schema = try toolsToGrammarSchema(allocator, tools_json);
    defer allocator.free(schema);

    // Should produce valid JSON with anyOf
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, schema, .{});
    defer parsed.deinit();

    try std.testing.expect(parsed.value == .object);
    const any_of = parsed.value.object.get("anyOf").?;
    try std.testing.expect(any_of == .array);
    try std.testing.expectEqual(@as(usize, 2), any_of.array.items.len);
}

test "generateCallId produces valid format" {
    const allocator = std.testing.allocator;

    const call_id = try generateCallId(allocator);
    defer allocator.free(call_id);

    try std.testing.expect(std.mem.startsWith(u8, call_id, "call_"));
    try std.testing.expectEqual(@as(usize, 37), call_id.len); // "call_" + 32 hex chars
}

test "parseToolCall valid input" {
    const allocator = std.testing.allocator;

    const json =
        \\{"name":"get_weather","arguments":{"location":"Paris"}}
    ;

    var parsed = try parseToolCall(allocator, json);
    defer parsed.deinit(allocator);

    try std.testing.expectEqualStrings("get_weather", parsed.name);
    try std.testing.expect(std.mem.indexOf(u8, parsed.arguments, "Paris") != null);
}

test "parseToolCall missing name" {
    const allocator = std.testing.allocator;

    const json =
        \\{"arguments":{"location":"Paris"}}
    ;

    const result = parseToolCall(allocator, json);
    try std.testing.expectError(ToolSchemaError.MissingNameField, result);
}

test "fuzz toolsToGrammarSchema" {
    try std.testing.fuzz({}, struct {
        fn testOne(_: void, input: []const u8) !void {
            const alloc = std.testing.allocator;
            const json_z = try alloc.dupeZ(u8, input);
            defer alloc.free(json_z);
            _ = toolsToGrammarSchema(alloc, json_z[0..input.len]) catch {};
        }
    }.testOne, .{});
}

test "fuzz parseToolCall" {
    try std.testing.fuzz({}, struct {
        fn testOne(_: void, input: []const u8) !void {
            const alloc = std.testing.allocator;
            const json_z = try alloc.dupeZ(u8, input);
            defer alloc.free(json_z);
            if (parseToolCall(alloc, json_z[0..input.len])) |parsed| {
                var tmp = parsed;
                tmp.deinit(alloc);
            } else |_| {}
        }
    }.testOne, .{});
}
