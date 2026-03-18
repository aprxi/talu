//! Tool Schema Conversion
//!
//! Converts OpenAI-style tool definitions into JSON Schema for grammar-based
//! constrained sampling. The generated schema enforces valid tool call JSON.
//!
//! Input format (nested or flat):
//!
//! Nested (OpenAI):
//! [{"type": "function", "function": {"name": "get_weather", "parameters": {...}}}]
//!
//! Flat:
//! [{"type": "function", "name": "get_weather", "parameters": {...}}]
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

        // Accept both nested format ({"function": {"name": ...}}) and
        // flat format ({"name": ..., "parameters": ...}).
        const source: std.json.ObjectMap = if (tool.object.get("function")) |func| blk: {
            if (func != .object) continue;
            break :blk func.object;
        } else tool.object;

        const name_val = source.get("name") orelse continue;
        if (name_val != .string) continue;

        const params = source.get("parameters");

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

/// Parse tool calls from model-generated text.
///
/// Supports multiple output formats:
///   1. XML (Qwen3.5): <tool_call>\n<function=NAME>\n<parameter=P>V</parameter>\n</function>\n</tool_call>
///   2. JSON: {"name": "...", "arguments": {...}}
///
/// Returns one or more parsed tool calls. Caller owns all returned memory.
pub fn parseToolCallsFromText(allocator: std.mem.Allocator, text: []const u8) ![]ParsedToolCall {
    // Try XML format first (Qwen3.5 native).
    if (parseXmlToolCalls(allocator, text)) |calls| {
        if (calls.len > 0) return calls;
    } else |_| {}

    // Fallback: try JSON (first complete JSON object in text).
    const json_slice = extractJsonObject(text) orelse return ToolSchemaError.InvalidToolsJson;
    const single = parseToolCall(allocator, json_slice) catch return ToolSchemaError.InvalidToolsJson;
    const result = try allocator.alloc(ParsedToolCall, 1);
    result[0] = single;
    return result;
}

/// Parse Qwen3.5-style XML tool calls from text.
///
/// Format: <tool_call>\n<function=NAME>\n<parameter=P>\nVALUE\n</parameter>\n...</function>\n</tool_call>
fn parseXmlToolCalls(allocator: std.mem.Allocator, text: []const u8) ![]ParsedToolCall {
    const tag_start = "<tool_call>";
    const tag_end = "</tool_call>";

    // Count tool_call blocks.
    var count: usize = 0;
    {
        var search = text;
        while (std.mem.indexOf(u8, search, tag_start)) |pos| {
            count += 1;
            search = search[pos + tag_start.len ..];
        }
    }
    if (count == 0) return ToolSchemaError.InvalidToolsJson;

    const calls = try allocator.alloc(ParsedToolCall, count);
    var ci: usize = 0;
    errdefer {
        for (calls[0..ci]) |*c| c.deinit(allocator);
        allocator.free(calls);
    }

    var rest = text;
    while (std.mem.indexOf(u8, rest, tag_start)) |start_pos| {
        const block_start = start_pos + tag_start.len;
        rest = rest[block_start..];
        const block_end = std.mem.indexOf(u8, rest, tag_end) orelse rest.len;
        const block = rest[0..block_end];

        if (try parseOneXmlBlock(allocator, block)) |parsed| {
            calls[ci] = parsed;
            ci += 1;
        }

        rest = rest[@min(block_end + tag_end.len, rest.len)..];
    }

    if (ci == 0) {
        allocator.free(calls);
        return ToolSchemaError.InvalidToolsJson;
    }

    // Shrink to actual count if fewer than expected.
    if (ci < count) {
        const shrunk = try allocator.alloc(ParsedToolCall, ci);
        @memcpy(shrunk, calls[0..ci]);
        allocator.free(calls);
        return shrunk;
    }
    return calls;
}

/// Parse a single <function=NAME>...</function> block into a ParsedToolCall.
fn parseOneXmlBlock(allocator: std.mem.Allocator, block: []const u8) !?ParsedToolCall {
    const func_prefix = "<function=";
    const func_start = std.mem.indexOf(u8, block, func_prefix) orelse return null;
    const name_start = func_start + func_prefix.len;
    const name_end = std.mem.indexOfScalarPos(u8, block, name_start, '>') orelse return null;
    const func_name = std.mem.trim(u8, block[name_start..name_end], " \t\r\n");
    if (func_name.len == 0) return null;

    // Build JSON arguments from <parameter=NAME>VALUE</parameter> pairs.
    const param_prefix = "<parameter=";
    const param_suffix = "</parameter>";

    var json_buf = std.ArrayList(u8).empty;
    defer json_buf.deinit(allocator);
    try json_buf.appendSlice(allocator, "{");

    var param_count: usize = 0;
    var search = block[name_end + 1 ..];
    while (std.mem.indexOf(u8, search, param_prefix)) |pstart| {
        const pname_start = pstart + param_prefix.len;
        const pname_end = std.mem.indexOfScalarPos(u8, search, pname_start, '>') orelse break;
        const param_name = std.mem.trim(u8, search[pname_start..pname_end], " \t\r\n");
        const value_start = pname_end + 1;
        const value_end = std.mem.indexOfPos(u8, search, value_start, param_suffix) orelse break;
        const raw_value = std.mem.trim(u8, search[value_start..value_end], "\n");

        if (param_count > 0) try json_buf.appendSlice(allocator, ",");
        try writeJsonStr(&json_buf, allocator, param_name);
        try json_buf.appendSlice(allocator, ":");
        try writeJsonVal(&json_buf, allocator, raw_value);
        param_count += 1;

        search = search[value_end + param_suffix.len ..];
    }
    try json_buf.appendSlice(allocator, "}");

    return ParsedToolCall{
        .name = try allocator.dupe(u8, func_name),
        .arguments = try allocator.dupe(u8, json_buf.items),
    };
}

/// Write a JSON string: "escaped_value"
fn writeJsonStr(buf: *std.ArrayList(u8), allocator: std.mem.Allocator, s: []const u8) !void {
    try buf.appendSlice(allocator, "\"");
    for (s) |c| {
        switch (c) {
            '"' => try buf.appendSlice(allocator, "\\\""),
            '\\' => try buf.appendSlice(allocator, "\\\\"),
            '\n' => try buf.appendSlice(allocator, "\\n"),
            '\r' => try buf.appendSlice(allocator, "\\r"),
            '\t' => try buf.appendSlice(allocator, "\\t"),
            else => try buf.append(allocator, c),
        }
    }
    try buf.appendSlice(allocator, "\"");
}

/// Write a JSON value: number/bool/null pass through, otherwise quoted string.
fn writeJsonVal(buf: *std.ArrayList(u8), allocator: std.mem.Allocator, raw: []const u8) !void {
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    // Boolean / null.
    if (std.mem.eql(u8, trimmed, "true") or std.mem.eql(u8, trimmed, "false") or
        std.mem.eql(u8, trimmed, "null"))
    {
        try buf.appendSlice(allocator, trimmed);
        return;
    }
    // Number.
    if (std.fmt.parseFloat(f64, trimmed)) |_| {
        try buf.appendSlice(allocator, trimmed);
        return;
    } else |_| {}
    // JSON object/array.
    if (trimmed.len > 0 and (trimmed[0] == '{' or trimmed[0] == '[')) {
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, trimmed, .{}) catch {
            try writeJsonStr(buf, allocator, trimmed);
            return;
        };
        parsed.deinit();
        try buf.appendSlice(allocator, trimmed);
        return;
    }
    // Default: string.
    try writeJsonStr(buf, allocator, trimmed);
}

/// Extract the first top-level JSON object from text.
fn extractJsonObject(text: []const u8) ?[]const u8 {
    const start = std.mem.indexOfScalar(u8, text, '{') orelse return null;
    var depth: usize = 0;
    var in_string = false;
    var escape = false;
    for (text[start..], 0..) |c, offset| {
        if (escape) {
            escape = false;
            continue;
        }
        if (c == '\\' and in_string) {
            escape = true;
            continue;
        }
        if (c == '"') {
            in_string = !in_string;
            continue;
        }
        if (in_string) continue;
        if (c == '{') depth += 1;
        if (c == '}') {
            depth -= 1;
            if (depth == 0) {
                return text[start .. start + offset + 1];
            }
        }
    }
    return null;
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

test "toolsToGrammarSchema flat format single tool" {
    const allocator = std.testing.allocator;

    const tools_json =
        \\[{"type":"function","name":"calculate_area","parameters":{"type":"object","properties":{"base":{"type":"number"},"height":{"type":"number"}},"required":["base","height"]}}]
    ;

    const schema = try toolsToGrammarSchema(allocator, tools_json);
    defer allocator.free(schema);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, schema, .{});
    defer parsed.deinit();

    try std.testing.expect(parsed.value == .object);
    const props = parsed.value.object.get("properties").?;
    const name_prop = props.object.get("name").?;
    const name_const = name_prop.object.get("const").?;
    try std.testing.expectEqualStrings("calculate_area", name_const.string);
}

test "toolsToGrammarSchema flat format multiple tools" {
    const allocator = std.testing.allocator;

    const tools_json =
        \\[
        \\  {"type":"function","name":"get_weather","parameters":{"type":"object"}},
        \\  {"type":"function","name":"search","parameters":{"type":"object"}}
        \\]
    ;

    const schema = try toolsToGrammarSchema(allocator, tools_json);
    defer allocator.free(schema);

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

test "toolsToGrammarSchema schema compiles to grammar" {
    const allocator = std.testing.allocator;
    const validate = @import("../validate/root.zig");
    const cache = validate.cache;

    // Flat format with realistic BFCL-style parameters
    const tools_json =
        \\[{"type":"function","name":"calculate_triangle_area","description":"Calculate the area of a triangle","parameters":{"type":"object","properties":{"base":{"type":"number","description":"Length of base"},"height":{"type":"number","description":"Height"}},"required":["base","height"]}}]
    ;

    const schema = try toolsToGrammarSchema(allocator, tools_json);
    defer allocator.free(schema);

    // Schema should compile into a grammar without error
    const grammar_cache = cache.getGlobalCache(allocator);
    const grammar = grammar_cache.getOrCompile(schema, .{}) catch |err| {
        std.debug.print("Grammar compile failed: {s}\nSchema: {s}\n", .{ @errorName(err), schema });
        return err;
    };
    _ = grammar;

    cache.cleanupGlobalCaches();
}

test "toolsToGrammarSchema multi-tool schema compiles to grammar" {
    const allocator = std.testing.allocator;
    const validate = @import("../validate/root.zig");
    const cache = validate.cache;

    const tools_json =
        \\[
        \\  {"type":"function","name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}},
        \\  {"type":"function","name":"search_web","description":"Search","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}
        \\]
    ;

    const schema = try toolsToGrammarSchema(allocator, tools_json);
    defer allocator.free(schema);

    const grammar_cache = cache.getGlobalCache(allocator);
    const grammar = grammar_cache.getOrCompile(schema, .{}) catch |err| {
        std.debug.print("Grammar compile failed: {s}\nSchema: {s}\n", .{ @errorName(err), schema });
        return err;
    };
    _ = grammar;

    cache.cleanupGlobalCaches();
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

// ============================================================================
// XML tool call parsing (Qwen3.5 format)
// ============================================================================

test "parseXmlToolCalls single call" {
    const allocator = std.testing.allocator;

    const text =
        \\<tool_call>
        \\<function=calculate_area>
        \\<parameter=base>10</parameter>
        \\<parameter=height>5</parameter>
        \\</function>
        \\</tool_call>
    ;

    const calls = try parseXmlToolCalls(allocator, text);
    defer {
        for (calls) |*c| {
            var call = c.*;
            call.deinit(allocator);
        }
        allocator.free(calls);
    }

    try std.testing.expectEqual(@as(usize, 1), calls.len);
    try std.testing.expectEqualStrings("calculate_area", calls[0].name);

    // Arguments should be valid JSON with base and height.
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try std.testing.expect(parsed.value == .object);
    const base = parsed.value.object.get("base").?;
    try std.testing.expect(base == .integer);
    try std.testing.expectEqual(@as(i64, 10), base.integer);
    const height = parsed.value.object.get("height").?;
    try std.testing.expect(height == .integer);
    try std.testing.expectEqual(@as(i64, 5), height.integer);
}

test "parseXmlToolCalls multiple calls" {
    const allocator = std.testing.allocator;

    const text =
        \\<tool_call>
        \\<function=get_weather>
        \\<parameter=city>Paris</parameter>
        \\</function>
        \\</tool_call>
        \\<tool_call>
        \\<function=search_web>
        \\<parameter=query>restaurants</parameter>
        \\</function>
        \\</tool_call>
    ;

    const calls = try parseXmlToolCalls(allocator, text);
    defer {
        for (calls) |*c| {
            var call = c.*;
            call.deinit(allocator);
        }
        allocator.free(calls);
    }

    try std.testing.expectEqual(@as(usize, 2), calls.len);
    try std.testing.expectEqualStrings("get_weather", calls[0].name);
    try std.testing.expectEqualStrings("search_web", calls[1].name);
}

test "parseXmlToolCalls string parameter values" {
    const allocator = std.testing.allocator;

    const text =
        \\<tool_call>
        \\<function=greet>
        \\<parameter=name>Alice</parameter>
        \\<parameter=greeting>Hello World</parameter>
        \\</function>
        \\</tool_call>
    ;

    const calls = try parseXmlToolCalls(allocator, text);
    defer {
        for (calls) |*c| {
            var call = c.*;
            call.deinit(allocator);
        }
        allocator.free(calls);
    }

    try std.testing.expectEqual(@as(usize, 1), calls.len);
    // String values should be quoted in the JSON arguments.
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    const name = parsed.value.object.get("name").?;
    try std.testing.expectEqualStrings("Alice", name.string);
    const greeting = parsed.value.object.get("greeting").?;
    try std.testing.expectEqualStrings("Hello World", greeting.string);
}

test "parseXmlToolCalls boolean and null values" {
    const allocator = std.testing.allocator;

    const text =
        \\<tool_call>
        \\<function=set_config>
        \\<parameter=enabled>true</parameter>
        \\<parameter=verbose>false</parameter>
        \\<parameter=default>null</parameter>
        \\</function>
        \\</tool_call>
    ;

    const calls = try parseXmlToolCalls(allocator, text);
    defer {
        for (calls) |*c| {
            var call = c.*;
            call.deinit(allocator);
        }
        allocator.free(calls);
    }

    try std.testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try std.testing.expect(parsed.value.object.get("enabled").? == .bool);
    try std.testing.expect(parsed.value.object.get("verbose").? == .bool);
    try std.testing.expect(parsed.value.object.get("default").? == .null);
}

test "parseXmlToolCalls no tool_call tags returns error" {
    const allocator = std.testing.allocator;
    const result = parseXmlToolCalls(allocator, "Just some text without tool calls.");
    try std.testing.expectError(ToolSchemaError.InvalidToolsJson, result);
}

test "parseToolCallsFromText prefers XML over JSON" {
    const allocator = std.testing.allocator;

    // Text with both XML tool call and a JSON object — XML should win.
    const text =
        \\<tool_call>
        \\<function=xml_func>
        \\<parameter=x>1</parameter>
        \\</function>
        \\</tool_call>
        \\{"name":"json_func","arguments":{"x":2}}
    ;

    const calls = try parseToolCallsFromText(allocator, text);
    defer {
        for (calls) |*c| {
            var call = c.*;
            call.deinit(allocator);
        }
        allocator.free(calls);
    }

    try std.testing.expectEqual(@as(usize, 1), calls.len);
    try std.testing.expectEqualStrings("xml_func", calls[0].name);
}

test "parseToolCallsFromText falls back to JSON" {
    const allocator = std.testing.allocator;

    const text =
        \\{"name":"calculate","arguments":{"a":1,"b":2}}
    ;

    const calls = try parseToolCallsFromText(allocator, text);
    defer {
        for (calls) |*c| {
            var call = c.*;
            call.deinit(allocator);
        }
        allocator.free(calls);
    }

    try std.testing.expectEqual(@as(usize, 1), calls.len);
    try std.testing.expectEqualStrings("calculate", calls[0].name);
}

test "parseToolCallsFromText no valid format returns error" {
    const allocator = std.testing.allocator;
    const result = parseToolCallsFromText(allocator, "I cannot help with that request.");
    try std.testing.expectError(ToolSchemaError.InvalidToolsJson, result);
}

test "parseXmlToolCalls with reasoning prefix" {
    const allocator = std.testing.allocator;

    // Simulates model output with <think> block before tool call.
    const text =
        \\<think>
        \\I need to calculate the area of a triangle with base 10 and height 5.
        \\The formula is (base * height) / 2.
        \\</think>
        \\
        \\<tool_call>
        \\<function=calculate_triangle_area>
        \\<parameter=base>10</parameter>
        \\<parameter=height>5</parameter>
        \\</function>
        \\</tool_call>
    ;

    const calls = try parseXmlToolCalls(allocator, text);
    defer {
        for (calls) |*c| {
            var call = c.*;
            call.deinit(allocator);
        }
        allocator.free(calls);
    }

    try std.testing.expectEqual(@as(usize, 1), calls.len);
    try std.testing.expectEqualStrings("calculate_triangle_area", calls[0].name);
}

test "extractJsonObject finds first complete object" {
    try std.testing.expectEqualStrings(
        "{\"name\":\"test\"}",
        extractJsonObject("prefix {\"name\":\"test\"} suffix").?,
    );
}

test "extractJsonObject handles nested braces" {
    try std.testing.expectEqualStrings(
        "{\"a\":{\"b\":1}}",
        extractJsonObject("{\"a\":{\"b\":1}}").?,
    );
}

test "extractJsonObject returns null for no object" {
    try std.testing.expect(extractJsonObject("no json here") == null);
}

test "extractJsonObject handles strings with braces" {
    try std.testing.expectEqualStrings(
        "{\"text\":\"a{b}c\"}",
        extractJsonObject("{\"text\":\"a{b}c\"}").?,
    );
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
