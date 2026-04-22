//! Tool - Runtime-polymorphic tool interface and registry for agent tool execution.
//!
//! Provides a vtable-based `Tool` interface (following the `StorageBackend` pattern)
//! and a `ToolRegistry` for managing named tools. Tools are registered at runtime,
//! enabling Python and other bindings to inject tool implementations via C callbacks.
//!
//! # Thread Safety
//!
//! NOT thread-safe. Callers must synchronize access externally.
//!
//! # Ownership
//!
//! `ToolRegistry.deinit()` calls `deinit()` on each registered tool.
//! The registry owns registered tools after `register()`.

const std = @import("std");
const Allocator = std.mem.Allocator;

// =============================================================================
// ToolResult - Output from a tool execution
// =============================================================================

/// Result of executing a tool.
///
/// `output` is the text returned to the LLM as the function_call_output content.
/// Error results still contain text describing the error for the LLM to interpret.
///
/// Caller owns `output` and must free it with `deinit()`.
pub const ToolResult = struct {
    /// Content returned to the LLM as function_call_output.
    output: []const u8,
    /// Whether this result represents an error. Error text is still in `output`.
    is_error: bool = false,

    /// Free the output buffer.
    pub fn deinit(self: *ToolResult, allocator: Allocator) void {
        allocator.free(self.output);
        self.* = undefined;
    }
};

// =============================================================================
// Tool - Vtable-based polymorphic tool interface
// =============================================================================

/// Runtime-polymorphic tool interface.
///
/// Follows the `StorageBackend` vtable pattern from `responses/backend.zig`.
/// Concrete tool implementations provide a pointer and a vtable with function
/// pointers for name, description, schema, execute, and cleanup.
pub const Tool = struct {
    /// Pointer to the concrete tool instance.
    ptr: *anyopaque,
    /// Virtual function table.
    vtable: *const VTable,

    pub const VTable = struct {
        /// Returns the tool name (e.g. "read_file"). Borrowed from the tool, not allocated.
        name: *const fn (ctx: *anyopaque) []const u8,

        /// Returns a human-readable description. Borrowed from the tool, not allocated.
        description: *const fn (ctx: *anyopaque) []const u8,

        /// Returns the JSON Schema string for the tool's parameters.
        /// Borrowed from the tool, not allocated.
        parametersSchema: *const fn (ctx: *anyopaque) []const u8,

        /// Execute the tool with the given arguments JSON string.
        /// Returns a ToolResult with output allocated using the provided allocator.
        /// Caller owns the returned ToolResult.output.
        execute: *const fn (ctx: *anyopaque, allocator: Allocator, arguments_json: []const u8) anyerror!ToolResult,

        /// Clean up the concrete tool instance.
        deinit: *const fn (ctx: *anyopaque) void,
    };

    /// Create a Tool from a concrete implementation pointer and vtable.
    pub fn init(ptr: anytype, vtable: *const VTable) Tool {
        const Ptr = @TypeOf(ptr);
        const ptr_info = @typeInfo(Ptr);

        comptime {
            if (ptr_info != .pointer) {
                @compileError("ptr must be a pointer type");
            }
        }

        return .{
            .ptr = @ptrCast(@alignCast(ptr)),
            .vtable = vtable,
        };
    }

    /// Returns the tool name.
    pub fn getName(self: Tool) []const u8 {
        return self.vtable.name(self.ptr);
    }

    /// Returns the tool description.
    pub fn getDescription(self: Tool) []const u8 {
        return self.vtable.description(self.ptr);
    }

    /// Returns the JSON Schema for parameters.
    pub fn getParametersSchema(self: Tool) []const u8 {
        return self.vtable.parametersSchema(self.ptr);
    }

    /// Execute the tool with arguments JSON. Caller owns returned ToolResult.output.
    pub fn execute(self: Tool, allocator: Allocator, arguments_json: []const u8) anyerror!ToolResult {
        return self.vtable.execute(self.ptr, allocator, arguments_json);
    }

    /// Clean up the tool instance.
    pub fn deinit(self: Tool) void {
        self.vtable.deinit(self.ptr);
    }
};

// =============================================================================
// ToolRegistry - Named tool collection
// =============================================================================

pub const ToolRegistryError = error{
    ToolAlreadyRegistered,
    ToolNotFound,
};

/// Registry of named tools for agent execution.
///
/// Tools are registered by name and looked up during the agent loop when the
/// LLM emits tool calls. The registry owns all registered tools and calls
/// `deinit()` on each when the registry itself is destroyed.
pub const ToolRegistry = struct {
    allocator: Allocator,
    tools: std.StringHashMapUnmanaged(Tool),

    pub fn init(allocator: Allocator) ToolRegistry {
        return .{
            .allocator = allocator,
            .tools = .{},
        };
    }

    /// Clean up all registered tools and free the map.
    pub fn deinit(self: *ToolRegistry) void {
        var it = self.tools.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.deinit();
            self.allocator.free(entry.key_ptr.*);
        }
        self.tools.deinit(self.allocator);
    }

    /// Register a tool. The registry takes ownership and will call tool.deinit() on cleanup.
    /// Returns error if a tool with the same name is already registered.
    pub fn register(self: *ToolRegistry, tool: Tool) !void {
        const name = tool.getName();
        const gop = try self.tools.getOrPut(self.allocator, name);
        if (gop.found_existing) {
            return ToolRegistryError.ToolAlreadyRegistered;
        }
        // Own the key so we don't depend on the tool's memory for map lookups
        const owned_key = try self.allocator.dupe(u8, name);
        gop.key_ptr.* = owned_key;
        gop.value_ptr.* = tool;
    }

    /// Look up a tool by name. Returns null if not found.
    pub fn get(self: *const ToolRegistry, name: []const u8) ?Tool {
        return self.tools.get(name);
    }

    /// Execute a tool by name with arguments JSON.
    /// Returns ToolNotFound if no tool with that name is registered.
    /// Caller owns the returned ToolResult.output.
    pub fn execute(self: *const ToolRegistry, allocator: Allocator, name: []const u8, arguments_json: []const u8) !ToolResult {
        const tool = self.tools.get(name) orelse return ToolRegistryError.ToolNotFound;
        return tool.execute(allocator, arguments_json);
    }

    /// Number of registered tools.
    pub fn count(self: *const ToolRegistry) usize {
        return self.tools.count();
    }

    /// Generate OpenAI-format tools JSON array.
    ///
    /// Produces: [{"type":"function","function":{"name":"...","description":"...","parameters":{...}}}]
    ///
    /// Caller owns returned memory.
    pub fn getToolDefinitionsJson(self: *const ToolRegistry, allocator: Allocator) ![]u8 {
        var buf = std.ArrayListUnmanaged(u8){};
        errdefer buf.deinit(allocator);
        const writer = buf.writer(allocator);

        try writer.writeByte('[');

        var first = true;
        var it = self.tools.iterator();
        while (it.next()) |entry| {
            if (!first) {
                try writer.writeByte(',');
            }
            first = false;

            const tool = entry.value_ptr.*;
            try writer.writeAll("{\"type\":\"function\",\"function\":{\"name\":\"");
            try writeJsonEscaped(writer, tool.getName());
            try writer.writeAll("\",\"description\":\"");
            try writeJsonEscaped(writer, tool.getDescription());
            try writer.writeAll("\",\"parameters\":");
            // Parameters schema is already valid JSON, append directly
            const schema = tool.getParametersSchema();
            if (schema.len == 0) {
                try writer.writeAll("{\"type\":\"object\",\"properties\":{}}");
            } else {
                try writer.writeAll(schema);
            }
            try writer.writeAll("}}");
        }

        try writer.writeByte(']');

        return buf.toOwnedSlice(allocator);
    }
};

/// Write a JSON-escaped string (without surrounding quotes) to the writer.
fn writeJsonEscaped(writer: anytype, s: []const u8) !void {
    for (s) |c| {
        switch (c) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            else => {
                if (c < 0x20) {
                    try writer.writeAll("\\u00");
                    try writer.writeByte(hexDigit(c >> 4));
                    try writer.writeByte(hexDigit(c & 0x0f));
                } else {
                    try writer.writeByte(c);
                }
            },
        }
    }
}

fn hexDigit(nibble: u8) u8 {
    return if (nibble < 10) '0' + nibble else 'a' + nibble - 10;
}

// =============================================================================
// Mock tool for testing
// =============================================================================

/// A simple mock tool for unit tests. Returns a fixed output string.
const MockTool = struct {
    tool_name: []const u8,
    tool_description: []const u8,
    tool_schema: []const u8,
    execute_output: []const u8,
    execute_is_error: bool = false,
    execute_count: usize = 0,
    deinit_called: bool = false,

    fn name(ctx: *anyopaque) []const u8 {
        const self: *MockTool = @ptrCast(@alignCast(ctx));
        return self.tool_name;
    }

    fn description(ctx: *anyopaque) []const u8 {
        const self: *MockTool = @ptrCast(@alignCast(ctx));
        return self.tool_description;
    }

    fn parametersSchema(ctx: *anyopaque) []const u8 {
        const self: *MockTool = @ptrCast(@alignCast(ctx));
        return self.tool_schema;
    }

    fn executeFn(ctx: *anyopaque, allocator: Allocator, _: []const u8) anyerror!ToolResult {
        const self: *MockTool = @ptrCast(@alignCast(ctx));
        self.execute_count += 1;
        return .{
            .output = try allocator.dupe(u8, self.execute_output),
            .is_error = self.execute_is_error,
        };
    }

    fn deinitFn(ctx: *anyopaque) void {
        const self: *MockTool = @ptrCast(@alignCast(ctx));
        self.deinit_called = true;
    }

    const vtable = Tool.VTable{
        .name = name,
        .description = description,
        .parametersSchema = parametersSchema,
        .execute = executeFn,
        .deinit = deinitFn,
    };

    fn asTool(self: *MockTool) Tool {
        return Tool.init(self, &vtable);
    }
};

// =============================================================================
// Tests
// =============================================================================

test "Tool.init creates tool from pointer and vtable" {
    var mock = MockTool{
        .tool_name = "test_tool",
        .tool_description = "A test tool",
        .tool_schema = "{}",
        .execute_output = "result",
    };

    const tool = Tool.init(&mock, &MockTool.vtable);
    try std.testing.expectEqualStrings("test_tool", tool.getName());
    try std.testing.expectEqualStrings("A test tool", tool.getDescription());
    try std.testing.expectEqualStrings("{}", tool.getParametersSchema());
}

test "Tool.execute delegates to vtable" {
    const allocator = std.testing.allocator;
    var mock = MockTool{
        .tool_name = "exec_tool",
        .tool_description = "Executes",
        .tool_schema = "{}",
        .execute_output = "hello world",
    };

    const tool = Tool.init(&mock, &MockTool.vtable);
    var result = try tool.execute(allocator, "{\"arg\":1}");
    defer result.deinit(allocator);

    try std.testing.expectEqualStrings("hello world", result.output);
    try std.testing.expect(!result.is_error);
    try std.testing.expectEqual(@as(usize, 1), mock.execute_count);
}

test "Tool.execute returns error result" {
    const allocator = std.testing.allocator;
    var mock = MockTool{
        .tool_name = "err_tool",
        .tool_description = "Errors",
        .tool_schema = "{}",
        .execute_output = "something went wrong",
        .execute_is_error = true,
    };

    const tool = Tool.init(&mock, &MockTool.vtable);
    var result = try tool.execute(allocator, "{}");
    defer result.deinit(allocator);

    try std.testing.expectEqualStrings("something went wrong", result.output);
    try std.testing.expect(result.is_error);
}

test "Tool.deinit delegates to vtable" {
    var mock = MockTool{
        .tool_name = "deinit_tool",
        .tool_description = "Tests deinit",
        .tool_schema = "{}",
        .execute_output = "",
    };

    const tool = Tool.init(&mock, &MockTool.vtable);
    tool.deinit();

    try std.testing.expect(mock.deinit_called);
}

test "ToolRegistry.register and get" {
    const allocator = std.testing.allocator;
    var mock = MockTool{
        .tool_name = "my_tool",
        .tool_description = "My tool",
        .tool_schema = "{}",
        .execute_output = "ok",
    };

    var registry = ToolRegistry.init(allocator);
    defer registry.deinit();

    try registry.register(mock.asTool());
    try std.testing.expectEqual(@as(usize, 1), registry.count());

    const found = registry.get("my_tool");
    try std.testing.expect(found != null);
    try std.testing.expectEqualStrings("my_tool", found.?.getName());

    const not_found = registry.get("nonexistent");
    try std.testing.expect(not_found == null);
}

test "ToolRegistry.register rejects duplicate names" {
    const allocator = std.testing.allocator;
    var mock1 = MockTool{
        .tool_name = "dup_tool",
        .tool_description = "First",
        .tool_schema = "{}",
        .execute_output = "1",
    };
    var mock2 = MockTool{
        .tool_name = "dup_tool",
        .tool_description = "Second",
        .tool_schema = "{}",
        .execute_output = "2",
    };

    var registry = ToolRegistry.init(allocator);
    defer registry.deinit();

    try registry.register(mock1.asTool());
    try std.testing.expectError(ToolRegistryError.ToolAlreadyRegistered, registry.register(mock2.asTool()));
    try std.testing.expectEqual(@as(usize, 1), registry.count());
}

test "ToolRegistry.execute dispatches by name" {
    const allocator = std.testing.allocator;
    var mock = MockTool{
        .tool_name = "exec_reg",
        .tool_description = "Exec test",
        .tool_schema = "{}",
        .execute_output = "executed",
    };

    var registry = ToolRegistry.init(allocator);
    defer registry.deinit();

    try registry.register(mock.asTool());

    var result = try registry.execute(allocator, "exec_reg", "{\"x\":1}");
    defer result.deinit(allocator);

    try std.testing.expectEqualStrings("executed", result.output);
    try std.testing.expectEqual(@as(usize, 1), mock.execute_count);
}

test "ToolRegistry.execute returns ToolNotFound for unknown tool" {
    const allocator = std.testing.allocator;
    var registry = ToolRegistry.init(allocator);
    defer registry.deinit();

    try std.testing.expectError(
        ToolRegistryError.ToolNotFound,
        registry.execute(allocator, "ghost", "{}"),
    );
}

test "ToolRegistry.deinit calls tool deinit" {
    const allocator = std.testing.allocator;
    var mock = MockTool{
        .tool_name = "cleanup_tool",
        .tool_description = "Cleanup test",
        .tool_schema = "{}",
        .execute_output = "",
    };

    var registry = ToolRegistry.init(allocator);
    try registry.register(mock.asTool());
    registry.deinit();

    try std.testing.expect(mock.deinit_called);
}

test "ToolRegistry.getToolDefinitionsJson produces valid JSON" {
    const allocator = std.testing.allocator;
    var mock = MockTool{
        .tool_name = "get_weather",
        .tool_description = "Get weather for a location",
        .tool_schema =
            \\{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}
        ,
        .execute_output = "",
    };

    var registry = ToolRegistry.init(allocator);
    defer registry.deinit();

    try registry.register(mock.asTool());

    const json = try registry.getToolDefinitionsJson(allocator);
    defer allocator.free(json);

    // Parse to verify valid JSON
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json, .{});
    defer parsed.deinit();

    // Should be an array with one element
    try std.testing.expect(parsed.value == .array);
    try std.testing.expectEqual(@as(usize, 1), parsed.value.array.items.len);

    // Check structure
    const tool_obj = parsed.value.array.items[0].object;
    try std.testing.expectEqualStrings("function", tool_obj.get("type").?.string);

    const func_obj = tool_obj.get("function").?.object;
    try std.testing.expectEqualStrings("get_weather", func_obj.get("name").?.string);
    try std.testing.expectEqualStrings("Get weather for a location", func_obj.get("description").?.string);

    // Parameters should be a valid object
    const params = func_obj.get("parameters").?.object;
    try std.testing.expectEqualStrings("object", params.get("type").?.string);
}

test "ToolRegistry.getToolDefinitionsJson empty registry" {
    const allocator = std.testing.allocator;
    var registry = ToolRegistry.init(allocator);
    defer registry.deinit();

    const json = try registry.getToolDefinitionsJson(allocator);
    defer allocator.free(json);

    try std.testing.expectEqualStrings("[]", json);
}

test "ToolRegistry.getToolDefinitionsJson handles empty schema" {
    const allocator = std.testing.allocator;
    var mock = MockTool{
        .tool_name = "no_params",
        .tool_description = "Takes no parameters",
        .tool_schema = "",
        .execute_output = "",
    };

    var registry = ToolRegistry.init(allocator);
    defer registry.deinit();

    try registry.register(mock.asTool());

    const json = try registry.getToolDefinitionsJson(allocator);
    defer allocator.free(json);

    // Should produce valid JSON with default empty object schema
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json, .{});
    defer parsed.deinit();

    const func_obj = parsed.value.array.items[0].object.get("function").?.object;
    const params = func_obj.get("parameters").?.object;
    try std.testing.expectEqualStrings("object", params.get("type").?.string);
}

test "writeJsonEscaped escapes special characters" {
    const allocator = std.testing.allocator;
    var buf = std.ArrayListUnmanaged(u8){};
    defer buf.deinit(allocator);

    try writeJsonEscaped(buf.writer(allocator), "hello \"world\"\nnew\\line");
    try std.testing.expectEqualStrings("hello \\\"world\\\"\\nnew\\\\line", buf.items);
}

test "ToolResult.deinit frees output" {
    const allocator = std.testing.allocator;
    var result = ToolResult{
        .output = try allocator.dupe(u8, "test output"),
        .is_error = false,
    };
    result.deinit(allocator);
    // std.testing.allocator will detect leak if deinit didn't free
}
