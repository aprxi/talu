//! Integration tests for agent.ToolRegistry
//!
//! Tests the ToolRegistry through its public interface: init, register, get,
//! execute, getToolDefinitionsJson, count, and deinit. Exercises end-to-end
//! tool registration and execution flows.

const std = @import("std");
const main = @import("main");

const agent = main.agent;
const Tool = agent.Tool;
const ToolResult = agent.ToolResult;
const ToolRegistry = agent.ToolRegistry;
const ToolRegistryError = agent.ToolRegistryError;

// =============================================================================
// Test helpers
// =============================================================================

/// Mock tool that echoes arguments back as output.
const EchoTool = struct {
    tool_name: []const u8,
    call_count: usize = 0,

    fn name(ctx: *anyopaque) []const u8 {
        const self: *EchoTool = @ptrCast(@alignCast(ctx));
        return self.tool_name;
    }

    fn description(_: *anyopaque) []const u8 {
        return "Echoes the input arguments";
    }

    fn parametersSchema(_: *anyopaque) []const u8 {
        return
            \\{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}
        ;
    }

    fn executeFn(ctx: *anyopaque, alloc: std.mem.Allocator, args_json: []const u8) anyerror!ToolResult {
        const self: *EchoTool = @ptrCast(@alignCast(ctx));
        self.call_count += 1;
        return .{
            .output = try alloc.dupe(u8, args_json),
            .is_error = false,
        };
    }

    fn deinitFn(_: *anyopaque) void {}

    const vtable = Tool.VTable{
        .name = name,
        .description = description,
        .parametersSchema = parametersSchema,
        .execute = executeFn,
        .deinit = deinitFn,
    };

    fn asTool(self: *EchoTool) Tool {
        return Tool.init(self, &vtable);
    }
};

/// Mock tool that always returns an error result.
const ErrorTool = struct {
    fn name(_: *anyopaque) []const u8 {
        return "error_tool";
    }

    fn description(_: *anyopaque) []const u8 {
        return "Always fails";
    }

    fn parametersSchema(_: *anyopaque) []const u8 {
        return "{}";
    }

    fn executeFn(_: *anyopaque, alloc: std.mem.Allocator, _: []const u8) anyerror!ToolResult {
        return .{
            .output = try alloc.dupe(u8, "Something went wrong"),
            .is_error = true,
        };
    }

    fn deinitFn(_: *anyopaque) void {}

    const vtable = Tool.VTable{
        .name = name,
        .description = description,
        .parametersSchema = parametersSchema,
        .execute = executeFn,
        .deinit = deinitFn,
    };
};

// =============================================================================
// ToolRegistry.register
// =============================================================================

test "ToolRegistry.register adds tool and count increases" {
    const allocator = std.testing.allocator;
    var echo = EchoTool{ .tool_name = "echo" };

    var registry = ToolRegistry.init(allocator);
    defer registry.deinit();

    try std.testing.expectEqual(@as(usize, 0), registry.count());
    try registry.register(echo.asTool());
    try std.testing.expectEqual(@as(usize, 1), registry.count());
}

test "ToolRegistry.register rejects duplicate tool names" {
    const allocator = std.testing.allocator;
    var echo1 = EchoTool{ .tool_name = "dup" };
    var echo2 = EchoTool{ .tool_name = "dup" };

    var registry = ToolRegistry.init(allocator);
    defer registry.deinit();

    try registry.register(echo1.asTool());
    try std.testing.expectError(ToolRegistryError.ToolAlreadyRegistered, registry.register(echo2.asTool()));
}

// =============================================================================
// ToolRegistry.get
// =============================================================================

test "ToolRegistry.get returns registered tool" {
    const allocator = std.testing.allocator;
    var echo = EchoTool{ .tool_name = "finder" };

    var registry = ToolRegistry.init(allocator);
    defer registry.deinit();

    try registry.register(echo.asTool());

    const found = registry.get("finder");
    try std.testing.expect(found != null);
    try std.testing.expectEqualStrings("finder", found.?.getName());
}

test "ToolRegistry.get returns null for unknown tool" {
    const allocator = std.testing.allocator;
    var registry = ToolRegistry.init(allocator);
    defer registry.deinit();

    try std.testing.expect(registry.get("nonexistent") == null);
}

// =============================================================================
// ToolRegistry.execute
// =============================================================================

test "ToolRegistry.execute dispatches to correct tool" {
    const allocator = std.testing.allocator;
    var echo = EchoTool{ .tool_name = "exec_test" };

    var registry = ToolRegistry.init(allocator);
    defer registry.deinit();

    try registry.register(echo.asTool());

    const args = "{\"text\":\"hello\"}";
    var result = try registry.execute(allocator, "exec_test", args);
    defer result.deinit(allocator);

    try std.testing.expectEqualStrings(args, result.output);
    try std.testing.expect(!result.is_error);
    try std.testing.expectEqual(@as(usize, 1), echo.call_count);
}

test "ToolRegistry.execute returns ToolNotFound for missing tool" {
    const allocator = std.testing.allocator;
    var registry = ToolRegistry.init(allocator);
    defer registry.deinit();

    try std.testing.expectError(
        ToolRegistryError.ToolNotFound,
        registry.execute(allocator, "ghost", "{}"),
    );
}

test "ToolRegistry.execute handles error result from tool" {
    const allocator = std.testing.allocator;
    var err_tool: u8 = 0; // dummy state

    var registry = ToolRegistry.init(allocator);
    defer registry.deinit();

    try registry.register(Tool.init(&err_tool, &ErrorTool.vtable));

    var result = try registry.execute(allocator, "error_tool", "{}");
    defer result.deinit(allocator);

    try std.testing.expect(result.is_error);
    try std.testing.expectEqualStrings("Something went wrong", result.output);
}

// =============================================================================
// ToolRegistry.getToolDefinitionsJson
// =============================================================================

test "ToolRegistry.getToolDefinitionsJson produces valid OpenAI format" {
    const allocator = std.testing.allocator;
    var echo = EchoTool{ .tool_name = "get_weather" };

    var registry = ToolRegistry.init(allocator);
    defer registry.deinit();

    try registry.register(echo.asTool());

    const json = try registry.getToolDefinitionsJson(allocator);
    defer allocator.free(json);

    // Parse and validate structure
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json, .{});
    defer parsed.deinit();

    // Array of tool definitions
    try std.testing.expect(parsed.value == .array);
    try std.testing.expectEqual(@as(usize, 1), parsed.value.array.items.len);

    // Each element has type + function
    const elem = parsed.value.array.items[0].object;
    try std.testing.expectEqualStrings("function", elem.get("type").?.string);

    const func = elem.get("function").?.object;
    try std.testing.expectEqualStrings("get_weather", func.get("name").?.string);
    try std.testing.expect(func.get("description") != null);
    try std.testing.expect(func.get("parameters") != null);
}

test "ToolRegistry.getToolDefinitionsJson with multiple tools" {
    const allocator = std.testing.allocator;
    var tool1 = EchoTool{ .tool_name = "tool_a" };
    var tool2 = EchoTool{ .tool_name = "tool_b" };

    var registry = ToolRegistry.init(allocator);
    defer registry.deinit();

    try registry.register(tool1.asTool());
    try registry.register(tool2.asTool());

    const json = try registry.getToolDefinitionsJson(allocator);
    defer allocator.free(json);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json, .{});
    defer parsed.deinit();

    try std.testing.expectEqual(@as(usize, 2), parsed.value.array.items.len);
}

test "ToolRegistry.getToolDefinitionsJson empty registry produces empty array" {
    const allocator = std.testing.allocator;
    var registry = ToolRegistry.init(allocator);
    defer registry.deinit();

    const json = try registry.getToolDefinitionsJson(allocator);
    defer allocator.free(json);

    try std.testing.expectEqualStrings("[]", json);
}

// =============================================================================
// ToolRegistry.deinit
// =============================================================================

test "ToolRegistry.deinit calls deinit on all tools" {
    const allocator = std.testing.allocator;

    const DeinitTracker = struct {
        deinit_called: bool = false,

        fn name(_: *anyopaque) []const u8 {
            return "tracked";
        }
        fn description(_: *anyopaque) []const u8 {
            return "Tracks deinit";
        }
        fn parametersSchema(_: *anyopaque) []const u8 {
            return "{}";
        }
        fn executeFn(_: *anyopaque, alloc: std.mem.Allocator, _: []const u8) anyerror!ToolResult {
            return .{ .output = try alloc.dupe(u8, "ok"), .is_error = false };
        }
        fn deinitFn(ctx: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.deinit_called = true;
        }

        const vtable = Tool.VTable{
            .name = name,
            .description = description,
            .parametersSchema = parametersSchema,
            .execute = executeFn,
            .deinit = deinitFn,
        };
    };

    var tracker = DeinitTracker{};
    var registry = ToolRegistry.init(allocator);
    try registry.register(Tool.init(&tracker, &DeinitTracker.vtable));
    registry.deinit();

    try std.testing.expect(tracker.deinit_called);
}
