//! Integration tests for agent.buildSystemPrompt
//!
//! Tests the system prompt builder through its public interface. Exercises
//! identity, workspace, tools, and extra instructions composition.

const std = @import("std");
const main = @import("main");

const agent = main.agent;
const Tool = agent.Tool;
const ToolResult = agent.ToolResult;
const ToolRegistry = agent.ToolRegistry;
const ContextConfig = agent.ContextConfig;
const buildSystemPrompt = agent.buildSystemPrompt;

// =============================================================================
// Test helpers
// =============================================================================

const SimpleTool = struct {
    tool_name: []const u8,
    tool_description: []const u8,

    fn name(ctx: *anyopaque) []const u8 {
        const self: *SimpleTool = @ptrCast(@alignCast(ctx));
        return self.tool_name;
    }

    fn description(ctx: *anyopaque) []const u8 {
        const self: *SimpleTool = @ptrCast(@alignCast(ctx));
        return self.tool_description;
    }

    fn parametersSchema(_: *anyopaque) []const u8 {
        return "{}";
    }

    fn executeFn(_: *anyopaque, alloc: std.mem.Allocator, _: []const u8) anyerror!ToolResult {
        return .{ .output = try alloc.dupe(u8, ""), .is_error = false };
    }

    fn deinitFn(_: *anyopaque) void {}

    const vtable = Tool.VTable{
        .name = name,
        .description = description,
        .parametersSchema = parametersSchema,
        .execute = executeFn,
        .deinit = deinitFn,
    };

    fn asTool(self: *SimpleTool) Tool {
        return Tool.init(self, &vtable);
    }
};

// =============================================================================
// buildSystemPrompt
// =============================================================================

test "buildSystemPrompt with identity only" {
    const allocator = std.testing.allocator;
    var registry = ToolRegistry.init(allocator);
    defer registry.deinit();

    const prompt = try buildSystemPrompt(allocator, &registry, .{
        .identity = "You are a coding assistant.",
    });
    defer allocator.free(prompt);

    try std.testing.expect(std.mem.indexOf(u8, prompt, "You are a coding assistant.") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "Available tools:") == null);
}

test "buildSystemPrompt includes all registered tools" {
    const allocator = std.testing.allocator;

    var read_file = SimpleTool{ .tool_name = "read_file", .tool_description = "Read a file" };
    var shell = SimpleTool{ .tool_name = "shell", .tool_description = "Execute shell commands" };

    var registry = ToolRegistry.init(allocator);
    defer registry.deinit();

    try registry.register(read_file.asTool());
    try registry.register(shell.asTool());

    const prompt = try buildSystemPrompt(allocator, &registry, .{
        .identity = "Agent",
    });
    defer allocator.free(prompt);

    try std.testing.expect(std.mem.indexOf(u8, prompt, "Available tools:") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "read_file: Read a file") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "shell: Execute shell commands") != null);
}

test "buildSystemPrompt includes workspace directory" {
    const allocator = std.testing.allocator;
    var registry = ToolRegistry.init(allocator);
    defer registry.deinit();

    const prompt = try buildSystemPrompt(allocator, &registry, .{
        .workspace_dir = "/home/user/project",
    });
    defer allocator.free(prompt);

    try std.testing.expect(std.mem.indexOf(u8, prompt, "Workspace: /home/user/project") != null);
}

test "buildSystemPrompt includes extra instructions" {
    const allocator = std.testing.allocator;
    var registry = ToolRegistry.init(allocator);
    defer registry.deinit();

    const prompt = try buildSystemPrompt(allocator, &registry, .{
        .extra_instructions = "Always explain your reasoning step by step.",
    });
    defer allocator.free(prompt);

    try std.testing.expect(std.mem.indexOf(u8, prompt, "Always explain your reasoning step by step.") != null);
}

test "buildSystemPrompt composes all sections" {
    const allocator = std.testing.allocator;

    var tool1 = SimpleTool{ .tool_name = "search", .tool_description = "Search codebase" };

    var registry = ToolRegistry.init(allocator);
    defer registry.deinit();
    try registry.register(tool1.asTool());

    const prompt = try buildSystemPrompt(allocator, &registry, .{
        .identity = "You are an expert.",
        .workspace_dir = "/tmp/work",
        .extra_instructions = "Be concise.",
    });
    defer allocator.free(prompt);

    // Verify ordering: identity comes before tools, tools before extra
    const identity_pos = std.mem.indexOf(u8, prompt, "You are an expert.").?;
    const tools_pos = std.mem.indexOf(u8, prompt, "Available tools:").?;
    const extra_pos = std.mem.indexOf(u8, prompt, "Be concise.").?;

    try std.testing.expect(identity_pos < tools_pos);
    try std.testing.expect(tools_pos < extra_pos);
}

test "buildSystemPrompt returns empty for all-null config" {
    const allocator = std.testing.allocator;
    var registry = ToolRegistry.init(allocator);
    defer registry.deinit();

    const prompt = try buildSystemPrompt(allocator, &registry, .{});
    defer allocator.free(prompt);

    try std.testing.expectEqual(@as(usize, 0), prompt.len);
}
