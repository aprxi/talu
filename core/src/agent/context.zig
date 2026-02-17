//! Context - System prompt builder for agent loops.
//!
//! Assembles a system prompt from an identity string, workspace info, and the
//! names/descriptions of available tools. The agent loop sets this as the Chat's
//! system prompt before generating.
//!
//! # Thread Safety
//!
//! Stateless functions only. No shared state.

const std = @import("std");
const Allocator = std.mem.Allocator;
const tool_mod = @import("tool.zig");
const ToolRegistry = tool_mod.ToolRegistry;

/// Configuration for system prompt construction.
pub const ContextConfig = struct {
    /// Agent identity / persona (e.g. "You are a helpful coding assistant.").
    identity: ?[]const u8 = null,
    /// Working directory path for workspace-aware tools.
    workspace_dir: ?[]const u8 = null,
    /// Additional instructions appended to the system prompt.
    extra_instructions: ?[]const u8 = null,
};

/// Build a system prompt incorporating identity, workspace, and available tools.
///
/// Concatenates:
///   1. Identity string (if set)
///   2. Workspace info (if set)
///   3. Tool names and descriptions from the registry
///   4. Extra instructions (if set)
///
/// Caller owns returned memory.
pub fn buildSystemPrompt(allocator: Allocator, registry: *const ToolRegistry, config: ContextConfig) ![]u8 {
    var buf = std.ArrayListUnmanaged(u8){};
    errdefer buf.deinit(allocator);
    const writer = buf.writer(allocator);

    // Identity
    if (config.identity) |identity| {
        try writer.writeAll(identity);
        try writer.writeByte('\n');
    }

    // Workspace
    if (config.workspace_dir) |dir| {
        try writer.writeAll("\nWorkspace: ");
        try writer.writeAll(dir);
        try writer.writeByte('\n');
    }

    // Available tools
    if (registry.count() > 0) {
        try writer.writeAll("\nAvailable tools:\n");
        var it = registry.tools.iterator();
        while (it.next()) |entry| {
            const tool = entry.value_ptr.*;
            try writer.writeAll("- ");
            try writer.writeAll(tool.getName());
            try writer.writeAll(": ");
            try writer.writeAll(tool.getDescription());
            try writer.writeByte('\n');
        }
    }

    // Extra instructions
    if (config.extra_instructions) |extra| {
        try writer.writeByte('\n');
        try writer.writeAll(extra);
        try writer.writeByte('\n');
    }

    return buf.toOwnedSlice(allocator);
}

// =============================================================================
// Tests
// =============================================================================

test "buildSystemPrompt includes identity" {
    const allocator = std.testing.allocator;
    var registry = ToolRegistry.init(allocator);
    defer registry.deinit();

    const prompt = try buildSystemPrompt(allocator, &registry, .{
        .identity = "You are a helpful assistant.",
    });
    defer allocator.free(prompt);

    try std.testing.expect(std.mem.indexOf(u8, prompt, "You are a helpful assistant.") != null);
}

test "buildSystemPrompt includes tool descriptions" {
    const allocator = std.testing.allocator;

    const MockTool = @import("tool.zig").Tool;
    var mock = struct {
        tool_name: []const u8 = "read_file",
        tool_description: []const u8 = "Read a file from disk",
        tool_schema: []const u8 = "{}",

        fn name(ctx: *anyopaque) []const u8 {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            return self.tool_name;
        }
        fn description(ctx: *anyopaque) []const u8 {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            return self.tool_description;
        }
        fn parametersSchema(ctx: *anyopaque) []const u8 {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            return self.tool_schema;
        }
        fn executeFn(_: *anyopaque, alloc: Allocator, _: []const u8) anyerror!tool_mod.ToolResult {
            return .{ .output = try alloc.dupe(u8, "ok"), .is_error = false };
        }
        fn deinitFn(_: *anyopaque) void {}

        const vtable = MockTool.VTable{
            .name = name,
            .description = description,
            .parametersSchema = parametersSchema,
            .execute = executeFn,
            .deinit = deinitFn,
        };
    }{};

    var registry = ToolRegistry.init(allocator);
    defer registry.deinit();
    try registry.register(MockTool.init(&mock, &@TypeOf(mock).vtable));

    const prompt = try buildSystemPrompt(allocator, &registry, .{
        .identity = "You are an agent.",
    });
    defer allocator.free(prompt);

    try std.testing.expect(std.mem.indexOf(u8, prompt, "Available tools:") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "read_file: Read a file from disk") != null);
}

test "buildSystemPrompt with no tools" {
    const allocator = std.testing.allocator;
    var registry = ToolRegistry.init(allocator);
    defer registry.deinit();

    const prompt = try buildSystemPrompt(allocator, &registry, .{
        .identity = "Hello",
    });
    defer allocator.free(prompt);

    // Should not contain "Available tools:" when no tools registered
    try std.testing.expect(std.mem.indexOf(u8, prompt, "Available tools:") == null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "Hello") != null);
}

test "buildSystemPrompt with workspace and extra instructions" {
    const allocator = std.testing.allocator;
    var registry = ToolRegistry.init(allocator);
    defer registry.deinit();

    const prompt = try buildSystemPrompt(allocator, &registry, .{
        .identity = "Agent",
        .workspace_dir = "/home/user/project",
        .extra_instructions = "Always explain your reasoning.",
    });
    defer allocator.free(prompt);

    try std.testing.expect(std.mem.indexOf(u8, prompt, "Workspace: /home/user/project") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "Always explain your reasoning.") != null);
}

test "buildSystemPrompt with all fields null" {
    const allocator = std.testing.allocator;
    var registry = ToolRegistry.init(allocator);
    defer registry.deinit();

    const prompt = try buildSystemPrompt(allocator, &registry, .{});
    defer allocator.free(prompt);

    try std.testing.expectEqual(@as(usize, 0), prompt.len);
}
