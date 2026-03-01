//! Agent built-in tools.
//!
//! This module provides a minimal built-in toolset for agentic execution:
//! - File tools: `read_file`, `write_file`, `edit_file`
//! - HTTP tool: `http_fetch`
//! - Shell tool: `execute_command` (opt-in)

const std = @import("std");
const Allocator = std.mem.Allocator;
const tool_mod = @import("../tool.zig");

pub const file = @import("file_tools.zig");
pub const http = @import("http_tool.zig");
pub const exec = @import("exec_tool.zig");

pub const BuiltinToolsConfig = struct {
    /// Root directory for file tool sandboxing.
    workspace_dir: []const u8,
    /// Maximum bytes read by `read_file` / `edit_file`.
    file_max_read_bytes: usize = 256 * 1024,
    /// Maximum bytes returned by `http_fetch`.
    http_max_response_bytes: usize = 1024 * 1024,
    /// Register the `execute_command` shell tool. Not all contexts need
    /// shell access, so this is opt-in.
    enable_shell: bool = false,
};

/// Register built-in tools into the provided registry.
///
/// Ownership of all tool instances transfers to the registry on success.
pub fn registerDefaultTools(
    allocator: Allocator,
    registry: *tool_mod.ToolRegistry,
    config: BuiltinToolsConfig,
) !void {
    if (config.workspace_dir.len == 0) return error.InvalidPath;

    const read_tool = try file.FileReadTool.init(
        allocator,
        config.workspace_dir,
        config.file_max_read_bytes,
    );
    try registerOwnedTool(registry, read_tool.asTool());

    const write_tool = try file.FileWriteTool.init(allocator, config.workspace_dir);
    try registerOwnedTool(registry, write_tool.asTool());

    const edit_tool = try file.FileEditTool.init(
        allocator,
        config.workspace_dir,
        config.file_max_read_bytes,
    );
    try registerOwnedTool(registry, edit_tool.asTool());

    const http_tool = try http.HttpFetchTool.init(
        allocator,
        config.http_max_response_bytes,
    );
    try registerOwnedTool(registry, http_tool.asTool());

    if (config.enable_shell) {
        const exec_tool = try exec.ExecCommandTool.init(allocator);
        try registerOwnedTool(registry, exec_tool.asTool());
    }
}

fn registerOwnedTool(registry: *tool_mod.ToolRegistry, tool: tool_mod.Tool) !void {
    registry.register(tool) catch |err| {
        tool.deinit();
        return err;
    };
}

test "registerDefaultTools registers file and http tools" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const workspace = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(workspace);

    var registry = tool_mod.ToolRegistry.init(allocator);
    defer registry.deinit();

    try registerDefaultTools(allocator, &registry, .{
        .workspace_dir = workspace,
    });

    try std.testing.expectEqual(@as(usize, 4), registry.count());
    try std.testing.expect(registry.get("read_file") != null);
    try std.testing.expect(registry.get("write_file") != null);
    try std.testing.expect(registry.get("edit_file") != null);
    try std.testing.expect(registry.get("http_fetch") != null);
}
