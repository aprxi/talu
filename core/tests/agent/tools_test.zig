//! Integration tests for agent built-in tools.

const std = @import("std");
const main = @import("main");

const agent = main.agent;
const ToolRegistry = agent.ToolRegistry;

test "agent.tools.registerDefaultTools registers and executes file tools" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const workspace = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(workspace);

    var registry = ToolRegistry.init(allocator);
    defer registry.deinit();

    try agent.tools.registerDefaultTools(allocator, &registry, .{
        .workspace_dir = workspace,
    });

    try std.testing.expect(registry.get("read_file") != null);
    try std.testing.expect(registry.get("write_file") != null);
    try std.testing.expect(registry.get("edit_file") != null);
    try std.testing.expect(registry.get("http_fetch") != null);

    var write_result = try registry.execute(
        allocator,
        "write_file",
        "{\"path\":\"note.txt\",\"content\":\"hello world\"}",
    );
    defer write_result.deinit(allocator);
    try std.testing.expect(!write_result.is_error);

    var read_result = try registry.execute(
        allocator,
        "read_file",
        "{\"path\":\"note.txt\"}",
    );
    defer read_result.deinit(allocator);
    try std.testing.expect(!read_result.is_error);
    try std.testing.expect(std.mem.indexOf(u8, read_result.output, "\"content\":\"hello world\"") != null);

    var edit_result = try registry.execute(
        allocator,
        "edit_file",
        "{\"path\":\"note.txt\",\"old_text\":\"world\",\"new_text\":\"agent\"}",
    );
    defer edit_result.deinit(allocator);
    try std.testing.expect(!edit_result.is_error);

    const contents = try tmp.dir.readFileAlloc(allocator, "note.txt", 64);
    defer allocator.free(contents);
    try std.testing.expectEqualStrings("hello agent", contents);
}
