//! Integration tests for `agent.fs.path`.

const std = @import("std");
const main = @import("main");
const fs = main.agent.fs;

test "resolveExistingPath resolves workspace-relative file" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(.{ .sub_path = "data.txt", .data = "x" });

    const workspace = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(workspace);

    const resolved = try fs.path.resolveExistingPath(allocator, workspace, "data.txt");
    defer allocator.free(resolved);

    try std.testing.expect(fs.path.isWithinWorkspace(workspace, resolved));
}
