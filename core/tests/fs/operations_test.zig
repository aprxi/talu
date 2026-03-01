//! Integration tests for `agent.fs.operations`.

const std = @import("std");
const main = @import("main");
const fs = main.agent.fs;

test "listDir returns recursive entries" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.makePath("a/b");
    try tmp.dir.writeFile(.{ .sub_path = "a/b/file.txt", .data = "ok" });
    const root = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(root);

    var list = try fs.operations.listDir(allocator, root, "*.txt", true, 100);
    defer list.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), list.entries.len);
    try std.testing.expectEqualStrings("file.txt", list.entries[0].name);
}
