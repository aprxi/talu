//! Integration tests for `agent.shell.exec`.

const std = @import("std");
const main = @import("main");
const shell = main.agent.shell;

test "execWithOptions captures output in configured cwd" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(.{ .sub_path = "probe.txt", .data = "ok" });

    const cwd = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(cwd);

    var result = try shell.exec.execWithOptions(
        allocator,
        "ls probe.txt",
        .{ .cwd = cwd },
    );
    defer result.deinit(allocator);

    try std.testing.expect(std.mem.indexOf(u8, result.stdout, "probe.txt") != null);
    try std.testing.expectEqual(@as(?i32, 0), result.exit_code);
}
