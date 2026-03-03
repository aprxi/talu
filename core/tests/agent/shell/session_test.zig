//! Integration tests for `agent.shell.session`.

const std = @import("std");
const main = @import("main");
const shell = main.agent.shell;

test "ShellSession open write read exit" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const cwd = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(cwd);

    var session = try shell.session.ShellSession.open(allocator, 80, 24, cwd, 64 * 1024, .host);
    defer session.close();

    _ = try session.write("echo hello\n");
    _ = try session.write("exit\n");

    var read_buf: [2048]u8 = undefined;
    var output = std.ArrayList(u8).empty;
    defer output.deinit(allocator);

    var guard: usize = 0;
    while (try session.isAlive() and guard < 50_000) : (guard += 1) {
        const n = try session.read(&read_buf);
        if (n > 0) try output.appendSlice(allocator, read_buf[0..n]);
    }

    while (true) {
        const n = try session.read(&read_buf);
        if (n == 0) break;
        try output.appendSlice(allocator, read_buf[0..n]);
    }

    try std.testing.expect(std.mem.indexOf(u8, output.items, "hello") != null);
    try std.testing.expect((try session.getExitCode()) != null);
}
