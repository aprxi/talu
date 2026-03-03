//! Integration tests for `agent.shell.session`.

const std = @import("std");
const builtin = @import("builtin");
const main = @import("main");
const shell = main.agent.shell;
const policy = main.agent.policy;
const sandbox = main.agent.sandbox;

test "ShellSession open write read exit" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const cwd = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(cwd);

    var session = try shell.session.ShellSession.open(
        allocator,
        80,
        24,
        cwd,
        64 * 1024,
        .host,
        .{},
    );
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

fn collectOutput(session: *shell.session.ShellSession, allocator: std.mem.Allocator) ![]u8 {
    var read_buf: [2048]u8 = undefined;
    var output = std.ArrayList(u8).empty;
    errdefer output.deinit(allocator);

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
    return try output.toOwnedSlice(allocator);
}

test "ShellSession strict mode allows whitelisted executable" {
    if (builtin.os.tag != .linux) return;
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(.{ .sub_path = "hello.txt", .data = "ok" });
    const cwd = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(cwd);

    const json =
        \\{
        \\  "default":"deny",
        \\  "statements":[
        \\    {"effect":"allow","action":"tool.shell","command":"ls *"}
        \\  ]
        \\}
    ;
    var p = try policy.parsePolicy(allocator, json);
    defer p.deinit();
    var exec_profile = try sandbox.profile.buildExecProfile(allocator, &p, .{
        .action = "tool.shell",
        .cwd = cwd,
        .include_shell_paths = true,
    });
    defer exec_profile.deinit();

    var session = shell.session.ShellSession.open(
        allocator,
        80,
        24,
        cwd,
        64 * 1024,
        .host,
        .{
            .mode = .strict,
            .backend = .linux_local,
            .exec_profile = &exec_profile,
        },
    ) catch |err| switch (err) {
        error.StrictUnavailable => return, // host lacks isolation support
        else => return err,
    };
    defer session.close();

    _ = try session.write("ls hello.txt\n");
    _ = try session.write("exit\n");

    const output = try collectOutput(session, allocator);
    defer allocator.free(output);
    try std.testing.expect(std.mem.indexOf(u8, output, "hello.txt") != null);
}

test "ShellSession strict mode blocks non-whitelisted executable in child shell" {
    if (builtin.os.tag != .linux) return;
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const cwd = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(cwd);

    const json =
        \\{
        \\  "default":"deny",
        \\  "statements":[
        \\    {"effect":"allow","action":"tool.shell","command":"echo *"}
        \\  ]
        \\}
    ;
    var p = try policy.parsePolicy(allocator, json);
    defer p.deinit();
    var exec_profile = try sandbox.profile.buildExecProfile(allocator, &p, .{
        .action = "tool.shell",
        .cwd = cwd,
        .include_shell_paths = true,
    });
    defer exec_profile.deinit();

    var session = shell.session.ShellSession.open(
        allocator,
        80,
        24,
        cwd,
        64 * 1024,
        .host,
        .{
            .mode = .strict,
            .backend = .linux_local,
            .exec_profile = &exec_profile,
        },
    ) catch |err| switch (err) {
        error.StrictUnavailable => return, // host lacks isolation support
        else => return err,
    };
    defer session.close();

    _ = try session.write("ls >/dev/null 2>&1\n");
    _ = try session.write("echo RC:$?\n");
    _ = try session.write("exit\n");

    const output = try collectOutput(session, allocator);
    defer allocator.free(output);
    try std.testing.expect(std.mem.indexOf(u8, output, "RC:") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "RC:0") == null);
}
