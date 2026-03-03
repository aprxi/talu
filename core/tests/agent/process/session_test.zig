//! Integration tests for `agent.process.session`.

const std = @import("std");
const builtin = @import("builtin");
const main = @import("main");
const process = main.agent.process;
const policy = main.agent.policy;
const sandbox = main.agent.sandbox;

test "ProcessSession open write read exit" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const cwd = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(cwd);

    var session = try process.session.ProcessSession.open(
        allocator,
        "while IFS= read -r line; do echo \"$line\"; [ \"$line\" = \"quit\" ] && break; done",
        cwd,
        .{},
    );
    defer session.close();

    _ = try session.write("hello\n");
    _ = try session.write("quit\n");

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

fn collectOutput(session: *process.session.ProcessSession, allocator: std.mem.Allocator) ![]u8 {
    var read_buf: [2048]u8 = undefined;
    var output = std.ArrayList(u8).empty;
    errdefer output.deinit(allocator);

    var guard: usize = 0;
    while (try session.isAlive() and guard < 5_000_000) : (guard += 1) {
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

test "ProcessSession strict mode blocks descendant executable not in allowlist" {
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
        \\    {"effect":"allow","action":"tool.process","command":"sh *"}
        \\  ]
        \\}
    ;
    var p = try policy.parsePolicy(allocator, json);
    defer p.deinit();
    var exec_profile = try sandbox.profile.buildExecProfile(allocator, &p, .{
        .action = "tool.process",
        .cwd = cwd,
        .include_shell_paths = true,
    });
    defer exec_profile.deinit();

    var session = process.session.ProcessSession.open(
        allocator,
        "sh -c 'ls >/dev/null 2>&1; echo CHILD:$?'",
        cwd,
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

    const output = try collectOutput(session, allocator);
    defer allocator.free(output);
    try std.testing.expect(std.mem.indexOf(u8, output, "CHILD:") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "CHILD:0") == null);
}

test "ProcessSession strict mode blocks file write outside allowed paths" {
    if (builtin.os.tag != .linux) return;
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.makePath("allowed");
    const cwd = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(cwd);

    const json =
        \\{
        \\  "default":"deny",
        \\  "statements":[
        \\    {"effect":"allow","action":"tool.process","command":"sh *"},
        \\    {"effect":"allow","action":"tool.fs.write","resource":"allowed/**"}
        \\  ]
        \\}
    ;
    var p = try policy.parsePolicy(allocator, json);
    defer p.deinit();
    var exec_profile = try sandbox.profile.buildExecProfile(allocator, &p, .{
        .action = "tool.process",
        .cwd = cwd,
        .include_shell_paths = true,
    });
    defer exec_profile.deinit();

    var session = process.session.ProcessSession.open(
        allocator,
        "echo blocked > denied.txt",
        cwd,
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

    const output = try collectOutput(session, allocator);
    defer allocator.free(output);
    try std.testing.expectError(error.FileNotFound, tmp.dir.statFile("denied.txt"));
}
