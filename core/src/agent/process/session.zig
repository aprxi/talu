//! Persistent non-PTY process session.

const std = @import("std");
const Allocator = std.mem.Allocator;
const signal = @import("../shell/signal.zig");

const c = @cImport({
    @cInclude("fcntl.h");
});

pub const ProcessSession = struct {
    allocator: Allocator,
    child_pid: std.posix.pid_t,
    stdin_file: ?std.fs.File,
    stdout_file: ?std.fs.File,
    stderr_file: ?std.fs.File,
    cwd: []u8,
    created_at: i64,
    last_access: i64,
    exit_code: ?i32 = null,
    closed: bool = false,

    pub fn open(
        allocator: Allocator,
        command: []const u8,
        cwd: ?[]const u8,
    ) !*ProcessSession {
        if (command.len == 0) return error.InvalidArgument;

        const resolved_cwd = try resolveCwd(allocator, cwd);
        errdefer allocator.free(resolved_cwd);

        const command_z = try allocator.dupeZ(u8, command);
        defer allocator.free(command_z);

        const argv = [_][]const u8{ "/bin/sh", "-c", command_z };
        var child = std.process.Child.init(&argv, allocator);
        child.cwd = resolved_cwd;
        child.stdin_behavior = .Pipe;
        child.stdout_behavior = .Pipe;
        child.stderr_behavior = .Pipe;
        try child.spawn();

        var stdin_file = child.stdin orelse return error.ExecFailed;
        errdefer stdin_file.close();
        var stdout_file = child.stdout orelse return error.ExecFailed;
        errdefer stdout_file.close();
        var stderr_file = child.stderr orelse return error.ExecFailed;
        errdefer stderr_file.close();

        try setNonBlocking(stdout_file.handle);
        try setNonBlocking(stderr_file.handle);

        const now = std.time.timestamp();
        const self = try allocator.create(ProcessSession);
        self.* = .{
            .allocator = allocator,
            .child_pid = @intCast(child.id),
            .stdin_file = stdin_file,
            .stdout_file = stdout_file,
            .stderr_file = stderr_file,
            .cwd = resolved_cwd,
            .created_at = now,
            .last_access = now,
            .exit_code = null,
            .closed = false,
        };
        return self;
    }

    /// Close the session. Idempotent — safe to call multiple times.
    /// Sends SIGTERM, polls for exit up to 100ms, then SIGKILL if needed.
    pub fn close(self: *ProcessSession) void {
        if (self.closed) return;
        self.closed = true;

        if (self.stdin_file) |*f| {
            f.close();
            self.stdin_file = null;
        }
        if (self.stdout_file) |*f| {
            f.close();
            self.stdout_file = null;
        }
        if (self.stderr_file) |*f| {
            f.close();
            self.stderr_file = null;
        }

        if (self.exit_code == null) {
            signal.send(self.child_pid, std.posix.SIG.TERM) catch {};

            var waited = false;
            for (0..10) |_| {
                const result = std.posix.waitpid(self.child_pid, std.c.W.NOHANG);
                if (result.pid != 0) {
                    self.recordExit(result.status);
                    waited = true;
                    break;
                }
                std.Thread.sleep(10 * std.time.ns_per_ms);
            }

            if (!waited) {
                signal.send(self.child_pid, std.posix.SIG.KILL) catch {};
                const result = std.posix.waitpid(self.child_pid, 0);
                self.recordExit(result.status);
            }
        }

        self.allocator.free(self.cwd);
        self.allocator.destroy(self);
    }

    pub fn write(self: *ProcessSession, data: []const u8) !usize {
        if (self.closed) return error.SessionClosed;
        self.last_access = std.time.timestamp();
        if (!self.pollAlive()) return error.ProcessExited;
        if (data.len == 0) return 0;

        const stdin_file = self.stdin_file orelse return error.ProcessExited;
        var written: usize = 0;
        while (written < data.len) {
            const n = std.posix.write(stdin_file.handle, data[written..]) catch |err| switch (err) {
                error.BrokenPipe => {
                    _ = self.pollAlive();
                    return error.ProcessExited;
                },
                else => return err,
            };
            if (n == 0) break;
            written += n;
        }
        return written;
    }

    pub fn read(self: *ProcessSession, buf: []u8) !usize {
        if (self.closed) return error.SessionClosed;
        self.last_access = std.time.timestamp();
        if (buf.len == 0) return 0;

        const stdout_n = try readPipe(self.stdout_file, buf);
        if (stdout_n > 0) return stdout_n;

        const stderr_n = try readPipe(self.stderr_file, buf);
        if (stderr_n > 0) return stderr_n;

        _ = self.pollAlive();
        return 0;
    }

    pub fn sendSignal(self: *ProcessSession, sig: u8) !void {
        if (self.closed) return error.SessionClosed;
        self.last_access = std.time.timestamp();
        if (!self.pollAlive()) return error.ProcessExited;
        try signal.send(self.child_pid, sig);
    }

    pub fn isAlive(self: *ProcessSession) !bool {
        if (self.closed) return error.SessionClosed;
        return self.pollAlive();
    }

    pub fn getExitCode(self: *ProcessSession) !?i32 {
        if (self.closed) return error.SessionClosed;
        _ = self.pollAlive();
        return self.exit_code;
    }

    fn pollAlive(self: *ProcessSession) bool {
        if (self.exit_code != null) return false;
        const result = std.posix.waitpid(self.child_pid, std.c.W.NOHANG);
        if (result.pid == 0) return true;
        self.recordExit(result.status);
        return false;
    }

    fn recordExit(self: *ProcessSession, status: u32) void {
        if (std.c.W.IFEXITED(status)) {
            self.exit_code = @as(i32, @intCast(std.c.W.EXITSTATUS(status)));
            return;
        }
        if (std.c.W.IFSIGNALED(status)) {
            self.exit_code = -@as(i32, @intCast(std.c.W.TERMSIG(status)));
            return;
        }
        self.exit_code = -1;
    }
};

fn readPipe(file_opt: ?std.fs.File, buf: []u8) !usize {
    const file = file_opt orelse return 0;
    return file.read(buf) catch |err| switch (err) {
        error.WouldBlock => 0,
        error.InputOutput => 0,
        else => err,
    };
}

fn setNonBlocking(fd: std.posix.fd_t) !void {
    const no_flags: i32 = 0;
    const current = c.fcntl(fd, c.F_GETFL, no_flags);
    if (current < 0) return error.FcntlFailed;
    if (c.fcntl(fd, c.F_SETFL, current | c.O_NONBLOCK) < 0) {
        return error.FcntlFailed;
    }
}

fn resolveCwd(allocator: Allocator, cwd: ?[]const u8) ![]u8 {
    if (cwd) |value| {
        if (value.len == 0) return error.InvalidPath;
        return std.fs.cwd().realpathAlloc(allocator, value);
    }
    return std.fs.cwd().realpathAlloc(allocator, ".");
}

test "ProcessSession open write read close roundtrip" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const cwd = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(cwd);

    var session = try ProcessSession.open(
        allocator,
        "while IFS= read -r line; do echo \"$line\"; [ \"$line\" = \"quit\" ] && break; done",
        cwd,
    );
    defer session.close();

    _ = try session.write("hello\n");
    _ = try session.write("quit\n");

    var buffer: [512]u8 = undefined;
    var output = std.ArrayList(u8).empty;
    defer output.deinit(allocator);

    var guard: usize = 0;
    while (try session.isAlive() and guard < 50_000) : (guard += 1) {
        const n = try session.read(&buffer);
        if (n > 0) try output.appendSlice(allocator, buffer[0..n]);
    }
    while (true) {
        const n = try session.read(&buffer);
        if (n == 0) break;
        try output.appendSlice(allocator, buffer[0..n]);
    }

    try std.testing.expect(std.mem.indexOf(u8, output.items, "hello") != null);
    try std.testing.expect((try session.getExitCode()) != null);
}
