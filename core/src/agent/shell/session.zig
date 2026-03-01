//! Persistent interactive shell session.

const std = @import("std");
const Allocator = std.mem.Allocator;
const pty_mod = @import("pty.zig");
const signal = @import("signal.zig");

pub const ShellSession = struct {
    allocator: Allocator,
    pty: pty_mod.Pty,
    child_pid: std.posix.pid_t,
    scrollback: std.ArrayListUnmanaged(u8),
    max_scrollback_bytes: usize,
    cwd: []u8,
    created_at: i64,
    last_access: i64,
    exit_code: ?i32 = null,
    closed: bool = false,

    pub fn open(
        allocator: Allocator,
        cols: u16,
        rows: u16,
        cwd: ?[]const u8,
        max_scrollback_bytes: usize,
    ) !*ShellSession {
        const resolved_cwd = try resolveCwd(allocator, cwd);
        errdefer allocator.free(resolved_cwd);

        var spawned = try pty_mod.spawnShell(cols, rows, resolved_cwd);
        errdefer spawned.pty.close();

        const now = std.time.timestamp();
        const self = try allocator.create(ShellSession);
        self.* = .{
            .allocator = allocator,
            .pty = spawned.pty,
            .child_pid = spawned.child_pid,
            .scrollback = .empty,
            .max_scrollback_bytes = max_scrollback_bytes,
            .cwd = resolved_cwd,
            .created_at = now,
            .last_access = now,
            .exit_code = null,
        };
        return self;
    }

    /// Close the session. Idempotent — safe to call multiple times.
    /// Sends SIGTERM, polls for exit up to 100ms, then SIGKILL if needed.
    pub fn close(self: *ShellSession) void {
        if (self.closed) return;
        self.closed = true;

        if (self.exit_code == null) {
            signal.send(self.child_pid, std.posix.SIG.TERM) catch {};

            // Poll with WNOHANG up to 100ms (10 × 10ms).
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

            // Escalate to SIGKILL if still alive.
            if (!waited) {
                signal.send(self.child_pid, std.posix.SIG.KILL) catch {};
                const result = std.posix.waitpid(self.child_pid, 0);
                self.recordExit(result.status);
            }
        }

        self.pty.close();
        self.scrollback.deinit(self.allocator);
        self.allocator.free(self.cwd);
        self.allocator.destroy(self);
    }

    pub fn write(self: *ShellSession, data: []const u8) !usize {
        if (self.closed) return error.SessionClosed;
        self.last_access = std.time.timestamp();
        if (!self.pollAlive()) return error.ProcessExited;
        return try self.pty.write(data);
    }

    pub fn read(self: *ShellSession, buf: []u8) !usize {
        if (self.closed) return error.SessionClosed;
        self.last_access = std.time.timestamp();
        if (buf.len == 0) return 0;

        const n = self.pty.read(buf) catch |err| switch (err) {
            error.WouldBlock => {
                _ = self.pollAlive();
                return 0;
            },
            error.InputOutput => {
                // PTY closed from peer side.
                _ = self.pollAlive();
                return 0;
            },
            else => return err,
        };

        if (n > 0) {
            try self.appendScrollback(buf[0..n]);
        } else {
            _ = self.pollAlive();
        }
        return n;
    }

    pub fn resize(self: *ShellSession, cols: u16, rows: u16) !void {
        if (self.closed) return error.SessionClosed;
        self.last_access = std.time.timestamp();
        if (!self.pollAlive()) return error.ProcessExited;
        try self.pty.resize(cols, rows);
    }

    pub fn sendSignal(self: *ShellSession, sig: u8) !void {
        if (self.closed) return error.SessionClosed;
        self.last_access = std.time.timestamp();
        if (!self.pollAlive()) return error.ProcessExited;
        try signal.send(self.child_pid, sig);
    }

    pub fn isAlive(self: *ShellSession) !bool {
        if (self.closed) return error.SessionClosed;
        return self.pollAlive();
    }

    pub fn scrollbackCopy(self: *ShellSession, allocator: Allocator) ![]u8 {
        if (self.closed) return error.SessionClosed;
        self.last_access = std.time.timestamp();
        return try allocator.dupe(u8, self.scrollback.items);
    }

    pub fn getExitCode(self: *ShellSession) !?i32 {
        if (self.closed) return error.SessionClosed;
        _ = self.pollAlive();
        return self.exit_code;
    }

    fn pollAlive(self: *ShellSession) bool {
        if (self.exit_code != null) return false;
        const result = std.posix.waitpid(self.child_pid, std.c.W.NOHANG);
        if (result.pid == 0) return true;
        self.recordExit(result.status);
        return false;
    }

    fn recordExit(self: *ShellSession, status: u32) void {
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

    fn appendScrollback(self: *ShellSession, chunk: []const u8) !void {
        if (self.max_scrollback_bytes == 0 or chunk.len == 0) return;

        if (chunk.len >= self.max_scrollback_bytes) {
            self.scrollback.clearRetainingCapacity();
            try self.scrollback.appendSlice(
                self.allocator,
                chunk[chunk.len - self.max_scrollback_bytes ..],
            );
            return;
        }

        const current_len = self.scrollback.items.len;
        if (current_len + chunk.len > self.max_scrollback_bytes) {
            const overflow = current_len + chunk.len - self.max_scrollback_bytes;
            std.mem.copyForwards(
                u8,
                self.scrollback.items[0 .. current_len - overflow],
                self.scrollback.items[overflow..current_len],
            );
            self.scrollback.items.len = current_len - overflow;
        }
        try self.scrollback.appendSlice(self.allocator, chunk);
    }
};

fn resolveCwd(allocator: Allocator, cwd: ?[]const u8) ![]u8 {
    if (cwd) |value| {
        if (value.len == 0) return error.InvalidPath;
        return std.fs.cwd().realpathAlloc(allocator, value);
    }
    return std.fs.cwd().realpathAlloc(allocator, ".");
}

test "ShellSession open write read close roundtrip" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const cwd = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(cwd);

    var session = try ShellSession.open(allocator, 80, 24, cwd, 64 * 1024);
    defer session.close();

    _ = try session.write("echo hello\n");
    _ = try session.write("exit\n");

    var buffer: [2048]u8 = undefined;
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

    const scrollback = try session.scrollbackCopy(allocator);
    defer allocator.free(scrollback);
    try std.testing.expect(std.mem.indexOf(u8, scrollback, "hello") != null);
}
