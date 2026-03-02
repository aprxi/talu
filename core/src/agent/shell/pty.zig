//! Linux PTY helpers for interactive shell sessions.

const std = @import("std");
const builtin = @import("builtin");

const c = @cImport({
    if (builtin.os.tag == .macos) {
        @cInclude("util.h");
    } else {
        @cInclude("pty.h");
    }
    @cInclude("fcntl.h");
    @cInclude("sys/ioctl.h");
});

pub const SpawnResult = struct {
    pty: Pty,
    child_pid: std.posix.pid_t,
};

pub const ShellSpawnMode = enum(u8) {
    host = 0,
    builtin = 1,
};

pub const Pty = struct {
    master_fd: std.posix.fd_t,

    /// Close the master fd. Idempotent — safe to call multiple times.
    pub fn close(self: *Pty) void {
        if (self.master_fd == -1) return;
        std.posix.close(self.master_fd);
        self.master_fd = -1;
    }

    pub fn isClosed(self: *const Pty) bool {
        return self.master_fd == -1;
    }

    pub fn resize(self: *Pty, cols: u16, rows: u16) !void {
        var winsize = std.mem.zeroes(c.struct_winsize);
        winsize.ws_col = cols;
        winsize.ws_row = rows;
        if (c.ioctl(self.master_fd, c.TIOCSWINSZ, &winsize) != 0) {
            return error.IoctlFailed;
        }
    }

    pub fn read(self: *Pty, buf: []u8) !usize {
        return try std.posix.read(self.master_fd, buf);
    }

    pub fn write(self: *Pty, data: []const u8) !usize {
        return try std.posix.write(self.master_fd, data);
    }
};

/// Spawn an interactive shell attached to a PTY and return the parent-side master fd.
pub fn spawnShell(cols: u16, rows: u16, cwd: ?[]const u8, mode: ShellSpawnMode) !SpawnResult {
    var winsize = std.mem.zeroes(c.struct_winsize);
    winsize.ws_col = cols;
    winsize.ws_row = rows;

    var master_fd: c_int = -1;
    const pid_raw = c.forkpty(&master_fd, null, null, &winsize);
    if (pid_raw < 0) return error.OpenPtyFailed;

    if (pid_raw == 0) {
        if (cwd) |dir| {
            std.posix.chdir(dir) catch std.posix.exit(1);
        }

        if (mode == .host) {
            if (resolveInteractiveShellFromEnv()) |shell_path| {
                execInteractiveShell(shell_path);
            }

            execInteractiveShell("/bin/bash");
            execInteractiveShell("/bin/zsh");
        }
        execInteractiveShell("/bin/sh");
        std.posix.exit(127);
        unreachable;
    }

    const fd: std.posix.fd_t = @intCast(master_fd);
    var pty_val = Pty{ .master_fd = fd };
    errdefer pty_val.close();
    try setNonBlocking(fd);
    return .{
        .pty = pty_val,
        .child_pid = @intCast(pid_raw),
    };
}

fn resolveInteractiveShellFromEnv() ?[*:0]const u8 {
    const shell_ptr = std.posix.getenv("SHELL") orelse return null;
    const shell = std.mem.sliceTo(shell_ptr, 0);
    if (!isValidShellPath(shell)) return null;
    return shell_ptr;
}

fn isValidShellPath(path: []const u8) bool {
    if (path.len == 0) return false;
    return std.fs.path.isAbsolute(path);
}

fn execInteractiveShell(shell_path: [*:0]const u8) void {
    const argv = [_:null]?[*:0]const u8{ shell_path, "-i" };
    std.posix.execvpeZ(shell_path, &argv, std.c.environ) catch {};
}

fn setNonBlocking(fd: std.posix.fd_t) !void {
    const current = c.fcntl(fd, c.F_GETFL, @as(c_int, 0));
    if (current < 0) return error.FcntlFailed;
    if (c.fcntl(fd, c.F_SETFL, current | c.O_NONBLOCK) < 0) {
        return error.FcntlFailed;
    }
}

test "isValidShellPath accepts absolute paths only" {
    try std.testing.expect(isValidShellPath("/bin/bash"));
    try std.testing.expect(!isValidShellPath(""));
    try std.testing.expect(!isValidShellPath("bash"));
    try std.testing.expect(!isValidShellPath("./bin/zsh"));
}
