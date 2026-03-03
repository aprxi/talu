//! Linux PTY helpers for interactive shell sessions.

const std = @import("std");
const builtin = @import("builtin");
const sandbox = @import("../sandbox/root.zig");
const helpers = @import("../sandbox/helpers.zig");
const cgroups = @import("../sandbox/cgroups.zig");

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

pub const SandboxConfig = struct {
    mode: sandbox.RuntimeMode = .host,
    backend: sandbox.Backend = .linux_local,
    exec_profile: ?*const sandbox.profile.ExecProfile = null,
    cgroup_config: ?sandbox.cgroups.CgroupConfig = null,

    /// Build the StrictRuntimeConfig with full isolation defaults for
    /// strict mode. `cwd` is the workspace directory to bind-mount.
    pub fn toStrictConfig(self: SandboxConfig, cwd: ?[]const u8) sandbox.StrictRuntimeConfig {
        if (self.mode == .strict) {
            var config = sandbox.StrictRuntimeConfig.defaultStrict(self.backend, cwd);
            config.exec_profile = self.exec_profile;
            if (self.cgroup_config) |cg| config.cgroup_config = cg;
            return config;
        }
        return .{
            .mode = self.mode,
            .backend = self.backend,
            .exec_profile = self.exec_profile,
        };
    }
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
pub fn spawnShell(
    cols: u16,
    rows: u16,
    cwd: ?[]const u8,
    mode: ShellSpawnMode,
    sandbox_config: SandboxConfig,
) !SpawnResult {
    var winsize = std.mem.zeroes(c.struct_winsize);
    winsize.ws_col = cols;
    winsize.ws_row = rows;

    var bootstrap: ?sandbox.launcher.BootstrapPipe = null;
    if (sandbox_config.mode == .strict) {
        bootstrap = try sandbox.launcher.openBootstrapPipe();
    }
    errdefer if (bootstrap) |*pipe| {
        pipe.closeRead();
        pipe.closeWrite();
    };

    var master_fd: c_int = -1;
    const pid_raw = c.forkpty(&master_fd, null, null, &winsize);
    if (pid_raw < 0) return error.OpenPtyFailed;

    if (pid_raw == 0) {
        if (bootstrap) |*pipe| pipe.closeChildRead();

        if (cwd) |dir| {
            std.posix.chdir(dir) catch |err| {
                if (bootstrap) |pipe| sandbox.launcher.childReportFailure(pipe.write_fd, err);
                std.posix.exit(1);
            };
        }
        sandbox.applyInChild(sandbox_config.toStrictConfig(cwd)) catch |err| {
            if (bootstrap) |pipe| sandbox.launcher.childReportFailure(pipe.write_fd, err);
            std.posix.exit(1);
        };

        if (sandbox_config.mode == .strict) {
            if (sandbox_config.exec_profile) |prof| {
                if (helpers.preferredStrictShellPath(prof)) |path| {
                    execInteractiveShellSlice(path);
                }
            }
        }

        if (mode == .host) {
            if (resolveInteractiveShellFromEnv()) |shell_path| {
                execInteractiveShell(shell_path);
            }

            execInteractiveShell("/bin/bash");
            execInteractiveShell("/bin/zsh");
        }
        execInteractiveShell("/bin/sh");
        if (bootstrap) |pipe| sandbox.launcher.childReportExecFailure(pipe.write_fd);
        std.posix.exit(127);
        unreachable;
    }

    // Parent-side cgroup setup: move child into cgroup before it execs.
    // Best-effort in 02e-1: silently ignored when host lacks cgroup
    // delegation. The server capability report surfaces cgroupv2_writable
    // at startup so operators can diagnose missing limits.
    if (sandbox_config.cgroup_config) |cg_config| {
        cgroups.createForPid(@intCast(pid_raw), cg_config) catch {};
    }

    if (bootstrap) |*pipe| {
        pipe.closeParentWrite();
        sandbox.launcher.waitForExecBoundary(pipe.read_fd) catch |err| {
            if (master_fd >= 0) std.posix.close(@as(std.posix.fd_t, @intCast(master_fd)));
            _ = std.posix.waitpid(@as(std.posix.pid_t, @intCast(pid_raw)), 0);
            return err;
        };
        pipe.closeRead();
    }

    const fd: std.posix.fd_t = @intCast(master_fd);
    var pty_val = Pty{ .master_fd = fd };
    errdefer pty_val.close();
    try helpers.setNonBlocking(fd);
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

fn execInteractiveShellSlice(shell_path: []const u8) void {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const shell_z = helpers.toPathZ(shell_path, &buf) catch return;
    execInteractiveShell(shell_z.ptr);
}

test "isValidShellPath accepts absolute paths only" {
    try std.testing.expect(isValidShellPath("/bin/bash"));
    try std.testing.expect(!isValidShellPath(""));
    try std.testing.expect(!isValidShellPath("bash"));
    try std.testing.expect(!isValidShellPath("./bin/zsh"));
}

test "spawnShell strict without exec profile fails setup" {
    if (builtin.os.tag != .linux) return;
    try std.testing.expectError(
        error.StrictSetupFailed,
        spawnShell(
            80,
            24,
            null,
            .host,
            .{
                .mode = .strict,
                .backend = .linux_local,
                .exec_profile = null,
            },
        ),
    );
}
