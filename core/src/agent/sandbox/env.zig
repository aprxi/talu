//! Environment variable sanitization for strict sandbox sessions.
//!
//! In strict mode, the child process receives a curated environment, not
//! the host's inherited variables. This prevents credential leakage
//! (SSH_AUTH_SOCK, AWS_*, GITHUB_TOKEN, etc.) and ensures PATH only
//! references directories present in the sandbox mount tree.
//!
//! All operations are fail-closed: if clearenv or setenv fails, the
//! function returns an error to abort sandbox setup.

const std = @import("std");
const builtin = @import("builtin");

const c = @cImport({
    @cInclude("stdlib.h");
});

pub const EnvConfig = struct {
    /// Override the default PATH.
    path: ?[]const u8 = null,
    /// Override the SHELL variable (e.g., resolved from ExecProfile).
    shell: ?[]const u8 = null,
    /// Override HOME (defaults to /workspace).
    home: ?[]const u8 = null,
};

/// Replace the process environment with curated strict-mode defaults.
///
/// Must be called in the forked child after mount setup and before exec.
/// After this call, only the curated variables are present. Returns error
/// if any libc call fails — the sandbox must not start with a partial env.
pub fn sanitize(config: EnvConfig) !void {
    // Clear all inherited environment variables.
    if (c.clearenv() != 0) return error.StrictSetupFailed;

    // Set curated defaults.
    try setenvSlice("PATH", config.path orelse "/usr/local/bin:/usr/bin:/bin");
    try setenvSlice("HOME", config.home orelse "/workspace");
    try setenvSlice("TERM", "xterm-256color");
    try setenvSlice("LANG", "C.UTF-8");
    try setenvSlice("SHELL", config.shell orelse "/bin/sh");
    try setenvSlice("USER", "sandbox");
}

fn setenvSlice(key: [*:0]const u8, value: []const u8) !void {
    var buf: [4096]u8 = undefined;
    if (value.len >= buf.len) return error.StrictSetupFailed;
    @memcpy(buf[0..value.len], value);
    buf[value.len] = 0;
    const value_z: [*:0]const u8 = buf[0..value.len :0];
    if (c.setenv(key, value_z, 1) != 0) return error.StrictSetupFailed;
}

test "sanitize sets PATH" {
    if (builtin.os.tag != .linux and builtin.os.tag != .macos) return;

    // Fork to avoid polluting the test process environment.
    const pid = std.posix.fork() catch return;
    if (pid == 0) {
        sanitize(.{}) catch std.posix.exit(99);
        const path = std.posix.getenv("PATH");
        if (path == null) std.posix.exit(1);
        const slice = std.mem.sliceTo(path.?, 0);
        if (!std.mem.eql(u8, slice, "/usr/local/bin:/usr/bin:/bin")) std.posix.exit(2);

        // Verify sensitive vars are cleared.
        if (std.posix.getenv("SSH_AUTH_SOCK") != null) std.posix.exit(3);

        // Verify USER is set.
        const user = std.posix.getenv("USER");
        if (user == null) std.posix.exit(4);
        const user_slice = std.mem.sliceTo(user.?, 0);
        if (!std.mem.eql(u8, user_slice, "sandbox")) std.posix.exit(5);

        std.posix.exit(0);
    }
    const result = std.posix.waitpid(pid, 0);
    try std.testing.expect(std.c.W.IFEXITED(result.status));
    try std.testing.expectEqual(@as(u32, 0), std.c.W.EXITSTATUS(result.status));
}

test "sanitize respects shell override" {
    if (builtin.os.tag != .linux and builtin.os.tag != .macos) return;

    const pid = std.posix.fork() catch return;
    if (pid == 0) {
        sanitize(.{ .shell = "/bin/bash" }) catch std.posix.exit(99);
        const shell = std.posix.getenv("SHELL");
        if (shell == null) std.posix.exit(1);
        const slice = std.mem.sliceTo(shell.?, 0);
        if (!std.mem.eql(u8, slice, "/bin/bash")) std.posix.exit(2);
        std.posix.exit(0);
    }
    const result = std.posix.waitpid(pid, 0);
    try std.testing.expect(std.c.W.IFEXITED(result.status));
    try std.testing.expectEqual(@as(u32, 0), std.c.W.EXITSTATUS(result.status));
}
