//! Shared helpers for sandbox spawn paths (PTY, exec, process session).

const std = @import("std");
const builtin = @import("builtin");
const profile = @import("profile.zig");

const c = @cImport({
    @cInclude("fcntl.h");
    if (builtin.os.tag == .linux) {
        @cInclude("sys/syscall.h");
        @cInclude("unistd.h");
    }
});

/// Select the best shell from an exec profile's allowed paths.
///
/// Prefers sh, bash, zsh in order. Falls back to the first allowed path.
/// Returns null only when the profile has no allowed paths.
pub fn preferredStrictShellPath(exec_profile: ?*const profile.ExecProfile) ?[]const u8 {
    const prof = exec_profile orelse return null;
    const preferred: []const []const u8 = &.{ "sh", "bash", "zsh" };
    for (preferred) |needle| {
        for (prof.paths) |path| {
            if (std.mem.eql(u8, std.fs.path.basename(path), needle)) return path;
        }
    }
    return if (prof.paths.len > 0) prof.paths[0] else null;
}

/// Copy a path slice into a null-terminated buffer.
pub fn toPathZ(path: []const u8, buf: *[std.fs.max_path_bytes]u8) ![:0]const u8 {
    if (path.len == 0 or path.len >= buf.len) return error.StrictSetupFailed;
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    return buf[0..path.len :0];
}

/// Set O_NONBLOCK on a file descriptor.
pub fn setNonBlocking(fd: std.posix.fd_t) !void {
    const current = c.fcntl(fd, c.F_GETFL, @as(c_int, 0));
    if (current < 0) return error.FcntlFailed;
    if (c.fcntl(fd, c.F_SETFL, current | c.O_NONBLOCK) < 0) {
        return error.FcntlFailed;
    }
}

/// Create a pipe whose endpoints are close-on-exec.
///
/// On Linux, this uses `pipe2(O_CLOEXEC)` to avoid descriptor-leak races in
/// multithreaded fork/exec paths. On other platforms it falls back to `pipe()`
/// plus `fcntl(FD_CLOEXEC)`.
pub fn pipeCloexec() ![2]std.posix.fd_t {
    if (builtin.os.tag == .linux and @hasDecl(c, "SYS_pipe2")) {
        var raw_fds: [2]c_int = undefined;
        if (c.syscall(c.SYS_pipe2, &raw_fds, @as(c_int, c.O_CLOEXEC)) == 0) {
            return .{
                @intCast(raw_fds[0]),
                @intCast(raw_fds[1]),
            };
        }
    }

    const fds = try std.posix.pipe();
    errdefer {
        std.posix.close(fds[0]);
        std.posix.close(fds[1]);
    }
    try setCloseOnExec(fds[0]);
    try setCloseOnExec(fds[1]);
    return fds;
}

fn setCloseOnExec(fd: std.posix.fd_t) !void {
    const current = c.fcntl(fd, c.F_GETFD, @as(c_int, 0));
    if (current < 0) return error.FcntlFailed;
    if (c.fcntl(fd, c.F_SETFD, current | c.FD_CLOEXEC) < 0) {
        return error.FcntlFailed;
    }
}

test "preferredStrictShellPath returns null for null profile" {
    try std.testing.expectEqual(@as(?[]const u8, null), preferredStrictShellPath(null));
}

test "toPathZ converts valid path" {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const result = try toPathZ("/bin/sh", &buf);
    try std.testing.expectEqualStrings("/bin/sh", result);
    try std.testing.expectEqual(@as(u8, 0), result.ptr[result.len]);
}

test "toPathZ rejects empty path" {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    try std.testing.expectError(error.StrictSetupFailed, toPathZ("", &buf));
}

test "pipeCloexec sets FD_CLOEXEC on both ends" {
    const fds = try pipeCloexec();
    defer std.posix.close(fds[0]);
    defer std.posix.close(fds[1]);

    const read_flags = c.fcntl(fds[0], c.F_GETFD, @as(c_int, 0));
    try std.testing.expect(read_flags >= 0);
    try std.testing.expect((read_flags & c.FD_CLOEXEC) != 0);

    const write_flags = c.fcntl(fds[1], c.F_GETFD, @as(c_int, 0));
    try std.testing.expect(write_flags >= 0);
    try std.testing.expect((write_flags & c.FD_CLOEXEC) != 0);
}
