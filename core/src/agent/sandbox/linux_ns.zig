//! Linux namespace setup for strict sandbox launches.
//!
//! This module is intentionally conservative for rootless operation. Namespace
//! changes are only attempted when explicitly enabled in config.

const std = @import("std");
const builtin = @import("builtin");

const c = if (builtin.os.tag == .linux) @cImport({
    @cInclude("errno.h");
    @cInclude("sys/syscall.h");
    @cInclude("unistd.h");
}) else struct {};

const CLONE_NEWNS: c_int = 0x00020000;
const CLONE_NEWUSER: c_int = 0x10000000;
const CLONE_NEWPID: c_int = 0x20000000;
const CLONE_NEWUTS: c_int = 0x04000000;
const CLONE_NEWIPC: c_int = 0x08000000;

pub const NamespaceOptions = struct {
    enable_mount_ns: bool = false,
    enable_pid_ns: bool = false,
    enable_user_ns: bool = false,
    enable_ipc_ns: bool = false,
    enable_uts_ns: bool = false,
};

pub fn apply(options: NamespaceOptions) !void {
    if (builtin.os.tag != .linux) return error.StrictUnavailable;

    var flags: c_int = 0;
    if (options.enable_mount_ns) flags |= CLONE_NEWNS;
    if (options.enable_ipc_ns) flags |= CLONE_NEWIPC;
    if (options.enable_uts_ns) flags |= CLONE_NEWUTS;
    if (options.enable_pid_ns) flags |= CLONE_NEWPID;
    if (options.enable_user_ns) flags |= CLONE_NEWUSER;
    if (flags == 0) return;

    if (c.syscall(c.SYS_unshare, @as(c_long, @intCast(flags))) != 0) return mapErrnoToError();

    // User namespace requires UID/GID mapping for the child to have a valid identity.
    if (options.enable_user_ns) try writeIdMappings();
}

/// Write UID/GID mappings for the new user namespace.
///
/// Maps the current real UID/GID to 0 (root) inside the namespace. This is
/// the standard rootless container pattern — the process appears as root
/// inside the namespace but has no host privileges. Mapping to 0 is the
/// most portable choice (works regardless of host UID) and enables
/// mount operations inside the namespace.
/// Write UID/GID mappings for the new user namespace.
///
/// EACCES/EPERM on /proc/self/setgroups indicates the kernel or a
/// security module (e.g. AppArmor) blocks user namespace ID mapping.
/// This is an availability issue, not a setup bug.
fn writeIdMappings() !void {
    const uid = std.os.linux.getuid();
    const gid = std.os.linux.getgid();

    // Deny setgroups (required before writing gid_map as unprivileged user).
    writeFile("/proc/self/setgroups", "deny\n") catch |err|
        return if (isPermissionError(err)) error.StrictUnavailable else error.StrictSetupFailed;
    // Map current UID -> 0 inside namespace.
    writeMapping("/proc/self/uid_map", 0, uid) catch |err|
        return if (isPermissionError(err)) error.StrictUnavailable else error.StrictSetupFailed;
    // Map current GID -> 0 inside namespace.
    writeMapping("/proc/self/gid_map", 0, gid) catch |err|
        return if (isPermissionError(err)) error.StrictUnavailable else error.StrictSetupFailed;
}

fn writeMapping(path: []const u8, inner_id: u32, outer_id: u32) !void {
    var buf: [64]u8 = undefined;
    const mapping = std.fmt.bufPrint(&buf, "{d} {d} 1\n", .{ inner_id, outer_id }) catch
        return error.StrictSetupFailed;
    try writeFile(path, mapping);
}

fn writeFile(path: []const u8, data: []const u8) !void {
    const fd = std.posix.open(path, .{ .ACCMODE = .WRONLY }, 0) catch
        return error.AccessDenied;
    defer std.posix.close(fd);
    _ = std.posix.write(fd, data) catch return error.AccessDenied;
}

fn isPermissionError(err: anyerror) bool {
    return err == error.AccessDenied;
}

fn mapErrnoToError() anyerror {
    const errno_value = std.c._errno().*;
    const err: std.posix.E = @enumFromInt(@as(u16, @intCast(errno_value)));
    return switch (err) {
        .PERM, .INVAL, .NOSYS, .OPNOTSUPP => error.StrictUnavailable,
        else => error.StrictSetupFailed,
    };
}

test "apply no-op when namespace options disabled" {
    if (builtin.os.tag != .linux) return;
    try apply(.{});
}

test "apply returns StrictUnavailable on non-linux" {
    if (builtin.os.tag == .linux) return;
    try std.testing.expectError(error.StrictUnavailable, apply(.{ .enable_mount_ns = true }));
}

test "NamespaceOptions defaults are all false" {
    const opts = NamespaceOptions{};
    try std.testing.expect(!opts.enable_mount_ns);
    try std.testing.expect(!opts.enable_pid_ns);
    try std.testing.expect(!opts.enable_user_ns);
    try std.testing.expect(!opts.enable_ipc_ns);
    try std.testing.expect(!opts.enable_uts_ns);
}
