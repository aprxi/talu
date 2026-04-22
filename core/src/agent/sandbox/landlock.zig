//! Linux Landlock runtime enforcement helpers.
//!
//! This module applies executable allowlists in strict runtime mode.

const std = @import("std");
const builtin = @import("builtin");
const profile = @import("profile.zig");
const helpers = @import("helpers.zig");

const c = @cImport({
    @cInclude("errno.h");
    @cInclude("fcntl.h");
    @cInclude("linux/landlock.h");
    @cInclude("sys/prctl.h");
    @cInclude("sys/syscall.h");
    @cInclude("unistd.h");
});

pub fn applyExecProfile(exec_profile: *const profile.ExecProfile) !void {
    if (builtin.os.tag != .linux) return error.StrictUnavailable;
    if (exec_profile.paths.len == 0) return error.StrictSetupFailed;

    var attr = std.mem.zeroes(c.struct_landlock_ruleset_attr);
    attr.handled_access_fs = @intCast(composeHandledAccess(exec_profile));

    const ruleset_fd = c.syscall(
        c.SYS_landlock_create_ruleset,
        &attr,
        @as(c_ulong, @sizeOf(c.struct_landlock_ruleset_attr)),
        @as(c_ulong, 0),
    );
    if (ruleset_fd < 0) return mapErrnoToError();
    defer _ = c.close(@intCast(ruleset_fd));

    for (exec_profile.paths) |path| {
        try addPathRule(@intCast(ruleset_fd), path, c.LANDLOCK_ACCESS_FS_EXECUTE);
    }
    for (exec_profile.fs_paths, exec_profile.fs_access) |path, mask| {
        const access = composeRuleAccess(mask);
        if (access == 0) continue;
        try addPathRule(@intCast(ruleset_fd), path, access);
    }

    if (c.prctl(
        c.PR_SET_NO_NEW_PRIVS,
        @as(c_ulong, 1),
        @as(c_ulong, 0),
        @as(c_ulong, 0),
        @as(c_ulong, 0),
    ) != 0) return mapErrnoToError();

    const restrict_rc = c.syscall(
        c.SYS_landlock_restrict_self,
        @as(c_int, @intCast(ruleset_fd)),
        @as(c_ulong, 0),
    );
    if (restrict_rc < 0) return mapErrnoToError();
}

pub fn validateAvailability() !void {
    if (builtin.os.tag != .linux) return error.StrictUnavailable;

    var attr = std.mem.zeroes(c.struct_landlock_ruleset_attr);
    attr.handled_access_fs = c.LANDLOCK_ACCESS_FS_EXECUTE;

    const ruleset_fd = c.syscall(
        c.SYS_landlock_create_ruleset,
        &attr,
        @as(c_ulong, @sizeOf(c.struct_landlock_ruleset_attr)),
        @as(c_ulong, 0),
    );
    if (ruleset_fd < 0) return mapErrnoToError();
    _ = c.close(@intCast(ruleset_fd));
}

fn mapErrnoToError() anyerror {
    const errno_value = std.c._errno().*;
    const err: std.posix.E = @enumFromInt(@as(u16, @intCast(errno_value)));
    return switch (err) {
        .PERM, .NOSYS, .OPNOTSUPP => error.StrictUnavailable,
        else => error.StrictSetupFailed,
    };
}

fn composeHandledAccess(exec_profile: *const profile.ExecProfile) u64 {
    var access: u64 = c.LANDLOCK_ACCESS_FS_EXECUTE;
    if (exec_profile.enforce_read) {
        access |= c.LANDLOCK_ACCESS_FS_READ_FILE;
        access |= c.LANDLOCK_ACCESS_FS_READ_DIR;
    }
    if (exec_profile.enforce_write) {
        access |= c.LANDLOCK_ACCESS_FS_WRITE_FILE;
        if (@hasDecl(c, "LANDLOCK_ACCESS_FS_TRUNCATE")) access |= c.LANDLOCK_ACCESS_FS_TRUNCATE;
        if (@hasDecl(c, "LANDLOCK_ACCESS_FS_MAKE_REG")) access |= c.LANDLOCK_ACCESS_FS_MAKE_REG;
        if (@hasDecl(c, "LANDLOCK_ACCESS_FS_MAKE_DIR")) access |= c.LANDLOCK_ACCESS_FS_MAKE_DIR;
        if (@hasDecl(c, "LANDLOCK_ACCESS_FS_MAKE_SYM")) access |= c.LANDLOCK_ACCESS_FS_MAKE_SYM;
    }
    if (exec_profile.enforce_delete) {
        access |= c.LANDLOCK_ACCESS_FS_REMOVE_DIR;
        access |= c.LANDLOCK_ACCESS_FS_REMOVE_FILE;
    }
    return access;
}

fn composeRuleAccess(mask: profile.FsAccessMask) u64 {
    var access: u64 = 0;
    if (mask.read) {
        access |= c.LANDLOCK_ACCESS_FS_READ_FILE;
        access |= c.LANDLOCK_ACCESS_FS_READ_DIR;
    }
    if (mask.write) {
        access |= c.LANDLOCK_ACCESS_FS_WRITE_FILE;
        if (@hasDecl(c, "LANDLOCK_ACCESS_FS_TRUNCATE")) access |= c.LANDLOCK_ACCESS_FS_TRUNCATE;
        if (@hasDecl(c, "LANDLOCK_ACCESS_FS_MAKE_REG")) access |= c.LANDLOCK_ACCESS_FS_MAKE_REG;
        if (@hasDecl(c, "LANDLOCK_ACCESS_FS_MAKE_DIR")) access |= c.LANDLOCK_ACCESS_FS_MAKE_DIR;
        if (@hasDecl(c, "LANDLOCK_ACCESS_FS_MAKE_SYM")) access |= c.LANDLOCK_ACCESS_FS_MAKE_SYM;
    }
    if (mask.delete) {
        access |= c.LANDLOCK_ACCESS_FS_REMOVE_DIR;
        access |= c.LANDLOCK_ACCESS_FS_REMOVE_FILE;
    }
    return access;
}

fn addPathRule(ruleset_fd: c_int, path: []const u8, allowed_access: u64) !void {
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_z = helpers.toPathZ(path, &path_buf) catch return error.StrictSetupFailed;

    const open_flags: c_int = (if (@hasDecl(c, "O_PATH")) c.O_PATH else c.O_RDONLY) | c.O_CLOEXEC;
    const fd = c.open(path_z.ptr, open_flags);
    if (fd < 0) return mapErrnoToError();
    defer _ = c.close(fd);

    var rule = std.mem.zeroes(c.struct_landlock_path_beneath_attr);
    rule.parent_fd = @intCast(fd);
    rule.allowed_access = @intCast(allowed_access);

    const rc = c.syscall(
        c.SYS_landlock_add_rule,
        ruleset_fd,
        @as(c_int, c.LANDLOCK_RULE_PATH_BENEATH),
        &rule,
        @as(c_ulong, 0),
    );
    if (rc < 0) return mapErrnoToError();
}

test "validateAvailability returns typed support errors on this platform" {
    if (builtin.os.tag != .linux) {
        try std.testing.expectError(error.StrictUnavailable, validateAvailability());
        return;
    }
    // On Linux hosts in CI/dev, either availability succeeds or returns a
    // typed unavailability/setup error without crashing.
    _ = validateAvailability() catch |err| switch (err) {
        error.StrictUnavailable, error.StrictSetupFailed => {},
        else => return err,
    };
}
