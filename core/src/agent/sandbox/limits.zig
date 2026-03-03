//! Runtime resource limits for strict sandbox launches.
//!
//! Limits are applied in the just-forked child process before `execve`.

const std = @import("std");

const c = @cImport({
    @cInclude("errno.h");
    @cInclude("sys/resource.h");
});

/// Runtime resource limits applied via setrlimit in the child process.
///
/// Note: RLIMIT_NPROC is per-UID, not per-sandbox. True per-sandbox process
/// limits require cgroups v2 (pids.max), which is the target for a follow-up.
pub const RuntimeLimits = struct {
    max_memory_bytes: ?u64 = null,
    max_processes: ?u64 = null,
    max_open_files: ?u64 = null,
    cpu_time_seconds: ?u64 = null,

    /// Sensible defaults for strict mode sessions.
    pub fn defaultStrict() RuntimeLimits {
        return .{
            .max_memory_bytes = 2 * 1024 * 1024 * 1024, // 2 GB
            .max_processes = 256,
            .max_open_files = 1024,
            .cpu_time_seconds = null, // sessions are long-lived
        };
    }
};

pub fn apply(limits: RuntimeLimits) !void {
    if (limits.max_memory_bytes) |value| try setRLimit(c.RLIMIT_AS, value);
    if (limits.max_processes) |value| try setRLimit(c.RLIMIT_NPROC, value);
    if (limits.max_open_files) |value| try setRLimit(c.RLIMIT_NOFILE, value);
    if (limits.cpu_time_seconds) |value| try setRLimit(c.RLIMIT_CPU, value);
}

fn setRLimit(resource: c_int, value: u64) !void {
    var rlim = std.mem.zeroes(c.struct_rlimit);
    const cast_value: c.rlim_t = @intCast(value);
    rlim.rlim_cur = cast_value;
    rlim.rlim_max = cast_value;
    if (c.setrlimit(resource, &rlim) != 0) return mapErrnoToError();
}

fn mapErrnoToError() anyerror {
    const errno_value = std.c._errno().*;
    const err: std.posix.E = @enumFromInt(@as(u16, @intCast(errno_value)));
    return switch (err) {
        .PERM, .INVAL => error.StrictUnavailable,
        else => error.StrictSetupFailed,
    };
}

test "apply accepts empty runtime limits" {
    try apply(.{});
}

test "defaultStrict returns non-null memory and process limits" {
    const defaults = RuntimeLimits.defaultStrict();
    try std.testing.expect(defaults.max_memory_bytes != null);
    try std.testing.expect(defaults.max_processes != null);
    try std.testing.expect(defaults.max_open_files != null);
    try std.testing.expectEqual(@as(?u64, null), defaults.cpu_time_seconds);
}
