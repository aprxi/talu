//! Signal forwarding helpers for shell sessions.

const std = @import("std");

/// Send a POSIX signal to a process.
pub fn send(pid: std.posix.pid_t, sig: u8) !void {
    try std.posix.kill(pid, sig);
}

/// Send a POSIX signal to a process group.
pub fn sendGroup(pgid: std.posix.pid_t, sig: u8) !void {
    try std.posix.kill(-pgid, sig);
}

test "send returns ProcessNotFound for invalid pid" {
    // Do NOT use -1: kill(-1, sig) sends sig to ALL processes.
    try std.testing.expectError(error.ProcessNotFound, send(999999999, std.posix.SIG.TERM));
}
