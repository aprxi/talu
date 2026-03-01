//! Signal forwarding helpers for shell sessions.

const std = @import("std");

/// Send a POSIX signal to a process.
pub fn send(pid: std.posix.pid_t, sig: u8) !void {
    try std.posix.kill(pid, sig);
}

test "send returns ProcessNotFound for invalid pid" {
    try std.testing.expectError(error.ProcessNotFound, send(-1, std.posix.SIG.TERM));
}
