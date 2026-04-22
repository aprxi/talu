//! Child bootstrap handshake for strict runtime launches.
//!
//! Strict enforcement must fail-closed. The parent blocks on a one-byte
//! bootstrap channel and only returns a live session/process after the child
//! successfully enters exec (CLOEXEC pipe EOF).

const std = @import("std");
const helpers = @import("helpers.zig");

pub const BootstrapPipe = struct {
    read_fd: std.posix.fd_t,
    write_fd: std.posix.fd_t,

    pub fn closeParentWrite(self: *BootstrapPipe) void {
        if (self.write_fd == -1) return;
        std.posix.close(self.write_fd);
        self.write_fd = -1;
    }

    pub fn closeChildRead(self: *BootstrapPipe) void {
        if (self.read_fd == -1) return;
        std.posix.close(self.read_fd);
        self.read_fd = -1;
    }

    pub fn closeRead(self: *BootstrapPipe) void {
        if (self.read_fd == -1) return;
        std.posix.close(self.read_fd);
        self.read_fd = -1;
    }

    pub fn closeWrite(self: *BootstrapPipe) void {
        if (self.write_fd == -1) return;
        std.posix.close(self.write_fd);
        self.write_fd = -1;
    }
};

const StatusCode = enum(u8) {
    strict_unavailable = 1,
    strict_setup_failed = 2,
    exec_failed = 3,
};

pub fn openBootstrapPipe() !BootstrapPipe {
    const fds = try helpers.pipeCloexec();

    return .{
        .read_fd = fds[0],
        .write_fd = fds[1],
    };
}

pub fn waitForExecBoundary(read_fd: std.posix.fd_t) !void {
    var byte: [1]u8 = undefined;
    while (true) {
        const n = std.posix.read(read_fd, &byte) catch return error.StrictSetupFailed;
        if (n == 0) return;
        return switch (byte[0]) {
            @intFromEnum(StatusCode.strict_unavailable) => error.StrictUnavailable,
            @intFromEnum(StatusCode.strict_setup_failed) => error.StrictSetupFailed,
            @intFromEnum(StatusCode.exec_failed) => error.ExecFailed,
            else => error.StrictSetupFailed,
        };
    }
}

pub fn childReportFailure(write_fd: std.posix.fd_t, err: anyerror) noreturn {
    var byte = [1]u8{statusCodeForError(err)};
    _ = std.posix.write(write_fd, &byte) catch {};
    std.posix.exit(1);
}

pub fn childReportExecFailure(write_fd: std.posix.fd_t) noreturn {
    var byte = [1]u8{@intFromEnum(StatusCode.exec_failed)};
    _ = std.posix.write(write_fd, &byte) catch {};
    std.posix.exit(127);
}

fn statusCodeForError(err: anyerror) u8 {
    return switch (err) {
        error.StrictUnavailable => @intFromEnum(StatusCode.strict_unavailable),
        error.StrictSetupFailed => @intFromEnum(StatusCode.strict_setup_failed),
        else => @intFromEnum(StatusCode.strict_setup_failed),
    };
}

test "statusCodeForError maps strict errors" {
    try std.testing.expectEqual(
        @as(u8, @intFromEnum(StatusCode.strict_unavailable)),
        statusCodeForError(error.StrictUnavailable),
    );
    try std.testing.expectEqual(
        @as(u8, @intFromEnum(StatusCode.strict_setup_failed)),
        statusCodeForError(error.StrictSetupFailed),
    );
}
