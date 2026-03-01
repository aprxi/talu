//! One-shot shell command execution.
//!
//! Spawns `sh -c <command>`, captures stdout and stderr, returns exit code.
//! This is the low-level execution primitive — no safety checks are applied.
//! Callers should validate commands with `safety.checkCommand()` first.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Result of a one-shot command execution. Caller owns stdout and stderr.
pub const ExecResult = struct {
    stdout: []const u8,
    stderr: []const u8,
    exit_code: ?i32,

    /// Free stdout and stderr buffers.
    pub fn deinit(self: *ExecResult, allocator: Allocator) void {
        if (self.stdout.len > 0) allocator.free(self.stdout);
        if (self.stderr.len > 0) allocator.free(self.stderr);
        self.* = .{ .stdout = &.{}, .stderr = &.{}, .exit_code = null };
    }
};

/// Execute a shell command and capture its output.
///
/// Spawns `/bin/sh -c <command>`, waits for completion, and returns
/// the captured stdout, stderr, and exit code. The caller owns the
/// returned buffers and must call `result.deinit(allocator)` to free them.
pub fn exec(allocator: Allocator, command: []const u8) !ExecResult {
    if (command.len == 0) return error.InvalidArgument;

    // We need a sentinel-terminated copy for the argv.
    const command_z = try allocator.dupeZ(u8, command);
    defer allocator.free(command_z);

    var child = std.process.Child.init(
        &.{ "/bin/sh", "-c", command_z },
        allocator,
    );
    child.stdout_behavior = .Pipe;
    child.stderr_behavior = .Pipe;

    try child.spawn();

    // Collect stdout and stderr via the polling API.
    const max_output: usize = 1024 * 1024; // 1 MB
    var stdout_buf = std.ArrayList(u8).empty;
    defer stdout_buf.deinit(allocator);
    var stderr_buf = std.ArrayList(u8).empty;
    defer stderr_buf.deinit(allocator);

    child.collectOutput(allocator, &stdout_buf, &stderr_buf, max_output) catch |err| {
        _ = child.wait() catch return err;
        return err;
    };

    const term = try child.wait();

    const exit_code: ?i32 = switch (term) {
        .Exited => |code| @as(i32, @intCast(code)),
        .Signal => |sig| -@as(i32, @intCast(sig)),
        else => null,
    };

    // Transfer ownership of buffers to caller.
    const stdout = try stdout_buf.toOwnedSlice(allocator);
    errdefer allocator.free(stdout);
    const stderr = try stderr_buf.toOwnedSlice(allocator);

    return ExecResult{
        .stdout = stdout,
        .stderr = stderr,
        .exit_code = exit_code,
    };
}

// =============================================================================
// Tests
// =============================================================================

test "exec runs valid command and captures stdout" {
    const allocator = std.testing.allocator;
    var result = try exec(allocator, "echo hello");
    defer result.deinit(allocator);

    try std.testing.expectEqualStrings("hello\n", result.stdout);
    try std.testing.expectEqual(@as(?i32, 0), result.exit_code);
}

test "exec captures stderr" {
    const allocator = std.testing.allocator;
    var result = try exec(allocator, "echo oops >&2");
    defer result.deinit(allocator);

    try std.testing.expectEqualStrings("oops\n", result.stderr);
    try std.testing.expectEqual(@as(?i32, 0), result.exit_code);
}

test "exec captures both stdout and stderr" {
    const allocator = std.testing.allocator;
    var result = try exec(allocator, "echo out && echo err >&2");
    defer result.deinit(allocator);

    try std.testing.expectEqualStrings("out\n", result.stdout);
    try std.testing.expectEqualStrings("err\n", result.stderr);
    try std.testing.expectEqual(@as(?i32, 0), result.exit_code);
}

test "exec returns non-zero exit code" {
    const allocator = std.testing.allocator;
    var result = try exec(allocator, "exit 42");
    defer result.deinit(allocator);

    try std.testing.expectEqual(@as(?i32, 42), result.exit_code);
}

test "exec handles empty output" {
    const allocator = std.testing.allocator;
    var result = try exec(allocator, "true");
    defer result.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), result.stdout.len);
    try std.testing.expectEqual(@as(usize, 0), result.stderr.len);
    try std.testing.expectEqual(@as(?i32, 0), result.exit_code);
}

test "exec rejects empty command" {
    const allocator = std.testing.allocator;
    const result = exec(allocator, "");
    try std.testing.expectError(error.InvalidArgument, result);
}

test "exec nonexistent command returns non-zero exit" {
    const allocator = std.testing.allocator;
    var result = try exec(allocator, "__nonexistent_command_12345__");
    defer result.deinit(allocator);

    // sh -c returns 127 for command not found
    try std.testing.expect(result.exit_code != null and result.exit_code.? != 0);
}
