//! One-shot shell command execution.
//!
//! Spawns `sh -c <command>`, captures stdout and stderr, returns exit code.
//! This is the low-level execution primitive — no safety checks are applied.
//! Callers should validate commands with `safety.checkCommand()` first.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Child = std.process.Child;

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

/// Execution options for shell commands.
pub const ExecOptions = struct {
    pub const EnvVar = struct {
        key: []const u8,
        value: []const u8,
    };

    cwd: ?[]const u8 = null,
    env: ?[]const EnvVar = null,
    timeout_ms: u64 = 120_000,
    max_output_bytes: usize = 1024 * 1024,
};

/// Streaming callback for incremental stdout/stderr forwarding.
/// Return false to abort processing.
pub const StreamCallback = *const fn (ctx: ?*anyopaque, data: []const u8) bool;

/// Execute a shell command and capture its output.
///
/// Spawns `/bin/sh -c <command>`, waits for completion, and returns
/// the captured stdout, stderr, and exit code. The caller owns the
/// returned buffers and must call `result.deinit(allocator)` to free them.
pub fn exec(allocator: Allocator, command: []const u8) !ExecResult {
    return execWithOptions(allocator, command, .{});
}

/// Execute a shell command with explicit options and capture output.
pub fn execWithOptions(
    allocator: Allocator,
    command: []const u8,
    options: ExecOptions,
) !ExecResult {
    return execStreaming(allocator, command, options, null, null, null, null);
}

/// Initialize a Child that runs `/bin/sh -c <command>`.
///
/// IMPORTANT: `argv` is stored by pointer, not copied. The caller must keep
/// `argv_buf` alive until after `child.spawn()` completes.
fn initShellChild(
    argv_buf: *const [3][]const u8,
    allocator: Allocator,
    options: ExecOptions,
    env_map: ?*std.process.EnvMap,
) Child {
    var child = std.process.Child.init(argv_buf, allocator);
    child.cwd = options.cwd;
    child.env_map = env_map;
    child.stdin_behavior = .Ignore;
    child.stdout_behavior = .Pipe;
    child.stderr_behavior = .Pipe;
    return child;
}

fn enforceOutputLimit(current_len: usize, chunk_len: usize, max_len: usize, which_stdout: bool) !void {
    const new_len = std.math.add(usize, current_len, chunk_len) catch {
        if (which_stdout) return error.StdoutStreamTooLong;
        return error.StderrStreamTooLong;
    };
    if (new_len > max_len) {
        if (which_stdout) return error.StdoutStreamTooLong;
        return error.StderrStreamTooLong;
    }
}

fn flushStreamChunk(
    allocator: Allocator,
    list: *std.ArrayList(u8),
    reader: *std.Io.Reader,
    callback: ?StreamCallback,
    callback_ctx: ?*anyopaque,
    max_output_bytes: usize,
    which_stdout: bool,
) !void {
    const chunk = reader.buffered();
    if (chunk.len == 0) return;

    if (callback) |cb| {
        if (!cb(callback_ctx, chunk)) return error.Aborted;
    }

    try enforceOutputLimit(list.items.len, chunk.len, max_output_bytes, which_stdout);
    try list.appendSlice(allocator, chunk);
    reader.tossBuffered();
}

/// Execute a command and forward incremental output chunks to callbacks.
///
/// This function captures output for the returned `ExecResult` while also
/// forwarding each chunk as it is read from stdout/stderr pipes.
pub fn execStreaming(
    allocator: Allocator,
    command: []const u8,
    options: ExecOptions,
    on_stdout: ?StreamCallback,
    on_stdout_ctx: ?*anyopaque,
    on_stderr: ?StreamCallback,
    on_stderr_ctx: ?*anyopaque,
) !ExecResult {
    if (command.len == 0) return error.InvalidArgument;

    // We need a sentinel-terminated copy for the argv.
    const command_z = try allocator.dupeZ(u8, command);
    defer allocator.free(command_z);

    var env_map: ?std.process.EnvMap = null;
    defer if (env_map) |*map| map.deinit();

    if (options.env) |pairs| {
        env_map = try std.process.getEnvMap(allocator);
        for (pairs) |pair| {
            if (pair.key.len == 0) return error.InvalidArgument;
            try env_map.?.put(pair.key, pair.value);
        }
    }

    const env_map_ptr: ?*std.process.EnvMap = if (env_map) |*map| map else null;
    // argv must outlive initShellChild — keep it on this stack frame.
    const argv = [_][]const u8{ "/bin/sh", "-c", command_z };
    var child = initShellChild(&argv, allocator, options, env_map_ptr);
    try child.spawn();
    errdefer {
        _ = child.kill() catch {};
    }

    var stdout_buf = std.ArrayList(u8).empty;
    defer stdout_buf.deinit(allocator);
    var stderr_buf = std.ArrayList(u8).empty;
    defer stderr_buf.deinit(allocator);

    var poller = std.Io.poll(allocator, enum { stdout, stderr }, .{
        .stdout = child.stdout.?,
        .stderr = child.stderr.?,
    });
    defer poller.deinit();

    const deadline_ns: ?i128 = if (options.timeout_ms == 0)
        null
    else
        std.time.nanoTimestamp() + @as(i128, @intCast(options.timeout_ms)) * std.time.ns_per_ms;

    while (true) {
        const keep_polling = if (deadline_ns) |deadline| blk: {
            const now = std.time.nanoTimestamp();
            if (now >= deadline) {
                _ = child.kill() catch {};
                return error.Timeout;
            }
            const remaining: u64 = @intCast(deadline - now);
            break :blk try poller.pollTimeout(remaining);
        } else try poller.poll();

        flushStreamChunk(
            allocator,
            &stdout_buf,
            poller.reader(.stdout),
            on_stdout,
            on_stdout_ctx,
            options.max_output_bytes,
            true,
        ) catch |err| {
            if (err == error.Aborted) {
                _ = child.kill() catch {};
            }
            return err;
        };

        flushStreamChunk(
            allocator,
            &stderr_buf,
            poller.reader(.stderr),
            on_stderr,
            on_stderr_ctx,
            options.max_output_bytes,
            false,
        ) catch |err| {
            if (err == error.Aborted) {
                _ = child.kill() catch {};
            }
            return err;
        };

        if (!keep_polling) break;
    }

    const term = try child.wait();
    const exit_code: ?i32 = switch (term) {
        .Exited => |code| @as(i32, @intCast(code)),
        .Signal => |sig| -@as(i32, @intCast(sig)),
        else => null,
    };

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

test "execWithOptions runs command in cwd" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{ .sub_path = "probe.txt", .data = "ok" });
    const cwd = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(cwd);

    var result = try execWithOptions(allocator, "ls probe.txt", .{ .cwd = cwd });
    defer result.deinit(allocator);
    try std.testing.expect(std.mem.indexOf(u8, result.stdout, "probe.txt") != null);
}

test "execStreaming forwards stdout and stderr" {
    const allocator = std.testing.allocator;

    var saw_stdout = false;
    var saw_stderr = false;

    const callbacks = struct {
        fn onStdout(ctx: ?*anyopaque, data: []const u8) bool {
            const flag: *bool = @ptrCast(@alignCast(ctx.?));
            if (std.mem.indexOf(u8, data, "out") != null) {
                flag.* = true;
            }
            return true;
        }

        fn onStderr(ctx: ?*anyopaque, data: []const u8) bool {
            const flag: *bool = @ptrCast(@alignCast(ctx.?));
            if (std.mem.indexOf(u8, data, "err") != null) {
                flag.* = true;
            }
            return true;
        }
    };

    var result = try execStreaming(
        allocator,
        "echo out && echo err >&2",
        .{},
        callbacks.onStdout,
        &saw_stdout,
        callbacks.onStderr,
        &saw_stderr,
    );
    defer result.deinit(allocator);

    try std.testing.expect(saw_stdout);
    try std.testing.expect(saw_stderr);
    try std.testing.expectEqual(@as(?i32, 0), result.exit_code);
}

test "execWithOptions applies env overrides" {
    const allocator = std.testing.allocator;
    const env = [_]ExecOptions.EnvVar{
        .{ .key = "TALU_SHELL_ENV_TEST", .value = "works" },
    };

    var result = try execWithOptions(
        allocator,
        "printf '%s' \"$TALU_SHELL_ENV_TEST\"",
        .{ .env = env[0..] },
    );
    defer result.deinit(allocator);

    try std.testing.expectEqualStrings("works", result.stdout);
}

test "execWithOptions enforces timeout" {
    const allocator = std.testing.allocator;
    const out = execWithOptions(
        allocator,
        "sleep 2",
        .{
            .timeout_ms = 20,
        },
    );
    try std.testing.expectError(error.Timeout, out);
}
