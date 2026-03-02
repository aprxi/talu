//! C API for shell command execution and safety validation.
//!
//! Stateless functions — no handle needed (unlike fs.zig).

const std = @import("std");
const shell = @import("../../agent/shell/root.zig");
const policy_api = @import("policy.zig");
const capi_error = @import("../error.zig");
const error_codes = @import("../error_codes.zig");

const allocator = std.heap.c_allocator;
const DEFAULT_SCROLLBACK_BYTES: usize = 64 * 1024;
const ACTION_EXEC = "tool.exec";
const ACTION_SHELL = "tool.shell";

pub const TaluShell = opaque {};

pub const StreamCallback = *const fn (?*anyopaque, [*]const u8, usize) callconv(.c) bool;

fn toShell(handle: ?*TaluShell) !*shell.session.ShellSession {
    return @ptrCast(@alignCast(handle orelse return error.InvalidHandle));
}

fn setPolicyDeniedError(reason: ?policy_api.ProcessDenyReason) i32 {
    if (reason != null and reason.? == .cwd) {
        capi_error.setErrorWithCode(.io_permission_denied, "agent policy denied cwd", .{});
        return @intFromEnum(error_codes.ErrorCode.policy_denied_cwd);
    }
    capi_error.setErrorWithCode(.shell_command_denied, "agent policy denied exec action", .{});
    return @intFromEnum(error_codes.ErrorCode.policy_denied_exec);
}

fn enforceShellExecPolicy(
    policy: ?*policy_api.TaluAgentPolicy,
    action: []const u8,
    command: []const u8,
    cwd: ?[]const u8,
) i32 {
    const safety_check = shell.safety.checkCommand(command);
    if (!safety_check.allowed) {
        const reason = safety_check.reason orelse "command denied by baseline shell safety";
        capi_error.setErrorWithCode(.shell_command_denied, "{s}", .{reason});
        return @intFromEnum(error_codes.ErrorCode.shell_command_denied);
    }

    var iter = shell.safety.ChainIterator.init(command);
    while (iter.next()) |segment_raw| {
        const segment = std.mem.trim(u8, segment_raw, &std.ascii.whitespace);
        if (segment.len == 0) continue;

        const normalized = shell.safety.normalizeCommand(allocator, segment) catch |err| {
            capi_error.setError(err, "failed to normalize command segment", .{});
            return @intFromEnum(error_codes.errorToCode(err));
        };
        defer allocator.free(normalized);

        const policy_result = policy_api.enforceProcessPolicy(policy, action, normalized, cwd);
        if (!policy_result.allowed) {
            return setPolicyDeniedError(policy_result.deny_reason);
        }
    }
    return 0;
}

/// Execute a shell command and capture output.
///
/// Enforces baseline shell safety checks and optional agent policy.
///
/// On success (return 0), `out_stdout` / `out_stderr` point to allocated
/// buffers that must be freed via `talu_shell_free_string`. `out_exit_code`
/// receives the process exit code (or a negative signal number).
pub fn talu_shell_exec(
    command: ?[*:0]const u8,
    cwd: ?[*:0]const u8,
    policy: ?*policy_api.TaluAgentPolicy,
    out_stdout: ?*?[*]const u8,
    out_stdout_len: ?*usize,
    out_stderr: ?*?[*]const u8,
    out_stderr_len: ?*usize,
    out_exit_code: ?*i32,
) callconv(.c) i32 {
    capi_error.clearError();
    if (out_stdout) |p| p.* = null;
    if (out_stdout_len) |l| l.* = 0;
    if (out_stderr) |p| p.* = null;
    if (out_stderr_len) |l| l.* = 0;
    if (out_exit_code) |c| c.* = -1;

    const cmd = std.mem.sliceTo(command orelse {
        capi_error.setErrorWithCode(.invalid_argument, "command is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }, 0);

    if (cmd.len == 0) {
        capi_error.setErrorWithCode(.invalid_argument, "command is empty", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }

    const cwd_slice: ?[]const u8 = if (cwd) |value| blk: {
        const slice = std.mem.sliceTo(value, 0);
        if (slice.len == 0) {
            capi_error.setErrorWithCode(.invalid_argument, "cwd is empty", .{});
            return @intFromEnum(error_codes.ErrorCode.invalid_argument);
        }
        break :blk slice;
    } else null;

    const policy_rc = enforceShellExecPolicy(policy, ACTION_EXEC, cmd, cwd_slice);
    if (policy_rc != 0) return policy_rc;

    const default_timeout_ms: u64 = 120_000;
    const effective_timeout_ms = policy_api.clampTimeoutMs(policy, default_timeout_ms);

    var result = shell.exec.execWithOptions(allocator, cmd, .{
        .cwd = cwd_slice,
        .timeout_ms = effective_timeout_ms,
    }) catch |err| {
        capi_error.setError(err, "shell exec failed", .{});
        return @intFromEnum(error_codes.ErrorCode.shell_exec_failed);
    };
    // We transfer ownership to the caller; do NOT deinit here.

    if (out_stdout) |p| p.* = if (result.stdout.len == 0) null else result.stdout.ptr;
    if (out_stdout_len) |l| l.* = result.stdout.len;
    if (out_stderr) |p| p.* = if (result.stderr.len == 0) null else result.stderr.ptr;
    if (out_stderr_len) |l| l.* = result.stderr.len;
    if (out_exit_code) |c| c.* = result.exit_code orelse -1;

    // Prevent deinit from freeing the buffers we just transferred.
    result.stdout = &.{};
    result.stderr = &.{};

    return 0;
}

/// Execute a command with execution options and stream chunks via callbacks.
pub fn talu_shell_exec_streaming(
    command: ?[*:0]const u8,
    cwd: ?[*:0]const u8,
    policy: ?*policy_api.TaluAgentPolicy,
    timeout_ms: u64,
    on_stdout: ?StreamCallback,
    on_stdout_ctx: ?*anyopaque,
    on_stderr: ?StreamCallback,
    on_stderr_ctx: ?*anyopaque,
    out_exit_code: ?*i32,
) callconv(.c) i32 {
    capi_error.clearError();
    if (out_exit_code) |code| code.* = -1;

    const cmd = std.mem.sliceTo(command orelse {
        capi_error.setErrorWithCode(.invalid_argument, "command is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }, 0);
    if (cmd.len == 0) {
        capi_error.setErrorWithCode(.invalid_argument, "command is empty", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }

    const cwd_slice: ?[]const u8 = if (cwd) |value| blk: {
        const slice = std.mem.sliceTo(value, 0);
        if (slice.len == 0) {
            capi_error.setErrorWithCode(.invalid_argument, "cwd is empty", .{});
            return @intFromEnum(error_codes.ErrorCode.invalid_argument);
        }
        break :blk slice;
    } else null;

    const policy_rc = enforceShellExecPolicy(policy, ACTION_EXEC, cmd, cwd_slice);
    if (policy_rc != 0) return policy_rc;

    const effective_timeout_ms = policy_api.clampTimeoutMs(policy, timeout_ms);

    var stdout_stream_ctx = CStreamCtx{ .callback = cNoopStreamCallback, .ctx = null };
    var stderr_stream_ctx = CStreamCtx{ .callback = cNoopStreamCallback, .ctx = null };
    if (on_stdout) |cb| {
        stdout_stream_ctx = .{ .callback = cb, .ctx = on_stdout_ctx };
    }
    if (on_stderr) |cb| {
        stderr_stream_ctx = .{ .callback = cb, .ctx = on_stderr_ctx };
    }

    var result = shell.exec.execStreaming(
        allocator,
        cmd,
        .{
            .cwd = cwd_slice,
            .timeout_ms = effective_timeout_ms,
            .max_output_bytes = 1024 * 1024,
        },
        if (on_stdout != null) cStreamForward else null,
        if (on_stdout != null) &stdout_stream_ctx else null,
        if (on_stderr != null) cStreamForward else null,
        if (on_stderr != null) &stderr_stream_ctx else null,
    ) catch |err| {
        capi_error.setError(err, "shell streaming exec failed", .{});
        return @intFromEnum(error_codes.ErrorCode.shell_exec_failed);
    };
    defer result.deinit(allocator);

    if (out_exit_code) |code| code.* = result.exit_code orelse -1;
    return 0;
}

const CStreamCtx = struct {
    callback: StreamCallback,
    ctx: ?*anyopaque,
};

fn cNoopStreamCallback(_: ?*anyopaque, _: [*]const u8, _: usize) callconv(.c) bool {
    return true;
}

fn cStreamForward(ctx: ?*anyopaque, data: []const u8) bool {
    const stream_ctx: *const CStreamCtx = @ptrCast(@alignCast(ctx orelse return true));
    if (data.len == 0) return true;
    return stream_ctx.callback(stream_ctx.ctx, data.ptr, data.len);
}

/// Check whether a command is allowed by the built-in whitelist.
///
/// On return, `out_allowed` is true if the command passes all checks.
/// If denied, `out_reason` / `out_reason_len` point to a static string
/// describing why (caller does NOT free it).
pub fn talu_shell_check_command(
    command: ?[*:0]const u8,
    out_allowed: ?*bool,
    out_reason: ?*?[*]const u8,
    out_reason_len: ?*usize,
) callconv(.c) i32 {
    capi_error.clearError();
    if (out_allowed) |a| a.* = false;
    if (out_reason) |r| r.* = null;
    if (out_reason_len) |l| l.* = 0;

    const cmd = std.mem.sliceTo(command orelse {
        capi_error.setErrorWithCode(.invalid_argument, "command is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }, 0);

    const check = shell.safety.checkCommand(cmd);

    if (out_allowed) |a| a.* = check.allowed;
    if (check.reason) |reason| {
        if (out_reason) |r| r.* = reason.ptr;
        if (out_reason_len) |l| l.* = reason.len;
    }
    return 0;
}

/// Get the default IAM-style policy JSON containing the whitelist.
///
/// The returned JSON must be freed via `talu_shell_free_string`.
pub fn talu_shell_default_policy_json(
    out_json: ?*?[*]const u8,
    out_json_len: ?*usize,
) callconv(.c) i32 {
    capi_error.clearError();
    if (out_json) |p| p.* = null;
    if (out_json_len) |l| l.* = 0;

    const json_ptr = out_json orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_json is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const json_len_ptr = out_json_len orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_json_len is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const json = shell.safety.defaultPolicyJson(allocator) catch |err| {
        capi_error.setError(err, "failed to build default policy JSON", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    json_ptr.* = if (json.len == 0) null else json.ptr;
    json_len_ptr.* = json.len;
    return 0;
}

/// Normalize a command for policy evaluation (absolute path → basename).
///
/// The returned string must be freed via `talu_shell_free_string`.
pub fn talu_shell_normalize_command(
    command: ?[*:0]const u8,
    out_normalized: ?*?[*]const u8,
    out_normalized_len: ?*usize,
) callconv(.c) i32 {
    capi_error.clearError();
    if (out_normalized) |p| p.* = null;
    if (out_normalized_len) |l| l.* = 0;

    const cmd = std.mem.sliceTo(command orelse {
        capi_error.setErrorWithCode(.invalid_argument, "command is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }, 0);

    const out_ptr = out_normalized orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_normalized is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const out_len_ptr = out_normalized_len orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_normalized_len is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const normalized = shell.safety.normalizeCommand(allocator, cmd) catch |err| {
        capi_error.setError(err, "failed to normalize command", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    out_ptr.* = if (normalized.len == 0) null else normalized.ptr;
    out_len_ptr.* = normalized.len;
    return 0;
}

/// Open an interactive shell session.
pub fn talu_shell_open(
    cols: u16,
    rows: u16,
    cwd: ?[*:0]const u8,
    policy: ?*policy_api.TaluAgentPolicy,
    out_shell: ?*?*TaluShell,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_shell orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_shell is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;

    const cwd_slice: ?[]const u8 = if (cwd) |value| blk: {
        const slice = std.mem.sliceTo(value, 0);
        if (slice.len == 0) {
            capi_error.setErrorWithCode(.invalid_argument, "cwd is empty", .{});
            return @intFromEnum(error_codes.ErrorCode.invalid_argument);
        }
        break :blk slice;
    } else null;

    const policy_result = policy_api.enforceProcessPolicy(policy, ACTION_SHELL, null, cwd_slice);
    if (!policy_result.allowed) {
        return setPolicyDeniedError(policy_result.deny_reason);
    }

    const session_ptr = shell.session.ShellSession.open(
        allocator,
        cols,
        rows,
        cwd_slice,
        DEFAULT_SCROLLBACK_BYTES,
    ) catch |err| {
        capi_error.setError(err, "failed to open shell session", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    out.* = @ptrCast(session_ptr);
    return 0;
}

/// Close an interactive shell session.
pub fn talu_shell_close(shell_handle: ?*TaluShell) callconv(.c) void {
    capi_error.clearError();
    const session_ptr: *shell.session.ShellSession = @ptrCast(@alignCast(shell_handle orelse return));
    session_ptr.close();
}

/// Write bytes to a shell session PTY.
pub fn talu_shell_write(
    shell_handle: ?*TaluShell,
    data: ?[*]const u8,
    len: usize,
) callconv(.c) i32 {
    capi_error.clearError();

    const session_ptr = toShell(shell_handle) catch |err| {
        capi_error.setError(err, "invalid shell handle", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    const data_slice: []const u8 = if (len == 0) &.{} else blk: {
        const ptr = data orelse {
            capi_error.setErrorWithCode(.invalid_argument, "data is null but len > 0", .{});
            return @intFromEnum(error_codes.ErrorCode.invalid_argument);
        };
        break :blk ptr[0..len];
    };

    _ = session_ptr.write(data_slice) catch |err| {
        capi_error.setError(err, "failed to write shell session", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Read bytes from a shell session PTY into caller buffer.
pub fn talu_shell_read(
    shell_handle: ?*TaluShell,
    buf: ?[*]u8,
    buf_len: usize,
    out_read: ?*usize,
) callconv(.c) i32 {
    capi_error.clearError();
    if (out_read) |n| n.* = 0;
    const out_read_ptr = out_read orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_read is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const session_ptr = toShell(shell_handle) catch |err| {
        capi_error.setError(err, "invalid shell handle", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    const read_buf: []u8 = if (buf_len == 0) &.{} else blk: {
        const ptr = buf orelse {
            capi_error.setErrorWithCode(.invalid_argument, "buf is null but buf_len > 0", .{});
            return @intFromEnum(error_codes.ErrorCode.invalid_argument);
        };
        break :blk ptr[0..buf_len];
    };

    const n = session_ptr.read(read_buf) catch |err| {
        capi_error.setError(err, "failed to read shell session", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    out_read_ptr.* = n;
    return 0;
}

/// Resize shell session PTY.
pub fn talu_shell_resize(
    shell_handle: ?*TaluShell,
    cols: u16,
    rows: u16,
) callconv(.c) i32 {
    capi_error.clearError();
    const session_ptr = toShell(shell_handle) catch |err| {
        capi_error.setError(err, "invalid shell handle", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    session_ptr.resize(cols, rows) catch |err| {
        capi_error.setError(err, "failed to resize shell session", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Send a POSIX signal to the shell session process.
pub fn talu_shell_signal(
    shell_handle: ?*TaluShell,
    sig: u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const session_ptr = toShell(shell_handle) catch |err| {
        capi_error.setError(err, "invalid shell handle", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    session_ptr.sendSignal(sig) catch |err| {
        capi_error.setError(err, "failed to signal shell session", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Return true if shell session process is still alive.
pub fn talu_shell_alive(shell_handle: ?*TaluShell) callconv(.c) bool {
    capi_error.clearError();
    const session_ptr = toShell(shell_handle) catch |err| {
        capi_error.setError(err, "invalid shell handle", .{});
        return false;
    };
    return session_ptr.isAlive() catch |err| {
        capi_error.setError(err, "shell session closed", .{});
        return false;
    };
}

/// Copy shell session scrollback buffer.
pub fn talu_shell_scrollback(
    shell_handle: ?*TaluShell,
    out_data: ?*?[*]const u8,
    out_len: ?*usize,
) callconv(.c) i32 {
    capi_error.clearError();
    if (out_data) |ptr| ptr.* = null;
    if (out_len) |len| len.* = 0;
    const out_data_ptr = out_data orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_data is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const out_len_ptr = out_len orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_len is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const session_ptr = toShell(shell_handle) catch |err| {
        capi_error.setError(err, "invalid shell handle", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    const copied = session_ptr.scrollbackCopy(allocator) catch |err| {
        capi_error.setError(err, "failed to copy shell scrollback", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    out_data_ptr.* = if (copied.len == 0) null else copied.ptr;
    out_len_ptr.* = copied.len;
    return 0;
}

/// Free bytes returned by shell APIs that transfer ownership to the caller,
/// including `talu_shell_exec`, `talu_shell_default_policy_json`,
/// `talu_shell_normalize_command`, and `talu_shell_scrollback`.
pub fn talu_shell_free_string(ptr: ?[*]const u8, len: usize) callconv(.c) void {
    if (ptr == null or len == 0) return;
    const mutable: [*]u8 = @constCast(ptr.?);
    allocator.free(mutable[0..len]);
}

// =============================================================================
// Tests
// =============================================================================

test "talu_shell_check_command allows whitelisted" {
    var allowed = false;
    var reason: ?[*]const u8 = null;
    var reason_len: usize = 0;
    const rc = talu_shell_check_command("ls -la", &allowed, &reason, &reason_len);
    try std.testing.expectEqual(@as(i32, 0), rc);
    try std.testing.expect(allowed);
    try std.testing.expect(reason == null);
}

test "talu_shell_check_command denies non-whitelisted" {
    var allowed = true;
    var reason: ?[*]const u8 = null;
    var reason_len: usize = 0;
    const rc = talu_shell_check_command("rm -rf /", &allowed, &reason, &reason_len);
    try std.testing.expectEqual(@as(i32, 0), rc);
    try std.testing.expect(!allowed);
    try std.testing.expect(reason != null);
    try std.testing.expect(reason_len > 0);
}

test "talu_shell_check_command null command returns error" {
    var allowed = false;
    const rc = talu_shell_check_command(null, &allowed, null, null);
    try std.testing.expect(rc != 0);
}

test "talu_shell_exec runs whitelisted command" {
    var stdout_ptr: ?[*]const u8 = null;
    var stdout_len: usize = 0;
    var stderr_ptr: ?[*]const u8 = null;
    var stderr_len: usize = 0;
    var exit_code: i32 = -1;

    const rc = talu_shell_exec(
        "echo hello",
        null,
        null,
        &stdout_ptr,
        &stdout_len,
        &stderr_ptr,
        &stderr_len,
        &exit_code,
    );
    defer talu_shell_free_string(stdout_ptr, stdout_len);
    defer talu_shell_free_string(stderr_ptr, stderr_len);

    try std.testing.expectEqual(@as(i32, 0), rc);
    try std.testing.expectEqual(@as(i32, 0), exit_code);
    try std.testing.expect(stdout_ptr != null);
    try std.testing.expectEqualStrings("hello\n", stdout_ptr.?[0..stdout_len]);
}

test "talu_shell_exec null command returns error" {
    const rc = talu_shell_exec(null, null, null, null, null, null, null, null);
    try std.testing.expect(rc != 0);
}

test "talu_shell_exec enforces agent policy" {
    var policy_handle: ?*policy_api.TaluAgentPolicy = null;
    const policy_json =
        \\{"default":"deny","statements":[{"effect":"allow","action":"tool.fs.read","resource":"**"}]}
    ;
    try std.testing.expectEqual(
        @as(i32, 0),
        policy_api.talu_agent_policy_create(policy_json.ptr, policy_json.len, &policy_handle),
    );
    defer policy_api.talu_agent_policy_free(policy_handle);

    const rc = talu_shell_exec("echo hello", null, policy_handle, null, null, null, null, null);
    try std.testing.expectEqual(
        @as(i32, @intFromEnum(error_codes.ErrorCode.policy_denied_exec)),
        rc,
    );
}

test "talu_shell_default_policy_json returns valid JSON" {
    var json_ptr: ?[*]const u8 = null;
    var json_len: usize = 0;
    const rc = talu_shell_default_policy_json(&json_ptr, &json_len);
    defer talu_shell_free_string(json_ptr, json_len);

    try std.testing.expectEqual(@as(i32, 0), rc);
    try std.testing.expect(json_ptr != null);
    try std.testing.expect(json_len > 0);

    const json = json_ptr.?[0..json_len];
    try std.testing.expect(std.mem.startsWith(u8, json, "{\"default\":\"deny\""));
}

test "talu_shell_normalize_command strips path" {
    var out_ptr: ?[*]const u8 = null;
    var out_len: usize = 0;
    const rc = talu_shell_normalize_command("/usr/bin/git status", &out_ptr, &out_len);
    defer talu_shell_free_string(out_ptr, out_len);

    try std.testing.expectEqual(@as(i32, 0), rc);
    try std.testing.expectEqualStrings("git status", out_ptr.?[0..out_len]);
}

test "talu_shell_exec_streaming invokes callbacks and returns exit" {
    const callbacks = struct {
        fn onStdout(ctx: ?*anyopaque, ptr: [*]const u8, len: usize) callconv(.c) bool {
            const saw: *bool = @ptrCast(@alignCast(ctx.?));
            if (len > 0 and std.mem.indexOf(u8, ptr[0..len], "out") != null) {
                saw.* = true;
            }
            return true;
        }

        fn onStderr(ctx: ?*anyopaque, ptr: [*]const u8, len: usize) callconv(.c) bool {
            const saw: *bool = @ptrCast(@alignCast(ctx.?));
            if (len > 0 and std.mem.indexOf(u8, ptr[0..len], "err") != null) {
                saw.* = true;
            }
            return true;
        }
    };

    var saw_stdout = false;
    var saw_stderr = false;
    var exit_code: i32 = -1;
    const rc = talu_shell_exec_streaming(
        "echo out && echo err >&2",
        null,
        null,
        30_000,
        callbacks.onStdout,
        &saw_stdout,
        callbacks.onStderr,
        &saw_stderr,
        &exit_code,
    );
    try std.testing.expectEqual(@as(i32, 0), rc);
    try std.testing.expect(saw_stdout);
    try std.testing.expect(saw_stderr);
    try std.testing.expectEqual(@as(i32, 0), exit_code);
}

test "talu_shell_open write read and scrollback" {
    var handle: ?*TaluShell = null;
    try std.testing.expectEqual(@as(i32, 0), talu_shell_open(80, 24, null, null, &handle));
    defer talu_shell_close(handle);

    const command = "echo hello\nexit\n";
    try std.testing.expectEqual(@as(i32, 0), talu_shell_write(handle, command.ptr, command.len));

    var read_buffer: [2048]u8 = undefined;
    var output = std.ArrayList(u8).empty;
    defer output.deinit(std.testing.allocator);

    var guard: usize = 0;
    while (talu_shell_alive(handle) and guard < 50_000) : (guard += 1) {
        var n: usize = 0;
        try std.testing.expectEqual(@as(i32, 0), talu_shell_read(handle, read_buffer[0..].ptr, read_buffer.len, &n));
        if (n > 0) try output.appendSlice(std.testing.allocator, read_buffer[0..n]);
    }
    while (true) {
        var n: usize = 0;
        try std.testing.expectEqual(@as(i32, 0), talu_shell_read(handle, read_buffer[0..].ptr, read_buffer.len, &n));
        if (n == 0) break;
        try output.appendSlice(std.testing.allocator, read_buffer[0..n]);
    }

    try std.testing.expect(std.mem.indexOf(u8, output.items, "hello") != null);

    var scroll_ptr: ?[*]const u8 = null;
    var scroll_len: usize = 0;
    try std.testing.expectEqual(@as(i32, 0), talu_shell_scrollback(handle, &scroll_ptr, &scroll_len));
    defer talu_shell_free_string(scroll_ptr, scroll_len);
    try std.testing.expect(scroll_ptr != null);
    try std.testing.expect(std.mem.indexOf(u8, scroll_ptr.?[0..scroll_len], "hello") != null);
}
