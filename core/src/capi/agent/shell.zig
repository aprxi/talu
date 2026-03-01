//! C API for shell command execution and safety validation.
//!
//! Stateless functions — no handle needed (unlike fs.zig).

const std = @import("std");
const shell = @import("../../agent/shell/root.zig");
const capi_error = @import("../error.zig");
const error_codes = @import("../error_codes.zig");

const allocator = std.heap.c_allocator;

/// Execute a shell command and capture output.
///
/// The command is NOT safety-checked — callers should call
/// `talu_shell_check_command` first if policy enforcement is needed.
///
/// On success (return 0), `out_stdout` / `out_stderr` point to allocated
/// buffers that must be freed via `talu_shell_free_string`. `out_exit_code`
/// receives the process exit code (or a negative signal number).
pub fn talu_shell_exec(
    command: ?[*:0]const u8,
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

    var result = shell.exec.exec(allocator, cmd) catch |err| {
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

/// Free a string returned by `talu_shell_exec`, `talu_shell_default_policy_json`,
/// or `talu_shell_normalize_command`.
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
    const rc = talu_shell_exec(null, null, null, null, null, null);
    try std.testing.expect(rc != 0);
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
