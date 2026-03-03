//! C API bridge for long-lived non-PTY process sessions (`talu_process_*`).

const std = @import("std");
const agent = @import("../../agent/root.zig");
const sandbox = @import("../../agent/sandbox/root.zig");
const policy_api = @import("policy.zig");
const runtime_api = @import("runtime.zig");
const capi_error = @import("../error.zig");
const error_codes = @import("../error_codes.zig");

const allocator = std.heap.c_allocator;
const process = agent.process;
const ACTION_PROCESS = "tool.process";

pub const TaluProcess = opaque {};

fn toProcess(handle: ?*TaluProcess) !*process.session.ProcessSession {
    const ptr = handle orelse return error.InvalidHandle;
    return @ptrCast(@alignCast(ptr));
}

fn setPolicyDeniedError(reason: ?policy_api.ProcessDenyReason) i32 {
    if (reason != null and reason.? == .cwd) {
        capi_error.setErrorWithCode(.io_permission_denied, "agent policy denied cwd", .{});
        return @intFromEnum(error_codes.ErrorCode.policy_denied_cwd);
    }
    capi_error.setErrorWithCode(.shell_command_denied, "agent policy denied process action", .{});
    return @intFromEnum(error_codes.ErrorCode.policy_denied_exec);
}

fn setStrictError(err: anyerror) i32 {
    return switch (err) {
        error.StrictUnavailable => blk: {
            capi_error.setErrorWithCode(.shell_command_denied, "strict runtime unavailable", .{});
            break :blk @intFromEnum(error_codes.ErrorCode.policy_strict_unavailable);
        },
        else => blk: {
            capi_error.setErrorWithCode(.shell_exec_failed, "strict runtime setup failed", .{});
            break :blk @intFromEnum(error_codes.ErrorCode.policy_strict_setup_failed);
        },
    };
}

fn enforceProcessOpenPolicy(
    policy: ?*policy_api.TaluAgentPolicy,
    command: []const u8,
    cwd: ?[]const u8,
) i32 {
    const safety_check = agent.shell.safety.checkCommand(command);
    if (!safety_check.allowed) {
        const reason = safety_check.reason orelse "command denied by baseline shell safety";
        capi_error.setErrorWithCode(.shell_command_denied, "{s}", .{reason});
        return @intFromEnum(error_codes.ErrorCode.shell_command_denied);
    }

    var iter = agent.shell.safety.ChainIterator.init(command);
    while (iter.next()) |segment_raw| {
        const segment = std.mem.trim(u8, segment_raw, &std.ascii.whitespace);
        if (segment.len == 0) continue;

        const normalized = agent.shell.safety.normalizeCommand(allocator, segment) catch |err| {
            capi_error.setError(err, "failed to normalize command segment", .{});
            return @intFromEnum(error_codes.errorToCode(err));
        };
        defer allocator.free(normalized);

        const policy_result = policy_api.enforceProcessPolicy(policy, ACTION_PROCESS, normalized, cwd);
        if (!policy_result.allowed) return setPolicyDeniedError(policy_result.deny_reason);
    }
    return 0;
}

/// Open a non-PTY process session using `/bin/sh -c <command>`.
pub fn talu_process_open(
    command: ?[*:0]const u8,
    cwd: ?[*:0]const u8,
    policy: ?*policy_api.TaluAgentPolicy,
    runtime_mode: c_int,
    sandbox_backend: c_int,
    out_process: ?*?*TaluProcess,
) callconv(.c) i32 {
    capi_error.clearError();

    const out = out_process orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_process is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;

    const command_slice = std.mem.sliceTo(command orelse {
        capi_error.setErrorWithCode(.invalid_argument, "command is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }, 0);
    if (command_slice.len == 0) {
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

    const policy_rc = enforceProcessOpenPolicy(policy, command_slice, cwd_slice);
    if (policy_rc != 0) return policy_rc;

    const mode = runtime_api.parseRuntimeMode(runtime_mode) catch |err| {
        capi_error.setError(err, "invalid runtime mode", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const backend = runtime_api.parseBackend(sandbox_backend) catch |err| {
        capi_error.setError(err, "invalid sandbox backend", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    runtime_api.validate(mode, backend) catch |err| {
        return setStrictError(err);
    };

    var owned_exec_profile: ?sandbox.profile.ExecProfile = null;
    defer if (owned_exec_profile) |*p| p.deinit();
    var exec_profile_ptr: ?*const sandbox.profile.ExecProfile = null;
    if (mode == .strict) {
        if (policy_api.preparedExecProfile(policy, ACTION_PROCESS, cwd_slice)) |prepared| {
            exec_profile_ptr = prepared;
        } else {
            owned_exec_profile = sandbox.profile.buildExecProfile(
                allocator,
                policy_api.toPolicyConst(policy),
                .{
                    .action = ACTION_PROCESS,
                    .cwd = cwd_slice,
                    .include_shell_paths = true,
                },
            ) catch |err| {
                capi_error.setError(err, "failed to build strict process profile", .{});
                return @intFromEnum(error_codes.ErrorCode.policy_strict_setup_failed);
            };
            exec_profile_ptr = &owned_exec_profile.?;
        }
    }

    const session_ptr = process.session.ProcessSession.open(
        allocator,
        command_slice,
        cwd_slice,
        .{
            .mode = mode,
            .backend = backend,
            .exec_profile = exec_profile_ptr,
        },
    ) catch |err| {
        capi_error.setError(err, "failed to open process session", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    out.* = @ptrCast(session_ptr);
    return 0;
}

/// Close a process session.
pub fn talu_process_close(process_handle: ?*TaluProcess) callconv(.c) void {
    capi_error.clearError();
    const session_ptr: *process.session.ProcessSession = @ptrCast(@alignCast(process_handle orelse return));
    session_ptr.close();
}

/// Write bytes to process stdin.
pub fn talu_process_write(
    process_handle: ?*TaluProcess,
    data: ?[*]const u8,
    len: usize,
) callconv(.c) i32 {
    capi_error.clearError();

    const session_ptr = toProcess(process_handle) catch |err| {
        capi_error.setError(err, "invalid process handle", .{});
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
        capi_error.setError(err, "failed to write process session", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Read bytes from process stdout/stderr into caller buffer.
pub fn talu_process_read(
    process_handle: ?*TaluProcess,
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

    const session_ptr = toProcess(process_handle) catch |err| {
        capi_error.setError(err, "invalid process handle", .{});
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
        capi_error.setError(err, "failed to read process session", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    out_read_ptr.* = n;
    return 0;
}

/// Send POSIX signal to process.
pub fn talu_process_signal(
    process_handle: ?*TaluProcess,
    sig: u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const session_ptr = toProcess(process_handle) catch |err| {
        capi_error.setError(err, "invalid process handle", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    session_ptr.sendSignal(sig) catch |err| {
        capi_error.setError(err, "failed to signal process session", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Return true if process is still alive.
pub fn talu_process_alive(process_handle: ?*TaluProcess) callconv(.c) bool {
    capi_error.clearError();
    const session_ptr = toProcess(process_handle) catch |err| {
        capi_error.setError(err, "invalid process handle", .{});
        return false;
    };
    return session_ptr.isAlive() catch |err| {
        capi_error.setError(err, "process session closed", .{});
        return false;
    };
}

/// Return process exit code when available.
pub fn talu_process_exit_code(
    process_handle: ?*TaluProcess,
    out_code: ?*i32,
    out_has_code: ?*bool,
) callconv(.c) i32 {
    capi_error.clearError();
    if (out_has_code) |p| p.* = false;
    if (out_code) |p| p.* = 0;

    const out_code_ptr = out_code orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_code is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const out_has_code_ptr = out_has_code orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_has_code is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const session_ptr = toProcess(process_handle) catch |err| {
        capi_error.setError(err, "invalid process handle", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    const code = session_ptr.getExitCode() catch |err| {
        capi_error.setError(err, "failed to read process exit code", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    if (code) |value| {
        out_code_ptr.* = value;
        out_has_code_ptr.* = true;
    }
    return 0;
}

test "talu_process_open write read close roundtrip" {
    var handle: ?*TaluProcess = null;
    try std.testing.expectEqual(
        @as(i32, 0),
        talu_process_open(
            "while IFS= read -r line; do echo \"$line\"; [ \"$line\" = \"quit\" ] && break; done",
            null,
            null,
            @intFromEnum(runtime_api.TaluAgentRuntimeMode.host),
            @intFromEnum(runtime_api.TaluSandboxBackend.linux_local),
            &handle,
        ),
    );
    defer talu_process_close(handle);

    const command = "hello\nquit\n";
    try std.testing.expectEqual(@as(i32, 0), talu_process_write(handle, command.ptr, command.len));

    var read_buffer: [512]u8 = undefined;
    var output = std.ArrayList(u8).empty;
    defer output.deinit(std.testing.allocator);

    var n: usize = 0;
    var guard: usize = 0;
    while (talu_process_alive(handle) and guard < 50_000) : (guard += 1) {
        try std.testing.expectEqual(@as(i32, 0), talu_process_read(handle, read_buffer[0..].ptr, read_buffer.len, &n));
        if (n > 0) try output.appendSlice(std.testing.allocator, read_buffer[0..n]);
    }
    while (true) {
        try std.testing.expectEqual(@as(i32, 0), talu_process_read(handle, read_buffer[0..].ptr, read_buffer.len, &n));
        if (n == 0) break;
        try output.appendSlice(std.testing.allocator, read_buffer[0..n]);
    }

    try std.testing.expect(std.mem.indexOf(u8, output.items, "hello") != null);
}

test "talu_process_open enforces agent policy" {
    var policy_handle: ?*policy_api.TaluAgentPolicy = null;
    const policy_json =
        \\{"default":"deny","statements":[{"effect":"allow","action":"tool.fs.read","resource":"**"}]}
    ;
    try std.testing.expectEqual(
        @as(i32, 0),
        policy_api.talu_agent_policy_create(policy_json.ptr, policy_json.len, &policy_handle),
    );
    defer policy_api.talu_agent_policy_free(policy_handle);

    var handle: ?*TaluProcess = null;
    const rc = talu_process_open(
        "echo hi",
        null,
        policy_handle,
        @intFromEnum(runtime_api.TaluAgentRuntimeMode.host),
        @intFromEnum(runtime_api.TaluSandboxBackend.linux_local),
        &handle,
    );
    try std.testing.expectEqual(
        @as(i32, @intFromEnum(error_codes.ErrorCode.policy_denied_exec)),
        rc,
    );
    try std.testing.expect(handle == null);
}
