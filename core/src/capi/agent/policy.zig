//! C API bridge for agent runtime policy handles.

const std = @import("std");
const policy_mod = @import("../../agent/policy/root.zig");
const sandbox = @import("../../agent/sandbox/root.zig");
const capi_error = @import("../error.zig");
const error_codes = @import("../error_codes.zig");

const allocator = std.heap.c_allocator;
const Policy = policy_mod.Policy;
const ACTION_EXEC = "tool.exec";
const ACTION_SHELL = "tool.shell";
const ACTION_PROCESS = "tool.process";

pub const TaluAgentPolicy = opaque {};
pub const ProcessDenyReason = policy_mod.ProcessDenyReason;
pub const TerminalShellMode = policy_mod.TerminalShellMode;

pub const ProcessPolicyResult = struct {
    allowed: bool,
    deny_reason: ?ProcessDenyReason = null,
};

pub const FileSubtreeDecision = enum {
    allow,
    deny,
};

const PreparedProfiles = struct {
    cwd: ?[]u8 = null,
    exec: ?sandbox.profile.ExecProfile = null,
    shell: ?sandbox.profile.ExecProfile = null,
    process: ?sandbox.profile.ExecProfile = null,

    fn deinit(self: *PreparedProfiles) void {
        if (self.exec) |*profile| profile.deinit();
        if (self.shell) |*profile| profile.deinit();
        if (self.process) |*profile| profile.deinit();
        if (self.cwd) |value| allocator.free(value);
        self.* = .{};
    }
};

const PolicyHandle = struct {
    policy: Policy,
    prepared: PreparedProfiles = .{},

    fn deinit(self: *PolicyHandle) void {
        self.prepared.deinit();
        self.policy.deinit();
    }
};

fn toHandle(handle: ?*TaluAgentPolicy) ?*PolicyHandle {
    if (handle) |h| return @ptrCast(@alignCast(h));
    return null;
}

pub fn toPolicyConst(handle: ?*TaluAgentPolicy) ?*const Policy {
    const h = toHandle(handle) orelse return null;
    return &h.policy;
}

pub fn preparedExecProfile(
    policy: ?*TaluAgentPolicy,
    action: []const u8,
    cwd: ?[]const u8,
) ?*const sandbox.profile.ExecProfile {
    const handle = toHandle(policy) orelse return null;
    if (!cwdMatches(handle.prepared.cwd, cwd)) return null;
    if (std.mem.eql(u8, action, ACTION_EXEC)) return if (handle.prepared.exec) |*p| p else null;
    if (std.mem.eql(u8, action, ACTION_SHELL)) return if (handle.prepared.shell) |*p| p else null;
    if (std.mem.eql(u8, action, ACTION_PROCESS)) return if (handle.prepared.process) |*p| p else null;
    return null;
}

pub fn prepareRuntimeProfiles(policy: ?*TaluAgentPolicy, cwd: ?[]const u8) !void {
    const handle = toHandle(policy) orelse return;
    var canonical_cwd: ?[]u8 = null;
    if (cwd) |value| {
        if (value.len == 0) return error.StrictSetupFailed;
        canonical_cwd = std.fs.cwd().realpathAlloc(allocator, value) catch return error.StrictSetupFailed;
    }
    errdefer if (canonical_cwd) |value| allocator.free(value);

    var exec_profile = sandbox.profile.buildExecProfile(allocator, &handle.policy, .{
        .action = ACTION_EXEC,
        .cwd = canonical_cwd,
        .include_shell_paths = true,
    }) catch |err| return mapStrictProfileBuildError(err);
    errdefer exec_profile.deinit();

    var shell_profile = sandbox.profile.buildExecProfile(allocator, &handle.policy, .{
        .action = ACTION_SHELL,
        .cwd = canonical_cwd,
        .include_shell_paths = true,
    }) catch |err| return mapStrictProfileBuildError(err);
    errdefer shell_profile.deinit();

    var process_profile = sandbox.profile.buildExecProfile(allocator, &handle.policy, .{
        .action = ACTION_PROCESS,
        .cwd = canonical_cwd,
        .include_shell_paths = true,
    }) catch |err| return mapStrictProfileBuildError(err);
    errdefer process_profile.deinit();

    handle.prepared.deinit();
    handle.prepared.cwd = canonical_cwd;
    handle.prepared.exec = exec_profile;
    handle.prepared.shell = shell_profile;
    handle.prepared.process = process_profile;
}

fn mapStrictProfileBuildError(err: anyerror) anyerror {
    return switch (err) {
        error.StrictDeferred => error.StrictDeferred,
        else => error.StrictSetupFailed,
    };
}

fn cwdMatches(prepared_cwd: ?[]const u8, requested_cwd: ?[]const u8) bool {
    if (prepared_cwd == null and requested_cwd == null) return true;
    if (prepared_cwd == null or requested_cwd == null) return false;
    return std.mem.eql(u8, prepared_cwd.?, requested_cwd.?);
}

pub fn clampTimeoutMs(policy: ?*TaluAgentPolicy, requested_ms: u64) u64 {
    const p = toPolicyConst(policy) orelse return requested_ms;
    return p.clampTimeoutMs(requested_ms);
}

pub fn enforceFilePolicy(
    policy: ?*TaluAgentPolicy,
    action: []const u8,
    resource: []const u8,
    is_dir: bool,
) bool {
    const p = toPolicyConst(policy) orelse return true;
    return policy_mod.checkFileAction(p, action, resource, is_dir);
}

pub fn fileSubtreeDecision(
    policy: ?*TaluAgentPolicy,
    action: []const u8,
    directory_resource: []const u8,
) ?FileSubtreeDecision {
    const p = toPolicyConst(policy) orelse return null;
    const effect = policy_mod.checkFileDescendantSubtree(p, action, directory_resource) orelse return null;
    return switch (effect) {
        .allow => .allow,
        .deny => .deny,
    };
}

pub fn enforceProcessPolicy(
    policy: ?*TaluAgentPolicy,
    action: []const u8,
    command: ?[]const u8,
    cwd: ?[]const u8,
) ProcessPolicyResult {
    const p = toPolicyConst(policy) orelse return .{ .allowed = true };
    const result = policy_mod.checkProcessAction(p, action, command, cwd);
    return .{
        .allowed = result.allowed,
        .deny_reason = result.deny_reason,
    };
}

pub fn terminalShellMode(policy: ?*TaluAgentPolicy) TerminalShellMode {
    const p = toPolicyConst(policy) orelse return .host;
    return p.terminal_shell_mode;
}

pub fn talu_agent_policy_create(
    json: ?[*]const u8,
    len: usize,
    out_policy: ?*?*TaluAgentPolicy,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_policy orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_policy is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;

    const json_slice = if (json) |ptr| ptr[0..len] else {
        capi_error.setErrorWithCode(.invalid_argument, "json is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    if (json_slice.len == 0) {
        capi_error.setErrorWithCode(.invalid_argument, "json is empty", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }

    const policy_ptr = allocator.create(PolicyHandle) catch |err| {
        capi_error.setError(err, "failed to allocate policy", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    const parsed = policy_mod.parsePolicy(allocator, json_slice) catch |err| {
        allocator.destroy(policy_ptr);
        capi_error.setError(err, "failed to parse agent policy", .{});
        return @intFromEnum(error_codes.ErrorCode.policy_invalid);
    };
    policy_ptr.* = .{ .policy = parsed };

    out.* = @ptrCast(policy_ptr);
    return 0;
}

pub fn talu_agent_policy_free(policy: ?*TaluAgentPolicy) callconv(.c) void {
    const p = toHandle(policy) orelse return;
    p.deinit();
    allocator.destroy(p);
}

pub fn talu_agent_policy_prepare_runtime(
    policy: ?*TaluAgentPolicy,
    cwd: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const cwd_slice: ?[]const u8 = if (cwd) |value| blk: {
        const slice = std.mem.sliceTo(value, 0);
        if (slice.len == 0) {
            capi_error.setErrorWithCode(.invalid_argument, "cwd is empty", .{});
            return @intFromEnum(error_codes.ErrorCode.invalid_argument);
        }
        break :blk slice;
    } else null;

    prepareRuntimeProfiles(policy, cwd_slice) catch |err| {
        capi_error.setError(err, "failed to precompile strict runtime profiles", .{});
        return @intFromEnum(error_codes.ErrorCode.policy_strict_setup_failed);
    };
    return 0;
}

pub fn talu_agent_policy_check_action(
    policy: ?*TaluAgentPolicy,
    action: ?[*:0]const u8,
    command: ?[*:0]const u8,
    cwd: ?[*:0]const u8,
    resource: ?[*:0]const u8,
    timeout_ms: u64,
    out_allowed: ?*bool,
    out_reason: ?*?[*]const u8,
    out_reason_len: ?*usize,
) callconv(.c) i32 {
    capi_error.clearError();
    if (out_allowed) |ptr| ptr.* = false;
    if (out_reason) |ptr| ptr.* = null;
    if (out_reason_len) |ptr| ptr.* = 0;

    const out_allowed_ptr = out_allowed orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_allowed is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const action_slice = std.mem.sliceTo(action orelse {
        capi_error.setErrorWithCode(.invalid_argument, "action is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }, 0);
    if (action_slice.len == 0) {
        capi_error.setErrorWithCode(.invalid_argument, "action is empty", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }

    const command_slice = if (command) |ptr| std.mem.sliceTo(ptr, 0) else null;
    const cwd_slice = if (cwd) |ptr| std.mem.sliceTo(ptr, 0) else null;
    const resource_slice = if (resource) |ptr| std.mem.sliceTo(ptr, 0) else null;

    const p = toPolicyConst(policy);
    if (p == null) {
        out_allowed_ptr.* = true;
        return 0;
    }

    _ = p.?.clampTimeoutMs(timeout_ms);
    const effect = p.?.evaluateDetailed(.{
        .action = action_slice,
        .command = command_slice,
        .cwd = cwd_slice,
        .resource = resource_slice,
    });
    out_allowed_ptr.* = effect == .allow;

    if (!out_allowed_ptr.*) {
        const reason = "policy denied action";
        if (out_reason) |ptr| ptr.* = reason.ptr;
        if (out_reason_len) |ptr| ptr.* = reason.len;
    }
    return 0;
}

pub fn talu_agent_policy_check_file(
    policy: ?*TaluAgentPolicy,
    action: ?[*:0]const u8,
    resource: ?[*:0]const u8,
    is_dir: bool,
    out_allowed: ?*bool,
) callconv(.c) i32 {
    capi_error.clearError();
    if (out_allowed) |ptr| ptr.* = false;

    const out = out_allowed orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_allowed is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const action_slice = std.mem.sliceTo(action orelse {
        capi_error.setErrorWithCode(.invalid_argument, "action is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }, 0);
    const resource_slice = std.mem.sliceTo(resource orelse {
        capi_error.setErrorWithCode(.invalid_argument, "resource is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }, 0);

    out.* = enforceFilePolicy(policy, action_slice, resource_slice, is_dir);
    return 0;
}

pub fn talu_agent_policy_check_process(
    policy: ?*TaluAgentPolicy,
    action: ?[*:0]const u8,
    command: ?[*:0]const u8,
    cwd: ?[*:0]const u8,
    out_allowed: ?*bool,
) callconv(.c) i32 {
    capi_error.clearError();
    if (out_allowed) |ptr| ptr.* = false;

    const out = out_allowed orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_allowed is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const action_slice = std.mem.sliceTo(action orelse {
        capi_error.setErrorWithCode(.invalid_argument, "action is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }, 0);
    const command_slice = if (command) |ptr| std.mem.sliceTo(ptr, 0) else null;
    const cwd_slice = if (cwd) |ptr| std.mem.sliceTo(ptr, 0) else null;

    out.* = enforceProcessPolicy(policy, action_slice, command_slice, cwd_slice).allowed;
    return 0;
}

test "talu_agent_policy_create check and free" {
    var handle: ?*TaluAgentPolicy = null;
    const policy_json =
        \\{"default":"deny","statements":[{"effect":"allow","action":"tool.exec","command":"echo *"}]}
    ;
    const rc = talu_agent_policy_create(policy_json.ptr, policy_json.len, &handle);
    defer talu_agent_policy_free(handle);
    try std.testing.expectEqual(@as(i32, 0), rc);
    try std.testing.expect(handle != null);

    var allowed = false;
    const check_rc = talu_agent_policy_check_action(
        handle,
        "tool.exec",
        "echo hi",
        null,
        null,
        1_000,
        &allowed,
        null,
        null,
    );
    try std.testing.expectEqual(@as(i32, 0), check_rc);
    try std.testing.expect(allowed);
}

test "talu_agent_policy_prepare_runtime precompiles strict profiles" {
    var handle: ?*TaluAgentPolicy = null;
    const policy_json =
        \\{
        \\  "default":"deny",
        \\  "statements":[
        \\    {"effect":"allow","action":"tool.exec","command":"echo *"},
        \\    {"effect":"allow","action":"tool.shell","command":"echo *"},
        \\    {"effect":"allow","action":"tool.process","command":"echo *"}
        \\  ]
        \\}
    ;
    const rc = talu_agent_policy_create(policy_json.ptr, policy_json.len, &handle);
    defer talu_agent_policy_free(handle);
    try std.testing.expectEqual(@as(i32, 0), rc);

    const prep_rc = talu_agent_policy_prepare_runtime(handle, null);
    try std.testing.expectEqual(@as(i32, 0), prep_rc);
    try std.testing.expect(preparedExecProfile(handle, ACTION_EXEC, null) != null);
    try std.testing.expect(preparedExecProfile(handle, ACTION_SHELL, null) != null);
    try std.testing.expect(preparedExecProfile(handle, ACTION_PROCESS, null) != null);
}

test "prepareRuntimeProfiles defers cwd-dependent missing allow paths" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const cwd = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(cwd);

    var handle: ?*TaluAgentPolicy = null;
    const policy_json =
        \\{
        \\  "default":"deny",
        \\  "statements":[
        \\    {"effect":"allow","action":"tool.exec","command":"echo *"},
        \\    {"effect":"allow","action":"tool.shell","command":"echo *"},
        \\    {"effect":"allow","action":"tool.process","command":"echo *"},
        \\    {"effect":"allow","action":"tool.fs.write","resource":"allowed/**"}
        \\  ]
        \\}
    ;
    const rc = talu_agent_policy_create(policy_json.ptr, policy_json.len, &handle);
    defer talu_agent_policy_free(handle);
    try std.testing.expectEqual(@as(i32, 0), rc);

    try std.testing.expectError(error.StrictDeferred, prepareRuntimeProfiles(handle, cwd));
}
