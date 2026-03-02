//! C API bridge for agent runtime policy handles.

const std = @import("std");
const policy_mod = @import("../../agent/policy/root.zig");
const capi_error = @import("../error.zig");
const error_codes = @import("../error_codes.zig");

const allocator = std.heap.c_allocator;
const Policy = policy_mod.Policy;

pub const TaluAgentPolicy = opaque {};
pub const ProcessDenyReason = policy_mod.ProcessDenyReason;

pub const ProcessPolicyResult = struct {
    allowed: bool,
    deny_reason: ?ProcessDenyReason = null,
};

pub fn toPolicyConst(handle: ?*TaluAgentPolicy) ?*const Policy {
    if (handle) |h| {
        return @ptrCast(@alignCast(h));
    }
    return null;
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

    const policy_ptr = allocator.create(Policy) catch |err| {
        capi_error.setError(err, "failed to allocate policy", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    policy_ptr.* = policy_mod.parsePolicy(allocator, json_slice) catch |err| {
        allocator.destroy(policy_ptr);
        capi_error.setError(err, "failed to parse agent policy", .{});
        return @intFromEnum(error_codes.ErrorCode.policy_invalid);
    };

    out.* = @ptrCast(policy_ptr);
    return 0;
}

pub fn talu_agent_policy_free(policy: ?*TaluAgentPolicy) callconv(.c) void {
    const p: *Policy = @ptrCast(@alignCast(policy orelse return));
    p.deinit();
    allocator.destroy(p);
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
