//! C API for the tool call firewall (policy engine).
//!
//! Provides create/free/evaluate/attach functions for IAM-style policies.
//! See core/src/policy/ for the implementation.

const std = @import("std");
const policy_mod = @import("../policy/root.zig");
const chat_mod = @import("../responses/chat.zig");
const capi_error = @import("error.zig");
const error_codes = @import("error_codes.zig");

const Policy = policy_mod.Policy;
const Chat = chat_mod.Chat;

const allocator = std.heap.c_allocator;

/// Opaque handle for a Policy object.
pub const PolicyHandle = opaque {};

/// Parse a policy from JSON. Caller owns the returned handle.
///
/// The JSON must follow the IAM-style schema:
/// ```json
/// {"default":"deny","statements":[{"effect":"allow","action":"ls *"}]}
/// ```
///
/// Returns 0 on success, error code on failure.
pub export fn talu_policy_create(
    json_ptr: ?[*]const u8,
    json_len: usize,
    out_policy: ?*?*PolicyHandle,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_policy orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_policy is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;

    const json = if (json_ptr) |ptr| ptr[0..json_len] else {
        capi_error.setError(error.InvalidArgument, "json_ptr is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const policy_ptr = allocator.create(Policy) catch |err| {
        capi_error.setError(err, "failed to allocate policy", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    policy_ptr.* = policy_mod.parsePolicy(allocator, json) catch |err| {
        allocator.destroy(policy_ptr);
        capi_error.setError(err, "failed to parse policy JSON", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    out.* = @ptrCast(policy_ptr);
    return 0;
}

/// Free a policy handle.
pub export fn talu_policy_free(handle: ?*PolicyHandle) callconv(.c) void {
    const policy: *Policy = @ptrCast(@alignCast(handle orelse return));
    policy.deinit();
    allocator.destroy(policy);
}

/// Evaluate an action against a policy.
///
/// Returns 0 for allow, 1 for deny.
pub export fn talu_policy_evaluate(
    handle: ?*const PolicyHandle,
    action_ptr: ?[*]const u8,
    action_len: usize,
) callconv(.c) u8 {
    const policy: *const Policy = @ptrCast(@alignCast(handle orelse return 1));
    const action = if (action_ptr) |ptr| ptr[0..action_len] else return 1;
    return @intFromEnum(policy.evaluate(action));
}

/// Get the policy mode (0 = enforce, 1 = audit).
pub export fn talu_policy_get_mode(handle: ?*const PolicyHandle) callconv(.c) u8 {
    const policy: *const Policy = @ptrCast(@alignCast(handle orelse return 0));
    return @intFromEnum(policy.mode);
}

/// Attach a policy to a chat. The policy must outlive the chat.
///
/// Pass null to detach the current policy.
/// Returns 0 on success, error code on failure.
pub export fn talu_chat_set_policy(
    chat_handle: ?*@import("responses.zig").ChatHandle,
    policy_handle: ?*const PolicyHandle,
) callconv(.c) i32 {
    capi_error.clearError();
    const chat_state: *Chat = @ptrCast(@alignCast(chat_handle orelse {
        capi_error.setError(error.InvalidHandle, "chat handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }));
    const policy: ?*const Policy = if (policy_handle) |h| @ptrCast(@alignCast(h)) else null;
    chat_state.policy = policy;
    return 0;
}

// =============================================================================
// Fuzz Tests
// =============================================================================

test "fuzz talu_policy_create" {
    try std.testing.fuzz({}, struct {
        fn testOne(_: void, input: []const u8) !void {
            var out_policy: ?*PolicyHandle = null;
            const rc = talu_policy_create(input.ptr, input.len, &out_policy);
            if (rc != 0) {
                try std.testing.expect(out_policy == null);
            }
            if (out_policy) |policy_handle| {
                talu_policy_free(policy_handle);
            }
        }
    }.testOne, .{});
}

test "talu_policy_create invalid json maps to invalid_argument" {
    var out_policy: ?*PolicyHandle = null;
    const rc = talu_policy_create("{", 1, &out_policy);
    try std.testing.expect(rc != 0);
    try std.testing.expect(out_policy == null);
    try std.testing.expectEqual(@as(i32, @intFromEnum(error_codes.ErrorCode.invalid_argument)), capi_error.talu_last_error_code());
}
