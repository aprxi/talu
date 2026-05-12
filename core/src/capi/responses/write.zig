//! Responses C API append, insert, remove, and pop operations.
//!
//! Thread safety: NOT thread-safe. All access must be from a single thread.

const std = @import("std");
const conversation_mod = @import("../../responses/conversation/root.zig");
const types = @import("types.zig");
const Conversation = conversation_mod.Conversation;
const Item = conversation_mod.Item;
const MessageRole = conversation_mod.MessageRole;
const ContentType = conversation_mod.ContentType;
const ResponsesHandle = types.ResponsesHandle;

const capi_error = @import("../error.zig");
const error_codes = @import("../error_codes.zig");
const signal_guard = @import("../signal_guard.zig");

/// Context for signal-guarded magic number check.
const MagicCheckContext = struct {
    ptr: *const u64,
    expected_magic: u64,
    result: bool,
};

/// Callback for signal-guarded magic check.
fn magicCheckCallback(ctx_ptr: *anyopaque) callconv(.c) i32 {
    const ctx: *MagicCheckContext = @ptrCast(@alignCast(ctx_ptr));
    ctx.result = (ctx.ptr.* == ctx.expected_magic);
    return 0;
}

/// Safely check if a Conversation pointer is valid.
/// Uses signal guard to catch SIGSEGV/SIGBUS if pointer is unmapped.
/// Returns true if valid, false if invalid or unmapped.
fn isConversationValidSafe(conv: *Conversation) bool {
    var guard = signal_guard.SignalGuard.init();
    defer guard.deinit();

    var ctx = MagicCheckContext{
        .ptr = &conv._magic,
        .expected_magic = Conversation.MAGIC_VALID,
        .result = false,
    };

    const call_result = guard.call(&magicCheckCallback, @ptrCast(&ctx));
    if (call_result == null) {
        // Signal was caught - pointer is unmapped
        return false;
    }
    return ctx.result;
}

/// Append message based on role code.
/// role: 0=system, 1=user, 2=assistant, 3=developer
/// Returns item handle on success, or error code (negative).
fn appendMessageByRole(conv: *Conversation, role: u8, content: []const u8) !*Item {
    return switch (role) {
        0 => try conv.appendSystemMessage(content),
        1 => try conv.appendUserMessage(content),
        3 => try conv.appendDeveloperMessage(content),
        2 => blk: {
            const msg = try conv.appendAssistantMessage();
            if (content.len > 0) try conv.appendTextContent(msg, content);
            break :blk msg;
        },
        else => error.InvalidArgument,
    };
}

/// Append message based on role code with hidden flag.
fn appendMessageByRoleHidden(conv: *Conversation, role: u8, content: []const u8, hidden: bool) !*Item {
    return switch (role) {
        0 => try conv.appendMessageWithHidden(.system, .input_text, content, hidden),
        1 => try conv.appendMessageWithHidden(.user, .input_text, content, hidden),
        3 => try conv.appendMessageWithHidden(.developer, .input_text, content, hidden),
        2 => blk: {
            const msg = try conv.appendAssistantMessage();
            if (content.len > 0) try conv.appendTextContent(msg, content);
            msg.hidden = hidden;
            break :blk msg;
        },
        else => error.InvalidArgument,
    };
}

/// Append a message item to the conversation.
/// role: 0=system, 1=user, 2=assistant, 3=developer
/// content: Text content (not null-terminated, use with length)
/// Returns the item index on success, negative error code on failure.
pub export fn talu_responses_append_message(
    handle: ?*ResponsesHandle,
    role: u8,
    content_ptr: ?[*]const u8,
    content_len: usize,
) callconv(.c) i64 {
    capi_error.clearError();
    const conv: *Conversation = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "conversation handle is null", .{});
        return -@as(i64, @intFromEnum(error_codes.ErrorCode.invalid_handle));
    }));

    // Validate magic number to detect use-after-free or memory corruption
    if (!conv.isValid()) {
        capi_error.setError(error.InvalidHandle, "conversation handle is corrupted or freed (magic=0x{x:016})", .{conv._magic});
        return -@as(i64, @intFromEnum(error_codes.ErrorCode.invalid_handle));
    }

    const content = if (content_ptr) |p| p[0..content_len] else "";

    _ = appendMessageByRole(conv, role, content) catch |err| {
        if (err == error.InvalidArgument) {
            capi_error.setError(error.InvalidArgument, "invalid role: {d}", .{role});
        } else {
            capi_error.setError(err, "failed to append message", .{});
        }
        return -@as(i64, @intFromEnum(error_codes.errorToCode(err)));
    };

    return @intCast(conv.items_list.items.len - 1);
}

/// Append a message item with a hidden flag.
/// role: 0=system, 1=user, 2=assistant, 3=developer
/// hidden: true to hide from UI history
/// Returns the item index on success, negative error code on failure.
pub export fn talu_responses_append_message_hidden(
    handle: ?*ResponsesHandle,
    role: u8,
    content_ptr: ?[*]const u8,
    content_len: usize,
    hidden: bool,
) callconv(.c) i64 {
    capi_error.clearError();
    const conv: *Conversation = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "conversation handle is null", .{});
        return -@as(i64, @intFromEnum(error_codes.ErrorCode.invalid_handle));
    }));

    // Validate magic number to detect use-after-free or memory corruption.
    // Use signal-safe validation to handle unmapped memory without crashing.
    if (!isConversationValidSafe(conv)) {
        capi_error.setError(error.InvalidHandle, "conversation handle corrupted, freed, or unmapped (ptr=0x{x:016})", .{@intFromPtr(conv)});
        return -@as(i64, @intFromEnum(error_codes.ErrorCode.invalid_handle));
    }

    const content = if (content_ptr) |p| p[0..content_len] else "";

    _ = appendMessageByRoleHidden(conv, role, content, hidden) catch |err| {
        if (err == error.InvalidArgument) {
            capi_error.setError(error.InvalidArgument, "invalid role: {d}", .{role});
        } else {
            capi_error.setError(err, "failed to append hidden message", .{});
        }
        return -@as(i64, @intFromEnum(error_codes.errorToCode(err)));
    };

    return @intCast(conv.items_list.items.len - 1);
}

/// Append a function call item to the conversation.
/// Returns the item index on success, negative error code on failure.
pub export fn talu_responses_append_function_call(
    handle: ?*ResponsesHandle,
    call_id: ?[*:0]const u8,
    name: ?[*:0]const u8,
    arguments_ptr: ?[*]const u8,
    arguments_len: usize,
) callconv(.c) i64 {
    capi_error.clearError();
    const conv: *Conversation = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "conversation handle is null", .{});
        return -@as(i64, @intFromEnum(error_codes.ErrorCode.invalid_handle));
    }));

    const cid = if (call_id) |c| std.mem.sliceTo(c, 0) else "";
    const nm = if (name) |n| std.mem.sliceTo(n, 0) else "";
    const args = if (arguments_ptr) |p| p[0..arguments_len] else "";

    const item = conv.appendFunctionCall(cid, nm) catch |err| {
        capi_error.setError(err, "failed to append function call", .{});
        return -@as(i64, @intFromEnum(error_codes.errorToCode(err)));
    };

    // Set arguments if provided
    if (args.len > 0) {
        conv.setFunctionCallArguments(item, args) catch |err| {
            capi_error.setError(err, "failed to set function call arguments", .{});
            return -@as(i64, @intFromEnum(error_codes.errorToCode(err)));
        };
    }

    return @intCast(conv.items_list.items.len - 1);
}

/// Append a function call output item to the conversation.
/// Returns the item index on success, negative error code on failure.
pub export fn talu_responses_append_function_call_output(
    handle: ?*ResponsesHandle,
    call_id: ?[*:0]const u8,
    output_ptr: ?[*]const u8,
    output_len: usize,
) callconv(.c) i64 {
    capi_error.clearError();
    const conv: *Conversation = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "conversation handle is null", .{});
        return -@as(i64, @intFromEnum(error_codes.ErrorCode.invalid_handle));
    }));

    const cid = if (call_id) |c| std.mem.sliceTo(c, 0) else "";
    const output = if (output_ptr) |p| p[0..output_len] else "";

    _ = conv.appendFunctionCallOutput(cid, output) catch |err| {
        capi_error.setError(err, "failed to append function call output", .{});
        return -@as(i64, @intFromEnum(error_codes.errorToCode(err)));
    };

    return @intCast(conv.items_list.items.len - 1);
}

/// Append text content to an existing message item (for streaming).
/// item_index: Index of the item to append to (must be a message)
/// Returns 0 on success, non-zero error code on failure.
pub export fn talu_responses_append_text_content(
    handle: ?*ResponsesHandle,
    item_index: usize,
    content_ptr: ?[*]const u8,
    content_len: usize,
) callconv(.c) i32 {
    capi_error.clearError();
    const conv: *Conversation = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "conversation handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }));

    if (item_index >= conv.items_list.items.len) {
        capi_error.setError(error.InvalidArgument, "item index {d} out of bounds", .{item_index});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }

    const content = if (content_ptr) |p| p[0..content_len] else "";
    const item = &conv.items_list.items[item_index];

    conv.appendTextContent(item, content) catch |err| {
        capi_error.setError(err, "failed to append text content", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    return 0;
}

/// Remove the last item from the conversation.
/// Returns 0 on success, non-zero error code on failure.
pub export fn talu_responses_pop(
    handle: ?*ResponsesHandle,
) callconv(.c) i32 {
    capi_error.clearError();
    const conv: *Conversation = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "conversation handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }));

    if (conv.items_list.items.len == 0) {
        capi_error.setError(error.InvalidArgument, "conversation is empty, cannot pop", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }

    const last_index = conv.items_list.items.len - 1;
    if (!conv.deleteItem(last_index)) {
        capi_error.setError(error.InternalError, "failed to delete last item", .{});
        return @intFromEnum(error_codes.ErrorCode.internal_error);
    }

    return 0;
}

/// Remove an item at the specified index.
/// Returns 0 on success, non-zero error code on failure.
pub export fn talu_responses_remove(
    handle: ?*ResponsesHandle,
    index: usize,
) callconv(.c) i32 {
    capi_error.clearError();
    const conv: *Conversation = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "conversation handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }));

    if (index >= conv.items_list.items.len) {
        capi_error.setError(error.InvalidArgument, "index {d} out of bounds", .{index});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }

    if (!conv.deleteItem(index)) {
        capi_error.setError(error.InternalError, "failed to delete item", .{});
        return @intFromEnum(error_codes.ErrorCode.internal_error);
    }

    return 0;
}

/// Insert a message item at the specified index.
/// role: 0=system, 1=user, 2=assistant, 3=developer
/// content: Text content (not null-terminated, use with length)
/// Returns the item index on success, negative error code on failure.
pub export fn talu_responses_insert_message(
    handle: ?*ResponsesHandle,
    index: usize,
    role: u8,
    content_ptr: ?[*]const u8,
    content_len: usize,
) callconv(.c) i64 {
    capi_error.clearError();
    const conv: *Conversation = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "conversation handle is null", .{});
        return -@as(i64, @intFromEnum(error_codes.ErrorCode.invalid_handle));
    }));

    const content = if (content_ptr) |p| p[0..content_len] else "";

    const msg_role: MessageRole = switch (role) {
        0 => .system,
        1 => .user,
        2 => .assistant,
        3 => .developer,
        else => {
            capi_error.setError(error.InvalidArgument, "invalid role: {d}", .{role});
            return -@as(i64, @intFromEnum(error_codes.ErrorCode.invalid_argument));
        },
    };

    // Determine content type based on role
    const content_type: ContentType = if (msg_role == .assistant) .output_text else .input_text;

    _ = conv.insertMessage(index, msg_role, content_type, content) catch |err| {
        capi_error.setError(err, "failed to insert message", .{});
        return -@as(i64, @intFromEnum(error_codes.errorToCode(err)));
    };

    return @intCast(index);
}

/// Insert a message item at the specified index with a hidden flag.
/// role: 0=system, 1=user, 2=assistant, 3=developer
/// hidden: true to hide from UI history
/// Returns the item index on success, negative error code on failure.
pub export fn talu_responses_insert_message_hidden(
    handle: ?*ResponsesHandle,
    index: usize,
    role: u8,
    content_ptr: ?[*]const u8,
    content_len: usize,
    hidden: bool,
) callconv(.c) i64 {
    capi_error.clearError();
    const conv: *Conversation = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "conversation handle is null", .{});
        return -@as(i64, @intFromEnum(error_codes.ErrorCode.invalid_handle));
    }));

    const content = if (content_ptr) |p| p[0..content_len] else "";

    const msg_role: MessageRole = switch (role) {
        0 => .system,
        1 => .user,
        2 => .assistant,
        3 => .developer,
        else => {
            capi_error.setError(error.InvalidArgument, "invalid role: {d}", .{role});
            return -@as(i64, @intFromEnum(error_codes.ErrorCode.invalid_argument));
        },
    };

    const content_type: ContentType = if (msg_role == .assistant) .output_text else .input_text;

    const item = conv.insertMessage(index, msg_role, content_type, content) catch |err| {
        capi_error.setError(err, "failed to insert message", .{});
        return -@as(i64, @intFromEnum(error_codes.errorToCode(err)));
    };
    item.hidden = hidden;

    return @intCast(index);
}
