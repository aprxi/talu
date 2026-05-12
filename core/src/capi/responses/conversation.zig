//! Responses C API conversation lifecycle, serialization, cloning, and status.
//!
//! Thread safety: NOT thread-safe. All access must be from a single thread.

const std = @import("std");
const conversation_mod = @import("../../responses/conversation/root.zig");
const responses_mod = @import("../../responses/root.zig");
const types = @import("types.zig");
const Conversation = conversation_mod.Conversation;
const ItemStatus = conversation_mod.ItemStatus;
const SerializationDirection = conversation_mod.SerializationDirection;
const completions_protocol = responses_mod.protocol.chat_completions;
const responses_protocol = responses_mod.protocol.openai_responses;
const ResponsesHandle = types.ResponsesHandle;

const allocator = std.heap.c_allocator;

const capi_error = @import("../error.zig");
const error_codes = @import("../error_codes.zig");

/// Create a new empty Conversation.
/// Returns null on allocation failure (check talu_last_error for details).
pub export fn talu_responses_create() callconv(.c) ?*ResponsesHandle {
    capi_error.clearError();
    const conv = Conversation.init(allocator) catch |err| {
        capi_error.setError(err, "failed to create Conversation", .{});
        return null;
    };
    return @ptrCast(conv);
}

/// Create a new Conversation with optional session ID.
/// Returns null on allocation failure.
pub export fn talu_responses_create_with_session(
    session_id: ?[*:0]const u8,
) callconv(.c) ?*ResponsesHandle {
    capi_error.clearError();
    const sid = if (session_id) |s| std.mem.sliceTo(s, 0) else null;
    const conv = Conversation.initWithSession(allocator, sid) catch |err| {
        capi_error.setError(err, "failed to create Conversation with session", .{});
        return null;
    };
    return @ptrCast(conv);
}

/// Free a Conversation.
pub export fn talu_responses_free(handle: ?*ResponsesHandle) callconv(.c) void {
    if (handle) |h| {
        const conv: *Conversation = @ptrCast(@alignCast(h));
        conv.deinit();
    }
}

/// Serialize conversation to Open Responses JSON format.
/// direction: 0 = request (ItemParam schemas), 1 = response (ItemField schemas)
/// Caller must free the returned string with talu_text_free().
pub export fn talu_responses_to_responses_json(
    handle: ?*ResponsesHandle,
    direction: u8,
) callconv(.c) ?[*:0]u8 {
    capi_error.clearError();
    const conv: *Conversation = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "conversation handle is null", .{});
        return null;
    }));

    const dir: SerializationDirection = if (direction == 0) .request else .response;
    const json_text = conv.toResponsesJsonWithOptions(.{ .direction = dir }) catch |err| {
        capi_error.setError(err, "failed to serialize to Responses JSON", .{});
        return null;
    };
    defer allocator.free(json_text);

    // Add null terminator
    const json_cstr = allocator.allocSentinel(u8, json_text.len, 0) catch {
        capi_error.setError(error.OutOfMemory, "failed to allocate JSON copy", .{});
        return null;
    };
    @memcpy(json_cstr, json_text);
    return json_cstr.ptr;
}

/// Serialize conversation to Chat Completions JSON format (role/content messages).
/// Caller must free the returned string with talu_text_free().
pub export fn talu_responses_to_completions_json(
    handle: ?*ResponsesHandle,
) callconv(.c) ?[*:0]u8 {
    capi_error.clearError();
    const conv: *Conversation = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "conversation handle is null", .{});
        return null;
    }));

    // Use the protocol adapter to serialize to Completions format
    const json_text = completions_protocol.serialize(allocator, conv, .{}) catch |err| {
        capi_error.setError(err, "failed to serialize to Completions JSON", .{});
        return null;
    };
    defer allocator.free(json_text);

    // Add null terminator
    const json_cstr = allocator.allocSentinel(u8, json_text.len, 0) catch {
        capi_error.setError(error.OutOfMemory, "failed to allocate JSON copy", .{});
        return null;
    };
    @memcpy(json_cstr, json_text);
    return json_cstr.ptr;
}

/// Clear all items from the conversation.
pub export fn talu_responses_clear(
    handle: ?*ResponsesHandle,
) callconv(.c) void {
    capi_error.clearError();
    const conv: *Conversation = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "conversation handle is null", .{});
        return;
    }));
    conv.clear();
}

/// Clear all items except the first system/developer message (if present).
pub export fn talu_responses_clear_keeping_system(
    handle: ?*ResponsesHandle,
) callconv(.c) void {
    capi_error.clearError();
    const conv: *Conversation = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "conversation handle is null", .{});
        return;
    }));
    conv.clearKeepingSystem();
}

/// Load conversation from OpenAI Completions JSON format: [{role, content}, ...]
/// This clears any existing items first.
/// Returns 0 on success, non-zero error code on failure.
///
/// Uses responses/protocol/chat_completions.zig for format conversion (One Correct Path).
pub export fn talu_responses_load_completions_json(
    handle: ?*ResponsesHandle,
    json: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const conv: *Conversation = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "conversation handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }));

    const json_slice = std.mem.sliceTo(json orelse {
        capi_error.setError(error.InvalidArgument, "json is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }, 0);

    // Use protocol layer for Completions format parsing
    completions_protocol.parse(conv, json_slice) catch |err| {
        capi_error.setError(err, "failed to load from Completions JSON", .{});
        return @intFromEnum(error_codes.ErrorCode.internal_error);
    };

    return 0;
}

/// Load conversation items from OpenResponses input format.
/// Input can be a JSON string (user message shorthand) or a JSON array of ItemParam objects.
/// Does NOT clear existing items — appends to the conversation.
/// Returns 0 on success, non-zero error code on failure.
pub export fn talu_responses_load_responses_json(
    handle: ?*ResponsesHandle,
    json: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const conv: *Conversation = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "conversation handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }));

    const json_slice = std.mem.sliceTo(json orelse {
        capi_error.setError(error.InvalidArgument, "json is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }, 0);

    responses_protocol.parse(conv, json_slice) catch |err| {
        capi_error.setError(err, "failed to load from Responses JSON", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    return 0;
}

/// Clone conversation items from source into destination.
/// Preserves item timestamps and metadata.
pub export fn talu_responses_clone(
    dest_handle: ?*ResponsesHandle,
    source_handle: ?*ResponsesHandle,
    batch_size: usize,
) callconv(.c) i32 {
    capi_error.clearError();
    const dest: *Conversation = @ptrCast(@alignCast(dest_handle orelse {
        capi_error.setError(error.InvalidHandle, "destination handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }));
    const source: *Conversation = @ptrCast(@alignCast(source_handle orelse {
        capi_error.setError(error.InvalidHandle, "source handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }));

    dest.cloneFrom(source, batch_size) catch |err| {
        capi_error.setError(err, "failed to clone conversation", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    return 0;
}

/// Clone a prefix of conversation items from source into destination.
/// last_index is inclusive. If out of range, clones all items.
pub export fn talu_responses_clone_prefix(
    dest_handle: ?*ResponsesHandle,
    source_handle: ?*ResponsesHandle,
    last_index: usize,
    batch_size: usize,
) callconv(.c) i32 {
    capi_error.clearError();
    const dest: *Conversation = @ptrCast(@alignCast(dest_handle orelse {
        capi_error.setError(error.InvalidHandle, "destination handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }));
    const source: *Conversation = @ptrCast(@alignCast(source_handle orelse {
        capi_error.setError(error.InvalidHandle, "source handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }));

    dest.cloneFromPrefix(source, last_index, batch_size) catch |err| {
        capi_error.setError(err, "failed to clone conversation prefix", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    return 0;
}

/// Truncate a conversation to keep items up to last_index (inclusive).
pub export fn talu_responses_truncate_after(
    handle: ?*ResponsesHandle,
    last_index: usize,
) callconv(.c) i32 {
    capi_error.clearError();
    const conv: *Conversation = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "conversation handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }));

    _ = conv.truncateAfterIndex(last_index);
    return 0;
}

/// Set the status of an item by index.
///
/// Status values: 0=in_progress, 1=waiting, 2=completed, 3=incomplete, 4=failed.
/// Returns 0 on success, error code on failure (invalid handle, out of bounds,
/// or invalid status value).
pub export fn talu_responses_set_item_status(
    handle: ?*ResponsesHandle,
    item_index: usize,
    status: u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const conv: *Conversation = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "conversation handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }));

    const item_status = std.meta.intToEnum(ItemStatus, status) catch {
        capi_error.setError(error.InvalidArgument, "invalid status value: {d}", .{status});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    conv.setItemStatus(item_index, item_status) catch |err| {
        capi_error.setError(err, "failed to set item status", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    return 0;
}
