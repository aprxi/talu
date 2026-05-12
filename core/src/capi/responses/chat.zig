//! Chat C API lifecycle and configuration functions.
//!
//! The Chat API provides a higher-level wrapper that combines an item-based
//! Conversation with generation configuration.
//!
//! Thread safety: NOT thread-safe. All access must be from a single thread.

const std = @import("std");
const conversation_mod = @import("../../responses/conversation/root.zig");
const session_id_mod = conversation_mod.session_id;
const responses_mod = @import("../../responses/root.zig");
const types = @import("types.zig");
const Conversation = conversation_mod.Conversation;
const Chat = conversation_mod.Chat;
const ChatHandle = types.ChatHandle;
const ChatCreateOptions = types.ChatCreateOptions;
const ResponsesHandle = types.ResponsesHandle;
const completions_protocol = responses_mod.protocol.chat_completions;

const allocator = std.heap.c_allocator;

const capi_error = @import("../error.zig");
const error_codes = @import("../error_codes.zig");
const signal_guard = @import("../signal_guard.zig");
const log = @import("log_pkg");

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

/// Safely check if a Chat pointer is valid.
/// Uses signal guard to catch SIGSEGV/SIGBUS if pointer is unmapped.
/// Returns true if valid, false if invalid or unmapped.
fn isChatValidSafe(chat: *Chat) bool {
    var guard = signal_guard.SignalGuard.init();
    defer guard.deinit();

    var ctx = MagicCheckContext{
        .ptr = &chat._magic,
        .expected_magic = Chat.MAGIC_VALID,
        .result = false,
    };

    const call_result = guard.call(&magicCheckCallback, @ptrCast(&ctx));
    if (call_result == null) {
        // Signal was caught - pointer is unmapped
        return false;
    }
    return ctx.result;
}

/// Create a new empty Chat.
/// Returns null on allocation failure (check talu_last_error for details).
pub export fn talu_chat_create(
    options: ?*const ChatCreateOptions,
) callconv(.c) ?*ChatHandle {
    capi_error.clearError();
    const chat_state = allocator.create(Chat) catch {
        capi_error.setError(error.OutOfMemory, "failed to allocate Chat", .{});
        return null;
    };
    chat_state.* = Chat.init(allocator) catch {
        allocator.destroy(chat_state);
        capi_error.setError(error.OutOfMemory, "failed to initialize Chat", .{});
        return null;
    };
    if (options) |opt| {
        chat_state.resolution_config = .{ .offline = opt.offline };
    }
    log.debug("chat", "Chat created", .{ .handle = @intFromPtr(chat_state) }, @src());
    return @ptrCast(chat_state);
}

/// Create a new Chat with a system prompt.
/// Returns null on allocation failure (check talu_last_error for details).
pub export fn talu_chat_create_with_system(
    system: [*:0]const u8,
    options: ?*const ChatCreateOptions,
) callconv(.c) ?*ChatHandle {
    capi_error.clearError();
    const chat_state = allocator.create(Chat) catch {
        capi_error.setError(error.OutOfMemory, "failed to allocate Chat", .{});
        return null;
    };
    const system_slice = std.mem.sliceTo(system, 0);
    chat_state.* = Chat.initWithSystem(allocator, system_slice) catch {
        allocator.destroy(chat_state);
        capi_error.setError(error.OutOfMemory, "failed to initialize Chat with system prompt", .{});
        return null;
    };
    if (options) |opt| {
        chat_state.resolution_config = .{ .offline = opt.offline };
    }
    log.debug("chat", "Chat created with system prompt", .{
        .handle = @intFromPtr(chat_state),
        .system_len = system_slice.len,
    }, @src());
    return @ptrCast(chat_state);
}

/// Create a new Chat with a session identifier.
/// session_id is used by clients to group messages by session.
/// Returns null on allocation failure (check talu_last_error for details).
pub export fn talu_chat_create_with_session(
    session_id: ?[*:0]const u8,
    options: ?*const ChatCreateOptions,
) callconv(.c) ?*ChatHandle {
    capi_error.clearError();
    const chat_state = allocator.create(Chat) catch {
        capi_error.setError(error.OutOfMemory, "failed to allocate Chat", .{});
        return null;
    };
    const sid = if (session_id) |s| std.mem.sliceTo(s, 0) else null;
    chat_state.* = Chat.initWithSession(allocator, sid) catch {
        allocator.destroy(chat_state);
        capi_error.setError(error.OutOfMemory, "failed to initialize Chat with session", .{});
        return null;
    };
    if (options) |opt| {
        chat_state.resolution_config = .{ .offline = opt.offline };
    }
    log.debug("chat", "Chat created with session", .{
        .handle = @intFromPtr(chat_state),
        .has_session_id = @as(u8, @intFromBool(sid != null)),
    }, @src());
    return @ptrCast(chat_state);
}

/// Create a new Chat with both a system prompt and session identifier.
/// This combines system prompt and session_id initialization in one call.
/// Returns null on allocation failure (check talu_last_error for details).
pub export fn talu_chat_create_with_system_and_session(
    system: ?[*:0]const u8,
    session_id: ?[*:0]const u8,
    options: ?*const ChatCreateOptions,
) callconv(.c) ?*ChatHandle {
    capi_error.clearError();
    const chat_state = allocator.create(Chat) catch {
        capi_error.setError(error.OutOfMemory, "failed to allocate Chat", .{});
        return null;
    };
    const system_slice = if (system) |s| std.mem.sliceTo(s, 0) else "";
    const sid = if (session_id) |s| std.mem.sliceTo(s, 0) else null;
    chat_state.* = Chat.initWithSystemAndSession(allocator, system_slice, sid) catch {
        allocator.destroy(chat_state);
        capi_error.setError(error.OutOfMemory, "failed to initialize Chat with system and session", .{});
        return null;
    };
    if (options) |opt| {
        chat_state.resolution_config = .{ .offline = opt.offline };
    }
    log.debug("chat", "Chat created with system and session", .{
        .handle = @intFromPtr(chat_state),
        .system_len = system_slice.len,
        .has_session_id = @as(u8, @intFromBool(sid != null)),
    }, @src());
    return @ptrCast(chat_state);
}

/// Free a Chat.
pub export fn talu_chat_free(handle: ?*ChatHandle) callconv(.c) void {
    if (handle) |chat_handle| {
        const chat_state: *Chat = @ptrCast(@alignCast(chat_handle));
        log.debug("chat", "Chat free", .{
            .handle = @intFromPtr(chat_state),
            .items = chat_state.conv.len(),
        }, @src());
        chat_state.deinit();
        allocator.destroy(chat_state);
    }
}

/// Get the Conversation from a Chat.
/// The returned handle is owned by the Chat - do NOT free it separately.
pub export fn talu_chat_get_conversation(handle: ?*ChatHandle) callconv(.c) ?*ResponsesHandle {
    capi_error.clearError();
    const chat_state: *Chat = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "chat handle is null", .{});
        return null;
    }));
    // Use signal-safe validation to handle unmapped memory without crashing.
    if (!isChatValidSafe(chat_state)) {
        capi_error.setError(error.InvalidHandle, "chat handle is corrupted, freed, or unmapped", .{});
        return null;
    }
    const conv = chat_state.getConversation();
    // Validate the Conversation pointer before returning using signal-safe check.
    if (!isConversationValidSafe(conv)) {
        capi_error.setError(error.InvalidHandle, "conversation inside chat is corrupted, freed, or unmapped", .{});
        return null;
    }
    return @ptrCast(conv);
}

/// Generate a new session ID.
/// Caller must free with talu_text_free.
pub export fn talu_session_id_new(out_session_id: *?[*:0]u8) callconv(.c) i32 {
    capi_error.clearError();
    out_session_id.* = null;

    const id = session_id_mod.generateSessionId(allocator) catch |err| {
        capi_error.setError(err, "failed to generate session_id", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer allocator.free(id);

    const id_cstr = allocator.allocSentinel(u8, id.len, 0) catch |err| {
        capi_error.setError(err, "failed to allocate session_id", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    @memcpy(id_cstr, id);
    out_session_id.* = id_cstr.ptr;
    return 0;
}

/// Sets the retention expiry for all items in this chat.
/// ttl_ts: Unix milliseconds, 0 means no expiry.
pub export fn talu_chat_set_ttl_ts(handle: ?*ChatHandle, ttl_ts: i64) callconv(.c) i32 {
    capi_error.clearError();
    const chat_state: *Chat = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "chat handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }));
    chat_state.conv.setTtlTs(ttl_ts);
    return 0;
}

/// Set the system prompt for a Chat.
/// Returns 0 on success, non-zero error code on failure.
pub export fn talu_chat_set_system(handle: ?*ChatHandle, system: ?[*:0]const u8) callconv(.c) i32 {
    capi_error.clearError();
    const chat_state: *Chat = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "chat handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }));
    const system_slice = if (system) |s| std.mem.sliceTo(s, 0) else "";
    chat_state.setSystem(system_slice) catch |err| {
        capi_error.setError(err, "failed to set system prompt", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Get the system prompt for a Chat.
/// Returns null-terminated string (must be freed with talu_text_free).
/// Returns null if no system prompt is set or handle is null.
pub export fn talu_chat_get_system(handle: ?*ChatHandle) callconv(.c) ?[*:0]u8 {
    const chat_state: *Chat = @ptrCast(@alignCast(handle orelse return null));
    const system = chat_state.getSystem() orelse return null;
    if (system.len == 0) return null;

    // Allocate null-terminated copy
    const result = allocator.allocSentinel(u8, system.len, 0) catch return null;
    @memcpy(result, system);
    return result;
}

/// Get the prompt document ID for lineage tracking.
/// Returns null-terminated string (must be freed with talu_text_free).
/// Returns null if no prompt_id is set or handle is null.
pub export fn talu_chat_get_prompt_id(handle: ?*ChatHandle) callconv(.c) ?[*:0]u8 {
    const chat_state: *Chat = @ptrCast(@alignCast(handle orelse return null));
    const pid = chat_state.getPromptId() orelse return null;
    if (pid.len == 0) return null;

    // Allocate null-terminated copy
    const result = allocator.allocSentinel(u8, pid.len, 0) catch return null;
    @memcpy(result, pid);
    return result;
}

/// Set the prompt document ID for lineage tracking.
/// Pass null or empty string to clear.
/// Returns 0 on success, non-zero error code on failure.
pub export fn talu_chat_set_prompt_id(handle: ?*ChatHandle, prompt_id: ?[*:0]const u8) callconv(.c) i32 {
    capi_error.clearError();
    const chat_state: *Chat = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "chat handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }));
    const pid_slice = if (prompt_id) |p| std.mem.sliceTo(p, 0) else null;
    chat_state.setPromptId(pid_slice) catch |err| {
        capi_error.setError(err, "failed to set prompt_id", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Set tool definitions on a Chat (opaque JSON blob).
/// Pass null to clear. Returns 0 on success.
pub export fn talu_chat_set_tools(
    handle: ?*ChatHandle,
    json: ?[*:0]const u8,
    json_len: usize,
) callconv(.c) i32 {
    capi_error.clearError();
    const chat_state: *Chat = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "chat handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }));
    if (json) |ptr| {
        const slice = ptr[0..json_len];
        chat_state.setTools(slice) catch |err| {
            capi_error.setError(err, "failed to set tools", .{});
            return @intFromEnum(error_codes.errorToCode(err));
        };
    } else {
        chat_state.clearTools();
    }
    return 0;
}

/// Get tool definitions JSON from a Chat.
/// Returns null-terminated string (must be freed with talu_text_free).
/// Returns null if no tools are set or handle is null.
pub export fn talu_chat_get_tools(handle: ?*ChatHandle) callconv(.c) ?[*:0]u8 {
    const chat_state: *Chat = @ptrCast(@alignCast(handle orelse return null));
    const tools = chat_state.getTools() orelse return null;
    if (tools.len == 0) return null;
    const result = allocator.allocSentinel(u8, tools.len, 0) catch return null;
    @memcpy(result, tools);
    return result;
}

/// Set tool_choice on a Chat (opaque JSON blob).
/// Pass null to clear. Returns 0 on success.
pub export fn talu_chat_set_tool_choice(
    handle: ?*ChatHandle,
    json: ?[*:0]const u8,
    json_len: usize,
) callconv(.c) i32 {
    capi_error.clearError();
    const chat_state: *Chat = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "chat handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }));
    if (json) |ptr| {
        const slice = ptr[0..json_len];
        chat_state.setToolChoice(slice) catch |err| {
            capi_error.setError(err, "failed to set tool_choice", .{});
            return @intFromEnum(error_codes.errorToCode(err));
        };
    } else {
        chat_state.clearToolChoice();
    }
    return 0;
}

/// Get tool_choice JSON from a Chat.
/// Returns null-terminated string (must be freed with talu_text_free).
/// Returns null if no tool_choice is set or handle is null.
pub export fn talu_chat_get_tool_choice(handle: ?*ChatHandle) callconv(.c) ?[*:0]u8 {
    const chat_state: *Chat = @ptrCast(@alignCast(handle orelse return null));
    const tc = chat_state.getToolChoice() orelse return null;
    if (tc.len == 0) return null;
    const result = allocator.allocSentinel(u8, tc.len, 0) catch return null;
    @memcpy(result, tc);
    return result;
}

/// Serialize Chat messages to JSON (OpenAI Completions format).
/// Returns allocated string that must be freed with talu_text_free.
/// Returns null on error (check talu_last_error for details).
///
/// Uses responses/protocol/chat_completions.zig for format conversion (One Correct Path).
pub export fn talu_chat_to_json(handle: ?*ChatHandle) callconv(.c) ?[*:0]u8 {
    capi_error.clearError();
    const chat_state: *Chat = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "chat handle is null", .{});
        return null;
    }));
    // Use protocol layer for Completions format serialization
    const json = completions_protocol.serialize(allocator, chat_state.getConversation(), .{}) catch |err| {
        capi_error.setError(err, "failed to serialize to JSON", .{});
        return null;
    };
    // Duplicate to allocator with sentinel
    const owned = allocator.dupeZ(u8, json) catch {
        allocator.free(json);
        capi_error.setError(error.OutOfMemory, "failed to allocate JSON string", .{});
        return null;
    };
    allocator.free(json);
    return owned.ptr;
}

/// Load messages from JSON (OpenAI Completions format).
/// Returns 0 on success, non-zero error code on failure.
///
/// Uses responses/protocol/chat_completions.zig for format conversion (One Correct Path).
pub export fn talu_chat_set_messages(handle: ?*ChatHandle, json: ?[*:0]const u8) callconv(.c) i32 {
    capi_error.clearError();
    const chat_state: *Chat = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "chat handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }));
    const json_slice = std.mem.sliceTo(json orelse {
        capi_error.setError(error.InvalidArgument, "json is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }, 0);
    // Use protocol layer for Completions format parsing
    completions_protocol.parse(chat_state.getConversation(), json_slice) catch |err| {
        capi_error.setError(err, "failed to load messages from JSON", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Load chat completions-format messages JSON into a chat conversation.
/// Format: [{"role":"system","content":"..."}, {"role":"user","content":"..."}, ...]
///
/// Clears existing items first, then parses the JSON array into the conversation
/// using the protocol completions parser. Uses ptr+len instead of sentinel-terminated
/// strings to support JSON payloads with embedded nulls.
///
/// Returns 0 on success, non-zero error code on failure.
pub export fn talu_chat_load_completions_json(
    handle: ?*ChatHandle,
    json_ptr: ?[*]const u8,
    json_len: usize,
) callconv(.c) i32 {
    capi_error.clearError();
    const chat_state: *Chat = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "chat handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }));
    const ptr = json_ptr orelse {
        capi_error.setError(error.InvalidArgument, "json_ptr is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const json_slice = ptr[0..json_len];

    completions_protocol.parse(chat_state.getConversation(), json_slice) catch |err| {
        capi_error.setError(err, "failed to load from Completions JSON", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    return 0;
}

/// Count tokens for the current chat history, optionally with an additional message.
///
/// This applies the chat template and tokenizes the result, returning the exact
/// token count that would be used for generation. Useful for context window management.
///
/// Args:
///   chat_handle: The chat handle
///   model: Model path (required for tokenizer and template)
///   additional_message: Optional message to include in count (null-terminated, can be null)
///   additional_message_len: Length of additional_message (0 if null)
///
/// Returns: Token count on success (>= 0), or negative error code on failure.
pub export fn talu_chat_count_tokens(
    chat_handle: ?*ChatHandle,
    model: ?[*:0]const u8,
    additional_message: ?[*]const u8,
    additional_message_len: usize,
) callconv(.c) i64 {
    capi_error.clearError();

    const chat: *Chat = @ptrCast(@alignCast(chat_handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "chat_handle is null", .{});
        return -@as(i64, @intFromEnum(error_codes.ErrorCode.invalid_argument));
    }));

    const model_path = std.mem.sliceTo(model orelse {
        capi_error.setErrorWithCode(.invalid_argument, "model is null", .{});
        return -@as(i64, @intFromEnum(error_codes.ErrorCode.invalid_argument));
    }, 0);

    // Get or create engine for this model
    const engine = responses_mod.getOrCreateEngine(allocator, model_path) catch |err| {
        capi_error.setError(err, "failed to load model for token counting", .{});
        return -@as(i64, @intFromEnum(error_codes.errorToCode(err)));
    };

    // Get additional message slice if provided
    const msg_slice: ?[]const u8 = if (additional_message != null and additional_message_len > 0)
        additional_message.?[0..additional_message_len]
    else
        null;

    // Count tokens
    const count = engine.countTokens(chat, msg_slice, .{}) catch |err| {
        capi_error.setError(err, "failed to count tokens", .{});
        return -@as(i64, @intFromEnum(error_codes.errorToCode(err)));
    };

    return @intCast(count);
}

/// Get the model's maximum context length.
///
/// Args:
///   model: Model path (required for loading config)
///
/// Returns: Maximum context length, or 0 if not specified in model config or on error.
pub export fn talu_chat_max_context_length(model: ?[*:0]const u8) callconv(.c) u64 {
    capi_error.clearError();

    const model_path = std.mem.sliceTo(model orelse {
        capi_error.setErrorWithCode(.invalid_argument, "model is null", .{});
        return 0;
    }, 0);

    // Get or create engine for this model
    const engine = responses_mod.getOrCreateEngine(allocator, model_path) catch |err| {
        capi_error.setError(err, "failed to load model for max context length", .{});
        return 0;
    };

    return engine.maxContextLength() orelse 0;
}
