//! Conversation C API - Item-Based Access
//!
//! C-callable functions for the Open Responses API architecture.
//! This module exposes the Item-based data model to language bindings,
//! enabling zero-copy access to conversation Items without flattening
//! to legacy message format.
//!
//! Architecture:
//!   - Items are accessed by index (0-based)
//!   - Item type is inspected via discriminator
//!   - Variant-specific accessors provide type-safe data access
//!   - Content parts within items are accessed separately
//!
//! Maps to Python: talu/chat/items.py
//!
//! Thread safety: NOT thread-safe. All access must be from a single thread.

const std = @import("std");
const responses_mod = @import("../responses/root.zig");
const session_id_mod = responses_mod.session_id;
const backend_mod = @import("../responses/backend.zig");
const router_mod = @import("../router/root.zig");
const db_capi = @import("db/ops.zig");
const Conversation = responses_mod.Conversation;
const Item = responses_mod.Item;
const ItemType = responses_mod.ItemType;
const ItemStatus = responses_mod.ItemStatus;
const MessageRole = responses_mod.MessageRole;
const ContentType = responses_mod.ContentType;
const ContentPart = responses_mod.ContentPart;
const ImageDetail = responses_mod.ImageDetail;
const SerializationDirection = responses_mod.SerializationDirection;
const completions_protocol = router_mod.protocol.completions;
const responses_protocol = router_mod.protocol.responses;

const allocator = std.heap.c_allocator;

// Error handling
const capi_error = @import("error.zig");
const error_codes = @import("error_codes.zig");
const signal_guard = @import("signal_guard.zig");
const log = @import("../log.zig");

// =============================================================================
// Signal-Safe Memory Validation
// =============================================================================

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

const ItemRecord = backend_mod.ItemRecord;
const ItemVariantRecord = backend_mod.ItemVariantRecord;
const ItemContentPartRecord = backend_mod.ItemContentPartRecord;
const CStorageRecord = db_capi.CStorageRecord;

// =============================================================================
// Opaque Handle
// =============================================================================

/// Opaque Conversation handle for C API.
/// Wraps the Zig Conversation pointer.
pub const ResponsesHandle = opaque {};

// =============================================================================
// C Structs - ABI-Stable Data Transfer
// =============================================================================

/// Item header (common fields for all item types).
/// Used for type inspection before calling variant-specific accessors.
pub const CItem = extern struct {
    /// Unique item ID (u64 internal format).
    id: u64,
    /// Item type discriminator (ItemType enum).
    /// 0=message, 1=function_call, 2=function_call_output, 3=reasoning, 4=item_reference, 255=unknown
    item_type: u8,
    /// Item status (ItemStatus enum).
    /// 0=in_progress, 1=completed, 2=incomplete, 3=failed
    status: u8,
    _padding: [6]u8 = .{0} ** 6,
    /// Creation timestamp (Unix milliseconds).
    created_at_ms: i64,
    /// Input token count (prompt tokens for this item).
    input_tokens: u32,
    /// Output token count (completion tokens for this item).
    output_tokens: u32,
    /// Prefill time in nanoseconds.
    prefill_ns: u64,
    /// Generation time in nanoseconds.
    generation_ns: u64,
    /// Finish reason string (e.g. "stop", "length"). Null if not set.
    finish_reason_ptr: ?[*:0]const u8,
};

/// Message item variant data.
/// Call talu_responses_item_as_message() after checking item_type == 0.
pub const CMessageItem = extern struct {
    /// Role discriminator (MessageRole enum).
    /// 0=system, 1=user, 2=assistant, 3=developer, 255=unknown
    role: u8,
    _padding: [7]u8 = .{0} ** 7,
    /// Number of content parts in this message.
    content_count: usize,
    /// Pointer to raw role string (for unknown roles). May be null.
    raw_role_ptr: ?[*:0]const u8,
};

/// Function call item variant data.
/// Call talu_responses_item_as_function_call() after checking item_type == 1.
pub const CFunctionCallItem = extern struct {
    /// Function name (null-terminated).
    name_ptr: ?[*:0]const u8,
    /// Function call ID (null-terminated).
    call_id_ptr: ?[*:0]const u8,
    /// Arguments as JSON string (not null-terminated, use with length).
    arguments_ptr: ?[*]const u8,
    /// Length of arguments string.
    arguments_len: usize,
};

/// Function call output item variant data.
/// Call talu_responses_item_as_function_call_output() after checking item_type == 2.
pub const CFunctionCallOutputItem = extern struct {
    /// The call_id this output is for (null-terminated).
    call_id_ptr: ?[*:0]const u8,
    /// Output text pointer (for simple text output).
    output_text_ptr: ?[*]const u8,
    /// Output text length.
    output_text_len: usize,
    /// Number of output parts (0 if simple text output).
    output_parts_count: usize,
    /// Whether output is simple text (true) or parts array (false).
    is_text_output: bool,
    _padding: [7]u8 = .{0} ** 7,
};

/// Reasoning item variant data.
/// Call talu_responses_item_as_reasoning() after checking item_type == 3.
pub const CReasoningItem = extern struct {
    /// Number of content parts in reasoning.
    content_count: usize,
    /// Number of summary parts.
    summary_count: usize,
    /// Encrypted content pointer (may be null).
    encrypted_content_ptr: ?[*]const u8,
    /// Encrypted content length.
    encrypted_content_len: usize,
};

/// Item reference variant data.
/// Call talu_responses_item_as_item_reference() after checking item_type == 4.
pub const CItemReferenceItem = extern struct {
    /// Referenced item ID (null-terminated string like "msg_123").
    id_ptr: ?[*:0]const u8,
};

/// Content part data.
/// Used for accessing content within messages, reasoning, and function call outputs.
pub const CContentPart = extern struct {
    /// Content type discriminator (ContentType enum).
    /// 0=input_text, 1=input_image, ..., 5=output_text, 7=text, etc.
    content_type: u8,
    /// Image detail level (0=auto, 1=low, 2=high). Only valid for input_image.
    image_detail: u8,
    _padding: [6]u8 = .{0} ** 6,
    /// Primary data pointer (text content, URL, or raw data).
    data_ptr: ?[*]const u8,
    /// Primary data length.
    data_len: usize,
    /// Secondary data pointer (e.g., filename for input_file, annotations_json for output_text).
    secondary_ptr: ?[*]const u8,
    /// Secondary data length.
    secondary_len: usize,
    /// Tertiary data pointer (e.g., logprobs_json for output_text).
    tertiary_ptr: ?[*]const u8,
    /// Tertiary data length.
    tertiary_len: usize,
    /// Quaternary data pointer (e.g., code_blocks_json for output_text).
    quaternary_ptr: ?[*]const u8,
    /// Quaternary data length.
    quaternary_len: usize,
};

// =============================================================================
// Conversation Lifecycle
// =============================================================================

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

// =============================================================================
// Item Count and Type Inspection
// =============================================================================

/// Get number of items in the conversation.
pub export fn talu_responses_item_count(handle: ?*ResponsesHandle) callconv(.c) usize {
    const conv: *Conversation = @ptrCast(@alignCast(handle orelse return 0));
    return conv.items_list.items.len;
}

/// Get item type discriminator at index.
/// Returns 255 (unknown) if index is out of bounds.
pub export fn talu_responses_item_type(
    handle: ?*ResponsesHandle,
    index: usize,
) callconv(.c) u8 {
    const conv: *Conversation = @ptrCast(@alignCast(handle orelse return 255));
    if (index >= conv.items_list.items.len) return 255;
    return @intFromEnum(conv.items_list.items[index].getType());
}

/// Get item header (common fields).
/// Returns 0 on success, non-zero error code on failure.
pub export fn talu_responses_get_item(
    handle: ?*ResponsesHandle,
    index: usize,
    out: ?*CItem,
) callconv(.c) i32 {
    capi_error.clearError();
    const conv: *Conversation = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "conversation handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }));
    const out_ptr = out orelse {
        capi_error.setError(error.InvalidArgument, "out pointer is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    if (index >= conv.items_list.items.len) {
        capi_error.setError(error.InvalidArgument, "index {d} out of bounds", .{index});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }

    const item = &conv.items_list.items[index];
    const status = getItemStatus(item);

    out_ptr.* = .{
        .id = item.id,
        .item_type = @intFromEnum(item.getType()),
        .status = @intFromEnum(status),
        .created_at_ms = item.created_at_ms,
        .input_tokens = item.input_tokens,
        .output_tokens = item.output_tokens,
        .prefill_ns = item.prefill_ns,
        .generation_ns = item.generation_ns,
        .finish_reason_ptr = if (item.finish_reason) |fr| fr.ptr else null,
    };
    return 0;
}

/// Get generation parameters JSON for an item (assistant messages only).
/// Returns the JSON string pointer and length.
/// out_ptr will be set to the data pointer (null if not set).
/// out_len will be set to the byte length (0 if not set).
/// The returned pointer is valid until the conversation is modified or freed.
/// Returns 0 on success, non-zero on error.
pub export fn talu_responses_item_get_generation_json(
    handle: ?*ResponsesHandle,
    index: usize,
    out_ptr: ?*?[*]const u8,
    out_len: ?*usize,
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
    const item = &conv.items_list.items[index];
    if (item.generation_json) |gen| {
        if (out_ptr) |p| p.* = gen.ptr;
        if (out_len) |l| l.* = gen.len;
    } else {
        if (out_ptr) |p| p.* = null;
        if (out_len) |l| l.* = 0;
    }
    return 0;
}

// =============================================================================
// Variant Accessors (Polymorphic)
// =============================================================================

/// Get message item data.
/// Returns 0 on success, non-zero error code if not a message or index out of bounds.
pub export fn talu_responses_item_as_message(
    handle: ?*ResponsesHandle,
    index: usize,
    out: ?*CMessageItem,
) callconv(.c) i32 {
    capi_error.clearError();
    const conv: *Conversation = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "conversation handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }));
    const out_ptr = out orelse {
        capi_error.setError(error.InvalidArgument, "out pointer is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    if (index >= conv.items_list.items.len) {
        capi_error.setError(error.InvalidArgument, "index {d} out of bounds", .{index});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }

    const item = &conv.items_list.items[index];
    const msg = item.asMessage() orelse {
        capi_error.setError(error.InvalidArgument, "item at index {d} is not a message", .{index});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    out_ptr.* = .{
        .role = @intFromEnum(msg.role),
        .content_count = msg.content.items.len,
        .raw_role_ptr = if (msg.raw_role) |r| r.ptr else null,
    };
    return 0;
}

/// Get function call item data.
/// Returns 0 on success, non-zero error code if not a function_call or index out of bounds.
pub export fn talu_responses_item_as_function_call(
    handle: ?*ResponsesHandle,
    index: usize,
    out: ?*CFunctionCallItem,
) callconv(.c) i32 {
    capi_error.clearError();
    const conv: *Conversation = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "conversation handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }));
    const out_ptr = out orelse {
        capi_error.setError(error.InvalidArgument, "out pointer is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    if (index >= conv.items_list.items.len) {
        capi_error.setError(error.InvalidArgument, "index {d} out of bounds", .{index});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }

    const item = &conv.items_list.items[index];
    const fc = item.asFunctionCall() orelse {
        capi_error.setError(error.InvalidArgument, "item at index {d} is not a function_call", .{index});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    out_ptr.* = .{
        .name_ptr = fc.name.ptr,
        .call_id_ptr = fc.call_id.ptr,
        .arguments_ptr = fc.arguments.items.ptr,
        .arguments_len = fc.arguments.items.len,
    };
    return 0;
}

/// Convert function call output to C struct.
fn convertFunctionCallOutput(fco: anytype, out_ptr: *CFunctionCallOutputItem) void {
    switch (fco.output) {
        .text => |t| out_ptr.* = .{
            .call_id_ptr = fco.call_id.ptr,
            .output_text_ptr = t.items.ptr,
            .output_text_len = t.items.len,
            .output_parts_count = 0,
            .is_text_output = true,
        },
        .parts => |p| out_ptr.* = .{
            .call_id_ptr = fco.call_id.ptr,
            .output_text_ptr = null,
            .output_text_len = 0,
            .output_parts_count = p.items.len,
            .is_text_output = false,
        },
    }
}

/// Get function call output item data.
/// Returns 0 on success, non-zero error code if not a function_call_output or index out of bounds.
pub export fn talu_responses_item_as_function_call_output(
    handle: ?*ResponsesHandle,
    index: usize,
    out: ?*CFunctionCallOutputItem,
) callconv(.c) i32 {
    capi_error.clearError();
    const conv: *Conversation = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "conversation handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }));
    const out_ptr = out orelse {
        capi_error.setError(error.InvalidArgument, "out pointer is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    if (index >= conv.items_list.items.len) {
        capi_error.setError(error.InvalidArgument, "index {d} out of bounds", .{index});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }

    const fco = conv.items_list.items[index].asFunctionCallOutput() orelse {
        capi_error.setError(error.InvalidArgument, "item at index {d} is not a function_call_output", .{index});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    convertFunctionCallOutput(fco, out_ptr);
    return 0;
}

/// Get reasoning item data.
/// Returns 0 on success, non-zero error code if not a reasoning or index out of bounds.
pub export fn talu_responses_item_as_reasoning(
    handle: ?*ResponsesHandle,
    index: usize,
    out: ?*CReasoningItem,
) callconv(.c) i32 {
    capi_error.clearError();
    const conv: *Conversation = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "conversation handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }));
    const out_ptr = out orelse {
        capi_error.setError(error.InvalidArgument, "out pointer is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    if (index >= conv.items_list.items.len) {
        capi_error.setError(error.InvalidArgument, "index {d} out of bounds", .{index});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }

    const item = &conv.items_list.items[index];
    const reasoning = item.asReasoning() orelse {
        capi_error.setError(error.InvalidArgument, "item at index {d} is not a reasoning", .{index});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    out_ptr.* = .{
        .content_count = reasoning.content.items.len,
        .summary_count = reasoning.summary.items.len,
        .encrypted_content_ptr = if (reasoning.encrypted_content) |e| @ptrCast(e.ptr) else null,
        .encrypted_content_len = if (reasoning.encrypted_content) |e| e.len else 0,
    };
    return 0;
}

/// Get item reference data.
/// Returns 0 on success, non-zero error code if not an item_reference or index out of bounds.
pub export fn talu_responses_item_as_item_reference(
    handle: ?*ResponsesHandle,
    index: usize,
    out: ?*CItemReferenceItem,
) callconv(.c) i32 {
    capi_error.clearError();
    const conv: *Conversation = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "conversation handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }));
    const out_ptr = out orelse {
        capi_error.setError(error.InvalidArgument, "out pointer is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    if (index >= conv.items_list.items.len) {
        capi_error.setError(error.InvalidArgument, "index {d} out of bounds", .{index});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }

    const item = &conv.items_list.items[index];
    const item_ref = item.asItemReference() orelse {
        capi_error.setError(error.InvalidArgument, "item at index {d} is not an item_reference", .{index});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    out_ptr.* = .{
        .id_ptr = item_ref.id.ptr,
    };
    return 0;
}

// =============================================================================
// Content Part Access
// =============================================================================

/// Get content part count for a message item.
/// Returns 0 if item is not a message or index is out of bounds.
pub export fn talu_responses_item_message_content_count(
    handle: ?*ResponsesHandle,
    index: usize,
) callconv(.c) usize {
    const conv: *Conversation = @ptrCast(@alignCast(handle orelse return 0));
    if (index >= conv.items_list.items.len) return 0;
    const item = &conv.items_list.items[index];
    const msg = item.asMessage() orelse return 0;
    return msg.content.items.len;
}

/// Get content part for a message item.
/// Returns 0 on success, non-zero error code on failure.
pub export fn talu_responses_item_message_get_content(
    handle: ?*ResponsesHandle,
    item_index: usize,
    part_index: usize,
    out: ?*CContentPart,
) callconv(.c) i32 {
    capi_error.clearError();
    const conv: *Conversation = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "conversation handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }));
    const out_ptr = out orelse {
        capi_error.setError(error.InvalidArgument, "out pointer is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    if (item_index >= conv.items_list.items.len) {
        capi_error.setError(error.InvalidArgument, "item index {d} out of bounds", .{item_index});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }

    const item = &conv.items_list.items[item_index];
    const msg = item.asMessage() orelse {
        capi_error.setError(error.InvalidArgument, "item at index {d} is not a message", .{item_index});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    if (part_index >= msg.content.items.len) {
        capi_error.setError(error.InvalidArgument, "part index {d} out of bounds", .{part_index});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }

    const part = &msg.content.items[part_index];
    fillContentPart(part, out_ptr);
    return 0;
}

/// Get content part count for a reasoning item's content.
pub export fn talu_responses_item_reasoning_content_count(
    handle: ?*ResponsesHandle,
    index: usize,
) callconv(.c) usize {
    const conv: *Conversation = @ptrCast(@alignCast(handle orelse return 0));
    if (index >= conv.items_list.items.len) return 0;
    const item = &conv.items_list.items[index];
    const reasoning = item.asReasoning() orelse return 0;
    return reasoning.content.items.len;
}

/// Get content part for a reasoning item's content.
pub export fn talu_responses_item_reasoning_get_content(
    handle: ?*ResponsesHandle,
    item_index: usize,
    part_index: usize,
    out: ?*CContentPart,
) callconv(.c) i32 {
    capi_error.clearError();
    const conv: *Conversation = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "conversation handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }));
    const out_ptr = out orelse {
        capi_error.setError(error.InvalidArgument, "out pointer is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    if (item_index >= conv.items_list.items.len) {
        capi_error.setError(error.InvalidArgument, "item index {d} out of bounds", .{item_index});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }

    const item = &conv.items_list.items[item_index];
    const reasoning = item.asReasoning() orelse {
        capi_error.setError(error.InvalidArgument, "item at index {d} is not a reasoning", .{item_index});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    if (part_index >= reasoning.content.items.len) {
        capi_error.setError(error.InvalidArgument, "part index {d} out of bounds", .{part_index});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }

    const part = &reasoning.content.items[part_index];
    fillContentPart(part, out_ptr);
    return 0;
}

/// Get summary content part count for a reasoning item.
pub export fn talu_responses_item_reasoning_summary_count(
    handle: ?*ResponsesHandle,
    index: usize,
) callconv(.c) usize {
    const conv: *Conversation = @ptrCast(@alignCast(handle orelse return 0));
    if (index >= conv.items_list.items.len) return 0;
    const item = &conv.items_list.items[index];
    const reasoning = item.asReasoning() orelse return 0;
    return reasoning.summary.items.len;
}

/// Get summary content part for a reasoning item.
pub export fn talu_responses_item_reasoning_get_summary(
    handle: ?*ResponsesHandle,
    item_index: usize,
    part_index: usize,
    out: ?*CContentPart,
) callconv(.c) i32 {
    capi_error.clearError();
    const conv: *Conversation = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "conversation handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }));
    const out_ptr = out orelse {
        capi_error.setError(error.InvalidArgument, "out pointer is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    if (item_index >= conv.items_list.items.len) {
        capi_error.setError(error.InvalidArgument, "item index {d} out of bounds", .{item_index});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }

    const item = &conv.items_list.items[item_index];
    const reasoning = item.asReasoning() orelse {
        capi_error.setError(error.InvalidArgument, "item at index {d} is not a reasoning", .{item_index});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    if (part_index >= reasoning.summary.items.len) {
        capi_error.setError(error.InvalidArgument, "part index {d} out of bounds", .{part_index});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }

    const part = &reasoning.summary.items[part_index];
    fillContentPart(part, out_ptr);
    return 0;
}

/// Get content part for a function_call_output's parts array.
/// Only valid if is_text_output is false.
pub export fn talu_responses_item_fco_get_part(
    handle: ?*ResponsesHandle,
    item_index: usize,
    part_index: usize,
    out: ?*CContentPart,
) callconv(.c) i32 {
    capi_error.clearError();
    const conv: *Conversation = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "conversation handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }));
    const out_ptr = out orelse {
        capi_error.setError(error.InvalidArgument, "out pointer is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    if (item_index >= conv.items_list.items.len) {
        capi_error.setError(error.InvalidArgument, "item index {d} out of bounds", .{item_index});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }

    const item = &conv.items_list.items[item_index];
    const fco = item.asFunctionCallOutput() orelse {
        capi_error.setError(error.InvalidArgument, "item at index {d} is not a function_call_output", .{item_index});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    switch (fco.output) {
        .text => {
            capi_error.setError(error.InvalidArgument, "function_call_output is text, not parts array", .{});
            return @intFromEnum(error_codes.ErrorCode.invalid_argument);
        },
        .parts => |p| {
            if (part_index >= p.items.len) {
                capi_error.setError(error.InvalidArgument, "part index {d} out of bounds", .{part_index});
                return @intFromEnum(error_codes.ErrorCode.invalid_argument);
            }
            const part = &p.items[part_index];
            fillContentPart(part, out_ptr);
        },
    }
    return 0;
}

// =============================================================================
// Serialization
// =============================================================================

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

/// Serialize conversation to legacy Completions JSON format (role/content messages).
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
/// Uses router/protocol/completions.zig for format conversion (One Correct Path).
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
/// Preserves item timestamps and metadata; emits storage events in batches.
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

    // Check for sticky storage error after successful clone.
    // Items are in memory but storage backend failed - propagate this to caller.
    if (dest.hasStorageError()) {
        // Use explicit error code - the captured error may be an arbitrary error from
        // the Python callback which would map to INTERNAL_ERROR (999) instead of STORAGE_ERROR (700).
        capi_error.setErrorWithCode(error_codes.ErrorCode.storage_error, "storage backend failed during clone (items in memory only)", .{});
        dest.clearStorageError();
        return @intFromEnum(error_codes.ErrorCode.storage_error);
    }

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

    // Check for sticky storage error after successful clone.
    // Items are in memory but storage backend failed - propagate this to caller.
    if (dest.hasStorageError()) {
        // Use explicit error code - the captured error may be an arbitrary error from
        // the Python callback which would map to INTERNAL_ERROR (999) instead of STORAGE_ERROR (700).
        capi_error.setErrorWithCode(error_codes.ErrorCode.storage_error, "storage backend failed during clone prefix (items in memory only)", .{});
        dest.clearStorageError();
        return @intFromEnum(error_codes.ErrorCode.storage_error);
    }

    return 0;
}

/// Begin a fork transaction boundary for storage backends.
/// Returns a fork_id (0 if no storage backend is configured).
pub export fn talu_responses_begin_fork(
    handle: ?*ResponsesHandle,
) callconv(.c) u64 {
    capi_error.clearError();
    const conv: *Conversation = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "conversation handle is null", .{});
        return 0;
    }));
    return conv.beginFork();
}

/// End a fork transaction boundary for storage backends.
pub export fn talu_responses_end_fork(
    handle: ?*ResponsesHandle,
    fork_id: u64,
) callconv(.c) i32 {
    capi_error.clearError();
    const conv: *Conversation = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "conversation handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }));
    conv.endFork(fork_id);
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

/// Set parent_item_id for an item by index.
pub export fn talu_responses_set_item_parent(
    handle: ?*ResponsesHandle,
    item_index: usize,
    parent_item_id: u64,
    has_parent: bool,
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

    const item = &conv.items_list.items[item_index];
    item.parent_item_id = if (has_parent) parent_item_id else null;
    return 0;
}

/// Set structured validation flags for an item by index.
pub export fn talu_responses_set_item_validation_flags(
    handle: ?*ResponsesHandle,
    item_index: usize,
    json_valid: bool,
    schema_valid: bool,
    repaired: bool,
) callconv(.c) i32 {
    capi_error.clearError();
    const conv: *Conversation = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "conversation handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }));

    conv.setItemValidationFlags(item_index, json_valid, schema_valid, repaired) catch |err| {
        capi_error.setError(err, "failed to set validation flags", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

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

/// Duplicates a C string to an owned slice, returns null if input is null.
fn dupeCString(cstr: ?[*:0]const u8) error{OutOfMemory}!?[]const u8 {
    const ptr = cstr orelse return null;
    return try allocator.dupe(u8, std.mem.sliceTo(ptr, 0));
}

/// Frees all ItemRecords in the list and deinitializes the list.
fn freeParsedRecords(records: *std.ArrayListUnmanaged(ItemRecord)) void {
    for (records.items) |*rec| rec.deinit(allocator);
    records.deinit(allocator);
}

/// Parses a single CStorageRecord into an ItemRecord.
/// On success, caller owns the returned ItemRecord. On error, all allocations are freed.
fn parseStorageRecord(record: CStorageRecord) !ItemRecord {
    const content_json = record.content_json orelse return error.InvalidArgument;
    const status = responses_mod.itemStatusFromU8(record.status);
    const variant = try responses_mod.parseItemVariantRecord(allocator, std.mem.sliceTo(content_json, 0), status);
    // Build ItemRecord with variant for proper cleanup via deinit on error
    var item_record = ItemRecord{
        .item_id = record.item_id,
        .created_at_ms = record.created_at_ms,
        .ttl_ts = record.ttl_ts,
        .status = status,
        .hidden = record.hidden,
        .pinned = record.pinned,
        .json_valid = record.json_valid,
        .schema_valid = record.schema_valid,
        .repaired = record.repaired,
        .parent_item_id = if (record.has_parent) record.parent_item_id else null,
        .origin_item_id = if (record.has_origin) record.origin_item_id else null,
        .prefill_ns = record.prefill_ns,
        .generation_ns = record.generation_ns,
        .input_tokens = record.input_tokens,
        .output_tokens = record.output_tokens,
        .item_type = @enumFromInt(record.item_type),
        .variant = variant,
    };
    errdefer item_record.deinit(allocator);

    item_record.origin_session_id = if (record.has_origin) try dupeCString(record.origin_session_id) else null;
    item_record.finish_reason = try dupeCString(record.finish_reason);
    item_record.metadata = try dupeCString(record.metadata_json);
    return item_record;
}

/// Load storage records into an empty conversation.
/// Uses storage JSON content to reconstruct Item variants.
pub export fn talu_responses_load_storage_records(
    handle: ?*ResponsesHandle,
    records_ptr: ?[*]const CStorageRecord,
    records_len: usize,
) callconv(.c) i32 {
    capi_error.clearError();
    const conv: *Conversation = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "conversation handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    }));
    const records = records_ptr orelse {
        capi_error.setError(error.InvalidArgument, "records pointer is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    if (records_len == 0) return 0;

    var parsed_records: std.ArrayListUnmanaged(ItemRecord) = .{};
    defer freeParsedRecords(&parsed_records);

    for (records[0..records_len]) |record| {
        const item_record = parseStorageRecord(record) catch |err| {
            capi_error.setError(err, "failed to parse storage record", .{});
            return @intFromEnum(error_codes.errorToCode(err));
        };
        parsed_records.append(allocator, item_record) catch |err| {
            capi_error.setError(err, "failed to allocate storage record", .{});
            return @intFromEnum(error_codes.errorToCode(err));
        };
    }

    conv.loadItemRecords(parsed_records.items) catch |err| {
        capi_error.setError(err, "failed to load storage records", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

// =============================================================================
// Write Operations (Append Items)
// =============================================================================

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

// =============================================================================
// Helper Functions
// =============================================================================

/// Get status from an item's variant.
fn getItemStatus(item: *const Item) ItemStatus {
    return switch (item.data) {
        .message => |m| m.status,
        .function_call => |f| f.status,
        .function_call_output => |f| f.status,
        .reasoning => |r| r.status,
        .item_reference => |i| i.status,
        .unknown => .completed,
    };
}

/// Fill CContentPart from a ContentPart.
fn fillContentPart(part: *const ContentPart, out: *CContentPart) void {
    const content_type = part.getContentType();
    out.content_type = @intFromEnum(content_type);
    out.image_detail = @intFromEnum(part.getImageDetail());

    // Initialize all pointers to null
    out.data_ptr = null;
    out.data_len = 0;
    out.secondary_ptr = null;
    out.secondary_len = 0;
    out.tertiary_ptr = null;
    out.tertiary_len = 0;
    out.quaternary_ptr = null;
    out.quaternary_len = 0;

    // Fill based on content type
    switch (part.variant) {
        .input_text => |v| {
            out.data_ptr = v.text.items.ptr;
            out.data_len = v.text.items.len;
        },
        .input_image => |v| {
            out.data_ptr = v.image_url.items.ptr;
            out.data_len = v.image_url.items.len;
        },
        .input_audio => |v| {
            out.data_ptr = v.audio_data.items.ptr;
            out.data_len = v.audio_data.items.len;
        },
        .input_video => |v| {
            out.data_ptr = v.video_url.items.ptr;
            out.data_len = v.video_url.items.len;
        },
        .input_file => |v| {
            if (v.file_data) |d| {
                out.data_ptr = d.items.ptr;
                out.data_len = d.items.len;
            } else if (v.file_url) |u| {
                out.data_ptr = u.items.ptr;
                out.data_len = u.items.len;
            }
            if (v.filename) |f| {
                out.secondary_ptr = @ptrCast(f.ptr);
                out.secondary_len = f.len;
            }
        },
        .output_text => |v| {
            out.data_ptr = v.text.items.ptr;
            out.data_len = v.text.items.len;
            if (v.annotations_json) |a| {
                out.secondary_ptr = @ptrCast(a.ptr);
                out.secondary_len = a.len;
            }
            if (v.logprobs_json) |l| {
                out.tertiary_ptr = @ptrCast(l.ptr);
                out.tertiary_len = l.len;
            }
            if (v.code_blocks_json) |c| {
                out.quaternary_ptr = @ptrCast(c.ptr);
                out.quaternary_len = c.len;
            }
        },
        .refusal => |v| {
            out.data_ptr = v.refusal.items.ptr;
            out.data_len = v.refusal.items.len;
        },
        .text => |v| {
            out.data_ptr = v.text.items.ptr;
            out.data_len = v.text.items.len;
        },
        .reasoning_text => |v| {
            out.data_ptr = v.text.items.ptr;
            out.data_len = v.text.items.len;
        },
        .summary_text => |v| {
            out.data_ptr = v.text.items.ptr;
            out.data_len = v.text.items.len;
        },
        .unknown => |v| {
            out.data_ptr = v.raw_data.items.ptr;
            out.data_len = v.raw_data.items.len;
            if (v.raw_type.len > 0) {
                out.secondary_ptr = @ptrCast(v.raw_type.ptr);
                out.secondary_len = v.raw_type.len;
            }
        },
    }
}

// =============================================================================
// Chat API (talu_chat_*) - Lifecycle and Configuration
// =============================================================================
//
// The Chat API provides a higher-level wrapper that combines:
// - A Conversation (Item-based message storage)
// - Generation configuration (temperature, max_tokens, etc.)
//
// Use talu_chat_* when you need sampling parameters.
// Use talu_responses_* when you only need Item storage.

const Chat = responses_mod.Chat;

/// Opaque Chat handle for C API.
/// Chat owns a Conversation and generation configuration.
pub const ChatHandle = opaque {};

/// Chat creation options.
pub const ChatCreateOptions = extern struct {
    offline: bool = false,
};

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
/// session_id is used by storage backends to group messages by session.
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

/// Validate a Conversation handle.
/// Returns 1 if valid, 0 if invalid or corrupted.
/// Safe to call with any pointer value - will not crash on invalid pointers.
/// Uses signal guard to safely probe unmapped memory.
pub export fn talu_responses_validate(handle: ?*ResponsesHandle) callconv(.c) i32 {
    capi_error.clearError();
    const conv: *Conversation = @ptrCast(@alignCast(handle orelse return 0));

    // Use signal-safe validation to handle unmapped memory without crashing.
    if (isConversationValidSafe(conv)) {
        return 1;
    }
    return 0;
}

/// Validate a Chat handle.
/// Returns 1 if valid, 0 if invalid or corrupted.
/// Safe to call with any pointer value - will not crash on invalid pointers.
/// Uses signal guard to safely probe unmapped memory.
pub export fn talu_chat_validate(handle: ?*ChatHandle) callconv(.c) i32 {
    capi_error.clearError();
    const chat_state: *Chat = @ptrCast(@alignCast(handle orelse return 0));

    // Use signal-safe validation to handle unmapped memory without crashing.
    if (isChatValidSafe(chat_state)) {
        return 1;
    }
    return 0;
}

/// Get the session identifier for a Chat.
/// Returns null if no session_id is set.
/// Caller must free with talu_text_free.
pub export fn talu_chat_get_session_id(handle: ?*ChatHandle) callconv(.c) ?[*:0]u8 {
    capi_error.clearError();
    const chat_state: *Chat = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "chat handle is null", .{});
        return null;
    }));
    if (chat_state.session_id) |sid| {
        const sid_cstr = allocator.allocSentinel(u8, sid.len, 0) catch {
            capi_error.setError(error.OutOfMemory, "failed to allocate session_id copy", .{});
            return null;
        };
        @memcpy(sid_cstr, sid);
        return sid_cstr.ptr;
    }
    return null; // No session_id set - not an error
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

// =============================================================================
// Chat Sampling Parameters
// =============================================================================

/// Gets the sampling temperature.
/// Returns the default value (0.7) if handle is null.
pub export fn talu_chat_get_temperature(handle: ?*ChatHandle) callconv(.c) f32 {
    const chat_state: *Chat = @ptrCast(@alignCast(handle orelse return 0.7));
    return chat_state.temperature;
}

/// Sets the sampling temperature.
pub export fn talu_chat_set_temperature(handle: ?*ChatHandle, value: f32) callconv(.c) void {
    const chat_state: *Chat = @ptrCast(@alignCast(handle orelse return));
    chat_state.temperature = value;
}

/// Gets the maximum tokens to generate.
/// Returns the default value (256) if handle is null.
pub export fn talu_chat_get_max_tokens(handle: ?*ChatHandle) callconv(.c) usize {
    const chat_state: *Chat = @ptrCast(@alignCast(handle orelse return 256));
    return chat_state.max_tokens;
}

/// Sets the maximum tokens to generate.
pub export fn talu_chat_set_max_tokens(handle: ?*ChatHandle, value: usize) callconv(.c) void {
    const chat_state: *Chat = @ptrCast(@alignCast(handle orelse return));
    chat_state.max_tokens = value;
}

/// Gets the top-k sampling parameter.
/// Returns the default value (50) if handle is null.
pub export fn talu_chat_get_top_k(handle: ?*ChatHandle) callconv(.c) usize {
    const chat_state: *Chat = @ptrCast(@alignCast(handle orelse return 50));
    return chat_state.top_k;
}

/// Sets the top-k sampling parameter.
pub export fn talu_chat_set_top_k(handle: ?*ChatHandle, value: usize) callconv(.c) void {
    const chat_state: *Chat = @ptrCast(@alignCast(handle orelse return));
    chat_state.top_k = value;
}

/// Gets the nucleus sampling (top-p) parameter.
/// Returns the default value (0.9) if handle is null.
pub export fn talu_chat_get_top_p(handle: ?*ChatHandle) callconv(.c) f32 {
    const chat_state: *Chat = @ptrCast(@alignCast(handle orelse return 0.9));
    return chat_state.top_p;
}

/// Sets the nucleus sampling (top-p) parameter.
pub export fn talu_chat_set_top_p(handle: ?*ChatHandle, value: f32) callconv(.c) void {
    const chat_state: *Chat = @ptrCast(@alignCast(handle orelse return));
    chat_state.top_p = value;
}

/// Gets the min-p sampling parameter.
/// Returns the default value (0.0) if handle is null.
pub export fn talu_chat_get_min_p(handle: ?*ChatHandle) callconv(.c) f32 {
    const chat_state: *Chat = @ptrCast(@alignCast(handle orelse return 0.0));
    return chat_state.min_p;
}

/// Sets the min-p sampling parameter.
pub export fn talu_chat_set_min_p(handle: ?*ChatHandle, value: f32) callconv(.c) void {
    const chat_state: *Chat = @ptrCast(@alignCast(handle orelse return));
    chat_state.min_p = value;
}

/// Gets the repetition penalty.
/// Returns the default value (1.0, no penalty) if handle is null.
pub export fn talu_chat_get_repetition_penalty(handle: ?*ChatHandle) callconv(.c) f32 {
    const chat_state: *Chat = @ptrCast(@alignCast(handle orelse return 1.0));
    return chat_state.repetition_penalty;
}

/// Sets the repetition penalty.
pub export fn talu_chat_set_repetition_penalty(handle: ?*ChatHandle, value: f32) callconv(.c) void {
    const chat_state: *Chat = @ptrCast(@alignCast(handle orelse return));
    chat_state.repetition_penalty = value;
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

/// Clear all messages from a Chat (preserves system prompt).
pub export fn talu_chat_clear(handle: ?*ChatHandle) callconv(.c) void {
    const chat_state: *Chat = @ptrCast(@alignCast(handle orelse return));
    chat_state.reset(); // reset preserves system, clear clears system
}

/// Reset a Chat completely (clears all messages including system prompt).
pub export fn talu_chat_reset(handle: ?*ChatHandle) callconv(.c) void {
    const chat_state: *Chat = @ptrCast(@alignCast(handle orelse return));
    chat_state.clear();
    chat_state.clearSystem();
}

/// Get the number of messages in a Chat.
pub export fn talu_chat_len(handle: ?*ChatHandle) callconv(.c) usize {
    const chat_state: *Chat = @ptrCast(@alignCast(handle orelse return 0));
    return chat_state.len();
}

/// Serialize Chat messages to JSON (OpenAI Completions format).
/// Returns allocated string that must be freed with talu_text_free.
/// Returns null on error (check talu_last_error for details).
///
/// Uses router/protocol/completions.zig for format conversion (One Correct Path).
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
/// Uses router/protocol/completions.zig for format conversion (One Correct Path).
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

/// Get the Messages handle from a Chat (legacy API).
/// Returns the underlying Messages structure pointer for zero-copy access.
/// The returned handle is owned by the Chat - do NOT free it separately.
/// Note: This returns the Conversation handle as the legacy Messages type is deprecated.
pub export fn talu_chat_get_messages(handle: ?*ChatHandle) callconv(.c) ?*anyopaque {
    capi_error.clearError();
    const chat_state: *Chat = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "chat handle is null", .{});
        return null;
    }));
    return @ptrCast(chat_state.getConversation());
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
    const engine = router_mod.getOrCreateEngine(allocator, model_path) catch |err| {
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
    const engine = router_mod.getOrCreateEngine(allocator, model_path) catch |err| {
        capi_error.setError(err, "failed to load model for max context length", .{});
        return 0;
    };

    return engine.maxContextLength() orelse 0;
}

// =============================================================================
// Session Update Notification
// =============================================================================

/// Notify the session that metadata fields have changed.
///
/// This is a lightweight notification to the conversation's storage backend.
/// The session record itself is stored by the table-plane APIs; this call
/// just pushes a `session_update` storage event so that backends can react.
///
/// Args:
///   chat_handle: Opaque Chat handle
///   model, title, system_prompt, config_json, marker,
///   parent_session_id, group_id, metadata_json, source_doc_id,
///   project_id:
///     New values (or null to leave unchanged).
///
/// Returns: 0 on success, negative error code on failure.
// lint:ignore capi-callconv - callconv(.c) on closing line
pub export fn talu_chat_notify_session_update(
    chat_handle: ?*ChatHandle,
    model: ?[*:0]const u8,
    title: ?[*:0]const u8,
    system_prompt: ?[*:0]const u8,
    config_json: ?[*:0]const u8,
    marker: ?[*:0]const u8,
    parent_session_id: ?[*:0]const u8,
    group_id: ?[*:0]const u8,
    metadata_json: ?[*:0]const u8,
    source_doc_id: ?[*:0]const u8,
    project_id: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();

    const chat: *Chat = @ptrCast(@alignCast(chat_handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "chat_handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const model_slice: ?[]const u8 = if (model) |m| std.mem.span(m) else null;
    const title_slice: ?[]const u8 = if (title) |t| std.mem.span(t) else null;
    const system_slice: ?[]const u8 = if (system_prompt) |s| std.mem.span(s) else null;
    const config_slice: ?[]const u8 = if (config_json) |c| std.mem.span(c) else null;
    const marker_slice: ?[]const u8 = if (marker) |s| std.mem.span(s) else null;
    const parent_session_id_slice: ?[]const u8 = if (parent_session_id) |p| std.mem.span(p) else null;
    const group_id_slice: ?[]const u8 = if (group_id) |g| std.mem.span(g) else null;
    const metadata_json_slice: ?[]const u8 = if (metadata_json) |m| std.mem.span(m) else null;
    const source_doc_id_slice: ?[]const u8 = if (source_doc_id) |s| std.mem.span(s) else null;
    const project_id_slice: ?[]const u8 = if (project_id) |p| std.mem.span(p) else null;

    chat.conv.notifySessionUpdate(
        model_slice,
        title_slice,
        system_slice,
        config_slice,
        marker_slice,
        parent_session_id_slice,
        group_id_slice,
        metadata_json_slice,
        source_doc_id_slice,
        project_id_slice,
    );

    return 0;
}

// =============================================================================
// Tests
// =============================================================================

test "CItem struct size and alignment" {
    // Verify ABI stability
    try std.testing.expectEqual(@as(usize, 56), @sizeOf(CItem));
    try std.testing.expectEqual(@as(usize, 8), @alignOf(CItem));
}

test "CMessageItem struct size and alignment" {
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(CMessageItem));
    try std.testing.expectEqual(@as(usize, 8), @alignOf(CMessageItem));
}

test "CContentPart struct size and alignment" {
    // Size increased from 56 to 72 with addition of quaternary_ptr/len for code_blocks_json
    try std.testing.expectEqual(@as(usize, 72), @sizeOf(CContentPart));
    try std.testing.expectEqual(@as(usize, 8), @alignOf(CContentPart));
}

test "talu_responses_create and free" {
    const handle = talu_responses_create();
    try std.testing.expect(handle != null);
    defer talu_responses_free(handle);

    const count = talu_responses_item_count(handle);
    try std.testing.expectEqual(@as(usize, 0), count);
}

test "talu_responses_item_type returns 255 for invalid index" {
    const handle = talu_responses_create();
    defer talu_responses_free(handle);

    const item_type = talu_responses_item_type(handle, 0);
    try std.testing.expectEqual(@as(u8, 255), item_type);
}

test "talu_responses_get_item returns error for invalid index" {
    const handle = talu_responses_create();
    defer talu_responses_free(handle);

    var item: CItem = undefined;
    const result = talu_responses_get_item(handle, 0, &item);
    try std.testing.expect(result != 0); // Should return error
}

// =============================================================================
// ABI Compatibility: Struct Size Validation
// =============================================================================
//
// ABI sizes are validated at compile time via comptime assertions.
// See core/src/capi/abi.zig for the canonical size definitions.
// Python bindings validate against the same sizes in tests/compat/abi/.

// =============================================================================
// Fuzz Tests
// =============================================================================

test "fuzz talu_responses_load_responses_json" {
    try std.testing.fuzz({}, struct {
        fn testOne(_: void, input: []const u8) !void {
            const handle = talu_responses_create();
            if (handle == null) return;
            defer talu_responses_free(handle);
            const alloc = std.testing.allocator;
            const json_z = try alloc.dupeZ(u8, input);
            defer alloc.free(json_z);
            const rc = talu_responses_load_responses_json(handle, json_z.ptr);
            if (rc != 0) {
                try std.testing.expect(capi_error.talu_last_error_code() != 0);
                try std.testing.expect(capi_error.talu_last_error() != null);
            }
        }
    }.testOne, .{});
}

test "fuzz talu_responses_load_completions_json" {
    try std.testing.fuzz({}, struct {
        fn testOne(_: void, input: []const u8) !void {
            const handle = talu_responses_create();
            if (handle == null) return;
            defer talu_responses_free(handle);
            const alloc = std.testing.allocator;
            const json_z = try alloc.dupeZ(u8, input);
            defer alloc.free(json_z);
            const rc = talu_responses_load_completions_json(handle, json_z.ptr);
            if (rc != 0) {
                try std.testing.expect(capi_error.talu_last_error_code() != 0);
                try std.testing.expect(capi_error.talu_last_error() != null);
            }
        }
    }.testOne, .{});
}

test "fuzz talu_chat_set_messages" {
    try std.testing.fuzz({}, struct {
        fn testOne(_: void, input: []const u8) !void {
            const chat_handle = talu_chat_create(null) orelse return;
            defer talu_chat_free(chat_handle);

            const alloc = std.testing.allocator;
            const json_z = try alloc.dupeZ(u8, input);
            defer alloc.free(json_z);
            const rc = talu_chat_set_messages(chat_handle, json_z.ptr);
            if (rc != 0) {
                try std.testing.expect(capi_error.talu_last_error_code() != 0);
                try std.testing.expect(capi_error.talu_last_error() != null);
            }
        }
    }.testOne, .{});
}

test "talu_responses_load_responses_json invalid json maps to invalid_argument" {
    const handle = talu_responses_create();
    if (handle == null) return error.OutOfMemory;
    defer talu_responses_free(handle);

    const rc = talu_responses_load_responses_json(handle, "{");
    try std.testing.expect(rc != 0);
    try std.testing.expectEqual(@as(i32, @intFromEnum(error_codes.ErrorCode.invalid_argument)), capi_error.talu_last_error_code());
}

test "talu_responses_load_completions_json invalid json maps to invalid_argument" {
    const handle = talu_responses_create();
    if (handle == null) return error.OutOfMemory;
    defer talu_responses_free(handle);

    const rc = talu_responses_load_completions_json(handle, "{");
    try std.testing.expect(rc != 0);
    try std.testing.expectEqual(@as(i32, @intFromEnum(error_codes.ErrorCode.invalid_argument)), capi_error.talu_last_error_code());
}

test "talu_chat_set_messages invalid json maps to invalid_argument" {
    const chat_handle = talu_chat_create(null) orelse return error.OutOfMemory;
    defer talu_chat_free(chat_handle);

    const rc = talu_chat_set_messages(chat_handle, "{");
    try std.testing.expect(rc != 0);
    try std.testing.expectEqual(@as(i32, @intFromEnum(error_codes.ErrorCode.invalid_argument)), capi_error.talu_last_error_code());
}
