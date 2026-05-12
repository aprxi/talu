//! Responses C API item and content accessors.
//!
//! Thread safety: NOT thread-safe. All access must be from a single thread.

const std = @import("std");
const conversation_mod = @import("../../responses/conversation/root.zig");
const types = @import("types.zig");
const Conversation = conversation_mod.Conversation;
const Item = conversation_mod.Item;
const ItemStatus = conversation_mod.ItemStatus;
const ContentPart = conversation_mod.ContentPart;
const ResponsesHandle = types.ResponsesHandle;
const CItem = types.CItem;
const CMessageItem = types.CMessageItem;
const CFunctionCallItem = types.CFunctionCallItem;
const CFunctionCallOutputItem = types.CFunctionCallOutputItem;
const CReasoningItem = types.CReasoningItem;
const CItemReferenceItem = types.CItemReferenceItem;
const CContentPart = types.CContentPart;

const capi_error = @import("../error.zig");
const error_codes = @import("../error_codes.zig");

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
