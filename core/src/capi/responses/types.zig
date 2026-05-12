//! Responses C API handles and ABI-stable data transfer structs.
//!
//! These declarations are shared by the conversation, item, write, chat, and
//! router-facing C API modules. They are layout-sensitive FFI contracts.

/// Opaque Conversation handle for C API.
/// Wraps the Zig Conversation pointer.
pub const ResponsesHandle = opaque {};

/// Opaque Chat handle for C API.
/// Chat owns a Conversation and generation configuration.
pub const ChatHandle = opaque {};

/// Chat creation options.
pub const ChatCreateOptions = extern struct {
    offline: bool = false,
};

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
