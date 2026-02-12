//! Conversation - Container for Open Responses Items.
//!
//! This module manages a linear history of Items (the atomic units of the
//! Open Responses architecture). It provides typed append functions for
//! each item type and handles ID generation.
//!
//! # Architecture
//!
//! The Conversation stores Items in insertion order. Each Item has a unique
//! u64 ID generated via a monotonic counter. The container supports:
//!   - Typed appenders for each ItemType
//!   - Lookup by ID (O(n) scan, future: hash map for O(1))
//!   - Serialization to Responses and Completions formats
//!
//! # Usage
//!
//! ```zig
//! const conv = try Conversation.init(allocator);
//! defer conv.deinit();
//!
//! const user_msg = try conv.appendUserMessage("Hello!");
//! const asst_msg = try conv.appendAssistantMessage();
//! try conv.appendTextContent(asst_msg, "Hi there!");
//! conv.finalizeItem(asst_msg);
//! ```
//!
//! Thread safety: NOT thread-safe. All access must be from a single thread.

const std = @import("std");
const json_mod = @import("../io/json/root.zig");
const items = @import("items.zig");
const backend_mod = @import("backend.zig");
const code_mod = @import("../validate/code/root.zig");

pub const Item = items.Item;
pub const ItemType = items.ItemType;
pub const ItemStatus = items.ItemStatus;
pub const ItemVariant = items.ItemVariant;
pub const MessageData = items.MessageData;
pub const MessageRole = items.MessageRole;
pub const ContentPart = items.ContentPart;
pub const ContentType = items.ContentType;
pub const FunctionCallData = items.FunctionCallData;
pub const FunctionCallOutputData = items.FunctionCallOutputData;
pub const ReasoningData = items.ReasoningData;
pub const ItemReferenceData = items.ItemReferenceData;
pub const ImageDetail = items.ImageDetail;

/// Serialization direction for schema-correct JSON output.
///
/// The Open Responses v2.3.0 schema distinguishes between:
/// - **Request (ItemParam):** What you send to the API. Constrained schemas.
/// - **Response (ItemField):** What the API returns. Richer schemas.
///
/// Key differences:
/// - `logprobs`: Only in OutputTextContent (response), NOT in OutputTextContentParam (request)
/// - `annotations`: UrlCitationParam[] in requests, Annotation[] in responses
/// - Reasoning `content`: OMIT for requests (ReasoningItemParam), allowed in responses
/// - `status`: OpenAI only accepts `in_progress`, `completed`, `incomplete` (not `failed`)
pub const SerializationDirection = enum {
    /// Serialize as request payload (ItemParam schemas).
    /// - Omit logprobs from output_text
    /// - Filter annotations to url_citation only
    /// - Omit reasoning content
    /// - Map status=failed to "incomplete"
    request,

    /// Serialize as response/storage (ItemField schemas).
    /// - Include logprobs if present
    /// - Include all annotations
    /// - Include reasoning content if present
    /// - Map status=failed to "incomplete"
    response,
};

/// Map ItemStatus to OpenAI-compatible string.
/// OpenAI FunctionCallItemStatus only accepts: in_progress, completed, incomplete.
/// Internal `failed` status maps to `incomplete` for API compatibility.
fn statusToOpenAIString(status: ItemStatus) []const u8 {
    return switch (status) {
        .in_progress => "in_progress",
        .waiting => "in_progress",
        .completed => "completed",
        .incomplete => "incomplete",
        .failed => "incomplete", // Map failed -> incomplete for OpenAI compliance
    };
}

/// Options for Responses format serialization.
pub const ResponsesSerializationOptions = struct {
    /// Direction determines which schema variant to use.
    direction: SerializationDirection = .request,
};

/// Get current Unix wall-clock timestamp in milliseconds since epoch.
fn currentTimeMs() i64 {
    return std.time.milliTimestamp();
}

/// Conversation - manages a linear history of Items.
///
/// Provides typed append functions for creating new Items, ID generation,
/// and serialization to different JSON formats (Responses vs Completions).
///
/// Thread safety: NOT thread-safe. All access must be from a single thread.
pub const Conversation = struct {
    /// Magic number for detecting use-after-free and memory corruption.
    /// Set to MAGIC_VALID on init, MAGIC_FREED on deinit.
    pub const MAGIC_VALID: u64 = 0xCAFE_BABE_DEAD_BEEF;
    pub const MAGIC_FREED: u64 = 0xDEAD_DEAD_DEAD_DEAD;
    _magic: u64 = MAGIC_VALID,

    allocator: std.mem.Allocator,

    /// Session identifier for this conversation.
    /// Used for persistence (hashed to SESSION_HASH in StoreFS).
    session_id: ?[]const u8 = null,

    /// Retention expiry (Unix ms). 0 = no expiry.
    ttl_ts: i64 = 0,

    /// Item storage (insertion order).
    items_list: std.ArrayListUnmanaged(Item),

    /// Monotonic counter for generating unique item IDs.
    _next_item_id: u64 = 0,

    /// Monotonic counter for fork transaction boundaries.
    _fork_sequence: u64 = 0,

    /// Optional storage backend for persistence.
    storage_backend: ?backend_mod.StorageBackend = null,

    /// Last storage error (sticky).
    last_storage_error: ?anyerror = null,

    /// Double-pointer for stable Python access.
    items_ptr: *[*]Item,
    _items_ptr_storage: [*]Item,

    /// Pointer to item count for live access.
    count_ptr: *usize,
    _count_storage: usize,

    /// Create a new empty Conversation.
    pub fn init(allocator: std.mem.Allocator) !*Conversation {
        return initWithStorage(allocator, null, null);
    }

    /// Create a new Conversation with optional session ID.
    pub fn initWithSession(allocator: std.mem.Allocator, session_id: ?[]const u8) !*Conversation {
        return initWithStorage(allocator, session_id, null);
    }

    /// Create a new Conversation with storage backend.
    pub fn initWithStorage(
        allocator: std.mem.Allocator,
        session_id_arg: ?[]const u8,
        storage_backend_arg: ?backend_mod.StorageBackend,
    ) !*Conversation {
        const self = try allocator.create(Conversation);
        errdefer allocator.destroy(self);

        const owned_session_id: ?[]const u8 = if (session_id_arg) |sid|
            try allocator.dupe(u8, sid)
        else
            null;
        errdefer if (owned_session_id) |sid| allocator.free(sid);

        self.* = .{
            ._magic = MAGIC_VALID,
            .allocator = allocator,
            .session_id = owned_session_id,
            .items_list = .{},
            ._next_item_id = 0,
            .storage_backend = storage_backend_arg,
            .last_storage_error = null,
            ._items_ptr_storage = undefined,
            .items_ptr = undefined,
            ._count_storage = 0,
            .count_ptr = undefined,
        };

        self.items_ptr = &self._items_ptr_storage;
        self.count_ptr = &self._count_storage;
        self._items_ptr_storage = self.items_list.items.ptr;

        return self;
    }

    /// Free all resources.
    pub fn deinit(self: *Conversation) void {
        // Validate magic number to detect double-free
        std.debug.assert(self._magic == MAGIC_VALID);

        if (self.storage_backend) |sb| {
            sb.deinit();
        }

        if (self.session_id) |sid| {
            self.allocator.free(sid);
        }

        for (self.items_list.items) |*item| {
            item.deinit(self.allocator);
        }
        self.items_list.deinit(self.allocator);

        // Mark as freed before destroying to detect use-after-free
        self._magic = MAGIC_FREED;
        self.allocator.destroy(self);
    }

    /// Validate that this Conversation is valid (not freed or corrupted).
    /// Returns true if valid, false if corrupted or freed.
    pub fn isValid(self: *const Conversation) bool {
        return self._magic == MAGIC_VALID;
    }

    /// Update pointers after list modification.
    fn updatePointers(self: *Conversation) void {
        self._items_ptr_storage = self.items_list.items.ptr;
        self._count_storage = self.items_list.items.len;
    }

    /// Get the number of items.
    pub fn len(self: *const Conversation) usize {
        return self.items_list.items.len;
    }

    /// Get an item by index.
    pub fn getItem(self: *const Conversation, index: usize) ?*const Item {
        if (index >= self.items_list.items.len) return null;
        return &self.items_list.items[index];
    }

    /// Get a mutable item by index.
    pub fn getItemMut(self: *Conversation, index: usize) ?*Item {
        if (index >= self.items_list.items.len) return null;
        return &self.items_list.items[index];
    }

    /// Get the last item.
    pub fn lastItem(self: *const Conversation) ?*const Item {
        if (self.items_list.items.len == 0) return null;
        return &self.items_list.items[self.items_list.items.len - 1];
    }

    /// Find an item by ID.
    pub fn findById(self: *const Conversation, id: u64) ?*const Item {
        for (self.items_list.items) |*item| {
            if (item.id == id) return item;
        }
        return null;
    }

    // =========================================================================
    // Message Item Appenders
    // =========================================================================

    /// Append a user message with text content.
    /// Returns the new Item for further modification.
    pub fn appendUserMessage(self: *Conversation, text: []const u8) !*Item {
        return self.appendMessage(.user, .input_text, text);
    }

    /// Append a system message with text content.
    pub fn appendSystemMessage(self: *Conversation, text: []const u8) !*Item {
        return self.appendMessage(.system, .input_text, text);
    }

    /// Append a developer message with text content.
    /// For o1/o3 models. Maps to system for legacy backends.
    pub fn appendDeveloperMessage(self: *Conversation, text: []const u8) !*Item {
        return self.appendMessage(.developer, .input_text, text);
    }

    /// Append an empty assistant message (for streaming).
    /// Add content via appendTextContent().
    pub fn appendAssistantMessage(self: *Conversation) !*Item {
        const item_id = self._next_item_id;
        self._next_item_id += 1;

        const item = Item{
            .id = item_id,
            .created_at_ms = 0, // Set at finalize time
            .ttl_ts = self.ttl_ts,
            .data = ItemVariant{
                .message = MessageData{
                    .role = .assistant,
                    .status = .in_progress,
                    .content = .{},
                },
            },
            .metadata = null,
        };

        try self.items_list.append(self.allocator, item);
        self.updatePointers();

        return &self.items_list.items[self.items_list.items.len - 1];
    }

    /// Internal: append a message with initial text content.
    fn appendMessage(self: *Conversation, role: MessageRole, content_type: ContentType, text: []const u8) !*Item {
        const item_id = self._next_item_id;
        self._next_item_id += 1;

        var content_list: std.ArrayListUnmanaged(ContentPart) = .{};
        errdefer {
            for (content_list.items) |*part| part.deinit(self.allocator);
            content_list.deinit(self.allocator);
        }

        // Create content part using factory function based on type
        var part = switch (content_type) {
            .input_text => ContentPart.initInputText(),
            .output_text => ContentPart.initOutputText(),
            .text => ContentPart.initText(),
            .input_image => ContentPart.initInputImage(.auto),
            .input_audio => ContentPart.initInputAudio(),
            .input_video => ContentPart.initInputVideo(),
            .input_file => ContentPart.initInputFile(),
            .refusal => ContentPart.initRefusal(),
            .reasoning_text => ContentPart.initReasoningText(),
            .summary_text => ContentPart.initSummaryText(),
            .unknown => ContentPart.initUnknown(),
        };
        try part.appendData(self.allocator, text);
        try content_list.append(self.allocator, part);

        const item = Item{
            .id = item_id,
            .created_at_ms = currentTimeMs(),
            .ttl_ts = self.ttl_ts,
            .data = ItemVariant{
                .message = MessageData{
                    .role = role,
                    .status = .completed,
                    .content = content_list,
                },
            },
            .metadata = null,
        };

        try self.items_list.append(self.allocator, item);
        self.updatePointers();

        const result = &self.items_list.items[self.items_list.items.len - 1];

        // For user/system/developer messages that are already complete,
        // notify storage immediately (assistant messages are finalized later during streaming).
        if (role != .assistant) {
            self.notifyStorage(result);
        }

        return result;
    }

    /// Internal: append a message with initial text content and hidden flag.
    pub fn appendMessageWithHidden(
        self: *Conversation,
        role: MessageRole,
        content_type: ContentType,
        text: []const u8,
        hidden: bool,
    ) !*Item {
        const item_id = self._next_item_id;
        self._next_item_id += 1;

        var content_list: std.ArrayListUnmanaged(ContentPart) = .{};
        errdefer {
            for (content_list.items) |*part| part.deinit(self.allocator);
            content_list.deinit(self.allocator);
        }

        var part = switch (content_type) {
            .input_text => ContentPart.initInputText(),
            .output_text => ContentPart.initOutputText(),
            .text => ContentPart.initText(),
            .input_image => ContentPart.initInputImage(.auto),
            .input_audio => ContentPart.initInputAudio(),
            .input_video => ContentPart.initInputVideo(),
            .input_file => ContentPart.initInputFile(),
            .refusal => ContentPart.initRefusal(),
            .reasoning_text => ContentPart.initReasoningText(),
            .summary_text => ContentPart.initSummaryText(),
            .unknown => ContentPart.initUnknown(),
        };
        try part.appendData(self.allocator, text);
        try content_list.append(self.allocator, part);

        const item = Item{
            .id = item_id,
            .created_at_ms = currentTimeMs(),
            .ttl_ts = self.ttl_ts,
            .hidden = hidden,
            .data = ItemVariant{
                .message = MessageData{
                    .role = role,
                    .status = .completed,
                    .content = content_list,
                },
            },
            .metadata = null,
        };

        try self.items_list.append(self.allocator, item);
        self.updatePointers();

        const result = &self.items_list.items[self.items_list.items.len - 1];

        if (role != .assistant) {
            self.notifyStorage(result);
        }

        return result;
    }

    /// Notify storage backend of a finalized item (internal helper).
    fn notifyStorage(self: *Conversation, item: *const Item) void {
        if (self.storage_backend) |sb| {
            // Create a deep copy ItemRecord for storage
            var record = backend_mod.ItemRecord.fromItem(self.allocator, item) catch |err| {
                self.last_storage_error = err;
                return;
            };
            defer record.deinit(self.allocator);

            const event = backend_mod.StorageEvent{ .PutItem = record };
            sb.onEvent(&event) catch |err| {
                self.last_storage_error = err;
            };
        }
    }

    /// Notify storage backend of session metadata update.
    ///
    /// Call this when session metadata changes (config, title, system_prompt, marker).
    /// The storage backend will receive a PutSession event with the current
    /// session state, enabling external persistence of session data.
    ///
    /// This ensures Zig is the single source of truth for all session data.
    pub fn notifySessionUpdate(
        self: *Conversation,
        model: ?[]const u8,
        title: ?[]const u8,
        system_prompt: ?[]const u8,
        config_json: ?[]const u8,
        marker: ?[]const u8,
        parent_session_id: ?[]const u8,
        group_id: ?[]const u8,
        metadata_json: ?[]const u8,
        source_doc_id: ?[]const u8,
    ) void {
        if (self.storage_backend) |sb| {
            // session_id is required for PutSession
            const session_id = self.session_id orelse return;

            const now_ms = currentTimeMs();
            const head_item_id: u64 = if (self.items_list.items.len > 0)
                self.items_list.items[self.items_list.items.len - 1].id
            else
                0;
            const record = backend_mod.SessionRecord{
                .session_id = session_id,
                .model = model,
                .title = title,
                .system_prompt = system_prompt,
                .config_json = config_json,
                .marker = marker,
                .parent_session_id = parent_session_id,
                .group_id = group_id,
                .head_item_id = head_item_id,
                .ttl_ts = self.ttl_ts,
                .metadata_json = metadata_json,
                .source_doc_id = source_doc_id,
                // New session events use now for both timestamps; storage can preserve prior created_at_ms.
                .created_at_ms = now_ms,
                .updated_at_ms = now_ms,
            };

            const event = backend_mod.StorageEvent{ .PutSession = record };
            sb.onEvent(&event) catch |err| {
                self.last_storage_error = err;
            };
        }
    }

    /// Insert a message with text content at the specified index.
    /// Returns the new Item for further modification.
    pub fn insertMessage(self: *Conversation, index: usize, role: MessageRole, content_type: ContentType, text: []const u8) !*Item {
        // Validate index (can insert at end, which is same as append)
        if (index > self.items_list.items.len) return error.IndexOutOfBounds;

        const item_id = self._next_item_id;
        self._next_item_id += 1;

        var content_list: std.ArrayListUnmanaged(ContentPart) = .{};
        errdefer {
            for (content_list.items) |*part| part.deinit(self.allocator);
            content_list.deinit(self.allocator);
        }

        // Create content part using factory function based on type
        var part = switch (content_type) {
            .input_text => ContentPart.initInputText(),
            .output_text => ContentPart.initOutputText(),
            .text => ContentPart.initText(),
            .input_image => ContentPart.initInputImage(.auto),
            .input_audio => ContentPart.initInputAudio(),
            .input_video => ContentPart.initInputVideo(),
            .input_file => ContentPart.initInputFile(),
            .refusal => ContentPart.initRefusal(),
            .reasoning_text => ContentPart.initReasoningText(),
            .summary_text => ContentPart.initSummaryText(),
            .unknown => ContentPart.initUnknown(),
        };
        try part.appendData(self.allocator, text);
        try content_list.append(self.allocator, part);

        const item = Item{
            .id = item_id,
            .created_at_ms = currentTimeMs(),
            .ttl_ts = self.ttl_ts,
            .data = ItemVariant{
                .message = MessageData{
                    .role = role,
                    .status = .completed,
                    .content = content_list,
                },
            },
            .metadata = null,
        };

        try self.items_list.insert(self.allocator, index, item);
        self.updatePointers();

        return &self.items_list.items[index];
    }

    /// Append text content to a message item (for streaming).
    pub fn appendTextContent(self: *Conversation, item: *Item, text: []const u8) !void {
        const msg = switch (item.data) {
            .message => |*m| m,
            else => return error.NotAMessage,
        };

        // If no content parts, create one
        if (msg.content.items.len == 0) {
            var part = if (msg.role == .assistant)
                ContentPart.initOutputText()
            else
                ContentPart.initInputText();
            try part.appendData(self.allocator, text);
            try msg.content.append(self.allocator, part);
        } else {
            // Append to last part
            const last_part = &msg.content.items[msg.content.items.len - 1];
            try last_part.appendData(self.allocator, text);
        }
    }

    /// Add a new content part to a message item.
    pub fn addContentPart(self: *Conversation, item: *Item, content_type: ContentType) !*ContentPart {
        const msg = switch (item.data) {
            .message => |*m| m,
            else => return error.NotAMessage,
        };

        // Create content part using factory function
        const part = switch (content_type) {
            .input_text => ContentPart.initInputText(),
            .output_text => ContentPart.initOutputText(),
            .text => ContentPart.initText(),
            .input_image => ContentPart.initInputImage(.auto),
            .input_audio => ContentPart.initInputAudio(),
            .input_video => ContentPart.initInputVideo(),
            .input_file => ContentPart.initInputFile(),
            .refusal => ContentPart.initRefusal(),
            .reasoning_text => ContentPart.initReasoningText(),
            .summary_text => ContentPart.initSummaryText(),
            .unknown => ContentPart.initUnknown(),
        };
        try msg.content.append(self.allocator, part);

        return &msg.content.items[msg.content.items.len - 1];
    }

    // =========================================================================
    // Function Call Appenders
    // =========================================================================

    /// Append a function call (tool intent) item.
    pub fn appendFunctionCall(self: *Conversation, call_id: []const u8, name: []const u8) !*Item {
        const item_id = self._next_item_id;
        self._next_item_id += 1;

        const item = Item{
            .id = item_id,
            .created_at_ms = currentTimeMs(),
            .ttl_ts = self.ttl_ts,
            .data = ItemVariant{
                .function_call = FunctionCallData{
                    .call_id = try self.allocator.dupeZ(u8, call_id),
                    .name = try self.allocator.dupeZ(u8, name),
                    .arguments = .{},
                    .status = .in_progress,
                },
            },
            .metadata = null,
        };

        try self.items_list.append(self.allocator, item);
        self.updatePointers();

        return &self.items_list.items[self.items_list.items.len - 1];
    }

    /// Set arguments for a function call item.
    pub fn setFunctionCallArguments(self: *Conversation, item: *Item, arguments: []const u8) !void {
        const fc = switch (item.data) {
            .function_call => |*f| f,
            else => return error.NotAFunctionCall,
        };

        fc.arguments.clearRetainingCapacity();
        try fc.arguments.appendSlice(self.allocator, arguments);
    }

    /// Append a function call output (tool result) item with text output.
    pub fn appendFunctionCallOutput(self: *Conversation, call_id: []const u8, output: []const u8) !*Item {
        const item_id = self._next_item_id;
        self._next_item_id += 1;

        // Create text output (most common case)
        var text_output: std.ArrayListUnmanaged(u8) = .{};
        errdefer text_output.deinit(self.allocator);
        try text_output.appendSlice(self.allocator, output);

        const item = Item{
            .id = item_id,
            .created_at_ms = currentTimeMs(),
            .ttl_ts = self.ttl_ts,
            .data = ItemVariant{
                .function_call_output = FunctionCallOutputData{
                    .call_id = try self.allocator.dupeZ(u8, call_id),
                    .output = .{ .text = text_output },
                    .status = .completed,
                },
            },
            .metadata = null,
        };

        try self.items_list.append(self.allocator, item);
        self.updatePointers();

        const result = &self.items_list.items[self.items_list.items.len - 1];

        // Function call outputs with text are immediately complete - notify storage
        self.notifyStorage(result);

        return result;
    }

    /// Append a function call output with multimodal content parts.
    pub fn appendFunctionCallOutputParts(self: *Conversation, call_id: []const u8) !*Item {
        const item_id = self._next_item_id;
        self._next_item_id += 1;

        const item = Item{
            .id = item_id,
            .created_at_ms = currentTimeMs(),
            .ttl_ts = self.ttl_ts,
            .data = ItemVariant{
                .function_call_output = FunctionCallOutputData{
                    .call_id = try self.allocator.dupeZ(u8, call_id),
                    .output = .{ .parts = .{} },
                    .status = .in_progress,
                },
            },
            .metadata = null,
        };

        try self.items_list.append(self.allocator, item);
        self.updatePointers();

        return &self.items_list.items[self.items_list.items.len - 1];
    }

    // =========================================================================
    // Reasoning Appenders
    // =========================================================================

    /// Append a reasoning item.
    pub fn appendReasoning(self: *Conversation) !*Item {
        const item_id = self._next_item_id;
        self._next_item_id += 1;

        const item = Item{
            .id = item_id,
            .created_at_ms = currentTimeMs(),
            .ttl_ts = self.ttl_ts,
            .data = ItemVariant{
                .reasoning = ReasoningData{
                    .content = .{},
                    .summary = .{},
                    .encrypted_content = null,
                },
            },
            .metadata = null,
        };

        try self.items_list.append(self.allocator, item);
        self.updatePointers();

        return &self.items_list.items[self.items_list.items.len - 1];
    }

    /// Add summary text to a reasoning item.
    pub fn addReasoningSummary(self: *Conversation, item: *Item, summary_text: []const u8) !void {
        const rd = switch (item.data) {
            .reasoning => |*r| r,
            else => return error.NotReasoning,
        };

        var part = ContentPart.initSummaryText();
        try part.appendData(self.allocator, summary_text);
        try rd.summary.append(self.allocator, part);
    }

    /// Add reasoning text content to a reasoning item.
    /// Parallel to addReasoningSummary but appends to content (not summary).
    pub fn addReasoningContent(self: *Conversation, item: *Item, text: []const u8) !void {
        const rd = switch (item.data) {
            .reasoning => |*r| r,
            else => return error.NotReasoning,
        };

        var part = ContentPart.initReasoningText();
        try part.appendData(self.allocator, text);
        try rd.content.append(self.allocator, part);
    }

    /// Set encrypted content for a reasoning item.
    pub fn setReasoningEncryptedContent(self: *Conversation, item: *Item, encrypted: []const u8) !void {
        const rd = switch (item.data) {
            .reasoning => |*r| r,
            else => return error.NotReasoning,
        };

        if (rd.encrypted_content) |old| self.allocator.free(old);
        rd.encrypted_content = try self.allocator.dupe(u8, encrypted);
    }

    // =========================================================================
    // Item Reference Appenders
    // =========================================================================

    /// Append an item reference (for context replay).
    pub fn appendItemReference(self: *Conversation, target_id: []const u8) !*Item {
        const item_id = self._next_item_id;
        self._next_item_id += 1;

        const item = Item{
            .id = item_id,
            .created_at_ms = currentTimeMs(),
            .ttl_ts = self.ttl_ts,
            .data = ItemVariant{
                .item_reference = ItemReferenceData{
                    .id = try self.allocator.dupeZ(u8, target_id),
                },
            },
            .metadata = null,
        };

        try self.items_list.append(self.allocator, item);
        self.updatePointers();

        return &self.items_list.items[self.items_list.items.len - 1];
    }

    // =========================================================================
    // Item Lifecycle
    // =========================================================================

    /// Finalize an item (mark as completed, set timestamp).
    /// Emits PutItem event to storage backend if configured.
    pub fn finalizeItem(self: *Conversation, item: *Item) void {
        if (item.created_at_ms == 0) {
            item.created_at_ms = currentTimeMs();
        }

        // Update status based on item type
        switch (item.data) {
            .message => |*m| m.status = .completed,
            .function_call => |*f| f.status = .completed,
            .function_call_output => |*f| f.status = .completed,
            else => {},
        }

        // Extract code blocks for output_text content (assistant messages)
        self.extractCodeBlocksForItem(item);

        // Notify storage backend if configured
        if (self.storage_backend) |sb| {
            // Create a deep copy ItemRecord for storage
            // Must use deep copy because ItemContentPartRecord is a different type than ContentPart
            var record = backend_mod.ItemRecord.fromItem(self.allocator, item) catch |err| {
                self.last_storage_error = err;
                return;
            };
            defer record.deinit(self.allocator); // Free after event consumed

            const event = backend_mod.StorageEvent{ .PutItem = record };
            sb.onEvent(&event) catch |err| {
                self.last_storage_error = err;
            };
        }
    }

    /// Extract code blocks from output_text content parts and store as JSON.
    /// Called at finalization time to populate code_blocks_json metadata.
    fn extractCodeBlocksForItem(self: *Conversation, item: *Item) void {
        const msg = switch (item.data) {
            .message => |*m| m,
            else => return, // Only messages have content parts
        };

        for (msg.content.items) |*part| {
            switch (part.variant) {
                .output_text => |*ot| {
                    // Skip if already has code blocks
                    if (ot.code_blocks_json != null) continue;

                    const text = ot.text.items;
                    if (text.len == 0) continue;

                    // Extract code blocks
                    var blocks = code_mod.extractCodeBlocks(self.allocator, text) catch continue;
                    defer blocks.deinit();

                    if (blocks.count() == 0) continue;

                    // Serialize to JSON
                    const json = blocks.toJson(self.allocator) catch continue;
                    ot.code_blocks_json = json;
                },
                else => {},
            }
        }
    }

    /// Set metadata on an item.
    pub fn setItemMetadata(self: *Conversation, item: *Item, json: []const u8) !void {
        if (item.metadata) |old| {
            self.allocator.free(old);
        }
        item.metadata = try self.allocator.dupe(u8, json);
    }

    /// Delete an item by index.
    /// Emits DeleteItem event to storage backend if configured.
    pub fn deleteItem(self: *Conversation, index: usize) bool {
        if (index >= self.items_list.items.len) return false;

        var item = self.items_list.items[index];
        const item_id = item.id; // Capture ID before destruction

        // Notify storage backend before destroying item
        if (self.storage_backend) |sb| {
            const event = backend_mod.StorageEvent{ .DeleteItem = .{
                .item_id = item_id,
                .deleted_at_ms = currentTimeMs(),
            } };
            sb.onEvent(&event) catch |err| {
                self.last_storage_error = err;
            };
        }

        item.deinit(self.allocator);

        _ = self.items_list.orderedRemove(index);
        self.updatePointers();

        return true;
    }

    /// Clone items from another conversation into this one.
    /// Preserves item timestamps and metadata and emits PutItems in batches.
    pub fn cloneFrom(self: *Conversation, source: *const Conversation, batch_size: usize) !void {
        try self.cloneFromRange(source, 0, source.items_list.items.len, batch_size);
    }

    /// Emit a fork begin event for storage backends.
    pub fn beginFork(self: *Conversation) u64 {
        if (self.storage_backend) |sb| {
            const session_id = self.session_id orelse return 0;
            self._fork_sequence += 1;
            const fork_id = self._fork_sequence;
            const event = backend_mod.StorageEvent{ .BeginFork = .{
                .fork_id = fork_id,
                .session_id = session_id,
            } };
            sb.onEvent(&event) catch |err| {
                self.last_storage_error = err;
            };
            return fork_id;
        }
        return 0;
    }

    /// Emit a fork end event for storage backends.
    pub fn endFork(self: *Conversation, fork_id: u64) void {
        if (fork_id == 0) return;
        if (self.storage_backend) |sb| {
            const session_id = self.session_id orelse return;
            const event = backend_mod.StorageEvent{ .EndFork = .{
                .fork_id = fork_id,
                .session_id = session_id,
            } };
            sb.onEvent(&event) catch |err| {
                self.last_storage_error = err;
            };
        }
    }

    /// Clone items from another conversation into this one, keeping only a prefix.
    /// last_index is inclusive; if out of range, clones all items.
    pub fn cloneFromPrefix(
        self: *Conversation,
        source: *const Conversation,
        last_index: usize,
        batch_size: usize,
    ) !void {
        const end_index = if (source.items_list.items.len == 0)
            0
        else if (last_index >= source.items_list.items.len)
            source.items_list.items.len - 1
        else
            last_index;
        try self.cloneFromRange(source, 0, end_index + 1, batch_size);
    }

    /// Clone a range of items from another conversation into this one.
    /// The range is [start_index, end_index) in the source list.
    ///
    /// # Performance Characteristics
    ///
    /// This performs a deep copy of all items including content (text, images, etc.).
    /// Cost is O(n) where n is the total content size. For conversations with large
    /// multimodal content (e.g., base64 images), this can be significant.
    ///
    /// Storage serialization adds overhead only when a persistence backend is configured.
    /// The default MemoryBackend is a no-op. Batching (default 1000 items) amortizes
    /// the per-event overhead for persistence backends.
    ///
    /// # Storage Consistency (Important)
    ///
    /// Items are cloned to memory BEFORE storage events are emitted. If storage
    /// fails mid-fork (e.g., database full, network error), items remain in memory
    /// but are not persisted. This is intentional:
    ///
    ///   - In-memory chat continues to work (fail-open for UX)
    ///   - Storage error is captured in `last_storage_error`
    ///   - Caller SHOULD check `hasStorageError()` after fork if persistence is critical
    ///   - On restart, unpersisted items are lost (storage is source of truth for restore)
    ///
    /// This design prioritizes availability over strict consistency. For strict
    /// consistency, wrap fork in BeginFork/EndFork and check errors.
    pub fn cloneFromRange(
        self: *Conversation,
        source: *const Conversation,
        start_index: usize,
        end_index: usize,
        batch_size: usize,
    ) !void {
        if (self.items_list.items.len != 0) return error.InvalidState;
        if (start_index > end_index) return error.InvalidArgument;

        const effective_batch_size: usize = if (batch_size == 0) 1000 else batch_size;
        var next_item_id: u64 = 0;
        const source_session_id = source.session_id;
        const clamped_end = @min(end_index, source.items_list.items.len);

        // Phase 1: Deep copy items to memory (always succeeds or fails atomically)
        var idx: usize = start_index;
        while (idx < clamped_end) : (idx += 1) {
            const source_item = &source.items_list.items[idx];
            const cloned = try source_item.cloneWithOrigin(self.allocator, source_session_id);
            try self.items_list.append(self.allocator, cloned);

            if (cloned.id >= next_item_id) {
                next_item_id = cloned.id + 1;
            }
        }

        // Phase 2: Emit storage events (errors are captured, not propagated)
        // Note: If storage fails here, items are in memory but not persisted.
        // Caller should check hasStorageError() if persistence is required.
        if (self.storage_backend) |sb| {
            var batch: std.ArrayListUnmanaged(backend_mod.ItemRecord) = .{};
            defer {
                for (batch.items) |*record| {
                    record.deinit(self.allocator);
                }
                batch.deinit(self.allocator);
            }

            for (self.items_list.items) |*item| {
                const record = try backend_mod.ItemRecord.fromItem(self.allocator, item);
                try batch.append(self.allocator, record);

                if (batch.items.len >= effective_batch_size) {
                    const event = backend_mod.StorageEvent{ .PutItems = batch.items };
                    sb.onEvent(&event) catch |err| {
                        self.last_storage_error = err;
                    };
                    for (batch.items) |*item_record| {
                        item_record.deinit(self.allocator);
                    }
                    batch.clearRetainingCapacity();
                }
            }

            if (batch.items.len > 0) {
                const event = backend_mod.StorageEvent{ .PutItems = batch.items };
                sb.onEvent(&event) catch |err| {
                    self.last_storage_error = err;
                };
                for (batch.items) |*item_record| {
                    item_record.deinit(self.allocator);
                }
                batch.clearRetainingCapacity();
            }
        }

        self._next_item_id = next_item_id;
        self.updatePointers();
    }

    /// Remove items after the specified index (inclusive).
    /// Emits DeleteItem events for removed items.
    pub fn truncateAfterIndex(self: *Conversation, last_index: usize) bool {
        if (self.items_list.items.len == 0) return true;
        if (last_index >= self.items_list.items.len) return true;

        var idx: usize = self.items_list.items.len;
        while (idx > last_index + 1) {
            idx -= 1;
            _ = self.deleteItem(idx);
        }

        self.updatePointers();
        return true;
    }

    /// Append a storage record directly without emitting storage events.
    pub fn appendItemRecord(self: *Conversation, record: *const backend_mod.ItemRecord) !void {
        const item = try backend_mod.itemFromRecord(self.allocator, record);
        try self.items_list.append(self.allocator, item);
        if (item.id >= self._next_item_id) {
            self._next_item_id = item.id + 1;
        }
        self.updatePointers();
    }

    /// Update structured validation flags for an item and emit storage update.
    pub fn setItemValidationFlags(
        self: *Conversation,
        item_index: usize,
        json_valid: bool,
        schema_valid: bool,
        repaired: bool,
    ) !void {
        if (item_index >= self.items_list.items.len) return error.InvalidArgument;

        const item = &self.items_list.items[item_index];
        item.json_valid = json_valid;
        item.schema_valid = schema_valid;
        item.repaired = repaired;
        self.notifyStorage(item);
    }

    /// Set the status of an item at the given index.
    /// Useful for marking items as incomplete (truncated) or failed (content filter).
    pub fn setItemStatus(self: *Conversation, item_index: usize, status: items.ItemStatus) !void {
        if (item_index >= self.items_list.items.len) return error.InvalidArgument;

        const item = &self.items_list.items[item_index];
        item.data.setStatus(status);
        self.notifyStorage(item);
    }

    /// Load storage records into an empty conversation without emitting storage events.
    pub fn loadItemRecords(self: *Conversation, records: []const backend_mod.ItemRecord) !void {
        if (self.items_list.items.len != 0) return error.InvalidState;

        for (records) |*record| {
            try self.appendItemRecord(record);
        }
    }

    /// Load items from the configured storage backend into an empty conversation.
    /// No-op if no storage backend is configured or it returns no records.
    pub fn loadFromStorageBackend(self: *Conversation) !void {
        const storage_backend = self.storage_backend orelse return;
        const records = try storage_backend.loadAll(self.allocator);
        defer backend_mod.freeItemRecords(self.allocator, records);

        if (records.len == 0) return;
        try self.loadItemRecords(records);
    }

    /// Clear all items.
    /// Emits ClearItems event to storage backend if configured.
    pub fn clear(self: *Conversation) void {
        // Notify storage backend before clearing
        if (self.storage_backend) |sb| {
            const event = backend_mod.StorageEvent{ .ClearItems = .{
                .cleared_at_ms = currentTimeMs(),
                .keep_context = false,
            } };
            sb.onEvent(&event) catch |err| {
                self.last_storage_error = err;
            };
        }

        for (self.items_list.items) |*item| {
            item.deinit(self.allocator);
        }
        self.items_list.clearRetainingCapacity();
        self.updatePointers();
    }

    /// Clear all items except the first system message (if present).
    /// Emits ClearItems event to storage backend if configured.
    pub fn clearKeepingSystem(self: *Conversation) void {
        if (self.items_list.items.len == 0) return;

        // Check if first item is a system message
        const first_item = &self.items_list.items[0];
        const has_system = switch (first_item.data) {
            .message => |m| m.role == .system or m.role == .developer,
            else => false,
        };

        // Notify storage backend before clearing
        if (self.storage_backend) |sb| {
            const event = backend_mod.StorageEvent{
                .ClearItems = .{
                    .cleared_at_ms = currentTimeMs(),
                    .keep_context = true, // clearKeepingSystem preserves system/developer if present
                },
            };
            sb.onEvent(&event) catch |err| {
                self.last_storage_error = err;
            };
        }

        if (has_system) {
            // Free all items except first
            for (self.items_list.items[1..]) |*item| {
                item.deinit(self.allocator);
            }
            self.items_list.shrinkRetainingCapacity(1);
        } else {
            // No storage notification needed - already sent above
            for (self.items_list.items) |*item| {
                item.deinit(self.allocator);
            }
            self.items_list.clearRetainingCapacity();
        }
        self.updatePointers();
    }

    // =========================================================================
    // Serialization
    // =========================================================================

    /// Serialize to Responses format JSON.
    /// Default: request direction (safe for sending to API).
    /// Caller owns returned memory.
    pub fn toResponsesJson(self: *const Conversation) ![]u8 {
        return self.toResponsesJsonWithOptions(.{});
    }

    /// Serialize to Responses format JSON with explicit options.
    /// Use `direction: .response` when storing/echoing API outputs.
    /// Caller owns returned memory.
    pub fn toResponsesJsonWithOptions(self: *const Conversation, opts: ResponsesSerializationOptions) ![]u8 {
        var json_bytes: std.ArrayListUnmanaged(u8) = .{};
        errdefer json_bytes.deinit(self.allocator);
        const writer = json_bytes.writer(self.allocator);

        try writer.writeByte('[');

        for (self.items_list.items, 0..) |*item, i| {
            if (i > 0) try writer.writeByte(',');
            try self.writeItemJson(writer, item, opts.direction);
        }

        try writer.writeByte(']');

        return json_bytes.toOwnedSlice(self.allocator);
    }

    /// Write a single item as JSON.
    fn writeItemJson(self: *const Conversation, writer: anytype, item: *const Item, direction: SerializationDirection) !void {
        switch (item.data) {
            .message => |*m| {
                try writer.print("{{\"type\":\"message\",\"id\":\"{s}{d}\",\"role\":\"{s}\",\"status\":\"{s}\",\"content\":", .{
                    "msg_",
                    item.id,
                    m.getRoleString(),
                    statusToOpenAIString(m.status),
                });
                try self.writeContentPartsJson(writer, m.content.items, direction);
                if (item.generation_json) |gen| {
                    try writer.writeAll(",\"generation\":");
                    try writer.writeAll(gen);
                }
                if (item.finish_reason) |fr| {
                    try writer.print(",\"finish_reason\":\"{s}\"", .{fr});
                }
                try writer.writeByte('}');
            },
            .function_call => |*f| {
                try writer.print("{{\"type\":\"function_call\",\"id\":\"{s}{d}\",\"call_id\":\"{s}\",\"name\":\"{s}\",\"arguments\":{f},\"status\":\"{s}\"}}", .{
                    "fc_",
                    item.id,
                    f.call_id,
                    f.name,
                    std.json.fmt(f.arguments.items, .{}),
                    statusToOpenAIString(f.status),
                });
            },
            .function_call_output => |*f| {
                try writer.print("{{\"type\":\"function_call_output\",\"id\":\"{s}{d}\",\"call_id\":\"{s}\",\"output\":", .{
                    "fco_",
                    item.id,
                    f.call_id,
                });
                // Output can be string or array (union type)
                switch (f.output) {
                    .text => |t| {
                        try writer.print("{f}", .{std.json.fmt(t.items, .{})});
                    },
                    .parts => |p| {
                        try self.writeContentPartsJson(writer, p.items, direction);
                    },
                }
                try writer.print(",\"status\":\"{s}\"}}", .{statusToOpenAIString(f.status)});
            },
            .reasoning => |*r| {
                // Reasoning serialization depends on direction:
                // - Request (ReasoningItemParam): content MUST be OMITTED, summary requires summary_text only
                // - Response (ReasoningBody): content CAN be included, summary accepts wider union
                try writer.print("{{\"type\":\"reasoning\",\"id\":\"{s}{d}\",\"summary\":", .{ "rs_", item.id });

                if (direction == .request) {
                    // Request direction: filter summary to only summary_text parts
                    // Schema: ReasoningItemParam.summary accepts ReasoningSummaryContentParam[] only
                    try writer.writeByte('[');
                    var first_summary = true;
                    for (r.summary.items) |*part| {
                        switch (part.variant) {
                            .summary_text => {
                                if (!first_summary) try writer.writeByte(',');
                                first_summary = false;
                                try writer.print("{{\"type\":\"summary_text\",\"text\":{f}}}", .{
                                    std.json.fmt(part.getData(), .{}),
                                });
                            },
                            else => {},
                        }
                    }
                    try writer.writeByte(']');
                } else {
                    // Response direction: preserve ALL content types in summary
                    // Schema: ReasoningBody.summary accepts wider union (output_text, text, etc.)
                    try self.writeContentPartsJson(writer, r.summary.items, direction);
                }

                // Content: only emit for response direction
                if (direction == .response and r.content.items.len > 0) {
                    try writer.writeAll(",\"content\":");
                    try self.writeContentPartsJson(writer, r.content.items, direction);
                }

                // Emit encrypted_content if present
                if (r.encrypted_content) |e| {
                    try writer.print(",\"encrypted_content\":{f}", .{std.json.fmt(e, .{})});
                }
                if (item.finish_reason) |fr| {
                    try writer.print(",\"finish_reason\":\"{s}\"", .{fr});
                }
                try writer.writeByte('}');
            },
            .item_reference => |*ref| {
                try writer.print("{{\"type\":\"item_reference\",\"id\":\"{s}\"}}", .{ref.id});
            },
            .unknown => |*u| {
                try writer.print("{{\"type\":\"{s}\",\"payload\":{s}}}", .{ u.raw_type, u.payload });
            },
        }
    }

    /// Write content parts array as JSON.
    /// Direction controls which schema variant is used (affects logprobs/annotations).
    fn writeContentPartsJson(self: *const Conversation, writer: anytype, parts: []const ContentPart, direction: SerializationDirection) !void {
        try writer.writeByte('[');

        var first = true;
        for (parts) |*part| {
            // Skip unknown content types in request direction (OpenAI will reject them)
            if (direction == .request and std.meta.activeTag(part.variant) == .unknown) {
                continue;
            }
            if (!first) try writer.writeByte(',');
            first = false;

            // Use variant to get the correct type and fields
            switch (part.variant) {
                .input_text => |v| {
                    try writer.print("{{\"type\":\"input_text\",\"text\":{f}}}", .{
                        std.json.fmt(v.text.items, .{}),
                    });
                },
                .text => |v| {
                    try writer.print("{{\"type\":\"text\",\"text\":{f}}}", .{
                        std.json.fmt(v.text.items, .{}),
                    });
                },
                .reasoning_text => |v| {
                    try writer.print("{{\"type\":\"reasoning_text\",\"text\":{f}}}", .{
                        std.json.fmt(v.text.items, .{}),
                    });
                },
                .output_text => |v| {
                    try writer.print("{{\"type\":\"output_text\",\"text\":{f}", .{
                        std.json.fmt(v.text.items, .{}),
                    });
                    // SCHEMA CONSTRAINT (Open Responses v2.3.0):
                    // - OutputTextContent (response): has logprobs field
                    // - OutputTextContentParam (request): does NOT have logprobs field
                    // Only emit logprobs for response direction to avoid 400 Bad Request.
                    if (direction == .response) {
                        if (v.logprobs_json) |l| {
                            try writer.print(",\"logprobs\":{s}", .{l});
                        }
                    }
                    // Annotations handling differs by direction:
                    // - Request: OutputTextContentParam.annotations is UrlCitationParam[] only
                    // - Response: OutputTextContent.annotations is Annotation[] (broader union)
                    // For request direction, we filter to url_citation only to avoid 400 errors.
                    if (v.annotations_json) |a| {
                        if (direction == .request) {
                            // Filter annotations to only include url_citation entries
                            try self.writeFilteredUrlCitationAnnotations(writer, a);
                        } else {
                            // Response: emit all annotations as-is
                            try writer.print(",\"annotations\":{s}", .{a});
                        }
                    }
                    try writer.writeByte('}');
                },
                .summary_text => |v| {
                    try writer.print("{{\"type\":\"summary_text\",\"text\":{f}}}", .{
                        std.json.fmt(v.text.items, .{}),
                    });
                },
                .input_image => |v| {
                    try writer.print("{{\"type\":\"input_image\",\"image_url\":{f},\"detail\":\"{s}\"}}", .{
                        std.json.fmt(v.image_url.items, .{}),
                        v.detail.toString(),
                    });
                },
                .input_file => |v| {
                    try writer.writeAll("{\"type\":\"input_file\"");
                    if (v.filename) |f| {
                        try writer.print(",\"filename\":{f}", .{std.json.fmt(f, .{})});
                    }
                    if (v.file_data) |d| {
                        try writer.print(",\"file_data\":{f}", .{std.json.fmt(d.items, .{})});
                    }
                    if (v.file_url) |u| {
                        try writer.print(",\"file_url\":{f}", .{std.json.fmt(u.items, .{})});
                    }
                    try writer.writeByte('}');
                },
                .refusal => |v| {
                    try writer.print("{{\"type\":\"refusal\",\"refusal\":{f}}}", .{
                        std.json.fmt(v.refusal.items, .{}),
                    });
                },
                .input_audio => |v| {
                    try writer.print("{{\"type\":\"input_audio\",\"data\":{f}}}", .{
                        std.json.fmt(v.audio_data.items, .{}),
                    });
                },
                .input_video => |v| {
                    try writer.print("{{\"type\":\"input_video\",\"video_url\":{f}}}", .{
                        std.json.fmt(v.video_url.items, .{}),
                    });
                },
                .unknown => |v| {
                    // Unknown content type: preserve raw_type and raw_data for round-tripping.
                    // Note: Request direction unknown parts are skipped before reaching here.
                    try writer.print("{{\"type\":\"{s}\",\"data\":{f}}}", .{
                        v.raw_type,
                        std.json.fmt(v.raw_data.items, .{}),
                    });
                },
            }
        }

        try writer.writeByte(']');
    }

    /// Filter annotations JSON to only include url_citation entries (for request direction).
    /// Schema: OutputTextContentParam.annotations must be UrlCitationParam[] only.
    fn writeFilteredUrlCitationAnnotations(self: *const Conversation, writer: anytype, annotations_json: []const u8) !void {
        // Parse the annotations array and filter to url_citation only
        const parsed = json_mod.parseValue(self.allocator, annotations_json, .{ .max_size_bytes = 1 * 1024 * 1024 }) catch {
            // If parsing fails, emit empty annotations array to avoid 400
            try writer.writeAll(",\"annotations\":[]");
            return;
        };
        defer parsed.deinit();

        var filtered_count: usize = 0;
        try writer.writeAll(",\"annotations\":[");

        if (parsed.value == .array) {
            for (parsed.value.array.items) |ann| {
                if (ann == .object) {
                    // Check if this is a url_citation entry
                    if (ann.object.get("type")) |type_val| {
                        if (type_val == .string and std.mem.eql(u8, type_val.string, "url_citation")) {
                            if (filtered_count > 0) try writer.writeByte(',');
                            filtered_count += 1;

                            // Re-serialize this annotation using json.fmt
                            try writer.print("{f}", .{std.json.fmt(ann, .{})});
                        }
                    }
                }
            }
        }

        try writer.writeByte(']');
    }

    /// Internal helper to create an empty message with a specific role (for multimodal loading).
    /// Used by the protocol layer for unfolding.
    pub fn appendEmptyMessage(self: *Conversation, role: MessageRole) !*Item {
        const item_id = self._next_item_id;
        self._next_item_id += 1;

        try self.items_list.append(self.allocator, .{
            .id = item_id,
            .created_at_ms = 0, // Will be set on finalize
            .ttl_ts = self.ttl_ts,
            .data = .{
                .message = .{
                    .role = role,
                    .content = std.ArrayListUnmanaged(ContentPart){},
                },
            },
        });
        self.updatePointers();
        return &self.items_list.items[self.items_list.items.len - 1];
    }

    /// Check if a storage error occurred.
    pub fn hasStorageError(self: *const Conversation) bool {
        return self.last_storage_error != null;
    }

    /// Get the last storage error.
    pub fn getStorageError(self: *const Conversation) ?anyerror {
        return self.last_storage_error;
    }

    /// Clear the stored storage error.
    pub fn clearStorageError(self: *Conversation) void {
        self.last_storage_error = null;
    }

    /// Set retention expiry for all items in this conversation.
    /// 0 means no expiry.
    pub fn setTtlTs(self: *Conversation, ttl_ts: i64) void {
        self.ttl_ts = ttl_ts;
    }
};

// =============================================================================
// Tests
// =============================================================================

test "Conversation.init" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    try std.testing.expectEqual(@as(usize, 0), conv.len());
}

test "Conversation.loadFromStorageBackend loads records" {
    const allocator = std.testing.allocator;

    const source = try Conversation.init(allocator);
    defer source.deinit();

    const item = try source.appendUserMessage("Hello!");
    const record = try backend_mod.ItemRecord.fromItem(allocator, item);

    var mock = struct {
        allocator: std.mem.Allocator,
        record: ?backend_mod.ItemRecord,

        const Self = @This();

        fn onEvent(ctx: *anyopaque, event: *const backend_mod.StorageEvent) anyerror!void {
            _ = ctx;
            _ = event;
        }

        fn loadAll(ctx: *anyopaque, alloc: std.mem.Allocator) anyerror![]backend_mod.ItemRecord {
            const self: *Self = @ptrCast(@alignCast(ctx));
            if (self.record == null) return alloc.alloc(backend_mod.ItemRecord, 0);

            var records = try alloc.alloc(backend_mod.ItemRecord, 1);
            records[0] = self.record.?;
            self.record = null;
            return records;
        }

        fn deinit(ctx: *anyopaque) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            if (self.record) |*owned| {
                owned.deinit(self.allocator);
                self.record = null;
            }
        }

        const vtable = backend_mod.StorageBackend.VTable{
            .onEvent = onEvent,
            .loadAll = loadAll,
            .deinit = deinit,
        };
    }{
        .allocator = allocator,
        .record = record,
    };

    const backend = backend_mod.StorageBackend.init(&mock, &@TypeOf(mock).vtable);
    const conv = try Conversation.initWithStorage(allocator, null, backend);
    defer conv.deinit();

    try conv.loadFromStorageBackend();

    try std.testing.expectEqual(@as(usize, 1), conv.len());
    const loaded = conv.getItem(0).?;
    const msg = loaded.asMessage().?;
    try std.testing.expectEqualStrings("Hello!", msg.getFirstText());
}

test "Conversation.appendUserMessage" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    const item = try conv.appendUserMessage("Hello!");
    try std.testing.expectEqual(ItemType.message, item.getType());

    const msg = item.asMessage().?;
    try std.testing.expectEqual(MessageRole.user, msg.role);
    try std.testing.expectEqual(ItemStatus.completed, msg.status);
    try std.testing.expectEqualStrings("Hello!", msg.getFirstText());
}

test "Conversation.appendSystemMessage" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    const item = try conv.appendSystemMessage("You are helpful.");
    const msg = item.asMessage().?;
    try std.testing.expectEqual(MessageRole.system, msg.role);
    try std.testing.expectEqualStrings("You are helpful.", msg.getFirstText());
}

test "Conversation.appendDeveloperMessage" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    const item = try conv.appendDeveloperMessage("Developer instruction");
    const msg = item.asMessage().?;
    try std.testing.expectEqual(MessageRole.developer, msg.role);
}

test "Conversation.appendAssistantMessage streaming" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    const item = try conv.appendAssistantMessage();
    const msg = item.asMessage().?;
    try std.testing.expectEqual(MessageRole.assistant, msg.role);
    try std.testing.expectEqual(ItemStatus.in_progress, msg.status);

    // Stream content
    try conv.appendTextContent(item, "Hello");
    try conv.appendTextContent(item, " World");

    try std.testing.expectEqualStrings("Hello World", msg.getFirstText());

    // Finalize
    conv.finalizeItem(item);
    try std.testing.expectEqual(ItemStatus.completed, msg.status);
}

test "Conversation.appendFunctionCall" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    const item = try conv.appendFunctionCall("call_123", "get_weather");
    try conv.setFunctionCallArguments(item, "{\"city\":\"NYC\"}");

    const fc = item.asFunctionCall().?;
    try std.testing.expectEqualStrings("call_123", fc.call_id);
    try std.testing.expectEqualStrings("get_weather", fc.name);
    try std.testing.expectEqualStrings("{\"city\":\"NYC\"}", fc.getArguments());
}

test "Conversation.appendFunctionCallOutput" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    const item = try conv.appendFunctionCallOutput("call_123", "Sunny, 72F");

    const fco = item.asFunctionCallOutput().?;
    try std.testing.expectEqualStrings("call_123", fco.call_id);
    try std.testing.expectEqualStrings("Sunny, 72F", fco.getOutputText());
}

test "Conversation.appendReasoning" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    const item = try conv.appendReasoning();
    try conv.addReasoningSummary(item, "I analyzed the problem...");

    const rd = item.asReasoning().?;
    try std.testing.expectEqualStrings("I analyzed the problem...", rd.getSummaryText());
}

test "Conversation.addReasoningContent" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    const item = try conv.appendReasoning();
    try conv.addReasoningContent(item, "step 1: think about it");

    const rd = item.asReasoning().?;
    try std.testing.expectEqual(@as(usize, 1), rd.content.items.len);
    try std.testing.expectEqualStrings("step 1: think about it", rd.content.items[0].getData());
}

test "Conversation.addReasoningContent rejects non-reasoning" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    const item = try conv.appendAssistantMessage();
    try std.testing.expectError(error.NotReasoning, conv.addReasoningContent(item, "text"));
}

test "Conversation.appendItemReference" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    const item = try conv.appendItemReference("msg_123");

    const ref = item.asItemReference().?;
    try std.testing.expectEqualStrings("msg_123", ref.id);
}

test "Conversation.getItem" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    _ = try conv.appendUserMessage("First");
    _ = try conv.appendUserMessage("Second");

    const item0 = conv.getItem(0).?;
    try std.testing.expectEqualStrings("First", item0.asMessage().?.getFirstText());

    const item1 = conv.getItem(1).?;
    try std.testing.expectEqualStrings("Second", item1.asMessage().?.getFirstText());

    try std.testing.expect(conv.getItem(2) == null);
}

test "Conversation.findById" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    // Store IDs immediately to avoid dangling pointer issues after reallocation
    const id1 = (try conv.appendUserMessage("First")).id;
    const id2 = (try conv.appendUserMessage("Second")).id;

    const found1 = conv.findById(id1).?;
    try std.testing.expectEqual(id1, found1.id);

    const found2 = conv.findById(id2).?;
    try std.testing.expectEqual(id2, found2.id);

    try std.testing.expect(conv.findById(999) == null);
}

test "Conversation.deleteItem" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    _ = try conv.appendUserMessage("First");
    _ = try conv.appendUserMessage("Second");
    _ = try conv.appendUserMessage("Third");

    try std.testing.expectEqual(@as(usize, 3), conv.len());

    try std.testing.expect(conv.deleteItem(1));
    try std.testing.expectEqual(@as(usize, 2), conv.len());

    // Verify remaining items
    try std.testing.expectEqualStrings("First", conv.getItem(0).?.asMessage().?.getFirstText());
    try std.testing.expectEqualStrings("Third", conv.getItem(1).?.asMessage().?.getFirstText());

    try std.testing.expect(!conv.deleteItem(10));
}

test "Conversation.clear" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    _ = try conv.appendUserMessage("First");
    _ = try conv.appendUserMessage("Second");

    try std.testing.expectEqual(@as(usize, 2), conv.len());

    conv.clear();
    try std.testing.expectEqual(@as(usize, 0), conv.len());
}

test "Conversation.setItemMetadata" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    const item = try conv.appendUserMessage("Hello");
    try conv.setItemMetadata(item, "{\"key\":\"value\"}");

    try std.testing.expectEqualStrings("{\"key\":\"value\"}", item.metadata.?);

    // Replace metadata
    try conv.setItemMetadata(item, "{\"new\":\"data\"}");
    try std.testing.expectEqualStrings("{\"new\":\"data\"}", item.metadata.?);
}

test "Conversation.toResponsesJson basic" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    _ = try conv.appendUserMessage("Hello");
    const json = try conv.toResponsesJson();
    defer allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"type\":\"message\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"role\":\"user\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"text\":\"Hello\"") != null);
}

test "Conversation.toResponsesJson function call" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    const fc = try conv.appendFunctionCall("call_1", "get_weather");
    try conv.setFunctionCallArguments(fc, "{\"city\":\"NYC\"}");
    conv.finalizeItem(fc);

    const json = try conv.toResponsesJson();
    defer allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"type\":\"function_call\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"call_id\":\"call_1\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"name\":\"get_weather\"") != null);
}

test "Conversation unique IDs" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    // Store IDs immediately to avoid dangling pointer issues after reallocation
    const id1 = (try conv.appendUserMessage("First")).id;
    const id2 = (try conv.appendUserMessage("Second")).id;
    const id3 = (try conv.appendUserMessage("Third")).id;

    try std.testing.expectEqual(@as(u64, 0), id1);
    try std.testing.expectEqual(@as(u64, 1), id2);
    try std.testing.expectEqual(@as(u64, 2), id3);
}

test "Conversation.addContentPart" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    // This is safe because we only have one append before using the pointer
    const item = try conv.appendAssistantMessage();
    const part = try conv.addContentPart(item, .output_text);
    try part.appendData(allocator, "Hello from new part");

    // Re-fetch item from list to ensure valid pointer
    const msg = conv.getItem(0).?.asMessage().?;
    try std.testing.expectEqual(@as(usize, 1), msg.partCount());
    try std.testing.expectEqualStrings("Hello from new part", msg.getPart(0).?.getData());
}

test "Conversation count_ptr updates" {
    const allocator = std.testing.allocator;
    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    try std.testing.expectEqual(@as(usize, 0), conv.count_ptr.*);

    _ = try conv.appendUserMessage("First");
    try std.testing.expectEqual(@as(usize, 1), conv.count_ptr.*);

    _ = try conv.appendUserMessage("Second");
    try std.testing.expectEqual(@as(usize, 2), conv.count_ptr.*);

    conv.clear();
    try std.testing.expectEqual(@as(usize, 0), conv.count_ptr.*);
}
