//! StorageBackend - Event-based interface for persistence backends.
//!
//! StorageBackend defines the contract that all storage implementations must follow.
//! It uses a vtable pattern for runtime polymorphism, allowing different backends
//! to be swapped without recompilation.
//!
//! # Events
//!
//! Backends receive StorageEvent unions for all mutations:
//! - PutMessage: A message was finalized (insert/update)
//! - DeleteMessage: A message was removed
//! - Clear: All messages were cleared (optionally keeping system)
//!
//! # TaluDB Column Mapping (env/features/storage/)
//!
//! When implementing a TaluDB backend, map fields as:
//!
//! ## SCHEMA_CHAT_EVENTS (schema_id = 1)
//! | Field | Column ID | Column Name | Type | Description |
//! |-------|-----------|-------------|------|-------------|
//! | msg_id | 1 | DOC_ID | u64 | Stable message identity |
//! | created_at_ms | 2 | TS | i64 | Unix timestamp (milliseconds) |
//! | (from session) | 3 | SESSION_HASH | u64 | SipHash of session_id |
//! | role | 4 | ROLE | u8 | MessageRole enum value |
//! | content+metadata | 20 | PAYLOAD | bytes | MsgPack-encoded MessageRecord |
//!
//! ## SCHEMA_CHAT_DELETES (schema_id = 2)
//! | Field | Column ID | Column Name | Type | Description |
//! |-------|-----------|-------------|------|-------------|
//! | msg_id | 1 | DELETE_TARGET | u64 | The msg_id being deleted |
//! | deleted_at_ms | 2 | TS | i64 | When the delete occurred |
//! | (from session) | 3 | SESSION_HASH | u64 | Session this delete applies to |
//!
//! ## Clear Event Mapping
//! Clear events generate delete markers targeting the SESSION_HASH:
//! - DELETE_KIND = SESSION_HASH (value 2)
//! - DELETE_TARGET = hash(session_id)
//! - TS = cleared_at_ms
//!
//! # Implementing a New Backend
//!
//! 1. Create a struct with your backend state (db connection, file handle, etc.)
//! 2. Implement the three VTable functions (onEvent, loadAll, deinit)
//! 3. Provide a `backend()` method that returns a StorageBackend
//!
//! Example (pseudocode for TaluDB backend):
//!
//!   pub const DbBackend = struct {
//!       writer: *DbWriter,
//!       session_hash: u64,
//!       allocator: Allocator,
//!
//!       pub fn init(allocator: Allocator, db_root: []const u8, session_id: []const u8) !DbBackend {
//!           const writer = try DbWriter.open(db_root, "chat");
//!           const session_hash = writer.hashSessionId(session_id);
//!           return .{ .writer = writer, .session_hash = session_hash, .allocator = allocator };
//!       }
//!
//!       fn onEvent(ctx: *anyopaque, event: *const StorageEvent) !void {
//!           const self: *DbBackend = @ptrCast(@alignCast(ctx));
//!           switch (event.*) {
//!               .PutMessage => |record| {
//!                   // Append row to chat/current.talu with columns:
//!                   // DOC_ID=msg_id, TS=created_at_ms, SESSION_HASH=self.session_hash,
//!                   // PAYLOAD=msgpack(record)
//!               },
//!               .DeleteMessage => |del| {
//!                   // Append delete marker: DELETE_TARGET=msg_id, TS=deleted_at_ms
//!               },
//!               .Clear => |clr| {
//!                   // Append session delete: DELETE_KIND=SESSION_HASH, DELETE_TARGET=session_hash
//!               },
//!           }
//!       }
//!       // ... other vtable implementations
//!   };
//!
//! # Thread Safety
//!
//! Backends should be thread-safe if used in multi-threaded contexts.
//! The MemoryBackend is trivially thread-safe (no state).
//! TaluDB backends use single-writer semantics via `<ns>/talu.lock`.

const std = @import("std");
const items_mod = @import("items.zig");

// Items module types
pub const ItemType = items_mod.ItemType;
pub const ItemStatus = items_mod.ItemStatus;
pub const MessageRole = items_mod.MessageRole;
pub const ContentType = items_mod.ContentType;
pub const ImageDetail = items_mod.ImageDetail;

// =============================================================================
// ItemRecord - Self-contained item for storage (Open Responses architecture)
// =============================================================================

/// ItemContentPartRecord - Portable snapshot of a content part for Items.
///
/// This is the serializable version of ContentPart from items.zig.
/// Uses a tagged union to mirror ContentVariant exactly, ensuring lossless
/// round-trip through storage.
///
/// # Schema Fidelity
///
/// Each content type stores EXACTLY the fields defined in the OpenAPI schema:
/// - `input_text` -> { text }
/// - `input_image` -> { image_url, detail }
/// - `input_file` -> { filename?, file_data?, file_url? }  (mutually exclusive data sources)
/// - `output_text` -> { text, logprobs_json?, annotations_json? }
/// - etc.
///
/// This prevents the "bag of optional fields" anti-pattern that loses type safety
/// and causes data loss (e.g., can't distinguish file_data from file_url).
pub const ItemContentPartRecord = union(ContentType) {
    /// Schema: InputTextContentParam { type: "input_text", text: string }
    input_text: struct {
        text: []const u8,
    },

    /// Schema: InputImageContentParamAutoParam { type: "input_image", image_url: string, detail?: enum }
    input_image: struct {
        image_url: []const u8,
        detail: items_mod.ImageDetail = .auto,
    },

    /// Extension for audio input
    input_audio: struct {
        audio_data: []const u8,
    },

    /// Schema: InputVideoContent { type: "input_video", video_url: string }
    input_video: struct {
        video_url: []const u8,
    },

    /// Schema: InputFileContentParam { type: "input_file", filename?: string, file_data?: string, file_url?: string }
    input_file: struct {
        filename: ?[]const u8 = null,
        file_data: ?[]const u8 = null,
        file_url: ?[]const u8 = null,
    },

    /// Schema: OutputTextContentParam { type: "output_text", text: string, logprobs?: array, annotations?: array }
    output_text: struct {
        text: []const u8,
        logprobs_json: ?[]const u8 = null,
        annotations_json: ?[]const u8 = null,
        code_blocks_json: ?[]const u8 = null,
    },

    /// Schema: RefusalContentParam { type: "refusal", refusal: string }
    refusal: struct {
        refusal: []const u8,
    },

    /// Schema: TextContent { type: "text", text: string }
    text: struct {
        text: []const u8,
    },

    /// Schema: ReasoningTextContent
    reasoning_text: struct {
        text: []const u8,
    },

    /// Schema: ReasoningSummaryContentParam { type: "summary_text", text: string }
    summary_text: struct {
        text: []const u8,
    },

    /// Unknown/unrecognized content type (for forward compatibility).
    unknown: struct {
        raw_type: []const u8,
        raw_data: []const u8,
    },

    /// Get the content type discriminator.
    pub fn getType(self: *const ItemContentPartRecord) ContentType {
        return std.meta.activeTag(self.*);
    }

    /// Get text content (for text-like types).
    pub fn getText(self: *const ItemContentPartRecord) []const u8 {
        return switch (self.*) {
            .input_text => |v| v.text,
            .output_text => |v| v.text,
            .text => |v| v.text,
            .reasoning_text => |v| v.text,
            .summary_text => |v| v.text,
            .refusal => |v| v.refusal,
            .input_image => |v| v.image_url,
            .input_audio => |v| v.audio_data,
            .input_video => |v| v.video_url,
            .input_file => |v| v.file_data orelse v.file_url orelse "",
            .unknown => |v| v.raw_data,
        };
    }
};

/// FunctionCallRecord - Portable function call for storage.
pub const FunctionCallRecord = struct {
    call_id: []const u8,
    name: []const u8,
    arguments: []const u8,
    status: ItemStatus,
};

/// FunctionCallOutputRecord - Portable function call output for storage.
pub const FunctionCallOutputRecord = struct {
    call_id: []const u8,
    output: []ItemContentPartRecord,
    status: ItemStatus,
};

/// ReasoningRecord - Portable reasoning item for storage.
pub const ReasoningRecord = struct {
    content: []ItemContentPartRecord,
    summary: []ItemContentPartRecord,
    encrypted_content: ?[]const u8,
    status: ItemStatus = .completed,
};

/// ItemReferenceRecord - Portable item reference for storage.
pub const ItemReferenceRecord = struct {
    id: []const u8,
    status: ItemStatus = .completed,
};

/// MessageItemRecord - Portable message item for storage.
pub const MessageItemRecord = struct {
    role: MessageRole,
    status: ItemStatus,
    content: []ItemContentPartRecord,
};

/// ItemVariantRecord - Tagged union for item variant storage.
pub const ItemVariantRecord = union(ItemType) {
    message: MessageItemRecord,
    function_call: FunctionCallRecord,
    function_call_output: FunctionCallOutputRecord,
    reasoning: ReasoningRecord,
    item_reference: ItemReferenceRecord,
    unknown: struct {
        raw_type: []const u8,
        payload: []const u8,
    },
};

/// Item type from items.zig (needed for fromItem).
const Item = items_mod.Item;
const ItemVariant = items_mod.ItemVariant;
const ContentPart = items_mod.ContentPart;
const MessageData = items_mod.MessageData;
const FunctionCallData = items_mod.FunctionCallData;
const FunctionCallOutputData = items_mod.FunctionCallOutputData;
const ReasoningData = items_mod.ReasoningData;
const ItemReferenceData = items_mod.ItemReferenceData;

/// ItemRecord - Frozen, portable snapshot of an Item for persistence.
///
/// This is the format used for the new Open Responses storage schema (ID=3).
/// It stores the polymorphic Item types (message, function_call, reasoning, etc.)
/// in a serializable format.
///
/// # StoreFS Column Mapping (SCHEMA_CHAT_ITEMS, schema_id = 3)
///
/// | Field | Column | Type | Notes |
/// |-------|--------|------|-------|
/// | item_id | DOC_ID (1) | u64 | Primary key, stable identity |
/// | created_at_ms | TS (2) | i64 | Unix millis |
/// | (session) | SESSION_HASH (3) | u64 | From session_id |
/// | item_type | TYPE (4) | u8 | ItemType enum value |
/// | (role) | ROLE (5) | u8 | MessageRole or 0xFF sentinel |
/// | status | STATUS (6) | u8 | ItemStatus enum value |
/// | flags | FLAGS (7) | u16 | bit0=hidden, bit1=pinned, bit2=deleted, bit3=json_valid, bit4=schema_valid, bit5=repaired |
/// | ttl_ts | TTL_TS (9) | i64 | Expiry timestamp (Unix ms) |
/// | variant+metadata | PAYLOAD (20) | bytes | MsgPack encoded |
///
/// # Ownership Rules
///
/// Same as MessageRecord - records in events are BORROWED, records from
/// loadAll() are OWNED and must be freed with deinit().
pub const ItemRecord = struct {
    /// Stable item identity.
    /// Session-scoped monotonic ID by design (not globally unique).
    /// This keeps storage clustered (items for a session are contiguous on disk)
    /// and allows O(1) indexing into Conversation.items without a UUID lookup.
    /// Cross-session lineage is tracked separately via origin_session_id/origin_item_id.
    item_id: u64 = 0,

    /// Creation timestamp (unix milliseconds).
    created_at_ms: i64 = 0,

    /// Expiration timestamp for retention (Unix ms). 0 = no expiry.
    ttl_ts: i64 = 0,

    /// Item status (in_progress, waiting, completed, incomplete, failed).
    status: ItemStatus = .completed,

    /// UI visibility flag (true = hidden).
    hidden: bool = false,

    /// Retention flag (true = pinned).
    pinned: bool = false,

    /// Structured output validation: JSON parsed successfully.
    json_valid: bool = false,

    /// Structured output validation: schema validation passed.
    schema_valid: bool = false,

    /// Structured output validation: output was repaired.
    repaired: bool = false,

    /// Optional parent item reference (edit/regenerate lineage).
    parent_item_id: ?u64 = null,

    /// Optional origin session identifier for forked items.
    origin_session_id: ?[]const u8 = null,

    /// Optional origin item ID for forked items.
    origin_item_id: ?u64 = null,

    /// Finish reason for generation (optional).
    finish_reason: ?[]const u8 = null,

    /// Prefill time in nanoseconds.
    prefill_ns: u64 = 0,

    /// Generation time in nanoseconds.
    generation_ns: u64 = 0,

    /// Input token count (prompt tokens).
    input_tokens: u32 = 0,

    /// Output token count (completion tokens).
    output_tokens: u32 = 0,

    /// Item type discriminator.
    item_type: ItemType,

    /// Polymorphic item data.
    variant: ItemVariantRecord,

    /// Developer metadata (JSON key-value pairs).
    metadata: ?[]const u8 = null,

    /// Generation parameters used to produce this item (assistant messages only).
    /// JSON object containing model, temperature, top_p, top_k, etc.
    generation_json: ?[]const u8 = null,

    /// Free allocated memory.
    pub fn deinit(self: *ItemRecord, allocator: std.mem.Allocator) void {
        switch (self.variant) {
            .message => |m| {
                for (m.content) |part| {
                    freeItemContentPartRecord(allocator, part);
                }
                allocator.free(m.content);
            },
            .function_call => |f| {
                allocator.free(f.call_id);
                allocator.free(f.name);
                allocator.free(f.arguments);
            },
            .function_call_output => |f| {
                allocator.free(f.call_id);
                for (f.output) |part| {
                    freeItemContentPartRecord(allocator, part);
                }
                allocator.free(f.output);
            },
            .reasoning => |r| {
                for (r.content) |part| {
                    freeItemContentPartRecord(allocator, part);
                }
                allocator.free(r.content);
                for (r.summary) |part| {
                    freeItemContentPartRecord(allocator, part);
                }
                allocator.free(r.summary);
                if (r.encrypted_content) |e| allocator.free(e);
            },
            .item_reference => |ref| {
                allocator.free(ref.id);
            },
            .unknown => |u| {
                allocator.free(u.raw_type);
                allocator.free(u.payload);
            },
        }
        if (self.metadata) |m| allocator.free(m);
        if (self.generation_json) |g| allocator.free(g);
        if (self.finish_reason) |r| allocator.free(r);
        if (self.origin_session_id) |sid| allocator.free(sid);
    }

    /// Create an ItemRecord from a live Item (deep copy for persistence).
    ///
    /// This creates a portable snapshot of the Item with all data owned.
    /// Used by Conversation.finalizeItem to emit PutItem events to storage.
    ///
    /// The returned record owns all allocated memory and must be freed with deinit().
    pub fn fromItem(allocator: std.mem.Allocator, item: *const Item) !ItemRecord {
        return ItemRecord{
            .item_id = item.id,
            .created_at_ms = item.created_at_ms,
            .ttl_ts = item.ttl_ts,
            .status = item.data.getStatus(),
            .hidden = item.hidden,
            .pinned = item.pinned,
            .json_valid = item.json_valid,
            .schema_valid = item.schema_valid,
            .repaired = item.repaired,
            .parent_item_id = item.parent_item_id,
            .origin_session_id = if (item.origin_session_id) |sid|
                try allocator.dupe(u8, sid)
            else
                null,
            .origin_item_id = item.origin_item_id,
            .finish_reason = if (item.finish_reason) |r| try allocator.dupe(u8, r) else null,
            .prefill_ns = item.prefill_ns,
            .generation_ns = item.generation_ns,
            .input_tokens = item.input_tokens,
            .output_tokens = item.output_tokens,
            .item_type = item.data.getType(),
            .variant = try copyItemVariant(allocator, &item.data),
            .metadata = if (item.metadata) |m| try allocator.dupe(u8, m) else null,
            .generation_json = if (item.generation_json) |g| try allocator.dupe(u8, g) else null,
        };
    }
};

fn inflateContentPart(allocator: std.mem.Allocator, record: ItemContentPartRecord) !ContentPart {
    const part: ContentPart = switch (record) {
        .input_text => |p| blk: {
            var value = ContentPart.initInputText();
            try value.appendData(allocator, p.text);
            break :blk value;
        },
        .input_image => |p| blk: {
            var value = ContentPart.initInputImage(p.detail);
            try value.appendData(allocator, p.image_url);
            break :blk value;
        },
        .input_audio => |p| blk: {
            var value = ContentPart.initInputAudio();
            try value.appendData(allocator, p.audio_data);
            break :blk value;
        },
        .input_video => |p| blk: {
            var value = ContentPart.initInputVideo();
            try value.appendData(allocator, p.video_url);
            break :blk value;
        },
        .input_file => |p| blk: {
            var value = ContentPart.initInputFile();
            if (p.filename) |f| {
                value.variant.input_file.filename = try allocator.dupe(u8, f);
            }
            if (p.file_data) |d| {
                var list: std.ArrayListUnmanaged(u8) = .{};
                try list.appendSlice(allocator, d);
                value.variant.input_file.file_data = list;
            }
            if (p.file_url) |u| {
                var list: std.ArrayListUnmanaged(u8) = .{};
                try list.appendSlice(allocator, u);
                value.variant.input_file.file_url = list;
            }
            break :blk value;
        },
        .output_text => |p| blk: {
            var value = ContentPart.initOutputText();
            try value.appendData(allocator, p.text);
            if (p.logprobs_json) |l| {
                value.variant.output_text.logprobs_json = try allocator.dupe(u8, l);
            }
            if (p.annotations_json) |a| {
                value.variant.output_text.annotations_json = try allocator.dupe(u8, a);
            }
            if (p.code_blocks_json) |c| {
                value.variant.output_text.code_blocks_json = try allocator.dupe(u8, c);
            }
            break :blk value;
        },
        .refusal => |p| blk: {
            var value = ContentPart.initRefusal();
            try value.appendData(allocator, p.refusal);
            break :blk value;
        },
        .text => |p| blk: {
            var value = ContentPart.initText();
            try value.appendData(allocator, p.text);
            break :blk value;
        },
        .reasoning_text => |p| blk: {
            var value = ContentPart.initReasoningText();
            try value.appendData(allocator, p.text);
            break :blk value;
        },
        .summary_text => |p| blk: {
            var value = ContentPart.initSummaryText();
            try value.appendData(allocator, p.text);
            break :blk value;
        },
        .unknown => |p| blk: {
            var value = ContentPart.initUnknown();
            if (p.raw_type.len > 0) {
                value.variant.unknown.raw_type = try allocator.dupe(u8, p.raw_type);
            }
            try value.appendData(allocator, p.raw_data);
            break :blk value;
        },
    };
    return part;
}

fn inflateContentParts(
    allocator: std.mem.Allocator,
    parts: []ItemContentPartRecord,
) !std.ArrayListUnmanaged(ContentPart) {
    var result: std.ArrayListUnmanaged(ContentPart) = .{};
    errdefer {
        for (result.items) |*part| {
            part.deinit(allocator);
        }
        result.deinit(allocator);
    }

    for (parts) |part_record| {
        const part = try inflateContentPart(allocator, part_record);
        try result.append(allocator, part);
    }

    return result;
}

fn inflateItemVariant(
    allocator: std.mem.Allocator,
    record: ItemVariantRecord,
) !ItemVariant {
    return switch (record) {
        .message => |m| .{
            .message = MessageData{
                .role = m.role,
                .status = m.status,
                .content = try inflateContentParts(allocator, m.content),
                .raw_role = null,
            },
        },
        .function_call => |f| blk: {
            var args: std.ArrayListUnmanaged(u8) = .{};
            errdefer args.deinit(allocator);
            try args.appendSlice(allocator, f.arguments);
            break :blk .{
                .function_call = FunctionCallData{
                    .call_id = try allocator.dupeZ(u8, f.call_id),
                    .name = try allocator.dupeZ(u8, f.name),
                    .arguments = args,
                    .status = f.status,
                },
            };
        },
        .function_call_output => |f| blk: {
            const parts = try inflateContentParts(allocator, f.output);
            break :blk .{
                .function_call_output = FunctionCallOutputData{
                    .call_id = try allocator.dupeZ(u8, f.call_id),
                    .output = .{ .parts = parts },
                    .status = f.status,
                },
            };
        },
        .reasoning => |r| .{
            .reasoning = ReasoningData{
                .content = try inflateContentParts(allocator, r.content),
                .summary = try inflateContentParts(allocator, r.summary),
                .encrypted_content = if (r.encrypted_content) |e|
                    try allocator.dupe(u8, e)
                else
                    null,
                .status = r.status,
            },
        },
        .item_reference => |ref| .{
            .item_reference = ItemReferenceData{
                .id = try allocator.dupeZ(u8, ref.id),
                .status = ref.status,
            },
        },
        .unknown => |u| .{
            .unknown = items_mod.UnknownData{
                .raw_type = try allocator.dupe(u8, u.raw_type),
                .payload = try allocator.dupe(u8, u.payload),
            },
        },
    };
}

/// Convert a storage ItemRecord into a runtime Item.
pub fn itemFromRecord(allocator: std.mem.Allocator, record: *const ItemRecord) !Item {
    const variant = try inflateItemVariant(allocator, record.variant);
    const item = Item{
        .id = record.item_id,
        .created_at_ms = record.created_at_ms,
        .ttl_ts = record.ttl_ts,
        .input_tokens = record.input_tokens,
        .output_tokens = record.output_tokens,
        .hidden = record.hidden,
        .pinned = record.pinned,
        .json_valid = record.json_valid,
        .schema_valid = record.schema_valid,
        .repaired = record.repaired,
        .parent_item_id = record.parent_item_id,
        .origin_session_id = if (record.origin_session_id) |sid|
            try allocator.dupe(u8, sid)
        else
            null,
        .origin_item_id = record.origin_item_id,
        .finish_reason = if (record.finish_reason) |reason|
            try allocator.dupeZ(u8, reason)
        else
            null,
        .prefill_ns = record.prefill_ns,
        .generation_ns = record.generation_ns,
        .data = variant,
        .metadata = if (record.metadata) |meta|
            try allocator.dupe(u8, meta)
        else
            null,
        .generation_json = if (record.generation_json) |g|
            try allocator.dupe(u8, g)
        else
            null,
    };
    return item;
}

/// Deep copy an ItemVariant to ItemVariantRecord.
fn copyItemVariant(allocator: std.mem.Allocator, variant: *const ItemVariant) !ItemVariantRecord {
    return switch (variant.*) {
        .message => |*msg| .{ .message = .{
            .role = msg.role,
            .status = msg.status,
            .content = try copyContentParts(allocator, msg.content.items),
        } },
        .function_call => |*fc| .{ .function_call = .{
            .call_id = try allocator.dupe(u8, fc.call_id),
            .name = try allocator.dupe(u8, fc.name),
            .arguments = try allocator.dupe(u8, fc.arguments.items),
            .status = fc.status,
        } },
        .function_call_output => |*fco| .{ .function_call_output = .{
            .call_id = try allocator.dupe(u8, fco.call_id),
            .output = try copyFunctionOutputValue(allocator, &fco.output),
            .status = fco.status,
        } },
        .reasoning => |*r| .{ .reasoning = .{
            .content = try copyContentParts(allocator, r.content.items),
            .summary = try copyContentParts(allocator, r.summary.items),
            .encrypted_content = if (r.encrypted_content) |e| try allocator.dupe(u8, e) else null,
            .status = r.status,
        } },
        .item_reference => |*ref| .{ .item_reference = .{
            .id = try allocator.dupe(u8, ref.id),
            .status = ref.status,
        } },
        .unknown => |*u| .{ .unknown = .{
            .raw_type = try allocator.dupe(u8, u.raw_type),
            .payload = try allocator.dupe(u8, u.payload),
        } },
    };
}

/// Copy ContentPart array to ItemContentPartRecord array.
fn copyContentParts(allocator: std.mem.Allocator, parts: []const ContentPart) ![]ItemContentPartRecord {
    const result = try allocator.alloc(ItemContentPartRecord, parts.len);
    errdefer allocator.free(result);

    for (parts, 0..) |*part, i| {
        result[i] = try copyContentPart(allocator, part);
    }
    return result;
}

/// Copy a single ContentPart to ItemContentPartRecord.
fn copyContentPart(allocator: std.mem.Allocator, part: *const ContentPart) !ItemContentPartRecord {
    return switch (part.variant) {
        .input_text => |v| .{ .input_text = .{
            .text = try allocator.dupe(u8, v.text.items),
        } },
        .input_image => |v| .{ .input_image = .{
            .image_url = try allocator.dupe(u8, v.image_url.items),
            .detail = v.detail,
        } },
        .input_audio => |v| .{ .input_audio = .{
            .audio_data = try allocator.dupe(u8, v.audio_data.items),
        } },
        .input_video => |v| .{ .input_video = .{
            .video_url = try allocator.dupe(u8, v.video_url.items),
        } },
        .input_file => |v| .{ .input_file = .{
            .filename = if (v.filename) |f| try allocator.dupe(u8, f) else null,
            .file_data = if (v.file_data) |d| try allocator.dupe(u8, d.items) else null,
            .file_url = if (v.file_url) |u| try allocator.dupe(u8, u.items) else null,
        } },
        .output_text => |v| .{ .output_text = .{
            .text = try allocator.dupe(u8, v.text.items),
            .logprobs_json = if (v.logprobs_json) |l| try allocator.dupe(u8, l) else null,
            .annotations_json = if (v.annotations_json) |a| try allocator.dupe(u8, a) else null,
            .code_blocks_json = if (v.code_blocks_json) |c| try allocator.dupe(u8, c) else null,
        } },
        .refusal => |v| .{ .refusal = .{
            .refusal = try allocator.dupe(u8, v.refusal.items),
        } },
        .text => |v| .{ .text = .{
            .text = try allocator.dupe(u8, v.text.items),
        } },
        .reasoning_text => |v| .{ .reasoning_text = .{
            .text = try allocator.dupe(u8, v.text.items),
        } },
        .summary_text => |v| .{ .summary_text = .{
            .text = try allocator.dupe(u8, v.text.items),
        } },
        .unknown => |v| .{ .unknown = .{
            .raw_type = try allocator.dupe(u8, v.raw_type),
            .raw_data = try allocator.dupe(u8, v.raw_data.items),
        } },
    };
}

/// Copy FunctionOutputValue to ItemContentPartRecord array.
fn copyFunctionOutputValue(allocator: std.mem.Allocator, output: *const items_mod.FunctionOutputValue) ![]ItemContentPartRecord {
    return switch (output.*) {
        .text => |t| blk: {
            // Convert text output to single input_text part
            const result = try allocator.alloc(ItemContentPartRecord, 1);
            errdefer allocator.free(result);
            result[0] = .{ .input_text = .{
                .text = try allocator.dupe(u8, t.items),
            } };
            break :blk result;
        },
        .parts => |p| try copyContentParts(allocator, p.items),
    };
}

/// Free an ItemContentPartRecord's allocated memory.
fn freeItemContentPartRecord(allocator: std.mem.Allocator, part: ItemContentPartRecord) void {
    switch (part) {
        .input_text => |v| {
            if (v.text.len > 0) allocator.free(v.text);
        },
        .input_image => |v| {
            if (v.image_url.len > 0) allocator.free(v.image_url);
        },
        .input_audio => |v| {
            if (v.audio_data.len > 0) allocator.free(v.audio_data);
        },
        .input_video => |v| {
            if (v.video_url.len > 0) allocator.free(v.video_url);
        },
        .input_file => |v| {
            if (v.filename) |f| allocator.free(f);
            if (v.file_data) |d| allocator.free(d);
            if (v.file_url) |u| allocator.free(u);
        },
        .output_text => |v| {
            if (v.text.len > 0) allocator.free(v.text);
            if (v.logprobs_json) |l| allocator.free(l);
            if (v.annotations_json) |a| allocator.free(a);
            if (v.code_blocks_json) |c| allocator.free(c);
        },
        .refusal => |v| {
            if (v.refusal.len > 0) allocator.free(v.refusal);
        },
        .text => |v| {
            if (v.text.len > 0) allocator.free(v.text);
        },
        .reasoning_text => |v| {
            if (v.text.len > 0) allocator.free(v.text);
        },
        .summary_text => |v| {
            if (v.text.len > 0) allocator.free(v.text);
        },
        .unknown => |v| {
            if (v.raw_type.len > 0) allocator.free(v.raw_type);
            if (v.raw_data.len > 0) allocator.free(v.raw_data);
        },
    }
}

/// Free item records returned by loadAll().
pub fn freeItemRecords(allocator: std.mem.Allocator, records: []ItemRecord) void {
    for (records) |*record| {
        record.deinit(allocator);
    }
    allocator.free(records);
}

// =============================================================================
// StorageEvent - Events emitted to storage backends (Item-based architecture)
// =============================================================================

/// SessionRecord - Portable session metadata for storage.
///
/// Contains session-level metadata that should be stored separately from
/// individual items. This ensures Zig is the single source of truth for
/// all session data (GenerationConfig, title, system_prompt).
///
/// # Fields
///
/// - session_id: Session identifier (user-provided or generated)
/// - model: Model identifier (optional)
/// - title: Human-readable title (optional)
/// - system_prompt: System prompt text (optional)
/// - config_json: GenerationConfig serialized as JSON (optional)
/// - marker: Session marker (optional, e.g. "pinned", "archived", "deleted")
/// - parent_session_id: Parent session identifier (optional)
/// - group_id: Application-defined group identifier (optional)
/// - head_item_id: Latest item_id in session (0 when no items)
/// - ttl_ts: Expiration timestamp (Unix ms). 0 = no expiry
/// - metadata_json: Session metadata as JSON (optional)
/// - created_at_ms: Unix timestamp when session was created
/// - updated_at_ms: Unix timestamp when session was last updated
pub const SessionRecord = struct {
    session_id: []const u8,
    model: ?[]const u8 = null,
    title: ?[]const u8 = null,
    system_prompt: ?[]const u8 = null,
    config_json: ?[]const u8 = null,
    marker: ?[]const u8 = null,
    parent_session_id: ?[]const u8 = null,
    group_id: ?[]const u8 = null,
    head_item_id: u64 = 0,
    ttl_ts: i64 = 0,
    metadata_json: ?[]const u8 = null,
    /// Source document ID for lineage tracking (prompt_id from API).
    /// Links this session to the document that spawned it.
    source_doc_id: ?[]const u8 = null,
    created_at_ms: i64,
    updated_at_ms: i64,
};

/// Storage events emitted by Conversation for persistence.
///
/// This enables correct persistence for:
/// - Item creation (PutItem)
/// - Item deletion (DeleteItem)
/// - Clear operations (ClearItems)
/// - Session metadata (PutSession)
///
/// Backends receive these events and can implement proper tombstones,
/// versioning, and ordering semantics.
///
/// Memory/Lifetime Rules (IMPORTANT):
/// ----------------------------------
/// Event payloads point into Conversation's internal memory. Backends MUST:
///
/// 1. **Consume synchronously**: Process the event data before returning from
///    onEvent(). The pointers become invalid after the call returns.
///
/// 2. **Copy if needed**: If async processing is required (e.g., background
///    write to database), copy all data before returning. Serialize to JSON/MsgPack.
///
/// 3. **Never retain pointers**: Do not store *const StorageEvent or any
///    slices from the event payload. They reference stack or transient memory.
pub const StorageEvent = union(enum) {
    /// Items were finalized and should be persisted (batched).
    PutItems: []ItemRecord,

    /// An item was finalized and should be persisted.
    PutItem: ItemRecord,

    /// An item was deleted and should be removed from storage.
    DeleteItem: struct {
        item_id: u64,
        deleted_at_ms: i64,
    },

    /// All items were cleared.
    ClearItems: struct {
        cleared_at_ms: i64,
        /// If true, keep system/developer messages (roles that set context).
        keep_context: bool,
    },

    /// Session metadata was set/updated.
    PutSession: SessionRecord,

    /// Begin a fork transaction boundary.
    BeginFork: struct {
        fork_id: u64,
        session_id: []const u8,
    },

    /// End a fork transaction boundary.
    EndFork: struct {
        fork_id: u64,
        session_id: []const u8,
    },
};

// =============================================================================
// StorageBackend - Event-based storage interface for Items
// =============================================================================

/// StoredItemEnvelope - Wrapper for ItemRecord storage with session_id.
///
/// This wrapper is required for TaluDB storage because:
/// 1. `ItemRecord` does not contain `session_id` (it's a session-level property)
/// 2. Hash collision validation requires comparing `session_id` strings
/// 3. Without this, readers cannot distinguish items from different sessions
///    that happen to have the same SESSION_HASH (collision)
///
/// # Privacy Critical
///
/// The `session_id` field enables collision validation:
/// ```
/// if (!std.mem.eql(u8, envelope.session_id, requested_session_id)) {
///     // Hash collision detected — skip silently (do not leak data)
///     continue;
/// }
/// ```
///
/// # TaluDB Column Mapping
///
/// When writing to SCHEMA_CHAT_ITEMS (schema_id = 3):
/// - Column 20 (PAYLOAD) contains MsgPack-encoded `StoredItemEnvelope`
/// - The `session_id` is stored inside PAYLOAD, not as a separate column
/// - SESSION_HASH (column 3) is computed from `session_id` for fast filtering
///
/// # Usage
///
/// ```zig
/// // Write path
/// const envelope = StoredItemEnvelope{
///     .session_id = self.session_id,
///     .record = ItemRecord.fromItem(item),
/// };
/// const bytes = try msgpack.encode(allocator, envelope);
///
/// // Read path
/// const envelope = try msgpack.decode(StoredItemEnvelope, allocator, bytes);
/// if (!std.mem.eql(u8, envelope.session_id, requested_session_id)) {
///     // Collision — skip
/// }
/// ```
pub const StoredItemEnvelope = struct {
    /// Session identifier (MUST match Conversation.session_id).
    /// Used for hash collision validation on read.
    session_id: []const u8,

    /// The actual item record.
    record: ItemRecord,

    /// Free allocated memory.
    pub fn deinit(self: *StoredItemEnvelope, allocator: std.mem.Allocator) void {
        allocator.free(self.session_id);
        self.record.deinit(allocator);
    }
};

/// StorageBackend - Event-based runtime polymorphic interface for Item persistence.
///
/// This is a type-erased wrapper around concrete backend implementations.
/// Use the `init()` function to create from a concrete backend pointer.
///
/// Backends receive events for all mutations:
/// - PutItem: An item was finalized
/// - DeleteItem: An item was removed
/// - ClearItems: All items were cleared (optionally keeping context)
///
/// This enables:
/// - Proper tombstones for deleted items
/// - Item versioning for edit history
/// - Ordering independent of array indices
/// - Multi-segment restore
///
/// # Thread Safety
///
/// Depends on underlying implementation. See individual backend documentation.
/// The interface itself is thread-safe to call but backends may have their
/// own synchronization requirements.
///
/// # Implementation Guide
///
/// See `core/src/messages/FUTURE.md` for the TaluDB implementation plan.
pub const StorageBackend = struct {
    /// Pointer to the concrete backend instance.
    ptr: *anyopaque,

    /// Virtual function table.
    vtable: *const VTable,

    /// Virtual function table for storage operations.
    pub const VTable = struct {
        /// Called for every storage mutation event.
        ///
        /// Events include:
        /// - PutItem: An item was finalized
        /// - DeleteItem: An item was removed
        /// - ClearItems: All items were cleared
        ///
        /// Returns error if the event could not be persisted.
        /// Callers should handle errors appropriately (retry, log, etc.).
        ///
        /// # Memory/Lifetime
        ///
        /// Event payloads point into Conversation's internal memory. Backends MUST:
        /// 1. Consume synchronously before returning
        /// 2. Copy/serialize if async processing is needed
        /// 3. Never retain pointers to event data
        ///
        /// Parameters:
        ///   ctx - Pointer to the concrete backend instance
        ///   event - The storage event to process
        onEvent: *const fn (ctx: *anyopaque, event: *const StorageEvent) anyerror!void,

        /// Load all persisted items for session restore.
        ///
        /// Returns all items in conversation order: `ORDER BY item_id ASC`.
        /// item_id is a monotonic counter, making it the authoritative ordering
        /// for replay. created_at_ms is metadata for display purposes only.
        ///
        /// The returned records include item_id and created_at_ms for
        /// proper identity restoration.
        ///
        /// # Collision Validation
        ///
        /// Implementations MUST verify `payload.session_id == requested_session_id`
        /// after SESSION_HASH match. Hash collisions should be skipped silently.
        ///
        /// Parameters:
        ///   ctx - Pointer to the concrete backend instance
        ///   allocator - Allocator for the returned slice and record data
        ///
        /// Returns:
        ///   Slice of ItemRecord on success (sorted by item_id ASC).
        ///   Returns error on IO failure.
        ///   Caller owns the returned memory and must free with freeItemRecords().
        loadAll: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) anyerror![]ItemRecord,

        /// Clean up backend resources.
        ///
        /// Called when the Conversation container is destroyed. Backends should:
        /// - Flush any pending writes
        /// - Close database connections / file handles
        /// - Free allocated memory
        ///
        /// Parameters:
        ///   ctx - Pointer to the concrete backend instance
        deinit: *const fn (ctx: *anyopaque) void,
    };

    /// Create a StorageBackend from a concrete implementation.
    ///
    /// Example:
    /// ```zig
    /// const backend = StorageBackend.init(&my_db_backend, &DbBackend.vtable);
    /// ```
    pub fn init(ptr: anytype, vtable: *const VTable) StorageBackend {
        const Ptr = @TypeOf(ptr);
        const ptr_info = @typeInfo(Ptr);

        comptime {
            if (ptr_info != .pointer) {
                @compileError("ptr must be a pointer type");
            }
        }

        return .{
            .ptr = @ptrCast(@alignCast(ptr)),
            .vtable = vtable,
        };
    }

    // =========================================================================
    // Public Interface
    // =========================================================================

    /// Emit a storage event.
    ///
    /// Call this for every mutation (finalize, delete, clear).
    /// Returns error if the event could not be persisted.
    pub fn onEvent(self: StorageBackend, event: *const StorageEvent) anyerror!void {
        return self.vtable.onEvent(self.ptr, event);
    }

    /// Load all items from storage.
    ///
    /// Returns error on IO failure.
    /// Returns empty slice if storage is empty.
    /// Caller owns returned memory.
    pub fn loadAll(self: StorageBackend, allocator: std.mem.Allocator) anyerror![]ItemRecord {
        return self.vtable.loadAll(self.ptr, allocator);
    }

    /// Clean up backend resources.
    pub fn deinit(self: StorageBackend) void {
        self.vtable.deinit(self.ptr);
    }
};

// =============================================================================
// Helper Functions
// =============================================================================

// Note: freeItemRecords() is defined above near ItemRecord.deinit()

/// Free a StoredItemEnvelope (including session_id and record).
///
/// Use this when manually decoding StoredItemEnvelope from storage.
pub fn freeStoredItemEnvelope(allocator: std.mem.Allocator, envelope: *StoredItemEnvelope) void {
    envelope.deinit(allocator);
}

// =============================================================================
// Backend Implementation Guide
// =============================================================================
//
// For the definitive storage specification, see:
// - env/features/storage/storage_layer_1.md (physical format, I/O protocol)
// - env/features/storage/schema_layer.md (policy parameters, column definitions)
// - env/features/storage/database_original.md (architecture, manifest, compaction)
// - core/src/responses/FUTURE.md (TaluDB implementation plan)
//
// =============================================================================
// SCHEMA_CHAT_ITEMS (schema_id = 3) — REQUIRED
// =============================================================================
//
// The chat namespace uses ONLY Schema ID 3 (Item-based storage).
// There is no legacy schema support. Any data with schema_id != 3 should be
// treated as corrupt/invalid.
//
// Location: To be implemented (src/db/table/sessions.zig)
//
// Write path:
//   1. Acquire <ns>/talu.lock
//   2. Append row to chat/current.talu with columns:
//      - ITEM_ID (1) = item_id
//      - TS (2) = created_at_ms (milliseconds)
//      - SESSION_HASH (3) = siphash(session_id, store.key)
//      - TYPE (4) = ItemType enum (0=message, 1=call, 2=output, 3=reasoning, 4=ref)
//      - ROLE (5) = MessageRole if TYPE=0, else 0xFF sentinel
//      - PAYLOAD (20) = msgpack(StoredItemEnvelope { session_id, record })
//   3. On seal: upload to remote (if configured)
//
// Read path (session restore):
//   1. Load manifest from authority
//   2. Compute SESSION_HASH from session_id
//   3. STRICT GUARD: if block.schema_id != 3, skip or error (no legacy support)
//   4. Jump-read: scan SESSION_HASH column across segments (Bloom filter pruning)
//   5. For matching rows: fetch ITEM_ID, TS, PAYLOAD
//   6. Decode PAYLOAD as StoredItemEnvelope
//   7. COLLISION CHECK: verify payload.session_id == requested_session_id
//   8. Filter by delete_map (item-level and session-level deletes)
//   9. Return in item_id ASC order (monotonic, authoritative for replay)
//
// =============================================================================

// =============================================================================
// Tests
// =============================================================================

test "StorageBackend.VTable has all required function pointers" {
    // Verify VTable struct has exactly 3 function pointer fields
    const info = @typeInfo(StorageBackend.VTable);
    try std.testing.expectEqual(@as(usize, 3), info.@"struct".fields.len);
    // Verify field names match expected interface
    try std.testing.expectEqualStrings("onEvent", info.@"struct".fields[0].name);
    try std.testing.expectEqualStrings("loadAll", info.@"struct".fields[1].name);
    try std.testing.expectEqualStrings("deinit", info.@"struct".fields[2].name);
}

test "StorageEvent union variants" {
    const allocator = std.testing.allocator;

    // Test PutItem
    const content = try allocator.alloc(ItemContentPartRecord, 1);
    defer allocator.free(content);
    content[0] = .{ .input_text = .{ .text = "test" } };

    const put = StorageEvent{ .PutItem = .{
        .item_id = 42,
        .created_at_ms = 1234567890,
        .item_type = .message,
        .variant = .{ .message = .{
            .role = .user,
            .status = .completed,
            .content = content,
        } },
    } };
    try std.testing.expect(put == .PutItem);
    try std.testing.expectEqual(@as(u64, 42), put.PutItem.item_id);
    try std.testing.expectEqual(ItemType.message, put.PutItem.item_type);

    // Test DeleteItem
    const del = StorageEvent{ .DeleteItem = .{
        .item_id = 42,
        .deleted_at_ms = 1234567890,
    } };
    try std.testing.expect(del == .DeleteItem);
    try std.testing.expectEqual(@as(u64, 42), del.DeleteItem.item_id);

    // Test ClearItems
    const clr = StorageEvent{ .ClearItems = .{
        .cleared_at_ms = 1234567890,
        .keep_context = true,
    } };
    try std.testing.expect(clr == .ClearItems);
    try std.testing.expect(clr.ClearItems.keep_context);
}

test "StorageBackend.init creates valid backend" {
    // Create a mock backend
    var mock = struct {
        event_count: usize = 0,

        const Self = @This();

        fn onEvent(ctx: *anyopaque, event: *const StorageEvent) anyerror!void {
            _ = event;
            const self: *Self = @ptrCast(@alignCast(ctx));
            self.event_count += 1;
        }

        fn loadAll(ctx: *anyopaque, allocator: std.mem.Allocator) anyerror![]ItemRecord {
            _ = ctx;
            return allocator.alloc(ItemRecord, 0);
        }

        fn deinit(ctx: *anyopaque) void {
            _ = ctx;
        }

        const vtable = StorageBackend.VTable{
            .onEvent = onEvent,
            .loadAll = loadAll,
            .deinit = deinit,
        };
    }{};

    const allocator = std.testing.allocator;
    const backend = StorageBackend.init(&mock, &@TypeOf(mock).vtable);

    // Test that vtable functions work through the backend interface
    const content = try allocator.alloc(ItemContentPartRecord, 1);
    defer allocator.free(content);
    content[0] = .{ .input_text = .{ .text = "test" } };

    const event = StorageEvent{ .PutItem = .{
        .item_id = 1,
        .created_at_ms = 123,
        .item_type = .message,
        .variant = .{ .message = .{
            .role = .user,
            .status = .completed,
            .content = content,
        } },
    } };
    try backend.onEvent(&event);
    try std.testing.expectEqual(@as(usize, 1), mock.event_count);

    // Test that backend stores correct vtable reference
    try std.testing.expectEqual(&@TypeOf(mock).vtable, backend.vtable);
}

test "StorageBackend.onEvent handles PutItem" {
    var mock = struct {
        last_item_id: u64 = 0,
        last_item_type: ?ItemType = null,
        call_count: usize = 0,

        const Self = @This();

        fn onEvent(ctx: *anyopaque, event: *const StorageEvent) anyerror!void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            switch (event.*) {
                .PutItem => |record| {
                    self.last_item_id = record.item_id;
                    self.last_item_type = record.item_type;
                },
                else => {},
            }
            self.call_count += 1;
        }

        fn loadAll(ctx: *anyopaque, allocator: std.mem.Allocator) anyerror![]ItemRecord {
            _ = ctx;
            return allocator.alloc(ItemRecord, 0);
        }

        fn deinit(ctx: *anyopaque) void {
            _ = ctx;
        }

        const vtable = StorageBackend.VTable{
            .onEvent = onEvent,
            .loadAll = loadAll,
            .deinit = deinit,
        };
    }{};

    const allocator = std.testing.allocator;
    const backend = StorageBackend.init(&mock, &@TypeOf(mock).vtable);

    // Test with message item
    const content = try allocator.alloc(ItemContentPartRecord, 1);
    defer allocator.free(content);
    content[0] = .{ .input_text = .{ .text = "Hello!" } };

    const msg_event = StorageEvent{ .PutItem = .{
        .item_id = 42,
        .created_at_ms = 1234567890,
        .item_type = .message,
        .variant = .{ .message = .{
            .role = .user,
            .status = .completed,
            .content = content,
        } },
    } };
    try backend.onEvent(&msg_event);

    try std.testing.expectEqual(@as(u64, 42), mock.last_item_id);
    try std.testing.expectEqual(ItemType.message, mock.last_item_type.?);
    try std.testing.expectEqual(@as(usize, 1), mock.call_count);

    // Test with function_call item
    const fc_event = StorageEvent{ .PutItem = .{
        .item_id = 43,
        .created_at_ms = 1234567891,
        .item_type = .function_call,
        .variant = .{ .function_call = .{
            .call_id = "call_123",
            .name = "get_weather",
            .arguments = "{}",
            .status = .completed,
        } },
    } };
    try backend.onEvent(&fc_event);

    try std.testing.expectEqual(@as(u64, 43), mock.last_item_id);
    try std.testing.expectEqual(ItemType.function_call, mock.last_item_type.?);
    try std.testing.expectEqual(@as(usize, 2), mock.call_count);
}

test "StorageBackend.onEvent handles DeleteItem" {
    var mock = struct {
        deleted_item_id: u64 = 0,
        delete_count: usize = 0,

        const Self = @This();

        fn onEvent(ctx: *anyopaque, event: *const StorageEvent) anyerror!void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            switch (event.*) {
                .DeleteItem => |del| {
                    self.deleted_item_id = del.item_id;
                    self.delete_count += 1;
                },
                else => {},
            }
        }

        fn loadAll(ctx: *anyopaque, allocator: std.mem.Allocator) anyerror![]ItemRecord {
            _ = ctx;
            return allocator.alloc(ItemRecord, 0);
        }

        fn deinit(ctx: *anyopaque) void {
            _ = ctx;
        }

        const vtable = StorageBackend.VTable{
            .onEvent = onEvent,
            .loadAll = loadAll,
            .deinit = deinit,
        };
    }{};

    const backend = StorageBackend.init(&mock, &@TypeOf(mock).vtable);

    const del_event = StorageEvent{ .DeleteItem = .{
        .item_id = 99,
        .deleted_at_ms = 1234567890,
    } };
    try backend.onEvent(&del_event);

    try std.testing.expectEqual(@as(u64, 99), mock.deleted_item_id);
    try std.testing.expectEqual(@as(usize, 1), mock.delete_count);
}

test "StorageBackend.onEvent handles ClearItems" {
    var mock = struct {
        cleared: bool = false,
        keep_context: bool = false,
        clear_count: usize = 0,

        const Self = @This();

        fn onEvent(ctx: *anyopaque, event: *const StorageEvent) anyerror!void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            switch (event.*) {
                .ClearItems => |clr| {
                    self.cleared = true;
                    self.keep_context = clr.keep_context;
                    self.clear_count += 1;
                },
                else => {},
            }
        }

        fn loadAll(ctx: *anyopaque, allocator: std.mem.Allocator) anyerror![]ItemRecord {
            _ = ctx;
            return allocator.alloc(ItemRecord, 0);
        }

        fn deinit(ctx: *anyopaque) void {
            _ = ctx;
        }

        const vtable = StorageBackend.VTable{
            .onEvent = onEvent,
            .loadAll = loadAll,
            .deinit = deinit,
        };
    }{};

    const backend = StorageBackend.init(&mock, &@TypeOf(mock).vtable);

    // Test clear without keep_context
    const clr_event1 = StorageEvent{ .ClearItems = .{
        .cleared_at_ms = 1234567890,
        .keep_context = false,
    } };
    try backend.onEvent(&clr_event1);

    try std.testing.expect(mock.cleared);
    try std.testing.expect(!mock.keep_context);
    try std.testing.expectEqual(@as(usize, 1), mock.clear_count);

    // Test clear with keep_context
    const clr_event2 = StorageEvent{ .ClearItems = .{
        .cleared_at_ms = 1234567891,
        .keep_context = true,
    } };
    try backend.onEvent(&clr_event2);

    try std.testing.expect(mock.keep_context);
    try std.testing.expectEqual(@as(usize, 2), mock.clear_count);
}

test "StoredItemEnvelope struct layout" {
    const allocator = std.testing.allocator;

    // Create a StoredItemEnvelope
    const content = try allocator.alloc(ItemContentPartRecord, 1);
    errdefer allocator.free(content);
    const text = try allocator.dupe(u8, "Hello!");
    errdefer allocator.free(text);
    content[0] = .{ .input_text = .{ .text = text } };

    var envelope = StoredItemEnvelope{
        .session_id = try allocator.dupe(u8, "user/123/chat/456"),
        .record = ItemRecord{
            .item_id = 42,
            .created_at_ms = 1234567890,
            .item_type = .message,
            .variant = .{ .message = .{
                .role = .user,
                .status = .completed,
                .content = content,
            } },
            .metadata = null,
        },
    };
    defer envelope.deinit(allocator);

    // Verify fields
    try std.testing.expectEqualStrings("user/123/chat/456", envelope.session_id);
    try std.testing.expectEqual(@as(u64, 42), envelope.record.item_id);
    try std.testing.expectEqual(ItemType.message, envelope.record.item_type);
}

test "freeItemRecords handles empty array" {
    const allocator = std.testing.allocator;

    const records = try allocator.alloc(ItemRecord, 0);
    freeItemRecords(allocator, records);
    // Should not crash on empty array
}

test "freeItemRecords handles records with content" {
    const allocator = std.testing.allocator;

    const records = try allocator.alloc(ItemRecord, 2);
    errdefer allocator.free(records);

    // First record: message
    const content1 = try allocator.alloc(ItemContentPartRecord, 1);
    errdefer allocator.free(content1);
    const text1 = try allocator.dupe(u8, "Content 1");
    content1[0] = .{ .input_text = .{ .text = text1 } };
    records[0] = ItemRecord{
        .item_id = 1,
        .created_at_ms = 1000,
        .item_type = .message,
        .variant = .{ .message = .{
            .role = .user,
            .status = .completed,
            .content = content1,
        } },
        .metadata = null,
    };

    // Second record: function_call
    records[1] = ItemRecord{
        .item_id = 2,
        .created_at_ms = 2000,
        .item_type = .function_call,
        .variant = .{ .function_call = .{
            .call_id = try allocator.dupe(u8, "call_1"),
            .name = try allocator.dupe(u8, "test_func"),
            .arguments = try allocator.dupe(u8, "{}"),
            .status = .completed,
        } },
        .metadata = null,
    };

    freeItemRecords(allocator, records);
    // Should handle all records correctly without leaks
}

test "freeStoredItemEnvelope handles full envelope" {
    const allocator = std.testing.allocator;

    const content = try allocator.alloc(ItemContentPartRecord, 1);
    errdefer allocator.free(content);
    const text = try allocator.dupe(u8, "Response text");
    errdefer allocator.free(text);
    const annotations = try allocator.dupe(u8, "[{\"url\":\"https://example.com\"}]");
    content[0] = .{ .output_text = .{
        .text = text,
        .logprobs_json = null,
        .annotations_json = annotations,
    } };

    var envelope = StoredItemEnvelope{
        .session_id = try allocator.dupe(u8, "session_abc"),
        .record = ItemRecord{
            .item_id = 100,
            .created_at_ms = 5000,
            .item_type = .message,
            .variant = .{ .message = .{
                .role = .assistant,
                .status = .completed,
                .content = content,
            } },
            .metadata = try allocator.dupe(u8, "{\"custom\":\"data\"}"),
        },
    };

    freeStoredItemEnvelope(allocator, &envelope);
    // Should handle session_id, metadata, content, and annotations without leaks
}
