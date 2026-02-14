//! Responses Module - Item-based data model for the Open Responses API.
//!
//! This module provides the core data types for the Open Responses architecture:
//!   - Item: Atomic unit (message, function_call, reasoning, etc.)
//!   - Conversation: Container for Items with serialization
//!   - Chat: Lightweight wrapper with sampling parameters
//!   - StorageBackend: Persistence interface for conversation history
//!
//! Architecture:
//!   Chat (sampling config, session identity)
//!       │
//!       ▼
//!   Conversation (Item container, serialization)
//!       │
//!       │ on item finalized (status → final)
//!       ▼
//!   StorageBackend (optional persistence)
//!       │
//!       └── MemoryBackend     (:memory: - default, no-op)
//!
//! Usage:
//!   const conv = try Conversation.init(allocator);
//!   defer conv.deinit();
//!
//!   const user_item = try conv.appendUserMessage("Hello!");
//!   const asst_item = try conv.appendAssistantMessage();
//!   try conv.appendTextContent(asst_item, "Hi there!");
//!   conv.finalizeItem(asst_item);
//!
//! Design Principles:
//!   1. Zig Memory is always source of truth during runtime
//!   2. Storage backends mirror data for persistence (write-through)
//!   3. Session restore loads from storage into Zig memory
//!   4. Zero overhead when no backend is configured (null check)
//!
//! See also:
//!   - capi/responses.zig - C API for Item access

const std = @import("std");

pub const backend = @import("backend.zig");
pub const memory = @import("memory.zig");
pub const chat = @import("chat.zig");
pub const storage_serializer = @import("storage_serializer.zig");
pub const record_serializer = @import("record_serializer.zig");
pub const record_parser = @import("record_parser.zig");
pub const reasoning_parser = @import("reasoning_parser.zig");
pub const session_id = @import("session_id.zig");

// Open Responses architecture modules
pub const items = @import("items.zig");
pub const conversation = @import("conversation.zig");

// Re-export main types
pub const StorageBackend = backend.StorageBackend;
pub const StorageEvent = backend.StorageEvent;
pub const MemoryBackend = memory.MemoryBackend;

// Item-based architecture types
pub const Conversation = conversation.Conversation;
pub const Item = items.Item;
pub const ItemType = items.ItemType;
pub const ItemStatus = items.ItemStatus;
pub const ItemVariant = items.ItemVariant;
pub const MessageData = items.MessageData;
pub const MessageRole = items.MessageRole;
pub const FunctionCallData = items.FunctionCallData;
pub const FunctionCallOutputData = items.FunctionCallOutputData;
pub const FunctionOutputValue = items.FunctionOutputValue;
pub const ReasoningData = items.ReasoningData;
pub const ItemReferenceData = items.ItemReferenceData;
pub const ContentType = items.ContentType;
pub const ContentPart = items.ContentPart;
pub const ContentVariant = items.ContentVariant;
pub const ImageDetail = items.ImageDetail;
pub const UrlCitation = items.UrlCitation;

// Reasoning parser
pub const ReasoningParser = reasoning_parser.ReasoningParser;
pub const ReasoningFormat = reasoning_parser.ReasoningFormat;
pub const generateSessionId = session_id.generateSessionId;

// Serialization types
pub const SerializationDirection = conversation.SerializationDirection;
pub const ResponsesSerializationOptions = conversation.ResponsesSerializationOptions;

// Item storage types
pub const ItemRecord = backend.ItemRecord;
pub const StoredItemEnvelope = backend.StoredItemEnvelope;

// Storage serialization (for Item - live in-memory objects)
pub const serializeItemToJson = storage_serializer.serializeItemToJson;
pub const serializeItemToJsonZ = storage_serializer.serializeItemToJsonZ;
pub const extractRole = storage_serializer.extractRole;

// Record serialization (for ItemRecord - portable snapshots)
pub const serializeItemRecordToJsonZ = record_serializer.serializeItemRecordToJsonZ;
pub const serializeItemRecordToKvBuf = record_serializer.serializeItemRecordToKvBuf;
pub const serializeItemRecordToKvBufWithStorage = record_serializer.serializeItemRecordToKvBufWithStorage;
pub const extractRoleFromRecord = record_serializer.extractRoleFromRecord;

// Record parsing (JSON → ItemRecord)
pub const parseItemVariantRecord = record_parser.parseItemVariantRecord;
pub const freeContentPartRecord = record_parser.freeContentPartRecord;
pub const itemStatusFromU8 = record_parser.itemStatusFromU8;

// Re-export chat types (canonical location)
pub const Chat = chat.Chat;
pub const ResolutionConfig = chat.ResolutionConfig;

/// Storage type enum for C API.
/// Used by talu_chat_set_storage() to select backend.
pub const StorageType = enum(u8) {
    /// In-memory only, no persistence (default).
    memory = 0,

    _reserved_1 = 1,
    _reserved_2 = 2,
    _reserved_3 = 3,
    _reserved_4 = 4,

    /// Reserved for custom/external backends.
    custom = 255,
};

// =============================================================================
// Tests
// =============================================================================

test "storage module re-exports types" {
    // Verify public API types are accessible
    try std.testing.expect(@TypeOf(backend.StorageBackend) == type);
    try std.testing.expect(@TypeOf(memory.MemoryBackend) == type);
    try std.testing.expect(@TypeOf(conversation.Conversation) == type);
    try std.testing.expect(@TypeOf(chat.Chat) == type);
}

test "StorageType values" {
    try std.testing.expectEqual(@as(u8, 0), @intFromEnum(StorageType.memory));
    try std.testing.expectEqual(@as(u8, 255), @intFromEnum(StorageType.custom));
}
