//! KvBuf (Key-Value Buffer) - Directory-based binary format.
//!
//! Separates field values from structural metadata to enable zero-copy
//! field access. Designed for TaluDB PAYLOAD columns where search needs
//! to skip metadata and scan only content bytes.
//!
//! Binary layout per blob:
//!   [ Values Area ]  [ Directory ]  [ Footer ]
//!
//! See writer.zig and reader.zig for encoding/decoding details.

pub const KvBufWriter = @import("writer.zig").KvBufWriter;
pub const KvBufReader = @import("reader.zig").KvBufReader;

pub const KVBUF_MAGIC = @import("writer.zig").KVBUF_MAGIC;
pub const FOOTER_SIZE = @import("writer.zig").FOOTER_SIZE;
pub const ENTRY_SIZE = @import("writer.zig").ENTRY_SIZE;

/// Stable field IDs for ItemRecord KvBuf encoding.
///
/// These IDs are persisted on disk — never change or reuse an existing value.
pub const FieldIds = struct {
    /// session_id string (UTF-8).
    pub const session_id: u16 = 1;
    /// item JSON record (the full record sub-object, JSON-encoded).
    pub const record_json: u16 = 2;
    /// Concatenated plain text from all content parts (for search).
    pub const content_text: u16 = 3;
    /// External blob reference for large record_json payloads (`sha256:<hex>`).
    pub const record_json_ref: u16 = 4;
    /// Optional trigram bloom (bytes) for externalized record_json prefiltering.
    pub const record_json_trigram_bloom: u16 = 5;
};

/// Stable field IDs for SessionRecord KvBuf encoding (Schema 5).
///
/// These IDs are persisted on disk — never change or reuse an existing value.
pub const SessionFieldIds = struct {
    pub const session_id: u16 = 1;
    pub const title: u16 = 2;
    pub const model: u16 = 3;
    pub const system_prompt: u16 = 4;
    pub const config_json: u16 = 5;
    pub const marker: u16 = 6;
    pub const parent_session_id: u16 = 7;
    pub const group_id: u16 = 8;
    pub const head_item_id: u16 = 9;
    pub const ttl_ts: u16 = 10;
    pub const metadata_json: u16 = 11;
    pub const created_at_ms: u16 = 12;
    pub const updated_at_ms: u16 = 13;
    /// Space-separated normalized tags extracted from metadata_json.tags.
    /// DEPRECATED: Use TagFieldIds for tag entities instead.
    pub const tags_text: u16 = 14;
    /// Source document ID for lineage tracking.
    /// Links this session to the prompt/persona document that spawned it.
    pub const source_doc_id: u16 = 15;
};

/// Stable field IDs for Tag entity KvBuf encoding (Schema 6).
///
/// These IDs are persisted on disk — never change or reuse an existing value.
pub const TagFieldIds = struct {
    /// Tag UUID string (primary key).
    pub const tag_id: u16 = 1;
    /// Tag name (unique within group, lowercase).
    pub const name: u16 = 2;
    /// Optional hex color (e.g., "#4a90d9").
    pub const color: u16 = 3;
    /// Optional description text.
    pub const description: u16 = 4;
    /// Optional group_id for multi-tenant isolation.
    pub const group_id: u16 = 5;
    /// Created timestamp (ms since epoch).
    pub const created_at_ms: u16 = 6;
    /// Updated timestamp (ms since epoch).
    pub const updated_at_ms: u16 = 7;
};

/// Stable field IDs for ConversationTag junction KvBuf encoding (Schema 7).
///
/// These IDs are persisted on disk — never change or reuse an existing value.
pub const ConversationTagFieldIds = struct {
    /// Session UUID string (foreign key to sessions).
    pub const session_id: u16 = 1;
    /// Tag UUID string (foreign key to tags).
    pub const tag_id: u16 = 2;
    /// Timestamp when tag was added (ms since epoch).
    pub const added_at_ms: u16 = 3;
};

/// Stable field IDs for Document KvBuf encoding (Schema 11).
///
/// These IDs are persisted on disk — never change or reuse an existing value.
pub const DocumentFieldIds = struct {
    /// Document UUID string (primary key).
    pub const doc_id: u16 = 1;
    /// Document type string ("prompt", "persona", "rag", "tool", "folder").
    pub const doc_type: u16 = 2;
    /// Human-readable title for UI lists.
    pub const title: u16 = 3;
    /// Space-separated normalized tags for search.
    pub const tags_text: u16 = 4;
    /// Full JSON payload (envelope: {_sys, data}).
    pub const doc_json: u16 = 5;
    /// Parent document ID string (optional, for versioning/hierarchy).
    pub const parent_id: u16 = 6;
    /// Marker string ("active", "archived", "deleted").
    pub const marker: u16 = 7;
    /// Group ID for multi-tenant isolation.
    pub const group_id: u16 = 8;
    /// Owner user ID string (for "My Docs" filtering).
    pub const owner_id: u16 = 9;
    /// Created timestamp (ms since epoch).
    pub const created_at_ms: u16 = 10;
    /// Updated timestamp (ms since epoch).
    pub const updated_at_ms: u16 = 11;
    /// Version type for delta versioning ("full" | "delta").
    pub const version_type: u16 = 12;
    /// Base document ID for delta versions.
    pub const base_doc_id: u16 = 13;
    /// External blob reference for large doc_json payloads (`sha256:<hex>`).
    pub const doc_json_ref: u16 = 14;
    /// Optional trigram bloom (bytes) for externalized doc_json prefiltering.
    pub const doc_json_trigram_bloom: u16 = 15;
};

/// Stable field IDs for DocumentTag junction KvBuf encoding (Schema 13).
///
/// These IDs are persisted on disk — never change or reuse an existing value.
pub const DocumentTagFieldIds = struct {
    /// Document UUID string (foreign key to documents).
    pub const doc_id: u16 = 1;
    /// Tag UUID string (foreign key to tags).
    pub const tag_id: u16 = 2;
    /// Timestamp when tag was added (ms since epoch).
    pub const added_at_ms: u16 = 3;
};

/// Stable field IDs for Repo Pin KvBuf encoding (Schema 20).
///
/// These IDs are persisted on disk — never change or reuse an existing value.
pub const RepoPinFieldIds = struct {
    /// Model URI string (canonical identity).
    pub const model_uri: u16 = 1;
    /// Initial pin timestamp (ms since epoch).
    pub const pinned_at_ms: u16 = 2;
    /// Cached local size in bytes (optional).
    pub const size_bytes: u16 = 3;
    /// Timestamp when size was last updated (optional, ms since epoch).
    pub const size_updated_at_ms: u16 = 4;
};

/// Stable field IDs for TagDocumentIndex (inverted index) KvBuf encoding (Schema 14).
///
/// This is the INVERTED index: tag → documents (for O(1) tag lookups).
/// These IDs are persisted on disk — never change or reuse an existing value.
pub const TagDocumentIndexFieldIds = struct {
    /// Tag UUID string (the lookup key).
    pub const tag_id: u16 = 1;
    /// Document UUID string (the value).
    pub const doc_id: u16 = 2;
};

/// Detect whether a payload blob is KvBuf-encoded by checking the trailing magic byte.
/// Safe to call on any non-empty slice; returns false for JSON payloads.
pub fn isKvBuf(payload: []const u8) bool {
    if (payload.len < FOOTER_SIZE) return false;
    return payload[payload.len - 1] == KVBUF_MAGIC;
}

// =============================================================================
// Tests
// =============================================================================

const std = @import("std");

test "isKvBuf detects KvBuf blob" {
    const allocator = std.testing.allocator;
    var w = KvBufWriter.init();
    defer w.deinit(allocator);

    try w.addString(allocator, FieldIds.content_text, "hello");
    const blob = try w.finish(allocator);
    defer allocator.free(blob);

    try std.testing.expect(isKvBuf(blob));
}

test "isKvBuf rejects JSON" {
    try std.testing.expect(!isKvBuf("{\"type\":\"message\"}"));
}

test "isKvBuf rejects empty" {
    try std.testing.expect(!isKvBuf(""));
}

test "isKvBuf rejects short" {
    try std.testing.expect(!isKvBuf("abc"));
}

test "FieldIds are stable" {
    try std.testing.expectEqual(@as(u16, 1), FieldIds.session_id);
    try std.testing.expectEqual(@as(u16, 2), FieldIds.record_json);
    try std.testing.expectEqual(@as(u16, 3), FieldIds.content_text);
    try std.testing.expectEqual(@as(u16, 4), FieldIds.record_json_ref);
    try std.testing.expectEqual(@as(u16, 5), FieldIds.record_json_trigram_bloom);
}

test "DocumentFieldIds include doc_json_ref" {
    try std.testing.expectEqual(@as(u16, 5), DocumentFieldIds.doc_json);
    try std.testing.expectEqual(@as(u16, 13), DocumentFieldIds.base_doc_id);
    try std.testing.expectEqual(@as(u16, 14), DocumentFieldIds.doc_json_ref);
    try std.testing.expectEqual(@as(u16, 15), DocumentFieldIds.doc_json_trigram_bloom);
}

test "RepoPinFieldIds are stable" {
    try std.testing.expectEqual(@as(u16, 1), RepoPinFieldIds.model_uri);
    try std.testing.expectEqual(@as(u16, 2), RepoPinFieldIds.pinned_at_ms);
    try std.testing.expectEqual(@as(u16, 3), RepoPinFieldIds.size_bytes);
    try std.testing.expectEqual(@as(u16, 4), RepoPinFieldIds.size_updated_at_ms);
}
