//! Session storage schema constants.
//!
//! Single source of truth for schema IDs, column IDs, and the compaction
//! policy used by the "chat" namespace.

const generic = @import("../db/table/generic.zig");
pub const CompactionPolicy = generic.CompactionPolicy;

// ============================================================================
// Schema IDs
// ============================================================================

pub const schema_deletes: u16 = 2;
pub const schema_items: u16 = 3;
pub const schema_sessions: u16 = 4;
pub const schema_sessions_kvbuf: u16 = 5;
pub const schema_embeddings: u16 = 10;

/// True if the schema represents session metadata (msgpack or kvbuf).
pub fn isSessionSchema(schema_id: u16) bool {
    return schema_id == schema_sessions or schema_id == schema_sessions_kvbuf;
}

// ============================================================================
// Column IDs
// ============================================================================

pub const col_item_id: u32 = 1;
pub const col_ts: u32 = 2;
pub const col_session_hash: u32 = 3;
pub const col_group_hash: u32 = 6;
pub const col_head_item_id: u32 = 7;
pub const col_created_ts: u32 = 8;
pub const col_ttl_ts: u32 = 9;
pub const col_embedding: u32 = 10;
pub const col_project_hash: u32 = 11;
pub const col_payload: u32 = 20;

// ============================================================================
// Compaction Policy
// ============================================================================

/// Compaction policy for the "chat" namespace.
///
/// Active schemas: items (3), sessions_kvbuf (5), embeddings (10).
/// Tombstone schema: deletes (2).
/// Dedup by session_hash (col 3) for sessions, item_id (col 1) for items.
pub const session_compaction_policy = CompactionPolicy{
    .active_schema_ids = &[_]u16{ schema_items, schema_sessions_kvbuf, schema_embeddings },
    .tombstone_schema_id = schema_deletes,
    .dedup_column_id = col_session_hash,
    .ts_column_id = col_ts,
    .ttl_column_id = col_ttl_ts,
};

// ============================================================================
// Tests
// ============================================================================

const std = @import("std");

test "schema constants are non-overlapping" {
    const schema_ids = [_]u16{
        schema_deletes,
        schema_items,
        schema_sessions,
        schema_sessions_kvbuf,
        schema_embeddings,
    };
    // No duplicates
    for (schema_ids, 0..) |a, i| {
        for (schema_ids[i + 1 ..]) |b| {
            try std.testing.expect(a != b);
        }
    }
}

test "column IDs are non-overlapping" {
    const col_ids = [_]u32{
        col_item_id,
        col_ts,
        col_session_hash,
        col_group_hash,
        col_head_item_id,
        col_created_ts,
        col_ttl_ts,
        col_embedding,
        col_project_hash,
        col_payload,
    };
    for (col_ids, 0..) |a, i| {
        for (col_ids[i + 1 ..]) |b| {
            try std.testing.expect(a != b);
        }
    }
}

test "isSessionSchema identifies session schemas" {
    try std.testing.expect(isSessionSchema(schema_sessions));
    try std.testing.expect(isSessionSchema(schema_sessions_kvbuf));
    try std.testing.expect(!isSessionSchema(schema_items));
    try std.testing.expect(!isSessionSchema(schema_deletes));
    try std.testing.expect(!isSessionSchema(schema_embeddings));
}
