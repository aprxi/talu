//! C-API Documents Module - Universal document storage.
//!
//! This module provides the C-compatible interface for:
//! - Document CRUD operations (`talu_documents_*`)
//! - Document-tag associations (`talu_documents_add_tag`, `talu_documents_get_tags`)
//! - Document search operations (`talu_documents_search`)
//! - External blob references (`talu_documents_get_blob_ref`)

const std = @import("std");
const capi_error = @import("error.zig");
const error_codes = @import("error_codes.zig");
const documents = @import("../db/table/documents.zig");
const document_tags = @import("../db/table/document_tags.zig");

const allocator = std.heap.c_allocator;

// =============================================================================
// C-ABI Document Records
// =============================================================================

/// C-ABI document record for external backends.
pub const CDocumentRecord = extern struct {
    /// Document UUID (null-terminated).
    doc_id: ?[*:0]const u8,
    /// Document type: "prompt", "persona", "rag", "tool", "folder" (null-terminated).
    doc_type: ?[*:0]const u8,
    /// Human-readable title (null-terminated).
    title: ?[*:0]const u8,
    /// Space-separated normalized tags (null-terminated, may be null).
    tags_text: ?[*:0]const u8,
    /// Full JSON payload (null-terminated).
    doc_json: ?[*:0]const u8,
    /// Parent document ID for versioning/hierarchy (null-terminated, may be null).
    parent_id: ?[*:0]const u8,
    /// Marker: "active", "archived", "deleted" (null-terminated, may be null).
    marker: ?[*:0]const u8,
    /// Group ID for multi-tenant isolation (null-terminated, may be null).
    group_id: ?[*:0]const u8,
    /// Owner user ID for "My Docs" filtering (null-terminated, may be null).
    owner_id: ?[*:0]const u8,
    /// Creation timestamp (Unix milliseconds).
    created_at_ms: i64,
    /// Last update timestamp (Unix milliseconds).
    updated_at_ms: i64,
    /// Expiration timestamp (Unix milliseconds). 0 = never expires.
    expires_at_ms: i64,
    /// Content hash for CAS/dedup (SipHash of doc_json).
    content_hash: u64,
    /// CDC sequence number.
    seq_num: u64,
    /// Reserved for future expansion.
    _reserved: [8]u8 = [_]u8{0} ** 8,
};

/// C-ABI document summary for list views (lighter than full record).
pub const CDocumentSummary = extern struct {
    /// Document UUID (null-terminated).
    doc_id: ?[*:0]const u8,
    /// Document type (null-terminated).
    doc_type: ?[*:0]const u8,
    /// Human-readable title (null-terminated).
    title: ?[*:0]const u8,
    /// Last update timestamp (Unix milliseconds).
    updated_at_ms: i64,
    /// Creation timestamp (Unix milliseconds).
    created_at_ms: i64,
    /// Marker (null-terminated, may be null).
    marker: ?[*:0]const u8,
};

/// Document list container for C API.
pub const CDocumentList = extern struct {
    /// Array of document summaries.
    items: ?[*]CDocumentSummary,
    /// Number of documents in the array.
    count: usize,
    /// Whether there are more results beyond this page.
    has_more: bool,
    /// Internal: backing arena for string data.
    _arena: ?*anyopaque,
};

/// String list container for C API (used for tag IDs, doc IDs).
pub const CStringList = extern struct {
    /// Array of null-terminated strings.
    items: ?[*]?[*:0]const u8,
    /// Number of strings in the array.
    count: usize,
    /// Internal: backing arena for string data.
    _arena: ?*anyopaque,
};

/// C-ABI search result containing document and matching snippet.
pub const CSearchResult = extern struct {
    /// Document UUID (null-terminated).
    doc_id: ?[*:0]const u8,
    /// Document type (null-terminated).
    doc_type: ?[*:0]const u8,
    /// Human-readable title (null-terminated).
    title: ?[*:0]const u8,
    /// Snippet of text surrounding the match (null-terminated).
    snippet: ?[*:0]const u8,
};

/// Search result list container for C API.
pub const CSearchResultList = extern struct {
    /// Array of search results.
    items: ?[*]CSearchResult,
    /// Number of results in the array.
    count: usize,
    /// Internal: backing arena for string data.
    _arena: ?*anyopaque,
};

/// C-ABI change record for CDC feed.
pub const CChangeRecord = extern struct {
    /// Change sequence number (monotonically increasing).
    seq_num: u64,
    /// Document UUID (null-terminated).
    doc_id: ?[*:0]const u8,
    /// Change action: 1=create, 2=update, 3=delete.
    action: u8,
    /// Timestamp of the change (Unix milliseconds).
    timestamp_ms: i64,
    /// Document type (null-terminated, may be null for deletes).
    doc_type: ?[*:0]const u8,
    /// Document title (null-terminated, may be null for deletes).
    title: ?[*:0]const u8,
    /// Reserved for future expansion.
    _reserved: [7]u8 = [_]u8{0} ** 7,
};

/// Change list container for C API (CDC feed).
pub const CChangeList = extern struct {
    /// Array of change records.
    items: ?[*]CChangeRecord,
    /// Number of changes in the array.
    count: usize,
    /// Next sequence number to query for continuation.
    next_seq: u64,
    /// Internal: backing arena for string data.
    _arena: ?*anyopaque,
};

// =============================================================================
// Validation Helpers
// =============================================================================

fn validateDbPath(db_path: ?[*:0]const u8) ?[]const u8 {
    const slice = std.mem.sliceTo(db_path orelse {
        capi_error.setErrorWithCode(.invalid_argument, "db_path is null", .{});
        return null;
    }, 0);
    if (slice.len == 0) {
        capi_error.setErrorWithCode(.invalid_argument, "db_path is empty", .{});
        return null;
    }
    return slice;
}

fn optSlice(s: ?[*:0]const u8) ?[]const u8 {
    return if (s) |p| std.mem.span(p) else null;
}

fn validateRequiredArg(s: ?[*:0]const u8, comptime arg_name: []const u8) ?[]const u8 {
    const slice = std.mem.sliceTo(s orelse {
        capi_error.setErrorWithCode(.invalid_argument, arg_name ++ " is null", .{});
        return null;
    }, 0);
    if (slice.len == 0) {
        capi_error.setErrorWithCode(.invalid_argument, arg_name ++ " is empty", .{});
        return null;
    }
    return slice;
}

// =============================================================================
// List Builder Helpers
// =============================================================================

/// Build a CDocumentList from scanned document records.
fn buildDocumentList(records: []documents.DocumentRecord, limit: usize) !*CDocumentList {
    const list = allocator.create(CDocumentList) catch return error.OutOfMemory;
    errdefer allocator.destroy(list);

    const arena_ptr = allocator.create(std.heap.ArenaAllocator) catch return error.OutOfMemory;
    errdefer allocator.destroy(arena_ptr);
    arena_ptr.* = std.heap.ArenaAllocator.init(allocator);
    errdefer arena_ptr.deinit();

    const arena = arena_ptr.allocator();

    const effective_count = @min(records.len, limit);
    const items_buf = arena.alloc(CDocumentSummary, effective_count) catch return error.OutOfMemory;

    for (records[0..effective_count], 0..) |record, i| {
        items_buf[i] = std.mem.zeroes(CDocumentSummary);
        items_buf[i].doc_id = (arena.dupeZ(u8, record.doc_id) catch return error.OutOfMemory).ptr;
        items_buf[i].doc_type = (arena.dupeZ(u8, record.doc_type) catch return error.OutOfMemory).ptr;
        items_buf[i].title = (arena.dupeZ(u8, record.title) catch return error.OutOfMemory).ptr;
        items_buf[i].updated_at_ms = record.updated_at_ms;
        items_buf[i].created_at_ms = record.created_at_ms;
        items_buf[i].marker = if (record.marker) |m|
            (arena.dupeZ(u8, m) catch return error.OutOfMemory).ptr
        else
            null;
    }

    list.* = .{
        .items = if (effective_count > 0) items_buf.ptr else null,
        .count = effective_count,
        .has_more = records.len > limit,
        ._arena = @ptrCast(arena_ptr),
    };

    return list;
}

/// Build a CStringList from string slices.
fn buildStringList(strings: [][]const u8) !*CStringList {
    const list = allocator.create(CStringList) catch return error.OutOfMemory;
    errdefer allocator.destroy(list);

    const arena_ptr = allocator.create(std.heap.ArenaAllocator) catch return error.OutOfMemory;
    errdefer allocator.destroy(arena_ptr);
    arena_ptr.* = std.heap.ArenaAllocator.init(allocator);
    errdefer arena_ptr.deinit();

    const arena = arena_ptr.allocator();

    const items_buf = arena.alloc(?[*:0]const u8, strings.len) catch return error.OutOfMemory;

    for (strings, 0..) |s, i| {
        items_buf[i] = (arena.dupeZ(u8, s) catch return error.OutOfMemory).ptr;
    }

    list.* = .{
        .items = if (strings.len > 0) items_buf.ptr else null,
        .count = strings.len,
        ._arena = @ptrCast(arena_ptr),
    };

    return list;
}

/// Build a CSearchResultList from search results.
fn buildSearchResultList(results: []documents.SearchResult, limit: usize) !*CSearchResultList {
    const list = allocator.create(CSearchResultList) catch return error.OutOfMemory;
    errdefer allocator.destroy(list);

    const arena_ptr = allocator.create(std.heap.ArenaAllocator) catch return error.OutOfMemory;
    errdefer allocator.destroy(arena_ptr);
    arena_ptr.* = std.heap.ArenaAllocator.init(allocator);
    errdefer arena_ptr.deinit();

    const arena = arena_ptr.allocator();

    const effective_count = @min(results.len, limit);
    const items_buf = arena.alloc(CSearchResult, effective_count) catch return error.OutOfMemory;

    for (results[0..effective_count], 0..) |result, i| {
        items_buf[i] = std.mem.zeroes(CSearchResult);
        items_buf[i].doc_id = (arena.dupeZ(u8, result.doc_id) catch return error.OutOfMemory).ptr;
        items_buf[i].doc_type = (arena.dupeZ(u8, result.doc_type) catch return error.OutOfMemory).ptr;
        items_buf[i].title = (arena.dupeZ(u8, result.title) catch return error.OutOfMemory).ptr;
        items_buf[i].snippet = (arena.dupeZ(u8, result.snippet) catch return error.OutOfMemory).ptr;
    }

    list.* = .{
        .items = if (effective_count > 0) items_buf.ptr else null,
        .count = effective_count,
        ._arena = @ptrCast(arena_ptr),
    };

    return list;
}

/// Build a CChangeList from change records.
fn buildChangeList(records: []documents.ChangeRecord, limit: usize) !*CChangeList {
    const list = allocator.create(CChangeList) catch return error.OutOfMemory;
    errdefer allocator.destroy(list);

    const arena_ptr = allocator.create(std.heap.ArenaAllocator) catch return error.OutOfMemory;
    errdefer allocator.destroy(arena_ptr);
    arena_ptr.* = std.heap.ArenaAllocator.init(allocator);
    errdefer arena_ptr.deinit();

    const arena = arena_ptr.allocator();

    const effective_count = @min(records.len, limit);
    const items_buf = arena.alloc(CChangeRecord, effective_count) catch return error.OutOfMemory;

    var max_seq: u64 = 0;
    for (records[0..effective_count], 0..) |record, i| {
        items_buf[i] = std.mem.zeroes(CChangeRecord);
        items_buf[i].seq_num = record.seq_num;
        items_buf[i].doc_id = (arena.dupeZ(u8, record.doc_id) catch return error.OutOfMemory).ptr;
        items_buf[i].action = @intFromEnum(record.action);
        items_buf[i].timestamp_ms = record.timestamp_ms;
        items_buf[i].doc_type = if (record.doc_type) |t|
            (arena.dupeZ(u8, t) catch return error.OutOfMemory).ptr
        else
            null;
        items_buf[i].title = if (record.title) |t|
            (arena.dupeZ(u8, t) catch return error.OutOfMemory).ptr
        else
            null;

        if (record.seq_num > max_seq) max_seq = record.seq_num;
    }

    list.* = .{
        .items = if (effective_count > 0) items_buf.ptr else null,
        .count = effective_count,
        .next_seq = max_seq + 1,
        ._arena = @ptrCast(arena_ptr),
    };

    return list;
}

// =============================================================================
// Document CRUD API
// =============================================================================

/// Create a new document.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - doc_id: Document UUID (null-terminated)
///   - doc_type: Document type (null-terminated)
///   - title: Human-readable title (null-terminated)
///   - doc_json: Full JSON payload (null-terminated)
///   - tags_text: Space-separated tags (null = no tags)
///   - parent_id: Parent document ID (null = no parent)
///   - marker: Marker string (null = "active")
///   - group_id: Group ID (null = no group)
///   - owner_id: Owner user ID (null = no owner)
///
/// Returns: 0 on success, negative error code on failure.
// lint:ignore capi-callconv - callconv(.c) on closing line
pub export fn talu_documents_create(
    db_path: ?[*:0]const u8,
    doc_id: ?[*:0]const u8,
    doc_type: ?[*:0]const u8,
    title: ?[*:0]const u8,
    doc_json: ?[*:0]const u8,
    tags_text: ?[*:0]const u8,
    parent_id: ?[*:0]const u8,
    marker: ?[*:0]const u8,
    group_id: ?[*:0]const u8,
    owner_id: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const doc_id_slice = validateRequiredArg(doc_id, "doc_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const doc_type_slice = validateRequiredArg(doc_type, "doc_type") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const title_slice = validateRequiredArg(title, "title") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const doc_json_slice = validateRequiredArg(doc_json, "doc_json") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);

    const now_ms = std.time.milliTimestamp();
    const record = documents.DocumentRecord{
        .doc_id = doc_id_slice,
        .doc_type = doc_type_slice,
        .title = title_slice,
        .doc_json = doc_json_slice,
        .tags_text = optSlice(tags_text),
        .parent_id = optSlice(parent_id),
        .marker = optSlice(marker),
        .group_id = optSlice(group_id),
        .owner_id = optSlice(owner_id),
        .created_at_ms = now_ms,
        .updated_at_ms = now_ms,
    };

    documents.createDocument(allocator, db_path_slice, record) catch |err| {
        capi_error.setErrorWithCode(.storage_error, "Failed to create document: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.ErrorCode.storage_error);
    };

    return 0;
}

/// Get a document by ID.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - doc_id: Document UUID to query (null-terminated)
///   - out_doc: Output parameter to receive document record
///
/// Returns: 0 on success, storage_error if document not found.
///
/// Note: String fields use borrowed pointers stored in a thread-local buffer.
/// They are valid only until the next talu_documents_get call.
pub export fn talu_documents_get(
    db_path: ?[*:0]const u8,
    doc_id: ?[*:0]const u8,
    out_doc: *CDocumentRecord,
) callconv(.c) i32 {
    capi_error.clearError();
    out_doc.* = std.mem.zeroes(CDocumentRecord);
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const doc_id_slice = validateRequiredArg(doc_id, "doc_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);

    var record = documents.getDocument(allocator, db_path_slice, doc_id_slice) catch |err| {
        capi_error.setErrorWithCode(.storage_error, "Failed to get document: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.ErrorCode.storage_error);
    } orelse {
        capi_error.setErrorWithCode(.storage_error, "Document not found", .{});
        return @intFromEnum(error_codes.ErrorCode.storage_error);
    };
    defer record.deinit(allocator);

    populateDocumentRecord(out_doc, record);
    return 0;
}

// =============================================================================
// Document Blob API
// =============================================================================

/// Get the external blob reference for a document's JSON payload.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - doc_id: Document UUID to query (null-terminated)
///   - out_blob_ref: Output buffer for `sha256:<hex>` or `multi:<hex>` (nullable)
///   - out_blob_ref_capacity: Output buffer capacity in bytes
///   - out_has_external_ref: Output flag (true when ref exists, false for inline payload)
///
/// Returns: 0 on success, negative error code on failure.
///
/// Notes:
///   - For inline documents, `out_has_external_ref=false` and `out_blob_ref` is set to empty if provided.
///   - If `out_has_external_ref=true`, `out_blob_ref` must be large enough for ref + NUL.
pub export fn talu_documents_get_blob_ref(
    db_path: ?[*:0]const u8,
    doc_id: ?[*:0]const u8,
    out_blob_ref: ?[*]u8,
    out_blob_ref_capacity: usize,
    out_has_external_ref: ?*bool,
) callconv(.c) i32 {
    capi_error.clearError();
    const has_ref = out_has_external_ref orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_has_external_ref is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    has_ref.* = false;
    if (out_blob_ref_capacity > 0) {
        if (out_blob_ref) |buf| {
            buf[0] = 0;
        } else {
            capi_error.setErrorWithCode(.invalid_argument, "out_blob_ref is null", .{});
            return @intFromEnum(error_codes.ErrorCode.invalid_argument);
        }
    }
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const doc_id_slice = validateRequiredArg(doc_id, "doc_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);

    const header = (documents.getDocumentHeader(allocator, db_path_slice, doc_id_slice) catch |err| {
        capi_error.setErrorWithCode(.storage_error, "Failed to get document header: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.ErrorCode.storage_error);
    }) orelse {
        capi_error.setErrorWithCode(.storage_error, "Document not found", .{});
        return @intFromEnum(error_codes.ErrorCode.storage_error);
    };
    defer documents.freeDocumentHeader(allocator, header);

    const blob_ref = header.doc_json_ref orelse return 0;
    if (out_blob_ref == null or out_blob_ref_capacity <= blob_ref.len) {
        capi_error.setErrorWithCode(.resource_exhausted, "out_blob_ref buffer too small (need at least {d} bytes)", .{blob_ref.len + 1});
        return @intFromEnum(error_codes.ErrorCode.resource_exhausted);
    }
    @memcpy(out_blob_ref.?[0..blob_ref.len], blob_ref);
    out_blob_ref.?[blob_ref.len] = 0;
    has_ref.* = true;
    return 0;
}

/// Update an existing document.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - doc_id: Document UUID to update (null-terminated)
///   - title: New title (null = no change)
///   - doc_json: New JSON payload (null = no change)
///   - tags_text: New tags (null = no change)
///   - marker: New marker (null = no change)
///
/// Returns: 0 on success, storage_error if document not found.
pub export fn talu_documents_update(
    db_path: ?[*:0]const u8,
    doc_id: ?[*:0]const u8,
    title: ?[*:0]const u8,
    doc_json: ?[*:0]const u8,
    tags_text: ?[*:0]const u8,
    marker: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const doc_id_slice = validateRequiredArg(doc_id, "doc_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);

    documents.updateDocument(
        allocator,
        db_path_slice,
        doc_id_slice,
        optSlice(title),
        optSlice(tags_text),
        optSlice(doc_json),
        optSlice(marker),
    ) catch |err| {
        if (err == error.DocumentNotFound) {
            capi_error.setErrorWithCode(.storage_error, "Document not found", .{});
        } else {
            capi_error.setErrorWithCode(.storage_error, "Failed to update document: {s}", .{@errorName(err)});
        }
        return @intFromEnum(error_codes.ErrorCode.storage_error);
    };

    return 0;
}

/// Delete a document (soft delete).
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - doc_id: Document UUID to delete (null-terminated)
///
/// Returns: 0 on success, negative error code on failure.
pub export fn talu_documents_delete(
    db_path: ?[*:0]const u8,
    doc_id: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const doc_id_slice = validateRequiredArg(doc_id, "doc_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);

    documents.deleteDocument(allocator, db_path_slice, doc_id_slice) catch |err| {
        if (err == error.DocumentNotFound) {
            capi_error.setErrorWithCode(.storage_error, "Document not found", .{});
        } else {
            capi_error.setErrorWithCode(.storage_error, "Failed to delete document: {s}", .{@errorName(err)});
        }
        return @intFromEnum(error_codes.ErrorCode.storage_error);
    };

    return 0;
}

// =============================================================================
// Batch Operations API
// =============================================================================

/// Batch-delete multiple documents (soft delete by writing tombstones).
/// Non-existent document IDs are silently skipped (idempotent batch semantics).
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - doc_ids: Array of null-terminated document ID strings
///   - doc_ids_count: Number of elements in doc_ids array
///   - out_deleted_count: Output parameter to receive the number of actually deleted documents (may be null)
///
/// Returns: 0 on success, negative error code on failure.
pub export fn talu_documents_delete_batch(
    db_path: ?[*:0]const u8,
    doc_ids: ?[*]const ?[*:0]const u8,
    doc_ids_count: usize,
    doc_type: ?[*:0]const u8,
    out_deleted_count: ?*usize,
) callconv(.c) i32 {
    capi_error.clearError();
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const doc_type_slice = optSlice(doc_type);

    if (doc_ids == null or doc_ids_count == 0) {
        if (out_deleted_count) |out| out.* = 0;
        return 0;
    }

    // Convert C string array to slices.
    var id_slices = allocator.alloc([]const u8, doc_ids_count) catch {
        capi_error.setErrorWithCode(.out_of_memory, "Failed to allocate id slice array", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };
    defer allocator.free(id_slices);

    const ids_ptr = doc_ids.?;
    for (0..doc_ids_count) |i| {
        const c_str = ids_ptr[i] orelse {
            capi_error.setErrorWithCode(.invalid_argument, "doc_ids[{d}] is null", .{i});
            return @intFromEnum(error_codes.ErrorCode.invalid_argument);
        };
        id_slices[i] = std.mem.sliceTo(c_str, 0);
    }

    const count = documents.deleteDocumentsBatch(allocator, db_path_slice, id_slices, doc_type_slice) catch |err| {
        capi_error.setErrorWithCode(.storage_error, "Batch delete failed: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.ErrorCode.storage_error);
    };

    if (out_deleted_count) |out| out.* = count;
    return 0;
}

/// Batch-update the marker field for multiple documents.
/// Non-existent document IDs are silently skipped.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - doc_ids: Array of null-terminated document ID strings
///   - doc_ids_count: Number of elements in doc_ids array
///   - marker: New marker value (null-terminated, e.g. "archived" or "active")
///   - out_updated_count: Output parameter to receive the number of actually updated documents (may be null)
///
/// Returns: 0 on success, negative error code on failure.
pub export fn talu_documents_set_marker_batch(
    db_path: ?[*:0]const u8,
    doc_ids: ?[*]const ?[*:0]const u8,
    doc_ids_count: usize,
    marker: ?[*:0]const u8,
    doc_type: ?[*:0]const u8,
    out_updated_count: ?*usize,
) callconv(.c) i32 {
    capi_error.clearError();
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const marker_slice = validateRequiredArg(marker, "marker") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const doc_type_slice = optSlice(doc_type);

    if (doc_ids == null or doc_ids_count == 0) {
        if (out_updated_count) |out| out.* = 0;
        return 0;
    }

    // Convert C string array to slices.
    var id_slices = allocator.alloc([]const u8, doc_ids_count) catch {
        capi_error.setErrorWithCode(.out_of_memory, "Failed to allocate id slice array", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };
    defer allocator.free(id_slices);

    const ids_ptr = doc_ids.?;
    for (0..doc_ids_count) |i| {
        const c_str = ids_ptr[i] orelse {
            capi_error.setErrorWithCode(.invalid_argument, "doc_ids[{d}] is null", .{i});
            return @intFromEnum(error_codes.ErrorCode.invalid_argument);
        };
        id_slices[i] = std.mem.sliceTo(c_str, 0);
    }

    const count = documents.setMarkerBatch(allocator, db_path_slice, id_slices, marker_slice, doc_type_slice) catch |err| {
        capi_error.setErrorWithCode(.storage_error, "Batch set marker failed: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.ErrorCode.storage_error);
    };

    if (out_updated_count) |out| out.* = count;
    return 0;
}

// =============================================================================
// Document Listing API
// =============================================================================

/// List documents with optional filters.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - doc_type: Filter by type (null = all types)
///   - group_id: Filter by group (null = all groups)
///   - owner_id: Filter by owner (null = all owners)
///   - marker: Filter by marker (null = all markers)
///   - limit: Maximum number of results
///   - out_list: Output parameter to receive document list handle
///
/// Returns: 0 on success, negative error code on failure.
/// On success, caller must free the handle with talu_documents_free_list().
pub export fn talu_documents_list(
    db_path: ?[*:0]const u8,
    doc_type: ?[*:0]const u8,
    group_id: ?[*:0]const u8,
    owner_id: ?[*:0]const u8,
    marker: ?[*:0]const u8,
    limit: u32,
    out_list: ?*?*CDocumentList,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_list orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_list is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);

    const effective_limit: usize = if (limit == 0) 100 else @intCast(limit);

    const records = documents.listDocuments(
        allocator,
        db_path_slice,
        optSlice(doc_type),
        optSlice(group_id),
        optSlice(owner_id),
        optSlice(marker),
    ) catch |err| {
        capi_error.setErrorWithCode(.storage_error, "Failed to list documents: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.ErrorCode.storage_error);
    };
    defer documents.freeDocumentRecords(allocator, @constCast(records));

    const list = buildDocumentList(records, effective_limit) catch |err| {
        capi_error.setErrorWithCode(.out_of_memory, "Failed to build document list: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };

    out.* = list;
    return 0;
}

/// Free a document list returned by talu_documents_list.
///
/// Parameters:
///   - list: Document list handle to free (may be null)
pub export fn talu_documents_free_list(list: ?*CDocumentList) callconv(.c) void {
    capi_error.clearError();
    const l = list orelse return;

    if (l._arena) |arena_ptr| {
        const arena: *std.heap.ArenaAllocator = @ptrCast(@alignCast(arena_ptr));
        arena.deinit();
        allocator.destroy(arena);
    }

    allocator.destroy(l);
}

// =============================================================================
// Document Search API
// =============================================================================

/// Search documents by content (title, tags, JSON payload).
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - query: Search query text (null-terminated)
///   - doc_type: Filter by document type (may be null for all types)
///   - limit: Maximum number of results (0 = use default of 100)
///   - out_list: Output parameter to receive search result list handle
///
/// Returns: 0 on success, negative error code on failure.
/// On success, caller must free the handle with talu_documents_free_search_results().
pub export fn talu_documents_search(
    db_path: ?[*:0]const u8,
    query: ?[*:0]const u8,
    doc_type: ?[*:0]const u8,
    limit: u32,
    out_list: ?*?*CSearchResultList,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_list orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_list is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const query_slice = validateRequiredArg(query, "query") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);

    const effective_limit: usize = if (limit == 0) 100 else @intCast(limit);

    const results = documents.searchDocuments(
        allocator,
        db_path_slice,
        query_slice,
        optSlice(doc_type),
    ) catch |err| {
        capi_error.setErrorWithCode(.storage_error, "Failed to search documents: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.ErrorCode.storage_error);
    };
    defer documents.freeSearchResults(allocator, @constCast(results));

    const list = buildSearchResultList(results, effective_limit) catch |err| {
        capi_error.setErrorWithCode(.out_of_memory, "Failed to build search result list: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };

    out.* = list;
    return 0;
}

/// Free a search result list returned by talu_documents_search.
///
/// Parameters:
///   - list: Search result list handle to free (may be null)
pub export fn talu_documents_free_search_results(list: ?*CSearchResultList) callconv(.c) void {
    capi_error.clearError();
    const l = list orelse return;

    if (l._arena) |arena_ptr| {
        const arena: *std.heap.ArenaAllocator = @ptrCast(@alignCast(arena_ptr));
        arena.deinit();
        allocator.destroy(arena);
    }

    allocator.destroy(l);
}

/// Internal: Execute batch search and return JSON result.
fn searchBatchImpl(db_path_slice: []const u8, queries_slice: []const u8) ![]u8 {
    const parsed_queries = try documents.parseBatchQueriesJson(allocator, queries_slice);
    defer documents.freeBatchQueries(allocator, @constCast(parsed_queries));
    const results = try documents.searchDocumentsBatch(allocator, db_path_slice, parsed_queries);
    defer documents.freeBatchSearchResults(allocator, @constCast(results));
    return buildBatchResultJson(results);
}

/// Batch search documents by content (multiple queries in single pass).
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - queries_json: JSON array of queries: [{"id": "q1", "text": "coding", "type": "prompt"}, ...]
///   - queries_len: Length of queries_json in bytes
///   - out_results_json: Output parameter to receive JSON map: {"q1": ["doc-a", "doc-b"], ...}
///   - out_results_len: Output parameter to receive length of results JSON
///
/// Returns: 0 on success, negative error code on failure.
/// On success, caller must free the results with talu_documents_free_json().
pub export fn talu_documents_search_batch(
    db_path: ?[*:0]const u8,
    queries_json: ?[*]const u8,
    queries_len: usize,
    out_results_json: ?*?[*]u8,
    out_results_len: ?*usize,
) callconv(.c) i32 {
    capi_error.clearError();
    const out_json = out_results_json orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_results_json is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out_json.* = null;
    const out_len = out_results_len orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_results_len is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    if (queries_json == null or queries_len == 0) {
        capi_error.setErrorWithCode(.invalid_argument, "queries_json is null or empty", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }
    const json_output = searchBatchImpl(db_path_slice, queries_json.?[0..queries_len]) catch |err| {
        const code: error_codes.ErrorCode = if (err == error.OutOfMemory) .out_of_memory else .storage_error;
        capi_error.setErrorWithCode(code, "Batch search failed: {s}", .{@errorName(err)});
        return @intFromEnum(code);
    };
    out_json.* = json_output.ptr;
    out_len.* = json_output.len;
    return 0;
}

/// Free JSON output from talu_documents_search_batch.
pub export fn talu_documents_free_json(ptr: ?[*]u8, len: usize) callconv(.c) void {
    capi_error.clearError();
    if (ptr) |p| {
        allocator.free(p[0..len]);
    }
}

// =============================================================================
// Document CDC (Change Data Capture) API
// =============================================================================

/// Get document changes since a given sequence number.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - since_seq: Sequence number to start from (0 = all changes)
///   - group_id: Filter by group (may be null for all groups)
///   - limit: Maximum number of changes to return (0 = use default of 100)
///   - out_list: Output parameter to receive change list handle
///
/// Returns: 0 on success, negative error code on failure.
/// On success, caller must free the handle with talu_documents_free_changes().
///
/// Usage example:
///   1. First call: talu_documents_get_changes(db, 0, NULL, 100, &list)
///   2. Process list.items[0..list.count]
///   3. Next call: talu_documents_get_changes(db, list.next_seq, NULL, 100, &list)
///   4. Repeat until list.count == 0
pub export fn talu_documents_get_changes(
    db_path: ?[*:0]const u8,
    since_seq: u64,
    group_id: ?[*:0]const u8,
    limit: u32,
    out_list: ?*?*CChangeList,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_list orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_list is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);

    const effective_limit: usize = if (limit == 0) 100 else @intCast(limit);

    const records = documents.getChanges(
        allocator,
        db_path_slice,
        since_seq,
        optSlice(group_id),
        effective_limit,
    ) catch |err| {
        capi_error.setErrorWithCode(.storage_error, "Failed to get changes: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.ErrorCode.storage_error);
    };
    defer documents.freeChangeRecords(allocator, @constCast(records));

    const list = buildChangeList(records, effective_limit) catch |err| {
        capi_error.setErrorWithCode(.out_of_memory, "Failed to build change list: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };

    out.* = list;
    return 0;
}

/// Free a change list returned by talu_documents_get_changes.
///
/// Parameters:
///   - list: Change list handle to free (may be null)
pub export fn talu_documents_free_changes(list: ?*CChangeList) callconv(.c) void {
    capi_error.clearError();
    const l = list orelse return;

    if (l._arena) |arena_ptr| {
        const arena: *std.heap.ArenaAllocator = @ptrCast(@alignCast(arena_ptr));
        arena.deinit();
        allocator.destroy(arena);
    }

    allocator.destroy(l);
}

// =============================================================================
// Document TTL (Time-To-Live) API
// =============================================================================

/// Set or update the TTL for a document.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - doc_id: Document UUID (null-terminated)
///   - ttl_seconds: Time-to-live in seconds from now. 0 = remove TTL (never expires)
///
/// Returns: 0 on success, storage_error if document not found.
///
/// Example:
///   // Set document to expire in 1 hour
///   talu_documents_set_ttl(db, doc_id, 3600);
///
///   // Remove TTL (document never expires)
///   talu_documents_set_ttl(db, doc_id, 0);
pub export fn talu_documents_set_ttl(
    db_path: ?[*:0]const u8,
    doc_id: ?[*:0]const u8,
    ttl_seconds: u64,
) callconv(.c) i32 {
    capi_error.clearError();
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const doc_id_slice = validateRequiredArg(doc_id, "doc_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);

    documents.setDocumentTTL(allocator, db_path_slice, doc_id_slice, ttl_seconds) catch |err| {
        if (err == error.DocumentNotFound) {
            capi_error.setErrorWithCode(.storage_error, "Document not found", .{});
        } else {
            capi_error.setErrorWithCode(.storage_error, "Failed to set TTL: {s}", .{@errorName(err)});
        }
        return @intFromEnum(error_codes.ErrorCode.storage_error);
    };

    return 0;
}

/// Count expired documents in the database.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - out_count: Output parameter to receive the count of expired documents
///
/// Returns: 0 on success, negative error code on failure.
///
/// Note: This counts documents that have an expires_at > 0 and expires_at < current time.
/// These documents are automatically filtered out of read operations but still exist in storage.
pub export fn talu_documents_count_expired(
    db_path: ?[*:0]const u8,
    out_count: ?*usize,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_count orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_count is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = 0;
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);

    const count = documents.countExpiredDocuments(allocator, db_path_slice) catch |err| {
        capi_error.setErrorWithCode(.storage_error, "Failed to count expired documents: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.ErrorCode.storage_error);
    };

    out.* = count;
    return 0;
}

// =============================================================================
// Delta Versioning Functions
// =============================================================================

/// Create a delta version of an existing document.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - base_doc_id: UUID of the base document to version (null-terminated)
///   - new_doc_id: UUID for the new delta version (null-terminated)
///   - delta_json: JSON content for the delta (null-terminated)
///   - title: Optional new title (null to inherit from base)
///   - tags_text: Optional new tags (null to inherit from base)
///   - marker: Optional new marker (null to inherit from base)
///
/// Returns: 0 on success, negative error code on failure.
/// The delta version will have version_type="delta" and base_doc_id set to the base document.
pub export fn talu_documents_create_delta(
    db_path: ?[*:0]const u8,
    base_doc_id: ?[*:0]const u8,
    new_doc_id: ?[*:0]const u8,
    delta_json: ?[*:0]const u8,
    title: ?[*:0]const u8,
    tags_text: ?[*:0]const u8,
    marker: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const base_id_slice = validateRequiredArg(base_doc_id, "base_doc_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const new_id_slice = validateRequiredArg(new_doc_id, "new_doc_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const delta_json_slice = validateRequiredArg(delta_json, "delta_json") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);

    const title_slice: ?[]const u8 = if (title) |t| std.mem.span(t) else null;
    const tags_slice: ?[]const u8 = if (tags_text) |t| std.mem.span(t) else null;
    const marker_slice: ?[]const u8 = if (marker) |m| std.mem.span(m) else null;

    documents.createDeltaVersion(
        allocator,
        db_path_slice,
        base_id_slice,
        new_id_slice,
        delta_json_slice,
        title_slice,
        tags_slice,
        marker_slice,
    ) catch |err| {
        if (err == error.DocumentNotFound) {
            capi_error.setErrorWithCode(.storage_error, "Base document not found", .{});
        } else {
            capi_error.setErrorWithCode(.storage_error, "Failed to create delta version: {s}", .{@errorName(err)});
        }
        return @intFromEnum(error_codes.ErrorCode.storage_error);
    };

    return 0;
}

/// Delta chain result structure.
pub const CDeltaChain = extern struct {
    items: ?[*]CDocumentRecord,
    count: usize,
    _arena: ?*anyopaque,
};

/// Get the delta chain for a document.
/// Returns documents in order from the requested document back to the base.
/// First element is the requested document, last element is the base (full version).
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - doc_id: UUID of the document to get the chain for (null-terminated)
///   - out_chain: Output parameter to receive the delta chain
///
/// Internal: Get delta chain and build C-ABI result.
fn getDeltaChainImpl(db_path_slice: []const u8, doc_id_slice: []const u8) !*CDeltaChain {
    const arena_ptr = try allocator.create(std.heap.ArenaAllocator);
    arena_ptr.* = std.heap.ArenaAllocator.init(allocator);
    errdefer {
        arena_ptr.deinit();
        allocator.destroy(arena_ptr);
    }
    const chain_records = try documents.getDeltaChain(allocator, db_path_slice, doc_id_slice);
    defer documents.freeDeltaChain(allocator, @constCast(chain_records));
    const c_records = try convertToCRecords(arena_ptr.allocator(), chain_records);
    const result = try allocator.create(CDeltaChain);
    result.* = .{
        .items = if (c_records.len > 0) c_records.ptr else null,
        .count = c_records.len,
        ._arena = @ptrCast(arena_ptr),
    };
    return result;
}

/// Returns: 0 on success, negative error code on failure.
/// Caller must free with talu_documents_free_delta_chain().
pub export fn talu_documents_get_delta_chain(
    db_path: ?[*:0]const u8,
    doc_id: ?[*:0]const u8,
    out_chain: ?*?*CDeltaChain,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_chain orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_chain is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const doc_id_slice = validateRequiredArg(doc_id, "doc_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    out.* = getDeltaChainImpl(db_path_slice, doc_id_slice) catch |err| {
        const code: error_codes.ErrorCode = if (err == error.OutOfMemory) .out_of_memory else .storage_error;
        capi_error.setErrorWithCode(code, "Failed to get delta chain: {s}", .{@errorName(err)});
        return @intFromEnum(code);
    };
    return 0;
}

/// Free a delta chain returned by talu_documents_get_delta_chain.
///
/// Parameters:
///   - chain: Delta chain handle to free (may be null)
pub export fn talu_documents_free_delta_chain(chain: ?*CDeltaChain) callconv(.c) void {
    capi_error.clearError();
    const c = chain orelse return;

    if (c._arena) |arena_ptr| {
        const arena: *std.heap.ArenaAllocator = @ptrCast(@alignCast(arena_ptr));
        arena.deinit();
        allocator.destroy(arena);
    }

    allocator.destroy(c);
}

/// Check if a document is a delta version.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - doc_id: UUID of the document to check (null-terminated)
///   - out_is_delta: Output parameter to receive true if delta, false otherwise
///
/// Returns: 0 on success, negative error code on failure.
pub export fn talu_documents_is_delta(
    db_path: ?[*:0]const u8,
    doc_id: ?[*:0]const u8,
    out_is_delta: ?*bool,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_is_delta orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_is_delta is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = false;
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const doc_id_slice = validateRequiredArg(doc_id, "doc_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);

    const is_delta = documents.isDeltaVersion(allocator, db_path_slice, doc_id_slice) catch |err| {
        if (err == error.DocumentNotFound) {
            capi_error.setErrorWithCode(.storage_error, "Document not found", .{});
        } else {
            capi_error.setErrorWithCode(.storage_error, "Failed to check delta status: {s}", .{@errorName(err)});
        }
        return @intFromEnum(error_codes.ErrorCode.storage_error);
    };

    out.* = is_delta;
    return 0;
}

/// Get the base document ID for a delta version.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - doc_id: UUID of the delta document (null-terminated)
///   - out_base_id: Output buffer to receive the base document ID
///   - out_base_id_len: Size of output buffer
///
/// Returns: 0 on success (base_id written to buffer), 1 if not a delta (buffer empty),
///          negative error code on failure.
pub export fn talu_documents_get_base_id(
    db_path: ?[*:0]const u8,
    doc_id: ?[*:0]const u8,
    out_base_id: ?[*]u8,
    out_base_id_len: usize,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_base_id orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_base_id is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    if (out_base_id_len == 0) {
        capi_error.setErrorWithCode(.invalid_argument, "out_base_id_len is 0", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }
    out[0] = 0; // Clear output
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const doc_id_slice = validateRequiredArg(doc_id, "doc_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);

    const base_id = documents.getBaseDocumentId(allocator, db_path_slice, doc_id_slice) catch |err| {
        capi_error.setErrorWithCode(.storage_error, "Failed to get base document ID: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.ErrorCode.storage_error);
    };

    if (base_id) |id| {
        defer allocator.free(id);
        const copy_len = @min(id.len, out_base_id_len - 1);
        @memcpy(out[0..copy_len], id[0..copy_len]);
        out[copy_len] = 0;
        return 0;
    }

    // Not a delta version
    return 1;
}

// =============================================================================
// Compaction/Garbage Collection Functions
// =============================================================================

/// Compaction statistics structure.
pub const CCompactionStats = extern struct {
    total_documents: usize,
    active_documents: usize,
    expired_documents: usize,
    deleted_documents: usize,
    tombstone_count: usize,
    delta_versions: usize,
    estimated_garbage_bytes: u64,
};

/// Get compaction statistics for the document storage.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - out_stats: Output parameter to receive the statistics
///
/// Returns: 0 on success, negative error code on failure.
pub export fn talu_documents_get_compaction_stats(
    db_path: ?[*:0]const u8,
    out_stats: ?*CCompactionStats,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_stats orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_stats is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);

    const stats = documents.getCompactionStats(allocator, db_path_slice) catch |err| {
        capi_error.setErrorWithCode(.storage_error, "Failed to get compaction stats: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.ErrorCode.storage_error);
    };

    out.* = .{
        .total_documents = stats.total_documents,
        .active_documents = stats.active_documents,
        .expired_documents = stats.expired_documents,
        .deleted_documents = stats.deleted_documents,
        .tombstone_count = stats.tombstone_count,
        .delta_versions = stats.delta_versions,
        .estimated_garbage_bytes = stats.estimated_garbage_bytes,
    };

    return 0;
}

/// Purge expired documents by writing tombstones for them.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - out_count: Output parameter to receive the number of documents purged
///
/// Returns: 0 on success, negative error code on failure.
pub export fn talu_documents_purge_expired(
    db_path: ?[*:0]const u8,
    out_count: ?*usize,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_count orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_count is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = 0;
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);

    const count = documents.purgeExpiredDocuments(allocator, db_path_slice) catch |err| {
        capi_error.setErrorWithCode(.storage_error, "Failed to purge expired documents: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.ErrorCode.storage_error);
    };

    out.* = count;
    return 0;
}

/// Get list of document IDs that are candidates for garbage collection.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - out_ids: Output parameter to receive string list of document IDs
///
/// Returns: 0 on success, negative error code on failure.
/// Caller must free with talu_documents_free_string_list().
pub export fn talu_documents_get_garbage_candidates(
    db_path: ?[*:0]const u8,
    out_ids: ?*?*CStringList,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_ids orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_ids is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);

    const ids = documents.getGarbageCandidates(allocator, db_path_slice) catch |err| {
        capi_error.setErrorWithCode(.storage_error, "Failed to get garbage candidates: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.ErrorCode.storage_error);
    };
    defer documents.freeStringIds(allocator, ids);

    // Convert to CStringList
    const list = buildStringList(ids) catch |err| {
        capi_error.setErrorWithCode(.out_of_memory, "Failed to build string list: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };

    out.* = list;
    return 0;
}

/// Build JSON map from batch search results: {"q1": ["doc-a"], "q2": ["doc-b", "doc-c"]}
fn buildBatchResultJson(results: []const documents.BatchSearchResult) ![]u8 {
    var buffer = std.ArrayList(u8).empty;
    errdefer buffer.deinit(allocator);

    try buffer.append(allocator, '{');

    for (results, 0..) |result, ri| {
        if (ri > 0) try buffer.append(allocator, ',');

        // Write key (query_id)
        try buffer.append(allocator, '"');
        try buffer.appendSlice(allocator, result.query_id);
        try buffer.appendSlice(allocator, "\":[");

        // Write value (doc_ids array)
        for (result.doc_ids, 0..) |doc_id, di| {
            if (di > 0) try buffer.append(allocator, ',');
            try buffer.append(allocator, '"');
            try buffer.appendSlice(allocator, doc_id);
            try buffer.append(allocator, '"');
        }
        try buffer.append(allocator, ']');
    }

    try buffer.append(allocator, '}');
    return buffer.toOwnedSlice(allocator);
}

// =============================================================================
// Document-Tag Junction API
// =============================================================================

/// Add a tag to a document.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - doc_id: Document UUID (null-terminated)
///   - tag_id: Tag UUID to add (null-terminated)
///   - group_id: Group ID for isolation (null = no group)
///
/// Returns: 0 on success, negative error code on failure.
pub export fn talu_documents_add_tag(
    db_path: ?[*:0]const u8,
    doc_id: ?[*:0]const u8,
    tag_id: ?[*:0]const u8,
    group_id: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const doc_id_slice = validateRequiredArg(doc_id, "doc_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const tag_id_slice = validateRequiredArg(tag_id, "tag_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);

    document_tags.addDocumentTag(allocator, db_path_slice, doc_id_slice, tag_id_slice, optSlice(group_id)) catch |err| {
        capi_error.setErrorWithCode(.storage_error, "Failed to add tag: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.ErrorCode.storage_error);
    };

    return 0;
}

/// Remove a tag from a document.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - doc_id: Document UUID (null-terminated)
///   - tag_id: Tag UUID to remove (null-terminated)
///   - group_id: Group ID for isolation (null = no group)
///
/// Returns: 0 on success, negative error code on failure.
pub export fn talu_documents_remove_tag(
    db_path: ?[*:0]const u8,
    doc_id: ?[*:0]const u8,
    tag_id: ?[*:0]const u8,
    group_id: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const doc_id_slice = validateRequiredArg(doc_id, "doc_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const tag_id_slice = validateRequiredArg(tag_id, "tag_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);

    document_tags.removeDocumentTag(allocator, db_path_slice, doc_id_slice, tag_id_slice, optSlice(group_id)) catch |err| {
        capi_error.setErrorWithCode(.storage_error, "Failed to remove tag: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.ErrorCode.storage_error);
    };

    return 0;
}

/// Get all tag IDs for a document.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - doc_id: Document UUID (null-terminated)
///   - out_tag_ids: Output parameter to receive string list of tag IDs
///
/// Returns: 0 on success, negative error code on failure.
/// On success, caller must free the handle with talu_documents_free_string_list().
pub export fn talu_documents_get_tags(
    db_path: ?[*:0]const u8,
    doc_id: ?[*:0]const u8,
    out_tag_ids: ?*?*CStringList,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_tag_ids orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_tag_ids is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const doc_id_slice = validateRequiredArg(doc_id, "doc_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);

    const tag_ids = document_tags.getDocumentTagIds(allocator, db_path_slice, doc_id_slice) catch |err| {
        capi_error.setErrorWithCode(.storage_error, "Failed to get document tags: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.ErrorCode.storage_error);
    };
    defer document_tags.freeStringSlice(allocator, @constCast(tag_ids));

    const list = buildStringList(tag_ids) catch |err| {
        capi_error.setErrorWithCode(.out_of_memory, "Failed to build string list: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };

    out.* = list;
    return 0;
}

/// Get all document IDs that have a specific tag.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - tag_id: Tag UUID to query (null-terminated)
///   - out_doc_ids: Output parameter to receive string list of document IDs
///
/// Returns: 0 on success, negative error code on failure.
/// On success, caller must free the handle with talu_documents_free_string_list().
pub export fn talu_documents_get_by_tag(
    db_path: ?[*:0]const u8,
    tag_id: ?[*:0]const u8,
    out_doc_ids: ?*?*CStringList,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_doc_ids orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_doc_ids is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const tag_id_slice = validateRequiredArg(tag_id, "tag_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);

    const doc_ids = document_tags.getTagDocumentIds(allocator, db_path_slice, tag_id_slice) catch |err| {
        capi_error.setErrorWithCode(.storage_error, "Failed to get documents by tag: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.ErrorCode.storage_error);
    };
    defer document_tags.freeStringSlice(allocator, @constCast(doc_ids));

    const list = buildStringList(doc_ids) catch |err| {
        capi_error.setErrorWithCode(.out_of_memory, "Failed to build string list: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };

    out.* = list;
    return 0;
}

/// Free a string list returned by talu_documents_get_tags or talu_documents_get_by_tag.
///
/// Parameters:
///   - list: String list handle to free (may be null)
pub export fn talu_documents_free_string_list(list: ?*CStringList) callconv(.c) void {
    capi_error.clearError();
    const l = list orelse return;

    if (l._arena) |arena_ptr| {
        const arena: *std.heap.ArenaAllocator = @ptrCast(@alignCast(arena_ptr));
        arena.deinit();
        allocator.destroy(arena);
    }

    allocator.destroy(l);
}

// =============================================================================
// Internal Helpers
// =============================================================================

/// Copy DocumentRecord fields to CDocumentRecord using thread-local static buffers.
fn populateDocumentRecord(out: *CDocumentRecord, record: documents.DocumentRecord) void {
    const S = struct {
        threadlocal var doc_id_buf: [256]u8 = .{0} ** 256;
        threadlocal var doc_type_buf: [64]u8 = .{0} ** 64;
        threadlocal var title_buf: [512]u8 = .{0} ** 512;
        threadlocal var tags_text_buf: [1024]u8 = .{0} ** 1024;
        threadlocal var doc_json_buf: [65536]u8 = .{0} ** 65536;
        threadlocal var parent_id_buf: [256]u8 = .{0} ** 256;
        threadlocal var marker_buf: [32]u8 = .{0} ** 32;
        threadlocal var group_id_buf: [256]u8 = .{0} ** 256;
        threadlocal var owner_id_buf: [256]u8 = .{0} ** 256;
    };
    out.doc_id = copyToStaticBuf(&S.doc_id_buf, record.doc_id);
    out.doc_type = copyToStaticBuf(&S.doc_type_buf, record.doc_type);
    out.title = copyToStaticBuf(&S.title_buf, record.title);
    out.tags_text = copyToStaticBufOpt(&S.tags_text_buf, record.tags_text);
    out.doc_json = copyToStaticBuf(&S.doc_json_buf, record.doc_json);
    out.parent_id = copyToStaticBufOpt(&S.parent_id_buf, record.parent_id);
    out.marker = copyToStaticBufOpt(&S.marker_buf, record.marker);
    out.group_id = copyToStaticBufOpt(&S.group_id_buf, record.group_id);
    out.owner_id = copyToStaticBufOpt(&S.owner_id_buf, record.owner_id);
    out.created_at_ms = record.created_at_ms;
    out.updated_at_ms = record.updated_at_ms;
    out.expires_at_ms = record.expires_at_ms;
    out.content_hash = record.content_hash;
    out.seq_num = record.seq_num;
}

fn copyToStaticBuf(buf: []u8, src: []const u8) ?[*:0]const u8 {
    if (src.len == 0) return null;
    const len = @min(src.len, buf.len - 1);
    @memcpy(buf[0..len], src[0..len]);
    buf[len] = 0;
    return @ptrCast(buf.ptr);
}

fn copyToStaticBufOpt(buf: []u8, src: ?[]const u8) ?[*:0]const u8 {
    const s = src orelse return null;
    return copyToStaticBuf(buf, s);
}

/// Copy string to arena-allocated buffer with null terminator.
fn copyToBuf(arena: std.mem.Allocator, src: []const u8) ?[*:0]const u8 {
    if (src.len == 0) return null;
    const buf = arena.allocSentinel(u8, src.len, 0) catch return null;
    @memcpy(buf, src);
    return buf.ptr;
}

/// Copy optional string to arena-allocated buffer with null terminator.
fn copyToBufOpt(arena: std.mem.Allocator, src: ?[]const u8) ?[*:0]const u8 {
    const s = src orelse return null;
    return copyToBuf(arena, s);
}

/// Convert DocumentRecord slice to CDocumentRecord slice using arena allocator.
fn convertToCRecords(arena: std.mem.Allocator, chain_records: []const documents.DocumentRecord) ![]CDocumentRecord {
    const c_records = try arena.alloc(CDocumentRecord, chain_records.len);
    for (chain_records, 0..) |rec, i| {
        c_records[i] = .{
            .doc_id = copyToBuf(arena, rec.doc_id),
            .doc_type = copyToBuf(arena, rec.doc_type),
            .title = copyToBuf(arena, rec.title),
            .tags_text = copyToBufOpt(arena, rec.tags_text),
            .doc_json = copyToBuf(arena, rec.doc_json),
            .parent_id = copyToBufOpt(arena, rec.parent_id),
            .marker = copyToBufOpt(arena, rec.marker),
            .group_id = copyToBufOpt(arena, rec.group_id),
            .owner_id = copyToBufOpt(arena, rec.owner_id),
            .created_at_ms = rec.created_at_ms,
            .updated_at_ms = rec.updated_at_ms,
            .expires_at_ms = rec.expires_at_ms,
            .content_hash = rec.content_hash,
            .seq_num = rec.seq_num,
        };
    }
    return c_records;
}
