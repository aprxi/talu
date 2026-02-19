//! TaluDB adapter for document entity persistence.
//!
//! Documents are the universal storage layer for prompts, personas, RAG sources,
//! tools, and any user-defined content. The hybrid columnar schema enables:
//! - SIMD-speed filtering on scalar columns (type, group, owner, marker)
//! - Zero-copy title/tag extraction from kvbuf headers
//! - Flexible JSON payloads for arbitrary user content
//!
//! Schema 11: Documents table
//! Schema 12: Document Deletes (tombstones)

const std = @import("std");
const kvbuf = @import("../../io/kvbuf/root.zig");
const db_writer = @import("../writer.zig");
const db_reader = @import("../reader.zig");
const block_reader = @import("../block_reader.zig");
const db_blob_store = @import("../blob/store.zig");
const types = @import("../types.zig");
const parallel = @import("../../compute/parallel.zig");

const Allocator = std.mem.Allocator;
const ColumnValue = db_writer.ColumnValue;
const DocumentFieldIds = kvbuf.DocumentFieldIds;

const schema_documents: u16 = 11;
const schema_document_deletes: u16 = 12;
const default_doc_json_externalize_threshold_bytes: usize = 1 * 1024 * 1024;
const env_doc_json_externalize_threshold_bytes = "TALU_DB_DOC_JSON_EXTERNALIZE_THRESHOLD_BYTES";
const env_shared_externalize_threshold_bytes = "TALU_DB_EXTERNALIZE_THRESHOLD_BYTES";
const doc_json_trigram_bloom_bytes: usize = 64;

// Column IDs for Documents table (Schema 11)
const col_doc_hash: u32 = 1;
const col_ts: u32 = 2;
const col_group_hash: u32 = 3;
const col_type_hash: u32 = 4;
const col_parent_hash: u32 = 5;
const col_marker_hash: u32 = 6;
const col_content_hash: u32 = 7;
const col_owner_hash: u32 = 8;
const col_expires_at: u32 = 9;
const col_meta_i1: u32 = 10;
const col_meta_i2: u32 = 11;
const col_meta_i3: u32 = 12;
const col_meta_i4: u32 = 13;
const col_meta_i5: u32 = 14;
const col_meta_f1: u32 = 15;
const col_meta_f2: u32 = 16;
const col_meta_f3: u32 = 17;
const col_meta_f4: u32 = 18;
const col_meta_f5: u32 = 19;
const col_payload: u32 = 20;
const col_seq_num: u32 = 21;

/// Document entity record for storage/retrieval.
pub const DocumentRecord = struct {
    doc_id: []const u8,
    doc_type: []const u8, // "prompt" | "persona" | "rag" | "tool" | "folder"
    title: []const u8,
    tags_text: ?[]const u8 = null, // Space-separated normalized tags
    doc_json: []const u8, // Full JSON payload (envelope: {_sys, data})
    parent_id: ?[]const u8 = null, // For versioning/hierarchy
    marker: ?[]const u8 = null, // "active" | "archived" | "deleted"
    group_id: ?[]const u8 = null,
    owner_id: ?[]const u8 = null,
    created_at_ms: i64,
    updated_at_ms: i64,
    expires_at_ms: i64 = 0, // 0 = never expires
    content_hash: u64 = 0, // SipHash of doc_json (computed on write)
    seq_num: u64 = 0, // CDC sequence number

    // Sparse columns for user-defined metrics
    meta_i1: ?i64 = null,
    meta_i2: ?i64 = null,
    meta_i3: ?i64 = null,
    meta_i4: ?i64 = null,
    meta_i5: ?i64 = null,
    meta_f1: ?f64 = null,
    meta_f2: ?f64 = null,
    meta_f3: ?f64 = null,
    meta_f4: ?f64 = null,
    meta_f5: ?f64 = null,

    // Delta versioning (optional)
    version_type: ?[]const u8 = null, // "full" | "delta"
    base_doc_id: ?[]const u8 = null, // Reference to base version

    pub fn deinit(self: *DocumentRecord, allocator: Allocator) void {
        allocator.free(self.doc_id);
        allocator.free(self.doc_type);
        allocator.free(self.title);
        if (self.tags_text) |t| allocator.free(t);
        allocator.free(self.doc_json);
        if (self.parent_id) |p| allocator.free(p);
        if (self.marker) |m| allocator.free(m);
        if (self.group_id) |g| allocator.free(g);
        if (self.owner_id) |o| allocator.free(o);
        if (self.version_type) |v| allocator.free(v);
        if (self.base_doc_id) |b| allocator.free(b);
    }
};

/// Lightweight document metadata without loading externalized doc_json content.
pub const DocumentHeader = struct {
    doc_id: []const u8,
    doc_type: []const u8,
    title: []const u8,
    tags_text: ?[]const u8 = null,
    parent_id: ?[]const u8 = null,
    marker: ?[]const u8 = null,
    group_id: ?[]const u8 = null,
    owner_id: ?[]const u8 = null,
    created_at_ms: i64,
    updated_at_ms: i64,
    expires_at_ms: i64 = 0,
    version_type: ?[]const u8 = null,
    base_doc_id: ?[]const u8 = null,
    doc_json_ref: ?[]const u8 = null,
    has_inline_doc_json: bool = false,

    pub fn deinit(self: *DocumentHeader, allocator: Allocator) void {
        allocator.free(self.doc_id);
        allocator.free(self.doc_type);
        allocator.free(self.title);
        if (self.tags_text) |t| allocator.free(t);
        if (self.parent_id) |p| allocator.free(p);
        if (self.marker) |m| allocator.free(m);
        if (self.group_id) |g| allocator.free(g);
        if (self.owner_id) |o| allocator.free(o);
        if (self.version_type) |v| allocator.free(v);
        if (self.base_doc_id) |b| allocator.free(b);
        if (self.doc_json_ref) |r| allocator.free(r);
    }
};

/// Search result containing document and matching snippet.
pub const SearchResult = struct {
    doc_id: []const u8,
    doc_type: []const u8,
    title: []const u8,
    snippet: []const u8,

    pub fn deinit(self: *SearchResult, allocator: Allocator) void {
        allocator.free(self.doc_id);
        allocator.free(self.doc_type);
        allocator.free(self.title);
        allocator.free(self.snippet);
    }
};

/// A single query in a batch search request.
pub const BatchQuery = struct {
    id: []const u8, // Query identifier for result mapping
    text: []const u8, // Search text
    doc_type: ?[]const u8 = null, // Optional type filter
};

/// Result of a batch search: maps query_id → [doc_ids].
pub const BatchSearchResult = struct {
    query_id: []const u8,
    doc_ids: [][]const u8,

    pub fn deinit(self: *BatchSearchResult, allocator: Allocator) void {
        allocator.free(self.query_id);
        for (self.doc_ids) |doc_id| allocator.free(doc_id);
        allocator.free(self.doc_ids);
    }
};

/// Change action type for CDC.
pub const ChangeAction = enum(u8) {
    create = 1,
    update = 2,
    delete = 3,
};

/// Change record for CDC feed.
/// Represents a single change event with sequence number for ordering.
pub const ChangeRecord = struct {
    seq_num: u64,
    doc_id: []const u8,
    action: ChangeAction,
    timestamp_ms: i64,
    doc_type: ?[]const u8 = null, // Available for create/update, null for delete
    title: ?[]const u8 = null, // Available for create/update, null for delete

    pub fn deinit(self: *ChangeRecord, allocator: Allocator) void {
        allocator.free(self.doc_id);
        if (self.doc_type) |t| allocator.free(t);
        if (self.title) |t| allocator.free(t);
    }
};

/// DocumentAdapter - TaluDB adapter for document persistence.
///
/// Provides CRUD operations for documents.
/// Thread safety: NOT thread-safe (single-writer semantics via lock).
pub const DocumentAdapter = struct {
    allocator: Allocator,
    fs_writer: *db_writer.Writer,
    fs_reader: *db_reader.Reader,
    blob_store: db_blob_store.BlobStore,
    read_only: bool,
    doc_json_externalize_threshold_bytes: usize,

    /// Initialize a TaluDB-backed document adapter with write capabilities.
    pub fn init(allocator: Allocator, db_root: []const u8) !DocumentAdapter {
        var writer_ptr = try allocator.create(db_writer.Writer);
        errdefer allocator.destroy(writer_ptr);
        writer_ptr.* = try db_writer.Writer.open(allocator, db_root, "docs");
        errdefer writer_ptr.deinit();

        var reader_ptr = try allocator.create(db_reader.Reader);
        errdefer allocator.destroy(reader_ptr);
        reader_ptr.* = try db_reader.Reader.open(allocator, db_root, "docs");
        errdefer reader_ptr.deinit();

        var blob_store = try db_blob_store.BlobStore.init(allocator, db_root);
        errdefer blob_store.deinit();

        return .{
            .allocator = allocator,
            .fs_writer = writer_ptr,
            .fs_reader = reader_ptr,
            .blob_store = blob_store,
            .read_only = false,
            .doc_json_externalize_threshold_bytes = resolveDocJsonExternalizeThresholdBytes(),
        };
    }

    /// Initialize a read-only adapter for scanning documents.
    pub fn initReadOnly(allocator: Allocator, db_root: []const u8) !DocumentAdapter {
        const reader_ptr = try allocator.create(db_reader.Reader);
        errdefer allocator.destroy(reader_ptr);
        reader_ptr.* = try db_reader.Reader.open(allocator, db_root, "docs");

        var blob_store = try db_blob_store.BlobStore.init(allocator, db_root);
        errdefer blob_store.deinit();

        return .{
            .allocator = allocator,
            .fs_writer = undefined,
            .fs_reader = reader_ptr,
            .blob_store = blob_store,
            .read_only = true,
            .doc_json_externalize_threshold_bytes = resolveDocJsonExternalizeThresholdBytes(),
        };
    }

    pub fn deinit(self: *DocumentAdapter) void {
        if (!self.read_only) {
            self.fs_writer.flushBlock() catch {};
            self.fs_writer.deinit();
            self.allocator.destroy(self.fs_writer);
        }
        self.fs_reader.deinit();
        self.allocator.destroy(self.fs_reader);
        self.blob_store.deinit();
    }

    pub fn deinitReadOnly(self: *DocumentAdapter) void {
        self.fs_reader.deinit();
        self.allocator.destroy(self.fs_reader);
        self.blob_store.deinit();
    }

    // =========================================================================
    // Document CRUD Operations
    // =========================================================================

    /// Write (create or update) a document record.
    pub fn writeDocument(self: *DocumentAdapter, record: DocumentRecord) !void {
        var doc_json_inline: ?[]const u8 = record.doc_json;
        var doc_json_ref: ?[]const u8 = null;
        var doc_json_ref_buf: [db_blob_store.ref_len]u8 = undefined; // Written by @memcpy below when externalized
        var doc_json_ref_len: usize = 0;
        var doc_json_trigram_bloom: ?[]const u8 = null;
        var doc_json_trigram_bloom_buf: [doc_json_trigram_bloom_bytes]u8 = undefined; // Written by computeTrigramBloom below when externalized

        if (record.doc_json.len > self.doc_json_externalize_threshold_bytes) {
            const blob_ref = try self.blob_store.putAuto(record.doc_json);
            const ref_slice = blob_ref.refSlice();
            @memcpy(doc_json_ref_buf[0..ref_slice.len], ref_slice);
            doc_json_ref_len = ref_slice.len;
            doc_json_inline = null;
            doc_json_ref = doc_json_ref_buf[0..doc_json_ref_len];
            doc_json_trigram_bloom_buf = buildDocJsonTrigramBloom(record.doc_json);
            doc_json_trigram_bloom = doc_json_trigram_bloom_buf[0..];
        }

        const payload = try encodeDocumentRecordKvBufWithStorage(
            self.allocator,
            record,
            doc_json_inline,
            doc_json_ref,
            doc_json_trigram_bloom,
        );
        defer self.allocator.free(payload);

        // Compute hashes for scalar columns
        const doc_hash = computeHash(record.doc_id);
        const group_hash = computeOptionalHash(record.group_id);
        const type_hash = computeHash(record.doc_type);
        const parent_hash = computeOptionalHash(record.parent_id);
        const marker_hash = computeOptionalHash(record.marker);
        const content_hash = if (record.content_hash != 0) record.content_hash else computeHash(record.doc_json);
        const owner_hash = computeOptionalHash(record.owner_id);

        // Prepare scalar column values
        var doc_hash_value = doc_hash;
        var ts_value = record.updated_at_ms;
        var group_hash_value = group_hash;
        var type_hash_value = type_hash;
        var parent_hash_value = parent_hash;
        var marker_hash_value = marker_hash;
        var content_hash_value = content_hash;
        var owner_hash_value = owner_hash;
        var expires_at_value = record.expires_at_ms;
        var seq_num_value = record.seq_num;

        // Sparse int columns
        var meta_i1_value: i64 = record.meta_i1 orelse 0;
        var meta_i2_value: i64 = record.meta_i2 orelse 0;
        var meta_i3_value: i64 = record.meta_i3 orelse 0;
        var meta_i4_value: i64 = record.meta_i4 orelse 0;
        var meta_i5_value: i64 = record.meta_i5 orelse 0;

        // Sparse float columns
        var meta_f1_value: f64 = record.meta_f1 orelse 0.0;
        var meta_f2_value: f64 = record.meta_f2 orelse 0.0;
        var meta_f3_value: f64 = record.meta_f3 orelse 0.0;
        var meta_f4_value: f64 = record.meta_f4 orelse 0.0;
        var meta_f5_value: f64 = record.meta_f5 orelse 0.0;

        const columns = [_]ColumnValue{
            .{ .column_id = col_doc_hash, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&doc_hash_value) },
            .{ .column_id = col_ts, .shape = .SCALAR, .phys_type = .I64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&ts_value) },
            .{ .column_id = col_group_hash, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&group_hash_value) },
            .{ .column_id = col_type_hash, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&type_hash_value) },
            .{ .column_id = col_parent_hash, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&parent_hash_value) },
            .{ .column_id = col_marker_hash, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&marker_hash_value) },
            .{ .column_id = col_content_hash, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&content_hash_value) },
            .{ .column_id = col_owner_hash, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&owner_hash_value) },
            .{ .column_id = col_expires_at, .shape = .SCALAR, .phys_type = .I64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&expires_at_value) },
            .{ .column_id = col_meta_i1, .shape = .SCALAR, .phys_type = .I64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&meta_i1_value) },
            .{ .column_id = col_meta_i2, .shape = .SCALAR, .phys_type = .I64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&meta_i2_value) },
            .{ .column_id = col_meta_i3, .shape = .SCALAR, .phys_type = .I64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&meta_i3_value) },
            .{ .column_id = col_meta_i4, .shape = .SCALAR, .phys_type = .I64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&meta_i4_value) },
            .{ .column_id = col_meta_i5, .shape = .SCALAR, .phys_type = .I64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&meta_i5_value) },
            .{ .column_id = col_meta_f1, .shape = .SCALAR, .phys_type = .F64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&meta_f1_value) },
            .{ .column_id = col_meta_f2, .shape = .SCALAR, .phys_type = .F64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&meta_f2_value) },
            .{ .column_id = col_meta_f3, .shape = .SCALAR, .phys_type = .F64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&meta_f3_value) },
            .{ .column_id = col_meta_f4, .shape = .SCALAR, .phys_type = .F64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&meta_f4_value) },
            .{ .column_id = col_meta_f5, .shape = .SCALAR, .phys_type = .F64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&meta_f5_value) },
            .{ .column_id = col_payload, .shape = .VARBYTES, .phys_type = .BINARY, .encoding = .RAW, .dims = 0, .data = payload },
            .{ .column_id = col_seq_num, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&seq_num_value) },
        };

        try self.fs_writer.appendRow(schema_documents, &columns);
    }

    /// Write a document deletion marker (tombstone).
    pub fn deleteDocument(self: *DocumentAdapter, doc_id: []const u8, deleted_at_ms: i64) !void {
        const doc_hash = computeHash(doc_id);
        var doc_hash_value = doc_hash;
        var ts_value = deleted_at_ms;

        const columns = [_]ColumnValue{
            .{ .column_id = col_doc_hash, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&doc_hash_value) },
            .{ .column_id = col_ts, .shape = .SCALAR, .phys_type = .I64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&ts_value) },
        };

        try self.fs_writer.appendRow(schema_document_deletes, &columns);
    }

    /// Scan all documents with optional filters.
    /// Expired documents (expires_at_ms > 0 and < current time) are automatically filtered out.
    pub fn scanDocuments(
        self: *DocumentAdapter,
        allocator: Allocator,
        doc_type: ?[]const u8,
        group_id: ?[]const u8,
        owner_id: ?[]const u8,
        marker: ?[]const u8,
    ) ![]DocumentRecord {
        const target_type_hash: ?u64 = if (doc_type) |t| computeHash(t) else null;
        const target_group_hash: ?u64 = if (group_id) |g| computeHash(g) else null;
        const target_owner_hash: ?u64 = if (owner_id) |o| computeHash(o) else null;
        const target_marker_hash: ?u64 = if (marker) |m| computeHash(m) else null;
        const now_ms = std.time.milliTimestamp();

        // Collect deleted document hashes
        var deleted = std.AutoHashMap(u64, i64).init(allocator);
        defer deleted.deinit();
        try self.collectDeletedDocuments(allocator, &deleted);

        // Track latest version of each document by hash
        var latest = std.AutoHashMap(u64, DocumentRecord).init(allocator);
        defer {
            var it = latest.valueIterator();
            while (it.next()) |v| {
                var doc = v.*;
                doc.deinit(allocator);
            }
            latest.deinit();
        }

        const blocks = try self.fs_reader.getBlocks(allocator);
        defer allocator.free(blocks);

        for (blocks) |block| {
            var file = self.fs_reader.dir.openFile(block.path, .{ .mode = .read_only }) catch continue;
            defer file.close();

            const reader = block_reader.BlockReader.init(file, allocator);
            const header = reader.readHeader(block.offset) catch continue;
            if (header.schema_id != schema_documents) continue;

            const descs = reader.readColumnDirectory(header, block.offset) catch continue;
            defer allocator.free(descs);

            const row_count = header.row_count;
            if (row_count == 0) continue;

            const hash_desc = findColumn(descs, col_doc_hash) orelse continue;
            const ts_desc = findColumn(descs, col_ts) orelse continue;
            const type_desc = findColumn(descs, col_type_hash);
            const group_desc = findColumn(descs, col_group_hash);
            const owner_desc = findColumn(descs, col_owner_hash);
            const marker_desc = findColumn(descs, col_marker_hash);
            const expires_desc = findColumn(descs, col_expires_at);
            const payload_desc = findColumn(descs, col_payload) orelse continue;

            const hash_bytes = reader.readColumnData(block.offset, hash_desc, allocator) catch continue;
            defer allocator.free(hash_bytes);

            const ts_bytes = reader.readColumnData(block.offset, ts_desc, allocator) catch continue;
            defer allocator.free(ts_bytes);

            var type_bytes: ?[]const u8 = null;
            defer if (type_bytes) |tb| allocator.free(tb);
            if (type_desc) |td| {
                type_bytes = reader.readColumnData(block.offset, td, allocator) catch null;
            }

            var group_bytes: ?[]const u8 = null;
            defer if (group_bytes) |gb| allocator.free(gb);
            if (group_desc) |gd| {
                group_bytes = reader.readColumnData(block.offset, gd, allocator) catch null;
            }

            var owner_bytes: ?[]const u8 = null;
            defer if (owner_bytes) |ob| allocator.free(ob);
            if (owner_desc) |od| {
                owner_bytes = reader.readColumnData(block.offset, od, allocator) catch null;
            }

            var marker_bytes: ?[]const u8 = null;
            defer if (marker_bytes) |mb| allocator.free(mb);
            if (marker_desc) |md| {
                marker_bytes = reader.readColumnData(block.offset, md, allocator) catch null;
            }

            var expires_bytes: ?[]const u8 = null;
            defer if (expires_bytes) |eb| allocator.free(eb);
            if (expires_desc) |ed| {
                expires_bytes = reader.readColumnData(block.offset, ed, allocator) catch null;
            }

            var payload_buffers = readVarBytesBuffers(file, block.offset, payload_desc, row_count, allocator) catch continue;
            defer payload_buffers.deinit(allocator);

            for (0..row_count) |row_idx| {
                const doc_hash = readU64At(hash_bytes, row_idx) catch continue;
                const ts = readI64At(ts_bytes, row_idx) catch continue;

                // Skip if deleted
                if (deleted.get(doc_hash)) |del_ts| {
                    if (del_ts >= ts) continue;
                }

                // Skip if expired (expires_at > 0 means TTL is set)
                if (expires_bytes) |eb| {
                    const expires_at = readI64At(eb, row_idx) catch 0;
                    if (expires_at > 0 and expires_at < now_ms) continue;
                }

                // Filter by type if specified
                if (target_type_hash) |target| {
                    if (type_bytes) |tb| {
                        const row_type = readU64At(tb, row_idx) catch continue;
                        if (row_type != target) continue;
                    }
                }

                // Filter by group if specified
                if (target_group_hash) |target| {
                    if (group_bytes) |gb| {
                        const row_group = readU64At(gb, row_idx) catch continue;
                        if (row_group != target and row_group != 0) continue;
                    }
                }

                // Filter by owner if specified
                if (target_owner_hash) |target| {
                    if (owner_bytes) |ob| {
                        const row_owner = readU64At(ob, row_idx) catch continue;
                        if (row_owner != target and row_owner != 0) continue;
                    }
                }

                // Filter by marker if specified
                if (target_marker_hash) |target| {
                    if (marker_bytes) |mb| {
                        const row_marker = readU64At(mb, row_idx) catch continue;
                        if (row_marker != target) continue;
                    }
                }

                const payload = payload_buffers.sliceForRow(row_idx) catch continue;
                const record_opt = decodeDocumentRecordWithBlobStore(allocator, payload, &self.blob_store) catch continue;
                if (record_opt) |record| {
                    // Check if we already have a newer version
                    if (latest.get(doc_hash)) |existing| {
                        if (existing.updated_at_ms >= record.updated_at_ms) {
                            var r = record;
                            r.deinit(allocator);
                            continue;
                        }
                        var old = existing;
                        old.deinit(allocator);
                    }
                    latest.put(doc_hash, record) catch {
                        var r = record;
                        r.deinit(allocator);
                        continue;
                    };
                }
            }
        }

        // Convert to slice
        var results = std.ArrayList(DocumentRecord).empty;
        errdefer {
            for (results.items) |*r| r.deinit(allocator);
            results.deinit(allocator);
        }

        var it = latest.valueIterator();
        while (it.next()) |v| {
            try results.append(allocator, v.*);
        }
        latest.clearRetainingCapacity();

        return results.toOwnedSlice(allocator);
    }

    /// Get a single document by ID.
    /// Returns null if document is not found or has expired.
    ///
    /// Uses reverse scanning (newest block first, last row first) so that the
    /// first matching row is the latest version.  This makes point lookups
    /// O(1) in the common case (document was recently written/updated).
    pub fn getDocument(self: *DocumentAdapter, allocator: Allocator, doc_id: []const u8) !?DocumentRecord {
        const target_hash = computeHash(doc_id);
        const now_ms = std.time.milliTimestamp();

        // Check if deleted
        var deleted = std.AutoHashMap(u64, i64).init(allocator);
        defer deleted.deinit();
        try self.collectDeletedDocuments(allocator, &deleted);

        const blocks = try self.fs_reader.getBlocks(allocator);
        defer allocator.free(blocks);

        // Reverse-scan: newest blocks first (current.talu blocks are at the
        // end of the slice, sealed segments at the front).
        var block_idx = blocks.len;
        while (block_idx > 0) {
            block_idx -= 1;
            const block = blocks[block_idx];

            var file = self.fs_reader.dir.openFile(block.path, .{ .mode = .read_only }) catch continue;
            defer file.close();

            const reader = block_reader.BlockReader.init(file, allocator);
            const header = reader.readHeader(block.offset) catch continue;
            if (header.schema_id != schema_documents) continue;

            const descs = reader.readColumnDirectory(header, block.offset) catch continue;
            defer allocator.free(descs);

            const row_count = header.row_count;
            if (row_count == 0) continue;

            const hash_desc = findColumn(descs, col_doc_hash) orelse continue;
            const ts_desc = findColumn(descs, col_ts) orelse continue;
            const expires_desc = findColumn(descs, col_expires_at);
            const payload_desc = findColumn(descs, col_payload) orelse continue;

            const hash_bytes = reader.readColumnData(block.offset, hash_desc, allocator) catch continue;
            defer allocator.free(hash_bytes);

            const ts_bytes = reader.readColumnData(block.offset, ts_desc, allocator) catch continue;
            defer allocator.free(ts_bytes);

            var expires_bytes: ?[]const u8 = null;
            defer if (expires_bytes) |eb| allocator.free(eb);
            if (expires_desc) |ed| {
                expires_bytes = reader.readColumnData(block.offset, ed, allocator) catch null;
            }

            var payload_buffers = readVarBytesBuffers(file, block.offset, payload_desc, row_count, allocator) catch continue;
            defer payload_buffers.deinit(allocator);

            // Reverse-scan rows within block (last written = highest index).
            var row_idx = row_count;
            while (row_idx > 0) {
                row_idx -= 1;

                const doc_hash = readU64At(hash_bytes, row_idx) catch continue;
                if (doc_hash != target_hash) continue;

                const ts = readI64At(ts_bytes, row_idx) catch continue;

                // Skip if deleted
                if (deleted.get(doc_hash)) |del_ts| {
                    if (del_ts >= ts) continue;
                }

                // First non-deleted match in reverse order is the latest version.
                var expires_at: i64 = 0;
                if (expires_bytes) |eb| {
                    expires_at = readI64At(eb, row_idx) catch 0;
                }

                const payload = payload_buffers.sliceForRow(row_idx) catch continue;
                const record_opt = decodeDocumentRecordWithBlobStore(allocator, payload, &self.blob_store) catch continue;
                if (record_opt) |record| {
                    // Check expiration before returning.
                    if (expires_at > 0 and expires_at < now_ms) {
                        var r = record;
                        r.deinit(allocator);
                        return null;
                    }
                    return record;
                }
            }
        }

        return null;
    }

    /// Get a single document header by ID without loading externalized doc_json.
    /// Returns null if document is not found or has expired.
    ///
    /// Uses reverse scanning (newest block first, last row first) for O(1)
    /// best-case point lookups.
    pub fn getDocumentHeader(self: *DocumentAdapter, allocator: Allocator, doc_id: []const u8) !?DocumentHeader {
        const target_hash = computeHash(doc_id);
        const now_ms = std.time.milliTimestamp();

        var deleted = std.AutoHashMap(u64, i64).init(allocator);
        defer deleted.deinit();
        try self.collectDeletedDocuments(allocator, &deleted);

        const blocks = try self.fs_reader.getBlocks(allocator);
        defer allocator.free(blocks);

        // Reverse-scan: newest blocks first.
        var block_idx = blocks.len;
        while (block_idx > 0) {
            block_idx -= 1;
            const block = blocks[block_idx];

            var file = self.fs_reader.dir.openFile(block.path, .{ .mode = .read_only }) catch continue;
            defer file.close();

            const reader = block_reader.BlockReader.init(file, allocator);
            const header = reader.readHeader(block.offset) catch continue;
            if (header.schema_id != schema_documents) continue;

            const descs = reader.readColumnDirectory(header, block.offset) catch continue;
            defer allocator.free(descs);

            if (header.row_count == 0) continue;

            const hash_desc = findColumn(descs, col_doc_hash) orelse continue;
            const ts_desc = findColumn(descs, col_ts) orelse continue;
            const expires_desc = findColumn(descs, col_expires_at);
            const payload_desc = findColumn(descs, col_payload) orelse continue;

            const hash_bytes = reader.readColumnData(block.offset, hash_desc, allocator) catch continue;
            defer allocator.free(hash_bytes);

            const ts_bytes = reader.readColumnData(block.offset, ts_desc, allocator) catch continue;
            defer allocator.free(ts_bytes);

            var expires_bytes: ?[]const u8 = null;
            defer if (expires_bytes) |eb| allocator.free(eb);
            if (expires_desc) |ed| {
                expires_bytes = reader.readColumnData(block.offset, ed, allocator) catch null;
            }

            var payload_buffers = readVarBytesBuffers(file, block.offset, payload_desc, header.row_count, allocator) catch continue;
            defer payload_buffers.deinit(allocator);

            // Reverse-scan rows within block.
            var row_idx = header.row_count;
            while (row_idx > 0) {
                row_idx -= 1;

                const row_hash = readU64At(hash_bytes, row_idx) catch continue;
                if (row_hash != target_hash) continue;

                const ts = readI64At(ts_bytes, row_idx) catch continue;

                if (deleted.get(row_hash)) |del_ts| {
                    if (del_ts >= ts) continue;
                }

                // First non-deleted match in reverse order is the latest version.
                var expires_at: i64 = 0;
                if (expires_bytes) |eb| {
                    expires_at = readI64At(eb, row_idx) catch 0;
                }

                const payload = payload_buffers.sliceForRow(row_idx) catch continue;
                const record_opt = decodeDocumentHeader(allocator, payload) catch continue;
                if (record_opt) |record| {
                    if (expires_at > 0 and expires_at < now_ms) {
                        var r = record;
                        r.deinit(allocator);
                        return null;
                    }
                    var result = record;
                    result.expires_at_ms = expires_at;
                    return result;
                }
            }
        }

        return null;
    }

    /// Load an externalized blob by reference (`sha256:<hex>` or `multi:<hex>`).
    pub fn loadBlob(self: *DocumentAdapter, allocator: Allocator, blob_ref: []const u8) ![]u8 {
        return self.blob_store.readAll(blob_ref, allocator);
    }

    /// Open a streaming reader for an externalized blob reference.
    pub fn openBlobStream(self: *DocumentAdapter, allocator: Allocator, blob_ref: []const u8) !db_blob_store.BlobReadStream {
        return self.blob_store.openReadStream(allocator, blob_ref);
    }

    /// Get document version history (all versions linked by parent_id).
    pub fn getDocumentVersions(self: *DocumentAdapter, allocator: Allocator, doc_id: []const u8) ![]DocumentRecord {
        const target_parent_hash = computeHash(doc_id);

        var results = std.ArrayList(DocumentRecord).empty;
        errdefer {
            for (results.items) |*r| r.deinit(allocator);
            results.deinit(allocator);
        }

        const blocks = try self.fs_reader.getBlocks(allocator);
        defer allocator.free(blocks);

        for (blocks) |block| {
            var file = self.fs_reader.dir.openFile(block.path, .{ .mode = .read_only }) catch continue;
            defer file.close();

            const reader = block_reader.BlockReader.init(file, allocator);
            const header = reader.readHeader(block.offset) catch continue;
            if (header.schema_id != schema_documents) continue;

            const descs = reader.readColumnDirectory(header, block.offset) catch continue;
            defer allocator.free(descs);

            const row_count = header.row_count;
            if (row_count == 0) continue;

            const parent_desc = findColumn(descs, col_parent_hash) orelse continue;
            const payload_desc = findColumn(descs, col_payload) orelse continue;

            const parent_bytes = reader.readColumnData(block.offset, parent_desc, allocator) catch continue;
            defer allocator.free(parent_bytes);

            var payload_buffers = readVarBytesBuffers(file, block.offset, payload_desc, row_count, allocator) catch continue;
            defer payload_buffers.deinit(allocator);

            for (0..row_count) |row_idx| {
                const parent_hash = readU64At(parent_bytes, row_idx) catch continue;
                if (parent_hash != target_parent_hash) continue;

                const payload = payload_buffers.sliceForRow(row_idx) catch continue;
                const record_opt = decodeDocumentRecordWithBlobStore(allocator, payload, &self.blob_store) catch continue;
                if (record_opt) |record| {
                    try results.append(allocator, record);
                }
            }
        }

        return results.toOwnedSlice(allocator);
    }

    /// Flush pending writes to disk.
    pub fn flush(self: *DocumentAdapter) !void {
        if (!self.read_only) {
            try self.fs_writer.flushBlock();
        }
    }

    // =========================================================================
    // CDC (Change Data Capture) Operations
    // =========================================================================

    /// Get changes since a given sequence number.
    /// Returns change records ordered by seq_num ascending.
    /// Use since_seq=0 to get all changes from the beginning.
    /// Caller owns returned records; free each with ChangeRecord.deinit().
    pub fn getChanges(
        self: *DocumentAdapter,
        allocator: Allocator,
        since_seq: u64,
        group_id: ?[]const u8,
        limit: usize,
    ) ![]ChangeRecord {
        const target_group_hash: ?u64 = if (group_id) |g| computeHash(g) else null;

        // Collect all changes (creates/updates from documents, deletes from tombstones)
        var changes = std.ArrayList(ChangeRecord).empty;
        errdefer {
            for (changes.items) |*c| c.deinit(allocator);
            changes.deinit(allocator);
        }

        // Track seen doc_ids to detect create vs update
        // We'll determine action based on whether we've seen earlier versions
        var doc_first_seq = std.AutoHashMap(u64, u64).init(allocator);
        defer doc_first_seq.deinit();

        const blocks = try self.fs_reader.getBlocks(allocator);
        defer allocator.free(blocks);

        // Phase 1: Collect document creates/updates
        for (blocks) |block| {
            var file = self.fs_reader.dir.openFile(block.path, .{ .mode = .read_only }) catch continue;
            defer file.close();

            const reader = block_reader.BlockReader.init(file, allocator);
            const header = reader.readHeader(block.offset) catch continue;

            if (header.schema_id == schema_documents) {
                try self.scanBlockForChanges(
                    file,
                    block.offset,
                    allocator,
                    since_seq,
                    target_group_hash,
                    &changes,
                    &doc_first_seq,
                );
            } else if (header.schema_id == schema_document_deletes) {
                try self.scanBlockForDeletes(
                    file,
                    block.offset,
                    allocator,
                    since_seq,
                    &changes,
                );
            }
        }

        // Sort by seq_num ascending
        std.mem.sort(ChangeRecord, changes.items, {}, struct {
            fn lessThan(_: void, a: ChangeRecord, b: ChangeRecord) bool {
                return a.seq_num < b.seq_num;
            }
        }.lessThan);

        // Apply limit
        const effective_limit = if (limit == 0) changes.items.len else @min(limit, changes.items.len);

        // Free excess items beyond limit
        for (changes.items[effective_limit..]) |*c| c.deinit(allocator);

        // Return limited slice
        if (changes.items.len > effective_limit) {
            const result = try allocator.dupe(ChangeRecord, changes.items[0..effective_limit]);
            changes.deinit(allocator);
            return result;
        }

        return changes.toOwnedSlice(allocator);
    }

    fn scanBlockForChanges(
        self: *DocumentAdapter,
        file: std.fs.File,
        block_offset: u64,
        alloc: Allocator,
        since_seq: u64,
        target_group_hash: ?u64,
        changes: *std.ArrayList(ChangeRecord),
        doc_first_seq: *std.AutoHashMap(u64, u64),
    ) !void {
        _ = self;
        const reader = block_reader.BlockReader.init(file, alloc);
        const header = try reader.readHeader(block_offset);
        if (header.row_count == 0) return;

        const descs = try reader.readColumnDirectory(header, block_offset);
        defer alloc.free(descs);

        const hash_desc = findColumn(descs, col_doc_hash) orelse return;
        const ts_desc = findColumn(descs, col_ts) orelse return;
        const seq_desc = findColumn(descs, col_seq_num) orelse return;
        const group_desc = findColumn(descs, col_group_hash);
        const payload_desc = findColumn(descs, col_payload) orelse return;

        const hash_bytes = try reader.readColumnData(block_offset, hash_desc, alloc);
        defer alloc.free(hash_bytes);

        const ts_bytes = try reader.readColumnData(block_offset, ts_desc, alloc);
        defer alloc.free(ts_bytes);

        const seq_bytes = try reader.readColumnData(block_offset, seq_desc, alloc);
        defer alloc.free(seq_bytes);

        var group_bytes: ?[]const u8 = null;
        defer if (group_bytes) |gb| alloc.free(gb);
        if (group_desc) |gd| {
            group_bytes = reader.readColumnData(block_offset, gd, alloc) catch null;
        }

        var payload_buffers = try readVarBytesBuffers(file, block_offset, payload_desc, header.row_count, alloc);
        defer payload_buffers.deinit(alloc);

        for (0..header.row_count) |row_idx| {
            const doc_hash = readU64At(hash_bytes, row_idx) catch continue;
            const ts = readI64At(ts_bytes, row_idx) catch continue;
            const seq_num = readU64At(seq_bytes, row_idx) catch continue;

            // Skip if seq_num <= since_seq (already seen)
            if (seq_num <= since_seq) {
                // But still track for create vs update detection
                const gop = try doc_first_seq.getOrPut(doc_hash);
                if (!gop.found_existing or seq_num < gop.value_ptr.*) {
                    gop.value_ptr.* = seq_num;
                }
                continue;
            }

            // Filter by group if specified
            if (target_group_hash) |target| {
                if (group_bytes) |gb| {
                    const row_group = readU64At(gb, row_idx) catch continue;
                    if (row_group != target and row_group != 0) continue;
                }
            }

            const payload = payload_buffers.sliceForRow(row_idx) catch continue;
            if (!kvbuf.isKvBuf(payload)) continue;

            const kv_reader = kvbuf.KvBufReader.init(payload) catch continue;
            const doc_id = kv_reader.get(DocumentFieldIds.doc_id) orelse continue;
            const doc_type = kv_reader.get(DocumentFieldIds.doc_type);
            const title = kv_reader.get(DocumentFieldIds.title);

            // Determine action: create if this is the first record for this doc_hash
            const gop = try doc_first_seq.getOrPut(doc_hash);
            const action: ChangeAction = if (!gop.found_existing or seq_num <= gop.value_ptr.*) blk: {
                gop.value_ptr.* = seq_num;
                break :blk .create;
            } else .update;

            try changes.append(alloc, .{
                .seq_num = seq_num,
                .doc_id = try alloc.dupe(u8, doc_id),
                .action = action,
                .timestamp_ms = ts,
                .doc_type = if (doc_type) |t| try alloc.dupe(u8, t) else null,
                .title = if (title) |t| try alloc.dupe(u8, t) else null,
            });
        }
    }

    fn scanBlockForDeletes(
        self: *DocumentAdapter,
        file: std.fs.File,
        block_offset: u64,
        alloc: Allocator,
        since_seq: u64,
        changes: *std.ArrayList(ChangeRecord),
    ) !void {
        _ = self;
        const reader = block_reader.BlockReader.init(file, alloc);
        const header = try reader.readHeader(block_offset);
        if (header.row_count == 0) return;

        const descs = try reader.readColumnDirectory(header, block_offset);
        defer alloc.free(descs);

        const hash_desc = findColumn(descs, col_doc_hash) orelse return;
        const ts_desc = findColumn(descs, col_ts) orelse return;

        const hash_bytes = try reader.readColumnData(block_offset, hash_desc, alloc);
        defer alloc.free(hash_bytes);

        const ts_bytes = try reader.readColumnData(block_offset, ts_desc, alloc);
        defer alloc.free(ts_bytes);

        // Delete tombstones don't have seq_num in the current schema,
        // so we use timestamp as a proxy for ordering.
        // For proper CDC, we should use the timestamp as seq_num for deletes.
        for (0..header.row_count) |row_idx| {
            const doc_hash = readU64At(hash_bytes, row_idx) catch continue;
            const ts = readI64At(ts_bytes, row_idx) catch continue;

            // Use timestamp as seq_num for delete records
            // This is a simplification; in production we'd add seq_num to tombstones
            const pseudo_seq: u64 = @bitCast(ts);
            if (pseudo_seq <= since_seq) continue;

            // For deletes, we only have the doc_hash, not the actual doc_id
            // We'll format it as a hex string for identification
            var doc_id_buf: [32]u8 = undefined;
            const doc_id_slice = std.fmt.bufPrint(&doc_id_buf, "{x:0>16}", .{doc_hash}) catch continue;

            try changes.append(alloc, .{
                .seq_num = pseudo_seq,
                .doc_id = try alloc.dupe(u8, doc_id_slice),
                .action = .delete,
                .timestamp_ms = ts,
                .doc_type = null,
                .title = null,
            });
        }
    }

    // =========================================================================
    // TTL (Time-To-Live) Operations
    // =========================================================================

    /// Count expired documents in the database.
    /// Returns the number of documents where expires_at > 0 and expires_at < current time.
    pub fn countExpired(self: *DocumentAdapter, allocator: Allocator) !usize {
        const now_ms = std.time.milliTimestamp();

        // Track latest version of each document by hash
        var latest_expires = std.AutoHashMap(u64, i64).init(allocator);
        defer latest_expires.deinit();

        var latest_ts = std.AutoHashMap(u64, i64).init(allocator);
        defer latest_ts.deinit();

        // Collect deleted document hashes
        var deleted = std.AutoHashMap(u64, i64).init(allocator);
        defer deleted.deinit();
        try self.collectDeletedDocuments(allocator, &deleted);

        const blocks = try self.fs_reader.getBlocks(allocator);
        defer allocator.free(blocks);

        for (blocks) |block| {
            var file = self.fs_reader.dir.openFile(block.path, .{ .mode = .read_only }) catch continue;
            defer file.close();

            const reader = block_reader.BlockReader.init(file, allocator);
            const header = reader.readHeader(block.offset) catch continue;
            if (header.schema_id != schema_documents) continue;

            const descs = reader.readColumnDirectory(header, block.offset) catch continue;
            defer allocator.free(descs);

            const row_count = header.row_count;
            if (row_count == 0) continue;

            const hash_desc = findColumn(descs, col_doc_hash) orelse continue;
            const ts_desc = findColumn(descs, col_ts) orelse continue;
            const expires_desc = findColumn(descs, col_expires_at) orelse continue;

            const hash_bytes = reader.readColumnData(block.offset, hash_desc, allocator) catch continue;
            defer allocator.free(hash_bytes);

            const ts_bytes = reader.readColumnData(block.offset, ts_desc, allocator) catch continue;
            defer allocator.free(ts_bytes);

            const expires_bytes = reader.readColumnData(block.offset, expires_desc, allocator) catch continue;
            defer allocator.free(expires_bytes);

            for (0..row_count) |row_idx| {
                const doc_hash = readU64At(hash_bytes, row_idx) catch continue;
                const ts = readI64At(ts_bytes, row_idx) catch continue;
                const expires_at = readI64At(expires_bytes, row_idx) catch continue;

                // Skip if deleted
                if (deleted.get(doc_hash)) |del_ts| {
                    if (del_ts >= ts) continue;
                }

                // Track the latest version's expiration time
                if (latest_ts.get(doc_hash)) |existing_ts| {
                    if (ts > existing_ts) {
                        try latest_ts.put(doc_hash, ts);
                        try latest_expires.put(doc_hash, expires_at);
                    }
                } else {
                    try latest_ts.put(doc_hash, ts);
                    try latest_expires.put(doc_hash, expires_at);
                }
            }
        }

        // Count documents where expires_at > 0 and expires_at < now_ms
        var count: usize = 0;
        var it = latest_expires.valueIterator();
        while (it.next()) |expires_at| {
            if (expires_at.* > 0 and expires_at.* < now_ms) {
                count += 1;
            }
        }

        return count;
    }

    // =========================================================================
    // Delta Versioning Operations
    // =========================================================================

    /// Create a delta version of an existing document.
    /// The delta contains only the changed content (title, tags_text, doc_json, marker).
    /// base_doc_id points to the document being versioned.
    /// Returns the new document ID.
    pub fn createDeltaVersion(
        self: *DocumentAdapter,
        allocator: Allocator,
        base_doc_id: []const u8,
        new_doc_id: []const u8,
        delta_json: []const u8,
        title: ?[]const u8,
        tags_text: ?[]const u8,
        marker: ?[]const u8,
    ) !void {
        // Get the base document to inherit metadata
        var base_doc = try self.getDocument(allocator, base_doc_id) orelse return error.DocumentNotFound;
        defer base_doc.deinit(allocator);

        const now_ms = std.time.milliTimestamp();

        // Create delta record
        const delta_record = DocumentRecord{
            .doc_id = new_doc_id,
            .doc_type = base_doc.doc_type,
            .title = title orelse base_doc.title,
            .tags_text = tags_text orelse base_doc.tags_text,
            .doc_json = delta_json,
            .parent_id = base_doc_id, // Chain to base
            .marker = marker orelse base_doc.marker,
            .group_id = base_doc.group_id,
            .owner_id = base_doc.owner_id,
            .created_at_ms = now_ms,
            .updated_at_ms = now_ms,
            .expires_at_ms = base_doc.expires_at_ms,
            .version_type = "delta",
            .base_doc_id = base_doc_id,
        };

        try self.writeDocument(delta_record);
    }

    /// Get the delta chain for a document.
    /// Returns documents in order from the requested document back to the base.
    /// First element is the requested document, last element is the base (full version).
    /// Caller owns returned records; free each with DocumentRecord.deinit().
    pub fn getDeltaChain(
        self: *DocumentAdapter,
        allocator: Allocator,
        doc_id: []const u8,
    ) ![]DocumentRecord {
        var chain = std.ArrayList(DocumentRecord).empty;
        errdefer {
            for (chain.items) |*r| r.deinit(allocator);
            chain.deinit(allocator);
        }

        var current_id = try allocator.dupe(u8, doc_id);
        defer allocator.free(current_id);

        // Follow the chain from current to base
        var iterations: usize = 0;
        const max_chain_depth = 100; // Prevent infinite loops

        while (iterations < max_chain_depth) : (iterations += 1) {
            const doc = try self.getDocument(allocator, current_id) orelse break;

            // Check if this is a full version (end of chain)
            const is_full = doc.version_type == null or
                std.mem.eql(u8, doc.version_type.?, "full") or
                doc.base_doc_id == null;

            try chain.append(allocator, doc);

            if (is_full) break;

            // Move to base document
            const base_id = doc.base_doc_id.?;
            allocator.free(current_id);
            current_id = try allocator.dupe(u8, base_id);
        }

        return chain.toOwnedSlice(allocator);
    }

    /// Check if a document is a delta version.
    pub fn isDeltaVersion(
        self: *DocumentAdapter,
        allocator: Allocator,
        doc_id: []const u8,
    ) !bool {
        var doc = try self.getDocument(allocator, doc_id) orelse return error.DocumentNotFound;
        defer doc.deinit(allocator);

        return doc.version_type != null and
            std.mem.eql(u8, doc.version_type.?, "delta") and
            doc.base_doc_id != null;
    }

    /// Get the base document ID for a delta version.
    /// Returns null if the document is not a delta or doesn't exist.
    pub fn getBaseDocumentId(
        self: *DocumentAdapter,
        allocator: Allocator,
        doc_id: []const u8,
    ) !?[]const u8 {
        var doc = try self.getDocument(allocator, doc_id) orelse return null;
        defer doc.deinit(allocator);

        if (doc.base_doc_id) |base_id| {
            return try allocator.dupe(u8, base_id);
        }
        return null;
    }

    // =========================================================================
    // Compaction/Garbage Collection Operations
    // =========================================================================

    /// Statistics about storage efficiency and compaction candidates.
    pub const CompactionStats = struct {
        total_documents: usize, // Total document records in storage
        active_documents: usize, // Documents visible to queries
        expired_documents: usize, // Documents with expired TTL
        deleted_documents: usize, // Documents with tombstones
        tombstone_count: usize, // Number of tombstone records
        delta_versions: usize, // Number of delta version records
        estimated_garbage_bytes: u64, // Rough estimate of reclaimable space
    };

    /// Get compaction statistics for the document storage.
    /// Shows how much data could be reclaimed by compaction.
    pub fn getCompactionStats(self: *DocumentAdapter, allocator: Allocator) !DocumentAdapter.CompactionStats {
        const now_ms = std.time.milliTimestamp();

        var stats = DocumentAdapter.CompactionStats{
            .total_documents = 0,
            .active_documents = 0,
            .expired_documents = 0,
            .deleted_documents = 0,
            .tombstone_count = 0,
            .delta_versions = 0,
            .estimated_garbage_bytes = 0,
        };

        // Collect deleted document hashes and count tombstones
        var deleted = std.AutoHashMap(u64, i64).init(allocator);
        defer deleted.deinit();

        const blocks = try self.fs_reader.getBlocks(allocator);
        defer allocator.free(blocks);

        // Phase 1: Count tombstones and collect deleted hashes
        for (blocks) |block| {
            var file = self.fs_reader.dir.openFile(block.path, .{ .mode = .read_only }) catch continue;
            defer file.close();

            const reader = block_reader.BlockReader.init(file, allocator);
            const header = reader.readHeader(block.offset) catch continue;
            if (header.schema_id != schema_document_deletes) continue;

            stats.tombstone_count += header.row_count;

            const descs = reader.readColumnDirectory(header, block.offset) catch continue;
            defer allocator.free(descs);

            const hash_desc = findColumn(descs, col_doc_hash) orelse continue;
            const ts_desc = findColumn(descs, col_ts) orelse continue;

            const hash_bytes = reader.readColumnData(block.offset, hash_desc, allocator) catch continue;
            defer allocator.free(hash_bytes);

            const ts_bytes = reader.readColumnData(block.offset, ts_desc, allocator) catch continue;
            defer allocator.free(ts_bytes);

            for (0..header.row_count) |row_idx| {
                const doc_hash = readU64At(hash_bytes, row_idx) catch continue;
                const ts = readI64At(ts_bytes, row_idx) catch continue;
                if (deleted.get(doc_hash)) |existing_ts| {
                    if (ts > existing_ts) try deleted.put(doc_hash, ts);
                } else {
                    try deleted.put(doc_hash, ts);
                }
            }
        }

        // Phase 2: Count document records and categorize
        var latest_ts = std.AutoHashMap(u64, i64).init(allocator);
        defer latest_ts.deinit();
        var latest_expired = std.AutoHashMap(u64, bool).init(allocator);
        defer latest_expired.deinit();
        var latest_deleted = std.AutoHashMap(u64, bool).init(allocator);
        defer latest_deleted.deinit();
        var latest_delta = std.AutoHashMap(u64, bool).init(allocator);
        defer latest_delta.deinit();

        for (blocks) |block| {
            var file = self.fs_reader.dir.openFile(block.path, .{ .mode = .read_only }) catch continue;
            defer file.close();

            const reader = block_reader.BlockReader.init(file, allocator);
            const header = reader.readHeader(block.offset) catch continue;
            if (header.schema_id != schema_documents) continue;

            stats.total_documents += header.row_count;

            const descs = reader.readColumnDirectory(header, block.offset) catch continue;
            defer allocator.free(descs);

            const hash_desc = findColumn(descs, col_doc_hash) orelse continue;
            const ts_desc = findColumn(descs, col_ts) orelse continue;
            const expires_desc = findColumn(descs, col_expires_at);
            const payload_desc = findColumn(descs, col_payload);

            const hash_bytes = reader.readColumnData(block.offset, hash_desc, allocator) catch continue;
            defer allocator.free(hash_bytes);

            const ts_bytes = reader.readColumnData(block.offset, ts_desc, allocator) catch continue;
            defer allocator.free(ts_bytes);

            var expires_bytes: ?[]const u8 = null;
            defer if (expires_bytes) |eb| allocator.free(eb);
            if (expires_desc) |ed| {
                expires_bytes = reader.readColumnData(block.offset, ed, allocator) catch null;
            }

            var payload_buffers: ?VarBytesBuffers = null;
            defer if (payload_buffers) |*pb| pb.deinit(allocator);
            if (payload_desc) |pd| {
                payload_buffers = readVarBytesBuffers(file, block.offset, pd, header.row_count, allocator) catch null;
            }

            for (0..header.row_count) |row_idx| {
                const doc_hash = readU64At(hash_bytes, row_idx) catch continue;
                const ts = readI64At(ts_bytes, row_idx) catch continue;

                // Check if deleted
                const is_deleted = if (deleted.get(doc_hash)) |del_ts| del_ts >= ts else false;

                // Check if expired
                var is_expired = false;
                if (expires_bytes) |eb| {
                    const expires_at = readI64At(eb, row_idx) catch 0;
                    is_expired = expires_at > 0 and expires_at < now_ms;
                }

                // Check if delta version
                var is_delta = false;
                if (payload_buffers) |pb| {
                    const payload = pb.sliceForRow(row_idx) catch continue;
                    if (kvbuf.isKvBuf(payload)) {
                        if (kvbuf.KvBufReader.init(payload)) |kv_reader| {
                            if (kv_reader.get(DocumentFieldIds.version_type)) |vt| {
                                is_delta = std.mem.eql(u8, vt, "delta");
                            }
                        } else |_| {}
                    }
                }

                // Track latest version of each document
                if (latest_ts.get(doc_hash)) |existing_ts| {
                    if (ts > existing_ts) {
                        try latest_ts.put(doc_hash, ts);
                        try latest_expired.put(doc_hash, is_expired);
                        try latest_deleted.put(doc_hash, is_deleted);
                        try latest_delta.put(doc_hash, is_delta);
                    }
                } else {
                    try latest_ts.put(doc_hash, ts);
                    try latest_expired.put(doc_hash, is_expired);
                    try latest_deleted.put(doc_hash, is_deleted);
                    try latest_delta.put(doc_hash, is_delta);
                }
            }
        }

        // Calculate final stats
        var it = latest_ts.keyIterator();
        while (it.next()) |doc_hash| {
            const is_expired = latest_expired.get(doc_hash.*) orelse false;
            const is_deleted = latest_deleted.get(doc_hash.*) orelse false;
            const is_delta = latest_delta.get(doc_hash.*) orelse false;

            if (is_deleted) {
                stats.deleted_documents += 1;
            } else if (is_expired) {
                stats.expired_documents += 1;
            } else {
                stats.active_documents += 1;
            }

            if (is_delta) {
                stats.delta_versions += 1;
            }
        }

        // Rough estimate: each garbage record ~500 bytes average
        const garbage_records = stats.total_documents - stats.active_documents + stats.tombstone_count;
        stats.estimated_garbage_bytes = garbage_records * 500;

        return stats;
    }

    /// Purge expired documents by writing tombstones for them.
    /// This makes the deletions explicit in the change feed.
    /// Returns the number of documents purged.
    pub fn purgeExpired(self: *DocumentAdapter, allocator: Allocator) !usize {
        const now_ms = std.time.milliTimestamp();

        // Collect expired document IDs
        var expired_ids = std.ArrayList([]const u8).empty;
        defer {
            for (expired_ids.items) |id| allocator.free(id);
            expired_ids.deinit(allocator);
        }

        // Collect deleted documents first
        var deleted = std.AutoHashMap(u64, i64).init(allocator);
        defer deleted.deinit();
        try self.collectDeletedDocuments(allocator, &deleted);

        // Track latest version's expiration
        var latest_ts = std.AutoHashMap(u64, i64).init(allocator);
        defer latest_ts.deinit();
        var latest_expires = std.AutoHashMap(u64, i64).init(allocator);
        defer latest_expires.deinit();
        var latest_id = std.AutoHashMap(u64, []const u8).init(allocator);
        defer {
            var vid = latest_id.valueIterator();
            while (vid.next()) |id| allocator.free(id.*);
            latest_id.deinit();
        }

        const blocks = try self.fs_reader.getBlocks(allocator);
        defer allocator.free(blocks);

        for (blocks) |block| {
            var file = self.fs_reader.dir.openFile(block.path, .{ .mode = .read_only }) catch continue;
            defer file.close();

            const reader = block_reader.BlockReader.init(file, allocator);
            const header = reader.readHeader(block.offset) catch continue;
            if (header.schema_id != schema_documents) continue;

            const descs = reader.readColumnDirectory(header, block.offset) catch continue;
            defer allocator.free(descs);

            const hash_desc = findColumn(descs, col_doc_hash) orelse continue;
            const ts_desc = findColumn(descs, col_ts) orelse continue;
            const expires_desc = findColumn(descs, col_expires_at) orelse continue;
            const payload_desc = findColumn(descs, col_payload) orelse continue;

            const hash_bytes = reader.readColumnData(block.offset, hash_desc, allocator) catch continue;
            defer allocator.free(hash_bytes);

            const ts_bytes = reader.readColumnData(block.offset, ts_desc, allocator) catch continue;
            defer allocator.free(ts_bytes);

            const expires_bytes = reader.readColumnData(block.offset, expires_desc, allocator) catch continue;
            defer allocator.free(expires_bytes);

            var payload_buffers = readVarBytesBuffers(file, block.offset, payload_desc, header.row_count, allocator) catch continue;
            defer payload_buffers.deinit(allocator);

            for (0..header.row_count) |row_idx| {
                const doc_hash = readU64At(hash_bytes, row_idx) catch continue;
                const ts = readI64At(ts_bytes, row_idx) catch continue;
                const expires_at = readI64At(expires_bytes, row_idx) catch continue;

                // Skip if already deleted
                if (deleted.get(doc_hash)) |del_ts| {
                    if (del_ts >= ts) continue;
                }

                // Skip if not expired
                if (expires_at == 0 or expires_at >= now_ms) continue;

                // Track latest expired version
                if (latest_ts.get(doc_hash)) |existing_ts| {
                    if (ts > existing_ts) {
                        try latest_ts.put(doc_hash, ts);
                        try latest_expires.put(doc_hash, expires_at);

                        // Update ID
                        if (latest_id.get(doc_hash)) |old_id| allocator.free(old_id);
                        const payload = payload_buffers.sliceForRow(row_idx) catch continue;
                        if (kvbuf.isKvBuf(payload)) {
                            if (kvbuf.KvBufReader.init(payload)) |kv_reader| {
                                if (kv_reader.get(DocumentFieldIds.doc_id)) |doc_id| {
                                    try latest_id.put(doc_hash, try allocator.dupe(u8, doc_id));
                                }
                            } else |_| {}
                        }
                    }
                } else {
                    try latest_ts.put(doc_hash, ts);
                    try latest_expires.put(doc_hash, expires_at);

                    const payload = payload_buffers.sliceForRow(row_idx) catch continue;
                    if (kvbuf.isKvBuf(payload)) {
                        if (kvbuf.KvBufReader.init(payload)) |kv_reader| {
                            if (kv_reader.get(DocumentFieldIds.doc_id)) |doc_id| {
                                try latest_id.put(doc_hash, try allocator.dupe(u8, doc_id));
                            }
                        } else |_| {}
                    }
                }
            }
        }

        // Collect IDs to purge (only expired, not already deleted)
        var id_it = latest_id.iterator();
        while (id_it.next()) |entry| {
            const doc_hash = entry.key_ptr.*;
            const doc_id = entry.value_ptr.*;

            // Verify it's actually expired and not already deleted
            const expires_at = latest_expires.get(doc_hash) orelse continue;
            if (expires_at > 0 and expires_at < now_ms) {
                if (deleted.get(doc_hash) == null) {
                    try expired_ids.append(allocator, try allocator.dupe(u8, doc_id));
                }
            }
        }

        // Write tombstones for all expired documents
        for (expired_ids.items) |doc_id| {
            try self.deleteDocument(doc_id, now_ms);
        }

        return expired_ids.items.len;
    }

    /// Get list of document IDs that are candidates for garbage collection.
    /// Returns IDs of expired and deleted documents.
    pub fn getGarbageCandidates(self: *DocumentAdapter, allocator: Allocator) ![][]const u8 {
        const now_ms = std.time.milliTimestamp();

        var candidates = std.ArrayList([]const u8).empty;
        errdefer {
            for (candidates.items) |id| allocator.free(id);
            candidates.deinit(allocator);
        }

        // Collect deleted documents
        var deleted = std.AutoHashMap(u64, i64).init(allocator);
        defer deleted.deinit();
        try self.collectDeletedDocuments(allocator, &deleted);

        // Track latest version info
        var latest_ts = std.AutoHashMap(u64, i64).init(allocator);
        defer latest_ts.deinit();
        var latest_expires = std.AutoHashMap(u64, i64).init(allocator);
        defer latest_expires.deinit();
        var latest_id = std.AutoHashMap(u64, []const u8).init(allocator);
        defer {
            var vid = latest_id.valueIterator();
            while (vid.next()) |id| allocator.free(id.*);
            latest_id.deinit();
        }

        const blocks = try self.fs_reader.getBlocks(allocator);
        defer allocator.free(blocks);

        for (blocks) |block| {
            var file = self.fs_reader.dir.openFile(block.path, .{ .mode = .read_only }) catch continue;
            defer file.close();

            const reader = block_reader.BlockReader.init(file, allocator);
            const header = reader.readHeader(block.offset) catch continue;
            if (header.schema_id != schema_documents) continue;

            const descs = reader.readColumnDirectory(header, block.offset) catch continue;
            defer allocator.free(descs);

            const hash_desc = findColumn(descs, col_doc_hash) orelse continue;
            const ts_desc = findColumn(descs, col_ts) orelse continue;
            const expires_desc = findColumn(descs, col_expires_at);
            const payload_desc = findColumn(descs, col_payload) orelse continue;

            const hash_bytes = reader.readColumnData(block.offset, hash_desc, allocator) catch continue;
            defer allocator.free(hash_bytes);

            const ts_bytes = reader.readColumnData(block.offset, ts_desc, allocator) catch continue;
            defer allocator.free(ts_bytes);

            var expires_bytes: ?[]const u8 = null;
            defer if (expires_bytes) |eb| allocator.free(eb);
            if (expires_desc) |ed| {
                expires_bytes = reader.readColumnData(block.offset, ed, allocator) catch null;
            }

            var payload_buffers = readVarBytesBuffers(file, block.offset, payload_desc, header.row_count, allocator) catch continue;
            defer payload_buffers.deinit(allocator);

            for (0..header.row_count) |row_idx| {
                const doc_hash = readU64At(hash_bytes, row_idx) catch continue;
                const ts = readI64At(ts_bytes, row_idx) catch continue;

                var expires_at: i64 = 0;
                if (expires_bytes) |eb| {
                    expires_at = readI64At(eb, row_idx) catch 0;
                }

                // Track latest version
                if (latest_ts.get(doc_hash)) |existing_ts| {
                    if (ts > existing_ts) {
                        try latest_ts.put(doc_hash, ts);
                        try latest_expires.put(doc_hash, expires_at);

                        if (latest_id.get(doc_hash)) |old_id| allocator.free(old_id);
                        const payload = payload_buffers.sliceForRow(row_idx) catch continue;
                        if (kvbuf.isKvBuf(payload)) {
                            if (kvbuf.KvBufReader.init(payload)) |kv_reader| {
                                if (kv_reader.get(DocumentFieldIds.doc_id)) |doc_id| {
                                    try latest_id.put(doc_hash, try allocator.dupe(u8, doc_id));
                                }
                            } else |_| {}
                        }
                    }
                } else {
                    try latest_ts.put(doc_hash, ts);
                    try latest_expires.put(doc_hash, expires_at);

                    const payload = payload_buffers.sliceForRow(row_idx) catch continue;
                    if (kvbuf.isKvBuf(payload)) {
                        if (kvbuf.KvBufReader.init(payload)) |kv_reader| {
                            if (kv_reader.get(DocumentFieldIds.doc_id)) |doc_id| {
                                try latest_id.put(doc_hash, try allocator.dupe(u8, doc_id));
                            }
                        } else |_| {}
                    }
                }
            }
        }

        // Collect garbage candidates (deleted or expired)
        var id_it = latest_id.iterator();
        while (id_it.next()) |entry| {
            const doc_hash = entry.key_ptr.*;
            const doc_id = entry.value_ptr.*;
            const ts = latest_ts.get(doc_hash) orelse continue;
            const expires_at = latest_expires.get(doc_hash) orelse 0;

            // Check if deleted
            const is_deleted = if (deleted.get(doc_hash)) |del_ts| del_ts >= ts else false;

            // Check if expired
            const is_expired = expires_at > 0 and expires_at < now_ms;

            if (is_deleted or is_expired) {
                try candidates.append(allocator, try allocator.dupe(u8, doc_id));
            }
        }

        return candidates.toOwnedSlice(allocator);
    }

    // =========================================================================
    // Internal Helpers
    // =========================================================================

    fn collectDeletedDocuments(self: *DocumentAdapter, allocator: Allocator, deleted: *std.AutoHashMap(u64, i64)) !void {
        const blocks = try self.fs_reader.getBlocks(allocator);
        defer allocator.free(blocks);

        for (blocks) |block| {
            var file = self.fs_reader.dir.openFile(block.path, .{ .mode = .read_only }) catch continue;
            defer file.close();

            const reader = block_reader.BlockReader.init(file, allocator);
            const header = reader.readHeader(block.offset) catch continue;
            if (header.schema_id != schema_document_deletes) continue;

            const descs = reader.readColumnDirectory(header, block.offset) catch continue;
            defer allocator.free(descs);

            const row_count = header.row_count;
            if (row_count == 0) continue;

            const hash_desc = findColumn(descs, col_doc_hash) orelse continue;
            const ts_desc = findColumn(descs, col_ts) orelse continue;

            const hash_bytes = reader.readColumnData(block.offset, hash_desc, allocator) catch continue;
            defer allocator.free(hash_bytes);

            const ts_bytes = reader.readColumnData(block.offset, ts_desc, allocator) catch continue;
            defer allocator.free(ts_bytes);

            for (0..row_count) |row_idx| {
                const doc_hash = readU64At(hash_bytes, row_idx) catch continue;
                const ts = readI64At(ts_bytes, row_idx) catch continue;

                if (deleted.get(doc_hash)) |existing| {
                    if (ts > existing) {
                        try deleted.put(doc_hash, ts);
                    }
                } else {
                    try deleted.put(doc_hash, ts);
                }
            }
        }
    }

    // =========================================================================
    // Search Operations
    // =========================================================================

    /// Search documents by content (title, tags_text, doc_json).
    /// Returns search results with matching snippets.
    /// Uses parallel block scanning for performance.
    /// Expired documents are automatically filtered out.
    /// Caller owns returned results; free each with SearchResult.deinit().
    pub fn searchDocuments(
        self: *DocumentAdapter,
        allocator: Allocator,
        query: []const u8,
        doc_type: ?[]const u8,
    ) ![]SearchResult {
        if (query.len == 0) return allocator.alloc(SearchResult, 0);

        const target_type_hash: ?u64 = if (doc_type) |t| computeHash(t) else null;
        const now_ms = std.time.milliTimestamp();

        // Collect deleted document hashes
        var deleted = std.AutoHashMap(u64, i64).init(allocator);
        defer deleted.deinit();
        try self.collectDeletedDocuments(allocator, &deleted);

        // Phase 1: Inventory — collect block offsets for document blocks.
        var opened_files = std.ArrayList(std.fs.File).empty;
        defer {
            for (opened_files.items) |f| f.close();
            opened_files.deinit(allocator);
        }

        var work_items = std.ArrayList(BlockWork).empty;
        defer work_items.deinit(allocator);

        // current.talu
        if (self.fs_reader.current_path) |current_path| {
            const file = self.fs_reader.dir.openFile(current_path, .{ .mode = .read_only }) catch |err| switch (err) {
                error.FileNotFound => null,
                else => return err,
            };
            if (file) |f| {
                const file_idx: u16 = @intCast(opened_files.items.len);
                try opened_files.append(allocator, f);

                const stat = try f.stat();
                const reader = block_reader.BlockReader.init(f, allocator);
                const block_index = try reader.getBlockIndex(stat.size);
                defer allocator.free(block_index);

                for (block_index) |entry| {
                    if (entry.schema_id != schema_documents) continue;
                    try work_items.append(allocator, .{
                        .file_idx = file_idx,
                        .block_offset = entry.block_off,
                    });
                }
            }
        }

        // Sealed segments
        if (self.fs_reader.manifest_data) |manifest| {
            for (manifest.segments) |segment| {
                const file = self.fs_reader.dir.openFile(segment.path, .{ .mode = .read_only }) catch |err| switch (err) {
                    error.FileNotFound => continue,
                    else => return err,
                };
                const file_idx: u16 = @intCast(opened_files.items.len);
                try opened_files.append(allocator, file);

                const stat = try file.stat();
                const reader = block_reader.BlockReader.init(file, allocator);
                const block_index = try reader.getBlockIndex(stat.size);
                defer allocator.free(block_index);

                for (block_index) |entry| {
                    if (entry.schema_id != schema_documents) continue;
                    try work_items.append(allocator, .{
                        .file_idx = file_idx,
                        .block_offset = entry.block_off,
                    });
                }
            }
        }

        const n_work = work_items.items.len;
        if (n_work == 0) return allocator.alloc(SearchResult, 0);

        // Phase 2: Allocate per-work-item result lists and run parallel scan.
        const result_lists = try allocator.alloc(std.ArrayList(SearchMatch), n_work);
        defer allocator.free(result_lists);
        for (result_lists) |*list| list.* = std.ArrayList(SearchMatch).empty;

        // Cleanup on error: free all allocated search matches.
        var results_owned = true;
        defer if (results_owned) {
            for (result_lists) |*list| {
                for (list.items) |*m| m.deinit(allocator);
                list.deinit(allocator);
            }
        };

        var ctx = ParallelSearchCtx{
            .opened_files = opened_files.items,
            .work_items = work_items.items,
            .query = query,
            .target_type_hash = target_type_hash,
            .deleted = &deleted,
            .now_ms = now_ms,
            .blob_store = &self.blob_store,
            .result_lists = result_lists,
            .alloc = allocator,
        };

        parallel.global().parallelFor(n_work, parallelSearchWorker, &ctx);

        // Phase 3: Merge and deduplicate (first match per doc_hash wins).
        var seen = std.AutoHashMap(u64, void).init(allocator);
        defer seen.deinit();

        var results = std.ArrayList(SearchResult).empty;
        errdefer {
            for (results.items) |*r| r.deinit(allocator);
            results.deinit(allocator);
        }

        for (result_lists) |*list| {
            for (list.items) |*match| {
                const gop = try seen.getOrPut(match.doc_hash);
                if (!gop.found_existing) {
                    // Transfer ownership
                    try results.append(allocator, .{
                        .doc_id = match.doc_id,
                        .doc_type = match.doc_type,
                        .title = match.title,
                        .snippet = match.snippet,
                    });
                    match.doc_id = "";
                    match.doc_type = "";
                    match.title = "";
                    match.snippet = "";
                } else {
                    // Duplicate — free the match
                    match.deinit(allocator);
                }
            }
            list.deinit(allocator);
        }
        results_owned = false;

        return results.toOwnedSlice(allocator);
    }

    /// Batch search: execute multiple queries in a single pass.
    /// Returns a map of query_id → [doc_ids].
    /// More efficient than multiple single searches for many queries.
    /// Expired documents are automatically filtered out.
    /// Caller owns returned results; free each with BatchSearchResult.deinit().
    pub fn searchDocumentsBatch(
        self: *DocumentAdapter,
        allocator: Allocator,
        queries: []const BatchQuery,
    ) ![]BatchSearchResult {
        if (queries.len == 0) return allocator.alloc(BatchSearchResult, 0);

        const now_ms = std.time.milliTimestamp();

        // Collect deleted document hashes
        var deleted = std.AutoHashMap(u64, i64).init(allocator);
        defer deleted.deinit();
        try self.collectDeletedDocuments(allocator, &deleted);

        // Phase 1: Inventory — collect block offsets for document blocks.
        var opened_files = std.ArrayList(std.fs.File).empty;
        defer {
            for (opened_files.items) |f| f.close();
            opened_files.deinit(allocator);
        }

        var work_items = std.ArrayList(BlockWork).empty;
        defer work_items.deinit(allocator);

        // current.talu
        if (self.fs_reader.current_path) |current_path| {
            const file = self.fs_reader.dir.openFile(current_path, .{ .mode = .read_only }) catch |err| switch (err) {
                error.FileNotFound => null,
                else => return err,
            };
            if (file) |f| {
                const file_idx: u16 = @intCast(opened_files.items.len);
                try opened_files.append(allocator, f);

                const stat = try f.stat();
                const reader = block_reader.BlockReader.init(f, allocator);
                const block_index = try reader.getBlockIndex(stat.size);
                defer allocator.free(block_index);

                for (block_index) |entry| {
                    if (entry.schema_id != schema_documents) continue;
                    try work_items.append(allocator, .{
                        .file_idx = file_idx,
                        .block_offset = entry.block_off,
                    });
                }
            }
        }

        // Sealed segments
        if (self.fs_reader.manifest_data) |manifest| {
            for (manifest.segments) |segment| {
                const file = self.fs_reader.dir.openFile(segment.path, .{ .mode = .read_only }) catch |err| switch (err) {
                    error.FileNotFound => continue,
                    else => return err,
                };
                const file_idx: u16 = @intCast(opened_files.items.len);
                try opened_files.append(allocator, file);

                const stat = try file.stat();
                const reader = block_reader.BlockReader.init(file, allocator);
                const block_index = try reader.getBlockIndex(stat.size);
                defer allocator.free(block_index);

                for (block_index) |entry| {
                    if (entry.schema_id != schema_documents) continue;
                    try work_items.append(allocator, .{
                        .file_idx = file_idx,
                        .block_offset = entry.block_off,
                    });
                }
            }
        }

        const n_work = work_items.items.len;
        if (n_work == 0) {
            // Return empty results for each query
            const results = try allocator.alloc(BatchSearchResult, queries.len);
            errdefer allocator.free(results);
            var init_count: usize = 0;
            errdefer for (results[0..init_count]) |r| {
                allocator.free(r.query_id);
                allocator.free(r.doc_ids);
            };
            for (queries, 0..) |q, i| {
                results[i] = .{
                    .query_id = try allocator.dupe(u8, q.id),
                    .doc_ids = try allocator.alloc([]const u8, 0),
                };
                init_count += 1;
            }
            return results;
        }

        // Phase 2: Allocate per-work-item result buckets and run parallel scan.
        // Each work item gets an array of doc_id lists, one per query.
        const batch_lists = try allocator.alloc([]std.ArrayList(BatchMatch), n_work);
        defer allocator.free(batch_lists);
        for (batch_lists, 0..) |*bl, i| {
            bl.* = try allocator.alloc(std.ArrayList(BatchMatch), queries.len);
            for (bl.*) |*list| list.* = std.ArrayList(BatchMatch).empty;
            _ = i;
        }

        // Cleanup on error
        var batch_owned = true;
        defer if (batch_owned) {
            for (batch_lists) |bl| {
                for (bl) |*list| {
                    for (list.items) |*m| m.deinit(allocator);
                    list.deinit(allocator);
                }
                allocator.free(bl);
            }
        };

        var ctx = ParallelBatchSearchCtx{
            .opened_files = opened_files.items,
            .work_items = work_items.items,
            .queries = queries,
            .deleted = &deleted,
            .now_ms = now_ms,
            .blob_store = &self.blob_store,
            .batch_lists = batch_lists,
            .alloc = allocator,
        };

        parallel.global().parallelFor(n_work, parallelBatchSearchWorker, &ctx);

        // Phase 3: Merge results per query and deduplicate.
        const results = try allocator.alloc(BatchSearchResult, queries.len);
        errdefer {
            for (results) |*r| r.deinit(allocator);
            allocator.free(results);
        }

        for (queries, 0..) |q, qi| {
            var seen = std.AutoHashMap(u64, void).init(allocator);
            defer seen.deinit();

            var doc_ids = std.ArrayList([]const u8).empty;
            errdefer {
                for (doc_ids.items) |d| allocator.free(d);
                doc_ids.deinit(allocator);
            }

            // Collect from all work items for this query
            for (batch_lists) |bl| {
                for (bl[qi].items) |*match| {
                    const gop = try seen.getOrPut(match.doc_hash);
                    if (!gop.found_existing) {
                        try doc_ids.append(allocator, match.doc_id);
                        match.doc_id = ""; // Transfer ownership
                    } else {
                        allocator.free(match.doc_id);
                        match.doc_id = "";
                    }
                }
                bl[qi].deinit(allocator);
            }

            results[qi] = .{
                .query_id = try allocator.dupe(u8, q.id),
                .doc_ids = try doc_ids.toOwnedSlice(allocator),
            };
        }

        // Free the outer batch_lists arrays
        for (batch_lists) |bl| allocator.free(bl);
        batch_owned = false;

        return results;
    }
};

/// Work item for parallel document search.
const BlockWork = struct {
    file_idx: u16,
    block_offset: u64,
};

/// Intermediate search match (before deduplication).
const SearchMatch = struct {
    doc_hash: u64,
    doc_id: []const u8,
    doc_type: []const u8,
    title: []const u8,
    snippet: []const u8,

    fn deinit(self: *SearchMatch, allocator: Allocator) void {
        if (self.doc_id.len > 0) allocator.free(self.doc_id);
        if (self.doc_type.len > 0) allocator.free(self.doc_type);
        if (self.title.len > 0) allocator.free(self.title);
        if (self.snippet.len > 0) allocator.free(self.snippet);
    }
};

/// Shared context for parallel search workers.
const ParallelSearchCtx = struct {
    opened_files: []std.fs.File,
    work_items: []const BlockWork,
    query: []const u8,
    target_type_hash: ?u64,
    deleted: *std.AutoHashMap(u64, i64),
    now_ms: i64,
    blob_store: *db_blob_store.BlobStore,
    result_lists: []std.ArrayList(SearchMatch),
    alloc: Allocator,
};

/// Worker function for parallel document search.
fn parallelSearchWorker(start: usize, end: usize, ctx: *ParallelSearchCtx) void {
    for (start..end) |i| {
        const work = ctx.work_items[i];
        const file = ctx.opened_files[work.file_idx];
        scanBlockForMatches(
            file,
            work.block_offset,
            ctx.alloc,
            ctx.query,
            ctx.target_type_hash,
            ctx.deleted,
            ctx.now_ms,
            ctx.blob_store,
            &ctx.result_lists[i],
        ) catch continue;
    }
}

/// Scan a single document block for content matches.
fn scanBlockForMatches(
    file: std.fs.File,
    block_offset: u64,
    alloc: Allocator,
    query: []const u8,
    target_type_hash: ?u64,
    deleted: *std.AutoHashMap(u64, i64),
    now_ms: i64,
    blob_store: *db_blob_store.BlobStore,
    results: *std.ArrayList(SearchMatch),
) !void {
    const reader = block_reader.BlockReader.init(file, alloc);
    const header = try reader.readHeader(block_offset);
    if (header.row_count == 0) return;

    const descs = try reader.readColumnDirectory(header, block_offset);
    defer alloc.free(descs);

    const hash_desc = findColumn(descs, col_doc_hash) orelse return;
    const ts_desc = findColumn(descs, col_ts) orelse return;
    const type_desc = findColumn(descs, col_type_hash);
    const expires_desc = findColumn(descs, col_expires_at);
    const payload_desc = findColumn(descs, col_payload) orelse return;

    const hash_bytes = try reader.readColumnData(block_offset, hash_desc, alloc);
    defer alloc.free(hash_bytes);

    const ts_bytes = try reader.readColumnData(block_offset, ts_desc, alloc);
    defer alloc.free(ts_bytes);

    var type_bytes: ?[]const u8 = null;
    defer if (type_bytes) |tb| alloc.free(tb);
    if (type_desc) |td| {
        type_bytes = reader.readColumnData(block_offset, td, alloc) catch null;
    }

    var expires_bytes: ?[]const u8 = null;
    defer if (expires_bytes) |eb| alloc.free(eb);
    if (expires_desc) |ed| {
        expires_bytes = reader.readColumnData(block_offset, ed, alloc) catch null;
    }

    var payload_buffers = try readVarBytesBuffers(file, block_offset, payload_desc, header.row_count, alloc);
    defer payload_buffers.deinit(alloc);

    for (0..header.row_count) |row_idx| {
        const doc_hash = readU64At(hash_bytes, row_idx) catch continue;
        const ts = readI64At(ts_bytes, row_idx) catch continue;

        // Skip if deleted
        if (deleted.get(doc_hash)) |del_ts| {
            if (del_ts >= ts) continue;
        }

        // Skip if expired
        if (expires_bytes) |eb| {
            const expires_at = readI64At(eb, row_idx) catch 0;
            if (expires_at > 0 and expires_at < now_ms) continue;
        }

        // Filter by type if specified
        if (target_type_hash) |target| {
            if (type_bytes) |tb| {
                const row_type = readU64At(tb, row_idx) catch continue;
                if (row_type != target) continue;
            }
        }

        const payload = payload_buffers.sliceForRow(row_idx) catch continue;
        if (!kvbuf.isKvBuf(payload)) continue;

        const kv_reader = kvbuf.KvBufReader.init(payload) catch continue;
        const doc_id = kv_reader.get(DocumentFieldIds.doc_id) orelse continue;
        const doc_type_str = kv_reader.get(DocumentFieldIds.doc_type) orelse continue;
        const title = kv_reader.get(DocumentFieldIds.title) orelse continue;

        // Search in title, tags_text, and doc_json
        var snippet: ?[]const u8 = null;

        // Try title first
        if (textFindInsensitive(title, query) != null) {
            snippet = try extractSnippet(title, query, alloc);
        }

        // Try tags_text
        if (snippet == null) {
            if (kv_reader.get(DocumentFieldIds.tags_text)) |tags| {
                if (textFindInsensitive(tags, query) != null) {
                    snippet = try extractSnippet(tags, query, alloc);
                }
            }
        }

        // Try doc_json
        if (snippet == null) {
            if (kv_reader.get(DocumentFieldIds.doc_json)) |json| {
                if (textFindInsensitive(json, query) != null) {
                    snippet = try extractSnippet(json, query, alloc);
                }
            } else if (kv_reader.get(DocumentFieldIds.doc_json_ref)) |doc_json_ref| {
                const trigram_bloom = kv_reader.get(DocumentFieldIds.doc_json_trigram_bloom);
                if (mayContainSubstringByTrigramBloom(trigram_bloom, query)) {
                    const loaded_json_opt = try readBlobForSearch(blob_store, doc_json_ref, alloc);
                    defer if (loaded_json_opt) |blob| alloc.free(blob);
                    if (loaded_json_opt) |json| {
                        if (textFindInsensitive(json, query) != null) {
                            snippet = try extractSnippet(json, query, alloc);
                        }
                    }
                }
            }
        }

        if (snippet) |s| {
            try results.append(alloc, .{
                .doc_hash = doc_hash,
                .doc_id = try alloc.dupe(u8, doc_id),
                .doc_type = try alloc.dupe(u8, doc_type_str),
                .title = try alloc.dupe(u8, title),
                .snippet = s,
            });
        }
    }
}

// =============================================================================
// Batch Search Infrastructure
// =============================================================================

/// Match for batch search (just doc_hash and doc_id, no snippet needed).
const BatchMatch = struct {
    doc_hash: u64,
    doc_id: []const u8,

    fn deinit(self: *BatchMatch, allocator: Allocator) void {
        if (self.doc_id.len > 0) allocator.free(self.doc_id);
    }
};

/// Shared context for parallel batch search workers.
const ParallelBatchSearchCtx = struct {
    opened_files: []std.fs.File,
    work_items: []const BlockWork,
    queries: []const BatchQuery,
    deleted: *std.AutoHashMap(u64, i64),
    now_ms: i64,
    blob_store: *db_blob_store.BlobStore,
    batch_lists: [][]std.ArrayList(BatchMatch), // [work_idx][query_idx]
    alloc: Allocator,
};

/// Worker function for parallel batch search.
fn parallelBatchSearchWorker(start: usize, end: usize, ctx: *ParallelBatchSearchCtx) void {
    for (start..end) |i| {
        const work = ctx.work_items[i];
        const file = ctx.opened_files[work.file_idx];
        scanBlockForBatchMatches(
            file,
            work.block_offset,
            ctx.alloc,
            ctx.queries,
            ctx.deleted,
            ctx.now_ms,
            ctx.blob_store,
            ctx.batch_lists[i],
        ) catch continue;
    }
}

/// Scan a single document block for batch matches (all queries at once).
fn scanBlockForBatchMatches(
    file: std.fs.File,
    block_offset: u64,
    alloc: Allocator,
    queries: []const BatchQuery,
    deleted: *std.AutoHashMap(u64, i64),
    now_ms: i64,
    blob_store: *db_blob_store.BlobStore,
    results: []std.ArrayList(BatchMatch), // One list per query
) !void {
    const reader = block_reader.BlockReader.init(file, alloc);
    const header = try reader.readHeader(block_offset);
    if (header.row_count == 0) return;

    const descs = try reader.readColumnDirectory(header, block_offset);
    defer alloc.free(descs);

    const hash_desc = findColumn(descs, col_doc_hash) orelse return;
    const ts_desc = findColumn(descs, col_ts) orelse return;
    const type_desc = findColumn(descs, col_type_hash);
    const expires_desc = findColumn(descs, col_expires_at);
    const payload_desc = findColumn(descs, col_payload) orelse return;

    const hash_bytes = try reader.readColumnData(block_offset, hash_desc, alloc);
    defer alloc.free(hash_bytes);

    const ts_bytes = try reader.readColumnData(block_offset, ts_desc, alloc);
    defer alloc.free(ts_bytes);

    var type_bytes: ?[]const u8 = null;
    defer if (type_bytes) |tb| alloc.free(tb);
    if (type_desc) |td| {
        type_bytes = reader.readColumnData(block_offset, td, alloc) catch null;
    }

    var expires_bytes: ?[]const u8 = null;
    defer if (expires_bytes) |eb| alloc.free(eb);
    if (expires_desc) |ed| {
        expires_bytes = reader.readColumnData(block_offset, ed, alloc) catch null;
    }

    var payload_buffers = try readVarBytesBuffers(file, block_offset, payload_desc, header.row_count, alloc);
    defer payload_buffers.deinit(alloc);

    // Pre-compute type hashes for queries that have type filters
    const query_type_hashes = try alloc.alloc(?u64, queries.len);
    defer alloc.free(query_type_hashes);
    for (queries, 0..) |q, qi| {
        query_type_hashes[qi] = if (q.doc_type) |t| computeHash(t) else null;
    }
    const pending_qi = try alloc.alloc(usize, queries.len);
    defer alloc.free(pending_qi);

    for (0..header.row_count) |row_idx| {
        const doc_hash = readU64At(hash_bytes, row_idx) catch continue;
        const ts = readI64At(ts_bytes, row_idx) catch continue;

        // Skip if deleted
        if (deleted.get(doc_hash)) |del_ts| {
            if (del_ts >= ts) continue;
        }

        // Skip if expired
        if (expires_bytes) |eb| {
            const expires_at = readI64At(eb, row_idx) catch 0;
            if (expires_at > 0 and expires_at < now_ms) continue;
        }

        const payload = payload_buffers.sliceForRow(row_idx) catch continue;
        if (!kvbuf.isKvBuf(payload)) continue;

        const kv_reader = kvbuf.KvBufReader.init(payload) catch continue;
        const doc_id = kv_reader.get(DocumentFieldIds.doc_id) orelse continue;
        const doc_type_str = kv_reader.get(DocumentFieldIds.doc_type);

        // Get row type hash for filtering
        var row_type_hash: ?u64 = null;
        if (type_bytes) |tb| {
            row_type_hash = readU64At(tb, row_idx) catch null;
        }

        // Get searchable content
        const title = kv_reader.get(DocumentFieldIds.title);
        const tags_text = kv_reader.get(DocumentFieldIds.tags_text);
        const inline_doc_json = kv_reader.get(DocumentFieldIds.doc_json);
        const doc_json_ref = if (inline_doc_json == null)
            kv_reader.get(DocumentFieldIds.doc_json_ref)
        else
            null;
        const doc_json_trigram_bloom = kv_reader.get(DocumentFieldIds.doc_json_trigram_bloom);
        var pending_count: usize = 0;

        var loaded_doc_json: ?[]u8 = null;
        defer if (loaded_doc_json) |blob| alloc.free(blob);

        // Test all queries against this row
        for (queries, 0..) |q, qi| {
            // Check type filter
            if (query_type_hashes[qi]) |target_hash| {
                if (row_type_hash) |rth| {
                    if (rth != target_hash) continue;
                } else if (doc_type_str) |dts| {
                    if (computeHash(dts) != target_hash) continue;
                } else {
                    continue; // No type info available
                }
            }

            // Skip empty queries
            if (q.text.len == 0) continue;

            // Search in title, tags_text, and doc_json
            var found = false;
            if (title) |t| {
                if (textFindInsensitive(t, q.text) != null) found = true;
            }
            if (!found and tags_text != null) {
                if (textFindInsensitive(tags_text.?, q.text) != null) found = true;
            }

            if (!found and inline_doc_json != null) {
                if (textFindInsensitive(inline_doc_json.?, q.text) != null) found = true;
            }

            if (!found and inline_doc_json == null and doc_json_ref != null) {
                if (mayContainSubstringByTrigramBloom(doc_json_trigram_bloom, q.text)) {
                    pending_qi[pending_count] = qi;
                    pending_count += 1;
                }
            }

            if (found) {
                try results[qi].append(alloc, .{
                    .doc_hash = doc_hash,
                    .doc_id = try alloc.dupe(u8, doc_id),
                });
            }
        }

        if (pending_count > 0 and doc_json_ref != null) {
            loaded_doc_json = try readBlobForSearch(blob_store, doc_json_ref.?, alloc);
            if (loaded_doc_json) |json| {
                for (pending_qi[0..pending_count]) |qi| {
                    if (textFindInsensitive(json, queries[qi].text) != null) {
                        try results[qi].append(alloc, .{
                            .doc_hash = doc_hash,
                            .doc_id = try alloc.dupe(u8, doc_id),
                        });
                    }
                }
            }
        }
    }
}

// =============================================================================
// KvBuf Encoding/Decoding
// =============================================================================

fn encodeDocumentRecordKvBuf(allocator: Allocator, record: DocumentRecord) ![]u8 {
    return encodeDocumentRecordKvBufWithStorage(allocator, record, record.doc_json, null, null);
}

fn encodeDocumentRecordKvBufWithStorage(
    allocator: Allocator,
    record: DocumentRecord,
    doc_json_inline: ?[]const u8,
    doc_json_ref: ?[]const u8,
    doc_json_trigram_bloom: ?[]const u8,
) ![]u8 {
    if (doc_json_inline == null and doc_json_ref == null) return error.InvalidPayload;

    var w = kvbuf.KvBufWriter.init();
    errdefer w.deinit(allocator);

    try w.addString(allocator, DocumentFieldIds.doc_id, record.doc_id);
    try w.addString(allocator, DocumentFieldIds.doc_type, record.doc_type);
    try w.addString(allocator, DocumentFieldIds.title, record.title);
    if (record.tags_text) |t| try w.addString(allocator, DocumentFieldIds.tags_text, t);
    if (doc_json_inline) |doc_json| {
        try w.addString(allocator, DocumentFieldIds.doc_json, doc_json);
    }
    if (doc_json_ref) |ref| {
        try w.addString(allocator, DocumentFieldIds.doc_json_ref, ref);
    }
    if (doc_json_trigram_bloom) |bloom| {
        try w.addBytes(allocator, DocumentFieldIds.doc_json_trigram_bloom, bloom);
    }
    if (record.parent_id) |p| try w.addString(allocator, DocumentFieldIds.parent_id, p);
    if (record.marker) |m| try w.addString(allocator, DocumentFieldIds.marker, m);
    if (record.group_id) |g| try w.addString(allocator, DocumentFieldIds.group_id, g);
    if (record.owner_id) |o| try w.addString(allocator, DocumentFieldIds.owner_id, o);
    try w.addI64(allocator, DocumentFieldIds.created_at_ms, record.created_at_ms);
    try w.addI64(allocator, DocumentFieldIds.updated_at_ms, record.updated_at_ms);
    if (record.version_type) |v| try w.addString(allocator, DocumentFieldIds.version_type, v);
    if (record.base_doc_id) |b| try w.addString(allocator, DocumentFieldIds.base_doc_id, b);

    const blob = try w.finish(allocator);
    w.deinit(allocator);
    return blob;
}

fn decodeDocumentRecord(allocator: Allocator, payload: []const u8) !?DocumentRecord {
    return decodeDocumentRecordWithBlobStore(allocator, payload, null);
}

fn decodeDocumentRecordWithBlobStore(
    allocator: Allocator,
    payload: []const u8,
    blob_store: ?*db_blob_store.BlobStore,
) !?DocumentRecord {
    if (!kvbuf.isKvBuf(payload)) return null;

    const reader = kvbuf.KvBufReader.init(payload) catch return null;

    const doc_id = reader.get(DocumentFieldIds.doc_id) orelse return null;
    const doc_type = reader.get(DocumentFieldIds.doc_type) orelse return null;
    const title = reader.get(DocumentFieldIds.title) orelse return null;
    const doc_json = if (reader.get(DocumentFieldIds.doc_json)) |inline_json| blk: {
        break :blk try allocator.dupe(u8, inline_json);
    } else blk: {
        const doc_json_ref = reader.get(DocumentFieldIds.doc_json_ref) orelse return null;
        const store = blob_store orelse return error.MissingBlobStore;
        break :blk try store.readAll(doc_json_ref, allocator);
    };

    var record = DocumentRecord{
        .doc_id = "",
        .doc_type = "",
        .title = "",
        .doc_json = "",
        .created_at_ms = reader.getI64(DocumentFieldIds.created_at_ms) orelse 0,
        .updated_at_ms = reader.getI64(DocumentFieldIds.updated_at_ms) orelse 0,
    };
    errdefer record.deinit(allocator);

    record.doc_id = try allocator.dupe(u8, doc_id);
    record.doc_type = try allocator.dupe(u8, doc_type);
    record.title = try allocator.dupe(u8, title);
    record.doc_json = doc_json;
    if (reader.get(DocumentFieldIds.tags_text)) |t| record.tags_text = try allocator.dupe(u8, t);
    if (reader.get(DocumentFieldIds.parent_id)) |p| record.parent_id = try allocator.dupe(u8, p);
    if (reader.get(DocumentFieldIds.marker)) |m| record.marker = try allocator.dupe(u8, m);
    if (reader.get(DocumentFieldIds.group_id)) |g| record.group_id = try allocator.dupe(u8, g);
    if (reader.get(DocumentFieldIds.owner_id)) |o| record.owner_id = try allocator.dupe(u8, o);
    if (reader.get(DocumentFieldIds.version_type)) |v| record.version_type = try allocator.dupe(u8, v);
    if (reader.get(DocumentFieldIds.base_doc_id)) |b| record.base_doc_id = try allocator.dupe(u8, b);

    return record;
}

fn decodeDocumentHeader(allocator: Allocator, payload: []const u8) !?DocumentHeader {
    if (!kvbuf.isKvBuf(payload)) return null;

    const reader = kvbuf.KvBufReader.init(payload) catch return null;

    const doc_id = reader.get(DocumentFieldIds.doc_id) orelse return null;
    const doc_type = reader.get(DocumentFieldIds.doc_type) orelse return null;
    const title = reader.get(DocumentFieldIds.title) orelse return null;

    var header = DocumentHeader{
        .doc_id = "",
        .doc_type = "",
        .title = "",
        .created_at_ms = reader.getI64(DocumentFieldIds.created_at_ms) orelse 0,
        .updated_at_ms = reader.getI64(DocumentFieldIds.updated_at_ms) orelse 0,
        .has_inline_doc_json = reader.get(DocumentFieldIds.doc_json) != null,
    };
    errdefer header.deinit(allocator);

    header.doc_id = try allocator.dupe(u8, doc_id);
    header.doc_type = try allocator.dupe(u8, doc_type);
    header.title = try allocator.dupe(u8, title);
    if (reader.get(DocumentFieldIds.tags_text)) |t| header.tags_text = try allocator.dupe(u8, t);
    if (reader.get(DocumentFieldIds.parent_id)) |p| header.parent_id = try allocator.dupe(u8, p);
    if (reader.get(DocumentFieldIds.marker)) |m| header.marker = try allocator.dupe(u8, m);
    if (reader.get(DocumentFieldIds.group_id)) |g| header.group_id = try allocator.dupe(u8, g);
    if (reader.get(DocumentFieldIds.owner_id)) |o| header.owner_id = try allocator.dupe(u8, o);
    if (reader.get(DocumentFieldIds.version_type)) |v| header.version_type = try allocator.dupe(u8, v);
    if (reader.get(DocumentFieldIds.base_doc_id)) |b| header.base_doc_id = try allocator.dupe(u8, b);
    if (reader.get(DocumentFieldIds.doc_json_ref)) |r| header.doc_json_ref = try allocator.dupe(u8, r);

    return header;
}

// =============================================================================
// Utility Functions
// =============================================================================

fn computeHash(s: []const u8) u64 {
    return std.hash.Wyhash.hash(0, s);
}

fn computeOptionalHash(s: ?[]const u8) u64 {
    if (s) |str| return computeHash(str);
    return 0;
}

fn findColumn(descs: []const types.ColumnDesc, col_id: u32) ?types.ColumnDesc {
    for (descs) |d| {
        if (d.column_id == col_id) return d;
    }
    return null;
}

fn readU64At(bytes: []const u8, row_idx: usize) !u64 {
    const offset = row_idx * 8;
    if (offset + 8 > bytes.len) return error.OutOfBounds;
    return std.mem.readInt(u64, bytes[offset..][0..8], .little);
}

fn readI64At(bytes: []const u8, row_idx: usize) !i64 {
    const offset = row_idx * 8;
    if (offset + 8 > bytes.len) return error.OutOfBounds;
    return std.mem.readInt(i64, bytes[offset..][0..8], .little);
}

fn resolveDocJsonExternalizeThresholdBytes() usize {
    if (readThresholdBytesFromEnvVar(env_doc_json_externalize_threshold_bytes)) |threshold| return threshold;
    if (readThresholdBytesFromEnvVar(env_shared_externalize_threshold_bytes)) |threshold| return threshold;
    return default_doc_json_externalize_threshold_bytes;
}

fn readThresholdBytesFromEnvVar(env_name: []const u8) ?usize {
    const env_ptr = std.posix.getenv(env_name) orelse return null;
    return parseThresholdBytes(std.mem.sliceTo(env_ptr, 0));
}

fn parseThresholdBytes(raw_value: []const u8) ?usize {
    const trimmed = std.mem.trim(u8, raw_value, " \t\r\n");
    if (trimmed.len == 0) return null;
    return std.fmt.parseUnsigned(usize, trimmed, 10) catch null;
}

fn buildDocJsonTrigramBloom(text: []const u8) [doc_json_trigram_bloom_bytes]u8 {
    var bloom = std.mem.zeroes([doc_json_trigram_bloom_bytes]u8);
    if (text.len < 3) return bloom;

    var i: usize = 0;
    while (i + 2 < text.len) : (i += 1) {
        const t0 = std.ascii.toLower(text[i]);
        const t1 = std.ascii.toLower(text[i + 1]);
        const t2 = std.ascii.toLower(text[i + 2]);
        const h1, const h2 = trigramHashes(t0, t1, t2);
        setBloomBit(&bloom, h1);
        setBloomBit(&bloom, h2);
    }

    return bloom;
}

fn mayContainSubstringByTrigramBloom(bloom_opt: ?[]const u8, query: []const u8) bool {
    if (query.len < 3) return true;
    const bloom = bloom_opt orelse return true;
    if (bloom.len != doc_json_trigram_bloom_bytes) return true;

    var i: usize = 0;
    while (i + 2 < query.len) : (i += 1) {
        const t0 = std.ascii.toLower(query[i]);
        const t1 = std.ascii.toLower(query[i + 1]);
        const t2 = std.ascii.toLower(query[i + 2]);
        const h1, const h2 = trigramHashes(t0, t1, t2);
        if (!isBloomBitSet(bloom, h1) or !isBloomBitSet(bloom, h2)) return false;
    }
    return true;
}

fn trigramHashes(t0: u8, t1: u8, t2: u8) struct { usize, usize } {
    const trigram = [_]u8{ t0, t1, t2 };
    const bit_count = doc_json_trigram_bloom_bytes * 8;
    const h1: usize = @intCast(std.hash.Wyhash.hash(0, &trigram) % bit_count);
    const h2: usize = @intCast(std.hash.Wyhash.hash(0x9e3779b97f4a7c15, &trigram) % bit_count);
    return .{ h1, h2 };
}

fn readBlobForSearch(
    blob_store: *db_blob_store.BlobStore,
    blob_ref: []const u8,
    alloc: Allocator,
) !?[]u8 {
    return blob_store.readAll(blob_ref, alloc) catch |err| switch (err) {
        // Search should remain best-effort for missing/corrupt refs.
        error.FileNotFound, error.InvalidBlobRef => null,
        // Unexpected failures (for example, OOM) should be surfaced.
        else => return err,
    };
}

fn setBloomBit(bloom: *[doc_json_trigram_bloom_bytes]u8, bit_idx: usize) void {
    const byte_idx = bit_idx / 8;
    const bit_mask: u8 = @as(u8, 1) << @as(u3, @intCast(bit_idx % 8));
    bloom[byte_idx] |= bit_mask;
}

fn isBloomBitSet(bloom: []const u8, bit_idx: usize) bool {
    const byte_idx = bit_idx / 8;
    const bit_mask: u8 = @as(u8, 1) << @as(u3, @intCast(bit_idx % 8));
    return (bloom[byte_idx] & bit_mask) != 0;
}

/// Case-insensitive substring search.
fn textFindInsensitive(haystack: []const u8, needle: []const u8) ?usize {
    if (needle.len == 0) return 0;
    if (haystack.len < needle.len) return null;
    for (0..haystack.len - needle.len + 1) |i| {
        var is_match = true;
        for (0..needle.len) |j| {
            if (std.ascii.toLower(haystack[i + j]) != std.ascii.toLower(needle[j])) {
                is_match = false;
                break;
            }
        }
        if (is_match) return i;
    }
    return null;
}

/// Extract a snippet around the match position.
fn extractSnippet(text: []const u8, query: []const u8, alloc: Allocator) !?[]const u8 {
    const match_pos = textFindInsensitive(text, query) orelse return null;
    const lead_in = 30; // bytes of context before the match
    const snippet_len = 200;
    const start = if (match_pos > lead_in) match_pos - lead_in else 0;
    const end = @min(text.len, start + snippet_len);
    return try alloc.dupe(u8, text[start..end]);
}

const VarBytesBuffers = struct {
    data: []u8,
    offsets: []u32,
    lengths: []u32,

    fn deinit(self: *VarBytesBuffers, allocator: Allocator) void {
        allocator.free(self.data);
        allocator.free(self.offsets);
        allocator.free(self.lengths);
    }

    fn sliceForRow(self: VarBytesBuffers, row_idx: usize) ![]const u8 {
        if (row_idx >= self.offsets.len or row_idx >= self.lengths.len) return error.InvalidColumnData;
        const offset = self.offsets[row_idx];
        const length = self.lengths[row_idx];
        const start = @as(usize, offset);
        const end = start + @as(usize, length);
        if (end > self.data.len) return error.InvalidColumnData;
        return self.data[start..end];
    }
};

fn readVarBytesBuffers(
    file: std.fs.File,
    block_offset: u64,
    desc: types.ColumnDesc,
    row_count: u32,
    allocator: Allocator,
) !VarBytesBuffers {
    if (desc.offsets_off == 0 or desc.lengths_off == 0) return error.InvalidColumnLayout;

    const reader = block_reader.BlockReader.init(file, allocator);
    const data = try reader.readColumnData(block_offset, desc, allocator);
    errdefer allocator.free(data);

    const offsets = try readU32Array(file, block_offset + @as(u64, desc.offsets_off), row_count, allocator);
    errdefer allocator.free(offsets);

    const lengths = try readU32Array(file, block_offset + @as(u64, desc.lengths_off), row_count, allocator);
    errdefer allocator.free(lengths);

    return .{ .data = data, .offsets = offsets, .lengths = lengths };
}

fn readU32Array(file: std.fs.File, offset: u64, count: u32, allocator: Allocator) ![]u32 {
    const total_bytes = @as(usize, count) * @sizeOf(u32);
    const buffer = try allocator.alloc(u8, total_bytes);
    defer allocator.free(buffer);

    const read_len = try file.preadAll(buffer, offset);
    if (read_len != buffer.len) return error.UnexpectedEof;

    const values = try allocator.alloc(u32, count);
    var i: usize = 0;
    while (i < values.len) : (i += 1) {
        const start = i * 4;
        values[i] = std.mem.readInt(u32, buffer[start..][0..4], .little);
    }
    return values;
}

// =============================================================================
// Tests
// =============================================================================

test "DocumentRecord encode/decode round-trip" {
    const allocator = std.testing.allocator;

    const original = DocumentRecord{
        .doc_id = "doc-uuid-123",
        .doc_type = "prompt",
        .title = "Code Review Assistant",
        .tags_text = "coding review rust",
        .doc_json = "{\"_sys\":{},\"data\":{\"system\":\"You are a code reviewer...\"}}",
        .parent_id = null,
        .marker = "active",
        .group_id = "tenant-1",
        .owner_id = "user-xyz",
        .created_at_ms = 1704067200000,
        .updated_at_ms = 1704153600000,
    };

    const blob = try encodeDocumentRecordKvBuf(allocator, original);
    defer allocator.free(blob);

    var decoded = (try decodeDocumentRecord(allocator, blob)).?;
    defer decoded.deinit(allocator);

    try std.testing.expectEqualStrings("doc-uuid-123", decoded.doc_id);
    try std.testing.expectEqualStrings("prompt", decoded.doc_type);
    try std.testing.expectEqualStrings("Code Review Assistant", decoded.title);
    try std.testing.expectEqualStrings("coding review rust", decoded.tags_text.?);
    try std.testing.expectEqualStrings("active", decoded.marker.?);
    try std.testing.expectEqualStrings("tenant-1", decoded.group_id.?);
    try std.testing.expectEqualStrings("user-xyz", decoded.owner_id.?);
    try std.testing.expectEqual(@as(i64, 1704067200000), decoded.created_at_ms);
    try std.testing.expectEqual(@as(i64, 1704153600000), decoded.updated_at_ms);
}

test "DocumentRecord encode/decode with optional fields null" {
    const allocator = std.testing.allocator;

    const original = DocumentRecord{
        .doc_id = "doc-minimal",
        .doc_type = "rag",
        .title = "Simple Document",
        .tags_text = null,
        .doc_json = "{}",
        .parent_id = null,
        .marker = null,
        .group_id = null,
        .owner_id = null,
        .created_at_ms = 1000,
        .updated_at_ms = 2000,
    };

    const blob = try encodeDocumentRecordKvBuf(allocator, original);
    defer allocator.free(blob);

    var decoded = (try decodeDocumentRecord(allocator, blob)).?;
    defer decoded.deinit(allocator);

    try std.testing.expectEqualStrings("doc-minimal", decoded.doc_id);
    try std.testing.expectEqualStrings("rag", decoded.doc_type);
    try std.testing.expectEqualStrings("Simple Document", decoded.title);
    try std.testing.expect(decoded.tags_text == null);
    try std.testing.expect(decoded.parent_id == null);
    try std.testing.expect(decoded.marker == null);
    try std.testing.expect(decoded.group_id == null);
    try std.testing.expect(decoded.owner_id == null);
}

test "DocumentRecord decode resolves external doc_json_ref" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(root_path);

    var blob_store = try db_blob_store.BlobStore.init(allocator, root_path);
    defer blob_store.deinit();

    const external_json = "{\"content\":\"externalized\"}";
    const blob_ref = try blob_store.put(external_json);

    const original = DocumentRecord{
        .doc_id = "doc-external",
        .doc_type = "prompt",
        .title = "External JSON",
        .doc_json = "",
        .created_at_ms = 1,
        .updated_at_ms = 2,
    };

    const blob = try encodeDocumentRecordKvBufWithStorage(
        allocator,
        original,
        null,
        blob_ref.refSlice(),
        null,
    );
    defer allocator.free(blob);

    var decoded = (try decodeDocumentRecordWithBlobStore(allocator, blob, &blob_store)).?;
    defer decoded.deinit(allocator);

    try std.testing.expectEqualStrings("doc-external", decoded.doc_id);
    try std.testing.expectEqualStrings(external_json, decoded.doc_json);
}

test "encodeDocumentRecordKvBufWithStorage stores doc_json_trigram_bloom when provided" {
    const allocator = std.testing.allocator;

    const record = DocumentRecord{
        .doc_id = "doc-bloom",
        .doc_type = "prompt",
        .title = "Bloom test",
        .doc_json = "",
        .created_at_ms = 1,
        .updated_at_ms = 1,
    };

    const bloom = buildDocJsonTrigramBloom("{\"content\":\"hello world\"}");
    const blob = try encodeDocumentRecordKvBufWithStorage(
        allocator,
        record,
        null,
        "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        bloom[0..],
    );
    defer allocator.free(blob);

    const reader = try kvbuf.KvBufReader.init(blob);
    const stored = reader.get(DocumentFieldIds.doc_json_trigram_bloom) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualSlices(u8, bloom[0..], stored);
}

test "DocumentAdapter.writeDocument externalizes large doc_json and reads transparently" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(root_path);

    var adapter = try DocumentAdapter.init(allocator, root_path);
    defer adapter.deinit();

    const large_len = default_doc_json_externalize_threshold_bytes + 128;
    const large_json = try allocator.alloc(u8, large_len);
    defer allocator.free(large_json);
    @memset(large_json, 'x');

    const record = DocumentRecord{
        .doc_id = "doc-large-1",
        .doc_type = "prompt",
        .title = "Large document",
        .doc_json = large_json,
        .created_at_ms = 100,
        .updated_at_ms = 100,
    };

    try adapter.writeDocument(record);
    try adapter.flush();

    var read_adapter = try DocumentAdapter.initReadOnly(allocator, root_path);
    defer read_adapter.deinitReadOnly();

    const loaded = try read_adapter.getDocument(allocator, "doc-large-1");
    try std.testing.expect(loaded != null);
    defer {
        var doc = loaded.?;
        doc.deinit(allocator);
    }

    try std.testing.expectEqual(large_json.len, loaded.?.doc_json.len);
    try std.testing.expectEqualSlices(u8, large_json, loaded.?.doc_json);
}

test "DocumentAdapter.getDocumentHeader and DocumentAdapter.loadBlob support lazy external doc_json" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(root_path);

    var adapter = try DocumentAdapter.init(allocator, root_path);
    defer adapter.deinit();
    adapter.doc_json_externalize_threshold_bytes = 64;

    const external_json =
        "{\"_sys\":{},\"data\":{\"text\":\"this payload should be externalized because it is longer than the configured threshold\"}}";
    const record = DocumentRecord{
        .doc_id = "doc-lazy-1",
        .doc_type = "prompt",
        .title = "Lazy header doc",
        .doc_json = external_json,
        .created_at_ms = 200,
        .updated_at_ms = 200,
    };

    try adapter.writeDocument(record);
    try adapter.flush();

    var read_adapter = try DocumentAdapter.initReadOnly(allocator, root_path);
    defer read_adapter.deinitReadOnly();

    const header_opt = try read_adapter.getDocumentHeader(allocator, "doc-lazy-1");
    try std.testing.expect(header_opt != null);
    var header = header_opt.?;
    defer header.deinit(allocator);

    try std.testing.expect(!header.has_inline_doc_json);
    try std.testing.expect(header.doc_json_ref != null);
    try std.testing.expect(std.mem.startsWith(u8, header.doc_json_ref.?, "sha256:"));

    const loaded_blob = try read_adapter.loadBlob(allocator, header.doc_json_ref.?);
    defer allocator.free(loaded_blob);
    try std.testing.expectEqualStrings(external_json, loaded_blob);
}

test "DocumentAdapter externalized doc_json can use multipart refs and loadBlob joins parts" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(root_path);

    var adapter = try DocumentAdapter.init(allocator, root_path);
    defer adapter.deinit();
    adapter.doc_json_externalize_threshold_bytes = 16;
    adapter.blob_store.multipart_chunk_size_bytes = 32;

    const external_json =
        "{\"_sys\":{},\"data\":{\"text\":\"this payload should be split into multipart chunks for blob storage integration testing\"}}";
    const record = DocumentRecord{
        .doc_id = "doc-multi-1",
        .doc_type = "prompt",
        .title = "Multipart header doc",
        .doc_json = external_json,
        .created_at_ms = 300,
        .updated_at_ms = 300,
    };

    try adapter.writeDocument(record);
    try adapter.flush();

    var read_adapter = try DocumentAdapter.initReadOnly(allocator, root_path);
    defer read_adapter.deinitReadOnly();

    const header_opt = try read_adapter.getDocumentHeader(allocator, "doc-multi-1");
    try std.testing.expect(header_opt != null);
    var header = header_opt.?;
    defer header.deinit(allocator);

    try std.testing.expect(!header.has_inline_doc_json);
    try std.testing.expect(header.doc_json_ref != null);
    try std.testing.expect(std.mem.startsWith(u8, header.doc_json_ref.?, db_blob_store.multipart_ref_prefix));

    const loaded_blob = try read_adapter.loadBlob(allocator, header.doc_json_ref.?);
    defer allocator.free(loaded_blob);
    try std.testing.expectEqualStrings(external_json, loaded_blob);
}

test "DocumentAdapter.openBlobStream streams externalized multipart doc_json" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(root_path);

    var adapter = try DocumentAdapter.init(allocator, root_path);
    defer adapter.deinit();
    adapter.doc_json_externalize_threshold_bytes = 16;
    adapter.blob_store.multipart_chunk_size_bytes = 24;

    const external_json =
        "{\"_sys\":{},\"data\":{\"text\":\"stream this multipart document payload without rebuilding API contracts later\"}}";
    const record = DocumentRecord{
        .doc_id = "doc-stream-1",
        .doc_type = "prompt",
        .title = "Stream doc",
        .doc_json = external_json,
        .created_at_ms = 400,
        .updated_at_ms = 400,
    };

    try adapter.writeDocument(record);
    try adapter.flush();

    var read_adapter = try DocumentAdapter.initReadOnly(allocator, root_path);
    defer read_adapter.deinitReadOnly();

    const header_opt = try read_adapter.getDocumentHeader(allocator, "doc-stream-1");
    try std.testing.expect(header_opt != null);
    var header = header_opt.?;
    defer header.deinit(allocator);
    try std.testing.expect(header.doc_json_ref != null);

    var stream = try read_adapter.openBlobStream(allocator, header.doc_json_ref.?);
    defer stream.deinit();

    var out = std.ArrayList(u8).empty;
    defer out.deinit(allocator);

    var buf: [13]u8 = undefined; // Scratch buffer, overwritten by stream.read each iteration
    while (true) {
        const read_len = try stream.read(&buf);
        if (read_len == 0) break;
        try out.appendSlice(allocator, buf[0..read_len]);
    }

    try std.testing.expectEqualStrings(external_json, out.items);
}

test "getDocumentHeader wrapper reports inline payload for small doc_json" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(root_path);

    {
        var adapter = try DocumentAdapter.init(allocator, root_path);
        defer adapter.deinit();

        const record = DocumentRecord{
            .doc_id = "doc-inline-1",
            .doc_type = "prompt",
            .title = "Inline header doc",
            .doc_json = "{\"small\":true}",
            .created_at_ms = 1,
            .updated_at_ms = 1,
        };
        try adapter.writeDocument(record);
        try adapter.flush();
    }

    const header_opt = try getDocumentHeader(allocator, root_path, "doc-inline-1");
    try std.testing.expect(header_opt != null);
    defer freeDocumentHeader(allocator, header_opt.?);

    try std.testing.expect(header_opt.?.has_inline_doc_json);
    try std.testing.expect(header_opt.?.doc_json_ref == null);
}

test "loadDocumentBlob wrapper reads externalized payload" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(root_path);

    const external_json = "{\"big\":true,\"payload\":\"abcdefghijklmnopqrstuvwxyz0123456789\"}";

    {
        var adapter = try DocumentAdapter.init(allocator, root_path);
        defer adapter.deinit();
        adapter.doc_json_externalize_threshold_bytes = 32;

        const record = DocumentRecord{
            .doc_id = "doc-wrapper-blob-1",
            .doc_type = "prompt",
            .title = "Wrapper blob doc",
            .doc_json = external_json,
            .created_at_ms = 2,
            .updated_at_ms = 2,
        };
        try adapter.writeDocument(record);
        try adapter.flush();
    }

    const header_opt = try getDocumentHeader(allocator, root_path, "doc-wrapper-blob-1");
    try std.testing.expect(header_opt != null);
    defer freeDocumentHeader(allocator, header_opt.?);

    try std.testing.expect(header_opt.?.doc_json_ref != null);
    const blob_ref = header_opt.?.doc_json_ref.?;
    const loaded = try loadDocumentBlob(allocator, root_path, blob_ref);
    defer allocator.free(loaded);

    try std.testing.expectEqualStrings(external_json, loaded);
}

test "computeHash is deterministic" {
    const h1 = computeHash("test-document");
    const h2 = computeHash("test-document");
    const h3 = computeHash("different-document");

    try std.testing.expectEqual(h1, h2);
    try std.testing.expect(h1 != h3);
}

test "computeOptionalHash handles null" {
    const h1 = computeOptionalHash(null);
    const h2 = computeOptionalHash("test");

    try std.testing.expectEqual(@as(u64, 0), h1);
    try std.testing.expect(h2 != 0);
}

test "parseThresholdBytes parses valid decimal thresholds" {
    try std.testing.expectEqual(@as(?usize, 0), parseThresholdBytes("0"));
    try std.testing.expectEqual(@as(?usize, 1048576), parseThresholdBytes("1048576"));
    try std.testing.expectEqual(@as(?usize, 256), parseThresholdBytes(" 256 "));
}

test "parseThresholdBytes rejects invalid threshold values" {
    try std.testing.expectEqual(@as(?usize, null), parseThresholdBytes(""));
    try std.testing.expectEqual(@as(?usize, null), parseThresholdBytes("  "));
    try std.testing.expectEqual(@as(?usize, null), parseThresholdBytes("-1"));
    try std.testing.expectEqual(@as(?usize, null), parseThresholdBytes("abc"));
    try std.testing.expectEqual(@as(?usize, null), parseThresholdBytes("12MB"));
}

test "buildDocJsonTrigramBloom indicates possible matches for present substrings" {
    const bloom = buildDocJsonTrigramBloom("{\"content\":\"hello world\"}");
    try std.testing.expect(mayContainSubstringByTrigramBloom(bloom[0..], "hello"));
    try std.testing.expect(mayContainSubstringByTrigramBloom(bloom[0..], "world"));
}

test "mayContainSubstringByTrigramBloom rejects impossible queries with empty bloom" {
    const bloom = std.mem.zeroes([doc_json_trigram_bloom_bytes]u8);
    try std.testing.expect(!mayContainSubstringByTrigramBloom(bloom[0..], "xyz"));
    try std.testing.expect(mayContainSubstringByTrigramBloom(bloom[0..], "xy"));
}

test "textFindInsensitive finds match" {
    // Basic match
    try std.testing.expectEqual(@as(?usize, 0), textFindInsensitive("Hello World", "hello"));
    try std.testing.expectEqual(@as(?usize, 6), textFindInsensitive("Hello World", "world"));
    try std.testing.expectEqual(@as(?usize, 0), textFindInsensitive("HELLO", "hello"));
    try std.testing.expectEqual(@as(?usize, 0), textFindInsensitive("hello", "HELLO"));

    // No match
    try std.testing.expect(textFindInsensitive("Hello", "xyz") == null);
    try std.testing.expect(textFindInsensitive("Hi", "Hello") == null);

    // Empty needle always matches at 0
    try std.testing.expectEqual(@as(?usize, 0), textFindInsensitive("Hello", ""));
}

test "extractSnippet returns context around match" {
    const allocator = std.testing.allocator;

    // Match at start
    const snippet1 = try extractSnippet("Hello World", "hello", allocator);
    defer if (snippet1) |s| allocator.free(s);
    try std.testing.expect(snippet1 != null);
    try std.testing.expect(std.mem.startsWith(u8, snippet1.?, "Hello"));

    // No match returns null
    const snippet2 = try extractSnippet("Hello World", "xyz", allocator);
    try std.testing.expect(snippet2 == null);
}

test "ChangeAction enum values" {
    try std.testing.expectEqual(@as(u8, 1), @intFromEnum(ChangeAction.create));
    try std.testing.expectEqual(@as(u8, 2), @intFromEnum(ChangeAction.update));
    try std.testing.expectEqual(@as(u8, 3), @intFromEnum(ChangeAction.delete));
}

test "ChangeRecord memory management" {
    const allocator = std.testing.allocator;

    var record = ChangeRecord{
        .seq_num = 100,
        .doc_id = try allocator.dupe(u8, "test-doc-123"),
        .action = .create,
        .timestamp_ms = 1704067200000,
        .doc_type = try allocator.dupe(u8, "prompt"),
        .title = try allocator.dupe(u8, "Test Title"),
    };
    defer record.deinit(allocator);

    try std.testing.expectEqual(@as(u64, 100), record.seq_num);
    try std.testing.expectEqualStrings("test-doc-123", record.doc_id);
    try std.testing.expectEqual(ChangeAction.create, record.action);
    try std.testing.expectEqualStrings("prompt", record.doc_type.?);
    try std.testing.expectEqualStrings("Test Title", record.title.?);
}

// =============================================================================
// High-Level API Functions (for capi thin wrappers)
// =============================================================================

/// List all documents, optionally filtered by type, group, owner, and marker.
/// Handles adapter lifecycle internally.
/// Caller owns returned records; free each with DocumentRecord.deinit().
pub fn listDocuments(
    alloc: Allocator,
    db_path: []const u8,
    doc_type: ?[]const u8,
    group_id: ?[]const u8,
    owner_id: ?[]const u8,
    marker: ?[]const u8,
) ![]DocumentRecord {
    var adapter = try DocumentAdapter.initReadOnly(alloc, db_path);
    defer adapter.deinitReadOnly();
    return adapter.scanDocuments(alloc, doc_type, group_id, owner_id, marker);
}

/// Get a single document by ID.
/// Handles adapter lifecycle internally.
/// Returns null if document not found.
/// Caller owns returned record; free with DocumentRecord.deinit().
pub fn getDocument(alloc: Allocator, db_path: []const u8, doc_id: []const u8) !?DocumentRecord {
    var adapter = try DocumentAdapter.initReadOnly(alloc, db_path);
    defer adapter.deinitReadOnly();
    return adapter.getDocument(alloc, doc_id);
}

/// Get a single document header by ID without loading externalized doc_json.
/// Handles adapter lifecycle internally.
/// Returns null if document not found.
/// Caller owns returned header; free with freeDocumentHeader().
pub fn getDocumentHeader(alloc: Allocator, db_path: []const u8, doc_id: []const u8) !?DocumentHeader {
    var adapter = try DocumentAdapter.initReadOnly(alloc, db_path);
    defer adapter.deinitReadOnly();
    return adapter.getDocumentHeader(alloc, doc_id);
}

/// Load an externalized blob by reference (`sha256:<hex>` or `multi:<hex>`).
/// Handles adapter lifecycle internally.
/// Caller owns returned bytes; free with allocator.free().
pub fn loadDocumentBlob(alloc: Allocator, db_path: []const u8, blob_ref: []const u8) ![]u8 {
    var adapter = try DocumentAdapter.initReadOnly(alloc, db_path);
    defer adapter.deinitReadOnly();
    return adapter.loadBlob(alloc, blob_ref);
}

/// Create a new document.
/// Handles adapter lifecycle internally.
/// Returns error.LockUnavailable if another process holds the database lock.
pub fn createDocument(alloc: Allocator, db_path: []const u8, record: DocumentRecord) !void {
    var adapter = try DocumentAdapter.init(alloc, db_path);
    defer adapter.deinit();
    try adapter.writeDocument(record);
    try adapter.flush();
}

/// Update an existing document.
/// Handles adapter lifecycle internally.
/// Returns error.DocumentNotFound if document doesn't exist.
/// Returns error.LockUnavailable if another process holds the database lock.
pub fn updateDocument(
    alloc: Allocator,
    db_path: []const u8,
    doc_id: []const u8,
    title: ?[]const u8,
    tags_text: ?[]const u8,
    doc_json: ?[]const u8,
    marker: ?[]const u8,
) !void {
    var adapter = try DocumentAdapter.init(alloc, db_path);
    defer adapter.deinit();

    var existing = try adapter.getDocument(alloc, doc_id) orelse return error.DocumentNotFound;
    defer existing.deinit(alloc);

    const updated = DocumentRecord{
        .doc_id = existing.doc_id,
        .doc_type = existing.doc_type,
        .title = title orelse existing.title,
        .tags_text = tags_text orelse existing.tags_text,
        .doc_json = doc_json orelse existing.doc_json,
        .parent_id = existing.parent_id,
        .marker = marker orelse existing.marker,
        .group_id = existing.group_id,
        .owner_id = existing.owner_id,
        .created_at_ms = existing.created_at_ms,
        .updated_at_ms = std.time.milliTimestamp(),
    };

    try adapter.writeDocument(updated);
    try adapter.flush();
}

/// Delete a document (soft delete by writing tombstone).
/// Handles adapter lifecycle internally.
/// Returns error.LockUnavailable if another process holds the database lock.
/// Returns error.DocumentNotFound if the document does not exist.
pub fn deleteDocument(alloc: Allocator, db_path: []const u8, doc_id: []const u8) !void {
    var adapter = try DocumentAdapter.init(alloc, db_path);
    defer adapter.deinit();

    // Check if document exists first
    var existing = try adapter.getDocument(alloc, doc_id) orelse return error.DocumentNotFound;
    existing.deinit(alloc);

    const now_ms = std.time.milliTimestamp();
    try adapter.deleteDocument(doc_id, now_ms);
    try adapter.flush();
}

/// Batch-delete multiple documents (soft delete by writing tombstones).
/// Handles adapter lifecycle internally — opens once, writes all tombstones, flushes once.
/// Non-existent document IDs are silently skipped (idempotent batch semantics).
/// When `doc_type` is non-null, only documents matching that type are deleted;
/// mismatched documents are silently skipped (same as non-existent).
/// Returns the number of documents actually deleted.
pub fn deleteDocumentsBatch(alloc: Allocator, db_path: []const u8, doc_ids: []const []const u8, doc_type: ?[]const u8) !usize {
    var adapter = try DocumentAdapter.init(alloc, db_path);
    defer adapter.deinit();

    const now_ms = std.time.milliTimestamp();
    var count: usize = 0;
    for (doc_ids) |doc_id| {
        var existing = try adapter.getDocument(alloc, doc_id) orelse continue;
        defer existing.deinit(alloc);
        if (doc_type) |dt| {
            if (!std.mem.eql(u8, existing.doc_type, dt)) continue;
        }
        try adapter.deleteDocument(doc_id, now_ms);
        count += 1;
    }
    try adapter.flush();
    return count;
}

/// Batch-update the marker field for multiple documents.
/// Handles adapter lifecycle internally — opens once, writes all updates, flushes once.
/// Non-existent document IDs are silently skipped.
/// When `doc_type` is non-null, only documents matching that type are updated;
/// mismatched documents are silently skipped (same as non-existent).
/// Returns the number of documents actually updated.
pub fn setMarkerBatch(alloc: Allocator, db_path: []const u8, doc_ids: []const []const u8, marker: []const u8, doc_type: ?[]const u8) !usize {
    var adapter = try DocumentAdapter.init(alloc, db_path);
    defer adapter.deinit();

    const now_ms = std.time.milliTimestamp();
    var count: usize = 0;
    for (doc_ids) |doc_id| {
        var existing = try adapter.getDocument(alloc, doc_id) orelse continue;
        defer existing.deinit(alloc);
        if (doc_type) |dt| {
            if (!std.mem.eql(u8, existing.doc_type, dt)) continue;
        }

        const updated = DocumentRecord{
            .doc_id = existing.doc_id,
            .doc_type = existing.doc_type,
            .title = existing.title,
            .tags_text = existing.tags_text,
            .doc_json = existing.doc_json,
            .parent_id = existing.parent_id,
            .marker = marker,
            .group_id = existing.group_id,
            .owner_id = existing.owner_id,
            .created_at_ms = existing.created_at_ms,
            .updated_at_ms = now_ms,
            .expires_at_ms = existing.expires_at_ms,
            .content_hash = existing.content_hash,
            .seq_num = existing.seq_num,
            .meta_i1 = existing.meta_i1,
            .meta_i2 = existing.meta_i2,
            .meta_i3 = existing.meta_i3,
            .meta_i4 = existing.meta_i4,
            .meta_i5 = existing.meta_i5,
            .meta_f1 = existing.meta_f1,
            .meta_f2 = existing.meta_f2,
            .meta_f3 = existing.meta_f3,
            .meta_f4 = existing.meta_f4,
            .meta_f5 = existing.meta_f5,
            .version_type = existing.version_type,
            .base_doc_id = existing.base_doc_id,
        };

        try adapter.writeDocument(updated);
        count += 1;
    }
    try adapter.flush();
    return count;
}

/// Get document version history.
/// Handles adapter lifecycle internally.
/// Caller owns returned records; free each with DocumentRecord.deinit().
pub fn listDocumentVersions(alloc: Allocator, db_path: []const u8, doc_id: []const u8) ![]DocumentRecord {
    var adapter = try DocumentAdapter.initReadOnly(alloc, db_path);
    defer adapter.deinitReadOnly();
    return adapter.getDocumentVersions(alloc, doc_id);
}

/// Free a slice of DocumentRecords.
pub fn freeDocumentRecords(alloc: Allocator, records: []DocumentRecord) void {
    for (records) |*r| {
        var rec = r.*;
        rec.deinit(alloc);
    }
    alloc.free(records);
}

/// Free a single DocumentHeader returned by getDocumentHeader().
pub fn freeDocumentHeader(alloc: Allocator, header: DocumentHeader) void {
    var owned = header;
    owned.deinit(alloc);
}

/// Search documents by content.
/// Handles adapter lifecycle internally.
/// Caller owns returned results; free each with SearchResult.deinit().
pub fn searchDocuments(
    alloc: Allocator,
    db_path: []const u8,
    query: []const u8,
    doc_type: ?[]const u8,
) ![]SearchResult {
    var adapter = try DocumentAdapter.initReadOnly(alloc, db_path);
    defer adapter.deinitReadOnly();
    return adapter.searchDocuments(alloc, query, doc_type);
}

/// Free a slice of SearchResults.
pub fn freeSearchResults(alloc: Allocator, results: []SearchResult) void {
    for (results) |*r| {
        var res = r.*;
        res.deinit(alloc);
    }
    alloc.free(results);
}

/// Batch search documents by content.
/// Handles adapter lifecycle internally.
/// Returns a map of query_id → [doc_ids].
/// Caller owns returned results; free each with BatchSearchResult.deinit().
pub fn searchDocumentsBatch(
    alloc: Allocator,
    db_path: []const u8,
    queries: []const BatchQuery,
) ![]BatchSearchResult {
    var adapter = try DocumentAdapter.initReadOnly(alloc, db_path);
    defer adapter.deinitReadOnly();
    return adapter.searchDocumentsBatch(alloc, queries);
}

/// Free a slice of BatchSearchResults.
pub fn freeBatchSearchResults(alloc: Allocator, results: []BatchSearchResult) void {
    for (results) |*r| {
        var res = r.*;
        res.deinit(alloc);
    }
    alloc.free(results);
}

/// Parse BatchQuery array from JSON: [{"id": "q1", "text": "search", "type": "prompt"}, ...]
/// Caller owns returned slice; free each query with freeBatchQueries.
pub fn parseBatchQueriesJson(alloc: Allocator, json: []const u8) ![]BatchQuery {
    const parsed = std.json.parseFromSlice(std.json.Value, alloc, json, .{}) catch return error.InvalidJson;
    defer parsed.deinit();
    const array = switch (parsed.value) {
        .array => |arr| arr,
        else => return error.InvalidJson,
    };
    const queries = try alloc.alloc(BatchQuery, array.items.len);
    errdefer alloc.free(queries);
    var initialized: usize = 0;
    errdefer for (queries[0..initialized]) |*q| {
        alloc.free(q.id);
        alloc.free(q.text);
        if (q.doc_type) |t| alloc.free(t);
    };
    for (array.items, 0..) |item, i| {
        const obj = switch (item) {
            .object => |o| o,
            else => return error.InvalidJson,
        };
        const id_val = obj.get("id") orelse return error.InvalidJson;
        const text_val = obj.get("text") orelse return error.InvalidJson;
        const id_str = switch (id_val) {
            .string => |s| s,
            else => return error.InvalidJson,
        };
        const text_str = switch (text_val) {
            .string => |s| s,
            else => return error.InvalidJson,
        };
        var doc_type: ?[]const u8 = null;
        if (obj.get("type")) |type_val| {
            doc_type = switch (type_val) {
                .string => |s| try alloc.dupe(u8, s),
                else => null,
            };
        }
        queries[i] = .{
            .id = try alloc.dupe(u8, id_str),
            .text = try alloc.dupe(u8, text_str),
            .doc_type = doc_type,
        };
        initialized += 1;
    }
    return queries;
}

/// Free a slice of BatchQueries returned by parseBatchQueriesJson.
pub fn freeBatchQueries(alloc: Allocator, queries: []BatchQuery) void {
    for (queries) |*q| {
        alloc.free(q.id);
        alloc.free(q.text);
        if (q.doc_type) |t| alloc.free(t);
    }
    alloc.free(queries);
}

/// Get changes since a given sequence number.
/// Handles adapter lifecycle internally.
/// Returns change records ordered by seq_num ascending.
/// Use since_seq=0 to get all changes from the beginning.
/// Caller owns returned records; free each with ChangeRecord.deinit().
pub fn getChanges(
    alloc: Allocator,
    db_path: []const u8,
    since_seq: u64,
    group_id: ?[]const u8,
    limit: usize,
) ![]ChangeRecord {
    var adapter = try DocumentAdapter.initReadOnly(alloc, db_path);
    defer adapter.deinitReadOnly();
    return adapter.getChanges(alloc, since_seq, group_id, limit);
}

/// Free a slice of ChangeRecords.
pub fn freeChangeRecords(alloc: Allocator, records: []ChangeRecord) void {
    for (records) |*r| {
        var rec = r.*;
        rec.deinit(alloc);
    }
    alloc.free(records);
}

/// Set or update the TTL for a document.
/// ttl_seconds: Time-to-live in seconds from now. 0 = remove TTL (never expires).
/// Handles adapter lifecycle internally.
/// Returns error.DocumentNotFound if document doesn't exist.
pub fn setDocumentTTL(
    alloc: Allocator,
    db_path: []const u8,
    doc_id: []const u8,
    ttl_seconds: u64,
) !void {
    var adapter = try DocumentAdapter.init(alloc, db_path);
    defer adapter.deinit();

    var existing = try adapter.getDocument(alloc, doc_id) orelse return error.DocumentNotFound;
    defer existing.deinit(alloc);

    // Calculate new expiration time
    const now_ms = std.time.milliTimestamp();
    const expires_at_ms: i64 = if (ttl_seconds == 0) 0 else now_ms + @as(i64, @intCast(ttl_seconds * 1000));

    // Create updated record with new expiration
    const updated = DocumentRecord{
        .doc_id = existing.doc_id,
        .doc_type = existing.doc_type,
        .title = existing.title,
        .tags_text = existing.tags_text,
        .doc_json = existing.doc_json,
        .parent_id = existing.parent_id,
        .marker = existing.marker,
        .group_id = existing.group_id,
        .owner_id = existing.owner_id,
        .created_at_ms = existing.created_at_ms,
        .updated_at_ms = now_ms,
        .expires_at_ms = expires_at_ms,
    };

    try adapter.writeDocument(updated);
    try adapter.flush();
}

/// Count expired documents in the database.
/// Returns the number of documents that have expired (expires_at > 0 and < current time).
/// Handles adapter lifecycle internally.
pub fn countExpiredDocuments(alloc: Allocator, db_path: []const u8) !usize {
    var adapter = try DocumentAdapter.initReadOnly(alloc, db_path);
    defer adapter.deinitReadOnly();
    return adapter.countExpired(alloc);
}

// =============================================================================
// Delta Versioning High-Level API
// =============================================================================

/// Create a delta version of an existing document.
/// The delta contains only the changed JSON content.
/// Handles adapter lifecycle internally.
/// Returns error.DocumentNotFound if base document doesn't exist.
/// Returns error.LockUnavailable if another process holds the database lock.
pub fn createDeltaVersion(
    alloc: Allocator,
    db_path: []const u8,
    base_doc_id: []const u8,
    new_doc_id: []const u8,
    delta_json: []const u8,
    title: ?[]const u8,
    tags_text: ?[]const u8,
    marker: ?[]const u8,
) !void {
    var adapter = try DocumentAdapter.init(alloc, db_path);
    defer adapter.deinit();
    try adapter.createDeltaVersion(alloc, base_doc_id, new_doc_id, delta_json, title, tags_text, marker);
    try adapter.flush();
}

/// Get the delta chain for a document.
/// Returns documents in order from the requested document back to the base.
/// First element is the requested document, last element is the base (full version).
/// Handles adapter lifecycle internally.
/// Caller owns returned records; free each with DocumentRecord.deinit().
pub fn getDeltaChain(
    alloc: Allocator,
    db_path: []const u8,
    doc_id: []const u8,
) ![]DocumentRecord {
    var adapter = try DocumentAdapter.initReadOnly(alloc, db_path);
    defer adapter.deinitReadOnly();
    return adapter.getDeltaChain(alloc, doc_id);
}

/// Free a delta chain returned by getDeltaChain.
pub fn freeDeltaChain(alloc: Allocator, records: []DocumentRecord) void {
    for (records) |*r| {
        var rec = r.*;
        rec.deinit(alloc);
    }
    alloc.free(records);
}

/// Check if a document is a delta version.
/// Handles adapter lifecycle internally.
/// Returns error.DocumentNotFound if document doesn't exist.
pub fn isDeltaVersion(alloc: Allocator, db_path: []const u8, doc_id: []const u8) !bool {
    var adapter = try DocumentAdapter.initReadOnly(alloc, db_path);
    defer adapter.deinitReadOnly();
    return adapter.isDeltaVersion(alloc, doc_id);
}

/// Get the base document ID for a delta version.
/// Returns null if the document is not a delta or doesn't exist.
/// Handles adapter lifecycle internally.
/// Caller owns returned string; free with allocator.free().
pub fn getBaseDocumentId(alloc: Allocator, db_path: []const u8, doc_id: []const u8) !?[]const u8 {
    var adapter = try DocumentAdapter.initReadOnly(alloc, db_path);
    defer adapter.deinitReadOnly();
    return adapter.getBaseDocumentId(alloc, doc_id);
}

// =============================================================================
// Compaction/Garbage Collection High-Level API
// =============================================================================

/// Re-export CompactionStats for external use.
pub const CompactionStats = DocumentAdapter.CompactionStats;

/// Get compaction statistics for the document storage.
/// Shows counts of active, expired, deleted documents and estimated reclaimable space.
/// Handles adapter lifecycle internally.
pub fn getCompactionStats(alloc: Allocator, db_path: []const u8) !CompactionStats {
    var adapter = try DocumentAdapter.initReadOnly(alloc, db_path);
    defer adapter.deinitReadOnly();
    return adapter.getCompactionStats(alloc);
}

/// Purge expired documents by writing tombstones for them.
/// This makes the deletions explicit and visible in the change feed.
/// Returns the number of documents purged.
/// Handles adapter lifecycle internally.
pub fn purgeExpiredDocuments(alloc: Allocator, db_path: []const u8) !usize {
    var adapter = try DocumentAdapter.init(alloc, db_path);
    defer adapter.deinit();
    const count = try adapter.purgeExpired(alloc);
    try adapter.flush();
    return count;
}

/// Get list of document IDs that are candidates for garbage collection.
/// Returns IDs of expired and deleted documents.
/// Handles adapter lifecycle internally.
/// Caller owns returned slice; free each string and the slice with allocator.
pub fn getGarbageCandidates(alloc: Allocator, db_path: []const u8) ![][]const u8 {
    var adapter = try DocumentAdapter.initReadOnly(alloc, db_path);
    defer adapter.deinitReadOnly();
    return adapter.getGarbageCandidates(alloc);
}

/// Free a slice of string IDs returned by getGarbageCandidates.
pub fn freeStringIds(alloc: Allocator, ids: [][]const u8) void {
    for (ids) |id| alloc.free(id);
    alloc.free(ids);
}
