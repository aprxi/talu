//! TaluDB adapter for document-tag junction persistence.
//!
//! Manages the many-to-many relationship between documents and tags.
//! Schema 13 stores doc_id → tag_id associations (forward direction).
//! Schema 14 stores tag_id → doc_id associations (inverted direction).
//!
//! Schema 13: DocumentTag junction table (forward: doc → tags)
//! Schema 14: TagDocumentIndex inverted table (inverted: tag → docs)

const std = @import("std");
const kvbuf = @import("../../io/kvbuf/root.zig");
const db_writer = @import("../writer.zig");
const db_reader = @import("../reader.zig");
const block_reader = @import("../block_reader.zig");
const types = @import("../types.zig");

const Allocator = std.mem.Allocator;
const ColumnValue = db_writer.ColumnValue;
const DocumentTagFieldIds = kvbuf.DocumentTagFieldIds;
const TagDocumentIndexFieldIds = kvbuf.TagDocumentIndexFieldIds;

const schema_document_tags: u16 = 13;
const schema_tag_document_index: u16 = 14;

// Column IDs for DocumentTag junction (Schema 13)
const col_doc_hash: u32 = 1;
const col_tag_hash: u32 = 2;
const col_ts: u32 = 3;
const col_group_hash: u32 = 4;
const col_payload: u32 = 10;

/// Document-Tag junction record.
pub const DocumentTagRecord = struct {
    doc_id: []const u8,
    tag_id: []const u8,
    added_at_ms: i64,

    pub fn deinit(self: *DocumentTagRecord, allocator: Allocator) void {
        allocator.free(self.doc_id);
        allocator.free(self.tag_id);
    }
};

/// DocumentTagAdapter - TaluDB adapter for document-tag associations.
///
/// Provides operations for managing document-tag relationships.
/// Maintains both forward (doc → tags) and inverted (tag → docs) indices.
/// Thread safety: NOT thread-safe (single-writer semantics via lock).
pub const DocumentTagAdapter = struct {
    allocator: Allocator,
    fs_writer: *db_writer.Writer,
    fs_reader: *db_reader.Reader,
    read_only: bool,

    /// Initialize a TaluDB-backed document-tag adapter with write capabilities.
    pub fn init(allocator: Allocator, db_root: []const u8) !DocumentTagAdapter {
        var writer_ptr = try allocator.create(db_writer.Writer);
        errdefer allocator.destroy(writer_ptr);
        writer_ptr.* = try db_writer.Writer.open(allocator, db_root, "docs");
        errdefer writer_ptr.deinit();

        var reader_ptr = try allocator.create(db_reader.Reader);
        errdefer allocator.destroy(reader_ptr);
        reader_ptr.* = try db_reader.Reader.open(allocator, db_root, "docs");
        errdefer reader_ptr.deinit();

        return .{
            .allocator = allocator,
            .fs_writer = writer_ptr,
            .fs_reader = reader_ptr,
            .read_only = false,
        };
    }

    /// Initialize a read-only adapter for scanning document-tag associations.
    pub fn initReadOnly(allocator: Allocator, db_root: []const u8) !DocumentTagAdapter {
        const reader_ptr = try allocator.create(db_reader.Reader);
        errdefer allocator.destroy(reader_ptr);
        reader_ptr.* = try db_reader.Reader.open(allocator, db_root, "docs");

        return .{
            .allocator = allocator,
            .fs_writer = undefined,
            .fs_reader = reader_ptr,
            .read_only = true,
        };
    }

    pub fn deinit(self: *DocumentTagAdapter) void {
        if (!self.read_only) {
            self.fs_writer.flushBlock() catch {};
            self.fs_writer.deinit();
            self.allocator.destroy(self.fs_writer);
        }
        self.fs_reader.deinit();
        self.allocator.destroy(self.fs_reader);
    }

    pub fn deinitReadOnly(self: *DocumentTagAdapter) void {
        self.fs_reader.deinit();
        self.allocator.destroy(self.fs_reader);
    }

    // =========================================================================
    // Document-Tag Junction Operations
    // =========================================================================

    /// Add a tag to a document.
    /// Writes to both Schema 13 (forward) and Schema 14 (inverted).
    pub fn addDocumentTag(self: *DocumentTagAdapter, doc_id: []const u8, tag_id: []const u8, added_at_ms: i64, group_id: ?[]const u8) !void {
        // Write forward index (Schema 13: doc → tag)
        try self.writeForwardIndex(doc_id, tag_id, added_at_ms, group_id);

        // Flush before switching schemas (13 → 14)
        try self.fs_writer.flushBlock();

        // Write inverted index (Schema 14: tag → doc)
        try self.writeInvertedIndex(tag_id, doc_id, added_at_ms, group_id);
    }

    /// Remove a tag from a document (writes tombstones with negative timestamp).
    /// Writes to both Schema 13 (forward) and Schema 14 (inverted).
    pub fn removeDocumentTag(self: *DocumentTagAdapter, doc_id: []const u8, tag_id: []const u8, removed_at_ms: i64, group_id: ?[]const u8) !void {
        // Write forward tombstone (negative timestamp)
        try self.writeForwardIndex(doc_id, tag_id, -removed_at_ms, group_id);

        // Flush before switching schemas (13 → 14)
        try self.fs_writer.flushBlock();

        // Write inverted tombstone (negative timestamp)
        try self.writeInvertedIndex(tag_id, doc_id, -removed_at_ms, group_id);
    }

    /// Get all tag IDs for a document.
    pub fn getDocumentTags(self: *DocumentTagAdapter, allocator: Allocator, doc_id: []const u8) ![][]const u8 {
        const target_doc_hash = computeHash(doc_id);

        // Track active tags (positive added_at_ms = active, negative = removed)
        var tag_states = std.StringHashMap(i64).init(allocator);
        defer {
            var it = tag_states.keyIterator();
            while (it.next()) |k| allocator.free(k.*);
            tag_states.deinit();
        }

        const blocks = try self.fs_reader.getBlocks(allocator);
        defer allocator.free(blocks);

        for (blocks) |block| {
            var file = self.fs_reader.dir.openFile(block.path, .{ .mode = .read_only }) catch continue;
            defer file.close();

            const reader = block_reader.BlockReader.init(file, allocator);
            const header = reader.readHeader(block.offset) catch continue;
            if (header.schema_id != schema_document_tags) continue;

            const descs = reader.readColumnDirectory(header, block.offset) catch continue;
            defer allocator.free(descs);

            const row_count = header.row_count;
            if (row_count == 0) continue;

            const doc_desc = findColumn(descs, col_doc_hash) orelse continue;
            const ts_desc = findColumn(descs, col_ts) orelse continue;
            const payload_desc = findColumn(descs, col_payload) orelse continue;

            const doc_bytes = reader.readColumnData(block.offset, doc_desc, allocator) catch continue;
            defer allocator.free(doc_bytes);

            const ts_bytes = reader.readColumnData(block.offset, ts_desc, allocator) catch continue;
            defer allocator.free(ts_bytes);

            var payload_buffers = readVarBytesBuffers(file, block.offset, payload_desc, row_count, allocator) catch continue;
            defer payload_buffers.deinit(allocator);

            for (0..row_count) |row_idx| {
                const doc_hash = readU64At(doc_bytes, row_idx) catch continue;
                if (doc_hash != target_doc_hash) continue;

                const ts = readI64At(ts_bytes, row_idx) catch continue;
                const payload = payload_buffers.sliceForRow(row_idx) catch continue;

                const record_opt = decodeDocumentTagRecord(allocator, payload) catch continue;
                if (record_opt) |record| {
                    defer allocator.free(record.doc_id);

                    // Check if this is newer than existing state
                    const abs_ts = if (ts >= 0) ts else -ts;
                    if (tag_states.get(record.tag_id)) |existing_ts| {
                        const existing_abs = if (existing_ts >= 0) existing_ts else -existing_ts;
                        if (abs_ts <= existing_abs) {
                            allocator.free(record.tag_id);
                            continue;
                        }
                        // Remove old key
                        if (tag_states.fetchRemove(record.tag_id)) |kv| {
                            allocator.free(kv.key);
                        }
                    }

                    tag_states.put(record.tag_id, record.added_at_ms) catch {
                        allocator.free(record.tag_id);
                        continue;
                    };
                }
            }
        }

        // Collect active tags (positive added_at_ms)
        var results = std.ArrayList([]const u8).empty;
        errdefer {
            for (results.items) |r| allocator.free(r);
            results.deinit(allocator);
        }

        var it = tag_states.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.* > 0) {
                const tag_copy = try allocator.dupe(u8, entry.key_ptr.*);
                try results.append(allocator, tag_copy);
            }
        }

        return results.toOwnedSlice(allocator);
    }

    /// Get all document IDs that have a specific tag (uses inverted index).
    pub fn getTagDocuments(self: *DocumentTagAdapter, allocator: Allocator, tag_id: []const u8) ![][]const u8 {
        const target_tag_hash = computeHash(tag_id);

        // Track active documents for this tag
        var doc_states = std.StringHashMap(i64).init(allocator);
        defer {
            var it = doc_states.keyIterator();
            while (it.next()) |k| allocator.free(k.*);
            doc_states.deinit();
        }

        const blocks = try self.fs_reader.getBlocks(allocator);
        defer allocator.free(blocks);

        for (blocks) |block| {
            var file = self.fs_reader.dir.openFile(block.path, .{ .mode = .read_only }) catch continue;
            defer file.close();

            const reader = block_reader.BlockReader.init(file, allocator);
            const header = reader.readHeader(block.offset) catch continue;
            // Use inverted index (Schema 14) for tag → docs lookup
            if (header.schema_id != schema_tag_document_index) continue;

            const descs = reader.readColumnDirectory(header, block.offset) catch continue;
            defer allocator.free(descs);

            const row_count = header.row_count;
            if (row_count == 0) continue;

            const tag_desc = findColumn(descs, col_tag_hash) orelse continue;
            const ts_desc = findColumn(descs, col_ts) orelse continue;
            const payload_desc = findColumn(descs, col_payload) orelse continue;

            const tag_bytes = reader.readColumnData(block.offset, tag_desc, allocator) catch continue;
            defer allocator.free(tag_bytes);

            const ts_bytes = reader.readColumnData(block.offset, ts_desc, allocator) catch continue;
            defer allocator.free(ts_bytes);

            var payload_buffers = readVarBytesBuffers(file, block.offset, payload_desc, row_count, allocator) catch continue;
            defer payload_buffers.deinit(allocator);

            for (0..row_count) |row_idx| {
                const tag_hash = readU64At(tag_bytes, row_idx) catch continue;
                if (tag_hash != target_tag_hash) continue;

                const ts = readI64At(ts_bytes, row_idx) catch continue;
                const payload = payload_buffers.sliceForRow(row_idx) catch continue;

                const record_opt = decodeTagDocumentIndexRecord(allocator, payload) catch continue;
                if (record_opt) |record| {
                    defer allocator.free(record.tag_id);

                    const abs_ts = if (ts >= 0) ts else -ts;
                    if (doc_states.get(record.doc_id)) |existing_ts| {
                        const existing_abs = if (existing_ts >= 0) existing_ts else -existing_ts;
                        if (abs_ts <= existing_abs) {
                            allocator.free(record.doc_id);
                            continue;
                        }
                        if (doc_states.fetchRemove(record.doc_id)) |kv| {
                            allocator.free(kv.key);
                        }
                    }

                    // For inverted index, we store the timestamp in the record
                    doc_states.put(record.doc_id, ts) catch {
                        allocator.free(record.doc_id);
                        continue;
                    };
                }
            }
        }

        // Collect active documents (positive timestamp)
        var results = std.ArrayList([]const u8).empty;
        errdefer {
            for (results.items) |r| allocator.free(r);
            results.deinit(allocator);
        }

        var it = doc_states.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.* > 0) {
                const doc_copy = try allocator.dupe(u8, entry.key_ptr.*);
                try results.append(allocator, doc_copy);
            }
        }

        return results.toOwnedSlice(allocator);
    }

    /// Flush pending writes to disk.
    pub fn flush(self: *DocumentTagAdapter) !void {
        if (!self.read_only) {
            try self.fs_writer.flushBlock();
        }
    }

    // =========================================================================
    // Internal Helpers
    // =========================================================================

    fn writeForwardIndex(self: *DocumentTagAdapter, doc_id: []const u8, tag_id: []const u8, added_at_ms: i64, group_id: ?[]const u8) !void {
        const payload = try encodeDocumentTagKvBuf(self.allocator, doc_id, tag_id, added_at_ms);
        defer self.allocator.free(payload);

        const doc_hash = computeHash(doc_id);
        const tag_hash = computeHash(tag_id);
        const group_hash = computeOptionalHash(group_id);

        var doc_hash_value = doc_hash;
        var tag_hash_value = tag_hash;
        var ts_value = if (added_at_ms >= 0) added_at_ms else -added_at_ms;
        var group_hash_value = group_hash;

        const columns = [_]ColumnValue{
            .{ .column_id = col_doc_hash, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&doc_hash_value) },
            .{ .column_id = col_tag_hash, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&tag_hash_value) },
            .{ .column_id = col_ts, .shape = .SCALAR, .phys_type = .I64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&ts_value) },
            .{ .column_id = col_group_hash, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&group_hash_value) },
            .{ .column_id = col_payload, .shape = .VARBYTES, .phys_type = .BINARY, .encoding = .RAW, .dims = 0, .data = payload },
        };

        try self.fs_writer.appendRow(schema_document_tags, &columns);
    }

    fn writeInvertedIndex(self: *DocumentTagAdapter, tag_id: []const u8, doc_id: []const u8, added_at_ms: i64, group_id: ?[]const u8) !void {
        const payload = try encodeTagDocumentIndexKvBuf(self.allocator, tag_id, doc_id);
        defer self.allocator.free(payload);

        const tag_hash = computeHash(tag_id);
        const doc_hash = computeHash(doc_id);
        const group_hash = computeOptionalHash(group_id);

        var tag_hash_value = tag_hash;
        var doc_hash_value = doc_hash;
        var ts_value = if (added_at_ms >= 0) added_at_ms else -added_at_ms;
        var group_hash_value = group_hash;

        // Schema 14 uses tag_hash as first column (lookup key)
        const columns = [_]ColumnValue{
            .{ .column_id = col_tag_hash, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&tag_hash_value) },
            .{ .column_id = col_doc_hash, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&doc_hash_value) },
            .{ .column_id = col_ts, .shape = .SCALAR, .phys_type = .I64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&ts_value) },
            .{ .column_id = col_group_hash, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&group_hash_value) },
            .{ .column_id = col_payload, .shape = .VARBYTES, .phys_type = .BINARY, .encoding = .RAW, .dims = 0, .data = payload },
        };

        try self.fs_writer.appendRow(schema_tag_document_index, &columns);
    }
};

// =============================================================================
// KvBuf Encoding/Decoding
// =============================================================================

fn encodeDocumentTagKvBuf(allocator: Allocator, doc_id: []const u8, tag_id: []const u8, added_at_ms: i64) ![]u8 {
    var w = kvbuf.KvBufWriter.init();
    errdefer w.deinit(allocator);

    try w.addString(allocator, DocumentTagFieldIds.doc_id, doc_id);
    try w.addString(allocator, DocumentTagFieldIds.tag_id, tag_id);
    try w.addI64(allocator, DocumentTagFieldIds.added_at_ms, added_at_ms);

    const blob = try w.finish(allocator);
    w.deinit(allocator);
    return blob;
}

fn decodeDocumentTagRecord(allocator: Allocator, payload: []const u8) !?DocumentTagRecord {
    if (!kvbuf.isKvBuf(payload)) return null;

    const reader = kvbuf.KvBufReader.init(payload) catch return null;

    const doc_id = reader.get(DocumentTagFieldIds.doc_id) orelse return null;
    const tag_id = reader.get(DocumentTagFieldIds.tag_id) orelse return null;

    return DocumentTagRecord{
        .doc_id = try allocator.dupe(u8, doc_id),
        .tag_id = try allocator.dupe(u8, tag_id),
        .added_at_ms = reader.getI64(DocumentTagFieldIds.added_at_ms) orelse 0,
    };
}

fn encodeTagDocumentIndexKvBuf(allocator: Allocator, tag_id: []const u8, doc_id: []const u8) ![]u8 {
    var w = kvbuf.KvBufWriter.init();
    errdefer w.deinit(allocator);

    try w.addString(allocator, TagDocumentIndexFieldIds.tag_id, tag_id);
    try w.addString(allocator, TagDocumentIndexFieldIds.doc_id, doc_id);

    const blob = try w.finish(allocator);
    w.deinit(allocator);
    return blob;
}

/// Inverted index record: tag → doc mapping.
const TagDocumentIndexRecord = struct {
    tag_id: []const u8,
    doc_id: []const u8,
};

fn decodeTagDocumentIndexRecord(allocator: Allocator, payload: []const u8) !?TagDocumentIndexRecord {
    if (!kvbuf.isKvBuf(payload)) return null;

    const reader = kvbuf.KvBufReader.init(payload) catch return null;

    const tag_id = reader.get(TagDocumentIndexFieldIds.tag_id) orelse return null;
    const doc_id = reader.get(TagDocumentIndexFieldIds.doc_id) orelse return null;

    return TagDocumentIndexRecord{
        .tag_id = try allocator.dupe(u8, tag_id),
        .doc_id = try allocator.dupe(u8, doc_id),
    };
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

test "DocumentTagRecord encode/decode round-trip" {
    const allocator = std.testing.allocator;

    const blob = try encodeDocumentTagKvBuf(allocator, "doc-123", "tag-456", 1704067200000);
    defer allocator.free(blob);

    var decoded = (try decodeDocumentTagRecord(allocator, blob)).?;
    defer decoded.deinit(allocator);

    try std.testing.expectEqualStrings("doc-123", decoded.doc_id);
    try std.testing.expectEqualStrings("tag-456", decoded.tag_id);
    try std.testing.expectEqual(@as(i64, 1704067200000), decoded.added_at_ms);
}

test "TagDocumentIndexRecord encode/decode round-trip" {
    const allocator = std.testing.allocator;

    const blob = try encodeTagDocumentIndexKvBuf(allocator, "tag-789", "doc-abc");
    defer allocator.free(blob);

    const decoded = (try decodeTagDocumentIndexRecord(allocator, blob)).?;
    defer allocator.free(@constCast(decoded.tag_id));
    defer allocator.free(@constCast(decoded.doc_id));

    try std.testing.expectEqualStrings("tag-789", decoded.tag_id);
    try std.testing.expectEqualStrings("doc-abc", decoded.doc_id);
}

test "computeHash is deterministic" {
    const h1 = computeHash("test-junction");
    const h2 = computeHash("test-junction");
    const h3 = computeHash("different-junction");

    try std.testing.expectEqual(h1, h2);
    try std.testing.expect(h1 != h3);
}

// =============================================================================
// High-Level API Functions (for capi thin wrappers)
// =============================================================================

/// Add a tag to a document.
/// Handles adapter lifecycle internally.
/// Returns error.LockUnavailable if another process holds the database lock.
pub fn addDocumentTag(alloc: Allocator, db_path: []const u8, doc_id: []const u8, tag_id: []const u8, group_id: ?[]const u8) !void {
    var adapter = try DocumentTagAdapter.init(alloc, db_path);
    defer adapter.deinit();
    try adapter.addDocumentTag(doc_id, tag_id, std.time.milliTimestamp(), group_id);
    try adapter.flush();
}

/// Remove a tag from a document.
/// Handles adapter lifecycle internally.
/// Returns error.LockUnavailable if another process holds the database lock.
pub fn removeDocumentTag(alloc: Allocator, db_path: []const u8, doc_id: []const u8, tag_id: []const u8, group_id: ?[]const u8) !void {
    var adapter = try DocumentTagAdapter.init(alloc, db_path);
    defer adapter.deinit();
    try adapter.removeDocumentTag(doc_id, tag_id, std.time.milliTimestamp(), group_id);
    try adapter.flush();
}

/// Get all tag IDs for a document.
/// Handles adapter lifecycle internally.
/// Caller owns returned strings; free each with allocator.free().
pub fn getDocumentTagIds(alloc: Allocator, db_path: []const u8, doc_id: []const u8) ![][]const u8 {
    var adapter = try DocumentTagAdapter.initReadOnly(alloc, db_path);
    defer adapter.deinitReadOnly();
    return adapter.getDocumentTags(alloc, doc_id);
}

/// Get all document IDs that have a specific tag.
/// Handles adapter lifecycle internally.
/// Caller owns returned strings; free each with allocator.free().
pub fn getTagDocumentIds(alloc: Allocator, db_path: []const u8, tag_id: []const u8) ![][]const u8 {
    var adapter = try DocumentTagAdapter.initReadOnly(alloc, db_path);
    defer adapter.deinitReadOnly();
    return adapter.getTagDocuments(alloc, tag_id);
}

/// Free a slice of strings.
pub fn freeStringSlice(alloc: Allocator, strings: [][]const u8) void {
    for (strings) |s| alloc.free(s);
    alloc.free(strings);
}
