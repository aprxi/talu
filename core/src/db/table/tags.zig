//! TaluDB adapter for tag entity persistence.
//!
//! Tags are first-class entities with their own table, enabling:
//! - Efficient tag CRUD operations
//! - Tag metadata (color, description)
//! - Fast tag discovery without scanning all conversations
//!
//! Schema 6: Tags table
//! Schema 7: ConversationTag junction table

const std = @import("std");
const kvbuf = @import("../../io/kvbuf/root.zig");
const db_writer = @import("../writer.zig");
const db_reader = @import("../reader.zig");
const block_reader = @import("../block_reader.zig");
const types = @import("../types.zig");

const Allocator = std.mem.Allocator;
const ColumnValue = db_writer.ColumnValue;
const TagFieldIds = kvbuf.TagFieldIds;
const ConversationTagFieldIds = kvbuf.ConversationTagFieldIds;

const schema_tags: u16 = 6;
const schema_conversation_tags: u16 = 7;
const schema_tag_deletes: u16 = 8;

// Column IDs for Tags table
const col_tag_hash: u32 = 1;
const col_ts: u32 = 2;
const col_group_hash: u32 = 3;
const col_name_hash: u32 = 4;
const col_payload: u32 = 20;

// Column IDs for ConversationTag junction
const col_session_hash: u32 = 5; // Different from col_tag_hash

/// Tag entity record for storage/retrieval.
pub const TagRecord = struct {
    tag_id: []const u8,
    name: []const u8,
    color: ?[]const u8 = null,
    description: ?[]const u8 = null,
    group_id: ?[]const u8 = null,
    created_at_ms: i64,
    updated_at_ms: i64,

    pub fn deinit(self: *TagRecord, allocator: Allocator) void {
        allocator.free(self.tag_id);
        allocator.free(self.name);
        if (self.color) |c| allocator.free(c);
        if (self.description) |d| allocator.free(d);
        if (self.group_id) |g| allocator.free(g);
    }
};

/// Conversation-Tag junction record.
pub const ConversationTagRecord = struct {
    session_id: []const u8,
    tag_id: []const u8,
    added_at_ms: i64,

    pub fn deinit(self: *ConversationTagRecord, allocator: Allocator) void {
        allocator.free(self.session_id);
        allocator.free(self.tag_id);
    }
};

/// TagAdapter - TaluDB adapter for tag persistence.
///
/// Provides CRUD operations for tags and conversation-tag associations.
/// Thread safety: NOT thread-safe (single-writer semantics via lock).
pub const TagAdapter = struct {
    allocator: Allocator,
    fs_writer: *db_writer.Writer,
    fs_reader: *db_reader.Reader,
    read_only: bool,

    /// Initialize a TaluDB-backed tag adapter with write capabilities.
    pub fn init(allocator: Allocator, db_root: []const u8) !TagAdapter {
        var writer_ptr = try allocator.create(db_writer.Writer);
        errdefer allocator.destroy(writer_ptr);
        writer_ptr.* = try db_writer.Writer.open(allocator, db_root, "chat");
        errdefer writer_ptr.deinit();

        var reader_ptr = try allocator.create(db_reader.Reader);
        errdefer allocator.destroy(reader_ptr);
        reader_ptr.* = try db_reader.Reader.open(allocator, db_root, "chat");
        errdefer reader_ptr.deinit();

        return .{
            .allocator = allocator,
            .fs_writer = writer_ptr,
            .fs_reader = reader_ptr,
            .read_only = false,
        };
    }

    /// Initialize a read-only adapter for scanning tags.
    pub fn initReadOnly(allocator: Allocator, db_root: []const u8) !TagAdapter {
        const reader_ptr = try allocator.create(db_reader.Reader);
        errdefer allocator.destroy(reader_ptr);
        reader_ptr.* = try db_reader.Reader.open(allocator, db_root, "chat");

        return .{
            .allocator = allocator,
            .fs_writer = undefined,
            .fs_reader = reader_ptr,
            .read_only = true,
        };
    }

    pub fn deinit(self: *TagAdapter) void {
        if (!self.read_only) {
            self.fs_writer.flushBlock() catch {};
            self.fs_writer.deinit();
            self.allocator.destroy(self.fs_writer);
        }
        self.fs_reader.deinit();
        self.allocator.destroy(self.fs_reader);
    }

    pub fn deinitReadOnly(self: *TagAdapter) void {
        self.fs_reader.deinit();
        self.allocator.destroy(self.fs_reader);
    }

    // =========================================================================
    // Tag CRUD Operations
    // =========================================================================

    /// Write (create or update) a tag record.
    pub fn writeTag(self: *TagAdapter, record: TagRecord) !void {
        const payload = try encodeTagRecordKvBuf(self.allocator, record);
        defer self.allocator.free(payload);

        const tag_hash = computeHash(record.tag_id);
        const group_hash = computeOptionalHash(record.group_id);
        const name_hash = computeHash(record.name);

        var tag_hash_value = tag_hash;
        var ts_value = record.updated_at_ms;
        var group_hash_value = group_hash;
        var name_hash_value = name_hash;

        const columns = [_]ColumnValue{
            .{ .column_id = col_tag_hash, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&tag_hash_value) },
            .{ .column_id = col_ts, .shape = .SCALAR, .phys_type = .I64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&ts_value) },
            .{ .column_id = col_group_hash, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&group_hash_value) },
            .{ .column_id = col_name_hash, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&name_hash_value) },
            .{ .column_id = col_payload, .shape = .VARBYTES, .phys_type = .BINARY, .encoding = .RAW, .dims = 0, .data = payload },
        };

        try self.fs_writer.appendRow(schema_tags, &columns);
    }

    /// Write a tag deletion marker.
    pub fn deleteTag(self: *TagAdapter, tag_id: []const u8, deleted_at_ms: i64) !void {
        const tag_hash = computeHash(tag_id);
        var tag_hash_value = tag_hash;
        var ts_value = deleted_at_ms;

        const columns = [_]ColumnValue{
            .{ .column_id = col_tag_hash, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&tag_hash_value) },
            .{ .column_id = col_ts, .shape = .SCALAR, .phys_type = .I64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&ts_value) },
        };

        try self.fs_writer.appendRow(schema_tag_deletes, &columns);
    }

    /// Scan all tags, optionally filtered by group.
    pub fn scanTags(self: *TagAdapter, allocator: Allocator, group_id: ?[]const u8) ![]TagRecord {
        const target_group_hash: ?u64 = if (group_id) |g| computeHash(g) else null;

        // Collect deleted tag hashes
        var deleted = std.AutoHashMap(u64, i64).init(allocator);
        defer deleted.deinit();
        try self.collectDeletedTags(allocator, &deleted);

        // Track latest version of each tag by hash
        var latest = std.AutoHashMap(u64, TagRecord).init(allocator);
        defer {
            var it = latest.valueIterator();
            while (it.next()) |v| {
                var tag = v.*;
                tag.deinit(allocator);
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
            if (header.schema_id != schema_tags) continue;

            const descs = reader.readColumnDirectory(header, block.offset) catch continue;
            defer allocator.free(descs);

            const row_count = header.row_count;
            if (row_count == 0) continue;

            const hash_desc = findColumn(descs, col_tag_hash) orelse continue;
            const ts_desc = findColumn(descs, col_ts) orelse continue;
            const group_desc = findColumn(descs, col_group_hash);
            const payload_desc = findColumn(descs, col_payload) orelse continue;

            const hash_bytes = reader.readColumnData(block.offset, hash_desc, allocator) catch continue;
            defer allocator.free(hash_bytes);

            const ts_bytes = reader.readColumnData(block.offset, ts_desc, allocator) catch continue;
            defer allocator.free(ts_bytes);

            var group_bytes: ?[]const u8 = null;
            defer if (group_bytes) |g| allocator.free(g);
            if (group_desc) |gd| {
                group_bytes = reader.readColumnData(block.offset, gd, allocator) catch null;
            }

            var payload_buffers = readVarBytesBuffers(file, block.offset, payload_desc, row_count, allocator) catch continue;
            defer payload_buffers.deinit(allocator);

            for (0..row_count) |row_idx| {
                const tag_hash = readU64At(hash_bytes, row_idx) catch continue;
                const ts = readI64At(ts_bytes, row_idx) catch continue;

                // Skip if deleted
                if (deleted.get(tag_hash)) |del_ts| {
                    if (del_ts >= ts) continue;
                }

                // Filter by group if specified
                if (target_group_hash) |target| {
                    if (group_bytes) |gb| {
                        const row_group = readU64At(gb, row_idx) catch continue;
                        if (row_group != target and row_group != 0) continue;
                    }
                }

                const payload = payload_buffers.sliceForRow(row_idx) catch continue;
                const record_opt = decodeTagRecord(allocator, payload) catch continue;
                if (record_opt) |record| {
                    // Check if we already have a newer version
                    if (latest.get(tag_hash)) |existing| {
                        if (existing.updated_at_ms >= record.updated_at_ms) {
                            var r = record;
                            r.deinit(allocator);
                            continue;
                        }
                        var old = existing;
                        old.deinit(allocator);
                    }
                    latest.put(tag_hash, record) catch {
                        var r = record;
                        r.deinit(allocator);
                        continue;
                    };
                }
            }
        }

        // Convert to slice
        var results = std.ArrayList(TagRecord).empty;
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

    /// Get a single tag by ID.
    pub fn getTag(self: *TagAdapter, allocator: Allocator, tag_id: []const u8) !?TagRecord {
        const target_hash = computeHash(tag_id);

        // Check if deleted
        var deleted = std.AutoHashMap(u64, i64).init(allocator);
        defer deleted.deinit();
        try self.collectDeletedTags(allocator, &deleted);

        var latest: ?TagRecord = null;
        var latest_ts: i64 = 0;

        const blocks = try self.fs_reader.getBlocks(allocator);
        defer allocator.free(blocks);

        for (blocks) |block| {
            var file = self.fs_reader.dir.openFile(block.path, .{ .mode = .read_only }) catch continue;
            defer file.close();

            const reader = block_reader.BlockReader.init(file, allocator);
            const header = reader.readHeader(block.offset) catch continue;
            if (header.schema_id != schema_tags) continue;

            const descs = reader.readColumnDirectory(header, block.offset) catch continue;
            defer allocator.free(descs);

            const row_count = header.row_count;
            if (row_count == 0) continue;

            const hash_desc = findColumn(descs, col_tag_hash) orelse continue;
            const ts_desc = findColumn(descs, col_ts) orelse continue;
            const payload_desc = findColumn(descs, col_payload) orelse continue;

            const hash_bytes = reader.readColumnData(block.offset, hash_desc, allocator) catch continue;
            defer allocator.free(hash_bytes);

            const ts_bytes = reader.readColumnData(block.offset, ts_desc, allocator) catch continue;
            defer allocator.free(ts_bytes);

            var payload_buffers = readVarBytesBuffers(file, block.offset, payload_desc, row_count, allocator) catch continue;
            defer payload_buffers.deinit(allocator);

            for (0..row_count) |row_idx| {
                const tag_hash = readU64At(hash_bytes, row_idx) catch continue;
                if (tag_hash != target_hash) continue;

                const ts = readI64At(ts_bytes, row_idx) catch continue;

                // Skip if deleted
                if (deleted.get(tag_hash)) |del_ts| {
                    if (del_ts >= ts) continue;
                }

                if (ts <= latest_ts) continue;

                const payload = payload_buffers.sliceForRow(row_idx) catch continue;
                const record_opt = decodeTagRecord(allocator, payload) catch continue;
                if (record_opt) |record| {
                    if (latest) |*old| old.deinit(allocator);
                    latest = record;
                    latest_ts = ts;
                }
            }
        }

        return latest;
    }

    /// Get a tag by name within a group.
    pub fn getTagByName(self: *TagAdapter, allocator: Allocator, name: []const u8, group_id: ?[]const u8) !?TagRecord {
        const target_name_hash = computeHash(name);
        const target_group_hash: ?u64 = if (group_id) |g| computeHash(g) else null;

        // Check deleted tags
        var deleted = std.AutoHashMap(u64, i64).init(allocator);
        defer deleted.deinit();
        try self.collectDeletedTags(allocator, &deleted);

        var latest: ?TagRecord = null;
        var latest_ts: i64 = 0;

        const blocks = try self.fs_reader.getBlocks(allocator);
        defer allocator.free(blocks);

        for (blocks) |block| {
            var file = self.fs_reader.dir.openFile(block.path, .{ .mode = .read_only }) catch continue;
            defer file.close();

            const reader = block_reader.BlockReader.init(file, allocator);
            const header = reader.readHeader(block.offset) catch continue;
            if (header.schema_id != schema_tags) continue;

            const descs = reader.readColumnDirectory(header, block.offset) catch continue;
            defer allocator.free(descs);

            const row_count = header.row_count;
            if (row_count == 0) continue;

            const hash_desc = findColumn(descs, col_tag_hash) orelse continue;
            const ts_desc = findColumn(descs, col_ts) orelse continue;
            const name_hash_desc = findColumn(descs, col_name_hash) orelse continue;
            const group_desc = findColumn(descs, col_group_hash);
            const payload_desc = findColumn(descs, col_payload) orelse continue;

            const hash_bytes = reader.readColumnData(block.offset, hash_desc, allocator) catch continue;
            defer allocator.free(hash_bytes);

            const ts_bytes = reader.readColumnData(block.offset, ts_desc, allocator) catch continue;
            defer allocator.free(ts_bytes);

            const name_hash_bytes = reader.readColumnData(block.offset, name_hash_desc, allocator) catch continue;
            defer allocator.free(name_hash_bytes);

            var group_bytes: ?[]const u8 = null;
            defer if (group_bytes) |g| allocator.free(g);
            if (group_desc) |gd| {
                group_bytes = reader.readColumnData(block.offset, gd, allocator) catch null;
            }

            var payload_buffers = readVarBytesBuffers(file, block.offset, payload_desc, row_count, allocator) catch continue;
            defer payload_buffers.deinit(allocator);

            for (0..row_count) |row_idx| {
                const name_hash = readU64At(name_hash_bytes, row_idx) catch continue;
                if (name_hash != target_name_hash) continue;

                // Filter by group
                if (target_group_hash) |target| {
                    if (group_bytes) |gb| {
                        const row_group = readU64At(gb, row_idx) catch continue;
                        if (row_group != target and row_group != 0) continue;
                    }
                }

                const tag_hash = readU64At(hash_bytes, row_idx) catch continue;
                const ts = readI64At(ts_bytes, row_idx) catch continue;

                // Skip if deleted
                if (deleted.get(tag_hash)) |del_ts| {
                    if (del_ts >= ts) continue;
                }

                if (ts <= latest_ts) continue;

                const payload = payload_buffers.sliceForRow(row_idx) catch continue;
                const record_opt = decodeTagRecord(allocator, payload) catch continue;
                if (record_opt) |record| {
                    // Verify name matches (hash collision check)
                    if (!std.ascii.eqlIgnoreCase(record.name, name)) {
                        var r = record;
                        r.deinit(allocator);
                        continue;
                    }
                    if (latest) |*old| old.deinit(allocator);
                    latest = record;
                    latest_ts = ts;
                }
            }
        }

        return latest;
    }

    // =========================================================================
    // Conversation-Tag Junction Operations
    // =========================================================================

    /// Add a tag to a conversation.
    pub fn addConversationTag(self: *TagAdapter, session_id: []const u8, tag_id: []const u8, added_at_ms: i64) !void {
        const payload = try encodeConversationTagKvBuf(self.allocator, session_id, tag_id, added_at_ms);
        defer self.allocator.free(payload);

        const session_hash = computeHash(session_id);
        const tag_hash = computeHash(tag_id);

        var session_hash_value = session_hash;
        var tag_hash_value = tag_hash;
        var ts_value = added_at_ms;

        const columns = [_]ColumnValue{
            .{ .column_id = col_session_hash, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&session_hash_value) },
            .{ .column_id = col_tag_hash, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&tag_hash_value) },
            .{ .column_id = col_ts, .shape = .SCALAR, .phys_type = .I64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&ts_value) },
            .{ .column_id = col_payload, .shape = .VARBYTES, .phys_type = .BINARY, .encoding = .RAW, .dims = 0, .data = payload },
        };

        try self.fs_writer.appendRow(schema_conversation_tags, &columns);
    }

    /// Remove a tag from a conversation (writes a tombstone with negative timestamp).
    pub fn removeConversationTag(self: *TagAdapter, session_id: []const u8, tag_id: []const u8, removed_at_ms: i64) !void {
        // Write a tombstone: same structure but with negative timestamp convention
        const payload = try encodeConversationTagKvBuf(self.allocator, session_id, tag_id, -removed_at_ms);
        defer self.allocator.free(payload);

        const session_hash = computeHash(session_id);
        const tag_hash = computeHash(tag_id);

        var session_hash_value = session_hash;
        var tag_hash_value = tag_hash;
        var ts_value = removed_at_ms;

        const columns = [_]ColumnValue{
            .{ .column_id = col_session_hash, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&session_hash_value) },
            .{ .column_id = col_tag_hash, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&tag_hash_value) },
            .{ .column_id = col_ts, .shape = .SCALAR, .phys_type = .I64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&ts_value) },
            .{ .column_id = col_payload, .shape = .VARBYTES, .phys_type = .BINARY, .encoding = .RAW, .dims = 0, .data = payload },
        };

        try self.fs_writer.appendRow(schema_conversation_tags, &columns);
    }

    /// Get all tag IDs for a conversation.
    pub fn getConversationTags(self: *TagAdapter, allocator: Allocator, session_id: []const u8) ![][]const u8 {
        const target_session_hash = computeHash(session_id);

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
            if (header.schema_id != schema_conversation_tags) continue;

            const descs = reader.readColumnDirectory(header, block.offset) catch continue;
            defer allocator.free(descs);

            const row_count = header.row_count;
            if (row_count == 0) continue;

            const session_desc = findColumn(descs, col_session_hash) orelse continue;
            const ts_desc = findColumn(descs, col_ts) orelse continue;
            const payload_desc = findColumn(descs, col_payload) orelse continue;

            const session_bytes = reader.readColumnData(block.offset, session_desc, allocator) catch continue;
            defer allocator.free(session_bytes);

            const ts_bytes = reader.readColumnData(block.offset, ts_desc, allocator) catch continue;
            defer allocator.free(ts_bytes);

            var payload_buffers = readVarBytesBuffers(file, block.offset, payload_desc, row_count, allocator) catch continue;
            defer payload_buffers.deinit(allocator);

            for (0..row_count) |row_idx| {
                const session_hash = readU64At(session_bytes, row_idx) catch continue;
                if (session_hash != target_session_hash) continue;

                const ts = readI64At(ts_bytes, row_idx) catch continue;
                const payload = payload_buffers.sliceForRow(row_idx) catch continue;

                const record_opt = decodeConversationTagRecord(allocator, payload) catch continue;
                if (record_opt) |record| {
                    defer allocator.free(record.session_id);

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

    /// Get all conversations that have a specific tag.
    pub fn getTagConversations(self: *TagAdapter, allocator: Allocator, tag_id: []const u8) ![][]const u8 {
        const target_tag_hash = computeHash(tag_id);

        // Track active sessions for this tag
        var session_states = std.StringHashMap(i64).init(allocator);
        defer {
            var it = session_states.keyIterator();
            while (it.next()) |k| allocator.free(k.*);
            session_states.deinit();
        }

        const blocks = try self.fs_reader.getBlocks(allocator);
        defer allocator.free(blocks);

        for (blocks) |block| {
            var file = self.fs_reader.dir.openFile(block.path, .{ .mode = .read_only }) catch continue;
            defer file.close();

            const reader = block_reader.BlockReader.init(file, allocator);
            const header = reader.readHeader(block.offset) catch continue;
            if (header.schema_id != schema_conversation_tags) continue;

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

                const record_opt = decodeConversationTagRecord(allocator, payload) catch continue;
                if (record_opt) |record| {
                    defer allocator.free(record.tag_id);

                    const abs_ts = if (ts >= 0) ts else -ts;
                    if (session_states.get(record.session_id)) |existing_ts| {
                        const existing_abs = if (existing_ts >= 0) existing_ts else -existing_ts;
                        if (abs_ts <= existing_abs) {
                            allocator.free(record.session_id);
                            continue;
                        }
                        if (session_states.fetchRemove(record.session_id)) |kv| {
                            allocator.free(kv.key);
                        }
                    }

                    session_states.put(record.session_id, record.added_at_ms) catch {
                        allocator.free(record.session_id);
                        continue;
                    };
                }
            }
        }

        // Collect active sessions
        var results = std.ArrayList([]const u8).empty;
        errdefer {
            for (results.items) |r| allocator.free(r);
            results.deinit(allocator);
        }

        var it = session_states.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.* > 0) {
                const session_copy = try allocator.dupe(u8, entry.key_ptr.*);
                try results.append(allocator, session_copy);
            }
        }

        return results.toOwnedSlice(allocator);
    }

    /// Flush pending writes to disk.
    pub fn flush(self: *TagAdapter) !void {
        if (!self.read_only) {
            try self.fs_writer.flushBlock();
        }
    }

    // =========================================================================
    // Internal Helpers
    // =========================================================================

    fn collectDeletedTags(self: *TagAdapter, allocator: Allocator, deleted: *std.AutoHashMap(u64, i64)) !void {
        const blocks = try self.fs_reader.getBlocks(allocator);
        defer allocator.free(blocks);

        for (blocks) |block| {
            var file = self.fs_reader.dir.openFile(block.path, .{ .mode = .read_only }) catch continue;
            defer file.close();

            const reader = block_reader.BlockReader.init(file, allocator);
            const header = reader.readHeader(block.offset) catch continue;
            if (header.schema_id != schema_tag_deletes) continue;

            const descs = reader.readColumnDirectory(header, block.offset) catch continue;
            defer allocator.free(descs);

            const row_count = header.row_count;
            if (row_count == 0) continue;

            const hash_desc = findColumn(descs, col_tag_hash) orelse continue;
            const ts_desc = findColumn(descs, col_ts) orelse continue;

            const hash_bytes = reader.readColumnData(block.offset, hash_desc, allocator) catch continue;
            defer allocator.free(hash_bytes);

            const ts_bytes = reader.readColumnData(block.offset, ts_desc, allocator) catch continue;
            defer allocator.free(ts_bytes);

            for (0..row_count) |row_idx| {
                const tag_hash = readU64At(hash_bytes, row_idx) catch continue;
                const ts = readI64At(ts_bytes, row_idx) catch continue;

                if (deleted.get(tag_hash)) |existing| {
                    if (ts > existing) {
                        try deleted.put(tag_hash, ts);
                    }
                } else {
                    try deleted.put(tag_hash, ts);
                }
            }
        }
    }
};

// =============================================================================
// KvBuf Encoding/Decoding
// =============================================================================

fn encodeTagRecordKvBuf(allocator: Allocator, record: TagRecord) ![]u8 {
    var w = kvbuf.KvBufWriter.init();
    errdefer w.deinit(allocator);

    try w.addString(allocator, TagFieldIds.tag_id, record.tag_id);
    try w.addString(allocator, TagFieldIds.name, record.name);
    if (record.color) |c| try w.addString(allocator, TagFieldIds.color, c);
    if (record.description) |d| try w.addString(allocator, TagFieldIds.description, d);
    if (record.group_id) |g| try w.addString(allocator, TagFieldIds.group_id, g);
    try w.addI64(allocator, TagFieldIds.created_at_ms, record.created_at_ms);
    try w.addI64(allocator, TagFieldIds.updated_at_ms, record.updated_at_ms);

    const blob = try w.finish(allocator);
    w.deinit(allocator);
    return blob;
}

fn decodeTagRecord(allocator: Allocator, payload: []const u8) !?TagRecord {
    if (!kvbuf.isKvBuf(payload)) return null;

    const reader = kvbuf.KvBufReader.init(payload) catch return null;

    const tag_id = reader.get(TagFieldIds.tag_id) orelse return null;
    const name = reader.get(TagFieldIds.name) orelse return null;

    return TagRecord{
        .tag_id = try allocator.dupe(u8, tag_id),
        .name = try allocator.dupe(u8, name),
        .color = if (reader.get(TagFieldIds.color)) |c| try allocator.dupe(u8, c) else null,
        .description = if (reader.get(TagFieldIds.description)) |d| try allocator.dupe(u8, d) else null,
        .group_id = if (reader.get(TagFieldIds.group_id)) |g| try allocator.dupe(u8, g) else null,
        .created_at_ms = reader.getI64(TagFieldIds.created_at_ms) orelse 0,
        .updated_at_ms = reader.getI64(TagFieldIds.updated_at_ms) orelse 0,
    };
}

fn encodeConversationTagKvBuf(allocator: Allocator, session_id: []const u8, tag_id: []const u8, added_at_ms: i64) ![]u8 {
    var w = kvbuf.KvBufWriter.init();
    errdefer w.deinit(allocator);

    try w.addString(allocator, ConversationTagFieldIds.session_id, session_id);
    try w.addString(allocator, ConversationTagFieldIds.tag_id, tag_id);
    try w.addI64(allocator, ConversationTagFieldIds.added_at_ms, added_at_ms);

    const blob = try w.finish(allocator);
    w.deinit(allocator);
    return blob;
}

fn decodeConversationTagRecord(allocator: Allocator, payload: []const u8) !?ConversationTagRecord {
    if (!kvbuf.isKvBuf(payload)) return null;

    const reader = kvbuf.KvBufReader.init(payload) catch return null;

    const session_id = reader.get(ConversationTagFieldIds.session_id) orelse return null;
    const tag_id = reader.get(ConversationTagFieldIds.tag_id) orelse return null;

    return ConversationTagRecord{
        .session_id = try allocator.dupe(u8, session_id),
        .tag_id = try allocator.dupe(u8, tag_id),
        .added_at_ms = reader.getI64(ConversationTagFieldIds.added_at_ms) orelse 0,
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

test "TagRecord encode/decode round-trip" {
    const allocator = std.testing.allocator;

    const original = TagRecord{
        .tag_id = "tag-uuid-123",
        .name = "work",
        .color = "#4a90d9",
        .description = "Work-related conversations",
        .group_id = "tenant-1",
        .created_at_ms = 1704067200000,
        .updated_at_ms = 1704153600000,
    };

    const blob = try encodeTagRecordKvBuf(allocator, original);
    defer allocator.free(blob);

    var decoded = (try decodeTagRecord(allocator, blob)).?;
    defer decoded.deinit(allocator);

    try std.testing.expectEqualStrings("tag-uuid-123", decoded.tag_id);
    try std.testing.expectEqualStrings("work", decoded.name);
    try std.testing.expectEqualStrings("#4a90d9", decoded.color.?);
    try std.testing.expectEqualStrings("Work-related conversations", decoded.description.?);
    try std.testing.expectEqualStrings("tenant-1", decoded.group_id.?);
    try std.testing.expectEqual(@as(i64, 1704067200000), decoded.created_at_ms);
    try std.testing.expectEqual(@as(i64, 1704153600000), decoded.updated_at_ms);
}

test "TagRecord encode/decode with optional fields null" {
    const allocator = std.testing.allocator;

    const original = TagRecord{
        .tag_id = "tag-minimal",
        .name = "simple",
        .color = null,
        .description = null,
        .group_id = null,
        .created_at_ms = 1000,
        .updated_at_ms = 2000,
    };

    const blob = try encodeTagRecordKvBuf(allocator, original);
    defer allocator.free(blob);

    var decoded = (try decodeTagRecord(allocator, blob)).?;
    defer decoded.deinit(allocator);

    try std.testing.expectEqualStrings("tag-minimal", decoded.tag_id);
    try std.testing.expectEqualStrings("simple", decoded.name);
    try std.testing.expect(decoded.color == null);
    try std.testing.expect(decoded.description == null);
    try std.testing.expect(decoded.group_id == null);
}

test "ConversationTagRecord encode/decode round-trip" {
    const allocator = std.testing.allocator;

    const blob = try encodeConversationTagKvBuf(allocator, "sess-123", "tag-456", 1704067200000);
    defer allocator.free(blob);

    var decoded = (try decodeConversationTagRecord(allocator, blob)).?;
    defer decoded.deinit(allocator);

    try std.testing.expectEqualStrings("sess-123", decoded.session_id);
    try std.testing.expectEqualStrings("tag-456", decoded.tag_id);
    try std.testing.expectEqual(@as(i64, 1704067200000), decoded.added_at_ms);
}

test "computeHash is deterministic" {
    const h1 = computeHash("test-string");
    const h2 = computeHash("test-string");
    const h3 = computeHash("different-string");

    try std.testing.expectEqual(h1, h2);
    try std.testing.expect(h1 != h3);
}

test "computeOptionalHash handles null" {
    const h1 = computeOptionalHash(null);
    const h2 = computeOptionalHash("test");

    try std.testing.expectEqual(@as(u64, 0), h1);
    try std.testing.expect(h2 != 0);
}

// =============================================================================
// High-Level API Functions (for capi thin wrappers)
// =============================================================================

/// List all tags, optionally filtered by group.
/// Handles adapter lifecycle internally.
/// Caller owns returned records; free each with TagRecord.deinit().
pub fn listTags(alloc: Allocator, db_path: []const u8, group_id: ?[]const u8) ![]TagRecord {
    var adapter = try TagAdapter.initReadOnly(alloc, db_path);
    defer adapter.deinitReadOnly();
    return adapter.scanTags(alloc, group_id);
}

/// Get a single tag by ID.
/// Handles adapter lifecycle internally.
/// Returns null if tag not found.
/// Caller owns returned record; free with TagRecord.deinit().
pub fn getTag(alloc: Allocator, db_path: []const u8, tag_id: []const u8) !?TagRecord {
    var adapter = try TagAdapter.initReadOnly(alloc, db_path);
    defer adapter.deinitReadOnly();
    return adapter.getTag(alloc, tag_id);
}

/// Get a tag by name within a group.
/// Handles adapter lifecycle internally.
/// Returns null if tag not found.
/// Caller owns returned record; free with TagRecord.deinit().
pub fn getTagByName(alloc: Allocator, db_path: []const u8, name: []const u8, group_id: ?[]const u8) !?TagRecord {
    var adapter = try TagAdapter.initReadOnly(alloc, db_path);
    defer adapter.deinitReadOnly();
    return adapter.getTagByName(alloc, name, group_id);
}

/// Create a new tag.
/// Handles adapter lifecycle internally.
/// Returns error.LockUnavailable if another process holds the database lock.
pub fn createTag(alloc: Allocator, db_path: []const u8, record: TagRecord) !void {
    var adapter = try TagAdapter.init(alloc, db_path);
    defer adapter.deinit();
    try adapter.writeTag(record);
    try adapter.flush();
}

/// Update an existing tag.
/// Handles adapter lifecycle internally.
/// Returns error.NotFound if tag doesn't exist.
/// Returns error.LockUnavailable if another process holds the database lock.
pub fn updateTag(
    alloc: Allocator,
    db_path: []const u8,
    tag_id: []const u8,
    name: ?[]const u8,
    color: ?[]const u8,
    description: ?[]const u8,
) !void {
    var adapter = try TagAdapter.init(alloc, db_path);
    defer adapter.deinit();

    var existing = try adapter.getTag(alloc, tag_id) orelse return error.TagNotFound;
    defer existing.deinit(alloc);

    const updated = TagRecord{
        .tag_id = existing.tag_id,
        .name = name orelse existing.name,
        .color = color orelse existing.color,
        .description = description orelse existing.description,
        .group_id = existing.group_id,
        .created_at_ms = existing.created_at_ms,
        .updated_at_ms = std.time.milliTimestamp(),
    };

    try adapter.writeTag(updated);
    try adapter.flush();
}

/// Delete a tag and remove it from all conversations.
/// Handles adapter lifecycle internally.
/// Returns error.LockUnavailable if another process holds the database lock.
/// Returns error.TagNotFound if the tag does not exist.
pub fn deleteTagAndAssociations(alloc: Allocator, db_path: []const u8, tag_id: []const u8) !void {
    var adapter = try TagAdapter.init(alloc, db_path);
    defer adapter.deinit();

    // Check if tag exists first
    var existing = try adapter.getTag(alloc, tag_id) orelse return error.TagNotFound;
    existing.deinit(alloc);

    const now_ms = std.time.milliTimestamp();

    // Remove from all conversations
    const sessions = try adapter.getTagConversations(alloc, tag_id);
    defer {
        for (sessions) |s| alloc.free(s);
        alloc.free(sessions);
    }
    for (sessions) |session_id| {
        try adapter.removeConversationTag(session_id, tag_id, now_ms);
    }

    // Delete the tag itself
    try adapter.deleteTag(tag_id, now_ms);
    try adapter.flush();
}

/// Add a tag to a conversation.
/// Handles adapter lifecycle internally.
/// Returns error.LockUnavailable if another process holds the database lock.
pub fn addConversationTag(alloc: Allocator, db_path: []const u8, session_id: []const u8, tag_id: []const u8) !void {
    var adapter = try TagAdapter.init(alloc, db_path);
    defer adapter.deinit();
    try adapter.addConversationTag(session_id, tag_id, std.time.milliTimestamp());
    try adapter.flush();
}

/// Remove a tag from a conversation.
/// Handles adapter lifecycle internally.
/// Returns error.LockUnavailable if another process holds the database lock.
pub fn removeConversationTag(alloc: Allocator, db_path: []const u8, session_id: []const u8, tag_id: []const u8) !void {
    var adapter = try TagAdapter.init(alloc, db_path);
    defer adapter.deinit();
    try adapter.removeConversationTag(session_id, tag_id, std.time.milliTimestamp());
    try adapter.flush();
}

/// Get all tag IDs for a conversation.
/// Handles adapter lifecycle internally.
/// Caller owns returned strings; free each with allocator.free().
pub fn getConversationTagIds(alloc: Allocator, db_path: []const u8, session_id: []const u8) ![][]const u8 {
    var adapter = try TagAdapter.initReadOnly(alloc, db_path);
    defer adapter.deinitReadOnly();
    return adapter.getConversationTags(alloc, session_id);
}

/// Get all conversation IDs that have a specific tag.
/// Handles adapter lifecycle internally.
/// Caller owns returned strings; free each with allocator.free().
pub fn getTagConversationIds(alloc: Allocator, db_path: []const u8, tag_id: []const u8) ![][]const u8 {
    var adapter = try TagAdapter.initReadOnly(alloc, db_path);
    defer adapter.deinitReadOnly();
    return adapter.getTagConversations(alloc, tag_id);
}

/// Free a slice of TagRecords.
pub fn freeTagRecords(alloc: Allocator, records: []TagRecord) void {
    for (records) |*r| {
        var rec = r.*;
        rec.deinit(alloc);
    }
    alloc.free(records);
}

/// Free a slice of owned strings (session IDs, tag IDs, etc.).
pub fn freeStringSlice(alloc: Allocator, strings: [][]const u8) void {
    for (strings) |s| alloc.free(s);
    alloc.free(strings);
}

// =============================================================================
// Tag Filter Resolution (domain logic for search)
// =============================================================================

/// Resolve a space-separated tag name filter to matching session IDs.
///
/// For AND logic (`is_and = true`): returns sessions that have ALL specified
/// tags. For OR logic (`is_and = false`): returns sessions that have ANY of
/// the specified tags.
///
/// Returns an empty slice if no tags match. Caller owns returned strings;
/// free each with `alloc.free()`, then free the slice.
pub fn resolveTagFilterSessionIds(
    alloc: Allocator,
    db_path: []const u8,
    filter: []const u8,
    is_and: bool,
    group_id: ?[]const u8,
) ![][]const u8 {
    var adapter = try TagAdapter.initReadOnly(alloc, db_path);
    defer adapter.deinitReadOnly();

    // Scan all tags once for case-insensitive name matching.
    const all_tags = try adapter.scanTags(alloc, group_id);
    defer {
        for (all_tags) |*t| {
            var tag = t.*;
            tag.deinit(alloc);
        }
        alloc.free(all_tags);
    }

    var result_set = std.StringHashMap(void).init(alloc);
    defer result_set.deinit();

    var first_tag = true;

    var it = std.mem.tokenizeScalar(u8, filter, ' ');
    while (it.next()) |tag_name| {
        // Find tag by name (case-insensitive).
        const tag_id = findTagIdByName(all_tags, tag_name) orelse {
            if (is_and) {
                // AND: unknown tag → intersection is empty.
                clearStringHashMap(alloc, &result_set);
                break;
            }
            continue; // OR: skip unknown tags.
        };

        // Get session IDs for this tag.
        const session_ids = try adapter.getTagConversations(alloc, tag_id);
        defer {
            for (session_ids) |s| alloc.free(s);
            alloc.free(session_ids);
        }

        if (is_and) {
            if (first_tag) {
                // First tag: seed the result set.
                for (session_ids) |sid| {
                    const key = try alloc.dupe(u8, sid);
                    result_set.put(key, {}) catch {
                        alloc.free(key);
                        continue;
                    };
                }
                first_tag = false;
            } else {
                // Subsequent tags: intersect.
                var tag_set = std.StringHashMap(void).init(alloc);
                defer tag_set.deinit();
                for (session_ids) |sid| {
                    tag_set.put(sid, {}) catch continue;
                }

                // Remove entries not in tag_set.
                var to_remove = std.ArrayList([]const u8).empty;
                defer to_remove.deinit(alloc);
                var rit = result_set.keyIterator();
                while (rit.next()) |key| {
                    if (!tag_set.contains(key.*)) {
                        to_remove.append(alloc, key.*) catch continue;
                    }
                }
                for (to_remove.items) |key| {
                    _ = result_set.remove(key);
                    alloc.free(key);
                }
            }
        } else {
            // OR: union.
            for (session_ids) |sid| {
                if (!result_set.contains(sid)) {
                    const key = try alloc.dupe(u8, sid);
                    result_set.put(key, {}) catch {
                        alloc.free(key);
                        continue;
                    };
                }
            }
        }
    }

    // Convert HashMap keys to owned slice.
    var results = std.ArrayList([]const u8).empty;
    errdefer {
        for (results.items) |s| alloc.free(s);
        results.deinit(alloc);
    }

    var kit = result_set.keyIterator();
    while (kit.next()) |key| {
        try results.append(alloc, key.*);
    }
    // Don't free keys — ownership transferred to results.
    // Just deinit the map structure (not the keys).
    result_set.clearRetainingCapacity();

    return results.toOwnedSlice(alloc);
}

/// Collect all session IDs that have at least one tag.
///
/// Scans all tags and their conversation associations. Returns deduplicated
/// session IDs. Caller owns returned strings; free each with `alloc.free()`.
pub fn collectAllTaggedSessionIds(alloc: Allocator, db_path: []const u8) ![][]const u8 {
    var adapter = try TagAdapter.initReadOnly(alloc, db_path);
    defer adapter.deinitReadOnly();

    // Get all tags.
    const tags = try adapter.scanTags(alloc, null);
    defer {
        for (tags) |*t| {
            var tag = t.*;
            tag.deinit(alloc);
        }
        alloc.free(tags);
    }

    var session_set = std.StringHashMap(void).init(alloc);
    defer session_set.deinit();

    for (tags) |tag| {
        const session_ids = try adapter.getTagConversations(alloc, tag.tag_id);
        defer {
            for (session_ids) |s| alloc.free(s);
            alloc.free(session_ids);
        }

        for (session_ids) |sid| {
            if (!session_set.contains(sid)) {
                const key = try alloc.dupe(u8, sid);
                session_set.put(key, {}) catch {
                    alloc.free(key);
                    continue;
                };
            }
        }
    }

    // Convert to owned slice.
    var results = std.ArrayList([]const u8).empty;
    errdefer {
        for (results.items) |s| alloc.free(s);
        results.deinit(alloc);
    }

    var kit = session_set.keyIterator();
    while (kit.next()) |key| {
        try results.append(alloc, key.*);
    }
    session_set.clearRetainingCapacity();

    return results.toOwnedSlice(alloc);
}

/// Clear a StringHashMap, freeing all owned keys.
fn clearStringHashMap(alloc: Allocator, map: *std.StringHashMap(void)) void {
    var it = map.keyIterator();
    while (it.next()) |key| alloc.free(key.*);
    map.clearRetainingCapacity();
}

/// Find a tag's ID by name (case-insensitive) from a pre-scanned list.
/// Returns a borrowed slice valid for the lifetime of `tags`.
fn findTagIdByName(tags: []const TagRecord, name: []const u8) ?[]const u8 {
    for (tags) |tag| {
        if (std.ascii.eqlIgnoreCase(tag.name, name)) return tag.tag_id;
    }
    return null;
}
