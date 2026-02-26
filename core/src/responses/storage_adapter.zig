//! TaluDB adapter for session-scoped item persistence.
//!
//! Translates ItemRecord events into TaluDB block columns and restores them.

const std = @import("std");
const json = @import("../io/json/root.zig");
const kvbuf = @import("../io/kvbuf/root.zig");
const db_writer = @import("../db/writer.zig");
const db_reader = @import("../db/reader.zig");
const block_reader = @import("../db/block_reader.zig");
const db_blob_store = @import("../db/blob/store.zig");
const types = @import("../db/types.zig");
const generic = @import("../db/table/generic.zig");
const responses = @import("root.zig");

const codec = @import("codec.zig");
const scan = @import("query.zig");
const search = @import("search.zig");
const helpers = @import("helpers.zig");

const Allocator = std.mem.Allocator;
const StorageBackend = responses.StorageBackend;
const StorageEvent = responses.StorageEvent;
const ItemRecord = responses.ItemRecord;
const SessionRecord = responses.backend.SessionRecord;
const ColumnValue = db_writer.ColumnValue;
const ItemStatus = responses.ItemStatus;
const ItemType = responses.ItemType;

// ============================================================================
// Re-exports from sub-modules
// ============================================================================

pub const computeSessionHash = codec.computeSessionHash;
pub const computeGroupHash = codec.computeGroupHash;
pub const computeOptionalHash = codec.computeOptionalHash;
pub const ScannedSessionRecord = codec.ScannedSessionRecord;
pub const freeScannedSessionRecord = codec.freeScannedSessionRecord;
pub const freeScannedSessionRecords = codec.freeScannedSessionRecords;
pub const encodeSessionRecordMsgpack = codec.encodeSessionRecordMsgpack;
pub const encodeSessionRecordKvBuf = codec.encodeSessionRecordKvBuf;
pub const decodeSessionRecordMsgpack = codec.decodeSessionRecordMsgpack;
pub const decodeSessionRecordKvBuf = codec.decodeSessionRecordKvBuf;
pub const textContainsInsensitive = search.textContainsInsensitive;
pub const textFindInsensitive = search.textFindInsensitive;
pub const extractSnippet = search.extractSnippet;
pub const extractTextFromPayload = search.extractTextFromPayload;
pub const markerMatchAny = search.markerMatchAny;
pub const modelMatchesFilter = search.modelMatchesFilter;
pub const caseInsensitiveEqual = search.caseInsensitiveEqual;

pub const findColumn = helpers.findColumn;
pub const checkedRowCount = helpers.checkedRowCount;
pub const readU64At = helpers.readU64At;
pub const readI64At = helpers.readI64At;
pub const VarBytesBuffers = helpers.VarBytesBuffers;
pub const readVarBytesBuffers = helpers.readVarBytesBuffers;
pub const readU32Array = helpers.readU32Array;
pub const parseItemType = helpers.parseItemType;
pub const parseStatus = helpers.parseStatus;
pub const ParsedUsage = helpers.ParsedUsage;
pub const parseUsage = helpers.parseUsage;
pub const parseGenerationJson = helpers.parseGenerationJson;
pub const sortByItemId = helpers.sortByItemId;

// ============================================================================
// Schema / Column Constants — canonical source: responses/schema.zig
// ============================================================================

const session_schema = responses.schema;
const schema_items = session_schema.schema_items;
const schema_deletes = session_schema.schema_deletes;
const schema_sessions = session_schema.schema_sessions;
const schema_sessions_kvbuf = session_schema.schema_sessions_kvbuf;
const schema_embeddings = session_schema.schema_embeddings;
const isSessionSchema = session_schema.isSessionSchema;

const col_item_id = session_schema.col_item_id;
const col_ts = session_schema.col_ts;
const col_session_hash = session_schema.col_session_hash;
const col_embedding = session_schema.col_embedding;
const col_group_hash = session_schema.col_group_hash;
const col_head_item_id = session_schema.col_head_item_id;
const col_created_ts = session_schema.col_created_ts;
const col_ttl_ts = session_schema.col_ttl_ts;
const col_project_hash = session_schema.col_project_hash;
const col_payload = session_schema.col_payload;

const default_item_json_externalize_threshold_bytes: usize = 1 * 1024 * 1024;
const record_json_trigram_bloom_bytes: usize = 64;
const env_item_json_externalize_threshold_bytes = "TALU_DB_ITEM_JSON_EXTERNALIZE_THRESHOLD_BYTES";
const env_shared_externalize_threshold_bytes = "TALU_DB_EXTERNALIZE_THRESHOLD_BYTES";

fn resolveItemJsonExternalizeThresholdBytes() usize {
    if (readThresholdBytesFromEnvVar(env_item_json_externalize_threshold_bytes)) |threshold| return threshold;
    if (readThresholdBytesFromEnvVar(env_shared_externalize_threshold_bytes)) |threshold| return threshold;
    return default_item_json_externalize_threshold_bytes;
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

fn buildRecordJsonTrigramBloom(text: []const u8) [record_json_trigram_bloom_bytes]u8 {
    var bloom = std.mem.zeroes([record_json_trigram_bloom_bytes]u8);
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

fn trigramHashes(t0: u8, t1: u8, t2: u8) struct { usize, usize } {
    const trigram = [_]u8{ t0, t1, t2 };
    const bit_count = record_json_trigram_bloom_bytes * 8;
    const h1: usize = @intCast(std.hash.Wyhash.hash(0, &trigram) % bit_count);
    const h2: usize = @intCast(std.hash.Wyhash.hash(0x9e3779b97f4a7c15, &trigram) % bit_count);
    return .{ h1, h2 };
}

fn setBloomBit(bloom: *[record_json_trigram_bloom_bytes]u8, bit_idx: usize) void {
    const byte_idx = bit_idx / 8;
    const bit_mask: u8 = @as(u8, 1) << @as(u3, @intCast(bit_idx % 8));
    bloom[byte_idx] |= bit_mask;
}

// ============================================================================
// TableAdapter
// ============================================================================

/// TableAdapter - session-scoped TaluDB adapter for item persistence.
///
/// Thread safety: NOT thread-safe (single-writer semantics via <ns>/talu.lock).
pub const TableAdapter = struct {
    allocator: Allocator,
    table: generic.Table,
    session_id: []const u8,
    session_hash: u64,
    /// Legacy SipHash for backward-compat reads of pre-migration data.
    /// Non-zero only when bound to a specific session (init or loadConversation).
    legacy_session_hash: u64 = 0,
    blob_store: db_blob_store.BlobStore,
    item_json_externalize_threshold_bytes: usize,

    /// Initialize a TaluDB-backed table adapter with write capabilities.
    /// Acquires a lock; returns error.LockUnavailable if another process holds the lock.
    pub fn init(allocator: Allocator, db_root: []const u8, session_id: []const u8) !TableAdapter {
        const session_copy = try allocator.dupe(u8, session_id);
        errdefer allocator.free(session_copy);

        const tables_root = try std.fs.path.join(allocator, &.{ db_root, "tables" });
        defer allocator.free(tables_root);

        var tbl = try generic.Table.open(
            allocator,
            tables_root,
            "chat",
            session_schema.session_compaction_policy,
        );
        errdefer tbl.deinit();

        var blob_store = try db_blob_store.BlobStore.init(allocator, db_root);
        errdefer blob_store.deinit();

        return .{
            .allocator = allocator,
            .table = tbl,
            .session_id = session_copy,
            .session_hash = computeSessionHash(session_id),
            .legacy_session_hash = codec.computeLegacySipHash(session_id),
            .blob_store = blob_store,
            .item_json_externalize_threshold_bytes = resolveItemJsonExternalizeThresholdBytes(),
        };
    }

    /// Initialize a read-only TaluDB adapter for scanning sessions.
    /// Does not acquire a lock; safe to run concurrently with writers.
    /// Returns a stack-allocated adapter. Caller must call deinitReadOnly() when done.
    pub fn initReadOnly(alloc: Allocator, db_root: []const u8) !TableAdapter {
        const tables_root = try std.fs.path.join(alloc, &.{ db_root, "tables" });
        defer alloc.free(tables_root);

        var tbl = try generic.Table.openReadOnly(alloc, tables_root, "chat", session_schema.session_compaction_policy);
        errdefer tbl.deinit();

        var blob_store = try db_blob_store.BlobStore.init(alloc, db_root);
        errdefer blob_store.deinit();

        return .{
            .allocator = alloc,
            .table = tbl,
            .session_id = "",
            .session_hash = 0,
            .blob_store = blob_store,
            .item_json_externalize_threshold_bytes = resolveItemJsonExternalizeThresholdBytes(),
        };
    }

    /// Create a heap-allocated TableAdapter and return its StorageBackend.
    /// The StorageBackend owns the adapter — calling deinit() on the backend
    /// frees the adapter and its heap allocation. Used by forkSession to
    /// attach a write-capable backend to a destination Conversation.
    pub fn initOwned(alloc: Allocator, db_root: []const u8, session_id: []const u8) !OwnedBackend {
        const self_ptr = try alloc.create(TableAdapter);
        errdefer alloc.destroy(self_ptr);
        self_ptr.* = try TableAdapter.init(alloc, db_root, session_id);
        return .{
            .adapter = self_ptr,
            .storage_backend = responses.StorageBackend{
                .ptr = self_ptr,
                .vtable = &owned_vtable,
            },
        };
    }

    pub const OwnedBackend = struct {
        adapter: *TableAdapter,
        storage_backend: responses.StorageBackend,
    };

    /// VTable that also frees the heap-allocated TableAdapter on deinit.
    const owned_vtable = responses.StorageBackend.VTable{
        .onEvent = onEvent,
        .loadAll = loadAll,
        .deinit = deinitOwned,
    };

    fn deinitOwned(ctx: *anyopaque) void {
        const self: *TableAdapter = @ptrCast(@alignCast(ctx));
        deinit(ctx); // flush + release internal resources
        self.allocator.destroy(self);
    }

    /// Deinitialize a read-only adapter (created with initReadOnly).
    pub fn deinitReadOnly(self: *TableAdapter) void {
        self.table.deinit();
        self.blob_store.deinit();
    }

    /// Get a StorageBackend interface for this adapter.
    pub fn backend(self: *TableAdapter) StorageBackend {
        return .{ .ptr = self, .vtable = &vtable };
    }

    const vtable = StorageBackend.VTable{
        .onEvent = onEvent,
        .loadAll = loadAll,
        .deinit = deinit,
    };

    fn onEvent(ctx: *anyopaque, event: *const StorageEvent) anyerror!void {
        const self: *TableAdapter = @ptrCast(@alignCast(ctx));
        switch (event.*) {
            .PutItems => |records| {
                for (records) |record| {
                    try self.writeItem(record);
                }
            },
            .PutItem => |record| try self.writeItem(record),
            .DeleteItem => |del| try self.writeDelete(del.item_id, del.deleted_at_ms),
            .ClearItems => |clr| try self.writeClear(clr.cleared_at_ms),
            .PutSession => |record| try self.writeSession(record),
            .BeginFork => {},
            .EndFork => {},
        }
    }

    fn loadAll(ctx: *anyopaque, allocator: Allocator) anyerror![]ItemRecord {
        const self: *TableAdapter = @ptrCast(@alignCast(ctx));
        _ = try self.table.fs_reader.refreshIfChanged();
        const blocks = try self.table.fs_reader.getBlocks(allocator);
        defer allocator.free(blocks);

        var items = std.ArrayList(ItemRecord).empty;
        errdefer {
            for (items.items) |*record| {
                record.deinit(allocator);
            }
            items.deinit(allocator);
        }

        for (blocks) |block| {
            var file = try self.table.fs_reader.dir.openFile(block.path, .{ .mode = .read_only });
            defer file.close();

            const reader = block_reader.BlockReader.init(file, allocator);
            const header = try reader.readHeader(block.offset);
            if (header.schema_id != schema_items) continue;

            const descs = try reader.readColumnDirectory(header, block.offset);
            defer allocator.free(descs);

            const row_count = header.row_count;
            if (row_count == 0) continue;

            const session_desc = findColumn(descs, col_session_hash) orelse return error.MissingColumn;
            const session_bytes = try reader.readColumnData(block.offset, session_desc, allocator);
            defer allocator.free(session_bytes);

            const rows = try checkedRowCount(row_count, session_bytes.len, 8);
            var matching = std.ArrayList(usize).empty;
            defer matching.deinit(allocator);

            for (0..rows) |row_idx| {
                const hash = try readU64At(session_bytes, row_idx);
                if (hash == self.session_hash or hash == self.legacy_session_hash) {
                    try matching.append(allocator, row_idx);
                }
            }

            if (matching.items.len == 0) continue;

            const id_desc = findColumn(descs, col_item_id) orelse return error.MissingColumn;
            const ts_desc = findColumn(descs, col_ts) orelse return error.MissingColumn;
            const payload_desc = findColumn(descs, col_payload) orelse return error.MissingColumn;

            const id_bytes = try reader.readColumnData(block.offset, id_desc, allocator);
            defer allocator.free(id_bytes);
            _ = try checkedRowCount(row_count, id_bytes.len, 8);

            const ts_bytes = try reader.readColumnData(block.offset, ts_desc, allocator);
            defer allocator.free(ts_bytes);
            _ = try checkedRowCount(row_count, ts_bytes.len, 8);

            var payload_buffers = try readVarBytesBuffers(file, block.offset, payload_desc, row_count, allocator);
            defer payload_buffers.deinit(allocator);

            for (matching.items) |row_idx| {
                const item_id = try readU64At(id_bytes, row_idx);
                const created_at_ms = try readI64At(ts_bytes, row_idx);
                const payload = try payload_buffers.sliceForRow(row_idx);
                const record_opt = try self.decodePayload(allocator, payload, item_id, created_at_ms);
                if (record_opt) |record| {
                    try items.append(allocator, record);
                }
            }
        }

        std.mem.sort(ItemRecord, items.items, {}, sortByItemId);
        return items.toOwnedSlice(allocator);
    }

    fn deinit(ctx: *anyopaque) void {
        const self: *TableAdapter = @ptrCast(@alignCast(ctx));
        self.table.deinit();
        self.blob_store.deinit();
        self.allocator.free(self.session_id);
    }

    /// Simulates a process crash for testing.
    ///
    /// Releases all resources (closing fds, releasing flocks) WITHOUT
    /// flushing pending data or deleting the WAL file. Leaves an
    /// orphaned WAL on disk that the next Writer.open will replay.
    pub fn simulateCrash(self: *TableAdapter) void {
        const writer = self.table.fs_writer orelse unreachable;
        writer.simulateCrash();
        self.allocator.destroy(writer);
        self.table.fs_writer = null;
        self.table.fs_reader.deinit();
        self.allocator.destroy(self.table.fs_reader);
        self.blob_store.deinit();
        self.allocator.free(self.session_id);
    }

    // ========================================================================
    // Write Operations
    // ========================================================================

    fn writeItem(self: *TableAdapter, record: ItemRecord) !void {

        const record_json_z = try responses.serializeItemRecordToJsonZ(self.allocator, record);
        defer self.allocator.free(record_json_z);
        const record_json = std.mem.sliceTo(record_json_z, 0);

        var record_json_inline: ?[]const u8 = record_json;
        var record_json_ref: ?[]const u8 = null;
        var record_json_ref_buf: [db_blob_store.ref_len]u8 = undefined; // Written by @memcpy below when externalized
        var record_json_ref_len: usize = 0;
        var record_json_trigram_bloom: ?[]const u8 = null;
        var record_json_trigram_bloom_buf: [record_json_trigram_bloom_bytes]u8 = undefined; // Written by computeTrigramBloom below when externalized

        if (record_json.len > self.item_json_externalize_threshold_bytes) {
            const blob_ref = try self.blob_store.putAuto(record_json);
            const ref_slice = blob_ref.refSlice();
            @memcpy(record_json_ref_buf[0..ref_slice.len], ref_slice);
            record_json_ref_len = ref_slice.len;
            record_json_trigram_bloom_buf = buildRecordJsonTrigramBloom(record_json);
            record_json_inline = null;
            record_json_ref = record_json_ref_buf[0..record_json_ref_len];
            record_json_trigram_bloom = record_json_trigram_bloom_buf[0..];
        }

        const payload = try responses.serializeItemRecordToKvBufWithStorage(
            self.allocator,
            record,
            self.session_id,
            record_json_inline,
            record_json_ref,
            record_json_trigram_bloom,
        );
        defer self.allocator.free(payload);

        var item_id_value = record.item_id;
        var ts_value = record.created_at_ms;
        var hash_value = self.session_hash;

        const columns = [_]ColumnValue{
            .{ .column_id = col_item_id, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&item_id_value) },
            .{ .column_id = col_ts, .shape = .SCALAR, .phys_type = .I64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&ts_value) },
            .{ .column_id = col_session_hash, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&hash_value) },
            .{ .column_id = col_payload, .shape = .VARBYTES, .phys_type = .BINARY, .encoding = .RAW, .dims = 0, .data = payload },
        };

        try self.table.appendRow(schema_items, &columns);
    }

    fn writeDelete(self: *TableAdapter, item_id: u64, deleted_at_ms: i64) !void {

        var item_id_value = item_id;
        var ts_value = deleted_at_ms;
        var hash_value = self.session_hash;

        const columns = [_]ColumnValue{
            .{ .column_id = col_item_id, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&item_id_value) },
            .{ .column_id = col_ts, .shape = .SCALAR, .phys_type = .I64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&ts_value) },
            .{ .column_id = col_session_hash, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&hash_value) },
        };

        try self.table.appendRow(schema_deletes, &columns);
    }

    fn writeClear(self: *TableAdapter, cleared_at_ms: i64) !void {

        var target = self.session_hash;
        var ts_value = cleared_at_ms;
        var hash_value = self.session_hash;

        const columns = [_]ColumnValue{
            .{ .column_id = col_item_id, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&target) },
            .{ .column_id = col_ts, .shape = .SCALAR, .phys_type = .I64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&ts_value) },
            .{ .column_id = col_session_hash, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&hash_value) },
        };

        try self.table.appendRow(schema_deletes, &columns);
    }

    fn writeSession(self: *TableAdapter, record: SessionRecord) !void {
        const payload = try encodeSessionRecordKvBuf(self.allocator, record);
        defer self.allocator.free(payload);

        const group_hash = computeOptionalHash(record.group_id);
        const project_hash = computeOptionalHash(record.project_id);
        var session_hash_value = self.session_hash;
        var ts_value = record.updated_at_ms;
        var group_hash_value = group_hash;
        var project_hash_value = project_hash;
        var head_item_id_value = record.head_item_id;
        var created_ts_value = record.created_at_ms;
        var ttl_ts_value = record.ttl_ts;

        const columns = [_]ColumnValue{
            .{ .column_id = col_session_hash, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&session_hash_value) },
            .{ .column_id = col_ts, .shape = .SCALAR, .phys_type = .I64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&ts_value) },
            .{ .column_id = col_group_hash, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&group_hash_value) },
            .{ .column_id = col_project_hash, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&project_hash_value) },
            .{ .column_id = col_head_item_id, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&head_item_id_value) },
            .{ .column_id = col_created_ts, .shape = .SCALAR, .phys_type = .I64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&created_ts_value) },
            .{ .column_id = col_ttl_ts, .shape = .SCALAR, .phys_type = .I64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&ttl_ts_value) },
            .{ .column_id = col_payload, .shape = .VARBYTES, .phys_type = .BINARY, .encoding = .RAW, .dims = 0, .data = payload },
        };

        try self.table.appendRow(schema_sessions_kvbuf, &columns);
    }

    /// Append an embedding row to TaluDB.
    pub fn writeEmbedding(
        self: *TableAdapter,
        doc_id: u64,
        vector: []const f32,
        payload: []const u8,
    ) !void {
        const vector_bytes = std.mem.sliceAsBytes(vector);
        var doc_id_value = doc_id;
        var ts_value = std.time.milliTimestamp();
        var group_hash_value: u64 = 0;
        var ttl_ts_value: i64 = 0;

        const columns = [_]ColumnValue{
            .{ .column_id = col_item_id, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&doc_id_value) },
            .{ .column_id = col_ts, .shape = .SCALAR, .phys_type = .I64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&ts_value) },
            .{ .column_id = col_group_hash, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&group_hash_value) },
            .{ .column_id = col_ttl_ts, .shape = .SCALAR, .phys_type = .I64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&ttl_ts_value) },
            .{ .column_id = col_embedding, .shape = .VECTOR, .phys_type = .F32, .encoding = .RAW, .dims = @intCast(vector.len), .data = vector_bytes },
            .{ .column_id = col_payload, .shape = .VARBYTES, .phys_type = .BINARY, .encoding = .RAW, .dims = 0, .data = payload },
        };

        try self.table.appendRow(schema_embeddings, &columns);
    }

    // ========================================================================
    // Scan Wrappers (delegate to scan.zig)
    // ========================================================================

    /// Re-export ScanParams from scan module.
    pub const ScanParams = scan.ScanParams;

    pub fn scanSessions(
        self: *TableAdapter,
        alloc: Allocator,
        target_hash: ?u64,
    ) ![]ScannedSessionRecord {
        return self.scanSessionsFiltered(alloc, .{ .target_hash = target_hash });
    }

    pub fn scanSessionsFiltered(
        self: *TableAdapter,
        alloc: Allocator,
        params: ScanParams,
    ) ![]ScannedSessionRecord {
        return scan.scanSessionsFiltered(self.table.fs_reader, &self.blob_store, alloc, params);
    }

    pub fn collectContentMatchHashes(
        self: *TableAdapter,
        alloc: Allocator,
        query: []const u8,
    ) !std.AutoHashMap(u64, []const u8) {
        return scan.collectContentMatchHashes(self.table.fs_reader, &self.blob_store, alloc, query);
    }

    // ========================================================================
    // Session Lookups and Mutations
    // ========================================================================

    /// Look up a single session by exact session_id match.
    /// Tries Wyhash first, falls back to legacy SipHash for pre-migration data.
    ///
    /// Returns the matched record. Caller owns the returned record and
    /// must free it with freeScannedSessionRecord().
    /// Returns error.SessionNotFound if no matching session exists.
    pub fn lookupSession(self: *TableAdapter, alloc: Allocator, session_id_slice: []const u8) !ScannedSessionRecord {
        const target_hash = computeSessionHash(session_id_slice);
        if (lookupSessionByHash(self, alloc, session_id_slice, target_hash)) |record| {
            return record;
        } else |err| switch (err) {
            error.SessionNotFound => {
                // Fall back to legacy SipHash for pre-migration data.
                const legacy_hash = codec.computeLegacySipHash(session_id_slice);
                if (legacy_hash != target_hash) {
                    return lookupSessionByHash(self, alloc, session_id_slice, legacy_hash);
                }
                return error.SessionNotFound;
            },
            else => return err,
        }
    }

    fn lookupSessionByHash(self: *TableAdapter, alloc: Allocator, session_id_slice: []const u8, target_hash: u64) !ScannedSessionRecord {
        const records = try self.scanSessions(alloc, target_hash);
        defer alloc.free(records);

        if (records.len == 0) return error.SessionNotFound;

        // Verify session_id matches (hash collision protection)
        if (!std.mem.eql(u8, records[0].session_id, session_id_slice)) {
            for (records) |*r| freeScannedSessionRecord(alloc, @constCast(r));
            return error.SessionNotFound;
        }

        // Transfer ownership of the first record; free the rest
        const result = records[0];
        for (records[1..]) |*r| freeScannedSessionRecord(alloc, @constCast(r));
        return result;
    }

    /// Update session metadata (read-modify-write).
    /// Requires write lock (must be initialized with init, not initReadOnly).
    ///
    /// Reads the current session head, merges non-null fields, and writes
    /// a new session record. Never touches conversation items.
    /// Returns error.SessionNotFound if the session does not exist.
    pub fn updateSession(
        self: *TableAdapter,
        alloc: Allocator,
        session_id: []const u8,
        new_title: ?[]const u8,
        new_marker: ?[]const u8,
        new_metadata_json: ?[]const u8,
    ) !void {
        return self.updateSessionEx(alloc, session_id, new_title, new_marker, new_metadata_json, null, null, false);
    }

    /// Extended update session with source_doc_id and project_id support.
    /// Same as updateSession but allows setting the source document link
    /// and project assignment. Set `clear_project_id` to explicitly remove
    /// a session from its project.
    pub fn updateSessionEx(
        self: *TableAdapter,
        alloc: Allocator,
        session_id: []const u8,
        new_title: ?[]const u8,
        new_marker: ?[]const u8,
        new_metadata_json: ?[]const u8,
        new_source_doc_id: ?[]const u8,
        new_project_id: ?[]const u8,
        clear_project_id: bool,
    ) !void {
        // Read current session head
        const current = try self.lookupSession(alloc, session_id);
        defer {
            var mutable = current;
            freeScannedSessionRecord(alloc, &mutable);
        }

        // Merge: replace only non-null fields, preserve everything else
        const now_ms = std.time.milliTimestamp();
        const merged = SessionRecord{
            .session_id = current.session_id,
            .model = current.model,
            .title = if (new_title) |t| t else current.title,
            .system_prompt = current.system_prompt,
            .config_json = current.config_json,
            .marker = if (new_marker) |s| s else current.marker,
            .parent_session_id = current.parent_session_id,
            .group_id = current.group_id,
            .project_id = if (clear_project_id) null else if (new_project_id) |p| p else current.project_id,
            .head_item_id = current.head_item_id,
            .ttl_ts = current.ttl_ts,
            .metadata_json = if (new_metadata_json) |m| m else current.metadata_json,
            .source_doc_id = if (new_source_doc_id) |d| d else current.source_doc_id,
            .created_at_ms = current.created_at_ms,
            .updated_at_ms = now_ms,
        };

        try self.writeSession(merged);
        try self.table.flush();
    }

    /// Delete a session by writing tombstone markers.
    /// Requires write lock (must be initialized with init, not initReadOnly).
    ///
    /// Writes:
    /// 1. SessionRecord with marker="deleted" (Schema 4)
    /// 2. ClearItems marker (Schema 2)
    pub fn deleteSession(self: *TableAdapter, session_id: []const u8) !void {
        // Verify this session actually exists
        const target_hash = computeSessionHash(session_id);

        // 1. Session Tombstone (Schema 4) - Hides from scanSessions
        const now_ms = std.time.milliTimestamp();
        const tombstone = SessionRecord{
            .session_id = session_id,
            .marker = "deleted",
            .model = null,
            .title = null,
            .system_prompt = null,
            .config_json = null,
            .parent_session_id = null,
            .group_id = null,
            .head_item_id = 0,
            .ttl_ts = 0,
            .metadata_json = null,
            .created_at_ms = now_ms,
            .updated_at_ms = now_ms,
        };
        try self.writeSession(tombstone);

        // 2. Clear Items marker (Schema 2) - Hides items from restoreSession
        // Use special marker: item_id = session_hash to indicate session-wide clear
        var item_id_value = target_hash;
        var ts_value = now_ms;
        var hash_value = target_hash;

        const columns = [_]db_writer.ColumnValue{
            .{ .column_id = col_item_id, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&item_id_value) },
            .{ .column_id = col_ts, .shape = .SCALAR, .phys_type = .I64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&ts_value) },
            .{ .column_id = col_session_hash, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&hash_value) },
        };

        try self.table.appendRow(schema_deletes, &columns);

        // 3. Flush to ensure both markers are durable
        try self.table.flush();
    }

    // ========================================================================
    // Payload Decoding
    // ========================================================================

    fn decodePayload(
        self: *TableAdapter,
        allocator: Allocator,
        payload: []const u8,
        item_id: u64,
        created_at_ms: i64,
    ) !?ItemRecord {
        if (kvbuf.isKvBuf(payload)) {
            return self.decodeKvBufPayload(allocator, payload, item_id, created_at_ms);
        }
        return self.decodeJsonPayload(allocator, payload, item_id, created_at_ms);
    }

    /// Decode a KvBuf-encoded payload blob.
    fn decodeKvBufPayload(
        self: *TableAdapter,
        allocator: Allocator,
        payload: []const u8,
        item_id: u64,
        created_at_ms: i64,
    ) !?ItemRecord {
        const reader = kvbuf.KvBufReader.init(payload) catch return error.InvalidPayload;

        const session_str = reader.get(kvbuf.FieldIds.session_id) orelse return error.InvalidPayload;
        if (!std.mem.eql(u8, session_str, self.session_id)) {
            return null;
        }

        if (reader.get(kvbuf.FieldIds.record_json)) |record_json| {
            return self.parseRecordJson(allocator, record_json, item_id, created_at_ms);
        }
        if (reader.get(kvbuf.FieldIds.record_json_ref)) |record_json_ref| {
            const loaded_json = try self.blob_store.readAll(record_json_ref, allocator);
            defer allocator.free(loaded_json);
            return self.parseRecordJson(allocator, loaded_json, item_id, created_at_ms);
        }
        return error.InvalidPayload;
    }

    /// Decode a legacy JSON-encoded payload (backward compatibility).
    fn decodeJsonPayload(
        self: *TableAdapter,
        allocator: Allocator,
        payload: []const u8,
        item_id: u64,
        created_at_ms: i64,
    ) !?ItemRecord {
        var parsed = json.parseValue(allocator, payload, .{
            .max_size_bytes = 50 * 1024 * 1024,
            .max_value_bytes = 50 * 1024 * 1024,
            .max_string_bytes = 50 * 1024 * 1024,
        }) catch |err| {
            return switch (err) {
                error.InputTooLarge => error.InvalidPayload,
                error.InputTooDeep => error.InvalidPayload,
                error.StringTooLong => error.InvalidPayload,
                error.InvalidJson => error.InvalidPayload,
                error.OutOfMemory => error.OutOfMemory,
            };
        };
        defer parsed.deinit();

        const obj = switch (parsed.value) {
            .object => |o| o,
            else => return error.InvalidPayload,
        };
        const session_value = obj.get("session_id") orelse return error.InvalidPayload;
        const record_value = obj.get("record") orelse return error.InvalidPayload;

        const session_str = switch (session_value) {
            .string => |s| s,
            else => return error.InvalidPayload,
        };

        if (!std.mem.eql(u8, session_str, self.session_id)) {
            return null;
        }

        const record_obj = switch (record_value) {
            .object => record_value,
            else => return error.InvalidPayload,
        };

        const record_json = try std.json.Stringify.valueAlloc(allocator, record_obj, .{});
        defer allocator.free(record_json);

        return self.parseRecordJson(allocator, record_json, item_id, created_at_ms);
    }

    /// Parse a record JSON string into an ItemRecord.
    /// Shared by both KvBuf and legacy JSON decode paths.
    fn parseRecordJson(
        self: *TableAdapter,
        allocator: Allocator,
        record_json: []const u8,
        item_id: u64,
        created_at_ms: i64,
    ) !?ItemRecord {
        _ = self;

        // Parse to extract status and item_type from the JSON
        var parsed = json.parseValue(allocator, record_json, .{
            .max_size_bytes = 50 * 1024 * 1024,
            .max_value_bytes = 50 * 1024 * 1024,
            .max_string_bytes = 50 * 1024 * 1024,
        }) catch |err| {
            return switch (err) {
                error.InputTooLarge => error.InvalidPayload,
                error.InputTooDeep => error.InvalidPayload,
                error.StringTooLong => error.InvalidPayload,
                error.InvalidJson => error.InvalidPayload,
                error.OutOfMemory => error.OutOfMemory,
            };
        };
        defer parsed.deinit();

        const status = parseStatus(parsed.value) orelse ItemStatus.completed;
        const item_type = parseItemType(parsed.value);

        var variant = try responses.parseItemVariantRecord(allocator, record_json, status);
        errdefer freeItemVariant(allocator, &variant);

        const usage = parseUsage(allocator, parsed.value);
        const generation_json = parseGenerationJson(allocator, parsed.value);

        return ItemRecord{
            .item_id = item_id,
            .created_at_ms = created_at_ms,
            .ttl_ts = 0,
            .status = status,
            .hidden = false,
            .pinned = false,
            .json_valid = false,
            .schema_valid = false,
            .repaired = false,
            .parent_item_id = null,
            .origin_session_id = null,
            .origin_item_id = null,
            .finish_reason = usage.finish_reason,
            .prefill_ns = usage.prefill_ns,
            .generation_ns = usage.generation_ns,
            .input_tokens = usage.input_tokens,
            .output_tokens = usage.output_tokens,
            .item_type = item_type,
            .variant = variant,
            .metadata = null,
            .generation_json = generation_json,
        };
    }
};

// =============================================================================
// Standalone Functions
// =============================================================================

fn freeItemVariant(allocator: Allocator, variant: *responses.backend.ItemVariantRecord) void {
    const record = ItemRecord{
        .item_id = 0,
        .created_at_ms = 0,
        .item_type = .message,
        .variant = variant.*,
    };
    var tmp = record;
    tmp.deinit(allocator);
}

pub fn loadConversation(alloc: Allocator, db_root: []const u8, session_id_slice: []const u8) !*responses.Conversation {
    var adapter = try TableAdapter.initReadOnly(alloc, db_root);
    adapter.session_id = session_id_slice;
    adapter.session_hash = computeSessionHash(session_id_slice);
    adapter.legacy_session_hash = codec.computeLegacySipHash(session_id_slice);

    const be = adapter.backend();
    const records = be.loadAll(alloc) catch |err| {
        adapter.deinitReadOnly();
        return err;
    };
    defer responses.backend.freeItemRecords(alloc, records);
    adapter.deinitReadOnly();

    const conv = try responses.Conversation.initWithSession(alloc, session_id_slice);
    errdefer conv.deinit();

    if (records.len > 0) {
        try conv.loadItemRecords(records);
    }
    return conv;
}

/// Fork a conversation at a specific item, creating a new session.
///
/// Loads the source conversation, clones items up to and including
/// `target_item_id` into a new session, and writes a session record with
/// `parent_session_id` pointing back to the source. Cloned items carry
/// origin lineage fields set by `cloneFromPrefix`.
///
/// Acquires a write lock for the new session. Returns error.LockUnavailable
/// if the database is locked by another process.
///
/// Returns error.ItemNotFound if `target_item_id` does not exist in the source.
pub fn forkSession(
    alloc: Allocator,
    db_root: []const u8,
    source_session_id: []const u8,
    target_item_id: u64,
    new_session_id: []const u8,
) !void {
    // 1. Load source conversation (read-only snapshot, no storage backend)
    const source = try loadConversation(alloc, db_root, source_session_id);
    defer source.deinit();

    // 2. Find item index by item_id
    const item = source.findById(target_item_id) orelse return error.ItemNotFound;
    const items_base = source.items_list.items.ptr;
    const item_ptr: [*]const responses.Item = @ptrCast(item);
    const last_index = (@intFromPtr(item_ptr) - @intFromPtr(items_base)) / @sizeOf(responses.Item);

    // 3. Create destination conversation with heap-owned TaluDB backend
    var owned = try TableAdapter.initOwned(alloc, db_root, new_session_id);
    // Conversation.deinit calls storage_backend.deinit which frees the adapter
    const dest = try responses.Conversation.initWithStorage(alloc, new_session_id, owned.storage_backend);
    defer dest.deinit();

    // 4. Fork transaction: beginFork → cloneFromPrefix → endFork
    const fork_id = dest.beginFork();
    try dest.cloneFromPrefix(source, last_index, 0);
    dest.endFork(fork_id);

    if (dest.hasStorageError()) return error.StorageForkFailed;

    // 5. Read source session metadata to inherit model/group_id
    var read_adapter = try TableAdapter.initReadOnly(alloc, db_root);
    defer read_adapter.deinitReadOnly();
    const source_session = read_adapter.lookupSession(alloc, source_session_id) catch |err| switch (err) {
        error.SessionNotFound => null,
        else => return err,
    };
    defer if (source_session) |s| {
        var mutable = s;
        freeScannedSessionRecord(alloc, &mutable);
    };

    // 6. Write session record for the new session
    const now_ms = std.time.milliTimestamp();
    const head_item_id: u64 = if (dest.items_list.items.len > 0)
        dest.items_list.items[dest.items_list.items.len - 1].id
    else
        0;

    const session_record = SessionRecord{
        .session_id = new_session_id,
        .model = if (source_session) |s| s.model else null,
        .title = null,
        .system_prompt = if (source_session) |s| s.system_prompt else null,
        .config_json = if (source_session) |s| s.config_json else null,
        .marker = "active",
        .parent_session_id = source_session_id,
        .group_id = if (source_session) |s| s.group_id else null,
        .project_id = if (source_session) |s| s.project_id else null,
        .head_item_id = head_item_id,
        .ttl_ts = 0,
        .metadata_json = null,
        .created_at_ms = now_ms,
        .updated_at_ms = now_ms,
    };

    try owned.adapter.writeSession(session_record);
    try owned.adapter.table.flush();
}

// =============================================================================
// High-Level API Functions (for capi thin wrappers)
// =============================================================================

/// List sessions from a TaluDB directory with filtering.
/// This is the high-level API intended for capi thin wrappers.
/// Handles adapter lifecycle internally.
/// Caller owns returned records; free with freeScannedSessionRecords().
pub fn listSessions(
    alloc: Allocator,
    db_path: []const u8,
    params: TableAdapter.ScanParams,
) ![]ScannedSessionRecord {
    var adapter = try TableAdapter.initReadOnly(alloc, db_path);
    defer adapter.deinitReadOnly();
    return adapter.scanSessionsFiltered(alloc, params);
}

/// Get a single session by ID from a TaluDB directory.
/// This is the high-level API intended for capi thin wrappers.
/// Handles adapter lifecycle internally.
/// Returns null if session not found.
/// Caller owns returned record; free with freeScannedSessionRecord().
pub fn getSessionInfo(
    alloc: Allocator,
    db_path: []const u8,
    session_id: []const u8,
) !?ScannedSessionRecord {
    var adapter = try TableAdapter.initReadOnly(alloc, db_path);
    defer adapter.deinitReadOnly();
    return adapter.lookupSession(alloc, session_id) catch |err| switch (err) {
        error.SessionNotFound => return null,
        else => return err,
    };
}

// =============================================================================
// Tests
// =============================================================================

test "TableAdapter.init computes session hash" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var be = try TableAdapter.init(std.testing.allocator, root_path, "session-1");
    defer be.backend().deinit();

    try std.testing.expectEqual(computeSessionHash("session-1"), be.session_hash);
    try std.testing.expectEqual(codec.computeLegacySipHash("session-1"), be.legacy_session_hash);
    try std.testing.expect(be.session_hash != be.legacy_session_hash);
}

test "TableAdapter.backend returns StorageBackend" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var be = try TableAdapter.init(std.testing.allocator, root_path, "session-2");
    const storage = be.backend();
    storage.deinit();
}

test "parseThresholdBytes parses valid decimal thresholds" {
    try std.testing.expectEqual(@as(?usize, 0), parseThresholdBytes("0"));
    try std.testing.expectEqual(@as(?usize, 1048576), parseThresholdBytes("1048576"));
    try std.testing.expectEqual(@as(?usize, 64), parseThresholdBytes(" 64 "));
}

test "parseThresholdBytes rejects invalid threshold values" {
    try std.testing.expectEqual(@as(?usize, null), parseThresholdBytes(""));
    try std.testing.expectEqual(@as(?usize, null), parseThresholdBytes(" "));
    try std.testing.expectEqual(@as(?usize, null), parseThresholdBytes("abc"));
    try std.testing.expectEqual(@as(?usize, null), parseThresholdBytes("-1"));
}

test "TableAdapter.onEvent writes wal" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var be = try TableAdapter.init(std.testing.allocator, root_path, "session-3");
    const storage = be.backend();

    const record = ItemRecord{
        .item_id = 1,
        .created_at_ms = 123,
        .item_type = .message,
        .variant = .{ .message = .{ .role = .user, .status = .completed, .content = &.{} } },
    };

    const event = StorageEvent{ .PutItem = record };
    try storage.onEvent(&event);

    const wal_path = try std.fmt.allocPrint(std.testing.allocator, "tables/chat/{s}", .{be.table.fs_writer.?.wal_name});
    defer std.testing.allocator.free(wal_path);
    const wal_stat = try tmp.dir.statFile(wal_path);
    try std.testing.expect(wal_stat.size > 0);

    storage.deinit();
}

test "TableAdapter.onEvent writes session metadata" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var be = try TableAdapter.init(std.testing.allocator, root_path, "session-5");
    const storage = be.backend();

    const session = SessionRecord{
        .session_id = "session-5",
        .model = "model-x",
        .title = "Title",
        .system_prompt = "System",
        .config_json = "{\"temperature\":0.5}",
        .marker = "active",
        .parent_session_id = null,
        .group_id = "group-1",
        .head_item_id = 7,
        .ttl_ts = 0,
        .metadata_json = "{\"source\":\"test\"}",
        .created_at_ms = 100,
        .updated_at_ms = 200,
    };

    const event = StorageEvent{ .PutSession = session };
    try storage.onEvent(&event);

    const wal_path = try std.fmt.allocPrint(std.testing.allocator, "tables/chat/{s}", .{be.table.fs_writer.?.wal_name});
    defer std.testing.allocator.free(wal_path);
    const wal_stat = try tmp.dir.statFile(wal_path);
    try std.testing.expect(wal_stat.size > 0);

    storage.deinit();
}

test "TableAdapter writes embeddings" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var be = try TableAdapter.init(std.testing.allocator, root_path, "session-6");
    defer be.backend().deinit();

    const vector = [_]f32{ 0.5, -1.25, 2.0 };
    const payload = "payload";
    try be.writeEmbedding(42, &vector, payload);

    const wal_path = try std.fmt.allocPrint(std.testing.allocator, "tables/chat/{s}", .{be.table.fs_writer.?.wal_name});
    defer std.testing.allocator.free(wal_path);
    const wal_stat = try tmp.dir.statFile(wal_path);
    try std.testing.expect(wal_stat.size > 0);
}

test "TableAdapter.loadAll returns records" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    {
        var be = try TableAdapter.init(std.testing.allocator, root_path, "session-4");
        defer be.backend().deinit();

        const record = ItemRecord{
            .item_id = 9,
            .created_at_ms = 500,
            .item_type = .message,
            .variant = .{ .message = .{ .role = .assistant, .status = .completed, .content = &.{} } },
        };

        const event = StorageEvent{ .PutItem = record };
        try be.backend().onEvent(&event);
        try be.table.fs_writer.?.flushBlock();
    }

    var backend2 = try TableAdapter.init(std.testing.allocator, root_path, "session-4");
    const storage2 = backend2.backend();
    const records = try storage2.loadAll(std.testing.allocator);
    defer responses.backend.freeItemRecords(std.testing.allocator, records);
    storage2.deinit();

    try std.testing.expectEqual(@as(usize, 1), records.len);
    try std.testing.expectEqual(@as(u64, 9), records[0].item_id);
    try std.testing.expectEqual(@as(i64, 500), records[0].created_at_ms);
}

test "TableAdapter externalizes large item record_json and loadAll resolves it" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    const long_text = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    {
        var be = try TableAdapter.init(std.testing.allocator, root_path, "session-large-json");
        defer be.backend().deinit();
        be.item_json_externalize_threshold_bytes = 64;

        var content = [_]responses.backend.ItemContentPartRecord{
            .{ .input_text = .{ .text = long_text } },
        };
        const record = ItemRecord{
            .item_id = 22,
            .created_at_ms = 777,
            .item_type = .message,
            .variant = .{ .message = .{ .role = .user, .status = .completed, .content = &content } },
        };
        try be.backend().onEvent(&StorageEvent{ .PutItem = record });
        try be.table.fs_writer.?.flushBlock();
    }

    // Verify stored row uses record_json_ref rather than inline record_json.
    {
        var ro = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
        defer ro.deinitReadOnly();

        const blocks = try ro.table.fs_reader.getBlocks(std.testing.allocator);
        defer std.testing.allocator.free(blocks);

        var found_items_block = false;
        for (blocks) |block| {
            var file = ro.table.fs_reader.dir.openFile(block.path, .{ .mode = .read_only }) catch continue;
            defer file.close();

            const reader = block_reader.BlockReader.init(file, std.testing.allocator);
            const header = reader.readHeader(block.offset) catch continue;
            if (header.schema_id != schema_items or header.row_count == 0) continue;
            found_items_block = true;

            const descs = try reader.readColumnDirectory(header, block.offset);
            defer std.testing.allocator.free(descs);
            const payload_desc = findColumn(descs, col_payload) orelse return error.TestUnexpectedResult;

            var payloads = try readVarBytesBuffers(file, block.offset, payload_desc, header.row_count, std.testing.allocator);
            defer payloads.deinit(std.testing.allocator);

            const payload = try payloads.sliceForRow(0);
            const kv_reader = try kvbuf.KvBufReader.init(payload);
            try std.testing.expect(kv_reader.get(kvbuf.FieldIds.record_json) == null);
            const ref_value = kv_reader.get(kvbuf.FieldIds.record_json_ref) orelse return error.TestUnexpectedResult;
            try std.testing.expect(std.mem.startsWith(u8, ref_value, "sha256:"));
            const bloom = kv_reader.get(kvbuf.FieldIds.record_json_trigram_bloom) orelse return error.TestUnexpectedResult;
            try std.testing.expectEqual(@as(usize, record_json_trigram_bloom_bytes), bloom.len);
            break;
        }
        try std.testing.expect(found_items_block);
    }

    // Verify loadAll decodes the externalized JSON transparently.
    {
        var be2 = try TableAdapter.init(std.testing.allocator, root_path, "session-large-json");
        const records = try be2.backend().loadAll(std.testing.allocator);
        defer responses.backend.freeItemRecords(std.testing.allocator, records);
        be2.backend().deinit();

        try std.testing.expectEqual(@as(usize, 1), records.len);
        try std.testing.expectEqual(@as(u64, 22), records[0].item_id);
        try std.testing.expectEqual(@as(i64, 777), records[0].created_at_ms);
    }
}

test "TableAdapter externalized item record_json can use multipart refs" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    const long_text = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    {
        var be = try TableAdapter.init(std.testing.allocator, root_path, "session-multi-json");
        defer be.backend().deinit();
        be.item_json_externalize_threshold_bytes = 16;
        be.blob_store.multipart_chunk_size_bytes = 32;

        var content = [_]responses.backend.ItemContentPartRecord{
            .{ .input_text = .{ .text = long_text } },
        };
        const record = ItemRecord{
            .item_id = 33,
            .created_at_ms = 888,
            .item_type = .message,
            .variant = .{ .message = .{ .role = .user, .status = .completed, .content = &content } },
        };
        try be.backend().onEvent(&StorageEvent{ .PutItem = record });
        try be.table.fs_writer.?.flushBlock();
    }

    {
        var ro = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
        defer ro.deinitReadOnly();

        const blocks = try ro.table.fs_reader.getBlocks(std.testing.allocator);
        defer std.testing.allocator.free(blocks);

        var found_items_block = false;
        for (blocks) |block| {
            var file = ro.table.fs_reader.dir.openFile(block.path, .{ .mode = .read_only }) catch continue;
            defer file.close();

            const reader = block_reader.BlockReader.init(file, std.testing.allocator);
            const header = reader.readHeader(block.offset) catch continue;
            if (header.schema_id != schema_items or header.row_count == 0) continue;
            found_items_block = true;

            const descs = try reader.readColumnDirectory(header, block.offset);
            defer std.testing.allocator.free(descs);
            const payload_desc = findColumn(descs, col_payload) orelse return error.TestUnexpectedResult;

            var payloads = try readVarBytesBuffers(file, block.offset, payload_desc, header.row_count, std.testing.allocator);
            defer payloads.deinit(std.testing.allocator);

            const payload = try payloads.sliceForRow(0);
            const kv_reader = try kvbuf.KvBufReader.init(payload);
            const ref_value = kv_reader.get(kvbuf.FieldIds.record_json_ref) orelse return error.TestUnexpectedResult;
            try std.testing.expect(std.mem.startsWith(u8, ref_value, db_blob_store.multipart_ref_prefix));
            break;
        }
        try std.testing.expect(found_items_block);
    }

    {
        var be2 = try TableAdapter.init(std.testing.allocator, root_path, "session-multi-json");
        const records = try be2.backend().loadAll(std.testing.allocator);
        defer responses.backend.freeItemRecords(std.testing.allocator, records);
        be2.backend().deinit();

        try std.testing.expectEqual(@as(usize, 1), records.len);
        try std.testing.expectEqual(@as(u64, 33), records[0].item_id);
        try std.testing.expectEqual(@as(i64, 888), records[0].created_at_ms);
    }
}

test "TableAdapter.scanSessions returns written sessions" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    // Write a session
    {
        var be = try TableAdapter.init(std.testing.allocator, root_path, "scan-test-1");
        defer be.backend().deinit();

        const session = SessionRecord{
            .session_id = "scan-test-1",
            .model = "test-model",
            .title = "Test Session",
            .system_prompt = null,
            .config_json = null,
            .marker = "active",
            .parent_session_id = null,
            .group_id = null,
            .head_item_id = 1,
            .ttl_ts = 0,
            .metadata_json = null,
            .created_at_ms = 100,
            .updated_at_ms = 200,
        };

        const event = StorageEvent{ .PutSession = session };
        try be.backend().onEvent(&event);
        try be.table.fs_writer.?.flushBlock();
    }

    // Read it back with scanSessions
    var adapter = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
    defer adapter.deinitReadOnly();

    const records = try adapter.scanSessions(std.testing.allocator, null);
    defer freeScannedSessionRecords(std.testing.allocator, records);

    try std.testing.expectEqual(@as(usize, 1), records.len);
    try std.testing.expectEqualStrings("scan-test-1", records[0].session_id);
    try std.testing.expectEqualStrings("test-model", records[0].model.?);
    try std.testing.expectEqualStrings("Test Session", records[0].title.?);
}

test "TableAdapter.scanSessions filters deleted sessions" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    // Write session 1
    {
        var be = try TableAdapter.init(std.testing.allocator, root_path, "session-del-1");
        defer be.backend().deinit();

        const session1 = SessionRecord{
            .session_id = "session-del-1",
            .model = null,
            .title = "Session 1",
            .system_prompt = null,
            .config_json = null,
            .marker = null,
            .parent_session_id = null,
            .group_id = null,
            .head_item_id = 0,
            .ttl_ts = 0,
            .metadata_json = null,
            .created_at_ms = 100,
            .updated_at_ms = 100,
        };
        try be.backend().onEvent(&StorageEvent{ .PutSession = session1 });
        try be.table.fs_writer.?.flushBlock();
    }

    // Write session 2 with a different adapter
    {
        var be = try TableAdapter.init(std.testing.allocator, root_path, "session-del-2");
        defer be.backend().deinit();

        const session2 = SessionRecord{
            .session_id = "session-del-2",
            .model = null,
            .title = "Session 2",
            .system_prompt = null,
            .config_json = null,
            .marker = null,
            .parent_session_id = null,
            .group_id = null,
            .head_item_id = 0,
            .ttl_ts = 0,
            .metadata_json = null,
            .created_at_ms = 200,
            .updated_at_ms = 200,
        };
        try be.backend().onEvent(&StorageEvent{ .PutSession = session2 });
        try be.table.fs_writer.?.flushBlock();
    }

    // Delete session 1
    {
        var be = try TableAdapter.init(std.testing.allocator, root_path, "session-del-1");
        defer be.backend().deinit();

        const deleted = SessionRecord{
            .session_id = "session-del-1",
            .model = null,
            .title = null,
            .system_prompt = null,
            .config_json = null,
            .marker = "deleted",
            .parent_session_id = null,
            .group_id = null,
            .head_item_id = 0,
            .ttl_ts = 0,
            .metadata_json = null,
            .created_at_ms = 100,
            .updated_at_ms = 300,
        };
        try be.backend().onEvent(&StorageEvent{ .PutSession = deleted });
        try be.table.fs_writer.?.flushBlock();
    }

    // Scan should only return session 2
    var adapter = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
    defer adapter.deinitReadOnly();

    const records = try adapter.scanSessions(std.testing.allocator, null);
    defer freeScannedSessionRecords(std.testing.allocator, records);

    try std.testing.expectEqual(@as(usize, 1), records.len);
    try std.testing.expectEqualStrings("session-del-2", records[0].session_id);
}

test "TableAdapter.scanSessions with target_hash returns single session" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    // Write session 1
    {
        var be = try TableAdapter.init(std.testing.allocator, root_path, "target-1");
        defer be.backend().deinit();

        const session1 = SessionRecord{
            .session_id = "target-1",
            .model = null,
            .title = "Target 1",
            .system_prompt = null,
            .config_json = null,
            .marker = null,
            .parent_session_id = null,
            .group_id = null,
            .head_item_id = 0,
            .ttl_ts = 0,
            .metadata_json = null,
            .created_at_ms = 100,
            .updated_at_ms = 100,
        };
        try be.backend().onEvent(&StorageEvent{ .PutSession = session1 });
        try be.table.fs_writer.?.flushBlock();
    }

    // Write session 2
    {
        var be = try TableAdapter.init(std.testing.allocator, root_path, "target-2");
        defer be.backend().deinit();

        const session2 = SessionRecord{
            .session_id = "target-2",
            .model = null,
            .title = "Target 2",
            .system_prompt = null,
            .config_json = null,
            .marker = null,
            .parent_session_id = null,
            .group_id = null,
            .head_item_id = 0,
            .ttl_ts = 0,
            .metadata_json = null,
            .created_at_ms = 200,
            .updated_at_ms = 200,
        };
        try be.backend().onEvent(&StorageEvent{ .PutSession = session2 });
        try be.table.fs_writer.?.flushBlock();
    }

    // Scan with target hash for target-1
    var adapter = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
    defer adapter.deinitReadOnly();

    const target_hash = computeSessionHash("target-1");
    const records = try adapter.scanSessions(std.testing.allocator, target_hash);
    defer freeScannedSessionRecords(std.testing.allocator, records);

    try std.testing.expectEqual(@as(usize, 1), records.len);
    try std.testing.expectEqualStrings("target-1", records[0].session_id);
}

test "TableAdapter.updateSession merges non-null fields" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    // Write initial session
    {
        var be = try TableAdapter.init(std.testing.allocator, root_path, "update-test");
        defer be.backend().deinit();

        const session = SessionRecord{
            .session_id = "update-test",
            .model = "model-abc",
            .title = "Original Title",
            .system_prompt = "Be helpful",
            .config_json = "{\"temp\":0.5}",
            .marker = "active",
            .parent_session_id = null,
            .group_id = "grp-1",
            .head_item_id = 5,
            .ttl_ts = 0,
            .metadata_json = "{\"key\":\"val\"}",
            .created_at_ms = 100,
            .updated_at_ms = 200,
        };
        try be.backend().onEvent(&StorageEvent{ .PutSession = session });
        try be.table.fs_writer.?.flushBlock();
    }

    // Update only title and metadata, leave marker unchanged
    {
        var be = try TableAdapter.init(std.testing.allocator, root_path, "update-test");
        defer be.backend().deinit();

        try be.updateSession(
            std.testing.allocator,
            "update-test",
            "New Title", // change title
            null, // marker unchanged
            "{\"starred\":true}", // replace metadata
        );
    }

    // Read back and verify
    var adapter = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
    defer adapter.deinitReadOnly();

    const record = try adapter.lookupSession(std.testing.allocator, "update-test");
    defer {
        var mutable = record;
        freeScannedSessionRecord(std.testing.allocator, &mutable);
    }

    // Changed fields
    try std.testing.expectEqualStrings("New Title", record.title.?);
    try std.testing.expectEqualStrings("{\"starred\":true}", record.metadata_json.?);
    // Preserved fields
    try std.testing.expectEqualStrings("active", record.marker.?);
    try std.testing.expectEqualStrings("model-abc", record.model.?);
    try std.testing.expectEqualStrings("Be helpful", record.system_prompt.?);
    try std.testing.expectEqualStrings("{\"temp\":0.5}", record.config_json.?);
    try std.testing.expectEqualStrings("grp-1", record.group_id.?);
    try std.testing.expectEqual(@as(u64, 5), record.head_item_id);
    try std.testing.expectEqual(@as(i64, 100), record.created_at_ms);
    // updated_at_ms should have changed (newer than original 200)
    try std.testing.expect(record.updated_at_ms > 200);
}

test "TableAdapter.updateSession returns error for missing session" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var be = try TableAdapter.init(std.testing.allocator, root_path, "nonexistent");
    defer be.backend().deinit();

    const result = be.updateSession(
        std.testing.allocator,
        "nonexistent",
        "New Title",
        null,
        null,
    );
    try std.testing.expectError(error.SessionNotFound, result);
}

test "TableAdapter.deleteSession writes tombstones" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    // Write a session
    {
        var be = try TableAdapter.init(std.testing.allocator, root_path, "delete-test");
        defer be.backend().deinit();

        const session = SessionRecord{
            .session_id = "delete-test",
            .model = null,
            .title = "To Delete",
            .system_prompt = null,
            .config_json = null,
            .marker = null,
            .parent_session_id = null,
            .group_id = null,
            .head_item_id = 0,
            .ttl_ts = 0,
            .metadata_json = null,
            .created_at_ms = 100,
            .updated_at_ms = 100,
        };
        try be.backend().onEvent(&StorageEvent{ .PutSession = session });
        try be.table.fs_writer.?.flushBlock();
    }

    // Delete it
    {
        var be = try TableAdapter.init(std.testing.allocator, root_path, "delete-test");
        defer be.backend().deinit();
        try be.deleteSession("delete-test");
    }

    // Verify it's no longer in scan results
    var adapter = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
    defer adapter.deinitReadOnly();

    const records = try adapter.scanSessions(std.testing.allocator, null);
    defer freeScannedSessionRecords(std.testing.allocator, records);

    try std.testing.expectEqual(@as(usize, 0), records.len);
}

test "forkSession clones items and sets parent" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    // Write source session with 3 items
    {
        var be = try TableAdapter.init(std.testing.allocator, root_path, "source-fork");
        const storage = be.backend();

        const session = SessionRecord{
            .session_id = "source-fork",
            .model = "test-model",
            .title = "Source Session",
            .system_prompt = "Be helpful",
            .config_json = null,
            .marker = "active",
            .parent_session_id = null,
            .group_id = "grp-fork",
            .head_item_id = 3,
            .ttl_ts = 0,
            .metadata_json = null,
            .created_at_ms = 100,
            .updated_at_ms = 200,
        };
        try storage.onEvent(&StorageEvent{ .PutSession = session });

        for ([_]u64{ 1, 2, 3 }) |item_id| {
            const item = ItemRecord{
                .item_id = item_id,
                .created_at_ms = @as(i64, @intCast(item_id)) * 100,
                .item_type = .message,
                .variant = .{ .message = .{ .role = .user, .status = .completed, .content = &.{} } },
            };
            try storage.onEvent(&StorageEvent{ .PutItem = item });
        }
        try be.table.fs_writer.?.flushBlock();
        storage.deinit();
    }

    // Fork at item 2 (should include items 1 and 2, not 3)
    try forkSession(std.testing.allocator, root_path, "source-fork", 2, "forked-session");

    // Verify the forked session exists and has correct metadata
    var adapter = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
    defer adapter.deinitReadOnly();

    const forked = try adapter.lookupSession(std.testing.allocator, "forked-session");
    defer {
        var mutable = forked;
        freeScannedSessionRecord(std.testing.allocator, &mutable);
    }

    try std.testing.expectEqualStrings("source-fork", forked.parent_session_id.?);
    try std.testing.expectEqualStrings("test-model", forked.model.?);
    try std.testing.expectEqualStrings("active", forked.marker.?);
    try std.testing.expectEqualStrings("grp-fork", forked.group_id.?);

    // Verify forked conversation has 2 items (not 3)
    const conv = try loadConversation(std.testing.allocator, root_path, "forked-session");
    defer conv.deinit();

    try std.testing.expectEqual(@as(usize, 2), conv.items_list.items.len);
    try std.testing.expectEqual(@as(u64, 1), conv.items_list.items[0].id);
    try std.testing.expectEqual(@as(u64, 2), conv.items_list.items[1].id);
}

test "forkSession returns ItemNotFound for invalid item_id" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    // Write source session with 1 item
    {
        var be = try TableAdapter.init(std.testing.allocator, root_path, "source-nf");
        const storage = be.backend();

        const session = SessionRecord{
            .session_id = "source-nf",
            .model = null,
            .title = null,
            .system_prompt = null,
            .config_json = null,
            .marker = null,
            .parent_session_id = null,
            .group_id = null,
            .head_item_id = 1,
            .ttl_ts = 0,
            .metadata_json = null,
            .created_at_ms = 100,
            .updated_at_ms = 100,
        };
        try storage.onEvent(&StorageEvent{ .PutSession = session });

        const item = ItemRecord{
            .item_id = 1,
            .created_at_ms = 100,
            .item_type = .message,
            .variant = .{ .message = .{ .role = .user, .status = .completed, .content = &.{} } },
        };
        try storage.onEvent(&StorageEvent{ .PutItem = item });
        try be.table.fs_writer.?.flushBlock();
        storage.deinit();
    }

    const result = forkSession(std.testing.allocator, root_path, "source-nf", 999, "forked-nf");
    try std.testing.expectError(error.ItemNotFound, result);
}

/// Helper: write N sessions with given IDs/timestamps into a single DB.
fn writeTestSessions(root_path: []const u8, sessions: []const struct {
    id: []const u8,
    group_id: ?[]const u8,
    updated_at_ms: i64,
}) !void {
    for (sessions) |s| {
        var be = try TableAdapter.init(std.testing.allocator, root_path, s.id);
        const storage = be.backend();
        const record = SessionRecord{
            .session_id = s.id,
            .model = null,
            .title = null,
            .system_prompt = null,
            .config_json = null,
            .marker = "active",
            .parent_session_id = null,
            .group_id = s.group_id,
            .head_item_id = 0,
            .ttl_ts = 0,
            .metadata_json = null,
            .created_at_ms = s.updated_at_ms,
            .updated_at_ms = s.updated_at_ms,
        };
        try storage.onEvent(&StorageEvent{ .PutSession = record });
        try be.table.fs_writer.?.flushBlock();
        storage.deinit();
    }
}

test "scanSessionsFiltered respects limit" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    try writeTestSessions(root_path, &.{
        .{ .id = "s1", .group_id = null, .updated_at_ms = 100 },
        .{ .id = "s2", .group_id = null, .updated_at_ms = 200 },
        .{ .id = "s3", .group_id = null, .updated_at_ms = 300 },
    });

    var adapter = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
    defer adapter.deinitReadOnly();

    // Limit 2
    const records = try adapter.scanSessionsFiltered(std.testing.allocator, .{ .limit = 2 });
    defer freeScannedSessionRecords(std.testing.allocator, records);

    try std.testing.expectEqual(@as(usize, 2), records.len);
}

test "scanSessionsFiltered filters by group_id" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    try writeTestSessions(root_path, &.{
        .{ .id = "g1-s1", .group_id = "tenant-a", .updated_at_ms = 100 },
        .{ .id = "g1-s2", .group_id = "tenant-a", .updated_at_ms = 200 },
        .{ .id = "g2-s1", .group_id = "tenant-b", .updated_at_ms = 300 },
    });

    var adapter = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
    defer adapter.deinitReadOnly();

    const group_hash = computeGroupHash("tenant-a");
    const records = try adapter.scanSessionsFiltered(std.testing.allocator, .{
        .target_group_hash = group_hash,
        .target_group_id = "tenant-a",
    });
    defer freeScannedSessionRecords(std.testing.allocator, records);

    try std.testing.expectEqual(@as(usize, 2), records.len);
    // Both should be tenant-a
    for (records) |r| {
        try std.testing.expectEqualStrings("tenant-a", r.group_id.?);
    }
}

test "scanSessionsFiltered composite cursor skips newer" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    try writeTestSessions(root_path, &.{
        .{ .id = "c-s1", .group_id = null, .updated_at_ms = 100 },
        .{ .id = "c-s2", .group_id = null, .updated_at_ms = 200 },
        .{ .id = "c-s3", .group_id = null, .updated_at_ms = 300 },
    });

    var adapter = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
    defer adapter.deinitReadOnly();

    // Cursor at ts=200 with hash of c-s2 → should only return c-s1
    const cursor_hash = computeSessionHash("c-s2");
    const records = try adapter.scanSessionsFiltered(std.testing.allocator, .{
        .before_ts = 200,
        .before_session_hash = cursor_hash,
    });
    defer freeScannedSessionRecords(std.testing.allocator, records);

    try std.testing.expectEqual(@as(usize, 1), records.len);
    try std.testing.expectEqualStrings("c-s1", records[0].session_id);
}

test "scanSessionsFiltered returns all without params" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    try writeTestSessions(root_path, &.{
        .{ .id = "a-s1", .group_id = null, .updated_at_ms = 100 },
        .{ .id = "a-s2", .group_id = null, .updated_at_ms = 200 },
    });

    var adapter = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
    defer adapter.deinitReadOnly();

    // No filters → return all
    const records = try adapter.scanSessionsFiltered(std.testing.allocator, .{});
    defer freeScannedSessionRecords(std.testing.allocator, records);

    try std.testing.expectEqual(@as(usize, 2), records.len);
}

test "scanSessionsFiltered row-reverse produces descending updated_at" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    // Write sessions with ascending timestamps — in a single adapter
    // (same session_id prefix to ensure all in one block)
    try writeTestSessions(root_path, &.{
        .{ .id = "r-oldest", .group_id = null, .updated_at_ms = 100 },
        .{ .id = "r-middle", .group_id = null, .updated_at_ms = 200 },
        .{ .id = "r-newest", .group_id = null, .updated_at_ms = 300 },
    });

    var adapter = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
    defer adapter.deinitReadOnly();

    const records = try adapter.scanSessionsFiltered(std.testing.allocator, .{});
    defer freeScannedSessionRecords(std.testing.allocator, records);

    try std.testing.expectEqual(@as(usize, 3), records.len);
    // With row-reverse, newest should come first
    try std.testing.expect(records[0].updated_at_ms >= records[1].updated_at_ms);
    try std.testing.expect(records[1].updated_at_ms >= records[2].updated_at_ms);
}

test "scanSessionsFiltered group + limit combined" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    try writeTestSessions(root_path, &.{
        .{ .id = "gl-1", .group_id = "grp-x", .updated_at_ms = 100 },
        .{ .id = "gl-2", .group_id = "grp-x", .updated_at_ms = 200 },
        .{ .id = "gl-3", .group_id = "grp-x", .updated_at_ms = 300 },
        .{ .id = "gl-4", .group_id = "grp-y", .updated_at_ms = 400 },
    });

    var adapter = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
    defer adapter.deinitReadOnly();

    const group_hash = computeGroupHash("grp-x");
    const records = try adapter.scanSessionsFiltered(std.testing.allocator, .{
        .target_group_hash = group_hash,
        .target_group_id = "grp-x",
        .limit = 2,
    });
    defer freeScannedSessionRecords(std.testing.allocator, records);

    try std.testing.expectEqual(@as(usize, 2), records.len);
    for (records) |r| {
        try std.testing.expectEqualStrings("grp-x", r.group_id.?);
    }
}

test "ScanParams.fromArgs sets cursor and group fields" {
    // No cursor, no group
    const p1 = TableAdapter.ScanParams.fromArgs(10, 0, null, null);
    try std.testing.expectEqual(@as(u32, 10), p1.limit);
    try std.testing.expect(p1.before_ts == null);
    try std.testing.expect(p1.before_session_hash == null);
    try std.testing.expect(p1.target_group_hash == null);
    try std.testing.expect(p1.target_group_id == null);

    // Cursor with session ID
    const p2 = TableAdapter.ScanParams.fromArgs(5, 12345, "sess-1", null);
    try std.testing.expectEqual(@as(u32, 5), p2.limit);
    try std.testing.expectEqual(@as(i64, 12345), p2.before_ts.?);
    try std.testing.expectEqual(computeSessionHash("sess-1"), p2.before_session_hash.?);
    try std.testing.expect(p2.target_group_hash == null);

    // Cursor without session ID
    const p3 = TableAdapter.ScanParams.fromArgs(0, 999, null, null);
    try std.testing.expectEqual(@as(i64, 999), p3.before_ts.?);
    try std.testing.expect(p3.before_session_hash == null);

    // Group filter
    const p4 = TableAdapter.ScanParams.fromArgs(0, 0, null, "grp-a");
    try std.testing.expectEqual(computeGroupHash("grp-a"), p4.target_group_hash.?);
    try std.testing.expectEqualStrings("grp-a", p4.target_group_id.?);
}

// ---------------------------------------------------------------------------
// scanSessionsFiltered search_query integration tests
// ---------------------------------------------------------------------------

fn writeTestSessionsWithMetadata(root_path: []const u8, sessions: []const struct {
    id: []const u8,
    title: ?[]const u8,
    model: ?[]const u8,
    system_prompt: ?[]const u8,
    group_id: ?[]const u8,
    updated_at_ms: i64,
}) !void {
    for (sessions) |s| {
        var be = try TableAdapter.init(std.testing.allocator, root_path, s.id);
        const storage = be.backend();
        const record = SessionRecord{
            .session_id = s.id,
            .model = s.model,
            .title = s.title,
            .system_prompt = s.system_prompt,
            .config_json = null,
            .marker = "active",
            .parent_session_id = null,
            .group_id = s.group_id,
            .head_item_id = 0,
            .ttl_ts = 0,
            .metadata_json = null,
            .created_at_ms = s.updated_at_ms,
            .updated_at_ms = s.updated_at_ms,
        };
        try storage.onEvent(&StorageEvent{ .PutSession = record });
        try be.table.fs_writer.?.flushBlock();
        storage.deinit();
    }
}

test "scanSessionsFiltered search_query filters by title" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    try writeTestSessionsWithMetadata(root_path, &.{
        .{ .id = "s1", .title = "Building a Rust CLI", .model = "gpt-4", .system_prompt = null, .group_id = null, .updated_at_ms = 100 },
        .{ .id = "s2", .title = "Python data analysis", .model = "claude-3", .system_prompt = null, .group_id = null, .updated_at_ms = 200 },
        .{ .id = "s3", .title = "TypeScript React app", .model = "gpt-4", .system_prompt = null, .group_id = null, .updated_at_ms = 300 },
    });

    var adapter = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
    defer adapter.deinitReadOnly();

    const records = try adapter.scanSessionsFiltered(std.testing.allocator, .{ .search_query = "rust" });
    defer freeScannedSessionRecords(std.testing.allocator, records);

    try std.testing.expectEqual(@as(usize, 1), records.len);
    try std.testing.expectEqualStrings("s1", records[0].session_id);
}

test "scanSessionsFiltered search_query matches model" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    try writeTestSessionsWithMetadata(root_path, &.{
        .{ .id = "s1", .title = "Chat 1", .model = "gpt-4", .system_prompt = null, .group_id = null, .updated_at_ms = 100 },
        .{ .id = "s2", .title = "Chat 2", .model = "claude-3", .system_prompt = null, .group_id = null, .updated_at_ms = 200 },
    });

    var adapter = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
    defer adapter.deinitReadOnly();

    const records = try adapter.scanSessionsFiltered(std.testing.allocator, .{ .search_query = "claude" });
    defer freeScannedSessionRecords(std.testing.allocator, records);

    try std.testing.expectEqual(@as(usize, 1), records.len);
    try std.testing.expectEqualStrings("s2", records[0].session_id);
}

test "scanSessionsFiltered search_query matches system_prompt" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    try writeTestSessionsWithMetadata(root_path, &.{
        .{ .id = "s1", .title = "Chat", .model = "m", .system_prompt = "You are a helpful coding assistant", .group_id = null, .updated_at_ms = 100 },
        .{ .id = "s2", .title = "Chat", .model = "m", .system_prompt = "You are a math tutor", .group_id = null, .updated_at_ms = 200 },
    });

    var adapter = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
    defer adapter.deinitReadOnly();

    const records = try adapter.scanSessionsFiltered(std.testing.allocator, .{ .search_query = "coding" });
    defer freeScannedSessionRecords(std.testing.allocator, records);

    try std.testing.expectEqual(@as(usize, 1), records.len);
    try std.testing.expectEqualStrings("s1", records[0].session_id);
}

test "scanSessionsFiltered search_query no match returns empty" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    try writeTestSessionsWithMetadata(root_path, &.{
        .{ .id = "s1", .title = "Building a Rust CLI", .model = "gpt-4", .system_prompt = null, .group_id = null, .updated_at_ms = 100 },
    });

    var adapter = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
    defer adapter.deinitReadOnly();

    const records = try adapter.scanSessionsFiltered(std.testing.allocator, .{ .search_query = "nonexistent" });
    defer freeScannedSessionRecords(std.testing.allocator, records);

    try std.testing.expectEqual(@as(usize, 0), records.len);
}

test "scanSessionsFiltered search_query null returns all" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    try writeTestSessionsWithMetadata(root_path, &.{
        .{ .id = "s1", .title = "A", .model = "m", .system_prompt = null, .group_id = null, .updated_at_ms = 100 },
        .{ .id = "s2", .title = "B", .model = "m", .system_prompt = null, .group_id = null, .updated_at_ms = 200 },
    });

    var adapter = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
    defer adapter.deinitReadOnly();

    // search_query = null (default) → no filtering
    const records = try adapter.scanSessionsFiltered(std.testing.allocator, .{});
    defer freeScannedSessionRecords(std.testing.allocator, records);

    try std.testing.expectEqual(@as(usize, 2), records.len);
}

test "scanSessionsFiltered search_query with limit" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    try writeTestSessionsWithMetadata(root_path, &.{
        .{ .id = "s1", .title = "Rust project 1", .model = "m", .system_prompt = null, .group_id = null, .updated_at_ms = 100 },
        .{ .id = "s2", .title = "Rust project 2", .model = "m", .system_prompt = null, .group_id = null, .updated_at_ms = 200 },
        .{ .id = "s3", .title = "Rust project 3", .model = "m", .system_prompt = null, .group_id = null, .updated_at_ms = 300 },
    });

    var adapter = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
    defer adapter.deinitReadOnly();

    // All 3 match "rust" but limit = 2
    const records = try adapter.scanSessionsFiltered(std.testing.allocator, .{
        .search_query = "rust",
        .limit = 2,
    });
    defer freeScannedSessionRecords(std.testing.allocator, records);

    try std.testing.expectEqual(@as(usize, 2), records.len);
}

test "scanSessionsFiltered max_scan budget stops early" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    // Write 5 sessions, none matching the search query.
    try writeTestSessionsWithMetadata(root_path, &.{
        .{ .id = "s1", .title = "A", .model = "m", .system_prompt = null, .group_id = null, .updated_at_ms = 100 },
        .{ .id = "s2", .title = "B", .model = "m", .system_prompt = null, .group_id = null, .updated_at_ms = 200 },
        .{ .id = "s3", .title = "C", .model = "m", .system_prompt = null, .group_id = null, .updated_at_ms = 300 },
        .{ .id = "s4", .title = "D", .model = "m", .system_prompt = null, .group_id = null, .updated_at_ms = 400 },
        .{ .id = "s5", .title = "E", .model = "m", .system_prompt = null, .group_id = null, .updated_at_ms = 500 },
    });

    var adapter = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
    defer adapter.deinitReadOnly();

    // max_scan = 3: should stop after decoding 3 rows, returning 0 matches
    // (without budget, it would decode all 5)
    const records = try adapter.scanSessionsFiltered(std.testing.allocator, .{
        .search_query = "nonexistent",
        .max_scan = 3,
    });
    defer freeScannedSessionRecords(std.testing.allocator, records);

    try std.testing.expectEqual(@as(usize, 0), records.len);
}

// ---------------------------------------------------------------------------
// Content search tests
// ---------------------------------------------------------------------------

/// Helper: write a session with a single message item containing the given text.
fn writeTestSessionWithMessage(root_path: []const u8, session_id: []const u8, title: []const u8, model: []const u8, message_text: []const u8, updated_at_ms: i64) !void {
    var be = try TableAdapter.init(std.testing.allocator, root_path, session_id);
    const storage = be.backend();

    const session = SessionRecord{
        .session_id = session_id,
        .model = model,
        .title = title,
        .system_prompt = null,
        .config_json = null,
        .marker = "active",
        .parent_session_id = null,
        .group_id = null,
        .head_item_id = 1,
        .ttl_ts = 0,
        .metadata_json = null,
        .created_at_ms = updated_at_ms,
        .updated_at_ms = updated_at_ms,
    };
    try storage.onEvent(&StorageEvent{ .PutSession = session });

    var content_parts = [_]responses.backend.ItemContentPartRecord{
        .{ .output_text = .{ .text = message_text } },
    };
    const item = ItemRecord{
        .item_id = 1,
        .created_at_ms = updated_at_ms,
        .item_type = .message,
        .variant = .{ .message = .{
            .role = .assistant,
            .status = .completed,
            .content = &content_parts,
        } },
    };
    try storage.onEvent(&StorageEvent{ .PutItem = item });
    try be.table.fs_writer.?.flushBlock();
    storage.deinit();
}

test "scanSessionsFiltered search_query matches item content" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    // Session with generic title but specific content
    try writeTestSessionWithMessage(root_path, "s1", "Chat", "m", "The quantum physics explanation is fascinating", 100);
    // Session with no matching content
    try writeTestSessionWithMessage(root_path, "s2", "Chat", "m", "Let me explain the weather forecast", 200);

    var adapter = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
    defer adapter.deinitReadOnly();

    // Search for content text that doesn't appear in title/model/system_prompt
    const records = try adapter.scanSessionsFiltered(std.testing.allocator, .{ .search_query = "quantum" });
    defer freeScannedSessionRecords(std.testing.allocator, records);

    try std.testing.expectEqual(@as(usize, 1), records.len);
    try std.testing.expectEqualStrings("s1", records[0].session_id);
}

test "scanSessionsFiltered search_query matches content or title" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    // Session matching by title only (content is unrelated)
    try writeTestSessionWithMessage(root_path, "s1", "Rust programming", "m", "Hello world", 100);
    // Session matching by content only (title is generic)
    try writeTestSessionWithMessage(root_path, "s2", "Chat", "m", "Let me explain Rust ownership semantics", 200);
    // Session matching neither
    try writeTestSessionWithMessage(root_path, "s3", "Chat", "m", "The weather is nice today", 300);

    var adapter = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
    defer adapter.deinitReadOnly();

    const records = try adapter.scanSessionsFiltered(std.testing.allocator, .{ .search_query = "rust" });
    defer freeScannedSessionRecords(std.testing.allocator, records);

    // Both s1 (title match) and s2 (content match) should be returned
    try std.testing.expectEqual(@as(usize, 2), records.len);
}

test "scanSessionsFiltered search_query content no match returns empty" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    try writeTestSessionWithMessage(root_path, "s1", "Chat", "m", "Hello world", 100);

    var adapter = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
    defer adapter.deinitReadOnly();

    const records = try adapter.scanSessionsFiltered(std.testing.allocator, .{ .search_query = "nonexistent" });
    defer freeScannedSessionRecords(std.testing.allocator, records);

    try std.testing.expectEqual(@as(usize, 0), records.len);
}

// ---------------------------------------------------------------------------
// search_snippet integration tests
// ---------------------------------------------------------------------------

test "scanSessionsFiltered content match populates search_snippet" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    try writeTestSessionWithMessage(root_path, "s1", "Chat", "m", "The quantum physics explanation is fascinating", 100);

    var adapter = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
    defer adapter.deinitReadOnly();

    const records = try adapter.scanSessionsFiltered(std.testing.allocator, .{ .search_query = "quantum" });
    defer freeScannedSessionRecords(std.testing.allocator, records);

    try std.testing.expectEqual(@as(usize, 1), records.len);
    // Content match should populate search_snippet
    try std.testing.expect(records[0].search_snippet != null);
    const snippet = records[0].search_snippet.?;
    try std.testing.expect(textContainsInsensitive(snippet, "quantum"));
    // Snippet should be clean text, not raw JSON
    try std.testing.expect(std.mem.indexOf(u8, snippet, "{") == null);
    try std.testing.expect(std.mem.indexOf(u8, snippet, "\"type\"") == null);
}

test "scanSessionsFiltered title match has null search_snippet" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    // Title matches "rust", content is unrelated
    try writeTestSessionWithMessage(root_path, "s1", "Rust programming", "m", "Hello world", 100);

    var adapter = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
    defer adapter.deinitReadOnly();

    const records = try adapter.scanSessionsFiltered(std.testing.allocator, .{ .search_query = "rust" });
    defer freeScannedSessionRecords(std.testing.allocator, records);

    try std.testing.expectEqual(@as(usize, 1), records.len);
    // Title-only match should NOT populate search_snippet
    try std.testing.expectEqual(@as(?[]const u8, null), records[0].search_snippet);
}

test "isSessionSchema helper" {
    try std.testing.expect(isSessionSchema(schema_sessions));
    try std.testing.expect(isSessionSchema(schema_sessions_kvbuf));
    try std.testing.expect(!isSessionSchema(schema_items));
    try std.testing.expect(!isSessionSchema(schema_deletes));
    try std.testing.expect(!isSessionSchema(schema_embeddings));
}
