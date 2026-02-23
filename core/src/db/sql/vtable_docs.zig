//! SQLite virtual table over TaluDB documents.
//!
//! Phase 2/3 status:
//! - registers as `taludb_docs`
//! - read-only table lifecycle (`xCreate/xConnect/xOpen/xClose/xDisconnect`)
//! - sequential scan over TaluDB document blocks (`xFilter/xNext/xEof/xColumn`)
//! - tombstone filtering and latest-version dedupe by `doc_hash`

const std = @import("std");
const block_reader = @import("../block_reader.zig");
const db_reader = @import("../reader.zig");
const db_blob_store = @import("../blob/store.zig");
const kvbuf = @import("../../io/kvbuf/root.zig");
const sqlite = @import("c.zig").sqlite3;
const types = @import("../types.zig");
const documents = @import("../table/documents.zig");

const allocator = std.heap.c_allocator;
const DocumentFieldIds = kvbuf.DocumentFieldIds;

const module_name: [:0]const u8 = "taludb_docs";
const schema_sql: [:0]const u8 =
    \\CREATE TABLE x(
    \\  doc_id TEXT,
    \\  doc_type TEXT,
    \\  group_id TEXT,
    \\  created_at_ms INTEGER,
    \\  payload TEXT
    \\)
;

const schema_documents: u16 = 11;
const schema_document_deletes: u16 = 12;
const col_doc_hash: u32 = 1;
const col_ts: u32 = 2;
const col_group_hash: u32 = 3;
const col_type_hash: u32 = 4;
const col_expires_at: u32 = 9;
const col_payload: u32 = 20;

const transient_destructor = sqlite.SQLITE_TRANSIENT;
const idx_group_eq: c_int = 1 << 0;
const idx_doc_type_eq: c_int = 1 << 1;

const ModuleAux = struct {
    db_root: []u8,
};

const DocsVTab = struct {
    base: sqlite.sqlite3_vtab = std.mem.zeroes(sqlite.sqlite3_vtab),
    db_root: []u8,
};

const RowView = struct {
    doc_id: []const u8 = "",
    doc_type: []const u8 = "",
    group_id: ?[]const u8 = null,
    created_at_ms: i64 = 0,
    payload_json: ?[]const u8 = null,
};

const VarBytesBuffers = struct {
    data: []u8,
    offsets: []u32,
    lengths: []u32,

    fn deinit(self: *VarBytesBuffers, alloc: std.mem.Allocator) void {
        alloc.free(self.data);
        alloc.free(self.offsets);
        alloc.free(self.lengths);
    }

    fn sliceForRow(self: VarBytesBuffers, row_idx: usize) ![]const u8 {
        if (row_idx >= self.offsets.len or row_idx >= self.lengths.len) return error.InvalidColumnData;
        const start = @as(usize, self.offsets[row_idx]);
        const end = start + @as(usize, self.lengths[row_idx]);
        if (end > self.data.len) return error.InvalidColumnData;
        return self.data[start..end];
    }
};

const DocsCursor = struct {
    base: sqlite.sqlite3_vtab_cursor = std.mem.zeroes(sqlite.sqlite3_vtab_cursor),
    table: *DocsVTab,
    reader: db_reader.Reader,
    blob_store: db_blob_store.BlobStore,
    blocks: []db_reader.BlockRef = &.{},
    deleted: std.AutoHashMap(u64, i64),
    latest: std.AutoHashMap(u64, i64),
    seen: std.AutoHashMap(u64, void),
    block_idx: isize = -1,
    row_idx: isize = -1,
    block_hash_bytes: []u8 = &.{},
    block_ts_bytes: []u8 = &.{},
    block_group_hash_bytes: ?[]u8 = null,
    block_type_hash_bytes: ?[]u8 = null,
    block_expires_bytes: ?[]u8 = null,
    block_payload: ?VarBytesBuffers = null,
    filter_group_id: ?[]u8 = null,
    filter_group_hash: ?u64 = null,
    filter_doc_type: ?[]u8 = null,
    filter_doc_type_hash: ?u64 = null,
    current_row: RowView = .{},
    owned_payload_json: ?[]u8 = null,
    rowid: sqlite.sqlite3_int64 = 0,
    query_now_ms: i64 = 0,
    eof: bool = true,

    fn init(table: *DocsVTab) !DocsCursor {
        var reader = try db_reader.Reader.open(allocator, table.db_root, "docs");
        errdefer reader.deinit();
        var blob_store = try db_blob_store.BlobStore.init(allocator, table.db_root);
        errdefer blob_store.deinit();

        return .{
            .base = std.mem.zeroes(sqlite.sqlite3_vtab_cursor),
            .table = table,
            .reader = reader,
            .blob_store = blob_store,
            .deleted = std.AutoHashMap(u64, i64).init(allocator),
            .latest = std.AutoHashMap(u64, i64).init(allocator),
            .seen = std.AutoHashMap(u64, void).init(allocator),
        };
    }

    fn deinit(self: *DocsCursor) void {
        self.clearLoadedBlock();
        self.clearOwnedPayloadJson();
        self.clearFilters();
        if (self.blocks.len > 0) allocator.free(self.blocks);
        self.deleted.deinit();
        self.latest.deinit();
        self.seen.deinit();
        self.blob_store.deinit();
        self.reader.deinit();
    }

    fn clearLoadedBlock(self: *DocsCursor) void {
        if (self.block_hash_bytes.len > 0) allocator.free(self.block_hash_bytes);
        if (self.block_ts_bytes.len > 0) allocator.free(self.block_ts_bytes);
        if (self.block_group_hash_bytes) |bytes| allocator.free(bytes);
        if (self.block_type_hash_bytes) |bytes| allocator.free(bytes);
        if (self.block_expires_bytes) |bytes| allocator.free(bytes);
        if (self.block_payload) |*payload| payload.deinit(allocator);

        self.block_hash_bytes = &.{};
        self.block_ts_bytes = &.{};
        self.block_group_hash_bytes = null;
        self.block_type_hash_bytes = null;
        self.block_expires_bytes = null;
        self.block_payload = null;
        self.row_idx = -1;
    }

    fn clearOwnedPayloadJson(self: *DocsCursor) void {
        if (self.owned_payload_json) |payload_json| allocator.free(payload_json);
        self.owned_payload_json = null;
    }

    fn clearFilters(self: *DocsCursor) void {
        if (self.filter_group_id) |s| allocator.free(s);
        if (self.filter_doc_type) |s| allocator.free(s);
        self.filter_group_id = null;
        self.filter_group_hash = null;
        self.filter_doc_type = null;
        self.filter_doc_type_hash = null;
    }

    fn setGroupFilter(self: *DocsCursor, value: []const u8) !void {
        if (self.filter_group_id) |old| allocator.free(old);
        self.filter_group_id = try allocator.dupe(u8, value);
        self.filter_group_hash = computeHash(value);
    }

    fn setDocTypeFilter(self: *DocsCursor, value: []const u8) !void {
        if (self.filter_doc_type) |old| allocator.free(old);
        self.filter_doc_type = try allocator.dupe(u8, value);
        self.filter_doc_type_hash = computeHash(value);
    }

    fn resetForFilter(self: *DocsCursor) !void {
        self.clearLoadedBlock();
        self.block_idx = -1;
        self.row_idx = -1;
        self.rowid = 0;
        self.current_row = .{};
        self.query_now_ms = std.time.milliTimestamp();
        self.eof = false;

        const changed = try self.reader.refreshIfChanged();
        if (changed or self.blocks.len == 0) {
            if (self.blocks.len > 0) allocator.free(self.blocks);
            self.blocks = try self.reader.getBlocks(allocator);

            self.deleted.clearRetainingCapacity();
            self.latest.clearRetainingCapacity();
            try self.collectDeletedDocuments();
            try self.collectLatestDocuments();
        }

        // Per-query dedupe state always resets for a fresh cursor scan.
        self.seen.clearRetainingCapacity();
        self.block_idx = @as(isize, @intCast(self.blocks.len)) - 1;
    }

    fn collectDeletedDocuments(self: *DocsCursor) !void {
        for (self.blocks) |block| {
            var handle = self.reader.openBlockReadOnly(block.path) catch continue;
            defer self.reader.closeBlock(&handle);

            const reader = block_reader.BlockReader.init(handle.file, allocator);
            const header = reader.readHeader(block.offset) catch continue;
            if (header.schema_id != schema_document_deletes) continue;
            if (header.row_count == 0) continue;

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

                if (self.deleted.get(doc_hash)) |existing| {
                    if (ts > existing) try self.deleted.put(doc_hash, ts);
                } else {
                    try self.deleted.put(doc_hash, ts);
                }
            }
        }
    }

    fn loadPreviousDocumentBlock(self: *DocsCursor) !bool {
        self.clearLoadedBlock();

        while (self.block_idx >= 0) : (self.block_idx -= 1) {
            const block = self.blocks[@intCast(self.block_idx)];

            var handle = self.reader.openBlockReadOnly(block.path) catch continue;
            defer self.reader.closeBlock(&handle);

            const reader = block_reader.BlockReader.init(handle.file, allocator);
            const header = reader.readHeader(block.offset) catch continue;
            if (header.schema_id != schema_documents) continue;
            if (header.row_count == 0) continue;

            const descs = reader.readColumnDirectory(header, block.offset) catch continue;
            defer allocator.free(descs);

            const hash_desc = findColumn(descs, col_doc_hash) orelse continue;
            const ts_desc = findColumn(descs, col_ts) orelse continue;
            const payload_desc = findColumn(descs, col_payload) orelse continue;
            const group_desc = findColumn(descs, col_group_hash);
            const type_desc = findColumn(descs, col_type_hash);
            const expires_desc = findColumn(descs, col_expires_at);

            self.block_hash_bytes = reader.readColumnData(block.offset, hash_desc, allocator) catch {
                self.block_hash_bytes = &.{};
                continue;
            };

            self.block_ts_bytes = reader.readColumnData(block.offset, ts_desc, allocator) catch {
                allocator.free(self.block_hash_bytes);
                self.block_hash_bytes = &.{};
                self.block_ts_bytes = &.{};
                continue;
            };

            if (group_desc) |gd| {
                self.block_group_hash_bytes = reader.readColumnData(block.offset, gd, allocator) catch null;
            } else {
                self.block_group_hash_bytes = null;
            }

            if (type_desc) |td| {
                self.block_type_hash_bytes = reader.readColumnData(block.offset, td, allocator) catch null;
            } else {
                self.block_type_hash_bytes = null;
            }

            if (expires_desc) |ed| {
                self.block_expires_bytes = reader.readColumnData(block.offset, ed, allocator) catch null;
            } else {
                self.block_expires_bytes = null;
            }

            self.block_payload = readVarBytesBuffers(
                handle.file,
                block.offset,
                payload_desc,
                header.row_count,
                allocator,
            ) catch {
                allocator.free(self.block_hash_bytes);
                allocator.free(self.block_ts_bytes);
                self.block_hash_bytes = &.{};
                self.block_ts_bytes = &.{};
                if (self.block_group_hash_bytes) |bytes| {
                    allocator.free(bytes);
                    self.block_group_hash_bytes = null;
                }
                if (self.block_type_hash_bytes) |bytes| {
                    allocator.free(bytes);
                    self.block_type_hash_bytes = null;
                }
                if (self.block_expires_bytes) |bytes| {
                    allocator.free(bytes);
                    self.block_expires_bytes = null;
                }
                continue;
            };

            self.row_idx = @as(isize, @intCast(header.row_count)) - 1;
            // Advance to the next older block so the next refill does not reload
            // the same block forever.
            self.block_idx -= 1;
            return true;
        }

        self.row_idx = -1;
        return false;
    }

    fn advance(self: *DocsCursor) !void {
        self.clearOwnedPayloadJson();
        self.current_row = .{};

        while (true) {
            if (self.row_idx < 0) {
                if (!try self.loadPreviousDocumentBlock()) {
                    self.eof = true;
                    return;
                }
            }

            const row_usize: usize = @intCast(self.row_idx);
            self.row_idx -= 1;

            const payload_buf = self.block_payload orelse continue;
            const payload = payload_buf.sliceForRow(row_usize) catch continue;
            if (!kvbuf.isKvBuf(payload)) continue;

            const doc_hash = readU64At(self.block_hash_bytes, row_usize) catch continue;
            const ts = readI64At(self.block_ts_bytes, row_usize) catch continue;

            if (self.latest.get(doc_hash)) |latest_ts| {
                if (ts < latest_ts) continue;
            }

            if (self.filter_group_hash) |target_hash| {
                if (self.block_group_hash_bytes) |group_hash_bytes| {
                    const row_group_hash = readU64At(group_hash_bytes, row_usize) catch continue;
                    if (row_group_hash != target_hash) continue;
                }
            }

            if (self.filter_doc_type_hash) |target_hash| {
                if (self.block_type_hash_bytes) |type_hash_bytes| {
                    const row_type_hash = readU64At(type_hash_bytes, row_usize) catch continue;
                    if (row_type_hash != target_hash) continue;
                }
            }

            if (self.deleted.get(doc_hash)) |del_ts| {
                if (del_ts >= ts) continue;
            }

            if (self.block_expires_bytes) |expires_bytes| {
                const expires_at = readI64At(expires_bytes, row_usize) catch 0;
                if (expires_at > 0 and expires_at < self.query_now_ms) continue;
            }

            if (self.seen.contains(doc_hash)) continue;
            try self.seen.put(doc_hash, {});

            const reader = kvbuf.KvBufReader.init(payload) catch continue;
            const doc_id = reader.get(DocumentFieldIds.doc_id) orelse continue;
            const doc_type = reader.get(DocumentFieldIds.doc_type) orelse continue;
            const group_id = reader.get(DocumentFieldIds.group_id);
            const created_at_ms = reader.getI64(DocumentFieldIds.created_at_ms) orelse ts;
            var payload_json = reader.get(DocumentFieldIds.doc_json);
            if (payload_json == null) {
                if (reader.get(DocumentFieldIds.doc_json_ref)) |doc_json_ref| {
                    if (self.blob_store.readAll(doc_json_ref, allocator) catch null) |loaded_json| {
                        self.owned_payload_json = loaded_json;
                        payload_json = loaded_json;
                    }
                }
            }

            if (self.filter_group_id) |target_group| {
                const row_group = group_id orelse continue;
                if (!std.mem.eql(u8, row_group, target_group)) continue;
            }

            if (self.filter_doc_type) |target_doc_type| {
                if (!std.mem.eql(u8, doc_type, target_doc_type)) continue;
            }

            self.current_row = .{
                .doc_id = doc_id,
                .doc_type = doc_type,
                .group_id = group_id,
                .created_at_ms = created_at_ms,
                .payload_json = payload_json,
            };
            self.rowid += 1;
            self.eof = false;
            return;
        }
    }

    fn collectLatestDocuments(self: *DocsCursor) !void {
        for (self.blocks) |block| {
            var handle = self.reader.openBlockReadOnly(block.path) catch continue;
            defer self.reader.closeBlock(&handle);

            const reader = block_reader.BlockReader.init(handle.file, allocator);
            const header = reader.readHeader(block.offset) catch continue;
            if (header.schema_id != schema_documents) continue;
            if (header.row_count == 0) continue;

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
                if (self.latest.get(doc_hash)) |existing| {
                    if (ts > existing) try self.latest.put(doc_hash, ts);
                } else {
                    try self.latest.put(doc_hash, ts);
                }
            }
        }
    }
};

const docs_module: sqlite.sqlite3_module = blk: {
    var m = std.mem.zeroes(sqlite.sqlite3_module);
    m.iVersion = 2;
    m.xCreate = xCreate;
    m.xConnect = xConnect;
    m.xBestIndex = xBestIndex;
    m.xDisconnect = xDisconnect;
    m.xDestroy = xDestroy;
    m.xOpen = xOpen;
    m.xClose = xClose;
    m.xFilter = xFilter;
    m.xNext = xNext;
    m.xEof = xEof;
    m.xColumn = xColumn;
    m.xRowid = xRowid;
    m.xUpdate = xUpdate;
    m.xBegin = xBegin;
    m.xSync = xSync;
    m.xCommit = xCommit;
    m.xRollback = xRollback;
    m.xFindFunction = xFindFunction;
    m.xRename = xRename;
    m.xSavepoint = xSavepoint;
    m.xRelease = xRelease;
    m.xRollbackTo = xRollbackTo;
    break :blk m;
};

/// Register the `taludb_docs` virtual table module on an open SQLite connection.
pub fn registerDocsModule(db: *sqlite.sqlite3, db_root: []const u8) !void {
    const aux = try allocator.create(ModuleAux);
    errdefer allocator.destroy(aux);
    aux.db_root = try allocator.dupe(u8, db_root);
    errdefer allocator.free(aux.db_root);

    const rc = sqlite.sqlite3_create_module_v2(
        db,
        module_name.ptr,
        &docs_module,
        aux,
        moduleDestroy,
    );
    if (rc != sqlite.SQLITE_OK) return error.SqliteModuleRegisterFailed;
}

fn moduleDestroy(p_aux: ?*anyopaque) callconv(.c) void {
    const aux: *ModuleAux = @ptrCast(@alignCast(p_aux orelse return));
    allocator.free(aux.db_root);
    allocator.destroy(aux);
}

fn tableFromBase(base: *sqlite.sqlite3_vtab) *DocsVTab {
    return @alignCast(@fieldParentPtr("base", @as(*sqlite.sqlite3_vtab, @alignCast(base))));
}

fn cursorFromBase(base: *sqlite.sqlite3_vtab_cursor) *DocsCursor {
    return @alignCast(@fieldParentPtr("base", @as(*sqlite.sqlite3_vtab_cursor, @alignCast(base))));
}

fn setErrMsg(pz_err: [*c][*c]u8, msg: [:0]const u8) void {
    if (pz_err == null) return;
    pz_err[0] = sqlite.sqlite3_mprintf(msg.ptr);
}

fn resultText(ctx: ?*sqlite.sqlite3_context, value: []const u8) void {
    if (value.len > std.math.maxInt(c_int)) {
        sqlite.sqlite3_result_error_toobig(ctx);
        return;
    }
    sqlite.sqlite3_result_text(ctx, value.ptr, @intCast(value.len), transient_destructor);
}

fn xCreate(
    db: ?*sqlite.sqlite3,
    p_aux: ?*anyopaque,
    argc: c_int,
    argv: [*c]const [*c]const u8,
    pp_vtab: [*c][*c]sqlite.sqlite3_vtab,
    pz_err: [*c][*c]u8,
) callconv(.c) c_int {
    return xConnect(db, p_aux, argc, argv, pp_vtab, pz_err);
}

fn xConnect(
    db: ?*sqlite.sqlite3,
    p_aux: ?*anyopaque,
    _: c_int,
    _: [*c]const [*c]const u8,
    pp_vtab: [*c][*c]sqlite.sqlite3_vtab,
    pz_err: [*c][*c]u8,
) callconv(.c) c_int {
    const db_handle = db orelse return sqlite.SQLITE_MISUSE;
    if (pp_vtab == null) return sqlite.SQLITE_MISUSE;
    const aux: *ModuleAux = @ptrCast(@alignCast(p_aux orelse {
        setErrMsg(pz_err, "missing taludb_docs module aux");
        return sqlite.SQLITE_ERROR;
    }));

    const decl_rc = sqlite.sqlite3_declare_vtab(db_handle, schema_sql.ptr);
    if (decl_rc != sqlite.SQLITE_OK) {
        setErrMsg(pz_err, "failed to declare taludb_docs schema");
        return decl_rc;
    }

    const table = allocator.create(DocsVTab) catch {
        setErrMsg(pz_err, "out of memory allocating taludb_docs table");
        return sqlite.SQLITE_NOMEM;
    };
    errdefer allocator.destroy(table);

    const db_root_copy = allocator.dupe(u8, aux.db_root) catch {
        setErrMsg(pz_err, "out of memory duplicating taludb_docs db_root");
        return sqlite.SQLITE_NOMEM;
    };

    table.* = .{
        .base = std.mem.zeroes(sqlite.sqlite3_vtab),
        .db_root = db_root_copy,
    };
    pp_vtab[0] = &table.base;
    return sqlite.SQLITE_OK;
}

fn xBestIndex(_: ?*sqlite.sqlite3_vtab, p_info: ?*sqlite.sqlite3_index_info) callconv(.c) c_int {
    if (p_info) |info| {
        info.idxNum = 0;
        info.orderByConsumed = 0;
        info.estimatedCost = 1_000_000.0;
        info.estimatedRows = 1_000_000;

        const n: usize = @intCast(info.nConstraint);
        if (n > 0) {
            const constraints = info.aConstraint[0..n];
            const usages = info.aConstraintUsage[0..n];

            var doc_type_idx: ?usize = null;
            var group_idx: ?usize = null;
            for (constraints, 0..) |constraint, idx| {
                if (constraint.usable == 0) continue;
                if (constraint.op != sqlite.SQLITE_INDEX_CONSTRAINT_EQ) continue;
                switch (constraint.iColumn) {
                    1 => {
                        if (doc_type_idx == null) doc_type_idx = idx;
                    },
                    2 => {
                        if (group_idx == null) group_idx = idx;
                    },
                    else => {},
                }
            }

            var matched: usize = 0;
            if (doc_type_idx) |idx| {
                usages[idx].argvIndex = 1;
                usages[idx].omit = 1;
                info.idxNum |= idx_doc_type_eq;
                matched += 1;
            }
            if (group_idx) |idx| {
                usages[idx].argvIndex = if (doc_type_idx != null) 2 else 1;
                usages[idx].omit = 1;
                info.idxNum |= idx_group_eq;
                matched += 1;
            }

            if (matched > 0) {
                info.estimatedCost = 10_000.0;
                info.estimatedRows = 10_000;
            }
            if (matched > 1) {
                info.estimatedCost = 1_000.0;
                info.estimatedRows = 1_000;
            }
        }
    }
    return sqlite.SQLITE_OK;
}

fn xDisconnect(p_vtab: ?*sqlite.sqlite3_vtab) callconv(.c) c_int {
    const base = p_vtab orelse return sqlite.SQLITE_MISUSE;
    const table = tableFromBase(base);
    allocator.free(table.db_root);
    allocator.destroy(table);
    return sqlite.SQLITE_OK;
}

fn xDestroy(p_vtab: ?*sqlite.sqlite3_vtab) callconv(.c) c_int {
    return xDisconnect(p_vtab);
}

fn xOpen(p_vtab: ?*sqlite.sqlite3_vtab, pp_cursor: [*c]?*sqlite.sqlite3_vtab_cursor) callconv(.c) c_int {
    if (pp_cursor == null) return sqlite.SQLITE_MISUSE;
    const base = p_vtab orelse return sqlite.SQLITE_MISUSE;
    const table = tableFromBase(base);

    const cursor = allocator.create(DocsCursor) catch return sqlite.SQLITE_NOMEM;
    cursor.* = DocsCursor.init(table) catch {
        allocator.destroy(cursor);
        return sqlite.SQLITE_ERROR;
    };
    cursor.base.pVtab = p_vtab;
    pp_cursor[0] = &cursor.base;
    return sqlite.SQLITE_OK;
}

fn xClose(p_cursor: ?*sqlite.sqlite3_vtab_cursor) callconv(.c) c_int {
    const base = p_cursor orelse return sqlite.SQLITE_MISUSE;
    const cursor = cursorFromBase(base);
    cursor.deinit();
    allocator.destroy(cursor);
    return sqlite.SQLITE_OK;
}

fn xFilter(
    p_cursor: ?*sqlite.sqlite3_vtab_cursor,
    idx_num: c_int,
    _: [*c]const u8,
    argc: c_int,
    argv: [*c]?*sqlite.sqlite3_value,
) callconv(.c) c_int {
    const base = p_cursor orelse return sqlite.SQLITE_MISUSE;
    const cursor = cursorFromBase(base);
    cursor.clearFilters();

    var arg_i: usize = 0;
    const arg_count: usize = if (argc <= 0) 0 else @intCast(argc);
    if ((idx_num & idx_doc_type_eq) != 0) {
        if (arg_i >= arg_count) return sqlite.SQLITE_MISUSE;
        const value = argv[arg_i] orelse return sqlite.SQLITE_MISUSE;
        arg_i += 1;
        if (sqlite.sqlite3_value_type(value) != sqlite.SQLITE_NULL) {
            const ptr = sqlite.sqlite3_value_text(value) orelse return sqlite.SQLITE_MISUSE;
            const len: usize = @intCast(sqlite.sqlite3_value_bytes(value));
            cursor.setDocTypeFilter(@as([*]const u8, @ptrCast(ptr))[0..len]) catch return sqlite.SQLITE_NOMEM;
        }
    }

    if ((idx_num & idx_group_eq) != 0) {
        if (arg_i >= arg_count) return sqlite.SQLITE_MISUSE;
        const value = argv[arg_i] orelse return sqlite.SQLITE_MISUSE;
        if (sqlite.sqlite3_value_type(value) != sqlite.SQLITE_NULL) {
            const ptr = sqlite.sqlite3_value_text(value) orelse return sqlite.SQLITE_MISUSE;
            const len: usize = @intCast(sqlite.sqlite3_value_bytes(value));
            cursor.setGroupFilter(@as([*]const u8, @ptrCast(ptr))[0..len]) catch return sqlite.SQLITE_NOMEM;
        }
    }

    cursor.resetForFilter() catch return sqlite.SQLITE_ERROR;
    cursor.advance() catch return sqlite.SQLITE_ERROR;
    return sqlite.SQLITE_OK;
}

fn xNext(p_cursor: ?*sqlite.sqlite3_vtab_cursor) callconv(.c) c_int {
    const base = p_cursor orelse return sqlite.SQLITE_MISUSE;
    const cursor = cursorFromBase(base);
    cursor.advance() catch return sqlite.SQLITE_ERROR;
    return sqlite.SQLITE_OK;
}

fn xEof(p_cursor: ?*sqlite.sqlite3_vtab_cursor) callconv(.c) c_int {
    const base = p_cursor orelse return 1;
    const cursor = cursorFromBase(base);
    return if (cursor.eof) 1 else 0;
}

fn xColumn(p_cursor: ?*sqlite.sqlite3_vtab_cursor, p_ctx: ?*sqlite.sqlite3_context, col: c_int) callconv(.c) c_int {
    const base = p_cursor orelse return sqlite.SQLITE_MISUSE;
    const cursor = cursorFromBase(base);
    if (cursor.eof) {
        sqlite.sqlite3_result_null(p_ctx);
        return sqlite.SQLITE_OK;
    }

    switch (col) {
        0 => resultText(p_ctx, cursor.current_row.doc_id),
        1 => resultText(p_ctx, cursor.current_row.doc_type),
        2 => {
            if (cursor.current_row.group_id) |group_id| {
                resultText(p_ctx, group_id);
            } else {
                sqlite.sqlite3_result_null(p_ctx);
            }
        },
        3 => sqlite.sqlite3_result_int64(p_ctx, cursor.current_row.created_at_ms),
        4 => {
            if (cursor.current_row.payload_json) |payload_json| {
                resultText(p_ctx, payload_json);
            } else {
                sqlite.sqlite3_result_null(p_ctx);
            }
        },
        else => sqlite.sqlite3_result_null(p_ctx),
    }
    return sqlite.SQLITE_OK;
}

fn xRowid(p_cursor: ?*sqlite.sqlite3_vtab_cursor, p_rowid: [*c]sqlite.sqlite3_int64) callconv(.c) c_int {
    const base = p_cursor orelse return sqlite.SQLITE_MISUSE;
    const cursor = cursorFromBase(base);
    if (p_rowid == null) return sqlite.SQLITE_MISUSE;
    p_rowid[0] = cursor.rowid;
    return sqlite.SQLITE_OK;
}

fn xUpdate(
    p_vtab: ?*sqlite.sqlite3_vtab,
    _: c_int,
    _: [*c]?*sqlite.sqlite3_value,
    _: [*c]sqlite.sqlite3_int64,
) callconv(.c) c_int {
    if (p_vtab) |vtab| {
        vtab.zErrMsg = sqlite.sqlite3_mprintf("taludb_docs is read-only");
    }
    return sqlite.SQLITE_READONLY;
}

fn xBegin(_: ?*sqlite.sqlite3_vtab) callconv(.c) c_int {
    return sqlite.SQLITE_OK;
}

fn xSync(_: ?*sqlite.sqlite3_vtab) callconv(.c) c_int {
    return sqlite.SQLITE_OK;
}

fn xCommit(_: ?*sqlite.sqlite3_vtab) callconv(.c) c_int {
    return sqlite.SQLITE_OK;
}

fn xRollback(_: ?*sqlite.sqlite3_vtab) callconv(.c) c_int {
    return sqlite.SQLITE_OK;
}

fn xFindFunction(
    _: ?*sqlite.sqlite3_vtab,
    _: c_int,
    _: [*c]const u8,
    _: [*c]?*const fn (?*sqlite.sqlite3_context, c_int, [*c]?*sqlite.sqlite3_value) callconv(.c) void,
    _: [*c]?*anyopaque,
) callconv(.c) c_int {
    return 0;
}

fn xRename(_: ?*sqlite.sqlite3_vtab, _: [*c]const u8) callconv(.c) c_int {
    return sqlite.SQLITE_OK;
}

fn xSavepoint(_: ?*sqlite.sqlite3_vtab, _: c_int) callconv(.c) c_int {
    return sqlite.SQLITE_OK;
}

fn xRelease(_: ?*sqlite.sqlite3_vtab, _: c_int) callconv(.c) c_int {
    return sqlite.SQLITE_OK;
}

fn xRollbackTo(_: ?*sqlite.sqlite3_vtab, _: c_int) callconv(.c) c_int {
    return sqlite.SQLITE_OK;
}

fn findColumn(descs: []const types.ColumnDesc, col_id: u32) ?types.ColumnDesc {
    for (descs) |d| {
        if (d.column_id == col_id) return d;
    }
    return null;
}

fn computeHash(value: []const u8) u64 {
    return std.hash.Wyhash.hash(0, value);
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

fn readVarBytesBuffers(
    file: std.fs.File,
    block_offset: u64,
    desc: types.ColumnDesc,
    row_count: u32,
    alloc: std.mem.Allocator,
) !VarBytesBuffers {
    if (desc.offsets_off == 0 or desc.lengths_off == 0) return error.InvalidColumnLayout;

    const reader = block_reader.BlockReader.init(file, alloc);
    const data = try reader.readColumnData(block_offset, desc, alloc);
    errdefer alloc.free(data);
    const offsets = try readU32Array(file, block_offset + @as(u64, desc.offsets_off), row_count, alloc);
    errdefer alloc.free(offsets);
    const lengths = try readU32Array(file, block_offset + @as(u64, desc.lengths_off), row_count, alloc);
    errdefer alloc.free(lengths);

    return .{
        .data = data,
        .offsets = offsets,
        .lengths = lengths,
    };
}

fn readU32Array(file: std.fs.File, offset: u64, count: u32, alloc: std.mem.Allocator) ![]u32 {
    const total_bytes = @as(usize, count) * @sizeOf(u32);
    const buffer = try alloc.alloc(u8, total_bytes);
    defer alloc.free(buffer);

    const read_len = try file.preadAll(buffer, offset);
    if (read_len != buffer.len) return error.UnexpectedEof;

    const values = try alloc.alloc(u32, count);
    var i: usize = 0;
    while (i < values.len) : (i += 1) {
        const start = i * 4;
        values[i] = std.mem.readInt(u32, buffer[start..][0..4], .little);
    }
    return values;
}

test "registerDocsModule creates taludb_docs virtual table skeleton" {
    var db: ?*sqlite.sqlite3 = null;
    try std.testing.expectEqual(@as(c_int, sqlite.SQLITE_OK), sqlite.sqlite3_open(":memory:", &db));
    defer _ = sqlite.sqlite3_close(db);

    try registerDocsModule(db.?, "/tmp/talu-sql-test");

    var err_msg: [*c]u8 = null;
    defer if (err_msg != null) sqlite.sqlite3_free(err_msg);

    const create_sql: [:0]const u8 = "CREATE VIRTUAL TABLE docs USING taludb_docs";
    try std.testing.expectEqual(
        @as(c_int, sqlite.SQLITE_OK),
        sqlite.sqlite3_exec(db, create_sql.ptr, null, null, &err_msg),
    );

    const scan_sql: [:0]const u8 = "SELECT count(*) FROM docs";
    try std.testing.expectEqual(
        @as(c_int, sqlite.SQLITE_OK),
        sqlite.sqlite3_exec(db, scan_sql.ptr, null, null, &err_msg),
    );
}

test "taludb_docs scan returns latest active documents" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    var adapter = try documents.DocumentAdapter.init(std.testing.allocator, root);
    defer adapter.deinit();

    try adapter.writeDocument(.{
        .doc_id = "doc-a",
        .doc_type = "prompt",
        .title = "A1",
        .doc_json = "{\"_sys\":{},\"data\":{\"v\":1}}",
        .created_at_ms = 1000,
        .updated_at_ms = 2000,
        .group_id = "g1",
    });
    try adapter.writeDocument(.{
        .doc_id = "doc-a",
        .doc_type = "prompt",
        .title = "A2",
        .doc_json = "{\"_sys\":{},\"data\":{\"v\":2}}",
        .created_at_ms = 1000,
        .updated_at_ms = 3000,
        .group_id = "g1",
    });
    try adapter.writeDocument(.{
        .doc_id = "doc-b",
        .doc_type = "prompt",
        .title = "B1",
        .doc_json = "{\"_sys\":{},\"data\":{\"v\":9}}",
        .created_at_ms = 1000,
        .updated_at_ms = 2500,
        .group_id = "g2",
    });
    try adapter.flush();
    adapter.fs_writer.resetSchema();
    try adapter.deleteDocument("doc-b", 4000);
    try adapter.flush();

    var db: ?*sqlite.sqlite3 = null;
    try std.testing.expectEqual(@as(c_int, sqlite.SQLITE_OK), sqlite.sqlite3_open(":memory:", &db));
    defer _ = sqlite.sqlite3_close(db);

    try registerDocsModule(db.?, root);

    var err_msg: [*c]u8 = null;
    defer if (err_msg != null) sqlite.sqlite3_free(err_msg);
    const create_sql: [:0]const u8 = "CREATE VIRTUAL TABLE docs USING taludb_docs";
    try std.testing.expectEqual(
        @as(c_int, sqlite.SQLITE_OK),
        sqlite.sqlite3_exec(db, create_sql.ptr, null, null, &err_msg),
    );

    var stmt: ?*sqlite.sqlite3_stmt = null;
    const query: [:0]const u8 =
        "SELECT doc_id, doc_type, group_id, created_at_ms, payload FROM docs ORDER BY doc_id";
    try std.testing.expectEqual(
        @as(c_int, sqlite.SQLITE_OK),
        sqlite.sqlite3_prepare_v2(db, query.ptr, -1, &stmt, null),
    );
    defer _ = sqlite.sqlite3_finalize(stmt);

    var row_count: usize = 0;
    while (true) {
        const rc = sqlite.sqlite3_step(stmt);
        if (rc == sqlite.SQLITE_ROW) {
            row_count += 1;
            try std.testing.expectEqual(@as(usize, 1), row_count);

            const doc_id_ptr = sqlite.sqlite3_column_text(stmt, 0) orelse return error.InvalidColumnData;
            const doc_id_len: usize = @intCast(sqlite.sqlite3_column_bytes(stmt, 0));
            try std.testing.expectEqualStrings("doc-a", @as([*]const u8, @ptrCast(doc_id_ptr))[0..doc_id_len]);

            const doc_type_ptr = sqlite.sqlite3_column_text(stmt, 1) orelse return error.InvalidColumnData;
            const doc_type_len: usize = @intCast(sqlite.sqlite3_column_bytes(stmt, 1));
            try std.testing.expectEqualStrings("prompt", @as([*]const u8, @ptrCast(doc_type_ptr))[0..doc_type_len]);

            const group_ptr = sqlite.sqlite3_column_text(stmt, 2) orelse return error.InvalidColumnData;
            const group_len: usize = @intCast(sqlite.sqlite3_column_bytes(stmt, 2));
            try std.testing.expectEqualStrings("g1", @as([*]const u8, @ptrCast(group_ptr))[0..group_len]);

            const created_at = sqlite.sqlite3_column_int64(stmt, 3);
            try std.testing.expectEqual(@as(i64, 1000), created_at);

            const payload_ptr = sqlite.sqlite3_column_text(stmt, 4) orelse return error.InvalidColumnData;
            const payload_len: usize = @intCast(sqlite.sqlite3_column_bytes(stmt, 4));
            const payload = @as([*]const u8, @ptrCast(payload_ptr))[0..payload_len];
            try std.testing.expect(std.mem.indexOf(u8, payload, "\"v\":2") != null);
        } else if (rc == sqlite.SQLITE_DONE) {
            break;
        } else {
            return error.UnexpectedSqliteResult;
        }
    }

    try std.testing.expectEqual(@as(usize, 1), row_count);
}

test "taludb_docs supports equality filtering on group_id and doc_type" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    var adapter = try documents.DocumentAdapter.init(std.testing.allocator, root);
    defer adapter.deinit();

    try adapter.writeDocument(.{
        .doc_id = "doc-1",
        .doc_type = "prompt",
        .title = "Prompt 1",
        .doc_json = "{\"_sys\":{},\"data\":{\"k\":1}}",
        .created_at_ms = 100,
        .updated_at_ms = 100,
        .group_id = "g1",
    });
    try adapter.writeDocument(.{
        .doc_id = "doc-2",
        .doc_type = "prompt",
        .title = "Prompt 2",
        .doc_json = "{\"_sys\":{},\"data\":{\"k\":2}}",
        .created_at_ms = 200,
        .updated_at_ms = 200,
        .group_id = "g2",
    });
    try adapter.writeDocument(.{
        .doc_id = "doc-3",
        .doc_type = "persona",
        .title = "Persona 3",
        .doc_json = "{\"_sys\":{},\"data\":{\"k\":3}}",
        .created_at_ms = 300,
        .updated_at_ms = 300,
        .group_id = "g1",
    });
    try adapter.flush();

    var db: ?*sqlite.sqlite3 = null;
    try std.testing.expectEqual(@as(c_int, sqlite.SQLITE_OK), sqlite.sqlite3_open(":memory:", &db));
    defer _ = sqlite.sqlite3_close(db);

    try registerDocsModule(db.?, root);

    var err_msg: [*c]u8 = null;
    defer if (err_msg != null) sqlite.sqlite3_free(err_msg);
    const create_sql: [:0]const u8 = "CREATE VIRTUAL TABLE docs USING taludb_docs";
    try std.testing.expectEqual(
        @as(c_int, sqlite.SQLITE_OK),
        sqlite.sqlite3_exec(db, create_sql.ptr, null, null, &err_msg),
    );

    var stmt: ?*sqlite.sqlite3_stmt = null;
    const query: [:0]const u8 =
        "SELECT doc_id FROM docs WHERE group_id = 'g1' AND doc_type = 'prompt'";
    try std.testing.expectEqual(
        @as(c_int, sqlite.SQLITE_OK),
        sqlite.sqlite3_prepare_v2(db, query.ptr, -1, &stmt, null),
    );
    defer _ = sqlite.sqlite3_finalize(stmt);

    var row_count: usize = 0;
    while (true) {
        const rc = sqlite.sqlite3_step(stmt);
        if (rc == sqlite.SQLITE_ROW) {
            row_count += 1;
            const doc_id_ptr = sqlite.sqlite3_column_text(stmt, 0) orelse return error.InvalidColumnData;
            const doc_id_len: usize = @intCast(sqlite.sqlite3_column_bytes(stmt, 0));
            try std.testing.expectEqualStrings(
                "doc-1",
                @as([*]const u8, @ptrCast(doc_id_ptr))[0..doc_id_len],
            );
        } else if (rc == sqlite.SQLITE_DONE) {
            break;
        } else {
            return error.UnexpectedSqliteResult;
        }
    }

    try std.testing.expectEqual(@as(usize, 1), row_count);
}
