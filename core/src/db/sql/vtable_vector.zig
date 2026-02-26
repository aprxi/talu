//! SQLite table-valued function for vector similarity search.
//!
//! Registers as `vector_search` (eponymous — no CREATE VIRTUAL TABLE needed).
//! Usage:
//!   SELECT id, score FROM vector_search('collection_name', ?query, 10)
//!   SELECT id, score FROM vector_search('coll', ?query, 10, '[1, 3]')
//!
//! Where ?query is a blob parameter (packed f32 array).
//! Arguments map to HIDDEN columns: collection (TEXT), query (BLOB), k (INTEGER),
//! filter_ids (TEXT — JSON array of integer IDs for pre-filtered search).

const std = @import("std");
const sqlite = @import("c.zig").sqlite3;
const vector_store = @import("../vector/store.zig");
const vector_filter = @import("../vector/filter.zig");

const allocator = std.heap.c_allocator;

const module_name: [:0]const u8 = "vector_search";
const default_k: u32 = 10;

// TVF schema: visible result columns + HIDDEN argument columns.
const schema_sql: [:0]const u8 =
    \\CREATE TABLE x(
    \\  id INTEGER,
    \\  score REAL,
    \\  collection TEXT HIDDEN,
    \\  query BLOB HIDDEN,
    \\  k INTEGER HIDDEN,
    \\  filter_ids TEXT HIDDEN
    \\)
;

// xBestIndex constraint bitmask for HIDDEN columns.
const idx_collection_eq: c_int = 1 << 0;
const idx_query_eq: c_int = 1 << 1;
const idx_k_eq: c_int = 1 << 2;
const idx_filter_ids_eq: c_int = 1 << 3;

const heap_destructor: sqlite.sqlite3_destructor_type = sqlite.sqlite3_free;

// =============================================================================
// Structures
// =============================================================================

const ModuleAux = struct {
    db_root: []u8,
};

const VectorVTab = struct {
    base: sqlite.sqlite3_vtab = std.mem.zeroes(sqlite.sqlite3_vtab),
    db_root: []u8,
};

const VectorCursor = struct {
    base: sqlite.sqlite3_vtab_cursor = std.mem.zeroes(sqlite.sqlite3_vtab_cursor),
    table: *VectorVTab,
    ids: []u64 = &.{},
    scores: []f32 = &.{},
    filter_id_buf: []u64 = &.{},
    row_idx: usize = 0,
    rowid: sqlite.sqlite3_int64 = 0,
    eof: bool = true,

    fn freeResults(self: *VectorCursor) void {
        if (self.ids.len > 0) allocator.free(self.ids);
        if (self.scores.len > 0) allocator.free(self.scores);
        if (self.filter_id_buf.len > 0) allocator.free(self.filter_id_buf);
        self.ids = &.{};
        self.scores = &.{};
        self.filter_id_buf = &.{};
    }
};

// =============================================================================
// Module definition (eponymous: xCreate = xConnect)
// =============================================================================

const vector_module: sqlite.sqlite3_module = blk: {
    var m = std.mem.zeroes(sqlite.sqlite3_module);
    m.iVersion = 2;
    m.xCreate = xConnect;
    m.xConnect = xConnect;
    m.xBestIndex = xBestIndex;
    m.xDisconnect = xDisconnect;
    m.xDestroy = xDisconnect;
    m.xOpen = xOpen;
    m.xClose = xClose;
    m.xFilter = xFilter;
    m.xNext = xNext;
    m.xEof = xEof;
    m.xColumn = xColumn;
    m.xRowid = xRowid;
    break :blk m;
};

/// Register the `vector_search` eponymous virtual table module.
pub fn registerVectorModule(db: *sqlite.sqlite3, db_root: []const u8) !void {
    const aux = try allocator.create(ModuleAux);
    errdefer allocator.destroy(aux);
    aux.db_root = try allocator.dupe(u8, db_root);
    errdefer allocator.free(aux.db_root);

    const rc = sqlite.sqlite3_create_module_v2(
        db,
        module_name.ptr,
        &vector_module,
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

// =============================================================================
// Helper casts
// =============================================================================

fn tableFromBase(base: *sqlite.sqlite3_vtab) *VectorVTab {
    return @alignCast(@fieldParentPtr("base", @as(*sqlite.sqlite3_vtab, @alignCast(base))));
}

fn cursorFromBase(base: *sqlite.sqlite3_vtab_cursor) *VectorCursor {
    return @alignCast(@fieldParentPtr("base", @as(*sqlite.sqlite3_vtab_cursor, @alignCast(base))));
}

fn setErrMsg(pz_err: [*c][*c]u8, msg: [:0]const u8) void {
    if (pz_err == null) return;
    pz_err[0] = sqlite.sqlite3_mprintf(msg.ptr);
}

// =============================================================================
// Virtual table lifecycle
// =============================================================================

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
        setErrMsg(pz_err, "missing vector_search module aux");
        return sqlite.SQLITE_ERROR;
    }));

    const decl_rc = sqlite.sqlite3_declare_vtab(db_handle, schema_sql.ptr);
    if (decl_rc != sqlite.SQLITE_OK) {
        setErrMsg(pz_err, "failed to declare vector_search schema");
        return decl_rc;
    }

    const table = allocator.create(VectorVTab) catch {
        setErrMsg(pz_err, "out of memory allocating vector_search table");
        return sqlite.SQLITE_NOMEM;
    };
    errdefer allocator.destroy(table);

    const db_root_copy = allocator.dupe(u8, aux.db_root) catch {
        setErrMsg(pz_err, "out of memory duplicating vector_search db_root");
        return sqlite.SQLITE_NOMEM;
    };

    table.* = .{
        .base = std.mem.zeroes(sqlite.sqlite3_vtab),
        .db_root = db_root_copy,
    };
    pp_vtab[0] = &table.base;
    return sqlite.SQLITE_OK;
}

fn xDisconnect(p_vtab: ?*sqlite.sqlite3_vtab) callconv(.c) c_int {
    const base = p_vtab orelse return sqlite.SQLITE_MISUSE;
    const table = tableFromBase(base);
    allocator.free(table.db_root);
    allocator.destroy(table);
    return sqlite.SQLITE_OK;
}

fn xOpen(p_vtab: ?*sqlite.sqlite3_vtab, pp_cursor: [*c]?*sqlite.sqlite3_vtab_cursor) callconv(.c) c_int {
    if (pp_cursor == null) return sqlite.SQLITE_MISUSE;
    const base = p_vtab orelse return sqlite.SQLITE_MISUSE;
    const table = tableFromBase(base);

    const cursor = allocator.create(VectorCursor) catch return sqlite.SQLITE_NOMEM;
    cursor.* = .{
        .base = std.mem.zeroes(sqlite.sqlite3_vtab_cursor),
        .table = table,
    };
    cursor.base.pVtab = p_vtab;
    pp_cursor[0] = &cursor.base;
    return sqlite.SQLITE_OK;
}

fn xClose(p_cursor: ?*sqlite.sqlite3_vtab_cursor) callconv(.c) c_int {
    const base = p_cursor orelse return sqlite.SQLITE_MISUSE;
    const cursor = cursorFromBase(base);
    cursor.freeResults();
    allocator.destroy(cursor);
    return sqlite.SQLITE_OK;
}

// =============================================================================
// Query planning
// =============================================================================

fn xBestIndex(_: ?*sqlite.sqlite3_vtab, p_info: ?*sqlite.sqlite3_index_info) callconv(.c) c_int {
    const info = p_info orelse return sqlite.SQLITE_MISUSE;

    info.idxNum = 0;
    info.orderByConsumed = 0;
    info.estimatedCost = 1e9;
    info.estimatedRows = 1_000_000;

    const n: usize = @intCast(info.nConstraint);
    if (n == 0) return sqlite.SQLITE_OK;

    const constraints = info.aConstraint[0..n];
    const usages = info.aConstraintUsage[0..n];

    var collection_idx: ?usize = null;
    var query_idx: ?usize = null;
    var k_idx: ?usize = null;
    var filter_ids_idx: ?usize = null;

    for (constraints, 0..) |constraint, idx| {
        if (constraint.usable == 0) continue;
        if (constraint.op != sqlite.SQLITE_INDEX_CONSTRAINT_EQ) continue;
        switch (constraint.iColumn) {
            2 => {
                if (collection_idx == null) collection_idx = idx;
            },
            3 => {
                if (query_idx == null) query_idx = idx;
            },
            4 => {
                if (k_idx == null) k_idx = idx;
            },
            5 => {
                if (filter_ids_idx == null) filter_ids_idx = idx;
            },
            else => {},
        }
    }

    // Both collection and query are required for a usable plan.
    if (collection_idx == null or query_idx == null) return sqlite.SQLITE_OK;

    // Assign argvIndex in order: collection=1, query=2, k=3.
    var next_arg: c_int = 1;

    usages[collection_idx.?].argvIndex = next_arg;
    usages[collection_idx.?].omit = 1;
    info.idxNum |= idx_collection_eq;
    next_arg += 1;

    usages[query_idx.?].argvIndex = next_arg;
    usages[query_idx.?].omit = 1;
    info.idxNum |= idx_query_eq;
    next_arg += 1;

    if (k_idx) |ki| {
        usages[ki].argvIndex = next_arg;
        usages[ki].omit = 1;
        info.idxNum |= idx_k_eq;
        next_arg += 1;
    }

    if (filter_ids_idx) |fi| {
        usages[fi].argvIndex = next_arg;
        usages[fi].omit = 1;
        info.idxNum |= idx_filter_ids_eq;
    }

    info.estimatedCost = 10.0;
    info.estimatedRows = 100;

    return sqlite.SQLITE_OK;
}

// =============================================================================
// Query execution
// =============================================================================

fn xFilter(
    p_cursor: ?*sqlite.sqlite3_vtab_cursor,
    idx_num: c_int,
    _: [*c]const u8,
    argc: c_int,
    argv: [*c]?*sqlite.sqlite3_value,
) callconv(.c) c_int {
    const base = p_cursor orelse return sqlite.SQLITE_MISUSE;
    const cursor = cursorFromBase(base);
    cursor.freeResults();
    cursor.row_idx = 0;
    cursor.rowid = 0;

    // Both collection and query constraints are required.
    if ((idx_num & idx_collection_eq) == 0 or (idx_num & idx_query_eq) == 0) {
        cursor.eof = true;
        return sqlite.SQLITE_OK;
    }

    const arg_count: usize = if (argc <= 0) 0 else @intCast(argc);
    var arg_i: usize = 0;

    // Arg 1: collection name (TEXT).
    if (arg_i >= arg_count) return sqlite.SQLITE_MISUSE;
    const collection_val = argv[arg_i] orelse return sqlite.SQLITE_MISUSE;
    arg_i += 1;
    if (sqlite.sqlite3_value_type(collection_val) != sqlite.SQLITE_TEXT) {
        cursor.eof = true;
        return sqlite.SQLITE_OK;
    }
    const collection_ptr = sqlite.sqlite3_value_text(collection_val) orelse return sqlite.SQLITE_MISUSE;
    const collection_len: usize = @intCast(sqlite.sqlite3_value_bytes(collection_val));
    const collection_name = @as([*]const u8, @ptrCast(collection_ptr))[0..collection_len];

    // Arg 2: query vector (BLOB — packed f32).
    if (arg_i >= arg_count) return sqlite.SQLITE_MISUSE;
    const query_val = argv[arg_i] orelse return sqlite.SQLITE_MISUSE;
    arg_i += 1;
    if (sqlite.sqlite3_value_type(query_val) != sqlite.SQLITE_BLOB) {
        cursor.eof = true;
        return sqlite.SQLITE_OK;
    }
    const query_blob_ptr = sqlite.sqlite3_value_blob(query_val) orelse {
        cursor.eof = true;
        return sqlite.SQLITE_OK;
    };
    const query_blob_len: usize = @intCast(sqlite.sqlite3_value_bytes(query_val));
    if (query_blob_len == 0 or query_blob_len % @sizeOf(f32) != 0) {
        cursor.eof = true;
        return sqlite.SQLITE_OK;
    }
    const query_blob = @as([*]const u8, @ptrCast(query_blob_ptr))[0..query_blob_len];

    // Arg 3: k (INTEGER, optional — default 10).
    var k: u32 = default_k;
    if ((idx_num & idx_k_eq) != 0) {
        if (arg_i < arg_count) {
            const k_val = argv[arg_i] orelse return sqlite.SQLITE_MISUSE;
            arg_i += 1;
            if (sqlite.sqlite3_value_type(k_val) == sqlite.SQLITE_INTEGER) {
                const k_raw = sqlite.sqlite3_value_int64(k_val);
                if (k_raw > 0 and k_raw <= std.math.maxInt(u32)) {
                    k = @intCast(k_raw);
                }
            }
        }
    }

    // Arg 4: filter_ids (TEXT, optional — JSON array of integer IDs).
    var has_filter_ids = false;
    if ((idx_num & idx_filter_ids_eq) != 0) {
        if (arg_i < arg_count) {
            const filter_val = argv[arg_i] orelse return sqlite.SQLITE_MISUSE;
            if (sqlite.sqlite3_value_type(filter_val) == sqlite.SQLITE_TEXT) {
                const filter_ptr = sqlite.sqlite3_value_text(filter_val) orelse {
                    cursor.eof = true;
                    return sqlite.SQLITE_OK;
                };
                const filter_len: usize = @intCast(sqlite.sqlite3_value_bytes(filter_val));
                const filter_text = @as([*]const u8, @ptrCast(filter_ptr))[0..filter_len];

                const parsed_ids = parseFilterIds(filter_text) orelse {
                    // Malformed JSON → graceful 0 rows.
                    cursor.eof = true;
                    return sqlite.SQLITE_OK;
                };
                has_filter_ids = true;
                cursor.filter_id_buf = parsed_ids;

                if (parsed_ids.len == 0) {
                    // Empty array → 0 rows.
                    cursor.eof = true;
                    return sqlite.SQLITE_OK;
                }
            }
        }
    }

    // Copy blob to aligned f32 buffer (SQLite blob may be unaligned).
    const num_floats = query_blob_len / @sizeOf(f32);
    const query_floats = allocator.alloc(f32, num_floats) catch return sqlite.SQLITE_NOMEM;
    defer allocator.free(query_floats);
    @memcpy(std.mem.sliceAsBytes(query_floats), query_blob);

    // Build collection path: {db_root}/vector/collections/{collection_name}
    const collection_path = std.fmt.allocPrint(
        allocator,
        "{s}/vector/collections/{s}",
        .{ cursor.table.db_root, collection_name },
    ) catch return sqlite.SQLITE_NOMEM;
    defer allocator.free(collection_path);

    // Open adapter, search, transfer results to cursor.
    var adapter = vector_store.VectorAdapter.init(allocator, collection_path) catch {
        cursor.eof = true;
        return sqlite.SQLITE_ERROR;
    };
    defer adapter.deinit();

    // Use searchWithOptions when filter_ids are provided.
    var search_opts = vector_store.SearchOptions{};
    var filter_expr: vector_filter.FilterExpr = undefined;
    if (has_filter_ids and cursor.filter_id_buf.len > 0) {
        filter_expr = .{ .id_in = cursor.filter_id_buf };
        search_opts.filter_expr = &filter_expr;
    }

    const result = adapter.searchWithOptions(allocator, query_floats, k, search_opts) catch {
        cursor.eof = true;
        return sqlite.SQLITE_ERROR;
    };

    // Transfer ownership of ids and scores to cursor.
    cursor.ids = result.ids;
    cursor.scores = result.scores;
    cursor.row_idx = 0;
    cursor.eof = (result.ids.len == 0);
    // Do NOT call result.deinit() — cursor owns the memory now.

    return sqlite.SQLITE_OK;
}

fn xNext(p_cursor: ?*sqlite.sqlite3_vtab_cursor) callconv(.c) c_int {
    const base = p_cursor orelse return sqlite.SQLITE_MISUSE;
    const cursor = cursorFromBase(base);
    cursor.row_idx += 1;
    cursor.rowid += 1;
    if (cursor.row_idx >= cursor.ids.len) {
        cursor.eof = true;
    }
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
    if (cursor.eof or cursor.row_idx >= cursor.ids.len) {
        sqlite.sqlite3_result_null(p_ctx);
        return sqlite.SQLITE_OK;
    }

    switch (col) {
        0 => sqlite.sqlite3_result_int64(p_ctx, @intCast(cursor.ids[cursor.row_idx])),
        1 => sqlite.sqlite3_result_double(p_ctx, @floatCast(cursor.scores[cursor.row_idx])),
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

// =============================================================================
// Helpers
// =============================================================================

/// Parse a JSON array of integer IDs (e.g. "[1, 2, 3]") into a u64 slice.
/// Returns null on malformed JSON, non-array, non-integer elements, or negative values.
/// Returns &.{} (empty slice) for "[]".
/// Caller owns returned slice via allocator.
fn parseFilterIds(text: []const u8) ?[]u64 {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, text, .{}) catch return null;
    defer parsed.deinit();

    const arr = switch (parsed.value) {
        .array => |a| a,
        else => return null,
    };

    if (arr.items.len == 0) {
        return allocator.alloc(u64, 0) catch return null;
    }

    const ids = allocator.alloc(u64, arr.items.len) catch return null;
    for (arr.items, 0..) |item, i| {
        switch (item) {
            .integer => |v| {
                if (v < 0) {
                    allocator.free(ids);
                    return null;
                }
                ids[i] = @intCast(v);
            },
            else => {
                allocator.free(ids);
                return null;
            },
        }
    }
    return ids;
}

// =============================================================================
// Tests
// =============================================================================

test "vector_search returns top-k results by score" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    // Create collection directory and populate with vectors.
    const collection_path = try std.fmt.allocPrint(
        std.testing.allocator,
        "{s}/vector/collections/emb",
        .{root},
    );
    defer std.testing.allocator.free(collection_path);
    try std.fs.cwd().makePath(collection_path);

    {
        var adapter = try vector_store.VectorAdapter.init(std.testing.allocator, collection_path);
        defer adapter.deinit(); // deinit flushes to disk
        const ids = [_]u64{ 1, 2, 3 };
        // dim=3: id=1→[1,0,0], id=2→[0,1,0], id=3→[0,0,1]
        const vectors = [_]f32{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        try adapter.appendBatch(&ids, &vectors, 3);
    }

    // Query via SQL engine (opens its own adapter).
    const engine = @import("engine.zig");
    const query_vec = [_]f32{ 1, 0, 0 };
    const query_blob = std.mem.sliceAsBytes(&query_vec);
    const params = [_]engine.SqlParam{
        engine.SqlParam.initText("emb"),
        engine.SqlParam.initBlob(query_blob),
        engine.SqlParam.initInt(2),
    };
    const result = try engine.queryJsonParams(
        std.testing.allocator,
        root,
        "SELECT id, score FROM vector_search(?1, ?2, ?3)",
        &params,
    );
    defer std.testing.allocator.free(result);

    // Should have 2 rows (k=2), top result is id=1.
    try std.testing.expect(std.mem.indexOf(u8, result, "\"row_count\":2") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"id\":1") != null);
}

test "vector_search respects k parameter" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    const collection_path = try std.fmt.allocPrint(
        std.testing.allocator,
        "{s}/vector/collections/emb2",
        .{root},
    );
    defer std.testing.allocator.free(collection_path);
    try std.fs.cwd().makePath(collection_path);

    {
        var adapter = try vector_store.VectorAdapter.init(std.testing.allocator, collection_path);
        defer adapter.deinit();
        const ids = [_]u64{ 1, 2, 3 };
        const vectors = [_]f32{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        try adapter.appendBatch(&ids, &vectors, 3);
    }

    const engine = @import("engine.zig");
    const query_vec = [_]f32{ 1, 0, 0 };
    const query_blob = std.mem.sliceAsBytes(&query_vec);
    const params = [_]engine.SqlParam{
        engine.SqlParam.initText("emb2"),
        engine.SqlParam.initBlob(query_blob),
        engine.SqlParam.initInt(1),
    };
    const result = try engine.queryJsonParams(
        std.testing.allocator,
        root,
        "SELECT id, score FROM vector_search(?1, ?2, ?3)",
        &params,
    );
    defer std.testing.allocator.free(result);

    try std.testing.expect(std.mem.indexOf(u8, result, "\"row_count\":1") != null);
}

test "vector_search returns empty for nonexistent collection" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    const engine = @import("engine.zig");
    const query_vec = [_]f32{ 1, 0, 0 };
    const query_blob = std.mem.sliceAsBytes(&query_vec);
    const params = [_]engine.SqlParam{
        engine.SqlParam.initText("nonexistent"),
        engine.SqlParam.initBlob(query_blob),
        engine.SqlParam.initInt(5),
    };

    // VectorAdapter.init auto-creates the path; search returns 0 results.
    const result = try engine.queryJsonParams(
        std.testing.allocator,
        root,
        "SELECT id, score FROM vector_search(?1, ?2, ?3)",
        &params,
    );
    defer std.testing.allocator.free(result);

    try std.testing.expect(std.mem.indexOf(u8, result, "\"row_count\":0") != null);
}

test "vector_search validates query blob alignment" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    const collection_path = try std.fmt.allocPrint(
        std.testing.allocator,
        "{s}/vector/collections/emb3",
        .{root},
    );
    defer std.testing.allocator.free(collection_path);
    try std.fs.cwd().makePath(collection_path);

    {
        var adapter = try vector_store.VectorAdapter.init(std.testing.allocator, collection_path);
        defer adapter.deinit();
        const ids = [_]u64{1};
        const vectors = [_]f32{ 1, 0, 0 };
        try adapter.appendBatch(&ids, &vectors, 3);
    }

    const engine = @import("engine.zig");
    // 5 bytes — not divisible by sizeof(f32)=4 → xFilter sets eof=true, returns 0 rows.
    const bad_blob = [_]u8{ 1, 2, 3, 4, 5 };
    const params = [_]engine.SqlParam{
        engine.SqlParam.initText("emb3"),
        engine.SqlParam.initBlob(&bad_blob),
        engine.SqlParam.initInt(5),
    };
    const result = try engine.queryJsonParams(
        std.testing.allocator,
        root,
        "SELECT id, score FROM vector_search(?1, ?2, ?3)",
        &params,
    );
    defer std.testing.allocator.free(result);

    // Misaligned blob → 0 results (not an error, just empty).
    try std.testing.expect(std.mem.indexOf(u8, result, "\"row_count\":0") != null);
}

test "vector_search with filter_ids returns only filtered results" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    const collection_path = try std.fmt.allocPrint(
        std.testing.allocator,
        "{s}/vector/collections/emb_filt",
        .{root},
    );
    defer std.testing.allocator.free(collection_path);
    try std.fs.cwd().makePath(collection_path);

    {
        var adapter = try vector_store.VectorAdapter.init(std.testing.allocator, collection_path);
        defer adapter.deinit();
        const ids = [_]u64{ 1, 2, 3 };
        const vectors = [_]f32{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        try adapter.appendBatch(&ids, &vectors, 3);
    }

    const engine = @import("engine.zig");
    const query_vec = [_]f32{ 1, 0, 0 };
    const query_blob = std.mem.sliceAsBytes(&query_vec);
    const params = [_]engine.SqlParam{
        engine.SqlParam.initText("emb_filt"),
        engine.SqlParam.initBlob(query_blob),
        engine.SqlParam.initInt(10),
        engine.SqlParam.initText("[1, 3]"),
    };
    const result = try engine.queryJsonParams(
        std.testing.allocator,
        root,
        "SELECT id, score FROM vector_search(?1, ?2, ?3, ?4)",
        &params,
    );
    defer std.testing.allocator.free(result);

    // Should return only ids 1 and 3 (not 2).
    try std.testing.expect(std.mem.indexOf(u8, result, "\"row_count\":2") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"id\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"id\":3") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"id\":2") == null);
}

test "vector_search with empty filter_ids returns no results" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    const collection_path = try std.fmt.allocPrint(
        std.testing.allocator,
        "{s}/vector/collections/emb_empty_filt",
        .{root},
    );
    defer std.testing.allocator.free(collection_path);
    try std.fs.cwd().makePath(collection_path);

    {
        var adapter = try vector_store.VectorAdapter.init(std.testing.allocator, collection_path);
        defer adapter.deinit();
        const ids = [_]u64{ 1, 2, 3 };
        const vectors = [_]f32{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        try adapter.appendBatch(&ids, &vectors, 3);
    }

    const engine = @import("engine.zig");
    const query_vec = [_]f32{ 1, 0, 0 };
    const query_blob = std.mem.sliceAsBytes(&query_vec);
    const params = [_]engine.SqlParam{
        engine.SqlParam.initText("emb_empty_filt"),
        engine.SqlParam.initBlob(query_blob),
        engine.SqlParam.initInt(10),
        engine.SqlParam.initText("[]"),
    };
    const result = try engine.queryJsonParams(
        std.testing.allocator,
        root,
        "SELECT id, score FROM vector_search(?1, ?2, ?3, ?4)",
        &params,
    );
    defer std.testing.allocator.free(result);

    try std.testing.expect(std.mem.indexOf(u8, result, "\"row_count\":0") != null);
}

test "vector_search with invalid filter_ids JSON returns no results" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    const collection_path = try std.fmt.allocPrint(
        std.testing.allocator,
        "{s}/vector/collections/emb_bad_filt",
        .{root},
    );
    defer std.testing.allocator.free(collection_path);
    try std.fs.cwd().makePath(collection_path);

    {
        var adapter = try vector_store.VectorAdapter.init(std.testing.allocator, collection_path);
        defer adapter.deinit();
        const ids = [_]u64{1};
        const vectors = [_]f32{ 1, 0, 0 };
        try adapter.appendBatch(&ids, &vectors, 3);
    }

    const engine = @import("engine.zig");
    const query_vec = [_]f32{ 1, 0, 0 };
    const query_blob = std.mem.sliceAsBytes(&query_vec);
    const params = [_]engine.SqlParam{
        engine.SqlParam.initText("emb_bad_filt"),
        engine.SqlParam.initBlob(query_blob),
        engine.SqlParam.initInt(10),
        engine.SqlParam.initText("not json"),
    };
    const result = try engine.queryJsonParams(
        std.testing.allocator,
        root,
        "SELECT id, score FROM vector_search(?1, ?2, ?3, ?4)",
        &params,
    );
    defer std.testing.allocator.free(result);

    // Malformed JSON → graceful 0 rows.
    try std.testing.expect(std.mem.indexOf(u8, result, "\"row_count\":0") != null);
}

test "vector_search with non-integer filter_ids returns no results" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    const collection_path = try std.fmt.allocPrint(
        std.testing.allocator,
        "{s}/vector/collections/emb_str_filt",
        .{root},
    );
    defer std.testing.allocator.free(collection_path);
    try std.fs.cwd().makePath(collection_path);

    {
        var adapter = try vector_store.VectorAdapter.init(std.testing.allocator, collection_path);
        defer adapter.deinit();
        const ids = [_]u64{1};
        const vectors = [_]f32{ 1, 0, 0 };
        try adapter.appendBatch(&ids, &vectors, 3);
    }

    const engine = @import("engine.zig");
    const query_vec = [_]f32{ 1, 0, 0 };
    const query_blob = std.mem.sliceAsBytes(&query_vec);
    const params = [_]engine.SqlParam{
        engine.SqlParam.initText("emb_str_filt"),
        engine.SqlParam.initBlob(query_blob),
        engine.SqlParam.initInt(10),
        engine.SqlParam.initText("[\"a\",\"b\"]"),
    };
    const result = try engine.queryJsonParams(
        std.testing.allocator,
        root,
        "SELECT id, score FROM vector_search(?1, ?2, ?3, ?4)",
        &params,
    );
    defer std.testing.allocator.free(result);

    // Non-integer elements → graceful 0 rows.
    try std.testing.expect(std.mem.indexOf(u8, result, "\"row_count\":0") != null);
}
