//! SQLite virtual table for any Table namespace.
//!
//! Registers as `taludb_generic`. Table parameters:
//!   CREATE VIRTUAL TABLE t USING taludb_generic(namespace, schema_id [, delete_schema_id])
//!
//! Discovers columns from the first matching block's column directory.
//! Exposes scalar columns as INTEGER and payload as TEXT.
//! Uses Table.scan() for block iteration, dedup, and tombstone filtering.

const std = @import("std");
const generic = @import("../table/generic.zig");
const db_reader = @import("../reader.zig");
const block_reader = @import("../block_reader.zig");
const types = @import("../types.zig");
const sqlite = @import("c.zig").sqlite3;

const allocator = std.heap.c_allocator;
const module_name: [:0]const u8 = "taludb_generic";
const heap_destructor: sqlite.sqlite3_destructor_type = sqlite.sqlite3_free;

// Filter pushdown bitmask: pk_hash = ? (vtable column 0 → block column 1).
const idx_pk_eq: c_int = 1 << 0;

// =============================================================================
// Structures
// =============================================================================

const ModuleAux = struct {
    db_root: []u8,
};

/// Column mapping: vtable column index → block column_id.
const ColumnMapping = struct {
    column_id: u32,
    is_payload: bool,
};

/// Return type for schema discovery functions.
const SchemaResult = struct {
    columns: []ColumnMapping,
    schema_sql: [:0]u8,
};

const GenericVTab = struct {
    base: sqlite.sqlite3_vtab = std.mem.zeroes(sqlite.sqlite3_vtab),
    db_root: []u8,
    namespace: []u8,
    schema_id: u16,
    delete_schema_id: ?u16,
    columns: []ColumnMapping,
};

const GenericCursor = struct {
    base: sqlite.sqlite3_vtab_cursor = std.mem.zeroes(sqlite.sqlite3_vtab_cursor),
    table: *GenericVTab,
    rows: []generic.Row = &.{},
    row_idx: usize = 0,
    rowid: sqlite.sqlite3_int64 = 0,
    eof: bool = true,

    fn freeRows(self: *GenericCursor) void {
        if (self.rows.len > 0) {
            generic.freeRows(allocator, self.rows);
            self.rows = &.{};
        }
    }
};

// =============================================================================
// Module registration
// =============================================================================

pub fn registerGenericModule(db: *sqlite.sqlite3, db_root: []const u8) !void {
    const aux = try allocator.create(ModuleAux);
    errdefer allocator.destroy(aux);
    aux.db_root = try allocator.dupe(u8, db_root);
    errdefer allocator.free(aux.db_root);

    const rc = sqlite.sqlite3_create_module_v2(
        db,
        module_name.ptr,
        &generic_module,
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

const generic_module: sqlite.sqlite3_module = blk: {
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

// =============================================================================
// Helpers
// =============================================================================

fn tableFromBase(base: *sqlite.sqlite3_vtab) *GenericVTab {
    return @alignCast(@fieldParentPtr("base", @as(*sqlite.sqlite3_vtab, @alignCast(base))));
}

fn cursorFromBase(base: *sqlite.sqlite3_vtab_cursor) *GenericCursor {
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
    const raw = sqlite.sqlite3_malloc64(@intCast(value.len + 1)) orelse {
        sqlite.sqlite3_result_error_nomem(ctx);
        return;
    };
    const copied: [*]u8 = @ptrCast(raw);
    @memcpy(copied[0..value.len], value);
    copied[value.len] = 0;
    sqlite.sqlite3_result_text(ctx, @ptrCast(copied), @intCast(value.len), heap_destructor);
}

fn resultBlob(ctx: ?*sqlite.sqlite3_context, value: []const u8) void {
    if (value.len > std.math.maxInt(c_int)) {
        sqlite.sqlite3_result_error_toobig(ctx);
        return;
    }
    const raw = sqlite.sqlite3_malloc64(@intCast(value.len)) orelse {
        sqlite.sqlite3_result_error_nomem(ctx);
        return;
    };
    const copied: [*]u8 = @ptrCast(raw);
    @memcpy(copied[0..value.len], value);
    sqlite.sqlite3_result_blob(ctx, @ptrCast(copied), @intCast(value.len), heap_destructor);
}

/// Parse a C-string argument, trimming surrounding quotes/whitespace.
fn parseArgStr(raw: [*c]const u8) ?[]const u8 {
    if (raw == null) return null;
    const ptr: [*]const u8 = @ptrCast(raw);
    var len: usize = 0;
    while (ptr[len] != 0) : (len += 1) {}
    var slice = std.mem.trim(u8, ptr[0..len], " \t\r\n");
    // Strip surrounding single or double quotes.
    if (slice.len >= 2) {
        if ((slice[0] == '\'' and slice[slice.len - 1] == '\'') or
            (slice[0] == '"' and slice[slice.len - 1] == '"'))
        {
            slice = slice[1 .. slice.len - 1];
        }
    }
    return if (slice.len > 0) slice else null;
}

fn parseArgU16(raw: [*c]const u8) ?u16 {
    const str = parseArgStr(raw) orelse return null;
    return std.fmt.parseUnsigned(u16, str, 10) catch null;
}

// =============================================================================
// Schema discovery
// =============================================================================

/// Discover columns from the first block matching the target schema_id.
/// Returns the column mappings and a heap-allocated SQL schema declaration.
fn discoverSchema(
    db_root: []const u8,
    namespace: []const u8,
    schema_id: u16,
) !SchemaResult {
    // Open a temporary reader to inspect blocks.
    var reader = db_reader.Reader.open(allocator, db_root, namespace) catch {
        return fallbackSchema();
    };
    defer reader.deinit();

    const blocks = reader.getBlocks(allocator) catch return fallbackSchema();
    defer allocator.free(blocks);

    // Find first block matching schema_id by reading headers.
    for (blocks) |block| {
        var file = reader.dir.openFile(block.path, .{ .mode = .read_only }) catch continue;
        defer file.close();

        const br = block_reader.BlockReader.init(file, allocator);
        const header = br.readHeader(block.offset) catch continue;
        if (header.schema_id != schema_id) continue;

        const descs = br.readColumnDirectory(header, block.offset) catch continue;
        defer allocator.free(descs);

        return buildSchemaFromDescs(descs);
    }

    return fallbackSchema();
}

fn fallbackSchema() !SchemaResult {
    const columns = try allocator.alloc(ColumnMapping, 3);
    columns[0] = .{ .column_id = 1, .is_payload = false };
    columns[1] = .{ .column_id = 2, .is_payload = false };
    columns[2] = .{ .column_id = 20, .is_payload = true };

    const sql = try allocator.dupeZ(u8, "CREATE TABLE x(pk_hash INTEGER, ts INTEGER, payload TEXT)");
    return .{ .columns = columns, .schema_sql = sql };
}

fn buildSchemaFromDescs(descs: []const types.ColumnDesc) !SchemaResult {
    // Sort by column_id for deterministic order.
    const sorted = try allocator.alloc(types.ColumnDesc, descs.len);
    defer allocator.free(sorted);
    @memcpy(sorted, descs);
    std.mem.sort(types.ColumnDesc, sorted, {}, struct {
        fn cmp(_: void, a: types.ColumnDesc, b: types.ColumnDesc) bool {
            return a.column_id < b.column_id;
        }
    }.cmp);

    var columns_list = std.ArrayList(ColumnMapping).empty;
    defer columns_list.deinit(allocator);
    var sql_buf = std.ArrayList(u8).empty;
    defer sql_buf.deinit(allocator);
    const writer = sql_buf.writer(allocator);

    try writer.writeAll("CREATE TABLE x(");

    var first = true;
    for (sorted) |desc| {
        const shape: types.ColumnShape = @enumFromInt(desc.shape);
        if (shape == .VECTOR) continue; // Skip vector columns (no SQL representation).

        const is_payload = (shape == .VARBYTES);

        if (!first) try writer.writeAll(", ");
        first = false;

        // Column name.
        switch (desc.column_id) {
            1 => try writer.writeAll("pk_hash"),
            2 => try writer.writeAll("ts"),
            20 => try writer.writeAll("payload"),
            else => try std.fmt.format(writer, "col_{d}", .{desc.column_id}),
        }

        // Column type.
        if (is_payload) {
            try writer.writeAll(" TEXT");
        } else {
            try writer.writeAll(" INTEGER");
        }

        try columns_list.append(allocator, .{ .column_id = desc.column_id, .is_payload = is_payload });
    }

    if (first) {
        // No columns found — should not happen, but use fallback.
        sql_buf.clearRetainingCapacity();
        columns_list.clearRetainingCapacity();
        return fallbackSchema();
    }

    try writer.writeAll(")");
    try sql_buf.append(allocator, 0); // null-terminate

    const sql_slice = try sql_buf.toOwnedSlice(allocator);
    const schema_sql: [:0]u8 = sql_slice[0 .. sql_slice.len - 1 :0];

    return .{
        .columns = try columns_list.toOwnedSlice(allocator),
        .schema_sql = schema_sql,
    };
}

// =============================================================================
// VTable callbacks
// =============================================================================

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
    argc: c_int,
    argv: [*c]const [*c]const u8,
    pp_vtab: [*c][*c]sqlite.sqlite3_vtab,
    pz_err: [*c][*c]u8,
) callconv(.c) c_int {
    const db_handle = db orelse return sqlite.SQLITE_MISUSE;
    if (pp_vtab == null) return sqlite.SQLITE_MISUSE;
    const aux: *ModuleAux = @ptrCast(@alignCast(p_aux orelse {
        setErrMsg(pz_err, "missing taludb_generic module aux");
        return sqlite.SQLITE_ERROR;
    }));

    const arg_count: usize = if (argc < 0) 0 else @intCast(argc);

    // Parse required args: namespace (argv[3]), schema_id (argv[4]).
    if (arg_count < 5) {
        setErrMsg(pz_err, "taludb_generic requires (namespace, schema_id [, delete_schema_id])");
        return sqlite.SQLITE_ERROR;
    }
    const namespace_str = parseArgStr(argv[3]) orelse {
        setErrMsg(pz_err, "taludb_generic: invalid namespace argument");
        return sqlite.SQLITE_ERROR;
    };
    const schema_id = parseArgU16(argv[4]) orelse {
        setErrMsg(pz_err, "taludb_generic: invalid schema_id argument");
        return sqlite.SQLITE_ERROR;
    };
    const delete_schema_id: ?u16 = if (arg_count > 5) parseArgU16(argv[5]) else null;

    // Discover schema from existing data.
    const discovered = discoverSchema(aux.db_root, namespace_str, schema_id) catch {
        setErrMsg(pz_err, "taludb_generic: schema discovery failed");
        return sqlite.SQLITE_ERROR;
    };
    errdefer {
        allocator.free(discovered.columns);
        allocator.free(discovered.schema_sql);
    }

    const decl_rc = sqlite.sqlite3_declare_vtab(db_handle, discovered.schema_sql.ptr);
    if (decl_rc != sqlite.SQLITE_OK) {
        setErrMsg(pz_err, "taludb_generic: failed to declare schema");
        allocator.free(discovered.columns);
        allocator.free(discovered.schema_sql);
        return decl_rc;
    }
    allocator.free(discovered.schema_sql);

    const table = allocator.create(GenericVTab) catch {
        setErrMsg(pz_err, "out of memory");
        allocator.free(discovered.columns);
        return sqlite.SQLITE_NOMEM;
    };
    errdefer allocator.destroy(table);

    const db_root_copy = allocator.dupe(u8, aux.db_root) catch {
        allocator.free(discovered.columns);
        return sqlite.SQLITE_NOMEM;
    };
    errdefer allocator.free(db_root_copy);

    const ns_copy = allocator.dupe(u8, namespace_str) catch {
        allocator.free(discovered.columns);
        allocator.free(db_root_copy);
        return sqlite.SQLITE_NOMEM;
    };

    table.* = .{
        .base = std.mem.zeroes(sqlite.sqlite3_vtab),
        .db_root = db_root_copy,
        .namespace = ns_copy,
        .schema_id = schema_id,
        .delete_schema_id = delete_schema_id,
        .columns = discovered.columns,
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
            var arg_i: c_int = 1;
            for (0..n) |ci| {
                const constraint = info.aConstraint[ci];
                if (constraint.usable == 0) continue;
                // pk_hash = ? (vtable column 0 → block column 1)
                if (constraint.iColumn == 0 and constraint.op == sqlite.SQLITE_INDEX_CONSTRAINT_EQ) {
                    info.idxNum |= idx_pk_eq;
                    info.aConstraintUsage[ci].argvIndex = arg_i;
                    info.aConstraintUsage[ci].omit = 1;
                    arg_i += 1;
                    info.estimatedCost = 10.0;
                    info.estimatedRows = 1;
                }
            }
        }
    }
    return sqlite.SQLITE_OK;
}

fn xDisconnect(p_vtab: ?*sqlite.sqlite3_vtab) callconv(.c) c_int {
    const base = p_vtab orelse return sqlite.SQLITE_MISUSE;
    const table = tableFromBase(base);
    allocator.free(table.db_root);
    allocator.free(table.namespace);
    allocator.free(table.columns);
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

    const cursor = allocator.create(GenericCursor) catch return sqlite.SQLITE_NOMEM;
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
    cursor.freeRows();
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
    cursor.freeRows();
    cursor.row_idx = 0;
    cursor.rowid = 0;

    const table = cursor.table;
    const arg_count: usize = if (argc <= 0) 0 else @intCast(argc);

    // Build filters from constraints.
    var filters_buf: [1]generic.ColumnFilter = undefined;
    var filter_count: usize = 0;

    var arg_i: usize = 0;
    if ((idx_num & idx_pk_eq) != 0) {
        if (arg_i >= arg_count) return sqlite.SQLITE_MISUSE;
        const value = argv[arg_i] orelse return sqlite.SQLITE_MISUSE;
        arg_i += 1;
        const pk_hash: u64 = @bitCast(sqlite.sqlite3_value_int64(value));
        filters_buf[filter_count] = .{ .column_id = 1, .op = .eq, .value = pk_hash };
        filter_count += 1;
    }

    // Build extra_columns list for all non-standard scalar columns.
    var extra_cols_buf: [16]u32 = undefined;
    var extra_count: usize = 0;
    for (table.columns) |col| {
        if (col.is_payload) continue;
        if (col.column_id == 1 or col.column_id == 2) continue;
        if (extra_count < extra_cols_buf.len) {
            extra_cols_buf[extra_count] = col.column_id;
            extra_count += 1;
        }
    }

    // Open a read-only Table, run scan, close immediately.
    var tbl = generic.Table.openReadOnly(
        allocator,
        table.db_root,
        table.namespace,
        .{ .active_schema_ids = &.{} },
    ) catch return sqlite.SQLITE_ERROR;
    defer tbl.deinit();

    const result = tbl.scan(allocator, .{
        .schema_id = table.schema_id,
        .delete_schema_id = table.delete_schema_id,
        .filters = filters_buf[0..filter_count],
        .extra_columns = extra_cols_buf[0..extra_count],
    }) catch return sqlite.SQLITE_ERROR;

    cursor.rows = result.rows;
    cursor.eof = (result.rows.len == 0);
    return sqlite.SQLITE_OK;
}

fn xNext(p_cursor: ?*sqlite.sqlite3_vtab_cursor) callconv(.c) c_int {
    const base = p_cursor orelse return sqlite.SQLITE_MISUSE;
    const cursor = cursorFromBase(base);
    cursor.row_idx += 1;
    cursor.rowid += 1;
    if (cursor.row_idx >= cursor.rows.len) {
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
    if (cursor.eof or cursor.row_idx >= cursor.rows.len) {
        sqlite.sqlite3_result_null(p_ctx);
        return sqlite.SQLITE_OK;
    }

    const col_idx: usize = if (col < 0) return sqlite.SQLITE_MISUSE else @intCast(col);
    if (col_idx >= cursor.table.columns.len) {
        sqlite.sqlite3_result_null(p_ctx);
        return sqlite.SQLITE_OK;
    }

    const mapping = cursor.table.columns[col_idx];
    const row = cursor.rows[cursor.row_idx];

    if (mapping.is_payload) {
        // Return payload as text.
        resultText(p_ctx, row.payload);
        return sqlite.SQLITE_OK;
    }

    // Find the scalar value for this column_id.
    for (row.scalars) |scalar| {
        if (scalar.column_id == mapping.column_id) {
            sqlite.sqlite3_result_int64(p_ctx, @bitCast(scalar.value_u64));
            return sqlite.SQLITE_OK;
        }
    }

    sqlite.sqlite3_result_null(p_ctx);
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
        vtab.zErrMsg = sqlite.sqlite3_mprintf("taludb_generic is read-only");
    }
    return sqlite.SQLITE_READONLY;
}

fn xBegin(_: ?*sqlite.sqlite3_vtab) callconv(.c) c_int { return sqlite.SQLITE_OK; }
fn xSync(_: ?*sqlite.sqlite3_vtab) callconv(.c) c_int { return sqlite.SQLITE_OK; }
fn xCommit(_: ?*sqlite.sqlite3_vtab) callconv(.c) c_int { return sqlite.SQLITE_OK; }
fn xRollback(_: ?*sqlite.sqlite3_vtab) callconv(.c) c_int { return sqlite.SQLITE_OK; }
fn xFindFunction(_: ?*sqlite.sqlite3_vtab, _: c_int, _: [*c]const u8, _: [*c]?*const fn (?*sqlite.sqlite3_context, c_int, [*c]?*sqlite.sqlite3_value) callconv(.c) void, _: [*c]?*anyopaque) callconv(.c) c_int { return 0; }
fn xRename(_: ?*sqlite.sqlite3_vtab, _: [*c]const u8) callconv(.c) c_int { return sqlite.SQLITE_OK; }
fn xSavepoint(_: ?*sqlite.sqlite3_vtab, _: c_int) callconv(.c) c_int { return sqlite.SQLITE_OK; }
fn xRelease(_: ?*sqlite.sqlite3_vtab, _: c_int) callconv(.c) c_int { return sqlite.SQLITE_OK; }
fn xRollbackTo(_: ?*sqlite.sqlite3_vtab, _: c_int) callconv(.c) c_int { return sqlite.SQLITE_OK; }

// =============================================================================
// Tests
// =============================================================================

test "registerGenericModule creates taludb_generic virtual table" {
    var db: ?*sqlite.sqlite3 = null;
    try std.testing.expectEqual(@as(c_int, sqlite.SQLITE_OK), sqlite.sqlite3_open(":memory:", &db));
    defer _ = sqlite.sqlite3_close(db);

    try registerGenericModule(db.?, "/tmp/talu-generic-test");

    var err_msg: [*c]u8 = null;
    defer if (err_msg != null) sqlite.sqlite3_free(err_msg);

    const create_sql: [:0]const u8 = "CREATE VIRTUAL TABLE t USING taludb_generic('docs', 11)";
    try std.testing.expectEqual(
        @as(c_int, sqlite.SQLITE_OK),
        sqlite.sqlite3_exec(db, create_sql.ptr, null, null, &err_msg),
    );

    const scan_sql: [:0]const u8 = "SELECT count(*) FROM t";
    try std.testing.expectEqual(
        @as(c_int, sqlite.SQLITE_OK),
        sqlite.sqlite3_exec(db, scan_sql.ptr, null, null, &err_msg),
    );
}

test "taludb_generic scan returns deduped rows from generic table" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    // Write test data through Table.
    const db_writer = @import("../writer.zig");
    const ColumnValue = db_writer.ColumnValue;

    var tbl = try generic.Table.open(std.testing.allocator, root, "test_ns", .{
        .active_schema_ids = &.{},
    });

    // Row 1: pk=100, ts=1000
    var pk1: u64 = 100;
    var ts1: i64 = 1000;
    const payload1 = "hello";
    try tbl.appendRow(42, &[_]ColumnValue{
        .{ .column_id = 1, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&pk1) },
        .{ .column_id = 2, .shape = .SCALAR, .phys_type = .I64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&ts1) },
        .{ .column_id = 20, .shape = .VARBYTES, .phys_type = .BINARY, .encoding = .RAW, .dims = 0, .data = payload1 },
    });

    // Row 2: same pk=100, newer ts=2000 (should dedup with row 1)
    var ts2: i64 = 2000;
    const payload2 = "world";
    try tbl.appendRow(42, &[_]ColumnValue{
        .{ .column_id = 1, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&pk1) },
        .{ .column_id = 2, .shape = .SCALAR, .phys_type = .I64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&ts2) },
        .{ .column_id = 20, .shape = .VARBYTES, .phys_type = .BINARY, .encoding = .RAW, .dims = 0, .data = payload2 },
    });

    // Row 3: different pk=200, ts=1500
    var pk3: u64 = 200;
    var ts3: i64 = 1500;
    const payload3 = "other";
    try tbl.appendRow(42, &[_]ColumnValue{
        .{ .column_id = 1, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&pk3) },
        .{ .column_id = 2, .shape = .SCALAR, .phys_type = .I64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&ts3) },
        .{ .column_id = 20, .shape = .VARBYTES, .phys_type = .BINARY, .encoding = .RAW, .dims = 0, .data = payload3 },
    });

    tbl.deinit();

    // Query via vtable.
    var db: ?*sqlite.sqlite3 = null;
    try std.testing.expectEqual(@as(c_int, sqlite.SQLITE_OK), sqlite.sqlite3_open(":memory:", &db));
    defer _ = sqlite.sqlite3_close(db);

    try registerGenericModule(db.?, root);

    var err_msg: [*c]u8 = null;
    defer if (err_msg != null) sqlite.sqlite3_free(err_msg);

    const create_sql: [:0]const u8 = "CREATE VIRTUAL TABLE t USING taludb_generic('test_ns', 42)";
    try std.testing.expectEqual(
        @as(c_int, sqlite.SQLITE_OK),
        sqlite.sqlite3_exec(db, create_sql.ptr, null, null, &err_msg),
    );

    var stmt: ?*sqlite.sqlite3_stmt = null;
    const query: [:0]const u8 = "SELECT pk_hash, ts, payload FROM t ORDER BY pk_hash";
    try std.testing.expectEqual(
        @as(c_int, sqlite.SQLITE_OK),
        sqlite.sqlite3_prepare_v2(db, query.ptr, -1, &stmt, null),
    );
    defer _ = sqlite.sqlite3_finalize(stmt);

    // Should get 2 rows (deduped): pk=100 with ts=2000, pk=200 with ts=1500.
    var row_count: usize = 0;
    while (sqlite.sqlite3_step(stmt) == sqlite.SQLITE_ROW) {
        row_count += 1;
        const pk = sqlite.sqlite3_column_int64(stmt, 0);
        const ts = sqlite.sqlite3_column_int64(stmt, 1);

        if (row_count == 1) {
            try std.testing.expectEqual(@as(i64, 100), pk);
            try std.testing.expectEqual(@as(i64, 2000), ts);
        } else if (row_count == 2) {
            try std.testing.expectEqual(@as(i64, 200), pk);
            try std.testing.expectEqual(@as(i64, 1500), ts);
        }
    }
    try std.testing.expectEqual(@as(usize, 2), row_count);
}
