//! SQLite compute engine over TaluDB virtual tables.
//!
//! Provides stateless SQL query execution over TaluDB data via in-memory
//! SQLite connections with virtual table modules. Supports parameterized
//! queries with typed binary parameters (no JSON at this layer).

const std = @import("std");
const sqlite = @import("c.zig").sqlite3;
const vtable_docs = @import("vtable_docs.zig");
const vtable_generic = @import("vtable_generic.zig");
const vtable_vector = @import("vtable_vector.zig");

const Allocator = std.mem.Allocator;

/// Typed SQL parameter for binding to prepared statements.
/// Matches CSqlParam layout across the FFI boundary.
pub const SqlParam = struct {
    tag: Tag,
    int_val: i64,
    float_val: f64,
    ptr: ?[*]const u8,
    len: usize,

    pub const Tag = enum(u8) {
        null_val = 0,
        int64 = 1,
        float64 = 2,
        text = 3,
        blob = 4,
    };

    pub fn initNull() SqlParam {
        return .{ .tag = .null_val, .int_val = 0, .float_val = 0, .ptr = null, .len = 0 };
    }

    pub fn initInt(val: i64) SqlParam {
        return .{ .tag = .int64, .int_val = val, .float_val = 0, .ptr = null, .len = 0 };
    }

    pub fn initFloat(val: f64) SqlParam {
        return .{ .tag = .float64, .int_val = 0, .float_val = val, .ptr = null, .len = 0 };
    }

    pub fn initText(text: []const u8) SqlParam {
        return .{ .tag = .text, .int_val = 0, .float_val = 0, .ptr = text.ptr, .len = text.len };
    }

    pub fn initBlob(data: []const u8) SqlParam {
        return .{ .tag = .blob, .int_val = 0, .float_val = 0, .ptr = data.ptr, .len = data.len };
    }
};

/// Open an in-memory SQLite connection with TaluDB virtual table modules registered.
fn openConnection(db_root: []const u8) !*sqlite.sqlite3 {
    var db: ?*sqlite.sqlite3 = null;
    if (sqlite.sqlite3_open(":memory:", &db) != sqlite.SQLITE_OK) {
        return error.SqliteOpenFailed;
    }
    errdefer _ = sqlite.sqlite3_close(db);

    try vtable_docs.registerDocsModule(db.?, db_root);
    try vtable_generic.registerGenericModule(db.?, db_root);
    try vtable_vector.registerVectorModule(db.?, db_root);

    var err_msg: [*c]u8 = null;
    defer if (err_msg != null) sqlite.sqlite3_free(err_msg);

    const create_docs: [:0]const u8 = "CREATE VIRTUAL TABLE docs USING taludb_docs";
    if (sqlite.sqlite3_exec(db, create_docs.ptr, null, null, &err_msg) != sqlite.SQLITE_OK) {
        return error.SqliteQueryFailed;
    }

    return db.?;
}

/// Bind a SqlParam array to a prepared statement.
fn bindParams(stmt: *sqlite.sqlite3_stmt, params: []const SqlParam) !void {
    for (params, 0..) |p, i| {
        const idx: c_int = @intCast(i + 1); // SQLite params are 1-indexed
        const rc: c_int = switch (p.tag) {
            .null_val => sqlite.sqlite3_bind_null(stmt, idx),
            .int64 => sqlite.sqlite3_bind_int64(stmt, idx, p.int_val),
            .float64 => sqlite.sqlite3_bind_double(stmt, idx, p.float_val),
            .text => sqlite.sqlite3_bind_text(
                stmt,
                idx,
                p.ptr,
                @intCast(p.len),
                sqlite.SQLITE_TRANSIENT,
            ),
            .blob => sqlite.sqlite3_bind_blob(
                stmt,
                idx,
                p.ptr,
                @intCast(p.len),
                sqlite.SQLITE_TRANSIENT,
            ),
        };
        if (rc != sqlite.SQLITE_OK) return error.SqliteBindFailed;
    }
}

/// Write a single result row as a JSON object into the output buffer.
fn writeRow(alloc: Allocator, out: *std.ArrayList(u8), stmt: *sqlite.sqlite3_stmt, column_count: usize) !void {
    try out.append(alloc, '{');
    for (0..column_count) |col_idx| {
        if (col_idx > 0) try out.append(alloc, ',');
        const c_col_idx: c_int = @intCast(col_idx);

        const name_ptr = sqlite.sqlite3_column_name(stmt, c_col_idx);
        const name = if (name_ptr != null) std.mem.span(name_ptr) else "";
        try out.writer(alloc).print("{f}", .{std.json.fmt(name, .{})});
        try out.append(alloc, ':');

        const col_type = sqlite.sqlite3_column_type(stmt, c_col_idx);
        switch (col_type) {
            sqlite.SQLITE_NULL => try out.appendSlice(alloc, "null"),
            sqlite.SQLITE_INTEGER => {
                try out.writer(alloc).print("{d}", .{sqlite.sqlite3_column_int64(stmt, c_col_idx)});
            },
            sqlite.SQLITE_FLOAT => {
                try out.writer(alloc).print("{d}", .{sqlite.sqlite3_column_double(stmt, c_col_idx)});
            },
            sqlite.SQLITE_TEXT => {
                const text_ptr = sqlite.sqlite3_column_text(stmt, c_col_idx) orelse {
                    try out.appendSlice(alloc, "null");
                    continue;
                };
                const text_len: usize = @intCast(sqlite.sqlite3_column_bytes(stmt, c_col_idx));
                const text = @as([*]const u8, @ptrCast(text_ptr))[0..text_len];
                try out.writer(alloc).print("{f}", .{std.json.fmt(text, .{})});
            },
            sqlite.SQLITE_BLOB => {
                const blob_ptr = sqlite.sqlite3_column_blob(stmt, c_col_idx) orelse {
                    try out.appendSlice(alloc, "null");
                    continue;
                };
                const blob_len: usize = @intCast(sqlite.sqlite3_column_bytes(stmt, c_col_idx));
                const blob = @as([*]const u8, @ptrCast(blob_ptr))[0..blob_len];
                const encoded_len = std.base64.standard.Encoder.calcSize(blob.len);
                const encoded = try alloc.alloc(u8, encoded_len);
                defer alloc.free(encoded);
                _ = std.base64.standard.Encoder.encode(encoded, blob);
                try out.writer(alloc).print("{f}", .{std.json.fmt(encoded, .{})});
            },
            else => try out.appendSlice(alloc, "null"),
        }
    }
    try out.append(alloc, '}');
}

/// Collect column names from a prepared statement as a JSON array.
fn writeColumnNames(alloc: Allocator, out: *std.ArrayList(u8), stmt: *sqlite.sqlite3_stmt, column_count: usize) !void {
    try out.append(alloc, '[');
    for (0..column_count) |col_idx| {
        if (col_idx > 0) try out.append(alloc, ',');
        const c_col_idx: c_int = @intCast(col_idx);
        const name_ptr = sqlite.sqlite3_column_name(stmt, c_col_idx);
        const name = if (name_ptr != null) std.mem.span(name_ptr) else "";
        try out.writer(alloc).print("{f}", .{std.json.fmt(name, .{})});
    }
    try out.append(alloc, ']');
}

/// Execute a SQL query with optional typed parameters.
/// Returns structured JSON: {"columns":[...],"rows":[...],"row_count":N}
pub fn queryJsonParams(alloc: Allocator, db_root: []const u8, query: []const u8, params: []const SqlParam) ![]u8 {
    const db = try openConnection(db_root);
    defer _ = sqlite.sqlite3_close(db);

    var stmt: ?*sqlite.sqlite3_stmt = null;
    const prepare_rc = sqlite.sqlite3_prepare_v2(db, query.ptr, @intCast(query.len), &stmt, null);
    if (prepare_rc != sqlite.SQLITE_OK) return error.SqliteQueryFailed;
    defer _ = sqlite.sqlite3_finalize(stmt);

    if (params.len > 0) {
        try bindParams(stmt.?, params);
    }

    const column_count: usize = @intCast(sqlite.sqlite3_column_count(stmt));
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(alloc);

    // Build structured response: {"columns":[...],"rows":[...],"row_count":N}
    try out.appendSlice(alloc, "{\"columns\":");
    writeColumnNames(alloc, &out, stmt.?, column_count) catch |err| {
        // Column names are available before stepping — this should not fail
        return err;
    };
    try out.appendSlice(alloc, ",\"rows\":[");

    var row_count: usize = 0;
    while (true) {
        const rc = sqlite.sqlite3_step(stmt);
        if (rc == sqlite.SQLITE_ROW) {
            if (row_count > 0) try out.append(alloc, ',');
            row_count += 1;
            try writeRow(alloc, &out, stmt.?, column_count);
            continue;
        }
        if (rc == sqlite.SQLITE_DONE) break;
        return error.SqliteQueryFailed;
    }

    try out.writer(alloc).print("],\"row_count\":{d}}}", .{row_count});
    return out.toOwnedSlice(alloc);
}

/// Execute a SQL query without parameters. Returns raw JSON array (backward compat).
/// Caller owns returned slice.
pub fn queryJson(alloc: Allocator, db_root: []const u8, query: []const u8) ![]u8 {
    const db = try openConnection(db_root);
    defer _ = sqlite.sqlite3_close(db);

    var stmt: ?*sqlite.sqlite3_stmt = null;
    const prepare_rc = sqlite.sqlite3_prepare_v2(db, query.ptr, @intCast(query.len), &stmt, null);
    if (prepare_rc != sqlite.SQLITE_OK) return error.SqliteQueryFailed;
    defer _ = sqlite.sqlite3_finalize(stmt);

    const column_count: usize = @intCast(sqlite.sqlite3_column_count(stmt));
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(alloc);

    try out.append(alloc, '[');
    var row_count: usize = 0;

    while (true) {
        const rc = sqlite.sqlite3_step(stmt);
        if (rc == sqlite.SQLITE_ROW) {
            if (row_count > 0) try out.append(alloc, ',');
            row_count += 1;
            try writeRow(alloc, &out, stmt.?, column_count);
            continue;
        }
        if (rc == sqlite.SQLITE_DONE) break;
        return error.SqliteQueryFailed;
    }

    try out.append(alloc, ']');
    return out.toOwnedSlice(alloc);
}

// =============================================================================
// Tests
// =============================================================================

test "queryJsonParams binds integer parameter and filters docs" {
    const documents = @import("../table/documents.zig");

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    var adapter = try documents.DocumentAdapter.init(std.testing.allocator, root);
    defer adapter.deinit();
    try adapter.writeDocument(.{
        .doc_id = "p1",
        .doc_type = "note",
        .title = "Alpha",
        .doc_json = "{\"_sys\":{},\"data\":{}}",
        .created_at_ms = 100,
        .updated_at_ms = 100,
        .group_id = "grp-a",
    });
    try adapter.writeDocument(.{
        .doc_id = "p2",
        .doc_type = "note",
        .title = "Beta",
        .doc_json = "{\"_sys\":{},\"data\":{}}",
        .created_at_ms = 200,
        .updated_at_ms = 200,
        .group_id = "grp-b",
    });
    try adapter.flush();

    const sql = "SELECT doc_id FROM docs WHERE group_id = ?";
    const params = [_]SqlParam{SqlParam.initText("grp-a")};
    const result = try queryJsonParams(std.testing.allocator, root, sql, &params);
    defer std.testing.allocator.free(result);

    // Structured response format
    try std.testing.expect(std.mem.indexOf(u8, result, "\"columns\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"rows\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"row_count\"") != null);
    // Contains the filtered doc
    try std.testing.expect(std.mem.indexOf(u8, result, "p1") != null);
    // Does not contain the excluded doc
    try std.testing.expect(std.mem.indexOf(u8, result, "p2") == null);
}

test "queryJsonParams with no params returns structured response" {
    const documents = @import("../table/documents.zig");

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    var adapter = try documents.DocumentAdapter.init(std.testing.allocator, root);
    defer adapter.deinit();
    try adapter.writeDocument(.{
        .doc_id = "d1",
        .doc_type = "memo",
        .title = "T",
        .doc_json = "{\"_sys\":{},\"data\":{}}",
        .created_at_ms = 1,
        .updated_at_ms = 1,
        .group_id = "g",
    });
    try adapter.flush();

    const sql = "SELECT doc_id FROM docs";
    const result = try queryJsonParams(std.testing.allocator, root, sql, &.{});
    defer std.testing.allocator.free(result);

    try std.testing.expect(std.mem.indexOf(u8, result, "\"row_count\":1") != null);
}

test "queryJsonParams binds null, float, and blob params" {
    const documents = @import("../table/documents.zig");

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    var adapter = try documents.DocumentAdapter.init(std.testing.allocator, root);
    defer adapter.deinit();
    try adapter.writeDocument(.{
        .doc_id = "x1",
        .doc_type = "t",
        .title = "T",
        .doc_json = "{\"_sys\":{},\"data\":{}}",
        .created_at_ms = 1,
        .updated_at_ms = 1,
        .group_id = "g",
    });
    try adapter.flush();

    // Use params of various types — null, float, blob — in a query that still works
    // The query itself just selects all docs; the params are bound but unused by the WHERE
    // because we want to verify binding doesn't crash, not filter logic.
    const sql = "SELECT doc_id FROM docs WHERE 1=1 AND ?1 IS NULL AND typeof(?2) = 'real' AND typeof(?3) = 'blob'";
    const blob_data = [_]u8{ 0x01, 0x02, 0x03 };
    const params = [_]SqlParam{
        SqlParam.initNull(),
        SqlParam.initFloat(3.14),
        SqlParam.initBlob(&blob_data),
    };
    const result = try queryJsonParams(std.testing.allocator, root, sql, &params);
    defer std.testing.allocator.free(result);

    try std.testing.expect(std.mem.indexOf(u8, result, "x1") != null);
}

test "queryJson returns raw JSON array (backward compat)" {
    const documents = @import("../table/documents.zig");

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    var adapter = try documents.DocumentAdapter.init(std.testing.allocator, root);
    defer adapter.deinit();
    try adapter.writeDocument(.{
        .doc_id = "bc1",
        .doc_type = "t",
        .title = "T",
        .doc_json = "{\"_sys\":{},\"data\":{}}",
        .created_at_ms = 1,
        .updated_at_ms = 1,
        .group_id = "g",
    });
    try adapter.flush();

    const result = try queryJson(std.testing.allocator, root, "SELECT doc_id FROM docs");
    defer std.testing.allocator.free(result);

    // Raw array format, not structured
    try std.testing.expect(result[0] == '[');
    try std.testing.expect(std.mem.indexOf(u8, result, "bc1") != null);
}
