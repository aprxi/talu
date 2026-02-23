//! SQLite compute engine over TaluDB virtual tables.

const std = @import("std");
const sqlite = @import("c.zig").sqlite3;
const vtable_docs = @import("vtable_docs.zig");

const Allocator = std.mem.Allocator;

pub fn queryJson(allocator: Allocator, db_root: []const u8, query: []const u8) ![]u8 {
    var db: ?*sqlite.sqlite3 = null;
    if (sqlite.sqlite3_open(":memory:", &db) != sqlite.SQLITE_OK) {
        return error.SqliteOpenFailed;
    }
    defer _ = sqlite.sqlite3_close(db);

    try vtable_docs.registerDocsModule(db.?, db_root);

    var err_msg: [*c]u8 = null;
    defer if (err_msg != null) sqlite.sqlite3_free(err_msg);

    const create_sql: [:0]const u8 = "CREATE VIRTUAL TABLE docs USING taludb_docs";
    const create_rc = sqlite.sqlite3_exec(db, create_sql.ptr, null, null, &err_msg);
    if (create_rc != sqlite.SQLITE_OK) return error.SqliteQueryFailed;

    var stmt: ?*sqlite.sqlite3_stmt = null;
    const prepare_rc = sqlite.sqlite3_prepare_v2(db, query.ptr, @intCast(query.len), &stmt, null);
    if (prepare_rc != sqlite.SQLITE_OK) return error.SqliteQueryFailed;
    defer _ = sqlite.sqlite3_finalize(stmt);

    const column_count: usize = @intCast(sqlite.sqlite3_column_count(stmt));
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);

    try out.append(allocator, '[');
    var row_count: usize = 0;

    while (true) {
        const rc = sqlite.sqlite3_step(stmt);
        if (rc == sqlite.SQLITE_ROW) {
            if (row_count > 0) try out.append(allocator, ',');
            row_count += 1;
            try out.append(allocator, '{');

            for (0..column_count) |col_idx| {
                if (col_idx > 0) try out.append(allocator, ',');
                const c_col_idx: c_int = @intCast(col_idx);

                const name_ptr = sqlite.sqlite3_column_name(stmt, c_col_idx);
                const name = if (name_ptr != null) std.mem.span(name_ptr) else "";
                try out.writer(allocator).print("{f}", .{std.json.fmt(name, .{})});
                try out.append(allocator, ':');

                const col_type = sqlite.sqlite3_column_type(stmt, c_col_idx);
                switch (col_type) {
                    sqlite.SQLITE_NULL => try out.appendSlice(allocator, "null"),
                    sqlite.SQLITE_INTEGER => {
                        try out.writer(allocator).print("{d}", .{sqlite.sqlite3_column_int64(stmt, c_col_idx)});
                    },
                    sqlite.SQLITE_FLOAT => {
                        try out.writer(allocator).print("{d}", .{sqlite.sqlite3_column_double(stmt, c_col_idx)});
                    },
                    sqlite.SQLITE_TEXT => {
                        const text_ptr = sqlite.sqlite3_column_text(stmt, c_col_idx) orelse {
                            try out.appendSlice(allocator, "null");
                            continue;
                        };
                        const text_len: usize = @intCast(sqlite.sqlite3_column_bytes(stmt, c_col_idx));
                        const text = @as([*]const u8, @ptrCast(text_ptr))[0..text_len];
                        try out.writer(allocator).print("{f}", .{std.json.fmt(text, .{})});
                    },
                    sqlite.SQLITE_BLOB => {
                        const blob_ptr = sqlite.sqlite3_column_blob(stmt, c_col_idx) orelse {
                            try out.appendSlice(allocator, "null");
                            continue;
                        };
                        const blob_len: usize = @intCast(sqlite.sqlite3_column_bytes(stmt, c_col_idx));
                        const blob = @as([*]const u8, @ptrCast(blob_ptr))[0..blob_len];
                        const encoded_len = std.base64.standard.Encoder.calcSize(blob.len);
                        const encoded = try allocator.alloc(u8, encoded_len);
                        defer allocator.free(encoded);
                        _ = std.base64.standard.Encoder.encode(encoded, blob);
                        try out.writer(allocator).print("{f}", .{std.json.fmt(encoded, .{})});
                    },
                    else => try out.appendSlice(allocator, "null"),
                }
            }

            try out.append(allocator, '}');
            continue;
        }

        if (rc == sqlite.SQLITE_DONE) break;
        return error.SqliteQueryFailed;
    }

    try out.append(allocator, ']');
    return out.toOwnedSlice(allocator);
}
