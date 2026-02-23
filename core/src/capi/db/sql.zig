//! Plane-specific DB C-API: SQL compute.

const sql_api = @import("../sql.zig");

pub export fn talu_db_sql_query(
    db_path: ?[*:0]const u8,
    query: ?[*:0]const u8,
    out_json: ?*?[*:0]u8,
) callconv(.c) i32 {
    return sql_api.talu_sql_query(db_path, query, out_json);
}

pub export fn talu_db_sql_query_free(ptr: ?[*:0]u8) callconv(.c) void {
    sql_api.talu_sql_query_free(ptr);
}

/// Scaffold for upcoming structured SQL execution API.
pub export fn talu_db_sql_execute(
    db_path: ?[*:0]const u8,
    query: ?[*:0]const u8,
    out_json: ?*?[*:0]u8,
) callconv(.c) i32 {
    return sql_api.talu_sql_query(db_path, query, out_json);
}
