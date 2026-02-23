//! C-API SQL module.
//!
//! Exposes compute-only SQLite queries over TaluDB virtual tables.

const std = @import("std");
const capi_error = @import("error.zig");
const error_codes = @import("error_codes.zig");
const sql_engine = @import("../db/sql/engine.zig");

const allocator = std.heap.c_allocator;

fn validateRequiredArg(value: ?[*:0]const u8, comptime arg_name: []const u8) ?[]const u8 {
    const slice = std.mem.sliceTo(value orelse {
        capi_error.setErrorWithCode(.invalid_argument, arg_name ++ " is null", .{});
        return null;
    }, 0);
    if (slice.len == 0) {
        capi_error.setErrorWithCode(.invalid_argument, arg_name ++ " is empty", .{});
        return null;
    }
    return slice;
}

/// Execute a SQL query against TaluDB virtual tables and return JSON rows.
///
/// Caller owns `out_json` and must free it with `talu_sql_query_free`.
pub export fn talu_sql_query(
    db_path: ?[*:0]const u8,
    query: ?[*:0]const u8,
    out_json: ?*?[*:0]u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_json orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_json is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;

    const db_path_slice = validateRequiredArg(db_path, "db_path") orelse {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const query_slice = validateRequiredArg(query, "query") orelse {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const json = sql_engine.queryJson(allocator, db_path_slice, query_slice) catch |err| {
        capi_error.setError(err, "failed to execute SQL query", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer allocator.free(json);

    const cstr = allocator.allocSentinel(u8, json.len, 0) catch {
        capi_error.setErrorWithCode(.out_of_memory, "failed to allocate SQL query result", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };
    @memcpy(cstr, json);
    out.* = cstr.ptr;
    return @intFromEnum(error_codes.ErrorCode.ok);
}

/// Free result returned by `talu_sql_query`.
pub export fn talu_sql_query_free(ptr: ?[*:0]u8) callconv(.c) void {
    capi_error.clearError();
    const cstr = ptr orelse return;
    allocator.free(std.mem.span(cstr));
}

test "talu_sql_query executes docs query and returns json" {
    const documents = @import("../db/table/documents.zig");

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    var adapter = try documents.DocumentAdapter.init(std.testing.allocator, root);
    defer adapter.deinit();
    try adapter.writeDocument(.{
        .doc_id = "doc-capi-1",
        .doc_type = "prompt",
        .title = "CAPI SQL",
        .doc_json = "{\"_sys\":{},\"data\":{\"ok\":true}}",
        .created_at_ms = 123,
        .updated_at_ms = 123,
        .group_id = "g-capi",
    });
    try adapter.flush();

    const query = "SELECT doc_id FROM docs WHERE group_id = 'g-capi'";

    const root_z = try std.testing.allocator.allocSentinel(u8, root.len, 0);
    defer std.testing.allocator.free(root_z);
    @memcpy(root_z, root);

    const query_z = try std.testing.allocator.allocSentinel(u8, query.len, 0);
    defer std.testing.allocator.free(query_z);
    @memcpy(query_z, query);

    var out_json: ?[*:0]u8 = null;
    try std.testing.expectEqual(@as(i32, 0), talu_sql_query(root_z.ptr, query_z.ptr, &out_json));
    defer talu_sql_query_free(out_json);

    const json_slice = std.mem.span(out_json.?);
    try std.testing.expect(std.mem.indexOf(u8, json_slice, "doc-capi-1") != null);
}
