//! C-API SQL module.
//!
//! Exposes compute-only SQLite queries over TaluDB virtual tables.
//! Supports parameterized queries via typed binary CSqlParam arrays.

const std = @import("std");
const capi_error = @import("error.zig");
const error_codes = @import("error_codes.zig");
const sql_engine = @import("../db/sql/engine.zig");

const alloc = std.heap.c_allocator;

/// Typed SQL parameter for FFI boundary.
/// Layout must match Rust `SqlParam` repr(C) struct exactly.
pub const CSqlParam = extern struct {
    tag: u8, // 0=null, 1=i64, 2=f64, 3=text, 4=blob
    int_val: i64,
    float_val: f64,
    ptr: ?[*]const u8,
    len: usize,
};

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

/// Convert a C param array into engine SqlParam slice (stack/arena allocated).
fn convertParams(c_params: [*]const CSqlParam, num_params: u32) ![]const sql_engine.SqlParam {
    if (num_params == 0) return &.{};

    const params = try alloc.alloc(sql_engine.SqlParam, num_params);
    for (0..num_params) |i| {
        const cp = c_params[i];
        params[i] = switch (cp.tag) {
            0 => sql_engine.SqlParam.initNull(),
            1 => sql_engine.SqlParam.initInt(cp.int_val),
            2 => sql_engine.SqlParam.initFloat(cp.float_val),
            3 => blk: {
                const p = cp.ptr orelse {
                    alloc.free(params);
                    return error.InvalidArgument;
                };
                break :blk sql_engine.SqlParam.initText(p[0..cp.len]);
            },
            4 => blk: {
                const p = cp.ptr orelse {
                    alloc.free(params);
                    return error.InvalidArgument;
                };
                break :blk sql_engine.SqlParam.initBlob(p[0..cp.len]);
            },
            else => {
                alloc.free(params);
                return error.InvalidArgument;
            },
        };
    }
    return params;
}

/// Copy engine result into a NUL-terminated C string.
fn resultToCString(json: []const u8) !?[*:0]u8 {
    const cstr = try alloc.allocSentinel(u8, json.len, 0);
    @memcpy(cstr, json);
    return cstr.ptr;
}

/// Execute a SQL query against TaluDB virtual tables and return JSON rows.
///
/// Returns raw JSON array. Caller owns `out_json` and must free with `talu_sql_query_free`.
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

    const json = sql_engine.queryJson(alloc, db_path_slice, query_slice) catch |err| {
        capi_error.setError(err, "failed to execute SQL query", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer alloc.free(json);

    out.* = resultToCString(json) catch {
        capi_error.setErrorWithCode(.out_of_memory, "failed to allocate SQL query result", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };
    return @intFromEnum(error_codes.ErrorCode.ok);
}

/// Execute a parameterized SQL query. Returns structured JSON.
///
/// Parameters are passed as a typed binary array — no JSON at this boundary.
/// Returns {"columns":[...],"rows":[...],"row_count":N}.
/// Caller owns `out_json` and must free with `talu_sql_query_free`.
pub export fn talu_sql_query_params(
    db_path: ?[*:0]const u8,
    query: ?[*:0]const u8,
    params: ?[*]const CSqlParam,
    num_params: u32,
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

    // Convert C params to engine params.
    const engine_params = convertParams(params orelse undefined, num_params) catch |err| {
        capi_error.setError(err, "invalid SQL parameter", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer if (engine_params.len > 0) alloc.free(engine_params);

    const json = sql_engine.queryJsonParams(alloc, db_path_slice, query_slice, engine_params) catch |err| {
        capi_error.setError(err, "failed to execute parameterized SQL query", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer alloc.free(json);

    out.* = resultToCString(json) catch {
        capi_error.setErrorWithCode(.out_of_memory, "failed to allocate SQL query result", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };
    return @intFromEnum(error_codes.ErrorCode.ok);
}

/// Free result returned by `talu_sql_query` or `talu_sql_query_params`.
pub export fn talu_sql_query_free(ptr: ?[*:0]u8) callconv(.c) void {
    capi_error.clearError();
    const cstr = ptr orelse return;
    alloc.free(std.mem.span(cstr));
}

// =============================================================================
// Tests
// =============================================================================

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

test "talu_sql_query_params binds text param and returns structured json" {
    const documents = @import("../db/table/documents.zig");

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    var adapter = try documents.DocumentAdapter.init(std.testing.allocator, root);
    defer adapter.deinit();
    try adapter.writeDocument(.{
        .doc_id = "doc-param-1",
        .doc_type = "note",
        .title = "Params Test",
        .doc_json = "{\"_sys\":{},\"data\":{}}",
        .created_at_ms = 50,
        .updated_at_ms = 50,
        .group_id = "grp-param",
    });
    try adapter.flush();

    const query = "SELECT doc_id FROM docs WHERE group_id = ?";

    const root_z = try std.testing.allocator.allocSentinel(u8, root.len, 0);
    defer std.testing.allocator.free(root_z);
    @memcpy(root_z, root);

    const query_z = try std.testing.allocator.allocSentinel(u8, query.len, 0);
    defer std.testing.allocator.free(query_z);
    @memcpy(query_z, query);

    const grp_text = "grp-param";
    var c_params = [_]CSqlParam{
        std.mem.zeroes(CSqlParam),
    };
    c_params[0].tag = 3; // text
    c_params[0].ptr = grp_text.ptr;
    c_params[0].len = grp_text.len;

    var out_json: ?[*:0]u8 = null;
    try std.testing.expectEqual(
        @as(i32, 0),
        talu_sql_query_params(root_z.ptr, query_z.ptr, &c_params, 1, &out_json),
    );
    defer talu_sql_query_free(out_json);

    const json_slice = std.mem.span(out_json.?);
    // Structured format
    try std.testing.expect(std.mem.indexOf(u8, json_slice, "\"columns\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json_slice, "\"row_count\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, json_slice, "doc-param-1") != null);
}
