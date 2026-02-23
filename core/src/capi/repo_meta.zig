//! C-API repository metadata module.
//!
//! Exposes pin metadata operations backed by io/repository/meta.zig.

const std = @import("std");
const capi_error = @import("error.zig");
const error_codes = @import("error_codes.zig");
const repo_meta_store = @import("../io/repository/meta.zig");

const allocator = std.heap.c_allocator;

/// Opaque handle for repo pin metadata adapter.
/// Thread safety: NOT thread-safe (single-writer semantics via lock).
pub const PinAdapterHandle = opaque {};

/// C-ABI pin record.
pub const CPinRecord = extern struct {
    model_uri: ?[*:0]const u8,
    pinned_at_ms: i64,
    size_bytes: u64,
    has_size_bytes: bool,
    size_updated_at_ms: i64,
    has_size_updated_at_ms: bool,
    _reserved: [6]u8 = [_]u8{0} ** 6,
};

/// C-ABI pin list with arena-backed string lifetime.
pub const CPinList = extern struct {
    items: ?[*]CPinRecord,
    count: usize,
    _arena: ?*anyopaque,
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

fn toAdapter(handle: ?*PinAdapterHandle) !*repo_meta_store.Store {
    return @ptrCast(@alignCast(handle orelse return error.InvalidHandle));
}

fn buildPinList(records: []repo_meta_store.PinRecord) !CPinList {
    const arena_ptr = allocator.create(std.heap.ArenaAllocator) catch return error.OutOfMemory;
    errdefer allocator.destroy(arena_ptr);
    arena_ptr.* = std.heap.ArenaAllocator.init(allocator);
    errdefer arena_ptr.deinit();

    const arena = arena_ptr.allocator();
    const items = arena.alloc(CPinRecord, records.len) catch return error.OutOfMemory;

    for (records, 0..) |record, idx| {
        items[idx] = std.mem.zeroes(CPinRecord);
        items[idx].model_uri = (arena.dupeZ(u8, record.model_uri) catch return error.OutOfMemory).ptr;
        items[idx].pinned_at_ms = record.pinned_at_ms;
        if (record.size_bytes) |size_bytes| {
            items[idx].size_bytes = size_bytes;
            items[idx].has_size_bytes = true;
        }
        if (record.size_updated_at_ms) |size_updated_at_ms| {
            items[idx].size_updated_at_ms = size_updated_at_ms;
            items[idx].has_size_updated_at_ms = true;
        }
    }

    return .{
        .items = if (records.len > 0) items.ptr else null,
        .count = records.len,
        ._arena = @ptrCast(arena_ptr),
    };
}

/// Initialize repo metadata adapter.
pub export fn talu_repo_meta_init(
    db_path: ?[*:0]const u8,
    out_handle: ?*?*PinAdapterHandle,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;

    const db_path_slice = validateRequiredArg(db_path, "db_path") orelse {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const adapter_ptr = allocator.create(repo_meta_store.Store) catch {
        capi_error.setErrorWithCode(.out_of_memory, "failed to allocate RepoMetaStore", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };
    errdefer allocator.destroy(adapter_ptr);

    adapter_ptr.* = repo_meta_store.Store.init(allocator, db_path_slice) catch |err| {
        capi_error.setError(err, "failed to initialize repo metadata adapter", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    out.* = @ptrCast(adapter_ptr);
    return 0;
}

/// Free repo metadata adapter handle.
pub export fn talu_repo_meta_free(handle: ?*PinAdapterHandle) callconv(.c) void {
    capi_error.clearError();
    const adapter: *repo_meta_store.Store = @ptrCast(@alignCast(handle orelse return));
    adapter.deinit();
    allocator.destroy(adapter);
}

/// Pin a model URI. Returns 0 whether newly pinned or already pinned.
pub export fn talu_repo_meta_pin(
    handle: ?*PinAdapterHandle,
    model_uri: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const adapter = toAdapter(handle) catch |err| {
        capi_error.setError(err, "invalid handle", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    const uri = validateRequiredArg(model_uri, "model_uri") orelse {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    _ = adapter.pinModel(uri) catch |err| {
        capi_error.setError(err, "failed to pin model", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Unpin a model URI. Returns 0 whether removed or already absent.
pub export fn talu_repo_meta_unpin(
    handle: ?*PinAdapterHandle,
    model_uri: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const adapter = toAdapter(handle) catch |err| {
        capi_error.setError(err, "invalid handle", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    const uri = validateRequiredArg(model_uri, "model_uri") orelse {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    _ = adapter.unpinModel(uri) catch |err| {
        capi_error.setError(err, "failed to unpin model", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Update cached size bytes for an already pinned model.
pub export fn talu_repo_meta_update_size(
    handle: ?*PinAdapterHandle,
    model_uri: ?[*:0]const u8,
    size_bytes: u64,
) callconv(.c) i32 {
    capi_error.clearError();
    const adapter = toAdapter(handle) catch |err| {
        capi_error.setError(err, "invalid handle", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    const uri = validateRequiredArg(model_uri, "model_uri") orelse {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    adapter.upsertSizeBytes(uri, size_bytes) catch |err| {
        capi_error.setError(err, "failed to update size bytes", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Clear cached size bytes for a pinned model. No-op when missing.
pub export fn talu_repo_meta_clear_size(
    handle: ?*PinAdapterHandle,
    model_uri: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const adapter = toAdapter(handle) catch |err| {
        capi_error.setError(err, "invalid handle", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    const uri = validateRequiredArg(model_uri, "model_uri") orelse {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    adapter.clearSizeBytes(uri) catch |err| {
        capi_error.setError(err, "failed to clear size bytes", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// List all pinned models in deterministic order.
pub export fn talu_repo_meta_list_pins(
    handle: ?*PinAdapterHandle,
    out_list: ?*CPinList,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_list orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_list is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = std.mem.zeroes(CPinList);

    const adapter = toAdapter(handle) catch |err| {
        capi_error.setError(err, "invalid handle", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    const records = adapter.listPins(allocator) catch |err| {
        capi_error.setError(err, "failed to list pins", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer repo_meta_store.freePinRecords(allocator, records);

    out.* = buildPinList(records) catch |err| {
        capi_error.setError(err, "failed to build pin list", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Free a pin list returned by talu_repo_meta_list_pins.
pub export fn talu_repo_meta_free_list(list: *CPinList) callconv(.c) void {
    capi_error.clearError();
    const value = list.*;
    if (value._arena) |arena_opaque| {
        const arena_ptr: *std.heap.ArenaAllocator = @ptrCast(@alignCast(arena_opaque));
        arena_ptr.deinit();
        allocator.destroy(arena_ptr);
    }
    list.* = std.mem.zeroes(CPinList);
}

test "repo_meta C-ABI list/free round-trip" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);
    const root_z = try std.testing.allocator.dupeZ(u8, root);
    defer std.testing.allocator.free(root_z);

    var handle: ?*PinAdapterHandle = null;
    try std.testing.expectEqual(@as(i32, 0), talu_repo_meta_init(root_z.ptr, &handle));
    defer talu_repo_meta_free(handle);

    try std.testing.expect(handle != null);
    try std.testing.expectEqual(@as(i32, 0), talu_repo_meta_pin(handle, "Qwen/Qwen3-0.6B"));
    try std.testing.expectEqual(@as(i32, 0), talu_repo_meta_update_size(handle, "Qwen/Qwen3-0.6B", 1234));
    try std.testing.expectEqual(@as(i32, 0), talu_repo_meta_pin(handle, "openai/gpt-oss-20b"));

    var list = std.mem.zeroes(CPinList);
    try std.testing.expectEqual(@as(i32, 0), talu_repo_meta_list_pins(handle, &list));
    defer talu_repo_meta_free_list(&list);

    try std.testing.expectEqual(@as(usize, 2), list.count);
    try std.testing.expect(list.items != null);

    const items = list.items.?[0..list.count];
    var saw_qwen = false;
    var saw_openai = false;
    for (items) |item| {
        const uri = std.mem.sliceTo(item.model_uri.?, 0);
        if (std.mem.eql(u8, uri, "Qwen/Qwen3-0.6B")) {
            saw_qwen = true;
            try std.testing.expect(item.has_size_bytes);
            try std.testing.expectEqual(@as(u64, 1234), item.size_bytes);
        } else if (std.mem.eql(u8, uri, "openai/gpt-oss-20b")) {
            saw_openai = true;
            try std.testing.expect(!item.has_size_bytes);
        }
    }
    try std.testing.expect(saw_qwen);
    try std.testing.expect(saw_openai);
}

test "repo_meta C-ABI invalid arguments and not-found map to stable codes" {
    const invalid_argument = @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const item_not_found = @intFromEnum(error_codes.ErrorCode.item_not_found);

    try std.testing.expectEqual(invalid_argument, talu_repo_meta_init(null, null));

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);
    const root_z = try std.testing.allocator.dupeZ(u8, root);
    defer std.testing.allocator.free(root_z);

    var handle: ?*PinAdapterHandle = null;
    try std.testing.expectEqual(@as(i32, 0), talu_repo_meta_init(root_z.ptr, &handle));
    defer talu_repo_meta_free(handle);

    try std.testing.expectEqual(invalid_argument, talu_repo_meta_pin(handle, null));
    try std.testing.expectEqual(item_not_found, talu_repo_meta_update_size(handle, "missing/model", 42));
}
