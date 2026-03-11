//! Plane-specific DB C-API: generic KV state storage.

const std = @import("std");
const capi_error = @import("../error.zig");
const error_codes = @import("../error_codes.zig");
const kv_store = @import("../../db/kv/store.zig");

const allocator = std.heap.c_allocator;

/// Opaque handle for KV storage.
pub const KvHandle = opaque {};

/// C-ABI single KV value payload.
pub const CKvValue = extern struct {
    data: ?[*]u8,
    len: usize,
    updated_at_ms: i64,
    found: bool,
    _reserved: [7]u8 = [_]u8{0} ** 7,
    _arena: ?*anyopaque,
};

/// C-ABI KV entry.
pub const CKvEntry = extern struct {
    key: ?[*:0]const u8,
    value: ?[*]const u8,
    value_len: usize,
    updated_at_ms: i64,
    _reserved: [8]u8 = [_]u8{0} ** 8,
};

/// C-ABI KV entry list with arena-backed lifetime.
pub const CKvList = extern struct {
    items: ?[*]CKvEntry,
    count: usize,
    _arena: ?*anyopaque,
};

/// C-ABI namespace runtime stats.
pub const CKvNamespaceStats = extern struct {
    batched_pending: usize,
    batched_max_pending: usize,
    batched_max_lag_ms: i64,
    batched_next_flush_deadline_ms: i64,
    batched_enqueued_writes: u64,
    batched_coalesced_writes: u64,
    batched_rejected_writes: u64,
    batched_flush_count: u64,
    batched_flushed_entries: u64,
    total_live_entries: usize,
    ephemeral_live_entries: usize,
    watch_published: u64,
    watch_overwritten: u64,
    watch_capacity: usize,
    _reserved: [8]u8 = [_]u8{0} ** 8,
};

pub const CKvWatchEvent = extern struct {
    seq: u64,
    event_type: u8,
    durability_class: u8,
    has_durability: bool,
    ttl_ms: u64,
    has_ttl: bool,
    _flags_reserved: [6]u8 = [_]u8{0} ** 6,
    key: ?[*:0]const u8,
    value_len: usize,
    updated_at_ms: i64,
    _reserved: [8]u8 = [_]u8{0} ** 8,
};

pub const CKvWatchBatch = extern struct {
    items: ?[*]CKvWatchEvent,
    count: usize,
    lost: bool,
    _flags_reserved: [7]u8 = [_]u8{0} ** 7,
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

fn validateBytes(bytes: ?[*]const u8, bytes_len: usize) ?[]const u8 {
    if (bytes_len == 0) return &.{};
    const ptr = bytes orelse {
        capi_error.setErrorWithCode(.invalid_argument, "value is null but value_len > 0", .{});
        return null;
    };
    return ptr[0..bytes_len];
}

fn toStore(handle: ?*KvHandle) !*kv_store.KVStore {
    return @ptrCast(@alignCast(handle orelse return error.InvalidHandle));
}

fn parseDurabilityClass(raw: u8) ?kv_store.DurabilityClass {
    return switch (raw) {
        0 => .strong,
        1 => .batched,
        2 => .ephemeral,
        else => null,
    };
}

fn buildKvValue(value: ?kv_store.ValueRecord) !CKvValue {
    var out = std.mem.zeroes(CKvValue);
    if (value) |record| {
        const arena_ptr = allocator.create(std.heap.ArenaAllocator) catch return error.OutOfMemory;
        errdefer allocator.destroy(arena_ptr);
        arena_ptr.* = std.heap.ArenaAllocator.init(allocator);
        errdefer arena_ptr.deinit();

        const arena = arena_ptr.allocator();
        const data = try arena.dupe(u8, record.value);
        out.data = if (data.len > 0) data.ptr else null;
        out.len = data.len;
        out.updated_at_ms = record.updated_at_ms;
        out.found = true;
        out._arena = @ptrCast(arena_ptr);
    }
    return out;
}

fn buildKvList(records: []const kv_store.EntryRecord) !CKvList {
    const arena_ptr = allocator.create(std.heap.ArenaAllocator) catch return error.OutOfMemory;
    errdefer allocator.destroy(arena_ptr);
    arena_ptr.* = std.heap.ArenaAllocator.init(allocator);
    errdefer arena_ptr.deinit();

    const arena = arena_ptr.allocator();
    const items = try arena.alloc(CKvEntry, records.len);
    for (records, 0..) |record, idx| {
        items[idx] = std.mem.zeroes(CKvEntry);
        items[idx].key = (try arena.dupeZ(u8, record.key)).ptr;
        const value_copy = try arena.dupe(u8, record.value);
        items[idx].value = if (value_copy.len > 0) value_copy.ptr else null;
        items[idx].value_len = value_copy.len;
        items[idx].updated_at_ms = record.updated_at_ms;
    }

    return .{
        .items = if (records.len > 0) items.ptr else null,
        .count = records.len,
        ._arena = @ptrCast(arena_ptr),
    };
}

fn buildKvNamespaceStats(stats: kv_store.NamespaceStats) CKvNamespaceStats {
    return .{
        .batched_pending = stats.batched_pending,
        .batched_max_pending = stats.batched_max_pending,
        .batched_max_lag_ms = stats.batched_max_lag_ms,
        .batched_next_flush_deadline_ms = stats.batched_next_flush_deadline_ms,
        .batched_enqueued_writes = stats.batched_enqueued_writes,
        .batched_coalesced_writes = stats.batched_coalesced_writes,
        .batched_rejected_writes = stats.batched_rejected_writes,
        .batched_flush_count = stats.batched_flush_count,
        .batched_flushed_entries = stats.batched_flushed_entries,
        .total_live_entries = stats.total_live_entries,
        .ephemeral_live_entries = stats.ephemeral_live_entries,
        .watch_published = stats.watch_published,
        .watch_overwritten = stats.watch_overwritten,
        .watch_capacity = stats.watch_capacity,
        ._reserved = [_]u8{0} ** 8,
    };
}

fn buildKvWatchBatch(events: []const kv_store.WatchEvent, lost: bool) !CKvWatchBatch {
    const arena_ptr = allocator.create(std.heap.ArenaAllocator) catch return error.OutOfMemory;
    errdefer allocator.destroy(arena_ptr);
    arena_ptr.* = std.heap.ArenaAllocator.init(allocator);
    errdefer arena_ptr.deinit();

    const arena = arena_ptr.allocator();
    const items = try arena.alloc(CKvWatchEvent, events.len);
    for (events, 0..) |event, idx| {
        items[idx] = std.mem.zeroes(CKvWatchEvent);
        items[idx].seq = event.seq;
        items[idx].event_type = @intFromEnum(event.event_type);
        if (event.durability) |durability| {
            items[idx].durability_class = @intFromEnum(durability);
            items[idx].has_durability = true;
        }
        if (event.ttl_ms) |ttl_ms| {
            items[idx].ttl_ms = ttl_ms;
            items[idx].has_ttl = true;
        }
        items[idx].key = (try arena.dupeZ(u8, event.key)).ptr;
        items[idx].value_len = event.value_len;
        items[idx].updated_at_ms = event.updated_at_ms;
    }

    return .{
        .items = if (events.len > 0) items.ptr else null,
        .count = events.len,
        .lost = lost,
        ._arena = @ptrCast(arena_ptr),
    };
}

/// Open generic KV namespace under db root.
pub export fn talu_db_kv_init(
    db_path: ?[*:0]const u8,
    namespace: ?[*:0]const u8,
    out_handle: ?*?*KvHandle,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;

    const db_root = validateRequiredArg(db_path, "db_path") orelse {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const ns = validateRequiredArg(namespace, "namespace") orelse {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const store_ptr = allocator.create(kv_store.KVStore) catch {
        capi_error.setErrorWithCode(.out_of_memory, "failed to allocate KVStore handle", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };
    errdefer allocator.destroy(store_ptr);

    store_ptr.* = kv_store.KVStore.init(allocator, db_root, ns) catch |err| {
        capi_error.setError(err, "failed to initialize KVStore", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    out.* = @ptrCast(store_ptr);
    return 0;
}

/// Free KV handle.
pub export fn talu_db_kv_free(handle: ?*KvHandle) callconv(.c) void {
    capi_error.clearError();
    const store: *kv_store.KVStore = @ptrCast(@alignCast(handle orelse return));
    store.deinit();
    allocator.destroy(store);
}

/// Upsert key/value entry.
pub export fn talu_db_kv_put(
    handle: ?*KvHandle,
    key: ?[*:0]const u8,
    value: ?[*]const u8,
    value_len: usize,
) callconv(.c) i32 {
    return talu_db_kv_put_ex(handle, key, value, value_len, 0, 0);
}

/// Upsert key/value entry with explicit durability class and TTL.
pub export fn talu_db_kv_put_ex(
    handle: ?*KvHandle,
    key: ?[*:0]const u8,
    value: ?[*]const u8,
    value_len: usize,
    durability_class: u8,
    ttl_ms: u64,
) callconv(.c) i32 {
    capi_error.clearError();
    const store = toStore(handle) catch |err| {
        capi_error.setError(err, "invalid handle", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    const key_slice = validateRequiredArg(key, "key") orelse {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const value_slice = validateBytes(value, value_len) orelse {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const durability = parseDurabilityClass(durability_class) orelse {
        capi_error.setErrorWithCode(.invalid_argument, "durability_class must be 0(strong), 1(batched), or 2(ephemeral)", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    store.putWithOptions(key_slice, value_slice, .{
        .durability = durability,
        .ttl_ms = ttl_ms,
    }) catch |err| {
        capi_error.setError(err, "failed to put KV entry", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Lookup key/value entry.
pub export fn talu_db_kv_get(
    handle: ?*KvHandle,
    key: ?[*:0]const u8,
    out_value: ?*CKvValue,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_value orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_value is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = std.mem.zeroes(CKvValue);

    const store = toStore(handle) catch |err| {
        capi_error.setError(err, "invalid handle", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    const key_slice = validateRequiredArg(key, "key") orelse {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    var value_record = store.getEntryCopy(allocator, key_slice) catch |err| {
        capi_error.setError(err, "failed to get KV entry", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer if (value_record) |*record| record.deinit(allocator);

    out.* = buildKvValue(value_record) catch |err| {
        capi_error.setError(err, "failed to build KV value", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Delete key/value entry. Sets `out_deleted=true` when entry existed.
pub export fn talu_db_kv_delete(
    handle: ?*KvHandle,
    key: ?[*:0]const u8,
    out_deleted: ?*bool,
) callconv(.c) i32 {
    capi_error.clearError();
    if (out_deleted) |deleted_ptr| deleted_ptr.* = false;

    const store = toStore(handle) catch |err| {
        capi_error.setError(err, "invalid handle", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    const key_slice = validateRequiredArg(key, "key") orelse {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const deleted = store.delete(key_slice) catch |err| {
        capi_error.setError(err, "failed to delete KV entry", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    if (out_deleted) |deleted_ptr| deleted_ptr.* = deleted;
    return 0;
}

/// List all KV entries in key-sorted order.
pub export fn talu_db_kv_list(
    handle: ?*KvHandle,
    out_list: ?*CKvList,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_list orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_list is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = std.mem.zeroes(CKvList);

    const store = toStore(handle) catch |err| {
        capi_error.setError(err, "invalid handle", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    const entries = store.listEntries(allocator) catch |err| {
        capi_error.setError(err, "failed to list KV entries", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer kv_store.freeEntryRecords(allocator, entries);

    out.* = buildKvList(entries) catch |err| {
        capi_error.setError(err, "failed to build KV list", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Flush in-memory block to WAL/current.talu.
pub export fn talu_db_kv_flush(handle: ?*KvHandle) callconv(.c) i32 {
    capi_error.clearError();
    const store = toStore(handle) catch |err| {
        capi_error.setError(err, "invalid handle", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    store.flush() catch |err| {
        capi_error.setError(err, "failed to flush KV namespace", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Flush only batched write-behind entries.
pub export fn talu_db_kv_flush_batched(handle: ?*KvHandle) callconv(.c) i32 {
    capi_error.clearError();
    const store = toStore(handle) catch |err| {
        capi_error.setError(err, "invalid handle", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    store.flushBatched() catch |err| {
        capi_error.setError(err, "failed to flush batched KV entries", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Return namespace runtime stats.
pub export fn talu_db_kv_stats(
    handle: ?*KvHandle,
    out_stats: ?*CKvNamespaceStats,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_stats orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_stats is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = std.mem.zeroes(CKvNamespaceStats);

    const store = toStore(handle) catch |err| {
        capi_error.setError(err, "invalid handle", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    out.* = buildKvNamespaceStats(store.stats());
    return 0;
}

/// Drain watch events with `seq > after_seq`, up to `max_events`.
pub export fn talu_db_kv_watch_drain(
    handle: ?*KvHandle,
    after_seq: u64,
    max_events: usize,
    out_batch: ?*CKvWatchBatch,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_batch orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_batch is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = std.mem.zeroes(CKvWatchBatch);

    const store = toStore(handle) catch |err| {
        capi_error.setError(err, "invalid handle", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    var drained = store.watchDrain(allocator, after_seq, max_events) catch |err| {
        capi_error.setError(err, "failed to drain KV watch events", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer drained.deinit(allocator);

    out.* = buildKvWatchBatch(drained.events, drained.lost) catch |err| {
        capi_error.setError(err, "failed to build KV watch batch", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Compact KV namespace to current active state.
pub export fn talu_db_kv_compact(handle: ?*KvHandle) callconv(.c) i32 {
    capi_error.clearError();
    const store = toStore(handle) catch |err| {
        capi_error.setError(err, "invalid handle", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    store.compact() catch |err| {
        capi_error.setError(err, "failed to compact KV namespace", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Free value returned by `talu_db_kv_get`.
pub export fn talu_db_kv_free_value(value: *CKvValue) callconv(.c) void {
    capi_error.clearError();
    const owned = value.*;
    if (owned._arena) |arena_opaque| {
        const arena_ptr: *std.heap.ArenaAllocator = @ptrCast(@alignCast(arena_opaque));
        arena_ptr.deinit();
        allocator.destroy(arena_ptr);
    }
    value.* = std.mem.zeroes(CKvValue);
}

/// Free list returned by `talu_db_kv_list`.
pub export fn talu_db_kv_free_list(list: *CKvList) callconv(.c) void {
    capi_error.clearError();
    const owned = list.*;
    if (owned._arena) |arena_opaque| {
        const arena_ptr: *std.heap.ArenaAllocator = @ptrCast(@alignCast(arena_opaque));
        arena_ptr.deinit();
        allocator.destroy(arena_ptr);
    }
    list.* = std.mem.zeroes(CKvList);
}

/// Free batch returned by `talu_db_kv_watch_drain`.
pub export fn talu_db_kv_free_watch_batch(batch: *CKvWatchBatch) callconv(.c) void {
    capi_error.clearError();
    const owned = batch.*;
    if (owned._arena) |arena_opaque| {
        const arena_ptr: *std.heap.ArenaAllocator = @ptrCast(@alignCast(arena_opaque));
        arena_ptr.deinit();
        allocator.destroy(arena_ptr);
    }
    batch.* = std.mem.zeroes(CKvWatchBatch);
}

test "db kv C-API put/get/delete/list roundtrip" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);
    const root_z = try std.testing.allocator.dupeZ(u8, root);
    defer std.testing.allocator.free(root_z);

    var handle: ?*KvHandle = null;
    try std.testing.expectEqual(@as(i32, 0), talu_db_kv_init(root_z.ptr, "kv_test", &handle));
    defer talu_db_kv_free(handle);

    try std.testing.expectEqual(@as(i32, 0), talu_db_kv_put(handle, "k1", "v1", 2));
    try std.testing.expectEqual(@as(i32, 0), talu_db_kv_put(handle, "k2", "v2", 2));

    var value = std.mem.zeroes(CKvValue);
    try std.testing.expectEqual(@as(i32, 0), talu_db_kv_get(handle, "k1", &value));
    defer talu_db_kv_free_value(&value);
    try std.testing.expect(value.found);
    try std.testing.expectEqualStrings("v1", value.data.?[0..value.len]);

    var list = std.mem.zeroes(CKvList);
    try std.testing.expectEqual(@as(i32, 0), talu_db_kv_list(handle, &list));
    defer talu_db_kv_free_list(&list);
    try std.testing.expectEqual(@as(usize, 2), list.count);
    try std.testing.expect(list.items != null);

    var deleted = false;
    try std.testing.expectEqual(@as(i32, 0), talu_db_kv_delete(handle, "k1", &deleted));
    try std.testing.expect(deleted);
}

test "db kv C-API get missing returns found=false" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);
    const root_z = try std.testing.allocator.dupeZ(u8, root);
    defer std.testing.allocator.free(root_z);

    var handle: ?*KvHandle = null;
    try std.testing.expectEqual(@as(i32, 0), talu_db_kv_init(root_z.ptr, "kv_test", &handle));
    defer talu_db_kv_free(handle);

    var value = std.mem.zeroes(CKvValue);
    try std.testing.expectEqual(@as(i32, 0), talu_db_kv_get(handle, "missing", &value));
    defer talu_db_kv_free_value(&value);
    try std.testing.expect(!value.found);
    try std.testing.expect(value.data == null);
    try std.testing.expectEqual(@as(usize, 0), value.len);
}

test "db kv C-API put_ex validates durability class" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);
    const root_z = try std.testing.allocator.dupeZ(u8, root);
    defer std.testing.allocator.free(root_z);

    var handle: ?*KvHandle = null;
    try std.testing.expectEqual(@as(i32, 0), talu_db_kv_init(root_z.ptr, "kv_test", &handle));
    defer talu_db_kv_free(handle);

    try std.testing.expectEqual(
        @as(i32, @intFromEnum(error_codes.ErrorCode.invalid_argument)),
        talu_db_kv_put_ex(handle, "k", "v", 1, 9, 0),
    );
}

test "db kv C-API stats and flush_batched" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);
    const root_z = try std.testing.allocator.dupeZ(u8, root);
    defer std.testing.allocator.free(root_z);

    var handle: ?*KvHandle = null;
    try std.testing.expectEqual(@as(i32, 0), talu_db_kv_init(root_z.ptr, "kv_test", &handle));
    defer talu_db_kv_free(handle);

    try std.testing.expectEqual(@as(i32, 0), talu_db_kv_put_ex(handle, "k1", "v1", 2, 1, 0));

    var stats = std.mem.zeroes(CKvNamespaceStats);
    try std.testing.expectEqual(@as(i32, 0), talu_db_kv_stats(handle, &stats));
    try std.testing.expectEqual(@as(usize, 1), stats.batched_pending);

    try std.testing.expectEqual(@as(i32, 0), talu_db_kv_flush_batched(handle));
    try std.testing.expectEqual(@as(i32, 0), talu_db_kv_stats(handle, &stats));
    try std.testing.expectEqual(@as(usize, 0), stats.batched_pending);
    try std.testing.expectEqual(@as(u64, 1), stats.batched_flushed_entries);
}

test "db kv C-API watch_drain returns core-generated events" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);
    const root_z = try std.testing.allocator.dupeZ(u8, root);
    defer std.testing.allocator.free(root_z);

    var handle: ?*KvHandle = null;
    try std.testing.expectEqual(@as(i32, 0), talu_db_kv_init(root_z.ptr, "kv_watch_test", &handle));
    defer talu_db_kv_free(handle);

    try std.testing.expectEqual(@as(i32, 0), talu_db_kv_put(handle, "alpha", "one", 3));

    var batch = std.mem.zeroes(CKvWatchBatch);
    try std.testing.expectEqual(@as(i32, 0), talu_db_kv_watch_drain(handle, 0, 16, &batch));
    defer talu_db_kv_free_watch_batch(&batch);
    try std.testing.expect(!batch.lost);
    try std.testing.expectEqual(@as(usize, 1), batch.count);
    const event = batch.items.?[0];
    try std.testing.expectEqual(@as(u8, @intFromEnum(kv_store.WatchEventType.put)), event.event_type);
    try std.testing.expectEqualStrings("alpha", std.mem.sliceTo(event.key.?, 0));
}
