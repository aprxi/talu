//! Integration tests for db.table.sessions.TableAdapter
//!
//! Tests session-specific methods: lookupSession, updateSession,
//! updateSessionEx, scanSessionsFiltered. Basic lifecycle tests
//! (init, scanSessions, deleteSession) are covered in db/adapters/.

const std = @import("std");
const main = @import("main");
const db = main.db;
const responses = main.responses;

const TableAdapter = db.table.sessions.TableAdapter;
const ScanParams = db.table.sessions.ScanParams;
const StorageBackend = responses.StorageBackend;
const StorageEvent = responses.StorageEvent;
const SessionRecord = responses.backend.SessionRecord;
const computeSessionHash = db.table.sessions.computeSessionHash;
const freeScannedSessionRecord = db.table.sessions.freeScannedSessionRecord;
const freeScannedSessionRecords = db.table.sessions.freeScannedSessionRecords;

/// Helper: write a session record via StorageBackend.onEvent + flushBlock.
fn writeSession(adapter: *TableAdapter, session: SessionRecord) !void {
    try adapter.backend().onEvent(&StorageEvent{ .PutSession = session });
    try adapter.fs_writer.flushBlock();
}

/// Helper: create a minimal SessionRecord.
fn makeSession(session_id: []const u8, title: ?[]const u8, marker: ?[]const u8) SessionRecord {
    return .{
        .session_id = session_id,
        .model = null,
        .title = title,
        .system_prompt = null,
        .config_json = null,
        .marker = marker,
        .parent_session_id = null,
        .group_id = null,
        .head_item_id = 0,
        .ttl_ts = 0,
        .metadata_json = null,
        .created_at_ms = 100,
        .updated_at_ms = 200,
    };
}

// ===== lookupSession =====

test "TableAdapter: lookupSession returns matching session" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    {
        var adapter = try TableAdapter.init(std.testing.allocator, root_path, "lookup-1");
        defer adapter.backend().deinit();
        try writeSession(&adapter, makeSession("lookup-1", "Lookup Test", null));
    }

    var reader = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
    defer reader.deinitReadOnly();

    var record = try reader.lookupSession(std.testing.allocator, "lookup-1");
    defer freeScannedSessionRecord(std.testing.allocator, &record);

    try std.testing.expectEqualStrings("lookup-1", record.session_id);
    try std.testing.expectEqualStrings("Lookup Test", record.title.?);
}

test "TableAdapter: lookupSession returns SessionNotFound for missing session" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var reader = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
    defer reader.deinitReadOnly();

    try std.testing.expectError(
        error.SessionNotFound,
        reader.lookupSession(std.testing.allocator, "nonexistent"),
    );
}

// ===== updateSession =====

test "TableAdapter: updateSession merges title" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    // Write initial session
    {
        var adapter = try TableAdapter.init(std.testing.allocator, root_path, "upd-1");
        defer adapter.backend().deinit();
        try writeSession(&adapter, makeSession("upd-1", "Original Title", "active"));
    }

    // Update title only
    {
        var adapter = try TableAdapter.init(std.testing.allocator, root_path, "upd-1");
        defer adapter.backend().deinit();
        try adapter.updateSession(std.testing.allocator, "upd-1", "New Title", null, null);
    }

    // Read back
    var reader = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
    defer reader.deinitReadOnly();

    var record = try reader.lookupSession(std.testing.allocator, "upd-1");
    defer freeScannedSessionRecord(std.testing.allocator, &record);

    try std.testing.expectEqualStrings("New Title", record.title.?);
    // Marker should be preserved
    try std.testing.expectEqualStrings("active", record.marker.?);
}

test "TableAdapter: updateSession merges marker" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    {
        var adapter = try TableAdapter.init(std.testing.allocator, root_path, "upd-2");
        defer adapter.backend().deinit();
        try writeSession(&adapter, makeSession("upd-2", "Keep Me", null));
    }

    {
        var adapter = try TableAdapter.init(std.testing.allocator, root_path, "upd-2");
        defer adapter.backend().deinit();
        try adapter.updateSession(std.testing.allocator, "upd-2", null, "pinned", null);
    }

    var reader = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
    defer reader.deinitReadOnly();

    var record = try reader.lookupSession(std.testing.allocator, "upd-2");
    defer freeScannedSessionRecord(std.testing.allocator, &record);

    try std.testing.expectEqualStrings("Keep Me", record.title.?);
    try std.testing.expectEqualStrings("pinned", record.marker.?);
}

test "TableAdapter: updateSession returns error for nonexistent session" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var adapter = try TableAdapter.init(std.testing.allocator, root_path, "ghost");
    defer adapter.backend().deinit();

    try std.testing.expectError(
        error.SessionNotFound,
        adapter.updateSession(std.testing.allocator, "ghost", "Title", null, null),
    );
}

// ===== scanSessionsFiltered =====

test "TableAdapter: scanSessionsFiltered with limit" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    // Write 3 sessions
    inline for (.{ "filt-a", "filt-b", "filt-c" }) |sid| {
        var adapter = try TableAdapter.init(std.testing.allocator, root_path, sid);
        defer adapter.backend().deinit();
        try writeSession(&adapter, makeSession(sid, sid, null));
    }

    var reader = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
    defer reader.deinitReadOnly();

    const params = ScanParams{ .limit = 2 };
    const records = try reader.scanSessionsFiltered(std.testing.allocator, params);
    defer freeScannedSessionRecords(std.testing.allocator, records);

    try std.testing.expectEqual(@as(usize, 2), records.len);
}

test "TableAdapter: scanSessionsFiltered with target_hash" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    inline for (.{ "hash-x", "hash-y" }) |sid| {
        var adapter = try TableAdapter.init(std.testing.allocator, root_path, sid);
        defer adapter.backend().deinit();
        try writeSession(&adapter, makeSession(sid, sid, null));
    }

    var reader = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
    defer reader.deinitReadOnly();

    const params = ScanParams{ .target_hash = computeSessionHash("hash-x") };
    const records = try reader.scanSessionsFiltered(std.testing.allocator, params);
    defer freeScannedSessionRecords(std.testing.allocator, records);

    try std.testing.expectEqual(@as(usize, 1), records.len);
    try std.testing.expectEqualStrings("hash-x", records[0].session_id);
}
