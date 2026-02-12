//! Integration tests for db.TableAdapter
//!
//! TableAdapter translates ItemRecord events into TaluDB block columns
//! and provides session scanning/deletion.

const std = @import("std");
const main = @import("main");
const db = main.db;
const responses = main.responses;

const TableAdapter = db.TableAdapter;
const StorageBackend = responses.StorageBackend;
const StorageEvent = responses.StorageEvent;
const ItemRecord = responses.ItemRecord;
const SessionRecord = responses.backend.SessionRecord;
const computeSessionHash = db.table.sessions.computeSessionHash;
const freeScannedSessionRecords = db.table.sessions.freeScannedSessionRecords;

// ===== init =====

test "TableAdapter: init creates adapter with session hash" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var adapter = try TableAdapter.init(std.testing.allocator, root_path, "test-session");
    defer adapter.backend().deinit();

    try std.testing.expectEqual(computeSessionHash("test-session"), adapter.session_hash);
}

// ===== initReadOnly =====

test "TableAdapter: initReadOnly opens without lock" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var adapter = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
    defer adapter.deinitReadOnly();
}

// ===== deinitReadOnly =====

test "TableAdapter: deinitReadOnly releases reader resources" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var adapter = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
    adapter.deinitReadOnly();

    // Verify resources were released: re-opening must succeed
    // (file handles closed, allocations freed — std.testing.allocator catches leaks).
    var adapter2 = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
    adapter2.deinitReadOnly();
}

// ===== backend =====

test "TableAdapter: backend returns StorageBackend interface" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var adapter = try TableAdapter.init(std.testing.allocator, root_path, "test-session");
    const storage = adapter.backend();
    storage.deinit();
}

// ===== writeEmbedding =====

test "TableAdapter: writeEmbedding writes embedding to WAL" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var adapter = try TableAdapter.init(std.testing.allocator, root_path, "embed-session");
    defer adapter.backend().deinit();

    const vector = [_]f32{ 0.5, -1.25, 2.0 };
    try adapter.writeEmbedding(42, &vector, "payload-data");

    const wal_stat = try tmp.dir.statFile("chat/current.wal");
    try std.testing.expect(wal_stat.size > 0);
}

// ===== scanSessions =====

test "TableAdapter: scanSessions returns written sessions" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    // Write a session
    {
        var adapter = try TableAdapter.init(std.testing.allocator, root_path, "scan-1");
        defer adapter.backend().deinit();

        const session = SessionRecord{
            .session_id = "scan-1",
            .model = "test-model",
            .title = "Test Session",
            .system_prompt = null,
            .config_json = null,
            .status = "active",
            .parent_session_id = null,
            .group_id = null,
            .head_item_id = 1,
            .ttl_ts = 0,
            .metadata_json = null,
            .created_at_ms = 100,
            .updated_at_ms = 200,
        };

        try adapter.backend().onEvent(&StorageEvent{ .PutSession = session });
        try adapter.fs_writer.flushBlock();
    }

    // Scan
    var reader = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
    defer reader.deinitReadOnly();

    const records = try reader.scanSessions(std.testing.allocator, null);
    defer freeScannedSessionRecords(std.testing.allocator, records);

    try std.testing.expectEqual(@as(usize, 1), records.len);
    try std.testing.expectEqualStrings("scan-1", records[0].session_id);
}

test "TableAdapter: scanSessions with target_hash returns single match" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    // Write two sessions
    inline for (.{ "target-a", "target-b" }) |sid| {
        var adapter = try TableAdapter.init(std.testing.allocator, root_path, sid);
        defer adapter.backend().deinit();

        const session = SessionRecord{
            .session_id = sid,
            .model = null,
            .title = sid,
            .system_prompt = null,
            .config_json = null,
            .status = null,
            .parent_session_id = null,
            .group_id = null,
            .head_item_id = 0,
            .ttl_ts = 0,
            .metadata_json = null,
            .created_at_ms = 100,
            .updated_at_ms = 100,
        };

        try adapter.backend().onEvent(&StorageEvent{ .PutSession = session });
        try adapter.fs_writer.flushBlock();
    }

    // Scan with target hash
    var reader = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
    defer reader.deinitReadOnly();

    const target_hash = computeSessionHash("target-a");
    const records = try reader.scanSessions(std.testing.allocator, target_hash);
    defer freeScannedSessionRecords(std.testing.allocator, records);

    try std.testing.expectEqual(@as(usize, 1), records.len);
    try std.testing.expectEqualStrings("target-a", records[0].session_id);
}

// ===== deleteSession =====

test "TableAdapter: deleteSession removes session from scan results" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    // Write session
    {
        var adapter = try TableAdapter.init(std.testing.allocator, root_path, "del-session");
        defer adapter.backend().deinit();

        const session = SessionRecord{
            .session_id = "del-session",
            .model = null,
            .title = "To Delete",
            .system_prompt = null,
            .config_json = null,
            .status = null,
            .parent_session_id = null,
            .group_id = null,
            .head_item_id = 0,
            .ttl_ts = 0,
            .metadata_json = null,
            .created_at_ms = 100,
            .updated_at_ms = 100,
        };

        try adapter.backend().onEvent(&StorageEvent{ .PutSession = session });
        try adapter.fs_writer.flushBlock();
    }

    // Delete
    {
        var adapter = try TableAdapter.init(std.testing.allocator, root_path, "del-session");
        defer adapter.backend().deinit();
        try adapter.deleteSession("del-session");
    }

    // Verify gone
    var reader = try TableAdapter.initReadOnly(std.testing.allocator, root_path);
    defer reader.deinitReadOnly();

    const records = try reader.scanSessions(std.testing.allocator, null);
    defer freeScannedSessionRecords(std.testing.allocator, records);

    try std.testing.expectEqual(@as(usize, 0), records.len);
}
