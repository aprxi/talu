//! Integration tests for db.Writer
//!
//! Writer is the namespace write path: opens a TaluDB namespace directory,
//! manages WAL durability, in-memory buffering, and block flush.

const std = @import("std");
const main = @import("main");
const db = main.db;

const Writer = db.Writer;
const types = db.types;

// ===== open =====

test "Writer: open creates namespace directory and files" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const tmp_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(tmp_path);

    var writer = try Writer.open(std.testing.allocator, tmp_path, "chat");
    defer writer.deinit();

    // Verify files were created
    var lock_file = try tmp.dir.openFile("chat/talu.lock", .{});
    defer lock_file.close();
    var data_file = try tmp.dir.openFile("chat/current.talu", .{});
    defer data_file.close();
    var wal_file = try tmp.dir.openFile("chat/current.wal", .{});
    defer wal_file.close();
}

// ===== appendRow =====

test "Writer: appendRow buffers data and writes WAL" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const tmp_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(tmp_path);

    var writer = try Writer.open(std.testing.allocator, tmp_path, "chat");
    defer writer.deinit();

    var value: u64 = 42;
    const col = db.writer.ColumnValue{
        .column_id = 1,
        .shape = .SCALAR,
        .phys_type = .U64,
        .encoding = .RAW,
        .dims = 1,
        .data = std.mem.asBytes(&value),
    };

    try writer.appendRow(3, &.{col});

    try std.testing.expectEqual(@as(u32, 1), writer.row_count);
    try std.testing.expect(writer.buffer_bytes > 0);

    // WAL should have data
    const wal_data = try tmp.dir.readFileAlloc(std.testing.allocator, "chat/current.wal", 1024);
    defer std.testing.allocator.free(wal_data);
    try std.testing.expect(wal_data.len > 0);
}

// ===== appendBatch =====

test "Writer: appendBatch buffers multiple rows" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var writer = try Writer.open(std.testing.allocator, root_path, "vector");
    defer writer.deinit();

    const ids = [_]u64{ 1, 2 };
    const ts = [_]i64{ 10, 20 };

    const columns = [_]db.writer.ColumnBatch{
        .{
            .column_id = 1,
            .shape = .SCALAR,
            .phys_type = .U64,
            .encoding = .RAW,
            .dims = 1,
            .data = std.mem.sliceAsBytes(&ids),
        },
        .{
            .column_id = 2,
            .shape = .SCALAR,
            .phys_type = .I64,
            .encoding = .RAW,
            .dims = 1,
            .data = std.mem.sliceAsBytes(&ts),
        },
    };

    try writer.appendBatch(10, 2, &columns);
    try std.testing.expectEqual(@as(u32, 2), writer.row_count);
}

// ===== flushBlock =====

test "Writer: flushBlock writes block and clears WAL" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const tmp_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(tmp_path);

    var writer = try Writer.open(std.testing.allocator, tmp_path, "chat");
    defer writer.deinit();

    var value: u64 = 7;
    const col = db.writer.ColumnValue{
        .column_id = 1,
        .shape = .SCALAR,
        .phys_type = .U64,
        .encoding = .RAW,
        .dims = 1,
        .data = std.mem.asBytes(&value),
    };

    try writer.appendRow(3, &.{col});
    try writer.flushBlock();

    // Data file should contain the block
    const data = try tmp.dir.readFileAlloc(std.testing.allocator, "chat/current.talu", 1024);
    defer std.testing.allocator.free(data);
    try std.testing.expect(data.len > 0);

    // WAL should be cleared
    const wal_data = try tmp.dir.readFileAlloc(std.testing.allocator, "chat/current.wal", 1024);
    defer std.testing.allocator.free(wal_data);
    try std.testing.expectEqual(@as(usize, 0), wal_data.len);
}

// ===== resetSchema =====

test "Writer: resetSchema clears buffered state" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const tmp_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(tmp_path);

    var writer = try Writer.open(std.testing.allocator, tmp_path, "chat");
    defer writer.deinit();

    var value: u64 = 7;
    const col = db.writer.ColumnValue{
        .column_id = 1,
        .shape = .SCALAR,
        .phys_type = .U64,
        .encoding = .RAW,
        .dims = 1,
        .data = std.mem.asBytes(&value),
    };

    try writer.appendRow(3, &.{col});
    writer.resetSchema();

    try std.testing.expect(writer.schema_id == null);
    try std.testing.expectEqual(@as(u32, 0), writer.row_count);
}

// ===== rotateSegment (via flushBlockLocked) =====

test "Writer: rotateSegment seals segment when threshold exceeded" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const tmp_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(tmp_path);

    var writer = try Writer.open(std.testing.allocator, tmp_path, "store");
    defer writer.deinit();

    // Set a tiny segment size so rotation triggers after one flush
    writer.max_segment_size = 1024;

    const col = db.writer.ColumnValue{
        .column_id = 1,
        .shape = .VARBYTES,
        .phys_type = .BINARY,
        .encoding = .RAW,
        .dims = 0,
        .data = "a" ** 512,
    };

    // First append+flush: creates initial data in current.talu
    try writer.appendRow(1, &.{col});
    try writer.flushBlock();

    const data_after_first = try tmp.dir.readFileAlloc(std.testing.allocator, "store/current.talu", 64 * 1024);
    defer std.testing.allocator.free(data_after_first);
    const first_block_size = data_after_first.len;
    try std.testing.expect(first_block_size > 0);

    // Second append+flush: should trigger rotation because
    // current.talu already has data and adding another block would exceed 1KB
    try writer.appendRow(1, &.{col});
    try writer.flushBlock();

    // current.talu should contain only the second block (post-rotation)
    const current_data = try tmp.dir.readFileAlloc(std.testing.allocator, "store/current.talu", 64 * 1024);
    defer std.testing.allocator.free(current_data);
    try std.testing.expect(current_data.len > 0);
    try std.testing.expect(current_data.len < first_block_size + 100);

    // A sealed segment file (seg-*.talu) should exist
    var seg_found = false;
    var iter = tmp.dir.openDir("store", .{ .iterate = true }) catch unreachable;
    defer iter.close();
    var dir_iter = iter.iterate();
    while (try dir_iter.next()) |entry| {
        if (std.mem.startsWith(u8, entry.name, "seg-") and std.mem.endsWith(u8, entry.name, ".talu")) {
            seg_found = true;
            break;
        }
    }
    try std.testing.expect(seg_found);

    // manifest.json should exist and contain one segment
    const manifest_path = try std.fs.path.join(std.testing.allocator, &.{ tmp_path, "store", "manifest.json" });
    defer std.testing.allocator.free(manifest_path);

    var manifest = try db.Manifest.load(std.testing.allocator, manifest_path);
    defer manifest.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u32, 1), manifest.version);
    try std.testing.expectEqual(@as(usize, 1), manifest.segments.len);
    try std.testing.expect(std.mem.startsWith(u8, manifest.segments[0].path, "seg-"));

    // WAL should be cleared after flush
    const wal_data = try tmp.dir.readFileAlloc(std.testing.allocator, "store/current.wal", 1024);
    defer std.testing.allocator.free(wal_data);
    try std.testing.expectEqual(@as(usize, 0), wal_data.len);
}

test "Writer: multiple rotations accumulate manifest segments" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const tmp_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(tmp_path);

    var writer = try Writer.open(std.testing.allocator, tmp_path, "multi");
    defer writer.deinit();

    writer.max_segment_size = 512;

    const col = db.writer.ColumnValue{
        .column_id = 1,
        .shape = .VARBYTES,
        .phys_type = .BINARY,
        .encoding = .RAW,
        .dims = 0,
        .data = "x" ** 256,
    };

    // Three flush cycles: first populates current.talu, second and third each rotate
    var flush_count: usize = 0;
    while (flush_count < 3) : (flush_count += 1) {
        try writer.appendRow(1, &.{col});
        try writer.flushBlock();
    }

    const manifest_path = try std.fs.path.join(std.testing.allocator, &.{ tmp_path, "multi", "manifest.json" });
    defer std.testing.allocator.free(manifest_path);

    var manifest = try db.Manifest.load(std.testing.allocator, manifest_path);
    defer manifest.deinit(std.testing.allocator);

    // Two rotations should have occurred (flush 2 and 3 each seal the previous)
    try std.testing.expectEqual(@as(usize, 2), manifest.segments.len);

    // Each segment should have a distinct ID
    try std.testing.expect(manifest.segments[0].id != manifest.segments[1].id);
}

// ===== deinit =====

test "Writer: deinit releases resources" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const tmp_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(tmp_path);

    var writer = try Writer.open(std.testing.allocator, tmp_path, "chat");
    writer.deinit();

    // Verify resources were released: re-opening the same namespace must succeed
    // (lock file released, allocations freed — std.testing.allocator catches leaks).
    var writer2 = try Writer.open(std.testing.allocator, tmp_path, "chat");
    writer2.deinit();
}
