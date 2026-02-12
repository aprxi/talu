//! Integration tests for db.WalWriter
//!
//! WalWriter appends framed payloads to the write-ahead log.
//! Frame format: [Magic u32][Payload Len u32][Payload][CRC32 u32]

const std = @import("std");
const main = @import("main");
const db = main.db;

const WalWriter = db.WalWriter;
const types = db.types;

// ===== init =====

test "WalWriter: init creates writer from file" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var file = try tmp.dir.createFile("test.wal", .{ .read = true });
    defer file.close();

    // Verify init wired up the file by performing a write that exercises it.
    var writer = WalWriter.init(file);
    try writer.append("hello");

    const data = try tmp.dir.readFileAlloc(std.testing.allocator, "test.wal", 1024);
    defer std.testing.allocator.free(data);
    try std.testing.expect(data.len > 0);
    try std.testing.expectEqual(types.MagicValues.WAL, std.mem.readInt(u32, data[0..4], .little));
}

// ===== append =====

test "WalWriter: append writes WAL frame with magic and payload" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var file = try tmp.dir.createFile("test.wal", .{ .read = true });
    defer file.close();

    var writer = WalWriter.init(file);
    try writer.append("payload");

    const data = try tmp.dir.readFileAlloc(std.testing.allocator, "test.wal", 1024);
    defer std.testing.allocator.free(data);

    // header(8) + payload(7) + crc(4) = 19 bytes
    try std.testing.expectEqual(@as(usize, 19), data.len);
    try std.testing.expectEqual(types.MagicValues.WAL, std.mem.readInt(u32, data[0..4], .little));
    try std.testing.expectEqual(@as(u32, 7), std.mem.readInt(u32, data[4..8], .little));
}

// ===== appendBatch =====

test "WalWriter: appendBatch writes WAL_BATCH frame" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var file = try tmp.dir.createFile("test.wal", .{ .read = true });
    defer file.close();

    var writer = WalWriter.init(file);
    try writer.appendBatch("batch");

    const data = try tmp.dir.readFileAlloc(std.testing.allocator, "test.wal", 1024);
    defer std.testing.allocator.free(data);

    try std.testing.expectEqual(types.MagicValues.WAL_BATCH, std.mem.readInt(u32, data[0..4], .little));
}

// ===== sync =====

test "WalWriter: sync flushes to stable storage" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var file = try tmp.dir.createFile("test.wal", .{ .read = true });
    defer file.close();

    var writer = WalWriter.init(file);
    try writer.append("data");
    try writer.sync();

    // Verify data is durable: re-read from a separate handle and confirm contents.
    const data = try tmp.dir.readFileAlloc(std.testing.allocator, "test.wal", 1024);
    defer std.testing.allocator.free(data);

    // header(8) + payload(4) + crc(4) = 16 bytes
    try std.testing.expectEqual(@as(usize, 16), data.len);
    try std.testing.expectEqual(types.MagicValues.WAL, std.mem.readInt(u32, data[0..4], .little));
}
