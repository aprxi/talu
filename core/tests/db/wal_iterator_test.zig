//! Integration tests for db.WalIterator
//!
//! WalIterator reads framed payloads from the WAL, expanding batch frames
//! into individual payloads.

const std = @import("std");
const main = @import("main");
const db = main.db;

const WalIterator = db.WalIterator;
const WalWriter = db.WalWriter;
const types = db.types;

// ===== init =====

test "WalIterator: init starts at offset zero" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var file = try tmp.dir.createFile("test.wal", .{ .read = true });
    defer file.close();

    const iter = WalIterator.init(file, std.testing.allocator);
    try std.testing.expectEqual(@as(u64, 0), iter.offset);
}

// ===== next =====

test "WalIterator: next reads sequential payloads" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var file = try tmp.dir.createFile("test.wal", .{ .read = true });
    defer file.close();

    var writer = WalWriter.init(file);
    try writer.append("one");
    try writer.append("two");

    var iter = WalIterator.init(file, std.testing.allocator);
    const first = (try iter.next()).?;
    defer std.testing.allocator.free(first);
    const second = (try iter.next()).?;
    defer std.testing.allocator.free(second);
    const third = try iter.next();

    try std.testing.expectEqualSlices(u8, "one", first);
    try std.testing.expectEqualSlices(u8, "two", second);
    try std.testing.expect(third == null);
}

test "WalIterator: next expands batch payloads" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var file = try tmp.dir.createFile("test.wal", .{ .read = true });
    defer file.close();

    var writer = WalWriter.init(file);

    // Build batch frame: [count:u32][len:u32][payload][len:u32][payload]
    const payloads = [_][]const u8{ "alpha", "beta" };
    var batch = std.ArrayList(u8).empty;
    defer batch.deinit(std.testing.allocator);

    var count_buf: [4]u8 = undefined;
    std.mem.writeInt(u32, count_buf[0..4], @intCast(payloads.len), .little);
    try batch.appendSlice(std.testing.allocator, &count_buf);

    for (payloads) |payload| {
        var len_buf: [4]u8 = undefined;
        std.mem.writeInt(u32, len_buf[0..4], @intCast(payload.len), .little);
        try batch.appendSlice(std.testing.allocator, &len_buf);
        try batch.appendSlice(std.testing.allocator, payload);
    }

    try writer.appendBatch(batch.items);

    var iter = WalIterator.init(file, std.testing.allocator);
    const first = (try iter.next()).?;
    defer std.testing.allocator.free(first);
    const second = (try iter.next()).?;
    defer std.testing.allocator.free(second);
    const third = try iter.next();

    try std.testing.expectEqualSlices(u8, "alpha", first);
    try std.testing.expectEqualSlices(u8, "beta", second);
    try std.testing.expect(third == null);
}

test "WalIterator: next returns null on empty file" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var file = try tmp.dir.createFile("test.wal", .{ .read = true });
    defer file.close();

    var iter = WalIterator.init(file, std.testing.allocator);
    const result = try iter.next();
    try std.testing.expect(result == null);
}

test "WalIterator: next returns null on partial frame" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var file = try tmp.dir.createFile("test.wal", .{ .read = true });
    defer file.close();

    // Write truncated frame: header says 100 bytes but file is short
    var header: [8]u8 = undefined;
    std.mem.writeInt(u32, header[0..4], types.MagicValues.WAL, .little);
    std.mem.writeInt(u32, header[4..8], 100, .little);
    try file.writeAll(&header);

    var iter = WalIterator.init(file, std.testing.allocator);
    const result = try iter.next();
    try std.testing.expect(result == null);
}
