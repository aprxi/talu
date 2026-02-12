//! StoreFS write-ahead log (WAL) implementation.
//!
//! Frame format: [Magic u32][Payload Len u32][Payload][CRC32 u32]

const std = @import("std");
const checksum = @import("checksum.zig");
const types = @import("types.zig");

const Allocator = std.mem.Allocator;

pub const WalWriter = struct {
    file: std.fs.File,

    pub fn init(file: std.fs.File) WalWriter {
        return .{ .file = file };
    }

    /// Appends a payload to the WAL with framing and CRC32C.
    pub fn append(self: *WalWriter, payload: []const u8) !void {
        try self.appendFrame(types.MagicValues.WAL, payload);
    }

    /// Appends a batch payload to the WAL with framing and CRC32C.
    pub fn appendBatch(self: *WalWriter, payload: []const u8) !void {
        try self.appendFrame(types.MagicValues.WAL_BATCH, payload);
    }

    /// Flushes WAL contents to stable storage.
    pub fn sync(self: *WalWriter) !void {
        try self.file.sync();
    }

    fn appendFrame(self: *WalWriter, magic: u32, payload: []const u8) !void {
        var header: [8]u8 = undefined;
        std.mem.writeInt(u32, header[0..4], magic, .little);
        if (payload.len > std.math.maxInt(u32)) return error.PayloadTooLarge;
        std.mem.writeInt(u32, header[4..8], @intCast(payload.len), .little);

        const crc = checksum.crc32c(payload);
        var crc_buf: [4]u8 = undefined;
        std.mem.writeInt(u32, crc_buf[0..4], crc, .little);

        try self.file.writeAll(&header);
        try self.file.writeAll(payload);
        try self.file.writeAll(&crc_buf);
    }
};

pub const WalIterator = struct {
    file: std.fs.File,
    allocator: Allocator,
    offset: u64,
    batch_buf: ?[]u8,
    batch_offset: usize,
    batch_remaining: u32,

    pub fn init(file: std.fs.File, allocator: Allocator) WalIterator {
        return .{
            .file = file,
            .allocator = allocator,
            .offset = 0,
            .batch_buf = null,
            .batch_offset = 0,
            .batch_remaining = 0,
        };
    }

    /// Returns the next payload, or null when the stream ends.
    /// Caller owns the returned slice.
    pub fn next(self: *WalIterator) !?[]u8 {
        if (self.batch_remaining > 0) {
            return try self.nextBatchPayload();
        }

        var header: [8]u8 = undefined;
        const header_read = try self.file.preadAll(&header, self.offset);
        if (header_read != header.len) return null;

        const magic = std.mem.readInt(u32, header[0..4], .little);
        if (magic != types.MagicValues.WAL and magic != types.MagicValues.WAL_BATCH) {
            return error.InvalidMagic;
        }

        const payload_len = std.mem.readInt(u32, header[4..8], .little);
        self.offset += header.len;

        const payload = try self.allocator.alloc(u8, payload_len);

        const payload_read = try self.file.preadAll(payload, self.offset);
        if (payload_read != payload.len) {
            self.allocator.free(payload);
            return null;
        }

        self.offset += payload.len;

        var crc_buf: [4]u8 = undefined;
        const crc_read = try self.file.preadAll(&crc_buf, self.offset);
        if (crc_read != crc_buf.len) {
            self.allocator.free(payload);
            return null;
        }

        self.offset += crc_buf.len;

        const expected_crc = std.mem.readInt(u32, crc_buf[0..4], .little);
        const actual_crc = checksum.crc32c(payload);
        if (expected_crc != actual_crc) {
            self.allocator.free(payload);
            return error.InvalidCrc;
        }

        if (magic == types.MagicValues.WAL) {
            return payload;
        }

        if (payload.len < 4) {
            self.allocator.free(payload);
            return error.InvalidWalEntry;
        }

        const count = std.mem.readInt(u32, payload[0..4], .little);
        if (count == 0) {
            self.allocator.free(payload);
            return error.InvalidWalEntry;
        }

        self.batch_buf = payload;
        self.batch_offset = 4;
        self.batch_remaining = count;
        return try self.nextBatchPayload();
    }

    fn nextBatchPayload(self: *WalIterator) !?[]u8 {
        if (self.batch_remaining == 0) return null;
        const payload = self.batch_buf orelse return error.InvalidWalEntry;
        errdefer {
            self.allocator.free(payload);
            self.batch_buf = null;
            self.batch_remaining = 0;
            self.batch_offset = 0;
        }

        var index = self.batch_offset;
        if (index + 4 > payload.len) return error.InvalidWalEntry;
        const entry_len = std.mem.readInt(u32, payload[index..][0..4], .little);
        index += 4;
        const end = index + @as(usize, entry_len);
        if (end > payload.len) return error.InvalidWalEntry;

        const out = try self.allocator.alloc(u8, entry_len);
        std.mem.copyForwards(u8, out, payload[index..end]);

        self.batch_offset = end;
        self.batch_remaining -= 1;

        if (self.batch_remaining == 0) {
            self.allocator.free(payload);
            self.batch_buf = null;
        }

        return out;
    }
};

test "WalWriter.init stores file" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var file = try tmp.dir.createFile("test.wal", .{ .read = true });
    defer file.close();

    const writer = WalWriter.init(file);
    _ = writer;
}

test "WalWriter.append writes frame" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var file = try tmp.dir.createFile("test.wal", .{ .read = true });
    defer file.close();

    var writer = WalWriter.init(file);
    try writer.append("payload");

    const data = try tmp.dir.readFileAlloc(std.testing.allocator, "test.wal", 1024);
    defer std.testing.allocator.free(data);

    try std.testing.expectEqual(@as(usize, 8 + "payload".len + 4), data.len);
    try std.testing.expectEqual(types.MagicValues.WAL, std.mem.readInt(u32, data[0..4], .little));
    try std.testing.expectEqual(@as(u32, "payload".len), std.mem.readInt(u32, data[4..8], .little));
}

test "WalWriter.sync flushes" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var file = try tmp.dir.createFile("test.wal", .{ .read = true });
    defer file.close();

    var writer = WalWriter.init(file);
    try writer.append("payload");
    try writer.sync();
}

test "WalIterator.init sets offset" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var file = try tmp.dir.createFile("test.wal", .{ .read = true });
    defer file.close();

    const iter = WalIterator.init(file, std.testing.allocator);
    try std.testing.expectEqual(@as(u64, 0), iter.offset);
}

test "WalIterator.next returns payloads" {
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

test "WalIterator.next expands batch payloads" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var file = try tmp.dir.createFile("test.wal", .{ .read = true });
    defer file.close();

    var writer = WalWriter.init(file);

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

test "WalIterator.next returns null on partial frame" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var file = try tmp.dir.createFile("test.wal", .{ .read = true });
    defer file.close();

    var header: [8]u8 = undefined;
    std.mem.writeInt(u32, header[0..4], types.MagicValues.WAL, .little);
    std.mem.writeInt(u32, header[4..8], 100, .little);
    try file.writeAll(&header);

    var iter = WalIterator.init(file, std.testing.allocator);
    const result = try iter.next();
    try std.testing.expect(result == null);
}
