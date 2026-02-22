//! Object-backed segment source with local cache materialization.
//!
//! This implementation uses object-store range reads to populate a local cached
//! copy, then serves reader requests from the cached immutable file.

const std = @import("std");
const db_cache = @import("cache.zig");
const segment_source = @import("segment_source.zig");

const Allocator = std.mem.Allocator;

pub const ObjectRangeReader = struct {
    context: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        size: *const fn (context: *anyopaque, object_key: []const u8) anyerror!u64,
        readAt: *const fn (context: *anyopaque, object_key: []const u8, offset: u64, dest: []u8) anyerror!usize,
    };

    pub fn size(self: ObjectRangeReader, object_key: []const u8) !u64 {
        return self.vtable.size(self.context, object_key);
    }

    pub fn readAt(self: ObjectRangeReader, object_key: []const u8, offset: u64, dest: []u8) !usize {
        return self.vtable.readAt(self.context, object_key, offset, dest);
    }
};

pub const ObjectSegmentSource = struct {
    allocator: Allocator,
    cache: db_cache.ObjectCache,
    reader: ObjectRangeReader,
    chunk_size: usize,
    expected_checksums: std.StringHashMapUnmanaged(u32),

    pub fn init(
        allocator: Allocator,
        cache_root: []const u8,
        reader: ObjectRangeReader,
    ) !ObjectSegmentSource {
        return .{
            .allocator = allocator,
            .cache = try db_cache.ObjectCache.init(allocator, cache_root),
            .reader = reader,
            .chunk_size = 1024 * 1024,
            .expected_checksums = .empty,
        };
    }

    pub fn deinit(self: *ObjectSegmentSource) void {
        var iter = self.expected_checksums.keyIterator();
        while (iter.next()) |key_ptr| {
            self.allocator.free(key_ptr.*);
        }
        self.expected_checksums.deinit(self.allocator);
        self.cache.deinit();
    }

    /// Registers an expected CRC32C checksum for a future materialized object.
    pub fn setExpectedChecksum(self: *ObjectSegmentSource, object_key: []const u8, checksum_crc32c: u32) !void {
        const key_copy = try self.allocator.dupe(u8, object_key);
        errdefer self.allocator.free(key_copy);
        if (try self.expected_checksums.fetchPut(self.allocator, key_copy, checksum_crc32c)) |prev| {
            self.allocator.free(prev.key);
        }
    }

    pub fn asSource(self: *ObjectSegmentSource) segment_source.SegmentSource {
        return .{
            .context = self,
            .vtable = &.{
                .openReadOnly = openReadOnly,
                .close = close,
            },
        };
    }

    fn openReadOnly(context: *anyopaque, rel_path: []const u8) !segment_source.SegmentHandle {
        const self: *ObjectSegmentSource = @ptrCast(@alignCast(context));
        const cache_rel = try self.cache.objectPath(self.allocator, rel_path);
        defer self.allocator.free(cache_rel);

        if (!self.cache.contains(cache_rel)) {
            try self.materializeToCache(rel_path, cache_rel);
        }

        const file = try self.cache.openReadOnly(cache_rel);
        errdefer file.close();
        const stat = try file.stat();
        return .{
            .file = file,
            .size = stat.size,
        };
    }

    fn close(_: *anyopaque, handle: *segment_source.SegmentHandle) void {
        handle.file.close();
    }

    fn materializeToCache(self: *ObjectSegmentSource, object_key: []const u8, cache_rel: []const u8) !void {
        const size = try self.reader.size(object_key);
        const tmp_rel = try std.fmt.allocPrint(self.allocator, "{s}.tmp", .{cache_rel});
        defer self.allocator.free(tmp_rel);

        var tmp = try self.cache.createWrite(tmp_rel);
        defer tmp.close();
        errdefer self.cache.dir.deleteFile(tmp_rel) catch {};

        var buf = try self.allocator.alloc(u8, self.chunk_size);
        defer self.allocator.free(buf);
        var crc = std.hash.crc.Crc32Iscsi.init();

        var offset: u64 = 0;
        while (offset < size) {
            const remaining = size - offset;
            const to_read = @min(@as(usize, @intCast(remaining)), buf.len);
            const read_len = try self.reader.readAt(object_key, offset, buf[0..to_read]);
            if (read_len != to_read) return error.UnexpectedEof;
            try tmp.writeAll(buf[0..read_len]);
            crc.update(buf[0..read_len]);
            offset += read_len;
        }

        if (self.expected_checksums.get(object_key)) |expected| {
            const actual = crc.final();
            if (actual != expected) return error.InvalidChecksum;
        }

        try tmp.sync();
        tmp.close();
        try self.cache.rename(tmp_rel, cache_rel);
    }
};

test "ObjectSegmentSource.asSource materializes once then serves from cache" {
    const Fixture = struct {
        const Self = @This();

        objects: std.StringHashMapUnmanaged([]const u8) = .empty,
        read_calls: usize = 0,

        fn size(context: *anyopaque, object_key: []const u8) !u64 {
            const self: *Self = @ptrCast(@alignCast(context));
            const bytes = self.objects.get(object_key) orelse return error.FileNotFound;
            return bytes.len;
        }

        fn readAt(context: *anyopaque, object_key: []const u8, offset: u64, dest: []u8) !usize {
            const self: *Self = @ptrCast(@alignCast(context));
            const bytes = self.objects.get(object_key) orelse return error.FileNotFound;
            const start = @as(usize, @intCast(offset));
            const end = start + dest.len;
            if (end > bytes.len) return error.UnexpectedEof;
            std.mem.copyForwards(u8, dest, bytes[start..end]);
            self.read_calls += 1;
            return dest.len;
        }
    };

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const cache_root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(cache_root);

    var fixture = Fixture{};
    defer fixture.objects.deinit(std.testing.allocator);
    try fixture.objects.put(std.testing.allocator, "tenant-a/seg-1.talu", "abcdef");

    var source_impl = try ObjectSegmentSource.init(
        std.testing.allocator,
        cache_root,
        .{
            .context = &fixture,
            .vtable = &.{
                .size = Fixture.size,
                .readAt = Fixture.readAt,
            },
        },
    );
    defer source_impl.deinit();

    const source = source_impl.asSource();
    var first = try source.openReadOnly("tenant-a/seg-1.talu");
    var buf: [6]u8 = undefined;
    _ = try first.file.preadAll(&buf, 0);
    try std.testing.expectEqualSlices(u8, "abcdef", &buf);
    source.close(&first);
    const calls_after_first = fixture.read_calls;
    try std.testing.expect(calls_after_first > 0);

    var second = try source.openReadOnly("tenant-a/seg-1.talu");
    _ = try second.file.preadAll(&buf, 0);
    source.close(&second);
    try std.testing.expectEqual(calls_after_first, fixture.read_calls);
}

test "ObjectRangeReader.size and ObjectRangeReader.readAt dispatch to vtable" {
    const Fixture = struct {
        fn size(_: *anyopaque, object_key: []const u8) !u64 {
            if (!std.mem.eql(u8, object_key, "obj")) return error.FileNotFound;
            return 4;
        }

        fn readAt(_: *anyopaque, object_key: []const u8, offset: u64, dest: []u8) !usize {
            if (!std.mem.eql(u8, object_key, "obj")) return error.FileNotFound;
            const bytes = "wxyz";
            const start = @as(usize, @intCast(offset));
            const end = start + dest.len;
            if (end > bytes.len) return error.UnexpectedEof;
            std.mem.copyForwards(u8, dest, bytes[start..end]);
            return dest.len;
        }
    };

    const reader = ObjectRangeReader{
        .context = undefined,
        .vtable = &.{
            .size = Fixture.size,
            .readAt = Fixture.readAt,
        },
    };

    try std.testing.expectEqual(@as(u64, 4), try reader.size("obj"));
    var buf: [2]u8 = undefined;
    const n = try reader.readAt("obj", 1, &buf);
    try std.testing.expectEqual(@as(usize, 2), n);
    try std.testing.expectEqualSlices(u8, "xy", &buf);
}

test "ObjectSegmentSource.init and ObjectSegmentSource.deinit lifecycle" {
    const Fixture = struct {
        fn size(_: *anyopaque, _: []const u8) !u64 {
            return 0;
        }

        fn readAt(_: *anyopaque, _: []const u8, _: u64, dest: []u8) !usize {
            return dest.len;
        }
    };

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const cache_root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(cache_root);

    var source = try ObjectSegmentSource.init(
        std.testing.allocator,
        cache_root,
        .{
            .context = undefined,
            .vtable = &.{
                .size = Fixture.size,
                .readAt = Fixture.readAt,
            },
        },
    );
    source.deinit();
}

test "ObjectSegmentSource.materializeToCache validates expected checksum when registered" {
    const Fixture = struct {
        fn size(_: *anyopaque, _: []const u8) !u64 {
            return 3;
        }

        fn readAt(_: *anyopaque, _: []const u8, offset: u64, dest: []u8) !usize {
            const bytes = "abc";
            const start = @as(usize, @intCast(offset));
            const end = start + dest.len;
            if (end > bytes.len) return error.UnexpectedEof;
            std.mem.copyForwards(u8, dest, bytes[start..end]);
            return dest.len;
        }
    };

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const cache_root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(cache_root);

    var source = try ObjectSegmentSource.init(
        std.testing.allocator,
        cache_root,
        .{
            .context = undefined,
            .vtable = &.{
                .size = Fixture.size,
                .readAt = Fixture.readAt,
            },
        },
    );
    defer source.deinit();

    // Force a mismatch and verify reads fail deterministically.
    try source.setExpectedChecksum("obj", 0);
    const err = source.asSource().openReadOnly("obj");
    try std.testing.expectError(error.InvalidChecksum, err);
}

test "ObjectSegmentSource.asSource retries cleanly after partial read failure" {
    const Fixture = struct {
        const Self = @This();
        partial_once: bool = true,

        fn size(_: *anyopaque, _: []const u8) !u64 {
            return 6;
        }

        fn readAt(context: *anyopaque, _: []const u8, offset: u64, dest: []u8) !usize {
            const self: *Self = @ptrCast(@alignCast(context));
            const bytes = "abcdef";
            const start = @as(usize, @intCast(offset));
            const end = start + dest.len;
            if (end > bytes.len) return error.UnexpectedEof;
            std.mem.copyForwards(u8, dest, bytes[start..end]);
            if (self.partial_once) {
                self.partial_once = false;
                return dest.len - 1; // short read simulates eventual/partial object fetch
            }
            return dest.len;
        }
    };

    var fixture = Fixture{};
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const cache_root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(cache_root);

    var source = try ObjectSegmentSource.init(
        std.testing.allocator,
        cache_root,
        .{
            .context = &fixture,
            .vtable = &.{
                .size = Fixture.size,
                .readAt = Fixture.readAt,
            },
        },
    );
    defer source.deinit();

    try std.testing.expectError(error.UnexpectedEof, source.asSource().openReadOnly("obj"));
    const rel = try source.cache.objectPath(std.testing.allocator, "obj");
    defer std.testing.allocator.free(rel);
    try std.testing.expect(!source.cache.contains(rel));

    var handle = try source.asSource().openReadOnly("obj");
    defer source.asSource().close(&handle);
    var buf: [6]u8 = undefined;
    _ = try handle.file.preadAll(&buf, 0);
    try std.testing.expectEqualSlices(u8, "abcdef", &buf);
}
