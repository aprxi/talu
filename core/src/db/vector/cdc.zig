//! CDC cursor primitives for deterministic resume behavior.

const std = @import("std");

const Allocator = std.mem.Allocator;

pub const Cursor = struct {
    since: u64,
    generation: u64 = 0,
};

pub fn encodeCursor(allocator: Allocator, cursor: Cursor) ![]u8 {
    return std.fmt.allocPrint(allocator, "v1:{d}:{d}", .{ cursor.since, cursor.generation });
}

pub fn decodeCursor(raw: []const u8) !Cursor {
    if (raw.len == 0) return .{ .since = 0 };

    // Backward compatible path for legacy integer cursors.
    if (!std.mem.startsWith(u8, raw, "v1:")) {
        const since = try std.fmt.parseUnsigned(u64, raw, 10);
        return .{ .since = since };
    }

    var parts = std.mem.splitScalar(u8, raw, ':');
    _ = parts.next() orelse return error.InvalidColumnData; // version prefix
    const since_raw = parts.next() orelse return error.InvalidColumnData;
    const generation_raw = parts.next() orelse return error.InvalidColumnData;
    if (parts.next() != null) return error.InvalidColumnData;

    return .{
        .since = try std.fmt.parseUnsigned(u64, since_raw, 10),
        .generation = try std.fmt.parseUnsigned(u64, generation_raw, 10),
    };
}

pub fn nextCursor(last_seen_seq: ?u64, current: Cursor, generation: u64) Cursor {
    var next = Cursor{
        .since = current.since,
        .generation = generation,
    };
    if (last_seen_seq) |seq| {
        if (seq > next.since) next.since = seq;
    }
    return next;
}

pub const CursorStore = struct {
    allocator: Allocator,
    path: []u8,

    pub fn init(allocator: Allocator, path: []const u8) !CursorStore {
        return .{
            .allocator = allocator,
            .path = try allocator.dupe(u8, path),
        };
    }

    pub fn deinit(self: *CursorStore) void {
        self.allocator.free(self.path);
        self.path = &[_]u8{};
    }

    pub fn load(self: *CursorStore) !Cursor {
        const bytes = std.fs.cwd().readFileAlloc(self.allocator, self.path, 1024) catch |err| switch (err) {
            error.FileNotFound => return .{ .since = 0, .generation = 0 },
            else => return err,
        };
        defer self.allocator.free(bytes);
        return decodeCursor(std.mem.trim(u8, bytes, " \r\n\t"));
    }

    pub fn save(self: *CursorStore, cursor: Cursor) !void {
        const parent = std.fs.path.dirname(self.path) orelse ".";
        try std.fs.cwd().makePath(parent);
        const encoded = try encodeCursor(self.allocator, cursor);
        defer self.allocator.free(encoded);
        const tmp_path = try std.fmt.allocPrint(self.allocator, "{s}.tmp", .{self.path});
        defer self.allocator.free(tmp_path);

        var tmp = try std.fs.cwd().createFile(tmp_path, .{ .truncate = true });
        errdefer std.fs.cwd().deleteFile(tmp_path) catch {};
        try tmp.writeAll(encoded);
        try tmp.sync();
        tmp.close();

        try std.fs.cwd().rename(tmp_path, self.path);
    }
};

test "encodeCursor and decodeCursor round-trip values" {
    const encoded = try encodeCursor(std.testing.allocator, .{ .since = 42, .generation = 7 });
    defer std.testing.allocator.free(encoded);

    const decoded = try decodeCursor(encoded);
    try std.testing.expectEqual(@as(u64, 42), decoded.since);
    try std.testing.expectEqual(@as(u64, 7), decoded.generation);
}

test "decodeCursor accepts empty cursor as zero" {
    const decoded = try decodeCursor("");
    try std.testing.expectEqual(@as(u64, 0), decoded.since);
    try std.testing.expectEqual(@as(u64, 0), decoded.generation);
}

test "nextCursor advances only when sequence increases" {
    const current = Cursor{ .since = 10 };
    try std.testing.expectEqual(@as(u64, 12), nextCursor(12, current, 4).since);
    try std.testing.expectEqual(@as(u64, 10), nextCursor(9, current, 4).since);
    try std.testing.expectEqual(@as(u64, 10), nextCursor(null, current, 4).since);
    try std.testing.expectEqual(@as(u64, 4), nextCursor(null, current, 4).generation);
}

test "decodeCursor accepts legacy numeric cursor format" {
    const decoded = try decodeCursor("77");
    try std.testing.expectEqual(@as(u64, 77), decoded.since);
    try std.testing.expectEqual(@as(u64, 0), decoded.generation);
}

test "CursorStore.save and CursorStore.load persist cursor atomically" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);
    const cursor_path = try std.fs.path.join(std.testing.allocator, &.{ root, "cdc", "cursor.txt" });
    defer std.testing.allocator.free(cursor_path);

    var store = try CursorStore.init(std.testing.allocator, cursor_path);
    defer store.deinit();

    const initial = try store.load();
    try std.testing.expectEqual(@as(u64, 0), initial.since);

    try store.save(.{ .since = 123, .generation = 5 });
    const loaded = try store.load();
    try std.testing.expectEqual(@as(u64, 123), loaded.since);
    try std.testing.expectEqual(@as(u64, 5), loaded.generation);
}
