//! Local on-disk cache utilities for object-backed segments.

const std = @import("std");

const Allocator = std.mem.Allocator;

pub const ObjectCache = struct {
    allocator: Allocator,
    dir: std.fs.Dir,
    root_path: []u8,

    pub fn init(allocator: Allocator, root_path: []const u8) !ObjectCache {
        var dir = try std.fs.cwd().makeOpenPath(root_path, .{});
        errdefer dir.close();
        return .{
            .allocator = allocator,
            .dir = dir,
            .root_path = try allocator.dupe(u8, root_path),
        };
    }

    pub fn deinit(self: *ObjectCache) void {
        self.dir.close();
        self.allocator.free(self.root_path);
        self.root_path = &[_]u8{};
    }

    pub fn objectPath(self: *ObjectCache, allocator: Allocator, object_key: []const u8) ![]u8 {
        _ = self;
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(object_key);
        const hash = hasher.final();
        return std.fmt.allocPrint(allocator, "{x}.obj", .{hash});
    }

    pub fn contains(self: *ObjectCache, rel_path: []const u8) bool {
        _ = self.dir.statFile(rel_path) catch return false;
        return true;
    }

    pub fn openReadOnly(self: *ObjectCache, rel_path: []const u8) !std.fs.File {
        return self.dir.openFile(rel_path, .{ .mode = .read_only });
    }

    pub fn createWrite(self: *ObjectCache, rel_path: []const u8) !std.fs.File {
        return self.dir.createFile(rel_path, .{ .truncate = true, .read = true });
    }

    pub fn rename(self: *ObjectCache, old_rel_path: []const u8, new_rel_path: []const u8) !void {
        try self.dir.rename(old_rel_path, new_rel_path);
    }
};

test "ObjectCache.objectPath is deterministic per object key" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    var cache = try ObjectCache.init(std.testing.allocator, root);
    defer cache.deinit();

    const path_a = try cache.objectPath(std.testing.allocator, "a/seg-1.talu");
    defer std.testing.allocator.free(path_a);
    const path_b = try cache.objectPath(std.testing.allocator, "a/seg-1.talu");
    defer std.testing.allocator.free(path_b);

    try std.testing.expectEqualStrings(path_a, path_b);
}

test "ObjectCache.contains reflects cached file presence" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    var cache = try ObjectCache.init(std.testing.allocator, root);
    defer cache.deinit();

    try std.testing.expect(!cache.contains("seg.obj"));
    var file = try cache.createWrite("seg.obj");
    try file.writeAll("ok");
    file.close();
    try std.testing.expect(cache.contains("seg.obj"));
}

test "ObjectCache.init and ObjectCache.deinit open and close cache root" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    var cache = try ObjectCache.init(std.testing.allocator, root);
    cache.deinit();
}

test "ObjectCache.createWrite and ObjectCache.openReadOnly round-trip bytes" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    var cache = try ObjectCache.init(std.testing.allocator, root);
    defer cache.deinit();

    var write_file = try cache.createWrite("write.obj");
    try write_file.writeAll("abc");
    write_file.close();

    var read_file = try cache.openReadOnly("write.obj");
    defer read_file.close();
    var buf: [3]u8 = undefined;
    _ = try read_file.preadAll(&buf, 0);
    try std.testing.expectEqualSlices(u8, "abc", &buf);
}

test "ObjectCache.rename moves cached files" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    var cache = try ObjectCache.init(std.testing.allocator, root);
    defer cache.deinit();

    var file = try cache.createWrite("old.obj");
    try file.writeAll("1");
    file.close();

    try cache.rename("old.obj", "new.obj");
    try std.testing.expect(!cache.contains("old.obj"));
    try std.testing.expect(cache.contains("new.obj"));
}
