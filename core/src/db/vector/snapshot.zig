//! Snapshot lifecycle helpers for vector namespaces.

const std = @import("std");

const Allocator = std.mem.Allocator;

const namespaces = [_][]const u8{
    "vector",
    "vector_changes",
    "vector_idempotency",
};

const snapshot_files = [_][]const u8{
    "manifest.json",
    "current.talu",
};

pub fn createSnapshot(allocator: Allocator, db_root: []const u8, snapshot_name: []const u8) !void {
    try validateSnapshotName(snapshot_name);

    const snapshot_root = try std.fs.path.join(allocator, &.{ db_root, "vector_snapshots", snapshot_name });
    defer allocator.free(snapshot_root);
    try std.fs.cwd().makePath(snapshot_root);

    for (namespaces) |namespace| {
        for (snapshot_files) |file_name| {
            const src = try std.fs.path.join(allocator, &.{ db_root, namespace, file_name });
            defer allocator.free(src);

            const bytes = std.fs.cwd().readFileAlloc(allocator, src, 64 * 1024 * 1024) catch |err| switch (err) {
                error.FileNotFound => continue,
                else => return err,
            };
            defer allocator.free(bytes);

            const dst = try std.fs.path.join(allocator, &.{ snapshot_root, namespace, file_name });
            defer allocator.free(dst);
            const parent = std.fs.path.dirname(dst) orelse ".";
            try std.fs.cwd().makePath(parent);
            try std.fs.cwd().writeFile(.{ .sub_path = dst, .data = bytes });
        }
    }
}

pub fn listSnapshots(allocator: Allocator, db_root: []const u8) ![][]u8 {
    const snapshots_dir_path = try std.fs.path.join(allocator, &.{ db_root, "vector_snapshots" });
    defer allocator.free(snapshots_dir_path);

    var dir = std.fs.cwd().openDir(snapshots_dir_path, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => return allocator.alloc([]u8, 0),
        else => return err,
    };
    defer dir.close();

    var names = std.ArrayList([]u8).empty;
    errdefer {
        for (names.items) |name| allocator.free(name);
        names.deinit(allocator);
    }

    var iter = dir.iterate();
    while (try iter.next()) |entry| {
        if (entry.kind != .directory) continue;
        try names.append(allocator, try allocator.dupe(u8, entry.name));
    }

    std.sort.pdq([]u8, names.items, {}, struct {
        fn lessThan(_: void, a: []u8, b: []u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.lessThan);

    return names.toOwnedSlice(allocator);
}

pub fn restoreSnapshot(allocator: Allocator, db_root: []const u8, snapshot_name: []const u8) !void {
    try validateSnapshotName(snapshot_name);

    const snapshot_root = try std.fs.path.join(allocator, &.{ db_root, "vector_snapshots", snapshot_name });
    defer allocator.free(snapshot_root);

    for (namespaces) |namespace| {
        const namespace_dir = try std.fs.path.join(allocator, &.{ db_root, namespace });
        defer allocator.free(namespace_dir);
        try std.fs.cwd().makePath(namespace_dir);

        for (snapshot_files) |file_name| {
            const src = try std.fs.path.join(allocator, &.{ snapshot_root, namespace, file_name });
            defer allocator.free(src);

            const bytes = std.fs.cwd().readFileAlloc(allocator, src, 64 * 1024 * 1024) catch |err| switch (err) {
                error.FileNotFound => continue,
                else => return err,
            };
            defer allocator.free(bytes);

            const dst = try std.fs.path.join(allocator, &.{ db_root, namespace, file_name });
            defer allocator.free(dst);
            try std.fs.cwd().writeFile(.{ .sub_path = dst, .data = bytes });
        }
    }
}

fn validateSnapshotName(snapshot_name: []const u8) !void {
    if (snapshot_name.len == 0) return error.InvalidColumnData;
    for (snapshot_name) |ch| {
        if (ch == '/' or ch == '\\' or ch == 0) return error.InvalidColumnData;
    }
}

test "createSnapshot and restoreSnapshot round-trip manifests" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    try tmp.dir.makePath("vector");
    try tmp.dir.writeFile(.{ .sub_path = "vector/manifest.json", .data = "{\"version\":1,\"segments\":[]}" });
    try tmp.dir.writeFile(.{ .sub_path = "vector/current.talu", .data = "abc" });
    try tmp.dir.makePath("vector_changes");
    try tmp.dir.writeFile(.{ .sub_path = "vector_changes/manifest.json", .data = "{\"version\":1,\"segments\":[{\"x\":1}]}" });
    try tmp.dir.writeFile(.{ .sub_path = "vector_changes/current.talu", .data = "def" });

    try createSnapshot(std.testing.allocator, root, "snap1");

    try tmp.dir.writeFile(.{ .sub_path = "vector/manifest.json", .data = "{\"version\":999}" });
    try tmp.dir.writeFile(.{ .sub_path = "vector/current.talu", .data = "zzz" });
    try restoreSnapshot(std.testing.allocator, root, "snap1");

    const restored = try tmp.dir.readFileAlloc(std.testing.allocator, "vector/manifest.json", 4096);
    defer std.testing.allocator.free(restored);
    try std.testing.expect(std.mem.indexOf(u8, restored, "\"version\":1") != null);

    const restored_current = try tmp.dir.readFileAlloc(std.testing.allocator, "vector/current.talu", 4096);
    defer std.testing.allocator.free(restored_current);
    try std.testing.expectEqualStrings("abc", restored_current);
}

test "listSnapshots returns sorted snapshot names" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.makePath("vector_snapshots/zeta");
    try tmp.dir.makePath("vector_snapshots/alpha");

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    const names = try listSnapshots(std.testing.allocator, root);
    defer {
        for (names) |name| std.testing.allocator.free(name);
        std.testing.allocator.free(names);
    }

    try std.testing.expectEqual(@as(usize, 2), names.len);
    try std.testing.expectEqualStrings("alpha", names[0]);
    try std.testing.expectEqualStrings("zeta", names[1]);
}
