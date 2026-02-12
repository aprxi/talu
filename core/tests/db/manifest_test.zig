//! Integration tests for db.Manifest
//!
//! Manifest is the root of trust for sealed (immutable) segments.
//! Stores version, segment metadata, and compaction timestamp as JSON.

const std = @import("std");
const main = @import("main");
const db = main.db;

const Manifest = db.Manifest;
const manifest = db.manifest;

// ===== load =====

test "Manifest: load parses valid manifest JSON" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const json =
        \\{"version":1,"last_compaction_ts":500,"segments":[{"id":"01234567-89ab-cdef-0123-456789abcdef","path":"seg-1.talu","min_ts":0,"max_ts":100,"row_count":10}]}
    ;
    try tmp.dir.writeFile(.{ .sub_path = "manifest.json", .data = json });

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "manifest.json");
    defer std.testing.allocator.free(path);

    var m = try Manifest.load(std.testing.allocator, path);
    defer m.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u32, 1), m.version);
    try std.testing.expectEqual(@as(i64, 500), m.last_compaction_ts);
    try std.testing.expectEqual(@as(usize, 1), m.segments.len);
    try std.testing.expectEqual(@as(u64, 10), m.segments[0].row_count);
}

test "Manifest: load returns error for missing file" {
    const result = Manifest.load(std.testing.allocator, "/nonexistent/path/manifest.json");
    try std.testing.expectError(error.FileNotFound, result);
}

// ===== save =====

test "Manifest: save writes JSON that load can read back" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(path);

    const manifest_path = try std.fs.path.join(std.testing.allocator, &.{ path, "manifest.json" });
    defer std.testing.allocator.free(manifest_path);

    // Create and save
    var segments = try std.testing.allocator.alloc(manifest.SegmentEntry, 1);
    segments[0] = .{
        .id = 0xAABBCCDDEEFF00110011223344556677,
        .path = try std.testing.allocator.dupe(u8, "seg-0.talu"),
        .min_ts = 10,
        .max_ts = 20,
        .row_count = 5,
    };

    var m = Manifest{
        .version = 1,
        .segments = segments,
        .last_compaction_ts = 42,
    };
    try m.save(manifest_path);
    m.deinit(std.testing.allocator);

    // Load back and verify
    var loaded = try Manifest.load(std.testing.allocator, manifest_path);
    defer loaded.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u32, 1), loaded.version);
    try std.testing.expectEqual(@as(i64, 42), loaded.last_compaction_ts);
    try std.testing.expectEqual(@as(usize, 1), loaded.segments.len);
    try std.testing.expectEqual(@as(u64, 5), loaded.segments[0].row_count);
}

// ===== deinit =====

test "Manifest: deinit frees segment memory" {
    var segments = try std.testing.allocator.alloc(manifest.SegmentEntry, 1);
    segments[0] = .{
        .id = 0,
        .path = try std.testing.allocator.dupe(u8, "test.talu"),
        .min_ts = 0,
        .max_ts = 0,
        .row_count = 0,
    };

    var m = Manifest{
        .version = 1,
        .segments = segments,
        .last_compaction_ts = 0,
    };

    try std.testing.expectEqual(@as(usize, 1), m.segments.len);
    m.deinit(std.testing.allocator);

    // deinit zeroes the segments slice; std.testing.allocator catches leaks.
    try std.testing.expectEqual(@as(usize, 0), m.segments.len);
}
