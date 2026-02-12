//! Integration tests for db.Reader
//!
//! Reader provides a unified view of blocks across the manifest (sealed segments)
//! and the active current.talu file.

const std = @import("std");
const main = @import("main");
const db = main.db;

const Reader = db.Reader;
const BlockBuilder = db.BlockBuilder;
const manifest = db.manifest;

fn buildTestBlock(allocator: std.mem.Allocator, value: u64) ![]u8 {
    var builder = BlockBuilder.init(allocator, 1, 1);
    defer builder.deinit();

    const payload = std.mem.asBytes(&value);
    try builder.addColumn(1, .SCALAR, .U64, .RAW, 1, payload, null, null);

    return builder.finalize();
}

// ===== open =====

test "Reader: open scans existing current.talu blocks" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    try tmp.dir.makePath("chat");

    const block1 = try buildTestBlock(std.testing.allocator, 1);
    defer std.testing.allocator.free(block1);
    const block2 = try buildTestBlock(std.testing.allocator, 2);
    defer std.testing.allocator.free(block2);

    var file = try tmp.dir.createFile("chat/current.talu", .{ .read = true });
    defer file.close();
    try file.writeAll(block1);
    try file.writeAll(block2);

    var reader = try Reader.open(std.testing.allocator, root_path, "chat");
    defer reader.deinit();

    try std.testing.expectEqual(@as(usize, 2), reader.current_blocks.items.len);
}

test "Reader: open handles missing current.talu gracefully" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var reader = try Reader.open(std.testing.allocator, root_path, "empty");
    defer reader.deinit();

    try std.testing.expectEqual(@as(usize, 0), reader.current_blocks.items.len);
}

// ===== getBlocks =====

test "Reader: getBlocks returns manifest segments and current blocks" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    try tmp.dir.makePath("chat");

    const block = try buildTestBlock(std.testing.allocator, 7);
    defer std.testing.allocator.free(block);

    var file = try tmp.dir.createFile("chat/current.talu", .{ .read = true });
    defer file.close();
    try file.writeAll(block);

    // Create a sealed segment file in the namespace directory.
    const seg_block = try buildTestBlock(std.testing.allocator, 42);
    defer std.testing.allocator.free(seg_block);

    var seg_file = try tmp.dir.createFile("chat/seg-1.talu", .{});
    try seg_file.writeAll(seg_block);
    seg_file.close();

    // Create manifest with one segment (path is filename only, within namespace).
    // lint:ignore errdefer-alloc - ownership transferred to manifest_data, freed via deinit()
    var segments = try std.testing.allocator.alloc(manifest.SegmentEntry, 1);
    segments[0] = .{
        .id = 0x0123456789abcdef0123456789abcdef,
        .path = try std.testing.allocator.dupe(u8, "seg-1.talu"), // lint:ignore errdefer-alloc - freed via manifest.deinit()
        .min_ts = 0,
        .max_ts = 1,
        .row_count = 1,
    };

    var manifest_data = manifest.Manifest{
        .version = 1,
        .segments = segments,
        .last_compaction_ts = 0,
    };

    // Manifest lives in the namespace directory.
    const manifest_path = try std.fs.path.join(std.testing.allocator, &.{ root_path, "chat", "manifest.json" });
    defer std.testing.allocator.free(manifest_path);
    try manifest_data.save(manifest_path);
    manifest_data.deinit(std.testing.allocator);

    var reader = try Reader.open(std.testing.allocator, root_path, "chat");
    defer reader.deinit();

    const blocks = try reader.getBlocks(std.testing.allocator);
    defer std.testing.allocator.free(blocks);

    // 1 block from sealed segment + 1 block from current.talu = 2 blocks.
    try std.testing.expectEqual(@as(usize, 2), blocks.len);
    try std.testing.expectEqualStrings("chat/seg-1.talu", blocks[0].path);
    try std.testing.expectEqualStrings("chat/current.talu", blocks[1].path);
}

// ===== refreshCurrent =====

test "Reader: refreshCurrent picks up newly written blocks" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    try tmp.dir.makePath("vector");

    const block1 = try buildTestBlock(std.testing.allocator, 3);
    defer std.testing.allocator.free(block1);
    const block2 = try buildTestBlock(std.testing.allocator, 4);
    defer std.testing.allocator.free(block2);

    var file = try tmp.dir.createFile("vector/current.talu", .{ .read = true });
    defer file.close();
    try file.writeAll(block1);

    var reader = try Reader.open(std.testing.allocator, root_path, "vector");
    defer reader.deinit();

    try std.testing.expectEqual(@as(usize, 1), reader.current_blocks.items.len);

    // Write another block while reader is open
    try file.writeAll(block2);
    try reader.refreshCurrent();

    try std.testing.expectEqual(@as(usize, 2), reader.current_blocks.items.len);
}

// ===== deinit =====

test "Reader: deinit releases all resources" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var reader = try Reader.open(std.testing.allocator, root_path, "chat");
    reader.deinit();

    // Verify resources were released: re-opening the same namespace must succeed
    // (file handles closed, allocations freed — std.testing.allocator catches leaks).
    var reader2 = try Reader.open(std.testing.allocator, root_path, "chat");
    reader2.deinit();
}
