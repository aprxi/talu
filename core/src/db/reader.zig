//! TaluDB reader that unifies sealed segments and the active file.
//!
//! Provides a block list across manifest segments and the current file.

const std = @import("std");
const block_writer = @import("block_writer.zig");
const manifest = @import("manifest.zig");
const types = @import("types.zig");

const Allocator = std.mem.Allocator;

pub const BlockRef = struct {
    path: []const u8,
    offset: u64,
};

pub const Reader = struct {
    allocator: Allocator,
    dir: std.fs.Dir,
    ns_prefix: []const u8,
    manifest_data: ?manifest.Manifest,
    current_file: ?std.fs.File,
    current_blocks: std.ArrayList(u64),
    current_path: ?[]const u8,
    /// Block references for sealed segments (path index + offset pairs).
    segment_blocks: std.ArrayList(BlockRef),

    /// Opens the TaluDB root and indexes the current file if present.
    pub fn open(allocator: Allocator, db_root: []const u8, namespace: []const u8) !Reader {
        // Manifest lives inside the namespace directory.
        const manifest_path = try std.fs.path.join(allocator, &.{ db_root, namespace, "manifest.json" });
        defer allocator.free(manifest_path);

        var dir = try std.fs.cwd().makeOpenPath(db_root, .{});
        errdefer dir.close();

        var manifest_data: ?manifest.Manifest = null;
        var current_file: ?std.fs.File = null;
        var current_path: ?[]const u8 = null;

        manifest_data = manifest.Manifest.load(allocator, manifest_path) catch |err| switch (err) {
            error.FileNotFound => blk: {
                const empty_segments = try allocator.alloc(manifest.SegmentEntry, 0);
                break :blk manifest.Manifest{
                    .version = 1,
                    .segments = empty_segments,
                    .last_compaction_ts = 0,
                };
            },
            else => return err,
        };
        errdefer if (manifest_data) |*data| data.deinit(allocator);

        const current_path_full = try std.fs.path.join(allocator, &.{ db_root, namespace, "current.talu" });
        defer allocator.free(current_path_full);

        current_file = std.fs.cwd().openFile(current_path_full, .{ .mode = .read_only }) catch |err| switch (err) {
            error.FileNotFound => null,
            else => return err,
        };

        if (current_file != null) {
            const rel_path = try std.fs.path.join(allocator, &.{ namespace, "current.talu" });
            defer allocator.free(rel_path);
            current_path = try allocator.dupe(u8, rel_path);
        }
        errdefer if (current_path) |path| allocator.free(path);

        const ns_prefix = try allocator.dupe(u8, namespace);
        errdefer allocator.free(ns_prefix);

        var reader = Reader{
            .allocator = allocator,
            .dir = dir,
            .ns_prefix = ns_prefix,
            .manifest_data = manifest_data,
            .current_file = current_file,
            .current_blocks = .empty,
            .current_path = current_path,
            .segment_blocks = .empty,
        };
        errdefer reader.deinit();

        // Scan sealed segments for block offsets.
        if (reader.manifest_data) |data| {
            try reader.scanSegments(data.segments);
        }

        if (reader.current_file) |file| {
            try reader.scanCurrent(file);
        }

        return reader;
    }

    /// Releases resources owned by the reader.
    pub fn deinit(self: *Reader) void {
        if (self.manifest_data) |*data| {
            data.deinit(self.allocator);
        }
        if (self.current_file) |file| file.close();
        if (self.current_path) |path| self.allocator.free(path);
        for (self.segment_blocks.items) |block| {
            self.allocator.free(block.path);
        }
        self.segment_blocks.deinit(self.allocator);
        self.current_blocks.deinit(self.allocator);
        self.allocator.free(self.ns_prefix);
        self.dir.close();
    }

    /// Returns a list of all blocks in sealed segments and current.talu.
    /// Caller owns the returned slice; paths borrow from reader-owned memory.
    pub fn getBlocks(self: *Reader, alloc: Allocator) ![]BlockRef {
        const total = self.segment_blocks.items.len + self.current_blocks.items.len;
        const blocks = try alloc.alloc(BlockRef, total);

        var index: usize = 0;

        // Sealed segment blocks (scanned with namespace-prefixed paths).
        for (self.segment_blocks.items) |block| {
            blocks[index] = .{ .path = block.path, .offset = block.offset };
            index += 1;
        }

        // Current file blocks.
        if (self.current_path) |path| {
            for (self.current_blocks.items) |offset| {
                blocks[index] = .{ .path = path, .offset = offset };
                index += 1;
            }
        }

        return blocks;
    }

    /// Refresh the current.talu block index.
    pub fn refreshCurrent(self: *Reader) !void {
        self.current_blocks.clearRetainingCapacity();
        if (self.current_file) |file| {
            try self.scanCurrent(file);
        }
    }

    fn scanCurrent(self: *Reader, file: std.fs.File) !void {
        try scanFileBlocks(file, self.allocator, &self.current_blocks);
    }

    /// Scans sealed segments from the manifest, recording all block
    /// offsets with namespace-prefixed paths (e.g., "chat/seg-xxx.talu").
    fn scanSegments(self: *Reader, segments: []const manifest.SegmentEntry) !void {
        for (segments) |segment| {
            // Build namespace-prefixed path (e.g., "chat/seg-xxx.talu").
            const ns_path = try std.fs.path.join(self.allocator, &.{ self.ns_prefix, segment.path });
            errdefer self.allocator.free(ns_path);

            var file = self.dir.openFile(ns_path, .{ .mode = .read_only }) catch |err| switch (err) {
                error.FileNotFound => {
                    self.allocator.free(ns_path);
                    continue; // Sealed segment may have been compacted away.
                },
                else => return err,
            };
            defer file.close();

            // Scan for block offsets within the sealed segment.
            var block_offsets = std.ArrayList(u64).empty;
            defer block_offsets.deinit(self.allocator);
            try scanFileBlocks(file, self.allocator, &block_offsets);

            // Record each block with its namespace-prefixed path.
            if (block_offsets.items.len == 0) {
                self.allocator.free(ns_path);
                continue;
            }

            for (block_offsets.items, 0..) |offset, i| {
                // First block owns ns_path; subsequent blocks get a dupe.
                const path = if (i == 0) ns_path else try self.allocator.dupe(u8, ns_path);
                try self.segment_blocks.append(self.allocator, .{
                    .path = path,
                    .offset = offset,
                });
            }
        }
    }
};

/// Scans a file for TaluDB block headers, appending offsets to `out`.
fn scanFileBlocks(file: std.fs.File, alloc: Allocator, out: *std.ArrayList(u64)) !void {
    const stat = try file.stat();
    const size = stat.size;
    const header_len = @sizeOf(types.BlockHeader);
    var offset: u64 = 0;

    while (offset + header_len <= size) {
        var header_bytes: [@sizeOf(types.BlockHeader)]u8 = undefined;
        const read_len = try file.preadAll(&header_bytes, offset);
        if (read_len != header_bytes.len) return;

        const header = std.mem.bytesToValue(types.BlockHeader, header_bytes[0..]);
        if (header.magic != types.MagicValues.BLOCK) return error.InvalidMagic;
        if (header.block_len < header_len) return error.InvalidBlock;

        const next_offset = offset + @as(u64, header.block_len);
        if (next_offset > size) return;

        try out.append(alloc, offset);
        offset = next_offset;
    }
}

fn buildTestBlock(allocator: Allocator, value: u64) ![]u8 {
    var builder = block_writer.BlockBuilder.init(allocator, 1, 1);
    defer builder.deinit();

    const payload = std.mem.asBytes(&value);
    try builder.addColumn(1, .SCALAR, .U64, .RAW, 1, payload, null, null);

    return builder.finalize();
}

// =============================================================================
// Tests
// =============================================================================

test "Reader.open scans current file" {
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
    try std.testing.expectEqual(@as(u64, 0), reader.current_blocks.items[0]);
    try std.testing.expect(reader.current_blocks.items[1] > reader.current_blocks.items[0]);
}

test "Reader.getBlocks returns manifest then current" {
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

test "Reader.refreshCurrent rescans file" {
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
    try file.writeAll(block2);

    try reader.refreshCurrent();
    try std.testing.expectEqual(@as(usize, 2), reader.current_blocks.items.len);
}

test "Reader.deinit releases resources" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var reader = try Reader.open(std.testing.allocator, root_path, "chat");
    reader.deinit();
}
