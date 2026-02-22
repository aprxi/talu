//! TaluDB reader that unifies sealed segments and the active file.
//!
//! Provides a block list across manifest segments and the current file.

const std = @import("std");
const block_writer = @import("block_writer.zig");
const block_reader = @import("block_reader.zig");
const checksum = @import("checksum.zig");
const manifest = @import("manifest.zig");
const segment_source = @import("segment_source.zig");
const types = @import("types.zig");

const Allocator = std.mem.Allocator;

pub const BlockRef = struct {
    path: []const u8,
    offset: u64,
};

pub const ManifestSnapshot = struct {
    generation: u64,
    segments: []manifest.SegmentEntry,

    pub fn deinit(self: *ManifestSnapshot, allocator: Allocator) void {
        for (self.segments) |segment| {
            allocator.free(segment.path);
            if (segment.index) |index_meta| allocator.free(index_meta.path);
        }
        allocator.free(self.segments);
        self.segments = &.{};
    }
};

const FileSignature = struct {
    exists: bool,
    size: u64,
    mtime_ns: i128,

    fn eql(a: FileSignature, b: FileSignature) bool {
        return a.exists == b.exists and a.size == b.size and a.mtime_ns == b.mtime_ns;
    }
};

pub const Reader = struct {
    allocator: Allocator,
    dir: std.fs.Dir,
    db_root: []const u8,
    ns_prefix: []const u8,
    manifest_rel_path: []const u8,
    current_rel_path: []const u8,
    manifest_data: ?manifest.Manifest,
    current_handle: ?segment_source.SegmentHandle,
    current_blocks: std.ArrayList(u64),
    current_path: ?[]const u8,
    source_override: ?segment_source.SegmentSource,
    manifest_sig: FileSignature,
    current_sig: FileSignature,
    /// Block references for sealed segments (path index + offset pairs).
    segment_blocks: std.ArrayList(BlockRef),

    /// Opens the TaluDB root and indexes the current file if present.
    pub fn open(allocator: Allocator, db_root: []const u8, namespace: []const u8) !Reader {
        return openWithSegmentSource(allocator, db_root, namespace, null);
    }

    /// Opens the TaluDB root and indexes the current file if present.
    /// If `source_override` is provided, sealed segments/current file reads are
    /// routed through that segment source instead of direct local opens.
    pub fn openWithSegmentSource(
        allocator: Allocator,
        db_root: []const u8,
        namespace: []const u8,
        source_override: ?segment_source.SegmentSource,
    ) !Reader {
        // Manifest lives inside the namespace directory.
        const manifest_path = try std.fs.path.join(allocator, &.{ db_root, namespace, "manifest.json" });
        defer allocator.free(manifest_path);

        var dir = try std.fs.cwd().makeOpenPath(db_root, .{});
        errdefer dir.close();

        var manifest_data: ?manifest.Manifest = null;
        var current_handle: ?segment_source.SegmentHandle = null;
        var current_path: ?[]const u8 = null;
        const manifest_rel_path = try std.fs.path.join(allocator, &.{ namespace, "manifest.json" });
        errdefer allocator.free(manifest_rel_path);
        const current_rel_path = try std.fs.path.join(allocator, &.{ namespace, "current.talu" });
        errdefer allocator.free(current_rel_path);

        manifest_data = manifest.Manifest.load(allocator, manifest_path) catch |err| switch (err) {
            error.FileNotFound => blk: {
                const empty_segments = try allocator.alloc(manifest.SegmentEntry, 0);
                break :blk manifest.Manifest{
                    .version = 1,
                    .generation = 0,
                    .segments = empty_segments,
                    .last_compaction_ts = 0,
                };
            },
            else => return err,
        };
        errdefer if (manifest_data) |*data| data.deinit(allocator);

        if (source_override) |source| {
            current_handle = source.openReadOnly(current_rel_path) catch |err| switch (err) {
                error.FileNotFound => null,
                else => return err,
            };
        } else {
            var local = segment_source.LocalSegmentSource.init(&dir);
            current_handle = local.asSource().openReadOnly(current_rel_path) catch |err| switch (err) {
                error.FileNotFound => null,
                else => return err,
            };
        }

        if (current_handle != null) {
            current_path = try allocator.dupe(u8, current_rel_path);
        }
        errdefer if (current_handle) |*handle| {
            if (source_override) |source| {
                source.close(handle);
            } else {
                handle.file.close();
            }
        };
        errdefer if (current_path) |path| allocator.free(path);

        const ns_prefix = try allocator.dupe(u8, namespace);
        errdefer allocator.free(ns_prefix);
        const db_root_copy = try allocator.dupe(u8, db_root);
        errdefer allocator.free(db_root_copy);

        var reader = Reader{
            .allocator = allocator,
            .dir = dir,
            .db_root = db_root_copy,
            .ns_prefix = ns_prefix,
            .manifest_rel_path = manifest_rel_path,
            .current_rel_path = current_rel_path,
            .manifest_data = manifest_data,
            .current_handle = current_handle,
            .current_blocks = .empty,
            .current_path = current_path,
            .source_override = source_override,
            .manifest_sig = .{ .exists = false, .size = 0, .mtime_ns = 0 },
            .current_sig = .{ .exists = false, .size = 0, .mtime_ns = 0 },
            .segment_blocks = .empty,
        };
        errdefer reader.deinit();

        // Scan sealed segments for block offsets.
        if (reader.manifest_data) |data| {
            try reader.scanSegments(data.segments);
        }

        if (reader.current_handle) |handle| {
            try reader.scanCurrent(handle);
        }

        reader.manifest_sig = try reader.readFileSignature(reader.manifest_rel_path);
        reader.current_sig = try reader.readFileSignature(reader.current_rel_path);

        return reader;
    }

    /// Releases resources owned by the reader.
    pub fn deinit(self: *Reader) void {
        if (self.manifest_data) |*data| {
            data.deinit(self.allocator);
        }
        if (self.current_handle) |*handle| self.closeHandle(handle);
        if (self.current_path) |path| self.allocator.free(path);
        for (self.segment_blocks.items) |block| {
            self.allocator.free(block.path);
        }
        self.segment_blocks.deinit(self.allocator);
        self.current_blocks.deinit(self.allocator);
        self.allocator.free(self.manifest_rel_path);
        self.allocator.free(self.current_rel_path);
        self.allocator.free(self.db_root);
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

    /// Returns the currently loaded manifest generation snapshot.
    pub fn snapshotGeneration(self: *Reader) u64 {
        if (self.manifest_data) |data| return data.generation;
        return 0;
    }

    /// Returns a deep-copied manifest snapshot that stays stable for callers.
    pub fn loadManifestSnapshot(self: *Reader, allocator: Allocator) !ManifestSnapshot {
        const data = self.manifest_data orelse return .{
            .generation = 0,
            .segments = try allocator.alloc(manifest.SegmentEntry, 0),
        };

        const out = try allocator.alloc(manifest.SegmentEntry, data.segments.len);
        var initialized: usize = 0;
        errdefer {
            var i: usize = 0;
            while (i < initialized) : (i += 1) {
                allocator.free(out[i].path);
                if (out[i].index) |index_meta| allocator.free(index_meta.path);
            }
            allocator.free(out);
        }
        for (data.segments, 0..) |segment, idx| {
            out[idx] = .{
                .id = segment.id,
                .path = try allocator.dupe(u8, segment.path),
                .min_ts = segment.min_ts,
                .max_ts = segment.max_ts,
                .row_count = segment.row_count,
                .checksum_crc32c = segment.checksum_crc32c,
                .index = if (segment.index) |index_meta|
                    .{
                        .kind = index_meta.kind,
                        .path = try allocator.dupe(u8, index_meta.path),
                        .checksum_crc32c = index_meta.checksum_crc32c,
                    }
                else
                    null,
            };
            initialized += 1;
        }

        return .{
            .generation = data.generation,
            .segments = out,
        };
    }

    /// Refresh the current.talu block index.
    pub fn refreshCurrent(self: *Reader) !void {
        self.current_blocks.clearRetainingCapacity();
        if (self.current_handle) |handle| {
            try self.scanCurrent(handle);
        }
    }

    /// Refreshes both manifest-backed segments and current.talu.
    ///
    /// This is required for cross-process visibility when another writer
    /// seals segments or replaces current.talu.
    pub fn refresh(self: *Reader) !void {
        const manifest_path = try std.fs.path.join(self.allocator, &.{ self.db_root, self.manifest_rel_path });
        defer self.allocator.free(manifest_path);

        var manifest_data = manifest.Manifest.load(self.allocator, manifest_path) catch |err| switch (err) {
            error.FileNotFound => blk: {
                const empty_segments = try self.allocator.alloc(manifest.SegmentEntry, 0);
                break :blk manifest.Manifest{
                    .version = 1,
                    .generation = 0,
                    .segments = empty_segments,
                    .last_compaction_ts = 0,
                };
            },
            else => return err,
        };
        errdefer manifest_data.deinit(self.allocator);

        for (self.segment_blocks.items) |block| {
            self.allocator.free(block.path);
        }
        self.segment_blocks.clearRetainingCapacity();
        if (manifest_data.segments.len > 0) {
            try self.scanSegments(manifest_data.segments);
        }

        if (self.manifest_data) |*old_manifest| {
            old_manifest.deinit(self.allocator);
        }
        self.manifest_data = manifest_data;
        // Ownership transferred to self.manifest_data.
        manifest_data.segments = &.{};

        try self.refreshCurrentOnly();

        self.manifest_sig = try self.readFileSignature(self.manifest_rel_path);
        self.current_sig = try self.readFileSignature(self.current_rel_path);
    }

    /// Refreshes reader state only when manifest/current files changed.
    ///
    /// Returns true when a refresh was applied, false when signatures match
    /// and no rescan was needed.
    pub fn refreshIfChanged(self: *Reader) !bool {
        const manifest_sig = try self.readFileSignature(self.manifest_rel_path);
        const current_sig = try self.readFileSignature(self.current_rel_path);
        if (FileSignature.eql(manifest_sig, self.manifest_sig) and
            FileSignature.eql(current_sig, self.current_sig))
        {
            return false;
        }

        // Fast path: only current.talu changed, manifest is unchanged.
        // Avoid rescanning sealed segments in this case.
        if (FileSignature.eql(manifest_sig, self.manifest_sig)) {
            try self.refreshCurrentOnly();
            self.current_sig = current_sig;
            return true;
        }

        try self.refresh();
        return true;
    }

    fn scanCurrent(self: *Reader, handle: segment_source.SegmentHandle) !void {
        try scanFileBlocks(handle.file, self.allocator, &self.current_blocks);
    }

    fn refreshCurrentOnly(self: *Reader) !void {
        if (self.current_handle) |*handle| self.closeHandle(handle);
        self.current_handle = null;
        self.current_blocks.clearRetainingCapacity();

        if (self.current_path == null) {
            self.current_path = try self.allocator.dupe(u8, self.current_rel_path);
        }

        self.current_handle = self.openReadOnlyHandle(self.current_rel_path) catch |err| switch (err) {
            error.FileNotFound => null,
            else => return err,
        };
        if (self.current_handle) |handle| {
            try self.scanCurrent(handle);
        }
    }

    fn readFileSignature(self: *Reader, rel_path: []const u8) !FileSignature {
        const stat = self.dir.statFile(rel_path) catch |err| switch (err) {
            error.FileNotFound => return .{ .exists = false, .size = 0, .mtime_ns = 0 },
            else => return err,
        };
        return .{
            .exists = true,
            .size = stat.size,
            .mtime_ns = stat.mtime,
        };
    }

    /// Scans sealed segments from the manifest, recording all block
    /// offsets with namespace-prefixed paths (e.g., "chat/seg-xxx.talu").
    fn scanSegments(self: *Reader, segments: []const manifest.SegmentEntry) !void {
        for (segments) |segment| {
            // Build namespace-prefixed path (e.g., "chat/seg-xxx.talu").
            const ns_path = try std.fs.path.join(self.allocator, &.{ self.ns_prefix, segment.path });
            errdefer self.allocator.free(ns_path);

            var handle = self.openReadOnlyHandle(ns_path) catch |err| switch (err) {
                error.FileNotFound => {
                    self.allocator.free(ns_path);
                    continue; // Sealed segment may have been compacted away.
                },
                else => return err,
            };
            defer self.closeHandle(&handle);

            // Prefer O(1) footer index for sealed segments, fallback to header scan.
            const idx_reader = block_reader.BlockReader.init(handle.file, self.allocator);
            const block_index = try idx_reader.getBlockIndex(handle.size);
            defer self.allocator.free(block_index);

            // Record each block with its namespace-prefixed path.
            if (block_index.len == 0) {
                self.allocator.free(ns_path);
                continue;
            }

            for (block_index, 0..) |entry, i| {
                // First block owns ns_path; subsequent blocks get a dupe.
                const path = if (i == 0) ns_path else try self.allocator.dupe(u8, ns_path);
                try self.segment_blocks.append(self.allocator, .{
                    .path = path,
                    .offset = entry.block_off,
                });
            }
        }
    }

    fn openReadOnlyHandle(self: *Reader, rel_path: []const u8) !segment_source.SegmentHandle {
        if (self.source_override) |source| {
            return source.openReadOnly(rel_path);
        }
        var local = segment_source.LocalSegmentSource.init(&self.dir);
        return local.asSource().openReadOnly(rel_path);
    }

    fn closeHandle(self: *Reader, handle: *segment_source.SegmentHandle) void {
        if (self.source_override) |source| {
            source.close(handle);
            return;
        }
        handle.file.close();
    }

    /// Open a block/segment path using the configured segment source.
    pub fn openBlockReadOnly(self: *Reader, rel_path: []const u8) !segment_source.SegmentHandle {
        return self.openReadOnlyHandle(rel_path);
    }

    /// Close a block/segment handle opened with `openBlockReadOnly`.
    pub fn closeBlock(self: *Reader, handle: *segment_source.SegmentHandle) void {
        self.closeHandle(handle);
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
        if (header.magic != types.MagicValues.BLOCK) break;
        if (header.block_len < header_len) break;

        const next_offset = offset + @as(u64, header.block_len);
        if (next_offset > size) break;

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
        .generation = 0,
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

test "Reader.getBlocks reads sealed segment footer index" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    try tmp.dir.makePath("chat");

    const block1 = try buildTestBlock(std.testing.allocator, 10);
    defer std.testing.allocator.free(block1);
    const block2 = try buildTestBlock(std.testing.allocator, 11);
    defer std.testing.allocator.free(block2);

    var seg_file = try tmp.dir.createFile("chat/seg-1.talu", .{ .read = true });
    defer seg_file.close();
    try seg_file.writeAll(block1);
    try seg_file.writeAll(block2);

    const entries = [_]types.FooterBlockEntry{
        .{
            .block_off = 0,
            .block_len = @intCast(block1.len),
            .schema_id = 1,
        },
        .{
            .block_off = @intCast(block1.len),
            .block_len = @intCast(block2.len),
            .schema_id = 1,
        },
    };
    const footer_payload = std.mem.sliceAsBytes(&entries);
    try seg_file.writeAll(footer_payload);

    const trailer = types.FooterTrailer{
        .magic = types.MagicValues.FOOTER,
        .version = 1,
        .flags = 0,
        .footer_len = @intCast(footer_payload.len),
        .footer_crc32c = checksum.crc32c(footer_payload),
        .segment_crc32c = 0,
        .reserved = [_]u8{0} ** 12,
    };
    try seg_file.writeAll(std.mem.asBytes(&trailer));

    const current_block = try buildTestBlock(std.testing.allocator, 99);
    defer std.testing.allocator.free(current_block);
    var current_file = try tmp.dir.createFile("chat/current.talu", .{ .read = true });
    defer current_file.close();
    try current_file.writeAll(current_block);

    // lint:ignore errdefer-alloc - ownership transferred to manifest_data, freed via deinit()
    var segments = try std.testing.allocator.alloc(manifest.SegmentEntry, 1);
    segments[0] = .{
        .id = 0x99999999888888887777777766666666,
        .path = try std.testing.allocator.dupe(u8, "seg-1.talu"), // lint:ignore errdefer-alloc - freed via manifest.deinit()
        .min_ts = 0,
        .max_ts = 1,
        .row_count = 2,
    };

    var manifest_data = manifest.Manifest{
        .version = 1,
        .generation = 0,
        .segments = segments,
        .last_compaction_ts = 0,
    };

    const manifest_path = try std.fs.path.join(std.testing.allocator, &.{ root_path, "chat", "manifest.json" });
    defer std.testing.allocator.free(manifest_path);
    try manifest_data.save(manifest_path);
    manifest_data.deinit(std.testing.allocator);

    var reader = try Reader.open(std.testing.allocator, root_path, "chat");
    defer reader.deinit();

    const blocks = try reader.getBlocks(std.testing.allocator);
    defer std.testing.allocator.free(blocks);

    try std.testing.expectEqual(@as(usize, 3), blocks.len);
    try std.testing.expectEqualStrings("chat/seg-1.talu", blocks[0].path);
    try std.testing.expectEqual(@as(u64, 0), blocks[0].offset);
    try std.testing.expectEqualStrings("chat/seg-1.talu", blocks[1].path);
    try std.testing.expectEqual(@as(u64, block1.len), blocks[1].offset);
    try std.testing.expectEqualStrings("chat/current.talu", blocks[2].path);
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

test "Reader.refresh reloads manifest segments and current file" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    try tmp.dir.makePath("vector");

    var reader = try Reader.open(std.testing.allocator, root_path, "vector");
    defer reader.deinit();

    const initial_blocks = try reader.getBlocks(std.testing.allocator);
    defer std.testing.allocator.free(initial_blocks);
    try std.testing.expectEqual(@as(usize, 0), initial_blocks.len);

    const seg_block = try buildTestBlock(std.testing.allocator, 9);
    defer std.testing.allocator.free(seg_block);
    var seg_file = try tmp.dir.createFile("vector/seg-1.talu", .{});
    try seg_file.writeAll(seg_block);
    seg_file.close();

    const current_block = try buildTestBlock(std.testing.allocator, 10);
    defer std.testing.allocator.free(current_block);
    var current_file = try tmp.dir.createFile("vector/current.talu", .{ .read = true });
    defer current_file.close();
    try current_file.writeAll(current_block);

    const segments = try std.testing.allocator.alloc(manifest.SegmentEntry, 1); // lint:ignore errdefer-alloc - freed via manifest_data.deinit()
    segments[0] = .{
        .id = 0x0123456789abcdef0123456789abcdef,
        .path = try std.testing.allocator.dupe(u8, "seg-1.talu"), // lint:ignore errdefer-alloc - freed via manifest_data.deinit()
        .min_ts = 0,
        .max_ts = 1,
        .row_count = 1,
    };

    var manifest_data = manifest.Manifest{
        .version = 1,
        .generation = 0,
        .segments = segments,
        .last_compaction_ts = 0,
    };
    const manifest_path = try std.fs.path.join(std.testing.allocator, &.{ root_path, "vector", "manifest.json" });
    defer std.testing.allocator.free(manifest_path);
    try manifest_data.save(manifest_path);
    manifest_data.deinit(std.testing.allocator);

    try reader.refresh();

    const refreshed = try reader.getBlocks(std.testing.allocator);
    defer std.testing.allocator.free(refreshed);
    try std.testing.expectEqual(@as(usize, 2), refreshed.len);
    try std.testing.expectEqualStrings("vector/seg-1.talu", refreshed[0].path);
    try std.testing.expectEqualStrings("vector/current.talu", refreshed[1].path);
}

test "Reader.refreshIfChanged skips when unchanged and refreshes on current append" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    try tmp.dir.makePath("vector");

    const block1 = try buildTestBlock(std.testing.allocator, 1);
    defer std.testing.allocator.free(block1);
    const block2 = try buildTestBlock(std.testing.allocator, 2);
    defer std.testing.allocator.free(block2);

    var current_file = try tmp.dir.createFile("vector/current.talu", .{ .read = true });
    defer current_file.close();
    try current_file.writeAll(block1);

    var reader = try Reader.open(std.testing.allocator, root_path, "vector");
    defer reader.deinit();
    try std.testing.expectEqual(@as(usize, 1), reader.current_blocks.items.len);

    const unchanged = try reader.refreshIfChanged();
    try std.testing.expect(!unchanged);
    try std.testing.expectEqual(@as(usize, 1), reader.current_blocks.items.len);

    try current_file.writeAll(block2);
    const changed = try reader.refreshIfChanged();
    try std.testing.expect(changed);
    try std.testing.expectEqual(@as(usize, 2), reader.current_blocks.items.len);
}

test "Reader.refreshIfChanged loads current created after open when manifest is unchanged" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    try tmp.dir.makePath("vector");

    const seg_block = try buildTestBlock(std.testing.allocator, 5);
    defer std.testing.allocator.free(seg_block);
    var seg_file = try tmp.dir.createFile("vector/seg-1.talu", .{});
    try seg_file.writeAll(seg_block);
    seg_file.close();

    const segments = try std.testing.allocator.alloc(manifest.SegmentEntry, 1); // lint:ignore errdefer-alloc - freed via manifest_data.deinit()
    segments[0] = .{
        .id = 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,
        .path = try std.testing.allocator.dupe(u8, "seg-1.talu"), // lint:ignore errdefer-alloc - freed via manifest_data.deinit()
        .min_ts = 0,
        .max_ts = 1,
        .row_count = 1,
    };

    var manifest_data = manifest.Manifest{
        .version = 1,
        .generation = 0,
        .segments = segments,
        .last_compaction_ts = 0,
    };
    const manifest_path = try std.fs.path.join(std.testing.allocator, &.{ root_path, "vector", "manifest.json" });
    defer std.testing.allocator.free(manifest_path);
    try manifest_data.save(manifest_path);
    manifest_data.deinit(std.testing.allocator);

    // No current.talu yet.
    var reader = try Reader.open(std.testing.allocator, root_path, "vector");
    defer reader.deinit();
    try std.testing.expectEqual(@as(usize, 1), reader.segment_blocks.items.len);
    try std.testing.expectEqual(@as(usize, 0), reader.current_blocks.items.len);

    const current_block = try buildTestBlock(std.testing.allocator, 6);
    defer std.testing.allocator.free(current_block);
    var current_file = try tmp.dir.createFile("vector/current.talu", .{ .read = true });
    defer current_file.close();
    try current_file.writeAll(current_block);

    const changed = try reader.refreshIfChanged();
    try std.testing.expect(changed);

    const blocks = try reader.getBlocks(std.testing.allocator);
    defer std.testing.allocator.free(blocks);
    try std.testing.expectEqual(@as(usize, 2), blocks.len);
    try std.testing.expectEqualStrings("vector/seg-1.talu", blocks[0].path);
    try std.testing.expectEqualStrings("vector/current.talu", blocks[1].path);
}

test "Reader.deinit releases resources" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var reader = try Reader.open(std.testing.allocator, root_path, "chat");
    reader.deinit();
}

test "Reader.open ignores trailing invalid bytes in current file" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    try tmp.dir.makePath("chat");

    const block = try buildTestBlock(std.testing.allocator, 123);
    defer std.testing.allocator.free(block);

    var file = try tmp.dir.createFile("chat/current.talu", .{ .read = true });
    defer file.close();
    try file.writeAll(block);
    const junk = [_]u8{0xaa} ** @sizeOf(types.BlockHeader);
    try file.writeAll(&junk);

    var reader = try Reader.open(std.testing.allocator, root_path, "chat");
    defer reader.deinit();

    try std.testing.expectEqual(@as(usize, 1), reader.current_blocks.items.len);
    try std.testing.expectEqual(@as(u64, 0), reader.current_blocks.items[0]);
}

test "Reader.openWithSegmentSource reads blocks via source override" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    try tmp.dir.makePath("chat");

    const block = try buildTestBlock(std.testing.allocator, 55);
    defer std.testing.allocator.free(block);

    var file = try tmp.dir.createFile("chat/current.talu", .{ .read = true });
    defer file.close();
    try file.writeAll(block);

    var source_dir = try tmp.dir.openDir(".", .{});
    defer source_dir.close();
    var local = segment_source.LocalSegmentSource.init(&source_dir);

    var reader = try Reader.openWithSegmentSource(std.testing.allocator, root_path, "chat", local.asSource());
    defer reader.deinit();

    try std.testing.expectEqual(@as(usize, 1), reader.current_blocks.items.len);
    try std.testing.expectEqual(@as(u64, 0), reader.current_blocks.items[0]);
}

test "Reader.openBlockReadOnly and Reader.closeBlock open and close segment handles" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    try tmp.dir.makePath("chat");
    var file = try tmp.dir.createFile("chat/seg-1.talu", .{ .read = true });
    defer file.close();
    try file.writeAll("abcd");

    var reader = try Reader.open(std.testing.allocator, root_path, "chat");
    defer reader.deinit();

    var handle = try reader.openBlockReadOnly("chat/seg-1.talu");
    var buf: [4]u8 = undefined;
    const n = try handle.file.preadAll(&buf, 0);
    try std.testing.expectEqual(@as(usize, 4), n);
    try std.testing.expectEqualSlices(u8, "abcd", &buf);

    reader.closeBlock(&handle);
}

test "Reader.loadManifestSnapshot returns deep-copied stable snapshot" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);
    try tmp.dir.makePath("vector");
    try tmp.dir.writeFile(.{
        .sub_path = "vector/manifest.json",
        .data = "{" ++
            "\"version\":1," ++
            "\"generation\":4," ++
            "\"last_compaction_ts\":0," ++
            "\"segments\":[{" ++
            "\"id\":\"0123456789abcdef0123456789abcdef\"," ++
            "\"path\":\"seg-1.talu\"," ++
            "\"min_ts\":1," ++
            "\"max_ts\":2," ++
            "\"row_count\":3," ++
            "\"checksum_crc32c\":123," ++
            "\"index\":{\"kind\":\"ivf_flat\",\"path\":\"seg-1.ivf\",\"checksum_crc32c\":9}" ++
            "}]" ++
            "}",
    });

    var reader = try Reader.open(std.testing.allocator, root_path, "vector");
    defer reader.deinit();

    var snapshot = try reader.loadManifestSnapshot(std.testing.allocator);
    defer snapshot.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u64, 4), snapshot.generation);
    try std.testing.expectEqual(@as(usize, 1), snapshot.segments.len);
    try std.testing.expectEqual(@as(u32, 123), snapshot.segments[0].checksum_crc32c);
    try std.testing.expect(snapshot.segments[0].index != null);
    try std.testing.expectEqualStrings("seg-1.ivf", snapshot.segments[0].index.?.path);
}
