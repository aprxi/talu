//! StoreFS block reader supporting jump reads.
//!
//! Reads header and column chunks without loading entire blocks into memory.

const std = @import("std");
const types = @import("types.zig");

const Allocator = std.mem.Allocator;

/// Block-level read helpers for StoreFS segment files.
///
/// Column access:
/// - `readColumnDirectory`: Read column directory for a block (caller owns slice).
/// - `readColumnData`: Read a column chunk without loading the whole block.
/// - `readColumnDataInto`: Read a column chunk into a caller-provided buffer.
///
/// Footer-based block index (sealed segments):
/// - `readFooter`: Read footer block index; returns null if the footer trailer
///   magic is not `MagicValues.FOOTER`, in which case the caller must fall back
///   to header-based scanning.
///
/// Header-scan fallback (files without footer):
/// - `scanBlocksFromHeaders`: O(blocks) header reads; prefer `readFooter` first.
///
/// Unified helper:
/// - `getBlockIndex`: Prefer footer, fall back to header scan. Caller owns slice.
pub const BlockReader = struct {
    file: std.fs.File,
    allocator: Allocator,

    pub fn init(file: std.fs.File, allocator: Allocator) BlockReader {
        return .{ .file = file, .allocator = allocator };
    }

    pub fn readHeader(self: BlockReader, offset: u64) !types.BlockHeader {
        var header_bytes: [@sizeOf(types.BlockHeader)]u8 = undefined;
        const read_len = try self.file.preadAll(&header_bytes, offset);
        if (read_len != header_bytes.len) return error.UnexpectedEof;

        const header = std.mem.bytesToValue(types.BlockHeader, header_bytes[0..]);
        if (header.magic != types.MagicValues.BLOCK) return error.InvalidMagic;
        return header;
    }

    /// Reads the column directory for a block. Caller owns the returned slice.
    pub fn readColumnDirectory(self: BlockReader, header: types.BlockHeader, block_offset: u64) ![]types.ColumnDesc {
        if (header.coldir_len == 0) {
            return self.allocator.alloc(types.ColumnDesc, 0);
        }

        const entry_size: u32 = @intCast(@sizeOf(types.ColumnDesc));
        if (header.coldir_len % entry_size != 0) {
            return error.InvalidDirectory;
        }

        const count = @as(usize, header.coldir_len / entry_size);
        const end = @as(u64, header.coldir_off) + @as(u64, header.coldir_len);
        if (end > @as(u64, header.block_len)) return error.InvalidDirectory;

        const descs = try self.allocator.alloc(types.ColumnDesc, count);
        errdefer self.allocator.free(descs);

        const dir_bytes = std.mem.sliceAsBytes(descs);
        const read_len = try self.file.preadAll(dir_bytes, block_offset + @as(u64, header.coldir_off));
        if (read_len != dir_bytes.len) return error.UnexpectedEof;

        return descs;
    }

    /// Reads a specific column chunk without reading the whole block.
    pub fn readColumnData(
        self: BlockReader,
        block_offset: u64,
        desc: types.ColumnDesc,
        allocator: Allocator,
    ) ![]u8 {
        if (desc.data_len == 0) {
            return allocator.alloc(u8, 0);
        }

        if (block_offset > std.math.maxInt(u64) - @as(u64, desc.data_off)) {
            return error.SizeOverflow;
        }

        const data_len = @as(usize, desc.data_len);
        const buffer = try allocator.alloc(u8, data_len);
        errdefer allocator.free(buffer);

        const read_len = try self.file.preadAll(buffer, block_offset + @as(u64, desc.data_off));
        if (read_len != buffer.len) return error.UnexpectedEof;

        return buffer;
    }

    /// Reads a specific column chunk into a caller-provided buffer.
    pub fn readColumnDataInto(
        self: BlockReader,
        block_offset: u64,
        desc: types.ColumnDesc,
        dest: []u8,
    ) !void {
        if (desc.data_len == 0) {
            if (dest.len != 0) return error.InvalidColumnData;
            return;
        }

        if (block_offset > std.math.maxInt(u64) - @as(u64, desc.data_off)) {
            return error.SizeOverflow;
        }

        if (dest.len != @as(usize, desc.data_len)) return error.InvalidColumnData;

        const read_len = try self.file.preadAll(dest, block_offset + @as(u64, desc.data_off));
        if (read_len != dest.len) return error.UnexpectedEof;
    }

    /// Read footer block index from a sealed segment file.
    /// Returns slice of FooterBlockEntry if valid footer found, null if no footer.
    /// Caller owns the returned slice.
    ///
    /// Footer layout (from end of file):
    ///   [block data...][footer payload: FooterBlockEntry[]][FooterTrailer: 32 bytes]
    ///
    /// If the file lacks a valid footer (trailer magic != MagicValues.FOOTER),
    /// returns null and caller must fall back to header-based scanning.
    pub fn readFooter(self: BlockReader, file_size: u64) !?[]types.FooterBlockEntry {
        const trailer_size: u64 = @sizeOf(types.FooterTrailer);
        if (file_size < trailer_size) return null;

        // 1. Read last 32 bytes (FooterTrailer)
        var trailer_bytes: [@sizeOf(types.FooterTrailer)]u8 = undefined;
        const trailer_offset = file_size - trailer_size;
        const read_len = try self.file.preadAll(&trailer_bytes, trailer_offset);
        if (read_len != trailer_bytes.len) return null;

        const trailer = std.mem.bytesToValue(types.FooterTrailer, trailer_bytes[0..]);

        // 2. Validate magic
        if (trailer.magic != types.MagicValues.FOOTER) return null;

        // 3. Calculate footer payload location
        const footer_len = trailer.footer_len;
        if (footer_len == 0) {
            return try self.allocator.alloc(types.FooterBlockEntry, 0);
        }

        const entry_size: u32 = @intCast(@sizeOf(types.FooterBlockEntry));
        if (footer_len % entry_size != 0) return error.InvalidFooter;

        const entry_count = footer_len / entry_size;
        const payload_offset = file_size - trailer_size - @as(u64, footer_len);

        // 4. Read footer payload
        const entries = try self.allocator.alloc(types.FooterBlockEntry, entry_count);
        errdefer self.allocator.free(entries);

        const payload_bytes = std.mem.sliceAsBytes(entries);
        const payload_read = try self.file.preadAll(payload_bytes, payload_offset);
        if (payload_read != payload_bytes.len) return error.UnexpectedEof;

        return entries;
    }

    /// Scan a file for block offsets by iterating headers (fallback for files without footer).
    /// Returns slice of FooterBlockEntry built from header scanning.
    /// Caller owns the returned slice.
    ///
    /// This is O(blocks) header reads - use readFooter() first for sealed segments.
    pub fn scanBlocksFromHeaders(self: BlockReader, file_size: u64) ![]types.FooterBlockEntry {
        const header_size = @sizeOf(types.BlockHeader);
        var entries = std.ArrayList(types.FooterBlockEntry).empty;
        errdefer entries.deinit(self.allocator);

        var offset: u64 = 0;
        while (offset + header_size <= file_size) {
            var header_bytes: [@sizeOf(types.BlockHeader)]u8 = undefined;
            const read_len = try self.file.preadAll(&header_bytes, offset);
            if (read_len != header_bytes.len) break;

            const header = std.mem.bytesToValue(types.BlockHeader, header_bytes[0..]);
            if (header.magic != types.MagicValues.BLOCK) break;
            if (header.block_len < header_size) break;

            const next_offset = offset + @as(u64, header.block_len);
            if (next_offset > file_size) break;

            try entries.append(self.allocator, .{
                .block_off = offset,
                .block_len = header.block_len,
                .schema_id = header.schema_id,
            });

            offset = next_offset;
        }

        return entries.toOwnedSlice(self.allocator);
    }

    /// Get block index for a file, preferring footer if available.
    /// Falls back to header scanning if no valid footer.
    /// Caller owns the returned slice.
    pub fn getBlockIndex(self: BlockReader, file_size: u64) ![]types.FooterBlockEntry {
        // Try footer first (O(1) read for sealed segments)
        if (try self.readFooter(file_size)) |entries| {
            return entries;
        }
        // Fallback to header scanning (O(blocks) reads)
        return self.scanBlocksFromHeaders(file_size);
    }
};

fn buildTestBlock(allocator: Allocator) ![]u8 {
    const writer = @import("block_writer.zig");

    var builder = writer.BlockBuilder.init(allocator, 2, 1);
    defer builder.deinit();

    try builder.addColumn(10, .SCALAR, .U8, .RAW, 1, "data", null, null);
    _ = try builder.appendArena("arena");

    return builder.finalize();
}

test "BlockReader.init stores allocator" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var file = try tmp.dir.createFile("block.bin", .{ .read = true });
    defer file.close();

    const reader = BlockReader.init(file, std.testing.allocator);
    _ = reader;
}

test "BlockReader.readHeader reads header" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const block = try buildTestBlock(std.testing.allocator);
    defer std.testing.allocator.free(block);

    var file = try tmp.dir.createFile("block.bin", .{ .read = true });
    defer file.close();
    try file.writeAll(block);

    const reader = BlockReader.init(file, std.testing.allocator);
    const header = try reader.readHeader(0);

    try std.testing.expectEqual(types.MagicValues.BLOCK, header.magic);
    try std.testing.expectEqual(@as(u16, 2), header.schema_id);
}

test "BlockReader.readColumnDirectory reads descriptors" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const block = try buildTestBlock(std.testing.allocator);
    defer std.testing.allocator.free(block);

    var file = try tmp.dir.createFile("block.bin", .{ .read = true });
    defer file.close();
    try file.writeAll(block);

    const reader = BlockReader.init(file, std.testing.allocator);
    const header = try reader.readHeader(0);
    const descs = try reader.readColumnDirectory(header, 0);
    defer std.testing.allocator.free(descs);

    try std.testing.expectEqual(@as(usize, 1), descs.len);
    try std.testing.expectEqual(@as(u32, 10), descs[0].column_id);
}

test "BlockReader.readColumnData reads data" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const block = try buildTestBlock(std.testing.allocator);
    defer std.testing.allocator.free(block);

    var file = try tmp.dir.createFile("block.bin", .{ .read = true });
    defer file.close();
    try file.writeAll(block);

    const reader = BlockReader.init(file, std.testing.allocator);
    const header = try reader.readHeader(0);
    const descs = try reader.readColumnDirectory(header, 0);
    defer std.testing.allocator.free(descs);

    const data = try reader.readColumnData(0, descs[0], std.testing.allocator);
    defer std.testing.allocator.free(data);

    try std.testing.expectEqualSlices(u8, "data", data);
}

test "BlockReader.readColumnDataInto reads data" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const block = try buildTestBlock(std.testing.allocator);
    defer std.testing.allocator.free(block);

    var file = try tmp.dir.createFile("block.bin", .{ .read = true });
    defer file.close();
    try file.writeAll(block);

    const reader = BlockReader.init(file, std.testing.allocator);
    const header = try reader.readHeader(0);
    const descs = try reader.readColumnDirectory(header, 0);
    defer std.testing.allocator.free(descs);

    var buffer: [4]u8 = undefined;
    try reader.readColumnDataInto(0, descs[0], &buffer);

    try std.testing.expectEqualSlices(u8, "data", &buffer);
}

test "BlockReader.readFooter returns null for file without footer" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const block = try buildTestBlock(std.testing.allocator);
    defer std.testing.allocator.free(block);

    var file = try tmp.dir.createFile("block.bin", .{ .read = true });
    defer file.close();
    try file.writeAll(block);

    const reader = BlockReader.init(file, std.testing.allocator);
    const stat = try file.stat();

    // File has no footer, should return null
    const result = try reader.readFooter(stat.size);
    try std.testing.expect(result == null);
}

test "BlockReader.scanBlocksFromHeaders finds blocks" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const block1 = try buildTestBlock(std.testing.allocator);
    defer std.testing.allocator.free(block1);
    const block2 = try buildTestBlock(std.testing.allocator);
    defer std.testing.allocator.free(block2);

    var file = try tmp.dir.createFile("blocks.bin", .{ .read = true });
    defer file.close();
    try file.writeAll(block1);
    try file.writeAll(block2);

    const reader = BlockReader.init(file, std.testing.allocator);
    const stat = try file.stat();

    const entries = try reader.scanBlocksFromHeaders(stat.size);
    defer std.testing.allocator.free(entries);

    try std.testing.expectEqual(@as(usize, 2), entries.len);
    try std.testing.expectEqual(@as(u64, 0), entries[0].block_off);
    try std.testing.expectEqual(@as(u16, 2), entries[0].schema_id);
    try std.testing.expect(entries[1].block_off > 0);
    try std.testing.expectEqual(@as(u16, 2), entries[1].schema_id);
}

test "BlockReader.getBlockIndex falls back to header scan" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const block = try buildTestBlock(std.testing.allocator);
    defer std.testing.allocator.free(block);

    var file = try tmp.dir.createFile("block.bin", .{ .read = true });
    defer file.close();
    try file.writeAll(block);

    const reader = BlockReader.init(file, std.testing.allocator);
    const stat = try file.stat();

    // getBlockIndex should return block entries even without footer
    const entries = try reader.getBlockIndex(stat.size);
    defer std.testing.allocator.free(entries);

    try std.testing.expectEqual(@as(usize, 1), entries.len);
    try std.testing.expectEqual(@as(u64, 0), entries[0].block_off);
}

test "BlockReader.readFooter parses valid footer" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const block = try buildTestBlock(std.testing.allocator);
    defer std.testing.allocator.free(block);

    // Build a file with block + footer
    var file = try tmp.dir.createFile("sealed.bin", .{ .read = true });
    defer file.close();
    try file.writeAll(block);

    // Write footer payload (one FooterBlockEntry)
    const entry = types.FooterBlockEntry{
        .block_off = 0,
        .block_len = @intCast(block.len),
        .schema_id = 2,
    };
    try file.writeAll(std.mem.asBytes(&entry));

    // Write footer trailer
    const trailer = types.FooterTrailer{
        .magic = types.MagicValues.FOOTER,
        .version = 1,
        .flags = 0,
        .footer_len = @sizeOf(types.FooterBlockEntry),
        .footer_crc32c = 0,
        .segment_crc32c = 0,
        .reserved = [_]u8{0} ** 12,
    };
    try file.writeAll(std.mem.asBytes(&trailer));

    const reader = BlockReader.init(file, std.testing.allocator);
    const stat = try file.stat();

    const entries = try reader.readFooter(stat.size);
    try std.testing.expect(entries != null);
    defer std.testing.allocator.free(entries.?);

    try std.testing.expectEqual(@as(usize, 1), entries.?.len);
    try std.testing.expectEqual(@as(u64, 0), entries.?[0].block_off);
    try std.testing.expectEqual(@as(u16, 2), entries.?[0].schema_id);
}
