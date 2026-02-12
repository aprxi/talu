//! Integration tests for db.BlockReader
//!
//! BlockReader deserializes blocks from files using jump reads (pread),
//! without loading entire blocks into memory.

const std = @import("std");
const main = @import("main");
const db = main.db;

const BlockReader = db.BlockReader;
const BlockBuilder = db.BlockBuilder;
const types = db.types;

fn buildTestBlock(allocator: std.mem.Allocator) ![]u8 {
    var builder = BlockBuilder.init(allocator, 2, 1);
    defer builder.deinit();

    try builder.addColumn(10, .SCALAR, .U8, .RAW, 1, "data", null, null);
    _ = try builder.appendArena("arena");

    return builder.finalize();
}

// ===== init =====

test "BlockReader: init stores file and allocator" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var file = try tmp.dir.createFile("block.bin", .{ .read = true });
    defer file.close();

    const reader = BlockReader.init(file, std.testing.allocator);

    // Verify init wired up correctly by performing a read that exercises
    // both stored fields (file handle for I/O, allocator for result alloc).
    const block = try buildTestBlock(std.testing.allocator);
    defer std.testing.allocator.free(block);
    try file.writeAll(block);

    const header = try reader.readHeader(0);
    try std.testing.expectEqual(types.MagicValues.BLOCK, header.magic);
}

// ===== readHeader =====

test "BlockReader: readHeader reads valid block header" {
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
    try std.testing.expectEqual(@as(u32, 1), header.row_count);
}

// ===== readColumnDirectory =====

test "BlockReader: readColumnDirectory reads column descriptors" {
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

// ===== readColumnData =====

test "BlockReader: readColumnData reads column bytes" {
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

// ===== readColumnDataInto =====

test "BlockReader: readColumnDataInto reads into caller buffer" {
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

// ===== readFooter =====

test "BlockReader: readFooter returns null for file without footer" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const block = try buildTestBlock(std.testing.allocator);
    defer std.testing.allocator.free(block);

    var file = try tmp.dir.createFile("block.bin", .{ .read = true });
    defer file.close();
    try file.writeAll(block);

    const reader = BlockReader.init(file, std.testing.allocator);
    const stat = try file.stat();

    const result = try reader.readFooter(stat.size);
    try std.testing.expect(result == null);
}

test "BlockReader: readFooter parses valid footer" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const block = try buildTestBlock(std.testing.allocator);
    defer std.testing.allocator.free(block);

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
}

// ===== scanBlocksFromHeaders =====

test "BlockReader: scanBlocksFromHeaders finds multiple blocks" {
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
    try std.testing.expect(entries[1].block_off > 0);
}

// ===== getBlockIndex =====

test "BlockReader: getBlockIndex falls back to header scan without footer" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const block = try buildTestBlock(std.testing.allocator);
    defer std.testing.allocator.free(block);

    var file = try tmp.dir.createFile("block.bin", .{ .read = true });
    defer file.close();
    try file.writeAll(block);

    const reader = BlockReader.init(file, std.testing.allocator);
    const stat = try file.stat();

    const entries = try reader.getBlockIndex(stat.size);
    defer std.testing.allocator.free(entries);

    try std.testing.expectEqual(@as(usize, 1), entries.len);
}
