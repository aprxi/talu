//! Integration tests for db.BlockBuilder
//!
//! BlockBuilder serializes in-memory column data into the on-disk block format:
//! [Header 64B] -> [Columns...] -> [Arena] -> [Directory]

const std = @import("std");
const main = @import("main");
const db = main.db;

const BlockBuilder = db.BlockBuilder;
const types = db.types;

// ===== Lifecycle =====

test "BlockBuilder: init and deinit" {
    var builder = BlockBuilder.init(std.testing.allocator, 7, 42);
    defer builder.deinit();

    try std.testing.expectEqual(@as(u16, 7), builder.schema_id);
    try std.testing.expectEqual(@as(u32, 42), builder.rows);
}

// ===== addColumn =====

test "BlockBuilder: addColumn stores scalar column" {
    var builder = BlockBuilder.init(std.testing.allocator, 1, 2);
    defer builder.deinit();

    try builder.addColumn(1, .SCALAR, .U8, .RAW, 1, "ab", null, null);
    try std.testing.expectEqual(@as(usize, 1), builder.columns.items.len);
}

test "BlockBuilder: addColumn stores varbytes with offsets and lengths" {
    var builder = BlockBuilder.init(std.testing.allocator, 1, 2);
    defer builder.deinit();

    const offsets = [_]u32{ 0, 3 };
    const lengths = [_]u32{ 3, 4 };
    try builder.addColumn(1, .VARBYTES, .BINARY, .RAW, 0, "abcdefg", &offsets, &lengths);

    try std.testing.expectEqual(@as(usize, 1), builder.columns.items.len);
}

test "BlockBuilder: addColumn rejects mismatched offsets and lengths" {
    var builder = BlockBuilder.init(std.testing.allocator, 1, 1);
    defer builder.deinit();

    // offsets provided without lengths -> error
    const offsets = [_]u32{0};
    const result = builder.addColumn(1, .VARBYTES, .BINARY, .RAW, 0, "abc", &offsets, null);
    try std.testing.expectError(error.InvalidColumnLayout, result);
}

// ===== appendArena =====

test "BlockBuilder: appendArena returns sequential offsets" {
    var builder = BlockBuilder.init(std.testing.allocator, 1, 0);
    defer builder.deinit();

    const offset0 = try builder.appendArena("abc");
    const offset1 = try builder.appendArena("defg");

    try std.testing.expectEqual(@as(u32, 0), offset0);
    try std.testing.expectEqual(@as(u32, 3), offset1);
}

// ===== finalize =====

test "BlockBuilder: finalize produces valid block with header and CRC" {
    var builder = BlockBuilder.init(std.testing.allocator, 3, 1);
    defer builder.deinit();

    try builder.addColumn(5, .SCALAR, .U8, .RAW, 1, "xyz", null, null);
    _ = try builder.appendArena("arena");

    const block = try builder.finalize();
    defer std.testing.allocator.free(block);

    const header_len = @sizeOf(types.BlockHeader);
    const header = std.mem.bytesToValue(types.BlockHeader, block[0..header_len]);

    try std.testing.expectEqual(types.MagicValues.BLOCK, header.magic);
    try std.testing.expectEqual(@as(u32, @intCast(block.len)), header.block_len);
    try std.testing.expectEqual(@as(u16, 3), header.schema_id);
    try std.testing.expectEqual(@as(u32, 1), header.row_count);
    try std.testing.expect(header.crc32c != 0);
}

test "BlockBuilder: finalize with no columns produces valid empty block" {
    var builder = BlockBuilder.init(std.testing.allocator, 1, 0);
    defer builder.deinit();

    const block = try builder.finalize();
    defer std.testing.allocator.free(block);

    const header = std.mem.bytesToValue(types.BlockHeader, block[0..@sizeOf(types.BlockHeader)]);
    try std.testing.expectEqual(types.MagicValues.BLOCK, header.magic);
    try std.testing.expectEqual(@as(u32, 0), header.coldir_len);
}
