//! StoreFS block writer for the physical-layer layout.
//!
//! Layout: [Header 64B] -> [Columns...] -> [Arena] -> [Directory]

const std = @import("std");
const checksum = @import("checksum.zig");
const types = @import("types.zig");

const Allocator = std.mem.Allocator;

pub const BlockBuilder = struct {
    allocator: Allocator,
    schema_id: u16,
    rows: u32,
    columns: std.ArrayList(ColumnData),
    arena: std.ArrayList(u8),
    flags: types.BlockFlags,
    min_ts: i64,
    max_ts: i64,
    primary_centroid_id: u32,
    primary_vec_dims: u16,
    primary_vec_type: u8,

    pub fn init(allocator: Allocator, schema_id: u16, rows: u32) BlockBuilder {
        return .{
            .allocator = allocator,
            .schema_id = schema_id,
            .rows = rows,
            .columns = .empty,
            .arena = .empty,
            .flags = .{
                .has_ts_range = false,
                .has_arena = false,
                .has_primary_vector_hints = false,
                .blob_encoding_zstd = false,
            },
            .min_ts = 0,
            .max_ts = 0,
            .primary_centroid_id = 0,
            .primary_vec_dims = 0,
            .primary_vec_type = 0,
        };
    }

    pub fn deinit(self: *BlockBuilder) void {
        for (self.columns.items) |*column| {
            column.deinit(self.allocator);
        }
        self.columns.deinit(self.allocator);
        self.arena.deinit(self.allocator);
    }

    pub fn addColumn(
        self: *BlockBuilder,
        column_id: u32,
        shape: types.ColumnShape,
        phys_type: types.PhysicalType,
        encoding: types.Encoding,
        dims: u16,
        data: []const u8,
        offsets: ?[]const u32,
        lengths: ?[]const u32,
    ) !void {
        if ((offsets == null) != (lengths == null)) {
            return error.InvalidColumnLayout;
        }
        if (offsets != null and offsets.?.len != lengths.?.len) {
            return error.InvalidColumnLayout;
        }

        var column = ColumnData.init(column_id, shape, phys_type, encoding, dims);
        errdefer column.deinit(self.allocator);

        try column.data.appendSlice(self.allocator, data);
        if (offsets) |offsets_slice| {
            try column.offsets.appendSlice(self.allocator, offsets_slice);
        }
        if (lengths) |lengths_slice| {
            try column.lengths.appendSlice(self.allocator, lengths_slice);
        }

        try self.columns.append(self.allocator, column);
    }

    pub fn appendArena(self: *BlockBuilder, bytes: []const u8) !u32 {
        const offset = try checkedU32(@as(u64, self.arena.items.len));
        try self.arena.appendSlice(self.allocator, bytes);
        return offset;
    }

    /// Finalizes the block buffer. Caller owns the returned slice.
    pub fn finalize(self: *BlockBuilder) ![]u8 {
        const header_len: usize = @sizeOf(types.BlockHeader);
        const column_count = self.columns.items.len;
        const coldir_len_u64 = @as(u64, column_count) * @sizeOf(types.ColumnDesc);
        if (coldir_len_u64 > std.math.maxInt(u32)) return error.SizeOverflow;

        var descs = try self.allocator.alloc(types.ColumnDesc, column_count);
        defer self.allocator.free(descs);

        var cursor: u64 = header_len;
        for (self.columns.items, 0..) |column, idx| {
            var desc = std.mem.zeroes(types.ColumnDesc);
            desc.column_id = column.column_id;
            desc.shape = @intFromEnum(column.shape);
            desc.phys_type = @intFromEnum(column.phys_type);
            desc.encoding = @intFromEnum(column.encoding);
            desc.dims = column.dims;

            desc.data_off = try checkedU32(cursor);
            desc.data_len = try checkedU32(@as(u64, column.data.items.len));
            cursor += @as(u64, column.data.items.len);

            if (column.offsets.items.len > 0) {
                const offsets_len_bytes = @as(u64, column.offsets.items.len) * @sizeOf(u32);
                if (offsets_len_bytes > std.math.maxInt(u32)) return error.SizeOverflow;
                desc.offsets_off = try checkedU32(cursor);
                cursor += offsets_len_bytes;
            }

            if (column.lengths.items.len > 0) {
                const lengths_len_bytes = @as(u64, column.lengths.items.len) * @sizeOf(u32);
                if (lengths_len_bytes > std.math.maxInt(u32)) return error.SizeOverflow;
                desc.lengths_off = try checkedU32(cursor);
                cursor += lengths_len_bytes;
            }

            descs[idx] = desc;
        }

        const arena_len = self.arena.items.len;
        const arena_off = if (arena_len > 0) try checkedU32(cursor) else 0;
        cursor += @as(u64, arena_len);

        const coldir_off = try checkedU32(cursor);
        cursor += coldir_len_u64;

        if (cursor > std.math.maxInt(u32)) return error.BlockTooLarge;
        const total_len: usize = @intCast(cursor);

        var buffer = try self.allocator.alloc(u8, total_len);
        errdefer self.allocator.free(buffer);

        var flags = self.flags;
        if (arena_len > 0) flags.has_arena = true;

        var header = std.mem.zeroes(types.BlockHeader);
        header.magic = types.MagicValues.BLOCK;
        header.version = 1;
        header.header_len = @intCast(header_len);
        header.flags = @bitCast(flags);
        header.schema_id = self.schema_id;
        header.row_count = self.rows;
        header.block_len = @intCast(total_len);
        header.crc32c = 0;
        header.coldir_off = coldir_off;
        header.coldir_len = @intCast(coldir_len_u64);
        header.arena_off = arena_off;
        header.arena_len = try checkedU32(@as(u64, arena_len));
        header.min_ts = self.min_ts;
        header.max_ts = self.max_ts;
        header.primary_centroid_id = self.primary_centroid_id;
        header.primary_vec_dims = self.primary_vec_dims;
        header.primary_vec_type = self.primary_vec_type;

        std.mem.copyForwards(u8, buffer[0..header_len], std.mem.asBytes(&header));

        for (self.columns.items, 0..) |column, idx| {
            const desc = descs[idx];
            const data_off = @as(usize, desc.data_off);
            const data_len = @as(usize, desc.data_len);
            std.mem.copyForwards(u8, buffer[data_off .. data_off + data_len], column.data.items);

            if (column.offsets.items.len > 0) {
                const offsets_off = @as(usize, desc.offsets_off);
                const offsets_bytes = std.mem.sliceAsBytes(column.offsets.items);
                std.mem.copyForwards(u8, buffer[offsets_off .. offsets_off + offsets_bytes.len], offsets_bytes);
            }

            if (column.lengths.items.len > 0) {
                const lengths_off = @as(usize, desc.lengths_off);
                const lengths_bytes = std.mem.sliceAsBytes(column.lengths.items);
                std.mem.copyForwards(u8, buffer[lengths_off .. lengths_off + lengths_bytes.len], lengths_bytes);
            }
        }

        if (arena_len > 0) {
            const arena_start = @as(usize, arena_off);
            std.mem.copyForwards(u8, buffer[arena_start .. arena_start + arena_len], self.arena.items);
        }

        const dir_bytes = std.mem.sliceAsBytes(descs);
        const dir_start = @as(usize, coldir_off);
        std.mem.copyForwards(u8, buffer[dir_start .. dir_start + dir_bytes.len], dir_bytes);

        const crc = checksum.crc32c(buffer);
        header.crc32c = crc;
        std.mem.copyForwards(u8, buffer[0..header_len], std.mem.asBytes(&header));

        return buffer;
    }
};

const ColumnData = struct {
    column_id: u32,
    shape: types.ColumnShape,
    phys_type: types.PhysicalType,
    encoding: types.Encoding,
    dims: u16,
    data: std.ArrayList(u8),
    offsets: std.ArrayList(u32),
    lengths: std.ArrayList(u32),

    fn init(
        column_id: u32,
        shape: types.ColumnShape,
        phys_type: types.PhysicalType,
        encoding: types.Encoding,
        dims: u16,
    ) ColumnData {
        return .{
            .column_id = column_id,
            .shape = shape,
            .phys_type = phys_type,
            .encoding = encoding,
            .dims = dims,
            .data = .empty,
            .offsets = .empty,
            .lengths = .empty,
        };
    }

    fn deinit(self: *ColumnData, allocator: Allocator) void {
        self.data.deinit(allocator);
        self.offsets.deinit(allocator);
        self.lengths.deinit(allocator);
    }
};

fn checkedU32(value: u64) !u32 {
    if (value > std.math.maxInt(u32)) return error.SizeOverflow;
    return @intCast(value);
}

test "BlockBuilder.init sets fields" {
    var builder = BlockBuilder.init(std.testing.allocator, 7, 42);
    defer builder.deinit();

    try std.testing.expectEqual(@as(u16, 7), builder.schema_id);
    try std.testing.expectEqual(@as(u32, 42), builder.rows);
    try std.testing.expectEqual(@as(usize, 0), builder.columns.items.len);
    try std.testing.expectEqual(@as(usize, 0), builder.arena.items.len);
}

test "BlockBuilder.addColumn stores column data" {
    var builder = BlockBuilder.init(std.testing.allocator, 1, 2);
    defer builder.deinit();

    const payload = "abcd";
    try builder.addColumn(1, .SCALAR, .U8, .RAW, 1, payload, null, null);

    try std.testing.expectEqual(@as(usize, 1), builder.columns.items.len);
    try std.testing.expectEqualSlices(u8, payload, builder.columns.items[0].data.items);
}

test "BlockBuilder.appendArena returns offset" {
    var builder = BlockBuilder.init(std.testing.allocator, 1, 0);
    defer builder.deinit();

    const offset0 = try builder.appendArena("abc");
    const offset1 = try builder.appendArena("defg");

    try std.testing.expectEqual(@as(u32, 0), offset0);
    try std.testing.expectEqual(@as(u32, 3), offset1);
    try std.testing.expectEqualSlices(u8, "abcdefg", builder.arena.items);
}

test "BlockBuilder.finalize writes header and crc" {
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
    try std.testing.expectEqual(@as(u32, @sizeOf(types.ColumnDesc)), header.coldir_len);
    try std.testing.expectEqual(@as(u32, "arena".len), header.arena_len);

    var scratch = try std.testing.allocator.dupe(u8, block);
    defer std.testing.allocator.free(scratch);

    var header_copy = std.mem.bytesToValue(types.BlockHeader, scratch[0..header_len]);
    header_copy.crc32c = 0;
    std.mem.copyForwards(u8, scratch[0..header_len], std.mem.asBytes(&header_copy));

    const crc = checksum.crc32c(scratch);
    try std.testing.expectEqual(crc, header.crc32c);
}
