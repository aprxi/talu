//! Generic TaluDB key/value state adapter.
//!
//! This adapter materializes tiny mutable state in memory while preserving
//! append-only durability on disk via TaluDB blocks + manifest generations.

const std = @import("std");
const block_writer = @import("../block_writer.zig");
const block_reader = @import("../block_reader.zig");
const checksum = @import("../checksum.zig");
const db_manifest = @import("../manifest.zig");
const db_writer = @import("../writer.zig");
const db_reader = @import("../reader.zig");
const types = @import("../types.zig");

const Allocator = std.mem.Allocator;
const ColumnValue = db_writer.ColumnValue;

// Generic KV state schema (append-only change log + replay).
pub const schema_kv_state: u16 = 20;

const col_key_hash: u32 = 1;
const col_ts: u32 = 2;
const col_op_type: u32 = 3;
const col_key: u32 = 10;
const col_value: u32 = 20;
const default_compact_threshold: usize = 500;

const KvOp = enum(u8) {
    put = 1,
    delete = 2,
};

const StoredValue = struct {
    value: []u8,
    updated_at_ms: i64,
};

const CompactedSegment = struct {
    name: []u8,
    id: u128,
    min_ts: i64,
    max_ts: i64,
    row_count: u64,
    checksum_crc32c: u32,
};

pub const EntryRecord = struct {
    key: []u8,
    value: []u8,
    updated_at_ms: i64,

    pub fn deinit(self: *EntryRecord, allocator: Allocator) void {
        allocator.free(self.key);
        allocator.free(self.value);
    }
};

pub const ValueRecord = struct {
    value: []u8,
    updated_at_ms: i64,

    pub fn deinit(self: *ValueRecord, allocator: Allocator) void {
        allocator.free(self.value);
    }
};

/// KVStore stores latest key/value state in-memory and appends every mutation.
/// Thread safety: NOT thread-safe (single-writer lock semantics in Writer).
pub const KVStore = struct {
    allocator: Allocator,
    db_root: []u8,
    namespace: []u8,
    fs_writer: *db_writer.Writer,
    fs_reader: *db_reader.Reader,
    values: std.StringHashMap(StoredValue),
    stale_count: usize,
    tombstone_count: usize,
    compact_threshold: usize,

    pub fn init(allocator: Allocator, db_root: []const u8, namespace: []const u8) !KVStore {
        if (namespace.len == 0) return error.InvalidArgument;

        const db_root_copy = try allocator.dupe(u8, db_root);
        errdefer allocator.free(db_root_copy);
        const namespace_copy = try allocator.dupe(u8, namespace);
        errdefer allocator.free(namespace_copy);

        var writer_ptr = try allocator.create(db_writer.Writer);
        errdefer allocator.destroy(writer_ptr);
        writer_ptr.* = try db_writer.Writer.open(allocator, db_root, namespace);
        errdefer writer_ptr.deinit();

        var reader_ptr = try allocator.create(db_reader.Reader);
        errdefer allocator.destroy(reader_ptr);
        reader_ptr.* = try db_reader.Reader.open(allocator, db_root, namespace);
        errdefer reader_ptr.deinit();

        var store = KVStore{
            .allocator = allocator,
            .db_root = db_root_copy,
            .namespace = namespace_copy,
            .fs_writer = writer_ptr,
            .fs_reader = reader_ptr,
            .values = std.StringHashMap(StoredValue).init(allocator),
            .stale_count = 0,
            .tombstone_count = 0,
            .compact_threshold = default_compact_threshold,
        };
        errdefer store.deinit();

        try store.loadAll();
        return store;
    }

    pub fn deinit(self: *KVStore) void {
        self.fs_writer.flushBlock() catch {};
        self.fs_writer.deinit();
        self.allocator.destroy(self.fs_writer);

        self.fs_reader.deinit();
        self.allocator.destroy(self.fs_reader);

        self.clearState();
        self.values.deinit();
        self.allocator.free(self.namespace);
        self.allocator.free(self.db_root);
    }

    pub fn contains(self: *KVStore, key: []const u8) bool {
        return self.values.contains(key);
    }

    pub fn getCopy(self: *KVStore, allocator: Allocator, key: []const u8) !?[]u8 {
        const value = self.values.get(key) orelse return null;
        return try allocator.dupe(u8, value.value);
    }

    pub fn getEntryCopy(self: *KVStore, allocator: Allocator, key: []const u8) !?ValueRecord {
        const value = self.values.get(key) orelse return null;
        return .{
            .value = try allocator.dupe(u8, value.value),
            .updated_at_ms = value.updated_at_ms,
        };
    }

    pub fn put(self: *KVStore, key: []const u8, value: []const u8) !void {
        if (key.len == 0) return error.InvalidArgument;

        const now_ms = std.time.milliTimestamp();
        try self.appendEvent(.put, key, value, now_ms);

        if (self.values.getPtr(key)) |existing| {
            self.allocator.free(existing.value);
            existing.value = try self.allocator.dupe(u8, value);
            existing.updated_at_ms = now_ms;
            self.stale_count += 1;
        } else {
            const key_copy = try self.allocator.dupe(u8, key);
            errdefer self.allocator.free(key_copy);
            const value_copy = try self.allocator.dupe(u8, value);
            errdefer self.allocator.free(value_copy);
            try self.values.putNoClobber(key_copy, .{
                .value = value_copy,
                .updated_at_ms = now_ms,
            });
        }

        try self.maybeAutoCompact();
    }

    pub fn delete(self: *KVStore, key: []const u8) !bool {
        if (key.len == 0) return error.InvalidArgument;
        if (!self.values.contains(key)) return false;

        const now_ms = std.time.milliTimestamp();
        try self.appendEvent(.delete, key, &.{}, now_ms);

        self.removeEntry(key);
        self.stale_count += 1;
        self.tombstone_count += 1;

        try self.maybeAutoCompact();
        return true;
    }

    pub fn listEntries(self: *KVStore, allocator: Allocator) ![]EntryRecord {
        var out = try allocator.alloc(EntryRecord, self.values.count());
        errdefer {
            for (out) |*entry| {
                if (entry.key.len > 0) allocator.free(entry.key);
                if (entry.value.len > 0) allocator.free(entry.value);
            }
            allocator.free(out);
        }
        @memset(out, .{ .key = "", .value = &.{}, .updated_at_ms = 0 });

        var idx: usize = 0;
        var iter = self.values.iterator();
        while (iter.next()) |entry| {
            out[idx] = .{
                .key = try allocator.dupe(u8, entry.key_ptr.*),
                .value = try allocator.dupe(u8, entry.value_ptr.value),
                .updated_at_ms = entry.value_ptr.updated_at_ms,
            };
            idx += 1;
        }

        std.sort.pdq(EntryRecord, out, {}, lessThanEntryRecord);
        return out;
    }

    pub fn compact(self: *KVStore) !void {
        try self.fs_writer.flushBlock();

        const now_ms = std.time.milliTimestamp();
        const entries = try self.listEntries(self.allocator);
        defer freeEntryRecords(self.allocator, entries);

        var compacted_segment: ?CompactedSegment = null;
        if (entries.len > 0) {
            compacted_segment = try self.writeCompactedSegment(entries);
            errdefer if (compacted_segment) |segment| {
                self.fs_writer.dir.deleteFile(segment.name) catch {};
                self.allocator.free(segment.name);
            };
        }

        const manifest_path = try std.fs.path.join(self.allocator, &.{ self.db_root, self.namespace, "manifest.json" });
        defer self.allocator.free(manifest_path);

        var old_segments = std.ArrayList([]u8).empty;
        defer {
            for (old_segments.items) |path| self.allocator.free(path);
            old_segments.deinit(self.allocator);
        }

        var attempt: usize = 0;
        while (true) {
            var current = loadManifestOrEmpty(self.allocator, manifest_path) catch |err| switch (err) {
                error.FileNotFound => blk: {
                    const empty = try self.allocator.alloc(db_manifest.SegmentEntry, 0);
                    break :blk db_manifest.Manifest{
                        .version = 1,
                        .generation = 0,
                        .segments = empty,
                        .last_compaction_ts = 0,
                    };
                },
                else => return err,
            };
            defer current.deinit(self.allocator);

            var next_segments = if (compacted_segment == null)
                try self.allocator.alloc(db_manifest.SegmentEntry, 0)
            else
                try self.allocator.alloc(db_manifest.SegmentEntry, 1);
            errdefer {
                for (next_segments) |entry| {
                    self.allocator.free(entry.path);
                    if (entry.index) |index_meta| self.allocator.free(index_meta.path);
                }
                self.allocator.free(next_segments);
            }

            if (compacted_segment) |segment| {
                next_segments[0] = .{
                    .id = segment.id,
                    .path = try self.allocator.dupe(u8, segment.name),
                    .min_ts = segment.min_ts,
                    .max_ts = segment.max_ts,
                    .row_count = segment.row_count,
                    .checksum_crc32c = segment.checksum_crc32c,
                    .index = null,
                };
            }

            var next = db_manifest.Manifest{
                .version = current.version,
                .generation = current.generation,
                .segments = next_segments,
                .last_compaction_ts = now_ms,
            };
            defer next.deinit(self.allocator);

            _ = next.saveNextGeneration(self.allocator, manifest_path, current.generation) catch |err| {
                if (err == error.ManifestGenerationConflict and attempt < 8) {
                    attempt += 1;
                    continue;
                }
                return err;
            };

            old_segments.clearRetainingCapacity();
            for (current.segments) |segment| {
                if (compacted_segment) |new_segment| {
                    if (std.mem.eql(u8, segment.path, new_segment.name)) continue;
                }
                try old_segments.append(self.allocator, try self.allocator.dupe(u8, segment.path));
            }
            break;
        }

        try self.fs_writer.data_file.setEndPos(0);
        try self.fs_writer.data_file.seekTo(0);
        try self.fs_writer.data_file.sync();
        try self.fs_writer.wal_writer.file.setEndPos(0);
        try self.fs_writer.wal_writer.file.seekTo(0);
        try self.fs_writer.wal_writer.file.sync();
        if (comptime @hasDecl(std.posix, "fsync")) {
            try std.posix.fsync(self.fs_writer.dir.fd);
        }

        for (old_segments.items) |segment_path| {
            self.fs_writer.dir.deleteFile(segment_path) catch {};
        }

        if (compacted_segment) |segment| {
            self.allocator.free(segment.name);
        }
        self.stale_count = 0;
        self.tombstone_count = 0;
        _ = try self.fs_reader.refreshIfChanged();
    }

    pub fn flush(self: *KVStore) !void {
        try self.fs_writer.flushBlock();
        try self.maybeAutoCompact();
    }

    fn loadAll(self: *KVStore) !void {
        self.clearState();
        self.stale_count = 0;
        self.tombstone_count = 0;

        _ = try self.fs_reader.refreshIfChanged();
        const blocks = try self.fs_reader.getBlocks(self.allocator);
        defer self.allocator.free(blocks);

        var row_count_total: usize = 0;

        for (blocks) |block| {
            var handle = self.fs_reader.openBlockReadOnly(block.path) catch continue;
            defer self.fs_reader.closeBlock(&handle);

            const reader = block_reader.BlockReader.init(handle.file, self.allocator);
            const header = reader.readHeader(block.offset) catch continue;
            if (header.schema_id != schema_kv_state or header.row_count == 0) continue;

            const descs = reader.readColumnDirectory(header, block.offset) catch continue;
            defer self.allocator.free(descs);

            const op_desc = findColumn(descs, col_op_type) orelse continue;
            const ts_desc = findColumn(descs, col_ts) orelse continue;
            const key_desc = findColumn(descs, col_key) orelse continue;
            const value_desc = findColumn(descs, col_value) orelse continue;

            const op_buf = reader.readColumnData(block.offset, op_desc, self.allocator) catch continue;
            defer self.allocator.free(op_buf);
            _ = try checkedRowCount(header.row_count, op_buf.len, @sizeOf(u8));

            const ts_buf = reader.readColumnData(block.offset, ts_desc, self.allocator) catch continue;
            defer self.allocator.free(ts_buf);
            _ = try checkedRowCount(header.row_count, ts_buf.len, @sizeOf(i64));

            var key_buffers = readVarBytesBuffers(handle.file, block.offset, key_desc, header.row_count, self.allocator) catch continue;
            defer key_buffers.deinit(self.allocator);
            var value_buffers = readVarBytesBuffers(handle.file, block.offset, value_desc, header.row_count, self.allocator) catch continue;
            defer value_buffers.deinit(self.allocator);

            const ops = @as([*]const u8, @ptrCast(@alignCast(op_buf.ptr)))[0..header.row_count];
            const ts = @as([*]const i64, @ptrCast(@alignCast(ts_buf.ptr)))[0..header.row_count];
            for (0..header.row_count) |row_idx| {
                row_count_total += 1;
                const op = try parseKvOp(ops[row_idx]);
                const key = try key_buffers.sliceForRow(row_idx);
                const value = try value_buffers.sliceForRow(row_idx);
                try self.applyReplay(op, key, value, ts[row_idx]);
            }
        }

        self.stale_count = row_count_total - self.values.count();
    }

    fn maybeAutoCompact(self: *KVStore) !void {
        if (self.stale_count < self.compact_threshold) return;
        try self.compact();
    }

    fn appendEvent(self: *KVStore, op: KvOp, key: []const u8, value: []const u8, ts_ms: i64) !void {
        var key_hash_value = computeHash(key);
        var ts_value = ts_ms;
        var op_value: u8 = @intFromEnum(op);

        const columns = [_]ColumnValue{
            .{ .column_id = col_key_hash, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&key_hash_value) },
            .{ .column_id = col_ts, .shape = .SCALAR, .phys_type = .I64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&ts_value) },
            .{ .column_id = col_op_type, .shape = .SCALAR, .phys_type = .U8, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&op_value) },
            .{ .column_id = col_key, .shape = .VARBYTES, .phys_type = .BINARY, .encoding = .RAW, .dims = 0, .data = key },
            .{ .column_id = col_value, .shape = .VARBYTES, .phys_type = .BINARY, .encoding = .RAW, .dims = 0, .data = value },
        };

        try self.fs_writer.appendRow(schema_kv_state, &columns);
    }

    fn removeEntry(self: *KVStore, key: []const u8) void {
        if (self.values.fetchRemove(key)) |removed| {
            self.allocator.free(removed.key);
            self.allocator.free(removed.value.value);
        }
    }

    fn applyReplay(self: *KVStore, op: KvOp, key: []const u8, value: []const u8, ts_ms: i64) !void {
        switch (op) {
            .put => {
                if (self.values.getPtr(key)) |existing| {
                    self.allocator.free(existing.value);
                    existing.value = try self.allocator.dupe(u8, value);
                    existing.updated_at_ms = ts_ms;
                    return;
                }

                const key_copy = try self.allocator.dupe(u8, key);
                errdefer self.allocator.free(key_copy);
                const value_copy = try self.allocator.dupe(u8, value);
                errdefer self.allocator.free(value_copy);
                try self.values.putNoClobber(key_copy, .{
                    .value = value_copy,
                    .updated_at_ms = ts_ms,
                });
            },
            .delete => self.removeEntry(key),
        }
    }

    fn clearState(self: *KVStore) void {
        var iter = self.values.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.value);
        }
        self.values.clearRetainingCapacity();
    }

    fn writeCompactedSegment(self: *KVStore, entries: []const EntryRecord) !CompactedSegment {
        var uuid_bytes: [16]u8 = undefined;
        std.crypto.random.bytes(&uuid_bytes);
        uuid_bytes[6] = (uuid_bytes[6] & 0x0f) | 0x40;
        uuid_bytes[8] = (uuid_bytes[8] & 0x3f) | 0x80;

        var hex_buf: [32]u8 = undefined;
        const hex_chars = "0123456789abcdef";
        for (uuid_bytes, 0..) |byte, i| {
            hex_buf[i * 2] = hex_chars[byte >> 4];
            hex_buf[i * 2 + 1] = hex_chars[byte & 0x0f];
        }

        const seg_name = try std.fmt.allocPrint(self.allocator, "seg-{s}.talu", .{hex_buf});
        errdefer self.allocator.free(seg_name);

        if (entries.len > std.math.maxInt(u32)) return error.RowCountOverflow;
        const row_count_u32: u32 = @intCast(entries.len);

        var hashes = try self.allocator.alloc(u64, entries.len);
        defer self.allocator.free(hashes);

        var ts = try self.allocator.alloc(i64, entries.len);
        defer self.allocator.free(ts);

        var ops = try self.allocator.alloc(u8, entries.len);
        defer self.allocator.free(ops);

        var key_data = std.ArrayList(u8).empty;
        defer key_data.deinit(self.allocator);
        var key_offsets = std.ArrayList(u32).empty;
        defer key_offsets.deinit(self.allocator);
        var key_lengths = std.ArrayList(u32).empty;
        defer key_lengths.deinit(self.allocator);

        var value_data = std.ArrayList(u8).empty;
        defer value_data.deinit(self.allocator);
        var value_offsets = std.ArrayList(u32).empty;
        defer value_offsets.deinit(self.allocator);
        var value_lengths = std.ArrayList(u32).empty;
        defer value_lengths.deinit(self.allocator);

        var min_ts: i64 = std.math.maxInt(i64);
        var max_ts: i64 = std.math.minInt(i64);

        for (entries, 0..) |entry, idx| {
            hashes[idx] = computeHash(entry.key);
            ts[idx] = entry.updated_at_ms;
            ops[idx] = @intFromEnum(KvOp.put);
            min_ts = @min(min_ts, ts[idx]);
            max_ts = @max(max_ts, ts[idx]);

            if (key_data.items.len > std.math.maxInt(u32)) return error.SizeOverflow;
            try key_offsets.append(self.allocator, @intCast(key_data.items.len));
            try key_lengths.append(self.allocator, @intCast(entry.key.len));
            try key_data.appendSlice(self.allocator, entry.key);

            if (value_data.items.len > std.math.maxInt(u32)) return error.SizeOverflow;
            try value_offsets.append(self.allocator, @intCast(value_data.items.len));
            try value_lengths.append(self.allocator, @intCast(entry.value.len));
            try value_data.appendSlice(self.allocator, entry.value);
        }

        var builder = block_writer.BlockBuilder.init(self.allocator, schema_kv_state, row_count_u32);
        defer builder.deinit();
        builder.flags.has_ts_range = entries.len > 0;
        builder.min_ts = if (entries.len > 0) min_ts else 0;
        builder.max_ts = if (entries.len > 0) max_ts else 0;

        try builder.addColumn(col_key_hash, .SCALAR, .U64, .RAW, 1, std.mem.sliceAsBytes(hashes), null, null);
        try builder.addColumn(col_ts, .SCALAR, .I64, .RAW, 1, std.mem.sliceAsBytes(ts), null, null);
        try builder.addColumn(col_op_type, .SCALAR, .U8, .RAW, 1, ops, null, null);
        try builder.addColumn(col_key, .VARBYTES, .BINARY, .RAW, 0, key_data.items, key_offsets.items, key_lengths.items);
        try builder.addColumn(col_value, .VARBYTES, .BINARY, .RAW, 0, value_data.items, value_offsets.items, value_lengths.items);

        const block = try builder.finalize();
        defer self.allocator.free(block);

        var file = try self.fs_writer.dir.createFile(seg_name, .{ .read = true, .truncate = true });
        errdefer {
            file.close();
            self.fs_writer.dir.deleteFile(seg_name) catch {};
        }
        defer file.close();

        try file.writeAll(block);
        try file.sync();

        var entry = types.FooterBlockEntry{
            .block_off = 0,
            .block_len = @intCast(block.len),
            .schema_id = schema_kv_state,
            ._reserved = 0,
        };
        const footer_payload = std.mem.asBytes(&entry);
        const segment_crc_before_footer = try crc32cFilePrefix(file, block.len);

        const trailer = types.FooterTrailer{
            .magic = types.MagicValues.FOOTER,
            .version = 1,
            .flags = 0,
            .footer_len = @intCast(footer_payload.len),
            .footer_crc32c = checksum.crc32c(footer_payload),
            .segment_crc32c = segment_crc_before_footer,
            .reserved = [_]u8{0} ** 12,
        };

        try file.seekTo(block.len);
        try file.writeAll(footer_payload);
        try file.writeAll(std.mem.asBytes(&trailer));
        try file.sync();

        const stat = try file.stat();
        const full_crc = try crc32cFilePrefix(file, stat.size);
        const id = std.mem.readInt(u128, &uuid_bytes, .big);

        return .{
            .name = seg_name,
            .id = id,
            .min_ts = if (entries.len > 0) min_ts else 0,
            .max_ts = if (entries.len > 0) max_ts else 0,
            .row_count = entries.len,
            .checksum_crc32c = full_crc,
        };
    }
};

fn lessThanEntryRecord(_: void, a: EntryRecord, b: EntryRecord) bool {
    return std.mem.lessThan(u8, a.key, b.key);
}

pub fn freeEntryRecords(allocator: Allocator, entries: []EntryRecord) void {
    for (entries) |*entry| {
        entry.deinit(allocator);
    }
    allocator.free(entries);
}

fn parseKvOp(raw: u8) !KvOp {
    return switch (raw) {
        1 => .put,
        2 => .delete,
        else => error.InvalidColumnData,
    };
}

fn computeHash(value: []const u8) u64 {
    return std.hash.Wyhash.hash(0, value);
}

fn findColumn(descs: []const types.ColumnDesc, column_id: u32) ?types.ColumnDesc {
    for (descs) |desc| {
        if (desc.column_id == column_id) return desc;
    }
    return null;
}

fn checkedRowCount(row_count: u32, data_len: usize, value_size: usize) !usize {
    const expected = @as(usize, row_count) * value_size;
    if (expected != data_len) return error.InvalidColumnData;
    return @as(usize, row_count);
}

const VarBytesBuffers = struct {
    data: []u8,
    offsets: []u32,
    lengths: []u32,

    fn deinit(self: *VarBytesBuffers, allocator: Allocator) void {
        allocator.free(self.data);
        allocator.free(self.offsets);
        allocator.free(self.lengths);
    }

    fn sliceForRow(self: VarBytesBuffers, row_idx: usize) ![]const u8 {
        if (row_idx >= self.offsets.len or row_idx >= self.lengths.len) return error.InvalidColumnData;
        const start = @as(usize, self.offsets[row_idx]);
        const length = @as(usize, self.lengths[row_idx]);
        const end = start + length;
        if (end > self.data.len) return error.InvalidColumnData;
        return self.data[start..end];
    }
};

fn readVarBytesBuffers(
    file: std.fs.File,
    block_offset: u64,
    desc: types.ColumnDesc,
    row_count: u32,
    allocator: Allocator,
) !VarBytesBuffers {
    if (desc.offsets_off == 0 or desc.lengths_off == 0) return error.InvalidColumnLayout;

    const reader = block_reader.BlockReader.init(file, allocator);
    const data = try reader.readColumnData(block_offset, desc, allocator);
    errdefer allocator.free(data);

    const offsets = try readU32Array(file, block_offset + @as(u64, desc.offsets_off), row_count, allocator);
    errdefer allocator.free(offsets);
    const lengths = try readU32Array(file, block_offset + @as(u64, desc.lengths_off), row_count, allocator);
    errdefer allocator.free(lengths);

    return .{
        .data = data,
        .offsets = offsets,
        .lengths = lengths,
    };
}

fn readU32Array(file: std.fs.File, offset: u64, count: u32, allocator: Allocator) ![]u32 {
    const bytes_len = @as(usize, count) * @sizeOf(u32);
    const bytes = try allocator.alloc(u8, bytes_len);
    defer allocator.free(bytes);

    const read_len = try file.preadAll(bytes, offset);
    if (read_len != bytes.len) return error.UnexpectedEof;

    const out = try allocator.alloc(u32, count);
    for (0..count) |idx| {
        out[idx] = std.mem.readInt(u32, bytes[idx * 4 ..][0..4], .little);
    }
    return out;
}

fn loadManifestOrEmpty(allocator: Allocator, path: []const u8) !db_manifest.Manifest {
    return db_manifest.Manifest.load(allocator, path) catch |err| switch (err) {
        error.FileNotFound => blk: {
            const empty = try allocator.alloc(db_manifest.SegmentEntry, 0);
            break :blk db_manifest.Manifest{
                .version = 1,
                .generation = 0,
                .segments = empty,
                .last_compaction_ts = 0,
            };
        },
        else => return err,
    };
}

fn crc32cFilePrefix(file: std.fs.File, len: u64) !u32 {
    var crc = std.hash.crc.Crc32Iscsi.init();
    var buf: [64 * 1024]u8 = undefined;
    var offset: u64 = 0;

    while (offset < len) {
        const remaining = len - offset;
        const chunk_len: usize = @intCast(@min(remaining, buf.len));
        const chunk = buf[0..chunk_len];
        const read_len = try file.preadAll(chunk, offset);
        if (read_len != chunk_len) return error.UnexpectedEof;
        crc.update(chunk);
        offset += @as(u64, @intCast(chunk_len));
    }

    return crc.final();
}

test "KVStore.put and KVStore.delete replay across restart" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    {
        var store = try KVStore.init(std.testing.allocator, root, "kv_state_test");
        defer store.deinit();

        try store.put("a", "1");
        try store.put("b", "2");
        try std.testing.expect(try store.delete("a"));
        try std.testing.expect(!(try store.delete("a")));
        try store.flush();
    }

    {
        var store = try KVStore.init(std.testing.allocator, root, "kv_state_test");
        defer store.deinit();

        try std.testing.expect(!store.contains("a"));
        try std.testing.expect(store.contains("b"));
        const v = (try store.getCopy(std.testing.allocator, "b")).?;
        defer std.testing.allocator.free(v);
        try std.testing.expectEqualStrings("2", v);
    }
}

test "KVStore.put overwrites existing value" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    var store = try KVStore.init(std.testing.allocator, root, "kv_state_test");
    defer store.deinit();

    try store.put("k", "v1");
    try store.put("k", "v2");

    const v = (try store.getCopy(std.testing.allocator, "k")).?;
    defer std.testing.allocator.free(v);
    try std.testing.expectEqualStrings("v2", v);
}

test "KVStore.getEntryCopy returns value with updated_at_ms" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    var store = try KVStore.init(std.testing.allocator, root, "kv_state_test");
    defer store.deinit();

    try store.put("k", "v");

    var entry = (try store.getEntryCopy(std.testing.allocator, "k")).?;
    defer entry.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("v", entry.value);
    try std.testing.expect(entry.updated_at_ms > 0);
}

test "KVStore.compact rewrites manifest to current active state only" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    {
        var store = try KVStore.init(std.testing.allocator, root, "kv_state_test");
        defer store.deinit();

        try store.put("q1", "111");
        try store.put("q1", "222");
        try store.put("q2", "x");
        try std.testing.expect(try store.delete("q2"));
        try store.compact();
    }

    {
        var store = try KVStore.init(std.testing.allocator, root, "kv_state_test");
        defer store.deinit();

        try std.testing.expect(store.contains("q1"));
        try std.testing.expect(!store.contains("q2"));
        const v = (try store.getCopy(std.testing.allocator, "q1")).?;
        defer std.testing.allocator.free(v);
        try std.testing.expectEqualStrings("222", v);
    }

    const manifest_path = try std.fs.path.join(std.testing.allocator, &.{ root, "kv_state_test", "manifest.json" });
    defer std.testing.allocator.free(manifest_path);
    var manifest = try db_manifest.Manifest.load(std.testing.allocator, manifest_path);
    defer manifest.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 1), manifest.segments.len);
}
