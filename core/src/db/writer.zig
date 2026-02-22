//! TaluDB write path: lock, WAL, in-memory buffer, and block flush.
//!
//! This module coordinates durability via a WAL and emits append-only blocks
//! into the active `current.talu` segment. When the active segment exceeds
//! `max_segment_size`, the writer seals it as `seg-<uuid>.talu` and updates
//! the manifest atomically before creating a fresh `current.talu`.

const std = @import("std");
const block_writer = @import("block_writer.zig");
const block_reader = @import("block_reader.zig");
const checksum = @import("checksum.zig");
const lock = @import("lock.zig");
const manifest_mod = @import("manifest.zig");
const types = @import("types.zig");
const wal = @import("wal.zig");

const Allocator = std.mem.Allocator;

pub const ColumnValue = struct {
    column_id: u32,
    shape: types.ColumnShape,
    phys_type: types.PhysicalType,
    encoding: types.Encoding,
    dims: u16,
    data: []const u8,
};

fn columnSlice(column: ColumnBatch, row_idx: u32, row_count: u32) ![]const u8 {
    if (column.shape == .VARBYTES) return error.InvalidColumnData;

    const element_size: usize = switch (column.phys_type) {
        .U8, .I8 => 1,
        .U16, .I16, .F16, .BF16 => 2,
        .U32, .I32, .F32 => 4,
        .U64, .I64, .F64 => 8,
        .BINARY => return error.InvalidColumnData,
    };

    const dims: usize = switch (column.shape) {
        .SCALAR => 1,
        .VECTOR => blk: {
            if (column.dims == 0) return error.InvalidColumnData;
            break :blk @as(usize, column.dims);
        },
        .VARBYTES => return error.InvalidColumnData,
    };

    const stride = element_size * dims;
    const total_len = stride * @as(usize, row_count);
    if (column.data.len != total_len) return error.InvalidColumnData;

    const start = @as(usize, row_idx) * stride;
    const end = start + stride;
    return column.data[start..end];
}
pub const ColumnBatch = struct {
    column_id: u32,
    shape: types.ColumnShape,
    phys_type: types.PhysicalType,
    encoding: types.Encoding,
    dims: u16,
    data: []const u8,
};

const SegmentStats = struct {
    min_ts: i64,
    max_ts: i64,
    row_count: u64,
};

pub const default_max_segment_size: u64 = 64 * 1024 * 1024;

/// Controls WAL fsync behavior after writes.
///
/// - `full`: Every WAL write is followed by fsync (~7ms per write on
///   typical SSDs). Survives both application crashes and OS/power failures.
/// - `async_os`: Skips fsync; the OS page cache buffers writes. Survives
///   application crashes (WAL data is in the kernel buffer and will be
///   replayed). Does NOT survive OS crash or power loss — unflushed WAL
///   entries may be lost.
///
/// Default: `full` (matches PostgreSQL/SQLite default behavior).
pub const Durability = enum(u8) {
    full = 0,
    async_os = 1,
};

pub const Writer = struct {
    allocator: Allocator,
    dir: std.fs.Dir,
    ns_path: []const u8,
    lock_file: std.fs.File,
    data_file: std.fs.File,
    wal_writer: wal.WalWriter,
    wal_name: []const u8,
    block_builder: block_writer.BlockBuilder,
    schema_id: ?u16,
    row_count: u32,
    columns: std.ArrayList(ColumnBuffer),
    buffer_bytes: usize,
    flush_threshold: usize,
    max_segment_size: u64,
    durability: Durability,

    /// Opens or creates the TaluDB namespace and replays orphaned WALs.
    ///
    /// Each writer instance creates a unique WAL file (wal-<hex>.wal) to
    /// avoid cross-writer interference. On open, any orphaned WAL files
    /// from prior crashed writers are replayed and cleaned up.
    ///
    /// Note: This does NOT acquire a persistent lock. Locking is done
    /// granularly during individual write operations (appendRow, flushBlock).
    /// Multiple writers (in-process or cross-process) can safely share
    /// the same namespace.
    pub fn open(allocator: Allocator, db_root: []const u8, namespace: []const u8) !Writer {
        const ns_path = try std.fs.path.join(allocator, &.{ db_root, namespace });
        errdefer allocator.free(ns_path);

        var dir_opt: ?std.fs.Dir = null;
        var lock_file_opt: ?std.fs.File = null;
        var data_file_opt: ?std.fs.File = null;

        dir_opt = try std.fs.cwd().makeOpenPath(ns_path, .{ .iterate = true });
        errdefer if (dir_opt) |*d| d.close();

        // Open lock file but do NOT acquire lock here.
        // Locks are acquired granularly during write operations.
        lock_file_opt = try dir_opt.?.createFile("talu.lock", .{ .read = true });
        errdefer if (lock_file_opt) |*f| f.close();

        data_file_opt = try dir_opt.?.createFile("current.talu", .{ .read = true, .truncate = false });
        errdefer if (data_file_opt) |*f| f.close();

        // Create a unique WAL file for this writer instance.
        // Hold an exclusive flock on it for the writer's lifetime so that
        // other writers' orphan scanners can distinguish active WALs from
        // orphaned ones (crashed writers release their flock on exit).
        const wal_name = try generateWalName(allocator);
        errdefer allocator.free(wal_name);

        var wal_file_opt: ?std.fs.File = try dir_opt.?.createFile(wal_name, .{ .read = true, .truncate = false });
        errdefer if (wal_file_opt) |*f| f.close();

        // Persistent exclusive lock on our WAL file.
        try lock.lock(wal_file_opt.?);

        var writer = Writer{
            .allocator = allocator,
            .dir = dir_opt.?,
            .ns_path = ns_path,
            .lock_file = lock_file_opt.?,
            .data_file = data_file_opt.?,
            .wal_writer = wal.WalWriter.init(wal_file_opt.?),
            .wal_name = wal_name,
            .block_builder = block_writer.BlockBuilder.init(allocator, 0, 0),
            .schema_id = null,
            .row_count = 0,
            .columns = .empty,
            .buffer_bytes = 0,
            .flush_threshold = 64 * 1024,
            .max_segment_size = default_max_segment_size,
            .durability = .full,
        };
        errdefer writer.deinit();

        dir_opt = null;
        lock_file_opt = null;
        data_file_opt = null;
        wal_file_opt = null;

        // Replay and clean up orphaned WAL files from crashed writers.
        try lock.lock(writer.lock_file);
        defer lock.unlock(writer.lock_file);
        try writer.replayOrphanWals();

        return writer;
    }

    /// Appends a single row, writing the WAL before updating in-memory buffers.
    ///
    /// Uses granular locking: acquires lock only for the WAL write duration.
    pub fn appendRow(self: *Writer, schema_id: u16, columns: []const ColumnValue) !void {
        if (columns.len == 0) return error.EmptyRow;
        try ensureNoDuplicates(columns);

        const payload = try self.encodeWalPayload(schema_id, columns);
        defer self.allocator.free(payload);

        // Granular lock: acquire, seek to end (other processes may have written), write, sync, release
        try lock.lock(self.lock_file);
        defer lock.unlock(self.lock_file);

        try self.wal_writer.file.seekFromEnd(0);
        try self.wal_writer.append(payload);
        if (self.durability == .full) try self.wal_writer.sync();

        try self.applyRow(schema_id, columns);

        if (self.buffer_bytes >= self.flush_threshold) {
            try self.flushBlockLocked();
        }
    }

    /// Append a batch of rows using columnar arrays (SoA).
    ///
    /// Note: VARBYTES columns are not supported in batch mode.
    /// Uses granular locking: acquires lock only for the WAL write duration.
    pub fn appendBatch(
        self: *Writer,
        schema_id: u16,
        row_count: u32,
        columns: []const ColumnBatch,
    ) !void {
        if (row_count == 0) return;
        if (columns.len == 0) return error.EmptyRow;
        try ensureNoDuplicatesBatch(columns);

        var column_values = try self.allocator.alloc(ColumnValue, columns.len);
        defer self.allocator.free(column_values);

        const batch_payload = try self.encodeWalBatchPayload(schema_id, row_count, columns, column_values);
        defer self.allocator.free(batch_payload);

        // Granular lock: acquire, seek to end, write, sync, release
        try lock.lock(self.lock_file);
        defer lock.unlock(self.lock_file);

        try self.wal_writer.file.seekFromEnd(0);
        try self.wal_writer.appendBatch(batch_payload);
        if (self.durability == .full) try self.wal_writer.sync();

        var row_idx: u32 = 0;
        while (row_idx < row_count) : (row_idx += 1) {
            for (columns, 0..) |column, idx| {
                column_values[idx] = .{
                    .column_id = column.column_id,
                    .shape = column.shape,
                    .phys_type = column.phys_type,
                    .encoding = column.encoding,
                    .dims = column.dims,
                    .data = try columnSlice(column, row_idx, row_count),
                };
            }

            try self.applyRow(schema_id, column_values);
        }

        if (self.buffer_bytes >= self.flush_threshold) {
            try self.flushBlockLocked();
        }
    }

    /// Flushes the current buffered rows into a block and clears the WAL.
    ///
    /// Uses granular locking: acquires lock only for the I/O duration.
    pub fn flushBlock(self: *Writer) !void {
        if (self.row_count == 0) return;

        try lock.lock(self.lock_file);
        defer lock.unlock(self.lock_file);

        try self.flushBlockLocked();
    }

    /// Internal: flush block when lock is already held.
    /// Rotates the segment if writing would exceed max_segment_size.
    fn flushBlockLocked(self: *Writer) !void {
        if (self.row_count == 0) return;
        const schema_id = self.schema_id orelse return error.SchemaMissing;

        self.block_builder.deinit();
        self.block_builder = block_writer.BlockBuilder.init(self.allocator, schema_id, self.row_count);
        self.setBlockTimestampRange();

        for (self.columns.items) |column| {
            const offsets = if (column.shape == .VARBYTES) column.offsets.items else null;
            const lengths = if (column.shape == .VARBYTES) column.lengths.items else null;
            try self.block_builder.addColumn(
                column.column_id,
                column.shape,
                column.phys_type,
                column.encoding,
                column.dims,
                column.data.items,
                offsets,
                lengths,
            );
        }

        const block = try self.block_builder.finalize();
        defer self.allocator.free(block);

        // Rotate if writing this block would exceed the segment size limit.
        const current_size = try self.currentFileSize();
        if (current_size > 0 and current_size + block.len > self.max_segment_size) {
            try self.rotateSegment();
        }

        // Seek to end (another process may have written) and write block
        try self.data_file.seekFromEnd(0);
        try self.data_file.writeAll(block);

        // Clear WAL
        try self.wal_writer.file.setEndPos(0);
        try self.wal_writer.file.seekTo(0);

        self.resetBuffers();
    }

    /// Seals the active segment and creates a fresh current.talu.
    ///
    /// Steps:
    /// 1. Close current data_file so it can be renamed.
    /// 2. Rename current.talu → seg-<uuid>.talu.
    /// 3. Load or create manifest.json, append the new segment entry, save atomically.
    /// 4. Open a new empty current.talu.
    ///
    /// Safety: called inside flushBlockLocked (lock is held). If manifest.save()
    /// fails, the renamed segment still exists on disk — no data is lost.
    fn rotateSegment(self: *Writer) !void {
        // Close current data file so the OS allows rename
        self.data_file.close();

        // Generate a random UUID for the sealed segment filename
        var uuid_bytes: [16]u8 = undefined;
        std.crypto.random.bytes(&uuid_bytes);
        // Set version (4) and variant (RFC 4122) bits
        uuid_bytes[6] = (uuid_bytes[6] & 0x0f) | 0x40;
        uuid_bytes[8] = (uuid_bytes[8] & 0x3f) | 0x80;

        var hex_buf: [32]u8 = undefined;
        const hex_chars = "0123456789abcdef";
        for (uuid_bytes, 0..) |byte, i| {
            hex_buf[i * 2] = hex_chars[byte >> 4];
            hex_buf[i * 2 + 1] = hex_chars[byte & 0x0f];
        }

        const seg_name = try std.fmt.allocPrint(self.allocator, "seg-{s}.talu", .{hex_buf});
        defer self.allocator.free(seg_name);

        // Rename current.talu → seg-<hex>.talu within the namespace dir
        self.dir.rename("current.talu", seg_name) catch |err| {
            // Re-open current.talu so the writer remains usable
            self.data_file = try self.dir.createFile("current.talu", .{ .read = true, .truncate = false });
            return err;
        };
        if (comptime @hasDecl(std.posix, "fsync")) {
            try std.posix.fsync(self.dir.fd);
        }
        errdefer {
            // If seal/manifest update fails after rename, try to roll back the
            // rename so existing data stays in current.talu.
            const rolled_back = blk: {
                self.dir.rename(seg_name, "current.talu") catch break :blk false;
                break :blk true;
            };
            var recovered = false;
            if (rolled_back) {
                if (self.dir.openFile("current.talu", .{ .mode = .read_write }) catch null) |file| {
                    self.data_file = file;
                    recovered = true;
                }
            }

            // Fallback: restore a writable current.talu even if rollback fails.
            if (!recovered) {
                if (self.dir.createFile("current.talu", .{ .read = true, .truncate = false }) catch null) |file| {
                    self.data_file = file;
                }
            }
        }

        const segment_stats = try self.writeFooterAndCollectSegmentStats(seg_name);

        // Update manifest (load existing or create empty)
        try self.updateManifest(seg_name, &uuid_bytes, segment_stats);

        // Open fresh current.talu
        self.data_file = try self.dir.createFile("current.talu", .{ .read = true, .truncate = true });
    }

    fn updateManifest(
        self: *Writer,
        seg_name: []const u8,
        uuid_bytes: *const [16]u8,
        segment_stats: SegmentStats,
    ) !void {
        const manifest_path = try std.fs.path.join(self.allocator, &.{ self.ns_path, "manifest.json" });
        defer self.allocator.free(manifest_path);

        var m = manifest_mod.Manifest.load(self.allocator, manifest_path) catch |err| switch (err) {
            error.FileNotFound => blk: {
                const empty = try self.allocator.alloc(manifest_mod.SegmentEntry, 0);
                break :blk manifest_mod.Manifest{
                    .version = 1,
                    .segments = empty,
                    .last_compaction_ts = 0,
                };
            },
            else => return err,
        };
        errdefer m.deinit(self.allocator);

        // Build u128 from uuid bytes (big-endian)
        const id: u128 = std.mem.readInt(u128, uuid_bytes, .big);

        const old_len = m.segments.len;
        const new_segments = try self.allocator.alloc(manifest_mod.SegmentEntry, old_len + 1);

        const seg_path = self.allocator.dupe(u8, seg_name) catch |err| {
            self.allocator.free(new_segments);
            return err;
        };

        @memcpy(new_segments[0..old_len], m.segments);
        new_segments[old_len] = .{
            .id = id,
            .path = seg_path,
            .min_ts = segment_stats.min_ts,
            .max_ts = segment_stats.max_ts,
            .row_count = segment_stats.row_count,
        };

        // Transfer ownership to manifest. After this, errdefer m.deinit
        // handles cleanup for the entire segments array including seg_path.
        self.allocator.free(m.segments);
        m.segments = new_segments;

        try m.save(manifest_path);
        m.deinit(self.allocator);
    }

    fn writeFooterAndCollectSegmentStats(self: *Writer, seg_name: []const u8) !SegmentStats {
        var file = try self.dir.openFile(seg_name, .{ .mode = .read_write });
        defer file.close();

        const stat = try file.stat();
        const reader = block_reader.BlockReader.init(file, self.allocator);
        const entries = try reader.scanBlocksFromHeaders(stat.size);
        defer self.allocator.free(entries);

        var row_count: u64 = 0;
        var min_ts: i64 = std.math.maxInt(i64);
        var max_ts: i64 = std.math.minInt(i64);
        var has_ts_range = false;

        for (entries) |entry| {
            const header = try reader.readHeader(entry.block_off);
            row_count += header.row_count;
            const flags: types.BlockFlags = @bitCast(header.flags);
            if (flags.has_ts_range) {
                if (!has_ts_range) {
                    min_ts = header.min_ts;
                    max_ts = header.max_ts;
                    has_ts_range = true;
                } else {
                    min_ts = @min(min_ts, header.min_ts);
                    max_ts = @max(max_ts, header.max_ts);
                }
            }
        }

        if (entries.len > 0) {
            const footer_bytes = std.mem.sliceAsBytes(entries);
            if (footer_bytes.len > std.math.maxInt(u32)) return error.SizeOverflow;
            const segment_crc32c = try crc32cFilePrefix(file, stat.size);
            const trailer = types.FooterTrailer{
                .magic = types.MagicValues.FOOTER,
                .version = 1,
                .flags = 0,
                .footer_len = @intCast(footer_bytes.len),
                .footer_crc32c = checksum.crc32c(footer_bytes),
                .segment_crc32c = segment_crc32c,
                .reserved = [_]u8{0} ** 12,
            };

            try file.seekTo(stat.size);
            try file.writeAll(footer_bytes);
            try file.writeAll(std.mem.asBytes(&trailer));
            try file.sync();
        }

        return .{
            .min_ts = if (has_ts_range) min_ts else 0,
            .max_ts = if (has_ts_range) max_ts else 0,
            .row_count = row_count,
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

    fn setBlockTimestampRange(self: *Writer) void {
        self.block_builder.flags.has_ts_range = false;
        self.block_builder.min_ts = 0;
        self.block_builder.max_ts = 0;

        const ts_column = for (self.columns.items) |column| {
            if (column.column_id == 2 and
                column.shape == .SCALAR and
                column.phys_type == .I64 and
                column.encoding == .RAW and
                column.data.items.len == @as(usize, self.row_count) * @sizeOf(i64))
            {
                break column;
            }
        } else return;

        if (self.row_count == 0) return;

        var min_ts: i64 = std.math.maxInt(i64);
        var max_ts: i64 = std.math.minInt(i64);
        var row_idx: usize = 0;
        const row_count: usize = self.row_count;
        while (row_idx < row_count) : (row_idx += 1) {
            const off = row_idx * @sizeOf(i64);
            const ts = std.mem.readInt(i64, ts_column.data.items[off..][0..8], .little);
            min_ts = @min(min_ts, ts);
            max_ts = @max(max_ts, ts);
        }

        self.block_builder.flags.has_ts_range = true;
        self.block_builder.min_ts = min_ts;
        self.block_builder.max_ts = max_ts;
    }

    fn currentFileSize(self: *Writer) !u64 {
        const stat = try self.data_file.stat();
        return stat.size;
    }

    /// Clears buffered schema state so a new schema can be written.
    pub fn resetSchema(self: *Writer) void {
        for (self.columns.items) |*column| {
            column.deinit(self.allocator);
        }
        self.columns.clearRetainingCapacity();
        self.schema_id = null;
        self.row_count = 0;
        self.buffer_bytes = 0;
    }

    /// Releases resources and closes open files.
    /// On clean close, deletes this writer's WAL file (data was flushed).
    pub fn deinit(self: *Writer) void {
        for (self.columns.items) |*column| {
            column.deinit(self.allocator);
        }
        self.columns.deinit(self.allocator);
        self.block_builder.deinit();
        // Delete WAL file BEFORE closing the fd (which releases the flock).
        // This prevents a race where another writer's orphan scanner finds
        // this WAL file in the gap between flock release and deletion.
        self.dir.deleteFile(self.wal_name) catch {};
        self.wal_writer.file.close();
        self.data_file.close();
        self.lock_file.close();
        self.dir.close();
        self.allocator.free(self.wal_name);
        self.allocator.free(self.ns_path);
    }

    /// Simulates a process crash for testing.
    ///
    /// Closes all fds (releasing flocks) WITHOUT flushing or deleting
    /// the WAL file. This accurately simulates what the OS does when
    /// a process dies: all locks are released, but files remain on disk.
    ///
    /// After calling this, the Writer is in an invalid state and must
    /// not be used. The caller is responsible for freeing the Writer
    /// struct itself (e.g., via allocator.destroy).
    pub fn simulateCrash(self: *Writer) void {
        for (self.columns.items) |*column| {
            column.deinit(self.allocator);
        }
        self.columns.deinit(self.allocator);
        self.block_builder.deinit();
        // Close WAL fd (releases flock) but do NOT delete the file.
        // This leaves the orphaned WAL on disk, just like a real crash.
        self.wal_writer.file.close();
        self.data_file.close();
        self.lock_file.close();
        self.dir.close();
        self.allocator.free(self.wal_name);
        self.allocator.free(self.ns_path);
    }

    fn replayWal(self: *Writer) !void {
        var iter = wal.WalIterator.init(self.wal_writer.file, self.allocator);
        while (try iter.next()) |payload| {
            defer self.allocator.free(payload);
            try self.decodeWalPayload(payload);
        }
    }

    /// Replays and deletes orphaned WAL files from crashed writers.
    ///
    /// Scans the namespace directory for `wal-*.wal` files that don't
    /// belong to this writer. Each such file represents a writer that
    /// crashed before flushing. We replay its entries into our buffer,
    /// flush to blocks, then delete the orphaned file.
    ///
    /// Must be called under lock.
    fn replayOrphanWals(self: *Writer) !void {
        var dir_iter = self.dir.iterate();
        // Collect orphan names first (can't modify dir while iterating).
        var orphans = std.ArrayList([]const u8).empty;
        defer {
            for (orphans.items) |name| self.allocator.free(name);
            orphans.deinit(self.allocator);
        }

        while (try dir_iter.next()) |entry| {
            if (entry.kind != .file) continue;
            if (!std.mem.startsWith(u8, entry.name, "wal-")) continue;
            if (!std.mem.endsWith(u8, entry.name, ".wal")) continue;
            // Skip our own WAL file.
            if (std.mem.eql(u8, entry.name, self.wal_name)) continue;
            try orphans.append(self.allocator, try self.allocator.dupe(u8, entry.name));
        }

        for (orphans.items) |orphan_name| {
            try self.replayWalFile(orphan_name);
            self.dir.deleteFile(orphan_name) catch {};
        }

        // Also handle legacy current.wal for backward compatibility.
        if (self.dir.statFile("current.wal")) |stat| {
            if (stat.size > 0) {
                try self.replayWalFile("current.wal");
                // Truncate legacy WAL after replaying.
                if (self.dir.openFile("current.wal", .{ .mode = .read_write })) |f| {
                    f.setEndPos(0) catch {};
                    f.close();
                } else |_| {}
            }
        } else |_| {}
    }

    /// Replays a single WAL file into blocks.
    /// Handles schema transitions: if a WAL entry has a different schema
    /// than the current buffer, flushes and resets before applying.
    /// For wal-*.wal files, acquires an exclusive flock to verify the
    /// writer is dead (orphaned). Skips files locked by active writers.
    fn replayWalFile(self: *Writer, name: []const u8) !void {
        var file = self.dir.openFile(name, .{ .mode = .read_only }) catch return;
        defer file.close();

        // For per-writer WAL files, try to lock to verify the writer is dead.
        // Active writers hold an exclusive flock on their WAL file.
        // If we can't lock it, the writer is still alive — skip.
        if (std.mem.startsWith(u8, name, "wal-")) {
            const locked = lock.tryLock(file) catch return;
            if (!locked) return;
            defer lock.unlock(file);
        }

        var iter = wal.WalIterator.init(file, self.allocator);
        while (try iter.next()) |payload| {
            defer self.allocator.free(payload);
            self.decodeWalPayload(payload) catch |err| {
                if (err == error.SchemaMismatch) {
                    // Schema changed mid-WAL (e.g., items → sessions).
                    // Flush current buffer and reset before retrying.
                    self.flushBlockLocked() catch continue;
                    self.resetSchema();
                    self.decodeWalPayload(payload) catch continue;
                } else {
                    continue;
                }
            };
        }

        // Flush recovered rows to blocks.
        try self.flushBlockLocked();
    }

    fn decodeWalPayload(self: *Writer, payload: []const u8) !void {
        var index: usize = 0;
        const schema_id = try readU16(payload, &index);
        const column_count = try readU16(payload, &index);
        if (column_count == 0) return error.InvalidWalEntry;

        var columns = try self.allocator.alloc(ColumnValue, column_count);
        defer self.allocator.free(columns);

        var i: usize = 0;
        while (i < column_count) : (i += 1) {
            const column_id = try readU32(payload, &index);
            const shape = try std.meta.intToEnum(types.ColumnShape, try readU8(payload, &index));
            const phys_type = try std.meta.intToEnum(types.PhysicalType, try readU8(payload, &index));
            const encoding = try std.meta.intToEnum(types.Encoding, try readU8(payload, &index));
            _ = try readU8(payload, &index);
            const dims = try readU16(payload, &index);
            const data_len = try readU32(payload, &index);
            const data = try readBytes(payload, &index, data_len);

            columns[i] = .{
                .column_id = column_id,
                .shape = shape,
                .phys_type = phys_type,
                .encoding = encoding,
                .dims = dims,
                .data = data,
            };
        }

        if (index != payload.len) return error.InvalidWalEntry;
        try self.applyRow(schema_id, columns);
    }

    fn encodeWalPayload(self: *Writer, schema_id: u16, columns: []const ColumnValue) ![]u8 {
        var payload_len: usize = 0;
        payload_len = try addUsize(payload_len, 4);

        for (columns) |column| {
            payload_len = try addUsize(payload_len, 4 + 1 + 1 + 1 + 1 + 2 + 4);
            payload_len = try addUsize(payload_len, column.data.len);
        }

        if (payload_len > std.math.maxInt(u32)) return error.PayloadTooLarge;
        if (columns.len > std.math.maxInt(u16)) return error.PayloadTooLarge;

        const payload = try self.allocator.alloc(u8, payload_len);
        errdefer self.allocator.free(payload);
        var stream = std.io.fixedBufferStream(payload);
        const writer = stream.writer();

        try writer.writeInt(u16, schema_id, .little);
        try writer.writeInt(u16, @intCast(columns.len), .little);

        for (columns) |column| {
            try writer.writeInt(u32, column.column_id, .little);
            try writer.writeByte(@intFromEnum(column.shape));
            try writer.writeByte(@intFromEnum(column.phys_type));
            try writer.writeByte(@intFromEnum(column.encoding));
            try writer.writeByte(0);
            try writer.writeInt(u16, column.dims, .little);
            if (column.data.len > std.math.maxInt(u32)) return error.PayloadTooLarge;
            try writer.writeInt(u32, @intCast(column.data.len), .little);
            try writer.writeAll(column.data);
        }

        return payload;
    }

    fn encodeWalBatchPayload(
        self: *Writer,
        schema_id: u16,
        row_count: u32,
        columns: []const ColumnBatch,
        column_values: []ColumnValue,
    ) ![]u8 {
        var batch = std.ArrayList(u8).empty;
        errdefer batch.deinit(self.allocator);

        var count_buf: [4]u8 = undefined;
        std.mem.writeInt(u32, count_buf[0..4], row_count, .little);
        try batch.appendSlice(self.allocator, &count_buf);

        var row_idx: u32 = 0;
        while (row_idx < row_count) : (row_idx += 1) {
            for (columns, 0..) |column, idx| {
                column_values[idx] = .{
                    .column_id = column.column_id,
                    .shape = column.shape,
                    .phys_type = column.phys_type,
                    .encoding = column.encoding,
                    .dims = column.dims,
                    .data = try columnSlice(column, row_idx, row_count),
                };
            }

            const payload = try self.encodeWalPayload(schema_id, column_values);
            defer self.allocator.free(payload);

            if (payload.len > std.math.maxInt(u32)) return error.PayloadTooLarge;
            var len_buf: [4]u8 = undefined;
            std.mem.writeInt(u32, len_buf[0..4], @intCast(payload.len), .little);
            try batch.appendSlice(self.allocator, &len_buf);
            try batch.appendSlice(self.allocator, payload);
        }

        return batch.toOwnedSlice(self.allocator);
    }

    fn applyRow(self: *Writer, schema_id: u16, columns: []const ColumnValue) !void {
        if (self.schema_id) |current_schema| {
            if (current_schema != schema_id) return error.SchemaMismatch;
        } else {
            self.schema_id = schema_id;
        }

        if (self.columns.items.len == 0) {
            try self.initializeColumns(columns);
        } else if (self.columns.items.len != columns.len) {
            return error.ColumnCountMismatch;
        }

        for (columns) |column| {
            const idx = self.findColumnIndex(column.column_id) orelse return error.UnknownColumn;
            var buffer = &self.columns.items[idx];

            if (buffer.shape != column.shape or
                buffer.phys_type != column.phys_type or
                buffer.encoding != column.encoding or
                buffer.dims != column.dims)
            {
                return error.ColumnMismatch;
            }

            try buffer.appendValue(self.allocator, column, self.row_count);
            self.buffer_bytes += buffer.last_append_bytes;
        }

        if (self.row_count == std.math.maxInt(u32)) return error.RowCountOverflow;
        self.row_count += 1;
    }

    fn initializeColumns(self: *Writer, columns: []const ColumnValue) !void {
        try self.columns.ensureTotalCapacity(self.allocator, columns.len);
        for (columns) |column| {
            var buffer = ColumnBuffer.init(column);
            errdefer buffer.deinit(self.allocator);
            try self.columns.append(self.allocator, buffer);
        }
    }

    fn findColumnIndex(self: *Writer, column_id: u32) ?usize {
        for (self.columns.items, 0..) |column, idx| {
            if (column.column_id == column_id) return idx;
        }
        return null;
    }

    fn resetBuffers(self: *Writer) void {
        for (self.columns.items) |*column| {
            column.clear();
        }
        self.schema_id = null;
        self.row_count = 0;
        self.buffer_bytes = 0;
    }
};

/// Generates a unique WAL filename: `wal-<32 hex chars>.wal`.
fn generateWalName(allocator: Allocator) ![]const u8 {
    var uuid_bytes: [16]u8 = undefined;
    std.crypto.random.bytes(&uuid_bytes);

    var hex_buf: [32]u8 = undefined;
    const hex_chars = "0123456789abcdef";
    for (uuid_bytes, 0..) |byte, i| {
        hex_buf[i * 2] = hex_chars[byte >> 4];
        hex_buf[i * 2 + 1] = hex_chars[byte & 0x0f];
    }

    return std.fmt.allocPrint(allocator, "wal-{s}.wal", .{hex_buf});
}

const ColumnBuffer = struct {
    column_id: u32,
    shape: types.ColumnShape,
    phys_type: types.PhysicalType,
    encoding: types.Encoding,
    dims: u16,
    data: std.ArrayList(u8),
    offsets: std.ArrayList(u32),
    lengths: std.ArrayList(u32),
    last_append_bytes: usize,

    fn init(column: ColumnValue) ColumnBuffer {
        return .{
            .column_id = column.column_id,
            .shape = column.shape,
            .phys_type = column.phys_type,
            .encoding = column.encoding,
            .dims = column.dims,
            .data = .empty,
            .offsets = .empty,
            .lengths = .empty,
            .last_append_bytes = 0,
        };
    }

    fn deinit(self: *ColumnBuffer, allocator: Allocator) void {
        self.data.deinit(allocator);
        self.offsets.deinit(allocator);
        self.lengths.deinit(allocator);
    }

    fn clear(self: *ColumnBuffer) void {
        self.data.clearRetainingCapacity();
        self.offsets.clearRetainingCapacity();
        self.lengths.clearRetainingCapacity();
    }

    fn appendValue(self: *ColumnBuffer, allocator: Allocator, column: ColumnValue, row_idx: u32) !void {
        _ = row_idx;
        self.last_append_bytes = 0;
        if (self.shape == .VARBYTES) {
            const offset = try checkedU32(self.data.items.len);
            const length = try checkedU32(column.data.len);
            try self.offsets.append(allocator, offset);
            try self.lengths.append(allocator, length);
            try self.data.appendSlice(allocator, column.data);
            self.last_append_bytes = column.data.len + 2 * @sizeOf(u32);
            return;
        }

        const expected = try expectedByteSize(self.shape, self.phys_type, self.dims);
        if (column.data.len != expected) return error.InvalidColumnData;
        try self.data.appendSlice(allocator, column.data);
        self.last_append_bytes = column.data.len;
    }
};

fn expectedByteSize(shape: types.ColumnShape, phys_type: types.PhysicalType, dims: u16) !usize {
    if (shape == .VARBYTES) return 0;

    const element_size: usize = switch (phys_type) {
        .U8, .I8 => 1,
        .U16, .I16, .F16, .BF16 => 2,
        .U32, .I32, .F32 => 4,
        .U64, .I64, .F64 => 8,
        .BINARY => return error.InvalidColumnData,
    };

    return switch (shape) {
        .SCALAR => element_size,
        .VECTOR => blk: {
            if (dims == 0) return error.InvalidColumnData;
            break :blk element_size * @as(usize, dims);
        },
        .VARBYTES => 0,
    };
}

fn ensureNoDuplicates(columns: []const ColumnValue) !void {
    for (columns, 0..) |column, idx| {
        var j: usize = 0;
        while (j < idx) : (j += 1) {
            if (columns[j].column_id == column.column_id) return error.DuplicateColumn;
        }
    }
}

fn ensureNoDuplicatesBatch(columns: []const ColumnBatch) !void {
    for (columns, 0..) |column, idx| {
        var j: usize = 0;
        while (j < idx) : (j += 1) {
            if (columns[j].column_id == column.column_id) return error.DuplicateColumn;
        }
    }
}

fn addUsize(base: usize, value: usize) !usize {
    const sum = base + value;
    if (sum < base) return error.SizeOverflow;
    return sum;
}

fn checkedU32(value: usize) !u32 {
    if (value > std.math.maxInt(u32)) return error.SizeOverflow;
    return @intCast(value);
}

fn readU8(payload: []const u8, index: *usize) !u8 {
    if (index.* + 1 > payload.len) return error.InvalidWalEntry;
    const value = payload[index.*];
    index.* += 1;
    return value;
}

fn readU16(payload: []const u8, index: *usize) !u16 {
    if (index.* + 2 > payload.len) return error.InvalidWalEntry;
    const value = std.mem.readInt(u16, payload[index.*..][0..2], .little);
    index.* += 2;
    return value;
}

fn readU32(payload: []const u8, index: *usize) !u32 {
    if (index.* + 4 > payload.len) return error.InvalidWalEntry;
    const value = std.mem.readInt(u32, payload[index.*..][0..4], .little);
    index.* += 4;
    return value;
}

fn readBytes(payload: []const u8, index: *usize, len: u32) ![]const u8 {
    const byte_len: usize = len;
    if (index.* + byte_len > payload.len) return error.InvalidWalEntry;
    const slice = payload[index.* .. index.* + byte_len];
    index.* += byte_len;
    return slice;
}

test "Writer.open creates files" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const tmp_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(tmp_path);

    var writer = try Writer.open(std.testing.allocator, tmp_path, "chat");
    defer writer.deinit();

    var lock_file = try tmp.dir.openFile("chat/talu.lock", .{});
    defer lock_file.close();
    var data_file = try tmp.dir.openFile("chat/current.talu", .{});
    defer data_file.close();
    // Writer creates a unique WAL file (wal-<hex>.wal).
    const wal_path = try std.fmt.allocPrint(std.testing.allocator, "chat/{s}", .{writer.wal_name});
    defer std.testing.allocator.free(wal_path);
    var wal_file = try tmp.dir.openFile(wal_path, .{});
    defer wal_file.close();
}

test "Writer.appendRow buffers data" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const tmp_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(tmp_path);

    var writer = try Writer.open(std.testing.allocator, tmp_path, "chat");
    defer writer.deinit();

    var value: u64 = 42;
    const col1 = ColumnValue{
        .column_id = 1,
        .shape = .SCALAR,
        .phys_type = .U64,
        .encoding = .RAW,
        .dims = 1,
        .data = std.mem.asBytes(&value),
    };
    const col2 = ColumnValue{
        .column_id = 2,
        .shape = .VARBYTES,
        .phys_type = .BINARY,
        .encoding = .RAW,
        .dims = 0,
        .data = "payload",
    };

    try writer.appendRow(3, &.{ col1, col2 });

    try std.testing.expectEqual(@as(u32, 1), writer.row_count);
    try std.testing.expect(writer.buffer_bytes > 0);

    const wal_path = try std.fmt.allocPrint(std.testing.allocator, "chat/{s}", .{writer.wal_name});
    defer std.testing.allocator.free(wal_path);
    const wal_data = try tmp.dir.readFileAlloc(std.testing.allocator, wal_path, 1024);
    defer std.testing.allocator.free(wal_data);
    try std.testing.expect(wal_data.len > 0);
}

test "Writer.appendBatch buffers data" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var writer = try Writer.open(std.testing.allocator, root_path, "vector");
    defer writer.deinit();

    const ids = [_]u64{ 1, 2 };
    const vectors = [_]f32{ 0.5, 1.0, 1.5, 2.0 };
    const ts = [_]i64{ 10, 20 };

    const columns = [_]ColumnBatch{
        .{
            .column_id = 1,
            .shape = .SCALAR,
            .phys_type = .U64,
            .encoding = .RAW,
            .dims = 1,
            .data = std.mem.sliceAsBytes(&ids),
        },
        .{
            .column_id = 2,
            .shape = .SCALAR,
            .phys_type = .I64,
            .encoding = .RAW,
            .dims = 1,
            .data = std.mem.sliceAsBytes(&ts),
        },
        .{
            .column_id = 10,
            .shape = .VECTOR,
            .phys_type = .F32,
            .encoding = .RAW,
            .dims = 2,
            .data = std.mem.sliceAsBytes(&vectors),
        },
    };

    try writer.appendBatch(10, 2, &columns);
    try std.testing.expectEqual(@as(u32, 2), writer.row_count);

    const wal_path = try std.fmt.allocPrint(std.testing.allocator, "vector/{s}", .{writer.wal_name});
    defer std.testing.allocator.free(wal_path);
    const wal_data = try tmp.dir.readFileAlloc(std.testing.allocator, wal_path, 4096);
    defer std.testing.allocator.free(wal_data);

    try std.testing.expect(wal_data.len >= 12);
    const magic = std.mem.readInt(u32, wal_data[0..4], .little);
    const payload_len = std.mem.readInt(u32, wal_data[4..8], .little);
    try std.testing.expectEqual(types.MagicValues.WAL_BATCH, magic);
    try std.testing.expectEqual(@as(usize, payload_len + 12), wal_data.len);
}

test "Writer.flushBlock writes block and clears wal" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const tmp_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(tmp_path);

    var writer = try Writer.open(std.testing.allocator, tmp_path, "chat");
    defer writer.deinit();

    var value: u64 = 7;
    const col1 = ColumnValue{
        .column_id = 1,
        .shape = .SCALAR,
        .phys_type = .U64,
        .encoding = .RAW,
        .dims = 1,
        .data = std.mem.asBytes(&value),
    };

    try writer.appendRow(3, &.{col1});
    try writer.flushBlock();

    const data = try tmp.dir.readFileAlloc(std.testing.allocator, "chat/current.talu", 1024);
    defer std.testing.allocator.free(data);
    try std.testing.expect(data.len > 0);

    const wal_path = try std.fmt.allocPrint(std.testing.allocator, "chat/{s}", .{writer.wal_name});
    defer std.testing.allocator.free(wal_path);
    const wal_data = try tmp.dir.readFileAlloc(std.testing.allocator, wal_path, 1024);
    defer std.testing.allocator.free(wal_data);
    try std.testing.expectEqual(@as(usize, 0), wal_data.len);
}

test "Writer.resetSchema clears columns" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const tmp_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(tmp_path);

    var writer = try Writer.open(std.testing.allocator, tmp_path, "chat");
    defer writer.deinit();

    var value: u64 = 7;
    const col1 = ColumnValue{
        .column_id = 1,
        .shape = .SCALAR,
        .phys_type = .U64,
        .encoding = .RAW,
        .dims = 1,
        .data = std.mem.asBytes(&value),
    };

    try writer.appendRow(3, &.{col1});
    writer.resetSchema();

    try std.testing.expect(writer.schema_id == null);
    try std.testing.expectEqual(@as(usize, 0), writer.columns.items.len);
    try std.testing.expectEqual(@as(u32, 0), writer.row_count);
}

test "Writer.deinit closes" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const tmp_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(tmp_path);

    var writer = try Writer.open(std.testing.allocator, tmp_path, "chat");
    writer.deinit();
}

test "Writer.rotateSegment seals current segment" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const tmp_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(tmp_path);

    var writer = try Writer.open(std.testing.allocator, tmp_path, "rot");
    defer writer.deinit();

    writer.max_segment_size = 1024;

    const col = ColumnValue{
        .column_id = 1,
        .shape = .VARBYTES,
        .phys_type = .BINARY,
        .encoding = .RAW,
        .dims = 0,
        .data = "a" ** 512,
    };

    // First flush populates current.talu
    try writer.appendRow(1, &.{col});
    try writer.flushBlock();

    // Second flush triggers rotation
    try writer.appendRow(1, &.{col});
    try writer.flushBlock();

    // A seg-*.talu file should exist in the namespace directory
    var seg_found = false;
    var seg_name_buf: [64]u8 = undefined;
    var seg_name_len: usize = 0;
    var dir = try tmp.dir.openDir("rot", .{ .iterate = true });
    defer dir.close();
    var dir_iter = dir.iterate();
    while (try dir_iter.next()) |entry| {
        if (std.mem.startsWith(u8, entry.name, "seg-") and std.mem.endsWith(u8, entry.name, ".talu")) {
            seg_found = true;
            if (entry.name.len <= seg_name_buf.len) {
                @memcpy(seg_name_buf[0..entry.name.len], entry.name);
                seg_name_len = entry.name.len;
            }
            break;
        }
    }
    try std.testing.expect(seg_found);

    const seg_name = seg_name_buf[0..seg_name_len];
    try std.testing.expect(seg_name_len > 0);

    // Sealed segment should carry a footer index.
    const seg_rel_path = try std.fmt.allocPrint(std.testing.allocator, "rot/{s}", .{seg_name});
    defer std.testing.allocator.free(seg_rel_path);
    var seg_file = try tmp.dir.openFile(seg_rel_path, .{ .mode = .read_only });
    defer seg_file.close();
    const seg_reader = block_reader.BlockReader.init(seg_file, std.testing.allocator);
    const seg_stat = try seg_file.stat();
    const footer = try seg_reader.readFooter(seg_stat.size);
    try std.testing.expect(footer != null);
    defer std.testing.allocator.free(footer.?);
    try std.testing.expect(footer.?.len > 0);

    var trailer_bytes: [@sizeOf(types.FooterTrailer)]u8 = undefined;
    const trailer_off = seg_stat.size - @sizeOf(types.FooterTrailer);
    const trailer_len = try seg_file.preadAll(&trailer_bytes, trailer_off);
    try std.testing.expectEqual(@as(usize, trailer_bytes.len), trailer_len);
    const trailer = std.mem.bytesToValue(types.FooterTrailer, trailer_bytes[0..]);
    try std.testing.expectEqual(types.MagicValues.FOOTER, trailer.magic);
    try std.testing.expect(trailer.segment_crc32c != 0);

    // Manifest entry should include non-zero segment row_count metadata.
    const manifest_path = try tmp.dir.realpathAlloc(std.testing.allocator, "rot/manifest.json");
    defer std.testing.allocator.free(manifest_path);
    var manifest_data = try manifest_mod.Manifest.load(std.testing.allocator, manifest_path);
    defer manifest_data.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 1), manifest_data.segments.len);
    try std.testing.expect(manifest_data.segments[0].row_count > 0);
}

test "Writer.rotateSegment records timestamp range in manifest metadata" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const tmp_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(tmp_path);

    var writer = try Writer.open(std.testing.allocator, tmp_path, "rot-ts");
    defer writer.deinit();

    writer.max_segment_size = 1;

    const ids_first = [_]u64{1};
    const ts_first = [_]i64{100};
    const cols_first = [_]ColumnBatch{
        .{
            .column_id = 1,
            .shape = .SCALAR,
            .phys_type = .U64,
            .encoding = .RAW,
            .dims = 1,
            .data = std.mem.sliceAsBytes(&ids_first),
        },
        .{
            .column_id = 2,
            .shape = .SCALAR,
            .phys_type = .I64,
            .encoding = .RAW,
            .dims = 1,
            .data = std.mem.sliceAsBytes(&ts_first),
        },
    };
    try writer.appendBatch(10, 1, &cols_first);
    try writer.flushBlock();

    const ids_second = [_]u64{2};
    const ts_second = [_]i64{200};
    const cols_second = [_]ColumnBatch{
        .{
            .column_id = 1,
            .shape = .SCALAR,
            .phys_type = .U64,
            .encoding = .RAW,
            .dims = 1,
            .data = std.mem.sliceAsBytes(&ids_second),
        },
        .{
            .column_id = 2,
            .shape = .SCALAR,
            .phys_type = .I64,
            .encoding = .RAW,
            .dims = 1,
            .data = std.mem.sliceAsBytes(&ts_second),
        },
    };
    try writer.appendBatch(10, 1, &cols_second);
    try writer.flushBlock();

    const manifest_path = try tmp.dir.realpathAlloc(std.testing.allocator, "rot-ts/manifest.json");
    defer std.testing.allocator.free(manifest_path);
    var manifest_data = try manifest_mod.Manifest.load(std.testing.allocator, manifest_path);
    defer manifest_data.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 1), manifest_data.segments.len);
    try std.testing.expectEqual(@as(i64, 100), manifest_data.segments[0].min_ts);
    try std.testing.expectEqual(@as(i64, 100), manifest_data.segments[0].max_ts);
    try std.testing.expectEqual(@as(u64, 1), manifest_data.segments[0].row_count);
}

test "Writer.rotateSegment failure keeps writer usable" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const tmp_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(tmp_path);

    var writer = try Writer.open(std.testing.allocator, tmp_path, "rot-fail");
    defer writer.deinit();

    writer.max_segment_size = 1;

    const first = ColumnValue{
        .column_id = 1,
        .shape = .VARBYTES,
        .phys_type = .BINARY,
        .encoding = .RAW,
        .dims = 0,
        .data = "a" ** 128,
    };

    // Initial block goes into current.talu.
    try writer.appendRow(1, &.{first});
    try writer.flushBlock();

    // Corrupt manifest to force rotateSegment -> updateManifest failure.
    var manifest_file = try tmp.dir.createFile("rot-fail/manifest.json", .{ .truncate = true });
    defer manifest_file.close();
    try manifest_file.writeAll("{not-json");

    // Next flush attempts rotation and should fail while keeping writer usable.
    const second = ColumnValue{
        .column_id = 1,
        .shape = .VARBYTES,
        .phys_type = .BINARY,
        .encoding = .RAW,
        .dims = 0,
        .data = "b" ** 128,
    };
    try writer.appendRow(1, &.{second});
    try std.testing.expectError(error.InvalidManifest, writer.flushBlock());

    // Failed rotation should roll back sealed rename and keep data in current.
    var seg_found = false;
    var dir = try tmp.dir.openDir("rot-fail", .{ .iterate = true });
    defer dir.close();
    var it = dir.iterate();
    while (try it.next()) |entry| {
        if (entry.kind == .file and std.mem.startsWith(u8, entry.name, "seg-") and std.mem.endsWith(u8, entry.name, ".talu")) {
            seg_found = true;
            break;
        }
    }
    try std.testing.expect(!seg_found);

    const current_before_retry = try tmp.dir.readFileAlloc(std.testing.allocator, "rot-fail/current.talu", 8192);
    defer std.testing.allocator.free(current_before_retry);
    try std.testing.expect(current_before_retry.len > 0);

    // Writer should still be able to flush buffered rows after the failure.
    writer.max_segment_size = 1024 * 1024;
    try writer.flushBlock();

    const current_data = try tmp.dir.readFileAlloc(std.testing.allocator, "rot-fail/current.talu", 8192);
    defer std.testing.allocator.free(current_data);
    try std.testing.expect(current_data.len > 0);
}

test "Writer.appendRow async_os durability skips fsync" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const tmp_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(tmp_path);

    var writer = try Writer.open(std.testing.allocator, tmp_path, "async");
    defer writer.deinit();

    writer.durability = .async_os;

    const col = ColumnValue{
        .column_id = 1,
        .shape = .SCALAR,
        .phys_type = .U64,
        .encoding = .RAW,
        .dims = 1,
        .data = std.mem.asBytes(&@as(u64, 42)),
    };

    // Write rows — should succeed without fsync
    try writer.appendRow(1, &.{col});
    try writer.appendRow(1, &.{col});

    // Data is in-memory buffer; flush to block to verify
    try writer.flushBlock();

    // Verify data survived the write path
    try std.testing.expectEqual(@as(u32, 0), writer.row_count);
}

test "Writer.appendBatch async_os durability skips fsync" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const tmp_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(tmp_path);

    var writer = try Writer.open(std.testing.allocator, tmp_path, "async_batch");
    defer writer.deinit();

    writer.durability = .async_os;

    const ids = [_]u64{ 1, 2, 3 };
    const columns = [_]ColumnBatch{.{
        .column_id = 1,
        .shape = .SCALAR,
        .phys_type = .U64,
        .encoding = .RAW,
        .dims = 1,
        .data = std.mem.sliceAsBytes(&ids),
    }};

    try writer.appendBatch(1, 3, &columns);

    // 3 rows should be buffered
    try std.testing.expectEqual(@as(u32, 3), writer.row_count);

    try writer.flushBlock();
    try std.testing.expectEqual(@as(u32, 0), writer.row_count);
}
