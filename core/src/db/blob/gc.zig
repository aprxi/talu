//! Offline blob mark-and-sweep tooling for TaluDB.
//!
//! This module scans document/session payload metadata for external blob refs
//! and removes unreferenced CAS files under `blobs/<shard>/<digest>`.
//!
//! Safety notes:
//! - This is an explicit/offline utility, not part of the hot write path.
//! - Mark phase includes both block files and WAL frames to preserve refs that
//!   are pending flush after a crash or while a writer is active.

const std = @import("std");
const kvbuf = @import("../../io/kvbuf/root.zig");
const db_reader = @import("../reader.zig");
const db_block_reader = @import("../block_reader.zig");
const db_blob_store = @import("store.zig");
const db_wal = @import("../wal.zig");
const db_writer = @import("../writer.zig");

const Allocator = std.mem.Allocator;

const docs_namespace = "docs";
const chat_namespace = "chat";
const schema_documents: u16 = 11;
const schema_items: u16 = 3;
const col_payload: u32 = 20;

const DigestKey = [db_blob_store.digest_hex_len]u8;
const DigestSet = std.AutoHashMap(DigestKey, void);
const RefKey = [db_blob_store.ref_len]u8;

const MultipartManifest = struct {
    v: u8,
    type: []const u8,
    total_size: u64,
    part_size: u64,
    parts: []const []const u8,
};

pub const SweepStats = struct {
    referenced_blob_count: usize = 0,
    total_blob_files: usize = 0,
    deleted_blob_files: usize = 0,
    reclaimed_bytes: u64 = 0,
    invalid_reference_count: usize = 0,
    skipped_invalid_entries: usize = 0,
    skipped_recent_blob_files: usize = 0,
};

pub const SweepOptions = struct {
    /// Minimum age (in seconds) a blob file must have before it can be deleted.
    /// Defaults to 15 minutes to avoid deleting blobs created shortly before
    /// their metadata reference is durably persisted.
    min_blob_age_seconds: u64 = 15 * 60,
    /// Optional deterministic clock override for tests.
    now_unix_seconds: ?i64 = null,
};

/// Mark referenced blobs and delete unreferenced CAS files in `blobs/*/*`.
/// Returns sweep statistics, including referenced/deleted counts.
pub fn sweepUnreferencedBlobs(allocator: Allocator, db_root: []const u8) !SweepStats {
    return sweepUnreferencedBlobsWithOptions(allocator, db_root, .{});
}

/// Mark referenced blobs and delete unreferenced CAS files in `blobs/*/*`.
/// Supports configurable grace-period behavior through SweepOptions.
pub fn sweepUnreferencedBlobsWithOptions(
    allocator: Allocator,
    db_root: []const u8,
    options: SweepOptions,
) !SweepStats {
    var stats = SweepStats{};
    const now_unix_seconds = options.now_unix_seconds orelse std.time.timestamp();

    var digests = try markReferencedBlobDigests(allocator, db_root, &stats.invalid_reference_count);
    defer digests.deinit();
    stats.referenced_blob_count = digests.count();

    var root_dir = std.fs.cwd().openDir(db_root, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => return stats,
        else => return err,
    };
    defer root_dir.close();

    var blobs_dir = root_dir.openDir("blobs", .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => return stats,
        else => return err,
    };
    defer blobs_dir.close();

    var shard_iter = blobs_dir.iterate();
    while (try shard_iter.next()) |shard_entry| {
        if (shard_entry.kind != .directory) {
            stats.skipped_invalid_entries += 1;
            continue;
        }

        var shard_dir = blobs_dir.openDir(shard_entry.name, .{ .iterate = true }) catch {
            stats.skipped_invalid_entries += 1;
            continue;
        };
        defer shard_dir.close();

        var to_delete = std.ArrayList(struct {
            name: [db_blob_store.digest_hex_len]u8,
            size: u64,
        }).empty;
        defer to_delete.deinit(allocator);

        var file_iter = shard_dir.iterate();
        while (try file_iter.next()) |file_entry| {
            if (file_entry.kind != .file) {
                stats.skipped_invalid_entries += 1;
                continue;
            }

            const digest_key = buildDigestFromName(shard_entry.name, file_entry.name) orelse {
                stats.skipped_invalid_entries += 1;
                continue;
            };
            stats.total_blob_files += 1;

            if (digests.contains(digest_key)) continue;

            const file_stat = shard_dir.statFile(file_entry.name) catch {
                stats.skipped_invalid_entries += 1;
                continue;
            };
            if (!isOldEnoughForSweep(file_stat, now_unix_seconds, options.min_blob_age_seconds)) {
                stats.skipped_recent_blob_files += 1;
                continue;
            }

            var name_buf: [db_blob_store.digest_hex_len]u8 = undefined;
            @memcpy(name_buf[0..], file_entry.name[0..db_blob_store.digest_hex_len]);
            try to_delete.append(allocator, .{
                .name = name_buf,
                .size = file_stat.size,
            });
        }

        for (to_delete.items) |item| {
            shard_dir.deleteFile(item.name[0..]) catch {
                continue;
            };
            stats.deleted_blob_files += 1;
            stats.reclaimed_bytes += item.size;
        }
    }

    return stats;
}

/// Collect all unique blob references discovered from docs/chat blocks and WALs.
/// Caller owns the returned slice.
pub fn collectReferencedBlobRefs(allocator: Allocator, db_root: []const u8) ![]RefKey {
    var invalid_ref_count: usize = 0;
    var digests = try markReferencedBlobDigests(allocator, db_root, &invalid_ref_count);
    defer digests.deinit();

    const out = try allocator.alloc(RefKey, digests.count());
    var i: usize = 0;
    var it = digests.iterator();
    while (it.next()) |entry| : (i += 1) {
        out[i] = db_blob_store.buildSha256RefString(entry.key_ptr.*);
    }

    std.mem.sort(RefKey, out, {}, struct {
        fn lessThan(_: void, lhs: RefKey, rhs: RefKey) bool {
            return std.mem.lessThan(u8, lhs[0..], rhs[0..]);
        }
    }.lessThan);

    return out;
}

fn isOldEnoughForSweep(stat: std.fs.File.Stat, now_unix_seconds: i64, min_blob_age_seconds: u64) bool {
    if (min_blob_age_seconds == 0) return true;
    if (stat.mtime <= 0) return false;

    const mtime_seconds: i64 = @intCast(@divTrunc(stat.mtime, std.time.ns_per_s));
    if (now_unix_seconds <= mtime_seconds) return false;

    const age_seconds: u64 = @intCast(now_unix_seconds - mtime_seconds);
    return age_seconds >= min_blob_age_seconds;
}

fn markReferencedBlobDigests(allocator: Allocator, db_root: []const u8, invalid_ref_count: *usize) !DigestSet {
    var digests = DigestSet.init(allocator);
    errdefer digests.deinit();

    var store = try db_blob_store.BlobStore.init(allocator, db_root);
    defer store.deinit();

    try markNamespaceBlockRefs(
        allocator,
        db_root,
        docs_namespace,
        schema_documents,
        kvbuf.DocumentFieldIds.doc_json_ref,
        &store,
        &digests,
        invalid_ref_count,
    );
    try markNamespaceBlockRefs(
        allocator,
        db_root,
        chat_namespace,
        schema_items,
        kvbuf.FieldIds.record_json_ref,
        &store,
        &digests,
        invalid_ref_count,
    );

    try markNamespaceWalRefs(
        allocator,
        db_root,
        docs_namespace,
        schema_documents,
        kvbuf.DocumentFieldIds.doc_json_ref,
        &store,
        &digests,
        invalid_ref_count,
    );
    try markNamespaceWalRefs(
        allocator,
        db_root,
        chat_namespace,
        schema_items,
        kvbuf.FieldIds.record_json_ref,
        &store,
        &digests,
        invalid_ref_count,
    );

    return digests;
}

fn markNamespaceBlockRefs(
    allocator: Allocator,
    db_root: []const u8,
    namespace: []const u8,
    schema_id: u16,
    ref_field_id: u16,
    store: *db_blob_store.BlobStore,
    digests: *DigestSet,
    invalid_ref_count: *usize,
) !void {
    var reader = try db_reader.Reader.open(allocator, db_root, namespace);
    defer reader.deinit();

    const blocks = try reader.getBlocks(allocator);
    defer allocator.free(blocks);

    for (blocks) |block| {
        var file = reader.dir.openFile(block.path, .{ .mode = .read_only }) catch continue;
        defer file.close();

        const block_reader = db_block_reader.BlockReader.init(file, allocator);
        const header = block_reader.readHeader(block.offset) catch continue;
        if (header.schema_id != schema_id) continue;
        if (header.row_count == 0) continue;

        const descs = block_reader.readColumnDirectory(header, block.offset) catch continue;
        defer allocator.free(descs);

        const payload_desc = findColumn(descs, col_payload) orelse continue;
        var payload_buffers = readVarBytesBuffers(file, block.offset, payload_desc, header.row_count, allocator) catch continue;
        defer payload_buffers.deinit(allocator);

        for (0..header.row_count) |row_idx| {
            const payload = payload_buffers.sliceForRow(row_idx) catch continue;
            try markRefInPayload(payload, ref_field_id, store, digests, invalid_ref_count);
        }
    }
}

fn markNamespaceWalRefs(
    allocator: Allocator,
    db_root: []const u8,
    namespace: []const u8,
    schema_id: u16,
    ref_field_id: u16,
    store: *db_blob_store.BlobStore,
    digests: *DigestSet,
    invalid_ref_count: *usize,
) !void {
    var root_dir = std.fs.cwd().openDir(db_root, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => return,
        else => return err,
    };
    defer root_dir.close();

    var namespace_dir = root_dir.openDir(namespace, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => return,
        else => return err,
    };
    defer namespace_dir.close();

    var iter = namespace_dir.iterate();
    while (try iter.next()) |entry| {
        if (entry.kind != .file) continue;
        if (!isWalName(entry.name)) continue;

        var wal_file = namespace_dir.openFile(entry.name, .{ .mode = .read_only }) catch continue;
        defer wal_file.close();

        var wal_iter = db_wal.WalIterator.init(wal_file, allocator);
        while (true) {
            const payload_opt = wal_iter.next() catch |err| switch (err) {
                error.InvalidMagic, error.InvalidCrc, error.InvalidWalEntry => break,
                else => return err,
            };
            const payload = payload_opt orelse break;
            defer allocator.free(payload);

            try markRefInWalPayload(payload, schema_id, ref_field_id, store, digests, invalid_ref_count);
        }
    }
}

fn markRefInWalPayload(
    payload: []const u8,
    schema_id: u16,
    ref_field_id: u16,
    store: *db_blob_store.BlobStore,
    digests: *DigestSet,
    invalid_ref_count: *usize,
) !void {
    var index: usize = 0;
    const row_schema = readU16(payload, &index) catch return;
    const column_count = readU16(payload, &index) catch return;
    if (column_count == 0) return;

    var payload_col_data: ?[]const u8 = null;
    var i: usize = 0;
    while (i < column_count) : (i += 1) {
        const column_id = readU32(payload, &index) catch return;
        _ = readU8(payload, &index) catch return;
        _ = readU8(payload, &index) catch return;
        _ = readU8(payload, &index) catch return;
        _ = readU8(payload, &index) catch return;
        _ = readU16(payload, &index) catch return;
        const data_len = readU32(payload, &index) catch return;
        const column_data = readBytes(payload, &index, data_len) catch return;

        if (row_schema == schema_id and column_id == col_payload) {
            payload_col_data = column_data;
        }
    }

    if (payload_col_data) |row_payload| {
        try markRefInPayload(row_payload, ref_field_id, store, digests, invalid_ref_count);
    }
}

fn markRefInPayload(
    payload: []const u8,
    ref_field_id: u16,
    store: *db_blob_store.BlobStore,
    digests: *DigestSet,
    invalid_ref_count: *usize,
) !void {
    if (!kvbuf.isKvBuf(payload)) return;
    const reader = kvbuf.KvBufReader.init(payload) catch return;
    const blob_ref = reader.get(ref_field_id) orelse return;

    const parsed = db_blob_store.parseRef(blob_ref) catch {
        invalid_ref_count.* += 1;
        return;
    };
    try markReferenceRecursive(store, parsed, digests, invalid_ref_count, store.allocator);
}

fn markReferenceRecursive(
    store: *db_blob_store.BlobStore,
    parsed_ref: db_blob_store.ParsedRef,
    digests: *DigestSet,
    invalid_ref_count: *usize,
    allocator: Allocator,
) !void {
    const gop = try digests.getOrPut(parsed_ref.digest_hex);
    if (gop.found_existing) return;

    if (parsed_ref.kind != .multipart) return;

    const manifest_sha_ref = db_blob_store.buildSha256RefString(parsed_ref.digest_hex);
    const manifest_bytes = store.readAll(manifest_sha_ref[0..db_blob_store.sha256_ref_len], allocator) catch {
        invalid_ref_count.* += 1;
        return;
    };
    defer allocator.free(manifest_bytes);

    var parsed_manifest = std.json.parseFromSlice(MultipartManifest, allocator, manifest_bytes, .{}) catch {
        invalid_ref_count.* += 1;
        return;
    };
    defer parsed_manifest.deinit();

    const manifest = parsed_manifest.value;
    if (manifest.v != 1 or !std.mem.eql(u8, manifest.type, "multipart")) {
        invalid_ref_count.* += 1;
        return;
    }

    for (manifest.parts) |part_ref| {
        const part_parsed = db_blob_store.parseRef(part_ref) catch {
            invalid_ref_count.* += 1;
            continue;
        };
        try markReferenceRecursive(store, part_parsed, digests, invalid_ref_count, allocator);
    }
}

fn buildDigestFromName(shard_name: []const u8, digest_name: []const u8) ?DigestKey {
    if (shard_name.len != 2) return null;
    if (digest_name.len != db_blob_store.digest_hex_len) return null;

    var out: DigestKey = undefined;

    for (digest_name, 0..) |ch, idx| {
        if (!std.ascii.isHex(ch)) return null;
        out[idx] = std.ascii.toLower(ch);
    }

    if (std.ascii.toLower(shard_name[0]) != out[0]) return null;
    if (std.ascii.toLower(shard_name[1]) != out[1]) return null;
    return out;
}

fn isWalName(name: []const u8) bool {
    if (std.mem.eql(u8, name, "current.wal")) return true;
    return std.mem.startsWith(u8, name, "wal-") and std.mem.endsWith(u8, name, ".wal");
}

fn findColumn(descs: []const @import("../types.zig").ColumnDesc, col_id: u32) ?@import("../types.zig").ColumnDesc {
    for (descs) |d| {
        if (d.column_id == col_id) return d;
    }
    return null;
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
        const end = start + @as(usize, self.lengths[row_idx]);
        if (end > self.data.len) return error.InvalidColumnData;
        return self.data[start..end];
    }
};

fn readVarBytesBuffers(
    file: std.fs.File,
    block_offset: u64,
    desc: @import("../types.zig").ColumnDesc,
    row_count: u32,
    allocator: Allocator,
) !VarBytesBuffers {
    if (desc.offsets_off == 0 or desc.lengths_off == 0) return error.InvalidColumnLayout;

    const reader = db_block_reader.BlockReader.init(file, allocator);
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
    const byte_len = @as(usize, count) * @sizeOf(u32);
    const bytes = try allocator.alloc(u8, byte_len);
    defer allocator.free(bytes);

    const read_len = try file.preadAll(bytes, offset);
    if (read_len != bytes.len) return error.UnexpectedEof;

    const out = try allocator.alloc(u32, count);
    var i: usize = 0;
    while (i < out.len) : (i += 1) {
        const start = i * 4;
        out[i] = std.mem.readInt(u32, bytes[start..][0..4], .little);
    }
    return out;
}

fn readU8(bytes: []const u8, index: *usize) !u8 {
    if (index.* >= bytes.len) return error.InvalidWalEntry;
    const v = bytes[index.*];
    index.* += 1;
    return v;
}

fn readU16(bytes: []const u8, index: *usize) !u16 {
    const start = index.*;
    if (start + 2 > bytes.len) return error.InvalidWalEntry;
    index.* = start + 2;
    return std.mem.readInt(u16, bytes[start..][0..2], .little);
}

fn readU32(bytes: []const u8, index: *usize) !u32 {
    const start = index.*;
    if (start + 4 > bytes.len) return error.InvalidWalEntry;
    index.* = start + 4;
    return std.mem.readInt(u32, bytes[start..][0..4], .little);
}

fn readBytes(bytes: []const u8, index: *usize, len: u32) ![]const u8 {
    const start = index.*;
    const end = start + @as(usize, len);
    if (end > bytes.len) return error.InvalidWalEntry;
    index.* = end;
    return bytes[start..end];
}

fn appendPayloadRow(
    allocator: Allocator,
    db_root: []const u8,
    namespace: []const u8,
    schema_id: u16,
    payload: []const u8,
    flush: bool,
) !db_writer.Writer {
    var writer = try db_writer.Writer.open(allocator, db_root, namespace);
    writer.setDurability(.full);

    const col = db_writer.ColumnValue{
        .column_id = col_payload,
        .shape = .VARBYTES,
        .phys_type = .BINARY,
        .encoding = .KVBUF,
        .dims = 1,
        .data = payload,
    };
    try writer.appendRow(schema_id, &.{col});
    if (flush) try writer.flushBlock();
    return writer;
}

fn containsRef(refs: []const RefKey, needle: []const u8) bool {
    for (refs) |item| {
        if (std.mem.eql(u8, item[0..], needle)) return true;
    }
    return false;
}

test "collectReferencedBlobRefs finds refs in blocks and WAL" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(root_path);

    var store = try db_blob_store.BlobStore.init(allocator, root_path);
    defer store.deinit();

    const doc_blob = try store.put("doc-block-ref");
    const chat_blob = try store.put("chat-block-ref");
    const wal_blob = try store.put("doc-wal-ref");

    {
        var doc_payload_writer = kvbuf.KvBufWriter.init();
        defer doc_payload_writer.deinit(allocator);
        try doc_payload_writer.addString(allocator, kvbuf.DocumentFieldIds.doc_json_ref, doc_blob.refSlice());
        const doc_payload = try doc_payload_writer.finish(allocator);
        defer allocator.free(doc_payload);
        var doc_writer = try appendPayloadRow(allocator, root_path, docs_namespace, schema_documents, doc_payload, true);
        doc_writer.deinit();
    }

    {
        var chat_payload_writer = kvbuf.KvBufWriter.init();
        defer chat_payload_writer.deinit(allocator);
        try chat_payload_writer.addString(allocator, kvbuf.FieldIds.record_json_ref, chat_blob.refSlice());
        const chat_payload = try chat_payload_writer.finish(allocator);
        defer allocator.free(chat_payload);
        var chat_writer = try appendPayloadRow(allocator, root_path, chat_namespace, schema_items, chat_payload, true);
        chat_writer.deinit();
    }

    var wal_writer: ?db_writer.Writer = null;
    defer if (wal_writer) |*w| w.deinit();
    {
        var wal_payload_writer = kvbuf.KvBufWriter.init();
        defer wal_payload_writer.deinit(allocator);
        try wal_payload_writer.addString(allocator, kvbuf.DocumentFieldIds.doc_json_ref, wal_blob.refSlice());
        const wal_payload = try wal_payload_writer.finish(allocator);
        defer allocator.free(wal_payload);
        wal_writer = try appendPayloadRow(allocator, root_path, docs_namespace, schema_documents, wal_payload, false);
    }

    const refs = try collectReferencedBlobRefs(allocator, root_path);
    defer allocator.free(refs);

    try std.testing.expectEqual(@as(usize, 3), refs.len);
    try std.testing.expect(containsRef(refs, doc_blob.refSlice()));
    try std.testing.expect(containsRef(refs, chat_blob.refSlice()));
    try std.testing.expect(containsRef(refs, wal_blob.refSlice()));
}

test "sweepUnreferencedBlobs deletes unreferenced blobs and preserves referenced blobs" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(root_path);

    var store = try db_blob_store.BlobStore.init(allocator, root_path);
    defer store.deinit();

    const doc_blob = try store.put("doc-sweep-ref");
    const chat_blob = try store.put("chat-sweep-ref");
    const orphan_blob = try store.put("orphan-sweep-ref");

    {
        var doc_payload_writer = kvbuf.KvBufWriter.init();
        defer doc_payload_writer.deinit(allocator);
        try doc_payload_writer.addString(allocator, kvbuf.DocumentFieldIds.doc_json_ref, doc_blob.refSlice());
        const doc_payload = try doc_payload_writer.finish(allocator);
        defer allocator.free(doc_payload);
        var doc_writer = try appendPayloadRow(allocator, root_path, docs_namespace, schema_documents, doc_payload, true);
        doc_writer.deinit();
    }

    {
        var chat_payload_writer = kvbuf.KvBufWriter.init();
        defer chat_payload_writer.deinit(allocator);
        try chat_payload_writer.addString(allocator, kvbuf.FieldIds.record_json_ref, chat_blob.refSlice());
        const chat_payload = try chat_payload_writer.finish(allocator);
        defer allocator.free(chat_payload);
        var chat_writer = try appendPayloadRow(allocator, root_path, chat_namespace, schema_items, chat_payload, true);
        chat_writer.deinit();
    }

    const stats = try sweepUnreferencedBlobsWithOptions(allocator, root_path, .{
        .min_blob_age_seconds = 0,
    });
    try std.testing.expectEqual(@as(usize, 2), stats.referenced_blob_count);
    try std.testing.expectEqual(@as(usize, 3), stats.total_blob_files);
    try std.testing.expectEqual(@as(usize, 1), stats.deleted_blob_files);
    try std.testing.expectEqual(@as(usize, 0), stats.skipped_recent_blob_files);
    try std.testing.expect(stats.reclaimed_bytes > 0);

    const doc_bytes = try store.readAll(doc_blob.refSlice(), allocator);
    defer allocator.free(doc_bytes);
    const chat_bytes = try store.readAll(chat_blob.refSlice(), allocator);
    defer allocator.free(chat_bytes);
    try std.testing.expectError(error.FileNotFound, store.readAll(orphan_blob.refSlice(), allocator));
}

test "sweepUnreferencedBlobs keeps blobs referenced only from WAL" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(root_path);

    var store = try db_blob_store.BlobStore.init(allocator, root_path);
    defer store.deinit();

    const wal_ref_blob = try store.put("wal-only-ref");
    const orphan_blob = try store.put("wal-orphan");

    var payload_writer = kvbuf.KvBufWriter.init();
    defer payload_writer.deinit(allocator);
    try payload_writer.addString(allocator, kvbuf.DocumentFieldIds.doc_json_ref, wal_ref_blob.refSlice());
    const payload = try payload_writer.finish(allocator);
    defer allocator.free(payload);

    var writer = try appendPayloadRow(allocator, root_path, docs_namespace, schema_documents, payload, false);
    defer writer.deinit();

    const stats = try sweepUnreferencedBlobsWithOptions(allocator, root_path, .{
        .min_blob_age_seconds = 0,
    });
    try std.testing.expectEqual(@as(usize, 1), stats.referenced_blob_count);
    try std.testing.expectEqual(@as(usize, 2), stats.total_blob_files);
    try std.testing.expectEqual(@as(usize, 1), stats.deleted_blob_files);
    try std.testing.expectEqual(@as(usize, 0), stats.skipped_recent_blob_files);

    const wal_bytes = try store.readAll(wal_ref_blob.refSlice(), allocator);
    defer allocator.free(wal_bytes);
    try std.testing.expectEqualStrings("wal-only-ref", wal_bytes);

    try std.testing.expectError(error.FileNotFound, store.readAll(orphan_blob.refSlice(), allocator));
}

test "sweepUnreferencedBlobs preserves multipart manifest and chunk blobs" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(root_path);

    var store = try db_blob_store.BlobStore.init(allocator, root_path);
    defer store.deinit();
    store.multipart_chunk_size_bytes = 16;

    const multi_blob = try store.putAuto("abcdefghijklmnopqrstuvwxyz0123456789--multipart-gc");
    try std.testing.expect(std.mem.startsWith(u8, multi_blob.refSlice(), db_blob_store.multipart_ref_prefix));
    const orphan_blob = try store.put("multipart-orphan");

    var payload_writer = kvbuf.KvBufWriter.init();
    defer payload_writer.deinit(allocator);
    try payload_writer.addString(allocator, kvbuf.DocumentFieldIds.doc_json_ref, multi_blob.refSlice());
    const payload = try payload_writer.finish(allocator);
    defer allocator.free(payload);

    var writer = try appendPayloadRow(allocator, root_path, docs_namespace, schema_documents, payload, true);
    defer writer.deinit();

    const stats = try sweepUnreferencedBlobsWithOptions(allocator, root_path, .{
        .min_blob_age_seconds = 0,
    });

    // multipart-gc should keep at least manifest + one chunk.
    try std.testing.expect(stats.referenced_blob_count >= 2);
    try std.testing.expect(stats.deleted_blob_files >= 1);

    const loaded = try store.readAll(multi_blob.refSlice(), allocator);
    defer allocator.free(loaded);
    try std.testing.expectEqualStrings("abcdefghijklmnopqrstuvwxyz0123456789--multipart-gc", loaded);
    try std.testing.expectError(error.FileNotFound, store.readAll(orphan_blob.refSlice(), allocator));
}

test "sweepUnreferencedBlobs default grace period protects recent unreferenced blobs" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(root_path);

    var store = try db_blob_store.BlobStore.init(allocator, root_path);
    defer store.deinit();

    _ = try store.put("referenced-doc");
    const orphan_blob = try store.put("recent-orphan");

    const stats = try sweepUnreferencedBlobs(allocator, root_path);
    try std.testing.expectEqual(@as(usize, 0), stats.referenced_blob_count);
    try std.testing.expectEqual(@as(usize, 2), stats.total_blob_files);
    try std.testing.expectEqual(@as(usize, 0), stats.deleted_blob_files);
    try std.testing.expect(stats.skipped_recent_blob_files >= 2);

    const orphan_bytes = try store.readAll(orphan_blob.refSlice(), allocator);
    defer allocator.free(orphan_bytes);
    try std.testing.expectEqualStrings("recent-orphan", orphan_bytes);
}
