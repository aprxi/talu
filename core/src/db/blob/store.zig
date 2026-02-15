//! Content-addressable blob storage for large payloads.
//!
//! Stores immutable blobs under:
//!   blobs/<first-two-hex>/<full-64-hex-sha256>
//!
//! References are encoded as:
//!   sha256:<64-hex>          (single blob)
//!   multi:<64-hex>           (multipart manifest blob digest)

const std = @import("std");

const Allocator = std.mem.Allocator;
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const sha256_ref_prefix = "sha256:";
pub const multipart_ref_prefix = "multi:";
pub const ref_prefix = sha256_ref_prefix; // Backward-compatible alias.
pub const digest_hex_len: usize = 64;
pub const sha256_ref_len: usize = sha256_ref_prefix.len + digest_hex_len;
pub const multipart_ref_len: usize = multipart_ref_prefix.len + digest_hex_len;
pub const ref_len: usize = @max(sha256_ref_len, multipart_ref_len);
pub const rel_path_len: usize = "blobs/".len + 2 + 1 + digest_hex_len;
pub const default_multipart_chunk_size_bytes: usize = 32 * 1024 * 1024;

const tmp_suffix_hex_len: usize = 16;
const tmp_path_len: usize = "blobs/".len + 2 + 1 + ".tmp-".len + tmp_suffix_hex_len;
const hex_chars = "0123456789abcdef";
const env_multipart_chunk_size_bytes = "TALU_DB_BLOB_MULTIPART_CHUNK_SIZE_BYTES";

pub const RefKind = enum {
    sha256,
    multipart,
};

pub const ParsedRef = struct {
    kind: RefKind,
    digest_hex: [digest_hex_len]u8,
};

const DigestKey = [digest_hex_len]u8;

const MultipartManifest = struct {
    v: u8,
    type: []const u8,
    total_size: u64,
    part_size: u64,
    parts: []const []const u8,
};

pub const BlobRef = struct {
    ref: [ref_len]u8,
    ref_len: u8,
    rel_path: [rel_path_len]u8,
    size_bytes: u64,

    pub fn refSlice(self: *const BlobRef) []const u8 {
        return self.ref[0..self.ref_len];
    }
};

pub const BlobReadStream = struct {
    store: *BlobStore,
    allocator: Allocator,
    part_digests: []DigestKey,
    part_sizes: []u64,
    total_size: u64,
    bytes_read: u64 = 0,
    current_part_idx: usize = 0,
    current_part_file: ?std.fs.File = null,

    pub fn deinit(self: *BlobReadStream) void {
        if (self.current_part_file) |file| {
            file.close();
            self.current_part_file = null;
        }
        self.allocator.free(self.part_digests);
        self.allocator.free(self.part_sizes);
    }

    /// Read bytes from the blob stream into `buffer`.
    /// Returns 0 when the stream is exhausted.
    pub fn read(self: *BlobReadStream, buffer: []u8) !usize {
        if (buffer.len == 0) return 0;

        var written: usize = 0;
        while (written < buffer.len) {
            if (self.current_part_file == null) {
                if (!try self.openNextPart()) break;
            }

            var file = self.current_part_file.?;
            const read_len = try file.read(buffer[written..]);
            if (read_len == 0) {
                file.close();
                self.current_part_file = null;
                self.current_part_idx += 1;
                continue;
            }

            written += read_len;
            self.bytes_read += read_len;
        }

        return written;
    }

    /// Seek to an absolute byte offset from the beginning of the blob.
    pub fn seek(self: *BlobReadStream, offset: u64) !void {
        if (offset > self.total_size) return error.InvalidSeekOffset;

        if (self.current_part_file) |file| {
            file.close();
            self.current_part_file = null;
        }

        self.bytes_read = offset;
        if (offset == self.total_size) {
            self.current_part_idx = self.part_digests.len;
            return;
        }

        var part_idx: usize = 0;
        var remaining = offset;
        while (part_idx < self.part_sizes.len and remaining >= self.part_sizes[part_idx]) : (part_idx += 1) {
            remaining -= self.part_sizes[part_idx];
        }
        if (part_idx >= self.part_digests.len) return error.InvalidSeekOffset;

        var file = try self.store.openSingleByDigest(self.part_digests[part_idx]);
        errdefer file.close();
        if (remaining > 0) try file.seekTo(remaining);

        self.current_part_idx = part_idx;
        self.current_part_file = file;
    }

    fn openNextPart(self: *BlobReadStream) !bool {
        if (self.current_part_idx >= self.part_digests.len) return false;
        self.current_part_file = try self.store.openSingleByDigest(self.part_digests[self.current_part_idx]);
        return true;
    }
};

pub const BlobPutStream = struct {
    store: *BlobStore,
    allocator: Allocator,
    chunk_size: usize,
    current_chunk: std.ArrayList(u8),
    part_digests: std.ArrayList(DigestKey),
    total_size: u64 = 0,

    pub fn init(store: *BlobStore, allocator: Allocator) BlobPutStream {
        const resolved_chunk_size = if (store.multipart_chunk_size_bytes == 0)
            default_multipart_chunk_size_bytes
        else
            store.multipart_chunk_size_bytes;
        return .{
            .store = store,
            .allocator = allocator,
            .chunk_size = resolved_chunk_size,
            .current_chunk = .empty,
            .part_digests = .empty,
        };
    }

    pub fn deinit(self: *BlobPutStream) void {
        self.current_chunk.deinit(self.allocator);
        self.part_digests.deinit(self.allocator);
    }

    pub fn write(self: *BlobPutStream, bytes: []const u8) !void {
        var remaining = bytes;
        while (remaining.len > 0) {
            if (self.current_chunk.items.len == self.chunk_size) {
                try self.flushCurrentChunk();
            }

            const space = self.chunk_size - self.current_chunk.items.len;
            const take_len = @min(space, remaining.len);
            try self.current_chunk.appendSlice(self.allocator, remaining[0..take_len]);
            self.total_size += take_len;
            remaining = remaining[take_len..];
        }
    }

    pub fn finish(self: *BlobPutStream) !BlobRef {
        if (self.part_digests.items.len == 0) {
            return self.store.putSingle(self.current_chunk.items);
        }

        if (self.current_chunk.items.len > 0) {
            try self.flushCurrentChunk();
        }
        return self.store.putMultipartFromDigests(
            self.part_digests.items,
            self.total_size,
            self.chunk_size,
        );
    }

    fn flushCurrentChunk(self: *BlobPutStream) !void {
        if (self.current_chunk.items.len == 0) return;
        const part_ref = try self.store.putSingle(self.current_chunk.items);
        const parsed = try parseRef(part_ref.refSlice());
        if (parsed.kind != .sha256) return error.InvalidMultipartManifest;
        try self.part_digests.append(self.allocator, parsed.digest_hex);
        self.current_chunk.clearRetainingCapacity();
    }
};

pub const BlobStore = struct {
    allocator: Allocator,
    root_dir: std.fs.Dir,
    multipart_chunk_size_bytes: usize,

    pub fn init(allocator: Allocator, db_root: []const u8) !BlobStore {
        const root_dir = try std.fs.cwd().makeOpenPath(db_root, .{});
        return .{
            .allocator = allocator,
            .root_dir = root_dir,
            .multipart_chunk_size_bytes = resolveMultipartChunkSizeBytes(),
        };
    }

    pub fn deinit(self: *BlobStore) void {
        self.root_dir.close();
    }

    /// Open a streaming reader over a single or multipart blob reference.
    pub fn openReadStream(self: *BlobStore, allocator: Allocator, blob_ref: []const u8) !BlobReadStream {
        const parsed = try parseRef(blob_ref);
        return switch (parsed.kind) {
            .sha256 => blk: {
                const part_digests = try allocator.alloc(DigestKey, 1);
                errdefer allocator.free(part_digests);
                const part_sizes = try allocator.alloc(u64, 1);
                errdefer allocator.free(part_sizes);
                part_digests[0] = parsed.digest_hex;
                const stat = try self.statSingleByDigest(parsed.digest_hex);
                part_sizes[0] = stat.size;
                break :blk .{
                    .store = self,
                    .allocator = allocator,
                    .part_digests = part_digests,
                    .part_sizes = part_sizes,
                    .total_size = @intCast(stat.size),
                };
            },
            .multipart => blk: {
                const parts = try self.loadMultipartParts(allocator, parsed.digest_hex);
                errdefer allocator.free(parts.part_digests);
                const part_sizes = try allocator.alloc(u64, parts.part_digests.len);
                errdefer allocator.free(part_sizes);

                var computed_total_size: u64 = 0;
                for (parts.part_digests, 0..) |part_digest, i| {
                    const part_stat = try self.statSingleByDigest(part_digest);
                    part_sizes[i] = part_stat.size;
                    computed_total_size += part_stat.size;
                }
                if (computed_total_size != parts.total_size) return error.InvalidMultipartManifest;

                break :blk .{
                    .store = self,
                    .allocator = allocator,
                    .part_digests = parts.part_digests,
                    .part_sizes = part_sizes,
                    .total_size = parts.total_size,
                };
            },
        };
    }

    /// Store bytes as either a single CAS blob or a multipart manifest.
    /// Uses `multipart_chunk_size_bytes` as the split threshold.
    pub fn putAuto(self: *BlobStore, bytes: []const u8) !BlobRef {
        if (self.multipart_chunk_size_bytes > 0 and bytes.len > self.multipart_chunk_size_bytes) {
            return self.putMultipart(bytes);
        }
        return self.put(bytes);
    }

    /// Store bytes as a single CAS blob (`sha256:<hex>`).
    pub fn put(self: *BlobStore, bytes: []const u8) !BlobRef {
        return self.putSingle(bytes);
    }

    /// Store bytes as multipart CAS using a manifest (`multi:<hex>`).
    /// The manifest blob itself is still stored as a normal `sha256:<hex>` blob.
    pub fn putMultipart(self: *BlobStore, bytes: []const u8) !BlobRef {
        const part_size = self.multipart_chunk_size_bytes;
        if (part_size == 0) return error.InvalidChunkSize;
        if (bytes.len <= part_size) return self.putSingle(bytes);

        var digests = std.ArrayList(DigestKey).empty;
        defer digests.deinit(self.allocator);

        var offset: usize = 0;
        while (offset < bytes.len) {
            const end = @min(offset + part_size, bytes.len);
            const part_ref = try self.putSingle(bytes[offset..end]);
            const parsed = try parseRef(part_ref.refSlice());
            if (parsed.kind != .sha256) return error.InvalidMultipartManifest;
            try digests.append(self.allocator, parsed.digest_hex);
            offset = end;
        }

        return self.putMultipartFromDigests(digests.items, bytes.len, part_size);
    }

    pub fn open(self: *BlobStore, blob_ref: []const u8) !std.fs.File {
        const parsed = try parseRef(blob_ref);
        if (parsed.kind != .sha256) return error.UnsupportedRefKind;
        const rel_path = buildRelativePath(parsed.digest_hex);
        return self.root_dir.openFile(rel_path[0..], .{ .mode = .read_only });
    }

    pub fn readAll(self: *BlobStore, blob_ref: []const u8, allocator: Allocator) ![]u8 {
        const parsed = try parseRef(blob_ref);
        return switch (parsed.kind) {
            .sha256 => self.readSingleByDigest(parsed.digest_hex, allocator),
            .multipart => self.readMultipartByDigest(parsed.digest_hex, allocator),
        };
    }

    pub fn verify(self: *BlobStore, blob_ref: []const u8) !void {
        const parsed = try parseRef(blob_ref);
        switch (parsed.kind) {
            .sha256 => try self.verifySingleByDigest(parsed.digest_hex),
            .multipart => try self.verifyMultipartByDigest(parsed.digest_hex),
        }
    }

    /// Check whether a blob reference currently resolves to stored content.
    pub fn exists(self: *BlobStore, blob_ref: []const u8) !bool {
        const parsed = try parseRef(blob_ref);
        return switch (parsed.kind) {
            .sha256 => try self.existsSingleByDigest(parsed.digest_hex),
            .multipart => try self.existsMultipartByDigest(parsed.digest_hex),
        };
    }

    /// List up to `limit` stored single-blob digests (`sha256` objects on disk).
    /// Pass `0` for `limit` to return all digests.
    pub fn listSingleDigests(self: *BlobStore, allocator: Allocator, limit: usize) ![]DigestKey {
        var out = std.ArrayList(DigestKey).empty;
        errdefer out.deinit(allocator);

        var blobs_dir = self.root_dir.openDir("blobs", .{ .iterate = true }) catch |err| switch (err) {
            error.FileNotFound => return out.toOwnedSlice(allocator),
            else => return err,
        };
        defer blobs_dir.close();

        var shard_iter = blobs_dir.iterate();
        shards: while (try shard_iter.next()) |shard_entry| {
            if (shard_entry.kind != .directory) continue;
            if (shard_entry.name.len != 2) continue;
            if (!std.ascii.isHex(shard_entry.name[0]) or !std.ascii.isHex(shard_entry.name[1])) continue;

            var shard_dir = blobs_dir.openDir(shard_entry.name, .{ .iterate = true }) catch continue;
            defer shard_dir.close();

            var file_iter = shard_dir.iterate();
            while (try file_iter.next()) |file_entry| {
                if (file_entry.kind != .file) continue;
                if (file_entry.name.len != digest_hex_len) continue;

                var digest: DigestKey = undefined; // Filled element-by-element in the loop below
                var valid = true;
                for (file_entry.name, 0..) |ch, i| {
                    if (!std.ascii.isHex(ch)) {
                        valid = false;
                        break;
                    }
                    digest[i] = std.ascii.toLower(ch);
                }
                if (!valid) continue;

                try out.append(allocator, digest);
                if (limit > 0 and out.items.len >= limit) break :shards;
            }
        }

        const SortCtx = struct {
            fn lessThan(_: void, a: DigestKey, b: DigestKey) bool {
                return std.mem.lessThan(u8, a[0..], b[0..]);
            }
        };
        std.mem.sort(DigestKey, out.items, {}, SortCtx.lessThan);
        return out.toOwnedSlice(allocator);
    }

    fn putSingle(self: *BlobStore, bytes: []const u8) !BlobRef {
        var digest: [32]u8 = undefined;
        Sha256.hash(bytes, &digest, .{});
        const digest_hex = digestToHex(digest);

        const blob_ref = BlobRef{
            .ref = buildSha256RefString(digest_hex),
            .ref_len = sha256_ref_len,
            .rel_path = buildRelativePath(digest_hex),
            .size_bytes = bytes.len,
        };

        try self.ensureShardDir(digest_hex[0], digest_hex[1]);

        var attempt: usize = 0;
        while (attempt < 8) : (attempt += 1) {
            const tmp_path = makeTempPath(digest_hex[0], digest_hex[1]);
            var tmp_file = self.root_dir.createFile(tmp_path[0..], .{ .read = true, .exclusive = true }) catch |err| switch (err) {
                error.PathAlreadyExists => continue,
                else => return err,
            };
            errdefer tmp_file.close();
            errdefer self.root_dir.deleteFile(tmp_path[0..]) catch {};

            try tmp_file.writeAll(bytes);
            try tmp_file.sync();
            tmp_file.close();

            self.root_dir.rename(tmp_path[0..], blob_ref.rel_path[0..]) catch |err| switch (err) {
                error.PathAlreadyExists => {
                    self.root_dir.deleteFile(tmp_path[0..]) catch |delete_err| switch (delete_err) {
                        error.FileNotFound => {},
                        else => return delete_err,
                    };
                    return blob_ref;
                },
                else => {
                    self.root_dir.deleteFile(tmp_path[0..]) catch {};
                    return err;
                },
            };

            return blob_ref;
        }

        return error.TempPathCollision;
    }

    fn readSingleByDigest(self: *BlobStore, digest_hex: [digest_hex_len]u8, allocator: Allocator) ![]u8 {
        var file = try self.openSingleByDigest(digest_hex);
        defer file.close();

        const stat = try file.stat();
        if (stat.size > std.math.maxInt(usize)) return error.FileTooLarge;
        return file.readToEndAlloc(allocator, @intCast(stat.size));
    }

    fn readMultipartByDigest(self: *BlobStore, manifest_digest: [digest_hex_len]u8, allocator: Allocator) ![]u8 {
        const parts = try self.loadMultipartParts(allocator, manifest_digest);
        defer allocator.free(parts.part_digests);

        if (parts.total_size > std.math.maxInt(usize)) return error.FileTooLarge;
        const total_size: usize = @intCast(parts.total_size);
        var out = try allocator.alloc(u8, total_size);
        errdefer allocator.free(out);

        var offset: usize = 0;
        for (parts.part_digests) |part_digest| {
            const part_bytes = try self.readSingleByDigest(part_digest, allocator);
            defer allocator.free(part_bytes);

            if (offset + part_bytes.len > out.len) return error.InvalidMultipartManifest;
            @memcpy(out[offset .. offset + part_bytes.len], part_bytes);
            offset += part_bytes.len;
        }

        if (offset != out.len) return error.InvalidMultipartManifest;
        return out;
    }

    fn verifySingleByDigest(self: *BlobStore, expected_hex: [digest_hex_len]u8) !void {
        var file = try self.openSingleByDigest(expected_hex);
        defer file.close();

        var hasher = Sha256.init(.{});
        var buf: [16 * 1024]u8 = undefined; // Scratch buffer, overwritten by file.read each iteration
        while (true) {
            const read_len = try file.read(&buf);
            if (read_len == 0) break;
            hasher.update(buf[0..read_len]);
        }

        var digest: [32]u8 = undefined; // Overwritten by hasher.final on the next line
        hasher.final(&digest);
        const actual_hex = digestToHex(digest);
        if (!std.mem.eql(u8, expected_hex[0..], actual_hex[0..])) {
            return error.IntegrityMismatch;
        }
    }

    fn verifyMultipartByDigest(self: *BlobStore, manifest_digest: [digest_hex_len]u8) !void {
        try self.verifySingleByDigest(manifest_digest);

        const parts = try self.loadMultipartParts(self.allocator, manifest_digest);
        defer self.allocator.free(parts.part_digests);
        for (parts.part_digests) |part_digest| {
            try self.verifySingleByDigest(part_digest);
        }
    }

    fn existsSingleByDigest(self: *BlobStore, digest_hex: DigestKey) !bool {
        _ = self.statSingleByDigest(digest_hex) catch |err| switch (err) {
            error.FileNotFound => return false,
            else => return err,
        };
        return true;
    }

    fn existsMultipartByDigest(self: *BlobStore, manifest_digest: DigestKey) !bool {
        if (!try self.existsSingleByDigest(manifest_digest)) return false;

        const parts = self.loadMultipartParts(self.allocator, manifest_digest) catch |err| switch (err) {
            error.FileNotFound,
            error.InvalidMultipartManifest,
            => return false,
            else => return err,
        };
        defer self.allocator.free(parts.part_digests);

        for (parts.part_digests) |part_digest| {
            if (!try self.existsSingleByDigest(part_digest)) return false;
        }
        return true;
    }

    const MultipartParts = struct {
        part_digests: []DigestKey,
        total_size: u64,
    };

    fn loadMultipartParts(self: *BlobStore, allocator: Allocator, manifest_digest: DigestKey) !MultipartParts {
        const manifest_bytes = try self.readSingleByDigest(manifest_digest, allocator);
        defer allocator.free(manifest_bytes);

        var parsed = std.json.parseFromSlice(MultipartManifest, allocator, manifest_bytes, .{}) catch return error.InvalidMultipartManifest;
        defer parsed.deinit();

        const manifest = parsed.value;
        if (manifest.v != 1) return error.InvalidMultipartManifest;
        if (!std.mem.eql(u8, manifest.type, "multipart")) return error.InvalidMultipartManifest;
        if (manifest.part_size == 0) return error.InvalidMultipartManifest;

        const part_digests = try allocator.alloc(DigestKey, manifest.parts.len);
        errdefer allocator.free(part_digests);
        for (manifest.parts, 0..) |part_ref, i| {
            const part_parsed = parseRef(part_ref) catch return error.InvalidMultipartManifest;
            if (part_parsed.kind != .sha256) return error.InvalidMultipartManifest;
            part_digests[i] = part_parsed.digest_hex;
        }

        return .{
            .part_digests = part_digests,
            .total_size = manifest.total_size,
        };
    }

    fn openSingleByDigest(self: *BlobStore, digest_hex: DigestKey) !std.fs.File {
        const rel_path = buildRelativePath(digest_hex);
        return self.root_dir.openFile(rel_path[0..], .{ .mode = .read_only });
    }

    fn statSingleByDigest(self: *BlobStore, digest_hex: DigestKey) !std.fs.File.Stat {
        const rel_path = buildRelativePath(digest_hex);
        return self.root_dir.statFile(rel_path[0..]);
    }

    fn ensureShardDir(self: *BlobStore, shard0: u8, shard1: u8) !void {
        var shard_path: [8]u8 = undefined; // "blobs/aa"
        @memcpy(shard_path[0.."blobs/".len], "blobs/");
        shard_path["blobs/".len] = shard0;
        shard_path["blobs/".len + 1] = shard1;
        try self.root_dir.makePath(shard_path[0..]);
    }

    fn putMultipartFromDigests(
        self: *BlobStore,
        part_digests: []const DigestKey,
        total_size: u64,
        part_size: usize,
    ) !BlobRef {
        if (part_digests.len == 0) return error.InvalidMultipartManifest;
        if (part_size == 0) return error.InvalidChunkSize;

        var manifest_json = std.ArrayList(u8).empty;
        defer manifest_json.deinit(self.allocator);

        const writer = manifest_json.writer(self.allocator);
        try writer.print(
            "{{\"v\":1,\"type\":\"multipart\",\"total_size\":{d},\"part_size\":{d},\"parts\":[",
            .{ total_size, part_size },
        );
        for (part_digests, 0..) |part_digest, idx| {
            if (idx > 0) try writer.writeByte(',');
            const ref_buf = buildSha256RefString(part_digest);
            try writer.print("\"{s}\"", .{ref_buf[0..sha256_ref_len]});
        }
        try writer.writeAll("]}");

        const manifest_bytes = try manifest_json.toOwnedSlice(self.allocator);
        defer self.allocator.free(manifest_bytes);

        const manifest_blob_ref = try self.putSingle(manifest_bytes);
        const parsed_manifest_ref = try parseRef(manifest_blob_ref.refSlice());
        if (parsed_manifest_ref.kind != .sha256) return error.InvalidMultipartManifest;

        return .{
            .ref = buildMultipartRefString(parsed_manifest_ref.digest_hex),
            .ref_len = multipart_ref_len,
            .rel_path = manifest_blob_ref.rel_path,
            .size_bytes = total_size,
        };
    }
};

pub fn parseRef(blob_ref: []const u8) !ParsedRef {
    var digest_hex: [digest_hex_len]u8 = undefined; // Filled element-by-element in the matching branch below

    if (blob_ref.len == sha256_ref_len and std.mem.startsWith(u8, blob_ref, sha256_ref_prefix)) {
        for (blob_ref[sha256_ref_prefix.len..], 0..) |ch, idx| {
            if (!std.ascii.isHex(ch)) return error.InvalidBlobRef;
            digest_hex[idx] = std.ascii.toLower(ch);
        }
        return .{ .kind = .sha256, .digest_hex = digest_hex };
    }
    if (blob_ref.len == multipart_ref_len and std.mem.startsWith(u8, blob_ref, multipart_ref_prefix)) {
        for (blob_ref[multipart_ref_prefix.len..], 0..) |ch, idx| {
            if (!std.ascii.isHex(ch)) return error.InvalidBlobRef;
            digest_hex[idx] = std.ascii.toLower(ch);
        }
        return .{ .kind = .multipart, .digest_hex = digest_hex };
    }
    return error.InvalidBlobRef;
}

fn digestToHex(digest: [32]u8) [digest_hex_len]u8 {
    var out: [digest_hex_len]u8 = undefined;
    for (digest, 0..) |byte, idx| {
        out[idx * 2] = hex_chars[byte >> 4];
        out[idx * 2 + 1] = hex_chars[byte & 0x0f];
    }
    return out;
}

pub fn buildSha256RefString(digest_hex: [digest_hex_len]u8) [ref_len]u8 {
    var out: [ref_len]u8 = undefined;
    @memcpy(out[0..sha256_ref_prefix.len], sha256_ref_prefix);
    @memcpy(out[sha256_ref_prefix.len .. sha256_ref_prefix.len + digest_hex_len], digest_hex[0..]);
    return out;
}

pub fn buildMultipartRefString(digest_hex: [digest_hex_len]u8) [ref_len]u8 {
    var out: [ref_len]u8 = undefined;
    @memcpy(out[0..multipart_ref_prefix.len], multipart_ref_prefix);
    @memcpy(out[multipart_ref_prefix.len .. multipart_ref_prefix.len + digest_hex_len], digest_hex[0..]);
    return out;
}

fn buildRelativePath(digest_hex: [digest_hex_len]u8) [rel_path_len]u8 {
    var out: [rel_path_len]u8 = undefined;
    // Keep blob references canonical with '/' so refs are portable across
    // filesystems/object stores and stable in persisted metadata.
    @memcpy(out[0.."blobs/".len], "blobs/");
    out["blobs/".len] = digest_hex[0];
    out["blobs/".len + 1] = digest_hex[1];
    out["blobs/".len + 2] = '/';
    @memcpy(out["blobs/".len + 3 ..], digest_hex[0..]);
    return out;
}

fn makeTempPath(shard0: u8, shard1: u8) [tmp_path_len]u8 {
    var out: [tmp_path_len]u8 = undefined;
    @memcpy(out[0.."blobs/".len], "blobs/");
    out["blobs/".len] = shard0;
    out["blobs/".len + 1] = shard1;
    // Intentionally use '/' for the same canonical-path reason as above.
    out["blobs/".len + 2] = '/';
    @memcpy(out["blobs/".len + 3 ..][0..".tmp-".len], ".tmp-");

    var nonce: [8]u8 = undefined;
    std.crypto.random.bytes(&nonce);
    for (nonce, 0..) |byte, idx| {
        const base = "blobs/".len + 3 + ".tmp-".len + idx * 2;
        out[base] = hex_chars[byte >> 4];
        out[base + 1] = hex_chars[byte & 0x0f];
    }

    return out;
}

fn resolveMultipartChunkSizeBytes() usize {
    const env_ptr = std.posix.getenv(env_multipart_chunk_size_bytes) orelse return default_multipart_chunk_size_bytes;
    const raw = std.mem.sliceTo(env_ptr, 0);
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    if (trimmed.len == 0) return default_multipart_chunk_size_bytes;
    return std.fmt.parseUnsigned(usize, trimmed, 10) catch default_multipart_chunk_size_bytes;
}

test "BlobStore.init opens db root" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var store = try BlobStore.init(std.testing.allocator, root_path);
    defer store.deinit();
}

test "BlobStore.deinit closes root dir" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var store = try BlobStore.init(std.testing.allocator, root_path);
    store.deinit();
}

test "BlobStore.put deduplicates identical payloads" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var store = try BlobStore.init(std.testing.allocator, root_path);
    defer store.deinit();

    const first = try store.put("same-content");
    const second = try store.put("same-content");

    try std.testing.expectEqualStrings(first.refSlice(), second.refSlice());
    try std.testing.expectEqualStrings(first.rel_path[0..], second.rel_path[0..]);

    const stored = try tmp.dir.readFileAlloc(std.testing.allocator, first.rel_path[0..], 1024);
    defer std.testing.allocator.free(stored);
    try std.testing.expectEqualStrings("same-content", stored);
}

test "BlobStore.open returns readable file for reference" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var store = try BlobStore.init(std.testing.allocator, root_path);
    defer store.deinit();

    const blob_ref = try store.put("open-test");
    var file = try store.open(blob_ref.refSlice());
    defer file.close();

    var buf: [16]u8 = undefined;
    const read_len = try file.readAll(&buf);
    try std.testing.expectEqual(@as(usize, "open-test".len), read_len);
    try std.testing.expectEqualStrings("open-test", buf[0..read_len]);

    try std.testing.expectError(error.InvalidBlobRef, store.open("bad-ref"));
}

test "BlobStore.readAll reads full blob bytes" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var store = try BlobStore.init(std.testing.allocator, root_path);
    defer store.deinit();

    const blob_ref = try store.put("read-all-payload");
    const payload = try store.readAll(blob_ref.refSlice(), std.testing.allocator);
    defer std.testing.allocator.free(payload);

    try std.testing.expectEqualStrings("read-all-payload", payload);
}

test "BlobStore.verify validates and detects tampering" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var store = try BlobStore.init(std.testing.allocator, root_path);
    defer store.deinit();

    const blob_ref = try store.put("verify-me");
    try store.verify(blob_ref.refSlice());

    var file = try tmp.dir.openFile(blob_ref.rel_path[0..], .{ .mode = .read_write });
    defer file.close();
    try file.pwriteAll("X", 0);
    try file.sync();

    try std.testing.expectError(error.IntegrityMismatch, store.verify(blob_ref.refSlice()));
}

test "BlobStore.exists returns true for stored refs and false for missing refs" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var store = try BlobStore.init(std.testing.allocator, root_path);
    defer store.deinit();

    const payload = "exists-check-payload";
    const blob_ref = try store.put(payload);
    try std.testing.expect(try store.exists(blob_ref.refSlice()));

    var missing_digest = std.mem.zeroes([digest_hex_len]u8);
    @memset(missing_digest[0..], '0');
    const missing_ref = buildSha256RefString(missing_digest);
    try std.testing.expect(!(try store.exists(missing_ref[0..sha256_ref_len])));
}

test "BlobStore.listSingleDigests returns sorted digest list" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var store = try BlobStore.init(std.testing.allocator, root_path);
    defer store.deinit();

    const blob_a = try store.put("alpha");
    const blob_b = try store.put("beta");
    const blob_c = try store.put("gamma");

    const digests = try store.listSingleDigests(std.testing.allocator, 0);
    defer std.testing.allocator.free(digests);
    try std.testing.expect(digests.len >= 3);

    var expected = std.ArrayList([digest_hex_len]u8).empty;
    defer expected.deinit(std.testing.allocator);
    try expected.append(std.testing.allocator, (try parseRef(blob_a.refSlice())).digest_hex);
    try expected.append(std.testing.allocator, (try parseRef(blob_b.refSlice())).digest_hex);
    try expected.append(std.testing.allocator, (try parseRef(blob_c.refSlice())).digest_hex);

    for (expected.items) |digest| {
        var found = false;
        for (digests) |listed| {
            if (std.mem.eql(u8, digest[0..], listed[0..])) {
                found = true;
                break;
            }
        }
        try std.testing.expect(found);
    }

    for (digests[1..], 1..) |digest, idx| {
        try std.testing.expect(!std.mem.lessThan(u8, digest[0..], digests[idx - 1][0..]));
    }
}

test "BlobStore.putAuto creates multipart ref for large payloads and readAll reassembles" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var store = try BlobStore.init(std.testing.allocator, root_path);
    defer store.deinit();
    store.multipart_chunk_size_bytes = 16;

    const large_payload =
        "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ--multipart-check";
    const blob_ref = try store.putAuto(large_payload);
    try std.testing.expect(std.mem.startsWith(u8, blob_ref.refSlice(), multipart_ref_prefix));

    const loaded = try store.readAll(blob_ref.refSlice(), std.testing.allocator);
    defer std.testing.allocator.free(loaded);
    try std.testing.expectEqualStrings(large_payload, loaded);
}

test "BlobStore.openReadStream streams single blob incrementally" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var store = try BlobStore.init(std.testing.allocator, root_path);
    defer store.deinit();

    const payload = "stream-single-payload";
    const blob_ref = try store.put(payload);
    var stream = try store.openReadStream(std.testing.allocator, blob_ref.refSlice());
    defer stream.deinit();

    var out = std.ArrayList(u8).empty;
    defer out.deinit(std.testing.allocator);

    var buf: [5]u8 = undefined; // Scratch buffer, overwritten by stream.read each iteration
    while (true) {
        const read_len = try stream.read(&buf);
        if (read_len == 0) break;
        try out.appendSlice(std.testing.allocator, buf[0..read_len]);
    }

    try std.testing.expectEqualStrings(payload, out.items);
    try std.testing.expectEqual(@as(u64, payload.len), stream.bytes_read);
    try std.testing.expectEqual(@as(u64, payload.len), stream.total_size);
}

test "BlobStore.openReadStream streams multipart blob incrementally" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var store = try BlobStore.init(std.testing.allocator, root_path);
    defer store.deinit();
    store.multipart_chunk_size_bytes = 12;

    const payload = "stream-multipart-payload-with-more-than-one-chunk";
    const blob_ref = try store.putAuto(payload);
    try std.testing.expect(std.mem.startsWith(u8, blob_ref.refSlice(), multipart_ref_prefix));

    var stream = try store.openReadStream(std.testing.allocator, blob_ref.refSlice());
    defer stream.deinit();

    var out = std.ArrayList(u8).empty;
    defer out.deinit(std.testing.allocator);

    var buf: [7]u8 = undefined; // Scratch buffer, overwritten by stream.read each iteration
    while (true) {
        const read_len = try stream.read(&buf);
        if (read_len == 0) break;
        try out.appendSlice(std.testing.allocator, buf[0..read_len]);
    }

    try std.testing.expectEqualStrings(payload, out.items);
    try std.testing.expectEqual(@as(u64, payload.len), stream.bytes_read);
    try std.testing.expectEqual(@as(u64, payload.len), stream.total_size);
}

test "BlobReadStream.seek repositions single blob reads" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var store = try BlobStore.init(std.testing.allocator, root_path);
    defer store.deinit();

    const payload = "0123456789abcdef";
    const blob_ref = try store.put(payload);
    var stream = try store.openReadStream(std.testing.allocator, blob_ref.refSlice());
    defer stream.deinit();

    try stream.seek(4);
    var out = std.ArrayList(u8).empty;
    defer out.deinit(std.testing.allocator);
    var buf: [6]u8 = undefined; // Scratch buffer, overwritten by stream.read each iteration
    while (true) {
        const read_len = try stream.read(&buf);
        if (read_len == 0) break;
        try out.appendSlice(std.testing.allocator, buf[0..read_len]);
    }
    try std.testing.expectEqualStrings(payload[4..], out.items);

    try stream.seek(0);
    var prefix_buf: [3]u8 = undefined; // Overwritten by stream.read on the next line
    const prefix_len = try stream.read(&prefix_buf);
    try std.testing.expectEqual(@as(usize, 3), prefix_len);
    try std.testing.expectEqualStrings("012", prefix_buf[0..prefix_len]);

    try stream.seek(payload.len);
    const eof_len = try stream.read(&prefix_buf);
    try std.testing.expectEqual(@as(usize, 0), eof_len);
    try std.testing.expectError(error.InvalidSeekOffset, stream.seek(payload.len + 1));
}

test "BlobReadStream.seek repositions multipart blob reads" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var store = try BlobStore.init(std.testing.allocator, root_path);
    defer store.deinit();
    store.multipart_chunk_size_bytes = 8;

    const payload = "multipart-seek-payload-check";
    const blob_ref = try store.putAuto(payload);
    try std.testing.expect(std.mem.startsWith(u8, blob_ref.refSlice(), multipart_ref_prefix));

    var stream = try store.openReadStream(std.testing.allocator, blob_ref.refSlice());
    defer stream.deinit();

    try stream.seek(10);
    var out = std.ArrayList(u8).empty;
    defer out.deinit(std.testing.allocator);
    var buf: [5]u8 = undefined; // Scratch buffer, overwritten by stream.read each iteration
    while (true) {
        const read_len = try stream.read(&buf);
        if (read_len == 0) break;
        try out.appendSlice(std.testing.allocator, buf[0..read_len]);
    }
    try std.testing.expectEqualStrings(payload[10..], out.items);

    try stream.seek(8);
    var boundary_buf: [4]u8 = undefined; // Overwritten by stream.read on the next line
    const boundary_len = try stream.read(&boundary_buf);
    try std.testing.expectEqual(@as(usize, 4), boundary_len);
    try std.testing.expectEqualStrings(payload[8..12], boundary_buf[0..boundary_len]);
}

test "BlobPutStream.finish stores small payload as single blob" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var store = try BlobStore.init(std.testing.allocator, root_path);
    defer store.deinit();
    store.multipart_chunk_size_bytes = 16;

    var writer = BlobPutStream.init(&store, std.testing.allocator);
    defer writer.deinit();

    try writer.write("tiny");
    const blob_ref = try writer.finish();
    try std.testing.expect(std.mem.startsWith(u8, blob_ref.refSlice(), sha256_ref_prefix));

    const loaded = try store.readAll(blob_ref.refSlice(), std.testing.allocator);
    defer std.testing.allocator.free(loaded);
    try std.testing.expectEqualStrings("tiny", loaded);
}

test "BlobPutStream.finish stores large payload as multipart blob" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var store = try BlobStore.init(std.testing.allocator, root_path);
    defer store.deinit();
    store.multipart_chunk_size_bytes = 12;

    var writer = BlobPutStream.init(&store, std.testing.allocator);
    defer writer.deinit();

    try writer.write("streaming-");
    try writer.write("multipart-");
    try writer.write("payload");

    const blob_ref = try writer.finish();
    try std.testing.expect(std.mem.startsWith(u8, blob_ref.refSlice(), multipart_ref_prefix));

    const loaded = try store.readAll(blob_ref.refSlice(), std.testing.allocator);
    defer std.testing.allocator.free(loaded);
    try std.testing.expectEqualStrings("streaming-multipart-payload", loaded);
}

test "parseRef accepts sha256 and multipart prefixes" {
    var digest = std.mem.zeroes([digest_hex_len]u8);
    @memset(digest[0..], 'a');

    const sha_ref = buildSha256RefString(digest);
    const parsed_sha = try parseRef(sha_ref[0..sha256_ref_len]);
    try std.testing.expectEqual(RefKind.sha256, parsed_sha.kind);

    const multi_ref = buildMultipartRefString(digest);
    const parsed_multi = try parseRef(multi_ref[0..multipart_ref_len]);
    try std.testing.expectEqual(RefKind.multipart, parsed_multi.kind);
}
