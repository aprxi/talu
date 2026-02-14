//! Content-addressable blob storage for large payloads.
//!
//! Stores immutable blobs under:
//!   blobs/<first-two-hex>/<full-64-hex-sha256>
//!
//! References are encoded as:
//!   sha256:<64-hex>

const std = @import("std");

const Allocator = std.mem.Allocator;
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const ref_prefix = "sha256:";
pub const digest_hex_len: usize = 64;
pub const ref_len: usize = ref_prefix.len + digest_hex_len;
pub const rel_path_len: usize = "blobs/".len + 2 + 1 + digest_hex_len;

const tmp_suffix_hex_len: usize = 16;
const tmp_path_len: usize = "blobs/".len + 2 + 1 + ".tmp-".len + tmp_suffix_hex_len;
const hex_chars = "0123456789abcdef";

pub const BlobRef = struct {
    ref: [ref_len]u8,
    rel_path: [rel_path_len]u8,
    size_bytes: u64,
};

pub const BlobStore = struct {
    allocator: Allocator,
    root_dir: std.fs.Dir,

    pub fn init(allocator: Allocator, db_root: []const u8) !BlobStore {
        const root_dir = try std.fs.cwd().makeOpenPath(db_root, .{});
        return .{
            .allocator = allocator,
            .root_dir = root_dir,
        };
    }

    pub fn deinit(self: *BlobStore) void {
        self.root_dir.close();
    }

    pub fn put(self: *BlobStore, bytes: []const u8) !BlobRef {
        var digest: [32]u8 = undefined;
        Sha256.hash(bytes, &digest, .{});
        const digest_hex = digestToHex(digest);

        const blob_ref = BlobRef{
            .ref = buildRefString(digest_hex),
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

    pub fn open(self: *BlobStore, blob_ref: []const u8) !std.fs.File {
        const digest_hex = try parseBlobRef(blob_ref);
        const rel_path = buildRelativePath(digest_hex);
        return self.root_dir.openFile(rel_path[0..], .{ .mode = .read_only });
    }

    pub fn readAll(self: *BlobStore, blob_ref: []const u8, allocator: Allocator) ![]u8 {
        var file = try self.open(blob_ref);
        defer file.close();

        const stat = try file.stat();
        if (stat.size > std.math.maxInt(usize)) return error.FileTooLarge;
        return file.readToEndAlloc(allocator, @intCast(stat.size));
    }

    pub fn verify(self: *BlobStore, blob_ref: []const u8) !void {
        const expected_hex = try parseBlobRef(blob_ref);

        var file = try self.open(blob_ref);
        defer file.close();

        var hasher = Sha256.init(.{});
        var buf: [16 * 1024]u8 = undefined;
        while (true) {
            const read_len = try file.read(&buf);
            if (read_len == 0) break;
            hasher.update(buf[0..read_len]);
        }

        var digest: [32]u8 = undefined;
        hasher.final(&digest);
        const actual_hex = digestToHex(digest);
        if (!std.mem.eql(u8, expected_hex[0..], actual_hex[0..])) {
            return error.IntegrityMismatch;
        }
    }

    fn ensureShardDir(self: *BlobStore, shard0: u8, shard1: u8) !void {
        var shard_path: [8]u8 = undefined; // "blobs/aa"
        @memcpy(shard_path[0.."blobs/".len], "blobs/");
        shard_path["blobs/".len] = shard0;
        shard_path["blobs/".len + 1] = shard1;
        try self.root_dir.makePath(shard_path[0..]);
    }
};

fn parseBlobRef(blob_ref: []const u8) ![digest_hex_len]u8 {
    if (blob_ref.len != ref_len) return error.InvalidBlobRef;
    if (!std.mem.startsWith(u8, blob_ref, ref_prefix)) return error.InvalidBlobRef;

    var digest_hex: [digest_hex_len]u8 = undefined;
    for (blob_ref[ref_prefix.len..], 0..) |ch, idx| {
        if (!std.ascii.isHex(ch)) return error.InvalidBlobRef;
        digest_hex[idx] = std.ascii.toLower(ch);
    }

    return digest_hex;
}

fn digestToHex(digest: [32]u8) [digest_hex_len]u8 {
    var out: [digest_hex_len]u8 = undefined;
    for (digest, 0..) |byte, idx| {
        out[idx * 2] = hex_chars[byte >> 4];
        out[idx * 2 + 1] = hex_chars[byte & 0x0f];
    }
    return out;
}

fn buildRefString(digest_hex: [digest_hex_len]u8) [ref_len]u8 {
    var out: [ref_len]u8 = undefined;
    @memcpy(out[0..ref_prefix.len], ref_prefix);
    @memcpy(out[ref_prefix.len..], digest_hex[0..]);
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

    try std.testing.expectEqualStrings(first.ref[0..], second.ref[0..]);
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
    var file = try store.open(blob_ref.ref[0..]);
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
    const payload = try store.readAll(blob_ref.ref[0..], std.testing.allocator);
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
    try store.verify(blob_ref.ref[0..]);

    var file = try tmp.dir.openFile(blob_ref.rel_path[0..], .{ .mode = .read_write });
    defer file.close();
    try file.pwriteAll("X", 0);
    try file.sync();

    try std.testing.expectError(error.IntegrityMismatch, store.verify(blob_ref.ref[0..]));
}
