//! Integration tests for db.BlobStore
//!
//! BlobStore provides content-addressable storage for large payloads,
//! supporting both single (sha256:) and multipart (multi:) blob references.

const std = @import("std");
const main = @import("main");
const db = main.db;

const BlobStore = db.BlobStore;
const blob_mod = db.blob;
const store_mod = blob_mod.store;

// =============================================================================
// init / deinit
// =============================================================================

test "BlobStore.init creates store and deinit releases it" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var store = try BlobStore.init(std.testing.allocator, root_path);
    store.deinit();
}

// =============================================================================
// put
// =============================================================================

test "BlobStore.put stores content and deduplicates identical payloads" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var store = try BlobStore.init(std.testing.allocator, root_path);
    defer store.deinit();

    const first = try store.put("dedup-payload");
    const second = try store.put("dedup-payload");
    try std.testing.expectEqualStrings(first.refSlice(), second.refSlice());
    try std.testing.expectEqual(first.size_bytes, second.size_bytes);

    // Different content yields different ref
    const third = try store.put("other-payload");
    try std.testing.expect(!std.mem.eql(u8, first.refSlice(), third.refSlice()));
}

// =============================================================================
// readAll
// =============================================================================

test "BlobStore.readAll round-trips stored content" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var store = try BlobStore.init(std.testing.allocator, root_path);
    defer store.deinit();

    const payload = "round-trip-payload-content";
    const blob_ref = try store.put(payload);
    const loaded = try store.readAll(blob_ref.refSlice(), std.testing.allocator);
    defer std.testing.allocator.free(loaded);

    try std.testing.expectEqualStrings(payload, loaded);
}

// =============================================================================
// open
// =============================================================================

test "BlobStore.open returns readable file and rejects invalid refs" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var store = try BlobStore.init(std.testing.allocator, root_path);
    defer store.deinit();

    const blob_ref = try store.put("open-test-content");
    var file = try store.open(blob_ref.refSlice());
    defer file.close();

    var buf: [64]u8 = undefined;
    const read_len = try file.readAll(&buf);
    try std.testing.expectEqualStrings("open-test-content", buf[0..read_len]);

    try std.testing.expectError(error.InvalidBlobRef, store.open("bad-ref"));
}

// =============================================================================
// verify
// =============================================================================

test "BlobStore.verify detects tampered content" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var store = try BlobStore.init(std.testing.allocator, root_path);
    defer store.deinit();

    const blob_ref = try store.put("verify-content");
    try store.verify(blob_ref.refSlice());

    // Tamper with the file
    var file = try tmp.dir.openFile(blob_ref.rel_path[0..], .{ .mode = .read_write });
    defer file.close();
    try file.pwriteAll("X", 0);
    try file.sync();

    try std.testing.expectError(error.IntegrityMismatch, store.verify(blob_ref.refSlice()));
}

// =============================================================================
// exists
// =============================================================================

test "BlobStore.exists returns true for stored and false for missing refs" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var store = try BlobStore.init(std.testing.allocator, root_path);
    defer store.deinit();

    const blob_ref = try store.put("exists-content");
    try std.testing.expect(try store.exists(blob_ref.refSlice()));

    var missing_digest = std.mem.zeroes([64]u8);
    @memset(missing_digest[0..], '0');
    const missing_ref = store_mod.buildSha256RefString(missing_digest);
    try std.testing.expect(!(try store.exists(missing_ref[0..store_mod.sha256_ref_len])));
}

// =============================================================================
// listSingleDigests
// =============================================================================

test "BlobStore.listSingleDigests returns sorted digest list" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var store = try BlobStore.init(std.testing.allocator, root_path);
    defer store.deinit();

    const a = try store.put("alpha");
    const b = try store.put("beta");

    const digests = try store.listSingleDigests(std.testing.allocator, 0);
    defer std.testing.allocator.free(digests);
    try std.testing.expect(digests.len >= 2);

    // Verify both stored digests are present
    const parsed_a = try store_mod.parseRef(a.refSlice());
    const parsed_b = try store_mod.parseRef(b.refSlice());
    var found_a = false;
    var found_b = false;
    for (digests) |d| {
        if (std.mem.eql(u8, d[0..], parsed_a.digest_hex[0..])) found_a = true;
        if (std.mem.eql(u8, d[0..], parsed_b.digest_hex[0..])) found_b = true;
    }
    try std.testing.expect(found_a);
    try std.testing.expect(found_b);

    // Verify sorted order
    for (digests[1..], 1..) |d, idx| {
        try std.testing.expect(!std.mem.lessThan(u8, d[0..], digests[idx - 1][0..]));
    }
}

// =============================================================================
// putAuto / putMultipart
// =============================================================================

test "BlobStore.putAuto creates multipart ref for large payloads and readAll reassembles" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var store = try BlobStore.init(std.testing.allocator, root_path);
    defer store.deinit();
    store.multipart_chunk_size_bytes = 16;

    const payload = "abcdefghijklmnopqrstuvwxyz0123456789-multipart-auto";
    const blob_ref = try store.putAuto(payload);
    try std.testing.expect(std.mem.startsWith(u8, blob_ref.refSlice(), store_mod.multipart_ref_prefix));

    const loaded = try store.readAll(blob_ref.refSlice(), std.testing.allocator);
    defer std.testing.allocator.free(loaded);
    try std.testing.expectEqualStrings(payload, loaded);
}

test "BlobStore.putMultipart stores and reassembles chunks" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var store = try BlobStore.init(std.testing.allocator, root_path);
    defer store.deinit();
    store.multipart_chunk_size_bytes = 16;

    const payload = "abcdefghijklmnopqrstuvwxyz0123456789-explicit-multipart";
    const blob_ref = try store.putMultipart(payload);
    try std.testing.expect(std.mem.startsWith(u8, blob_ref.refSlice(), store_mod.multipart_ref_prefix));

    const loaded = try store.readAll(blob_ref.refSlice(), std.testing.allocator);
    defer std.testing.allocator.free(loaded);
    try std.testing.expectEqualStrings(payload, loaded);
}

// =============================================================================
// openReadStream
// =============================================================================

test "BlobStore.openReadStream streams single blob incrementally" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var store = try BlobStore.init(std.testing.allocator, root_path);
    defer store.deinit();

    const payload = "stream-single-integration-test";
    const blob_ref = try store.put(payload);
    var stream = try store.openReadStream(std.testing.allocator, blob_ref.refSlice());
    defer stream.deinit();

    var out = std.ArrayList(u8).empty;
    defer out.deinit(std.testing.allocator);

    var buf: [5]u8 = undefined;
    while (true) {
        const read_len = try stream.read(&buf);
        if (read_len == 0) break;
        try out.appendSlice(std.testing.allocator, buf[0..read_len]);
    }

    try std.testing.expectEqualStrings(payload, out.items);
    try std.testing.expectEqual(@as(u64, payload.len), stream.bytes_read);
}
