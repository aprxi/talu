//! Integration tests for db.BlobReadStream
//!
//! BlobReadStream provides incremental streaming reads over single or
//! multipart blob references, transparently handling part boundaries.

const std = @import("std");
const main = @import("main");
const db = main.db;

const BlobReadStream = db.BlobReadStream;
const BlobStore = db.BlobStore;
const store_mod = db.blob.store;

// =============================================================================
// read (single blob)
// =============================================================================

test "BlobReadStream.read streams single blob and tracks bytes_read" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var store = try BlobStore.init(std.testing.allocator, root_path);
    defer store.deinit();

    const payload = "single-stream-content";
    const blob_ref = try store.put(payload);
    var stream = try store.openReadStream(std.testing.allocator, blob_ref.refSlice());
    defer stream.deinit();

    try std.testing.expectEqual(@as(u64, payload.len), stream.total_size);
    try std.testing.expectEqual(@as(u64, 0), stream.bytes_read);

    var out = std.ArrayList(u8).empty;
    defer out.deinit(std.testing.allocator);

    var buf: [4]u8 = undefined;
    while (true) {
        const n = try stream.read(&buf);
        if (n == 0) break;
        try out.appendSlice(std.testing.allocator, buf[0..n]);
    }

    try std.testing.expectEqualStrings(payload, out.items);
    try std.testing.expectEqual(@as(u64, payload.len), stream.bytes_read);
}

test "BlobReadStream.read returns zero on empty buffer" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var store = try BlobStore.init(std.testing.allocator, root_path);
    defer store.deinit();

    const blob_ref = try store.put("some-data");
    var stream = try store.openReadStream(std.testing.allocator, blob_ref.refSlice());
    defer stream.deinit();

    var empty_buf: [0]u8 = undefined;
    const n = try stream.read(&empty_buf);
    try std.testing.expectEqual(@as(usize, 0), n);
}

// =============================================================================
// read (multipart blob)
// =============================================================================

test "BlobReadStream.read streams multipart blob across chunk boundaries" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var store = try BlobStore.init(std.testing.allocator, root_path);
    defer store.deinit();
    store.multipart_chunk_size_bytes = 12;

    const payload = "stream-multipart-across-multiple-chunk-boundaries";
    const blob_ref = try store.putAuto(payload);
    try std.testing.expect(std.mem.startsWith(u8, blob_ref.refSlice(), store_mod.multipart_ref_prefix));

    var stream = try store.openReadStream(std.testing.allocator, blob_ref.refSlice());
    defer stream.deinit();

    try std.testing.expectEqual(@as(u64, payload.len), stream.total_size);

    var out = std.ArrayList(u8).empty;
    defer out.deinit(std.testing.allocator);

    // Use a read buffer that doesn't align with chunk boundaries
    var buf: [7]u8 = undefined;
    while (true) {
        const n = try stream.read(&buf);
        if (n == 0) break;
        try out.appendSlice(std.testing.allocator, buf[0..n]);
    }

    try std.testing.expectEqualStrings(payload, out.items);
    try std.testing.expectEqual(@as(u64, payload.len), stream.bytes_read);
}

// =============================================================================
// deinit
// =============================================================================

test "BlobReadStream.deinit cleans up after partial read" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var store = try BlobStore.init(std.testing.allocator, root_path);
    defer store.deinit();

    const blob_ref = try store.put("partial-read-deinit-test");
    var stream = try store.openReadStream(std.testing.allocator, blob_ref.refSlice());

    // Read partially, then deinit — must not leak file handles or memory.
    var buf: [3]u8 = undefined;
    const n = try stream.read(&buf);
    try std.testing.expect(n > 0);
    stream.deinit();
}
