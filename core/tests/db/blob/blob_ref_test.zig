//! Integration tests for db.BlobRef
//!
//! BlobRef is a value type holding a content-addressable reference string
//! and the relative filesystem path to the blob file.

const std = @import("std");
const main = @import("main");
const db = main.db;

const BlobRef = db.BlobRef;
const BlobStore = db.BlobStore;
const store_mod = db.blob.store;

// =============================================================================
// refSlice
// =============================================================================

test "BlobRef.refSlice returns sha256-prefixed reference of correct length" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var store = try BlobStore.init(std.testing.allocator, root_path);
    defer store.deinit();

    const blob_ref = try store.put("ref-test-payload");
    const slice = blob_ref.refSlice();

    try std.testing.expect(std.mem.startsWith(u8, slice, store_mod.sha256_ref_prefix));
    try std.testing.expectEqual(store_mod.sha256_ref_len, slice.len);
}

test "BlobRef.refSlice is deterministic for identical content" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var store = try BlobStore.init(std.testing.allocator, root_path);
    defer store.deinit();

    const a = try store.put("deterministic-content");
    const b = try store.put("deterministic-content");
    try std.testing.expectEqualStrings(a.refSlice(), b.refSlice());
}

test "BlobRef.refSlice differs for different content" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var store = try BlobStore.init(std.testing.allocator, root_path);
    defer store.deinit();

    const a = try store.put("content-a");
    const b = try store.put("content-b");
    try std.testing.expect(!std.mem.eql(u8, a.refSlice(), b.refSlice()));
}

test "BlobRef size_bytes matches stored payload length" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var store = try BlobStore.init(std.testing.allocator, root_path);
    defer store.deinit();

    const payload = "size-check-payload";
    const blob_ref = try store.put(payload);
    try std.testing.expectEqual(@as(u64, payload.len), blob_ref.size_bytes);
}
