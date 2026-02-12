//! Integration tests for db.adapters.vector.VectorBatch
//!
//! VectorBatch is the return type of VectorAdapter.loadVectors().
//! It owns allocated ids and vectors slices, freed via deinit.

const std = @import("std");
const main = @import("main");
const db = main.db;

const VectorAdapter = db.adapters.vector.VectorAdapter;
const VectorBatch = db.adapters.vector.VectorBatch;

// ===== deinit =====

test "VectorBatch: deinit frees loaded ids and vectors" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    // Write vectors through the adapter
    {
        var adapter = try VectorAdapter.init(std.testing.allocator, root_path);
        defer adapter.deinit();

        const doc_ids = [_]u64{ 10, 20, 30 };
        const vectors = [_]f32{ 1.0, 0.0, 0.0, 1.0, 0.5, 0.5 };
        try adapter.appendBatch(&doc_ids, &vectors, 2);
        try adapter.fs_writer.flushBlock();
    }

    // Load into a VectorBatch, verify contents, then deinit
    {
        var adapter = try VectorAdapter.init(std.testing.allocator, root_path);
        defer adapter.deinit();

        var batch = try adapter.loadVectors(std.testing.allocator);

        try std.testing.expectEqual(@as(usize, 3), batch.ids.len);
        try std.testing.expectEqual(@as(u32, 2), batch.dims);
        try std.testing.expectEqual(@as(usize, 6), batch.vectors.len);
        try std.testing.expectEqual(@as(u64, 10), batch.ids[0]);
        try std.testing.expectEqual(@as(u64, 20), batch.ids[1]);
        try std.testing.expectEqual(@as(u64, 30), batch.ids[2]);
        try std.testing.expectEqual(@as(f32, 1.0), batch.vectors[0]);
        try std.testing.expectEqual(@as(f32, 0.0), batch.vectors[1]);

        // deinit frees ids and vectors; std.testing.allocator catches leaks.
        batch.deinit(std.testing.allocator);
    }
}

test "VectorBatch: deinit handles empty store" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var adapter = try VectorAdapter.init(std.testing.allocator, root_path);
    defer adapter.deinit();

    // Load from empty store — no vectors written
    var batch = try adapter.loadVectors(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 0), batch.ids.len);
    try std.testing.expectEqual(@as(usize, 0), batch.vectors.len);
    try std.testing.expectEqual(@as(u32, 0), batch.dims);

    batch.deinit(std.testing.allocator);
}
