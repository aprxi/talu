//! Integration tests for db.VectorAdapter
//!
//! VectorAdapter provides batch append and bulk load for embedding vectors,
//! plus top-k search and streaming score callbacks.

const std = @import("std");
const main = @import("main");
const db = main.db;

const VectorAdapter = db.adapters.vector.VectorAdapter;
const VectorBatch = db.adapters.vector.VectorBatch;
const SearchBatchResult = db.adapters.vector.SearchBatchResult;

// ===== init / deinit =====

test "VectorAdapter: init opens vector namespace" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var adapter = try VectorAdapter.init(std.testing.allocator, root_path);
    defer adapter.deinit();

    // Verify init created the namespace: vector/ directory with lock, data, and WAL files.
    var lock_file = try tmp.dir.openFile("vector/talu.lock", .{});
    lock_file.close();
    var data_file = try tmp.dir.openFile("vector/current.talu", .{});
    data_file.close();
    var wal_file = try tmp.dir.openFile("vector/current.wal", .{});
    wal_file.close();
}

// ===== appendBatch =====

test "VectorAdapter: appendBatch stores vectors" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var adapter = try VectorAdapter.init(std.testing.allocator, root_path);
    defer adapter.deinit();

    const doc_ids = [_]u64{ 1, 2 };
    const vectors = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    try adapter.appendBatch(&doc_ids, &vectors, 2);

    // Verify data was persisted: WAL should contain the appended batch.
    const wal_stat = try tmp.dir.statFile("vector/current.wal");
    try std.testing.expect(wal_stat.size > 0);
}

// ===== loadVectors =====

test "VectorAdapter: appendBatch and loadVectors roundtrip" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    {
        var adapter = try VectorAdapter.init(std.testing.allocator, root_path);
        defer adapter.deinit();

        const doc_ids = [_]u64{ 10, 20, 30 };
        const vectors = [_]f32{ 1.0, 0.0, 0.0, 1.0, 0.5, 0.5 };
        try adapter.appendBatch(&doc_ids, &vectors, 2);
        try adapter.fs_writer.flushBlock();
    }

    {
        var adapter = try VectorAdapter.init(std.testing.allocator, root_path);
        defer adapter.deinit();

        var batch = try adapter.loadVectors(std.testing.allocator);
        defer batch.deinit(std.testing.allocator);

        try std.testing.expectEqual(@as(usize, 3), batch.ids.len);
        try std.testing.expectEqual(@as(u32, 2), batch.dims);
        try std.testing.expectEqual(@as(u64, 10), batch.ids[0]);
        try std.testing.expectEqual(@as(u64, 20), batch.ids[1]);
        try std.testing.expectEqual(@as(u64, 30), batch.ids[2]);

        // Verify vector data roundtripped correctly
        try std.testing.expectEqual(@as(usize, 6), batch.vectors.len);
        try std.testing.expectEqual(@as(f32, 1.0), batch.vectors[0]);
        try std.testing.expectEqual(@as(f32, 0.0), batch.vectors[1]);
        try std.testing.expectEqual(@as(f32, 0.0), batch.vectors[2]);
        try std.testing.expectEqual(@as(f32, 1.0), batch.vectors[3]);
        try std.testing.expectEqual(@as(f32, 0.5), batch.vectors[4]);
        try std.testing.expectEqual(@as(f32, 0.5), batch.vectors[5]);
    }
}

// ===== search =====

test "VectorAdapter: search returns top-k results" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    {
        var adapter = try VectorAdapter.init(std.testing.allocator, root_path);
        defer adapter.deinit();

        // 3 vectors of dim=2
        const doc_ids = [_]u64{ 1, 2, 3 };
        const vectors = [_]f32{ 1.0, 0.0, 0.0, 1.0, 0.7, 0.7 };
        try adapter.appendBatch(&doc_ids, &vectors, 2);
        try adapter.fs_writer.flushBlock();
    }

    {
        var adapter = try VectorAdapter.init(std.testing.allocator, root_path);
        defer adapter.deinit();

        const query = [_]f32{ 1.0, 0.0 };
        var result = try adapter.search(std.testing.allocator, &query, 2);
        defer result.deinit(std.testing.allocator);

        try std.testing.expectEqual(@as(usize, 2), result.ids.len);
        try std.testing.expectEqual(@as(usize, 2), result.scores.len);
        // Best match for [1,0] should be doc_id=1 (dot product = 1.0)
        try std.testing.expectEqual(@as(u64, 1), result.ids[0]);
        try std.testing.expectEqual(@as(f32, 1.0), result.scores[0]);
        // Second match is doc_id=3 (dot product = 0.7)
        try std.testing.expectEqual(@as(u64, 3), result.ids[1]);
        try std.testing.expectApproxEqAbs(@as(f32, 0.7), result.scores[1], 1e-6);
    }
}

// ===== searchScores =====

test "VectorAdapter: searchScores streams all scores via callback" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    {
        var adapter = try VectorAdapter.init(std.testing.allocator, root_path);
        defer adapter.deinit();

        const doc_ids = [_]u64{ 100, 200 };
        const vectors = [_]f32{ 1.0, 0.0, 0.5, 0.5 };
        try adapter.appendBatch(&doc_ids, &vectors, 2);
        try adapter.fs_writer.flushBlock();
    }

    {
        var adapter = try VectorAdapter.init(std.testing.allocator, root_path);
        defer adapter.deinit();

        const query = [_]f32{ 1.0, 0.0 };

        const Counter = struct {
            count: usize = 0,
            fn callback(ctx: *anyopaque, _: u64, _: f32) void {
                const self: *@This() = @ptrCast(@alignCast(ctx));
                self.count += 1;
            }
        };

        var counter = Counter{};
        try adapter.searchScores(std.testing.allocator, &query, &counter, Counter.callback);

        try std.testing.expectEqual(@as(usize, 2), counter.count);
    }
}

// ===== countEmbeddingRows =====

test "VectorAdapter: countEmbeddingRows returns total row count" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    {
        var adapter = try VectorAdapter.init(std.testing.allocator, root_path);
        defer adapter.deinit();

        const doc_ids = [_]u64{ 1, 2, 3 };
        const vectors = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
        try adapter.appendBatch(&doc_ids, &vectors, 2);
        try adapter.fs_writer.flushBlock();
    }

    {
        var adapter = try VectorAdapter.init(std.testing.allocator, root_path);
        defer adapter.deinit();

        const count = try adapter.countEmbeddingRows(std.testing.allocator, 2);
        try std.testing.expectEqual(@as(usize, 3), count);
    }
}

// ===== searchBatch =====

test "VectorAdapter: searchBatch returns ranked results per query" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    {
        var adapter = try VectorAdapter.init(std.testing.allocator, root_path);
        defer adapter.deinit();

        // 3 vectors of dim=2: [1,0], [0,1], [0.7,0.7]
        const doc_ids = [_]u64{ 1, 2, 3 };
        const vectors = [_]f32{ 1.0, 0.0, 0.0, 1.0, 0.7, 0.7 };
        try adapter.appendBatch(&doc_ids, &vectors, 2);
        try adapter.fs_writer.flushBlock();
    }

    {
        var adapter = try VectorAdapter.init(std.testing.allocator, root_path);
        defer adapter.deinit();

        // Two queries: [1,0] and [0,1], top-2 each
        const queries = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
        var result = try adapter.searchBatch(std.testing.allocator, &queries, 2, 2, 2);
        defer result.deinit(std.testing.allocator);

        try std.testing.expectEqual(@as(u32, 2), result.query_count);
        try std.testing.expectEqual(@as(u32, 2), result.count_per_query);
        try std.testing.expectEqual(@as(usize, 4), result.ids.len);
        try std.testing.expectEqual(@as(usize, 4), result.scores.len);

        // Query [1,0]: best match is doc_id=1 (dot=1.0), second is doc_id=3 (dot=0.7)
        try std.testing.expectEqual(@as(u64, 1), result.ids[0]);
        try std.testing.expectEqual(@as(f32, 1.0), result.scores[0]);
        try std.testing.expectEqual(@as(u64, 3), result.ids[1]);
        try std.testing.expectApproxEqAbs(@as(f32, 0.7), result.scores[1], 1e-6);

        // Query [0,1]: best match is doc_id=2 (dot=1.0), second is doc_id=3 (dot=0.7)
        try std.testing.expectEqual(@as(u64, 2), result.ids[2]);
        try std.testing.expectEqual(@as(f32, 1.0), result.scores[2]);
        try std.testing.expectEqual(@as(u64, 3), result.ids[3]);
        try std.testing.expectApproxEqAbs(@as(f32, 0.7), result.scores[3], 1e-6);
    }
}
