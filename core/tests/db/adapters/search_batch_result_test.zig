//! Integration tests for db.adapters.vector.SearchBatchResult
//!
//! SearchBatchResult is the return type of VectorAdapter.searchBatch().
//! It owns allocated ids and scores slices, freed via deinit.

const std = @import("std");
const main = @import("main");
const db = main.db;

const VectorAdapter = db.adapters.vector.VectorAdapter;
const SearchBatchResult = db.adapters.vector.SearchBatchResult;

// ===== deinit =====

test "SearchBatchResult: deinit frees batch search results" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    // Write vectors
    {
        var adapter = try VectorAdapter.init(std.testing.allocator, root_path);
        defer adapter.deinit();

        const doc_ids = [_]u64{ 1, 2, 3 };
        const vectors = [_]f32{ 1.0, 0.0, 0.0, 1.0, 0.7, 0.7 };
        try adapter.appendBatch(&doc_ids, &vectors, 2);
        try adapter.fs_writer.flushBlock();
    }

    // Batch search with 2 queries, top-2 each, then deinit
    {
        var adapter = try VectorAdapter.init(std.testing.allocator, root_path);
        defer adapter.deinit();

        const queries = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
        var result = try adapter.searchBatch(std.testing.allocator, &queries, 2, 2, 2);

        try std.testing.expectEqual(@as(u32, 2), result.query_count);
        try std.testing.expectEqual(@as(u32, 2), result.count_per_query);
        try std.testing.expectEqual(@as(usize, 4), result.ids.len);
        try std.testing.expectEqual(@as(usize, 4), result.scores.len);

        // Query [1,0]: doc_id=1 (dot=1.0) ranked first, doc_id=3 (dot=0.7) second
        try std.testing.expectEqual(@as(u64, 1), result.ids[0]);
        try std.testing.expectEqual(@as(f32, 1.0), result.scores[0]);
        try std.testing.expectEqual(@as(u64, 3), result.ids[1]);
        try std.testing.expectApproxEqAbs(@as(f32, 0.7), result.scores[1], 1e-6);

        // Query [0,1]: doc_id=2 (dot=1.0) ranked first, doc_id=3 (dot=0.7) second
        try std.testing.expectEqual(@as(u64, 2), result.ids[2]);
        try std.testing.expectEqual(@as(f32, 1.0), result.scores[2]);
        try std.testing.expectEqual(@as(u64, 3), result.ids[3]);
        try std.testing.expectApproxEqAbs(@as(f32, 0.7), result.scores[3], 1e-6);

        // deinit frees ids and scores; std.testing.allocator catches leaks.
        result.deinit(std.testing.allocator);
    }
}

test "SearchBatchResult: deinit handles k=0 empty batch" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    {
        var adapter = try VectorAdapter.init(std.testing.allocator, root_path);
        defer adapter.deinit();

        const doc_ids = [_]u64{1};
        const vectors = [_]f32{ 1.0, 0.0 };
        try adapter.appendBatch(&doc_ids, &vectors, 2);
        try adapter.fs_writer.flushBlock();
    }

    {
        var adapter = try VectorAdapter.init(std.testing.allocator, root_path);
        defer adapter.deinit();

        const queries = [_]f32{ 1.0, 0.0 };
        // k=0: returns empty batch result
        var result = try adapter.searchBatch(std.testing.allocator, &queries, 2, 1, 0);

        try std.testing.expectEqual(@as(u32, 0), result.count_per_query);
        try std.testing.expectEqual(@as(u32, 1), result.query_count);
        try std.testing.expectEqual(@as(usize, 0), result.ids.len);
        try std.testing.expectEqual(@as(usize, 0), result.scores.len);

        result.deinit(std.testing.allocator);
    }
}
