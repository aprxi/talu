//! Integration tests for db.adapters.vector.SearchResult
//!
//! SearchResult is the return type of VectorAdapter.search().
//! It owns allocated ids and scores slices, freed via deinit.

const std = @import("std");
const main = @import("main");
const db = main.db;

const VectorAdapter = db.adapters.vector.VectorAdapter;
const SearchResult = db.adapters.vector.SearchResult;

// ===== deinit =====

test "SearchResult: deinit frees search results" {
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

    // Search and verify SearchResult, then deinit
    {
        var adapter = try VectorAdapter.init(std.testing.allocator, root_path);
        defer adapter.deinit();

        const query = [_]f32{ 1.0, 0.0 };
        var result = try adapter.search(std.testing.allocator, &query, 2);

        try std.testing.expectEqual(@as(usize, 2), result.ids.len);
        try std.testing.expectEqual(@as(usize, 2), result.scores.len);
        // Ranked: doc_id=1 (dot=1.0) first, doc_id=3 (dot=0.7) second
        try std.testing.expectEqual(@as(u64, 1), result.ids[0]);
        try std.testing.expectEqual(@as(f32, 1.0), result.scores[0]);
        try std.testing.expectEqual(@as(u64, 3), result.ids[1]);
        try std.testing.expectApproxEqAbs(@as(f32, 0.7), result.scores[1], 1e-6);

        // deinit frees ids and scores; std.testing.allocator catches leaks.
        result.deinit(std.testing.allocator);
    }
}

test "SearchResult: deinit handles k=0 empty result" {
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

        const query = [_]f32{ 1.0, 0.0 };
        // k=0: returns empty result
        var result = try adapter.search(std.testing.allocator, &query, 0);

        try std.testing.expectEqual(@as(usize, 0), result.ids.len);
        try std.testing.expectEqual(@as(usize, 0), result.scores.len);

        result.deinit(std.testing.allocator);
    }
}
