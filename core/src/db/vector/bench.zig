//! Lightweight benchmark helpers for vector index backends.

const std = @import("std");
const flat_index = @import("index/flat.zig").FlatIndex;
const ivf_mod = @import("index/ivfflat.zig");
const ivf_index = ivf_mod.IvfFlatIndex;

const Allocator = std.mem.Allocator;

pub const BenchResult = struct {
    elapsed_ns: u64,
    rows_scanned: usize,
    queries: usize,
};

pub const RecallBenchResult = struct {
    flat_elapsed_ns: u64,
    ivf_elapsed_ns: u64,
    recall_at_k: f32,
    rows_scanned: usize,
    queries: usize,
};

pub fn benchmarkFlatSearch(
    allocator: Allocator,
    row_count: usize,
    dims: u32,
    query_count: u32,
    k: u32,
) !BenchResult {
    return benchmarkIndex(allocator, .flat, row_count, dims, query_count, k);
}

pub fn benchmarkIvfSearch(
    allocator: Allocator,
    row_count: usize,
    dims: u32,
    query_count: u32,
    k: u32,
) !BenchResult {
    return benchmarkIndex(allocator, .ivf, row_count, dims, query_count, k);
}

pub fn benchmarkIvfRecall(
    allocator: Allocator,
    row_count: usize,
    dims: u32,
    query_count: u32,
    k: u32,
) !RecallBenchResult {
    if (row_count == 0 or dims == 0 or query_count == 0 or k == 0) return error.InvalidColumnData;

    const dims_usize = @as(usize, dims);
    const ids = try allocator.alloc(u64, row_count);
    defer allocator.free(ids);
    const vectors = try allocator.alloc(f32, row_count * dims_usize);
    defer allocator.free(vectors);
    const queries = try allocator.alloc(f32, @as(usize, query_count) * dims_usize);
    defer allocator.free(queries);

    for (0..row_count) |i| ids[i] = @intCast(i + 1);
    for (vectors, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i % 97))) / 97.0;
    for (queries, 0..) |*q, i| q.* = @as(f32, @floatFromInt((i % 31))) / 31.0;

    var flat = flat_index{};
    var timer_flat = try std.time.Timer.start();
    var flat_result = try flat.searchBatch(allocator, ids, vectors, dims, queries, query_count, k, dims, null, null);
    defer flat_result.deinit(allocator);
    const flat_elapsed = timer_flat.read();

    var ivf = ivf_index{};
    var segment_index = try ivf_mod.buildSegmentIndex(allocator, ids, vectors, dims);
    defer segment_index.deinit(allocator);
    const segment_indexes = [_]ivf_mod.IvfSegmentIndex{segment_index};
    var timer_ivf = try std.time.Timer.start();
    var ivf_result = try ivf.searchBatch(allocator, ids, vectors, dims, queries, query_count, k, dims, null, null, &segment_indexes, null);
    defer ivf_result.deinit(allocator);
    const ivf_elapsed = timer_ivf.read();

    const per_query = @as(usize, flat_result.count_per_query);
    if (per_query == 0) {
        return .{
            .flat_elapsed_ns = flat_elapsed,
            .ivf_elapsed_ns = ivf_elapsed,
            .recall_at_k = 1.0,
            .rows_scanned = row_count,
            .queries = query_count,
        };
    }
    var hit_count: usize = 0;
    for (0..@as(usize, query_count)) |qi| {
        const off = qi * per_query;
        const flat_ids = flat_result.ids[off .. off + per_query];
        const ivf_ids = ivf_result.ids[off .. off + per_query];
        for (ivf_ids) |id| {
            if (std.mem.indexOfScalar(u64, flat_ids, id) != null) hit_count += 1;
        }
    }
    const denom = @as(f32, @floatFromInt(per_query * @as(usize, query_count)));
    const recall = @as(f32, @floatFromInt(hit_count)) / denom;

    return .{
        .flat_elapsed_ns = flat_elapsed,
        .ivf_elapsed_ns = ivf_elapsed,
        .recall_at_k = recall,
        .rows_scanned = row_count,
        .queries = query_count,
    };
}

const Backend = enum {
    flat,
    ivf,
};

fn benchmarkIndex(
    allocator: Allocator,
    backend: Backend,
    row_count: usize,
    dims: u32,
    query_count: u32,
    k: u32,
) !BenchResult {
    if (row_count == 0 or dims == 0 or query_count == 0 or k == 0) return error.InvalidColumnData;

    const dims_usize = @as(usize, dims);
    const ids = try allocator.alloc(u64, row_count);
    defer allocator.free(ids);
    const vectors = try allocator.alloc(f32, row_count * dims_usize);
    defer allocator.free(vectors);
    const queries = try allocator.alloc(f32, @as(usize, query_count) * dims_usize);
    defer allocator.free(queries);

    for (0..row_count) |i| {
        ids[i] = @intCast(i + 1);
    }
    for (vectors, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt((i % 97))) / 97.0;
    }
    for (queries, 0..) |*q, i| {
        q.* = @as(f32, @floatFromInt((i % 31))) / 31.0;
    }

    var timer = try std.time.Timer.start();
    switch (backend) {
        .flat => {
            var index = flat_index{};
            var result = try index.searchBatch(allocator, ids, vectors, dims, queries, query_count, k, dims, null, null);
            defer result.deinit(allocator);
        },
        .ivf => {
            var index = ivf_index{};
            var segment_index = try ivf_mod.buildSegmentIndex(allocator, ids, vectors, dims);
            defer segment_index.deinit(allocator);
            const segment_indexes = [_]ivf_mod.IvfSegmentIndex{segment_index};
            var result = try index.searchBatch(allocator, ids, vectors, dims, queries, query_count, k, dims, null, null, &segment_indexes, null);
            defer result.deinit(allocator);
        },
    }

    return .{
        .elapsed_ns = timer.read(),
        .rows_scanned = row_count,
        .queries = query_count,
    };
}

test "benchmarkFlatSearch runs with valid synthetic input" {
    const result = try benchmarkFlatSearch(std.testing.allocator, 256, 8, 2, 5);
    try std.testing.expect(result.elapsed_ns > 0);
    try std.testing.expectEqual(@as(usize, 256), result.rows_scanned);
}

test "benchmarkIvfSearch runs with valid synthetic input" {
    const result = try benchmarkIvfSearch(std.testing.allocator, 1024, 8, 2, 5);
    try std.testing.expect(result.elapsed_ns > 0);
    try std.testing.expectEqual(@as(usize, 1024), result.rows_scanned);
}

test "benchmarkIvfRecall reports bounded recall metric" {
    const result = try benchmarkIvfRecall(std.testing.allocator, 2048, 8, 4, 8);
    try std.testing.expect(result.flat_elapsed_ns > 0);
    try std.testing.expect(result.ivf_elapsed_ns > 0);
    try std.testing.expect(result.recall_at_k >= 0.0);
    try std.testing.expect(result.recall_at_k <= 1.0);
    try std.testing.expectEqual(@as(usize, 2048), result.rows_scanned);
}
