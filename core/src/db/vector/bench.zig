//! Lightweight benchmark helpers for vector index backends.

const std = @import("std");
const flat_index = @import("index/flat.zig").FlatIndex;
const ivf_index = @import("index/ivfflat.zig").IvfFlatIndex;

const Allocator = std.mem.Allocator;

pub const BenchResult = struct {
    elapsed_ns: u64,
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
            var result = try index.searchBatch(allocator, ids, vectors, dims, queries, query_count, k, dims);
            defer result.deinit(allocator);
        },
        .ivf => {
            var index = ivf_index{};
            var result = try index.searchBatch(allocator, ids, vectors, dims, queries, query_count, k, dims);
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
