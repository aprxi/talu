//! Approximate IVF-Flat index with exact-fallback guarantees.
//!
//! This backend builds a transient coarse partition over in-memory vectors.
//! If probing does not yield enough candidates for `top_k`, it falls back to
//! exact scan semantics so the caller still gets deterministic top-k results.

const std = @import("std");
const dot_product = @import("../../../compute/cpu/linalg.zig").dot;
const flat_mod = @import("flat.zig");

const Allocator = std.mem.Allocator;

pub const IvfFlatIndex = struct {
    pub fn searchBatch(
        _: IvfFlatIndex,
        allocator: Allocator,
        ids: []const u64,
        vectors: []const f32,
        input_dims: u32,
        queries: []const f32,
        query_count: u32,
        k: u32,
        dims: u32,
    ) !flat_mod.FlatIndex.SearchBatchResult {
        // Degenerate cases are delegated to the exact backend.
        if (ids.len <= 512) {
            var flat_index = flat_mod.FlatIndex{};
            return flat_index.searchBatch(
                allocator,
                ids,
                vectors,
                input_dims,
                queries,
                query_count,
                k,
                dims,
            );
        }

        if (query_count == 0 or k == 0) {
            return .{
                .ids = try allocator.alloc(u64, 0),
                .scores = try allocator.alloc(f32, 0),
                .count_per_query = 0,
                .query_count = query_count,
            };
        }
        if (dims == 0 or input_dims == 0) return error.InvalidColumnData;
        if (dims != input_dims) return error.InvalidColumnData;
        if (queries.len != @as(usize, dims) * @as(usize, query_count)) return error.InvalidColumnData;
        if (ids.len * @as(usize, input_dims) != vectors.len) return error.InvalidColumnData;

        const row_count = ids.len;
        const dims_usize = @as(usize, dims);
        const nlist: usize = @min(@as(usize, 64), row_count);
        const probes: usize = @min(@as(usize, 8), nlist);
        const per_query = @min(@as(usize, k), row_count);

        var inverted_lists = try allocator.alloc(std.ArrayListUnmanaged(u32), nlist);
        defer {
            for (inverted_lists) |*list| list.deinit(allocator);
            allocator.free(inverted_lists);
        }
        for (inverted_lists) |*list| {
            list.* = .empty;
        }

        // Assignment pass: each row goes to the closest centroid.
        for (0..row_count) |row_idx| {
            const base = row_idx * dims_usize;
            const row = vectors[base .. base + dims_usize];
            var best_centroid: usize = 0;
            var best_score: f32 = -std.math.inf(f32);
            for (0..nlist) |ci| {
                const cbase = ci * dims_usize;
                const centroid = vectors[cbase .. cbase + dims_usize];
                const score = dot_product.dotProductF32(row, centroid);
                if (score > best_score) {
                    best_score = score;
                    best_centroid = ci;
                }
            }
            try inverted_lists[best_centroid].append(allocator, @intCast(row_idx));
        }

        const total_out = per_query * @as(usize, query_count);
        const ids_out = try allocator.alloc(u64, total_out);
        errdefer allocator.free(ids_out);
        const scores_out = try allocator.alloc(f32, total_out);
        errdefer allocator.free(scores_out);

        var centroid_scores = try allocator.alloc(CentroidScore, nlist);
        defer allocator.free(centroid_scores);

        var qi: usize = 0;
        while (qi < query_count) : (qi += 1) {
            const query = queries[qi * dims_usize .. (qi + 1) * dims_usize];

            for (0..nlist) |ci| {
                const cbase = ci * dims_usize;
                const centroid = vectors[cbase .. cbase + dims_usize];
                centroid_scores[ci] = .{
                    .score = dot_product.dotProductF32(query, centroid),
                    .centroid_idx = ci,
                };
            }
            std.sort.pdq(CentroidScore, centroid_scores, {}, lessCentroidScore);

            var candidate_count: usize = 0;
            for (0..probes) |pi| {
                candidate_count += inverted_lists[centroid_scores[pi].centroid_idx].items.len;
            }

            // Guarantee exact top-k semantics via flat fallback when approximate
            // probing cannot produce enough candidates.
            if (candidate_count < per_query) {
                var flat_index = flat_mod.FlatIndex{};
                var flat_result = try flat_index.searchBatch(
                    allocator,
                    ids,
                    vectors,
                    input_dims,
                    query,
                    1,
                    @intCast(per_query),
                    dims,
                );
                defer flat_result.deinit(allocator);

                const out_off = qi * per_query;
                std.mem.copyForwards(u64, ids_out[out_off .. out_off + per_query], flat_result.ids);
                std.mem.copyForwards(f32, scores_out[out_off .. out_off + per_query], flat_result.scores);
                continue;
            }

            var heap = try MinHeap.init(allocator, per_query);
            defer heap.deinit(allocator);

            for (0..probes) |pi| {
                const list = inverted_lists[centroid_scores[pi].centroid_idx].items;
                for (list) |row_idx_u32| {
                    const row_idx = @as(usize, row_idx_u32);
                    const base = row_idx * dims_usize;
                    const row = vectors[base .. base + dims_usize];
                    const score = dot_product.dotProductF32(query, row);
                    heap.push(score, ids[row_idx]);
                }
            }

            const out_off = qi * per_query;
            try heap.writeSorted(
                allocator,
                ids_out[out_off .. out_off + per_query],
                scores_out[out_off .. out_off + per_query],
            );
        }

        return .{
            .ids = ids_out,
            .scores = scores_out,
            .count_per_query = @intCast(per_query),
            .query_count = query_count,
        };
    }
};

const CentroidScore = struct {
    score: f32,
    centroid_idx: usize,
};

fn lessCentroidScore(_: void, a: CentroidScore, b: CentroidScore) bool {
    if (a.score > b.score) return true;
    if (a.score < b.score) return false;
    return a.centroid_idx < b.centroid_idx;
}

const SearchEntry = struct {
    score: f32,
    id: u64,
};

const MinHeap = struct {
    entries: []SearchEntry,
    size: usize,

    fn init(allocator: Allocator, capacity: usize) !MinHeap {
        if (capacity == 0) return error.InvalidColumnData;
        const entries = try allocator.alloc(SearchEntry, capacity);
        return .{ .entries = entries, .size = 0 };
    }

    fn deinit(self: *MinHeap, allocator: Allocator) void {
        allocator.free(self.entries);
    }

    fn push(self: *MinHeap, score: f32, id: u64) void {
        const candidate = SearchEntry{ .score = score, .id = id };
        if (self.size < self.entries.len) {
            self.entries[self.size] = candidate;
            self.siftUp(self.size);
            self.size += 1;
            return;
        }
        if (!betterThan(candidate, self.entries[0])) return;
        self.entries[0] = candidate;
        self.siftDown(0);
    }

    fn siftUp(self: *MinHeap, start_idx: usize) void {
        var idx = start_idx;
        while (idx > 0) {
            const parent = (idx - 1) / 2;
            if (!betterThan(self.entries[parent], self.entries[idx])) break;
            std.mem.swap(SearchEntry, &self.entries[parent], &self.entries[idx]);
            idx = parent;
        }
    }

    fn siftDown(self: *MinHeap, start_idx: usize) void {
        var idx = start_idx;
        const size = self.size;
        while (true) {
            const left = idx * 2 + 1;
            if (left >= size) break;
            const right = left + 1;
            var smallest = left;
            if (right < size and betterThan(self.entries[left], self.entries[right])) {
                smallest = right;
            }
            if (!betterThan(self.entries[idx], self.entries[smallest])) break;
            std.mem.swap(SearchEntry, &self.entries[idx], &self.entries[smallest]);
            idx = smallest;
        }
    }

    fn writeSorted(self: *MinHeap, allocator: Allocator, ids: []u64, scores: []f32) !void {
        if (ids.len != self.size or scores.len != self.size) return error.InvalidColumnData;
        const items = try allocator.alloc(SearchEntry, self.size);
        defer allocator.free(items);
        std.mem.copyForwards(SearchEntry, items, self.entries[0..self.size]);

        std.sort.pdq(SearchEntry, items, {}, struct {
            fn lessThan(_: void, a: SearchEntry, b: SearchEntry) bool {
                return betterThan(a, b);
            }
        }.lessThan);

        for (items, 0..) |entry, idx| {
            ids[idx] = entry.id;
            scores[idx] = entry.score;
        }
    }
};

fn betterThan(a: SearchEntry, b: SearchEntry) bool {
    if (a.score > b.score) return true;
    if (a.score < b.score) return false;
    return a.id < b.id;
}

test "IvfFlatIndex.searchBatch falls back to exact for small datasets" {
    const ids = [_]u64{ 1, 2, 3 };
    const vectors = [_]f32{
        1.0, 0.0,
        0.8, 0.0,
        0.1, 0.0,
    };
    const queries = [_]f32{ 1.0, 0.0 };

    var index = IvfFlatIndex{};
    var result = try index.searchBatch(
        std.testing.allocator,
        &ids,
        &vectors,
        2,
        &queries,
        1,
        2,
        2,
    );
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u32, 2), result.count_per_query);
    try std.testing.expectEqual(@as(usize, 2), result.ids.len);
    try std.testing.expectEqual(@as(u64, 1), result.ids[0]);
    try std.testing.expectEqual(@as(u64, 2), result.ids[1]);
}
