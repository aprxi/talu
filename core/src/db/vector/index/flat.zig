//! Exact flat index implementation (dot-product scan).

const std = @import("std");
const dot_product = @import("../../../compute/cpu/linalg.zig").dot;
const parallel = @import("../../../system/parallel.zig");

const Allocator = std.mem.Allocator;

pub const FlatIndex = struct {
    pub const SearchBatchResult = struct {
        ids: []u64,
        scores: []f32,
        count_per_query: u32,
        query_count: u32,

        pub fn deinit(self: *SearchBatchResult, allocator: Allocator) void {
            allocator.free(self.ids);
            allocator.free(self.scores);
        }
    };

    pub fn searchBatch(
        _: FlatIndex,
        allocator: Allocator,
        ids: []const u64,
        vectors: []const f32,
        input_dims: u32,
        queries: []const f32,
        query_count: u32,
        k: u32,
        dims: u32,
    ) !SearchBatchResult {
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
        if (row_count == 0) {
            return .{
                .ids = try allocator.alloc(u64, 0),
                .scores = try allocator.alloc(f32, 0),
                .count_per_query = 0,
                .query_count = query_count,
            };
        }

        const per_query_capacity = @min(@as(usize, k), row_count);
        var heaps = try allocator.alloc(MinHeap, query_count);
        errdefer {
            for (heaps) |*heap| heap.deinit(allocator);
            allocator.free(heaps);
        }
        for (heaps) |*heap| {
            heap.* = try MinHeap.init(allocator, per_query_capacity);
        }

        const all_scores = try allocator.alloc(f32, row_count * @as(usize, query_count));
        defer allocator.free(all_scores);
        parallelScoreBatchRows(
            queries,
            vectors,
            @as(usize, dims),
            query_count,
            row_count,
            row_count,
            0,
            all_scores,
        );

        for (0..row_count) |row_idx| {
            for (0..@as(usize, query_count)) |query_idx| {
                heaps[query_idx].push(
                    all_scores[query_idx * row_count + row_idx],
                    ids[row_idx],
                );
            }
        }

        const count_per_query: u32 = @intCast(heaps[0].size);
        const total = @as(usize, count_per_query) * @as(usize, query_count);
        const ids_out = try allocator.alloc(u64, total);
        errdefer allocator.free(ids_out);
        const scores_out = try allocator.alloc(f32, total);
        errdefer allocator.free(scores_out);

        var query_idx: usize = 0;
        while (query_idx < query_count) : (query_idx += 1) {
            const offset = query_idx * @as(usize, count_per_query);
            const ids_slice = ids_out[offset .. offset + @as(usize, count_per_query)];
            const scores_slice = scores_out[offset .. offset + @as(usize, count_per_query)];
            try heaps[query_idx].writeSorted(allocator, ids_slice, scores_slice);
        }

        for (heaps) |*heap| heap.deinit(allocator);
        allocator.free(heaps);

        return .{
            .ids = ids_out,
            .scores = scores_out,
            .count_per_query = count_per_query,
            .query_count = query_count,
        };
    }
};

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

const BatchScoreScanCtx = struct {
    queries: []const f32,
    vectors: []const f32,
    dims: usize,
    query_count: u32,
    total_rows: usize,
    row_offset: usize,
    scores_out: []f32,
};

fn batchScoreScanWorker(start: usize, end: usize, ctx: *BatchScoreScanCtx) void {
    const dims = ctx.dims;
    const query_count = @as(usize, ctx.query_count);
    for (start..end) |row_idx| {
        const base = row_idx * dims;
        const row_slice = ctx.vectors[base .. base + dims];

        var qi: usize = 0;
        while (qi + 4 <= query_count) : (qi += 4) {
            const q0 = ctx.queries[(qi + 0) * dims .. (qi + 1) * dims];
            const q1 = ctx.queries[(qi + 1) * dims .. (qi + 2) * dims];
            const q2 = ctx.queries[(qi + 2) * dims .. (qi + 3) * dims];
            const q3 = ctx.queries[(qi + 3) * dims .. (qi + 4) * dims];
            const scores = dot_product.dotProductF32Batch4(q0, q1, q2, q3, row_slice);
            const global_row = ctx.row_offset + row_idx;
            ctx.scores_out[(qi + 0) * ctx.total_rows + global_row] = scores[0];
            ctx.scores_out[(qi + 1) * ctx.total_rows + global_row] = scores[1];
            ctx.scores_out[(qi + 2) * ctx.total_rows + global_row] = scores[2];
            ctx.scores_out[(qi + 3) * ctx.total_rows + global_row] = scores[3];
        }
        while (qi < query_count) : (qi += 1) {
            const q_base = qi * dims;
            const score = dot_product.dotProductF32(ctx.queries[q_base .. q_base + dims], row_slice);
            ctx.scores_out[qi * ctx.total_rows + ctx.row_offset + row_idx] = score;
        }
    }
}

fn parallelScoreBatchRows(
    queries: []const f32,
    vectors: []const f32,
    dims: usize,
    query_count: u32,
    row_count: usize,
    total_rows: usize,
    row_offset: usize,
    scores_out: []f32,
) void {
    var ctx = BatchScoreScanCtx{
        .queries = queries,
        .vectors = vectors,
        .dims = dims,
        .query_count = query_count,
        .total_rows = total_rows,
        .row_offset = row_offset,
        .scores_out = scores_out,
    };
    parallel.global().parallelFor(row_count, batchScoreScanWorker, &ctx);
}

test "FlatIndex.searchBatch returns deterministic score order" {
    const ids = [_]u64{ 2, 1 };
    const vectors = [_]f32{
        1.0, 0.0,
        1.0, 0.0,
    };
    const queries = [_]f32{
        1.0, 0.0,
    };

    var index = FlatIndex{};
    var result = try index.searchBatch(
        std.testing.allocator,
        &ids,
        &vectors,
        2,
        &queries,
        1,
        1,
        2,
    );
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 1), result.ids.len);
    try std.testing.expectEqual(@as(u64, 1), result.ids[0]);
}
