//! Persisted IVF-Flat index backend with exact fallback guarantees.

const std = @import("std");
const dot_product = @import("../../../compute/cpu/linalg.zig").dot;
const vector_filter = @import("../filter.zig");
const flat_mod = @import("flat.zig");

const Allocator = std.mem.Allocator;
const ivf_magic = [_]u8{ 'T', 'A', 'L', 'U', 'I', 'V', 'F', '1' };
const ivf_version: u32 = 1;
const max_training_rows: usize = 4096;
const kmeans_iterations: usize = 6;

pub const IvfSegmentIndex = struct {
    dims: u32,
    nlist: u32,
    row_count: u64,
    centroids: []f32,
    list_offsets: []u64,
    list_ids: []u64,

    pub fn deinit(self: *IvfSegmentIndex, allocator: Allocator) void {
        allocator.free(self.centroids);
        allocator.free(self.list_offsets);
        allocator.free(self.list_ids);
        self.* = .{
            .dims = 0,
            .nlist = 0,
            .row_count = 0,
            .centroids = &[_]f32{},
            .list_offsets = &[_]u64{},
            .list_ids = &[_]u64{},
        };
    }
};

pub fn buildSegmentIndex(
    allocator: Allocator,
    ids: []const u64,
    vectors: []const f32,
    dims: u32,
) !IvfSegmentIndex {
    if (dims == 0) return error.InvalidColumnData;
    const dims_usize = @as(usize, dims);
    if (vectors.len != ids.len * dims_usize) return error.InvalidColumnData;
    if (ids.len == 0) return error.InvalidColumnData;

    const row_count = ids.len;
    const nlist_usize = @min(@as(usize, 64), row_count);
    const nlist: u32 = @intCast(nlist_usize);

    const centroids = try allocator.alloc(f32, nlist_usize * dims_usize);
    errdefer allocator.free(centroids);
    try trainCentroids(
        allocator,
        vectors,
        dims_usize,
        row_count,
        centroids,
        nlist_usize,
    );

    var lists = try allocator.alloc(std.ArrayListUnmanaged(u64), nlist_usize);
    defer {
        for (lists) |*list| list.deinit(allocator);
        allocator.free(lists);
    }
    for (lists) |*list| list.* = .empty;

    for (0..row_count) |row_idx| {
        const row = vectors[row_idx * dims_usize .. (row_idx + 1) * dims_usize];
        var best_centroid: usize = 0;
        var best_score: f32 = -std.math.inf(f32);
        for (0..nlist_usize) |ci| {
            const centroid = centroids[ci * dims_usize .. (ci + 1) * dims_usize];
            const score = dot_product.dotProductF32(row, centroid);
            if (score > best_score) {
                best_score = score;
                best_centroid = ci;
            }
        }
        try lists[best_centroid].append(allocator, ids[row_idx]);
    }

    const offsets = try allocator.alloc(u64, nlist_usize + 1);
    errdefer allocator.free(offsets);
    offsets[0] = 0;
    for (0..nlist_usize) |ci| {
        offsets[ci + 1] = offsets[ci] + @as(u64, @intCast(lists[ci].items.len));
    }

    const list_ids = try allocator.alloc(u64, row_count);
    errdefer allocator.free(list_ids);
    var write_idx: usize = 0;
    for (lists) |list| {
        std.mem.copyForwards(u64, list_ids[write_idx .. write_idx + list.items.len], list.items);
        write_idx += list.items.len;
    }

    return .{
        .dims = dims,
        .nlist = nlist,
        .row_count = row_count,
        .centroids = centroids,
        .list_offsets = offsets,
        .list_ids = list_ids,
    };
}

pub fn encodeSegmentIndex(allocator: Allocator, index: IvfSegmentIndex) ![]u8 {
    if (index.dims == 0 or index.nlist == 0) return error.InvalidColumnData;
    const nlist_usize = @as(usize, index.nlist);
    const row_count_usize = try usizeFromU64(index.row_count);
    if (index.centroids.len != nlist_usize * @as(usize, index.dims)) return error.InvalidColumnData;
    if (index.list_offsets.len != nlist_usize + 1) return error.InvalidColumnData;
    if (index.list_ids.len != row_count_usize) return error.InvalidColumnData;
    if (index.list_offsets[index.list_offsets.len - 1] != index.row_count) return error.InvalidColumnData;

    var size: usize = 0;
    size = try addUsize(size, ivf_magic.len);
    size = try addUsize(size, @sizeOf(u32) * 4 + @sizeOf(u64));
    size = try addUsize(size, index.centroids.len * @sizeOf(f32));
    size = try addUsize(size, index.list_offsets.len * @sizeOf(u64));
    size = try addUsize(size, index.list_ids.len * @sizeOf(u64));

    const out = try allocator.alloc(u8, size);
    errdefer allocator.free(out);
    var stream = std.io.fixedBufferStream(out);
    const writer = stream.writer();

    try writer.writeAll(&ivf_magic);
    try writer.writeInt(u32, ivf_version, .little);
    try writer.writeInt(u32, index.dims, .little);
    try writer.writeInt(u32, index.nlist, .little);
    try writer.writeInt(u32, 0, .little);
    try writer.writeInt(u64, index.row_count, .little);
    try writer.writeAll(std.mem.sliceAsBytes(index.centroids));
    try writer.writeAll(std.mem.sliceAsBytes(index.list_offsets));
    try writer.writeAll(std.mem.sliceAsBytes(index.list_ids));
    return out;
}

pub fn decodeSegmentIndex(allocator: Allocator, bytes: []const u8) !IvfSegmentIndex {
    var at: usize = 0;
    if (bytes.len < ivf_magic.len) return error.InvalidColumnData;
    if (!std.mem.eql(u8, bytes[0..ivf_magic.len], &ivf_magic)) return error.InvalidColumnData;
    at += ivf_magic.len;

    const version = try readU32(bytes, &at);
    if (version != ivf_version) return error.InvalidColumnData;
    const dims = try readU32(bytes, &at);
    const nlist = try readU32(bytes, &at);
    _ = try readU32(bytes, &at); // reserved
    const row_count = try readU64(bytes, &at);
    if (dims == 0 or nlist == 0) return error.InvalidColumnData;

    const nlist_usize = @as(usize, nlist);
    const row_count_usize = try usizeFromU64(row_count);
    const centroid_count = try mulUsize(nlist_usize, @as(usize, dims));
    const centroids = try allocator.alloc(f32, centroid_count);
    errdefer allocator.free(centroids);
    const centroids_bytes = std.mem.sliceAsBytes(centroids);
    if (at + centroids_bytes.len > bytes.len) return error.InvalidColumnData;
    @memcpy(centroids_bytes, bytes[at .. at + centroids_bytes.len]);
    at += centroids_bytes.len;

    const offsets = try allocator.alloc(u64, nlist_usize + 1);
    errdefer allocator.free(offsets);
    const offsets_bytes = std.mem.sliceAsBytes(offsets);
    if (at + offsets_bytes.len > bytes.len) return error.InvalidColumnData;
    @memcpy(offsets_bytes, bytes[at .. at + offsets_bytes.len]);
    at += offsets_bytes.len;

    const list_ids = try allocator.alloc(u64, row_count_usize);
    errdefer allocator.free(list_ids);
    const ids_bytes = std.mem.sliceAsBytes(list_ids);
    if (at + ids_bytes.len > bytes.len) return error.InvalidColumnData;
    @memcpy(ids_bytes, bytes[at .. at + ids_bytes.len]);
    at += ids_bytes.len;

    if (at != bytes.len) return error.InvalidColumnData;
    if (offsets[0] != 0) return error.InvalidColumnData;
    if (offsets[offsets.len - 1] != row_count) return error.InvalidColumnData;
    for (1..offsets.len) |idx| {
        if (offsets[idx] < offsets[idx - 1]) return error.InvalidColumnData;
    }

    return .{
        .dims = dims,
        .nlist = nlist,
        .row_count = row_count,
        .centroids = centroids,
        .list_offsets = offsets,
        .list_ids = list_ids,
    };
}

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
        allow_list: ?*const vector_filter.AllowList,
        allowed_count: ?usize,
        segment_indexes: ?[]const IvfSegmentIndex,
        id_to_row: ?*const std.AutoHashMap(u64, usize),
    ) !flat_mod.FlatIndex.SearchBatchResult {
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
        if (ids.len == 0) {
            return .{
                .ids = try allocator.alloc(u64, 0),
                .scores = try allocator.alloc(f32, 0),
                .count_per_query = 0,
                .query_count = query_count,
            };
        }
        if (allow_list) |allow| {
            if (allow.len != ids.len) return error.InvalidColumnData;
        }

        const candidate_count = if (allow_list != null)
            allowed_count orelse vector_filter.allowListCount(allow_list.?)
        else
            ids.len;
        if (candidate_count > ids.len) return error.InvalidColumnData;
        if (candidate_count == 0) {
            return .{
                .ids = try allocator.alloc(u64, 0),
                .scores = try allocator.alloc(f32, 0),
                .count_per_query = 0,
                .query_count = query_count,
            };
        }

        const indexes = segment_indexes orelse {
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
                allow_list,
                allowed_count,
            );
        };
        if (indexes.len == 0) {
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
                allow_list,
                allowed_count,
            );
        }

        for (indexes) |index| {
            if (index.dims != dims) return error.InvalidColumnData;
            if (index.nlist == 0) return error.InvalidColumnData;
            if (index.list_offsets.len != @as(usize, index.nlist) + 1) return error.InvalidColumnData;
            if (index.list_offsets[index.list_offsets.len - 1] != index.row_count) return error.InvalidColumnData;
            if (index.list_ids.len != try usizeFromU64(index.row_count)) return error.InvalidColumnData;
            if (index.centroids.len != @as(usize, index.nlist) * @as(usize, dims)) return error.InvalidColumnData;
        }

        const dims_usize = @as(usize, dims);
        const per_query = @min(@as(usize, k), candidate_count);
        const total_out = per_query * @as(usize, query_count);
        const ids_out = try allocator.alloc(u64, total_out);
        errdefer allocator.free(ids_out);
        const scores_out = try allocator.alloc(f32, total_out);
        errdefer allocator.free(scores_out);

        var local_id_to_row = std.AutoHashMap(u64, usize).init(allocator);
        defer local_id_to_row.deinit();
        const id_to_row_map: *const std.AutoHashMap(u64, usize) = if (id_to_row) |external|
            external
        else blk: {
            try local_id_to_row.ensureTotalCapacity(@intCast(ids.len));
            for (ids, 0..) |id, row_idx| {
                try local_id_to_row.put(id, row_idx);
            }
            break :blk &local_id_to_row;
        };

        const row_seen = try allocator.alloc(u32, ids.len);
        defer allocator.free(row_seen);
        @memset(row_seen, 0);
        var seen_epoch: u32 = 1;

        var centroid_scores = std.ArrayList(CentroidScore).empty;
        defer centroid_scores.deinit(allocator);

        var flat_index = flat_mod.FlatIndex{};
        var qi: usize = 0;
        while (qi < query_count) : (qi += 1) {
            var heap = try MinHeap.init(allocator, per_query);
            defer heap.deinit(allocator);
            if (seen_epoch == std.math.maxInt(u32)) {
                @memset(row_seen, 0);
                seen_epoch = 1;
            }
            const current_epoch = seen_epoch;
            seen_epoch += 1;

            const query = queries[qi * dims_usize .. (qi + 1) * dims_usize];
            for (indexes) |index| {
                const nlist_usize = @as(usize, index.nlist);
                try centroid_scores.resize(allocator, nlist_usize);
                for (0..nlist_usize) |ci| {
                    const centroid = index.centroids[ci * dims_usize .. (ci + 1) * dims_usize];
                    centroid_scores.items[ci] = .{
                        .score = dot_product.dotProductF32(query, centroid),
                        .centroid_idx = ci,
                    };
                }
                std.sort.pdq(CentroidScore, centroid_scores.items, {}, lessCentroidScore);

                const probes = chooseProbeCount(index, centroid_scores.items, per_query, candidate_count, ids.len);
                for (0..probes) |pi| {
                    const centroid_idx = centroid_scores.items[pi].centroid_idx;
                    const start = try usizeFromU64(index.list_offsets[centroid_idx]);
                    const end = try usizeFromU64(index.list_offsets[centroid_idx + 1]);
                    for (index.list_ids[start..end]) |id| {
                        const row_idx = id_to_row_map.get(id) orelse continue;
                        if (row_idx >= row_seen.len) continue;
                        if (row_seen[row_idx] == current_epoch) continue;
                        row_seen[row_idx] = current_epoch;
                        if (allow_list) |allow| {
                            if (!vector_filter.allowListContains(allow, row_idx)) continue;
                        }
                        const row = vectors[row_idx * dims_usize .. (row_idx + 1) * dims_usize];
                        const score = dot_product.dotProductF32(query, row);
                        heap.push(score, id);
                    }
                }
            }

            if (heap.size < per_query) {
                var flat = try flat_index.searchBatch(
                    allocator,
                    ids,
                    vectors,
                    input_dims,
                    query,
                    1,
                    @intCast(per_query),
                    dims,
                    allow_list,
                    allowed_count,
                );
                defer flat.deinit(allocator);
                const out_off = qi * per_query;
                std.mem.copyForwards(u64, ids_out[out_off .. out_off + per_query], flat.ids);
                std.mem.copyForwards(f32, scores_out[out_off .. out_off + per_query], flat.scores);
                continue;
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

fn trainCentroids(
    allocator: Allocator,
    vectors: []const f32,
    dims: usize,
    row_count: usize,
    centroids: []f32,
    nlist: usize,
) !void {
    const sample_count = @min(row_count, max_training_rows);
    const sample_rows = try allocator.alloc(usize, sample_count);
    defer allocator.free(sample_rows);
    buildUniformSampleRows(sample_rows, row_count);

    for (0..nlist) |ci| {
        const sample_idx = (ci * sample_count) / nlist;
        const row_idx = sample_rows[sample_idx];
        std.mem.copyForwards(
            f32,
            centroids[ci * dims .. (ci + 1) * dims],
            vectors[row_idx * dims .. (row_idx + 1) * dims],
        );
    }
    if (sample_count <= nlist) return;

    const sums = try allocator.alloc(f64, nlist * dims);
    defer allocator.free(sums);
    const counts = try allocator.alloc(u32, nlist);
    defer allocator.free(counts);

    for (0..kmeans_iterations) |iter| {
        @memset(sums, 0);
        @memset(counts, 0);

        for (sample_rows) |row_idx| {
            const row = vectors[row_idx * dims .. (row_idx + 1) * dims];
            var best_centroid: usize = 0;
            var best_score: f32 = -std.math.inf(f32);
            for (0..nlist) |ci| {
                const centroid = centroids[ci * dims .. (ci + 1) * dims];
                const score = dot_product.dotProductF32(row, centroid);
                if (score > best_score) {
                    best_score = score;
                    best_centroid = ci;
                }
            }
            counts[best_centroid] += 1;
            for (0..dims) |d| {
                sums[best_centroid * dims + d] += @as(f64, row[d]);
            }
        }

        for (0..nlist) |ci| {
            if (counts[ci] == 0) {
                const replacement = sample_rows[(iter + ci) % sample_count];
                std.mem.copyForwards(
                    f32,
                    centroids[ci * dims .. (ci + 1) * dims],
                    vectors[replacement * dims .. (replacement + 1) * dims],
                );
                continue;
            }
            const inv_count = 1.0 / @as(f64, @floatFromInt(counts[ci]));
            for (0..dims) |d| {
                centroids[ci * dims + d] = @floatCast(sums[ci * dims + d] * inv_count);
            }
        }
    }
}

fn buildUniformSampleRows(out_rows: []usize, row_count: usize) void {
    if (out_rows.len == 0) return;
    if (out_rows.len == 1) {
        out_rows[0] = 0;
        return;
    }

    var previous: usize = 0;
    for (0..out_rows.len) |idx| {
        const numer = idx * (row_count - 1);
        const denom = out_rows.len - 1;
        var row = numer / denom;
        if (idx > 0 and row <= previous and previous + 1 < row_count) {
            row = previous + 1;
        }
        out_rows[idx] = row;
        previous = row;
    }
}

fn chooseProbeCount(
    index: IvfSegmentIndex,
    ranked_centroids: []const CentroidScore,
    per_query: usize,
    candidate_count: usize,
    total_count: usize,
) usize {
    if (ranked_centroids.len == 0) return 0;
    const nlist = ranked_centroids.len;
    const selectivity = if (total_count == 0)
        1.0
    else
        @as(f64, @floatFromInt(candidate_count)) / @as(f64, @floatFromInt(total_count));
    const multiplier: usize = blk: {
        if (selectivity >= 0.5) break :blk 2;
        if (selectivity >= 0.2) break :blk 3;
        if (selectivity >= 0.1) break :blk 4;
        if (selectivity >= 0.05) break :blk 6;
        break :blk 8;
    };
    const target_candidates = blk: {
        const base = std.math.mul(usize, per_query, multiplier) catch std.math.maxInt(usize);
        const max_rows = usizeFromU64(index.row_count) catch per_query;
        break :blk @max(@as(usize, 1), @min(base, max_rows));
    };

    var probes: usize = 0;
    var candidates: usize = 0;
    while (probes < nlist) : (probes += 1) {
        const centroid_idx = ranked_centroids[probes].centroid_idx;
        const start = usizeFromU64(index.list_offsets[centroid_idx]) catch continue;
        const end = usizeFromU64(index.list_offsets[centroid_idx + 1]) catch continue;
        candidates += end - start;
        if (candidates >= target_candidates and probes + 1 >= @min(@as(usize, 2), nlist)) break;
    }

    const min_probe = @min(@as(usize, 2), nlist);
    const chosen = @max(min_probe, probes + 1);
    return @min(chosen, nlist);
}

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

fn usizeFromU64(value: u64) !usize {
    if (value > std.math.maxInt(usize)) return error.InvalidColumnData;
    return @intCast(value);
}

fn mulUsize(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch return error.InvalidColumnData;
}

fn addUsize(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch return error.InvalidColumnData;
}

fn readU32(bytes: []const u8, at: *usize) !u32 {
    if (at.* + @sizeOf(u32) > bytes.len) return error.InvalidColumnData;
    const value = std.mem.readInt(u32, bytes[at.*..][0..4], .little);
    at.* += 4;
    return value;
}

fn readU64(bytes: []const u8, at: *usize) !u64 {
    if (at.* + @sizeOf(u64) > bytes.len) return error.InvalidColumnData;
    const value = std.mem.readInt(u64, bytes[at.*..][0..8], .little);
    at.* += 8;
    return value;
}

test "buildSegmentIndex and encode/decode round trip" {
    const ids = [_]u64{ 1, 2, 3, 4 };
    const vectors = [_]f32{
        1.0, 0.0,
        0.9, 0.0,
        0.0, 1.0,
        0.0, 0.9,
    };

    var built = try buildSegmentIndex(std.testing.allocator, &ids, &vectors, 2);
    defer built.deinit(std.testing.allocator);
    const encoded = try encodeSegmentIndex(std.testing.allocator, built);
    defer std.testing.allocator.free(encoded);

    var decoded = try decodeSegmentIndex(std.testing.allocator, encoded);
    defer decoded.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u32, 2), decoded.dims);
    try std.testing.expectEqual(@as(u64, 4), decoded.row_count);
    try std.testing.expectEqual(@as(usize, 4), decoded.list_ids.len);
    try std.testing.expectEqual(@as(u64, 4), decoded.list_offsets[decoded.list_offsets.len - 1]);
}

test "IvfFlatIndex.searchBatch uses segment indexes when available" {
    const ids = [_]u64{ 1, 2, 3 };
    const vectors = [_]f32{
        1.0, 0.0,
        0.8, 0.0,
        0.1, 0.0,
    };
    const queries = [_]f32{ 1.0, 0.0 };

    var seg_index = try buildSegmentIndex(std.testing.allocator, &ids, &vectors, 2);
    defer seg_index.deinit(std.testing.allocator);
    const indexes = [_]IvfSegmentIndex{seg_index};

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
        null,
        null,
        &indexes,
        null,
    );
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u32, 2), result.count_per_query);
    try std.testing.expectEqual(@as(usize, 2), result.ids.len);
    try std.testing.expectEqual(@as(u64, 1), result.ids[0]);
}

test "buildUniformSampleRows produces monotonic bounded rows" {
    var rows: [11]usize = undefined;
    buildUniformSampleRows(&rows, 100);
    try std.testing.expectEqual(@as(usize, 0), rows[0]);
    try std.testing.expectEqual(@as(usize, 99), rows[rows.len - 1]);
    for (1..rows.len) |idx| {
        try std.testing.expect(rows[idx] > rows[idx - 1]);
        try std.testing.expect(rows[idx] < 100);
    }
}

test "chooseProbeCount scales with filter selectivity target" {
    var offsets = [_]u64{ 0, 40, 70, 90, 100 };
    var empty_centroids: [0]f32 = .{};
    var empty_ids: [0]u64 = .{};
    const index = IvfSegmentIndex{
        .dims = 2,
        .nlist = 4,
        .row_count = 100,
        .centroids = empty_centroids[0..],
        .list_offsets = offsets[0..],
        .list_ids = empty_ids[0..],
    };
    const ranked = [_]CentroidScore{
        .{ .score = 1.0, .centroid_idx = 0 },
        .{ .score = 0.9, .centroid_idx = 1 },
        .{ .score = 0.8, .centroid_idx = 2 },
        .{ .score = 0.7, .centroid_idx = 3 },
    };

    const probes_no_filter = chooseProbeCount(index, &ranked, 10, 100, 100);
    const probes_with_filter = chooseProbeCount(index, &ranked, 10, 5, 100);
    try std.testing.expect(probes_with_filter >= probes_no_filter);
    try std.testing.expect(probes_no_filter >= 2);
    try std.testing.expect(probes_with_filter <= ranked.len);
}
