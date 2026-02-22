//! Pluggable vector index interface.

const std = @import("std");
const flat_mod = @import("flat.zig");
const ivf_flat_mod = @import("ivfflat.zig");
const vector_filter = @import("../filter.zig");

const Allocator = std.mem.Allocator;

pub const QueryInput = struct {
    ids: []const u64,
    vectors: []const f32,
    dims: u32,
};

pub const IvfSegmentIndex = ivf_flat_mod.IvfSegmentIndex;

pub const SearchParams = struct {
    queries: []const f32,
    query_count: u32,
    k: u32,
    dims: u32,
    allow_list: ?*const vector_filter.AllowList = null,
    allowed_count: ?usize = null,
    segment_indexes: ?[]const ivf_flat_mod.IvfSegmentIndex = null,
    id_to_row: ?*const std.AutoHashMap(u64, usize) = null,
};

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

pub const VectorIndex = union(enum) {
    flat: flat_mod.FlatIndex,
    ivf_flat: ivf_flat_mod.IvfFlatIndex,

    pub fn searchBatch(
        self: VectorIndex,
        allocator: Allocator,
        input: QueryInput,
        params: SearchParams,
    ) !SearchBatchResult {
        return switch (self) {
            .flat => |index| blk: {
                const flat_result = try index.searchBatch(
                    allocator,
                    input.ids,
                    input.vectors,
                    input.dims,
                    params.queries,
                    params.query_count,
                    params.k,
                    params.dims,
                    params.allow_list,
                    params.allowed_count,
                );
                break :blk .{
                    .ids = flat_result.ids,
                    .scores = flat_result.scores,
                    .count_per_query = flat_result.count_per_query,
                    .query_count = flat_result.query_count,
                };
            },
            .ivf_flat => |index| blk: {
                const ivf_result = try index.searchBatch(
                    allocator,
                    input.ids,
                    input.vectors,
                    input.dims,
                    params.queries,
                    params.query_count,
                    params.k,
                    params.dims,
                    params.allow_list,
                    params.allowed_count,
                    params.segment_indexes,
                    params.id_to_row,
                );
                break :blk .{
                    .ids = ivf_result.ids,
                    .scores = ivf_result.scores,
                    .count_per_query = ivf_result.count_per_query,
                    .query_count = ivf_result.query_count,
                };
            },
        };
    }
};

test "VectorIndex.searchBatch dispatches to FlatIndex" {
    const ids = [_]u64{ 10, 11, 12 };
    const vectors = [_]f32{
        1.0, 0.0,
        0.5, 0.0,
        0.1, 0.0,
    };
    const queries = [_]f32{
        1.0, 0.0,
    };

    const index = VectorIndex{ .flat = .{} };
    var result = try index.searchBatch(
        std.testing.allocator,
        .{
            .ids = &ids,
            .vectors = &vectors,
            .dims = 2,
        },
        .{
            .queries = &queries,
            .query_count = 1,
            .k = 2,
            .dims = 2,
        },
    );
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u32, 2), result.count_per_query);
    try std.testing.expectEqual(@as(usize, 2), result.ids.len);
    try std.testing.expectEqual(@as(u64, 10), result.ids[0]);
    try std.testing.expectEqual(@as(u64, 11), result.ids[1]);
}

test "VectorIndex.searchBatch dispatches to IvfFlatIndex" {
    const ids = [_]u64{ 21, 22, 23 };
    const vectors = [_]f32{
        1.0, 0.0,
        0.8, 0.0,
        0.1, 0.0,
    };
    const queries = [_]f32{
        1.0, 0.0,
    };

    const index = VectorIndex{ .ivf_flat = .{} };
    var result = try index.searchBatch(
        std.testing.allocator,
        .{
            .ids = &ids,
            .vectors = &vectors,
            .dims = 2,
        },
        .{
            .queries = &queries,
            .query_count = 1,
            .k = 2,
            .dims = 2,
        },
    );
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u32, 2), result.count_per_query);
    try std.testing.expectEqual(@as(usize, 2), result.ids.len);
    try std.testing.expectEqual(@as(u64, 21), result.ids[0]);
}
