//! Query planning IR for vector search.
//!
//! Planner decisions stay in core so boundary layers only pass inputs.

const std = @import("std");
const vector_filter = @import("filter.zig");

pub const CandidateSource = enum {
    visible_cache,
};

pub const FilterMode = enum {
    none,
    pre_filter,
};

pub const TopKStrategy = enum {
    exact_heap,
};

pub const IndexKind = enum {
    flat,
    ivf_flat,
};

pub const QueryPlan = struct {
    candidate_source: CandidateSource,
    filter_mode: FilterMode,
    top_k_strategy: TopKStrategy,
    index_kind: IndexKind,
    k: u32,
    query_count: u32,
};

pub fn buildSearchPlan(
    k: u32,
    query_count: u32,
    filter_expr: ?*const vector_filter.FilterExpr,
    prefer_approximate: bool,
) !QueryPlan {
    if (query_count == 0) return error.InvalidColumnData;
    return .{
        .candidate_source = .visible_cache,
        .filter_mode = if (filter_expr == null) .none else .pre_filter,
        .top_k_strategy = .exact_heap,
        .index_kind = if (prefer_approximate) .ivf_flat else .flat,
        .k = k,
        .query_count = query_count,
    };
}

test "buildSearchPlan selects none filter mode when filter is absent" {
    const plan = try buildSearchPlan(10, 2, null, false);
    try std.testing.expectEqual(FilterMode.none, plan.filter_mode);
    try std.testing.expectEqual(IndexKind.flat, plan.index_kind);
    try std.testing.expectEqual(TopKStrategy.exact_heap, plan.top_k_strategy);
}

test "buildSearchPlan selects pre-filter mode when filter exists" {
    const expr = vector_filter.FilterExpr{ .id_eq = 42 };
    const plan = try buildSearchPlan(5, 1, &expr, false);
    try std.testing.expectEqual(FilterMode.pre_filter, plan.filter_mode);
    try std.testing.expectEqual(@as(u32, 5), plan.k);
}

test "buildSearchPlan selects ivf-flat when approximate mode is requested" {
    const plan = try buildSearchPlan(5, 1, null, true);
    try std.testing.expectEqual(IndexKind.ivf_flat, plan.index_kind);
}

test "buildSearchPlan rejects zero query_count" {
    try std.testing.expectError(error.InvalidColumnData, buildSearchPlan(1, 0, null, false));
}
