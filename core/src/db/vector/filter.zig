//! Core-owned vector filter AST and allowlist evaluation.
//!
//! This module intentionally stays storage-agnostic: it evaluates filters
//! against candidate IDs and emits an allowlist bitset consumed by search.

const std = @import("std");

const Allocator = std.mem.Allocator;

pub const FilterExpr = union(enum) {
    all: void,
    id_eq: u64,
    id_in: []const u64,
    and_expr: []const FilterExpr,
    or_expr: []const FilterExpr,
    not_expr: *const FilterExpr,
};

pub const AllowList = struct {
    words: []u64,
    len: usize,
};

pub fn evaluateAllowList(allocator: Allocator, expr: ?*const FilterExpr, ids: []const u64) !AllowList {
    var list = try initAllowList(allocator, ids.len);
    errdefer deinitAllowList(allocator, &list);

    if (ids.len == 0) return list;
    if (expr == null) {
        setAll(&list);
        return list;
    }

    for (ids, 0..) |id, idx| {
        if (matches(expr.?, id)) {
            setBit(&list, idx);
        }
    }
    return list;
}

pub fn allowListCount(list: *const AllowList) usize {
    var total: usize = 0;
    for (list.words) |word| {
        total += @popCount(word);
    }
    return total;
}

pub fn allowListContains(list: *const AllowList, index: usize) bool {
    if (index >= list.len) return false;
    const word_idx = index / 64;
    const bit_idx = @as(u6, @intCast(index % 64));
    return (list.words[word_idx] & (@as(u64, 1) << bit_idx)) != 0;
}

pub fn deinitAllowList(allocator: Allocator, list: *AllowList) void {
    allocator.free(list.words);
    list.words = &[_]u64{};
    list.len = 0;
}

fn initAllowList(allocator: Allocator, len: usize) !AllowList {
    const words_len = if (len == 0) 0 else (len + 63) / 64;
    const words = try allocator.alloc(u64, words_len);
    @memset(words, 0);
    return .{
        .words = words,
        .len = len,
    };
}

fn setBit(list: *AllowList, index: usize) void {
    if (index >= list.len) return;
    const word_idx = index / 64;
    const bit_idx = @as(u6, @intCast(index % 64));
    list.words[word_idx] |= (@as(u64, 1) << bit_idx);
}

fn setAll(list: *AllowList) void {
    if (list.words.len == 0) return;
    @memset(list.words, std.math.maxInt(u64));
    const tail_bits = list.len % 64;
    if (tail_bits == 0) return;
    const mask = (@as(u64, 1) << @as(u6, @intCast(tail_bits))) - 1;
    list.words[list.words.len - 1] = mask;
}

fn matches(expr: *const FilterExpr, id: u64) bool {
    return switch (expr.*) {
        .all => true,
        .id_eq => |target| id == target,
        .id_in => |targets| std.mem.indexOfScalar(u64, targets, id) != null,
        .and_expr => |children| blk: {
            for (children) |child| {
                if (!matches(&child, id)) break :blk false;
            }
            break :blk true;
        },
        .or_expr => |children| blk: {
            for (children) |child| {
                if (matches(&child, id)) break :blk true;
            }
            break :blk false;
        },
        .not_expr => |child| !matches(child, id),
    };
}

test "evaluateAllowList handles null as match-all" {
    const ids = [_]u64{ 10, 20, 30 };
    var allow = try evaluateAllowList(std.testing.allocator, null, &ids);
    defer deinitAllowList(std.testing.allocator, &allow);

    try std.testing.expectEqual(@as(usize, 3), allowListCount(&allow));
    try std.testing.expect(allowListContains(&allow, 0));
    try std.testing.expect(allowListContains(&allow, 1));
    try std.testing.expect(allowListContains(&allow, 2));
}

test "allowListContains returns false for out-of-range index" {
    const ids = [_]u64{1};
    const expr = FilterExpr{ .all = {} };
    var allow = try evaluateAllowList(std.testing.allocator, &expr, &ids);
    defer deinitAllowList(std.testing.allocator, &allow);

    try std.testing.expect(!allowListContains(&allow, 1));
}

test "allowListCount counts selected rows" {
    const ids = [_]u64{ 1, 2, 3, 4 };
    const expr = FilterExpr{ .id_in = &[_]u64{ 2, 4 } };
    var allow = try evaluateAllowList(std.testing.allocator, &expr, &ids);
    defer deinitAllowList(std.testing.allocator, &allow);

    try std.testing.expectEqual(@as(usize, 2), allowListCount(&allow));
}

test "evaluateAllowList supports and/or/not expressions" {
    const ids = [_]u64{ 1, 2, 3, 4 };
    const not_three = FilterExpr{ .not_expr = &FilterExpr{ .id_eq = 3 } };
    const children = [_]FilterExpr{
        .{ .id_in = &[_]u64{ 2, 3, 4 } },
        not_three,
    };
    const expr = FilterExpr{ .and_expr = &children };

    var allow = try evaluateAllowList(std.testing.allocator, &expr, &ids);
    defer deinitAllowList(std.testing.allocator, &allow);

    try std.testing.expect(!allowListContains(&allow, 0));
    try std.testing.expect(allowListContains(&allow, 1));
    try std.testing.expect(!allowListContains(&allow, 2));
    try std.testing.expect(allowListContains(&allow, 3));
}

test "deinitAllowList can be called on populated list" {
    const ids = [_]u64{ 7, 8 };
    const expr = FilterExpr{ .id_eq = 8 };
    var allow = try evaluateAllowList(std.testing.allocator, &expr, &ids);
    deinitAllowList(std.testing.allocator, &allow);

    try std.testing.expectEqual(@as(usize, 0), allow.len);
    try std.testing.expectEqual(@as(usize, 0), allow.words.len);
}
