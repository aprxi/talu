//! Integration tests for io.repository.SearchResultListC
//!
//! SearchResultListC is a C-compatible struct that converts SearchResult slices
//! into C-compatible entries with null-terminated strings suitable for FFI.

const std = @import("std");
const main = @import("main");

const SearchResultListC = main.io.repository.SearchResultListC;
const SearchResultC = main.io.repository.SearchResultC;
const SearchResult = main.io.repository.SearchResult;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "SearchResultListC type is accessible" {
    const T = SearchResultListC;
    _ = T;
}

test "SearchResultListC is a struct" {
    const info = @typeInfo(SearchResultListC);
    try std.testing.expect(info == .@"struct");
}

test "SearchResultListC has expected fields" {
    const info = @typeInfo(SearchResultListC);
    const fields = info.@"struct".fields;

    var has_entries = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "entries")) has_entries = true;
    }

    try std.testing.expect(has_entries);
}

// =============================================================================
// Method Tests
// =============================================================================

test "SearchResultListC has fromResults method" {
    try std.testing.expect(@hasDecl(SearchResultListC, "fromResults"));
}

test "SearchResultListC has deinit method" {
    try std.testing.expect(@hasDecl(SearchResultListC, "deinit"));
}

test "SearchResultListC has count method" {
    try std.testing.expect(@hasDecl(SearchResultListC, "count"));
}

// =============================================================================
// fromResults Tests
// =============================================================================

test "SearchResultListC.fromResults with empty slice" {
    const allocator = std.testing.allocator;
    const results: []const SearchResult = &.{};

    var list = try SearchResultListC.fromResults(allocator, results);
    defer list.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), list.count());
    try std.testing.expectEqual(@as(usize, 0), list.entries.len);
}

test "SearchResultListC.fromResults with single entry" {
    const allocator = std.testing.allocator;
    const results = [_]SearchResult{
        .{
            .model_id = "Qwen/Qwen3-0.6B",
            .downloads = 1000,
            .likes = 50,
            .last_modified = "2024-01-15T10:30:00Z",
            .pipeline_tag = "text-generation",
            .params_total = 600_000_000,
        },
    };

    var list = try SearchResultListC.fromResults(allocator, &results);
    defer list.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), list.count());
    try std.testing.expectEqualStrings("Qwen/Qwen3-0.6B", list.entries[0].model_id);
    try std.testing.expectEqual(@as(i64, 1000), list.entries[0].downloads);
    try std.testing.expectEqual(@as(i64, 50), list.entries[0].likes);
    try std.testing.expectEqualStrings("2024-01-15T10:30:00Z", list.entries[0].last_modified);
    try std.testing.expectEqualStrings("text-generation", list.entries[0].pipeline_tag);
    try std.testing.expectEqual(@as(i64, 600_000_000), list.entries[0].params_total);
}

test "SearchResultListC.fromResults with multiple entries" {
    const allocator = std.testing.allocator;
    const results = [_]SearchResult{
        .{
            .model_id = "org1/model1",
            .downloads = 100,
            .likes = 10,
            .last_modified = "2024-01-01",
            .pipeline_tag = "text-generation",
            .params_total = 1_000_000,
        },
        .{
            .model_id = "org2/model2",
            .downloads = 200,
            .likes = 20,
            .last_modified = "2024-02-01",
            .pipeline_tag = "feature-extraction",
            .params_total = 2_000_000,
        },
        .{
            .model_id = "org3/model3",
            .downloads = 300,
            .likes = 30,
            .last_modified = "2024-03-01",
            .pipeline_tag = "text-classification",
            .params_total = 3_000_000,
        },
    };

    var list = try SearchResultListC.fromResults(allocator, &results);
    defer list.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 3), list.count());

    try std.testing.expectEqualStrings("org1/model1", list.entries[0].model_id);
    try std.testing.expectEqual(@as(i64, 100), list.entries[0].downloads);
    try std.testing.expectEqualStrings("text-generation", list.entries[0].pipeline_tag);

    try std.testing.expectEqualStrings("org2/model2", list.entries[1].model_id);
    try std.testing.expectEqual(@as(i64, 200), list.entries[1].downloads);
    try std.testing.expectEqualStrings("feature-extraction", list.entries[1].pipeline_tag);

    try std.testing.expectEqualStrings("org3/model3", list.entries[2].model_id);
    try std.testing.expectEqual(@as(i64, 300), list.entries[2].downloads);
    try std.testing.expectEqualStrings("text-classification", list.entries[2].pipeline_tag);
}

// =============================================================================
// deinit Tests
// =============================================================================

test "SearchResultListC.deinit clears entries" {
    const allocator = std.testing.allocator;
    const results = [_]SearchResult{
        .{
            .model_id = "test/model",
            .downloads = 0,
            .likes = 0,
            .last_modified = "2024-01-01",
            .pipeline_tag = "text-generation",
            .params_total = 0,
        },
    };

    var list = try SearchResultListC.fromResults(allocator, &results);
    list.deinit(allocator);

    // After deinit, entries should be empty
    try std.testing.expectEqual(@as(usize, 0), list.entries.len);
}

// =============================================================================
// count Tests
// =============================================================================

test "SearchResultListC.count returns correct count" {
    const allocator = std.testing.allocator;
    const results = [_]SearchResult{
        .{
            .model_id = "a/a",
            .downloads = 0,
            .likes = 0,
            .last_modified = "",
            .pipeline_tag = "",
            .params_total = 0,
        },
        .{
            .model_id = "b/b",
            .downloads = 0,
            .likes = 0,
            .last_modified = "",
            .pipeline_tag = "",
            .params_total = 0,
        },
    };

    var list = try SearchResultListC.fromResults(allocator, &results);
    defer list.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 2), list.count());
}

// =============================================================================
// SearchResultC Tests
// =============================================================================

test "SearchResultC type is accessible" {
    const T = SearchResultC;
    _ = T;
}

test "SearchResultC has expected fields" {
    const info = @typeInfo(SearchResultC);
    const fields = info.@"struct".fields;

    var has_model_id = false;
    var has_downloads = false;
    var has_likes = false;
    var has_last_modified = false;
    var has_pipeline_tag = false;
    var has_params_total = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "model_id")) has_model_id = true;
        if (comptime std.mem.eql(u8, field.name, "downloads")) has_downloads = true;
        if (comptime std.mem.eql(u8, field.name, "likes")) has_likes = true;
        if (comptime std.mem.eql(u8, field.name, "last_modified")) has_last_modified = true;
        if (comptime std.mem.eql(u8, field.name, "pipeline_tag")) has_pipeline_tag = true;
        if (comptime std.mem.eql(u8, field.name, "params_total")) has_params_total = true;
    }

    try std.testing.expect(has_model_id);
    try std.testing.expect(has_downloads);
    try std.testing.expect(has_likes);
    try std.testing.expect(has_last_modified);
    try std.testing.expect(has_pipeline_tag);
    try std.testing.expect(has_params_total);
}
