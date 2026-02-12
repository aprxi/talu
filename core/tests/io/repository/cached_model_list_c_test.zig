//! Integration tests for io.repository.CachedModelListC
//!
//! CachedModelListC is a C-compatible struct that converts CachedModel slices
//! into C-compatible entries with null-terminated strings suitable for FFI.

const std = @import("std");
const main = @import("main");

const CachedModelListC = main.io.repository.CachedModelListC;
const CachedModelC = main.io.repository.CachedModelC;
const CachedModel = main.io.repository.CachedModel;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "CachedModelListC type is accessible" {
    const T = CachedModelListC;
    _ = T;
}

test "CachedModelListC is a struct" {
    const info = @typeInfo(CachedModelListC);
    try std.testing.expect(info == .@"struct");
}

test "CachedModelListC has expected fields" {
    const info = @typeInfo(CachedModelListC);
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

test "CachedModelListC has fromModels method" {
    try std.testing.expect(@hasDecl(CachedModelListC, "fromModels"));
}

test "CachedModelListC has deinit method" {
    try std.testing.expect(@hasDecl(CachedModelListC, "deinit"));
}

test "CachedModelListC has count method" {
    try std.testing.expect(@hasDecl(CachedModelListC, "count"));
}

// =============================================================================
// fromModels Tests
// =============================================================================

test "CachedModelListC.fromModels with empty slice" {
    const allocator = std.testing.allocator;
    const models: []const CachedModel = &.{};

    var list = try CachedModelListC.fromModels(allocator, models);
    defer list.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), list.count());
    try std.testing.expectEqual(@as(usize, 0), list.entries.len);
}

test "CachedModelListC.fromModels with single entry" {
    const allocator = std.testing.allocator;
    const models = [_]CachedModel{
        .{
            .model_id = "Qwen/Qwen3-0.6B",
            .cache_dir = "/home/user/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B",
        },
    };

    var list = try CachedModelListC.fromModels(allocator, &models);
    defer list.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), list.count());
    try std.testing.expectEqualStrings("Qwen/Qwen3-0.6B", list.entries[0].model_id);
    try std.testing.expectEqualStrings(
        "/home/user/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B",
        list.entries[0].cache_dir,
    );
}

test "CachedModelListC.fromModels with multiple entries" {
    const allocator = std.testing.allocator;
    const models = [_]CachedModel{
        .{ .model_id = "org1/model1", .cache_dir = "/cache/models--org1--model1" },
        .{ .model_id = "org2/model2", .cache_dir = "/cache/models--org2--model2" },
        .{ .model_id = "org3/model3", .cache_dir = "/cache/models--org3--model3" },
    };

    var list = try CachedModelListC.fromModels(allocator, &models);
    defer list.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 3), list.count());

    try std.testing.expectEqualStrings("org1/model1", list.entries[0].model_id);
    try std.testing.expectEqualStrings("/cache/models--org1--model1", list.entries[0].cache_dir);

    try std.testing.expectEqualStrings("org2/model2", list.entries[1].model_id);
    try std.testing.expectEqualStrings("/cache/models--org2--model2", list.entries[1].cache_dir);

    try std.testing.expectEqualStrings("org3/model3", list.entries[2].model_id);
    try std.testing.expectEqualStrings("/cache/models--org3--model3", list.entries[2].cache_dir);
}

// =============================================================================
// deinit Tests
// =============================================================================

test "CachedModelListC.deinit clears entries" {
    const allocator = std.testing.allocator;
    const models = [_]CachedModel{
        .{ .model_id = "test/model", .cache_dir = "/tmp/cache" },
    };

    var list = try CachedModelListC.fromModels(allocator, &models);
    list.deinit(allocator);

    // After deinit, entries should be empty
    try std.testing.expectEqual(@as(usize, 0), list.entries.len);
}

// =============================================================================
// count Tests
// =============================================================================

test "CachedModelListC.count returns correct count" {
    const allocator = std.testing.allocator;
    const models = [_]CachedModel{
        .{ .model_id = "a/a", .cache_dir = "/a" },
        .{ .model_id = "b/b", .cache_dir = "/b" },
    };

    var list = try CachedModelListC.fromModels(allocator, &models);
    defer list.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 2), list.count());
}

// =============================================================================
// CachedModelC Tests
// =============================================================================

test "CachedModelC type is accessible" {
    const T = CachedModelC;
    _ = T;
}

test "CachedModelC has expected fields" {
    const info = @typeInfo(CachedModelC);
    const fields = info.@"struct".fields;

    var has_model_id = false;
    var has_cache_dir = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "model_id")) has_model_id = true;
        if (comptime std.mem.eql(u8, field.name, "cache_dir")) has_cache_dir = true;
    }

    try std.testing.expect(has_model_id);
    try std.testing.expect(has_cache_dir);
}
