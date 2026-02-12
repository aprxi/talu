//! Integration tests for HttpListModelsResult
//!
//! HttpListModelsResult contains the response from HttpEngine.listModels(),
//! wrapping an array of HttpModelInfo structs.

const std = @import("std");
const main = @import("main");
const HttpListModelsResult = main.router.HttpListModelsResult;
const HttpModelInfo = main.router.HttpModelInfo;

// =============================================================================
// Struct Layout Tests
// =============================================================================

test "HttpListModelsResult has expected fields" {
    const fields = @typeInfo(HttpListModelsResult).@"struct".fields;

    var has_models = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "models")) has_models = true;
    }

    try std.testing.expect(has_models);
}

test "HttpListModelsResult can be constructed" {
    // Create mock models
    var models = try std.testing.allocator.alloc(HttpModelInfo, 2);
    errdefer std.testing.allocator.free(models);

    models[0] = HttpModelInfo{
        .id = try std.testing.allocator.dupe(u8, "gpt-4o"),
        .object = try std.testing.allocator.dupe(u8, "model"),
        .created = 1699000000,
        .owned_by = try std.testing.allocator.dupe(u8, "openai"),
    };

    models[1] = HttpModelInfo{
        .id = try std.testing.allocator.dupe(u8, "gpt-4o-mini"),
        .object = try std.testing.allocator.dupe(u8, "model"),
        .created = 1700000000,
        .owned_by = try std.testing.allocator.dupe(u8, "openai"),
    };

    const result = HttpListModelsResult{
        .models = models,
    };

    try std.testing.expectEqual(@as(usize, 2), result.models.len);
    try std.testing.expectEqualStrings("gpt-4o", result.models[0].id);
    try std.testing.expectEqualStrings("gpt-4o-mini", result.models[1].id);

    result.deinit(std.testing.allocator);
}

// =============================================================================
// Method Tests
// =============================================================================

test "HttpListModelsResult has deinit method" {
    try std.testing.expect(@hasDecl(HttpListModelsResult, "deinit"));
}

test "HttpListModelsResult.deinit frees all memory" {
    var models = try std.testing.allocator.alloc(HttpModelInfo, 1);
    errdefer std.testing.allocator.free(models);

    models[0] = HttpModelInfo{
        .id = try std.testing.allocator.dupe(u8, "test-model"),
        .object = try std.testing.allocator.dupe(u8, "model"),
        .created = null,
        .owned_by = try std.testing.allocator.dupe(u8, "test"),
    };

    const result = HttpListModelsResult{
        .models = models,
    };

    result.deinit(std.testing.allocator);
    // No leak = test passes
}

// =============================================================================
// Edge Case Tests
// =============================================================================

test "HttpListModelsResult with empty models list" {
    const models = try std.testing.allocator.alloc(HttpModelInfo, 0);

    const result = HttpListModelsResult{
        .models = models,
    };

    try std.testing.expectEqual(@as(usize, 0), result.models.len);

    result.deinit(std.testing.allocator);
}

test "HttpListModelsResult with many models" {
    // Test with larger model list (e.g., from OpenAI which has many models)
    const count = 10;
    var models = try std.testing.allocator.alloc(HttpModelInfo, count);
    errdefer std.testing.allocator.free(models);

    for (0..count) |i| {
        var name_buf: [32]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "model-{d}", .{i}) catch unreachable;

        models[i] = HttpModelInfo{
            .id = try std.testing.allocator.dupe(u8, name),
            .object = try std.testing.allocator.dupe(u8, "model"),
            .created = @as(i64, @intCast(1699000000 + i)),
            .owned_by = try std.testing.allocator.dupe(u8, "test-org"),
        };
    }

    const result = HttpListModelsResult{
        .models = models,
    };

    try std.testing.expectEqual(@as(usize, count), result.models.len);

    result.deinit(std.testing.allocator);
}
