//! Integration tests for HttpModelInfo
//!
//! HttpModelInfo represents a single model returned by the /v1/models endpoint.
//! Contains model ID, object type, creation timestamp, and owner information.

const std = @import("std");
const main = @import("main");
const HttpModelInfo = main.router.HttpModelInfo;

// =============================================================================
// Struct Layout Tests
// =============================================================================

test "HttpModelInfo has expected fields" {
    const fields = @typeInfo(HttpModelInfo).@"struct".fields;

    var has_id = false;
    var has_object = false;
    var has_created = false;
    var has_owned_by = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "id")) has_id = true;
        if (comptime std.mem.eql(u8, field.name, "object")) has_object = true;
        if (comptime std.mem.eql(u8, field.name, "created")) has_created = true;
        if (comptime std.mem.eql(u8, field.name, "owned_by")) has_owned_by = true;
    }

    try std.testing.expect(has_id);
    try std.testing.expect(has_object);
    try std.testing.expect(has_created);
    try std.testing.expect(has_owned_by);
}

test "HttpModelInfo can be constructed" {
    const id = try std.testing.allocator.dupe(u8, "gpt-4o");
    const object = try std.testing.allocator.dupe(u8, "model");
    const owned_by = try std.testing.allocator.dupe(u8, "openai");

    const info = HttpModelInfo{
        .id = id,
        .object = object,
        .created = 1699000000,
        .owned_by = owned_by,
    };

    try std.testing.expectEqualStrings("gpt-4o", info.id);
    try std.testing.expectEqualStrings("model", info.object);
    try std.testing.expectEqual(@as(?i64, 1699000000), info.created);
    try std.testing.expectEqualStrings("openai", info.owned_by);

    info.deinit(std.testing.allocator);
}

// =============================================================================
// Method Tests
// =============================================================================

test "HttpModelInfo has deinit method" {
    try std.testing.expect(@hasDecl(HttpModelInfo, "deinit"));
}

test "HttpModelInfo.deinit frees memory" {
    const id = try std.testing.allocator.dupe(u8, "test-model");
    const object = try std.testing.allocator.dupe(u8, "model");
    const owned_by = try std.testing.allocator.dupe(u8, "test-org");

    const info = HttpModelInfo{
        .id = id,
        .object = object,
        .created = null,
        .owned_by = owned_by,
    };

    info.deinit(std.testing.allocator);
    // No leak = test passes
}

// =============================================================================
// Edge Case Tests
// =============================================================================

test "HttpModelInfo with null created timestamp" {
    const id = try std.testing.allocator.dupe(u8, "local-model");
    const object = try std.testing.allocator.dupe(u8, "model");
    const owned_by = try std.testing.allocator.dupe(u8, "system");

    const info = HttpModelInfo{
        .id = id,
        .object = object,
        .created = null, // Some APIs don't report creation time
        .owned_by = owned_by,
    };

    try std.testing.expect(info.created == null);

    info.deinit(std.testing.allocator);
}

test "HttpModelInfo with model path as ID" {
    // vLLM often uses full model paths as IDs
    const id = try std.testing.allocator.dupe(u8, "Qwen/Qwen3-0.6B");
    const object = try std.testing.allocator.dupe(u8, "model");
    const owned_by = try std.testing.allocator.dupe(u8, "vllm");

    const info = HttpModelInfo{
        .id = id,
        .object = object,
        .created = 1700000000,
        .owned_by = owned_by,
    };

    try std.testing.expectEqualStrings("Qwen/Qwen3-0.6B", info.id);

    info.deinit(std.testing.allocator);
}
