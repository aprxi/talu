//! Integration tests for HttpEngine
//!
//! HttpEngine handles remote inference via OpenAI-compatible APIs.
//! These tests verify the type structure, initialization, and exports.
//!
//! Note: Network-dependent functions (generate, stream, listModels) cannot be
//! unit tested without a real server. Their parsing logic is tested via mock
//! tests in http_engine.zig. Real HTTP integration tests are in bindings/python.

const std = @import("std");
const main = @import("main");

const HttpEngine = main.router.HttpEngine;
const HttpEngineConfig = main.router.HttpEngineConfig;

// =============================================================================
// HttpEngine Type Export Tests
// =============================================================================

test "HttpEngine is exported from router" {
    const type_info = @typeInfo(HttpEngine);
    try std.testing.expect(type_info == .@"struct");
}

test "HttpEngine has expected fields" {
    const fields = @typeInfo(HttpEngine).@"struct".fields;

    var has_allocator = false;
    var has_base_url = false;
    var has_endpoint_url = false;
    var has_api_key = false;
    var has_model = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "allocator")) has_allocator = true;
        if (comptime std.mem.eql(u8, field.name, "base_url")) has_base_url = true;
        if (comptime std.mem.eql(u8, field.name, "endpoint_url")) has_endpoint_url = true;
        if (comptime std.mem.eql(u8, field.name, "api_key")) has_api_key = true;
        if (comptime std.mem.eql(u8, field.name, "model")) has_model = true;
    }

    try std.testing.expect(has_allocator);
    try std.testing.expect(has_base_url);
    try std.testing.expect(has_endpoint_url);
    try std.testing.expect(has_api_key);
    try std.testing.expect(has_model);
}

// =============================================================================
// HttpEngine Method Existence Tests
// =============================================================================

test "HttpEngine has init method" {
    try std.testing.expect(@hasDecl(HttpEngine, "init"));
}

test "HttpEngine has deinit method" {
    try std.testing.expect(@hasDecl(HttpEngine, "deinit"));
}

test "HttpEngine has generate method" {
    try std.testing.expect(@hasDecl(HttpEngine, "generate"));
}

test "HttpEngine has stream method" {
    try std.testing.expect(@hasDecl(HttpEngine, "stream"));
}

test "HttpEngine has listModels method" {
    try std.testing.expect(@hasDecl(HttpEngine, "listModels"));
}

test "HttpEngine has parseResponse method" {
    try std.testing.expect(@hasDecl(HttpEngine, "parseResponse"));
}

test "HttpEngine has parseModelsResponse method" {
    try std.testing.expect(@hasDecl(HttpEngine, "parseModelsResponse"));
}

// =============================================================================
// HttpEngine Initialization Tests
// =============================================================================

test "HttpEngine.init creates engine with correct fields" {
    const allocator = std.testing.allocator;

    var engine = try HttpEngine.init(allocator, .{
        .base_url = "http://localhost:8000/v1",
        .api_key = "test-key",
        .model = "test-model",
        .timeout_ms = 30000,
    });
    defer engine.deinit();

    // Verify fields are set correctly
    try std.testing.expectEqualStrings("test-model", engine.model);
    try std.testing.expect(engine.api_key != null);
    try std.testing.expectEqualStrings("test-key", engine.api_key.?);
    try std.testing.expectEqual(@as(i32, 30000), engine.timeout_ms);

    // Verify endpoint URL is constructed correctly
    try std.testing.expect(std.mem.endsWith(u8, engine.endpoint_url, "/chat/completions"));
}

test "HttpEngine.init without api_key" {
    const allocator = std.testing.allocator;

    var engine = try HttpEngine.init(allocator, .{
        .base_url = "http://localhost:8000/v1",
        .api_key = null,
        .model = "local-model",
    });
    defer engine.deinit();

    try std.testing.expect(engine.api_key == null);
    try std.testing.expectEqualStrings("local-model", engine.model);
}

test "HttpEngine.init appends /chat/completions to base_url" {
    const allocator = std.testing.allocator;

    // Without trailing slash
    var engine1 = try HttpEngine.init(allocator, .{
        .base_url = "http://localhost:8000/v1",
        .model = "test",
    });
    defer engine1.deinit();
    try std.testing.expect(std.mem.endsWith(u8, engine1.endpoint_url, "/v1/chat/completions"));

    // With trailing slash
    var engine2 = try HttpEngine.init(allocator, .{
        .base_url = "http://localhost:8000/v1/",
        .model = "test",
    });
    defer engine2.deinit();
    try std.testing.expect(std.mem.endsWith(u8, engine2.endpoint_url, "/chat/completions"));
}

test "HttpEngine.deinit frees all memory" {
    const allocator = std.testing.allocator;

    var engine = try HttpEngine.init(allocator, .{
        .base_url = "http://localhost:8000/v1",
        .api_key = "test-key",
        .org_id = "org-123",
        .model = "test-model",
    });

    // deinit should free all owned strings without crash
    engine.deinit();
    // No leak = test passes (testing allocator will catch leaks)
}

// =============================================================================
// HttpEngineConfig Tests
// =============================================================================

test "HttpEngineConfig has expected fields" {
    const fields = @typeInfo(HttpEngineConfig).@"struct".fields;

    var has_base_url = false;
    var has_api_key = false;
    var has_model = false;
    var has_timeout_ms = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "base_url")) has_base_url = true;
        if (comptime std.mem.eql(u8, field.name, "api_key")) has_api_key = true;
        if (comptime std.mem.eql(u8, field.name, "model")) has_model = true;
        if (comptime std.mem.eql(u8, field.name, "timeout_ms")) has_timeout_ms = true;
    }

    try std.testing.expect(has_base_url);
    try std.testing.expect(has_api_key);
    try std.testing.expect(has_model);
    try std.testing.expect(has_timeout_ms);
}

test "HttpEngineConfig has sensible defaults" {
    // Verify default values exist
    const config = HttpEngineConfig{
        .base_url = "http://test",
        .model = "test",
    };

    // api_key should default to null
    try std.testing.expect(config.api_key == null);

    // timeout_ms should have a reasonable default
    try std.testing.expect(config.timeout_ms > 0);
}
