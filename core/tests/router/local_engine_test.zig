//! Integration tests for LocalEngine
//!
//! LocalEngine loads a model and handles all local inference.
//! Multiple Chats can share a single engine for efficient multi-user serving.
//!
//! Note: Full integration tests require a model to be present.
//! These tests verify the type structure and exports without loading models.

const std = @import("std");
const main = @import("main");
const LocalEngine = main.router.LocalEngine;
const GenerateOptions = main.router.GenerateOptions;
const GenerationResult = main.router.GenerationResult;

// =============================================================================
// Type Export Tests
// =============================================================================

test "LocalEngine is exported from router" {
    // Verify the type exists and is accessible
    const type_info = @typeInfo(LocalEngine);
    try std.testing.expect(type_info == .@"struct");
}

test "LocalEngine has expected fields" {
    const fields = @typeInfo(LocalEngine).@"struct".fields;

    // Should have allocator, loaded, tok, samp, backend, gen_config, model_path
    try std.testing.expectEqual(@as(usize, 7), fields.len);

    var has_allocator = false;
    var has_loaded = false;
    var has_tok = false;
    var has_samp = false;
    var has_backend = false;
    var has_gen_config = false;
    var has_model_path = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "allocator")) has_allocator = true;
        if (comptime std.mem.eql(u8, field.name, "loaded")) has_loaded = true;
        if (comptime std.mem.eql(u8, field.name, "tok")) has_tok = true;
        if (comptime std.mem.eql(u8, field.name, "samp")) has_samp = true;
        if (comptime std.mem.eql(u8, field.name, "backend")) has_backend = true;
        if (comptime std.mem.eql(u8, field.name, "gen_config")) has_gen_config = true;
        if (comptime std.mem.eql(u8, field.name, "model_path")) has_model_path = true;
    }

    try std.testing.expect(has_allocator);
    try std.testing.expect(has_loaded);
    try std.testing.expect(has_tok);
    try std.testing.expect(has_samp);
    try std.testing.expect(has_backend);
    try std.testing.expect(has_gen_config);
    try std.testing.expect(has_model_path);
}

// =============================================================================
// Method Existence Tests
// =============================================================================

test "LocalEngine has init method" {
    // Verify init exists and has correct signature
    try std.testing.expect(@hasDecl(LocalEngine, "init"));

    const init_fn = @typeInfo(@TypeOf(LocalEngine.init)).@"fn";
    try std.testing.expectEqual(@as(usize, 2), init_fn.params.len);
}

test "LocalEngine has initWithSeed method" {
    try std.testing.expect(@hasDecl(LocalEngine, "initWithSeed"));

    const init_fn = @typeInfo(@TypeOf(LocalEngine.initWithSeed)).@"fn";
    try std.testing.expectEqual(@as(usize, 3), init_fn.params.len); // allocator, path, seed
}

test "LocalEngine has deinit method" {
    try std.testing.expect(@hasDecl(LocalEngine, "deinit"));
}

test "LocalEngine has generate method" {
    try std.testing.expect(@hasDecl(LocalEngine, "generate"));
}

test "LocalEngine has getEosTokens method" {
    try std.testing.expect(@hasDecl(LocalEngine, "getEosTokens"));
}

test "LocalEngine has vocabSize method" {
    try std.testing.expect(@hasDecl(LocalEngine, "vocabSize"));
}

test "LocalEngine has encode method" {
    try std.testing.expect(@hasDecl(LocalEngine, "encode"));
}

test "LocalEngine has decode method" {
    try std.testing.expect(@hasDecl(LocalEngine, "decode"));
}

test "LocalEngine has tokenizer method" {
    try std.testing.expect(@hasDecl(LocalEngine, "tokenizer"));
}

// =============================================================================
// Init Error Tests
// =============================================================================

test "LocalEngine.init returns error for missing model" {
    const result = LocalEngine.init(std.testing.allocator, "/nonexistent/path/to/model");

    // Should return an error, not crash
    try std.testing.expectError(error.WeightsNotFound, result);
}

test "LocalEngine.initWithSeed returns error for missing model" {
    const result = LocalEngine.initWithSeed(std.testing.allocator, "/nonexistent/path", 12345);

    try std.testing.expectError(error.WeightsNotFound, result);
}

// =============================================================================
// Type Relationship Tests
// =============================================================================

test "LocalEngine.generate returns GenerationResult" {
    // Verify return type through reflection
    const generate_fn = @typeInfo(@TypeOf(LocalEngine.generate)).@"fn";
    const ReturnType = generate_fn.return_type.?;

    // Should be an error union with GenerationResult
    const error_union_info = @typeInfo(ReturnType);
    try std.testing.expect(error_union_info == .error_union);
    try std.testing.expect(error_union_info.error_union.payload == GenerationResult);
}

test "LocalEngine.generate accepts GenerateOptions" {
    const generate_fn = @typeInfo(@TypeOf(LocalEngine.generate)).@"fn";

    // Third parameter should be GenerateOptions
    try std.testing.expectEqual(@as(usize, 3), generate_fn.params.len);
    try std.testing.expect(generate_fn.params[2].type == GenerateOptions);
}
