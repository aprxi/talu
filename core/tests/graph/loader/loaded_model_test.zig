//! Integration tests for graph.loader.LoadedModel
//!
//! LoadedModel contains the loaded model weights and configuration.

const std = @import("std");
const main = @import("main");
const LoadedModel = main.graph.loader.LoadedModel;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "LoadedModel type is accessible" {
    const T = LoadedModel;
    _ = T;
}

test "LoadedModel is a struct" {
    const info = @typeInfo(LoadedModel);
    try std.testing.expect(info == .@"struct");
}

test "LoadedModel has expected fields" {
    const info = @typeInfo(LoadedModel);
    const fields = info.@"struct".fields;

    var has_arena = false;
    var has_config = false;
    var has_runtime = false;
    var has_ln_final = false;
    var has_lm_head = false;
    var has_token_embeddings = false;
    var has_blocks = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "arena")) has_arena = true;
        if (comptime std.mem.eql(u8, field.name, "config")) has_config = true;
        if (comptime std.mem.eql(u8, field.name, "runtime")) has_runtime = true;
        if (comptime std.mem.eql(u8, field.name, "ln_final")) has_ln_final = true;
        if (comptime std.mem.eql(u8, field.name, "lm_head")) has_lm_head = true;
        if (comptime std.mem.eql(u8, field.name, "token_embeddings")) has_token_embeddings = true;
        if (comptime std.mem.eql(u8, field.name, "blocks")) has_blocks = true;
    }

    try std.testing.expect(has_arena);
    try std.testing.expect(has_config);
    try std.testing.expect(has_runtime);
    try std.testing.expect(has_ln_final);
    try std.testing.expect(has_lm_head);
    try std.testing.expect(has_token_embeddings);
    try std.testing.expect(has_blocks);
}

// =============================================================================
// Method Tests
// =============================================================================

test "LoadedModel has deinit method" {
    try std.testing.expect(@hasDecl(LoadedModel, "deinit"));
}

test "LoadedModel has ensureCpuBlocks method" {
    try std.testing.expect(@hasDecl(LoadedModel, "ensureCpuBlocks"));
}

// =============================================================================
// Optional Field Tests
// =============================================================================

test "LoadedModel has runtime architecture fields" {
    const info = @typeInfo(LoadedModel);
    const fields = info.@"struct".fields;

    var has_runtime_arch = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "runtime_arch")) has_runtime_arch = true;
    }

    try std.testing.expect(has_runtime_arch);
}

test "LoadedModel has file metadata fields" {
    const info = @typeInfo(LoadedModel);
    const fields = info.@"struct".fields;

    var has_file_size = false;
    var has_tensor_count = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "file_size")) has_file_size = true;
        if (comptime std.mem.eql(u8, field.name, "tensor_count")) has_tensor_count = true;
    }

    try std.testing.expect(has_file_size);
    try std.testing.expect(has_tensor_count);
}
