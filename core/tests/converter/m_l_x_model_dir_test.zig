//! Integration tests for converter.MLXModelDir
//!
//! MLXModelDir manages MLX model directory structure.

const std = @import("std");
const main = @import("main");
const MLXModelDir = main.converter.MLXModelDir;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "MLXModelDir type is accessible" {
    const T = MLXModelDir;
    _ = T;
}

test "MLXModelDir is a struct" {
    const info = @typeInfo(MLXModelDir);
    try std.testing.expect(info == .@"struct");
}

test "MLXModelDir has expected fields" {
    const info = @typeInfo(MLXModelDir);
    const fields = info.@"struct".fields;

    var has_allocator = false;
    var has_path = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "allocator")) has_allocator = true;
        if (comptime std.mem.eql(u8, field.name, "path")) has_path = true;
    }

    try std.testing.expect(has_allocator);
    try std.testing.expect(has_path);
}

// =============================================================================
// Method Tests
// =============================================================================

test "MLXModelDir has init method" {
    try std.testing.expect(@hasDecl(MLXModelDir, "init"));
}

test "MLXModelDir has deinit method" {
    try std.testing.expect(@hasDecl(MLXModelDir, "deinit"));
}

test "MLXModelDir has configPath method" {
    try std.testing.expect(@hasDecl(MLXModelDir, "configPath"));
}

test "MLXModelDir has weightsPath method" {
    try std.testing.expect(@hasDecl(MLXModelDir, "weightsPath"));
}

test "MLXModelDir has tokenizerPath method" {
    try std.testing.expect(@hasDecl(MLXModelDir, "tokenizerPath"));
}

test "MLXModelDir has copyTokenizerFiles method" {
    try std.testing.expect(@hasDecl(MLXModelDir, "copyTokenizerFiles"));
}
