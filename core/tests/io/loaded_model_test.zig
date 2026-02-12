//! Integration tests for io.LoadedModel
//!
//! LoadedModel is re-exported from io/root.zig for convenience.
//! The full tests are in io/loader/loaded_model_test.zig.

const std = @import("std");
const main = @import("main");
const LoadedModel = main.io.LoadedModel;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "LoadedModel type is accessible from io" {
    const T = LoadedModel;
    _ = T;
}

test "LoadedModel is a struct" {
    const info = @typeInfo(LoadedModel);
    try std.testing.expect(info == .@"struct");
}

test "LoadedModel has deinit method" {
    try std.testing.expect(@hasDecl(LoadedModel, "deinit"));
}
