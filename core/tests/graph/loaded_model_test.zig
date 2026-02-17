//! Integration tests for graph.LoadedModel
//!
//! LoadedModel is re-exported from graph/root.zig.

const std = @import("std");
const main = @import("main");
const LoadedModel = main.graph.LoadedModel;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "LoadedModel type is accessible from graph" {
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
