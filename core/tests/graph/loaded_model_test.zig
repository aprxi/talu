//! Integration tests for graph.LoadedModel
//!
//! LoadedModel is exported from models root (`main.models.dispatcher`).

const std = @import("std");
const main = @import("main");
const LoadedModel = main.models.dispatcher.LoadedModel;

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
