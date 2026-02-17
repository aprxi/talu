//! Integration tests for inference.model_loader.ArchitectureCheck
//!
//! ArchitectureCheck is the result of checking model architecture support.

const std = @import("std");
const main = @import("main");
const ArchitectureCheck = main.inference.model_loader.ArchitectureCheck;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "ArchitectureCheck type is accessible" {
    const T = ArchitectureCheck;
    _ = T;
}

test "ArchitectureCheck is a struct" {
    const info = @typeInfo(ArchitectureCheck);
    try std.testing.expect(info == .@"struct");
}

test "ArchitectureCheck has expected fields" {
    const info = @typeInfo(ArchitectureCheck);
    const fields = info.@"struct".fields;

    var has_supported = false;
    var has_model_type_buf = false;
    var has_model_type_len = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "supported")) has_supported = true;
        if (comptime std.mem.eql(u8, field.name, "model_type_buf")) has_model_type_buf = true;
        if (comptime std.mem.eql(u8, field.name, "model_type_len")) has_model_type_len = true;
    }

    try std.testing.expect(has_supported);
    try std.testing.expect(has_model_type_buf);
    try std.testing.expect(has_model_type_len);
}

// =============================================================================
// Method Tests
// =============================================================================

test "ArchitectureCheck has getModelType method" {
    try std.testing.expect(@hasDecl(ArchitectureCheck, "getModelType"));
}

test "ArchitectureCheck has getArchitecture method" {
    try std.testing.expect(@hasDecl(ArchitectureCheck, "getArchitecture"));
}

test "ArchitectureCheck getModelType returns null for empty" {
    const check = ArchitectureCheck{ .supported = false };
    try std.testing.expect(check.getModelType() == null);
}

test "ArchitectureCheck getArchitecture returns null for empty" {
    const check = ArchitectureCheck{ .supported = false };
    try std.testing.expect(check.getArchitecture() == null);
}
