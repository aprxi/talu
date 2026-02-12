//! Integration tests for compute.DLDataType
//!
//! DLDataType is the DLPack data type representation for tensor exchange.

const std = @import("std");
const main = @import("main");
const DLDataType = main.compute.DLDataType;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "DLDataType type is accessible" {
    const T = DLDataType;
    _ = T;
}

test "DLDataType is a struct" {
    const info = @typeInfo(DLDataType);
    try std.testing.expect(info == .@"struct");
}

test "DLDataType has expected fields" {
    const info = @typeInfo(DLDataType);
    const fields = info.@"struct".fields;

    var has_code = false;
    var has_bits = false;
    var has_lanes = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "code")) has_code = true;
        if (comptime std.mem.eql(u8, field.name, "bits")) has_bits = true;
        if (comptime std.mem.eql(u8, field.name, "lanes")) has_lanes = true;
    }

    try std.testing.expect(has_code);
    try std.testing.expect(has_bits);
    try std.testing.expect(has_lanes);
}

// =============================================================================
// Factory Tests
// =============================================================================

test "DLDataType has fromDType method" {
    try std.testing.expect(@hasDecl(DLDataType, "fromDType"));
}

test "DLDataType has factory methods" {
    // DLDataType has type-specific factory methods
    try std.testing.expect(@hasDecl(DLDataType, "float32"));
    try std.testing.expect(@hasDecl(DLDataType, "float64"));
    try std.testing.expect(@hasDecl(DLDataType, "int32"));
    try std.testing.expect(@hasDecl(DLDataType, "int64"));
}
