//! Integration tests for converter.F32Result
//!
//! F32Result tracks ownership for F32 tensor data conversion results.

const std = @import("std");
const main = @import("main");
const converter = main.converter;
const F32Result = converter.F32Result;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "F32Result type is accessible" {
    const T = F32Result;
    _ = T;
}

test "F32Result is a struct" {
    const info = @typeInfo(F32Result);
    try std.testing.expect(info == .@"struct");
}

test "F32Result has expected fields" {
    const info = @typeInfo(F32Result);
    const fields = info.@"struct".fields;

    var has_data = false;
    var has_owned = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "data")) has_data = true;
        if (comptime std.mem.eql(u8, field.name, "owned")) has_owned = true;
    }

    try std.testing.expect(has_data);
    try std.testing.expect(has_owned);
}

// =============================================================================
// Method Tests
// =============================================================================

test "F32Result has deinit method" {
    try std.testing.expect(@hasDecl(F32Result, "deinit"));
}

test "F32Result has asF32Slice method" {
    try std.testing.expect(@hasDecl(F32Result, "asF32Slice"));
}

// =============================================================================
// Ownership Tests
// =============================================================================

test "F32Result borrowed data does not require deallocation" {
    // A borrowed result has owned = null
    const borrowed = F32Result{
        .data = &[_]u8{},
        .owned = null,
    };

    // Deinit should be safe to call (no-op for borrowed data)
    borrowed.deinit(std.testing.allocator);
}
