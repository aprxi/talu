//! Integration tests for io.safetensors.SafeTensors
//!
//! SafeTensors provides zero-copy tensor loading from SafeTensors binary files.

const std = @import("std");
const main = @import("main");
const SafeTensors = main.io.safetensors.root.SafeTensors;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "SafeTensors type is accessible" {
    const T = SafeTensors;
    _ = T;
}

test "SafeTensors is a struct" {
    const info = @typeInfo(SafeTensors);
    try std.testing.expect(info == .@"struct");
}

test "SafeTensors has expected fields" {
    const info = @typeInfo(SafeTensors);
    const fields = info.@"struct".fields;

    var has_allocator = false;
    var has_entries = false;
    var has_data_start = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "allocator")) has_allocator = true;
        if (comptime std.mem.eql(u8, field.name, "entries")) has_entries = true;
        if (comptime std.mem.eql(u8, field.name, "data_start")) has_data_start = true;
    }

    try std.testing.expect(has_allocator);
    try std.testing.expect(has_entries);
    try std.testing.expect(has_data_start);
}

// =============================================================================
// Method Tests
// =============================================================================

test "SafeTensors has load method" {
    try std.testing.expect(@hasDecl(SafeTensors, "load"));
}

test "SafeTensors has deinit method" {
    try std.testing.expect(@hasDecl(SafeTensors, "deinit"));
}

test "SafeTensors has getTensor method" {
    try std.testing.expect(@hasDecl(SafeTensors, "getTensor"));
}

test "SafeTensors has entries iterator" {
    // Entries is a StringHashMapUnmanaged, which has iterator method
    const info = @typeInfo(SafeTensors);
    var has_entries = false;

    inline for (info.@"struct".fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "entries")) {
            has_entries = true;
            // StringHashMapUnmanaged has iterator
            try std.testing.expect(@hasDecl(field.type, "iterator"));
        }
    }

    try std.testing.expect(has_entries);
}

// =============================================================================
// Entry Type Tests
// =============================================================================

test "SafeTensors.Entry type is accessible" {
    const T = SafeTensors.Entry;
    _ = T;
}

test "SafeTensors.Entry has expected fields" {
    const info = @typeInfo(SafeTensors.Entry);
    const fields = info.@"struct".fields;

    var has_dtype = false;
    var has_shape = false;
    var has_data = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "dtype")) has_dtype = true;
        if (comptime std.mem.eql(u8, field.name, "shape")) has_shape = true;
        if (comptime std.mem.eql(u8, field.name, "data")) has_data = true;
    }

    try std.testing.expect(has_dtype);
    try std.testing.expect(has_shape);
    try std.testing.expect(has_data);
}
