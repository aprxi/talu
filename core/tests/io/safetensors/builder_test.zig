//! Integration tests for io.safetensors.Builder
//!
//! Builder is used for writing tensors to SafeTensors files.

const std = @import("std");
const main = @import("main");
const writer = main.io.safetensors.root.writer;
const Builder = writer.Builder;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "Builder type is accessible" {
    const T = Builder;
    _ = T;
}

test "Builder is a struct" {
    const info = @typeInfo(Builder);
    try std.testing.expect(info == .@"struct");
}

test "Builder has expected fields" {
    const info = @typeInfo(Builder);
    const fields = info.@"struct".fields;

    var has_allocator = false;
    var has_entries = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "allocator")) has_allocator = true;
        if (comptime std.mem.eql(u8, field.name, "entries")) has_entries = true;
    }

    try std.testing.expect(has_allocator);
    try std.testing.expect(has_entries);
}

// =============================================================================
// Method Tests
// =============================================================================

test "Builder has init method" {
    try std.testing.expect(@hasDecl(Builder, "init"));
}

test "Builder has deinit method" {
    try std.testing.expect(@hasDecl(Builder, "deinit"));
}

test "Builder has addTensor method" {
    try std.testing.expect(@hasDecl(Builder, "addTensor"));
}

test "Builder has save method" {
    try std.testing.expect(@hasDecl(Builder, "save"));
}

// =============================================================================
// TensorEntry Type Tests
// =============================================================================

test "TensorEntry type is accessible" {
    const T = writer.TensorEntry;
    _ = T;
}

test "TensorEntry has expected fields" {
    const info = @typeInfo(writer.TensorEntry);
    const fields = info.@"struct".fields;

    var has_name = false;
    var has_dtype = false;
    var has_shape = false;
    var has_data = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "name")) has_name = true;
        if (comptime std.mem.eql(u8, field.name, "dtype")) has_dtype = true;
        if (comptime std.mem.eql(u8, field.name, "shape")) has_shape = true;
        if (comptime std.mem.eql(u8, field.name, "data")) has_data = true;
    }

    try std.testing.expect(has_name);
    try std.testing.expect(has_dtype);
    try std.testing.expect(has_shape);
    try std.testing.expect(has_data);
}
