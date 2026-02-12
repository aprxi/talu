//! Integration tests for tokenizer.BatchEncodeResult
//!
//! BatchEncodeResult contains the result of batch encoding operation.
//! It uses CSR-style layout with flattened token IDs and offsets.

const std = @import("std");
const main = @import("main");

const BatchEncodeResult = main.tokenizer.BatchEncodeResult;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "BatchEncodeResult type is accessible" {
    const T = BatchEncodeResult;
    _ = T;
}

test "BatchEncodeResult is a struct" {
    const info = @typeInfo(BatchEncodeResult);
    try std.testing.expect(info == .@"struct");
}

test "BatchEncodeResult has expected fields" {
    const info = @typeInfo(BatchEncodeResult);
    const fields = info.@"struct".fields;

    var has_ids = false;
    var has_offsets = false;
    var has_total_tokens = false;
    var has_num_sequences = false;
    var has_allocator = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "ids")) has_ids = true;
        if (comptime std.mem.eql(u8, field.name, "offsets")) has_offsets = true;
        if (comptime std.mem.eql(u8, field.name, "total_tokens")) has_total_tokens = true;
        if (comptime std.mem.eql(u8, field.name, "num_sequences")) has_num_sequences = true;
        if (comptime std.mem.eql(u8, field.name, "allocator")) has_allocator = true;
    }

    try std.testing.expect(has_ids);
    try std.testing.expect(has_offsets);
    try std.testing.expect(has_total_tokens);
    try std.testing.expect(has_num_sequences);
    try std.testing.expect(has_allocator);
}

// =============================================================================
// Method Tests
// =============================================================================

test "BatchEncodeResult has deinit method" {
    try std.testing.expect(@hasDecl(BatchEncodeResult, "deinit"));
}

// =============================================================================
// deinit Tests
// =============================================================================

test "BatchEncodeResult.deinit is safe on empty result" {
    const allocator = std.testing.allocator;
    var result = BatchEncodeResult{
        .ids = &.{},
        .offsets = &.{},
        .total_tokens = 0,
        .num_sequences = 0,
        .allocator = allocator,
    };
    result.deinit();
}
