//! Integration tests for tokenizer.PaddedTensorResult
//!
//! PaddedTensorResult contains padded token tensors ready for model input.
//! It includes input_ids and optional attention_mask arrays.

const std = @import("std");
const main = @import("main");

const PaddedTensorResult = main.tokenizer.PaddedTensorResult;
const PaddingSide = main.tokenizer.PaddingSide;
const PaddedTensorOptions = main.tokenizer.PaddedTensorOptions;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "PaddedTensorResult type is accessible" {
    const T = PaddedTensorResult;
    _ = T;
}

test "PaddedTensorResult is a struct" {
    const info = @typeInfo(PaddedTensorResult);
    try std.testing.expect(info == .@"struct");
}

test "PaddedTensorResult has expected fields" {
    const info = @typeInfo(PaddedTensorResult);
    const fields = info.@"struct".fields;

    var has_input_ids = false;
    var has_attention_mask = false;
    var has_num_sequences = false;
    var has_padded_length = false;
    var has_allocator = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "input_ids")) has_input_ids = true;
        if (comptime std.mem.eql(u8, field.name, "attention_mask")) has_attention_mask = true;
        if (comptime std.mem.eql(u8, field.name, "num_sequences")) has_num_sequences = true;
        if (comptime std.mem.eql(u8, field.name, "padded_length")) has_padded_length = true;
        if (comptime std.mem.eql(u8, field.name, "allocator")) has_allocator = true;
    }

    try std.testing.expect(has_input_ids);
    try std.testing.expect(has_attention_mask);
    try std.testing.expect(has_num_sequences);
    try std.testing.expect(has_padded_length);
    try std.testing.expect(has_allocator);
}

// =============================================================================
// Method Tests
// =============================================================================

test "PaddedTensorResult has deinit method" {
    try std.testing.expect(@hasDecl(PaddedTensorResult, "deinit"));
}

// =============================================================================
// deinit Tests
// =============================================================================

test "PaddedTensorResult.deinit is safe on empty result" {
    const allocator = std.testing.allocator;
    var result = PaddedTensorResult{
        .input_ids = &.{},
        .attention_mask = null,
        .num_sequences = 0,
        .padded_length = 0,
        .allocator = allocator,
    };
    result.deinit();
}

// =============================================================================
// PaddingSide Tests
// =============================================================================

test "PaddingSide type is accessible" {
    const T = PaddingSide;
    _ = T;
}

test "PaddingSide is an enum" {
    const info = @typeInfo(PaddingSide);
    try std.testing.expect(info == .@"enum");
}

test "PaddingSide has right and left values" {
    try std.testing.expectEqual(@as(u8, 0), @intFromEnum(PaddingSide.right));
    try std.testing.expectEqual(@as(u8, 1), @intFromEnum(PaddingSide.left));
}

// =============================================================================
// PaddedTensorOptions Tests
// =============================================================================

test "PaddedTensorOptions type is accessible" {
    const T = PaddedTensorOptions;
    _ = T;
}

test "PaddedTensorOptions is a struct" {
    const info = @typeInfo(PaddedTensorOptions);
    try std.testing.expect(info == .@"struct");
}

test "PaddedTensorOptions has expected fields" {
    const info = @typeInfo(PaddedTensorOptions);
    const fields = info.@"struct".fields;

    var has_pad_id = false;
    var has_padding_side = false;
    var has_max_length = false;
    var has_truncate = false;
    var has_return_attention_mask = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "pad_id")) has_pad_id = true;
        if (comptime std.mem.eql(u8, field.name, "padding_side")) has_padding_side = true;
        if (comptime std.mem.eql(u8, field.name, "max_length")) has_max_length = true;
        if (comptime std.mem.eql(u8, field.name, "truncate")) has_truncate = true;
        if (comptime std.mem.eql(u8, field.name, "return_attention_mask")) has_return_attention_mask = true;
    }

    try std.testing.expect(has_pad_id);
    try std.testing.expect(has_padding_side);
    try std.testing.expect(has_max_length);
    try std.testing.expect(has_truncate);
    try std.testing.expect(has_return_attention_mask);
}

test "PaddedTensorOptions has sensible defaults" {
    const opts = PaddedTensorOptions{};
    try std.testing.expectEqual(@as(u32, 0), opts.pad_id);
    try std.testing.expectEqual(PaddingSide.right, opts.padding_side);
    try std.testing.expectEqual(@as(usize, 0), opts.max_length);
    try std.testing.expect(!opts.truncate);
    try std.testing.expect(opts.return_attention_mask);
}
