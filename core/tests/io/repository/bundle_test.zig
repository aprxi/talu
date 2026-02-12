//! Integration tests for io.repository.Bundle
//!
//! Bundle represents a validated bundle of model resources ready for loading.
//! It abstracts local paths and HF cache format.

const std = @import("std");
const main = @import("main");
const Bundle = main.io.repository.Bundle;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "Bundle type is accessible" {
    const T = Bundle;
    _ = T;
}

test "Bundle is a struct" {
    const info = @typeInfo(Bundle);
    try std.testing.expect(info == .@"struct");
}

test "Bundle has expected fields" {
    const info = @typeInfo(Bundle);
    const fields = info.@"struct".fields;

    var has_allocator = false;
    var has_dir = false;
    var has_config = false;
    var has_weights = false;
    var has_tokenizer = false;
    var has_format = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "allocator")) has_allocator = true;
        if (comptime std.mem.eql(u8, field.name, "dir")) has_dir = true;
        if (comptime std.mem.eql(u8, field.name, "config")) has_config = true;
        if (comptime std.mem.eql(u8, field.name, "weights")) has_weights = true;
        if (comptime std.mem.eql(u8, field.name, "tokenizer")) has_tokenizer = true;
        if (comptime std.mem.eql(u8, field.name, "format")) has_format = true;
    }

    try std.testing.expect(has_allocator);
    try std.testing.expect(has_dir);
    try std.testing.expect(has_config);
    try std.testing.expect(has_weights);
    try std.testing.expect(has_tokenizer);
    try std.testing.expect(has_format);
}

// =============================================================================
// Nested Type Tests
// =============================================================================

test "Bundle.Format enum exists" {
    const Format = Bundle.Format;
    try std.testing.expect(@typeInfo(Format) == .@"enum");

    // Check variants
    _ = Format.mlx;
    _ = Format.safetensors;
}

test "Bundle.ConfigSource union exists" {
    const ConfigSource = Bundle.ConfigSource;
    try std.testing.expect(@typeInfo(ConfigSource) == .@"union");
}

test "Bundle.WeightsSource union exists" {
    const WeightsSource = Bundle.WeightsSource;
    try std.testing.expect(@typeInfo(WeightsSource) == .@"union");
}

test "Bundle.TokenizerSource union exists" {
    const TokenizerSource = Bundle.TokenizerSource;
    try std.testing.expect(@typeInfo(TokenizerSource) == .@"union");
}

// =============================================================================
// Method Existence Tests
// =============================================================================

test "Bundle has config_path method" {
    try std.testing.expect(@hasDecl(Bundle, "config_path"));
}

test "Bundle has deinit method" {
    try std.testing.expect(@hasDecl(Bundle, "deinit"));
}
