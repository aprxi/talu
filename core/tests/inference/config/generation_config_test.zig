//! Integration tests for inference.config.GenerationConfig
//!
//! GenerationConfig contains generation parameters loaded from model directory.

const std = @import("std");
const main = @import("main");
const GenerationConfig = main.inference.config.GenerationConfig;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "GenerationConfig type is accessible" {
    const T = GenerationConfig;
    _ = T;
}

test "GenerationConfig is a struct" {
    const info = @typeInfo(GenerationConfig);
    try std.testing.expect(info == .@"struct");
}

test "GenerationConfig has expected fields" {
    const info = @typeInfo(GenerationConfig);
    const fields = info.@"struct".fields;

    var has_temperature = false;
    var has_top_k = false;
    var has_top_p = false;
    var has_do_sample = false;
    var has_eos_token_ids = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "temperature")) has_temperature = true;
        if (comptime std.mem.eql(u8, field.name, "top_k")) has_top_k = true;
        if (comptime std.mem.eql(u8, field.name, "top_p")) has_top_p = true;
        if (comptime std.mem.eql(u8, field.name, "do_sample")) has_do_sample = true;
        if (comptime std.mem.eql(u8, field.name, "eos_token_ids")) has_eos_token_ids = true;
    }

    try std.testing.expect(has_temperature);
    try std.testing.expect(has_top_k);
    try std.testing.expect(has_top_p);
    try std.testing.expect(has_do_sample);
    try std.testing.expect(has_eos_token_ids);
}

// =============================================================================
// Default Value Tests
// =============================================================================

test "GenerationConfig default temperature is 1.0" {
    const config = GenerationConfig{};
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), config.temperature, 0.001);
}

test "GenerationConfig default top_k is 50" {
    const config = GenerationConfig{};
    try std.testing.expectEqual(@as(usize, 50), config.top_k);
}

test "GenerationConfig default top_p is 1.0" {
    const config = GenerationConfig{};
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), config.top_p, 0.001);
}

test "GenerationConfig default do_sample is true" {
    const config = GenerationConfig{};
    try std.testing.expect(config.do_sample);
}

test "GenerationConfig default add_bos_token is true" {
    const config = GenerationConfig{};
    try std.testing.expect(config.add_bos_token);
}

// =============================================================================
// Method Tests
// =============================================================================

test "GenerationConfig has deinit method" {
    try std.testing.expect(@hasDecl(GenerationConfig, "deinit"));
}
