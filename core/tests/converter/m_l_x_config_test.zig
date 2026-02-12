//! Integration tests for converter.MLXConfig
//!
//! MLXConfig represents MLX-specific model configuration.

const std = @import("std");
const main = @import("main");
const MLXConfig = main.converter.MLXConfig;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "MLXConfig type is accessible" {
    const T = MLXConfig;
    _ = T;
}

test "MLXConfig is a struct" {
    const info = @typeInfo(MLXConfig);
    try std.testing.expect(info == .@"struct");
}

test "MLXConfig has expected standard fields" {
    const info = @typeInfo(MLXConfig);
    const fields = info.@"struct".fields;

    var has_vocab_size = false;
    var has_hidden_size = false;
    var has_num_hidden_layers = false;
    var has_num_attention_heads = false;
    var has_num_key_value_heads = false;
    var has_intermediate_size = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "vocab_size")) has_vocab_size = true;
        if (comptime std.mem.eql(u8, field.name, "hidden_size")) has_hidden_size = true;
        if (comptime std.mem.eql(u8, field.name, "num_hidden_layers")) has_num_hidden_layers = true;
        if (comptime std.mem.eql(u8, field.name, "num_attention_heads")) has_num_attention_heads = true;
        if (comptime std.mem.eql(u8, field.name, "num_key_value_heads")) has_num_key_value_heads = true;
        if (comptime std.mem.eql(u8, field.name, "intermediate_size")) has_intermediate_size = true;
    }

    try std.testing.expect(has_vocab_size);
    try std.testing.expect(has_hidden_size);
    try std.testing.expect(has_num_hidden_layers);
    try std.testing.expect(has_num_attention_heads);
    try std.testing.expect(has_num_key_value_heads);
    try std.testing.expect(has_intermediate_size);
}

test "MLXConfig has position and norm fields" {
    const info = @typeInfo(MLXConfig);
    const fields = info.@"struct".fields;

    var has_max_position_embeddings = false;
    var has_head_dim = false;
    var has_rms_norm_eps = false;
    var has_rope_theta = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "max_position_embeddings")) has_max_position_embeddings = true;
        if (comptime std.mem.eql(u8, field.name, "head_dim")) has_head_dim = true;
        if (comptime std.mem.eql(u8, field.name, "rms_norm_eps")) has_rms_norm_eps = true;
        if (comptime std.mem.eql(u8, field.name, "rope_theta")) has_rope_theta = true;
    }

    try std.testing.expect(has_max_position_embeddings);
    try std.testing.expect(has_head_dim);
    try std.testing.expect(has_rms_norm_eps);
    try std.testing.expect(has_rope_theta);
}

test "MLXConfig has quantization field" {
    const info = @typeInfo(MLXConfig);
    const fields = info.@"struct".fields;

    var has_quantization = false;
    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "quantization")) has_quantization = true;
    }

    try std.testing.expect(has_quantization);
}

// =============================================================================
// Method Tests
// =============================================================================

test "MLXConfig has fromModelConfig method" {
    try std.testing.expect(@hasDecl(MLXConfig, "fromModelConfig"));
}

test "MLXConfig has writeToFile method" {
    try std.testing.expect(@hasDecl(MLXConfig, "writeToFile"));
}

test "MLXConfig has jsonStringify method" {
    try std.testing.expect(@hasDecl(MLXConfig, "jsonStringify"));
}

// =============================================================================
// QuantizationConfig Type Tests
// =============================================================================

test "MLXConfig.QuantizationConfig type is accessible" {
    const T = MLXConfig.QuantizationConfig;
    _ = T;
}

test "MLXConfig.QuantizationConfig has expected fields" {
    const info = @typeInfo(MLXConfig.QuantizationConfig);
    const fields = info.@"struct".fields;

    var has_group_size = false;
    var has_bits = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "group_size")) has_group_size = true;
        if (comptime std.mem.eql(u8, field.name, "bits")) has_bits = true;
    }

    try std.testing.expect(has_group_size);
    try std.testing.expect(has_bits);
}
