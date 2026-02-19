//! Integration tests for graph.config.ModelDescription
//!
//! ModelDescription is a C-compatible struct containing model metadata
//! for C API introspection. It owns allocated architecture strings.

const std = @import("std");
const main = @import("main");

const ModelDescription = main.models.dispatcher.config.ModelDescription;
const QuantMethod = main.models.dispatcher.config.QuantMethod;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "ModelDescription type is accessible" {
    const T = ModelDescription;
    _ = T;
}

test "ModelDescription is a struct" {
    const info = @typeInfo(ModelDescription);
    try std.testing.expect(info == .@"struct");
}

test "ModelDescription has expected dimension fields" {
    const info = @typeInfo(ModelDescription);
    const fields = info.@"struct".fields;

    var has_vocab_size = false;
    var has_hidden_size = false;
    var has_num_layers = false;
    var has_num_heads = false;
    var has_num_kv_heads = false;
    var has_intermediate_size = false;
    var has_max_seq_len = false;
    var has_head_dim = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "vocab_size")) has_vocab_size = true;
        if (comptime std.mem.eql(u8, field.name, "hidden_size")) has_hidden_size = true;
        if (comptime std.mem.eql(u8, field.name, "num_layers")) has_num_layers = true;
        if (comptime std.mem.eql(u8, field.name, "num_heads")) has_num_heads = true;
        if (comptime std.mem.eql(u8, field.name, "num_kv_heads")) has_num_kv_heads = true;
        if (comptime std.mem.eql(u8, field.name, "intermediate_size")) has_intermediate_size = true;
        if (comptime std.mem.eql(u8, field.name, "max_seq_len")) has_max_seq_len = true;
        if (comptime std.mem.eql(u8, field.name, "head_dim")) has_head_dim = true;
    }

    try std.testing.expect(has_vocab_size);
    try std.testing.expect(has_hidden_size);
    try std.testing.expect(has_num_layers);
    try std.testing.expect(has_num_heads);
    try std.testing.expect(has_num_kv_heads);
    try std.testing.expect(has_intermediate_size);
    try std.testing.expect(has_max_seq_len);
    try std.testing.expect(has_head_dim);
}

test "ModelDescription has expected quantization fields" {
    const info = @typeInfo(ModelDescription);
    const fields = info.@"struct".fields;

    var has_quant_bits = false;
    var has_quant_group_size = false;
    var has_quant_method = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "quant_bits")) has_quant_bits = true;
        if (comptime std.mem.eql(u8, field.name, "quant_group_size")) has_quant_group_size = true;
        if (comptime std.mem.eql(u8, field.name, "quant_method")) has_quant_method = true;
    }

    try std.testing.expect(has_quant_bits);
    try std.testing.expect(has_quant_group_size);
    try std.testing.expect(has_quant_method);
}

test "ModelDescription has expected architecture fields" {
    const info = @typeInfo(ModelDescription);
    const fields = info.@"struct".fields;

    var has_model_type = false;
    var has_architecture = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "model_type")) has_model_type = true;
        if (comptime std.mem.eql(u8, field.name, "architecture")) has_architecture = true;
    }

    try std.testing.expect(has_model_type);
    try std.testing.expect(has_architecture);
}

test "ModelDescription has expected flag fields" {
    const info = @typeInfo(ModelDescription);
    const fields = info.@"struct".fields;

    var has_tie_word_embeddings = false;
    var has_use_gelu = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "tie_word_embeddings")) has_tie_word_embeddings = true;
        if (comptime std.mem.eql(u8, field.name, "use_gelu")) has_use_gelu = true;
    }

    try std.testing.expect(has_tie_word_embeddings);
    try std.testing.expect(has_use_gelu);
}

test "ModelDescription has expected MoE fields" {
    const info = @typeInfo(ModelDescription);
    const fields = info.@"struct".fields;

    var has_num_experts = false;
    var has_experts_per_token = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "num_experts")) has_num_experts = true;
        if (comptime std.mem.eql(u8, field.name, "experts_per_token")) has_experts_per_token = true;
    }

    try std.testing.expect(has_num_experts);
    try std.testing.expect(has_experts_per_token);
}

// =============================================================================
// Method Tests
// =============================================================================

test "ModelDescription has fromDir method" {
    try std.testing.expect(@hasDecl(ModelDescription, "fromDir"));
}

test "ModelDescription has deinit method" {
    try std.testing.expect(@hasDecl(ModelDescription, "deinit"));
}

// =============================================================================
// deinit Tests
// =============================================================================

test "ModelDescription.deinit clears architecture strings" {
    const allocator = std.testing.allocator;

    // Create a description with allocated strings
    const model_type_buf = try allocator.allocSentinel(u8, 5, 0);
    @memcpy(model_type_buf, "qwen3");

    const arch_buf = try allocator.allocSentinel(u8, 10, 0);
    @memcpy(arch_buf, "Qwen2Model");

    var desc = ModelDescription{
        .vocab_size = 32000,
        .hidden_size = 4096,
        .num_layers = 32,
        .num_heads = 32,
        .num_kv_heads = 8,
        .intermediate_size = 11008,
        .max_seq_len = 4096,
        .head_dim = 128,
        .rope_theta = 10000.0,
        .norm_eps = 1e-5,
        .quant_bits = 4,
        .quant_group_size = 64,
        .quant_method = .gaffine,
        .model_type = model_type_buf,
        .architecture = arch_buf,
        .tie_word_embeddings = false,
        .use_gelu = false,
        .num_experts = 0,
        .experts_per_token = 0,
    };

    desc.deinit(allocator);

    // After deinit, strings should be null
    try std.testing.expectEqual(@as(?[:0]u8, null), desc.model_type);
    try std.testing.expectEqual(@as(?[:0]u8, null), desc.architecture);
}

test "ModelDescription.deinit handles null strings" {
    const allocator = std.testing.allocator;

    var desc = ModelDescription{
        .vocab_size = 32000,
        .hidden_size = 4096,
        .num_layers = 32,
        .num_heads = 32,
        .num_kv_heads = 8,
        .intermediate_size = 11008,
        .max_seq_len = 4096,
        .head_dim = 128,
        .rope_theta = 10000.0,
        .norm_eps = 1e-5,
        .quant_bits = 16,
        .quant_group_size = 0,
        .quant_method = .none,
        .model_type = null,
        .architecture = null,
        .tie_word_embeddings = true,
        .use_gelu = false,
        .num_experts = 0,
        .experts_per_token = 0,
    };

    // Should not panic
    desc.deinit(allocator);

    try std.testing.expectEqual(@as(?[:0]u8, null), desc.model_type);
    try std.testing.expectEqual(@as(?[:0]u8, null), desc.architecture);
}

// =============================================================================
// QuantMethod Tests
// =============================================================================

test "QuantMethod type is accessible" {
    const T = QuantMethod;
    _ = T;
}

test "QuantMethod has expected variants" {
    try std.testing.expect(@intFromEnum(QuantMethod.none) == 0);
    try std.testing.expect(@intFromEnum(QuantMethod.gaffine) == 1);
    try std.testing.expect(@intFromEnum(QuantMethod.mxfp4) == 2);
    try std.testing.expect(@intFromEnum(QuantMethod.native) == 3);
}
