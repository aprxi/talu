//! Integration tests for inference.executor.Transformer
//!
//! Tests the Transformer type from the executor module, including:
//! - Transformer configuration and fields
//! - Type verification

const std = @import("std");
const main = @import("main");

const Transformer = main.inference.executor.Transformer;
const Block = main.inference.executor.Block;
const RMSNorm = main.inference.executor.RMSNorm;
const Linear = main.inference.executor.Linear;
const Embedding = main.inference.executor.Embedding;
const Tensor = main.Tensor;
const DType = main.DType;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "Transformer type is accessible" {
    const T = Transformer;
    _ = T;
}

test "Transformer is a struct" {
    const info = @typeInfo(Transformer);
    try std.testing.expect(info == .@"struct");
}

test "Transformer has expected fields" {
    const info = @typeInfo(Transformer);
    const fields = info.@"struct".fields;

    var has_model_type = false;
    var has_embed_tokens = false;
    var has_layers = false;
    var has_norm = false;
    var has_hidden_size = false;
    var has_vocab_size = false;
    var has_num_hidden_layers = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "model_type")) has_model_type = true;
        if (comptime std.mem.eql(u8, field.name, "embed_tokens")) has_embed_tokens = true;
        if (comptime std.mem.eql(u8, field.name, "layers")) has_layers = true;
        if (comptime std.mem.eql(u8, field.name, "norm")) has_norm = true;
        if (comptime std.mem.eql(u8, field.name, "hidden_size")) has_hidden_size = true;
        if (comptime std.mem.eql(u8, field.name, "vocab_size")) has_vocab_size = true;
        if (comptime std.mem.eql(u8, field.name, "num_hidden_layers")) has_num_hidden_layers = true;
    }

    try std.testing.expect(has_model_type);
    try std.testing.expect(has_embed_tokens);
    try std.testing.expect(has_layers);
    try std.testing.expect(has_norm);
    try std.testing.expect(has_hidden_size);
    try std.testing.expect(has_vocab_size);
    try std.testing.expect(has_num_hidden_layers);
}

test "Transformer has lm_head field" {
    const info = @typeInfo(Transformer);
    const fields = info.@"struct".fields;

    var has_lm_head = false;
    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "lm_head")) has_lm_head = true;
    }

    try std.testing.expect(has_lm_head);
}

test "Transformer has tie_word_embeddings field" {
    const info = @typeInfo(Transformer);
    const fields = info.@"struct".fields;

    var has_tie = false;
    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "tie_word_embeddings")) has_tie = true;
    }

    try std.testing.expect(has_tie);
}

// =============================================================================
// Method Existence Tests
// =============================================================================

test "Transformer has forward method" {
    try std.testing.expect(@hasDecl(Transformer, "forward"));
}

test "Transformer has summary method" {
    try std.testing.expect(@hasDecl(Transformer, "summary"));
}

test "Transformer has estimate method" {
    try std.testing.expect(@hasDecl(Transformer, "estimate"));
}

test "Transformer has build method" {
    try std.testing.expect(@hasDecl(Transformer, "build"));
}
