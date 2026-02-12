//! Integration tests for inference.executor.Embedding
//!
//! Tests the Embedding type from the executor module.

const std = @import("std");
const main = @import("main");

const Embedding = main.inference.executor.Embedding;

test "Embedding type is accessible" {
    const T = Embedding;
    _ = T;
}

test "Embedding has expected structure" {
    const info = @typeInfo(Embedding);
    try std.testing.expect(info == .@"struct");

    const fields = info.@"struct".fields;
    var has_weight = false;
    var has_vocab_size = false;
    var has_embed_dim = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "weight")) has_weight = true;
        if (comptime std.mem.eql(u8, field.name, "vocab_size")) has_vocab_size = true;
        if (comptime std.mem.eql(u8, field.name, "embed_dim")) has_embed_dim = true;
    }

    try std.testing.expect(has_weight);
    try std.testing.expect(has_vocab_size);
    try std.testing.expect(has_embed_dim);
}
