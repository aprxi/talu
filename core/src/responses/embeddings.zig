//! Embedding extraction for local response-serving models.
//!
//! Embeddings use the same LocalEngine ownership, model loading, and backend
//! execution contracts as text generation while exposing a small vector API.

const std = @import("std");
const local_mod = @import("local.zig");

pub const PoolingStrategy = local_mod.PoolingStrategy;

pub const Result = struct {
    values: []f32,

    pub fn deinit(self: Result, allocator: std.mem.Allocator) void {
        allocator.free(self.values);
    }
};

pub fn dimension(engine: *local_mod.LocalEngine) usize {
    return engine.embeddingDim();
}

pub fn extract(
    allocator: std.mem.Allocator,
    engine: *local_mod.LocalEngine,
    text: []const u8,
    pooling: PoolingStrategy,
    normalize: bool,
) !Result {
    const dim = engine.embeddingDim();
    const values = try allocator.alloc(f32, dim);
    errdefer allocator.free(values);

    try engine.embed(text, pooling, normalize, values);
    return .{ .values = values };
}
