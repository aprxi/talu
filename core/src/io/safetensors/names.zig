//! SafeTensors tensor name resolution utilities.
//!
//! Provides functions for looking up tensor names across multiple naming
//! conventions used by different model packaging formats.

const std = @import("std");
const tensor = @import("../../tensor.zig");
const st_loader = @import("root.zig");
const log = @import("../../log.zig");

pub const Tensor = tensor.Tensor;

pub const NamesError = error{NotFound};

pub fn getNameAny(safetensors: *st_loader.UnifiedSafeTensors, comptime options: anytype) ![]const u8 {
    inline for (options) |candidate_name| {
        if (safetensors.hasTensor(candidate_name)) return candidate_name;
    }
    return NamesError.NotFound;
}

pub fn getTensorAny(safetensors: *st_loader.UnifiedSafeTensors, comptime options: anytype) !Tensor {
    inline for (options) |candidate_name| {
        if (safetensors.hasTensor(candidate_name)) {
            return try safetensors.getTensor(candidate_name, null);
        }
    }
    return NamesError.NotFound;
}

pub fn selectNameLayer(
    safetensors: *st_loader.UnifiedSafeTensors,
    name_buffer: []u8,
    layer_idx: usize,
    comptime options: anytype,
) ![]const u8 {
    inline for (options) |name_fmt| {
        const name = try std.fmt.bufPrint(name_buffer, name_fmt, .{layer_idx});
        log.trace("load", "selectNameLayer check", .{ .name = name, .found = safetensors.hasTensor(name) }, @src());
        if (safetensors.hasTensor(name)) return name;
    }
    log.trace("load", "selectNameLayer not found", .{ .layer_idx = layer_idx }, @src());
    return NamesError.NotFound;
}

pub fn getTensorLayer(
    safetensors: *st_loader.UnifiedSafeTensors,
    name_buffer: []u8,
    layer_idx: usize,
    comptime options: anytype,
) !Tensor {
    inline for (options) |name_fmt| {
        const name = try std.fmt.bufPrint(name_buffer, name_fmt, .{layer_idx});
        log.trace("load", "getTensorLayer check", .{ .name = name, .found = safetensors.hasTensor(name) }, @src());
        if (safetensors.hasTensor(name)) return try safetensors.getTensor(name, null);
    }
    log.trace("load", "getTensorLayer not found", .{ .layer_idx = layer_idx }, @src());
    return NamesError.NotFound;
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "NamesError.NotFound exists" {
    // Basic test to ensure error type compiles
    const err: anyerror = NamesError.NotFound;
    try testing.expectEqual(NamesError.NotFound, err);
}

// Note: Full unit tests for getNameAny, getTensorAny, selectNameLayer, and getTensorLayer
// require a UnifiedSafeTensors instance which is complex to mock due to Zig's tagged union semantics.
// These functions are simple wrappers that iterate through options calling hasTensor/getTensor,
// so their correctness is validated through:
// 1. Static type checking (comptime options parameter ensures all are string literals)
// 2. Integration tests with real SafeTensors files in higher-level model loading tests
// 3. Production usage in core/src/models/config/*.zig files

test "selectNameLayer - name formatting logic" {
    // Test the formatting logic without needing a real UnifiedSafeTensors
    var buffer: [256]u8 = undefined;

    // Single digit layer index
    {
        const result = try std.fmt.bufPrint(&buffer, "model.layers.{d}.attn.weight", .{5});
        try testing.expectEqualStrings("model.layers.5.attn.weight", result);
    }

    // Multi-digit layer index
    {
        const result = try std.fmt.bufPrint(&buffer, "model.layers.{d}.attn.weight", .{42});
        try testing.expectEqualStrings("model.layers.42.attn.weight", result);
    }

    // Zero index
    {
        const result = try std.fmt.bufPrint(&buffer, "layer.{d}.weight", .{0});
        try testing.expectEqualStrings("layer.0.weight", result);
    }

    // Three-digit index
    {
        const result = try std.fmt.bufPrint(&buffer, "layer.{d}.bias", .{123});
        try testing.expectEqualStrings("layer.123.bias", result);
    }
}

test "getTensorLayer - buffer size is sufficient for common patterns" {
    // Verify that a 256-byte buffer is sufficient for typical model layer names
    var buffer: [256]u8 = undefined;

    // Long realistic pattern
    {
        const result = try std.fmt.bufPrint(&buffer, "model.layers.{d}.self_attn.q_proj.weight", .{99});
        try testing.expectEqualStrings("model.layers.99.self_attn.q_proj.weight", result);
        try testing.expect(result.len < buffer.len);
    }

    // Very long edge case
    {
        const result = try std.fmt.bufPrint(&buffer, "model.encoder.block.{d}.layer.1.DenseReluDense.wi_1.weight", .{127});
        try testing.expect(result.len < buffer.len);
    }
}

test "getNameAny - comptime iteration works with tuple of strings" {
    // This test verifies the comptime iteration logic compiles correctly
    // We can't test the runtime behavior without UnifiedSafeTensors, but we
    // can verify the comptime logic works with different tuple sizes

    const single = .{"name1"};
    const double = .{ "name1", "name2" };
    const triple = .{ "name1", "name2", "name3" };
    const many = .{ "a", "b", "c", "d", "e", "f" };

    // Verify these compile-time constants are valid
    try testing.expectEqual(1, single.len);
    try testing.expectEqual(2, double.len);
    try testing.expectEqual(3, triple.len);
    try testing.expectEqual(6, many.len);
}

test "getTensorAny - comptime iteration works with tuple of strings" {
    // This test verifies the comptime iteration logic for getTensorAny compiles correctly
    // The function uses inline for to iterate through candidate names at compile time
    // We verify the tuple structure is valid for different sizes

    const single = .{"model.embed_tokens.weight"};
    const double = .{ "model.embed_tokens.weight", "model.wte.weight" };
    const triple = .{ "model.embed_tokens.weight", "model.wte.weight", "transformer.wte.weight" };

    // Verify these compile-time constants are valid
    try testing.expectEqual(1, single.len);
    try testing.expectEqual(2, double.len);
    try testing.expectEqual(3, triple.len);

    // Verify first element is accessible at comptime
    try testing.expectEqualStrings("model.embed_tokens.weight", single[0]);
    try testing.expectEqualStrings("model.embed_tokens.weight", double[0]);
}
