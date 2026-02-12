//! Integration tests for graph.Architecture

const std = @import("std");
const main = @import("main");

const graph = main.graph;
const Architecture = graph.Architecture;
const Op = graph.Op;

test "Architecture type is accessible" {
    _ = Architecture;
}

test "Architecture.isHeterogeneous returns false for homogeneous model" {
    const arch = Architecture{
        .name = "test_arch",
        .model_types = &.{},
        .block_ops = &.{},
        .block_variants = null,
    };

    try std.testing.expect(!arch.isHeterogeneous());
}

test "Architecture.isHeterogeneous returns true when block_variants set" {
    var variants: [1]graph.types.BlockVariant = .{.{
        .name = "attention",
        .ops = &.{},
    }};

    const arch = Architecture{
        .name = "test_arch",
        .model_types = &.{},
        .block_ops = &.{},
        .block_variants = &variants,
    };

    try std.testing.expect(arch.isHeterogeneous());
}

test "Architecture.getVariantIndex returns 0 for homogeneous model" {
    const arch = Architecture{
        .name = "test_arch",
        .model_types = &.{},
        .block_ops = &.{},
        .layer_map = null,
    };

    try std.testing.expectEqual(@as(u8, 0), arch.getVariantIndex(0));
    try std.testing.expectEqual(@as(u8, 0), arch.getVariantIndex(5));
    try std.testing.expectEqual(@as(u8, 0), arch.getVariantIndex(100));
}

test "Architecture.getVariantIndex uses layer_map when present" {
    const layer_map = [_]u8{ 0, 0, 1, 0, 1, 1 };

    const arch = Architecture{
        .name = "test_arch",
        .model_types = &.{},
        .block_ops = &.{},
        .layer_map = &layer_map,
    };

    try std.testing.expectEqual(@as(u8, 0), arch.getVariantIndex(0));
    try std.testing.expectEqual(@as(u8, 0), arch.getVariantIndex(1));
    try std.testing.expectEqual(@as(u8, 1), arch.getVariantIndex(2));
    try std.testing.expectEqual(@as(u8, 0), arch.getVariantIndex(3));
    try std.testing.expectEqual(@as(u8, 1), arch.getVariantIndex(4));
    try std.testing.expectEqual(@as(u8, 1), arch.getVariantIndex(5));
    // Out of bounds returns 0
    try std.testing.expectEqual(@as(u8, 0), arch.getVariantIndex(100));
}

test "Architecture.getVariant returns null for homogeneous model" {
    const arch = Architecture{
        .name = "test_arch",
        .model_types = &.{},
        .block_ops = &.{},
        .block_variants = null,
    };

    try std.testing.expect(arch.getVariant(0) == null);
}
