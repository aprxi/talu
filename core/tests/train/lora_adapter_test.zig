//! Integration tests for LoRA adapter.

const std = @import("std");
const lib = @import("main");
const train = lib.train;
const LoraLayer = train.LoraLayer;
const LoraAdapter = train.LoraAdapter;
const LoraConfig = train.LoraConfig;

test "LoraLayer zero-initialized B means no initial effect" {
    const allocator = std.testing.allocator;
    var layer = try LoraLayer.init(allocator, "self_attn.q_proj.weight", 0, 64, 64, .{ .rank = 8, .alpha = 16.0 });
    defer layer.deinit();

    // B is all zeros, so any forward contribution would be zero
    for (layer.B.asSlice(f32)) |v| {
        try std.testing.expectEqual(@as(f32, 0.0), v);
    }

    // A should have non-zero values (Kaiming init)
    var any_nonzero = false;
    for (layer.A.asSlice(f32)) |v| {
        if (v != 0.0) {
            any_nonzero = true;
            break;
        }
    }
    try std.testing.expect(any_nonzero);
}

test "LoraAdapter explicit creation and lookup" {
    const allocator = std.testing.allocator;
    var adapter = LoraAdapter.initExplicit(allocator, .{ .rank = 4, .alpha = 8.0 });
    defer adapter.deinit();

    // Add layers for two different weights at different transformer blocks
    var l0 = try LoraLayer.init(allocator, "self_attn.q_proj.weight", 0, 32, 32, .{ .rank = 4, .alpha = 8.0 });
    errdefer l0.deinit();
    try adapter.addLayer(l0);

    var l1 = try LoraLayer.init(allocator, "self_attn.q_proj.weight", 1, 32, 32, .{ .rank = 4, .alpha = 8.0 });
    errdefer l1.deinit();
    try adapter.addLayer(l1);

    var l2 = try LoraLayer.init(allocator, "self_attn.v_proj.weight", 0, 32, 32, .{ .rank = 4, .alpha = 8.0 });
    errdefer l2.deinit();
    try adapter.addLayer(l2);

    try std.testing.expectEqual(@as(usize, 3), adapter.layerCount());

    // Lookup by weight_id + layer_index
    try std.testing.expect(adapter.getLayer("self_attn.q_proj.weight", 0) != null);
    try std.testing.expect(adapter.getLayer("self_attn.q_proj.weight", 1) != null);
    try std.testing.expect(adapter.getLayer("self_attn.v_proj.weight", 0) != null);

    // Should not find at wrong layer or wrong id
    try std.testing.expect(adapter.getLayer("self_attn.q_proj.weight", 2) == null);
    try std.testing.expect(adapter.getLayer("mlp.gate_proj.weight", 0) == null);
}

test "LoraAdapter trainableParamCount" {
    const allocator = std.testing.allocator;
    var adapter = LoraAdapter.initExplicit(allocator, .{ .rank = 4, .alpha = 8.0 });
    defer adapter.deinit();

    // in=16, out=16, rank=4 -> A=64, B=64, total per layer = 128
    var l1 = try LoraLayer.init(allocator, "a", 0, 16, 16, .{ .rank = 4, .alpha = 8.0 });
    errdefer l1.deinit();
    try adapter.addLayer(l1);

    var l2 = try LoraLayer.init(allocator, "b", 0, 16, 16, .{ .rank = 4, .alpha = 8.0 });
    errdefer l2.deinit();
    try adapter.addLayer(l2);

    try std.testing.expectEqual(@as(usize, 256), adapter.trainableParamCount());
}

test "LoraLayer scaling computation" {
    const allocator = std.testing.allocator;
    var layer = try LoraLayer.init(allocator, "test", 0, 32, 32, .{ .rank = 16, .alpha = 32.0 });
    defer layer.deinit();

    // scaling = alpha / rank = 32 / 16 = 2.0
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), layer.scaling, 1e-6);
}

test "LoraLayer A values bounded by Kaiming range" {
    const allocator = std.testing.allocator;
    const rank: u32 = 8;
    var layer = try LoraLayer.init(allocator, "test", 0, 64, 64, .{ .rank = rank, .alpha = 16.0 });
    defer layer.deinit();

    const bound: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(rank)));

    for (layer.A.asSlice(f32)) |v| {
        try std.testing.expect(v >= -bound);
        try std.testing.expect(v <= bound);
    }
}
