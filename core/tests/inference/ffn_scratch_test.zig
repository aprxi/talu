//! Integration tests for inference.FfnScratch
//!
//! Tests the FfnScratch struct which provides temporary buffers for FFN computation.

const std = @import("std");
const main = @import("main");

const FfnScratch = main.inference.backend.FfnScratch;

// =============================================================================
// Initialization Tests
// =============================================================================

test "FfnScratch initializes with empty slices" {
    const scratch = FfnScratch{};

    try std.testing.expectEqual(@as(usize, 0), scratch.gate.len);
    try std.testing.expectEqual(@as(usize, 0), scratch.gate_act.len);
    try std.testing.expectEqual(@as(usize, 0), scratch.up.len);
    try std.testing.expectEqual(@as(usize, 0), scratch.hidden.len);
}

// =============================================================================
// Memory Management Tests
// =============================================================================

test "FfnScratch.deinit frees all buffers" {
    const allocator = std.testing.allocator;

    var scratch = FfnScratch{};
    scratch.gate = try allocator.alloc(f32, 256);
    scratch.gate_act = try allocator.alloc(f32, 256);
    scratch.up = try allocator.alloc(f32, 256);
    scratch.hidden = try allocator.alloc(f32, 256);

    scratch.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), scratch.gate.len);
    try std.testing.expectEqual(@as(usize, 0), scratch.gate_act.len);
    try std.testing.expectEqual(@as(usize, 0), scratch.up.len);
    try std.testing.expectEqual(@as(usize, 0), scratch.hidden.len);
}

test "FfnScratch.deinit is safe on empty struct" {
    const allocator = std.testing.allocator;

    var scratch = FfnScratch{};
    scratch.deinit(allocator);
}

test "FfnScratch buffers can be allocated independently" {
    const allocator = std.testing.allocator;

    var scratch = FfnScratch{};
    scratch.gate = try allocator.alloc(f32, 512);
    scratch.up = try allocator.alloc(f32, 512);

    try std.testing.expectEqual(@as(usize, 512), scratch.gate.len);
    try std.testing.expectEqual(@as(usize, 512), scratch.up.len);
    try std.testing.expectEqual(@as(usize, 0), scratch.gate_act.len);
    try std.testing.expectEqual(@as(usize, 0), scratch.hidden.len);

    scratch.deinit(allocator);
}

test "FfnScratch handles large FFN dimensions" {
    const allocator = std.testing.allocator;
    const ffn_dim = 11008; // Typical 7B model FFN dimension

    var scratch = FfnScratch{};
    scratch.gate = try allocator.alloc(f32, ffn_dim);
    scratch.gate_act = try allocator.alloc(f32, ffn_dim);
    scratch.up = try allocator.alloc(f32, ffn_dim);
    scratch.hidden = try allocator.alloc(f32, ffn_dim);

    try std.testing.expectEqual(ffn_dim, scratch.gate.len);

    scratch.deinit(allocator);
}
