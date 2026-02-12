//! Integration tests for inference.SamplingWorkspace
//!
//! Tests the Workspace struct used by the Sampler for efficient
//! per-sample allocation-free operation.

const std = @import("std");
const main = @import("main");
const SamplingWorkspace = main.inference.SamplingWorkspace;

// =============================================================================
// Initialization Tests
// =============================================================================

test "SamplingWorkspace.init allocates correct buffer sizes" {
    const allocator = std.testing.allocator;
    const vocab_size = 1000;

    var workspace = try SamplingWorkspace.init(allocator, vocab_size);
    defer workspace.deinit();

    // Verify buffers are allocated with correct size
    try std.testing.expectEqual(vocab_size, workspace.probabilities.len);
    try std.testing.expectEqual(vocab_size, workspace.sorted_indices.len);
}

test "SamplingWorkspace.init with small vocab" {
    const allocator = std.testing.allocator;
    const vocab_size = 10;

    var workspace = try SamplingWorkspace.init(allocator, vocab_size);
    defer workspace.deinit();

    try std.testing.expectEqual(vocab_size, workspace.probabilities.len);
    try std.testing.expectEqual(vocab_size, workspace.sorted_indices.len);
}

test "SamplingWorkspace.init with large vocab" {
    const allocator = std.testing.allocator;
    const vocab_size = 152064; // Typical LLM vocab size (Qwen)

    var workspace = try SamplingWorkspace.init(allocator, vocab_size);
    defer workspace.deinit();

    try std.testing.expectEqual(vocab_size, workspace.probabilities.len);
    try std.testing.expectEqual(vocab_size, workspace.sorted_indices.len);
}

test "SamplingWorkspace.init with zero vocab size" {
    const allocator = std.testing.allocator;

    var workspace = try SamplingWorkspace.init(allocator, 0);
    defer workspace.deinit();

    try std.testing.expectEqual(@as(usize, 0), workspace.probabilities.len);
    try std.testing.expectEqual(@as(usize, 0), workspace.sorted_indices.len);
}

// =============================================================================
// Memory Management Tests
// =============================================================================

test "SamplingWorkspace.deinit frees memory correctly" {
    const allocator = std.testing.allocator;

    var workspace = try SamplingWorkspace.init(allocator, 100);
    workspace.deinit();

    // After deinit, workspace is undefined - this test passes if no memory leak
}

test "SamplingWorkspace buffers are writable" {
    const allocator = std.testing.allocator;
    const vocab_size = 100;

    var workspace = try SamplingWorkspace.init(allocator, vocab_size);
    defer workspace.deinit();

    // Should be able to write to probabilities buffer
    for (workspace.probabilities, 0..) |*p, i| {
        p.* = @floatFromInt(i);
    }

    // Verify writes
    try std.testing.expectEqual(@as(f32, 0.0), workspace.probabilities[0]);
    try std.testing.expectEqual(@as(f32, 99.0), workspace.probabilities[99]);
}

test "SamplingWorkspace stores allocator reference" {
    const allocator = std.testing.allocator;

    var workspace = try SamplingWorkspace.init(allocator, 50);
    defer workspace.deinit();

    // Allocator should be stored for deinit
    try std.testing.expect(@intFromPtr(workspace.allocator.ptr) == @intFromPtr(allocator.ptr));
}

// =============================================================================
// Thread Safety Tests (each thread gets own workspace)
// =============================================================================

test "SamplingWorkspace can be created per-thread" {
    const allocator = std.testing.allocator;
    const vocab_size = 100;

    // Simulate multiple threads each with their own workspace
    var workspaces: [4]SamplingWorkspace = undefined;
    var initialized: usize = 0;

    errdefer {
        for (workspaces[0..initialized]) |*ws| {
            ws.deinit();
        }
    }

    for (&workspaces) |*ws| {
        ws.* = try SamplingWorkspace.init(allocator, vocab_size);
        initialized += 1;
    }

    // Each workspace should have independent buffers
    for (&workspaces, 0..) |*ws, i| {
        for (ws.probabilities) |*p| {
            p.* = @floatFromInt(i);
        }
    }

    // Verify independence
    try std.testing.expectEqual(@as(f32, 0.0), workspaces[0].probabilities[0]);
    try std.testing.expectEqual(@as(f32, 1.0), workspaces[1].probabilities[0]);
    try std.testing.expectEqual(@as(f32, 2.0), workspaces[2].probabilities[0]);
    try std.testing.expectEqual(@as(f32, 3.0), workspaces[3].probabilities[0]);

    for (&workspaces) |*ws| {
        ws.deinit();
    }
}
