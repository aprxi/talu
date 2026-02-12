//! Integration tests for dump.Capture.
//!
//! Tests the Capture type exported from core/src/xray/dump/root.zig.
//! Capture is for dev-only full tensor dumping during inference.

const std = @import("std");
const main = @import("main");
const dump = main.dump;
const Capture = dump.Capture;

// ============================================================================
// Lifecycle Tests
// ============================================================================

test "Capture init and deinit" {
    var cap = Capture.init(std.testing.allocator);
    defer cap.deinit();

    try std.testing.expectEqual(@as(usize, 0), cap.tensors.items.len);
    try std.testing.expectEqual(@as(usize, 0), cap.total_bytes);
    try std.testing.expect(!cap.enabled);
}

// ============================================================================
// Enable/Disable Tests
// ============================================================================

test "Capture enable and disable" {
    var cap = Capture.init(std.testing.allocator);
    defer cap.deinit();

    try std.testing.expect(!cap.enabled);

    cap.enable();
    try std.testing.expect(cap.enabled);

    cap.disable();
    try std.testing.expect(!cap.enabled);
}

// ============================================================================
// Clear Tests
// ============================================================================

test "Capture clear resets state" {
    var cap = Capture.init(std.testing.allocator);
    defer cap.deinit();

    // Manually add a tensor to test clear
    const name = try std.testing.allocator.dupe(u8, "test_tensor");
    errdefer std.testing.allocator.free(name);
    const data = try std.testing.allocator.alloc(f32, 4);
    errdefer std.testing.allocator.free(data);
    @memset(data, 1.0);

    try cap.tensors.append(std.testing.allocator, .{
        .name = name,
        .data = data,
        .shape = .{ 4, 0, 0, 0 },
        .ndim = 1,
    });
    cap.total_bytes = 16;

    try std.testing.expectEqual(@as(usize, 1), cap.tensors.items.len);
    try std.testing.expectEqual(@as(usize, 16), cap.total_bytes);

    cap.clear();

    try std.testing.expectEqual(@as(usize, 0), cap.tensors.items.len);
    try std.testing.expectEqual(@as(usize, 0), cap.total_bytes);
}

// ============================================================================
// Layer Filter Tests
// ============================================================================

test "Capture setLayerFilter single layer" {
    var cap = Capture.init(std.testing.allocator);
    defer cap.deinit();

    try std.testing.expect(cap.layer_filter == null);
    try std.testing.expect(cap.layer_range_end == null);

    cap.setLayerFilter(5);

    try std.testing.expectEqual(@as(u16, 5), cap.layer_filter.?);
    try std.testing.expect(cap.layer_range_end == null);
}

test "Capture setLayerFilter null clears filter" {
    var cap = Capture.init(std.testing.allocator);
    defer cap.deinit();

    cap.setLayerFilter(5);
    try std.testing.expect(cap.layer_filter != null);

    cap.setLayerFilter(null);
    try std.testing.expect(cap.layer_filter == null);
}

test "Capture setLayerRange sets range" {
    var cap = Capture.init(std.testing.allocator);
    defer cap.deinit();

    cap.setLayerRange(2, 10);

    try std.testing.expectEqual(@as(u16, 2), cap.layer_filter.?);
    try std.testing.expectEqual(@as(u16, 10), cap.layer_range_end.?);
}

test "Capture setLayerFilter clears range" {
    var cap = Capture.init(std.testing.allocator);
    defer cap.deinit();

    cap.setLayerRange(2, 10);
    try std.testing.expect(cap.layer_range_end != null);

    cap.setLayerFilter(5);
    try std.testing.expect(cap.layer_range_end == null);
}

// ============================================================================
// Point Filter Tests
// ============================================================================

test "Capture setPointFilters empty allows all" {
    var cap = Capture.init(std.testing.allocator);
    defer cap.deinit();

    try std.testing.expectEqual(@as(usize, 0), cap.point_filters.len);

    cap.setPointFilters(&.{});
    try std.testing.expectEqual(@as(usize, 0), cap.point_filters.len);
}

test "Capture setPointFilters with values" {
    var cap = Capture.init(std.testing.allocator);
    defer cap.deinit();

    const filters = [_][]const u8{ "attn", "ffn" };
    cap.setPointFilters(&filters);

    try std.testing.expectEqual(@as(usize, 2), cap.point_filters.len);
    try std.testing.expectEqualStrings("attn", cap.point_filters[0]);
    try std.testing.expectEqualStrings("ffn", cap.point_filters[1]);
}

// ============================================================================
// Stop After Layer Tests
// ============================================================================

test "Capture setStopAfterLayer and shouldStop" {
    var cap = Capture.init(std.testing.allocator);
    defer cap.deinit();

    try std.testing.expect(!cap.shouldStop());
    try std.testing.expect(cap.stop_after_layer == null);

    cap.setStopAfterLayer(5);

    try std.testing.expectEqual(@as(u16, 5), cap.stop_after_layer.?);
    try std.testing.expect(!cap.stopped);
    try std.testing.expect(!cap.shouldStop());

    // Manually set stopped to test shouldStop
    cap.stopped = true;
    try std.testing.expect(cap.shouldStop());
}

test "Capture setStopAfterLayer null clears" {
    var cap = Capture.init(std.testing.allocator);
    defer cap.deinit();

    cap.setStopAfterLayer(5);
    cap.stopped = true;

    cap.setStopAfterLayer(null);

    try std.testing.expect(cap.stop_after_layer == null);
    try std.testing.expect(!cap.stopped);
}

// ============================================================================
// Record Tests (when disabled)
// ============================================================================

test "Capture record does nothing when disabled" {
    var cap = Capture.init(std.testing.allocator);
    defer cap.deinit();

    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    // Record when disabled should be a no-op
    try cap.record(
        "test_tensor",
        @ptrCast(&data),
        .f32,
        .{ 4, 0, 0, 0 },
        1,
    );

    try std.testing.expectEqual(@as(usize, 0), cap.tensors.items.len);
}

// ============================================================================
// Global Capture Tests
// ============================================================================

test "setGlobalCapture and clearGlobalCapture" {
    var cap = Capture.init(std.testing.allocator);
    defer cap.deinit();

    dump.capture.setGlobalCapture(&cap);
    defer dump.capture.clearGlobalCapture();

    // recordGlobal is a void function, just verify it doesn't crash
    const data = [_]f32{1.0};
    dump.capture.recordGlobal(
        .lm_head,
        0,
        @ptrCast(&data),
        .f32,
        .{ 1, 0, 0, 0 },
        1,
    );
}

test "isDumpEnabled returns build option" {
    // This just tests that the function is callable
    _ = dump.capture.isDumpEnabled();
}

test "shouldStopGlobal returns false when no global capture" {
    dump.capture.clearGlobalCapture();
    try std.testing.expect(!dump.capture.shouldStopGlobal());
}
