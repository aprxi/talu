//! Integration tests for xray.TraceCapture
//!
//! TraceCapture is the main capture storage that receives emissions from the
//! trace system and stores captured tensor data according to configuration.

const std = @import("std");
const main = @import("main");
const xray = main.xray;

const TraceCapture = xray.TraceCapture;
const TraceCaptureConfig = xray.TraceCaptureConfig;
const TracePointSet = xray.TracePointSet;
const TraceEmission = xray.TraceEmission;
const TracePoint = xray.TracePoint;

test "TraceCapture: init and deinit" {
    var config = TraceCaptureConfig{};
    config.points.logits = true;
    config.mode = .stats;

    var capture = TraceCapture.init(std.testing.allocator, config);
    defer capture.deinit();

    try std.testing.expectEqual(@as(usize, 0), capture.count());
}

test "TraceCapture: captures emissions matching filter" {
    var config = TraceCaptureConfig{};
    config.points.logits = true;
    config.mode = .stats;

    var capture = TraceCapture.init(std.testing.allocator, config);
    defer capture.deinit();

    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    capture.handleEmission(.{
        .point = .logits,
        .layer = TraceEmission.NO_LAYER,
        .token = 0,
        .position = 0,
        .tensor = .{
            .ptr = @ptrCast(&data),
            .dtype = .f32,
            .shape = .{ 4, 0, 0, 0 },
            .ndim = 1,
        },
        .timestamp_ns = 0,
    });

    try std.testing.expectEqual(@as(usize, 1), capture.count());

    const record = capture.get(0).?;
    try std.testing.expectEqual(TracePoint.logits, record.point);
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), record.stats.mean(), 0.001);
}

test "TraceCapture: filters by point" {
    var config = TraceCaptureConfig{};
    config.points.logits = true; // Only capture logits
    config.mode = .stats;

    var capture = TraceCapture.init(std.testing.allocator, config);
    defer capture.deinit();

    const data = [_]f32{ 1.0, 2.0 };

    // Emit embed (should NOT be captured)
    capture.handleEmission(.{
        .point = .embed,
        .layer = TraceEmission.NO_LAYER,
        .token = 0,
        .position = 0,
        .tensor = .{ .ptr = @ptrCast(&data), .dtype = .f32, .shape = .{ 2, 0, 0, 0 }, .ndim = 1 },
        .timestamp_ns = 0,
    });

    try std.testing.expectEqual(@as(usize, 0), capture.count());
}

test "TraceCapture: clear resets state" {
    var config = TraceCaptureConfig{};
    config.points = TracePointSet.all();
    config.mode = .stats;

    var capture = TraceCapture.init(std.testing.allocator, config);
    defer capture.deinit();

    const data = [_]f32{ 1.0 };
    capture.handleEmission(.{
        .point = .logits,
        .layer = TraceEmission.NO_LAYER,
        .token = 0,
        .position = 0,
        .tensor = .{ .ptr = @ptrCast(&data), .dtype = .f32, .shape = .{ 1, 0, 0, 0 }, .ndim = 1 },
        .timestamp_ns = 0,
    });

    try std.testing.expect(capture.count() > 0);

    capture.clear();

    try std.testing.expectEqual(@as(usize, 0), capture.count());
}

test "TraceCapture: get returns record by index" {
    var config = TraceCaptureConfig{};
    config.points = TracePointSet.all();
    config.mode = .stats;

    var capture = TraceCapture.init(std.testing.allocator, config);
    defer capture.deinit();

    const data = [_]f32{ 1.0, 2.0, 3.0 };

    capture.handleEmission(.{
        .point = .layer_attn_out,
        .layer = 5,
        .token = 0,
        .position = 0,
        .tensor = .{ .ptr = @ptrCast(&data), .dtype = .f32, .shape = .{ 3, 0, 0, 0 }, .ndim = 1 },
        .timestamp_ns = 0,
    });

    const record = capture.get(0);
    try std.testing.expect(record != null);
    try std.testing.expectEqual(TracePoint.layer_attn_out, record.?.point);
    try std.testing.expectEqual(@as(u16, 5), record.?.layer);

    // Out of bounds returns null
    try std.testing.expect(capture.get(1) == null);
}
