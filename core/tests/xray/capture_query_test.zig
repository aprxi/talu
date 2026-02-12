//! Integration tests for xray.CaptureQuery
//!
//! CaptureQuery provides query operations on a TraceCapture instance.
//! Allows getting records by point/layer/token, counting, and finding
//! records matching conditions.

const std = @import("std");
const main = @import("main");
const xray = main.xray;

const CaptureQuery = xray.CaptureQuery;
const TraceCapture = xray.TraceCapture;
const TraceCaptureConfig = xray.TraceCaptureConfig;
const TracePointSet = xray.TracePointSet;
const TraceEmission = xray.TraceEmission;
const TracePoint = xray.TracePoint;

test "CaptureQuery: get by point and layer" {
    var config = TraceCaptureConfig{};
    config.points = TracePointSet.all();
    config.mode = .stats;

    var capture = TraceCapture.init(std.testing.allocator, config);
    defer capture.deinit();

    const data = [_]f32{ 1.0, 2.0, 3.0 };

    // Add records at different layers
    capture.handleEmission(.{
        .point = .layer_attn_out,
        .layer = 0,
        .token = 0,
        .position = 0,
        .tensor = .{ .ptr = @ptrCast(&data), .dtype = .f32, .shape = .{ 3, 0, 0, 0 }, .ndim = 1 },
        .timestamp_ns = 0,
    });

    capture.handleEmission(.{
        .point = .layer_attn_out,
        .layer = 1,
        .token = 0,
        .position = 0,
        .tensor = .{ .ptr = @ptrCast(&data), .dtype = .f32, .shape = .{ 3, 0, 0, 0 }, .ndim = 1 },
        .timestamp_ns = 0,
    });

    const query = CaptureQuery.init(&capture);

    // Query specific layer
    const result = query.get(.layer_attn_out, 1, 0);
    try std.testing.expect(result != null);
    try std.testing.expectEqual(@as(u16, 1), result.?.layer);
}

test "CaptureQuery: count with no filters" {
    var config = TraceCaptureConfig{};
    config.points = TracePointSet.all();
    config.mode = .stats;

    var capture = TraceCapture.init(std.testing.allocator, config);
    defer capture.deinit();

    const data = [_]f32{ 1.0 };

    // Add 3 layer records
    for (0..3) |layer| {
        capture.handleEmission(.{
            .point = .layer_attn_out,
            .layer = @intCast(layer),
            .token = 0,
            .position = 0,
            .tensor = .{ .ptr = @ptrCast(&data), .dtype = .f32, .shape = .{ 1, 0, 0, 0 }, .ndim = 1 },
            .timestamp_ns = 0,
        });
    }

    // Add 1 logits record
    capture.handleEmission(.{
        .point = .logits,
        .layer = TraceEmission.NO_LAYER,
        .token = 0,
        .position = 0,
        .tensor = .{ .ptr = @ptrCast(&data), .dtype = .f32, .shape = .{ 1, 0, 0, 0 }, .ndim = 1 },
        .timestamp_ns = 0,
    });

    const query = CaptureQuery.init(&capture);

    // Count all
    try std.testing.expectEqual(@as(usize, 4), query.count(null, null, null));
}

test "CaptureQuery: count with point filter" {
    var config = TraceCaptureConfig{};
    config.points = TracePointSet.all();
    config.mode = .stats;

    var capture = TraceCapture.init(std.testing.allocator, config);
    defer capture.deinit();

    const data = [_]f32{ 1.0 };

    // Add 3 layer_attn_out records
    for (0..3) |layer| {
        capture.handleEmission(.{
            .point = .layer_attn_out,
            .layer = @intCast(layer),
            .token = 0,
            .position = 0,
            .tensor = .{ .ptr = @ptrCast(&data), .dtype = .f32, .shape = .{ 1, 0, 0, 0 }, .ndim = 1 },
            .timestamp_ns = 0,
        });
    }

    // Add 1 logits record
    capture.handleEmission(.{
        .point = .logits,
        .layer = TraceEmission.NO_LAYER,
        .token = 0,
        .position = 0,
        .tensor = .{ .ptr = @ptrCast(&data), .dtype = .f32, .shape = .{ 1, 0, 0, 0 }, .ndim = 1 },
        .timestamp_ns = 0,
    });

    const query = CaptureQuery.init(&capture);

    // Count by point
    try std.testing.expectEqual(@as(usize, 3), query.count(.layer_attn_out, null, null));
    try std.testing.expectEqual(@as(usize, 1), query.count(.logits, null, null));
}

test "CaptureQuery: count with layer filter" {
    var config = TraceCaptureConfig{};
    config.points = TracePointSet.all();
    config.mode = .stats;

    var capture = TraceCapture.init(std.testing.allocator, config);
    defer capture.deinit();

    const data = [_]f32{ 1.0 };

    // Add records at different layers
    for (0..3) |layer| {
        capture.handleEmission(.{
            .point = .layer_attn_out,
            .layer = @intCast(layer),
            .token = 0,
            .position = 0,
            .tensor = .{ .ptr = @ptrCast(&data), .dtype = .f32, .shape = .{ 1, 0, 0, 0 }, .ndim = 1 },
            .timestamp_ns = 0,
        });
    }

    const query = CaptureQuery.init(&capture);

    // Count by layer
    try std.testing.expectEqual(@as(usize, 1), query.count(.layer_attn_out, 0, null));
    try std.testing.expectEqual(@as(usize, 1), query.count(.layer_attn_out, 1, null));
    try std.testing.expectEqual(@as(usize, 1), query.count(.layer_attn_out, 2, null));
}
