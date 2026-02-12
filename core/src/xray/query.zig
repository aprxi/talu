//! Query Engine for Captured Tensors
//!
//! Provides query operations over captured tensor data.

const std = @import("std");
const trace = @import("trace.zig");
const capture = @import("capture.zig");

pub const TensorStats = capture.TensorStats;
pub const CapturedTensor = capture.CapturedTensor;

/// Condition for finding tensors.
pub const Condition = enum {
    /// Has any NaN values
    has_nan,
    /// Has any Inf values
    has_inf,
    /// Has any anomalies (NaN or Inf)
    has_anomaly,
    /// Max value above threshold
    max_above,
    /// Min value below threshold
    min_below,
    /// Mean outside range
    mean_outside,
};

/// Location of a found tensor.
pub const Location = struct {
    point: trace.TracePoint,
    layer: u16,
    token: u32,
    position: u32,
};

/// Comparison result between two tensors.
pub const CompareResult = struct {
    /// Maximum absolute difference
    max_abs_diff: f32,
    /// Mean absolute difference
    mean_abs_diff: f32,
    /// Cosine similarity (-1 to 1)
    cosine_similarity: f32,
    /// Stats of first tensor
    a_stats: TensorStats,
    /// Stats of second tensor
    b_stats: TensorStats,
};

/// Query operations on a capture.
pub const CaptureQuery = struct {
    cap: *const capture.TraceCapture,

    pub fn init(cap: *const capture.TraceCapture) CaptureQuery {
        return .{ .cap = cap };
    }

    /// Get a specific tensor by point, layer, token.
    pub fn get(
        self: CaptureQuery,
        point: trace.TracePoint,
        layer: ?u16,
        token: ?u32,
    ) ?*const CapturedTensor {
        var iter = self.cap.find(point, layer, token);
        return iter.next();
    }

    /// Find first tensor matching a condition.
    pub fn findFirst(
        self: CaptureQuery,
        condition: Condition,
        threshold: f32,
    ) ?Location {
        for (self.cap.records.items) |*record| {
            if (matchesCondition(record.stats, condition, threshold)) {
                return .{
                    .point = record.point,
                    .layer = record.layer,
                    .token = record.token,
                    .position = record.position,
                };
            }
        }
        return null;
    }

    /// Find all tensors matching a condition.
    pub fn findAll(
        self: CaptureQuery,
        condition: Condition,
        threshold: f32,
        allocator: std.mem.Allocator,
    ) ![]Location {
        var results = std.ArrayList(Location).init(allocator);
        errdefer results.deinit();

        for (self.cap.records.items) |*record| {
            if (matchesCondition(record.stats, condition, threshold)) {
                try results.append(.{
                    .point = record.point,
                    .layer = record.layer,
                    .token = record.token,
                    .position = record.position,
                });
            }
        }

        return results.toOwnedSlice();
    }

    /// Count tensors matching criteria.
    pub fn count(
        self: CaptureQuery,
        point: ?trace.TracePoint,
        layer: ?u16,
        token: ?u32,
    ) usize {
        var n: usize = 0;
        for (self.cap.records.items) |*record| {
            if (point) |p| {
                if (record.point != p) continue;
            }
            if (layer) |l| {
                if (record.layer != l) continue;
            }
            if (token) |t| {
                if (record.token != t) continue;
            }
            n += 1;
        }
        return n;
    }

    /// Get all layer outputs for a specific token.
    pub fn layerOutputs(self: CaptureQuery, token: u32) LayerOutputIterator {
        return .{
            .cap = self.cap,
            .token = token,
            .index = 0,
        };
    }

    pub const LayerOutputIterator = struct {
        cap: *const capture.TraceCapture,
        token: u32,
        index: usize,

        pub fn next(self: *LayerOutputIterator) ?*const CapturedTensor {
            while (self.index < self.cap.records.items.len) {
                const record = &self.cap.records.items[self.index];
                self.index += 1;

                if (record.point == .block_out and record.token == self.token) {
                    return record;
                }
            }
            return null;
        }
    };

    /// Compare two captured tensors by stats.
    pub fn compareStats(a: *const CapturedTensor, b: *const CapturedTensor) CompareResult {
        return .{
            .max_abs_diff = @abs(a.stats.max - b.stats.max),
            .mean_abs_diff = @abs(a.stats.mean() - b.stats.mean()),
            .cosine_similarity = computeCosineSimilarity(a, b),
            .a_stats = a.stats,
            .b_stats = b.stats,
        };
    }
};

fn matchesCondition(stats: TensorStats, condition: Condition, threshold: f32) bool {
    return switch (condition) {
        .has_nan => stats.nan_count > 0,
        .has_inf => stats.inf_count > 0,
        .has_anomaly => stats.nan_count > 0 or stats.inf_count > 0,
        .max_above => stats.max > threshold,
        .min_below => stats.min < threshold,
        .mean_outside => @abs(stats.mean()) > threshold,
    };
}

fn computeCosineSimilarity(a: *const CapturedTensor, b: *const CapturedTensor) f32 {
    // If we have samples, use them for cosine similarity
    if (a.samples != null and b.samples != null) {
        const as = a.samples.?;
        const bs = b.samples.?;
        const n = @min(as.len, bs.len);

        var dot: f64 = 0;
        var a_sq: f64 = 0;
        var b_sq: f64 = 0;

        for (0..n) |i| {
            const av: f64 = as[i];
            const bv: f64 = bs[i];
            dot += av * bv;
            a_sq += av * av;
            b_sq += bv * bv;
        }

        const denom = @sqrt(a_sq) * @sqrt(b_sq);
        if (denom == 0) return 0;
        return @floatCast(dot / denom);
    }

    // Fall back to comparing l2 norms
    const a_norm = a.stats.l2Norm();
    const b_norm = b.stats.l2Norm();
    if (a_norm == 0 or b_norm == 0) return 0;

    // Can't compute real cosine similarity without samples
    // Return 1.0 if norms are similar, less otherwise
    const ratio = @min(a_norm, b_norm) / @max(a_norm, b_norm);
    return ratio;
}

// ============================================================================
// Tests
// ============================================================================

test "CaptureQuery get" {
    var config = capture.TraceCaptureConfig{};
    config.points = capture.TracePointSet.all();

    var cap = capture.TraceCapture.init(std.testing.allocator, config);
    defer cap.deinit();

    const data = [_]f32{ 1.0, 2.0, 3.0 };

    cap.handleEmission(.{
        .point = .lm_head,
        .layer = trace.TraceEmission.NO_LAYER,
        .token = 0,
        .position = 5,
        .tensor = .{ .ptr = @ptrCast(&data), .dtype = .f32, .shape = .{ 3, 0, 0, 0 }, .ndim = 1 },
        .timestamp_ns = 0,
        .kernel_name = std.mem.zeroes([48]u8),
    });

    const q = CaptureQuery.init(&cap);
    const result = q.get(.lm_head, null, 0);
    try std.testing.expect(result != null);
    try std.testing.expectEqual(trace.TracePoint.lm_head, result.?.point);
}

test "CaptureQuery findFirst" {
    var config = capture.TraceCaptureConfig{};
    config.points = capture.TracePointSet.all();

    var cap = capture.TraceCapture.init(std.testing.allocator, config);
    defer cap.deinit();

    // Add normal tensor
    const normal = [_]f32{ 1.0, 2.0, 3.0 };
    cap.handleEmission(.{
        .point = .attn_out,
        .layer = 0,
        .token = 0,
        .position = 0,
        .tensor = .{ .ptr = @ptrCast(&normal), .dtype = .f32, .shape = .{ 3, 0, 0, 0 }, .ndim = 1 },
        .timestamp_ns = 0,
        .kernel_name = std.mem.zeroes([48]u8),
    });

    // Add tensor with NaN
    const with_nan = [_]f32{ 1.0, std.math.nan(f32), 3.0 };
    cap.handleEmission(.{
        .point = .attn_out,
        .layer = 5,
        .token = 0,
        .position = 0,
        .tensor = .{ .ptr = @ptrCast(&with_nan), .dtype = .f32, .shape = .{ 3, 0, 0, 0 }, .ndim = 1 },
        .timestamp_ns = 0,
        .kernel_name = std.mem.zeroes([48]u8),
    });

    const q = CaptureQuery.init(&cap);
    const nan_loc = q.findFirst(.has_nan, 0);
    try std.testing.expect(nan_loc != null);
    try std.testing.expectEqual(@as(u16, 5), nan_loc.?.layer);
}

test "CaptureQuery count" {
    var config = capture.TraceCaptureConfig{};
    config.points = capture.TracePointSet.all();

    var cap = capture.TraceCapture.init(std.testing.allocator, config);
    defer cap.deinit();

    const data = [_]f32{ 1.0 };

    // Add 3 layer_attn_out records
    for ([_]u16{ 0, 1, 2 }) |layer| {
        cap.handleEmission(.{
            .point = .attn_out,
            .layer = layer,
            .token = 0,
            .position = 0,
            .tensor = .{ .ptr = @ptrCast(&data), .dtype = .f32, .shape = .{ 1, 0, 0, 0 }, .ndim = 1 },
            .timestamp_ns = 0,
            .kernel_name = std.mem.zeroes([48]u8),
        });
    }

    // Add 1 logits record
    cap.handleEmission(.{
        .point = .lm_head,
        .layer = trace.TraceEmission.NO_LAYER,
        .token = 0,
        .position = 0,
        .tensor = .{ .ptr = @ptrCast(&data), .dtype = .f32, .shape = .{ 1, 0, 0, 0 }, .ndim = 1 },
        .timestamp_ns = 0,
        .kernel_name = std.mem.zeroes([48]u8),
    });

    const q = CaptureQuery.init(&cap);
    try std.testing.expectEqual(@as(usize, 4), q.count(null, null, null));
    try std.testing.expectEqual(@as(usize, 3), q.count(.attn_out, null, null));
    try std.testing.expectEqual(@as(usize, 1), q.count(.lm_head, null, null));
    try std.testing.expectEqual(@as(usize, 1), q.count(.attn_out, 1, null));
}
