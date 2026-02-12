//! Tensor Capture System
//!
//! Captures tensor data during inference based on configuration.
//! Uses own allocator - does not share memory with inference.

const std = @import("std");
const trace = @import("trace.zig");
const stats_mod = @import("stats.zig");

pub const TensorStats = stats_mod.TensorStats;

/// What data to capture from each tensor.
pub const TraceCaptureMode = enum(u8) {
    /// Statistics only: min, max, mean, rms, nan_count, inf_count
    stats,
    /// Statistics + first N tensor values
    sample,
    /// Complete tensor data copy
    full,
};

/// Filter for which layers to capture.
pub const LayerFilter = union(enum) {
    /// Capture all layers
    all,
    /// Capture specific layer range [start, end)
    range: struct { start: u16, end: u16 },
    /// Capture specific layers
    list: []const u16,

    pub fn matches(self: LayerFilter, layer: u16) bool {
        if (layer == trace.TraceEmission.NO_LAYER) return true; // Always capture non-layer points
        return switch (self) {
            .all => true,
            .range => |r| layer >= r.start and layer < r.end,
            .list => |list| for (list) |l| {
                if (l == layer) return true;
            } else false,
        };
    }
};

/// Filter for which tokens to capture.
pub const TokenFilter = union(enum) {
    /// Capture all tokens
    all,
    /// Capture token range [start, end)
    range: struct { start: u32, end: u32 },
    /// Capture specific tokens
    list: []const u32,

    pub fn matches(self: TokenFilter, token: u32) bool {
        return switch (self) {
            .all => true,
            .range => |r| token >= r.start and token < r.end,
            .list => |list| for (list) |t| {
                if (t == token) return true;
            } else false,
        };
    }
};

/// Bitset for which trace points to capture.
pub const TracePointSet = packed struct {
    embed: bool = false,
    embed_pos: bool = false,
    layer_input: bool = false,
    layer_attn_norm: bool = false,
    attn_q: bool = false,
    attn_k: bool = false,
    attn_v: bool = false,
    attn_qk: bool = false,
    attn_weights: bool = false,
    attn_out: bool = false,
    layer_ffn_norm: bool = false,
    ffn_gate: bool = false,
    ffn_up: bool = false,
    ffn_act: bool = false,
    ffn_down: bool = false,
    block_out: bool = false,
    mamba_out: bool = false,
    conv_in_proj: bool = false,
    conv_conv: bool = false,
    conv_out_proj: bool = false,
    final_norm: bool = false,
    lm_head: bool = false,
    logits_scaled: bool = false,
    _padding: u9 = 0,

    pub fn all() TracePointSet {
        return .{
            .embed = true,
            .embed_pos = true,
            .layer_input = true,
            .layer_attn_norm = true,
            .attn_q = true,
            .attn_k = true,
            .attn_v = true,
            .attn_qk = true,
            .attn_weights = true,
            .attn_out = true,
            .layer_ffn_norm = true,
            .ffn_gate = true,
            .ffn_up = true,
            .ffn_act = true,
            .ffn_down = true,
            .block_out = true,
            .mamba_out = true,
            .conv_in_proj = true,
            .conv_conv = true,
            .conv_out_proj = true,
            .final_norm = true,
            .lm_head = true,
            .logits_scaled = true,
        };
    }

    pub fn none() TracePointSet {
        return .{};
    }

    pub fn contains(self: TracePointSet, point: trace.TracePoint) bool {
        return switch (point) {
            .embed => self.embed,
            .embed_pos => self.embed_pos,
            .layer_input => self.layer_input,
            .layer_attn_norm => self.layer_attn_norm,
            .attn_q => self.attn_q,
            .attn_k => self.attn_k,
            .attn_v => self.attn_v,
            .attn_qk => self.attn_qk,
            .attn_weights => self.attn_weights,
            .attn_out => self.attn_out,
            .layer_ffn_norm => self.layer_ffn_norm,
            .ffn_gate => self.ffn_gate,
            .ffn_up => self.ffn_up,
            .ffn_act => self.ffn_act,
            .ffn_down => self.ffn_down,
            .block_out => self.block_out,
            .mamba_out => self.mamba_out,
            .conv_in_proj => self.conv_in_proj,
            .conv_conv => self.conv_conv,
            .conv_out_proj => self.conv_out_proj,
            .final_norm => self.final_norm,
            .lm_head => self.lm_head,
            .logits_scaled => self.logits_scaled,
            _ => false, // Custom points not in standard set
        };
    }
};

/// Configuration for what to capture.
pub const TraceCaptureConfig = struct {
    /// Which trace points to capture
    points: TracePointSet = TracePointSet.none(),
    /// Which layers (null = all)
    layers: LayerFilter = .all,
    /// Which tokens (null = all)
    tokens: TokenFilter = .all,
    /// What data to capture
    mode: TraceCaptureMode = .stats,
    /// For sample mode: how many values to capture
    sample_count: u32 = 8,
    /// Memory limit in bytes (null = unlimited)
    memory_limit: ?usize = null,
};

/// A captured tensor record.
pub const CapturedTensor = struct {
    /// Where in the pipeline
    point: trace.TracePoint,
    /// Which layer (NO_LAYER for non-layer points)
    layer: u16,
    /// Token index
    token: u32,
    /// Position in sequence
    position: u32,
    /// Timestamp
    timestamp_ns: i128,
    /// Shape
    shape: [4]u32,
    /// Number of dimensions
    ndim: u8,
    /// Data type
    dtype: trace.DType,
    /// Kernel name that produced this tensor (null-terminated)
    kernel_name: [48]u8,
    /// Computed statistics
    stats: TensorStats,
    /// Sample values (if mode was sample or full)
    samples: ?[]f32,
    /// Full tensor data (if mode was full)
    data: ?[]u8,
};

/// Capture storage - holds all captured tensors.
pub const TraceCapture = struct {
    allocator: std.mem.Allocator,
    config: TraceCaptureConfig,
    records: std.ArrayList(CapturedTensor),
    memory_used: usize,
    overflow: bool,

    pub fn init(allocator: std.mem.Allocator, config: TraceCaptureConfig) TraceCapture {
        return .{
            .allocator = allocator,
            .config = config,
            .records = .empty,
            .memory_used = 0,
            .overflow = false,
        };
    }

    pub fn deinit(self: *TraceCapture) void {
        for (self.records.items) |*record| {
            if (record.samples) |samples| {
                self.allocator.free(samples);
            }
            if (record.data) |data| {
                self.allocator.free(data);
            }
        }
        self.records.deinit(self.allocator);
    }

    /// Clear all captured data (for reuse).
    pub fn clear(self: *TraceCapture) void {
        for (self.records.items) |*record| {
            if (record.samples) |samples| {
                self.allocator.free(samples);
            }
            if (record.data) |data| {
                self.allocator.free(data);
            }
        }
        self.records.clearRetainingCapacity();
        self.memory_used = 0;
        self.overflow = false;
    }

    /// Handle an emission from the trace system.
    pub fn handleEmission(self: *TraceCapture, emission: trace.TraceEmission) void {
        // Check if we should capture this emission
        if (!self.config.points.contains(emission.point)) return;
        if (!self.config.layers.matches(emission.layer)) return;
        if (!self.config.tokens.matches(emission.token)) return;

        // Check memory limit
        const record_size = self.estimateRecordSize(emission.tensor);
        if (self.config.memory_limit) |limit| {
            if (self.memory_used + record_size > limit) {
                self.overflow = true;
                return;
            }
        }

        // Compute statistics
        const tensor_stats = stats_mod.compute(emission.tensor);

        // Capture samples if requested
        var samples: ?[]f32 = null;
        if (self.config.mode == .sample or self.config.mode == .full) {
            const sample_len = @min(self.config.sample_count, @as(u32, @intCast(emission.tensor.elementCount())));
            if (sample_len > 0) {
                samples = self.allocator.alloc(f32, sample_len) catch null;
                if (samples) |s| {
                    self.copySamples(emission.tensor, s);
                }
            }
        }

        // Capture full data if requested
        var data: ?[]u8 = null;
        if (self.config.mode == .full) {
            const byte_size = emission.tensor.byteSize();
            if (byte_size > 0) {
                data = self.allocator.alloc(u8, byte_size) catch null;
                if (data) |d| {
                    @memcpy(d, emission.tensor.ptr[0..byte_size]);
                }
            }
        }

        // Create record
        const record = CapturedTensor{
            .point = emission.point,
            .layer = emission.layer,
            .token = emission.token,
            .position = emission.position,
            .timestamp_ns = emission.timestamp_ns,
            .shape = emission.tensor.shape,
            .ndim = emission.tensor.ndim,
            .dtype = emission.tensor.dtype,
            .kernel_name = emission.kernel_name,
            .stats = tensor_stats,
            .samples = samples,
            .data = data,
        };

        self.records.append(self.allocator, record) catch {
            // Failed to store - free any allocated memory
            if (samples) |s| self.allocator.free(s);
            if (data) |d| self.allocator.free(d);
            self.overflow = true;
            return;
        };

        self.memory_used += record_size;
    }

    fn estimateRecordSize(self: *const TraceCapture, tensor: trace.TracedTensor) usize {
        var size: usize = @sizeOf(CapturedTensor);
        if (self.config.mode == .sample or self.config.mode == .full) {
            const sample_len = @min(self.config.sample_count, @as(u32, @intCast(tensor.elementCount())));
            size += sample_len * @sizeOf(f32);
        }
        if (self.config.mode == .full) {
            size += tensor.byteSize();
        }
        return size;
    }

    fn copySamples(self: *const TraceCapture, tensor: trace.TracedTensor, dest: []f32) void {
        _ = self;
        const n = dest.len;
        switch (tensor.dtype) {
            .f32 => {
                const src: [*]const f32 = @ptrCast(@alignCast(tensor.ptr));
                for (0..n) |i| {
                    dest[i] = src[i];
                }
            },
            .f16 => {
                const src: [*]const f16 = @ptrCast(@alignCast(tensor.ptr));
                for (0..n) |i| {
                    dest[i] = @floatCast(src[i]);
                }
            },
            .bf16 => {
                const src: [*]const u16 = @ptrCast(@alignCast(tensor.ptr));
                for (0..n) |i| {
                    const bits: u32 = @as(u32, src[i]) << 16;
                    dest[i] = @bitCast(bits);
                }
            },
            else => {
                // Unsupported dtype - fill with zeros
                @memset(dest, 0);
            },
        }
    }

    /// Get number of captured records.
    pub fn count(self: *const TraceCapture) usize {
        return self.records.items.len;
    }

    /// Get a captured record by index.
    pub fn get(self: *const TraceCapture, index: usize) ?*const CapturedTensor {
        if (index >= self.records.items.len) return null;
        return &self.records.items[index];
    }

    /// Find records matching criteria.
    pub fn find(
        self: *const TraceCapture,
        point: ?trace.TracePoint,
        layer: ?u16,
        token: ?u32,
    ) FindIterator {
        return FindIterator{
            .capture = self,
            .point = point,
            .layer = layer,
            .token = token,
            .index = 0,
        };
    }

    pub const FindIterator = struct {
        capture: *const TraceCapture,
        point: ?trace.TracePoint,
        layer: ?u16,
        token: ?u32,
        index: usize,

        pub fn next(self: *FindIterator) ?*const CapturedTensor {
            while (self.index < self.capture.records.items.len) {
                const record = &self.capture.records.items[self.index];
                self.index += 1;

                if (self.point) |p| {
                    if (record.point != p) continue;
                }
                if (self.layer) |l| {
                    if (record.layer != l) continue;
                }
                if (self.token) |t| {
                    if (record.token != t) continue;
                }
                return record;
            }
            return null;
        }
    };
};

/// Global capture instance (set by Inspector).
/// Single-threaded: only accessed from main thread via enable()/disable().
var global_capture: ?*TraceCapture = null;

/// Handler function that routes to global capture.
fn globalHandler(emission: trace.TraceEmission) void {
    if (global_capture) |cap| {
        cap.handleEmission(emission);
    }
}

/// Enable capturing with the given capture instance.
pub fn enable(cap: *TraceCapture) void {
    global_capture = cap;
    trace.setHandler(&globalHandler);
}

/// Disable capturing.
pub fn disable() void {
    trace.setHandler(null);
    global_capture = null;
}

/// Check if capturing is enabled.
pub fn isEnabled() bool {
    return global_capture != null;
}

// ============================================================================
// Tests
// ============================================================================

test "TracePointSet operations" {
    const all = TracePointSet.all();
    try std.testing.expect(all.contains(.embed));
    try std.testing.expect(all.contains(.lm_head));
    try std.testing.expect(all.contains(.attn_out));

    const none = TracePointSet.none();
    try std.testing.expect(!none.contains(.embed));
    try std.testing.expect(!none.contains(.lm_head));

    var custom = TracePointSet.none();
    custom.lm_head = true;
    custom.attn_out = true;
    try std.testing.expect(custom.contains(.lm_head));
    try std.testing.expect(custom.contains(.attn_out));
    try std.testing.expect(!custom.contains(.embed));
}

test "LayerFilter" {
    const all: LayerFilter = .all;
    try std.testing.expect(LayerFilter.matches(all, 0));
    try std.testing.expect(LayerFilter.matches(all, 31));

    const range: LayerFilter = .{ .range = .{ .start = 10, .end = 20 } };
    try std.testing.expect(!LayerFilter.matches(range, 9));
    try std.testing.expect(LayerFilter.matches(range, 10));
    try std.testing.expect(LayerFilter.matches(range, 15));
    try std.testing.expect(!LayerFilter.matches(range, 20));

    const list: LayerFilter = .{ .list = &[_]u16{ 0, 10, 31 } };
    try std.testing.expect(LayerFilter.matches(list, 0));
    try std.testing.expect(LayerFilter.matches(list, 10));
    try std.testing.expect(LayerFilter.matches(list, 31));
    try std.testing.expect(!LayerFilter.matches(list, 5));
}

test "TraceCapture basic flow" {
    var config = TraceCaptureConfig{};
    config.points.lm_head = true;
    config.mode = .stats;

    var cap = TraceCapture.init(std.testing.allocator, config);
    defer cap.deinit();

    // Simulate an emission
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const emission = trace.TraceEmission{
        .point = .lm_head,
        .layer = trace.TraceEmission.NO_LAYER,
        .token = 0,
        .position = 5,
        .tensor = .{
            .ptr = @ptrCast(&data),
            .dtype = .f32,
            .shape = .{ 4, 0, 0, 0 },
            .ndim = 1,
        },
        .timestamp_ns = 12345,
        .kernel_name = std.mem.zeroes([48]u8),
    };

    cap.handleEmission(emission);

    try std.testing.expectEqual(@as(usize, 1), cap.count());
    const record = cap.get(0).?;
    try std.testing.expectEqual(trace.TracePoint.lm_head, record.point);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), record.stats.min, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), record.stats.max, 0.001);
}

test "TraceCapture filtering" {
    var config = TraceCaptureConfig{};
    config.points.attn_out = true; // Only capture attention output
    config.layers = .{ .range = .{ .start = 5, .end = 10 } }; // Only layers 5-9

    var cap = TraceCapture.init(std.testing.allocator, config);
    defer cap.deinit();

    const data = [_]f32{ 1.0, 2.0 };

    // Should NOT capture: wrong point
    cap.handleEmission(.{
        .point = .lm_head,
        .layer = 5,
        .token = 0,
        .position = 0,
        .tensor = .{ .ptr = @ptrCast(&data), .dtype = .f32, .shape = .{ 2, 0, 0, 0 }, .ndim = 1 },
        .timestamp_ns = 0,
        .kernel_name = std.mem.zeroes([48]u8),
    });
    try std.testing.expectEqual(@as(usize, 0), cap.count());

    // Should NOT capture: layer out of range
    cap.handleEmission(.{
        .point = .attn_out,
        .layer = 4,
        .token = 0,
        .position = 0,
        .tensor = .{ .ptr = @ptrCast(&data), .dtype = .f32, .shape = .{ 2, 0, 0, 0 }, .ndim = 1 },
        .timestamp_ns = 0,
        .kernel_name = std.mem.zeroes([48]u8),
    });
    try std.testing.expectEqual(@as(usize, 0), cap.count());

    // Should capture: right point, right layer
    cap.handleEmission(.{
        .point = .attn_out,
        .layer = 7,
        .token = 0,
        .position = 0,
        .tensor = .{ .ptr = @ptrCast(&data), .dtype = .f32, .shape = .{ 2, 0, 0, 0 }, .ndim = 1 },
        .timestamp_ns = 0,
        .kernel_name = std.mem.zeroes([48]u8),
    });
    try std.testing.expectEqual(@as(usize, 1), cap.count());
}

test "TraceCapture with samples" {
    var config = TraceCaptureConfig{};
    config.points.lm_head = true;
    config.mode = .sample;
    config.sample_count = 4;

    var cap = TraceCapture.init(std.testing.allocator, config);
    defer cap.deinit();

    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    cap.handleEmission(.{
        .point = .lm_head,
        .layer = trace.TraceEmission.NO_LAYER,
        .token = 0,
        .position = 0,
        .tensor = .{ .ptr = @ptrCast(&data), .dtype = .f32, .shape = .{ 8, 0, 0, 0 }, .ndim = 1 },
        .timestamp_ns = 0,
        .kernel_name = std.mem.zeroes([48]u8),
    });

    const record = cap.get(0).?;
    try std.testing.expect(record.samples != null);
    try std.testing.expectEqual(@as(usize, 4), record.samples.?.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), record.samples.?[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), record.samples.?[3], 0.001);
}

test "TraceCapture find iterator" {
    var config = TraceCaptureConfig{};
    config.points = TracePointSet.all();

    var cap = TraceCapture.init(std.testing.allocator, config);
    defer cap.deinit();

    const data = [_]f32{ 1.0 };

    // Add multiple records
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

    try std.testing.expectEqual(@as(usize, 3), cap.count());

    // Find specific layer
    var iter = cap.find(.attn_out, 1, null);
    const found = iter.next();
    try std.testing.expect(found != null);
    try std.testing.expectEqual(@as(u16, 1), found.?.layer);
    try std.testing.expect(iter.next() == null);
}

test "enable/disable global capture" {
    var config = TraceCaptureConfig{};
    config.points.lm_head = true;

    var cap = TraceCapture.init(std.testing.allocator, config);
    defer cap.deinit();

    try std.testing.expect(!isEnabled());

    enable(&cap);
    try std.testing.expect(isEnabled());
    try std.testing.expect(trace.isEnabled());

    disable();
    try std.testing.expect(!isEnabled());
    try std.testing.expect(!trace.isEnabled());
}
