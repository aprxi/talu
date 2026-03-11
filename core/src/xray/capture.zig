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
    /// Metadata/timing only: no tensor stats or value sampling.
    timing,
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
    logits_ready: bool = false,
    token_select: bool = false,
    ffn_act_map: bool = false,
    ffn_act_mix: bool = false,
    gdelta_in_proj: bool = false,
    gdelta_conv: bool = false,
    gdelta_ssm: bool = false,
    gdelta_norm: bool = false,
    gdelta_out: bool = false,

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
            .ffn_act_map = true,
            .ffn_act_mix = true,
            .ffn_down = true,
            .block_out = true,
            .mamba_out = true,
            .conv_in_proj = true,
            .conv_conv = true,
            .conv_out_proj = true,
            .final_norm = true,
            .lm_head = true,
            .logits_scaled = true,
            .logits_ready = true,
            .token_select = true,
            .gdelta_in_proj = true,
            .gdelta_conv = true,
            .gdelta_ssm = true,
            .gdelta_norm = true,
            .gdelta_out = true,
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
            .ffn_act_map => self.ffn_act_map,
            .ffn_act_mix => self.ffn_act_mix,
            .ffn_down => self.ffn_down,
            .block_out => self.block_out,
            .mamba_out => self.mamba_out,
            .conv_in_proj => self.conv_in_proj,
            .conv_conv => self.conv_conv,
            .conv_out_proj => self.conv_out_proj,
            .final_norm => self.final_norm,
            .lm_head => self.lm_head,
            .logits_scaled => self.logits_scaled,
            .logits_ready => self.logits_ready,
            .token_select => self.token_select,
            .gdelta_in_proj => self.gdelta_in_proj,
            .gdelta_conv => self.gdelta_conv,
            .gdelta_ssm => self.gdelta_ssm,
            .gdelta_norm => self.gdelta_norm,
            .gdelta_out => self.gdelta_out,
            _ => false, // Custom points not in standard set
        };
    }

    pub fn builtinMask(self: TracePointSet) u64 {
        var mask: u64 = 0;
        if (self.embed) mask |= @as(u64, 1) << 0;
        if (self.embed_pos) mask |= @as(u64, 1) << 1;
        if (self.layer_input) mask |= @as(u64, 1) << 2;
        if (self.layer_attn_norm) mask |= @as(u64, 1) << 3;
        if (self.attn_q) mask |= @as(u64, 1) << 4;
        if (self.attn_k) mask |= @as(u64, 1) << 5;
        if (self.attn_v) mask |= @as(u64, 1) << 6;
        if (self.attn_qk) mask |= @as(u64, 1) << 7;
        if (self.attn_weights) mask |= @as(u64, 1) << 8;
        if (self.attn_out) mask |= @as(u64, 1) << 9;
        if (self.layer_ffn_norm) mask |= @as(u64, 1) << 10;
        if (self.ffn_gate) mask |= @as(u64, 1) << 11;
        if (self.ffn_up) mask |= @as(u64, 1) << 12;
        if (self.ffn_act) mask |= @as(u64, 1) << 13;
        if (self.ffn_down) mask |= @as(u64, 1) << 14;
        if (self.block_out) mask |= @as(u64, 1) << 15;
        if (self.mamba_out) mask |= @as(u64, 1) << 16;
        if (self.conv_in_proj) mask |= @as(u64, 1) << 17;
        if (self.conv_conv) mask |= @as(u64, 1) << 18;
        if (self.conv_out_proj) mask |= @as(u64, 1) << 19;
        if (self.final_norm) mask |= @as(u64, 1) << 20;
        if (self.lm_head) mask |= @as(u64, 1) << 21;
        if (self.logits_scaled) mask |= @as(u64, 1) << 22;
        if (self.logits_ready) mask |= @as(u64, 1) << 23;
        if (self.token_select) mask |= @as(u64, 1) << 24;
        if (self.ffn_act_map) mask |= @as(u64, 1) << 25;
        if (self.ffn_act_mix) mask |= @as(u64, 1) << 26;
        if (self.gdelta_in_proj) mask |= @as(u64, 1) << 27;
        if (self.gdelta_conv) mask |= @as(u64, 1) << 28;
        if (self.gdelta_ssm) mask |= @as(u64, 1) << 29;
        if (self.gdelta_norm) mask |= @as(u64, 1) << 30;
        if (self.gdelta_out) mask |= @as(u64, 1) << 31;
        return mask;
    }

    pub fn fromBuiltinMask(mask: u64) TracePointSet {
        return .{
            .embed = (mask & (@as(u64, 1) << 0)) != 0,
            .embed_pos = (mask & (@as(u64, 1) << 1)) != 0,
            .layer_input = (mask & (@as(u64, 1) << 2)) != 0,
            .layer_attn_norm = (mask & (@as(u64, 1) << 3)) != 0,
            .attn_q = (mask & (@as(u64, 1) << 4)) != 0,
            .attn_k = (mask & (@as(u64, 1) << 5)) != 0,
            .attn_v = (mask & (@as(u64, 1) << 6)) != 0,
            .attn_qk = (mask & (@as(u64, 1) << 7)) != 0,
            .attn_weights = (mask & (@as(u64, 1) << 8)) != 0,
            .attn_out = (mask & (@as(u64, 1) << 9)) != 0,
            .layer_ffn_norm = (mask & (@as(u64, 1) << 10)) != 0,
            .ffn_gate = (mask & (@as(u64, 1) << 11)) != 0,
            .ffn_up = (mask & (@as(u64, 1) << 12)) != 0,
            .ffn_act = (mask & (@as(u64, 1) << 13)) != 0,
            .ffn_down = (mask & (@as(u64, 1) << 14)) != 0,
            .block_out = (mask & (@as(u64, 1) << 15)) != 0,
            .mamba_out = (mask & (@as(u64, 1) << 16)) != 0,
            .conv_in_proj = (mask & (@as(u64, 1) << 17)) != 0,
            .conv_conv = (mask & (@as(u64, 1) << 18)) != 0,
            .conv_out_proj = (mask & (@as(u64, 1) << 19)) != 0,
            .final_norm = (mask & (@as(u64, 1) << 20)) != 0,
            .lm_head = (mask & (@as(u64, 1) << 21)) != 0,
            .logits_scaled = (mask & (@as(u64, 1) << 22)) != 0,
            .logits_ready = (mask & (@as(u64, 1) << 23)) != 0,
            .token_select = (mask & (@as(u64, 1) << 24)) != 0,
            .ffn_act_map = (mask & (@as(u64, 1) << 25)) != 0,
            .ffn_act_mix = (mask & (@as(u64, 1) << 26)) != 0,
            .gdelta_in_proj = (mask & (@as(u64, 1) << 27)) != 0,
            .gdelta_conv = (mask & (@as(u64, 1) << 28)) != 0,
            .gdelta_ssm = (mask & (@as(u64, 1) << 29)) != 0,
            .gdelta_norm = (mask & (@as(u64, 1) << 30)) != 0,
            .gdelta_out = (mask & (@as(u64, 1) << 31)) != 0,
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
    /// Trust that non-CPU backend emissions are host-readable pointers.
    /// Use only when emitters guarantee host accessibility (e.g. verify mode).
    allow_non_cpu_host_data: bool = false,
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
    /// Backend that emitted this trace point.
    backend: trace.Backend,
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
    /// Runtime-provided exact work counters.
    work_flops: u64,
    work_bytes: u64,
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

    fn kernelNameSlice(name: [48]u8) []const u8 {
        const len = std.mem.indexOfScalar(u8, &name, 0) orelse name.len;
        return name[0..len];
    }

    /// Handle an emission from the trace system.
    pub fn handleEmission(self: *TraceCapture, emission: trace.TraceEmission) void {
        // Check if we should capture this emission
        if (!self.config.points.contains(emission.point)) return;
        if (!self.config.layers.matches(emission.layer)) return;
        if (!self.config.tokens.matches(emission.token)) return;

        // Check memory limit
        const record_size = self.estimateRecordSize(emission.backend, emission.tensor);
        if (self.config.memory_limit) |limit| {
            if (self.memory_used + record_size > limit) {
                self.overflow = true;
                return;
            }
        }

        // Non-CPU backends may emit device pointers that are not host-readable.
        // Only read them when emitter explicitly marks host accessibility.
        const kernel_name = kernelNameSlice(emission.kernel_name);
        const non_cpu_host_readable = emission.backend != .cpu and
            self.config.allow_non_cpu_host_data and
            std.mem.endsWith(u8, kernel_name, "_host");
        const can_read_tensor = emission.backend == .cpu or non_cpu_host_readable;
        const tensor_stats = if (can_read_tensor and self.config.mode != .timing)
            stats_mod.compute(emission.tensor)
        else
            TensorStats.EMPTY;

        // Capture samples if requested
        var samples: ?[]f32 = null;
        if (can_read_tensor and (self.config.mode == .sample or self.config.mode == .full)) {
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
        if (can_read_tensor and self.config.mode == .full) {
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
            .backend = emission.backend,
            .timestamp_ns = emission.timestamp_ns,
            .shape = emission.tensor.shape,
            .ndim = emission.tensor.ndim,
            .dtype = emission.tensor.dtype,
            .kernel_name = emission.kernel_name,
            .work_flops = emission.work_flops,
            .work_bytes = emission.work_bytes,
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

    fn estimateRecordSize(self: *const TraceCapture, backend: trace.Backend, tensor: trace.TracedTensor) usize {
        var size: usize = @sizeOf(CapturedTensor);
        const can_read_tensor = backend == .cpu or self.config.allow_non_cpu_host_data;
        if (can_read_tensor and (self.config.mode == .sample or self.config.mode == .full)) {
            const sample_len = @min(self.config.sample_count, @as(u32, @intCast(tensor.elementCount())));
            size += sample_len * @sizeOf(f32);
        }
        if (can_read_tensor and self.config.mode == .full) {
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

/// Global capture instance used by trace handler callbacks.
/// Must be atomic because backend emissions can race with enable/disable.
var global_capture_atomic: std.atomic.Value(?*TraceCapture) = std.atomic.Value(?*TraceCapture).init(null);

/// Handler function that routes to global capture.
fn globalHandler(emission: trace.TraceEmission) void {
    if (global_capture_atomic.load(.acquire)) |cap| {
        cap.handleEmission(emission);
    }
}

/// Enable capturing with the given capture instance.
pub fn enable(cap: *TraceCapture) void {
    global_capture_atomic.store(cap, .release);
    trace.setActiveBuiltInPointMask(cap.config.points.builtinMask());
    trace.setHandler(&globalHandler);
}

/// Disable capturing.
pub fn disable() void {
    global_capture_atomic.store(null, .release);
    trace.setHandler(null);
}

/// Check if capturing is enabled.
pub fn isEnabled() bool {
    return global_capture_atomic.load(.acquire) != null;
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

test "TracePointSet builtinMask/fromBuiltinMask round-trip" {
    var points = TracePointSet.none();
    points.layer_attn_norm = true;
    points.block_out = true;
    points.lm_head = true;
    points.token_select = true;
    points.gdelta_out = true;

    const mask = points.builtinMask();
    const round_trip = TracePointSet.fromBuiltinMask(mask);

    try std.testing.expect(round_trip.contains(.layer_attn_norm));
    try std.testing.expect(round_trip.contains(.block_out));
    try std.testing.expect(round_trip.contains(.lm_head));
    try std.testing.expect(round_trip.contains(.token_select));
    try std.testing.expect(round_trip.contains(.gdelta_out));
    try std.testing.expect(!round_trip.contains(.attn_out));
    try std.testing.expect(!round_trip.contains(.ffn_act));
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
        .backend = .cpu,
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
        .backend = .cpu,
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
        .backend = .cpu,
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
        .backend = .cpu,
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
        .backend = .cpu,
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

    const data = [_]f32{1.0};

    // Add multiple records
    for ([_]u16{ 0, 1, 2 }) |layer| {
        cap.handleEmission(.{
            .point = .attn_out,
            .layer = layer,
            .token = 0,
            .position = 0,
            .backend = .cpu,
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
