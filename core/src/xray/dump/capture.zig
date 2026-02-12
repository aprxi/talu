//! Tensor Capture
//!
//! Captures full tensor data during inference for offline comparison.
//! This is heavy-weight instrumentation - only used in the dump binary.
//!
//! Design: Thread-local global capture state. When enabled, kernel code
//! calls capture.record() to store tensor data.

const std = @import("std");
const build_options = @import("build_options");
const trace = @import("../trace.zig");
const dtype_mod = @import("../../dtype.zig");

pub const DumpEnabled = build_options.dump_tensors;

/// Maximum number of tensors that can be captured in one run.
const MAX_CAPTURES = 4096;

/// Maximum total bytes across all captures (1GB limit).
const MAX_TOTAL_BYTES: usize = 1024 * 1024 * 1024;

/// A captured tensor record.
pub const CapturedTensor = struct {
    /// Trace point name (e.g., "layer0.attn_out")
    name: []const u8,
    /// Tensor data (owned, f32 for consistency)
    data: []f32,
    /// Shape dimensions
    shape: [4]usize,
    /// Number of dimensions
    ndim: u8,

    pub fn deinit(self: *CapturedTensor, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
        allocator.free(self.name);
    }
};

/// Global capture state.
pub const Capture = struct {
    allocator: std.mem.Allocator,
    tensors: std.ArrayListUnmanaged(CapturedTensor),
    total_bytes: usize,
    enabled: bool,

    // Filters (null = capture all)
    layer_filter: ?u16, // Only capture this layer (null = all)
    layer_range_end: ?u16, // If set with layer_filter, capture range [layer_filter, layer_range_end]
    point_filters: []const []const u8, // Only capture points containing these substrings (empty = all)
    stop_after_layer: ?u16, // Signal to stop execution after this layer
    stopped: bool, // Set when stop_after_layer is reached

    pub fn init(allocator: std.mem.Allocator) Capture {
        return .{
            .allocator = allocator,
            .tensors = .{},
            .total_bytes = 0,
            .enabled = false,
            .layer_filter = null,
            .layer_range_end = null,
            .point_filters = &.{},
            .stop_after_layer = null,
            .stopped = false,
        };
    }

    pub fn deinit(self: *Capture) void {
        for (self.tensors.items) |*t| {
            t.deinit(self.allocator);
        }
        self.tensors.deinit(self.allocator);
    }

    pub fn enable(self: *Capture) void {
        self.enabled = true;
    }

    pub fn disable(self: *Capture) void {
        self.enabled = false;
    }

    pub fn clear(self: *Capture) void {
        for (self.tensors.items) |*t| {
            t.deinit(self.allocator);
        }
        self.tensors.clearRetainingCapacity();
        self.total_bytes = 0;
    }

    /// Set layer filter (only capture this layer, or null for all).
    pub fn setLayerFilter(self: *Capture, layer: ?u16) void {
        self.layer_filter = layer;
        self.layer_range_end = null; // Single layer mode
    }

    /// Set layer range filter (capture layers in [start, end] inclusive).
    pub fn setLayerRange(self: *Capture, start: u16, end: u16) void {
        self.layer_filter = start;
        self.layer_range_end = end;
    }

    /// Set point filters (capture only points containing any of these substrings).
    /// Pass empty slice to capture all points.
    pub fn setPointFilters(self: *Capture, filters: []const []const u8) void {
        self.point_filters = filters;
    }

    /// Set stop-after-layer (signals executor to stop after this layer).
    pub fn setStopAfterLayer(self: *Capture, layer: ?u16) void {
        self.stop_after_layer = layer;
        self.stopped = false;
    }

    /// Check if we should stop execution (called by executor after each layer).
    pub fn shouldStop(self: *const Capture) bool {
        return self.stopped;
    }

    /// Check if a layer passes the layer filter.
    fn passesLayerFilter(self: *const Capture, layer: u16) bool {
        if (self.layer_filter) |start| {
            if (self.layer_range_end) |end| {
                // Range mode
                return layer >= start and layer <= end;
            } else {
                // Single layer mode
                return layer == start;
            }
        }
        return true; // No filter
    }

    /// Check if a point name passes the point filters.
    fn passesPointFilter(self: *const Capture, name: []const u8) bool {
        if (self.point_filters.len == 0) return true;
        for (self.point_filters) |filter| {
            if (std.mem.indexOf(u8, name, filter) != null) return true;
        }
        return false;
    }

    /// Record a tensor at a trace point.
    /// The name format is "layer{N}.{point}" or just "{point}" for non-layer points.
    pub fn record(
        self: *Capture,
        name: []const u8,
        data_ptr: [*]const u8,
        dtype: dtype_mod.DType,
        shape: [4]usize,
        ndim: u8,
    ) !void {
        if (!self.enabled) return;
        if (self.tensors.items.len >= MAX_CAPTURES) return error.TooManyCaptures;

        // Calculate element count and byte size
        var numel: usize = 1;
        for (0..ndim) |i| {
            numel *= shape[i];
        }

        const src_byte_size = numel * dtype.elementSize();
        const dst_byte_size = numel * @sizeOf(f32);

        if (self.total_bytes + dst_byte_size > MAX_TOTAL_BYTES) return error.CaptureLimitExceeded;

        // Allocate and copy name
        const name_copy = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_copy);

        // Allocate destination buffer (always f32)
        const dst_data = try self.allocator.alloc(f32, numel);
        errdefer self.allocator.free(dst_data);

        // Convert source data to f32
        switch (dtype) {
            .f32 => {
                const src: [*]const f32 = @ptrCast(@alignCast(data_ptr));
                @memcpy(dst_data, src[0..numel]);
            },
            .bf16 => {
                const src: [*]const u16 = @ptrCast(@alignCast(data_ptr));
                for (0..numel) |i| {
                    dst_data[i] = bf16ToF32(src[i]);
                }
            },
            .f16 => {
                const src: [*]const u16 = @ptrCast(@alignCast(data_ptr));
                for (0..numel) |i| {
                    dst_data[i] = fp16ToF32(src[i]);
                }
            },
            else => {
                // Unsupported dtype - zero fill
                @memset(dst_data, 0);
            },
        }

        try self.tensors.append(self.allocator, .{
            .name = name_copy,
            .data = dst_data,
            .shape = shape,
            .ndim = ndim,
        });

        self.total_bytes += dst_byte_size;
        _ = src_byte_size;
    }

    /// Record a tensor using trace point enum and layer index.
    pub fn recordTrace(
        self: *Capture,
        point: trace.TracePoint,
        layer: u16,
        data_ptr: [*]const u8,
        dtype: dtype_mod.DType,
        shape: [4]usize,
        ndim: u8,
    ) !void {
        if (!self.enabled) return;
        if (self.stopped) return;

        // Apply layer filter for layer-specific trace points
        const is_layer_point = layer != trace.TraceEmission.NO_LAYER;
        if (is_layer_point and !self.passesLayerFilter(layer)) return;

        // Build name string
        var name_buf: [64]u8 = undefined;
        const point_name = point.name();
        const name = if (!is_layer_point)
            std.fmt.bufPrint(&name_buf, "{s}", .{point_name}) catch return error.NameTooLong
        else
            std.fmt.bufPrint(&name_buf, "layer{d}.{s}", .{ layer, point_name }) catch return error.NameTooLong;

        // Apply point filter
        if (!self.passesPointFilter(name)) return;

        // Check stop-after-layer (set stopped flag after recording this layer's last point)
        if (is_layer_point) {
            if (self.stop_after_layer) |stop_layer| {
                if (layer >= stop_layer and point == .block_out) {
                    self.stopped = true;
                }
            }
        }

        try self.record(name, data_ptr, dtype, shape, ndim);
    }
};

// Conversion helpers
fn bf16ToF32(bits: u16) f32 {
    const result: u32 = @as(u32, bits) << 16;
    return @bitCast(result);
}

fn fp16ToF32(bits: u16) f32 {
    // IEEE 754 half-precision to single-precision conversion
    const sign: u32 = (@as(u32, bits) & 0x8000) << 16;
    const exp: u32 = (@as(u32, bits) >> 10) & 0x1F;
    const mant: u32 = @as(u32, bits) & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            return @bitCast(sign); // Zero
        }
        // Denormalized
        var m = mant;
        var e: u32 = 0;
        while ((m & 0x400) == 0) {
            m <<= 1;
            e += 1;
        }
        const result = sign | ((127 - 15 - e) << 23) | ((m & 0x3FF) << 13);
        return @bitCast(result);
    } else if (exp == 31) {
        // Inf or NaN
        const result = sign | 0x7F800000 | (mant << 13);
        return @bitCast(result);
    }
    // Normalized
    const result = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    return @bitCast(result);
}

// Thread-local global capture instance
var global_capture: ?*Capture = null;

/// Set the global capture instance (call before inference).
pub fn setGlobalCapture(cap: *Capture) void {
    global_capture = cap;
}

/// Clear the global capture instance.
pub fn clearGlobalCapture() void {
    global_capture = null;
}

/// Record to global capture (called from kernel code).
/// This is the entry point for instrumentation.
pub fn recordGlobal(
    point: trace.TracePoint,
    layer: u16,
    data_ptr: [*]const u8,
    dtype: dtype_mod.DType,
    shape: [4]usize,
    ndim: u8,
) void {
    if (global_capture) |cap| {
        cap.recordTrace(point, layer, data_ptr, dtype, shape, ndim) catch {};
    }
}

/// Check if dump capture is enabled (compile-time check).
pub fn isDumpEnabled() bool {
    return DumpEnabled;
}

/// Check if we should stop execution (called by executor after each layer).
/// Returns true if stop-after-layer has been reached.
pub fn shouldStopGlobal() bool {
    if (global_capture) |cap| {
        return cap.shouldStop();
    }
    return false;
}
