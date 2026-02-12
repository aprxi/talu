//! X-Ray C API
//!
//! C-callable functions for tensor inspection during inference.
//! Allows capturing tensor values at defined trace points for debugging
//! and validation.

const std = @import("std");
const xray = @import("../xray/root.zig");
const trace = xray.trace;
const capture = xray.capture;
const query = xray.query;
const stats_mod = xray.stats;
const capi_error = @import("error.zig");
const error_codes = @import("error_codes.zig");

const allocator = std.heap.c_allocator;

// =============================================================================
// Types
// =============================================================================

/// Opaque capture handle for tensor inspection/debugging.
///
/// Thread Safety: NOT thread-safe. Each CaptureHandle must be accessed from
/// a single thread at a time. Callers must provide external synchronization
/// if sharing across threads.
pub const CaptureHandle = opaque {};

/// Statistics returned from captured tensors
pub const TensorStats = extern struct {
    count: u64,
    min: f32,
    max: f32,
    mean: f32,
    rms: f32,
    nan_count: u32,
    inf_count: u32,
};

/// Captured tensor info
pub const CapturedTensorInfo = extern struct {
    point: u8,
    layer: u16,
    token: u32,
    position: u32,
    shape: [4]u32,
    ndim: u8,
    dtype: u8,
    kernel_name: [48]u8,
    stats: TensorStats,
    timestamp_ns: i64,
};

/// Query result
pub const QueryResult = extern struct {
    /// Pointer to array of CapturedTensorInfo (caller must free)
    results: ?[*]CapturedTensorInfo,
    /// Number of results
    count: usize,
    /// Error message if failed (null on success)
    error_msg: ?[*:0]const u8,
};

// =============================================================================
// Capture Management
// =============================================================================

/// Create a new capture with configuration.
/// Returns null on failure (check talu_error_message for details).
pub export fn talu_xray_capture_create(
    points_mask: u32,
    mode: u8,
    sample_count: u32,
) callconv(.c) ?*CaptureHandle {
    capi_error.clearError();
    var config = capture.TraceCaptureConfig{
        .mode = @enumFromInt(mode),
        .sample_count = sample_count,
    };

    // Set points from bitmask
    const points_ptr: *capture.TracePointSet = &config.points;
    const mask_bytes: *u32 = @ptrCast(points_ptr);
    mask_bytes.* = points_mask;

    const cap = allocator.create(capture.TraceCapture) catch {
        capi_error.setError(error.OutOfMemory, "failed to allocate trace capture", .{});
        return null;
    };
    cap.* = capture.TraceCapture.init(allocator, config);
    return @ptrCast(cap);
}

/// Create capture with all points enabled (convenience).
/// Returns null on failure (check talu_error_message for details).
pub export fn talu_xray_capture_create_all(mode: u8, sample_count: u32) callconv(.c) ?*CaptureHandle {
    capi_error.clearError();
    const config = capture.TraceCaptureConfig{
        .points = capture.TracePointSet.all(),
        .mode = @enumFromInt(mode),
        .sample_count = sample_count,
    };

    const cap = allocator.create(capture.TraceCapture) catch {
        capi_error.setError(error.OutOfMemory, "failed to allocate trace capture", .{});
        return null;
    };
    cap.* = capture.TraceCapture.init(allocator, config);
    return @ptrCast(cap);
}

/// Enable capture (start receiving trace emissions).
pub export fn talu_xray_capture_enable(handle: ?*CaptureHandle) callconv(.c) void {
    // Note: void return - silently ignore null handle
    const cap = getCapture(handle) orelse return;
    capture.enable(cap);
}

/// Disable capture (stop receiving trace emissions).
pub export fn talu_xray_capture_disable() callconv(.c) void {
    capture.disable();
}

/// Check if capture is enabled.
pub export fn talu_xray_capture_is_enabled() callconv(.c) bool {
    return capture.isEnabled();
}

/// Clear all captured data (keep configuration).
pub export fn talu_xray_capture_clear(handle: ?*CaptureHandle) callconv(.c) void {
    // Note: void return - silently ignore null handle
    const cap = getCapture(handle) orelse return;
    cap.clear();
}

/// Get number of captured records.
/// Returns 0 if handle is null.
pub export fn talu_xray_capture_count(handle: ?*CaptureHandle) callconv(.c) usize {
    // Note: Query function - returns 0 rather than error for null handle
    const cap = getCapture(handle) orelse return 0;
    return cap.count();
}

/// Check if capture overflowed (memory limit exceeded).
/// Returns false if handle is null.
pub export fn talu_xray_capture_overflow(handle: ?*CaptureHandle) callconv(.c) bool {
    // Note: Query function - returns false rather than error for null handle
    const cap = getCapture(handle) orelse return false;
    return cap.overflow;
}

/// Destroy capture and free memory.
pub export fn talu_xray_capture_destroy(handle: ?*CaptureHandle) callconv(.c) void {
    // Note: void return - silently ignore null handle
    const cap = getCapture(handle) orelse return;
    // Disable if this capture is active
    if (capture.isEnabled()) {
        capture.disable();
    }
    cap.deinit();
    allocator.destroy(cap);
}

// =============================================================================
// Query Functions
// =============================================================================

/// Get captured tensor info by index.
/// Returns true on success, false on error (check talu_error_message).
pub export fn talu_xray_get(
    handle: ?*CaptureHandle,
    index: usize,
    out: ?*CapturedTensorInfo,
) callconv(.c) bool {
    capi_error.clearError();
    const cap = getCapture(handle) orelse {
        capi_error.setError(error.InvalidHandle, "capture handle is null", .{});
        return false;
    };
    const record = cap.get(index) orelse {
        capi_error.setError(error.InvalidArgument, "index {d} out of bounds", .{index});
        return false;
    };
    const info = out orelse {
        capi_error.setError(error.InvalidArgument, "out parameter is null", .{});
        return false;
    };

    info.* = capturedToInfo(record);
    return true;
}

/// Find first tensor with anomalies (NaN or Inf).
/// Returns true if found, false if not found or error.
pub export fn talu_xray_find_anomaly(
    handle: ?*CaptureHandle,
    out_point: ?*u8,
    out_layer: ?*u16,
    out_token: ?*u32,
) callconv(.c) bool {
    // Note: This is a search function - null handle or not-found returns false (not an error)
    const cap = getCapture(handle) orelse return false;
    const q = query.CaptureQuery.init(cap);

    const loc = q.findFirst(.has_anomaly, 0) orelse return false;

    if (out_point) |p| p.* = @intFromEnum(loc.point);
    if (out_layer) |l| l.* = loc.layer;
    if (out_token) |t| t.* = loc.token;
    return true;
}

/// Count captured tensors matching criteria.
/// Pass 0xFF for point to match any point.
/// Pass 0xFFFF for layer to match any layer.
/// Pass 0xFFFFFFFF for token to match any token.
/// Returns 0 if handle is null.
pub export fn talu_xray_count_matching(
    handle: ?*CaptureHandle,
    point: u8,
    layer: u16,
    token: u32,
) callconv(.c) usize {
    // Note: Query function - returns 0 rather than error for null handle
    const cap = getCapture(handle) orelse return 0;
    const q = query.CaptureQuery.init(cap);

    const point_filter: ?trace.TracePoint = if (point == 0xFF) null else @enumFromInt(point);
    const layer_filter: ?u16 = if (layer == 0xFFFF) null else layer;
    const token_filter: ?u32 = if (token == 0xFFFFFFFF) null else token;

    return q.count(point_filter, layer_filter, token_filter);
}

/// Get samples from a captured tensor.
/// Returns number of samples copied, or 0 if no samples available or error.
pub export fn talu_xray_get_samples(
    handle: ?*CaptureHandle,
    index: usize,
    out_samples: [*]f32,
    max_samples: usize,
) callconv(.c) usize {
    // Note: Query function - returns 0 rather than error for null handle / no samples
    const cap = getCapture(handle) orelse return 0;
    const record = cap.get(index) orelse return 0;

    const samples = record.samples orelse return 0;
    const copy_count = @min(samples.len, max_samples);
    @memcpy(out_samples[0..copy_count], samples[0..copy_count]);
    return copy_count;
}

/// Get size of full captured tensor data (bytes). Returns 0 if not available.
pub export fn talu_xray_get_data_size(handle: ?*CaptureHandle, index: usize) callconv(.c) usize {
    const cap = getCapture(handle) orelse return 0;
    const record = cap.get(index) orelse return 0;
    if (record.data) |data| {
        return data.len;
    }
    return 0;
}

/// Copy full captured tensor data into caller-provided buffer.
/// Returns number of bytes copied (0 if not available).
pub export fn talu_xray_get_data(
    handle: ?*CaptureHandle,
    index: usize,
    out_data: [*]u8,
    max_len: usize,
) callconv(.c) usize {
    const cap = getCapture(handle) orelse return 0;
    const record = cap.get(index) orelse return 0;
    if (record.data) |data| {
        const copy_count = @min(data.len, max_len);
        @memcpy(out_data[0..copy_count], data[0..copy_count]);
        return copy_count;
    }
    return 0;
}

// =============================================================================
// Trace Point Names
// =============================================================================

/// Get name of a trace point.
pub export fn talu_xray_point_name(point: u8) callconv(.c) [*:0]const u8 {
    const p: trace.TracePoint = @enumFromInt(point);
    const name = p.name();
    // Return pointer to static string
    return @ptrCast(name.ptr);
}

// =============================================================================
// Helpers
// =============================================================================

fn getCapture(handle: ?*CaptureHandle) ?*capture.TraceCapture {
    const h = handle orelse return null;
    return @ptrCast(@alignCast(h));
}

fn capturedToInfo(record: *const capture.CapturedTensor) CapturedTensorInfo {
    var info = std.mem.zeroes(CapturedTensorInfo);
    info.point = @intFromEnum(record.point);
    info.layer = record.layer;
    info.token = record.token;
    info.position = record.position;
    info.shape = record.shape;
    info.ndim = record.ndim;
    info.dtype = @intFromEnum(record.dtype);
    info.kernel_name = record.kernel_name;
    info.stats.count = record.stats.count;
    info.stats.min = record.stats.min;
    info.stats.max = record.stats.max;
    info.stats.mean = record.stats.mean();
    info.stats.rms = record.stats.rms();
    info.stats.nan_count = record.stats.nan_count;
    info.stats.inf_count = record.stats.inf_count;
    info.timestamp_ns = @intCast(record.timestamp_ns);
    return info;
}

// =============================================================================
// Capture Mode Constants
// =============================================================================

/// CaptureMode.stats - statistics only
pub const CAPTURE_MODE_STATS: u8 = 0;
/// CaptureMode.sample - statistics + first N values
pub const CAPTURE_MODE_SAMPLE: u8 = 1;
/// CaptureMode.full - complete tensor copy
pub const CAPTURE_MODE_FULL: u8 = 2;

// =============================================================================
// Point Constants (for bitmask)
// =============================================================================

pub const POINT_EMBED: u32 = 1 << 0;
pub const POINT_EMBED_POS: u32 = 1 << 1;
pub const POINT_LAYER_INPUT: u32 = 1 << 2;
pub const POINT_LAYER_ATTN_NORM: u32 = 1 << 3;
pub const POINT_LAYER_Q: u32 = 1 << 4;
pub const POINT_LAYER_K: u32 = 1 << 5;
pub const POINT_LAYER_V: u32 = 1 << 6;
pub const POINT_LAYER_QK: u32 = 1 << 7;
pub const POINT_LAYER_ATTN_WEIGHTS: u32 = 1 << 8;
pub const POINT_LAYER_ATTN_OUT: u32 = 1 << 9;
pub const POINT_LAYER_FFN_NORM: u32 = 1 << 10;
pub const POINT_LAYER_FFN_GATE: u32 = 1 << 11;
pub const POINT_LAYER_FFN_UP: u32 = 1 << 12;
pub const POINT_LAYER_FFN_ACT: u32 = 1 << 13;
pub const POINT_LAYER_FFN_DOWN: u32 = 1 << 14;
pub const POINT_LAYER_RESIDUAL: u32 = 1 << 15; // block_out
pub const POINT_MAMBA_OUT: u32 = 1 << 16;
pub const POINT_CONV_IN_PROJ: u32 = 1 << 17;
pub const POINT_CONV_CONV: u32 = 1 << 18;
pub const POINT_CONV_OUT_PROJ: u32 = 1 << 19;
pub const POINT_FINAL_NORM: u32 = 1 << 20;
pub const POINT_LOGITS: u32 = 1 << 21; // lm_head
pub const POINT_LOGITS_SCALED: u32 = 1 << 22;
pub const POINT_ALL: u32 = 0x7FFFFF; // All 23 points
