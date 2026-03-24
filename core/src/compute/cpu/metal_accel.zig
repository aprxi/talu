//! Metal acceleration for CPU backend attention operations.
//!
//! On macOS, uses Metal GPU for large attention matmuls.
//! Falls back gracefully on other platforms or when Metal unavailable.
//!
//! To disable: set TALU_NO_METAL_ACCEL=1

const std = @import("std");
const builtin = @import("builtin");

const metal = if (builtin.os.tag == .macos) @import("../metal/root.zig") else struct {
    pub const device = struct {
        pub const Device = void;
    };
    pub const matmul = struct {
        pub fn matmulF32TransBScaled(_: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) !void {}
        pub fn matmulF32(_: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) !void {}
    };
};

pub const Device = metal.device.Device;

var global_device: ?Device = null;
var initialized: bool = false;

/// Initialize Metal acceleration (automatically enabled on macOS).
/// Set TALU_NO_METAL_ACCEL=1 to disable.
pub fn init() void {
    if (initialized) return;
    initialized = true;

    if (comptime builtin.os.tag != .macos) return;

    // Check if Metal acceleration is disabled via env var
    const disabled = std.process.hasEnvVar(std.heap.page_allocator, "TALU_NO_METAL_ACCEL") catch false;
    if (disabled) return;

    initDevice();
}

fn initDevice() void {
    if (global_device != null) return;

    // Try to create Metal device
    global_device = Device.init() catch |err| {
        std.log.warn("Metal acceleration failed to initialize: {}", .{err});
        return;
    };

    if (global_device) |dev| {
        var d = dev;
        std.log.info("Metal acceleration enabled: {s}", .{d.name()});
    }
}

/// Get the Metal device if available.
pub fn getDevice() ?*Device {
    if (!initialized) init();
    if (global_device) |*dev| return dev;
    return null;
}

/// Check if Metal acceleration is available.
pub fn isAvailable() bool {
    if (!initialized) init();
    return global_device != null;
}

/// Perform Q @ K^T with scaling using Metal.
/// Returns true if Metal was used, false if caller should use CPU fallback.
pub fn matmulTransBScaled(
    a: []const f32, // [m, k]
    b: []const f32, // [n, k]
    c: []f32, // [m, n]
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
) bool {
    if (comptime builtin.os.tag != .macos) return false;

    const dev = getDevice() orelse return false;

    metal.matmul.matmulF32TransBScaled(dev, a, m, k, b, n, c, alpha) catch return false;
    return true;
}

/// Perform A @ B using Metal.
/// Returns true if Metal was used, false if caller should use CPU fallback.
pub fn matmul(
    a: []const f32, // [m, k]
    b: []const f32, // [k, n]
    c: []f32, // [m, n]
    m: usize,
    n: usize,
    k: usize,
) bool {
    if (comptime builtin.os.tag != .macos) return false;

    const dev = getDevice() orelse return false;

    metal.matmul.matmulF32(dev, a, m, k, b, n, c) catch return false;
    return true;
}

/// Perform Q @ K^T with K as INT8 and on-the-fly dequant using Metal.
/// Q: [m, k] f32, K: [n, k] i8, K_scales: [n] f32, C: [m, n] f32.
/// Returns true if Metal was used, false if caller should use CPU fallback.
pub fn matmulI8TransBScaled(
    a: []const f32, // [m, k] - Q
    b: []const i8, // [n, k] - K (INT8)
    b_scales: []const f32, // [n] - K scales
    c: []f32, // [m, n] - scores
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
) bool {
    if (comptime builtin.os.tag != .macos) return false;

    const dev = getDevice() orelse return false;

    metal.matmul.matmulF32I8TransBScaled(dev, a, m, k, b, n, b_scales, c, alpha) catch return false;
    return true;
}
