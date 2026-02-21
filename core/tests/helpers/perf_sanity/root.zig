//! Performance and lifecycle sanity checks.
//!
//! Intended for automation/nightly usage with explicit thresholds.
//!
//! Usage:
//!   zig run core/tests/helpers/perf_sanity/root.zig
//!   zig run core/tests/helpers/perf_sanity/root.zig -- cpu
//!   zig run core/tests/helpers/perf_sanity/root.zig -- lifecycle
//!
//! Environment:
//!   TALU_PERF_CPU_MIN_TOKENS_PER_SEC=<float>   optional minimum threshold
//!   TALU_PERF_CPU_ITERS=<usize>                optional iteration count (default: 2048)
//!   TALU_LIFECYCLE_CYCLES=<usize>              optional cycle count (default: 128)

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const root_main = @import("main");
const rowwise = root_main.compute.cpu.rowwise;
const softmax = root_main.compute.cpu.softmax;
const metal = root_main.compute.metal;
const runtime = root_main.inference.backend.cpu.executor.runtime;

fn envUsize(allocator: std.mem.Allocator, name: []const u8, default_value: usize) usize {
    const raw = std.process.getEnvVarOwned(allocator, name) catch |err| switch (err) {
        error.EnvironmentVariableNotFound => return default_value,
        else => return default_value,
    };
    defer allocator.free(raw);
    return std.fmt.parseUnsigned(usize, raw, 10) catch default_value;
}

fn envF64(allocator: std.mem.Allocator, name: []const u8) ?f64 {
    const raw = std.process.getEnvVarOwned(allocator, name) catch |err| switch (err) {
        error.EnvironmentVariableNotFound => return null,
        else => return null,
    };
    defer allocator.free(raw);
    return std.fmt.parseFloat(f64, raw) catch null;
}

pub fn benchmarkCpuDecodeProxy(allocator: std.mem.Allocator, iters: usize, hidden_size: usize, vocab_size: usize) !f64 {
    const hidden = try allocator.alloc(f32, hidden_size);
    defer allocator.free(hidden);
    const residual = try allocator.alloc(f32, hidden_size);
    defer allocator.free(residual);
    const logits = try allocator.alloc(f32, vocab_size);
    defer allocator.free(logits);

    for (hidden, 0..) |*v, idx| v.* = @as(f32, @floatFromInt((idx % 31) + 1)) * 0.01;
    for (residual, 0..) |*v, idx| v.* = @as(f32, @floatFromInt((idx % 17) + 1)) * 0.005;
    for (logits, 0..) |*v, idx| v.* = @as(f32, @floatFromInt((idx % 29) + 1)) * 0.02;

    const start = std.time.nanoTimestamp();
    for (0..iters) |_| {
        rowwise.addScaledInPlace(hidden, residual, 0.125);
        softmax.stableInPlace(logits);
    }
    const end = std.time.nanoTimestamp();

    const elapsed_ns: f64 = @floatFromInt(end - start);
    if (elapsed_ns <= 0) return 0.0;
    const elapsed_s = elapsed_ns / 1_000_000_000.0;
    return @as(f64, @floatFromInt(iters)) / elapsed_s;
}

pub fn benchmarkMetalDecodeProxy(
    allocator: std.mem.Allocator,
    iters: usize,
    hidden_size: usize,
) !?f64 {
    if (comptime builtin.os.tag != .macos) return null;
    if (!build_options.enable_metal) return null;
    if (!metal.isAvailable()) return null;

    var device = metal.device.Device.init() catch return null;
    defer device.deinit();

    const a = try allocator.alloc(f32, hidden_size);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, hidden_size * hidden_size);
    defer allocator.free(b);
    const c = try allocator.alloc(f32, hidden_size);
    defer allocator.free(c);

    for (a, 0..) |*v, idx| v.* = @as(f32, @floatFromInt((idx % 23) + 1)) * 0.02;
    for (b, 0..) |*v, idx| v.* = @as(f32, @floatFromInt((idx % 19) + 1)) * 0.01;
    @memset(c, 0.0);

    const start = std.time.nanoTimestamp();
    for (0..iters) |_| {
        try metal.matmul.matmulF32(&device, a, 1, hidden_size, b, hidden_size, c);
        device.synchronize();
    }
    const end = std.time.nanoTimestamp();

    const elapsed_ns: f64 = @floatFromInt(end - start);
    if (elapsed_ns <= 0) return 0.0;
    const elapsed_s = elapsed_ns / 1_000_000_000.0;
    return @as(f64, @floatFromInt(iters)) / elapsed_s;
}

pub fn checkCpuScratchLifecycle(cycles: usize) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    for (0..cycles) |_| {
        var scratch = try runtime.ScratchBuffer.init(allocator, 1024, 4096, 4);
        try scratch.ensure(64);
        scratch.resetCaches();
        scratch.deinit();
    }

    const status = gpa.deinit();
    if (status == .leak) return error.MemoryLeak;
}

fn runCpu(writer: anytype, allocator: std.mem.Allocator) !void {
    const iters = envUsize(allocator, "TALU_PERF_CPU_ITERS", 2048);
    const min_tps = envF64(allocator, "TALU_PERF_CPU_MIN_TOKENS_PER_SEC");
    const measured = try benchmarkCpuDecodeProxy(allocator, iters, 4096, 4096);
    var threshold_buf: [64]u8 = undefined;
    const threshold_str: []const u8 = if (min_tps) |threshold|
        try std.fmt.bufPrint(&threshold_buf, "{d:.3}", .{threshold})
    else
        "none";

    const status: []const u8 = if (min_tps) |threshold|
        if (measured >= threshold) "pass" else "fail"
    else
        "pass";

    try writer.print(
        "PERF_CHECK name=cpu_decode_proxy tokens_per_sec={d:.3} threshold={s} status={s}\n",
        .{
            measured,
            threshold_str,
            status,
        },
    );

    if (std.mem.eql(u8, status, "fail")) return error.PerfRegression;
}

fn runMetal(writer: anytype, allocator: std.mem.Allocator) !void {
    const iters = envUsize(allocator, "TALU_PERF_METAL_ITERS", 256);
    const min_tps = envF64(allocator, "TALU_PERF_METAL_MIN_TOKENS_PER_SEC");
    const maybe_measured = try benchmarkMetalDecodeProxy(allocator, iters, 1024);

    if (maybe_measured == null) {
        try writer.writeAll("PERF_CHECK name=metal_decode_proxy tokens_per_sec=0.000 threshold=none status=skip reason=unavailable\n");
        return;
    }

    const measured = maybe_measured.?;
    var threshold_buf: [64]u8 = undefined;
    const threshold_str: []const u8 = if (min_tps) |threshold|
        try std.fmt.bufPrint(&threshold_buf, "{d:.3}", .{threshold})
    else
        "none";

    const status: []const u8 = if (min_tps) |threshold|
        if (measured >= threshold) "pass" else "fail"
    else
        "pass";

    try writer.print(
        "PERF_CHECK name=metal_decode_proxy tokens_per_sec={d:.3} threshold={s} status={s}\n",
        .{
            measured,
            threshold_str,
            status,
        },
    );

    if (std.mem.eql(u8, status, "fail")) return error.PerfRegression;
}

fn runLifecycle(writer: anytype, allocator: std.mem.Allocator) !void {
    const cycles = envUsize(allocator, "TALU_LIFECYCLE_CYCLES", 128);
    try checkCpuScratchLifecycle(cycles);
    try writer.print("LIFECYCLE_CHECK name=cpu_scratch status=pass cycles={}\n", .{cycles});
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const status = gpa.deinit();
        if (status == .leak) @panic("perf_sanity main leaked memory");
    }
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    const stdout = std.fs.File.stdout().deprecatedWriter();

    if (args.len == 1) {
        try runCpu(stdout, allocator);
        try runMetal(stdout, allocator);
        try runLifecycle(stdout, allocator);
        return;
    }

    if (std.mem.eql(u8, args[1], "cpu")) {
        try runCpu(stdout, allocator);
        return;
    }

    if (std.mem.eql(u8, args[1], "lifecycle")) {
        try runLifecycle(stdout, allocator);
        return;
    }

    if (std.mem.eql(u8, args[1], "metal")) {
        try runMetal(stdout, allocator);
        return;
    }

    return error.InvalidArguments;
}

test "benchmarkCpuDecodeProxy returns positive throughput" {
    const tps = try benchmarkCpuDecodeProxy(std.testing.allocator, 32, 256, 256);
    try std.testing.expect(tps > 0);
}

test "benchmarkMetalDecodeProxy returns optional throughput" {
    const tps = try benchmarkMetalDecodeProxy(std.testing.allocator, 4, 64);
    if (tps) |value| {
        try std.testing.expect(value > 0);
    }
}

test "checkCpuScratchLifecycle does not leak" {
    try checkCpuScratchLifecycle(8);
}
