//! Shared CPU helpers for compute primitives.

const std = @import("std");
const builtin = @import("builtin");
const simd = @import("simd/arch/root.zig");

const VEC_LEN = simd.f32_vec_len;
const F32Vec = simd.F32Vec;

/// Page size for page-aligned allocations (required for Metal zero-copy buffers).
/// macOS uses 16KB pages on Apple Silicon, 4KB on Intel.
pub const page_size: usize = if (builtin.os.tag == .macos and builtin.cpu.arch == .aarch64)
    16 * 1024
else
    std.heap.page_size_min;

/// Alignment in f32 elements for page-aligned buffers.
const page_align_f32s = page_size / @sizeOf(f32);

/// Heap overflow diagnostic: when true, ensureF32Slice and ensureAligned64F32Slice
/// append a guard zone of GUARD_PATTERN after each buffer. Use verifyF32Guard() to
/// detect overflows. Set to false for zero overhead in production.
pub const heap_debug = false;

pub const GUARD_F32S: usize = if (heap_debug) 64 else 0;
pub const GUARD_PATTERN: u32 = 0xDEADBEEF;

fn fillGuard(buf: []f32, data_len: usize) void {
    for (buf[data_len..]) |*slot| {
        slot.* = @bitCast(GUARD_PATTERN);
    }
}

/// Returns true if the guard zone appended by ensureF32Slice / ensureAligned64F32Slice
/// is intact. Always returns true when heap_debug is false.
pub fn verifyF32Guard(buf: []const f32) bool {
    if (!heap_debug) return true;
    if (buf.len < GUARD_F32S) return true;
    const guard_start = buf.len - GUARD_F32S;
    for (buf[guard_start..]) |val| {
        if (@as(u32, @bitCast(val)) != GUARD_PATTERN) return false;
    }
    return true;
}

/// Ensure a reusable f32 buffer has at least `needed` elements.
/// When heap_debug is true, appends a guard zone after the data region.
pub fn ensureF32Slice(allocator: std.mem.Allocator, storage: *[]f32, needed: usize) !void {
    const total = needed + GUARD_F32S;
    if (storage.*.len >= total) return;
    if (storage.*.len > 0) {
        allocator.free(storage.*);
        storage.* = &.{};
    }
    storage.* = try allocator.alloc(f32, total);
    if (heap_debug) fillGuard(storage.*, needed);
}

/// Ensure a reusable u32 buffer has at least `needed` elements.
pub fn ensureU32Slice(allocator: std.mem.Allocator, storage: *[]u32, needed: usize) !void {
    if (storage.*.len >= needed) return;
    if (storage.*.len > 0) {
        allocator.free(storage.*);
        storage.* = &.{};
    }
    storage.* = try allocator.alloc(u32, needed);
}

/// Ensure a reusable f32 buffer with 64-byte alignment (cache-line / SIMD).
/// Free via freeAligned64F32Slice.
/// When heap_debug is true, appends a guard zone after the data region.
pub fn ensureAligned64F32Slice(allocator: std.mem.Allocator, storage: *[]f32, needed: usize) !void {
    const total = needed + GUARD_F32S;
    if (storage.*.len >= total) return;
    if (storage.*.len > 0) {
        freeAligned64F32Slice(allocator, storage.*);
        storage.* = &.{};
    }
    const aligned = try allocator.alignedAlloc(f32, .@"64", total);
    storage.* = aligned;
    if (heap_debug) fillGuard(storage.*, needed);
}

/// Free a []f32 that was allocated with 64-byte alignment.
pub fn freeAligned64F32Slice(allocator: std.mem.Allocator, slice: []f32) void {
    const aligned: []align(64) f32 = @alignCast(slice);
    allocator.free(aligned);
}

/// Ensure a reusable f32 buffer with page alignment (for Metal zero-copy buffers).
/// Free via freePageAlignedF32Slice.
/// When heap_debug is true, appends a guard zone after the data region.
pub fn ensurePageAlignedF32Slice(storage: *[]f32, needed: usize) !void {
    const total = needed + GUARD_F32S;
    if (storage.*.len >= total) return;
    if (storage.*.len > 0) {
        freePageAlignedF32Slice(storage.*);
        storage.* = &.{};
    }
    // Use page_allocator for guaranteed page-aligned memory.
    const byte_count = total * @sizeOf(f32);
    const aligned_bytes = std.mem.alignForward(usize, byte_count, page_size);
    const bytes = try std.heap.page_allocator.alloc(u8, aligned_bytes);
    storage.* = @as([*]f32, @alignCast(@ptrCast(bytes.ptr)))[0..total];
    if (heap_debug) fillGuard(storage.*, needed);
}

/// Free a []f32 that was allocated with page alignment.
pub fn freePageAlignedF32Slice(slice: []f32) void {
    if (slice.len == 0) return;
    const byte_count = slice.len * @sizeOf(f32);
    const aligned_bytes = std.mem.alignForward(usize, byte_count, page_size);
    const bytes: [*]u8 = @ptrCast(slice.ptr);
    std.heap.page_allocator.free(bytes[0..aligned_bytes]);
}

/// Add a 1-D bias vector to each row of a [rows, dim] f32 buffer.
pub fn addBiasRows(data: []f32, bias: []const f32, rows: usize, dim: usize) void {
    std.debug.assert(bias.len == dim);
    std.debug.assert(data.len >= rows * dim);

    for (0..rows) |row_idx| {
        const row = data[row_idx * dim ..][0..dim];
        var vec_idx: usize = 0;
        while (vec_idx + VEC_LEN - 1 < dim) : (vec_idx += VEC_LEN) {
            const row_vec: F32Vec = row[vec_idx..][0..VEC_LEN].*;
            const bias_vec: F32Vec = bias[vec_idx..][0..VEC_LEN].*;
            row[vec_idx..][0..VEC_LEN].* = row_vec + bias_vec;
        }
        while (vec_idx < dim) : (vec_idx += 1) {
            row[vec_idx] += bias[vec_idx];
        }
    }
}

test "ensureF32Slice grows buffer" {
    const allocator = std.testing.allocator;
    var storage: []f32 = &.{};
    defer if (storage.len > 0) allocator.free(storage);

    try ensureF32Slice(allocator, &storage, 16);
    try std.testing.expect(storage.len >= 16);
}

test "addBiasRows adds per-row bias" {
    var data = [_]f32{
        1, 2, 3, 4,
        5, 6, 7, 8,
    };
    const bias = [_]f32{ 0.5, -1.0, 2.0, 0.0 };

    addBiasRows(&data, &bias, 2, 4);

    try std.testing.expectApproxEqAbs(@as(f32, 1.5), data[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), data[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), data[3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.5), data[4], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), data[5], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), data[6], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), data[7], 1e-6);
}

test "ensureU32Slice grows buffer" {
    const allocator = std.testing.allocator;
    var storage: []u32 = &.{};
    defer if (storage.len > 0) allocator.free(storage);

    try ensureU32Slice(allocator, &storage, 8);
    try std.testing.expect(storage.len >= 8);
}

test "ensureAligned64F32Slice provides 64-byte alignment" {
    const allocator = std.testing.allocator;
    var storage: []f32 = &.{};
    defer if (storage.len > 0) freeAligned64F32Slice(allocator, storage);

    try ensureAligned64F32Slice(allocator, &storage, 32);
    try std.testing.expect(storage.len >= 32);
    try std.testing.expect(@intFromPtr(storage.ptr) % 64 == 0);
}

test "ensurePageAlignedF32Slice provides page alignment" {
    var storage: []f32 = &.{};
    defer if (storage.len > 0) freePageAlignedF32Slice(storage);

    try ensurePageAlignedF32Slice(&storage, 1024);
    try std.testing.expect(storage.len >= 1024);
    try std.testing.expect(@intFromPtr(storage.ptr) % page_size == 0);
}

test "ensurePageAlignedF32Slice grows buffer" {
    var storage: []f32 = &.{};
    defer if (storage.len > 0) freePageAlignedF32Slice(storage);

    try ensurePageAlignedF32Slice(&storage, 16);
    try std.testing.expect(storage.len >= 16);
    const first_ptr = storage.ptr;

    // Asking for same or smaller size reuses buffer
    try ensurePageAlignedF32Slice(&storage, 8);
    try std.testing.expectEqual(first_ptr, storage.ptr);

    // Asking for larger size reallocates
    try ensurePageAlignedF32Slice(&storage, 8192);
    try std.testing.expect(storage.len >= 8192);
    try std.testing.expect(@intFromPtr(storage.ptr) % page_size == 0);
}

test "ensureF32Slice clears storage on allocation failure" {
    // Regression: before the fix, a failed grow left storage pointing to
    // freed memory. The defer would then double-free on cleanup.
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 1 });
    const alloc = failing.allocator();
    var storage: []f32 = &.{};
    defer if (storage.len > 0) alloc.free(storage);

    try ensureF32Slice(alloc, &storage, 16);
    try std.testing.expect(storage.len >= 16);

    // Second alloc fails after the old buffer is freed.
    try std.testing.expectError(error.OutOfMemory, ensureF32Slice(alloc, &storage, 32));
    try std.testing.expectEqual(@as(usize, 0), storage.len);
}

test "ensureU32Slice clears storage on allocation failure" {
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 1 });
    const alloc = failing.allocator();
    var storage: []u32 = &.{};
    defer if (storage.len > 0) alloc.free(storage);

    try ensureU32Slice(alloc, &storage, 8);
    try std.testing.expect(storage.len >= 8);

    try std.testing.expectError(error.OutOfMemory, ensureU32Slice(alloc, &storage, 16));
    try std.testing.expectEqual(@as(usize, 0), storage.len);
}

test "ensureAligned64F32Slice clears storage on allocation failure" {
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 1 });
    const alloc = failing.allocator();
    var storage: []f32 = &.{};
    defer if (storage.len > 0) freeAligned64F32Slice(alloc, storage);

    try ensureAligned64F32Slice(alloc, &storage, 32);
    try std.testing.expect(storage.len >= 32);

    try std.testing.expectError(error.OutOfMemory, ensureAligned64F32Slice(alloc, &storage, 64));
    try std.testing.expectEqual(@as(usize, 0), storage.len);
}

test "verifyF32Guard detects overflow past ensureF32Slice data region" {
    const allocator = std.testing.allocator;
    var storage: []f32 = &.{};
    defer if (storage.len > 0) allocator.free(storage);

    try ensureF32Slice(allocator, &storage, 16);
    // Guard zone intact after allocation.
    try std.testing.expect(verifyF32Guard(storage));

    if (heap_debug) {
        // Corrupt the first guard element and verify detection.
        storage[16] = 0.0;
        try std.testing.expect(!verifyF32Guard(storage));
    }
}

test "verifyF32Guard detects overflow past ensureAligned64F32Slice data region" {
    const allocator = std.testing.allocator;
    var storage: []f32 = &.{};
    defer if (storage.len > 0) freeAligned64F32Slice(allocator, storage);

    try ensureAligned64F32Slice(allocator, &storage, 32);
    try std.testing.expect(verifyF32Guard(storage));

    if (heap_debug) {
        storage[32] = 0.0;
        try std.testing.expect(!verifyF32Guard(storage));
    }
}
