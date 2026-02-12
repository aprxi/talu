//! SharedBuffer - Refcounted buffer for zero-copy DLPack interop.
//!
//! SharedBuffer enables safe zero-copy export to PyTorch/JAX via DLPack while
//! allowing the source TokenArray to remain valid after export. Multiple exports
//! and multiple TokenArray views can share the same underlying buffer.
//!
//! Memory Safety Contract:
//! - DLPack export shares storage with the TokenArray
//! - If a consumer mutates the exported tensor, the underlying storage is mutated
//! - TokenArray APIs treat tokens as immutable; mutation from consumers is allowed
//!   but not recommended
//! - If isolation is needed, clone/copy in the consumer framework
//!
//! Refcount Pattern:
//! - retain: fetch_add(1, Relaxed)
//! - release: fetch_sub(1, AcqRel), if hits zero: Acquire fence then free

const std = @import("std");
const Atomic = std.atomic.Value;
const capi_error = @import("error.zig");

/// Allocator for buffer operations (C allocator for FFI compatibility)
const allocator = std.heap.c_allocator;

/// Opaque handle for C API.
///
/// Thread safety: retain/release use atomic refcounting and are safe to call
/// from any thread. All other access (reads, slicing, mutation) is NOT
/// thread-safe; callers must provide external synchronization if sharing
/// across threads.
pub const BufferHandle = opaque {};

/// A refcounted buffer that can be shared across TokenArrays and DLPack exports.
///
/// Design notes:
/// - Owns a contiguous allocation of u32 token IDs
/// - Refcount starts at 1 on creation
/// - Each DLPack export increments refcount
/// - Each release decrements refcount
/// - Buffer is freed when refcount reaches 0
///
/// Future-proofing for views:
/// - TokenArray will store (buffer, offset_elems, len)
/// - This allows zero-copy slicing in the future
/// - For v1, offset_elems is always 0
pub const SharedBuffer = struct {
    /// The token data (owned by this buffer)
    data: [*]u32,
    /// Total number of elements in the buffer
    capacity: usize,
    /// Atomic reference count (starts at 1)
    refcount: Atomic(usize),

    const Self = @This();

    /// Create a new SharedBuffer by taking ownership of existing data.
    /// The caller must NOT free the data after this call.
    /// Returns null on allocation failure.
    pub fn createFromOwned(data: [*]u32, capacity: usize) ?*Self {
        const self = allocator.create(Self) catch return null;
        self.* = .{
            .data = data,
            .capacity = capacity,
            .refcount = Atomic(usize).init(1),
        };
        return self;
    }

    /// Create a new SharedBuffer by allocating and copying data.
    /// Returns null on allocation failure.
    pub fn createFromCopy(src: [*]const u32, len: usize) ?*Self {
        if (len == 0) return null;

        const data = allocator.alloc(u32, len) catch return null;
        @memcpy(data, src[0..len]);

        const self = allocator.create(Self) catch {
            allocator.free(data);
            return null;
        };

        self.* = .{
            .data = data.ptr,
            .capacity = len,
            .refcount = Atomic(usize).init(1),
        };
        return self;
    }

    /// Create a new SharedBuffer with uninitialized data.
    /// Caller should fill the data after creation.
    /// Returns null on allocation failure.
    pub fn createUninitialized(capacity: usize) ?*Self {
        if (capacity == 0) return null;

        const data = allocator.alloc(u32, capacity) catch return null;

        const self = allocator.create(Self) catch {
            allocator.free(data);
            return null;
        };

        self.* = .{
            .data = data.ptr,
            .capacity = capacity,
            .refcount = Atomic(usize).init(1),
        };
        return self;
    }

    /// Increment reference count.
    /// Safe to call from any thread.
    pub fn retain(self: *Self) void {
        // Relaxed is sufficient for increment - we don't need to synchronize
        // with any particular memory operations here
        _ = self.refcount.fetchAdd(1, .monotonic);
    }

    /// Decrement reference count and free if it reaches zero.
    /// Safe to call from any thread (including non-Python threads in DLPack deleters).
    /// Returns true if the buffer was freed.
    pub fn release(self: *Self) bool {
        // AcqRel ensures:
        // - All prior writes are visible before the decrement (Release)
        // - We see all writes from other threads before potentially freeing (Acquire)
        const prev = self.refcount.fetchSub(1, .acq_rel);

        if (prev == 1) {
            // We were the last holder - free the buffer
            // The AcqRel on fetchSub provides the necessary synchronization
            allocator.free(self.data[0..self.capacity]);
            allocator.destroy(self);
            return true;
        }
        return false;
    }

    /// Get current reference count (for debugging/testing only).
    /// The value may be stale by the time it's used.
    pub fn getRefcount(self: *const Self) usize {
        return self.refcount.load(.monotonic);
    }

    /// Get a slice of the buffer data.
    pub fn slice(self: *const Self, start: usize, end: usize) []const u32 {
        const actual_end = @min(end, self.capacity);
        const actual_start = @min(start, actual_end);
        return self.data[actual_start..actual_end];
    }

    /// Get mutable slice (use with caution - mutation affects all views).
    pub fn sliceMut(self: *Self, start: usize, end: usize) []u32 {
        const actual_end = @min(end, self.capacity);
        const actual_start = @min(start, actual_end);
        return self.data[actual_start..actual_end];
    }
};

// =============================================================================
// C API
// =============================================================================

/// Create a SharedBuffer by taking ownership of existing token data.
/// The caller must NOT free the tokens after this call.
/// Returns null on failure (check talu_error_message for details).
pub export fn talu_buffer_create_from_owned(
    tokens: ?[*]u32,
    num_tokens: usize,
) callconv(.c) ?*BufferHandle {
    capi_error.clearError();
    const data = tokens orelse {
        capi_error.setError(error.InvalidArgument, "tokens is null", .{});
        return null;
    };
    if (num_tokens == 0) {
        capi_error.setError(error.InvalidArgument, "num_tokens is 0", .{});
        return null;
    }

    const buffer = SharedBuffer.createFromOwned(data, num_tokens) orelse {
        capi_error.setError(error.OutOfMemory, "failed to allocate SharedBuffer", .{});
        return null;
    };
    return @ptrCast(buffer);
}

/// Create a SharedBuffer by copying token data.
/// Returns null on failure (check talu_error_message for details).
pub export fn talu_buffer_create_from_copy(
    tokens: ?[*]const u32,
    num_tokens: usize,
) callconv(.c) ?*BufferHandle {
    capi_error.clearError();
    const data = tokens orelse {
        capi_error.setError(error.InvalidArgument, "tokens is null", .{});
        return null;
    };
    if (num_tokens == 0) {
        capi_error.setError(error.InvalidArgument, "num_tokens is 0", .{});
        return null;
    }

    const buffer = SharedBuffer.createFromCopy(data, num_tokens) orelse {
        capi_error.setError(error.OutOfMemory, "failed to allocate SharedBuffer", .{});
        return null;
    };
    return @ptrCast(buffer);
}

/// Create an uninitialized SharedBuffer of given capacity.
/// Caller should fill via talu_buffer_get_data_ptr.
/// Returns null on failure (check talu_error_message for details).
pub export fn talu_buffer_create_uninitialized(
    capacity: usize,
) callconv(.c) ?*BufferHandle {
    capi_error.clearError();
    if (capacity == 0) {
        capi_error.setError(error.InvalidArgument, "capacity is 0", .{});
        return null;
    }

    const buffer = SharedBuffer.createUninitialized(capacity) orelse {
        capi_error.setError(error.OutOfMemory, "failed to allocate SharedBuffer", .{});
        return null;
    };
    return @ptrCast(buffer);
}

/// Increment buffer reference count.
/// Safe to call from any thread. Silently ignores null handle.
pub export fn talu_buffer_retain(handle: ?*BufferHandle) callconv(.c) void {
    // Note: Lightweight accessor - no error context for performance
    const buffer: *SharedBuffer = @ptrCast(@alignCast(handle orelse return));
    buffer.retain();
}

/// Decrement buffer reference count. Frees buffer if count reaches zero.
/// Safe to call from any thread (including DLPack deleter threads).
/// Returns 1 if the buffer was freed, 0 otherwise (or null handle).
pub export fn talu_buffer_release(handle: ?*BufferHandle) callconv(.c) u8 {
    // Note: Lightweight accessor - no error context for performance
    const buffer: *SharedBuffer = @ptrCast(@alignCast(handle orelse return 0));
    return if (buffer.release()) 1 else 0;
}

/// Get the data pointer from a buffer.
/// Returns null if handle is null.
pub export fn talu_buffer_get_data_ptr(handle: ?*BufferHandle) callconv(.c) ?[*]u32 {
    // Note: Lightweight accessor - no error context for performance
    const buffer: *SharedBuffer = @ptrCast(@alignCast(handle orelse return null));
    return buffer.data;
}

/// Get the capacity (number of elements) of a buffer.
/// Returns 0 if handle is null.
pub export fn talu_buffer_get_capacity(handle: ?*BufferHandle) callconv(.c) usize {
    // Note: Lightweight accessor - no error context for performance
    const buffer: *SharedBuffer = @ptrCast(@alignCast(handle orelse return 0));
    return buffer.capacity;
}

/// Get the current reference count (for debugging/testing).
/// Returns 0 if handle is null.
pub export fn talu_buffer_get_refcount(handle: ?*BufferHandle) callconv(.c) usize {
    // Note: Lightweight accessor - no error context for performance
    const buffer: *SharedBuffer = @ptrCast(@alignCast(handle orelse return 0));
    return buffer.getRefcount();
}

// =============================================================================
// Tests
// =============================================================================

test "SharedBuffer basic lifecycle" {
    // Create buffer
    const data = try allocator.alloc(u32, 4);
    data[0] = 100;
    data[1] = 200;
    data[2] = 300;
    data[3] = 400;

    const buffer = SharedBuffer.createFromOwned(data.ptr, 4) orelse return error.CreateFailed;

    // Initial refcount is 1
    try std.testing.expectEqual(@as(usize, 1), buffer.getRefcount());

    // Data is accessible
    try std.testing.expectEqual(@as(u32, 100), buffer.data[0]);
    try std.testing.expectEqual(@as(u32, 400), buffer.data[3]);

    // Retain increases refcount
    buffer.retain();
    try std.testing.expectEqual(@as(usize, 2), buffer.getRefcount());

    // First release doesn't free
    try std.testing.expect(!buffer.release());
    try std.testing.expectEqual(@as(usize, 1), buffer.getRefcount());

    // Second release frees
    try std.testing.expect(buffer.release());
    // buffer is now invalid - don't access it
}

test "SharedBuffer createFromCopy" {
    const src = [_]u32{ 1, 2, 3, 4, 5 };
    const buffer = SharedBuffer.createFromCopy(&src, 5) orelse return error.CreateFailed;

    // Data is copied
    try std.testing.expectEqual(@as(u32, 1), buffer.data[0]);
    try std.testing.expectEqual(@as(u32, 5), buffer.data[4]);

    // Modifying original doesn't affect buffer (it's a copy)
    // (can't actually test this since src is const, but the semantics are clear)

    _ = buffer.release();
}

test "SharedBuffer multiple retains and releases" {
    const data = try allocator.alloc(u32, 2);
    data[0] = 42;
    data[1] = 43;

    const buffer = SharedBuffer.createFromOwned(data.ptr, 2) orelse return error.CreateFailed;

    // Simulate multiple DLPack exports
    buffer.retain(); // Export 1
    buffer.retain(); // Export 2
    buffer.retain(); // Export 3
    try std.testing.expectEqual(@as(usize, 4), buffer.getRefcount());

    // Release in various orders (simulating torch tensors being GC'd)
    try std.testing.expect(!buffer.release()); // refcount: 3
    try std.testing.expect(!buffer.release()); // refcount: 2
    try std.testing.expect(!buffer.release()); // refcount: 1

    // Final release frees
    try std.testing.expect(buffer.release());
}

test "SharedBuffer slice" {
    const src = [_]u32{ 10, 20, 30, 40, 50 };
    const buffer = SharedBuffer.createFromCopy(&src, 5) orelse return error.CreateFailed;
    defer _ = buffer.release();

    const s = buffer.slice(1, 4);
    try std.testing.expectEqual(@as(usize, 3), s.len);
    try std.testing.expectEqual(@as(u32, 20), s[0]);
    try std.testing.expectEqual(@as(u32, 40), s[2]);
}

test "C API basic operations" {
    const src = [_]u32{ 1, 2, 3 };
    const handle = talu_buffer_create_from_copy(&src, 3);
    try std.testing.expect(handle != null);

    // Check capacity
    try std.testing.expectEqual(@as(usize, 3), talu_buffer_get_capacity(handle));

    // Check refcount
    try std.testing.expectEqual(@as(usize, 1), talu_buffer_get_refcount(handle));

    // Retain
    talu_buffer_retain(handle);
    try std.testing.expectEqual(@as(usize, 2), talu_buffer_get_refcount(handle));

    // Get data
    const data_ptr = talu_buffer_get_data_ptr(handle);
    try std.testing.expect(data_ptr != null);
    try std.testing.expectEqual(@as(u32, 1), data_ptr.?[0]);

    // Release twice
    try std.testing.expectEqual(@as(u8, 0), talu_buffer_release(handle));
    try std.testing.expectEqual(@as(u8, 1), talu_buffer_release(handle));
}
