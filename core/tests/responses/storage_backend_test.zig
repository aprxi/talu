//! Integration tests for responses.StorageBackend
//!
//! StorageBackend is the vtable interface for persistence backends.
//! It provides runtime polymorphism for different storage implementations.

const std = @import("std");
const main = @import("main");
const StorageBackend = main.responses.StorageBackend;
const StorageEvent = main.responses.StorageEvent;
const ItemRecord = main.responses.ItemRecord;
const MemoryBackend = main.responses.MemoryBackend;

// =============================================================================
// VTable Interface Tests
// =============================================================================

test "StorageBackend can be created from MemoryBackend" {
    var mem = MemoryBackend{};
    const backend = mem.backend();

    // Should have non-null pointers
    try std.testing.expect(@intFromPtr(backend.vtable) != 0);
    try std.testing.expect(@intFromPtr(backend.ptr) != 0);
}

test "StorageBackend.onEvent accepts StorageEvent" {
    var mem = MemoryBackend{ .debug_stats = true };
    const backend = mem.backend();

    const event = StorageEvent{
        .put_item = .{
            .item_id = 1,
            .created_at_ms = 1000,
            .item_type = .message,
            .role = .user,
        },
    };

    // Should not error
    try backend.onEvent(&event);

    // MemoryBackend tracks calls
    try std.testing.expectEqual(@as(usize, 1), mem._debug_persist_count);
}

test "StorageBackend.loadAll returns empty slice for empty MemoryBackend" {
    var mem = MemoryBackend{};
    const backend = mem.backend();

    const loaded = try backend.loadAll(std.testing.allocator);
    defer std.testing.allocator.free(loaded);
    try std.testing.expectEqual(@as(usize, 0), loaded.len);
}

test "StorageBackend.deinit is callable" {
    var mem = MemoryBackend{};
    const backend = mem.backend();

    // Should not crash
    backend.deinit();
}

// =============================================================================
// Custom Backend Implementation Test
// =============================================================================

/// Test backend that records all operations for verification.
const TestBackend = struct {
    events_received: usize = 0,
    deinit_called: bool = false,

    fn backend(self: *TestBackend) StorageBackend {
        return StorageBackend.init(self, &vtable);
    }

    const vtable = StorageBackend.VTable{
        .onEvent = onEvent,
        .loadAll = loadAll,
        .deinit = deinit,
    };

    fn onEvent(ctx: *anyopaque, event: *const StorageEvent) anyerror!void {
        _ = event;
        const self: *TestBackend = @ptrCast(@alignCast(ctx));
        self.events_received += 1;
    }

    fn loadAll(ctx: *anyopaque, allocator: std.mem.Allocator) anyerror![]ItemRecord {
        _ = ctx;
        return allocator.alloc(ItemRecord, 0);
    }

    fn deinit(ctx: *anyopaque) void {
        const self: *TestBackend = @ptrCast(@alignCast(ctx));
        self.deinit_called = true;
    }
};

test "StorageBackend works with custom implementation" {
    var test_backend = TestBackend{};
    const backend = test_backend.backend();

    // Send events
    const event1 = StorageEvent{
        .put_item = .{
            .item_id = 1,
            .created_at_ms = 1000,
            .item_type = .message,
            .role = .user,
        },
    };
    const event2 = StorageEvent{
        .put_item = .{
            .item_id = 2,
            .created_at_ms = 2000,
            .item_type = .message,
            .role = .assistant,
        },
    };
    const event3 = StorageEvent{ .clear_items = .{ .keep_context = false } };

    try backend.onEvent(&event1);
    try backend.onEvent(&event2);
    try backend.onEvent(&event3);

    try std.testing.expectEqual(@as(usize, 3), test_backend.events_received);

    // Deinit
    backend.deinit();
    try std.testing.expect(test_backend.deinit_called);
}
