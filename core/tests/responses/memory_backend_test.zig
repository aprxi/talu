//! Integration tests for MemoryBackend
//!
//! MemoryBackend is the default no-op storage backend.
//! All operations are no-ops since data lives in Conversation's in-memory arrays.

const std = @import("std");
const main = @import("main");
const MemoryBackend = main.responses.MemoryBackend;
const StorageEvent = main.responses.StorageEvent;
const MessageRole = main.responses.MessageRole;

// =============================================================================
// Basic Functionality Tests
// =============================================================================

test "MemoryBackend default construction" {
    const mem = MemoryBackend{};

    // Debug stats should be off by default
    try std.testing.expect(!mem.debug_stats);
    try std.testing.expectEqual(@as(usize, 0), mem._debug_persist_count);
    try std.testing.expectEqual(@as(usize, 0), mem._debug_load_count);
    try std.testing.expectEqual(@as(usize, 0), mem._debug_clear_count);
}

test "MemoryBackend.backend returns valid StorageBackend" {
    var mem = MemoryBackend{};
    const backend = mem.backend();

    // Should have non-null vtable
    try std.testing.expect(@intFromPtr(backend.vtable) != 0);
}

test "MemoryBackend.onEvent is no-op" {
    var mem = MemoryBackend{};
    const backend = mem.backend();

    const event = StorageEvent{
        .put_item = .{
            .item_id = 1,
            .created_at_ms = 1000,
            .item_type = .message,
            .role = .user,
        },
    };

    // Multiple calls should not fail
    try backend.onEvent(&event);
    try backend.onEvent(&event);
    try backend.onEvent(&event);

    // Without debug_stats, counters stay at 0
    try std.testing.expectEqual(@as(usize, 0), mem._debug_persist_count);
}

test "MemoryBackend.loadAll returns empty slice" {
    var mem = MemoryBackend{};
    const backend = mem.backend();

    // Should always return empty - memory backend has nothing to load
    const result = try backend.loadAll(std.testing.allocator);
    defer std.testing.allocator.free(result);
    try std.testing.expectEqual(@as(usize, 0), result.len);
}

test "MemoryBackend.deinit is no-op" {
    var mem = MemoryBackend{};
    const backend = mem.backend();

    // Should not crash
    backend.deinit();
}

// =============================================================================
// Debug Statistics Tests
// =============================================================================

test "MemoryBackend tracks operations with debug_stats" {
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

    // Track persist calls
    try backend.onEvent(&event);
    try backend.onEvent(&event);
    try std.testing.expectEqual(@as(usize, 2), mem._debug_persist_count);

    // Track load calls
    var result = try backend.loadAll(std.testing.allocator);
    std.testing.allocator.free(result);
    result = try backend.loadAll(std.testing.allocator);
    std.testing.allocator.free(result);
    result = try backend.loadAll(std.testing.allocator);
    std.testing.allocator.free(result);
    try std.testing.expectEqual(@as(usize, 3), mem._debug_load_count);

    // Track clear calls
    const clear_event = StorageEvent{ .clear_items = .{ .keep_context = false } };
    try backend.onEvent(&clear_event);
    try std.testing.expectEqual(@as(usize, 1), mem._debug_clear_count);
}

test "MemoryBackend debug stats are independent" {
    // Create multiple backends
    var mem1 = MemoryBackend{ .debug_stats = true };
    var mem2 = MemoryBackend{ .debug_stats = true };

    const backend1 = mem1.backend();
    const backend2 = mem2.backend();

    const event = StorageEvent{
        .put_item = .{
            .item_id = 1,
            .created_at_ms = 1000,
            .item_type = .message,
            .role = .assistant,
        },
    };

    // Operations on one don't affect the other
    try backend1.onEvent(&event);
    try backend1.onEvent(&event);
    try backend2.onEvent(&event);

    try std.testing.expectEqual(@as(usize, 2), mem1._debug_persist_count);
    try std.testing.expectEqual(@as(usize, 1), mem2._debug_persist_count);
}

// =============================================================================
// Multiple Role Tests
// =============================================================================

test "MemoryBackend handles all roles" {
    var mem = MemoryBackend{ .debug_stats = true };
    const backend = mem.backend();

    // All roles should work
    const roles = [_]MessageRole{ .system, .user, .assistant, .developer };

    for (roles, 0..) |role, i| {
        const event = StorageEvent{
            .put_item = .{
                .item_id = @intCast(i + 1),
                .created_at_ms = 1000,
                .item_type = .message,
                .role = role,
            },
        };
        try backend.onEvent(&event);
    }

    try std.testing.expectEqual(@as(usize, 4), mem._debug_persist_count);
}
