//! MemoryBackend - In-memory storage (no persistence).
//!
//! This is the default storage backend. It keeps all items in RAM only.
//! When the process exits, all items are lost.
//!
//! Use this for:
//!   - Development and testing
//!   - Ephemeral chat sessions
//!
//! This backend is a no-op - it doesn't actually store anything.
//! All data lives in the Conversation struct's in-memory arrays.

const std = @import("std");
const backend_mod = @import("backend.zig");

const StorageBackend = backend_mod.StorageBackend;
const StorageEvent = backend_mod.StorageEvent;
const ItemRecord = backend_mod.ItemRecord;

/// MemoryBackend - No-op storage backend.
///
/// This backend does nothing - all storage operations are no-ops.
/// It exists to satisfy the StorageBackend interface when no persistence
/// is needed.
///
/// Thread safety: Fully thread-safe (stateless no-op implementation).
///
/// Example:
///   var mem = MemoryBackend{};
///   const storage = mem.backend();
///   // storage.onEvent() does nothing
///   // storage.loadAll() returns empty slice
pub const MemoryBackend = struct {
    // No state needed - this is a stateless no-op backend.
    // We keep an empty struct so it can be extended in the future
    // (e.g., for debugging/logging purposes).

    /// Optional: track statistics for debugging.
    /// Set to true to count operations (useful for testing).
    debug_stats: bool = false,

    /// Debug counter: number of PutItem events received.
    _debug_put_count: usize = 0,

    /// Debug counter: number of DeleteItem events received.
    _debug_delete_count: usize = 0,

    /// Debug counter: number of ClearItems events received.
    _debug_clear_count: usize = 0,

    /// Debug counter: number of loadAll calls.
    _debug_load_count: usize = 0,

    /// Get a StorageBackend interface for this MemoryBackend.
    ///
    /// Example:
    ///   var mem = MemoryBackend{};
    ///   const conv = try Conversation.initWithStorage(allocator, "session-123", mem.backend());
    pub fn backend(self: *MemoryBackend) StorageBackend {
        return .{
            .ptr = self,
            .vtable = &vtable,
        };
    }

    /// VTable for StorageBackend interface.
    const vtable = StorageBackend.VTable{
        .onEvent = onEvent,
        .loadAll = loadAll,
        .deinit = deinit,
    };

    /// No-op: MemoryBackend doesn't persist events.
    fn onEvent(ctx: *anyopaque, event: *const StorageEvent) anyerror!void {
        const self: *MemoryBackend = @ptrCast(@alignCast(ctx));
        if (self.debug_stats) {
            switch (event.*) {
                .PutItems => |records| self._debug_put_count += records.len,
                .PutItem => self._debug_put_count += 1,
                .DeleteItem => self._debug_delete_count += 1,
                .ClearItems => self._debug_clear_count += 1,
                .PutSession => {}, // Session events are no-ops for memory backend
                .BeginFork => {}, // Fork events are no-ops for memory backend
                .EndFork => {}, // Fork events are no-ops for memory backend
            }
        }
        // No-op: memory backend doesn't persist anything.
        // The data lives in Conversation's in-memory arrays.
    }

    /// No-op: MemoryBackend has nothing to load.
    fn loadAll(ctx: *anyopaque, allocator: std.mem.Allocator) anyerror![]ItemRecord {
        const self: *MemoryBackend = @ptrCast(@alignCast(ctx));
        if (self.debug_stats) {
            self._debug_load_count += 1;
        }
        // Return empty slice - memory backend has no persistence.
        return allocator.alloc(ItemRecord, 0);
    }

    /// No-op: MemoryBackend has no resources to clean up.
    fn deinit(ctx: *anyopaque) void {
        _ = ctx;
        // No-op: nothing to clean up.
    }
};

// =============================================================================
// Tests
// =============================================================================

test "backend MemoryBackend no-op" {
    const allocator = std.testing.allocator;
    var mem = MemoryBackend{};
    const storage = mem.backend();

    // All operations should be no-ops
    const content = try allocator.alloc(backend_mod.ItemContentPartRecord, 1);
    defer allocator.free(content);
    content[0] = .{ .input_text = .{ .text = "test" } };

    const put_event = StorageEvent{ .PutItem = .{
        .item_id = 1,
        .created_at_ms = 123,
        .item_type = .message,
        .variant = .{ .message = .{
            .role = .user,
            .status = .completed,
            .content = content,
        } },
    } };
    try storage.onEvent(&put_event);

    const loaded = try storage.loadAll(allocator);
    defer allocator.free(loaded);
    try std.testing.expectEqual(@as(usize, 0), loaded.len);

    storage.deinit();
}

test "backend MemoryBackend debug stats" {
    const allocator = std.testing.allocator;
    var mem = MemoryBackend{ .debug_stats = true };
    const storage = mem.backend();

    const content = try allocator.alloc(backend_mod.ItemContentPartRecord, 1);
    defer allocator.free(content);
    content[0] = .{ .input_text = .{ .text = "test" } };

    // Test PutItem events
    const put_event = StorageEvent{ .PutItem = .{
        .item_id = 1,
        .created_at_ms = 123,
        .item_type = .message,
        .variant = .{ .message = .{
            .role = .user,
            .status = .completed,
            .content = content,
        } },
    } };
    try storage.onEvent(&put_event);
    try storage.onEvent(&put_event);
    try std.testing.expectEqual(@as(usize, 2), mem._debug_put_count);

    // Test DeleteItem event
    const del_event = StorageEvent{ .DeleteItem = .{
        .item_id = 1,
        .deleted_at_ms = 456,
    } };
    try storage.onEvent(&del_event);
    try std.testing.expectEqual(@as(usize, 1), mem._debug_delete_count);

    // Test ClearItems event
    const clr_event = StorageEvent{ .ClearItems = .{
        .cleared_at_ms = 789,
        .keep_context = false,
    } };
    try storage.onEvent(&clr_event);
    try std.testing.expectEqual(@as(usize, 1), mem._debug_clear_count);

    // Test loadAll
    const loaded = try storage.loadAll(allocator);
    defer allocator.free(loaded);
    try std.testing.expectEqual(@as(usize, 1), mem._debug_load_count);
}

test "backend returns valid StorageBackend interface" {
    var mem = MemoryBackend{};
    const storage = mem.backend();

    // Verify ptr points to the MemoryBackend instance
    const self_ptr: *MemoryBackend = @ptrCast(@alignCast(storage.ptr));
    try std.testing.expect(self_ptr == &mem);

    // Verify that the returned interface is a StorageBackend
    // (compile-time check via type coercion)
    _ = @as(StorageBackend, storage);
}

test "backend interface methods work correctly" {
    const allocator = std.testing.allocator;
    var mem = MemoryBackend{ .debug_stats = true };
    const storage = mem.backend();

    // Test onEvent via interface with PutItem
    const content = try allocator.alloc(backend_mod.ItemContentPartRecord, 1);
    defer allocator.free(content);
    content[0] = .{ .output_text = .{ .text = "hello", .logprobs_json = null, .annotations_json = null } };

    const put_event = StorageEvent{ .PutItem = .{
        .item_id = 42,
        .created_at_ms = 1234567890,
        .item_type = .message,
        .variant = .{ .message = .{
            .role = .assistant,
            .status = .completed,
            .content = content,
        } },
    } };
    try storage.onEvent(&put_event);
    try std.testing.expectEqual(@as(usize, 1), mem._debug_put_count);

    // Test loadAll via interface
    const loaded = try storage.loadAll(allocator);
    defer allocator.free(loaded);
    try std.testing.expectEqual(@as(usize, 0), loaded.len);
    try std.testing.expectEqual(@as(usize, 1), mem._debug_load_count);

    // Test deinit via interface (should be no-op)
    storage.deinit();
}

test "backend with multiple MemoryBackend instances" {
    const allocator = std.testing.allocator;
    var mem1 = MemoryBackend{ .debug_stats = true };
    var mem2 = MemoryBackend{ .debug_stats = true };

    const storage1 = mem1.backend();
    const storage2 = mem2.backend();

    const content = try allocator.alloc(backend_mod.ItemContentPartRecord, 1);
    defer allocator.free(content);
    content[0] = .{ .input_text = .{ .text = "test" } };

    const put_event = StorageEvent{ .PutItem = .{
        .item_id = 1,
        .created_at_ms = 123,
        .item_type = .message,
        .variant = .{ .message = .{
            .role = .user,
            .status = .completed,
            .content = content,
        } },
    } };

    // Operations on storage1 should only affect mem1
    try storage1.onEvent(&put_event);
    try storage1.onEvent(&put_event);
    try std.testing.expectEqual(@as(usize, 2), mem1._debug_put_count);
    try std.testing.expectEqual(@as(usize, 0), mem2._debug_put_count);

    // Operations on storage2 should only affect mem2
    try storage2.onEvent(&put_event);
    try std.testing.expectEqual(@as(usize, 2), mem1._debug_put_count);
    try std.testing.expectEqual(@as(usize, 1), mem2._debug_put_count);
}

test "backend without debug stats does not track operations" {
    const allocator = std.testing.allocator;
    var mem = MemoryBackend{ .debug_stats = false };
    const storage = mem.backend();

    const content = try allocator.alloc(backend_mod.ItemContentPartRecord, 1);
    defer allocator.free(content);
    content[0] = .{ .input_text = .{ .text = "system prompt" } };

    // Operations should complete without updating counters
    const put_event = StorageEvent{ .PutItem = .{
        .item_id = 1,
        .created_at_ms = 123,
        .item_type = .message,
        .variant = .{ .message = .{
            .role = .system,
            .status = .completed,
            .content = content,
        } },
    } };
    try storage.onEvent(&put_event);
    try storage.onEvent(&put_event);

    const loaded = try storage.loadAll(allocator);
    defer allocator.free(loaded);

    const clr_event = StorageEvent{ .ClearItems = .{
        .cleared_at_ms = 456,
        .keep_context = false,
    } };
    try storage.onEvent(&clr_event);

    // Counters remain at initial values
    try std.testing.expectEqual(@as(usize, 0), mem._debug_put_count);
    try std.testing.expectEqual(@as(usize, 0), mem._debug_delete_count);
    try std.testing.expectEqual(@as(usize, 0), mem._debug_clear_count);
    try std.testing.expectEqual(@as(usize, 0), mem._debug_load_count);
}

test "backend interface handles all event types" {
    const allocator = std.testing.allocator;
    var mem = MemoryBackend{ .debug_stats = true };
    const storage = mem.backend();

    // Test PutItem with different roles
    const user_content = try allocator.alloc(backend_mod.ItemContentPartRecord, 1);
    defer allocator.free(user_content);
    user_content[0] = .{ .input_text = .{ .text = "user message" } };

    const user_event = StorageEvent{ .PutItem = .{
        .item_id = 1,
        .created_at_ms = 1000,
        .item_type = .message,
        .variant = .{ .message = .{
            .role = .user,
            .status = .completed,
            .content = user_content,
        } },
    } };
    try storage.onEvent(&user_event);

    const asst_content = try allocator.alloc(backend_mod.ItemContentPartRecord, 1);
    defer allocator.free(asst_content);
    asst_content[0] = .{ .output_text = .{ .text = "assistant response", .logprobs_json = null, .annotations_json = null } };

    const asst_event = StorageEvent{ .PutItem = .{
        .item_id = 2,
        .created_at_ms = 2000,
        .item_type = .message,
        .variant = .{ .message = .{
            .role = .assistant,
            .status = .completed,
            .content = asst_content,
        } },
    } };
    try storage.onEvent(&asst_event);

    const sys_content = try allocator.alloc(backend_mod.ItemContentPartRecord, 1);
    defer allocator.free(sys_content);
    sys_content[0] = .{ .input_text = .{ .text = "system prompt" } };

    const sys_event = StorageEvent{ .PutItem = .{
        .item_id = 0,
        .created_at_ms = 500,
        .item_type = .message,
        .variant = .{ .message = .{
            .role = .system,
            .status = .completed,
            .content = sys_content,
        } },
    } };
    try storage.onEvent(&sys_event);

    // Test DeleteItem
    const del_event = StorageEvent{ .DeleteItem = .{
        .item_id = 1,
        .deleted_at_ms = 3000,
    } };
    try storage.onEvent(&del_event);

    // Test ClearItems
    const clr_event = StorageEvent{ .ClearItems = .{
        .cleared_at_ms = 4000,
        .keep_context = true,
    } };
    try storage.onEvent(&clr_event);

    // Verify all event types were tracked
    try std.testing.expectEqual(@as(usize, 3), mem._debug_put_count);
    try std.testing.expectEqual(@as(usize, 1), mem._debug_delete_count);
    try std.testing.expectEqual(@as(usize, 1), mem._debug_clear_count);

    // loadAll should still return empty (no persistence)
    const loaded = try storage.loadAll(allocator);
    defer allocator.free(loaded);
    try std.testing.expectEqual(@as(usize, 0), loaded.len);
}
