//! File locking primitives for StoreFS.
//!
//! Provides both blocking and non-blocking exclusive lock helpers.
//! The granular locking model uses blocking locks held only for
//! microseconds during actual I/O operations.

const std = @import("std");
const builtin = @import("builtin");

/// Acquires an exclusive, blocking lock on the file.
/// Blocks until the lock is available.
pub fn lock(file: std.fs.File) !void {
    if (builtin.os.tag == .windows) {
        // Windows: LockFileEx would be ideal, but for now no-op
        return;
    }

    // LOCK.EX without LOCK.NB = blocking exclusive lock
    try std.posix.flock(file.handle, std.posix.LOCK.EX);
}

/// Releases an exclusive lock on the file.
pub fn unlock(file: std.fs.File) void {
    if (builtin.os.tag == .windows) {
        return;
    }

    std.posix.flock(file.handle, std.posix.LOCK.UN) catch {};
}

/// Attempts to acquire an exclusive, non-blocking lock on the file.
/// Returns true on success, false if the lock is already held.
pub fn tryLock(file: std.fs.File) !bool {
    if (builtin.os.tag == .windows) {
        return true;
    }

    std.posix.flock(file.handle, std.posix.LOCK.EX | std.posix.LOCK.NB) catch |err| switch (err) {
        error.WouldBlock => return false,
        else => return err,
    };
    return true;
}

test "lock acquires and unlock releases" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var file = try tmp.dir.createFile("lock.test", .{ .read = true });
    defer file.close();

    // Lock should succeed
    try lock(file);

    // Unlock should not error
    unlock(file);

    // Should be able to lock again after unlock
    try lock(file);
    unlock(file);
}

test "tryLock returns true for unlocked file" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var file = try tmp.dir.createFile("lock.test", .{ .read = true });
    defer file.close();

    const locked = try tryLock(file);
    try std.testing.expect(locked);
}
