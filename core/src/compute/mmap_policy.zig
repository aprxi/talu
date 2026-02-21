//! Platform policy for read-only file mmap behavior.
//!
//! Keeps OS-specific mmap tuning out of io/ modules.

const std = @import("std");
const builtin = @import("builtin");

pub fn applyReadOnlyMapPolicy(flags: *std.posix.MAP, size: usize) void {
    if (comptime builtin.os.tag == .linux) {
        if (shouldPopulatePages(size)) {
            flags.POPULATE = true;
        }
    }
}

pub fn shouldPopulatePages(size: usize) bool {
    if (builtin.os.tag != .linux) return false;

    // Explicit override via environment variable
    if (std.posix.getenv("TALU_MMAP_POPULATE")) |v| {
        if (std.mem.eql(u8, v, "0") or std.mem.eql(u8, v, "false")) return false;
        if (std.mem.eql(u8, v, "1") or std.mem.eql(u8, v, "true")) return true;
    }

    // MAP_POPULATE eagerly faults all pages and can look like a "hang" for very large shards.
    // Keep it enabled for smaller files where it improves latency.
    const max_bytes: usize = 512 * 1024 * 1024;
    return size <= max_bytes;
}

test "shouldPopulatePages threshold policy" {
    if (builtin.os.tag != .linux) return;

    try std.testing.expect(shouldPopulatePages(1));
    try std.testing.expect(!shouldPopulatePages((512 * 1024 * 1024) + 1));
}

test "applyReadOnlyMapPolicy toggles MAP_POPULATE according to policy" {
    if (builtin.os.tag != .linux) return;

    var flags: std.posix.MAP = undefined;
    @memset(std.mem.asBytes(&flags), 0);
    applyReadOnlyMapPolicy(&flags, 1);

    if (shouldPopulatePages(1)) {
        try std.testing.expect(@field(flags, "POPULATE"));
    } else {
        try std.testing.expect(!@field(flags, "POPULATE"));
    }
}
